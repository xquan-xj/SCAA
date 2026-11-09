"""Content-and-relative-position axial skip cross attention (SCAA).

By default cross attention uses the selected Q source and K/V from the opposite source.
This isolated experiment may override key_source only, retaining the Q and V
definitions, all axial operators and the E/D gate. key_source='auto' preserves
the earlier initialization, forward and gradient paths exactly.
cross_attention=False keeps the Q source and uses that same source for K/V.
The selected Full uses decoder Q and encoder K/V. All sources
are explicitly projected to heads*key_dim channels. After pooled attention,
each output is restored to the skip's original H,W. Rows index theta and columns
index radius: radial attention traverses columns, circumferential traverses rows.

For each axis the logits are QK^T/sqrt(key_dim) + learned relative-position bias,
followed by ONE softmax over key positions. No concatenation or channel slicing
can discard the content term. The content_attention=False ablation uses only the
position logits. A sigmoid channel/spatial gate blends the two axis outputs;
gated=False uses their fixed arithmetic mean. Single-axis modes skip the gate.
axes='global' flattens the same pooled rectangular grid, adds the two learned
relative-axis biases, and chunks queries at128 positions. It has one branch and
therefore no dual-axis gate; this is a structural dependency of that ablation.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import positive_int


def scaled_attention(q, k, v, position_bias=None, content_attention=True):
    """Return softmax(QK^T/sqrt(d) + position_bias) V.

    Inputs have identical leading dimensions and trailing (query/key length,
    channel) dimensions. Q/K channel widths and K/V lengths must agree; V may
    have a different channel width. Position bias can broadcast to the logits.
    Half precision content products/softmax accumulate in float32 under AMP.
    """
    if (q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim
            or q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]
            or q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]
            or min(q.shape[-2:]) < 1 or k.shape[-2] < 1):
        raise ValueError("incompatible Q/K/V shape, feature dimension or key/value length")
    dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    # Disable autocast here so reduced-precision matmul cannot overflow before
    # softmax even though the outer training step may use autocast.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if content_attention:
            logits = torch.matmul(q.to(dtype), k.to(dtype).transpose(-1, -2))
            logits = logits / math.sqrt(q.shape[-1])
        else:
            logits = torch.zeros((*q.shape[:-2], q.shape[-2], k.shape[-2]),
                                 device=q.device, dtype=dtype)
        if position_bias is not None:
            try:
                bias_shape = torch.broadcast_shapes(logits.shape, position_bias.shape)
            except RuntimeError as error:
                raise ValueError("position bias shape cannot broadcast to attention logits") from error
            if bias_shape != logits.shape:
                raise ValueError("position bias must not expand attention batch/head dimensions")
            logits = logits + position_bias.to(device=q.device, dtype=dtype)
        output = torch.matmul(logits.softmax(dim=-1), v.to(dtype))
    return output.to(v.dtype)


class RelativePositionBias(nn.Module):
    """Per-head displacement table with a fixed linear interpolation matrix.

    Values/gradients match linear align_corners=True interpolation (including
    output length1 selecting source index0). Matrix multiplication avoids
    CUDA's unsupported deterministic upsample_linear1d backward. The matrix is
    fixed by the axis length and has no trainable parameters.
    """

    def __init__(self, heads, bins=127):
        super().__init__()
        self.table = nn.Parameter(torch.empty(heads, bins))
        nn.init.normal_(self.table, std=.02)

    def forward(self, length):
        target_bins = 2 * length - 1
        source_bins = self.table.shape[-1]
        scale = (source_bins - 1) / (target_bins - 1) if target_bins > 1 else 0.
        coordinates = torch.arange(target_bins, device=self.table.device,
                                    dtype=torch.float64) * scale
        left = coordinates.floor().long().clamp(max=source_bins - 1)
        right = (left + 1).clamp(max=source_bins - 1)
        fraction = (coordinates - left).to(self.table.dtype)
        weights = (F.one_hot(left, num_classes=source_bins).T.to(self.table.dtype)
                   * (1 - fraction).unsqueeze(0)
                   + F.one_hot(right, num_classes=source_bins).T.to(self.table.dtype)
                   * fraction.unsqueeze(0))
        with torch.autocast(device_type=self.table.device.type, enabled=False):
            table = self.table @ weights
        coords = torch.arange(length, device=table.device)
        offsets = coords[:, None] - coords[None, :] + length - 1
        return table[:, offsets]


class SCAA(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels,
                 axes="both", gated=True, content_attention=True,
                 pool_factor=4, attention_heads=4, key_dim=8,
                 cross_attention=True, query_source="encoder", key_source="auto"):
        super().__init__()
        for name, value in (("encoder_channels", encoder_channels),
                            ("decoder_channels", decoder_channels),
                            ("out_channels", out_channels),
                            ("pool_factor", pool_factor),
                            ("attention_heads", attention_heads), ("key_dim", key_dim)):
            positive_int(value, name)
        if axes not in ("both", "radial", "circumferential", "global"):
            raise ValueError("axes must be both, radial, circumferential or global")
        if any(not isinstance(value, bool) for value in (gated, content_attention, cross_attention)):
            raise ValueError("gated, content_attention and cross_attention must be booleans")
        if query_source not in ("encoder", "decoder"):
            raise ValueError("query_source must be encoder or decoder")
        if key_source not in ("auto", "encoder", "decoder"):
            raise ValueError("key_source must be auto, encoder or decoder")
        self.encoder_channels, self.decoder_channels = encoder_channels, decoder_channels
        self.axes, self.gated, self.content_attention = axes, gated, content_attention
        self.cross_attention = cross_attention
        self.query_source = query_source
        self.key_source = key_source
        self.pool_factor, self.heads, self.key_dim = pool_factor, attention_heads, key_dim
        self.global_query_chunk = 128
        # Recompute each exact global query block during backward, including
        # its positional bias, instead of retaining the full N x N softmax.
        self.global_activation_checkpoint = True
        hidden = attention_heads * key_dim
        query_channels = decoder_channels if query_source == "decoder" else encoder_channels
        self.query = nn.Conv2d(query_channels, hidden, 1)
        source_channels = (decoder_channels if query_source == "encoder" else encoder_channels) if cross_attention else query_channels
        key_channels = source_channels if key_source == "auto" else (encoder_channels if key_source == "encoder" else decoder_channels)
        self.key = nn.Conv2d(key_channels, hidden, 1)
        self.value = nn.Conv2d(source_channels, hidden, 1)
        self.output = nn.Conv2d(hidden, out_channels, 1)
        selected = ("radial", "circumferential") if axes in ("both", "global") else (axes,)
        self.relative = nn.ModuleDict({axis: RelativePositionBias(attention_heads) for axis in selected})
        self.gate = (nn.Conv2d(encoder_channels + decoder_channels, out_channels, 1)
                     if axes == "both" and gated else None)

    def _attend_axis(self, q, k, v, axis):
        batch, _, height, width = q.shape
        # radial: B*theta x heads x radius x d;
        # circumferential: B*radius x heads x theta x d.
        def sequence(x):
            x = x.reshape(batch, self.heads, self.key_dim, height, width)
            permutation = (0, 3, 1, 4, 2) if axis == "radial" else (0, 4, 1, 3, 2)
            length = width if axis == "radial" else height
            return x.permute(permutation).reshape(-1, self.heads, length, self.key_dim)
        length = width if axis == "radial" else height
        result = scaled_attention(sequence(q), sequence(k), sequence(v),
                                  self.relative[axis](length).unsqueeze(0),
                                  self.content_attention)
        if axis == "radial":
            result = result.reshape(batch, height, self.heads, width, self.key_dim)
            result = result.permute(0, 2, 4, 1, 3)
        else:
            result = result.reshape(batch, width, self.heads, height, self.key_dim)
            result = result.permute(0, 2, 4, 3, 1)
        return self.output(result.reshape(batch, self.heads * self.key_dim, height, width))

    def _attend_global(self, q, k, v):
        batch, _, height, width = q.shape
        total = height * width
        def sequence(x):
            return x.reshape(batch, self.heads, self.key_dim, total).transpose(-1, -2)
        q, k, v = sequence(q), sequence(k), sequence(v)
        radial = self.relative["radial"](width)
        circumferential = self.relative["circumferential"](height)
        key_positions = torch.arange(total, device=q.device)
        key_radius, key_theta = key_positions % width, key_positions // width
        chunks = []
        for start in range(0, total, self.global_query_chunk):
            end = min(start + self.global_query_chunk, total)
            query_positions = torch.arange(start, end, device=q.device)
            # Bind the positions now: a late-bound loop closure would silently
            # recompute earlier chunks with the final chunk's positions.
            def attend(q_chunk, keys, values, radial_bias, angular_bias,
                       positions=query_positions):
                query_radius, query_theta = positions % width, positions // width
                bias = (radial_bias[:, query_radius[:, None], key_radius[None, :]]
                        + angular_bias[:, query_theta[:, None], key_theta[None, :]])
                return scaled_attention(q_chunk, keys, values, bias.unsqueeze(0),
                                        self.content_attention)
            args = (q[:, :, start:end], k, v, radial, circumferential)
            chunks.append(checkpoint(attend, *args, use_reentrant=False)
                          if self.global_activation_checkpoint and torch.is_grad_enabled()
                          else attend(*args))
        output = torch.cat(chunks, dim=-2).transpose(-1, -2)
        return self.output(output.reshape(batch, self.heads * self.key_dim, height, width))

    def forward(self, skip, decoder):
        if (skip.ndim != 4 or decoder.ndim != 4
                or skip.shape[0] != decoder.shape[0]
                or skip.shape[1] != self.encoder_channels
                or decoder.shape[1] != self.decoder_channels
                or skip.shape[-2:] != decoder.shape[-2:]):
            raise ValueError("SCAA expects channel-compatible, spatially aligned encoder/decoder shapes")
        height, width = skip.shape[-2:]
        pooled_size = (max(1, height // self.pool_factor), max(1, width // self.pool_factor))
        pooled_skip = F.adaptive_avg_pool2d(skip, pooled_size)
        pooled_decoder = F.adaptive_avg_pool2d(decoder, pooled_size)
        query = pooled_decoder if self.query_source == "decoder" else pooled_skip
        source = ((pooled_decoder if self.query_source == "encoder" else pooled_skip)
                  if self.cross_attention else query)
        key_input = source if self.key_source == "auto" else (pooled_skip if self.key_source == "encoder" else pooled_decoder)
        q, k, v = self.query(query), self.key(key_input), self.value(source)
        if self.axes == "global":
            return F.interpolate(self._attend_global(q, k, v), size=(height, width),
                                  mode="bilinear", align_corners=False)
        outputs = [F.interpolate(self._attend_axis(q, k, v, axis), size=(height, width),
                                 mode="bilinear", align_corners=False)
                   for axis in self.relative]
        if len(outputs) == 1:
            return outputs[0]
        if self.gate is None:
            return (outputs[0] + outputs[1]) * .5
        gate = torch.sigmoid(self.gate(torch.cat((skip, decoder), dim=1)))
        return gate * outputs[0] + (1 - gate) * outputs[1]
