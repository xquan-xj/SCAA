"""Shared boundary, decoder and output controls for controlled comparisons."""

from numbers import Integral

import torch
from torch import nn
import torch.nn.functional as F


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def output_shape(size):
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("output_size must contain (height, width)")
    return tuple(positive_int(x, "output_size dimension") for x in size)


def validate_controls(config):
    controls = dict(padding=config.get("padding", "zeros"),
                    fusion=config.get("fusion", "concat"),
                    axes=config.get("axes", "both"),
                    cross_attention=config.get("cross_attention", True),
                    query_source=config.get("query_source", "encoder"),
                    key_source=config.get("key_source", "auto"),
                    attention_placement=config.get("attention_placement", "skip"),
                    gated=config.get("gated", True),
                    content_attention=config.get("content_attention", True),
                    smoothing=config.get("smoothing", False),
                    smoothing_kernels=config.get("smoothing_kernels", [7, 5, 3]),
                    pool_factor=config.get("pool_factor", 4),
                    attention_heads=config.get("attention_heads", 4),
                    key_dim=config.get("key_dim", 8))
    for key, allowed in (("padding", ("zeros", "reflect")),
                         ("fusion", ("concat", "scaa", "concat_scaa", "scaa_residual_sum")),
                         ("query_source", ("encoder", "decoder")),
                         ("key_source", ("auto", "encoder", "decoder")),
                         ("axes", ("both", "radial", "circumferential", "global")),
                         ("attention_placement", ("skip", "reconstruction", "dual"))):
        if controls[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    if controls["fusion"] in ("concat_scaa", "scaa_residual_sum"):
        if controls["attention_placement"] in ("reconstruction", "dual") and config.get("kind", "unet") != "unet":
            raise ValueError("Reconstruction/dual concat/residual is defined for U-Net only")
    if controls["key_source"] != "auto" and config.get("kind", "unet") != "unet":
        raise ValueError("Explicit key_source is supported for this U-Net experiment only")
    for key in ("gated", "content_attention", "cross_attention", "smoothing"):
        if not isinstance(controls[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("pool_factor", "attention_heads", "key_dim"):
        controls[key] = positive_int(controls[key], key)
    kernels = controls["smoothing_kernels"]
    if not isinstance(kernels, (list, tuple)) or not kernels:
        raise ValueError("smoothing_kernels must be a nonempty sequence of odd integers")
    kernels = tuple(positive_int(x, "smoothing kernel") for x in kernels)
    if any(x % 2 == 0 for x in kernels):
        raise ValueError("smoothing_kernels must be odd")
    controls["smoothing_kernels"] = kernels
    return controls


def norm(channels):
    # Keep at least two channels/group when possible so valid tiny bottlenecks
    # (batch1, spatial1x1) are supported as well as the normal research grids.
    groups = max(1, min(8, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding="zeros"):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,
                      padding_mode=padding, bias=False),
            norm(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      padding_mode=padding, bias=False),
            norm(out_channels), nn.ReLU(inplace=True))


class OutputSmoothing(nn.Module):
    """Optional sequential mean filters; identical for all model/fusion kinds.

    Default active kernel sequence is (7, 5, 3). Padding is explicit, including
    zeros in the denominator, and every stage preserves the input dimensions.
    No learned parameters or normalization/physical-unit conversion is applied.
    """

    def __init__(self, enabled=False, padding="zeros", kernels=(7, 5, 3)):
        super().__init__()
        self.enabled, self.padding, self.kernels = enabled, padding, tuple(kernels)

    def forward(self, x):
        if not self.enabled:
            return x
        for kernel in self.kernels:
            pad = kernel // 2
            if self.padding == "reflect" and min(x.shape[-2:]) <= pad:
                raise ValueError("output size is too small for reflection smoothing kernels")
            mode = "constant" if self.padding == "zeros" else "reflect"
            x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode=mode),
                             kernel, stride=1)
        return x


class DecoderBlock(nn.Module):
    """Bilinear alignment, explicit skip fusion, then the same ConvBlock.

    concat_scaa retains both original value paths and appends the attention
    output: Conv(cat(encoder, aligned_decoder, SCAA(encoder, decoder))).
    """

    def __init__(self, decoder_channels, skip_channels, out_channels, controls):
        super().__init__()
        self.skip_channels = skip_channels
        self.fusion = controls["fusion"]
        if skip_channels and self.fusion in ("scaa", "concat_scaa", "scaa_residual_sum"):
            from .attention import SCAA
            self.attention = SCAA(
                skip_channels, decoder_channels, out_channels,
                **{key: controls[key] for key in
                   ("axes", "gated", "content_attention", "cross_attention", "query_source", "key_source", "pool_factor", "attention_heads", "key_dim")})
            fused_channels = (out_channels if self.fusion in ("scaa", "scaa_residual_sum") else
                              skip_channels + decoder_channels + out_channels)
        else:
            self.attention = None
            fused_channels = decoder_channels + skip_channels
        self.local_encoder = self.local_decoder = None
        if skip_channels and self.fusion == "scaa_residual_sum":
            self.local_encoder = (nn.Identity() if skip_channels == out_channels else
                                  nn.Conv2d(skip_channels, out_channels, 1, bias=False))
            self.local_decoder = nn.Conv2d(decoder_channels, out_channels, 1, bias=False)
        self.refine = ConvBlock(fused_channels, out_channels, controls["padding"])

    def forward(self, decoder, skip=None, size=None):
        if self.skip_channels:
            if skip is None or skip.ndim != 4 or skip.shape[1] != self.skip_channels:
                raise ValueError("decoder skip shape/channels do not match configuration")
            size = skip.shape[-2:]
        if size is None:
            raise ValueError("decoder needs an explicit target size")
        decoder = F.interpolate(decoder, size=size, mode="bilinear", align_corners=False)
        if self.skip_channels:
            if self.fusion == "concat_scaa":
                decoder = torch.cat((skip, decoder, self.attention(skip, decoder)), dim=1)
            elif self.fusion == "scaa_residual_sum":
                decoder = (self.local_encoder(skip) + self.local_decoder(decoder)
                           + self.attention(skip, decoder))
            else:
                decoder = (self.attention(skip, decoder) if self.attention is not None
                           else torch.cat((skip, decoder), dim=1))
        return self.refine(decoder)


def validate_field(x, channels, minimum):
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(f"field input shape must be B x {channels} channels x H x W")
    if min(x.shape[-2:]) < minimum:
        raise ValueError(f"field size is too small; minimum dimension is {minimum}")
