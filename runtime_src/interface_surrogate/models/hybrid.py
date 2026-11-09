"""A self-contained R50V2--ViT hybrid encoder and U-shaped decoder.

Default encoder: weight-standardized 7x7/stride-2 stem (64 channels), 3x3
max-pool/stride-2, GroupNorm bottleneck stages with (3,4,9) units and
(256,512,1024) output channels. The last two stages downsample, giving stride16.
This is the three-stage R50V2 hybrid used in TransUNet-style R50-ViT-B/16,
not the four-stage classification ResNet50. Bottleneck conv/GN ordering follows
the supplied architectural reference's ResNetV2 implementation. Transformer:
1x1 patch projection, learned 2D position grid, 12 pre-norm ViT blocks with
hidden size768, 12 heads, MLP ratio4, final LayerNorm. The 2D grid is tracked
explicitly, including rectangular inputs; no square-root token reshaping.

Decoder: hidden->512 3x3 head, (256,128,64,16) convolutional stages, bilinear
alignment to encoder skips (512,256,64) and original input size; one linear
normalized-target output channel. Decoder normalization is GroupNorm. The
same padding, interpolation, skips, optional smoothing, and max pooling apply
to concat/SCAA variants. Max pooling uses the same native 3x3/pad1 operation
in both variants; convolution and smoothing use the chosen zeros/reflect mode.

Architectural reference only: the supplied xnet-orin files
vit_seg_modeling_resnet_skip.py and vit_seg_modeling.py. This clean, modified
implementation is not byte-identical or weight-compatible with official
pretrained TransUNet. It contains no pretrained loaders, network downloads,
global CLI config, historical weights, or ml_collections dependency. Compact
transformer_hidden/layers/heads and resnet_units overrides support smoke tests.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import DecoderBlock, OutputSmoothing, norm, positive_int, validate_field
from .attention import SCAA


class StdConv2d(nn.Conv2d):
    """Per-output-channel weight standardization before each convolution."""

    def forward(self, x):
        variance, mean = torch.var_mean(self.weight, dim=(1, 2, 3),
                                       keepdim=True, unbiased=False)
        weight = (self.weight - mean) * torch.rsqrt(variance + 1e-5)
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv2d(x, weight, self.bias, self.stride, padding,
                        self.dilation, self.groups)


class ResNetV2Bottleneck(nn.Module):
    """Reference-style weight-standardized residual bottleneck with GroupNorm."""

    def __init__(self, cin, cout, middle, stride, padding):
        super().__init__()
        self.conv1 = StdConv2d(cin, middle, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, middle, eps=1e-6)
        self.conv2 = StdConv2d(middle, middle, 3, stride=stride, padding=1,
                              padding_mode=padding, bias=False)
        self.gn2 = nn.GroupNorm(32, middle, eps=1e-6)
        self.conv3 = StdConv2d(middle, cout, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.projection = (nn.Sequential(StdConv2d(cin, cout, 1, stride=stride, bias=False),
                                         nn.GroupNorm(cout, cout, eps=1e-6))
                           if cin != cout or stride != 1 else nn.Identity())

    def forward(self, x):
        residual = self.projection(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        return F.relu(x + residual)


class ResNetV2(nn.Module):
    output_stride = 16

    def __init__(self, input_channels, units=(3, 4, 9), padding="zeros"):
        super().__init__()
        self.stem = nn.Sequential(
            StdConv2d(input_channels, 64, 7, stride=2, padding=3,
                      padding_mode=padding, bias=False),
            nn.GroupNorm(32, 64, eps=1e-6), nn.ReLU())
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        stages, cin = [], 64
        for index, (count, cout, middle) in enumerate(zip(units, (256, 512, 1024), (64, 128, 256))):
            blocks = [ResNetV2Bottleneck(cin, cout, middle, 1 if index == 0 else 2, padding)]
            blocks += [ResNetV2Bottleneck(cout, cout, middle, 1, padding)
                       for _ in range(count - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        stem = self.stem(x)
        stage1 = self.stages[0](self.pool(stem))
        stage2 = self.stages[1](stage1)
        deepest = self.stages[2](stage2)
        return deepest, (stage2, stage1, stem)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio, dropout, attention_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=attention_dropout,
                                              batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(hidden * mlp_ratio), hidden),
                                 nn.Dropout(dropout))

    def forward(self, x):
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class HybridTransformer(nn.Module):
    def __init__(self, output_size, hidden=768, layers=12, heads=12, mlp_ratio=4.,
                 dropout=.1, attention_dropout=0.):
        super().__init__()
        self.hidden_size = hidden
        self.patch_embedding = nn.Conv2d(1024, hidden, 1)
        grid = tuple(math.ceil(size / 16) for size in output_size)
        self.position_embedding = nn.Parameter(torch.empty(1, hidden, *grid))
        nn.init.trunc_normal_(self.position_embedding, std=.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(hidden, heads, mlp_ratio,
                                                      dropout, attention_dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, features):
        features = self.patch_embedding(features)
        batch, hidden, height, width = features.shape
        position = F.interpolate(self.position_embedding, size=(height, width),
                                  mode="bilinear", align_corners=False)
        tokens = self.dropout((features + position).flatten(2).transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).transpose(1, 2).reshape(batch, hidden, height, width)


class HybridTransUNet(nn.Module):
    def __init__(self, input_channels, output_size, controls, config):
        super().__init__()
        hidden = positive_int(config.get("transformer_hidden", 768), "transformer_hidden")
        layers = positive_int(config.get("transformer_layers", 12), "transformer_layers")
        heads = positive_int(config.get("transformer_heads", 12), "transformer_heads")
        if hidden % heads:
            raise ValueError("transformer_hidden must be divisible by transformer_heads")
        units = config.get("resnet_units", (3, 4, 9))
        if not isinstance(units, (list, tuple)) or len(units) != 3:
            raise ValueError("resnet_units must contain three positive stage lengths")
        units = tuple(positive_int(x, "resnet unit count") for x in units)
        mlp_ratio = config.get("transformer_mlp_ratio", 4.)
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0 or not math.isfinite(mlp_ratio):
            raise ValueError("transformer_mlp_ratio must be positive and finite")
        if int(hidden * mlp_ratio) < 1:
            raise ValueError("transformer MLP hidden width must be positive")
        dropout = config.get("transformer_dropout", .1)
        attention_dropout = config.get("transformer_attention_dropout", 0.)
        if any(not isinstance(p, (int, float)) or not 0 <= p < 1
               for p in (dropout, attention_dropout)):
            raise ValueError("transformer dropout probabilities must be in [0, 1)")
        decoder_channels = config.get("decoder_channels", (256, 128, 64, 16))
        if not isinstance(decoder_channels, (list, tuple)) or len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four positive widths")
        decoder_channels = tuple(positive_int(c, "decoder channel") for c in decoder_channels)
        self.input_channels = input_channels
        # Reflection convolutions at the bottleneck need at least 2 cells/axis.
        self.minimum = 32 if controls["padding"] == "reflect" else 17
        self.encoder = ResNetV2(input_channels, units, controls["padding"])
        self.transformer = HybridTransformer(output_size, hidden, layers, heads, mlp_ratio,
                                             dropout, attention_dropout)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(hidden, 512, 3, padding=1, padding_mode=controls["padding"], bias=False),
            norm(512), nn.ReLU())
        skip_controls = dict(controls)
        if controls["attention_placement"] == "reconstruction":
            skip_controls["fusion"] = "concat"
        self.decoder = nn.ModuleList([
            DecoderBlock(cin, skip, cout, skip_controls)
            for cin, skip, cout in zip((512,) + decoder_channels[:-1],
                                      (512, 256, 64, 0), decoder_channels)])
        self.reconstruction_attention = None
        if controls["fusion"] == "scaa" and controls["attention_placement"] in ("reconstruction", "dual"):
            self.reconstruction_attention = SCAA(64, decoder_channels[-1], decoder_channels[-1],
                **{key: controls[key] for key in ("axes", "gated", "content_attention",
                   "cross_attention", "pool_factor", "attention_heads", "key_dim")})
        self.head = nn.Conv2d(decoder_channels[-1], 1, 3, padding=1,
                              padding_mode=controls["padding"])
        self.smoothing = OutputSmoothing(controls["smoothing"], controls["padding"],
                                         controls["smoothing_kernels"])

    def forward(self, x):
        validate_field(x, self.input_channels, self.minimum)
        input_size = x.shape[-2:]
        features, skips = self.encoder(x)
        x = self.decoder_head(self.transformer(features))
        for index, block in enumerate(self.decoder):
            x = block(x, skips[index]) if index < 3 else block(x, size=input_size)
        if self.reconstruction_attention is not None:
            reconstruction_skip = F.interpolate(skips[-1], size=input_size,
                                                 mode="bilinear", align_corners=False)
            x = self.reconstruction_attention(reconstruction_skip, x)
        return self.smoothing(self.head(x))
