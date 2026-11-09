"""U-Net with shared max pooling and boundary policy across fusion variants."""

import torch
from torch import nn

from .common import ConvBlock, DecoderBlock, OutputSmoothing, validate_field
from .attention import SCAA


class UNet(nn.Module):
    def __init__(self, input_channels, width, depth, controls):
        super().__init__()
        self.input_channels, self.depth = input_channels, depth
        self.minimum = 2 ** depth * (2 if controls["padding"] == "reflect" else 1)
        channels = [width * 2 ** i for i in range(depth + 1)]
        self.encoder = nn.ModuleList([
            ConvBlock(input_channels if i == 0 else channels[i - 1], channels[i],
                      controls["padding"]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2, stride=2)
        skip_controls = dict(controls)
        if controls["attention_placement"] == "reconstruction":
            skip_controls["fusion"] = "concat"
        self.decoder = nn.ModuleList([
            DecoderBlock(channels[i + 1], channels[i], channels[i], skip_controls)
            for i in reversed(range(depth))])
        self.reconstruction_attention = None
        self.reconstruction_fusion = controls["fusion"]
        self.reconstruction_refine = None
        self.reconstruction_local_encoder = None
        self.reconstruction_local_decoder = None
        if controls["fusion"] in ("scaa", "concat_scaa", "scaa_residual_sum") and controls["attention_placement"] in ("reconstruction", "dual"):
            self.reconstruction_attention = SCAA(channels[0], channels[0], channels[0],
                **{key: controls[key] for key in ("axes", "gated", "content_attention",
                   "cross_attention", "query_source", "key_source", "pool_factor", "attention_heads", "key_dim")})
            if controls["fusion"] == "concat_scaa":
                self.reconstruction_refine = ConvBlock(3 * channels[0], channels[0], controls["padding"])
            elif controls["fusion"] == "scaa_residual_sum":
                self.reconstruction_local_encoder = nn.Identity()
                self.reconstruction_local_decoder = nn.Conv2d(channels[0], channels[0], 1, bias=False)
                self.reconstruction_refine = ConvBlock(channels[0], channels[0], controls["padding"])
        self.head = nn.Conv2d(channels[0], 1, 1)
        self.smoothing = OutputSmoothing(controls["smoothing"], controls["padding"],
                                         controls["smoothing_kernels"])

    def forward(self, x):
        validate_field(x, self.input_channels, self.minimum)
        skips = []
        for index, block in enumerate(self.encoder):
            x = block(x if index == 0 else self.pool(x))
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)
        if self.reconstruction_attention is not None:
            attention = self.reconstruction_attention(skips[0], x)
            if self.reconstruction_fusion == "concat_scaa":
                x = self.reconstruction_refine(torch.cat((skips[0], x, attention), dim=1))
            elif self.reconstruction_fusion == "scaa_residual_sum":
                x = self.reconstruction_refine(self.reconstruction_local_encoder(skips[0])
                                              + self.reconstruction_local_decoder(x) + attention)
            else:
                x = attention
        return self.smoothing(self.head(x))
