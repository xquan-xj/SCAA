"""Explicit-parameter MLP with a full-field output in normalized target units."""

import math

from torch import nn

from .common import OutputSmoothing, positive_int


class ParameterMLP(nn.Module):
    def __init__(self, output_size, controls, hidden=(128, 128), parameter_dim=2):
        super().__init__()
        if not isinstance(hidden, (list, tuple)) or not hidden:
            raise ValueError("hidden must be a nonempty sequence of positive widths")
        hidden = [positive_int(value, "hidden width") for value in hidden]
        self.parameter_dim = positive_int(parameter_dim, "parameter_dim")
        widths = [self.parameter_dim] + hidden + [math.prod(output_size)]
        layers = []
        for index, (before, after) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_size = tuple(output_size)
        self.smoothing = OutputSmoothing(controls["smoothing"], controls["padding"],
                                         controls["smoothing_kernels"])

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.parameter_dim:
            raise ValueError(f"parameter MLP input shape must be B x {self.parameter_dim}")
        return self.smoothing(self.layers(x).reshape(x.shape[0], 1, *self.output_size))
