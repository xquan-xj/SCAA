"""Fresh neural surrogates predicting train-normalized morphology fields.

build_model(config, input_channels=4, output_size=(224,224), parameter_dim=2) supports kinds
unet, hybrid_transunet and parameter_mlp. Common controls: padding='zeros' or
'reflect', fusion='concat' or 'scaa', axes='both'/'radial'/'circumferential'/'global',
cross_attention=True (Q encoder,K/V decoder), gated=True, content_attention=True,
attention_placement='skip'/'reconstruction'/'dual', smoothing=False, smoothing_kernels=[7,5,3],
pool_factor=4, attention_heads=4, key_dim=8. U-Net width=32/depth=4; MLP
hidden=[128,128]. The legacy defaults support FP tests; training/evaluation
explicitly pass six field channels and parameter_dim=4 for main FTPS metadata.
Hybrid defaults and compact
overrides are documented in hybrid.py. Field models preserve input H,W;
output_size fixes the MLP output and initializes the hybrid position grid.

Outputs are Bx1xHxW, unrestricted in normalized target units. Inverse target
normalization belongs to the shared training/evaluation layer. Models neither
fit normalization nor load weights at construction. PyTorch's standard strict
state loading remains enabled by default. Concat/SCAA parameter counts differ;
they share all encoder, pooling, boundary and optional smoothing settings.
"""

from .attention import SCAA, scaled_attention
from .common import output_shape, positive_int, validate_controls
from .hybrid import HybridTransUNet
from .parameter import ParameterMLP
from .unet import UNet


def build_model(config, input_channels=4, output_size=(224, 224), parameter_dim=2):
    if not isinstance(config, dict):
        raise ValueError("model config must be a dictionary")
    kind = config.get("kind", "unet")
    if kind not in ("unet", "hybrid_transunet", "parameter_mlp"):
        raise ValueError(f"unknown model kind: {kind}")
    controls = validate_controls(config)
    output_size = output_shape(output_size)
    input_channels = positive_int(input_channels, "input_channels")
    if kind == "parameter_mlp":
        return ParameterMLP(output_size, controls, config.get("hidden", (128, 128)), parameter_dim)
    if kind == "hybrid_transunet":
        return HybridTransUNet(input_channels, output_size, controls, config)
    width = positive_int(config.get("width", 32), "width")
    depth = positive_int(config.get("depth", 4), "depth")
    return UNet(input_channels, width, depth, controls)


__all__ = ["build_model", "scaled_attention", "SCAA", "UNet", "HybridTransUNet", "ParameterMLP"]
