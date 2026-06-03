"""Backbone networks for feature extraction.

Supported architectures (via timm):
    - resnet50, resnet101, resnet152
    - convnext_tiny, convnext_small, convnext_base
    - swin_tiny_patch4_window7_224, swin_small/base/large
    - vit_base_patch16_224, vit_large_patch16_224
    - efficientnet_b0 through b7
    - And any other timm model with features_only=True support
"""

from adatile.registry import BACKBONE
from adatile.core import SparseImportancePredictor  # noqa: F401 — used in typing

# Import implementations to trigger @BACKBONE.register() decorators
from adatile.backbone.base import (
    TimmBackbone,
    ResNet50Backbone,
)


def build_backbone(name: str, **kwargs):
    """Factory: instantiate a registered backbone by name.

    Available names:
        - "TimmBackbone" — general-purpose timm backbone (pass model_name="...")
        - "ResNet50Backbone" — ResNet50 with optional FPN (backward-compat)
        - "fastsam_vit_b" — alias for ResNet50Backbone
        - "resnet50" — alias for ResNet50Backbone
    """
    _aliases = {
        "fastsam_vit_b": "ResNet50Backbone",
        "resnet50": "ResNet50Backbone",
    }
    name = _aliases.get(name, name)
    return BACKBONE.build(name, **kwargs)


__all__ = [
    "BACKBONE",
    "build_backbone",
    "TimmBackbone",
    "ResNet50Backbone",
]
