"""Backbone networks.

Active (Stage A/B/C):
    - FastSAMHookBackbone: hook-based FastSAM feature extractor (P4/P8)

Legacy (train.py compatibility):
    - FastSAMBackbone: YOLOv8 API-based (superseded by hook version)
    - TimmBackbone, ResNet50Backbone: timm-based backbones
"""

from adatile.registry import BACKBONE

from adatile.backbone.base import TimmBackbone, ResNet50Backbone
from adatile.backbone.fastsam_hook import FastSAMHookBackbone
from adatile.backbone.fpn import MultiScaleFPNFusion, LightweightFPNFusion

# Legacy alias (moved to legacy/)
FastSAMBackbone = None


def build_backbone(name: str, **kwargs):
    """Factory: instantiate a registered backbone by name."""
    _aliases = {
        "fastsam": "FastSAMBackbone",
        "fastsam_vit_b": "ResNet50Backbone",
        "resnet50": "ResNet50Backbone",
        "fastsam_hook": "FastSAMHookBackbone",
    }
    name = _aliases.get(name, name)
    return BACKBONE.build(name, **kwargs)


__all__ = [
    "FastSAMHookBackbone",
    "FastSAMBackbone",
    "TimmBackbone",
    "ResNet50Backbone",
    "MultiScaleFPNFusion",
    "LightweightFPNFusion",
    "build_backbone",
]
