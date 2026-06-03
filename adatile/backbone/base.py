"""Backbone implementations backed by timm and torchvision.

Uses timm for feature extraction (multi-scale) and torchvision for
optional FPN neck. Replaces custom FPNNeck and ResNet wrapper.

Architecture:
    - TimmBackbone: timm.create_model(name, features_only=True) → multi-scale features
    - ResNet50Backbone: backward-compat alias using TimmBackbone("resnet50") + optional FPN

Feature map convention (timm features_only):
    Stride 4, 8, 16, 32 → {"p2", "p3", "p4", "p5"}
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adatile.registry import BACKBONE


# ── Timm Backbone ──────────────────────────────────────────────────────


@BACKBONE.register()
class TimmBackbone(nn.Module):
    """Multi-scale backbone backed by timm.

    Uses timm.create_model(name, features_only=True, out_indices=...) to
    extract features at multiple strides. Returns a dict keyed by "p2".."p5".

    Args:
        model_name: timm model name (e.g. "resnet50", "convnext_tiny",
                    "swin_tiny_patch4_window7_224").
        pretrained: Load pretrained weights.
        out_indices: Which feature levels to return (0-indexed).
                     Default (1,2,3,4) for resnet-style 4-stage models.
        add_fpn: If True, attach a torchvision FeaturePyramidNetwork to unify
                 channel dimensions. If False (default), return raw features;
                 Ada-SPM's internal FPN fusion handles channel unification.
        **kwargs: Passed to timm.create_model (e.g., features_only=True).
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        add_fpn: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.add_fpn = add_fpn

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for TimmBackbone. Install with: pip install timm"
            )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **kwargs,
        )

        self._feature_channels = self.backbone.feature_info.channels()
        self._feature_strides = self.backbone.feature_info.reduction()

        if add_fpn:
            from torchvision.ops import FeaturePyramidNetwork
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=self._feature_channels,
                out_channels=256,
            )
        else:
            self.fpn = None

    @property
    def stage_channels(self) -> Dict[str, int]:
        stride_to_key = {4: "p2", 8: "p3", 16: "p4", 32: "p5"}
        return {
            stride_to_key.get(s, f"p{int(s**0.5)}"): c
            for c, s in zip(self._feature_channels, self._feature_strides)
        }

    @property
    def out_channels(self) -> int:
        return 256 if self.add_fpn else self._feature_channels[0]

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feats = self.backbone(x)

        if self.fpn is not None:
            od = OrderedDict((str(i), f) for i, f in enumerate(feats))
            od = self.fpn(od)
            key_map = {str(i): f"p{i + 2}" for i in range(len(feats))}
            return {key_map[k]: v for k, v in od.items()}

        return {
            f"p{i + 2}": feats[i]
            for i in range(len(feats))
        }


# ── ResNet50 Backbone (backward-compat) ─────────────────────────────────


@BACKBONE.register()
class ResNet50Backbone(TimmBackbone):
    """ResNet50 backbone with optional FPN (backward-compat wrapper).

    Delegates to TimmBackbone("resnet50"). Accepts old-style config params
    (freeze_stages, out_channels, output_keys) for backward compatibility.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: Optional[List[int]] = None,
        out_channels: int = 256,
        output_keys: Optional[Tuple[str, ...]] = None,
        fpn: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name="resnet50",
            pretrained=pretrained,
            add_fpn=fpn,
        )
        self._output_keys = output_keys or (
            ("p2", "p3", "p4", "p5") if fpn else ("p2", "p3", "p4", "p5")
        )

        if freeze_stages:
            stages = [
                self.backbone.conv1, self.backbone.bn1,
                self.backbone.act1, self.backbone.maxpool,
                self.backbone.layer1, self.backbone.layer2,
                self.backbone.layer3, self.backbone.layer4,
            ]
            for idx in freeze_stages:
                if idx < len(stages):
                    stage = stages[idx]
                    if isinstance(stage, nn.Module):
                        for p in stage.parameters():
                            p.requires_grad = False

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = super().forward(x)
        return {k: features[k] for k in self._output_keys if k in features}
