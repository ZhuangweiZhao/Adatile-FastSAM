"""Prototype Feature Adapter — class-conditioned feature modulation.

ProtoNet/CAT-SAM style: prototype interacts with query features BEFORE
the detection head, enhancing class-discriminative power.

Architecture:
    query_features [N, C, H, W] + class_prototypes {cls_id: [C]}
        ↓
    Per-class cosine similarity maps
        ↓
    Channel attention: prototype → per-channel gate
        ↓
    Enhanced features [N, C, H, W] → YOLOv8 Head

Why before head (not after): Few-shot most lacks category discrimination,
not mask quality. Modulating features before cls+bbox heads fixes this.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PrototypeAdapter(nn.Module):
    """Class-conditioned feature enhancement for few-shot detection.

    Two modes:
      1. Standard (prototypes=None):  identity pass-through
      2. Few-shot (prototypes given): prototype × query → enhanced features

    Args:
        feature_dim: Feature channel dimension.
        proto_dim: Prototype embedding dimension.
        num_prototypes: Number of prototypes per class.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        proto_dim: int = 256,
        num_prototypes: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes

        # Prototype → channel gate
        self.proto_project = nn.Sequential(
            nn.Linear(proto_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid(),
        )

        # Feature refinement after prototype modulation
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Zero-init the last Linear (index -2, before Sigmoid) for identity-like start
        nn.init.zeros_(self.proto_project[-2].weight)
        nn.init.zeros_(self.proto_project[-2].bias)

    def forward(
        self,
        features: Tensor,
        prototypes: Optional[Dict[int, Tensor]] = None,
    ) -> Tensor:
        """Enhance features with class prototype information.

        Args:
            features: [B, C, H, W] query features.
            prototypes: Optional {class_id: [proto_dim]} dict.
                        If None, returns features unchanged.

        Returns:
            [B, C, H, W] enhanced features.
        """
        if prototypes is None or len(prototypes) == 0:
            return features

        B, C, H, W = features.shape

        # ── Global prototype pooling ────────────────────────
        # Average all class prototypes → single representative embedding
        proto_list = list(prototypes.values())
        proto_stack = torch.stack(proto_list, dim=0)  # [N_cls, proto_dim]
        proto_global = proto_stack.mean(dim=0)  # [proto_dim]

        # ── Channel gate from prototype ─────────────────────
        gate = self.proto_project(proto_global)  # [C]
        gate = gate.view(1, C, 1, 1).expand(B, C, H, W)

        # ── Modulate + residual ─────────────────────────────
        features = features * gate + features
        features = self.refine(features)

        return features

    def build_prototypes(
        self,
        support_features: Tensor,
        support_masks: Tensor,
        class_ids: List[int],
    ) -> Dict[int, Tensor]:
        """Build class prototypes from support set.

        Args:
            support_features: [S, C, H, W] support image features.
            support_masks: [S, H, W] binary instance masks.
            class_ids: [S] class labels.

        Returns:
            {class_id: [proto_dim]} dict.
        """
        prototypes: Dict[int, List[Tensor]] = {}

        for i, cls_id in enumerate(class_ids):
            feat = support_features[i]   # [C, H, W]
            mask = support_masks[i]      # [H, W]

            # Resize mask to match feature spatial size
            if mask.shape != feat.shape[-2:]:
                mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0),
                    size=feat.shape[-2:], mode="nearest",
                ).squeeze(0).squeeze(0)

            # Masked average pooling → prototype
            masked_feat = feat * mask.unsqueeze(0)
            proto = masked_feat.sum(dim=(1, 2)) / (mask.sum() + 1e-8)  # [C]

            if cls_id not in prototypes:
                prototypes[cls_id] = []
            prototypes[cls_id].append(proto)

        # Average per-class
        return {
            cls_id: torch.stack(protos).mean(dim=0)
            for cls_id, protos in prototypes.items()
        }
