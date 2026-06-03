"""Prototype memory implementations.

Masked Average Pooling:
    Standard few-shot prototype: mask-weighted average of support features.

Learnable Prototype Aggregation:
    Attention-based prototype refinement across multiple support examples.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import PrototypeMemory
from adatile.registry import PROTOTYPE


@PROTOTYPE.register()
class MaskedAveragePrototype(PrototypeMemory):
    """Classic masked average pooling (MAP) prototype.

    Computes prototype = mean(feature * mask) / mean(mask) for each class.

    Extensions (PANet/HSNet-style):
        - Multiple prototypes per class via K-means on support features
        - Online prototype update during inference (momentum-based)
    """

    def __init__(self, prototype_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.temperature = temperature

    def forward(
        self,
        support_features: Tensor,
        support_masks: Tensor,
        class_ids: Optional[List[int]] = None,
    ) -> Dict[int, Tensor]:
        # Masked average pooling
        masks = F.interpolate(
            support_masks.unsqueeze(1).float(),
            size=support_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, H, W]
        masked_features = support_features * masks  # [B, C, H, W]
        prototypes = masked_features.sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-8)
        if class_ids is None:
            class_ids = list(range(len(prototypes)))
        return {cid: prototypes[i] for i, cid in enumerate(class_ids)}

    def retrieve(
        self,
        query_features: Tensor,
        prototypes: Dict[int, Tensor],
        temperature: Optional[float] = None,
    ) -> Tensor:
        temp = temperature or self.temperature
        B, C, H, W = query_features.shape
        proto_list = [p for _, p in sorted(prototypes.items())]
        proto_stack = torch.stack(proto_list)  # [N_cls, C]

        similarity = F.cosine_similarity(
            query_features.unsqueeze(1),  # [B, 1, C, H, W]
            proto_stack.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),  # [1, N_cls, C, 1, 1]
            dim=2,
        )  # [B, N_cls, H, W]
        return similarity / temp

    def update(
        self,
        prototypes: Dict[int, Tensor],
        cache_path: Optional[str] = None,
    ) -> None:
        if cache_path is not None:
            state = {str(k): v.cpu() for k, v in prototypes.items()}
            torch.save(state, cache_path)
