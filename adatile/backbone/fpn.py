"""FPN-style multi-scale feature fusion for Ada-SPM.

Self-contained FPN implementation — no external dependency.
Lazy-built on first forward pass to detect actual input channels.

Architecture (standard FPN):
    p5 (H/32) ← lateral_conv5(c5)
        ↓ upsample
    p4 (H/16) ← lateral_conv4(c4) + upsample(p5)
        ↓ upsample
    p3 (H/8)  ← lateral_conv3(c3) + upsample(p4)
        ↓ upsample
    p2 (H/4)  ← lateral_conv2(c2) + upsample(p3) → fused output
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiScaleFPNFusion(nn.Module):
    """FPN multi-scale fusion — self-contained."""

    def __init__(
        self,
        in_channels_list: Optional[List[int]] = None,
        out_dim: int = 256,
        use_bifpn: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self._in_channels_list = in_channels_list
        self._lateral = nn.ModuleList()
        self._smooth = nn.ModuleList()
        self._built = False
        self._sorted_keys: List[str] = []
        self._num_levels = 0

    def _ensure_built(self, features: Dict[str, Tensor]) -> None:
        """Lazy-build lateral/smooth convs on first forward pass.

        Always creates weights in fp32 regardless of input dtype
        (AMP autocast may produce fp16 features, but fp16 conv weights
         are numerically unstable and cause NaN).
        """
        if self._built:
            return

        # Sort keys by spatial resolution — finest first
        sorted_keys = sorted(
            features.keys(),
            key=lambda k: features[k].shape[-1],
            reverse=True,
        )
        self._sorted_keys = sorted_keys
        self._num_levels = len(sorted_keys)

        if self._in_channels_list is not None:
            in_channels = list(self._in_channels_list)
        else:
            in_channels = [features[k].shape[1] for k in sorted_keys]

        # Always use fp32 for weights — AMP will handle compute precision
        device = features[sorted_keys[0]].device

        for ch in in_channels:
            self._lateral.append(
                nn.Conv2d(ch, self.out_dim, 1, bias=False,
                          dtype=torch.float32, device=device)
            )
        for _ in range(self._num_levels):
            self._smooth.append(
                nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1, bias=False,
                          dtype=torch.float32, device=device)
            )

        self._built = True

    def forward(
        self, features: Dict[str, Tensor]
    ) -> Tuple[Tensor, List[Tensor]]:
        """Fuse multi-scale features via top-down FPN."""
        self._ensure_built(features)

        # Get features in consistent order (finest→coarsest)
        feats = [features[k] for k in self._sorted_keys]

        # Lateral 1×1 convolutions
        laterals = [conv(f) for conv, f in zip(self._lateral, feats)]

        # Top-down pathway (coarsest → finest)
        pyramid = []
        prev = None
        for i in range(self._num_levels - 1, -1, -1):
            if prev is None:
                p = laterals[i]
            else:
                up = F.interpolate(prev, size=laterals[i].shape[-2:],
                                   mode='bilinear', align_corners=False)
                p = laterals[i] + up
            p = F.relu(self._smooth[i](p))
            pyramid.insert(0, p)
            prev = p

        fused = pyramid[0]  # finest scale
        return fused, pyramid


class LightweightFPNFusion(MultiScaleFPNFusion):
    """Reduced-channel variant for latency-critical applications."""

    def __init__(
        self,
        in_channels_list: Optional[List[int]] = None,
        out_dim: int = 128,
    ):
        super().__init__(
            in_channels_list=in_channels_list,
            out_dim=out_dim,
            use_bifpn=False,
        )
