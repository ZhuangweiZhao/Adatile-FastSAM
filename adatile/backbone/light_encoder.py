"""Lightweight pre-SPM encoder — ~0.1M params, stride=8.

Sits BEFORE SPM in the pipeline:
    Image [B,3,H,W] → LightEncoder → [B,C,H/8,W/8] → SPM → importance

Why needed: SPM at H/32 has cells covering 32×32 original pixels.
Small objects (15×15 planes in iSAID) would be invisible. A tiny
stride-8 encoder provides enough spatial detail for SPM to make
informed decisions.

Architecture:
    Conv3x3(s=2) → Conv3x3 → Conv3x3 → [B, 64, H/8, W/8]
"""

import torch
import torch.nn as nn


class LightEncoder(nn.Module):
    """Tiny feature extractor for pre-SPM encoding.

    Input:  [B, 3, H, W] image in [0,1]
    Output: [B, 64, H/8, W/8] features
    Params: ~0.12M
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        # stride-2: H→H/2, stride-4: H/2→H/4, stride-8: H/4→H/8
        self.stem = nn.Sequential(
            # H → H/2, 3→32
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # H/2 → H/4, 32→64
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # H/4 → H/8, 64→out_channels
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Lightweight refinement (no downsampling)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features at stride 8.

        Args:
            image: [B, 3, H, W] in [0, 1].

        Returns:
            [B, out_channels, H/8, W/8]
        """
        x = self.stem(image)
        x = self.refine(x)
        return x

    @property
    def output_stride(self) -> int:
        return 8
