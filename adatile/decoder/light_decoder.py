"""Lightweight Decoder for Stage A — binary semantic segmentation.

Simple U-Net-like decoder: upsample + skip connection + conv blocks.
Output: [B, 1, H/4, W/4] binary mask logits.

Reference: CS-FastSAM StructureDecoder (simplified).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Conv3x3 → BN → ReLU × 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightDecoder(nn.Module):
    """Simple decoder for binary segmentation.

    Architecture:
        P4 [B, 128, H/16, W/16] ──→ upsample ──→ conv ──→ mask [B, 1, H/4, W/4]
        P8 [B, 128, H/8, W/8]  ──→ skip connection ──┘

    Args:
        in_channels: Feature channels from backbone (default 128).
        decoder_channels: Internal decoder channels.
        num_classes: Output channels (1 for binary, K for multi-class).
    """

    def __init__(
        self,
        in_channels: int = 128,
        decoder_channels: int = 64,
        num_classes: int = 1,
    ):
        super().__init__()

        # P4 (low-res, high-semantic) → upsample path
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, decoder_channels, 2, stride=2),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # P8 skip projection
        self.skip_proj = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # Skip fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder conv blocks
        self.dec_conv = ConvBlock(decoder_channels, decoder_channels)

        # Upsample to H/4
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.final_conv = ConvBlock(decoder_channels, decoder_channels // 2)

        # Output head
        self.head = nn.Conv2d(decoder_channels // 2, num_classes, 1)

    def forward(self, features: dict, proto_guide=None) -> torch.Tensor:
        """proto_guide ignored — here for API compatibility with ProtoGuidedDecoder."""
        return self._forward(features)

    def _forward(self, features: dict) -> torch.Tensor:
        p4 = features["P4"]  # low-res
        p8 = features["P8"]  # mid-res

        # Upsample P4 → match P8 spatial size
        up = self.up_conv(p4)  # [B, dec_c, H/8, W/8]

        # Skip from P8
        skip = self.skip_proj(p8)  # [B, dec_c, H/8, W/8]

        # Gated fusion
        fused = self.fusion_gate(torch.cat([up, skip], dim=1))

        # Conv refine
        refined = self.dec_conv(fused)

        # Upsample to H/4
        out = self.final_up(refined)
        out = self.final_conv(out)

        # Output mask logits
        return self.head(out)  # [B, num_classes, H/4, W/4]
