"""Lightweight Ada-SPM for Stage B.

Takes P8 backbone features → predicts importance map.
Minimal design: 3 conv layers + sigmoid. No FPN, no transformer.

Decoupled from decoder — importance trained via GT-driven losses only.
Decoder always receives full features (unchanged from Stage A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightSPM(nn.Module):
    """Lightweight Sparse Perception Module.

    Input: P8 features [B, C, H/8, W/8]
    Output: importance [B, 1, H/32, W/32]

    Args:
        in_channels: Feature channels from backbone.
        hidden_channels: Internal conv channels.
        importance_stride: Downsample factor for importance grid (default 32).
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        importance_stride: int = 32,
    ):
        super().__init__()
        self.stride = importance_stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

        # CAT: PromptBridge bias (set externally)
        self._spm_bias: Optional[torch.Tensor] = None

    def set_spm_bias(self, bias: Optional[torch.Tensor]):
        self._spm_bias = bias

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: {"P8": [B, C, H/8, W/8]}  (and optionally "P4")

        Returns:
            importance: [B, 1, H/32, W/32]  (downsampled from H/8)
        """
        x = features["P8"]  # [B, C, H/8, W/8]

        # Apply PromptBridge bias (CAT-style)
        if self._spm_bias is not None:
            bias = self._spm_bias
            if bias.shape[-2:] != x.shape[-2:]:
                bias = F.interpolate(bias, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if bias.shape[1] != x.shape[1]:
                if bias.shape[1] > x.shape[1]:
                    bias = bias[:, :x.shape[1]]
                else:
                    bias = F.pad(bias, (0, 0, 0, 0, 0, x.shape[1] - bias.shape[1]))
            x = x + bias

        # Conv → sigmoid
        imp = self.conv(x)  # [B, 1, H/8, W/8]

        # Downsample to target stride
        target_h = max(x.shape[2] // (self.stride // 8), 1)
        target_w = max(x.shape[3] // (self.stride // 8), 1)
        if imp.shape[-2:] != (target_h, target_w):
            imp = F.interpolate(imp, size=(target_h, target_w), mode="area")

        return imp  # [B, 1, H/32, W/32]
