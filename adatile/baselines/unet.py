"""
Original UNet — 严格符合 Ronneberger et al., MICCAI 2015.
================================================================

Ronneberger et al., MICCAI 2015: "U-Net: Convolutional Networks for Biomedical
Image Segmentation" (https://arxiv.org/abs/1505.04597)

严格遵循原论文架构 | Strictly follows the original architecture:
  - Conv3x3 -> ReLU -> Conv3x3 -> ReLU (no BatchNorm)
  - 2x2 MaxPool (encoder) / 2x2 TransposedConv (decoder)
  - Skip connections via concatenation (channel dim)
  - 1x1 Conv classification head
  - bias=True (original paper uses bias)
  - padding=1 (same conv) -- sole modern adaptation

与 2015 原论文的唯一偏差 | Single deviation from 2015:
  padding=1 (same convolution) instead of padding=0 (valid convolution).
  Reason: All compared baseline methods (DA-FRN variants) use same-padding,
  producing same-size output. Valid convolution would shrink output by ~184px,
  requiring GT cropping or prediction padding, introducing unfair evaluation
  bias. This deviation is explicitly declared in the paper.

Channels: 64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64
Params: ~31.0M (base=64, no BN, bias=True)
Input: H, W must be multiples of 16 (four 2x downsamplings).

Usage::
    >>> from adatile.baselines import UNet
    >>> model = UNet(num_classes=4)
    >>> logits = model(torch.randn(2, 3, 224, 224))   # [2, 4, 224, 224]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 Conv-ReLU | UNet basic building block (no BatchNorm).

    Strictly per Ronneberger et al. 2015: Conv3x3 -> ReLU -> Conv3x3 -> ReLU.
    No BatchNorm, bias=True (required when no BN), same padding.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Original paper uses bias (required without BatchNorm)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    """
    Original U-Net | Ronneberger et al., MICCAI 2015.

    Strictly follows the original architecture (except padding=1, documented above).

    :param in_channels: Input channels (default 3).
    :param num_classes: Number of output classes.
    :param base: Channels at first level (64 -> ~31.0M params).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, base: int = 64):
        super().__init__()
        b = base
        # -- Encoder (Contracting Path) --
        # Original: Conv3x3-ReLU x2 -> 2x2 MaxPool, double channels
        self.enc1 = DoubleConv(in_channels, b)          # 64
        self.enc2 = DoubleConv(b, b * 2)                # 128
        self.enc3 = DoubleConv(b * 2, b * 4)            # 256
        self.enc4 = DoubleConv(b * 4, b * 8)            # 512
        self.bottleneck = DoubleConv(b * 8, b * 16)     # 1024
        self.pool = nn.MaxPool2d(2)

        # -- Decoder (Expansive Path) --
        # Original: 2x2 Up-Conv -> concat skip -> Conv3x3-ReLU x2, halve channels
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)           # 1024->512
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)            # 512->256
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)            # 256->128
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)                # 128->64

        # -- Classification Head --
        # Original: 1x1 Conv maps 64-channel features to num_classes
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: [B, 3, H, W] -- H, W must be multiples of 16.
        :return: [B, C, H, W] raw logits (no softmax).
        """
        # -- Contracting Path (Encoder) --
        e1 = self.enc1(x)                      # [B, b,    H,   W]
        e2 = self.enc2(self.pool(e1))          # [B, 2b,   H/2, W/2]
        e3 = self.enc3(self.pool(e2))          # [B, 4b,   H/4, W/4]
        e4 = self.enc4(self.pool(e3))          # [B, 8b,   H/8, W/8]
        bt = self.bottleneck(self.pool(e4))    # [B, 16b,  H/16, W/16]

        # -- Expansive Path (Decoder) --
        # Up-conv -> concatenate skip (channel dim) -> DoubleConv
        d4 = self.dec4(torch.cat([self.up4(bt), e4], dim=1))   # [B, 8b,  H/8,  W/8]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # [B, 4b,  H/4,  W/4]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # [B, 2b,  H/2,  W/2]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # [B, b,   H,    W]

        return self.head(d1)  # [B, C, H, W] logits
