"""
Vanilla UNet — 经典编解码分割基线 | Classic encoder-decoder segmentation baseline.
==================================================================================

Ronneberger et al., MICCAI 2015: "U-Net: Convolutional Networks for Biomedical
Image Segmentation" (https://arxiv.org/abs/1505.04597)

标准实现 (base=64, 4 级下采样, ~31M 参数), 加 BatchNorm (现代通用做法)。
Standard implementation (base=64, 4 down levels, ~31M params) with BatchNorm
(common modern practice). 用作 NEU-Seg 论文的 from-scratch 经典基线。
Used as the from-scratch classic baseline for the NEU-Seg paper.

输入要求 | Input requirement: H, W 需为 16 的倍数 (4 次 2× 下采样)。
H, W must be multiples of 16 (four 2x downsamplings).

Usage::
    >>> from adatile.baselines import UNet
    >>> model = UNet(num_classes=4)
    >>> logits = model(torch.randn(2, 3, 224, 224))   # [2, 4, 224, 224]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """两次 3×3 Conv-BN-ReLU | Two 3x3 Conv-BN-ReLU blocks (UNet 基本单元)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Vanilla UNet | 原版 UNet。

    :param in_channels: 输入通道 | Input channels (default 3).
    :param num_classes: 输出类别数 | Number of output classes.
    :param base: 第一级通道数 | Channels at first level (64 → ~31M params).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, base: int = 64):
        super().__init__()
        b = base
        # ── 编码器 | Encoder ──
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)

        # ── 解码器 (转置卷积上采样 + skip 拼接) | Decoder (deconv + skip concat) ──
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        # ── 分类头 | Classification head ──
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: [B, 3, H, W] (H, W 为 16 倍数). :return: [B, C, H, W] logits."""
        e1 = self.enc1(x)                      # [B, b,  H,   W]
        e2 = self.enc2(self.pool(e1))          # [B, 2b, H/2, W/2]
        e3 = self.enc3(self.pool(e2))          # [B, 4b, H/4, W/4]
        e4 = self.enc4(self.pool(e3))          # [B, 8b, H/8, W/8]
        bn = self.bottleneck(self.pool(e4))    # [B, 16b, H/16, W/16]

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
