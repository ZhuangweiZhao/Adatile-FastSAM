"""
LightDecoder — 轻量解码器 | Lightweight Segmentation Decoder.
===============================================================

P4 特征 → 渐进上采样 + 多层 Conv → 分割掩码。
P4 features → gradual upsampling + multi-layer Conv → segmentation mask.

支持两种模式 | Two modes:
    Binary (num_classes=1):  1280→64→64→32→32→1, ~716K params
    Multi-class (num_classes>1): 1280→256→128→64→32→C, ~716K params

用法 | Usage::
    # Binary segmentation (Paper A, MassBuildings)
    decoder = LightDecoder(in_channels=1280, num_classes=1)

    # Multi-class (Paper B, iSAID 15-class)
    decoder = LightDecoder(in_channels=1280, num_classes=16)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger


class LightDecoder(nn.Module):
    """
    轻量解码器 | Lightweight Decoder.

    从 P4 特征（stride-16）渐进上采样到原始分辨率。
    Gradually upsamples P4 features (stride-16) to original resolution.

    num_classes=1:  binary mode, ~716K params
    num_classes>1:  multi-class mode, ~716K params (wider early layers)
    """

    def __init__(self, in_channels: int = 1280, num_classes: int = 1):
        super().__init__()
        self.logger = get_logger("decoder.light")
        self.num_classes = num_classes
        self._is_binary = (num_classes == 1)

        if self._is_binary:
            # Binary mode: thinner, deeper (Paper A) | 二值模式
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        else:
            # Multi-class mode: wider Stage1, fewer upsamples (Paper B) | 多类模式
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage4 = None  # multi-class mode has no stage4
            self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

        # 参数统计 | Parameter stats
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_info(
            "light_decoder/init",
            f"LightDecoder: mode={'binary' if self._is_binary else 'multi'}, "
            f"num_classes={num_classes}, "
            f"params={n_total:,} (trainable={n_trainable:,})",
        )

    def forward(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param features: {"p4": [B, 1280, H/16, W/16]}
        :type features: dict[str, torch.Tensor]

        :param target_size: (H, W) 目标尺寸。None → 返回 stride-2 (binary) 或 stride-4 (multi) 的 logit。 Target size. None → return logits at output stride.
        :type target_size: tuple[int, int] | None

        :return: logit [B, C, *, *] — raw logits (before sigmoid/softmax). Binary mode: C=1, for BCEWithLogitsLoss. Multi-class mode: C=num_classes, for CrossEntropyLoss.
        :rtype: torch.Tensor
        """
        x = features["p4"]  # [B, in_channels, H/16, W/16]

        # Stage 1: compress at stride-16 | 压缩特征
        x = self.stage1(x)  # [B, ch, H/16, W/16]

        # Stage 2: H/16 → H/8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)  # [B, ch, H/8, W/8]

        # Stage 3: H/8 → H/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)  # [B, ch, H/4, W/4]

        # Stage 4 (binary only): H/4 → H/2
        if self.stage4 is not None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.stage4(x)  # [B, 32, H/2, W/2]

        # Final upsample to target
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        # Head: 投影到输出通道 | Project to output channels
        logit = self.head(x)  # [B, C, *, *]

        return logit  # raw logit, NO activation

    def predict(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int]) -> torch.Tensor:
        """
        预测分割掩码 | Predict segmentation mask.

        :param features: backbone 输出 | Backbone output.
        :type features: dict[str, torch.Tensor]

        :param target_size: (H, W) 目标尺寸 | Target size.
        :type target_size: tuple[int, int]

        :return: Binary mode: [B, 1, H, W] float mask. Multi-class mode: [B, H, W] int64 class indices.
        :rtype: torch.Tensor
        """
        logit = self.forward(features, target_size=target_size)
        if self._is_binary:
            return (torch.sigmoid(logit) > 0.5).float()
        else:
            return logit.argmax(dim=1)
