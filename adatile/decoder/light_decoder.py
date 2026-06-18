"""
LightDecoder — 轻量解码器 | Lightweight Segmentation Decoder.
===============================================================

P4 特征 → 渐进上采样 + 多层 Conv → 分割掩码。
P4 features → gradual upsampling + multi-layer Conv → segmentation mask.

结构 | Architecture (~800K params):
    P4 [B, 1280, H/16, W/16]
         │
    Conv(1280→64, 3×3) + BN + ReLU
         │
    Upsample 2×  →  H/8
         │
    Conv(64→64, 3×3) + BN + ReLU
         │
    Upsample 2×  →  H/4
         │
    Conv(64→32, 3×3) + BN + ReLU
         │
    Upsample 2×  →  H/2
         │
    Conv(32→32, 3×3) + BN + ReLU
         │
    Upsample → target_size
         │
    Conv(32→1, 1×1)
         │
    Sigmoid → Binary Mask
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

    ~800K 可训练参数 | ~800K trainable parameters.
    """

    def __init__(self, in_channels: int = 1280):
        super().__init__()
        self.logger = get_logger("decoder.light")

        # Stage 1: 压缩特征 | Compress features
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 2: H/16 → H/8
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 3: H/8 → H/4
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stage 4: H/4 → H/2
        self.stage4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 最终投影 | Final projection
        self.head = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        # 参数统计 | Parameter stats
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_info(
            "light_decoder/init",
            f"LightDecoder: params={n_total:,} (trainable={n_trainable:,})",
        )

    def forward(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        Args:
            features:    {"p4": [B, 1280, H/16, W/16]}
            target_size: (H, W) 目标尺寸。None → 返回 stride-2 的 logit。
                        Target size. None → return stride-2 logits.

        Returns:
            logit [B, 1, *, *] — raw logits (before sigmoid) for BCEWithLogitsLoss.
        """
        x = features["p4"]  # [B, 1280, H/16, W/16]

        # Stage 1: compress at stride-16
        x = self.stage1(x)   # [B, 64, H/16, W/16]

        # Stage 2: H/16 → H/8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)   # [B, 64, H/8, W/8]

        # Stage 3: H/8 → H/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)   # [B, 32, H/4, W/4]

        # Stage 4: H/4 → H/2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage4(x)   # [B, 32, H/2, W/2]

        # Final upsample to target
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        # 投影到单通道 logit | Project to single-channel logit
        logit = self.head(x)  # [B, 1, *, *]

        return logit  # raw logit, NO sigmoid

    def predict(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int]) -> torch.Tensor:
        """
        预测二值掩码 | Predict binary mask.

        Args:
            features:    backbone 输出 | Backbone output.
            target_size: (H, W) 目标尺寸 | Target size.

        Returns:
            [B, 1, H, W] binary mask.
        """
        logit = self.forward(features, target_size=target_size)
        return (torch.sigmoid(logit) > 0.5).float()
