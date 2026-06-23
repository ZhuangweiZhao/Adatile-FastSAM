"""
LinearProbe — 线性探针解码器 | 1×1 Conv Linear Probe Decoder.
===============================================================

最简化解码器：单个 1×1 卷积 + 上采样 + Sigmoid。
Minimal decoder: single 1×1 Conv + Upsample + Sigmoid.

目的：验证 FastSAM 冻结特征是否已经包含足够的分割信息。
Purpose: verify whether frozen FastSAM features already contain sufficient segmentation info.

结构 | Architecture:
    P4 [B, C, H/16, W/16]  (C=1280 for FastSAM-x)
     │
    Conv2d(C, 1, kernel_size=1)  ← 仅 C+1 个可训练参数 | only C+1 trainable params
     │
    Bilinear Upsample → [B, 1, H, W]
     │
    Sigmoid → [B, 1, H, W] ∈ [0, 1]

如果 Dice > 0.6 → FastSAM P4 特征已经包含分割信息，缺的只是映射。
如果 Dice ≈ 0.12（不变）→ P4 本身不适合建筑分割，需要更高层特征或融合。

If Dice > 0.6 → P4 features already encode segmentation info, only mapping needed.
If Dice ≈ 0.12 (unchanged) → P4 alone insufficient, need higher-level features or fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger


class LinearProbe(nn.Module):
    """
    线性探针解码器 | Linear Probe Decoder.

    极简参数量的分割头，用于验证 backbone 特征质量。
    Ultra-low-parameter segmentation head for verifying backbone feature quality.

    ----------
    in_channels : int
        输入通道数（P4 的通道数）| Input channels (P4 channels).
        FastSAM-x: 1280, FastSAM-s: 640.
    """

    def __init__(self, in_channels: int = 1280) -> None:
        super().__init__()
        self.logger = get_logger("decoder.linear_probe")

        # 1×1 卷积：通道映射 → 单通道 logit | 1×1 Conv: channel → single logit
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        # 参数统计 | Parameter stats
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_info(
            "linear_probe/init",
            f"LinearProbe: in={in_channels}, "
            f"params={n_params} (trainable={n_trainable})",
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param features: backbone 输出 {"p4": [B, C, H/16, W/16], "p8": ...} Backbone output.
        :type features: dict[str, torch.Tensor]

        :return: torch.Tensor [B, 1, H_orig, W_orig] 二值预测 | Binary prediction.
        :rtype: torch.Tensor
        """
        x = features["p4"]  # [B, C, H/16, W/16]

        # 1×1 卷积 → logit | 1×1 Conv → logit
        logit = self.conv(x)  # [B, 1, H/16, W/16]

        # Sigmoid → 概率 | Sigmoid → probability
        prob = torch.sigmoid(logit)  # [B, 1, H/16, W/16]

        return prob

    def predict(
        self, features: dict[str, torch.Tensor], target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        预测并上采样到目标尺寸 | Predict and upsample to target size.

        :param features: backbone 输出 | Backbone output.
        :type features: dict[str, torch.Tensor]

        :param target_size: (H, W) 目标尺寸 | Target spatial size.
        :type target_size: tuple[int, int]

        :return: torch.Tensor [B, 1, H, W] 二值 mask (0/1) | Binary mask.
        :rtype: torch.Tensor
        """
        prob = self.forward(features)  # [B, 1, H_feat, W_feat]

        # 上采样 | Upsample
        prob_up = F.interpolate(
            prob, size=target_size, mode="bilinear", align_corners=False,
        )

        # 二值化 | Binarize
        binary = (prob_up > 0.5).float()

        return binary
