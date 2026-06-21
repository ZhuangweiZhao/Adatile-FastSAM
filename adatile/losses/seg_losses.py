"""
分割损失函数 | Segmentation Loss Functions.
=============================================

Focal Loss + Dice Loss + 组合损失。
Extracted from train_b04.py inline definitions (2026-06-21).

用法 | Usage:
    >>> loss_fn = CombinedLoss(num_classes=16, gamma=5.0, ignore_index=255)
    >>> loss = loss_fn(logits, targets)  # logits: [B,C,H,W], targets: [B,H,W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation | 多类分割 Focal Loss.

    FL(pt) = (1 - pt)^γ * CE(pt)
    对低置信度样本（难例）施加更大权重，缓解类别不平衡。

    Args:
        gamma: 聚焦参数 | focusing parameter (default 5.0 for extreme imbalance)
        ignore_index: 忽略标签值 | label value to ignore (default 255)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(self, gamma: float = 5.0, ignore_index: int = 255,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] raw logits (before softmax)
            targets: [B, H, W] int64 class indices

        Returns:
            scalar loss if reduction="mean", else per-pixel loss
        """
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index,
                             reduction="none")
        focal = ((1 - torch.exp(-ce)) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class DiceLoss(nn.Module):
    """
    Dice Loss — 逐前景类 Soft Dice | Per-Foreground-Class Soft Dice Loss.

    对每个前景类 c (1..C-1) 计算 Dice:
        Dice_c = 2 * |P_c ∩ T_c| / (|P_c| + |T_c|)
    最终: loss = 1 - mean(Dice_c over classes where T_c exists)

    忽略 BG (c=0) 和 GT 中不存在的类别。

    Args:
        num_classes: 总类别数 (含 BG) | total classes including background
        smooth: 数值稳定项 | numerical stability term
    """

    def __init__(self, num_classes: int = 16, smooth: float = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] raw logits
            targets: [B, H, W] int64 class indices

        Returns:
            scalar: 1 - mean(Dice over valid foreground classes)
        """
        probs = F.softmax(logits, dim=1)

        dice_sum, valid = 0.0, 0
        for c in range(1, self.num_classes):
            p_c = probs[:, c]          # [B, H, W]
            t_c = (targets == c).float()
            inter = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum() + self.smooth
            if t_c.sum() > 0:          # 只在 GT 中存在时才计入 | only count if GT present
                dice_sum += (2 * inter / union)
                valid += 1

        return 1.0 - (dice_sum / max(valid, 1))


class CombinedLoss(nn.Module):
    """
    Focal + Dice 组合损失 | Combined Focal + Dice Loss.

    Loss = α * FocalLoss + (1-α) * DiceLoss

    Paper B 默认: γ=5.0, α=0.5, num_classes=16, ignore_index=255
    """

    def __init__(self, num_classes: int = 16, gamma: float = 5.0,
                 alpha: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] raw logits
            targets: [B, H, W] int64 class indices

        Returns:
            scalar combined loss
        """
        return (self.alpha * self.focal(logits, targets) +
                (1 - self.alpha) * self.dice(logits, targets))
