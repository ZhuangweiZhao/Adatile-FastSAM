"""
Dice 系数计算 | Dice coefficient computation.
===============================================

修复 v1 中 unsqueeze(0) 导致的 broadcast 爆炸问题。
Fixes v1 unsqueeze(0) broadcast explosion when batch > 1.

Dice = 2 * |pred ∩ target| / (|pred| + |target|)
"""

from __future__ import annotations

import torch


def compute_dice(
    pred: torch.Tensor,             # [B, C, H, W] or [B, H, W]
    target: torch.Tensor,           # [B, C, H, W] or [B, H, W]
    smooth: float = 1e-6,
    per_class: bool = False,
) -> torch.Tensor:
    """
    计算 Dice 系数 | Compute Dice coefficient.

    Dice = 2 * |pred ∩ target| / (|pred| + |target| + ε)

    V1 教训 | V1 lesson:
        不使用 unsqueeze(0) — 当 batch>1 时会产生 [1,B,H,W] 而不是 [B,1,H,W]，
        导致广播爆炸使 Dice 值远超 1.0。
        Do NOT use unsqueeze(0) — with batch>1 it produces [1,B,H,W]
        instead of [B,1,H,W], causing broadcast explosion with Dice >> 1.0.

        修复方案：使用 sum(dim=...) 直接沿空间维度求和，
        避免 reshape/unsqueeze 陷阱。
        Fix: use sum(dim=...) directly along spatial dims,
        avoiding reshape/unsqueeze pitfalls.

    :param pred: 预测张量 [B, C, H, W] 或 [B, H, W] | Prediction tensor.
    :type pred: torch.Tensor

    :param target: 目标张量 [B, C, H, W] 或 [B, H, W] | Target tensor.
    :type target: torch.Tensor

    :param smooth: 平滑项防止除零 | Smoothing term to prevent division by zero.
    :type smooth: float

    :param per_class: 如果 True，返回每类的 Dice [C]；否则返回均值标量。 If True, return per-class Dice [C]; else return mean scalar.
    :type per_class: bool

    :return: Scalar tensor (mean Dice) or [C] tensor (per-class Dice).
    :rtype: torch.Tensor
    """
    # ── 输入标准化 | Input normalization ──
    # 确保是 float 类型 | Ensure float type
    pred = pred.float()
    target = target.float()

    # ── 处理 {0, 255} 掩码（V1 教训 | V1 lesson）──
    target = _normalize_binary_mask(target)

    if pred.dim() == 3:
        # [B, H, W] → [B, 1, H, W]
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # ── 计算 Dice（修复 v1 broadcast bug）──
    # 直接用 sum(dim=(2,3)) 计算每个样本的空间交集和并集
    # Use sum(dim=(2,3)) to compute per-sample spatial intersection and union
    # dims: [B, C, H, W] → sum over H,W → [B, C]

    intersection = (pred * target).sum(dim=(2, 3))  # [B, C]
    pred_sum = pred.sum(dim=(2, 3))                  # [B, C]
    target_sum = target.sum(dim=(2, 3))               # [B, C]

    # Dice per sample, per class: [B, C]
    dice_per_sample = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    if per_class:
        # 对所有 batch 取平均 → [C]
        # Average over batch → [C]
        return dice_per_sample.mean(dim=0)
    else:
        # 全局平均 → scalar
        # Global average → scalar
        return dice_per_sample.mean()


def _normalize_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    标准化二值掩码 | Normalize binary mask.

    V1 教训 | V1 lesson:
        {0, 255} 掩码需要检测唯一值 ≤ 2，而非检查 max_val ≤ 1。
        {0, 255} masks need n_unique ≤ 2 check, not max_val ≤ 1.
    """
    unique_vals = mask.unique()
    if len(unique_vals) <= 2 and unique_vals.max() > 1:
        # {0, 255} → {0, 1} 通过阈值 | via threshold
        mask = (mask > 128).float()
    return mask
