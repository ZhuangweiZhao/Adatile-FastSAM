"""
Hungarian 匹配器 — 预测掩码与 GT 实例的一对一最优匹配.
Hungarian Matcher — optimal one-to-one matching between predicted masks and GT instances.
========================================================================================

用于训练 DynamicKernelDecoder 时将 N 个预测掩码与 M 个 GT 实例进行最优配对.
Used during DynamicKernelDecoder training to assign N predicted masks to M GT instances.

算法 | Algorithm:
    1. 构建代价矩阵 C ∈ R^{N×M}: C[i,j] = 1 - Dice(pred_i, gt_j) + λ_bce × BCE(pred_i, gt_j)
    2. Scipy linear_sum_assignment → 最小代价配对
    3. 返回 (matched_pairs, unmatched_preds, unmatched_gts)

用法 | Usage::

    from adatile.metrics.hungarian_matcher import hungarian_match_instances

    matched, unmatched_pred, unmatched_gt = hungarian_match_instances(pred_masks, gt_masks)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional
from scipy.optimize import linear_sum_assignment


def _dice_coeff(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐对 Dice 系数 | Pairwise Dice coefficient.

    :param pred: [N, H, W] float32 predicted masks (probability, ∈ [0,1]).
    :param gt: [M, H, W] float32 GT binary masks (∈ {0,1}).
    :return: [N, M] Dice coefficients.
    """
    N, H, W = pred.shape
    M = gt.shape[0]

    pred_flat = pred.view(N, -1)  # [N, H*W]
    gt_flat = gt.view(M, -1)      # [M, H*W]

    # 逐对计算 | Compute pairwise
    # inter[n, m] = sum(pred_n * gt_m)
    inter = pred_flat @ gt_flat.T  # [N, M]

    pred_sum = pred_flat.sum(dim=1, keepdim=True)   # [N, 1]
    gt_sum = gt_flat.sum(dim=1, keepdim=True)        # [M, 1] → need [1, M]

    union = pred_sum + gt_sum.T  # [N, M]

    dice = (2.0 * inter + eps) / (union + eps)
    return dice


def _bce_cost(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """逐对 BCE 代价 | Pairwise BCE cost.

    :param pred: [N, H, W] float32 predicted masks.
    :param gt: [M, H, W] float32 GT binary masks.
    :return: [N, M] mean BCE per pair.
    """
    N, H, W = pred.shape
    M = gt.shape[0]

    # BCE = -(y*log(p) + (1-y)*log(1-p))
    # 对每对 (n, m) 求均值 | Average over spatial dims for each pair
    pred_clamp = pred.clamp(eps, 1 - eps)
    pred_flat = pred_clamp.view(N, 1, -1)  # [N, 1, H*W]
    gt_flat = gt.view(1, M, -1)            # [1, M, H*W]

    bce = -(gt_flat * pred_flat.log() + (1 - gt_flat) * (1 - pred_flat).log())
    return bce.mean(dim=2)  # [N, M]


def hungarian_match_instances(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 0.5,
    min_dice: float = 0.05,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian 一对一最优匹配 | Hungarian one-to-one optimal matching.

    将 N 个预测掩码与 M 个 GT 实例进行配对。
    Assign N predicted masks to M GT instances optimally.

    :param pred_masks: [N, H, W] float32 predicted masks (probability ∈ [0,1]).
    :param gt_masks: [M, H, W] float32 GT binary masks (∈ {0,1}).
        M 可以 = 0 (图像中没有该类实例).
    :param dice_weight: Dice 代价权重 | Dice cost weight.
    :param bce_weight: BCE 代价权重 | BCE cost weight.
    :param min_dice: 最小 Dice 阈值: 低于此值的配对强制为无效 | Minimum Dice threshold for valid match.
    :return: (matched_pairs, unmatched_preds, unmatched_gts)
        - matched_pairs: list of (pred_idx, gt_idx) 有效配对.
        - unmatched_preds: list of pred_idx 未匹配的预测.
        - unmatched_gts: list of gt_idx 未匹配的 GT 实例.
    """
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]

    # Edge case: 无 GT 实例 | No GT instances → all predictions unmatched
    if M == 0:
        return [], list(range(N)), []

    # Edge case: 无预测 | No predictions → all GTs unmatched
    if N == 0:
        return [], [], list(range(M))

    # 构建代价矩阵 | Build cost matrix
    # cost = (1 - Dice) + λ * BCE
    # Lower cost = better match
    dice = _dice_coeff(pred_masks, gt_masks)  # [N, M]
    bce = _bce_cost(pred_masks, gt_masks)      # [N, M]

    cost = dice_weight * (1.0 - dice) + bce_weight * bce  # [N, M]

    # Hungarian assignment (minimize cost)
    cost_np = cost.detach().cpu().numpy()
    pred_indices, gt_indices = linear_sum_assignment(cost_np)

    # 过滤低质量配对 | Filter low-quality matches
    matched_pairs = []
    matched_pred_set = set()
    matched_gt_set = set()

    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if dice[p_idx, g_idx] >= min_dice:
            matched_pairs.append((p_idx, g_idx))
            matched_pred_set.add(p_idx)
            matched_gt_set.add(g_idx)

    # 未匹配的预测和 GT | Unmatched predictions and GTs
    unmatched_preds = [i for i in range(N) if i not in matched_pred_set]
    unmatched_gts = [i for i in range(M) if i not in matched_gt_set]

    return matched_pairs, unmatched_preds, unmatched_gts


def multi_instance_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    matched_pairs: list[tuple[int, int]],
    unmatched_preds: list[int],
    unmatched_gts: list[int],
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    empty_weight: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    多实例损失 | Multi-instance loss.

    对匹配的预测计算 Dice+BCE, 对未匹配的预测惩罚其输出→0。
    For matched pairs: Dice + BCE. For unmatched predictions: pull toward zero.

    :param pred_masks: [N, H, W] float32 predicted masks (probability ∈ [0,1]).
    :param gt_masks: [M, H, W] float32 GT binary masks.
    :param matched_pairs: list of (pred_idx, gt_idx) valid matches.
    :param unmatched_preds: list of pred_idx unmatched predictions.
    :param unmatched_gts: list of gt_idx unmatched GT instances (not used in loss directly;
        unmatched GTs are missed instances — they contribute to FN but not to loss gradient).
    :param dice_weight: Dice loss weight.
    :param bce_weight: BCE loss weight.
    :param empty_weight: Weight for pulling unmatched predictions to zero.
    :param eps: Numerical stability epsilon.
    :return: (total_loss, loss_dict)
    """
    total_loss = torch.tensor(0.0, device=pred_masks.device)
    n_dice = 0
    n_zero = 0

    # ── 匹配的预测 → Dice + BCE | Matched predictions → Dice + BCE ──
    for p_idx, g_idx in matched_pairs:
        pred = pred_masks[p_idx]  # [H, W]
        gt = gt_masks[g_idx]      # [H, W]

        # Dice loss
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        dice = (2.0 * inter + eps) / (union + eps)
        dice_loss = 1.0 - dice

        # BCE loss
        pred_clamp = pred.clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(pred_clamp, gt, reduction="mean")

        total_loss = total_loss + dice_weight * dice_loss + bce_weight * bce
        n_dice += 1

    # ── 未匹配的预测 → 拉向零 | Unmatched predictions → pull toward zero ──
    for p_idx in unmatched_preds:
        pred = pred_masks[p_idx]  # [H, W]
        # Mean squared: encourage all pixels to be 0
        zero_loss = (pred ** 2).mean()
        total_loss = total_loss + empty_weight * zero_loss
        n_zero += 1

    # ── 归一化 | Normalize by number of terms ──
    n_terms = n_dice + n_zero
    if n_terms > 0:
        total_loss = total_loss / n_terms

    loss_dict = {
        "n_matched": n_dice,
        "n_unmatched_pred": len(unmatched_preds),
        "n_unmatched_gt": len(unmatched_gts),
        "n_zero": n_zero,
    }

    return total_loss, loss_dict


def masks_to_binary(masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """概率掩码 → 二值掩码 | Probability masks → binary masks.

    :param masks: [N, H, W] float32 ∈ [0,1].
    :param threshold: 二值化阈值 | Binarization threshold.
    :return: [N, H, W] bool.
    """
    return masks > threshold


def mask_scores(masks: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """计算每个掩码的置信度 | Compute confidence score per mask.

    :param masks: [N, H, W] float32 probability masks.
    :param method: "mean" (mean prob within mask) or "max" (max prob).
    :return: [N] confidence scores.
    """
    if method == "mean":
        # Mean probability of the binarized mask region
        binary = masks > 0.5
        scores = []
        for i in range(masks.shape[0]):
            if binary[i].sum() > 0:
                scores.append(masks[i][binary[i]].mean())
            else:
                scores.append(torch.tensor(0.0, device=masks.device))
        return torch.stack(scores)
    elif method == "max":
        return masks.max(dim=2)[0].max(dim=1)[0]
    else:
        raise ValueError(f"Unknown score method: {method}")
