"""
mIoU 计算 | Mean Intersection over Union computation.
=======================================================

支持多类别、整数标签和 one-hot 两种输入格式。
Supports multi-class, integer label and one-hot input formats.

V1 教训 | V1 lesson:
    正确处理 {0, 255} 二值掩码——检测 n_unique ≤ 2，而非 max_val ≤ 1。
    Correctly handle {0, 255} binary masks — check n_unique ≤ 2, not max_val ≤ 1.
"""

from __future__ import annotations

import torch


def compute_miou(
    pred: torch.Tensor,         # [B, C, H, W] or [B, H, W]
    target: torch.Tensor,       # [B, C, H, W] or [B, H, W]
    num_classes: int,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    计算平均交并比 | Compute mean Intersection over Union.

    IoU = |pred ∩ target| / |pred ∪ target|
        = TP / (TP + FP + FN)

    支持两种输入格式 | Supports two input formats:
        - 标签格式：pred [B,H,W] (argmax 结果), target [B,H,W]
          Label format: pred [B,H,W] (argmax), target [B,H,W]
        - One-hot：pred [B,C,H,W], target [B,C,H,W]
          One-hot: pred [B,C,H,W], target [B,C,H,W]

    :param pred: 预测张量 | Prediction tensor.
    :type pred: torch.Tensor

    :param target: 目标张量 | Target tensor.
    :type target: torch.Tensor

    :param num_classes: 类别总数 | Total number of classes.
    :type num_classes: int

    :param ignore_index: 忽略的标签值 | Label value to ignore (default -1 = none).
    :type ignore_index: int

    :param eps: 数值稳定项 | Numerical stability term.
    :type eps: float

    :return: dict with: "miou":      float — 所有类别的平均 IoU | Mean IoU over all classes "iou_cls_i": float — 每个类别的 IoU | Per-class IoU
    :rtype: dict[str, float]
    """
    # ── 输入标准化：统一为 [B, H, W] 标签格式 ──
    # Normalize input: convert to [B, H, W] label format

    if pred.dim() == 4 and pred.shape[1] > 1:
        # One-hot [B, C, H, W] → labels [B, H, W]
        pred_labels = pred.argmax(dim=1)
    else:
        pred_labels = pred.squeeze(1) if pred.dim() == 4 else pred

    if target.dim() == 4 and target.shape[1] > 1:
        target_labels = target.argmax(dim=1)
    else:
        target_labels = target.squeeze(1) if target.dim() == 4 else target

    # 确保是 long 类型 | Ensure long type
    pred_labels = pred_labels.long()
    target_labels = target_labels.long()

    # ── 处理 {0, 255} 二值掩码（V1 教训 | V1 lesson）──
    # 如果输入是二值的但值域异常，自动标准化 | Auto-normalize if binary with odd range
    target_labels = _normalize_mask_values(target_labels)

    # ── 逐类别计算 IoU | Per-class IoU ──
    per_class: list[dict] = []
    total_iou = 0.0
    valid_classes = 0

    for cls_idx in range(num_classes):
        # 预测为该类别的像素 | Pixels predicted as this class
        pred_cls = (pred_labels == cls_idx)
        # 真值为该类别的像素 | Pixels where ground truth is this class
        target_cls = (target_labels == cls_idx)

        # 排除 ignore_index | Exclude ignore_index
        if ignore_index >= 0:
            valid_mask = (target_labels != ignore_index)
            pred_cls = pred_cls & valid_mask
            target_cls = target_cls & valid_mask

        # 交集和并集 | Intersection and union
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            iou_val = (intersection + eps) / (union + eps)
        else:
            # 该类别在 pred 和 target 中都不存在 → 跳过
            # Class not present in pred or target → skip
            continue

        per_class.append({"class": cls_idx, "iou": iou_val.item()})
        total_iou += iou_val.item()
        valid_classes += 1

    # 平均 IoU | Mean IoU
    miou = total_iou / max(valid_classes, 1)

    return {"miou": miou, "per_class": per_class}


def _normalize_mask_values(mask: torch.Tensor) -> torch.Tensor:
    """
    标准化掩码值 | Normalize mask values.

    V1 教训 | V1 lesson:
        {0, 255} 掩码必须被检测和标准化为 {0, 1}。
        {0, 255} masks must be detected and normalized to {0, 1}.

    检测逻辑：如果唯一值数量 ≤ 2 且最大值 > 1，
    将其除以 255 标准化。
    Detection logic: if n_unique ≤ 2 and max > 1,
    normalize by dividing by 255.
    """
    unique_vals = mask.unique()
    if len(unique_vals) <= 2 and unique_vals.max() > 1:
        # {0, 255} → {0, 1}
        mask = (mask > 128).long()
    return mask
