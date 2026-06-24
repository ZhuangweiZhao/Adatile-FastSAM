"""
Prototype Computation — shared across few-shot experiments.
==============================================================
原型计算: 从 support P4 特征和 binary mask 中提取 L2-normalized prototype。

Prototype computation: extract L2-normalized prototype from support P4 features
and binary masks. Shared across C-02A/B, C-03, C-04, B-09.

Canonical definition (2026-07). Previously duplicated in 4 files.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_fg_prototype(
    p4_features: list[torch.Tensor],
    masks: list[torch.Tensor],
    feat_dim: int = 1280,
) -> torch.Tensor:
    """
    masked_mean(P4) → L2-normalized prototype | 掩码平均 → L2 归一化原型。

    对每张 support image 的 P4 特征做 foreground-masked average pooling，
    然后对多张 support 的结果取平均并 L2 归一化。

    Foreground-masked average pooling over P4 features from support images,
    then mean-pool across supports and L2-normalize.

    :param p4_features: list of [C, H_p4, W_p4] tensors (one per support image, on device)
    :param masks: list of [H_orig, W_orig] binary float tensors (one per support image)
    :param feat_dim: P4 feature channel dimension (1280 for FastSAM)
    :return: [feat_dim] L2-normalized prototype vector, or zero if all masks empty
    """
    p4_h, p4_w = p4_features[0].shape[1], p4_features[0].shape[2]
    all_feats: list[torch.Tensor] = []

    for i in range(len(p4_features)):
        m = masks[i]
        if m.dim() == 3:
            m = m.squeeze(0)
        mask_4d = m.unsqueeze(0).unsqueeze(0).float()
        mask_p4 = F.interpolate(
            mask_4d, size=(p4_h, p4_w), mode="nearest"
        ).squeeze(0)

        fg_area = mask_p4.sum()
        if fg_area > 0:
            weighted = (p4_features[i] * mask_p4).sum(dim=(1, 2)) / (fg_area + 1e-8)
            all_feats.append(weighted)

    if not all_feats:
        return torch.zeros(feat_dim, device=p4_features[0].device)

    return F.normalize(torch.stack(all_feats).mean(dim=0), dim=0, p=2)
