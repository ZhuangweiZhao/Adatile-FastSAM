"""
Pixel-Level Supervised Contrastive Loss for Few-Shot Feature Learning.
========================================================================
像素级监督对比损失 — 直接在特征空间上做度量学习, 而非仅依赖 mask prediction loss.

核心思想 | Core Idea:
    Mask BCE/Dice loss 通过 decoder 反向传播到 backbone 是间接的。
    对比损失直接在 P3/P4 特征上约束: FG 像素靠近 support prototype, BG 像素远离。
    这让 backbone/adapter 直接学习 "什么样的特征是 few-shot matchable 的"。

    Mask BCE/Dice backprop through decoder → backbone is indirect.
    Contrastive loss directly constrains P3/P4 features: FG pixels near prototype, BG far.
    This teaches backbone/adapter directly what makes features "few-shot matchable".

损失设计 | Loss Design:
    L_total = L_mask + λ * L_contrastive

    L_contrastive:
        1. Attraction: query FG pixels → pulled toward support prototype
        2. Repulsion:  query BG pixels → pushed away from support prototype
        3. Uniformity:  prevent feature collapse (optional, via variance regularization)

用法 | Usage:
    >>> from adatile.losses.contrastive import PixelSupConLoss
    >>> contrastive_loss = PixelSupConLoss(temperature=0.1)
    >>> loss = contrastive_loss(query_p3, query_mask, support_prototype)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelSupConLoss(nn.Module):
    """
    Pixel-level supervised contrastive loss for few-shot feature learning.
    像素级监督对比损失.

    在 L2-normalized 特征空间上:
    - FG 像素与 support prototype 的余弦相似度应最大化
    - BG 像素与 support prototype 的余弦相似度应最小化
    - 特征分布应有足够方差, 防止 collapse

    On L2-normalized feature space:
    - FG pixels should have high cosine similarity to support prototype
    - BG pixels should have low cosine similarity to support prototype
    - Feature distribution should maintain variance to prevent collapse

    Parameters | 参数:
        temperature: 对比温度 (default 0.1, lower = sharper distinction).
        margin: FG/BG 相似度最小差距 | Minimum similarity gap between FG and BG.
        uniform_weight: 均匀性正则化权重 | Uniformity regularization weight.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.3,
                 uniform_weight: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.uniform_weight = uniform_weight

    def forward(
        self,
        query_feat: torch.Tensor,
        query_mask: torch.Tensor,
        support_prototype: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        :param query_feat: [B, C, H, W] query 特征图 (P3 或 P4) | Query feature map.
        :param query_mask: [B, H, W] query GT mask (binary, 1=FG) — 需与 query_feat 同分辨率.
        :param support_prototype: [C] support FG prototype (mean of support FG features).
        :return: (loss, components_dict)
        """
        B, C, H, W = query_feat.shape

        # ── 1. L2 normalize & cosine similarity | L2 归一化 + 余弦相似度 ──
        q_norm = F.normalize(query_feat, p=2, dim=1)  # [B, C, H, W]
        p_norm = F.normalize(support_prototype, p=2, dim=0)  # [C]
        sim = (q_norm * p_norm.view(1, C, 1, 1)).sum(dim=1)  # [B, H, W]
        sim = sim / self.temperature

        # ── 2. Resize mask to feature resolution | 对齐 mask 分辨率 ──
        if query_mask.shape[-2:] != (H, W):
            mask = F.interpolate(
                query_mask.unsqueeze(1).float(),
                size=(H, W), mode="nearest"
            ).squeeze(1)  # [B, H, W]
        else:
            mask = query_mask.float()

        fg_mask = mask > 0.5
        bg_mask = mask <= 0.5

        # ── 3. Attraction loss: FG pixels → high similarity ──
        # 使用 softplus 鼓励 FG 相似度 > margin
        # Use softplus to encourage FG similarity > margin
        if fg_mask.sum() > 0:
            fg_sim = sim[fg_mask]
            # L_attract = softplus(margin - fg_sim) → penalize fg_sim < margin
            attract_loss = F.softplus(self.margin - fg_sim).mean()
        else:
            attract_loss = torch.tensor(0.0, device=sim.device)

        # ── 4. Repulsion loss: BG pixels → low similarity ──
        # 使用 softplus 鼓励 BG 相似度 < 0 (或 < margin)
        if bg_mask.sum() > 0:
            bg_sim = sim[bg_mask]
            # L_repel = softplus(bg_sim) → penalize bg_sim > 0
            repel_loss = F.softplus(bg_sim).mean()
        else:
            repel_loss = torch.tensor(0.0, device=sim.device)

        # ── 5. Uniformity regularization: 防止特征 collapse | Prevent collapse ──
        # 度量: query 像素余弦相似度的标准差 (越高越好 = 特征不塌缩)
        q_flat = q_norm.reshape(B, C, -1)  # [B, C, N]
        # 随机采样避免 O(N²) | Random sample to avoid O(N²)
        n_sample = min(256, q_flat.shape[-1])
        indices = torch.randperm(q_flat.shape[-1], device=q_flat.device)[:n_sample]
        q_sample = q_flat[:, :, indices]  # [B, C, n_sample]
        # Pairwise cosine similarity within sample → 均值应接近 0 (分散)
        sim_matrix = torch.bmm(q_sample.transpose(1, 2), q_sample)  # [B, n, n]
        # Exclude diagonal (self-similarity = 1.0)
        mask_offdiag = ~torch.eye(n_sample, dtype=torch.bool, device=sim_matrix.device)
        sim_offdiag = sim_matrix[:, mask_offdiag]
        # 均匀性损失: 平均 pairwise similarity 越高 → 越 collapse → 惩罚
        uniformity_loss = sim_offdiag.pow(2).mean()

        # ── 6. Total | 总损失 ──
        loss = attract_loss + repel_loss + self.uniform_weight * uniformity_loss

        components = {
            "contrast_attract": attract_loss.item(),
            "contrast_repel": repel_loss.item(),
            "contrast_uniform": uniformity_loss.item(),
            "contrast_fg_sim_mean": fg_sim.mean().item() if fg_mask.sum() > 0 else 0.0,
            "contrast_bg_sim_mean": bg_sim.mean().item() if bg_mask.sum() > 0 else 0.0,
        }
        return loss, components


class PixelSupConLossMultiScale(nn.Module):
    """
    Multi-scale pixel contrastive loss: applies PixelSupConLoss at P3 and P4.
    多尺度像素对比损失: 在 P3 和 P4 分别应用 PixelSupConLoss.

    P3 (stride 8, fine spatial): 擅长小物体边界 | Good for small object boundaries.
    P4 (stride 16, coarse semantic): 擅长语义区分 | Good for semantic distinction.
    """

    def __init__(self, temperature: float = 0.1, p3_weight: float = 0.5,
                 p4_weight: float = 0.5):
        super().__init__()
        self.p3_loss = PixelSupConLoss(temperature=temperature)
        self.p4_loss = PixelSupConLoss(temperature=temperature)
        self.p3_weight = p3_weight
        self.p4_weight = p4_weight

    def forward(self, query_p3, query_p4, query_mask, support_prototype):
        """Compute contrastive loss at both scales."""
        # Resize mask for each scale
        mask_p3 = F.interpolate(
            query_mask.unsqueeze(1).float(),
            size=query_p3.shape[2:], mode="nearest"
        ).squeeze(1)
        mask_p4 = F.interpolate(
            query_mask.unsqueeze(1).float(),
            size=query_p4.shape[2:], mode="nearest"
        ).squeeze(1)

        loss_p3, comp_p3 = self.p3_loss(query_p3, mask_p3, support_prototype)
        loss_p4, comp_p4 = self.p4_loss(query_p4, mask_p4, support_prototype)

        loss = self.p3_weight * loss_p3 + self.p4_weight * loss_p4

        components = {}
        for k, v in comp_p3.items():
            components[f"{k}_p3"] = v
        for k, v in comp_p4.items():
            components[f"{k}_p4"] = v
        components["contrast_loss"] = loss.item()

        return loss, components
