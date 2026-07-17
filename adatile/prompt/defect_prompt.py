"""
缺陷感知 Prompt 生成器 | Defect-aware Prompt Generator.
========================================================

P2 特征 → K 个可学习 heatmap → 空间加权 → K 个 prompt token。
P2 features → K learnable heatmaps → spatial softmax weighting → K prompt tokens.

每个 heatmap 自主学习关注不同类型缺陷区域 (Inclusion/Patch/Scratch)。
Each heatmap autonomously learns to attend to different defect types.

设计理念 | Design Philosophy:
    不微调 FastSAM backbone，而是教模型"缺陷在哪里"。
    Don't fine-tune the backbone — teach the model "where defects are".

参数: ~30K | Params: ~30K
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DefectPromptGenerator(nn.Module):
    """
    缺陷感知 Prompt 生成器 | Defect-aware Prompt Generator.

    从 FastSAM P2 特征 (stride=4, 最高分辨率) 预测 K 个 attention heatmap，
    用空间 softmax 聚合 P2 特征 → K 个 prompt token。
    Predicts K attention heatmaps from FastSAM P2 features (stride=4, highest resolution),
    spatially aggregates P2 features via softmax → K prompt tokens.

    完全可导，heatmap 可视化 → 可解释性强。
    Fully differentiable, heatmap visualization → strong interpretability.

    Parameters
    ----------
    in_channels : int
        P2 特征通道数 (FastSAM-x: 160, FastSAM-s: ~80).
        P2 feature channels.
    prompt_dim : int
        Prompt token 输出维度 | Prompt token output dimension.
    num_prompts : int
        Prompt 数量 K。K 个 heatmap 各自学习关注不同区域。
        Number of prompt tokens. K heatmaps each learn to focus on different regions.
    """

    def __init__(self, in_channels: int = 160, prompt_dim: int = 256, num_prompts: int = 5):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        # ── Heatmap 头: P2 → K 通道 heatmap | Heatmap Head: P2 → K-channel heatmap ──
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_prompts, kernel_size=1),
        )

        # ── Prompt 投影: in_channels → prompt_dim | Prompt Projector ──
        self.proj = nn.Linear(in_channels, prompt_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化，最后一层 Conv 小权重让 heatmap 初始接近均匀。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 最后一层 Conv1x1 初始化为小值，让 heatmap 接近均匀
        last_conv = self.heatmap_head[-1]
        nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward pass.

        :param p2: [B, C, H, W] FastSAM P2 特征 (e.g., [1, 160, 50, 50]).
        :return:
            prompt_embeddings: [B, K, prompt_dim] — 缺陷感知 prompt token。
            heatmaps: [B, K, H, W] — 注意力 heatmap，用于可视化。
        """
        B, C, H, W = p2.shape

        # ── 生成 K 个 heatmap | Generate K heatmaps ──
        heatmaps = self.heatmap_head(p2)  # [B, K, H, W]

        # ── 空间 softmax: 每个 heatmap 关注不同空间位置 ──
        # Spatial softmax: each heatmap attends to different spatial locations
        attn = heatmaps.flatten(2).softmax(dim=-1)  # [B, K, H*W]

        # ── 加权聚合: prompt_k = Σ attn_k[h,w] * P2[:, :, h, w] ──
        # Weighted aggregation: prompt_k = sum over spatial positions
        p2_flat = p2.flatten(2)  # [B, C, H*W]
        prompt_tokens = torch.bmm(attn, p2_flat.transpose(1, 2))  # [B, K, C]

        # ── 投影到 prompt_dim | Project to prompt_dim ──
        prompt_embeddings = self.proj(prompt_tokens)  # [B, K, prompt_dim]

        return prompt_embeddings, heatmaps


def prompt_global_conditioning(
    prompt_embeddings: torch.Tensor,
    target_features: torch.Tensor,
    mode: str = "concat",
) -> torch.Tensor:
    """
    将 prompt embedding 广播为全局空间条件 | Broadcast prompt as global spatial conditioning.

    聚合 K 个 prompt → 全局向量 → 拼接到目标特征的空间维度。
    Aggregate K prompts → global vector → concatenate to target feature's spatial dims.

    :param prompt_embeddings: [B, K, D] prompt tokens from DefectPromptGenerator.
    :param target_features: [B, C, H, W] feature map to condition.
    :param mode: "concat" (拼接) or "film" (FiLM modulation, not yet implemented).
    :return: conditioned features [B, C+D, H, W] if concat, or [B, C, H, W] if film.
    """
    B, _, D = prompt_embeddings.shape
    _, _, H, W = target_features.shape

    # 聚合: 均值 | Aggregate: mean over K prompts
    prompt_global = prompt_embeddings.mean(dim=1)  # [B, D]

    if mode == "concat":
        prompt_spatial = prompt_global.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)
        return torch.cat([target_features, prompt_spatial], dim=1)  # [B, C+D, H, W]

    # FiLM mode (未来扩展 | future extension):
    # split into gamma, beta → feature * gamma + beta
    raise NotImplementedError(f"Unknown conditioning mode: {mode}")


def prompt_diversity_loss(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Prompt 多样性损失 — 强制 K 个 heatmap 关注不同区域。
    Prompt Diversity Loss — forces K heatmaps to attend to different regions.

    计算 softmax 注意力分布之间的余弦相似度，惩罚非对角线相似度。
    Computes cosine similarity between softmax attention distributions,
    penalizing off-diagonal similarity (encourages orthogonality).

    :param heatmaps: [B, K, H, W] raw heatmap logits (will be softmaxed internally).
    :return: scalar diversity loss. 0 = all prompts attend to same region,
             1 = fully orthogonal attention patterns.
    """
    B, K, H, W = heatmaps.shape

    # Spatial softmax → attention distribution | 空间 softmax → 注意力分布
    attn = heatmaps.flatten(2).softmax(dim=-1)  # [B, K, H*W]

    # L2 normalize each attention vector | L2 归一化
    attn_norm = torch.nn.functional.normalize(attn, dim=-1)  # [B, K, H*W]

    # Pairwise cosine similarity | 两两余弦相似度
    sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # [B, K, K]

    # Penalize off-diagonal: want cos(p_i, p_j) ≈ 0 for i≠j
    eye = torch.eye(K, device=sim.device, dtype=sim.dtype).unsqueeze(0)  # [1, K, K]
    off_diag_mask = 1.0 - eye  # [1, K, K]
    diversity = (sim.abs() * off_diag_mask).sum() / (B * K * max(K - 1, 1))

    return diversity
