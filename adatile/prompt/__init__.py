"""
Prompt Token — 类别条件编码 (CAT-SAM Prompt Token 迁移)
========================================================
Class-Conditional Prompt Encoding — CAT-SAM Prompt Token concept.

CAT-SAM 使用可学习的 Prompt Token 编码"找什么类别"的信息。
本模块将其适配为：通用 Prompt (遥感先验) + Few-Shot Prototype (具体类别)。
CAT-SAM uses learnable Prompt Tokens to encode "what to look for".
This module adapts it as: Generic Prompt (remote sensing prior) + Few-Shot Prototype (specific class).

设计 | Design:
    GenericPrompt:   可学习 embedding, 编码"遥感目标"的通用概念.
    GenericPrompt:   learnable embeddings encoding the general concept of "remote sensing objects."
    PrototypePrompt: 从 Support Set 提取的类别原型, 编码"这一个具体类别".
    PrototypePrompt: class prototype extracted from Support Set, encoding "this specific class."
    Fusion:          concat + linear projection → unified condition.

对应 CAT-SAM: Prompt Token → 类别嵌入 → Cross-Attention 到图像特征.
CAT-SAM analog: Prompt Token → class embedding → Cross-Attention to image features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger

logger = get_logger("prompt")


class GenericPrompt(nn.Module):
    """
    可学习的通用类别提示 — 编码"遥感目标"的通用先验.
    Learnable generic class prompts — encode general "remote sensing object" prior.

    这些 token 在训练中学习遥感域中"什么是前景目标"的通用概念，
    不绑定特定类别（类别无关，与 FDR 同哲学）。
    These tokens learn during training the general concept of "what is a foreground
    object" in remote sensing, not tied to specific classes (category-agnostic,
    same philosophy as FDR).

    ----------
    num_tokens : int
        Prompt token 数量 | Number of prompt tokens.
    dim : int
        每个 token 的维度 | Dimension per token.
    """

    def __init__(self, num_tokens: int = 8, dim: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        # 从正态分布初始化，std=0.02 (Transformer 惯例)
        # Init from normal with std=0.02 (Transformer convention)
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

        logger.log_info("prompt/init",
                        f"GenericPrompt: {num_tokens} tokens, dim={dim}, "
                        f"{num_tokens * dim:,} params")

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        :param batch_size: 批次大小 | Batch size.
        :type batch_size: int
        :return: prompt tokens [B, num_tokens, dim]
        :rtype: torch.Tensor
        """
        return self.tokens.expand(batch_size, -1, -1)


class PrototypePrompt(nn.Module):
    """
    从 Support Set 特征提取类别原型，编码"这个具体类别"的信息.
    Extract class prototype from Support Set features, encoding "this specific class."

    原型计算方法：masked average pooling over support features.
    Prototype computation: masked average pooling over support features.

    ----------
    feat_dim : int
        Support 特征维度 | Support feature dimension (e.g., 1280 for P4).
    proto_dim : int
        投影后维度 | Projected dimension (usually 256).
    """

    def __init__(self, feat_dim: int = 1280, proto_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.proto_dim = proto_dim
        # 投影: raw feature → prompt space | Projection: raw feature → prompt space
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proto_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proto_dim, proto_dim),
        )

        logger.log_info("prompt/init",
                        f"PrototypePrompt: feat={feat_dim} → proto={proto_dim}")

    def forward(
        self,
        support_feats: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 support 特征 + mask 提取原型.
        Extract prototype from support features + masks.

        :param support_feats: [B, C, H, W] support 特征.
        :param support_feats: [B, C, H, W] support features.
        :param support_masks: [B, H, W] support 二值 mask.
        :param support_masks: [B, H, W] binary support masks.
        :return: prototype [B, 1, proto_dim]
        :rtype: torch.Tensor
        """
        B, C, H, W = support_feats.shape
        # Resize mask to feature resolution | Resize mask to feature spatial size
        if support_masks.shape[-2:] != (H, W):
            support_masks = F.interpolate(
                support_masks.unsqueeze(1).float(),
                size=(H, W), mode="nearest",
            ).squeeze(1)  # [B, H, W]

        # Masked average pooling | 掩码平均池化
        mask_flat = support_masks.view(B, 1, -1)  # [B, 1, H*W]
        feat_flat = support_feats.view(B, C, -1)    # [B, C, H*W]
        fg_sum = (feat_flat * mask_flat).sum(dim=-1)  # [B, C]
        fg_count = mask_flat.sum(dim=-1).clamp(min=1)  # [B, 1]
        proto_raw = fg_sum / fg_count  # [B, C]

        # 投影到 prompt 空间 | Project to prompt space
        proto = self.proj(proto_raw)  # [B, proto_dim]
        return proto.unsqueeze(1)  # [B, 1, proto_dim]


class PromptFusion(nn.Module):
    """
    融合 GenericPrompt + PrototypePrompt → 统一的条件向量.
    Fuse GenericPrompt + PrototypePrompt → unified condition vector.

    融合方式: concat → Linear → LayerNorm → condition.
    Fusion: concat → Linear → LayerNorm → condition.

    ----------
    generic_dim : int
        通用 prompt 维度 | Generic prompt dim.
    proto_dim : int
        原型 prompt 维度 | Prototype prompt dim.
    output_dim : int
        融合后输出维度 | Fused output dim.
    """

    def __init__(
        self,
        generic_dim: int = 256,
        proto_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(generic_dim + proto_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        logger.log_info("prompt/init",
                        f"PromptFusion: ({generic_dim}+{proto_dim}) → {output_dim}")

    def forward(
        self,
        generic: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param generic: [B, N_gen, D_gen] 通用 prompt.
        :param prototype: [B, N_proto, D_proto] 原型 prompt.
        :return: [B, N_gen + N_proto, D_out] 融合后的条件.
        """
        # Concat along token dimension (每个 token 独立融合)
        # Concat along token dimension (fuse each token independently)
        combined = torch.cat([generic, prototype], dim=1)  # [B, N_gen+N_proto, D]
        # 需要先确保维度对齐 | Ensure dimension alignment first
        # generic dim + proto dim → output dim
        # 如果维度不同，用线性层对齐 | If dims differ, align with linear
        return self.fusion(combined)
