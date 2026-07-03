"""
Proto Mask 系数预测器 | Proto Mask Coefficient Predictor.
===========================================================

核心创新：将 Few-shot support prototype 映射为 FastSAM proto mask 系数，
通过线性组合 proto masks 生成实例掩码。

Core innovation: Maps few-shot support prototypes to FastSAM proto mask coefficients,
generating instance masks via linear combination of proto masks.

原理 | Principle:
    FastSAM 预训练了 32 个 proto masks (基函数)，任意实例掩码可通过线性组合生成：
        instance_mask = sigmoid(coefficients @ proto_masks)
    其中 coefficients 是 32 维向量，proto_masks 是 [32, H, W] 基函数。

    FastSAM pretrained 32 proto masks (basis functions). Any instance mask
    can be generated via linear combination:
        instance_mask = sigmoid(coefficients @ proto_masks)
    where coefficients is a 32-d vector, proto_masks is [32, H, W] basis.

本模块将 support prototype 映射为 coefficients:
    Support Image → FG prototype [1280] → MLP → coefficients [32]

    Then: mask = sigmoid(coefficients @ proto_masks)

用法 | Usage::

    from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor

    predictor = ProtoCoeffPredictor(proto_dim=32, feat_dim=1280)
    coeffs = predictor(prototype)           # [B, 32] coefficients
    mask = predictor.generate_mask(coeffs, proto_masks)  # [B, H, W] mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoCoeffPredictor(nn.Module):
    """
    Few-shot 条件掩码系数预测器 | Few-Shot Conditioned Mask Coefficient Predictor.

    将 support prototype 映射为 32 维 mask 系数，然后通过线性组合
    FastSAM proto masks 生成实例掩码。

    Maps support prototype → 32-d mask coefficients, then generates
    instance masks via linear combination of FastSAM proto masks.

    架构 | Architecture:
        Support Prototype [feat_dim] → MLP → [proto_dim] coefficients
        coeffs @ proto_masks → sigmoid → binary mask

    Parameters
    ----------
    proto_dim : int
        Proto mask 数量（FastSAM 默认 32）| Number of proto masks (FastSAM default 32).
    feat_dim : int
        Support prototype 特征维度（FastSAM P4 = 1280）| Support prototype feature dim.
    hidden_dim : int
        MLP 隐藏层维度 | MLP hidden dimension.

    参数量 | Parameter Count: ~400K (feat_dim=1280, hidden_dim=256, proto_dim=32).
    """

    def __init__(
        self,
        proto_dim: int = 32,
        feat_dim: int = 1280,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.proto_dim = proto_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # ── 三层 MLP: prototype → coefficients | 3-layer MLP: prototype → coefficients ──
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proto_dim),
        )

        # ── 可选: 多原型支持（K-means 聚类得到的 K 个原型）| Optional: multi-prototype support ──
        self.multi_proto_fusion = nn.Sequential(
            nn.Linear(proto_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proto_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重 | Initialize weights."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prototype: torch.Tensor) -> torch.Tensor:
        """
        预测 mask 系数 | Predict mask coefficients.

        :param prototype: [B, feat_dim] L2-normalized support prototype.
        :return: [B, proto_dim] mask coefficients (no sigmoid — applied in generate_mask).
        """
        return self.mlp(prototype)

    def forward_multi_proto(
        self,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        多原型系数预测 | Multi-prototype coefficient prediction.

        对 K 个原型分别预测系数，然后融合（取平均或 MLP 融合）。
        Predict coefficients for K prototypes separately, then fuse (mean or MLP fusion).

        :param prototypes: [K, feat_dim] K L2-normalized prototypes.
        :return: [1, proto_dim] fused coefficients.
        """
        K = prototypes.shape[0]
        coeffs = self.forward(prototypes)  # [K, proto_dim]

        if K == 1:
            return coeffs

        # ── 简单平均融合 + MLP 精炼 | Simple mean fusion + MLP refine ──
        mean_coeff = coeffs.mean(dim=0, keepdim=True)   # [1, proto_dim]
        max_coeff = coeffs.max(dim=0, keepdim=True)[0]   # [1, proto_dim]
        fused = torch.cat([mean_coeff, max_coeff], dim=-1)  # [1, proto_dim*2]
        return self.multi_proto_fusion(fused)  # [1, proto_dim]

    @staticmethod
    def generate_mask(
        coeffs: torch.Tensor,
        proto_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        从系数和 proto masks 生成实例掩码 | Generate instance mask from coefficients and proto masks.

        核心公式 | Core Formula:
            mask = sigmoid(coeffs @ flatten(proto_masks))

        :param coeffs: [N, proto_dim] mask coefficients.
        :param proto_masks: [proto_dim, H, W] 或 [1, proto_dim, H, W] proto basis masks.
            FastSAM backbone 输出的 proto masks 可能带 batch 维度。
            Proto masks from FastSAM backbone may have a batch dimension.
        :return: [N, H, W] instance masks in [0, 1] (after sigmoid).
        """
        # ── 处理可选的 batch 维度 | Handle optional batch dimension ──
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)  # [1, proto_dim, H, W] → [proto_dim, H, W]

        proto_dim = proto_masks.shape[0]
        spatial_shape = proto_masks.shape[1:]
        proto_flat = proto_masks.view(proto_dim, -1)  # [proto_dim, H*W]

        # 矩阵乘法: coeffs @ proto_flat → per-instance mask
        # Matrix multiplication: coeffs @ proto_flat → per-instance mask
        masks_flat = coeffs @ proto_flat  # [N, H*W]
        masks = masks_flat.view(-1, *spatial_shape)  # [N, H, W]

        return torch.sigmoid(masks)

    def predict_mask(
        self,
        prototype: torch.Tensor,
        proto_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        端到端预测: prototype → mask | End-to-end prediction: prototype → mask.

        :param prototype: [B, feat_dim] L2-normalized support prototype.
        :param proto_masks: [proto_dim, H, W] proto basis masks.
        :return: [B, H, W] instance masks in [0, 1].
        """
        coeffs = self.forward(prototype)  # [B, proto_dim]
        return self.generate_mask(coeffs, proto_masks)  # [B, H, W]

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"ProtoCoeffPredictor(feat_dim={self.feat_dim}, "
                f"proto_dim={self.proto_dim}, hidden_dim={self.hidden_dim}, "
                f"params={n_params/1e3:.1f}K)")
