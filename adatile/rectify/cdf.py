"""
CDF — Cross-scale Defect Fusion | 跨尺度缺陷融合.
===================================================

动态、输入自适应的多尺度融合：不同于 BiFPN 的静态可学习权重，
CDF 根据输入内容动态预测各尺度的融合权重。
Dynamic, input-adaptive multi-scale fusion: unlike BiFPN's static learnable weights,
CDF dynamically predicts per-scale fusion weights based on input content.

核心思想 | Core Idea:
    不同缺陷类型需要不同尺度的特征：
    - 划痕 (Scratch): 细长、高频 → P3 (H/8, 高分辨率) 权重高
    - 斑块 (Patch): 中等、纹理 → P3+P4 平衡
    - 夹杂物 (Inclusion): 小目标 → P3 (H/8) 权重高
    静态权重 (如 BiFPN) 无法适应这种变化, 需要动态权重。
    Different defect types need different scale features:
    - Scratch: fine, high-freq → P3 (H/8, high-res) higher weight
    - Patch: medium, textural → P3+P4 balanced
    - Inclusion: small objects → P3 (H/8) higher weight
    Static weights (like BiFPN) can't adapt; dynamic weights needed.

与纯 BiFPN 的对比 | vs Pure BiFPN:
    PureDecoderP2P3P4: w_p2, w_p3, w_p4 是 nn.Parameter (静态, 所有样本共享)
    CDF:               w3, w4 = MLP(GAP(p3) ⊕ GAP(p4)) (动态, 逐样本预测)
    PureDecoderP2P3P4: w_p2, w_p3, w_p4 are nn.Parameter (static, shared across samples)
    CDF:               w3, w4 = MLP(GAP(p3) ⊕ GAP(p4)) (dynamic, per-sample predicted)

参数量 | Params: ~20K (small MLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleDefectFusion(nn.Module):
    """
    跨尺度缺陷融合 | Cross-scale Defect Fusion.

    输入 P3/P4 特征 → 预测每样本动态融合权重 → 加权输出。
    Input P3/P4 features → predict per-sample dynamic fusion weights → weighted output.

    注意: 该模块只做权重预测和加权, 不上采样/融合。
    上采样和融合由 Decoder 内部完成。
    Note: This module only predicts weights and applies them,
    upsampling and fusion are done inside the Decoder.

    Parameters
    ----------
    p3_channels : int
        P3 通道数 | P3 channels (e.g., 960 for FastSAM-x).
    p4_channels : int
        P4 通道数 | P4 channels (e.g., 1280 for FastSAM-x).
    hidden_dim : int
        权重预测 MLP 隐层维度 | Weight predictor MLP hidden dim (default 64).
    """

    def __init__(
        self,
        p3_channels: int,
        p4_channels: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        total_channels = p3_channels + p4_channels

        # ── 权重预测器 | Weight Predictor ──
        # GAP(p3) ⊕ GAP(p4) → MLP → 2 scalars (w3, w4)
        # [B, C_p3+C_p4] → [B, hidden] → [B, 2]
        self.predictor = nn.Sequential(
            nn.Linear(total_channels, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2, bias=True),
            nn.Softplus(),  # 确保输出 ≥ 0 | Ensure output ≥ 0 (like BiFPN's ReLU)
        )

        # ── 初始化 | Init: 最后一个 Linear 层的权重零初始化 → 接近等权 ──
        # Softplus(0) ≈ 0.693,归一化后 w3≈w4≈1.0
        # Zero-init last Linear → Softplus(0) ≈ 0.693, normalized → w3≈w4≈1.0
        last_linear = self.predictor[-2]  # predictor[-1] 是 Softplus
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        # ── 统计 tracker | Statistics tracker (for interpretability) ──
        self.register_buffer("_w3_mean", torch.tensor(0.0))
        self.register_buffer("_w4_mean", torch.tensor(0.0))
        self.register_buffer("_step_count", torch.tensor(0))

        self._init_params_count()

    def _init_params_count(self) -> None:
        n = sum(p.numel() for p in self.parameters())
        from adatile.logging import get_logger
        get_logger("rectify.cdf").log_info(
            "cdf/init",
            f"CDF(p3={self.p3_channels}, p4={self.p4_channels}): {n/1e3:.1f}K params",
        )

    def forward(
        self,
        p3_features: torch.Tensor,
        p4_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        动态权重融合 | Dynamic weight fusion.

        :param p3_features: [B, C3, H3, W3] P3 特征.
        :param p4_features: [B, C4, H4, W4] P4 特征.
        :return: (p3_weighted, p4_weighted) — 形状不变 | same shape.
        """
        B = p3_features.shape[0]

        # ── 1. 全局统计量 | Global Statistics ──
        # GAP + 可选: 方差 (捕捉纹理信息) | GAP + optional: variance (captures texture)
        p3_gap = F.adaptive_avg_pool2d(p3_features, 1).view(B, -1)  # [B, C3]
        p4_gap = F.adaptive_avg_pool2d(p4_features, 1).view(B, -1)  # [B, C4]

        # ── 2. 拼接 → 预测权重 | Concat → Predict Weights ──
        combined = torch.cat([p3_gap, p4_gap], dim=1)  # [B, C3+C4]
        raw_weights = self.predictor(combined)  # [B, 2]

        # ── 3. 归一化 (保持总能量) | Normalize (preserve total energy) ──
        # w_i = raw_i / sum(raw) × num_scales → 均值=1.0, 保持特征幅度
        # w_i = raw_i / sum(raw) × 2 → mean=1.0, preserves feature magnitude
        w_sum = raw_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w3 = (raw_weights[:, 0] / w_sum.squeeze(1)) * 2.0  # [B]
        w4 = (raw_weights[:, 1] / w_sum.squeeze(1)) * 2.0  # [B]

        # ── 4. 应用权重 | Apply Weights ──
        p3_out = p3_features * w3.view(B, 1, 1, 1)
        p4_out = p4_features * w4.view(B, 1, 1, 1)

        # ── 5. 更新统计 (eval 模式下不更新) | Update stats (skip in eval) ──
        if self.training:
            with torch.no_grad():
                decay = 0.99
                self._w3_mean.mul_(decay).add_(w3.mean().detach(), alpha=1 - decay)
                self._w4_mean.mul_(decay).add_(w4.mean().detach(), alpha=1 - decay)
                self._step_count.add_(1)

        return p3_out, p4_out

    def get_weight_stats(self) -> dict[str, float]:
        """
        获取融合权重统计 (用于可解释性) | Get fusion weight stats (for interpretability).

        :return: {"w3_mean": float, "w4_mean": float, "steps": int}
        """
        return {
            "w3_mean": float(self._w3_mean.item()),
            "w4_mean": float(self._w4_mean.item()),
            "steps": int(self._step_count.item()),
        }

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"CDF(p3={self.p3_channels}, p4={self.p4_channels}, params={n/1e3:.1f}K)"
