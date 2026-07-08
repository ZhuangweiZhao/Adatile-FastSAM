"""
三层协同自适应解码器 | Three-Layer Collaborative Adaptive Decoder.
==================================================================

P8/P4/P3 分工协作的少样本实例分割解码器。
Three-layer collaborative decoder with staged responsibilities:
    P8 → Prototype generation (global semantics, class identity)
    P4 → Main decoder (semantic-spatial balance, workhorse)
    P3 → Boundary refinement (high-res detail, small objects)

设计动机 | Design Motivation:
    uf=8 解冻了 P3+P4+P8，但 AdaptiveSparseDecoder 只消费 P4。
    P3 的高分辨细节和 P8 的全局语义未被 decoder 直接利用 → Feature-Decoder 不匹配。
    uf=8 unfreezes P3+P4+P8, but AdaptiveSparseDecoder only consumes P4.
    P3's high-res detail and P8's global semantics are unused → feature-head mismatch.

架构总览 | Architecture Overview::

    Support Image → Backbone ─┐
                              │
    ┌─────────────────────────┘
    │
    │  P8 [1280, H/32, W/32]  ← 语义最强，全局类别表征
    │       ↓                    Best semantics, global class identity
    │  MAP + L2-norm → support_proto [1280]
    │       ↓
    │  ProtoCoeffPredictor → coeffs [32]
    │
    └──────────────────────────┐
                               │
    Query Image → Backbone ────┤
    │                          │
    ├─ P3 [960, H/8, W/8]     │  ← 高分辨率，边界/细节
    │       ↓                  │     High-res, boundaries/details
    │  p3_proj → p3_refine → p3_logit → upsample(H/4)
    │                          │
    ├─ P4 [1280, H/16, W/16]  │  ← 空间+语义平衡，主力
    │       ↓                  │     Spatial-semantic balance, workhorse
    │  feat_proj → feat_refine → p4_logit → upsample(H/4)
    │                          │
    └─ Proto [32, H/4, W/4] ──┤  ← 预训练基函数
                               │     Pretrained basis functions
                               │
    coeffs @ Proto → proto_mask [1, H/4, W/4]  ← 类别形状先验
                                                   Class shape prior

    Fusion at H/4:
        concat(proto_mask, p3_logit_up, p4_logit_up) → [3, H/4, W/4]
        → fusion_conv → sigmoid → [H/4, W/4] instance mask

参数量 | Params: ~1.65M
    ProtoCoeffPredictor: 427K
    P4 path:            715K (feat_proj + feat_refine + p4_head)
    P3 path:            289K (p3_proj + p3_refine + p3_head)
    Fusion:              ~5K (3→16→1)

用法 | Usage::

    from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4

    decoder = AdaptiveDecoderP3P4(p3_channels=960, p4_channels=1280)
    mask = decoder(
        p3_features=p3,            # [1, 960, H/8, W/8]
        p4_features=p4,            # [1, 1280, H/16, W/16]
        proto_masks=proto,         # [32, H/4, W/4]
        support_proto=prototype,   # [1280] P8-derived, L2-normalized
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor


class AdaptiveDecoderP3P4(nn.Module):
    """
    三层协同自适应解码器 | Three-Layer Collaborative Adaptive Decoder.

    整合 P3/P4/P8 分工协作:
        P8 → Prototype generation (经由 compute_support_prototype, 外部完成)
        P4 → Main decoder (语义-空间平衡，主力特征精炼)
        P3 → Boundary refinement (高分辨率边界恢复，小目标检测)

    与 AdaptiveSparseDecoder 的关键差异:
    1. Prototype 源从 P4 改为 P8 (更好语义) — 在外部 compute_support_prototype 完成
    2. 新增 P3 精炼路径 (边界细节)
    3. 三路可学习融合替代逐元素相加

    Parameters
    ----------
    p3_channels : int
        FastSAM P3 特征通道数 (默认 960) | P3 feature channels (default 960).
    p4_channels : int
        FastSAM P4 特征通道数 (默认 1280) | P4 feature channels (default 1280).
    proto_dim : int
        Proto mask 数量 (默认 32) | Number of proto masks (default 32).
    hidden_dim : int
        系数预测器隐藏层维度 (默认 256) | Coefficient predictor hidden dim.
    """

    def __init__(
        self,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        proto_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.proto_dim = proto_dim

        # ═══════════════════════════════════════════════════════════
        # 1. 系数预测器 | Coefficient Predictor (~427K)
        # ═══════════════════════════════════════════════════════════
        # 输入: P8-derived support_proto [p4_channels] (P8=1280=P4, 维度兼容)
        # Input: P8-derived support_proto [p4_channels] (P8=1280=P4, dim compatible)
        self.coeff_predictor = ProtoCoeffPredictor(
            proto_dim=proto_dim,
            feat_dim=p4_channels,  # P8=1280=P4
            hidden_dim=hidden_dim,
        )

        # ═══════════════════════════════════════════════════════════
        # 2. P4 精炼路径 | P4 Refinement Path (~715K)
        # ═══════════════════════════════════════════════════════════
        # 主力解码层: 1280 → 256 → 128 → 64 → 1 (logit)
        # Main decoder: semantic-spatial balance, workhorse
        self.feat_proj = nn.Sequential(
            nn.Conv2d(p4_channels, 256, kernel_size=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )

        self.feat_refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.p4_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # ═══════════════════════════════════════════════════════════
        # 3. P3 精炼路径 | P3 Refinement Path (~289K)
        # ═══════════════════════════════════════════════════════════
        # 高分辨率边界恢复: 960 → 128 → 64 → 64 → 1 (logit)
        # High-res boundary recovery: 960 → 128 → 64 → 64 → 1 (logit)
        self.p3_proj = nn.Sequential(
            nn.Conv2d(p3_channels, 128, kernel_size=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        self.p3_refine = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.p3_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # ═══════════════════════════════════════════════════════════
        # 4. 三路融合层 | Three-Way Fusion Layer (~5K)
        # ═══════════════════════════════════════════════════════════
        # 输入: proto_mask + p3_logit + p4_logit → [3, H/4, W/4]
        # 学习三路权重，自动决定依赖程度
        # Learns to weight three paths automatically
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重 (跳过 ProtoCoeffPredictor, 其自行初始化)
        Initialize weights (skip ProtoCoeffPredictor, self-initializing).
        """
        for name, module in self.named_modules():
            if 'coeff_predictor' in name:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        p3_features: torch.Tensor,
        p4_features: torch.Tensor,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param p3_features: [1, 960, H/8, W/8] FastSAM P3 特征 (高分辨率细节).
        :param p4_features: [1, 1280, H/16, W/16] FastSAM P4 特征 (语义-空间平衡).
        :param proto_masks: [32, H/4, W/4] 或 [1, 32, H/4, W/4] Proto basis masks.
        :param support_proto: [1280] 或 [1, 1280] P8-derived L2-normalized support prototype.
        :return: [H/4, W/4] 实例掩码 (sigmoid, ∈ [0, 1]).
        """
        # ── 输入标准化 | Input normalization ──
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)  # [1, 32, H/4, W/4] → [32, H/4, W/4]
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)  # [1, 1280] → [1280]

        # ═══════════════════════════════════════════════════════════
        # Step 1: Proto 系数预测 → 粗掩码
        # Proto coefficient prediction → coarse mask
        # ═══════════════════════════════════════════════════════════
        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))  # [1, 32]
        proto_mask = self.coeff_predictor.generate_mask(
            coeffs, proto_masks
        )  # [1, H/4, W/4] — generate_mask returns [N, H, W]

        H_out, W_out = proto_masks.shape[1:]  # H/4, W/4

        # ═══════════════════════════════════════════════════════════
        # Step 2: P4 路径 — 主力解码 | P4 Path — Main Decoder
        # ═══════════════════════════════════════════════════════════
        p4_proj = self.feat_proj(p4_features)     # [1, 256, H/16, W/16]
        p4_ref = self.feat_refine(p4_proj)         # [1, 64, H/16, W/16]
        p4_logit = self.p4_head(p4_ref)            # [1, 1, H/16, W/16]
        p4_up = F.interpolate(
            p4_logit, size=(H_out, W_out),
            mode='bilinear', align_corners=False,
        )  # [1, 1, H/4, W/4]

        # ═══════════════════════════════════════════════════════════
        # Step 3: P3 路径 — 边界细化 | P3 Path — Boundary Refinement
        # ═══════════════════════════════════════════════════════════
        p3_proj = self.p3_proj(p3_features)        # [1, 128, H/8, W/8]
        p3_ref = self.p3_refine(p3_proj)            # [1, 64, H/8, W/8]
        p3_logit = self.p3_head(p3_ref)             # [1, 1, H/8, W/8]
        p3_up = F.interpolate(
            p3_logit, size=(H_out, W_out),
            mode='bilinear', align_corners=False,
        )  # [1, 1, H/4, W/4]

        # ═══════════════════════════════════════════════════════════
        # Step 4: 三路融合 | Three-Way Fusion
        # ═══════════════════════════════════════════════════════════
        # proto_mask: 类别形状先验 (全局, 粗粒度) — [1, H/4, W/4] → [1, 1, H/4, W/4]
        # p3_up:      高分辨边界细节 (局部, 细粒度) — [1, 1, H/4, W/4]
        # p4_up:      语义-空间平衡 (主力) — [1, 1, H/4, W/4]
        proto_mask_4d = proto_mask.unsqueeze(1) if proto_mask.dim() == 3 else proto_mask
        fused = torch.cat([proto_mask_4d, p3_up, p4_up], dim=1)  # [1, 3, H/4, W/4]
        final_logit = self.fusion(fused)                       # [1, 1, H/4, W/4]
        final_mask = torch.sigmoid(final_logit)                # [1, 1, H/4, W/4]

        return final_mask.squeeze(0)  # [H/4, W/4] — 与 AdaptiveSparseDecoder 接口一致

    def forward_with_proto_only(
        self,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """
        仅使用 proto mask 生成 (无 P3/P4 精炼) | Proto-only mask generation.
        用于快速推理或消融实验 | For fast inference or ablation.

        :param proto_masks: [32, H/4, W/4] proto basis masks.
        :param support_proto: [1280] support prototype.
        :return: [H/4, W/4] instance mask in [0, 1].
        """
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))
        return self.coeff_predictor.generate_mask(coeffs, proto_masks).squeeze(0)

    def get_submodule_params(self) -> dict[str, int]:
        """
        获取各子模块参数量 | Get parameter count per submodule.

        :return: {submodule_name: param_count} 映射.
        """
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "coeff_predictor": _count(self.coeff_predictor),
            "p4_feat_proj": _count(self.feat_proj),
            "p4_feat_refine": _count(self.feat_refine),
            "p4_head": _count(self.p4_head),
            "p3_proj": _count(self.p3_proj),
            "p3_refine": _count(self.p3_refine),
            "p3_head": _count(self.p3_head),
            "fusion": _count(self.fusion),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"AdaptiveDecoderP3P4(p3_ch={self.p3_channels}, "
                f"p4_ch={self.p4_channels}, proto={self.proto_dim}, "
                f"total_params={params['total']/1e3:.1f}K)")
