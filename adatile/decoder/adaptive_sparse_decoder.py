"""
自适应稀疏解码器 | Adaptive Sparse Decoder.
=============================================

统一的少样本实例分割解码器，整合三大核心创新：
Unified few-shot instance segmentation decoder integrating three core innovations:

1. **FDR (Foreground Density Router)**: tile-level sparse routing
   → P8 features → density map → tile importance → Top-K tile selection

2. **ProtoCoeffPredictor**: few-shot conditioned mask generation
   → Support prototype → 32-d mask coefficients → sigmoid(coeffs @ proto_masks)

3. **Feature Refinement**: P4-based mask quality improvement
   → P4 features → CNN refine → fuse with proto mask → final instance mask

架构总览 | Architecture Overview::

    Support Image ──→ FG Prototype [1280]
                           │
                    ┌──────▼──────────┐
                    │ ProtoCoeff       │
                    │ Predictor (~400K)│
                    │ proto → coeffs   │
                    └──────┬──────────┘
                           │ coeffs [32]
                    ┌──────▼──────────┐
    Query Image ────│                 │
    → Backbone      │  Mask Generator │
    → Proto [32,H,W]│  coeffs@proto   │
    → P4 [1280,H/16]│  → coarse mask  │
    → P8 [1280,H/32]│                 │
         │          └─────────────────┘
         │                   │
    ┌────▼────┐        ┌─────▼──────┐
    │ FDR     │        │ P4 Refine  │
    │ P8→density│      │ CNN        │
    │ →gate   │───────→│ →fine mask │
    └─────────┘        └─────┬──────┘
                             │
                      ┌──────▼──────┐
                      │  Final Mask │
                      │  [H, W]     │
                      └─────────────┘

用法 | Usage::

    from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder

    decoder = AdaptiveSparseDecoder(in_channels=1280, proto_dim=32)
    mask = decoder(
        p4_features=p4,            # [B, 1280, H/16, W/16]
        proto_masks=proto,         # [32, H/4, W/4]
        support_proto=prototype,   # [1280]
        fdr_map=fdr_density,       # [1, H/32, W/32] (optional)
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor


class AdaptiveSparseDecoder(nn.Module):
    """
    统一的少样本实例分割自适应稀疏解码器 | Unified Adaptive Sparse Decoder.

    整合三大创新 | Three Innovations Integrated:
    1. FDR → tile-level sparse routing（通过 gating 机制引导计算聚焦）
    2. ProtoCoeffPredictor → few-shot conditioned mask generation
    3. Feature refinement → P4-based mask quality improvement

    Parameters
    ----------
    in_channels : int
        FastSAM P4 特征通道数 | FastSAM P4 feature channels (default=1280).
    proto_dim : int
        Proto mask 数量 | Number of proto masks (default=32).
    hidden_dim : int
        系数预测器隐藏层维度 | Coefficient predictor hidden dimension.
    use_fdr : bool
        是否启用 FDR 引导的注意力 | Whether to enable FDR-guided attention.
    """

    def __init__(
        self,
        in_channels: int = 1280,
        proto_dim: int = 32,
        hidden_dim: int = 256,
        use_fdr: bool = True,
        normalize_proto: str = "none",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.proto_dim = proto_dim
        self.use_fdr = use_fdr
        # ── proto basis 归一化 (修复未约束 basis 幅值导致的 sigmoid 饱和) | proto-basis normalization ──
        #    none=identity(默认,零影响); l2=逐 basis 单位 L2; layernorm=逐 basis 标准化; scale=固定缩放
        self.normalize_proto = normalize_proto
        # ── 诊断: forward 时收集前向统计 (默认关闭) | optional forward-stat collection ──
        self.collect_stats = False
        self.last_stats: dict = {}

        # ═══════════════════════════════════════════════════════════
        # 1. 系数预测器 | Coefficient Predictor (~400K params)
        # ═══════════════════════════════════════════════════════════
        self.coeff_predictor = ProtoCoeffPredictor(
            proto_dim=proto_dim,
            feat_dim=in_channels,
            hidden_dim=hidden_dim,
        )

        # ═══════════════════════════════════════════════════════════
        # 2. P4 特征精炼 | P4 Feature Refinement
        # ═══════════════════════════════════════════════════════════
        # 将 P4 特征压缩为精炼特征，用于提升 proto mask 的细节质量
        # Compress P4 features to refined features for improving proto mask detail
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),  # InstanceNorm: bs=1 安全 | bs=1 safe
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

        # ═══════════════════════════════════════════════════════════
        # 3. FDR 引导的注意力门控 | FDR-Guided Attention Gate
        # ═══════════════════════════════════════════════════════════
        # 将 FDR 密度图上采样后与精炼特征融合，生成空间注意力门控
        # Upsample FDR density map, fuse with refined features, generate spatial attention gate
        if use_fdr:
            self.fdr_gate = nn.Sequential(
                nn.Conv2d(64 + 1, 32, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        # ═══════════════════════════════════════════════════════════
        # 4. 掩码头 | Mask Head
        # ═══════════════════════════════════════════════════════════
        # 将精炼特征映射为逐像素 FG logit
        # Map refined features to per-pixel FG logit
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # ── proto basis 归一化模块 (仅 layernorm 模式需可学习参数) | learnable norm (layernorm only) ──
        #    逐 basis 零均值单位方差 + 可学习仿射; InstanceNorm2d 对空间尺寸无关 (size-agnostic layernorm-per-basis)
        self.proto_norm = (nn.InstanceNorm2d(proto_dim, affine=True)
                           if normalize_proto == "layernorm" else None)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化非 ProtoCoeffPredictor 的权重 | Initialize non-Predictor weights."""
        for name, module in self.named_modules():
            if 'coeff_predictor' in name:
                continue  # ProtoCoeffPredictor handles its own init
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_proto(self, proto_masks: torch.Tensor) -> torch.Tensor:
        """归一化 proto basis 幅值, 防止 coeffs@proto 进入 sigmoid 饱和死区。
        Normalize the proto-basis magnitude to keep coeffs@proto out of the sigmoid dead zone.

        根因: 解冻微调使 FastSAM proto basis 幅值爆炸 (523→1e11), 无归一化时 pre-sigmoid→1e9 → 全饱和
        → ∂L/∂coeffs=0。本方法把 basis 幅值约束回可训练范围。
        Root cause: unfrozen fine-tuning explodes the proto basis (523→1e11); without normalization the
        pre-sigmoid hits ~1e9 → full saturation → zero gradient. This bounds the basis magnitude.

        :param proto_masks: [proto_dim, H, W] (已 squeeze) | squeezed proto basis.
        :return: 同形状归一化结果 | normalized, same shape.
        """
        mode = self.normalize_proto
        if mode == "none":
            return proto_masks
        C, H, W = proto_masks.shape
        if mode == "l2":
            # 每个 basis 单位 L2 (over spatial) → ‖row‖=1 → pre-sigmoid 受 ‖coeffs‖ 约束
            flat = F.normalize(proto_masks.reshape(C, -1), p=2, dim=1)
            return flat.view(C, H, W)
        if mode == "scale":
            # 朴素固定缩放 (预期弱于自适应归一化, 作对照行) | naive fixed scale (control row)
            return proto_masks / (float(H * W) ** 0.5)
        if mode == "layernorm":
            # 逐 basis 零均值单位方差 + 可学习仿射 (InstanceNorm2d = size-agnostic)
            return self.proto_norm(proto_masks.unsqueeze(0)).squeeze(0)
        raise ValueError(f"unknown normalize_proto: {mode}")

    def forward(
        self,
        p4_features: torch.Tensor,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
        fdr_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param p4_features: [B, in_channels, H/16, W/16] FastSAM P4 特征.
        :param proto_masks: [proto_dim, H/4, W/4] 或 [1, proto_dim, H/4, W/4] FastSAM proto masks.
            Proto masks 共享于所有 query 图像，冻结不训练。
            Proto masks shared across query images, frozen (not trained).
        :param support_proto: [in_channels] 或 [1, in_channels] L2-normalized support prototype.
        :param fdr_map: [1, 1, H/32, W/32] 或 None. FDR 密度图 (可选).
            FDR density map (optional). When None, FDR gating is skipped.
        :return: [H, W] 实例掩码 in [0, 1] | Instance mask in [0, 1].
        """
        # ── 输入标准化 | Input normalization ──
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)  # [1, 32, H, W] → [32, H, W]
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        # ── proto basis 归一化 (防饱和; 默认 none=identity, 零影响) | normalize proto basis ──
        proto_masks = self._normalize_proto(proto_masks)  # [1, 1280] → [1280]

        # ═══════════════════════════════════════════════════════════
        # Step 1: 从 support prototype 预测 proto mask 系数
        # Predict proto mask coefficients from support prototype
        # ═══════════════════════════════════════════════════════════
        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))  # [1, proto_dim]
        proto_mask = self.coeff_predictor.generate_mask(
            coeffs, proto_masks
        )  # [1, H/4, W/4] — coarse mask at stride 4

        # ── 可选: 收集前向统计 (机制诊断: basis 幅值/系数/饱和度) | optional forward-stat collection ──
        if self.collect_stats:
            with torch.no_grad():
                pf = proto_masks.reshape(proto_masks.shape[0], -1)
                pre = coeffs @ pf                      # pre-sigmoid logit
                pm = proto_mask.detach()
                self.last_stats = {
                    "proto_basis_l2": float(pf.norm(dim=1).mean().item()),   # post-norm 每基 L2 均值
                    "coeff_l2": float(coeffs.detach().norm().item()),
                    "pre_sigmoid_absmax": float(pre.abs().max().item()),
                    "sat_frac": float(((pm < 1e-6) | (pm > 1 - 1e-6)).float().mean().item()),
                }

        # ═══════════════════════════════════════════════════════════
        # Step 2: P4 特征精炼 | P4 Feature Refinement
        # ═══════════════════════════════════════════════════════════
        feat_proj = self.feat_proj(p4_features)   # [B, 256, H/16, W/16]
        feat_refined = self.feat_refine(feat_proj)  # [B, 64, H/16, W/16]

        # ═══════════════════════════════════════════════════════════
        # Step 3: FDR 引导的特征调制 | FDR-Guided Feature Modulation
        # ═══════════════════════════════════════════════════════════
        if self.use_fdr and fdr_map is not None:
            # 上采样 FDR 密度图到 P4 分辨率 | Upsample FDR density to P4 resolution
            fdr_up = F.interpolate(
                fdr_map, size=feat_refined.shape[2:],
                mode='bilinear', align_corners=False,
            )  # [1, 1, H/16, W/16]

            # FDR 门控: 高密度区域获得更多 attention | FDR gate: high-density regions get more attention
            gate_input = torch.cat([feat_refined, fdr_up], dim=1)  # [B, 65, H/16, W/16]
            gate = self.fdr_gate(gate_input)  # [B, 1, H/16, W/16]
            feat_refined = feat_refined * gate  # 空间注意力 | Spatial attention

        # ═══════════════════════════════════════════════════════════
        # Step 4: 生成精细掩码 | Generate Refined Mask
        # ═══════════════════════════════════════════════════════════
        refined_logit = self.mask_head(feat_refined)  # [B, 1, H/16, W/16]

        # 上采样到 proto mask 分辨率 (stride 4)
        # Upsample to proto mask resolution (stride 4)
        refined_logit_up = F.interpolate(
            refined_logit, size=proto_mask.shape[1:],
            mode='bilinear', align_corners=False,
        )  # [B, 1, H/4, W/4]

        # ═══════════════════════════════════════════════════════════
        # Step 5: 融合 Proto Mask + Refined Features
        # Fuse Proto Mask + Refined Features
        # ═══════════════════════════════════════════════════════════
        # Proto mask 提供全局形状先验（来自预训练基函数）
        # Refined logit 提供局部细节（来自 P4 特征）
        # Proto mask provides global shape prior (from pretrained basis)
        # Refined logit provides local details (from P4 features)
        final_logit = refined_logit_up.squeeze(1) + proto_mask.squeeze(0)  # [H/4, W/4]
        final_mask = torch.sigmoid(final_logit)  # [H/4, W/4]

        return final_mask

    def forward_with_proto_only(
        self,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """
        仅使用 proto mask 生成（无 P4 精炼）| Proto-only mask generation (no P4 refinement).

        用于快速推理或 ablating P4 refinement 的贡献。
        For fast inference or ablating P4 refinement contribution.

        :param proto_masks: [proto_dim, H/4, W/4] proto basis masks.
        :param support_proto: [in_channels] support prototype.
        :return: [H/4, W/4] instance mask in [0, 1].
        """
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))  # [1, proto_dim]
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
            "feat_proj": _count(self.feat_proj),
            "feat_refine": _count(self.feat_refine),
            "fdr_gate": _count(self.fdr_gate) if self.use_fdr else 0,
            "mask_head": _count(self.mask_head),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"AdaptiveSparseDecoder(in={self.in_channels}, "
                f"proto={self.proto_dim}, use_fdr={self.use_fdr}, "
                f"total_params={params['total']/1e3:.1f}K)")


# ═══════════════════════════════════════════════════════════════════
# 轻量变体：仅 ProtoCoeff (ablating P4 refinement + FDR)
# Lightweight variant: ProtoCoeff only (ablating P4 refinement + FDR)
# ═══════════════════════════════════════════════════════════════════

class ProtoOnlyDecoder(nn.Module):
    """
    仅使用 proto coefficient 预测的轻量解码器 | Lightweight decoder using only proto coefficients.

    用于消融实验：测量 P4 refinement 和 FDR gating 的独立贡献。
    For ablation studies: measuring the standalone contribution of P4 refinement and FDR gating.

    参数量: ~400K (仅 ProtoCoeffPredictor)。
    Parameter count: ~400K (ProtoCoeffPredictor only).
    """

    def __init__(
        self,
        proto_dim: int = 32,
        feat_dim: int = 1280,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.predictor = ProtoCoeffPredictor(
            proto_dim=proto_dim,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param proto_masks: [proto_dim, H, W] proto basis masks.
        :param support_proto: [feat_dim] support prototype.
        :return: [H, W] instance mask in [0, 1].
        """
        return self.predictor.predict_mask(
            support_proto.unsqueeze(0), proto_masks
        ).squeeze(0)

    def __repr__(self) -> str:
        return repr(self.predictor).replace("ProtoCoeffPredictor", "ProtoOnlyDecoder")
