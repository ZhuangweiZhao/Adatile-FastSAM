"""
Center-Affinity Instance Decoder | 中心-亲和力实例解码器.
==========================================================

将单通道语义概率图升级为三通道实例感知输出:
Upgrades single-channel semantic output to 3-channel instance-aware output:
    Channel 0: Center Heatmap — instance center likelihood
    Channel 1: dx Offset     — pixel→center x-direction
    Channel 2: dy Offset     — pixel→center y-direction

核心洞察 | Core Insight (from SOLO/CenterNet):
    Spatial location IS the instance differentiator.
    If each pixel knows its displacement to its instance center,
    grouping becomes a simple nearest-center assignment.

为什么能分离三个粘连小车 | Why It Separates 3 Adjacent Small Cars:
    Before: [H, W] FG prob → 3 touching cars → 1 merged blob → CC fails
    After:  offset[x,y] = (dx, dy) to center → each pixel votes for its center
            → 3 distinct center peaks → 3 Voronoi regions → 3 instance masks

架构 | Architecture::

    P3 [C3, H/8]  P4 [C4, H/16]
         │              │
    ┌────┴──────────────┴────┐
    │  FPN Fusion (~110K)    │  共享特征提取 | Shared feature extraction
    │  → [fpn_dim, H/8]     │
    └────────┬───────────────┘
             │
    ┌────────┼──────────────────────────┐
    │        │                          │
    ▼        ▼                          ▼
┌────────┐ ┌──────────┐    ┌────────────────────────┐
│ Center │ │ Offset   │    │ Proto Path (kept)      │
│ Head   │ │ Head     │    │ CoeffPredictor [427K]  │
│ ~18K   │ │ ~18K     │    │ support_proto → coeffs │
│ [1,H/8]│ │ [2,H/8]  │    │ coeffs@proto → FG map  │
└────┬───┘ └────┬─────┘    └───────────┬────────────┘
     │          │                      │
     │          │    ┌─────────────────┘
     │          │    │
     ▼          ▼    ▼
  Inference: find center peaks → voronoi-group pixels by offset proximity
  → N instance masks filtered by FG prior

用法 | Usage::

    from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder

    decoder = CenterAffinityDecoder(p3_channels=320, p4_channels=640, fpn_dim=64)
    center, offset, proto_mask = decoder(p3, p4, proto_masks, support_proto)
    # center:     [H/8, W/8]    center heatmap
    # offset:     [2, H/8, W/8] displacement field
    # proto_mask: [H/4, W/4]    semantic FG prior (class-conditioned)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor


class CenterAffinityDecoder(nn.Module):
    """
    中心-亲和力实例解码器 | Center-Affinity Instance Decoder.

    输出 3 通道而非 1 通道:
    - Center Heatmap [H/8, W/8]: 实例中心位置 | instance center locations
    - Offset Field [2, H/8, W/8]: 逐像素到中心的偏移 | per-pixel displacement to center
    - Proto Mask [H/4, W/4]: 语义前景 (保留, 类别条件) | semantic FG (kept, class-conditioned)

    Parameters
    ----------
    p3_channels : int
        P3 特征通道数 (auto-detected, ~320 for FastSAM-x).
    p4_channels : int
        P4 特征通道数 (auto-detected, ~640 for FastSAM-x).
    proto_dim : int
        Proto mask 数量 (默认 32).
    fpn_dim : int
        FPN 融合后通道数 (默认 64, 小以控参数量).
    normalize_proto : str
        Proto basis 归一化模式.
    """

    def __init__(
        self,
        p3_channels: int = 320,
        p4_channels: int = 640,
        proto_dim: int = 32,
        fpn_dim: int = 64,
        normalize_proto: str = "none",
    ):
        super().__init__()

        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.proto_dim = proto_dim
        self.fpn_dim = fpn_dim
        self.normalize_proto = normalize_proto

        # ── FPN: 多尺度特征融合 | Multi-scale feature fusion ──
        # P4 (H/16) 提供语义, P3 (H/8) 提供高分辨率细节
        # Lateral connections project to fpn_dim, then top-down fuse
        self.lateral_p3 = nn.Sequential(
            nn.Conv2d(p3_channels, fpn_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        self.lateral_p4 = nn.Sequential(
            nn.Conv2d(p4_channels, fpn_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        # Fuse: P3_lat + P4_up → fused features
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── Center Heatmap Head: 预测实例中心位置 | Predicts instance center locations ──
        # 输出 1 通道: 每个像素是实例中心的概率
        # GT: Gaussian kernel at each instance centroid
        self.center_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 1, kernel_size=1),
        )

        # ── Offset Head: 预测逐像素到实例中心的偏移 | Predicts per-pixel center displacement ──
        # 输出 2 通道: (dx, dy) 从当前像素到所属实例中心
        # GT: (center_x - x, center_y - y) for FG pixels, (0, 0) for BG
        self.offset_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 2, kernel_size=1),
        )

        # ── Proto Coefficient Predictor (kept, ~427K): 类别条件前景先验 ──
        self.coeff_predictor = ProtoCoeffPredictor(
            proto_dim=proto_dim,
            feat_dim=p4_channels,
            hidden_dim=256,
        )

        # ── Proto basis normalization ──
        self.proto_norm = (
            nn.InstanceNorm2d(proto_dim, affine=True)
            if normalize_proto == "layernorm"
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (skip ProtoCoeffPredictor, self-initializing)."""
        for name, module in self.named_modules():
            if "coeff_predictor" in name:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_proto(self, proto_masks: torch.Tensor) -> torch.Tensor:
        """Normalize proto basis magnitude (same logic as AdaptiveSparseDecoder)."""
        mode = self.normalize_proto
        if mode == "none":
            return proto_masks
        C, H, W = proto_masks.shape
        if mode == "l2":
            flat = F.normalize(proto_masks.reshape(C, -1), p=2, dim=1)
            return flat.view(C, H, W)
        if mode == "scale":
            return proto_masks / (float(H * W) ** 0.5)
        if mode == "layernorm":
            return self.proto_norm(proto_masks.unsqueeze(0)).squeeze(0)
        raise ValueError(f"unknown normalize_proto: {mode}")

    def _build_fpn_features(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        """
        FPN multi-scale fusion | FPN 多尺度融合.

        P3 [B, C3, H/8, W/8] + P4 [B, C4, H/16, W/16] → [B, fpn_dim, H/8, W/8]
        """
        lat_p4 = self.lateral_p4(p4)  # [B, fpn_dim, H/16, W/16]
        lat_p3 = self.lateral_p3(p3)  # [B, fpn_dim, H/8, W/8]

        # Top-down: upsample P4 to P3 resolution
        p4_up = F.interpolate(
            lat_p4, size=lat_p3.shape[2:],
            mode="bilinear", align_corners=False,
        )  # [B, fpn_dim, H/8, W/8]

        fused = torch.cat([lat_p3, p4_up], dim=1)  # [B, 2*fpn_dim, H/8, W/8]
        return self.fpn_fuse(fused)  # [B, fpn_dim, H/8, W/8]

    def forward(
        self,
        p3_features: torch.Tensor,
        p4_features: torch.Tensor,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass | 前向传播.

        :param p3_features: [B, C3, H/8, W/8] FastSAM P3 features.
        :param p4_features: [B, C4, H/16, W/16] FastSAM P4 features.
        :param proto_masks: [proto_dim, H/4, W/4] or [1, proto_dim, H/4, W/4] Proto basis.
        :param support_proto: [C4] or [1, C4] L2-normalized support prototype.
        :return: (center_heatmap, offset_field, proto_mask)
            - center_heatmap: [H/8, W/8] ∈ [0, 1] center likelihood
            - offset_field:   [2, H/8, W/8] (dx, dy) displacement per pixel
            - proto_mask:     [H/4, W/4] ∈ [0, 1] semantic FG prior (class-conditioned)
        """
        # ── Input normalization ──
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Proto mask (class-conditioned FG prior)
        # ═══════════════════════════════════════════════════════════════
        proto_masks_norm = self._normalize_proto(proto_masks)
        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))  # [1, proto_dim]
        proto_mask = self.coeff_predictor.generate_mask(
            coeffs, proto_masks_norm
        ).squeeze(0)  # [H/4, W/4]

        # ═══════════════════════════════════════════════════════════════
        # Step 2: FPN fusion (class-agnostic shared features)
        # ═══════════════════════════════════════════════════════════════
        fpn_feat = self._build_fpn_features(p3_features, p4_features)
        # → [B, fpn_dim, H/8, W/8]

        # ═══════════════════════════════════════════════════════════════
        # Step 3: Center heatmap (class-agnostic)
        # ═══════════════════════════════════════════════════════════════
        center_logit = self.center_head(fpn_feat)  # [B, 1, H/8, W/8]
        center_heatmap = torch.sigmoid(center_logit)  # [B, 1, H/8, W/8]

        # ═══════════════════════════════════════════════════════════════
        # Step 4: Offset field (class-agnostic)
        # ═══════════════════════════════════════════════════════════════
        offset_field = self.offset_head(fpn_feat)  # [B, 2, H/8, W/8]

        # Squeeze batch dim
        return (
            center_heatmap.squeeze(0).squeeze(0),  # [H/8, W/8]
            offset_field.squeeze(0),                # [2, H/8, W/8]
            proto_mask,                              # [H/4, W/4]
        )

    def forward_proto_only(
        self,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """Proto-only mask generation (for ablation / FG prior)."""
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))
        return self.coeff_predictor.generate_mask(coeffs, proto_masks).squeeze(0)

    def get_submodule_params(self) -> dict[str, int]:
        """Get parameter count per submodule."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "fpn": _count(self.lateral_p3) + _count(self.lateral_p4) + _count(self.fpn_fuse),
            "center_head": _count(self.center_head),
            "offset_head": _count(self.offset_head),
            "coeff_predictor": _count(self.coeff_predictor),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (
            f"CenterAffinityDecoder(p3_ch={self.p3_channels}, p4_ch={self.p4_channels}, "
            f"fpn_dim={self.fpn_dim}, total_params={params['total']/1e3:.1f}K)"
        )
