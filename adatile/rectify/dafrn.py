"""
DA-FRN — Defect-Aware Feature Rectification Network.
=====================================================
缺陷感知特征校正网络 — 完整模块.
Defect-Aware Feature Rectification Network — full module.

组合 DCR + FDE + CDF 三个子模块，构成完整的"特征翻译器"。
Combines DCR + FDE + CDF sub-modules into a complete "feature translator."

架构 | Architecture::

         FastSAM P3 [B, 960, H/8, W/8]
         FastSAM P4 [B, 1280, H/16, W/16]
                │              │
                ▼              ▼
         ┌────────────┐ ┌────────────┐
         │  DCR (P3)  │ │  DCR (P4)  │  ← 通道重加权 | Channel Reweighting
         └─────┬──────┘ └─────┬──────┘
               │              │
               ▼              ▼
         ┌────────────┐ ┌────────────┐
         │  FDE (P3)  │ │  FDE (P4)  │  ← 频率增强 | Frequency Enhancement
         └─────┬──────┘ └─────┬──────┘
               │              │
               └──────┬───────┘
                      │
                      ▼
              ┌──────────────┐
              │     CDF      │          ← 动态跨尺度融合 | Dynamic Cross-scale Fusion
              └──────┬───────┘
                     │
              ┌──────┴──────┐
              │             │
         P3' [same]   P4' [same]
              │             │
              ▼             ▼
         ┌─────────────────────────┐
         │  PureDecoderP3P4 (现有)  │
         └─────────────────────────┘
                     │
                     ▼
              Mask Prediction

即插即用 | Plug-and-Play:
    DA-FRN 输出与输入相同形状的特征，可直接接入任何现有的
    P3P4 Decoder (PureDecoderP3P4, AdaptiveSparseDecoder 等)。
    DA-FRN outputs features with same shapes as input, can directly
    connect to any existing P3P4 Decoder.

模块独立性 | Module Independence:
    DCR, FDE, CDF 可独立启用/禁用 (用于消融实验)。
    DCR, FDE, CDF can be independently enabled/disabled (for ablation).

用法 | Usage::

    from adatile.rectify import DA_FRN
    from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4

    frn = DA_FRN(p3_channels=960, p4_channels=1280)
    decoder = PureDecoderP3P4(p3_channels=960, p4_channels=1280)

    p3, p4 = backbone(img)
    p3_r, p4_r = frn(p3, p4)       # Feature rectification
    mask = decoder(p3_r, p4_r)      # Decode as usual

参数量 | Total Params: ~80K (lightweight!)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adatile.rectify.dcr import DefectChannelReweighting
from adatile.rectify.fde import FreqDefectEnhance
from adatile.rectify.cdf import CrossScaleDefectFusion
from adatile.logging import get_logger

logger = get_logger("rectify.dafrn")


class DA_FRN(nn.Module):
    """
    缺陷感知特征校正网络 | Defect-Aware Feature Rectification Network.

    完整的即插即用特征翻译器：不修改 Backbone，只翻译特征空间。
    Complete plug-and-play feature translator: no backbone modification,
    only translates feature space.

    Parameters
    ----------
    p3_channels : int
        P3 通道数 | P3 channels.
    p4_channels : int
        P4 通道数 | P4 channels.
    dcr_reduction : int
        DCR 压缩比 | DCR reduction ratio (default 16).
    fde_kernel_sizes : tuple[int, ...]
        FDE 多尺度池化核大小 | FDE multi-scale pooling kernel sizes.
        Default: (3, 7, 15) — fine texture, mid texture, BG suppression.
    cdf_hidden : int
        CDF 权重预测 MLP 隐层维度 | CDF weight predictor MLP hidden dim (default 64).
    enable_dcr : bool
        启用 DCR | Enable DCR.
    enable_fde : bool
        启用 FDE | Enable FDE.
    enable_cdf : bool
        启用 CDF | Enable CDF.
    """

    def __init__(
        self,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        dcr_reduction: int = 16,
        fde_kernel_sizes: tuple[int, ...] = (3, 7, 15),
        fde_alpha_init: float = 0.5,
        cdf_hidden: int = 64,
        enable_dcr: bool = True,
        enable_fde: bool = True,
        enable_cdf: bool = True,
    ) -> None:
        super().__init__()
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.enable_dcr = enable_dcr
        self.enable_fde = enable_fde
        self.enable_cdf = enable_cdf
        self.fde_alpha_init = fde_alpha_init

        # ── DCR: 通道重加权 | Channel Reweighting ──
        if enable_dcr:
            self.dcr_p3 = DefectChannelReweighting(p3_channels, reduction=dcr_reduction)
            self.dcr_p4 = DefectChannelReweighting(p4_channels, reduction=dcr_reduction)
        else:
            self.dcr_p3 = None
            self.dcr_p4 = None

        # ── FDE: 频率增强 | Frequency Enhancement ──
        if enable_fde:
            self.fde_p3 = FreqDefectEnhance(
                p3_channels, kernel_sizes=fde_kernel_sizes, alpha_init=fde_alpha_init,
            )
            self.fde_p4 = FreqDefectEnhance(
                p4_channels, kernel_sizes=fde_kernel_sizes, alpha_init=fde_alpha_init,
            )
        else:
            self.fde_p3 = None
            self.fde_p4 = None

        # ── CDF: 跨尺度动态融合 | Cross-scale Dynamic Fusion ──
        if enable_cdf:
            self.cdf = CrossScaleDefectFusion(p3_channels, p4_channels, hidden_dim=cdf_hidden)
        else:
            self.cdf = None

        # ── 统计 | Statistics ──
        total_params = sum(p.numel() for p in self.parameters())
        logger.log_info(
            "dafrn/init",
            f"DA-FRN(p3={p3_channels}, p4={p4_channels}, "
            f"DCR={enable_dcr}, FDE={enable_fde}, CDF={enable_cdf}): "
            f"{total_params/1e3:.1f}K params",
        )

    def forward(
        self,
        p3_features: torch.Tensor,
        p4_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        特征校正前向 | Feature rectification forward.

        :param p3_features: [B, C3, H/8, W/8] P3 特征.
        :param p4_features: [B, C4, H/16, W/16] P4 特征.
        :return: (p3_rectified, p4_rectified) — 形状不变 | same shape.
        """
        p3 = p3_features
        p4 = p4_features

        # ── 阶段 1: DCR — 通道重加权 | Stage 1: DCR — Channel Reweighting ──
        if self.enable_dcr:
            p3 = self.dcr_p3(p3)
            p4 = self.dcr_p4(p4)

        # ── 阶段 2: FDE — 频率增强 | Stage 2: FDE — Frequency Enhancement ──
        if self.enable_fde:
            p3 = self.fde_p3(p3)
            p4 = self.fde_p4(p4)

        # ── 阶段 3: CDF — 跨尺度动态融合 | Stage 3: CDF — Cross-scale Dynamic Fusion ──
        if self.enable_cdf:
            p3, p4 = self.cdf(p3, p4)

        return p3, p4

    def get_stats(self) -> dict:
        """
        获取所有子模块统计 (用于日志/可视化) | Get all submodule stats (for logging/vis).

        :return: dict with per-module stats.
        """
        stats = {
            "dcr_enabled": self.enable_dcr,
            "fde_enabled": self.enable_fde,
            "cdf_enabled": self.enable_cdf,
        }

        if self.enable_cdf and self.cdf is not None:
            stats.update(self.cdf.get_weight_stats())

        return stats

    def get_submodule_params(self) -> dict[str, int]:
        """各子模块参数量 | Parameter count per submodule."""

        def _count(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0

        return {
            "dcr_p3": _count(self.dcr_p3),
            "dcr_p4": _count(self.dcr_p4),
            "fde_p3": _count(self.fde_p3),
            "fde_p4": _count(self.fde_p4),
            "cdf": _count(self.cdf),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"DA_FRN(p3={self.p3_channels}, p4={self.p4_channels}, "
                f"DCR={self.enable_dcr}, FDE={self.enable_fde}, CDF={self.enable_cdf}, "
                f"params={params['total']/1e3:.1f}K)")
