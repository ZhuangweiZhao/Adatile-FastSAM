"""
纯 CNN 解码器 (无 Prototype) | Pure CNN Decoder (No Prototype).
===============================================================

基于原型消融发现：ProtoCoeffPredictor 通路功能上已死亡 (Zero/Random proto = Normal)。
完全移除 prototype conditioning，纯 query-driven instance segmentation。
原型 ablation found ProtoCoeffPredictor pathway functionally dead → remove entirely.

设计理念 | Design Philosophy:
    "The model discovers that query-driven refinement suffices.
     Architectural self-pruning — prototype pathway becomes vestigial."

架构 | Architecture:
    PureDecoder (P4-only, ~553K):
        P4 [640, H/16, W/16] → proj → refine → mask_head → upsample → sigmoid → mask

    PureDecoderP3P4 (P3+P4, ~840K):
        P3 [320, H/8, W/8]  → p3_proj → p3_refine → p3_head → upsample → [H/4]
        P4 [640, H/16, W/16] → p4_proj → p4_refine → p4_head → upsample → [H/4]
        concat → fusion → sigmoid → mask

对比 AdaptiveSparseDecoder | vs AdaptiveSparseDecoder:
    - 移除 ProtoCoeffPredictor (427K 死参数) | Remove ProtoCoeffPredictor (427K dead params)
    - 移除 proto_mask 融合 | Remove proto_mask fusion
    - 移除 support_proto 输入 — 纯 query-driven | Remove support_proto input
    - 参数量减少 ~44% | Parameter count reduced ~44%

用法 | Usage::

    from adatile.decoder.pure_cnn_decoder import PureDecoder

    decoder = PureDecoder(in_channels=640)
    mask = decoder(p4_features)  # [1, H/4, W/4] — no support needed!
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureDecoder(nn.Module):
    """
    纯 P4 CNN 解码器 | Pure P4 CNN Decoder.

    无 prototype, 无 conditioning — 纯 query-driven 实例分割。
    No prototype, no conditioning — pure query-driven instance segmentation.

    P4 features → CNN refine → per-pixel FG logit → sigmoid → mask.

    Parameters
    ----------
    in_channels : int
        P4 特征通道数 (FastSAM: 640 或 1280) | P4 feature channels.
    """

    def __init__(self, in_channels: int = 640):
        super().__init__()
        self.in_channels = in_channels

        # ── 特征投影 (降维) | Feature Projection (channel reduction) ──
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── 特征精炼 | Feature Refinement ──
        self.feat_refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── 掩码头 | Mask Head ──
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化 | Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p4_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param p4_features: [B, in_channels, H/16, W/16] P4 特征.
        :return: [H/4, W/4] 实例 mask (sigmoid, ∈ [0,1]).
        """
        # ── P4 精炼 | P4 Refinement ──
        x = self.feat_proj(p4_features)     # [B, 256, H/16, W/16]
        x = self.feat_refine(x)              # [B, 64, H/16, W/16]
        logit = self.mask_head(x)            # [B, 1, H/16, W/16]

        # ── 上采样到 H/4 | Upsample to stride-4 ──
        logit_up = F.interpolate(
            logit, scale_factor=4, mode='bilinear', align_corners=False
        )  # [B, 1, H/4, W/4]

        mask = torch.sigmoid(logit_up)
        return mask.squeeze(0)  # [H/4, W/4]

    def get_submodule_params(self) -> dict[str, int]:
        """各子模块参数量 | Parameter count per submodule."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "feat_proj": _count(self.feat_proj),
            "feat_refine": _count(self.feat_refine),
            "mask_head": _count(self.mask_head),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"PureDecoder(in={self.in_channels}, "
                f"params={params['total']/1e3:.1f}K)")


class PureDecoderP3P4(nn.Module):
    """
    纯 P3+P4 CNN 解码器 | Pure P3+P4 CNN Decoder.

    双尺度 query-driven 实例分割: P3 (高分辨率边界) + P4 (语义-空间平衡)。
    Dual-scale query-driven instance segmentation: P3 (high-res boundary) + P4 (semantic-spatial).

    与 PureDecoder 的区别: 新增 P3 路径做边界细化。
    Difference from PureDecoder: adds P3 pathway for boundary refinement.

    Parameters
    ----------
    p3_channels : int
        P3 特征通道数 (FastSAM: 320 或 960) | P3 feature channels.
    p4_channels : int
        P4 特征通道数 (FastSAM: 640 或 1280) | P4 feature channels.
    """

    def __init__(self, p3_channels: int = 320, p4_channels: int = 640):
        super().__init__()
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels

        # ── P4 路径 (主力) | P4 Path (workhorse) ──
        self.p4_proj = nn.Sequential(
            nn.Conv2d(p4_channels, 256, kernel_size=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )
        self.p4_refine = nn.Sequential(
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

        # ── P3 路径 (边界细化) | P3 Path (boundary refinement) ──
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

        # ── 双路融合 | Two-Way Fusion ──
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化 | Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p3_features: torch.Tensor,
                p4_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param p3_features: [B, p3_channels, H/8, W/8] P3 特征.
        :param p4_features: [B, p4_channels, H/16, W/16] P4 特征.
        :return: [H/4, W/4] 实例 mask (sigmoid, ∈ [0,1]).
        """
        # ── P4 路径: H/16 → H/4 | P4 Path: H/16 → H/4 ──
        p4_x = self.p4_proj(p4_features)     # [B, 256, H/16, W/16]
        p4_x = self.p4_refine(p4_x)           # [B, 64, H/16, W/16]
        p4_logit = self.p4_head(p4_x)         # [B, 1, H/16, W/16]

        # ── P3 路径: H/8 → H/4 | P3 Path: H/8 → H/4 ──
        p3_x = self.p3_proj(p3_features)      # [B, 128, H/8, W/8]
        p3_x = self.p3_refine(p3_x)            # [B, 64, H/8, W/8]
        p3_logit = self.p3_head(p3_x)          # [B, 1, H/8, W/8]

        # ── 统一上采样到 H/4 | Unified upsample to H/4 ──
        H_out = p3_logit.shape[2] * 2  # P3 is H/8, target H/4
        W_out = p3_logit.shape[3] * 2

        p4_up = F.interpolate(p4_logit, size=(H_out, W_out),
                              mode='bilinear', align_corners=False)  # [B, 1, H/4, W/4]
        p3_up = F.interpolate(p3_logit, size=(H_out, W_out),
                              mode='bilinear', align_corners=False)  # [B, 1, H/4, W/4]

        # ── 双路融合 | Two-Way Fusion ──
        fused = torch.cat([p3_up, p4_up], dim=1)  # [B, 2, H/4, W/4]
        final_logit = self.fusion(fused)            # [B, 1, H/4, W/4]
        mask = torch.sigmoid(final_logit)

        return mask.squeeze(0)  # [H/4, W/4]

    def get_submodule_params(self) -> dict[str, int]:
        """各子模块参数量 | Parameter count per submodule."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "p4_proj": _count(self.p4_proj),
            "p4_refine": _count(self.p4_refine),
            "p4_head": _count(self.p4_head),
            "p3_proj": _count(self.p3_proj),
            "p3_refine": _count(self.p3_refine),
            "p3_head": _count(self.p3_head),
            "fusion": _count(self.fusion),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"PureDecoderP3P4(p3_ch={self.p3_channels}, p4_ch={self.p4_channels}, "
                f"params={params['total']/1e3:.1f}K)")
