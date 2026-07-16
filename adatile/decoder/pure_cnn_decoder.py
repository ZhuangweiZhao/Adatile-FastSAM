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

    支持二值和多类别输出 | Supports binary and multi-class output:
        out_channels=1 → sigmoid → binary mask [H, W]
        out_channels=C → softmax → multi-class mask [C, H, W]

    Parameters
    ----------
    in_channels : int
        P4 特征通道数 (FastSAM: 640 或 1280) | P4 feature channels.
    out_channels : int
        输出通道数: 1=二值分割, C=多类别 | Output channels: 1=binary, C=multi-class.
    """

    def __init__(self, in_channels: int = 640, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
            nn.Conv2d(32, out_channels, kernel_size=1),
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
        :return: out_channels=1 → [H/4, W/4] sigmoid mask ∈ [0,1].
                 out_channels=C → [C, H/4, W/4] softmax probabilities.
        """
        # ── P4 精炼 | P4 Refinement ──
        x = self.feat_proj(p4_features)     # [B, 256, H/16, W/16]
        x = self.feat_refine(x)              # [B, 64, H/16, W/16]
        logit = self.mask_head(x)            # [B, out_channels, H/16, W/16]

        # ── 上采样到 H/4 | Upsample to stride-4 ──
        logit_up = F.interpolate(
            logit, scale_factor=4, mode='bilinear', align_corners=False
        )  # [B, out_channels, H/4, W/4]

        if self.out_channels == 1:
            mask = torch.sigmoid(logit_up)
            return mask.squeeze(0)  # [H/4, W/4]
        else:
            # 多类别: softmax across channel dim | Multi-class: softmax across channels
            mask = torch.softmax(logit_up, dim=1)
            return mask.squeeze(0)  # [C, H/4, W/4]

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

    支持二值和多类别输出 | Supports binary and multi-class output:
        out_channels=1 → sigmoid → binary mask [H, W]
        out_channels=C → softmax → multi-class mask [C, H, W]

    Parameters
    ----------
    p3_channels : int
        P3 特征通道数 (FastSAM: 320 或 960) | P3 feature channels.
    p4_channels : int
        P4 特征通道数 (FastSAM: 640 或 1280) | P4 feature channels.
    out_channels : int
        输出通道数: 1=二值分割, C=多类别 | Output channels: 1=binary, C=multi-class.
    """

    def __init__(self, p3_channels: int = 320, p4_channels: int = 640, out_channels: int = 1):
        super().__init__()
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.out_channels = out_channels

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
            nn.Conv2d(32, out_channels, kernel_size=1),
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
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        # ── 双路融合 | Two-Way Fusion ──
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
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
        :return: out_channels=1 → [H/4, W/4] sigmoid mask.
                 out_channels=C → [C, H/4, W/4] softmax probabilities.
        """
        # ── P4 路径: H/16 → H/4 | P4 Path: H/16 → H/4 ──
        p4_x = self.p4_proj(p4_features)     # [B, 256, H/16, W/16]
        p4_x = self.p4_refine(p4_x)           # [B, 64, H/16, W/16]
        p4_logit = self.p4_head(p4_x)         # [B, out_channels, H/16, W/16]

        # ── P3 路径: H/8 → H/4 | P3 Path: H/8 → H/4 ──
        p3_x = self.p3_proj(p3_features)      # [B, 128, H/8, W/8]
        p3_x = self.p3_refine(p3_x)            # [B, 64, H/8, W/8]
        p3_logit = self.p3_head(p3_x)          # [B, out_channels, H/8, W/8]

        # ── 统一上采样到 H/4 | Unified upsample to H/4 ──
        H_out = p3_logit.shape[2] * 2  # P3 is H/8, target H/4
        W_out = p3_logit.shape[3] * 2

        p4_up = F.interpolate(p4_logit, size=(H_out, W_out),
                              mode='bilinear', align_corners=False)  # [B, out_channels, H/4, W/4]
        p3_up = F.interpolate(p3_logit, size=(H_out, W_out),
                              mode='bilinear', align_corners=False)  # [B, out_channels, H/4, W/4]

        # ── 双路融合 | Two-Way Fusion ──
        fused = torch.cat([p3_up, p4_up], dim=1)  # [B, out_channels*2, H/4, W/4]
        final_logit = self.fusion(fused)            # [B, out_channels, H/4, W/4]

        if self.out_channels == 1:
            mask = torch.sigmoid(final_logit)
        else:
            mask = torch.softmax(final_logit, dim=1)

        return mask.squeeze(0)  # [out_channels, H/4, W/4]

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


# ═══════════════════════════════════════════════════════════════════
# PureDecoderP2P3P4 — P2+P3+P4 多尺度 BiFPN 解码器
# Multi-scale BiFPN decoder fusing P2 (stride 4) + P3 (stride 8) + P4 (stride 16)
# ═══════════════════════════════════════════════════════════════════

class PureDecoderP2P3P4(nn.Module):
    """
    P2+P3+P4 多尺度解码器 (BiFPN 可学习加权融合) | Multi-scale Decoder with BiFPN fusion.

    将 P2/P3/P4 三个尺度特征投影到统一维度, 使用 BiFPN 风格的
    可学习权重融合, 在 P2 (H/4) 分辨率输出。
    Projects P2/P3/P4 to common dim, fuses with BiFPN-style learnable weights,
    outputs at P2 (H/4) resolution.

    设计 | Design:
        P2 (H/4, fine detail) ──┐
        P3 (H/8, structure) ────┼── BiFPN weighted fusion → Refine → Mask Head
        P4 (H/16, semantic) ───┘

    BiFPN 融合 | BiFPN Fusion:
        fused = Σ(w_i · ReLU · f_i) / Σ(w_i · ReLU)
        每个尺度有独立可学习权重, ReLU 确保非负, 归一化防止数值爆炸。
        Each scale has independent learnable weight, ReLU ensures non-negativity,
        normalization prevents explosion.

    Parameters
    ----------
    p2_channels : int
        P2 层通道数 (FastSAM-x 典型值 ~320) | P2 channels.
    p3_channels : int
        P3 层通道数 (FastSAM-x 典型值 960) | P3 channels.
    p4_channels : int
        P4 层通道数 (FastSAM-x 典型值 1280) | P4 channels.
    out_channels : int
        输出类别数 | Output classes (4 for multi-class NEU_Seg).
    mid_channels : int
        融合中间通道数 | Fusion mid channels (default 128).
    """

    def __init__(
        self,
        p2_channels: int = 320,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        out_channels: int = 4,
        mid_channels: int = 128,
    ) -> None:
        super().__init__()
        self.p2_channels = p2_channels
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        # ── 尺度投影 | Scale Projectors (1×1 Conv → IN → ReLU) ──
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(p2_channels, mid_channels, 1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(p3_channels, mid_channels, 1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(p4_channels, mid_channels, 1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── BiFPN 可学习融合权重 | Learnable Fusion Weights ──
        # 初始化为 1.0, ReLU 保证非负 | Init to 1.0, ReLU ensures non-neg
        self.w_p2 = nn.Parameter(torch.ones(1))
        self.w_p3 = nn.Parameter(torch.ones(1))
        self.w_p4 = nn.Parameter(torch.ones(1))

        # ── 融合后精炼 | Post-fusion Refinement ──
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── Mask Head (在 P2 分辨率 H/4) ──
        self.mask_head = nn.Sequential(
            nn.Conv2d(mid_channels, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(
        self,
        p2: torch.Tensor,   # [1, C2, H/4, W/4]
        p3: torch.Tensor,   # [1, C3, H/8, W/8]
        p4: torch.Tensor,   # [1, C4, H/16, W/16]
        freq_fusion: torch.nn.Module | None = None,  # 可选频率引导融合
    ) -> torch.Tensor:
        """
        多尺度融合前向 (BiFPN 或 Frequency-guided) | Multi-scale fusion forward.

        :param freq_fusion: 可选 FrequencyGuidedFusion, 提供时替代 BiFPN 权重。
            Optional FrequencyGuidedFusion, replaces BiFPN weights when provided.
        :return: [H/4, W/4] (binary, sigmoid) 或 [C, H/4, W/4] (multi-class, softmax).
        """
        # ── 投影到统一维度 | Project to common dim ──
        f2 = self.proj_p2(p2)  # [B, C_mid, H/4, W/4]
        f3 = self.proj_p3(p3)  # [B, C_mid, H/8, W/8]
        f4 = self.proj_p4(p4)  # [B, C_mid, H/16, W/16]

        # ── 上采样到 P2 分辨率 | Upsample to P2 resolution ──
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        f4_up = F.interpolate(f4, size=f2.shape[2:], mode="bilinear", align_corners=False)

        # ── 融合 | Fusion ──
        if freq_fusion is not None:
            # 频率引导的动态权重 | Frequency-guided dynamic weights
            w2, w3, w4 = freq_fusion(f2, f3, f4)  # [B, 1] each
            w2 = w2.view(-1, 1, 1, 1)
            w3 = w3.view(-1, 1, 1, 1)
            w4 = w4.view(-1, 1, 1, 1)
            w_sum = w2 + w3 + w4 + 1e-4
        else:
            # BiFPN 可学习权重 | Learnable weights
            w2 = torch.relu(self.w_p2)
            w3 = torch.relu(self.w_p3)
            w4 = torch.relu(self.w_p4)
            w_sum = w2 + w3 + w4 + 1e-4
        fused = (w2 * f2 + w3 * f3_up + w4 * f4_up) / w_sum

        # ── 精炼 + Mask Head | Refine + Mask Head ──
        refined = self.refine(fused)
        logit = self.mask_head(refined)  # [B, out_channels, H/4, W/4]

        if self.out_channels == 1:
            return torch.sigmoid(logit).squeeze(0)        # [H/4, W/4]
        else:
            return torch.softmax(logit, dim=1).squeeze(0)  # [C, H/4, W/4]

    # ── 参数量统计 | Parameter Count ──
    def get_submodule_params(self) -> dict[str, int]:
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "proj_p2": _count(self.proj_p2),
            "proj_p3": _count(self.proj_p3),
            "proj_p4": _count(self.proj_p4),
            "refine": _count(self.refine),
            "mask_head": _count(self.mask_head),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (f"PureDecoderP2P3P4(p2={self.p2_channels}, p3={self.p3_channels}, "
                f"p4={self.p4_channels}, mid={self.mid_channels}, "
                f"params={params['total']/1e3:.1f}K)")
