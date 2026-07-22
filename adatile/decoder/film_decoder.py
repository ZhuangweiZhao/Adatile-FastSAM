"""
FiLM Decoder — Prototype-Conditioned Feature Modulation | 原型条件化特征调制解码器.
=====================================================================================

将 Prototype 通过 FiLM (Feature-wise Linear Modulation) 直接调制 P4 特征，
消除"Proto 分支可被绕过"的结构缺陷。

Modulates P4 features directly via FiLM, eliminating the structural flaw
where the Proto branch can be bypassed.

架构 | Architecture::

    Prototype [1280]                    P4 Features [1280, H/16, W/16]
        │                                       │
        ├─ Linear(1280→C) → γ [C]               │
        └─ Linear(1280→C) → β [C]               │
              │                                  │
              └──────────→ FiLM ←───────────────┘
                           │
                    feat = γ · feat_proj(P4) + β
                           │
                    Refinement CNN → Mask Head
                           │
                    Final Mask [H/4, W/4]

关键创新 | Key Innovation:
    与 AdaptiveSparseDecoder 不同, FiLM Decoder 没有独立的 Proto 分支。
    Prototype 通过 γ/β 调制所有 P4 特征 → 网络无法绕过 Prototype。
    Unlike AdaptiveSparseDecoder, FiLM Decoder has NO independent Proto branch.
    Prototype modulates ALL P4 features via γ/β → network CANNOT bypass it.

用法 | Usage::

    from adatile.decoder.film_decoder import FiLMDecoder

    decoder = FiLMDecoder(in_channels=1280, hidden_dim=256)
    mask = decoder(p4_features, support_proto)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMDecoder(nn.Module):
    """
    FiLM 条件化解码器 | FiLM-conditioned Decoder.

    Prototype → (γ, β) → 调制 P4 特征 → Refinement → Mask.
    无 ProtoCoeffPredictor, 无 ProtoMask, 无分支加法。
    No ProtoCoeffPredictor, no ProtoMask, no branch addition.

    Parameters
    ----------
    in_channels : int
        P4 特征通道数 (FastSAM-x = 1280).
    hidden_dim : int
        FiLM 调制通道数 & 特征投影输出通道数.
    refine_channels : tuple
        Refinement CNN 各层通道数.
    """

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_dim: int = 256,
        refine_channels: tuple = (128, 64),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # ═══════════════════════════════════════════════════════════
        # FiLM 参数生成器 | FiLM Parameter Generator
        # ═══════════════════════════════════════════════════════════
        # Prototype [1280] → γ [hidden_dim] + β [hidden_dim]
        # 两个独立 Linear, 各自学习映射
        self.film_gamma = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ═══════════════════════════════════════════════════════════
        # P4 特征投影 | P4 Feature Projection
        # ═══════════════════════════════════════════════════════════
        # [in_channels, H/16, W/16] → [hidden_dim, H/16, W/16]
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ═══════════════════════════════════════════════════════════
        # Refinement CNN (同原版, 但输入从 hidden_dim 开始)
        # ═══════════════════════════════════════════════════════════
        layers = []
        in_ch = hidden_dim
        for out_ch in refine_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.feat_refine = nn.Sequential(*layers)

        # ═══════════════════════════════════════════════════════════
        # Mask Head | 掩码头
        # ═══════════════════════════════════════════════════════════
        final_ch = refine_channels[-1] if refine_channels else hidden_dim
        self.mask_head = nn.Sequential(
            nn.Conv2d(final_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        权重初始化 | Weight Initialization.

        FiLM γ 最后层 → 零初始化 (初始不改变特征).
        FiLM β 最后层 → 零初始化 (初始不加偏置).
        Conv 层 → Kaiming Normal.
        FiLM gamma final layer → zero init (identity at start).
        FiLM beta final layer → zero init (no bias at start).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # 区分 FiLM 最后层 vs 中间层
                # Distinguish FiLM final layer vs intermediate
                is_film_final = (
                    name.endswith("film_gamma.2") or name.endswith("film_beta.2")
                )
                if is_film_final:
                    # 零初始化 → FiLM 初始为恒等变换
                    # Zero init → FiLM starts as identity
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        p4_features: torch.Tensor,     # [B, in_channels, H/16, W/16]
        proto_masks: torch.Tensor,     # [32, H/4, W/4] — 忽略, 仅兼容旧 API | ignored, compat only
        support_proto: torch.Tensor,   # [in_channels] or [1, in_channels]
        fdr_map=None,                  # 忽略 | ignored (no FDR)
    ) -> torch.Tensor:
        """
        FiLM 前向传播 | FiLM Forward.

        参数顺序与 AdaptiveSparseDecoder 一致 (p4, proto_masks, proto).
        proto_masks 被忽略 — FiLM 不使用 Proto Basis.

        :param p4_features: P4 特征图 [B, in_channels, H/16, W/16].
        :param proto_masks: 忽略 (兼容性). Ignored — FiLM does not use Proto Basis.
        :param support_proto: Support prototype [in_channels].
        :param fdr_map: 忽略 | ignored.
        :return: Binary mask [B, H/4, W/4] in [0,1].
        """
        # ── 输入标准化 | Input normalization ──
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        B = p4_features.shape[0]

        # ═══════════════════════════════════════════════════════════
        # Step 1: 生成 FiLM 参数 | Generate FiLM parameters
        # ═══════════════════════════════════════════════════════════
        gamma = self.film_gamma(support_proto)  # [hidden_dim]
        beta = self.film_beta(support_proto)     # [hidden_dim]

        # ═══════════════════════════════════════════════════════════
        # Step 2: P4 特征投影 | Project P4 features
        # ═══════════════════════════════════════════════════════════
        feat = self.feat_proj(p4_features)  # [B, hidden_dim, H/16, W/16]

        # ═══════════════════════════════════════════════════════════
        # Step 3: FiLM 调制 | FiLM Modulation
        # ═══════════════════════════════════════════════════════════
        # γ [hidden_dim] → [1, hidden_dim, 1, 1] → 逐通道缩放
        # β [hidden_dim] → [1, hidden_dim, 1, 1] → 逐通道偏移
        gamma = gamma.view(1, -1, 1, 1).expand(B, -1, feat.shape[2], feat.shape[3])
        beta = beta.view(1, -1, 1, 1).expand(B, -1, feat.shape[2], feat.shape[3])
        feat = gamma * feat + beta

        # ═══════════════════════════════════════════════════════════
        # Step 4: Refinement + Mask Head
        # ═══════════════════════════════════════════════════════════
        feat = self.feat_refine(feat)          # [B, final_ch, H/16, W/16]
        logit = self.mask_head(feat)            # [B, 1, H/16, W/16]

        # ═══════════════════════════════════════════════════════════
        # Step 5: 上采样到 stride 4 + Sigmoid
        # ═══════════════════════════════════════════════════════════
        final_up = F.interpolate(
            logit, scale_factor=4, mode="bilinear", align_corners=False,
        )  # [B, 1, H/4, W/4]

        return torch.sigmoid(final_up.squeeze(1))  # [B, H/4, W/4] — 兼容 AdaptiveSparseDecoder 形状


class FiLMDecoderP3P4(nn.Module):
    """
    FiLM Decoder with P3+P4 multi-scale features.
    双尺度 FiLM: Prototype 同时调制 P3 和 P4 特征.

    用于捕捉细长 scratch 等需要高空间分辨率的小目标。
    For capturing thin scratches that need higher spatial resolution.

    参数 | Parameters:
        p3_channels: P3 通道数 (960 for FastSAM-x).
        p4_channels: P4 通道数 (1280 for FastSAM-x).
        hidden_dim:  FiLM 调制通道数.
    """

    def __init__(
        self,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        hidden_dim: int = 256,
        refine_channels: tuple = (128, 64),
    ):
        super().__init__()

        # ── P4 分支 (主通路) | P4 branch (main pathway) ──
        self.film_gamma_p4 = nn.Sequential(
            nn.Linear(p4_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film_beta_p4 = nn.Sequential(
            nn.Linear(p4_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj_p4 = nn.Sequential(
            nn.Conv2d(p4_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── P3 分支 (细节补偿) | P3 branch (detail compensation) ──
        p3_hidden = hidden_dim // 2
        self.film_gamma_p3 = nn.Sequential(
            nn.Linear(p4_channels, p3_hidden),  # proto 仍是 P4 的
            nn.ReLU(inplace=True),
            nn.Linear(p3_hidden, p3_hidden),
        )
        self.film_beta_p3 = nn.Sequential(
            nn.Linear(p4_channels, p3_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(p3_hidden, p3_hidden),
        )
        self.feat_proj_p3 = nn.Sequential(
            nn.Conv2d(p3_channels, p3_hidden, kernel_size=1, bias=False),
            nn.InstanceNorm2d(p3_hidden, affine=True),
            nn.ReLU(inplace=True),
        )
        # P3 → P4 分辨率对齐 | P3 → P4 spatial alignment
        self.p3_to_p4 = nn.Sequential(
            nn.Conv2d(p3_hidden, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── Refinement (P3+P4 融合后) | Refinement after P3+P4 fusion ──
        in_ch = hidden_dim + hidden_dim  # P4 + P3 features
        layers = []
        for out_ch in refine_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.feat_refine = nn.Sequential(*layers)

        # ── Mask Head ──
        final_ch = refine_channels[-1] if refine_channels else in_ch
        self.mask_head = nn.Sequential(
            nn.Conv2d(final_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """同 FiLMDecoder 初始化策略."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                is_film_final = any(x in name for x in [
                    "film_gamma_p4.2", "film_beta_p4.2",
                    "film_gamma_p3.2", "film_beta_p3.2",
                ])
                if is_film_final:
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        p4_features: torch.Tensor,
        proto_masks: torch.Tensor,         # 忽略, 兼容性
        support_proto: torch.Tensor,
        p3_features: torch.Tensor | None = None,
        fdr_map=None,
    ) -> torch.Tensor:
        """前向传播 (参数顺序兼容 AdaptiveSparseDecoder)."""
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)
        B = p4_features.shape[0]

        # ── P4 FiLM ──
        gamma_p4 = self.film_gamma_p4(support_proto).view(1, -1, 1, 1)
        beta_p4 = self.film_beta_p4(support_proto).view(1, -1, 1, 1)
        feat_p4 = self.feat_proj_p4(p4_features)
        feat_p4 = gamma_p4 * feat_p4 + beta_p4

        # ── P3 FiLM (if available) ──
        if p3_features is not None:
            gamma_p3 = self.film_gamma_p3(support_proto).view(1, -1, 1, 1)
            beta_p3 = self.film_beta_p3(support_proto).view(1, -1, 1, 1)
            feat_p3 = self.feat_proj_p3(p3_features)
            feat_p3 = gamma_p3 * feat_p3 + beta_p3
            feat_p3_down = self.p3_to_p4(feat_p3)
            feat = torch.cat([feat_p4, feat_p3_down], dim=1)  # [B, hidden_dim*2, H/16, W/16]
        else:
            feat = feat_p4

        # ── Refinement + Mask ──
        feat = self.feat_refine(feat)
        logit = self.mask_head(feat)

        final_up = F.interpolate(
            logit, scale_factor=4, mode="bilinear", align_corners=False,
        )
        return torch.sigmoid(final_up.squeeze(1))  # [B, H/4, W/4]
