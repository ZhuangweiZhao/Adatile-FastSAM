"""
Cross-Attention Conditioned Decoder — CAT-SAM 条件化解码迁移到 FastSAM
=======================================================================
Cross-Attention Conditioned Decoder — CAT-SAM conditioned decoding for FastSAM.

CAT-SAM 核心洞察: Decoder 应根据 "Prompt Token" 只输出目标类别 mask。
本模块完整实现: Cross-Attention (条件化) + Gate Fusion (P3+P4) + FiLM (原型引导).
CAT-SAM core insight: Decoder should output target-class mask based on Prompt Tokens.
This module implements: Cross-Attention (conditioning) + Gate Fusion (P3+P4) + FiLM (prototype).

架构 | Architecture:
    P4 [B,1280,H/16,W/16]              Prompt Tokens [B, N, D]
         │                                    │
         ├─ proj_p4 → [B,256,H,W]             │
         │                                    │
         │  ┌── SpatialCrossAttention ────────┘
         │  │   Q = P4 spatial, K/V = Prompt
         │  │   → "Where is the target in this feature?"
         │  └→ p4_cond [B,256,H,W]
         │
    P3 [B,640,H/8,W/8]
         │
         ├─ proj_p3 → [B,256,H/8,W/8] (upsampled P4 to match)
         │
         ├─ Gate Fusion: α·P3 + (1-α)·P4_cond  (proto-guided α)
         │
         ├─ FiLM: γ·fused + β  (proto-guided modulation)
         │
         └─ Upsample → Mask [B,1,H,W]

训练 | Training:
    Stage 1: Freeze FastSAM backbone, train Decoder (CrossAttn + Gate + FiLM)
    Stage 2: Joint fine-tune with Adapters

用法 | Usage::
    >>> from adatile.decoder.conditioned_decoder import CATStyleDecoder
    >>> decoder = CATStyleDecoder(feat_dim_p3=640, feat_dim_p4=1280)
    >>> mask = decoder(query_p3, query_p4, prompt_tokens, target_size=(896, 896))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger

logger = get_logger("cond_decoder")


class SpatialCrossAttention(nn.Module):
    """
    空间特征 ↔ Prompt Token 交叉注意力.
    Spatial Feature ↔ Prompt Token cross-attention.

    每个空间位置查询所有 prompt token，获取"这个位置有什么目标"的信息。
    Each spatial position queries all prompt tokens for "what target is here".
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert feat_dim % num_heads == 0, f"feat_dim {feat_dim} must be divisible by num_heads {num_heads}"
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Multi-head QKV projections
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self.dropout = nn.Dropout(dropout)

        # 零初始化残差权重 | Zero-init residual weight
        self.res_weight = nn.Parameter(torch.zeros(1))

        n = sum(p.numel() for p in self.parameters())
        logger.log_info("cross_attn/init",
                        f"SpatialCrossAttention: dim={feat_dim}, heads={num_heads}, {n:,} params")

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, C] → [B*H, N, head_dim] for multi-head attention."""
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, self.head_dim)

    def _reshape_from_attention(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """[B*H, N, head_dim] → [B, N, C]."""
        x = x.reshape(B, self.num_heads, N, self.head_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        return x

    def forward(
        self,
        features: torch.Tensor,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param features: [B, C, H, W] spatial features (Q).
        :param prompt: [B, N_p, D_p] prompt tokens (K, V).
        :return: [B, C, H, W] conditioned features.
        """
        B, C, H, W = features.shape
        N_tokens = H * W
        N_prompt = prompt.shape[1]

        # 如果 prompt dim ≠ feat_dim, 对齐 | Align prompt dim to feat_dim
        if prompt.shape[-1] != C:
            prompt = F.pad(prompt, (0, max(0, C - prompt.shape[-1])))[..., :C]

        # Spatial flatten
        feat_seq = features.permute(0, 2, 3, 1).reshape(B, N_tokens, C)

        # QKV
        Q = self.q_proj(feat_seq)        # [B, N_tokens, C]
        K = self.k_proj(prompt)           # [B, N_prompt, C]
        V = self.v_proj(prompt)           # [B, N_prompt, C]

        # Multi-head reshape
        Q = self._reshape_for_attention(Q)
        K = self._reshape_for_attention(K)
        V = self._reshape_for_attention(V)

        # Attention
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B*H, N_tokens, N_prompt]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, V)  # [B*H, N_tokens, head_dim]

        # Output reshape
        out = self._reshape_from_attention(out, B, N_tokens)  # [B, N_tokens, C]
        out = self.out_proj(out)
        out = self.dropout(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # 可学习残差 | Learnable residual
        return features + self.res_weight * out


class CATStyleDecoder(nn.Module):
    """
    CAT-SAM 风格条件化解码器 | CAT-SAM Style Conditioned Decoder.

    融合三个机制 | Combines three mechanisms:
        1. SpatialCrossAttention: 空间特征条件化 (来自 CAT-SAM)
           Spatial feature conditioning (from CAT-SAM)
        2. Proto-Guided Gate Fusion: P3+P4 自适应融合 (来自 AdaTile)
           Adaptive P3+P4 fusion (from AdaTile)
        3. FiLM Modulation: 原型引导的特征调制 (来自 AdaTile)
           Prototype-guided feature modulation (from AdaTile)

    ----------
    feat_dim_p3 : int
        P3 特征通道数 (640 for FastSAM-x).
    feat_dim_p4 : int
        P4 特征通道数 (1280 for FastSAM-x).
    fusion_dim : int
        融合后特征维度 (256).
    prompt_dim : int
        Prompt token 维度 (256).
    num_heads : int
        Cross-attention 头数.
    """

    def __init__(
        self,
        feat_dim_p3: int = 640,
        feat_dim_p4: int = 1280,
        fusion_dim: int = 256,
        prompt_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # ── 通道投影 | Channel projection ──
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(feat_dim_p3, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(feat_dim_p4, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # ── Cross-Attention: P4 spatial ↔ Prompt tokens ──
        # 只在 P4 上做 (P4 是语义层, P3 是纹理层)
        # Only on P4 (semantic layer, P3 = texture layer)
        self.cross_attn = SpatialCrossAttention(
            feat_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
        )

        # ── Prompt → Proto bridge (for gate + FiLM) ──
        self.prompt_to_gate = nn.Sequential(
            nn.Linear(prompt_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, fusion_dim),
            nn.Sigmoid(),
        )
        self.prompt_to_film = nn.Sequential(
            nn.Linear(prompt_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fusion_dim * 2),
        )

        # ── Upsample path | 上采样路径 ──
        self.up1 = nn.Sequential(
            nn.Conv2d(fusion_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1, bias=True)

        # ── 统计 | Stats ──
        n_ca = sum(p.numel() for p in self.cross_attn.parameters())
        n_gate = sum(p.numel() for p in self.prompt_to_gate.parameters())
        n_film = sum(p.numel() for p in self.prompt_to_film.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        logger.log_info("decoder/init",
                        f"CATStyleDecoder: {n_total:,} params "
                        f"(CrossAttn={n_ca:,}, Gate={n_gate:,}, FiLM={n_film:,})")

    def forward(
        self,
        query_p3: torch.Tensor,
        query_p4: torch.Tensor,
        prompt: torch.Tensor,
        target_size: tuple | None = None,
    ) -> torch.Tensor:
        """
        :param query_p3: [B, 640, H/8, W/8] P3 features.
        :param query_p4: [B, 1280, H/16, W/16] P4 features.
        :param prompt: [B, N, D] Prompt tokens (generic + prototype fused).
        :param target_size: (H, W) output mask resolution.
        :return: [B, 1, H, W] mask logits.
        """
        B = query_p3.shape[0]

        # ── Step 1: 通道投影 | Channel projection ──
        f3 = self.proj_p3(query_p3)  # [B, 256, H/8, W/8]
        f4 = self.proj_p4(query_p4)  # [B, 256, H/16, W/16]

        # ── Step 2: Cross-Attention 条件化 P4 | Condition P4 with prompt ──
        # 这是 CAT-SAM 的核心: "根据 prompt, P4 特征应该关注什么?"
        # CAT-SAM core: "Based on prompt, what should P4 features focus on?"
        f4_cond = self.cross_attn(f4, prompt)  # [B, 256, H/16, W/16]

        # ── Step 3: P3+P4 Gate Fusion | 自适应融合 ──
        # Upsample P4 to P3 spatial size | 上采样 P4 匹配 P3
        f4_up = F.interpolate(f4_cond, size=f3.shape[2:],
                              mode="bilinear", align_corners=False)  # [B, 256, H/8, W/8]

        # Prompt → gate α
        prompt_for_gate = prompt.mean(dim=1)  # [B, D]
        alpha = self.prompt_to_gate(prompt_for_gate)  # [B, fusion_dim]
        alpha = alpha.view(B, -1, 1, 1)

        # α·P3 + (1-α)·P4 (per-channel adaptive fusion)
        fused = alpha * f3 + (1.0 - alpha) * f4_up  # [B, 256, H/8, W/8]

        # ── Step 4: FiLM Modulation | 原型引导调制 ──
        film_out = self.prompt_to_film(prompt_for_gate)  # [B, fusion_dim*2]
        gamma, beta = film_out.chunk(2, dim=1)  # 2 × [B, fusion_dim]
        fused = gamma.view(B, -1, 1, 1) * fused + beta.view(B, -1, 1, 1)

        # ── Step 5: Upsample → Mask | 上采样 → 掩码 ──
        x = self.up1(fused)                                         # H/8
        x = F.interpolate(x, scale_factor=2, mode="bilinear",       # H/4
                         align_corners=False)
        x = self.up2(x)                                             # H/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear",       # H/2
                         align_corners=False)
        x = self.up3(x)                                             # H/2
        x = F.interpolate(x, scale_factor=2, mode="bilinear",       # H
                         align_corners=False)
        x = self.mask_head(x)                                       # [B, 1, H, W]

        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear",
                            align_corners=False)
        return x
