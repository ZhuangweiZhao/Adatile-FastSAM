"""
Cross-Attention Prototype Decoder | 交叉注意力原型解码器.
============================================================

用 Cross-Attention 替代 Global Mean + MLP, 保留 K 个独立的 support prototype,
让 Decoder 自动学习选择最相关的 prototype 特征。

Replaces Global Mean + MLP with Cross-Attention over K independent support
prototypes. The decoder learns to attend to the most relevant prototype(s)
at each spatial position.

架构 | Architecture::

    K Support Images
        │
        ▼
    K Prototypes [K, 1280]  ← 不做 mean, 保留多样性
        │
        │  ┌────────────────────────────┐
        │  │ Cross-Attention            │
        │  │ Q = P4_feat [B, N, 256]    │
        │  │ K = Prototypes [K, 1280]   │
        │  │ V = Prototypes [K, 1280]   │
        │  │ → Attended [B, N, 1280]    │
        │  └────────────┬───────────────┘
        │               │
        │               ▼
        │         Reshape + Conv → Condition P4
        │               │
        ▼               ▼
    P4 Features ──→ FiLM-like modulation ──→ Refinement ──→ Mask

关键创新 | Key Innovation:
    无 MLP 压缩, 无信息瓶颈. Attention 权重自动决定每个位置
    参考哪个 support 原型. 类间差异来自 attention 分布, 不是
    一个全局向量.

用法 | Usage::

    from adatile.decoder.cross_attn_decoder import CrossAttnDecoder

    decoder = CrossAttnDecoder(in_channels=1280, hidden_dim=256, num_heads=4)
    mask = decoder(p4_features, prototypes)  # prototypes: [K, 1280]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力 | Multi-Head Cross-Attention.

    Q: Query features [B, N, d_model]
    K,V: Support prototypes [K, d_model]
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(1280, d_model, bias=False)  # K,V 从 1280-d prototype 投影
        self.W_v = nn.Linear(1280, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        query: torch.Tensor,       # [B, N, d_model]
        prototypes: torch.Tensor,  # [K, 1280]
    ) -> torch.Tensor:
        """
        :param query: P4 特征投影 [B, N, 256].
        :param prototypes: K 个 support prototype [K, 1280].
        :return: attended features [B, N, d_model].
        """
        B, N, _ = query.shape
        K, _ = prototypes.shape

        # ── 投影 | Project ──
        Q = self.W_q(query)                        # [B, N, d_model]
        K_p = self.W_k(prototypes)                  # [K, d_model]
        V = self.W_v(prototypes)                    # [K, d_model]

        # ── 多头重塑 | Multi-head reshape ──
        Q = Q.view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, N, d_k]
        K_p = K_p.view(K, self.n_heads, self.d_k).transpose(0, 1)  # [h, K, d_k]
        V = V.view(K, self.n_heads, self.d_k).transpose(0, 1)      # [h, K, d_k]

        # 扩展 batch 维度 | Expand batch dim for K, V
        K_p = K_p.unsqueeze(0).expand(B, -1, -1, -1)  # [B, h, K, d_k]
        V = V.unsqueeze(0).expand(B, -1, -1, -1)      # [B, h, K, d_k]

        # ── Scaled Dot-Product Attention ──
        attn_scores = torch.matmul(Q, K_p.transpose(-2, -1)) / self.scale  # [B, h, N, K]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # [B, h, N, d_k]

        # ── 合并多头 | Merge heads ──
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.d_model)  # [B, N, d_model]

        # ── 输出投影 | Output projection ──
        out = self.W_o(attended)
        return out, attn_weights  # 返回 attention weights 用于分析


class CrossAttnDecoder(nn.Module):
    """
    Cross-Attention Prototype Decoder.

    P4 features attend to K support prototypes via multi-head cross-attention.
    Attended features are used to modulate (FiLM-style) the P4 refinement pathway.

    Parameters
    ----------
    in_channels : P4 feature channels (1280 for FastSAM-x).
    hidden_dim : Internal feature dimension.
    n_heads : Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_dim: int = 256,
        n_heads: int = 4,
        refine_channels: tuple = (128, 64),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # ── P4 特征投影 (生成 Query) | P4 feature projection (→ Query) ──
        self.query_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── Cross-Attention ──
        self.cross_attn = MultiHeadCrossAttention(
            d_model=hidden_dim, n_heads=n_heads, dropout=0.1,
        )

        # ── Attention 输出融合 | Attention output fusion ──
        # Attended features [B, N, hidden_dim] → reshape → [B, hidden_dim, H, W]
        # → Conv fusion with P4 features
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ── Refinement CNN ──
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

        # ── Mask Head ──
        final_ch = refine_channels[-1] if refine_channels else hidden_dim
        self.mask_head = nn.Sequential(
            nn.Conv2d(final_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
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
            # Linear layers are initialized in MultiHeadCrossAttention

    def forward(
        self,
        p4_features: torch.Tensor,     # [B, in_channels, H/16, W/16]
        proto_masks: torch.Tensor,     # 忽略, 兼容旧 API | ignored, old API compat
        support_protos: torch.Tensor,  # [K, 1280] — K independent prototypes
        fdr_map=None,
    ) -> torch.Tensor:
        """
        :param p4_features: P4 features [B, in_channels, H/16, W/16].
        :param proto_masks: Ignored (compatibility).
        :param support_protos: K prototypes [K, 1280]. K≥1.
        :return: Binary mask [B, H/4, W/4] in [0,1].
        """
        B, C, H, W = p4_features.shape
        if support_protos.dim() == 1:
            # 仅 1 个 prototype → 扩展为 [1, 1280]
            support_protos = support_protos.unsqueeze(0)

        # ── Step 1: Query projection ──
        query_feat = self.query_proj(p4_features)  # [B, hidden_dim, H, W]

        # ── Step 2: Reshape for attention ──
        N = H * W
        query_flat = query_feat.view(B, self.hidden_dim, N).transpose(1, 2)  # [B, N, hidden_dim]

        # ── Step 3: Cross-Attention ──
        attended, attn_weights = self.cross_attn(query_flat, support_protos)
        # attended: [B, N, hidden_dim]

        # ── Step 4: Reshape back + Fusion with original P4 features ──
        attended_spatial = attended.transpose(1, 2).view(B, self.hidden_dim, H, W)
        fused = self.attn_fusion(torch.cat([query_feat, attended_spatial], dim=1))
        # fused: [B, hidden_dim, H, W]

        # ── Step 5: Refinement + Mask Head ──
        refined = self.feat_refine(fused)
        logit = self.mask_head(refined)  # [B, 1, H, W]

        # ── Step 6: Upsample + Sigmoid ──
        final_up = F.interpolate(
            logit, scale_factor=4, mode="bilinear", align_corners=False,
        )  # [B, 1, H/4, W/4]

        # 保存 attention weights 用于分析 | Save for analysis
        self._last_attn = attn_weights.detach()

        return torch.sigmoid(final_up.squeeze(1))  # [B, H/4, W/4] — 兼容 AdaptiveSparseDecoder
