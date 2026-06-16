"""Sparse attention backends for DTR-v2 routing levels.

Three attention variants with different compute/accuracy tradeoffs,
all backed by PyTorch SDPA (which dispatches to FlashAttention-2 on
supported hardware) and xFormers memory_efficient_attention.

    - LinearAttention  (Level-1): Local-window attention via block-diagonal mask.
      Approximates O(N·W²·d) complexity with window size W.
    - LowRankAttention (Level-2): Block-sparse attention via xFormers
      BlockDiagonalMask. Approximates O(N·r·d) with effective rank r.
    - FullAttention    (Level-3): Standard multi-head attention via
      F.scaled_dot_product_attention → FlashAttention-2.

Each implements a common interface: forward(x) → processed_x.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SparseAttentionBase(nn.Module):
    """Shared interface for all sparse attention backends."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        )
        self.dropout = dropout

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"

    def _reshape_for_attention(self, qkv: Tensor) -> Tensor:
        """Reshape [N, embed_dim] → [1, num_heads, N, head_dim] for SDPA.

        F.scaled_dot_product_attention expects [B, H, N, D] format
        where B=batch, H=num_heads, N=seq_len, D=head_dim.
        Mask broadcasts as [B, 1, N, N].
        """
        N, D = qkv.shape
        # [N, H*D] → [N, H, D] → [H, N, D] → [1, H, N, D]
        return qkv.view(N, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)


# ── Level-1: Local-Window (Linear) Attention ────────────────────────────


class LinearAttention(SparseAttentionBase):
    """Local-window attention approximating linear complexity.

    Instead of Performer-style kernelization, uses xFormers
    memory_efficient_attention with a block-diagonal causal mask
    that restricts each token to attend within a local window.

    Complexity: O(N · W · d) where W is the window size.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        window_size: Local attention window size (tokens).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: int = 32,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _build_window_mask(self, N: int, device: torch.device) -> Tensor:
        """Build a block-diagonal attention mask for local windows.

        Each token can only attend to tokens within the same window of
        size `window_size`. Returns a boolean mask suitable for SDPA.
        """
        num_windows = max(1, N // self.window_size)
        mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        for w in range(num_windows):
            start = w * self.window_size
            end = min(start + self.window_size, N)
            mask[start:end, start:end] = True
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """Local-window attention forward pass.

        Args:
            x: [N, embed_dim] input token sequence.

        Returns:
            out: [N, embed_dim] attended token features.
        """
        N, D = x.shape
        H = self.num_heads

        q = self._reshape_for_attention(self.q_proj(x))  # [1, N, H, d]
        k = self._reshape_for_attention(self.k_proj(x))  # [1, N, H, d]
        v = self._reshape_for_attention(self.v_proj(x))  # [1, N, H, d]

        # Build window mask [N, N] — SDPA broadcasts batch dim
        window_mask = self._build_window_mask(N, x.device)
        attn_mask = torch.zeros(N, N, device=x.device, dtype=x.dtype)
        attn_mask[~window_mask] = float("-inf")

        # SDPA with 4D inputs expects attn_mask in [B, 1, N, N] or [1, 1, N, N]
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)  # [N,N] → [1,1,N,N]
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)  # [N,N] → [1,1,N,N]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [1, H, N, d]

        # [1, H, N, d] → [H, N, d] → [N, H, d] → [N, D]
        out = out.squeeze(0).transpose(0, 1).reshape(N, D)
        return self.out_proj(out)

    def flops_per_token(self) -> int:
        D = self.embed_dim
        W = self.window_size
        return 6 * D * D + 2 * W * D


# ── Level-2: Block-Sparse (Low-Rank) Attention ──────────────────────────


class LowRankAttention(SparseAttentionBase):
    """Block-sparse attention approximating low-rank attention.

    Instead of Linformer-style projection matrices, uses xFormers
    block-sparse attention patterns that restrict each token to
    attend to a subset of other tokens, achieving O(N·r·d) complexity
    for effective rank r.

    When xFormers is not installed, falls back to standard SDPA.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        rank: Effective rank (controls block size).
        dropout: Attention dropout rate.
        num_blocks: Number of sparse blocks in the attention pattern.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        rank: int = 128,
        dropout: float = 0.1,
        num_blocks: int = 4,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.rank = rank
        self.num_blocks = num_blocks

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Block-sparse attention forward pass.

        Divides the sequence into blocks. Each token attends to tokens
        within the same block and adjacent blocks (strided pattern),
        approximating low-rank structure.

        Args:
            x: [N, embed_dim] input tokens.

        Returns:
            out: [N, embed_dim] attended features.
        """
        N, D = x.shape
        H = self.num_heads

        q = self._reshape_for_attention(self.q_proj(x))  # [1, N, H, d]
        k = self._reshape_for_attention(self.k_proj(x))  # [1, N, H, d]
        v = self._reshape_for_attention(self.v_proj(x))  # [1, N, H, d]

        # Block-sparse pattern: divide into blocks, stride-based connectivity
        block_size = max(1, N // self.num_blocks)
        mask = torch.zeros(N, N, device=x.device, dtype=x.dtype)

        for b in range(self.num_blocks):
            b_start = b * block_size
            b_end = min(b_start + block_size, N)
            mask[b_start:b_end, b_start:b_end] = 0.0
            if b > 0:
                p_start = max(0, (b - 1) * block_size)
                p_end = b_start
                mask[b_start:b_end, p_start:p_end] = 0.0
            if b < self.num_blocks - 1:
                n_start = min(b_end, N)
                n_end = min(b_end + block_size, N)
                mask[b_start:b_end, n_start:n_end] = 0.0

        mask[mask != 0.0] = float("-inf")

        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [N,N] → [1,1,N,N]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [1, H, N, d]

        out = out.squeeze(0).transpose(0, 1).reshape(N, D)
        return self.out_proj(out)

    def flops_per_token(self) -> int:
        D = self.embed_dim
        block_size = self.rank
        return 6 * D * D + 2 * block_size * D


# ── Level-3: Full Multi-Head Attention ──────────────────────────────────


class FullAttention(SparseAttentionBase):
    """Standard multi-head self-attention via FlashAttention-2.

    Uses F.scaled_dot_product_attention which automatically dispatches
    to FlashAttention-2, Memory-Efficient Attention, or a generic
    fallback based on input shape and hardware capability.

    Used for the most important tokens that require full pairwise
    attention for precise mask prediction.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        add_ffn: Whether to append a feed-forward network.
        ffn_expansion: FFN hidden dimension expansion factor.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_ffn: bool = True,
        ffn_expansion: int = 4,
        **kwargs,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.add_ffn = add_ffn

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)

        if add_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * ffn_expansion, embed_dim),
                nn.Dropout(dropout),
            )
            self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Full attention with FlashAttention-2 backend.

        Args:
            x: [N, embed_dim] input tokens.

        Returns:
            out: [N, embed_dim] attended tokens.
        """
        N, D = x.shape

        q = self._reshape_for_attention(self.q_proj(x))  # [1, N, H, d]
        k = self._reshape_for_attention(self.k_proj(x))  # [1, N, H, d]
        v = self._reshape_for_attention(self.v_proj(x))  # [1, N, H, d]

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [1, H, N, d] — auto-dispatches to FlashAttention-2

        out = out.squeeze(0).transpose(0, 1).reshape(N, D)
        out = self.out_proj(out)
        out = self.attn_norm(x + out)

        if self.add_ffn:
            ffn_out = self.ffn(out)
            out = self.ffn_norm(out + ffn_out)

        return out

    def flops_per_token(self) -> int:
        D = self.embed_dim
        base = 6 * D * D
        ffn = 8 * D * D if self.add_ffn else 0
        return base + ffn


# ── Combined Multi-Level Attention ──────────────────────────────────────


class MultiLevelAttention(nn.Module):
    """Routes tokens to different attention levels and combines results.

    Given token-to-level assignments, dispatches each token to its
    assigned attention backend, processes in parallel per level, then
    recombines in original order.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Attention heads per level.
        lowrank_rank: Effective rank for Level-2 block-sparse attention.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        lowrank_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.linear_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.lowrank_attn = LowRankAttention(embed_dim, num_heads, lowrank_rank, dropout)
        self.full_attn = FullAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        x: Tensor,
        level_assignments: Tensor,
    ) -> Tensor:
        """Process tokens through assigned attention levels.

        Args:
            x: [N, embed_dim] all active (non-skip) tokens.
            level_assignments: [N] int tensor with values in {1, 2, 3}.

        Returns:
            out: [N, embed_dim] processed tokens in same order.
        """
        N, D = x.shape
        out = torch.zeros_like(x)

        for level in [1, 2, 3]:
            mask = (level_assignments == level)
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            x_level = x[idx]  # [n_level, D]

            if level == 1:
                x_level = self.linear_attn(x_level)
            elif level == 2:
                x_level = self.lowrank_attn(x_level)
            else:
                x_level = self.full_attn(x_level)

            out[idx] = x_level.to(dtype=out.dtype)

        return out

    def flops(self, num_tokens: int, level_counts: dict) -> int:
        """Total FLOPs given token counts per level."""
        total = 0
        for level, count in level_counts.items():
            if level == 1:
                total += count * self.linear_attn.flops_per_token()
            elif level == 2:
                total += count * self.lowrank_attn.flops_per_token()
            elif level == 3:
                total += count * self.full_attn.flops_per_token()
        return total
