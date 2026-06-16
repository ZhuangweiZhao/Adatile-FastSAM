"""CAT-AdaTile: Conditional Adaptive Tiling modules.

Fuses CAT-SAM (Conditional Tuning) with AdaTile-FastSAM's sparse tiling.
Core modules for few-shot domain adaptation with minimal trainable parameters.

Modules:
    - FPNAdapter: Lightweight adapter per FPN level (captures domain-specific features)
    - SpatialAdapter: Adapter for Ada-SPM's SpatialTransformer
    - PromptBridge: Maps decoder CAT-Token → Ada-SPM importance bias
    - CATToken: Learnable token that interacts with prototypes for mask generation

Reference:
    CAT-SAM (Xiao et al., 2024): Decoder-conditioned joint tuning for SAM
    FastSAM (Zhao et al., 2023): CNN-based segment anything
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Lightweight Adapters ────────────────────────────────────────────────


class FPNAdapter(nn.Module):
    """Lightweight adapter inserted at each FPN output level.

    Architecture: Linear(dim → dim//r) → GELU → Linear(dim//r → dim)
    with residual connection. Lazy initialization on first forward pass
    to auto-detect input channels.

    Args:
        in_dim: Input channel dimension (optional, auto-detected if 0).
        reduction: Reduction ratio (default 4).
    """

    def __init__(self, in_dim: int = 0, reduction: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.reduction = reduction
        self._built = in_dim > 0
        if self._built:
            self._build_layers(in_dim)

    def _build_layers(self, in_dim: int, device=None, dtype=None):
        hidden_dim = max(in_dim // self.reduction, 16)
        self.down = nn.Linear(in_dim, hidden_dim, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, in_dim, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.in_dim = in_dim
        self._built = True

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if not self._built:
            self._build_layers(C, device=x.device, dtype=x.dtype)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
        adapted = self.up(self.act(self.down(x_flat)))   # [BHW, C]
        adapted = adapted.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return x + adapted  # residual


class MultiScaleFPNAdapters(nn.Module):
    """Collection of FPNAdapters with lazy channel detection.

    Args:
        channels: Dict mapping level name → channel count (for initialization).
                  Actual channels are auto-detected on first forward.
        reduction: Reduction ratio for each adapter.
    """

    def __init__(self, channels: Dict[str, int], reduction: int = 4):
        super().__init__()
        # Create adapters as a list — keys are matched at forward time
        self.expected_keys = list(channels.keys())
        self.adapters = nn.ModuleList([
            FPNAdapter(in_dim=0, reduction=reduction)  # lazy init
            for _ in channels
        ])

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply per-level adapter to FPN features. Auto-detects channels."""
        result = {}
        adapter_idx = 0
        for name, feat in features.items():
            if adapter_idx < len(self.adapters):
                result[name] = self.adapters[adapter_idx](feat)
                adapter_idx += 1
            else:
                result[name] = feat
        return result


class SpatialTransformerAdapter(nn.Module):
    """Adapter inserted after Ada-SPM's SpatialTransformerRefine.

    Captures domain-specific spatial patterns (e.g., building shapes in aerial
    imagery, organ boundaries in medical images).

    Architecture:
        Conv3x3(dim → dim//r) → GELU → Conv3x3(dim//r → dim) + residual

    Args:
        dim: Feature dimension.
        reduction: Reduction ratio.
    """

    def __init__(self, dim: int = 128, reduction: int = 2):
        super().__init__()
        hidden = max(dim // reduction, 32)
        self.conv_down = nn.Conv2d(dim, hidden, 3, padding=1)
        self.act = nn.GELU()
        self.conv_up = nn.Conv2d(hidden, dim, 3, padding=1)

        # Zero-init for identity-like start
        nn.init.zeros_(self.conv_up.weight)
        nn.init.zeros_(self.conv_up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_up(self.act(self.conv_down(x)))


# ── Prompt Bridge ────────────────────────────────────────────────────────


class PromptBridge(nn.Module):
    """Maps decoder CAT-Token → Ada-SPM importance bias.

    This is the core CAT-SAM innovation: the lightweight decoder's learnable
    token guides Ada-SPM's attention. The bridge tells Ada-SPM "which feature
    channels to emphasize for target classes".

    Architecture (parameter-efficient):
        CAT-Token [C] → MLP → per-channel bias [spm_dim]
        bias is broadcast spatially and added to Ada-SPM's fused features.

    Args:
        token_dim: CAT-Token dimension (from decoder).
        spm_dim: Ada-SPM hidden dimension.
    """

    def __init__(
        self,
        token_dim: int = 256,
        spm_dim: int = 128,
        spatial_size: int = 32,  # kept for API compatibility, unused in lightweight mode
    ):
        super().__init__()
        self.spm_dim = spm_dim

        # Project CAT-Token → per-channel bias (NOT per-spatial-location)
        # This is key: the CAT-Token learns which feature channels matter
        # for the target class, and this bias gets broadcast spatially.
        hidden = max(token_dim // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, spm_dim),
        )

        # Zero-init last layer for neutral start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cat_token: Tensor) -> Tensor:
        """Generate per-channel bias from CAT-Token.

        Args:
            cat_token: [B, token_dim] or [1, token_dim] learnable token.

        Returns:
            bias: [B, spm_dim, 1, 1] per-channel bias (broadcast-ready).
        """
        if cat_token.dim() == 1:
            cat_token = cat_token.unsqueeze(0)  # [1, token_dim]

        B = cat_token.shape[0]
        bias = self.mlp(cat_token)  # [B, spm_dim]
        return bias.unsqueeze(-1).unsqueeze(-1)  # [B, spm_dim, 1, 1]


# ── CAT Token + MLP Head (Decoder-side) ──────────────────────────────────


class CATToken(nn.Module):
    """Learnable Conditional Adaptation Token for the mask decoder.

    Similar to CAT-SAM's CAT-Token: a learnable token that is concatenated
    with decoder inputs. During training, it captures domain-specific features.
    During inference, it remains fixed and generates dynamic weights for the
    mask prediction head.

    The token is also used as input to PromptBridge to guide Ada-SPM.

    Args:
        dim: Token dimension (should match decoder embed_dim).
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        # Single learnable token
        self.token = nn.Parameter(torch.zeros(1, dim))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(self, batch_size: int = 1) -> Tensor:
        """Return CAT-Token expanded to batch size."""
        return self.token.expand(batch_size, -1)  # [B, dim]


class CATMaskHead(nn.Module):
    """Lightweight mask head conditioned on CAT-Token.

    Simplified design: CAT-Token generates a class-specific channel attention
    vector that modulates decoder features, followed by a lightweight conv head.

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

        # CAT-Token → channel attention weights
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

        # Lightweight conv head for mask prediction
        self.conv_head = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
        )

        # Zero-init final conv
        nn.init.zeros_(self.conv_head[-1].weight)
        nn.init.zeros_(self.conv_head[-1].bias)

    def forward(self, features: Tensor, cat_token: Tensor) -> Tensor:
        """Generate mask from features conditioned on CAT-Token.

        Args:
            features: [BHW, dim] decoder features.
            cat_token: [B, dim] CAT-Token.

        Returns:
            mask: [B, 1, H, W] predicted mask.
        """
        B = cat_token.shape[0]
        if features.dim() == 3:
            N, D = features.shape
            H = W = int(N ** 0.5)
            features = features.reshape(B, H, W, D)

        # Channel gating: CAT-Token selects relevant channels
        gate = self.channel_gate(cat_token)  # [B, dim]
        features = features * gate.unsqueeze(-1).unsqueeze(-1)  # [B, H, W, dim]

        # Convert to channel-first for conv
        x = features.permute(0, 3, 1, 2)  # [B, dim, H, W]
        mask = self.conv_head(x)  # [B, 1, H, W]
        return mask.sigmoid()


# ── Full CAT Module ──────────────────────────────────────────────────────


class CATModule(nn.Module):
    """Complete CAT (Conditional Adaptive Tuning) module for AdaTile-FastSAM.

    Wraps all CAT components into a single module that:
    1. Adapts FPN features via lightweight adapters
    2. Bridges decoder CAT-Token → Ada-SPM importance bias
    3. Generates CAT-Token for decoder and prompt bridge

    Usage:
        cat = CATModule(
            fpn_channels={"p3": 128, "p4": 256, "p5": 512},
            token_dim=256,
            spm_dim=128,
        )
        adapted_features = cat.adapt_fpn(fpn_features)
        spm_bias = cat.prompt_bridge(cat.cat_token())
    """

    def __init__(
        self,
        fpn_channels: Dict[str, int],
        token_dim: int = 256,
        spm_dim: int = 128,
        spatial_size: int = 32,
        adapter_reduction: int = 4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.spm_dim = spm_dim

        # Sub-modules
        self.fpn_adapters = MultiScaleFPNAdapters(fpn_channels, adapter_reduction)
        self.spatial_adapter = SpatialTransformerAdapter(spm_dim, reduction=2)
        self.prompt_bridge = PromptBridge(token_dim, spm_dim, spatial_size)
        self.cat_token = CATToken(token_dim)
        self.mask_head = CATMaskHead(token_dim)

        # Track total trainable parameters
        self._n_params = sum(p.numel() for p in self.parameters())

    def adapt_fpn(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply FPN adapters to multi-scale features.

        Handles feature dicts with different key names (p2/p3/p4 vs p3/p4/p5)
        by matching adapters to available feature keys in order.
        """
        result = {}
        adapter_idx = 0
        adapters = list(self.fpn_adapters.adapters)
        for name, feat in features.items():
            if adapter_idx < len(adapters):
                result[name] = adapters[adapter_idx](feat)
                adapter_idx += 1
            else:
                result[name] = feat
        return result

    def adapt_spatial(self, x: Tensor) -> Tensor:
        """Apply spatial adapter to refined features."""
        return self.spatial_adapter(x)

    def get_spm_bias(self, batch_size: int = 1) -> Tensor:
        """Get Ada-SPM importance bias from CAT-Token via PromptBridge."""
        token = self.cat_token(batch_size)
        return self.prompt_bridge(token)

    def get_cat_token(self, batch_size: int = 1) -> Tensor:
        """Get CAT-Token for decoder."""
        return self.cat_token(batch_size)

    @property
    def num_trainable_params(self) -> int:
        return self._n_params

    def trainable_parameters(self):
        """Yield only trainable parameters (for optimizer)."""
        yield from self.parameters()
