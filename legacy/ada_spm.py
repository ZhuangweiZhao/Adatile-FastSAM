"""Ada-SPM: Adaptive Spatial Partition Module.

Core module of AdaTile-FastSAM. Predicts per-region importance and
tile granularity from multi-scale backbone features.

Architecture:
    Multi-scale features
      ↓
    [FPN Fusion Neck] ──→ Unified feature map (H/4 × W/4)
      ↓
    [Optional Transformer Refinement] ──→ Context-aware features
      ↓
    ┌─────────────────────────────────────────┐
    │  Density Head          Granularity Head  │
    │  Conv → Sigmoid        Conv → Gumbel     │
    │  S ∈ [0,1]^(H×W)      T ∈ {0..3}^(H×W)  │
    └─────────────────────────────────────────┘

Outputs:
    - importance:  S ∈ [0,1]^(H_s×W_s)  — predicted spatial importance
    - granularity: T ∈ {0,…,K}^(H_s×W_s) — recommended tile size index
    - aux: dict with diagnostics for analysis hooks

Training losses:
    - Density MSE:    ||S - GT_density||²
    - Entropy reg:    -Σ p log p over tile-size distribution
    - Sparsity loss:  L1 on S to encourage sparsity
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import SparseImportancePredictor, SparsePrediction
from adatile.registry import SPARSE
from adatile.backbone.fpn import MultiScaleFPNFusion, LightweightFPNFusion


# ── Transformer Refinement Block ─────────────────────────────────────


class SpatialTransformerRefine(nn.Module):
    """Lightweight spatial transformer for importance refinement.

    Applies a windowed self-attention block on the fused feature map
    to capture long-range dependencies before density/granularity prediction.

    Uses depthwise-separable convolutions to minimize overhead.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads

        # Window-wise multi-head attention
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)

        # Lightweight FFN (depthwise-separable conv)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, mlp_hidden, 3, padding=1, groups=mlp_hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, dim, 1, bias=False),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Windowed self-attention — forced fp32 for numerical stability.

        The manual q@k matmul and subsequent softmax overflow in fp16
        (intermediate dot products exceed 65504 → INF → NaN).
        We force fp32 computation and cast back to input dtype.
        """
        # Force fp32 computation for the entire attention block.
        # autocast(enabled=False) prevents AMP from downcasting ops.
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_impl(x.float()).to(dtype=x.dtype)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Actual forward logic — always runs in fp32."""
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)

        # QKV projection
        qkv = self.qkv(x)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # [B, C, H, W] each

        # Reshape for window attention
        # Fold H,W into (H/ws, ws, W/ws, ws) then rearrange to windows
        ws = self.window_size
        # Pad if necessary
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = q.shape[-2], q.shape[-1]
        num_windows_h = Hp // ws
        num_windows_w = Wp // ws

        def window_partition(t: Tensor) -> Tensor:
            # [B, C, H, W] → [B*nWin, C, ws, ws]
            B, C, Ht, Wt = t.shape
            t = t.reshape(B, C, num_windows_h, ws, num_windows_w, ws)
            t = t.permute(0, 2, 4, 1, 3, 5)  # [B, nWh, nWw, C, ws, ws]
            t = t.reshape(B * num_windows_h * num_windows_w, C, ws, ws)
            return t

        def window_reverse(t: Tensor) -> Tensor:
            # [B*nWin, C, ws, ws] → [B, C, Hp, Wp]
            t = t.reshape(B, num_windows_h, num_windows_w, C, ws, ws)
            t = t.permute(0, 3, 1, 4, 2, 5)  # [B, C, nWh, ws, nWw, ws]
            t = t.reshape(B, C, Hp, Wp)
            return t

        q_w = window_partition(q)  # [B*nWin, C, ws, ws]
        k_w = window_partition(k)
        v_w = window_partition(v)

        # Multi-head attention per window
        q_w = q_w.reshape(-1, self.num_heads, self.head_dim, ws * ws)
        k_w = k_w.reshape(-1, self.num_heads, self.head_dim, ws * ws)
        v_w = v_w.reshape(-1, self.num_heads, self.head_dim, ws * ws)

        scale = self.head_dim ** -0.5
        attn = (q_w.transpose(-2, -1) @ k_w) * scale  # [*, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ v_w.transpose(-2, -1)).transpose(-2, -1)
        attn_out = attn_out.reshape(-1, C, ws, ws)

        # Reverse window partition
        attn_out = window_reverse(attn_out)  # [B, C, Hp, Wp]
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :, :H, :W]

        x = shortcut + self.dropout(self.proj(attn_out))

        # FFN
        shortcut2 = x
        x = self.norm2(x)
        x = shortcut2 + self.dropout(self.mlp(x))

        return x


# ── Density Head ─────────────────────────────────────────────────────


class DensityHead(nn.Module):
    """Predicts per-pixel importance S ∈ [0, 1].

    Lightweight conv stack: 3×3 → 3×3 → 1×1 → Sigmoid.
    Output resolution matches the input feature map.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 1, bias=True)  # Single-channel output
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # Initialize bias so sigmoid(bias) ≈ 0.85 → model starts "optimistic"
        # (high density everywhere). It learns to suppress low-importance regions.
        nn.init.constant_(self.conv3.bias, 1.8)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return torch.sigmoid(x)


# ── Granularity Head ─────────────────────────────────────────────────


class GranularityHead(nn.Module):
    """Predicts tile size recommendation T at each spatial location.

    Output: logits over K tile size categories.
    During training: returns Gumbel-Softmax samples (differentiable).
    During inference: returns argmax (hard assignment).

    Default tile sizes: [384, 768, 1536, 3072]
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 128,
        num_tile_sizes: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tile_sizes = num_tile_sizes

        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, num_tile_sizes, 1, bias=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Predict tile granularity.

        Args:
            x: [B, C, H, W] fused feature map.
            temperature: Gumbel-Softmax temperature.
            hard: If True, use straight-through Gumbel (one-hot output).

        Returns:
            soft_assignment: [B, K, H, W] softmax probabilities (always).
            hard_index: [B, 1, H, W] integer tile size index (argmax).
        """
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x)))
        logits = self.conv3(x)  # [B, K, H, W]

        if self.training and not hard:
            # Gumbel-Softmax for differentiable sampling
            soft = F.gumbel_softmax(
                logits, tau=temperature, hard=False, dim=1
            )  # [B, K, H, W]
        elif hard:
            soft = F.gumbel_softmax(
                logits, tau=temperature, hard=True, dim=1
            )  # straight-through, gradient like soft
        else:
            soft = F.softmax(logits, dim=1)

        hard_index = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, H, W]
        return soft, hard_index


# ── Main Ada-SPM Module ──────────────────────────────────────────────


@SPARSE.register()
class AdaSPM(SparseImportancePredictor):
    """Adaptive Spatial Partition Module — full implementation.

    Inputs:
        features: {"c2": [B,256,H/4,W/4], "c3": [B,512,H/8,W/8],
                   "c4": [B,1024,H/16,W/16], "c5": [B,2048,H/32,W/32]}

    Outputs:
        importance:  [B, 1, H_s, W_s] predicted importance (alias for density output).
        granularity: {"soft": [B,K,H_s,W_s], "hard": [B,1,H_s,W_s]} tile assignments.
        density:     [B, 1, H_s, W_s] continuous density prediction (same as importance).
        aux:         dict with intermediate outputs for analysis hooks.

    Args:
        in_channels_list: Channel dims for [c2, c3, c4, c5].
        fusion_dim: FPN output dimension.
        num_tile_sizes: Number of tile size categories.
        tile_sizes: Actual tile side lengths (for metadata).
        use_transformer: Enable self-attention refinement.
        importance_threshold: Default threshold for routing (configurable at runtime).
        gumbel_temperature: Gumbel-Softmax temperature.
    """

    def __init__(
        self,
        in_channels_list: Optional[List[int]] = None,
        fusion_dim: int = 256,
        hidden_dim: int = 128,
        num_tile_sizes: int = 4,
        tile_sizes: Optional[List[int]] = None,
        use_transformer: bool = True,
        transformer_window: int = 8,
        transformer_heads: int = 4,
        importance_threshold: float = 0.5,
        gumbel_temperature: float = 1.0,
        lightweight: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.num_tile_sizes = num_tile_sizes
        self.tile_sizes = tile_sizes or [384, 768, 1536, 3072]
        self.importance_threshold = importance_threshold
        self.gumbel_temperature = gumbel_temperature

        # Feature fusion
        fusion_cls = LightweightFPNFusion if lightweight else MultiScaleFPNFusion
        self.fusion = fusion_cls(
            in_channels_list=in_channels_list,
            out_dim=fusion_dim,
        )

        # Optional transformer refinement
        self.transformer = None
        if use_transformer:
            self.transformer = SpatialTransformerRefine(
                dim=fusion_dim,
                num_heads=transformer_heads,
                window_size=transformer_window,
                dropout=dropout,
            )

        # Prediction heads
        self.density_head = DensityHead(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # CAT: PromptBridge bias (set externally by pipeline)
        self._spm_bias: Optional[torch.Tensor] = None

        self.granularity_head = GranularityHead(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_tile_sizes=num_tile_sizes,
            dropout=dropout,
        )

        # Learnable per-tile-size importance bias
        self.tile_importance_bias = nn.Parameter(
            torch.linspace(0.2, 0.8, num_tile_sizes).view(1, num_tile_sizes, 1, 1)
        )

    # ── CAT PromptBridge ──────────────────────────────────────────

    def set_spm_bias(self, bias: Optional[torch.Tensor]):
        """Set PromptBridge bias for CAT-style conditional tuning.

        Args:
            bias: [B, spm_dim, H_s, W_s] spatial bias from PromptBridge,
                  or None to clear.
        """
        self._spm_bias = bias

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        features: Dict[str, Tensor],
        return_aux: bool = False,
    ) -> "SparsePrediction":
        """Forward pass — forced fp32 for numerical stability.

        AMP fp16 causes NaN in FPN fusion, custom window attention,
        and density/granularity conv heads. The Ada-SPM operates on
        small feature maps (16×16 to 128×128) so fp32 overhead is
        negligible (< 5ms) while guaranteeing no NaN.
        """
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_impl(features, return_aux)

    def _forward_impl(
        self,
        features: Dict[str, Tensor],
        return_aux: bool = False,
    ) -> "SparsePrediction":
        """Internal forward — always runs in fp32."""

        import logging
        _dbg = logging.getLogger("adatile.ada_spm")

        # Convert all input features to fp32
        fp32_features = {k: v.float() for k, v in features.items()}

        # ── NaN source tracking ──
        for k, v in fp32_features.items():
            if torch.isnan(v).any():
                _dbg.error("[NaN-TRACK] Backbone feature '%s' contains NaN! "
                           "Backbone produced NaN under fp16.", k)

        # 1. Multi-scale fusion
        fused, pyramid = self.fusion(fp32_features)
        if torch.isnan(fused).any():
            _dbg.error("[NaN-TRACK] FPN output is NaN!")

        # 2. Transformer refinement (optional)
        if self.transformer is not None:
            fused = self.transformer(fused)
            if torch.isnan(fused).any():
                _dbg.error("[NaN-TRACK] Transformer output is NaN!")

        # ── CAT: Apply PromptBridge bias ──
        if self._spm_bias is not None:
            # Bias: [B, spm_dim, H_s, W_s] — pad/trim to match fused shape
            bias = self._spm_bias
            if bias.shape[-2:] != fused.shape[-2:]:
                bias = nn.functional.interpolate(
                    bias, size=fused.shape[-2:], mode="bilinear", align_corners=False
                )
            if bias.shape[1] != fused.shape[1]:
                # Channel mismatch: use a 1×1 conv projection
                if bias.shape[1] > fused.shape[1]:
                    bias = bias[:, :fused.shape[1]]
                else:
                    bias = nn.functional.pad(bias, (0, 0, 0, 0, 0, fused.shape[1] - bias.shape[1]))
            fused = fused + bias

        # 3. Density prediction
        density = self.density_head(fused)  # [B, 1, H_s, W_s]
        if torch.isnan(density).any():
            _dbg.error("[NaN-TRACK] DensityHead output is NaN!")

        # 4. Granularity prediction — use higher temperature for stability
        # Gumbel-Softmax with low tau: exp(logits/tau) can overflow → NaN
        # Using tau=2.0 prevents overflow even with large logits
        granularity_soft, granularity_hard = self.granularity_head(
            fused,
            temperature=max(self.gumbel_temperature, 2.0),
            hard=not self.training,
        )
        if torch.isnan(granularity_soft).any():
            _dbg.error("[NaN-TRACK] GranularityHead (Gumbel) output is NaN! "
                       "temperature=%.1f", max(self.gumbel_temperature, 2.0))

        # Combine density and granularity for routing importance
        importance = self._compute_importance(density, granularity_soft)

        prediction = SparsePrediction(
            importance=importance,
            density=density,
            granularity_soft=granularity_soft,
            granularity_hard=granularity_hard,
        )

        if return_aux:
            aux = {
                "fused_features": fused,
                "pyramid": pyramid,
                "density_raw": density,
            }
            return prediction, aux

        return prediction

    # ── Importance Computation ───────────────────────────────────

    def _compute_importance(
        self,
        density: Tensor,
        granularity_soft: Tensor,
    ) -> Tensor:
        """Compute combined importance from density and granularity.

        importance = density ⊙ (Σ_k (1 - tile_size_ratio[k]) × granularity[k])

        where tile_size_ratio = tile_size / max_tile_size.
        Smaller tiles contribute higher importance (they represent regions
        needing fine-grained attention).

        NaN-safe: if density contains NaN (AMP fp16 instability or bad data),
        replaces with 0.5 uniform and logs a warning.

        Args:
            density: [B, 1, H, W] predicted density.
            granularity_soft: [B, K, H, W] soft tile size assignment.

        Returns:
            importance: [B, 1, H, W] in [0, 1].
        """
        import logging
        _log = logging.getLogger("adatile.ada_spm")

        # ── NaN guard: replace NaN density with 0.5 uniform ─────
        nan_mask = torch.isnan(density)
        if nan_mask.any():
            _log.warning(
                "[AdaSPM] ⚠ NaN detected in density! %d/%d values. "
                "Replacing with uniform 0.5. Check AMP stability.",
                nan_mask.sum().item(), density.numel(),
            )
            density = torch.where(nan_mask, torch.full_like(density, 0.5), density)

        # ── NaN guard: replace NaN granularity with uniform ─────
        nan_g = torch.isnan(granularity_soft)
        if nan_g.any():
            _log.warning(
                "[AdaSPM] ⚠ NaN detected in granularity! %d/%d values. "
                "Replacing with uniform 1/K.",
                nan_g.sum().item(), granularity_soft.numel(),
            )
            K = granularity_soft.shape[1]
            granularity_soft = torch.where(
                nan_g,
                torch.full_like(granularity_soft, 1.0 / K),
                granularity_soft,
            )

        max_size = max(self.tile_sizes)
        # Weight smaller tiles more → higher importance (heuristic baseline)
        size_weights = 1.0 - torch.tensor(
            [s / max_size for s in self.tile_sizes],
            device=density.device,
            dtype=density.dtype,
        ).view(1, -1, 1, 1)  # [1, K, 1, 1]

        # Learned per-tile-size importance modulation (Reviewer Q9 fix).
        learned_bias = self.tile_importance_bias.to(device=density.device, dtype=density.dtype)

        granularity_weight = (granularity_soft * size_weights * learned_bias).sum(dim=1, keepdim=True)
        importance = density * (0.5 + 0.5 * granularity_weight)

        # Final clamp + NaN guard (clamp does NOT fix NaN)
        importance = torch.clamp(importance, 0.0, 1.0)
        final_nan = torch.isnan(importance)
        if final_nan.any():
            _log.warning(
                "[AdaSPM] ⚠ importance still NaN after clamp! "
                "Falling back to uniform 0.5."
            )
            importance = torch.where(final_nan, torch.full_like(importance, 0.5), importance)

        return importance

    # ── Loss Functions ───────────────────────────────────────────

    def compute_density_loss(
        self,
        pred_density: Tensor,
        target_density: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """MSE loss between predicted and GT density.

        Args:
            pred_density: [B, 1, H, W] predicted.
            target_density: [B, 1, H, W] ground truth.
            valid_mask: [B, 1, H, W] optional mask.

        Returns:
            Scalar loss.
        """
        loss = F.mse_loss(pred_density, target_density, reduction="none")
        if valid_mask is not None:
            loss = loss * valid_mask
            return loss.sum() / (valid_mask.sum() + 1e-8)
        return loss.mean()

    def compute_entropy_loss(
        self,
        granularity_soft: Tensor,
    ) -> Tensor:
        """Entropy regularization: discourage uniform tile-size distribution.

        Low entropy → model is confident about which tile size to use.

        Args:
            granularity_soft: [B, K, H, W] softmax probabilities.

        Returns:
            Scalar loss (mean entropy across spatial locations).
        """
        # H = -Σ p_k log(p_k)
        log_probs = torch.log(granularity_soft + 1e-8)
        entropy = -(granularity_soft * log_probs).sum(dim=1)  # [B, H, W]
        # Normalize by max entropy (log K)
        max_entropy = torch.log(torch.tensor(
            self.num_tile_sizes, dtype=entropy.dtype, device=entropy.device
        ))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy.mean()

    def compute_sparsity_loss(
        self,
        importance: Tensor,
        target_sparsity: float = 0.5,
    ) -> Tensor:
        """Sparsity loss: encourage a target fraction of low-importance regions.

        Uses L1 on importance values to bias toward sparsity.
        Adjust target_sparsity to control how many regions are skipped.

        Args:
            importance: [B, 1, H, W].
            target_sparsity: desired mean importance (lower = sparser).

        Returns:
            Scalar loss.
        """
        return F.l1_loss(importance, torch.full_like(importance, target_sparsity))

    def compute_loss(
        self,
        pred_importance: Tensor,
        target_density: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Legacy interface: density MSE loss.

        For full loss, use compute_all_losses().
        """
        return self.compute_density_loss(pred_importance, target_density, valid_mask)

    def compute_all_losses(
        self,
        output: "SparsePrediction",
        target_density: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        density_weight: float = 1.0,
        entropy_weight: float = 0.1,
        sparsity_weight: float = 0.05,
        target_sparsity: float = 0.5,
    ) -> Dict[str, Tensor]:
        """Compute all Ada-SPM losses.

        Args:
            output: SparsePrediction from forward().
            target_density: GT density map.
            valid_mask: Optional validity mask.
            density_weight: Weight for density MSE.
            entropy_weight: Weight for entropy regularization.
            sparsity_weight: Weight for sparsity loss.
            target_sparsity: Target mean importance.

        Returns:
            Dict of named losses and a "loss_spm" total.
        """
        losses = {}

        if target_density is not None:
            losses["loss_density"] = density_weight * self.compute_density_loss(
                output.density, target_density, valid_mask
            )

        if output.granularity_soft is not None:
            losses["loss_entropy"] = entropy_weight * self.compute_entropy_loss(
                output.granularity_soft
            )

        losses["loss_sparsity"] = sparsity_weight * self.compute_sparsity_loss(
            output.importance,
            target_sparsity,
        )

        losses["loss_spm"] = sum(losses.values())
        return losses

    # ── Threshold Methods ────────────────────────────────────────

    def get_binary_importance(
        self,
        importance: Tensor,
        threshold: Optional[float] = None,
    ) -> Tensor:
        """Binarize importance map for routing decisions.

        Args:
            importance: [B, 1, H, W] importance.
            threshold: Override default threshold.

        Returns:
            binary: [B, 1, H, W] boolean.
        """
        thresh = threshold if threshold is not None else self.importance_threshold
        return importance > thresh

    def set_threshold(self, threshold: float) -> None:
        """Update routing threshold (e.g., for curriculum learning)."""
        self.importance_threshold = threshold

    def set_gumbel_temperature(self, temperature: float) -> None:
        """Update Gumbel temperature (anneal during training)."""
        self.gumbel_temperature = temperature

    # ── Granularity Helpers ──────────────────────────────────────

    def get_tile_size_map(
        self,
        granularity_hard: Tensor,
    ) -> Tensor:
        """Convert hard tile-size indices to actual tile sizes.

        Args:
            granularity_hard: [B, 1, H, W] integer indices.

        Returns:
            [B, 1, H, W] tile sizes in pixels.
        """
        size_tensor = torch.tensor(
            self.tile_sizes,
            device=granularity_hard.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)

        # One-hot + matmul
        B, _, H, W = granularity_hard.shape
        one_hot = F.one_hot(
            granularity_hard.squeeze(1).long(),
            num_classes=len(self.tile_sizes),
        ).permute(0, 3, 1, 2).float()  # [B, K, H, W]

        tile_map = (one_hot * size_tensor).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        return tile_map

    def get_granularity_stats(
        self,
        granularity_soft: Tensor,
    ) -> Dict[str, float]:
        """Aggregate tile-size distribution statistics.

        Args:
            granularity_soft: [B, K, H, W].

        Returns:
            Dict with per-size ratio and mean assignment entropy.
        """
        B, K = granularity_soft.shape[:2]
        # Per-size average across batch and spatial dims
        size_avg = granularity_soft.mean(dim=(0, 2, 3))  # [K]

        stats = {}
        for i, size in enumerate(self.tile_sizes):
            stats[f"granularity_{size}_ratio"] = float(size_avg[i])
            stats[f"granularity_{size}_label"] = {
                0: "fine", 1: "moderate", 2: "coarse", 3: "context"
            }.get(i, str(i))

        # Mean entropy
        log_p = torch.log(granularity_soft + 1e-8)
        entropy_map = -(granularity_soft * log_p).sum(dim=1)
        stats["granularity_mean_entropy"] = float(entropy_map.mean())

        return stats


# ── Baseline Variants ────────────────────────────────────────────────


@SPARSE.register()
class AdaSPMLite(AdaSPM):
    """Lightweight Ada-SPM variant — no transformer, fewer channels.

    Suitable for latency-sensitive applications and ablation studies.
    """

    def __init__(
        self,
        in_channels_list: Optional[List[int]] = None,
        fusion_dim: int = 128,
        hidden_dim: int = 64,
        num_tile_sizes: int = 4,
        **kwargs,
    ):
        super().__init__(
            in_channels_list=in_channels_list,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_tile_sizes=num_tile_sizes,
            use_transformer=False,
            lightweight=True,
            dropout=0.0,
            **kwargs,
        )


@SPARSE.register()
class AdaSPMFull(AdaSPM):
    """Full Ada-SPM — transformer refinement + larger hidden dim.

    Maximum accuracy variant for primary experiments.
    """

    def __init__(
        self,
        in_channels_list: Optional[List[int]] = None,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_tile_sizes: int = 4,
        **kwargs,
    ):
        super().__init__(
            in_channels_list=in_channels_list,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_tile_sizes=num_tile_sizes,
            use_transformer=True,
            lightweight=False,
            dropout=0.1,
            **kwargs,
        )


@SPARSE.register()
class DensityOnlySPM(AdaSPM):
    """Ablation variant: density prediction only (no granularity).

    The granularity head is disabled; tile sizes are uniform.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, features: Dict[str, Tensor], return_aux: bool = False):
        fused, pyramid = self.fusion(features)
        if self.transformer is not None:
            fused = self.transformer(fused)
        density = self.density_head(fused)
        B, _, H, W = density.shape
        # Uniform granularity
        uniform = torch.ones(B, self.num_tile_sizes, H, W, device=density.device)
        uniform = uniform / self.num_tile_sizes
        importance = self._compute_importance(density, uniform)

        return SparsePrediction(
            importance=importance,
            density=density,
            granularity_soft=uniform,
            granularity_hard=torch.zeros(B, 1, H, W, dtype=torch.long, device=density.device),
        )
