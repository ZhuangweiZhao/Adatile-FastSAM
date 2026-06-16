"""DTR-v2 Dynamic Token Router with 4-level adaptive routing.

Core routing pipeline:
    1. RoutingHead: token → 4-class routing logits (skip/L1/L2/L3)
    2. BudgetController: enforce compute budget constraints per level
    3. PrototypeRouter: bias routing toward known classes
    4. ConfidenceEstimator: per-token routing confidence
    5. TokenSparsifier: prune low-importance tokens (Level-0)
    6. MultiLevelAttention: dispatch tokens to appropriate attention backends

All routers follow the BaseRouter interface:
    forward(tokens, metadata=None) -> RoutingOutput
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import BaseRouter, RoutingOutput
from adatile.registry import ROUTER
from adatile.routing.attention import MultiLevelAttention


# ── Routing Head ───────────────────────────────────────────────────────


class RoutingHead(nn.Module):
    """Lightweight MLP that predicts per-token routing decisions.

    Takes token features and outputs 4-class logits:
    [skip, linear_attn, lowrank_attn, full_attn].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.level_bias = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        tokens: Tensor,
        importance: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict routing logits and soft probabilities.

        Args:
            tokens: [N, C] token features.
            importance: Optional [N] importance scores (unused, reserved).

        Returns:
            logits: [N, 4] raw routing logits.
            probs: [N, 4] softmax probability distribution.
        """
        logits = self.net(tokens) + self.level_bias
        probs = (logits / self.temperature).softmax(dim=-1)
        return logits, probs


# ── Budget Controller ──────────────────────────────────────────────────


class BudgetController(nn.Module):
    """Enforces compute budget constraints via differentiable logit biasing.

    Instead of post-hoc assignment modification (which breaks gradients),
    applies biases to routing logits BEFORE gumbel_softmax. This preserves
    straight-through gradient flow from the hard one-hot tensor back to
    the routing head parameters.

    Constraints:
        - Level 0 (skip):  at most max_skip_ratio fraction
        - Level 3 (full):  at most max_full_ratio fraction
        - Level 1 (linear): at least min_linear_ratio fraction
    """

    def __init__(
        self,
        max_skip_ratio: float = 0.15,
        max_full_ratio: float = 0.25,
        min_linear_ratio: float = 0.20,
        bias_strength: float = 8.0,
    ):
        super().__init__()
        self.max_skip_ratio = max_skip_ratio
        self.max_full_ratio = max_full_ratio
        self.min_linear_ratio = min_linear_ratio
        self.bias_strength = bias_strength

    def forward(
        self,
        logits: Tensor,
        probs: Tensor,
        importance: Optional[Tensor] = None,
        training: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply budget constraints via differentiable logit biasing.

        Args:
            logits: [N, 4] raw routing logits (before softmax).
            probs: [N, 4] soft routing probabilities (for analysis only).
            importance: Optional [N] importance scores for skip priority.
            training: Whether in training mode.

        Returns:
            biased_probs: [N, 4] budget-adjusted soft probabilities.
            hard: [N, 4] one-hot hard assignment (straight-through gradient).
            assignments: [N] integer level indices (for control flow).
            weights: [N] differentiable routing weights.
        """
        N = logits.shape[0]
        device = logits.device

        max_skip = int(N * self.max_skip_ratio)
        max_full = int(N * self.max_full_ratio)
        min_linear = int(N * self.min_linear_ratio)

        budget_bias = torch.zeros(N, 4, device=device)

        # ── Skip budget (differentiable logit biasing) ────────────
        if max_skip > 0 and importance is not None:
            skip_order = importance.argsort()
            forced_skip = skip_order[:max_skip]
            # Boost level-0 logit, suppress others → differentiable bias
            budget_bias[forced_skip, 0] += self.bias_strength
            budget_bias[forced_skip, 1:] -= self.bias_strength * 0.5

        # ── Full-attention budget ─────────────────────────────────
        # First pass: estimate which tokens would go to level-3
        if max_full < N and training:
            # Use a soft approximation: find tokens with high L3 probability
            with torch.no_grad():
                pre_biased = logits + budget_bias
                pre_probs = pre_biased.softmax(dim=-1)
                l3_probs = pre_probs[:, 3]
                if l3_probs.argsort(descending=True).numel() > max_full:
                    _, l3_top_idx = l3_probs.topk(max_full, largest=True)
                    l3_keep = torch.zeros(N, dtype=torch.bool, device=device)
                    l3_keep[l3_top_idx] = True
                    l3_downgrade = ~l3_keep & (l3_probs > 0.2)
                    # Penalize level-3 for downgrade candidates
                    budget_bias[l3_downgrade, 3] -= self.bias_strength * 0.7
                    budget_bias[l3_downgrade, 2] += self.bias_strength * 0.3

        # ── Minimum linear-ratio budget ───────────────────────────
        if min_linear > 0 and training:
            with torch.no_grad():
                pre_biased = logits + budget_bias
                pre_probs = pre_biased.softmax(dim=-1)
                l1_probs = pre_probs[:, 1]
                l1_estimated = int((l1_probs > 0.15).sum().item())
                if l1_estimated < min_linear:
                    non_l1_best = ((l1_probs > 0.08) & (l1_probs < 0.3))
                    probs_sorted = l1_probs[non_l1_best].argsort(descending=True)
                    n_needed = min(min_linear - l1_estimated, probs_sorted.numel())
                    if n_needed > 0:
                        promote_idx = non_l1_best.nonzero(as_tuple=True)[0][probs_sorted[:n_needed]]
                        budget_bias[promote_idx, 1] += self.bias_strength * 0.5

        # ── Apply bias and compute assignments ─────────────────────
        biased_logits = logits + budget_bias
        biased_probs = (biased_logits / 1.0).softmax(dim=-1)

        if training:
            # Gumbel-Softmax with straight-through estimator
            # Forward: one-hot. Backward: gradient through soft probabilities.
            hard = F.gumbel_softmax(biased_logits, tau=0.5, hard=True, dim=-1)
            # Integer assignments for control flow (no gradient needed)
            assignments = hard.argmax(dim=-1)
            # Differentiable weights: hard selects which prob value to use
            # hard has straight-through gradient → weights connected to logits
            weights = (hard * biased_probs).sum(dim=-1)
        else:
            # Inference: deterministic argmax
            assignments = biased_probs.argmax(dim=-1)
            hard = torch.zeros_like(biased_probs)
            hard.scatter_(1, assignments.unsqueeze(-1), 1.0)
            weights = biased_probs.max(dim=-1).values

        # ── Hard enforcement for inference-critical constraints ────
        # Full-attention: hard-downgrade excess L3 (post-hoc, no grad)
        if max_full < N:
            l3_mask = (assignments == 3)
            l3_idx = l3_mask.nonzero(as_tuple=True)[0]
            if l3_idx.numel() > max_full:
                # Keep top-max_full most confident, downgrade rest to L2
                l3_idx_sorted = l3_idx[biased_probs[l3_idx, 3].argsort(descending=True)]
                keep = l3_idx_sorted[:max_full]
                downgrade = l3_idx_sorted[max_full:]
                assignments[downgrade] = 2
                weights[downgrade] = biased_probs[downgrade, 2]

        # Linear minimum: hard-promote if still below threshold
        l1_count = (assignments == 1).sum().item()
        if l1_count < min_linear:
            candidates = ((assignments == 0) | (assignments == 2))
            cand_idx = candidates.nonzero(as_tuple=True)[0]
            n_needed = min_linear - l1_count
            if cand_idx.numel() > n_needed:
                l1_scores = biased_probs[cand_idx, 1]
                _, promote = l1_scores.topk(min(n_needed, cand_idx.numel()))
                assignments[cand_idx[promote]] = 1
                weights[cand_idx[promote]] = biased_probs[cand_idx[promote], 1]

        return biased_probs, hard, assignments, weights


# ── Prototype Router ───────────────────────────────────────────────────


class PrototypeRouter(nn.Module):
    """Biases token routing based on prototype similarity.

    Tokens highly similar to known class prototypes are more likely
    to be routed to higher attention levels.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        bias_strength: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_strength = bias_strength

        self.bias_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
        )

    def forward(
        self,
        logits: Tensor,
        tokens: Tensor,
        prototypes: Dict[int, Tensor],
    ) -> Tensor:
        """Add prototype-similarity bias to routing logits.

        Args:
            logits: [N, 4] current routing logits.
            tokens: [N, C] token features.
            prototypes: Dict of class_id → prototype [C].

        Returns:
            logits: [N, 4] biased routing logits.
        """
        if not prototypes:
            return logits

        proto_list = [p for p in prototypes.values()]
        proto_stack = torch.stack(proto_list, dim=0)  # [K, C]

        similarity = F.cosine_similarity(
            tokens.unsqueeze(1), proto_stack.unsqueeze(0), dim=-1,
        )  # [N, K]

        max_sim, _ = similarity.max(dim=-1)       # [N]
        mean_sim = similarity.mean(dim=-1)         # [N]
        ent = -(similarity.softmax(dim=-1) * (similarity.softmax(dim=-1) + 1e-8).log()).sum(dim=-1)  # [N]

        bias_in = torch.stack([max_sim, mean_sim, ent], dim=-1)  # [N, 3]
        bias = self.bias_net(bias_in) * self.bias_strength        # [N, 4]

        return logits + bias


# ── Confidence Estimator ───────────────────────────────────────────────


class ConfidenceEstimator(nn.Module):
    """Estimates per-token routing confidence."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: Tensor,
        probs: Tensor,
        assignments: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Estimate confidence scores.

        Args:
            tokens: [N, C] processed token features.
            probs: [N, 4] routing probability distribution.
            assignments: [N] assigned levels.

        Returns:
            confidence: [N] confidence scores.
            conf_aux: [N] auxiliary confidence.
            entropy: [N] normalized entropy.
        """
        inp = torch.cat([tokens, probs], dim=-1)
        conf = self.conf_head(inp).squeeze(-1)  # [N]
        max_prob = probs.max(dim=-1).values
        conf_aux = max_prob
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1) / math.log(4)
        return conf, conf_aux, entropy


# ── DTR-v2 Main Router ─────────────────────────────────────────────────


@ROUTER.register()
class DTRv2Router(BaseRouter):
    """Dynamic Token Router v2 — 4-level adaptive routing with budget control.

    Pipeline:
        1. RoutingHead: token features → 4-class routing logits
        2. Prototype bias (if prototypes in metadata)
        3. BudgetController: enforce per-level capacity
        4. Token sparsification: prune Level-0
        5. MultiLevelAttention: process at assigned levels
        6. ConfidenceEstimator: per-token routing confidence
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        lowrank_rank: int = 128,
        max_skip_ratio: float = 0.15,
        max_full_ratio: float = 0.25,
        min_linear_ratio: float = 0.20,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
        confidence_weight: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.load_balance_weight = load_balance_weight
        self.confidence_weight = confidence_weight

        self.routing_head = RoutingHead(embed_dim=embed_dim, temperature=temperature)
        self.prototype_router = PrototypeRouter(embed_dim=embed_dim)
        self.budget_controller = BudgetController(
            max_skip_ratio=max_skip_ratio,
            max_full_ratio=max_full_ratio,
            min_linear_ratio=min_linear_ratio,
        )
        self.confidence_estimator = ConfidenceEstimator(embed_dim=embed_dim)
        self.multi_level_attn = MultiLevelAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            lowrank_rank=lowrank_rank,
            dropout=dropout,
        )

        self._last_probs: Optional[Tensor] = None
        self._last_assignments: Optional[Tensor] = None

    def forward(
        self,
        tokens: Tensor,
        metadata: Optional[Dict] = None,
    ) -> RoutingOutput:
        """Route tokens through adaptive 4-level system.

        Gradient flow: routing head logits → budget bias → gumbel_softmax
        (straight-through) → hard one-hot → weights → routed_tokens
        → decoder → loss → backprop to routing head.

        Args:
            tokens: [B, N, C] tile token features.
            metadata: optional dict with keys:
                "importance" — [B, N] importance scores from Ada-SPM.
                "prototypes" — dict of class_id → prototype vector.

        Returns:
            RoutingOutput with processed tokens, assignments, weights, skip mask.
        """
        B, N, C = tokens.shape
        device = tokens.device
        training = self.training

        importance = metadata.get("importance") if metadata else None
        prototypes = metadata.get("prototypes") if metadata else None

        all_routed = []
        all_assignments = []
        all_weights = []
        all_skip_masks = []
        all_confidence = []
        all_conf_aux = []
        all_probs = []
        entropy_sum = torch.tensor(0.0, device=device)

        for b in range(B):
            tok_b = tokens[b]                                                # [N, C]
            imp_b = importance[b] if importance is not None else None        # [N]

            # Step 1: Routing logits and probabilities
            logits_b, probs_b = self.routing_head(tok_b, imp_b)

            # Step 2: Prototype bias
            if prototypes is not None and len(prototypes) > 0:
                logits_b = self.prototype_router(logits_b, tok_b, prototypes)
                probs_b = (logits_b / self.routing_head.temperature).softmax(dim=-1)

            # Step 3: Budget constraints (differentiable logit biasing)
            biased_probs_b, hard_b, assignments_b, weights_b = \
                self.budget_controller(logits_b, probs_b, imp_b, training)
            all_probs.append(biased_probs_b)

            # Step 4: Token sparsification
            skip_mask_b = (assignments_b == 0)                        # [N] bool
            active_idx = (~skip_mask_b).nonzero(as_tuple=True)[0]

            if active_idx.numel() == 0:
                all_routed.append(torch.empty(0, C, device=device))
                all_assignments.append(torch.empty(0, dtype=torch.long, device=device))
                all_weights.append(torch.empty(0, device=device))
                all_skip_masks.append(skip_mask_b)
                all_confidence.append(torch.empty(0, device=device))
                all_conf_aux.append(torch.empty(0, device=device))
                continue

            tok_active = tok_b[active_idx]
            assign_active = assignments_b[active_idx]
            weights_active = weights_b[active_idx]
            probs_active = biased_probs_b[active_idx]

            # Step 5: Multi-level attention
            tok_routed = self.multi_level_attn(tok_active, assign_active)

            # Step 5b: Multiply by routing weights so main loss gradient
            # flows through weights → hard → gumbel_softmax → logits → routing head
            tok_routed = tok_routed * weights_active.unsqueeze(-1)

            # Step 6: Confidence estimation
            conf, conf_aux, _ = self.confidence_estimator(tok_routed, probs_active, assign_active)

            all_routed.append(tok_routed)
            all_assignments.append(assign_active)
            all_weights.append(weights_active)
            all_skip_masks.append(skip_mask_b)
            all_confidence.append(conf)
            all_conf_aux.append(conf_aux)

            entropy_b = -(probs_active * (probs_active + 1e-8).log()).sum(dim=-1).mean()
            entropy_sum = entropy_sum + entropy_b

        # Concatenate across batch
        routed = torch.cat(all_routed, dim=0) if all_routed else torch.empty(0, C, device=device)
        assignments = torch.cat(all_assignments, dim=0).unsqueeze(-1) if all_assignments else torch.empty(0, 1, dtype=torch.long, device=device)
        weights = torch.cat(all_weights, dim=0).unsqueeze(-1) if all_weights else torch.empty(0, 1, device=device)
        skip_mask = torch.cat(all_skip_masks, dim=0)  # [B*N] bool

        # Level distribution stats
        total = assignments.numel()
        level_dist = {
            0: float((assignments.squeeze(-1) == 0).sum().item() + skip_mask.sum().item()) / max(B * N, 1),
            1: float((assignments.squeeze(-1) == 1).sum().item()) / max(B * N, 1),
            2: float((assignments.squeeze(-1) == 2).sum().item()) / max(B * N, 1),
            3: float((assignments.squeeze(-1) == 3).sum().item()) / max(B * N, 1),
        }

        # Aux loss: entropy + load balance
        aux_loss = entropy_sum * self.load_balance_weight
        aux_loss = aux_loss + self.compute_load_balance_loss(assignments.float()) * self.load_balance_weight

        self._last_probs = torch.cat(all_probs, dim=0) if all_probs else None
        self._last_assignments = assignments

        return RoutingOutput(
            routed_tokens=routed,
            assignments=assignments,
            routing_weights=weights,
            skipped_mask=skip_mask,
            aux_loss=aux_loss,
            stats={
                "level_distribution": level_dist,
                "entropy": float(entropy_sum.item() / max(B, 1)),
                "mean_weight": float(weights.mean().item()) if weights.numel() > 0 else 0.0,
            },
        )

    def compute_load_balance_loss(self, routing_weights: Tensor) -> Tensor:
        """Load-balancing loss encouraging uniform level utilization."""
        N = routing_weights.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=routing_weights.device)

        assigns = routing_weights.squeeze(-1).long()
        fractions = []
        for l in range(4):
            fractions.append((assigns == l).float().mean())
        f = torch.stack(fractions)
        return 4.0 * (f ** 2).sum() - 1.0

    @property
    def last_probs(self) -> Optional[Tensor]:
        """Return the last biased routing probabilities [N, 4]."""
        return self._last_probs

    @property
    def last_assignments(self) -> Optional[Tensor]:
        """Return the last routing assignments [N, 1]."""
        return self._last_assignments


# ── Uniform Router Baseline ─────────────────────────────────────────────


@ROUTER.register()
class UniformRouter(BaseRouter):
    """Baseline: route all tokens to the same attention level.

    Useful for ablation studies comparing adaptive vs. fixed routing.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        route_level: int = 2,
        num_heads: int = 8,
        lowrank_rank: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.route_level = route_level

        self.multi_level_attn = MultiLevelAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            lowrank_rank=lowrank_rank,
        )

    def forward(
        self,
        tokens: Tensor,
        metadata: Optional[Dict] = None,
    ) -> RoutingOutput:
        B, N, C = tokens.shape
        device = tokens.device

        all_routed = []
        for b in range(B):
            tok_b = tokens[b]
            assign_b = torch.full((N,), self.route_level, device=device, dtype=torch.long)
            routed = self.multi_level_attn(tok_b, assign_b)
            all_routed.append(routed)

        routed = torch.cat(all_routed, dim=0)
        assignments = torch.full((routed.shape[0], 1), self.route_level, device=device, dtype=torch.long)
        weights = torch.ones(routed.shape[0], 1, device=device)
        skip_mask = torch.zeros(B * N, dtype=torch.bool, device=device)

        return RoutingOutput(
            routed_tokens=routed,
            assignments=assignments,
            routing_weights=weights,
            skipped_mask=skip_mask,
            aux_loss=None,
            stats={"route_level": self.route_level},
        )

    def compute_load_balance_loss(self, routing_weights: Tensor) -> Tensor:
        return torch.tensor(0.0, device=routing_weights.device)


# ── Identity Router Baseline ────────────────────────────────────────────


@ROUTER.register()
class IdentityRouter(BaseRouter):
    """Pass-through router: returns tokens unchanged, no routing.

    Simplest possible baseline. Every token is kept, no levels assigned.
    Useful as a "no routing" control for ablation studies.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: Tensor,
        metadata: Optional[Dict] = None,
    ) -> RoutingOutput:
        B, N, C = tokens.shape
        device = tokens.device

        routed = tokens.reshape(B * N, C)
        assignments = torch.zeros(B * N, 1, dtype=torch.long, device=device)
        weights = torch.ones(B * N, 1, device=device)
        skip_mask = torch.zeros(B * N, dtype=torch.bool, device=device)

        return RoutingOutput(
            routed_tokens=routed,
            assignments=assignments,
            routing_weights=weights,
            skipped_mask=skip_mask,
            aux_loss=None,
            stats={"mode": "identity"},
        )

    def compute_load_balance_loss(self, routing_weights: Tensor) -> Tensor:
        return torch.tensor(0.0, device=routing_weights.device)
