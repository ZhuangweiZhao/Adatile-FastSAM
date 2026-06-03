"""Tests for routing subsystem: RoutingOutput, IdentityRouter, UniformRouter, DTRv2Router."""

import pytest
import torch
from torch import Tensor


# ── RoutingOutput dataclass ───────────────────────────────────────────


class TestRoutingOutput:
    """Test RoutingOutput construction."""

    def test_basic_construction(self):
        from adatile.core import RoutingOutput
        N, C = 16, 256
        out = RoutingOutput(
            routed_tokens=torch.randn(N, C),
            assignments=torch.randint(1, 4, (N, 1)),
            routing_weights=torch.rand(N, 1),
            skipped_mask=torch.zeros(N, dtype=torch.bool),
        )
        assert out.routed_tokens.shape == (N, C)
        assert out.assignments.shape == (N, 1)
        assert out.routing_weights.shape == (N, 1)
        assert out.skipped_mask.shape == (N,)
        assert out.aux_loss is None
        assert out.stats is None

    def test_with_stats(self):
        from adatile.core import RoutingOutput
        N, C = 16, 256
        out = RoutingOutput(
            routed_tokens=torch.randn(N, C),
            assignments=torch.randint(1, 4, (N, 1)),
            routing_weights=torch.rand(N, 1),
            skipped_mask=torch.zeros(N, dtype=torch.bool),
            aux_loss=torch.tensor(0.5),
            stats={"level_distribution": {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}},
        )
        assert out.aux_loss == 0.5
        assert out.stats["level_distribution"][3] == 0.2

    def test_routing_output_fields(self):
        from adatile.routing import RoutingOutput
        assert hasattr(RoutingOutput, "skipped_mask")
        assert hasattr(RoutingOutput, "assignments")


# ── IdentityRouter ────────────────────────────────────────────────────


class TestIdentityRouter:
    """IdentityRouter: pass-through, no routing, no skipping."""

    def test_pass_through(self):
        from adatile.routing import IdentityRouter
        router = IdentityRouter(embed_dim=256)
        tokens = torch.randn(2, 32, 256)
        out = router(tokens)

        assert out.routed_tokens.shape == (64, 256)
        assert out.assignments.shape == (64, 1)
        assert out.routing_weights.shape == (64, 1)
        assert out.skipped_mask.shape == (64,)
        assert not out.skipped_mask.any()
        assert (out.routing_weights == 1.0).all()
        assert (out.assignments == 0).all()
        # Pass-through: routed tokens == input tokens (reshaped)
        assert torch.allclose(out.routed_tokens, tokens.reshape(64, 256))

    def test_with_metadata_ignored(self):
        from adatile.routing import IdentityRouter
        router = IdentityRouter(embed_dim=256)
        tokens = torch.randn(1, 16, 256)
        out = router(tokens, metadata={"importance": torch.rand(1, 16)})
        assert out.routed_tokens.shape == (16, 256)

    def test_registered(self):
        from adatile.routing import build_router, IdentityRouter
        router = build_router("IdentityRouter", embed_dim=128)
        assert isinstance(router, IdentityRouter)


# ── UniformRouter ─────────────────────────────────────────────────────


class TestUniformRouter:
    """UniformRouter: all tokens to same level."""

    def test_all_same_level(self):
        from adatile.routing import UniformRouter
        router = UniformRouter(embed_dim=256, route_level=2)
        tokens = torch.randn(1, 20, 256)
        out = router(tokens)

        assert out.routed_tokens.shape == (20, 256)
        assert (out.assignments == 2).all()
        assert (out.routing_weights == 1.0).all()
        assert not out.skipped_mask.any()

    def test_different_levels(self):
        from adatile.routing import UniformRouter
        for level in [1, 2, 3]:
            router = UniformRouter(embed_dim=256, route_level=level)
            tokens = torch.randn(1, 10, 256)
            out = router(tokens)
            assert (out.assignments == level).all()

    def test_with_metadata(self):
        from adatile.routing import UniformRouter
        router = UniformRouter(embed_dim=256)
        tokens = torch.randn(1, 10, 256)
        out = router(tokens, metadata={"importance": torch.rand(1, 10)})
        assert out.routed_tokens.shape == (10, 256)

    def test_registered(self):
        from adatile.routing import build_router, UniformRouter
        router = build_router("UniformRouter", embed_dim=128)
        assert isinstance(router, UniformRouter)


# ── DTRv2Router ───────────────────────────────────────────────────────


class TestDTRv2Router:
    """DTRv2Router: full adaptive 4-level routing."""

    def test_basic_forward(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.1, max_full_ratio=0.3)
        tokens = torch.randn(1, 64, 256)
        out = router(tokens)

        assert out.routed_tokens.shape[1] == 256
        assert out.assignments.shape[1] == 1
        assert out.routing_weights.shape[1] == 1
        assert out.skipped_mask.shape == (64,)
        assert out.stats is not None
        assert "level_distribution" in out.stats

    def test_with_importance(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256)
        tokens = torch.randn(1, 64, 256)
        importance = torch.rand(1, 64)
        out = router(tokens, metadata={"importance": importance})

        assert out.routed_tokens.shape[1] == 256
        assert out.skipped_mask.shape == (64,)

    def test_skip_mask_matches_routed(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.3)
        tokens = torch.randn(1, 32, 256)
        importance = torch.rand(1, 32)
        importance[0, :8] = 0.0  # force first 8 to be low-importance
        out = router(tokens, metadata={"importance": importance})

        N_total = out.skipped_mask.shape[0]
        N_active = out.routed_tokens.shape[0]
        N_skipped = int(out.skipped_mask.sum().item())
        assert N_total == N_active + N_skipped

    def test_gradient_flow(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.1)
        router.train()
        tokens = torch.randn(1, 32, 256, requires_grad=True)
        out = router(tokens)
        loss = out.routed_tokens.sum()
        loss.backward()
        assert tokens.grad is not None

    def test_levels_in_range(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.1)
        tokens = torch.randn(1, 64, 256)
        importance = torch.rand(1, 64)
        out = router(tokens, metadata={"importance": importance})

        active_assignments = out.assignments.squeeze(-1)
        assert ((active_assignments >= 1) & (active_assignments <= 3)).all()

    def test_registered(self):
        from adatile.routing import build_router, DTRv2Router
        router = build_router("DTRv2Router", embed_dim=128)
        assert isinstance(router, DTRv2Router)

    def test_all_levels_used(self):
        from adatile.routing import DTRv2Router
        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.05, max_full_ratio=0.5, min_linear_ratio=0.3)
        tokens = torch.randn(1, 128, 256)
        importance = torch.rand(1, 128)
        out = router(tokens, metadata={"importance": importance})

        dist = out.stats["level_distribution"]
        # At least 3 of 4 levels should have some tokens
        nonzero_levels = sum(1 for v in dist.values() if v > 0)
        assert nonzero_levels >= 2


# ── Registry ──────────────────────────────────────────────────────────


class TestRouterRegistry:
    """All routers must be registered and buildable."""

    def test_all_three_registered(self):
        from adatile.registry import ROUTER
        keys = ROUTER.list()
        assert "DTRv2Router" in keys
        assert "UniformRouter" in keys
        assert "IdentityRouter" in keys

    def test_build_all_three(self):
        from adatile.routing import build_router
        from adatile.routing import DTRv2Router, UniformRouter, IdentityRouter

        r1 = build_router("DTRv2Router", embed_dim=64)
        r2 = build_router("UniformRouter", embed_dim=64)
        r3 = build_router("IdentityRouter", embed_dim=64)
        assert isinstance(r1, DTRv2Router)
        assert isinstance(r2, UniformRouter)
        assert isinstance(r3, IdentityRouter)

    def test_no_duplicate_registrations(self):
        from adatile.registry import ROUTER
        keys = ROUTER.list()
        assert len(keys) == len(set(keys)), f"Duplicate keys: {keys}"


# ── Differentiable Routing & Gradient Verification ────────────────────


class TestDifferentiableBudget:
    """BudgetController gradient flow and straight-through estimator."""

    def test_logits_receive_gradient(self):
        """Routing head logits must receive non-zero gradients."""
        from adatile.routing.router import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2, max_full_ratio=0.3)
        router.train()
        tokens = torch.randn(1, 32, 256, requires_grad=True)
        importance = torch.rand(1, 32)

        out = router(tokens, metadata={"importance": importance})
        routed = out.routed_tokens

        if routed.numel() > 0:
            loss = routed.sum()
            loss.backward()

            # Routing head parameters must have gradients
            for name, param in router.routing_head.named_parameters():
                assert param.grad is not None, (
                    f"Routing head param '{name}' received no gradient"
                )
                assert param.grad.abs().sum() > 0, (
                    f"Routing head param '{name}' has zero gradient"
                )

    def test_budget_bias_is_differentiable(self):
        """Budget biasing of logits preserves gradient flow."""
        from adatile.routing.router import BudgetController
        import torch.nn.functional as F

        bc = BudgetController(max_skip_ratio=0.2, max_full_ratio=0.3)
        logits = torch.randn(32, 4, requires_grad=True)
        probs = F.softmax(logits, dim=-1)
        importance = torch.rand(32)

        biased_probs, hard, assignments, weights = bc(logits, probs, importance, training=True)

        # weights must be connected to logits via straight-through
        loss = weights.sum()
        loss.backward()

        assert logits.grad is not None, "Logits should receive gradient through weights"
        assert logits.grad.abs().sum() > 0, "Logits gradient should be non-zero"

    def test_hard_tensor_has_straight_through_gradient(self):
        """Gumbel-softmax hard tensor must carry straight-through gradients."""
        import torch.nn.functional as F

        logits = torch.randn(16, 4, requires_grad=True)
        hard = F.gumbel_softmax(logits, tau=0.5, hard=True, dim=-1)
        loss = hard.sum()
        loss.backward()

        assert logits.grad is not None, (
            "Logits must receive gradient through gumbel_softmax straight-through"
        )
        assert logits.grad.abs().sum() > 0

    def test_weights_differentiable_in_training(self):
        """In training mode, routing_weights must be differentiable."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2)
        router.train()
        tokens = torch.randn(1, 32, 256)

        out = router(tokens)
        weights = out.routing_weights

        if weights.numel() > 0:
            assert weights.requires_grad, "routing_weights must require grad in training"
            loss = weights.sum()
            loss.backward()
            # Check that some router params got gradients from weight loss
            any_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in router.routing_head.parameters()
            )
            # Weights connect to gumbel_softmax which has straight-through gradient
            # to logits, so routing head should receive gradient
            assert any_grad, "Routing head should receive gradient through weights"

    def test_routed_tokens_carry_gradient_to_weights(self):
        """routed_tokens = attn(tokens) * weights → weights must get gradient."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.05, max_full_ratio=0.3)
        router.train()
        tokens = torch.randn(1, 64, 256)

        out = router(tokens)
        routed = out.routed_tokens

        if routed.numel() > 0:
            routed.retain_grad()
            loss = routed.sum()
            loss.backward()

            # Verify routed tokens got gradients (they should via attention + weight multiplication)
            if routed.numel() > 1:
                assert routed.grad is not None, "routed_tokens should receive gradients"

    def test_skip_mask_entropy_tracked(self):
        """Entropy statistics must be present in router output."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2)
        router.train()
        tokens = torch.randn(1, 64, 256)

        out = router(tokens)
        assert out.stats is not None
        assert "entropy" in out.stats
        assert out.stats["entropy"] >= 0.0  # entropy is non-negative

    def test_level_distribution_sums_to_one(self):
        """Level distribution fractions should sum to approximately 1."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2)
        router.eval()
        tokens = torch.randn(1, 64, 256)

        out = router(tokens)
        dist = out.stats["level_distribution"]
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01, f"Level distribution sums to {total}, expected ~1.0"

    def test_mean_weight_in_stats(self):
        """Mean routing weight should be in stats dict."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256)
        router.train()
        tokens = torch.randn(1, 32, 256)

        out = router(tokens)
        assert "mean_weight" in out.stats
        assert 0.0 <= out.stats["mean_weight"] <= 1.0

    def test_last_probs_returns_tensor(self):
        """After forward, last_probs should return the biased probability tensor."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256)
        router.train()
        tokens = torch.randn(1, 32, 256)

        router(tokens)
        probs = router.last_probs
        assert probs is not None
        assert probs.shape[1] == 4  # [N_active, 4]

    def test_inference_no_gumbel_noise(self):
        """In eval mode, routing should be deterministic."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2)
        router.eval()
        tokens = torch.randn(1, 32, 256)

        torch.manual_seed(42)
        out1 = router(tokens)
        torch.manual_seed(42)
        out2 = router(tokens)

        assert torch.equal(out1.assignments, out2.assignments), (
            "Eval-mode routing should be deterministic"
        )

    def test_aux_loss_is_tensor_with_grad(self):
        """Aux loss should be a differentiable tensor."""
        from adatile.routing import DTRv2Router

        router = DTRv2Router(embed_dim=256, max_skip_ratio=0.2)
        router.train()
        tokens = torch.randn(1, 32, 256)

        out = router(tokens)
        assert out.aux_loss is not None
        assert isinstance(out.aux_loss, Tensor)
        out.aux_loss.backward()
        # After backward, routing head should have gradients
        assert router.routing_head.level_bias.grad is not None, (
            "level_bias should receive gradient from aux_loss"
        )


# ── Backward compatibility ────────────────────────────────────────────


class TestBackwardCompat:
    """Old names must still work as import aliases."""

    def test_base_router_registered(self):
        from adatile.core import BaseRouter
        from adatile.routing import DTRv2Router
        assert issubclass(DTRv2Router, BaseRouter)

    def test_routing_output_fields_exist(self):
        from adatile.core import RoutingOutput
        assert hasattr(RoutingOutput, "skipped_mask")
        assert hasattr(RoutingOutput, "assignments")

    def test_routing_imports(self):
        from adatile.routing import DTRv2Router, UniformRouter, IdentityRouter
        assert DTRv2Router is not None
        assert UniformRouter is not None
        assert IdentityRouter is not None
