"""Tests for TilePlanner skip modes: threshold, hard, topk + PlannerStats."""

import pytest
import torch
import numpy as np

from adatile.tokenizer.tile_planner import TilePlanner, TilePlan, TileSpec, PlannerStats


def _make_importance(H_s=8, W_s=8, pattern="mixed"):
    """Create synthetic importance maps for testing."""
    if pattern == "all_high":
        return torch.full((H_s, W_s), 0.9)
    elif pattern == "all_low":
        return torch.full((H_s, W_s), 0.1)
    elif pattern == "mixed":
        imp = torch.full((H_s, W_s), 0.1)
        imp[:H_s // 2, :W_s // 2] = 0.9  # top-left high
        imp[H_s // 2:, W_s // 2:] = 0.6  # bottom-right medium
        return imp
    elif pattern == "gradient":
        y = torch.linspace(0, 1, H_s)
        x = torch.linspace(0, 1, W_s)
        return y.unsqueeze(1) * x.unsqueeze(0)


class TestPlannerStats:
    """PlannerStats dataclass."""

    def test_basic_construction(self):
        stats = PlannerStats(
            total_cells=64, skipped_cells=16, borderline_cells=8,
            cells_with_tiles=40, skip_mode="hard",
        )
        assert stats.total_cells == 64
        assert stats.skipped_cells == 16
        assert stats.borderline_cells == 8
        assert stats.cells_with_tiles == 40
        assert stats.skip_ratio == 0.25

    def test_skip_ratio_zero(self):
        stats = PlannerStats(
            total_cells=100, skipped_cells=0, borderline_cells=0,
            cells_with_tiles=100,
        )
        assert stats.skip_ratio == 0.0

    def test_skip_ratio_all(self):
        stats = PlannerStats(
            total_cells=64, skipped_cells=64, borderline_cells=0,
            cells_with_tiles=0,
        )
        assert stats.skip_ratio == 1.0

    def test_memory_saved_mb(self):
        stats = PlannerStats(
            total_cells=64, skipped_cells=32, borderline_cells=0,
            cells_with_tiles=32,
            estimated_memory_saved_bytes=32 * 256 * 4,
        )
        expected_mb = (32 * 256 * 4) / (1024 * 1024)
        assert abs(stats.memory_saved_mb - expected_mb) < 0.001


class TestThresholdMode:
    """Threshold mode — backward compatible current behavior."""

    def test_backward_compatible(self):
        """Threshold mode produces coarse tiles for borderline cells."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.5,
            skip_mode="threshold",
        )
        # Half high, half borderline (0.3 > 0.5*0.5 and < 0.5)
        imp = torch.full((4, 4), 0.3)
        plan = planner.plan(imp, image_size=(1024, 1024))

        # Borderline cells should get coarse tiles (size_idx=3, 3072)
        for spec in plan.specs:
            assert spec.scale_level == 3
            assert spec.tile_size == 3072

        assert plan.planner_stats is not None
        assert plan.planner_stats.skip_mode == "threshold"
        assert plan.planner_stats.borderline_cells == 16  # all 16 are borderline

    def test_definite_skip(self):
        """imp < 0.5*threshold → definitely skipped."""
        planner = TilePlanner(
            importance_threshold=0.5,
            skip_mode="threshold",
        )
        imp = torch.full((4, 4), 0.1)  # < 0.5*0.5 = 0.25
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) == 0
        assert plan.planner_stats.skipped_cells == 16

    def test_high_importance_fine_tiles(self):
        """High importance → fine tiles (size 384)."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.3,
            skip_mode="threshold",
        )
        imp = torch.full((4, 4), 0.9)  # > 0.75 → size_idx 0 (384)
        plan = planner.plan(imp, image_size=(1024, 1024))

        for spec in plan.specs:
            assert spec.scale_level == 0
            assert spec.tile_size == 384


class TestHardMode:
    """Hard skip mode — truly skip cells below threshold."""

    def test_skips_borderline(self):
        """imp < threshold → NO tile at all (not even coarse)."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.5,
            skip_mode="hard",
        )
        # 0.3 < 0.5 → should all be skipped
        imp = torch.full((4, 4), 0.3)
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) == 0
        assert plan.planner_stats.skipped_cells == 16
        assert plan.planner_stats.borderline_cells == 0

    def test_keeps_high_importance(self):
        """High importance cells still get tiles in hard mode."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.5,
            skip_mode="hard",
        )
        imp = torch.full((4, 4), 0.9)
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) == 16
        assert plan.planner_stats.skipped_cells == 0
        assert plan.planner_stats.cells_with_tiles == 16

    def test_fewer_tiles_than_threshold(self):
        """Hard mode produces strictly fewer tiles for mixed importance."""
        # Create map with values in the borderline range [0.25, 0.5)
        # for threshold=0.5: threshold mode gives coarse tiles, hard mode skips
        imp = torch.full((8, 8), 0.35)  # borderline: >= 0.5*0.5 and < 0.5
        imp[:4, :4] = 0.9  # top-left: high importance (fine tiles in both modes)
        imp[4:, 4:] = 0.1  # bottom-right: below threshold (skipped in both modes)

        planner_t = TilePlanner(
            importance_threshold=0.5,
            skip_mode="threshold",
        )
        plan_t = planner_t.plan(imp, image_size=(1024, 1024))

        planner_h = TilePlanner(
            importance_threshold=0.5,
            skip_mode="hard",
        )
        plan_h = planner_h.plan(imp, image_size=(1024, 1024))

        # Hard mode skips the 0.35 borderline cells (32 cells),
        # while threshold mode keeps them as coarse tiles
        assert len(plan_h.specs) < len(plan_t.specs), (
            f"Hard mode ({len(plan_h.specs)}) should have fewer tiles "
            f"than threshold mode ({len(plan_t.specs)})"
        )

    def test_hard_skip_multiplier(self):
        """hard_skip_multiplier=0.5 only skips cells below 0.5*threshold."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.6,
            skip_mode="hard",
            hard_skip_multiplier=0.5,
        )
        # 0.4: < 0.6 but > 0.6*0.5=0.3 → NOT skipped
        imp = torch.full((4, 4), 0.4)
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) > 0  # should still generate tiles

    def test_all_zero_importance(self):
        """All-zero importance → 0 tiles in hard mode."""
        planner = TilePlanner(
            importance_threshold=0.5,
            skip_mode="hard",
        )
        imp = torch.zeros(4, 4)
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) == 0
        assert plan.planner_stats.skipped_cells == 16


class TestTopKMode:
    """Top-K mode — keep top K% by importance."""

    def test_respects_ratio(self):
        """With hard_skip_multiplier=0.5, roughly half the cells get tiles."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.3,
            skip_mode="topk",
            hard_skip_multiplier=0.5,  # keep top 50%
        )
        imp = torch.rand(8, 8)
        plan = planner.plan(imp, image_size=(1024, 1024))

        total = 8 * 8  # 64
        expected = int(total * 0.5)  # 32
        assert abs(len(plan.specs) - expected) <= 2, (
            f"Expected ~{expected} tiles, got {len(plan.specs)}"
        )

    def test_all_high_still_truncated(self):
        """Top-K mode still enforces ratio even if all tiles have high importance."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            skip_mode="topk",
            hard_skip_multiplier=0.25,  # keep top 25%
        )
        imp = torch.full((4, 4), 0.9)
        plan = planner.plan(imp, image_size=(1024, 1024))

        total = 16
        expected = max(1, int(total * 0.25))  # 4
        assert len(plan.specs) == expected


class TestTilePlan:
    """TilePlan construction."""

    def test_planner_stats_attached(self):
        """TilePlan always has planner_stats after plan()."""
        planner = TilePlanner(skip_mode="threshold")
        imp = torch.rand(4, 4)
        plan = planner.plan(imp, image_size=(512, 512))

        assert plan.planner_stats is not None
        assert isinstance(plan.planner_stats, PlannerStats)
        assert plan.planner_stats.total_cells == 16

    def test_default_skip_mode_is_threshold(self):
        """Default constructor uses threshold mode."""
        planner = TilePlanner()
        assert planner.skip_mode == "threshold"

    def test_token_budget_limits(self):
        """Max tokens enforces an upper bound."""
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            max_tokens=8,
            skip_mode="threshold",
        )
        imp = torch.full((8, 8), 0.9)  # 64 high-importance cells
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert len(plan.specs) <= 8


class TestQuadtreeMode:
    """Quadtree plan should also have planner_stats."""

    def test_quadtree_has_planner_stats(self):
        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
        )
        imp = torch.rand(8, 8)
        plan = planner.plan_quadtree(imp, image_size=(1024, 1024))

        assert plan.planner_stats is not None
        assert plan.planner_stats.skip_mode == "quadtree"


class TestConfigRoundtrip:
    """skip_mode survives Config → build_tokenizer → TilePlanner."""

    def test_build_tokenizer_passes_skip_mode(self):
        from adatile.tokenizer import build_tokenizer
        tokenizer = build_tokenizer(
            "DynamicTileTokenizerImpl",
            tile_sizes=[384, 768, 1536],
            strides=[0.5, 0.5, 0.5],
            max_tokens=4096,
            skip_mode="hard",
            hard_skip_multiplier=0.8,
        )
        assert tokenizer.planner.skip_mode == "hard"
        assert tokenizer.planner.hard_skip_multiplier == 0.8

    def test_build_tokenizer_default_is_threshold(self):
        from adatile.tokenizer import build_tokenizer
        tokenizer = build_tokenizer(
            "DynamicTileTokenizerImpl",
            tile_sizes=[384, 768],
            strides=[0.5, 0.5],
            max_tokens=1024,
        )
        assert tokenizer.planner.skip_mode == "threshold"
        assert tokenizer.planner.hard_skip_multiplier == 1.0
