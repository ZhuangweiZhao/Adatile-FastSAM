"""Importance-guided adaptive tile planner for dynamic token allocation.

Given an importance map S ∈ [0,1] and granularity map T from Ada-SPM,
determines WHERE to place tiles and at WHAT resolution.

Algorithms:
    1. Threshold-based: simple importance binarization + fixed tile sizes
    2. Quadtree decomposition: recursive splitting based on region density
    3. Token-budget-aware: top-K selection by importance to enforce budget

Output: List of TileSpec (position, size, stride) for tile extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from adatile.core import TileInfo


# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class TileSpec:
    """Planned tile: position, size, priority for extraction."""

    x1: int
    y1: int
    x2: int
    y2: int
    tile_size: int          # actual tile dimension in pixels
    stride: int             # effective stride (for overlap control)
    importance: float       # average importance in this region
    priority: float         # sort key: higher = extract first
    density: float = 0.0    # estimated object density in this tile
    scale_level: int = 0    # 0=finest(384), 1=moderate(768), 2=coarse(1536), 3=context(3072)

    def to_tile_info(self, image_id: str, idx: int) -> TileInfo:
        """Convert to TileInfo for cache / metadata systems."""
        return TileInfo(
            tile_id=f"{image_id}_tile_{idx:04d}_{self.tile_size}_{self.x1}_{self.y1}",
            image_id=image_id,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            tile_size=self.tile_size,
            object_density=self.density,
        )


@dataclass
class PlannerStats:
    """Real sparsity statistics from the tile planning stage.

    Tracks how many cells were truly skipped BEFORE token generation,
    so that compute savings are real, not fake.
    """

    total_cells: int
    skipped_cells: int
    borderline_cells: int
    cells_with_tiles: int
    estimated_flops_saved: int = 0
    estimated_memory_saved_bytes: int = 0
    skip_mode: str = "threshold"

    @property
    def skip_ratio(self) -> float:
        return self.skipped_cells / max(self.total_cells, 1)

    @property
    def memory_saved_mb(self) -> float:
        return self.estimated_memory_saved_bytes / (1024 * 1024)


@dataclass
class TilePlan:
    """Complete tile allocation plan for one image.

    Contains the ordered list of tiles to extract plus allocation metadata.
    """

    specs: List[TileSpec]
    image_size: Tuple[int, int]
    total_tiles: int
    active_tiles: int
    skipped_regions: int
    token_budget_used: int
    token_budget_max: int
    planner_stats: Optional[PlannerStats] = None

    @property
    def skip_ratio(self) -> float:
        return self.skipped_regions / max(self.total_tiles, 1)

    @property
    def budget_utilization(self) -> float:
        return self.token_budget_used / max(self.token_budget_max, 1)


# ── Tile Planner Base ────────────────────────────────────────────────


class TilePlanner:
    """Importance-guided tile planner.

    Converts Ada-SPM output (importance + granularity) into a concrete
    tile extraction plan.

    Args:
        tile_sizes: Available tile sizes in pixels [384, 768, 1536, 3072].
        strides: Overlap stride per tile size (fraction of tile_size).
        importance_threshold: Skip regions with importance below this.
        max_tokens: Hard limit on number of tiles per image.
        min_tile_coverage: Minimum fraction of image that must be covered.
        overlap_ratio: Overlap between adjacent tiles as fraction of tile_size.
    """

    def __init__(
        self,
        tile_sizes: Optional[List[int]] = None,
        strides: Optional[List[float]] = None,
        importance_threshold: float = 0.3,
        max_tokens: int = 4096,
        min_tile_coverage: float = 0.05,
        overlap_ratio: float = 0.25,
        skip_mode: str = "threshold",
        hard_skip_multiplier: float = 1.0,
    ):
        self.tile_sizes = tile_sizes or [384, 768, 1536, 3072]
        self.strides = strides or [0.75, 0.50, 0.25, 0.125]  # finer tiles overlap more
        self.importance_threshold = importance_threshold
        self.max_tokens = max_tokens
        self.min_tile_coverage = min_tile_coverage
        self.overlap_ratio = overlap_ratio
        self.skip_mode = skip_mode
        self.hard_skip_multiplier = hard_skip_multiplier

        # Configurable tile-size thresholds for the _imp_to_size_idx() fallback.
        # These map importance values to tile size indices (index 0 = finest).
        # Default: [0.75, 0.50, 0.30] — verified as reasonable in ablation (Q7).
        # Override via set_tile_size_thresholds() or by passing tile_size_thresholds.
        self._tile_size_thresholds = [0.75, 0.50, 0.30]

        # Calibrated per-tile costs — initialized with reasonable defaults
        # based on a 256-dim token through a lightweight conv stem.
        # These are overwritten when set_calibrated_costs() is called
        # with empirically measured values from PipelineProfiler.
        # Default: ~0.5 GFLOPs per 224×224 tile patch, ~2 MB memory.
        self._calibrated_flops_per_tile: float = 500_000_000  # 0.5 GFLOPs
        self._calibrated_mem_per_tile: float = 2_000_000      # 2 MB

    # ── Main Planning API ────────────────────────────────────────

    def plan(
        self,
        importance: Tensor,
        image_size: Tuple[int, int],
        granularity_hard: Optional[Tensor] = None,
        granularity_soft: Optional[Tensor] = None,
        image_id: str = "",
    ) -> TilePlan:
        """Generate a tile allocation plan.

        Args:
            importance: [H_s, W_s] importance map in [0, 1].
            image_size: (H, W) of the original image.
            granularity_hard: Optional [H_s, W_s] tile-size index per cell.
            granularity_soft: Optional [K, H_s, W_s] soft tile-size probs.
            image_id: Identifier for debug logging.

        Returns:
            TilePlan with ordered tile specs and PlannerStats.
        """
        H, W = image_size
        H_s, W_s = importance.shape[-2:]
        total_cells = H_s * W_s

        # Compute per-cell tile decisions
        tile_specs, skipped_cells, borderline_cells, cells_with_tiles = \
            self._allocate_by_importance(importance, image_size, granularity_hard)

        # Top-K mode: post-hoc filtering by importance
        if self.skip_mode == "topk":
            n_before = len(tile_specs)
            tile_specs.sort(key=lambda t: t.priority, reverse=True)
            keep_n = max(1, int(total_cells * self.hard_skip_multiplier))
            tile_specs = tile_specs[:keep_n]
            skipped_cells = total_cells - len(tile_specs)
            borderline_cells = n_before - len(tile_specs)
            cells_with_tiles = len(tile_specs)
        else:
            # Enforce token budget: sort by priority, keep top-K
            tile_specs.sort(key=lambda t: t.priority, reverse=True)
            skipped_cells += max(0, len(tile_specs) - self.max_tokens)

        active_specs = tile_specs[:self.max_tokens]

        # Compute PlannerStats with savings estimates.
        # Uses calibrated costs when available (from PipelineProfiler),
        # otherwise sensible defaults (0.5 GFLOPs/tile, 2 MB/tile).
        flops_per_tile = self._calibrated_flops_per_tile
        mem_per_tile = self._calibrated_mem_per_tile
        planner_stats = PlannerStats(
            total_cells=total_cells,
            skipped_cells=skipped_cells,
            borderline_cells=borderline_cells,
            cells_with_tiles=cells_with_tiles,
            estimated_flops_saved=skipped_cells * flops_per_tile,
            estimated_memory_saved_bytes=skipped_cells * mem_per_tile,
            skip_mode=self.skip_mode,
        )

        return TilePlan(
            specs=active_specs,
            image_size=image_size,
            total_tiles=total_cells,
            active_tiles=len(active_specs),
            skipped_regions=skipped_cells,
            token_budget_used=len(active_specs),
            token_budget_max=self.max_tokens,
            planner_stats=planner_stats,
        )

    # ── Allocation Strategies ────────────────────────────────────

    def _allocate_by_importance(
        self,
        importance: Tensor,
        image_size: Tuple[int, int],
        granularity_hard: Optional[Tensor] = None,
    ) -> Tuple[List[TileSpec], int, int, int]:
        """Cell-by-cell tile allocation based on importance + granularity.

        Each cell in the importance map corresponds to a region in the image.
        Low-importance cells → coarse or skip; high-importance → fine tiles.

        Args:
            importance: [H_s, W_s] in [0, 1].
            image_size: (H, W) full image size.
            granularity_hard: Optional [H_s, W_s] int tile-size index.

        Returns:
            (specs, skipped_cells, borderline_cells, cells_with_tiles)
        """
        H, W = image_size
        H_s, W_s = importance.shape[-2:]

        # NOTE: Tile planning is a discrete spatial decision (which cells get
        # tiles and at what size). This cannot be backpropagated through
        # directly — we use .detach() for the NumPy cell loop. Gradients
        # from the downstream segmentation loss reach Ada-SPM through two
        # differentiable paths:
        #   1. compute_planning_alignment_loss() (below) — loss on plan vs. importance
        #   2. The router's per-token importance extraction (segmentation/base.py:197)
        imp = importance.squeeze().detach().cpu().numpy()
        if granularity_hard is not None:
            gr_hard = granularity_hard.squeeze().detach().cpu().numpy()
        else:
            gr_hard = None

        specs = []
        skipped_cells = 0
        borderline_cells = 0
        cell_h = H / H_s
        cell_w = W / W_s

        for y_cell in range(H_s):
            for x_cell in range(W_s):
                imp_val = float(imp[y_cell, x_cell])

                if self.skip_mode == "hard":
                    # Hard mode: imp < threshold → truly skip (no tile at all)
                    if imp_val < self.importance_threshold * self.hard_skip_multiplier:
                        skipped_cells += 1
                        continue
                    if gr_hard is not None:
                        size_idx = int(gr_hard[y_cell, x_cell])
                    else:
                        size_idx = self._imp_to_size_idx(imp_val)

                elif self.skip_mode == "topk":
                    # Top-K mode: collect all cells, filter in plan()
                    if gr_hard is not None:
                        size_idx = int(gr_hard[y_cell, x_cell])
                    else:
                        size_idx = self._imp_to_size_idx(imp_val)

                else:  # "threshold" — current behavior, backward compatible
                    if imp_val < self.importance_threshold:
                        if imp_val < self.importance_threshold * 0.5:
                            skipped_cells += 1
                            continue  # definitely skip
                        # Borderline: place context-level tile
                        borderline_cells += 1
                        size_idx = 3  # 3072
                    else:
                        if gr_hard is not None:
                            size_idx = int(gr_hard[y_cell, x_cell])
                        else:
                            size_idx = self._imp_to_size_idx(imp_val)

                tile_size = self.tile_sizes[min(size_idx, len(self.tile_sizes) - 1)]
                stride = int(tile_size * self.strides[min(size_idx, len(self.strides) - 1)])

                # Compute pixel coordinates
                # Center tile on this cell's region
                cx = int((x_cell + 0.5) * cell_w)
                cy = int((y_cell + 0.5) * cell_h)

                x1 = max(0, cx - tile_size // 2)
                y1 = max(0, cy - tile_size // 2)
                x2 = min(W, x1 + tile_size)
                y2 = min(H, y1 + tile_size)

                # Shift back if we went past the right edge
                if x2 == W:
                    x1 = max(0, W - tile_size)
                if y2 == H:
                    y1 = max(0, H - tile_size)

                specs.append(TileSpec(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    tile_size=tile_size,
                    stride=stride,
                    importance=imp_val,
                    priority=imp_val,  # simplicity: priority = importance
                    density=imp_val,
                    scale_level=size_idx,
                ))

        cells_with_tiles = len(specs)
        return specs, skipped_cells, borderline_cells, cells_with_tiles

    def _imp_to_size_idx(self, importance: float) -> int:
        """Map continuous importance to discrete tile size index.

        Uses configurable tile_size_thresholds (set in __init__).
        High importance → fine tiles (index 0).
        Low importance → coarse tiles (index N-1).

        This is a FALLBACK used only when granularity_hard from
        Ada-SPM's GranularityHead is not available. The learned
        GranularityHead is the primary mechanism.
        """
        for i, thresh in enumerate(self._tile_size_thresholds):
            if importance > thresh:
                return i
        return len(self._tile_size_thresholds)  # last (coarsest) index

    # ── Quadtree Decomposition ───────────────────────────────────

    def plan_quadtree(
        self,
        importance: Tensor,
        image_size: Tuple[int, int],
        max_depth: int = 4,
        density_threshold: float = 0.5,
        granularity_hard: Optional[Tensor] = None,
    ) -> TilePlan:
        """Recursive quadtree-based tile allocation.

        Splits regions recursively until:
            - The region's mean importance is above threshold (dense enough for fine tiles)
            - Max depth reached
            - Tile size would be below minimum

        This produces a multi-scale tiling that concentrates fine tiles
        on high-density regions and coarse tiles on low-density background.

        Args:
            importance: [H_s, W_s] in [0, 1].
            image_size: (H, W).
            max_depth: Max recursion depth.
            density_threshold: Split if importance < this.
            granularity_hard: Optional per-cell tile-size index.

        Returns:
            TilePlan with hierarchically allocated tiles.
        """
        H, W = image_size
        specs = []
        depth_tile_size = {
            0: self.tile_sizes[-1],  # 3072 (full image)
            1: self.tile_sizes[-2],  # 1536
            2: self.tile_sizes[1],   # 768
            3: self.tile_sizes[0],   # 384
        }

        def _split(x1: int, y1: int, x2: int, y2: int, depth: int) -> None:
            """Recursive region splitting."""
            if depth >= max_depth:
                tile_size = self.tile_sizes[0]  # minimum
                specs.append(TileSpec(
                    x1=x1, y1=y1, x2=min(x2, W), y2=min(y2, H),
                    tile_size=tile_size,
                    stride=tile_size // 2,
                    importance=1.0,
                    priority=1.0,
                    scale_level=0,
                ))
                return

            # Map region to importance grid
            gx1 = int(x1 / W * importance.shape[-1])
            gy1 = int(y1 / H * importance.shape[-2])
            gx2 = int(x2 / W * importance.shape[-1])
            gy2 = int(y2 / H * importance.shape[-2])

            gx1 = max(0, min(gx1, importance.shape[-1] - 1))
            gy1 = max(0, min(gy1, importance.shape[-2] - 1))
            gx2 = max(gx1 + 1, min(gx2, importance.shape[-1]))
            gy2 = max(gy1 + 1, min(gy2, importance.shape[-2]))

            region_imp = importance[..., gy1:gy2, gx1:gx2].mean().item()

            if region_imp > density_threshold or depth >= max_depth - 1:
                # This region is dense enough — place a tile
                ts = depth_tile_size.get(depth, self.tile_sizes[0])
                specs.append(TileSpec(
                    x1=x1, y1=y1, x2=min(x2, W), y2=min(y2, H),
                    tile_size=ts,
                    stride=ts // 2,
                    importance=region_imp,
                    priority=region_imp,
                    density=region_imp,
                    scale_level=max(0, 3 - depth),
                ))
            else:
                # Split into 4 quadrants
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2
                _split(x1, y1, mx, my, depth + 1)
                _split(mx, y1, x2, my, depth + 1)
                _split(x1, my, mx, y2, depth + 1)
                _split(mx, my, x2, y2, depth + 1)

        _split(0, 0, W, H, 0)

        # Sort by priority and enforce budget
        specs.sort(key=lambda t: t.priority, reverse=True)
        skipped = max(0, len(specs) - self.max_tokens)
        active_specs = specs[:self.max_tokens]

        flops_per_tile = self._calibrated_flops_per_tile
        mem_per_tile = self._calibrated_mem_per_tile
        planner_stats = PlannerStats(
            total_cells=importance.shape[-2] * importance.shape[-1],
            skipped_cells=skipped,
            borderline_cells=0,
            cells_with_tiles=len(active_specs),
            estimated_flops_saved=skipped * flops_per_tile,
            estimated_memory_saved_bytes=skipped * mem_per_tile,
            skip_mode="quadtree",
        )

        return TilePlan(
            specs=active_specs,
            image_size=image_size,
            total_tiles=len(specs),
            active_tiles=len(active_specs),
            skipped_regions=skipped,
            token_budget_used=len(active_specs),
            token_budget_max=self.max_tokens,
            planner_stats=planner_stats,
        )

    # ── Overlap-Aware Planning ───────────────────────────────────

    def add_overlap(
        self,
        specs: List[TileSpec],
        overlap_fraction: float = 0.25,
    ) -> List[TileSpec]:
        """Augment tile plan with overlapping tiles at boundaries.

        For each tile in specs, adds shifted copies at the boundaries
        to ensure smooth merges across tile edges.

        Args:
            specs: Original tile specs.
            overlap_fraction: Overlap as fraction of tile_size.

        Returns:
            Augmented list with overlap tiles appended.
        """
        overlap_specs = []
        for spec in specs:
            ts = spec.tile_size
            shift = int(ts * overlap_fraction)

            if shift <= 0:
                continue

            # Half-shifted tile (right/down)
            overlap_specs.append(TileSpec(
                x1=min(spec.x1 + shift, spec.x2 - ts),
                y1=spec.y1,
                x2=min(spec.x1 + shift + ts, spec.x2),
                y2=spec.y2,
                tile_size=ts,
                stride=shift,
                importance=spec.importance * 0.8,
                priority=spec.priority * 0.5,  # lower priority
                scale_level=spec.scale_level,
            ))
            overlap_specs.append(TileSpec(
                x1=spec.x1,
                y1=min(spec.y1 + shift, spec.y2 - ts),
                x2=spec.x2,
                y2=min(spec.y1 + shift + ts, spec.y2),
                tile_size=ts,
                stride=shift,
                importance=spec.importance * 0.8,
                priority=spec.priority * 0.5,
                scale_level=spec.scale_level,
            ))

        return specs + overlap_specs

    # ── Differentiable Planning Loss ────────────────────────────

    def compute_planning_alignment_loss(
        self,
        importance: Tensor,
        plan: "TilePlan",
    ) -> Tensor:
        """Differentiable loss bridging the planner → Ada-SPM gradient gap.

        Computes how well the discrete tile plan covers high-importance
        regions. This provides a gradient path from planning decisions
        back to Ada-SPM's importance predictor.

        The loss is computed on the importance tensor (which retains
        gradients) using the plan's spatial coverage as a soft target.

        Loss = BCE(importance[covered_cells], 1.0) + BCE(importance[skipped_cells], 0.0)

        This encourages Ada-SPM to predict high importance for regions
        that the planner chose to tile, and low importance for regions
        that were skipped.

        Args:
            importance: [H_s, W_s] soft importance with gradients.
            plan: TilePlan from the discrete planning step.

        Returns:
            Scalar differentiable loss tensor.
        """
        H_s, W_s = importance.shape[-2:]
        device = importance.device

        # Build coverage map: 1.0 for cells that received tiles, 0.0 for skipped
        coverage = torch.zeros(H_s, W_s, device=device, dtype=importance.dtype)
        H_img, W_img = plan.image_size
        cell_h = H_img / H_s
        cell_w = W_img / W_s

        for spec in plan.specs:
            cx = (spec.x1 + spec.x2) / 2
            cy = (spec.y1 + spec.y2) / 2
            gx = min(int(cx / cell_w), W_s - 1)
            gy = min(int(cy / cell_h), H_s - 1)
            coverage[gy, gx] = 1.0

        # Binary cross-entropy: importance should match coverage.
        # BCE requires input in [0,1].
        imp = importance.squeeze().float()
        cov = coverage.float()
        with torch.cuda.amp.autocast(enabled=False):
            # CRITICAL: clamp at 1e-4, not 1e-7!
            # BCE gradient = 1/(imp*(1-imp)) → with clamp(1e-7) the gradient
            # can reach 1e7, which overflows fp16 in the backward pass
            # through the backbone → NaN weights → permanent corruption.
            imp = torch.nan_to_num(imp, nan=0.5).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(imp, cov)
        return loss

    # ── Utility ──────────────────────────────────────────────────

    def set_threshold(self, threshold: float) -> None:
        self.importance_threshold = threshold

    def set_tile_size_thresholds(self, thresholds: list) -> None:
        """Update the tile-size importance thresholds.

        Args:
            thresholds: List of descending floats, e.g. [0.9, 0.7, 0.5].
                        Maps importance > thresholds[0] → size_idx=0, etc.
        """
        self._tile_size_thresholds = list(thresholds)

    def get_tile_size_thresholds(self) -> list:
        """Return current tile-size thresholds (for ablation reporting)."""
        return list(self._tile_size_thresholds)

    def set_calibrated_costs(
        self, flops_per_tile: float, mem_bytes_per_tile: float,
    ) -> None:
        """Calibrate cost estimates from real profiler measurements.

        Call this after running PipelineProfiler to replace placeholder
        analytical estimates with empirically measured per-tile costs.

        Args:
            flops_per_tile: Measured FLOPs per tile from profiler.
            mem_bytes_per_tile: Measured memory per tile from profiler.
        """
        self._calibrated_flops_per_tile = flops_per_tile
        self._calibrated_mem_per_tile = mem_bytes_per_tile

    def get_effective_coverage(
        self, plan: TilePlan, image_size: Tuple[int, int]
    ) -> float:
        """Compute what fraction of image pixels are covered by at least one tile."""
        H, W = image_size
        coverage = torch.zeros(H, W, dtype=torch.bool)

        for spec in plan.specs:
            x1 = max(0, spec.x1)
            y1 = max(0, spec.y1)
            x2 = min(W, spec.x2)
            y2 = min(H, spec.y2)
            coverage[y1:y2, x1:x2] = True

        return float(coverage.float().mean())
