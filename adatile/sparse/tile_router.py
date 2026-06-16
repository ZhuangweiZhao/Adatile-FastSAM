"""TileRouter — Dynamic Tile Routing (DTR simplified).

Takes SPM importance map at H/32, selects top-K tiles, assigns granularity
(small/medium/large tile sizes), and outputs tile coordinates.

Granularity logic:
  - Small objects (high importance, small region)  → small tile  (384px)
  - Medium objects                                  → medium tile (768px)
  - Large objects (broad importance region)         → large tile  (1536px)

This replaces the old DTRv2Router (3-level attention routing) with a
simpler but more effective spatial tile planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TileSpec:
    """Single tile specification."""
    x: int          # top-left x in SPM grid coordinates
    y: int          # top-left y in SPM grid coordinates
    size: int       # tile side length (384, 768, or 1536)
    score: float    # mean importance within tile
    level: int      # 0=small, 1=medium, 2=large


class TileRouter(nn.Module):
    """Dynamic tile router: importance → tile coordinates + granularity.

    Input:  importance [B, 1, H_s, W_s] from SPM
    Output: List[TileSpec] per image

    Config:
        tile_sizes: [small, medium, large] pixel sizes. Default [384, 768, 1536].
        keep_ratio: fraction of grid cells to keep.
        overlap: tile overlap in SPM grid cells.
    """

    TILE_SIZES = [384, 768, 1536]   # small, medium, large (pixels at stride=1)

    def __init__(
        self,
        tile_sizes: Optional[List[int]] = None,
        keep_ratio: float = 0.15,
        overlap: int = 0,
        spm_stride: int = 32,
        image_size: int = 1024,
    ):
        super().__init__()
        self.tile_sizes = tile_sizes or self.TILE_SIZES
        self.keep_ratio = keep_ratio
        self.overlap = overlap
        self.spm_stride = spm_stride
        self.image_size = image_size

    def forward(
        self, importance: torch.Tensor
    ) -> Tuple[List[List[TileSpec]], torch.Tensor]:
        """Route tiles based on importance.

        Args:
            importance: [B, 1, H_s, W_s] in [0, 1].

        Returns:
            tile_specs: List (per batch) of List[TileSpec].
            keep_mask: [B, H_s, W_s] boolean mask of selected cells.
        """
        B, _, H_s, W_s = importance.shape
        device = importance.device

        all_specs: List[List[TileSpec]] = []
        all_masks: List[torch.Tensor] = []

        for b in range(B):
            imp = importance[b, 0]  # [H_s, W_s]

            # ── Step 1: Top-K cell selection ─────────────────
            n_cells = H_s * W_s
            n_keep = max(1, int(n_cells * self.keep_ratio))
            imp_flat = imp.reshape(-1)
            _, top_indices = imp_flat.topk(n_keep)

            keep_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)
            keep_mask[top_indices] = True
            keep_mask_2d = keep_mask.reshape(H_s, W_s)
            all_masks.append(keep_mask_2d)

            # ── Step 2: Assign granularity per kept cell ──────
            specs: List[TileSpec] = []
            for idx in top_indices:
                y = (idx // W_s).item()
                x = (idx % W_s).item()
                score = imp[y, x].item()

                # Granularity based on local importance context
                y0 = max(0, y - 1)
                y1 = min(H_s, y + 2)
                x0 = max(0, x - 1)
                x1 = min(W_s, x + 2)
                local_imp = imp[y0:y1, x0:x1]
                local_mean = local_imp.mean().item()
                local_std = local_imp.std().item()

                # High mean + low std → concentrated object → small tile
                # Low mean + high std → scattered objects → large tile
                # Otherwise → medium tile
                if local_mean > 0.7 and local_std < 0.15:
                    level = 0   # small
                elif local_mean > 0.4:
                    level = 1   # medium
                else:
                    level = 2   # large

                specs.append(TileSpec(
                    x=x, y=y,
                    size=self.tile_sizes[level],
                    score=score,
                    level=level,
                ))

            # ── Step 3: Merge overlapping tiles ──────────────
            specs = self._merge_overlapping(specs, H_s, W_s)
            all_specs.append(specs)

        keep_mask_batch = torch.stack(all_masks, dim=0)  # [B, H_s, W_s]
        return all_specs, keep_mask_batch

    # ── Tile merging ────────────────────────────────────────

    def _merge_overlapping(
        self, specs: List[TileSpec], H_s: int, W_s: int
    ) -> List[TileSpec]:
        """Merge tiles with IoU > 0.5. Larger tile absorbs smaller."""
        if len(specs) <= 1:
            return specs

        # Sort by score descending — keep highest-importance tiles first
        specs = sorted(specs, key=lambda s: s.score, reverse=True)
        merged: List[TileSpec] = []

        for s in specs:
            absorbed = False
            for m in merged:
                iou = self._tile_iou(s, m, H_s, W_s)
                if iou > 0.5:
                    # Absorb: use larger tile size, keep higher score
                    if s.score > m.score:
                        m.score = s.score
                    m.size = max(m.size, s.size)
                    m.level = min(m.level, s.level)  # smaller level = more detailed
                    absorbed = True
                    break
            if not absorbed:
                merged.append(s)

        return merged

    @staticmethod
    def _tile_iou(a: TileSpec, b: TileSpec, H_s: int, W_s: int) -> float:
        """Compute IoU of two tiles in SPM grid coordinates."""
        # Tile extent in SPM grid (approximate: tile_size / spm_stride / 2 for radius)
        r_a = max(1, a.size // 64)  # rough radius in grid cells
        r_b = max(1, b.size // 64)

        ax1, ay1 = max(0, a.x - r_a), max(0, a.y - r_a)
        ax2, ay2 = min(W_s, a.x + r_a), min(H_s, a.y + r_a)
        bx1, by1 = max(0, b.x - r_b), max(0, b.y - r_b)
        bx2, by2 = min(W_s, b.x + r_b), min(H_s, b.y + r_b)

        # Intersection
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter

        return inter / max(union, 1)

    # ── Coordinate conversion ───────────────────────────────

    def grid_to_pixel(
        self, x: int, y: int, tile_size: int, h_img: int, w_img: int
    ) -> Tuple[int, int, int, int]:
        """Convert SPM grid tile to pixel coordinates.

        Args:
            x, y: SPM grid cell coordinates.
            tile_size: Tile side length in pixels.
            h_img, w_img: Original image dimensions.

        Returns:
            (x1, y1, x2, y2) pixel coordinates, clamped to image bounds.
        """
        # SPM cell center in pixel space
        cx = int((x + 0.5) * self.spm_stride * w_img / (self.image_size * self.spm_stride / 32))
        cy = int((y + 0.5) * self.spm_stride * h_img / (self.image_size * self.spm_stride / 32))

        # Simpler: tile centered on SPM cell
        cx = x * self.spm_stride * w_img // (self.image_size)
        cy = y * self.spm_stride * h_img // (self.image_size)

        half = tile_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w_img, cx + half)
        y2 = min(h_img, cy + half)

        return x1, y1, x2, y2
