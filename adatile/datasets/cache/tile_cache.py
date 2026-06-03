"""Adaptive tile cache system with entropy tracking and sparse statistics.

Pre-computes and caches image tiles at 4 resolution levels:
    384  — fine detail (small objects)
    768  — moderate (medium objects)
    1536 — coarse overview (large objects)
    3072 — full-image context

Each tile stores:
    - RGB tensor
    - TileInfo metadata (position, size, density)
    - Shannon entropy of the tile region
    - Sparse region flag (background-dominated)

Disk layout:
    tiles/{size}/{split}/{image_id}/
        {hash}.pt       — tile tensor [3, H, W]
        meta.json       — all tiles' metadata for this image
        stats.json      — aggregated stats per size
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from adatile.core import TileInfo

# ── Constants ───────────────────────────────────────────────────────

DEFAULT_TILE_SIZES = [384, 768, 1536, 3072]
TILE_SIZE_LABELS = {384: "fine", 768: "moderate", 1536: "coarse", 3072: "context"}


# ── Tile Cache ──────────────────────────────────────────────────────


class TileCache:
    """Multi-resolution disk-backed tile cache.

    Features:
        - 4-resolution pyramid (384, 768, 1536, 3072)
        - Shannon entropy per tile
        - Sparse region detection
        - GT density from annotations
        - Fast hash-based path indexing
        - Batch precomputation with progress tracking
    """

    def __init__(
        self,
        cache_root: str,
        tile_sizes: Optional[List[int]] = None,
        track_entropy: bool = True,
        track_sparsity: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.tile_sizes = tile_sizes or DEFAULT_TILE_SIZES
        self.track_entropy = track_entropy
        self.track_sparsity = track_sparsity

    # ── Path Resolution ──────────────────────────────────────────

    @staticmethod
    def _make_hash(seed: str) -> str:
        return hashlib.md5(seed.encode()).hexdigest()[:12]

    def get_tile_dir(
        self,
        image_id: str,
        tile_size: int,
        split: str = "train",
    ) -> Path:
        """Directory for a specific image's tiles at a given size."""
        return self.cache_root / str(tile_size) / split / str(image_id)

    def get_tile_path(
        self,
        image_id: str,
        tile_size: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        split: str = "train",
    ) -> Path:
        """Cache file path for a tile."""
        tile_key = f"{image_id}_{tile_size}_{x1}_{y1}_{x2}_{y2}"
        filename = self._make_hash(tile_key) + ".pt"
        return self.get_tile_dir(image_id, tile_size, split) / filename

    def has_tile(
        self,
        image_id: str,
        tile_size: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        split: str = "train",
    ) -> bool:
        return self.get_tile_path(image_id, tile_size, x1, y1, x2, y2, split).exists()

    # ── Save / Load ──────────────────────────────────────────────

    def save_tile(
        self,
        tile: Tensor,
        tile_info: TileInfo,
        split: str = "train",
        entropy: Optional[float] = None,
        is_sparse: Optional[bool] = None,
    ) -> str:
        """Save a tile and update metadata.

        Args:
            tile: [3, H, W] float32 tensor in [0, 1] or [0, 255].
            tile_info: Tile metadata.
            split: Data split.
            entropy: Pre-computed Shannon entropy (computed if None and track_entropy=True).
            is_sparse: Whether this tile is background-dominated.

        Returns:
            Path where tile was saved.
        """
        tile_path = self.get_tile_path(
            tile_info.image_id, tile_info.tile_size,
            tile_info.x1, tile_info.y1, tile_info.x2, tile_info.y2,
            split,
        )
        tile_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute entropy if tracking
        if entropy is None and self.track_entropy:
            entropy = self._compute_tile_entropy(tile)

        # Compute sparsity if tracking
        if is_sparse is None and self.track_sparsity:
            is_sparse = self._is_tile_sparse(tile)

        # Save tile
        torch.save(tile, tile_path)

        # Update metadata
        self._update_meta(tile_info, tile_path.parent, entropy, is_sparse)

        return str(tile_path)

    def load_tile(
        self,
        image_id: str,
        tile_size: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        split: str = "train",
    ) -> Tuple[Tensor, Optional[TileInfo]]:
        """Load a tile and its metadata from cache.

        Returns:
            tile: [3, H, W] float32 tensor.
            tile_info: TileInfo with metadata (None if meta missing).
        """
        tile_path = self.get_tile_path(
            image_id, tile_size, x1, y1, x2, y2, split,
        )
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not cached: {tile_path}")

        tile = torch.load(tile_path, weights_only=True)

        # Load metadata
        tile_info = self._load_tile_info(image_id, tile_size, x1, y1, x2, y2, split)
        return tile, tile_info

    def load_tiles_batch(
        self,
        requests: List[Tuple[str, int, int, int, int, int, str]],
    ) -> List[Tuple[Tensor, Optional[TileInfo]]]:
        """Batch-load multiple tiles.

        Args:
            requests: List of (image_id, tile_size, x1, y1, x2, y2, split) tuples.

        Returns:
            List of (tile, tile_info) tuples.
        """
        results = []
        for req in requests:
            try:
                results.append(self.load_tile(*req))
            except FileNotFoundError:
                results.append((None, None))
        return results

    # ── Metadata Persistence ─────────────────────────────────────

    def _update_meta(
        self,
        tile_info: TileInfo,
        tile_dir: Path,
        entropy: Optional[float] = None,
        is_sparse: Optional[bool] = None,
    ) -> None:
        """Update the meta.json and stats.json for a tile directory."""
        meta_path = tile_dir / "meta.json"

        # Load existing
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

        # Update entry
        entry = asdict(tile_info)
        if entropy is not None:
            entry["entropy"] = entropy
        if is_sparse is not None:
            entry["is_sparse"] = is_sparse

        meta[tile_info.tile_id] = entry

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update aggregated stats
        self._update_stats(tile_dir, tile_info.tile_size)

    def _update_stats(self, tile_dir: Path, tile_size: int) -> None:
        """Recalculate aggregated stats for a tile directory."""
        meta_path = tile_dir / "meta.json"
        stats_path = tile_dir / "stats.json"
        if not meta_path.exists():
            return

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Aggregate
        densities = []
        entropies = []
        sparse_count = 0
        for entry in meta.values():
            if "object_density" in entry:
                densities.append(entry["object_density"])
            if "entropy" in entry:
                entropies.append(entry["entropy"])
            if entry.get("is_sparse", False):
                sparse_count += 1

        stats = {
            "tile_size": tile_size,
            "total_tiles": len(meta),
            "sparse_count": sparse_count,
            "sparse_ratio": sparse_count / max(len(meta), 1),
            "avg_density": float(np.mean(densities)) if densities else 0.0,
            "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def _load_tile_info(
        self,
        image_id: str,
        tile_size: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        split: str = "train",
    ) -> Optional[TileInfo]:
        """Load metadata for a specific tile."""
        meta_path = self.get_tile_dir(image_id, tile_size, split) / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Search by coordinates
        for entry in meta.values():
            if (entry.get("x1") == x1 and entry.get("y1") == y1 and
                    entry.get("x2") == x2 and entry.get("y2") == y2):
                return TileInfo(
                    tile_id=entry.get("tile_id", ""),
                    image_id=entry.get("image_id", image_id),
                    x1=entry["x1"],
                    y1=entry["y1"],
                    x2=entry["x2"],
                    y2=entry["y2"],
                    tile_size=entry.get("tile_size", tile_size),
                    object_density=entry.get("object_density", 0.0),
                )
        return None

    # ── Analysis Methods ─────────────────────────────────────────

    @staticmethod
    def _compute_tile_entropy(tile: Tensor, num_bins: int = 64) -> float:
        """Compute Shannon entropy for a tile.

        Args:
            tile: [3, H, W] or [1, H, W] tensor.
            num_bins: Histogram resolution.

        Returns:
            Mean per-channel Shannon entropy.
        """
        import numpy as np
        arr = tile.cpu().numpy()
        if arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)  # CHW → HWC

        # Normalize to 0–255
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)

        from scipy.stats import entropy as scipy_entropy
        total_ent = 0.0
        channels = arr.shape[2] if arr.ndim == 3 else 1

        for c in range(channels):
            channel = arr[:, :, c].ravel() if arr.ndim == 3 else arr.ravel()
            hist, _ = np.histogram(channel, bins=num_bins, range=(0, 255))
            hist = hist.astype(np.float64)
            hist = hist / (hist.sum() + 1e-12)
            total_ent += scipy_entropy(hist + 1e-12) / channels

        return float(total_ent)

    @staticmethod
    def _is_tile_sparse(
        tile: Tensor,
        bg_ratio_threshold: float = 0.95,
        edge_threshold: float = 0.02,
    ) -> bool:
        """Determine if a tile is background-dominated (sparse).

        Uses two signals:
            1. Low edge density (smooth / uniform regions)
            2. Low variance

        Args:
            tile: [3, H, W] float32 [0, 1].
            bg_ratio_threshold: Edge density below this → sparse.
            edge_threshold: Gradient magnitude threshold.

        Returns:
            True if tile is background-dominated.
        """
        arr = tile.cpu().numpy()
        if arr.ndim == 3:
            gray = arr.mean(axis=0)  # [H, W]
        else:
            gray = arr

        # Edge density
        gy = np.abs(np.diff(gray, axis=0))
        gx = np.abs(np.diff(gray, axis=1))
        # Align shapes
        edge_density = (gy[:, :-1].mean() + gx[:-1, :].mean()) / 2.0

        # Variance
        variance = gray.var()

        is_sparse = (edge_density < edge_threshold) and (variance < 0.01)
        return bool(is_sparse)

    # ── Precomputation ───────────────────────────────────────────

    def precompute_tiles(
        self,
        image_path: str,
        image_id: str,
        tile_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        split: str = "train",
        density_generator: Optional[callable] = None,
        annotations: Optional[List[Dict]] = None,
    ) -> List[TileInfo]:
        """Pre-compute and cache tiles at multiple resolutions.

        Args:
            image_path: Source image file path.
            image_id: Unique identifier.
            tile_sizes: Tile sizes (default: self.tile_sizes).
            strides: Per-size strides (default: tile_size // 2).
            split: Data split.
            density_generator: Optional callable(tile_tensor, annotations)
                               that returns per-tile object density.
            annotations: COCO annotations for GT density computation.

        Returns:
            List of all TileInfo entries created.
        """
        if tile_sizes is None:
            tile_sizes = self.tile_sizes
        if strides is None:
            strides = [max(ts // 2, 128) for ts in tile_sizes]

        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))
        H, W = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        all_infos: List[TileInfo] = []

        for tile_size, stride in zip(tile_sizes, strides):
            # Skip if image is too small for this tile size
            if tile_size > H and tile_size > W:
                continue

            for y in range(0, max(H - tile_size + 1, 1), stride):
                for x in range(0, max(W - tile_size + 1, 1), stride):
                    x2 = min(x + tile_size, W)
                    y2 = min(y + tile_size, H)
                    actual_w = x2 - x
                    actual_h = y2 - y

                    # Skip very small residual tiles
                    if actual_w < tile_size // 4 or actual_h < tile_size // 4:
                        continue

                    # Skip if already cached
                    if self.has_tile(image_id, tile_size, x, y, x2, y2, split):
                        continue

                    tile = image_tensor[:, y:y2, x:x2]

                    # Object density: GT-based if annotations provided, else edge proxy
                    if density_generator is not None:
                        density = density_generator(tile, annotations)
                    elif annotations is not None:
                        density = self._gt_tile_density(
                            annotations, x, y, x2, y2, H, W
                        )
                    else:
                        density = self._edge_density_proxy(tile)

                    # Entropy and sparsity
                    entropy = self._compute_tile_entropy(tile) if self.track_entropy else None
                    is_sparse = self._is_tile_sparse(tile) if self.track_sparsity else None

                    info = TileInfo(
                        tile_id=f"{image_id}_t_{tile_size}_{x}_{y}",
                        image_id=image_id,
                        x1=x,
                        y1=y,
                        x2=x2,
                        y2=y2,
                        tile_size=tile_size,
                        object_density=density,
                    )
                    self.save_tile(tile, info, split, entropy, is_sparse)
                    all_infos.append(info)

        return all_infos

    # ── Density Helpers ──────────────────────────────────────────

    @staticmethod
    def _edge_density_proxy(tile: Tensor) -> float:
        """Simple edge-based object density estimate.

        Returns value in [0, 1].
        """
        gray = tile.mean(dim=0)
        gy = torch.abs(gray[1:] - gray[:-1]).mean()
        gx = torch.abs(gray[:, 1:] - gray[:, :-1]).mean()
        return float(((gy + gx) / 2.0).clamp(0, 1))

    @staticmethod
    def _gt_tile_density(
        annotations: List[Dict],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_h: int,
        img_w: int,
    ) -> float:
        """Compute object density from GT annotations within a tile region.

        Density = number of instances overlapping the tile / tile area.
        """
        tile_area = (x2 - x1) * (y2 - y1)
        if tile_area == 0:
            return 0.0

        count = 0
        for ann in annotations:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            bx, by, bw, bh = bbox
            if (bx + bw > x1 and bx < x2 and by + bh > y1 and by < y2):
                count += 1

        return min(count / max(tile_area, 1), 1.0)

    # ── Statistics ───────────────────────────────────────────────

    def get_cache_stats(self) -> Dict[str, float]:
        """Aggregate statistics across the entire cache.

        Returns:
            Dict with per-size counts, ratios, skip estimates, and entropy averages.
        """
        stats: Dict[str, float] = {}
        total = 0
        size_counts: Dict[int, int] = {}
        sparse_total = 0

        for size_dir in sorted(self.cache_root.iterdir()):
            if not size_dir.is_dir():
                continue
            try:
                size = int(size_dir.name)
            except ValueError:
                continue

            # Count .pt files
            pt_count = 0
            entropies = []
            for pt_file in size_dir.rglob("*.pt"):
                pt_count += 1

            # Read aggregated stats per split
            for stats_file in size_dir.rglob("stats.json"):
                with open(stats_file, "r") as f:
                    s = json.load(f)
                entropies.append(s.get("avg_entropy", 0.0))
                sparse_total += s.get("sparse_count", 0)

            size_counts[size] = pt_count
            total += pt_count

            stats[f"{size}_count"] = pt_count
            if entropies:
                stats[f"{size}_avg_entropy"] = float(np.mean(entropies))

        # Ratios
        if total > 0:
            for size, count in size_counts.items():
                stats[f"{size}_ratio"] = count / total
            stats["skip_estimate"] = sparse_total / max(total, 1)

        stats["total_tiles"] = total
        return stats

    def get_tile_stats_for_image(
        self, image_id: str, tile_size: int, split: str = "train"
    ) -> Dict[str, Any]:
        """Load aggregated tile stats for a specific image."""
        stats_path = self.get_tile_dir(image_id, tile_size, split) / "stats.json"
        if not stats_path.exists():
            return {}
        with open(stats_path, "r") as f:
            return json.load(f)

    def get_all_meta_for_image(
        self, image_id: str, tile_size: int, split: str = "train"
    ) -> Dict[str, Any]:
        """Load full tile metadata for an image."""
        meta_path = self.get_tile_dir(image_id, tile_size, split) / "meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r") as f:
            return json.load(f)

    def query_tiles_by_density(
        self,
        image_id: str,
        tile_size: int,
        min_density: float = 0.0,
        max_density: float = 1.0,
        split: str = "train",
    ) -> List[TileInfo]:
        """Query tiles within a density range."""
        meta = self.get_all_meta_for_image(image_id, tile_size, split)
        results = []
        for entry in meta.values():
            d = entry.get("object_density", 0.0)
            if min_density <= d <= max_density:
                results.append(TileInfo(
                    tile_id=entry.get("tile_id", ""),
                    image_id=image_id,
                    x1=entry["x1"],
                    y1=entry["y1"],
                    x2=entry["x2"],
                    y2=entry["y2"],
                    tile_size=tile_size,
                    object_density=d,
                ))
        return results

    def query_sparse_tiles(
        self,
        image_id: str,
        tile_size: int,
        split: str = "train",
    ) -> List[TileInfo]:
        """Return only tiles marked as sparse."""
        meta = self.get_all_meta_for_image(image_id, tile_size, split)
        results = []
        for entry in meta.values():
            if entry.get("is_sparse", False):
                results.append(TileInfo(
                    tile_id=entry.get("tile_id", ""),
                    image_id=image_id,
                    x1=entry["x1"],
                    y1=entry["y1"],
                    x2=entry["x2"],
                    y2=entry["y2"],
                    tile_size=tile_size,
                    object_density=entry.get("object_density", 0.0),
                ))
        return results

    # ── Cache Maintenance ────────────────────────────────────────

    def clear_size(self, tile_size: int, split: Optional[str] = None) -> int:
        """Remove all tiles of a specific size.

        Args:
            tile_size: Tile size to clear.
            split: Optional split filter.

        Returns:
            Number of files removed.
        """
        import shutil

        size_dir = self.cache_root / str(tile_size)
        if not size_dir.exists():
            return 0

        if split:
            target = size_dir / split
            count = sum(1 for _ in target.rglob("*.pt"))
            shutil.rmtree(target, ignore_errors=True)
        else:
            count = sum(1 for _ in size_dir.rglob("*.pt"))
            shutil.rmtree(size_dir, ignore_errors=True)

        return count

    def validate(self, fix: bool = False) -> Dict[str, Any]:
        """Validate cache integrity.

        Args:
            fix: If True, remove corrupted files.

        Returns:
            Dict with validation report.
        """
        corrupted_files = []
        missing_meta = []
        total = 0

        for pt_file in sorted(self.cache_root.rglob("*.pt")):
            total += 1
            try:
                t = torch.load(pt_file, weights_only=True)
                if not isinstance(t, torch.Tensor):
                    corrupted_files.append(str(pt_file))
                    if fix:
                        pt_file.unlink()
            except Exception:
                corrupted_files.append(str(pt_file))
                if fix:
                    pt_file.unlink()

        # Check meta.json presence
        for meta_file in sorted(self.cache_root.rglob("meta.json")):
            tile_dir = meta_file.parent
            pt_files = list(tile_dir.glob("*.pt"))
            if not pt_files:
                missing_meta.append(str(meta_file))

        return {
            "total_files": total,
            "corrupted": corrupted_files,
            "num_corrupted": len(corrupted_files),
            "missing_meta_dirs": missing_meta,
            "num_missing_meta": len(missing_meta),
            "healthy": total - len(corrupted_files),
        }
