"""Cache Manager — orchestrates tile caching, metadata, and statistics.

Central orchestration for:
    - Tile precomputation at scale (384, 768, 1536, 3072)
    - Density map batch generation
    - Metadata aggregation and persistence
    - Cache validation and cleanup
    - Lazy dataset-level statistics

Usage:
    from adatile.datasets import CacheManager, CocoDataset

    dataset = CocoDataset("datasets/iSAID", split="train")
    cache_mgr = CacheManager("datasets/iSAID", dataset)
    cache_mgr.precompute_all_tiles()
    cache_mgr.generate_all_density_maps()
    cache_mgr.save_all_metadata()
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from adatile.datasets.cache.tile_cache import TileCache
from adatile.datasets.metadata import MetadataManager, DatasetMeta


class CacheManager:
    """Orchestrates tile caching, density maps, and metadata for a dataset.

    Provides a high-level API for preparing a dataset for AdaTile-FastSAM training.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset=None,
        tile_sizes: Optional[List[int]] = None,
        cache_subdir: str = "tiles",
        density_subdir: str = "density_maps",
        meta_subdir: str = "metadata",
        num_workers: int = 4,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset = dataset
        self.tile_sizes = tile_sizes or [384, 768, 1536, 3072]
        self.num_workers = num_workers

        # Sub-caches
        self.tile_cache = TileCache(str(self.dataset_root / cache_subdir))
        self.density_dir = self.dataset_root / density_subdir
        self.density_dir.mkdir(parents=True, exist_ok=True)
        self.meta_manager = MetadataManager(str(self.dataset_root / meta_subdir))

    # ── Tile Precomputation ──────────────────────────────────────

    def precompute_all_tiles(
        self,
        splits: Optional[List[str]] = None,
        strides: Optional[List[int]] = None,
        resume: bool = True,
    ) -> Dict[str, int]:
        """Precompute tile cache for all images across splits.

        Args:
            splits: Data splits to process (default: all in cache).
            strides: Per-size stride override (default: tile_size // 2).
            resume: Skip already-cached tiles.

        Returns:
            Dict of {split: num_tiles_generated}.
        """
        if strides is None:
            strides = [ts // 2 for ts in self.tile_sizes]

        if splits is None:
            splits = ["train", "val"]
        if self.dataset is not None:
            splits = [self.dataset.split]

        stats = {}

        for split in splits:
            image_dir = self.dataset_root / "images" / split
            if not image_dir.exists():
                print(f"  [skip] {image_dir} does not exist")
                stats[split] = 0
                continue

            image_paths = sorted(image_dir.glob("*"))
            valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            image_paths = [p for p in image_paths if p.suffix.lower() in valid_exts]

            print(f"Precomputing tiles for {split}: {len(image_paths)} images")
            print(f"  Tile sizes: {self.tile_sizes}")
            print(f"  Strides: {strides}")

            total_tiles = 0

            def _process(img_path: Path) -> int:
                image_id = img_path.stem
                try:
                    infos = self.tile_cache.precompute_tiles(
                        str(img_path),
                        image_id,
                        self.tile_sizes,
                        strides,
                        split=split,
                    )
                    return len(infos)
                except Exception as exc:
                    print(f"  Error {img_path.name}: {exc}")
                    return 0

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_process, p): p for p in image_paths}
                with tqdm(total=len(futures), desc=f"  {split}") as pbar:
                    for future in as_completed(futures):
                        total_tiles += future.result()
                        pbar.update(1)

            stats[split] = total_tiles
            print(f"  -> {total_tiles} tiles cached")

        # Save aggregate tile stats
        tile_stats = self.tile_cache.get_cache_stats()
        tile_stats["splits"] = stats
        self._save_tile_stats(tile_stats)

        return stats

    def precompute_tiles_for_image(
        self,
        image_path: str,
        image_id: Optional[str] = None,
        split: str = "train",
        strides: Optional[List[int]] = None,
    ) -> List:
        """Precompute tiles for a single image.

        Args:
            image_path: Path to the image file.
            image_id: Identifier (default: filename stem).
            split: Data split name.
            strides: Per-size strides.

        Returns:
            List of TileInfo for generated tiles.
        """
        if image_id is None:
            image_id = Path(image_path).stem
        if strides is None:
            strides = [ts // 2 for ts in self.tile_sizes]
        return self.tile_cache.precompute_tiles(
            image_path, image_id, self.tile_sizes, strides, split
        )

    # ── Density Map Generation ───────────────────────────────────

    def generate_all_density_maps(
        self,
        stride: int = 16,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Generate and save density maps for all dataset images.

        Requires self.dataset to be set.

        Args:
            stride: Downsample factor.
            splits: Data splits (default: from dataset).

        Returns:
            {split: num_maps_generated}.
        """
        if self.dataset is None:
            raise ValueError("CacheManager.dataset must be set for density map generation.")

        save_dir = str(self.density_dir / self.dataset.split)
        count = 0
        for idx in tqdm(range(len(self.dataset)), desc="  Density maps"):
            self.dataset.save_density_map(idx, save_dir, stride)
            count += 1

        return {self.dataset.split: count}

    # ── Metadata ─────────────────────────────────────────────────

    def compute_and_save_metadata(self) -> DatasetMeta:
        """Compute full dataset metadata and persist to disk.

        Returns:
            DatasetMeta with all statistics.
        """
        if self.dataset is None:
            raise ValueError("CacheManager.dataset must be set.")

        meta = self.meta_manager.compute_from_dataset(self.dataset)
        self.meta_manager.compute_tile_stats(self.tile_cache, self.tile_sizes)
        self.meta_manager.save()
        return meta

    def save_all_metadata(self) -> None:
        """Convenience method: compute + save all metadata."""
        self.compute_and_save_metadata()

    def load_metadata(self) -> DatasetMeta:
        """Load cached metadata from disk."""
        return self.meta_manager.load()

    def print_summary(self) -> None:
        """Print a human-readable metadata summary."""
        try:
            meta = self.meta_manager.load()
            if meta.total_images == 0 and self.dataset is not None:
                meta = self.compute_and_save_metadata()
            print(self.meta_manager.summary())
        except Exception as e:
            print(f"Could not load metadata: {e}")

    # ── Validation ───────────────────────────────────────────────

    def validate_cache(self) -> Dict[str, Any]:
        """Validate tile cache integrity.

        Returns:
            Dict with: missing_count, corrupted_count, total_checked, valid_ratio.
        """
        missing = 0
        corrupted = 0
        checked = 0

        for pt_file in self.tile_cache.cache_root.rglob("*.pt"):
            checked += 1
            try:
                import torch
                t = torch.load(pt_file, weights_only=True)
                if not isinstance(t, torch.Tensor):
                    corrupted += 1
            except Exception:
                corrupted += 1

        # Check for missing tiles referenced in metadata
        for meta_file in self.tile_cache.cache_root.rglob("meta.json"):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            for tile_key, info in meta.items():
                tile_path = meta_file.parent / f"{tile_key}.pt"
                if not tile_path.exists():
                    missing += 1

        valid = checked - corrupted
        return {
            "total_checked": checked,
            "missing_count": missing,
            "corrupted_count": corrupted,
            "valid_count": valid,
            "valid_ratio": valid / max(checked, 1),
        }

    # ── Helpers ──────────────────────────────────────────────────

    def _save_tile_stats(self, stats: Dict) -> None:
        path = self.tile_cache.cache_root / "cache_meta" / "tile_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
