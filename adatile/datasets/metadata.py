"""Metadata manager for dataset-level statistics and tile caching information.

Tracks and persists:
    - Image-level stats (resolution, instance count, density)
    - Tile-level stats (size distribution, entropy, skip ratio)
    - Class distribution
    - Dataset splits and few-shot configurations
    - Cache validation and integrity
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ImageMeta:
    """Per-image statistics."""
    image_id: int
    file_name: str
    width: int
    height: int
    instances: int = 0
    density: float = 0.0
    tile_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TileMeta:
    """Aggregated tile statistics across the dataset."""
    total_tiles: int = 0
    size_distribution: Dict[int, int] = field(default_factory=dict)
    avg_density: Dict[int, float] = field(default_factory=dict)
    avg_entropy: float = 0.0
    sparse_ratio: float = 0.0


@dataclass
class DatasetMeta:
    """Complete dataset metadata."""
    name: str = ""
    version: str = "1.0"
    created: str = ""
    total_images: int = 0
    total_instances: int = 0
    num_classes: int = 0
    categories: Dict[int, str] = field(default_factory=dict)
    class_distribution: Dict[int, int] = field(default_factory=dict)
    class_distribution_norm: Dict[str, float] = field(default_factory=dict)
    image_stats: Dict[str, float] = field(default_factory=dict)
    tile_stats: Dict[str, float] = field(default_factory=dict)
    splits: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MetadataManager:
    """Manages dataset-level metadata: computation, persistence, querying.

    Usage:
        mgr = MetadataManager("datasets/iSAID/metadata")
        mgr.compute_from_dataset(dataset)
        mgr.save()
        stats = mgr.load()
    """

    def __init__(self, meta_dir: str):
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._meta: Optional[DatasetMeta] = None

    # ── Property ─────────────────────────────────────────────────

    @property
    def meta(self) -> DatasetMeta:
        if self._meta is None:
            self._meta = self.load()
        return self._meta

    # ── Compute ──────────────────────────────────────────────────

    def compute_from_dataset(self, dataset) -> DatasetMeta:
        """Compute full metadata from a BaseDataset instance.

        Args:
            dataset: Any BaseDataset subclass with image and annotation access.

        Returns:
            DatasetMeta with all computed statistics.
        """
        # Category mapping
        categories = {}
        try:
            categories = dataset._load_category_mapping()
        except (NotImplementedError, AttributeError, FileNotFoundError) as e:
            # Dataset may not have category mapping implemented yet
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Could not load category mapping from {type(dataset).__name__}: {e}"
            )

        # Image-level stats
        image_stats = dataset.compute_image_stats()

        # Class distribution
        class_dist = dataset.compute_class_distribution()
        class_dist_norm = dataset.compute_class_distribution_normalized()
        class_dist_norm_str = {str(k): v for k, v in class_dist_norm.items()}

        # Per-image metadata
        all_images = []
        total_instances = 0
        for idx in range(len(dataset)):
            iid = dataset.image_ids[idx] if idx < len(dataset.image_ids) else idx
            info = dataset.get_image_info(dataset.image_ids[idx]) if idx < len(dataset.image_ids) else {}
            n_inst = dataset.get_instance_count(idx)
            total_instances += n_inst

            all_images.append(ImageMeta(
                image_id=int(iid),
                file_name=info.get("file_name", str(iid)),
                width=info.get("width", 0),
                height=info.get("height", 0),
                instances=n_inst,
                density=dataset.get_instance_density(idx),
            ))

        self._meta = DatasetMeta(
            name=dataset.__class__.__name__,
            created=datetime.now().isoformat(),
            total_images=len(dataset),
            total_instances=total_instances,
            num_classes=len(categories),
            categories=categories,
            class_distribution=class_dist,
            class_distribution_norm=class_dist_norm_str,
            image_stats=image_stats,
        )
        return self._meta

    def compute_tile_stats(
        self,
        tile_cache: Any,
        tile_sizes: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Aggregate tile-level statistics from a TileCache.

        Args:
            tile_cache: TileCache instance.
            tile_sizes: Tile sizes to aggregate (auto-detects from cache dirs).

        Returns:
            Dict with per-size ratios, skip_ratio, avg_entropy.
        """
        raw_stats = tile_cache.get_cache_stats()

        # Compute derived metrics
        sizes = tile_sizes or [384, 768, 1536, 3072]
        tile_stats = {}

        total = sum(
            raw_stats.get(f"{s}_count", 0) for s in sizes
        )

        for s in sizes:
            count = raw_stats.get(f"{s}_count", 0)
            tile_stats[f"{s}_count"] = count
            tile_stats[f"{s}_ratio"] = count / max(total, 1)

        # Estimate skip ratio (tiles that would be pruned by sparsity)
        # Approximated from low-density tiles
        skip_est = raw_stats.get("skip_estimate", 0.0)
        tile_stats["skip_ratio"] = skip_est

        if self._meta:
            self._meta.tile_stats = tile_stats

        return tile_stats

    # ── Persistence ──────────────────────────────────────────────

    def save(self, meta: Optional[DatasetMeta] = None) -> None:
        """Save metadata as JSON files.

        Creates:
            - {meta_dir}/dataset_meta.json        (main metadata)
            - {meta_dir}/category_mapping.json     (category map)
            - {meta_dir}/image_stats.json          (image-level stats)
            - {meta_dir}/tile_stats.json           (tile-level stats)
            - {meta_dir}/class_distribution.json   (class distribution)
        """
        if meta is not None:
            self._meta = meta
        if self._meta is None:
            raise ValueError("No metadata to save. Call compute_from_dataset() first.")

        # Main metadata
        self._save_json("dataset_meta.json", {
            "name": self._meta.name,
            "version": self._meta.version,
            "created": self._meta.created,
            "total_images": self._meta.total_images,
            "total_instances": self._meta.total_instances,
            "num_classes": self._meta.num_classes,
        })

        # Category mapping
        self._save_json("category_mapping.json", self._meta.categories)

        # Image stats
        self._save_json("image_stats.json", self._meta.image_stats)

        # Tile stats
        if self._meta.tile_stats:
            self._save_json("tile_stats.json", self._meta.tile_stats)

        # Class distribution
        self._save_json("class_distribution.json", {
            "counts": self._meta.class_distribution,
            "normalized": self._meta.class_distribution_norm,
        })

    def load(self) -> DatasetMeta:
        """Load metadata from disk.

        Returns:
            DatasetMeta (with empty/default if files not found).
        """
        main = self._load_json("dataset_meta.json")
        categories = self._load_json("category_mapping.json")
        image_stats = self._load_json("image_stats.json")
        tile_stats = self._load_json("tile_stats.json")
        class_dist = self._load_json("class_distribution.json")

        return DatasetMeta(
            name=main.get("name", ""),
            version=main.get("version", "1.0"),
            created=main.get("created", ""),
            total_images=main.get("total_images", 0),
            total_instances=main.get("total_instances", 0),
            num_classes=main.get("num_classes", 0),
            categories={int(k): v for k, v in categories.items()} if categories else {},
            class_distribution=class_dist.get("counts", {}) if class_dist else {},
            class_distribution_norm=class_dist.get("normalized", {}) if class_dist else {},
            image_stats=image_stats if image_stats else {},
            tile_stats=tile_stats if tile_stats else {},
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _save_json(self, filename: str, data: Dict) -> None:
        path = self.meta_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_json(self, filename: str) -> Dict:
        path = self.meta_dir / filename
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    # ── Summary ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable metadata summary."""
        m = self.meta
        lines = [
            f"=== {m.name} Dataset Metadata ===",
            f"  Total images:     {m.total_images}",
            f"  Total instances:  {m.total_instances}",
            f"  Classes:          {m.num_classes}",
            f"  Avg instances:    {m.image_stats.get('avg_instances', 'N/A')}",
            f"  Avg density:      {m.image_stats.get('avg_density', 'N/A')}",
        ]
        if m.tile_stats:
            lines.append("  Tile distribution:")
            for k, v in m.tile_stats.items():
                if "ratio" in k:
                    lines.append(f"    {k}: {v:.4f}")
        if m.class_distribution:
            lines.append("  Class distribution:")
            for cid, count in sorted(m.class_distribution.items()):
                name = m.categories.get(cid, str(cid))
                ratio = m.class_distribution_norm.get(str(cid), 0.0)
                lines.append(f"    [{cid}] {name}: {count} ({ratio:.2%})")
        return "\n".join(lines)
