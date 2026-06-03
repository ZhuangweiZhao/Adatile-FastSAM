"""COCO-style instance segmentation datasets for AdaTile-FastSAM.

Provides:
    - CocoDataset: general COCO JSON loader with full mask/density support
    - ISAIDDataset: iSAID remote sensing (15 classes, up to 4000×4000 px)
    - LoveDADataset: LoveDA land-cover (7 classes, 1024×1024 px)
    - Tile entropy and sparse-region statistics per image
    - Class distribution analysis tools
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from adatile.datasets.base import BaseDataset
from adatile.registry import DATASET


@DATASET.register()
class CocoDataset(BaseDataset):
    """COCO-style instance segmentation dataset with full analysis support.

    Loads images and polygon/RLE annotations from COCO JSON format.

    Features:
        - Lazy annotation loading via cached indices
        - Polygon → mask conversion with pycocotools
        - Density map generation (per-instance + mask-aggregation)
        - Tile entropy computation (Shannon entropy per tile region)
        - Sparse region ratio (background fraction)
        - Class distribution aggregation

    Args:
        root_dir: Dataset root (e.g., "datasets/COCO").
        image_dir: Subdirectory for images (appended with split).
        anno_file: Path to COCO JSON annotation file.
        split: "train", "val", or "test".
        transforms: Optional albumentations Compose transform.
        filter_empty: Drop images with zero annotations.
        min_visibility: Minimum annotation visibility (not currently used).
    """

    COCO_CATEGORIES = None  # Override in subclasses

    def __init__(
        self,
        root_dir: str,
        image_dir: str = "images",
        anno_file: Optional[str] = None,
        split: str = "train",
        transforms: Optional[callable] = None,
        filter_empty: bool = True,
        min_visibility: float = 0.0,
    ):
        super().__init__(
            root_dir=root_dir,
            image_dir=image_dir,
            split=split,
            transforms=transforms,
        )
        self.anno_file = Path(anno_file) if anno_file else self.anno_dir / f"instances_{split}.json"
        self.filter_empty = filter_empty
        self.min_visibility = min_visibility

        # Lazy-loaded state
        self._coco_data: Optional[Dict] = None
        self._image_to_anns: Optional[Dict[int, List[Dict]]] = None
        self._filename_to_image_id: Optional[Dict[str, int]] = None

    # ── Lazy Loading ─────────────────────────────────────────────

    @property
    def coco_data(self) -> Dict:
        if self._coco_data is None:
            if not self.anno_file.exists():
                raise FileNotFoundError(
                    f"COCO annotation file not found: {self.anno_file}"
                )
            with open(self.anno_file, "r") as f:
                self._coco_data = json.load(f)
            self._build_indices()
        return self._coco_data

    def _build_indices(self) -> None:
        """Build all lookup indices from the COCO JSON."""
        data = self._coco_data

        # Images
        self._image_ids = [img["id"] for img in data["images"]]
        self._image_info = {img["id"]: img for img in data["images"]}
        self._filename_to_image_id = {
            img["file_name"]: img["id"] for img in data["images"]
        }

        # Handle case where file_name contains path separators
        for img in data["images"]:
            name = Path(img["file_name"]).name
            self._filename_to_image_id[name] = img["id"]

        # Categories
        if "categories" in data:
            self._category_mapping = {
                cat["id"]: cat["name"] for cat in data["categories"]
            }

        # Annotations indexed by image_id
        self._image_to_anns = defaultdict(list)
        for ann in data.get("annotations", []):
            self._image_to_anns[ann["image_id"]].append(ann)

        # Filter empty images
        if self.filter_empty:
            self._image_ids = [
                iid for iid in self._image_ids
                if len(self._image_to_anns.get(iid, [])) > 0
            ]

    # ── Core I/O ─────────────────────────────────────────────────

    def _load_image(self, index: int) -> np.ndarray:
        image_id = self._image_ids[index]
        info = self._image_info[image_id]
        file_name = info["file_name"]

        # Try multiple possible paths
        candidates = [
            self.image_dir / file_name,
            self.image_dir / Path(file_name).name,
            self.root_dir / file_name,
            self.root_dir / "images" / file_name,
        ]
        for p in candidates:
            if p.exists():
                with Image.open(p) as img:
                    return np.array(img.convert("RGB"))

        raise FileNotFoundError(
            f"Image not found for {file_name}. Tried: {candidates}"
        )

    def _load_annotations(self, index: int) -> List[Dict[str, Any]]:
        image_id = self._image_ids[index]
        return self._image_to_anns.get(image_id, [])

    def _load_category_mapping(self) -> Dict[int, str]:
        _ = self.coco_data  # trigger lazy load
        return self._category_mapping

    # ── Filename Indexing ────────────────────────────────────────

    @property
    def filename_to_image_id(self) -> Dict[str, int]:
        """Mapping from filename → COCO image_id."""
        _ = self.coco_data
        return self._filename_to_image_id or {}

    def get_index_by_filename(self, filename: str) -> Optional[int]:
        """Get dataset index from an image filename."""
        name = Path(filename).name
        img_id = self.filename_to_image_id.get(name)
        if img_id is None:
            return None
        try:
            return self._image_ids.index(img_id)
        except ValueError:
            return None

    def get_index_by_image_id(self, image_id: int) -> Optional[int]:
        """Get dataset index from a COCO image_id."""
        try:
            return self._image_ids.index(image_id)
        except ValueError:
            return None

    def get_image_by_id(self, image_id: int) -> Dict:
        """Retrieve a specific item by COCO image_id."""
        idx = self.get_index_by_image_id(image_id)
        if idx is None:
            raise KeyError(f"image_id={image_id} not in {self.split} split")
        return self[idx]

    def get_image_by_filename(self, filename: str) -> Dict:
        """Retrieve a specific item by filename."""
        idx = self.get_index_by_filename(filename)
        if idx is None:
            raise KeyError(f"filename={filename} not in {self.split} split")
        return self[idx]

    # ── Tile Analysis ────────────────────────────────────────────

    def compute_tile_entropy(
        self,
        index: int,
        tile_size: int = 768,
        num_bins: int = 64,
    ) -> Dict[str, float]:
        """Compute Shannon entropy for tiles of the image.

        Args:
            index: Dataset index.
            tile_size: Side length of analysis tiles.
            num_bins: Histogram bins for entropy calculation.

        Returns:
            Dict with:
                mean_entropy: Average entropy across tiles.
                std_entropy: Standard deviation of tile entropies.
                max_entropy, min_entropy: Extrema.
                num_tiles: Total tiles analyzed.
        """
        from scipy.stats import entropy as scipy_entropy

        image = self._load_image(index)
        H, W = image.shape[:2]
        entropies = []

        for y in range(0, H - tile_size // 2, tile_size // 2):
            for x in range(0, W - tile_size // 2, tile_size // 2):
                x2 = min(x + tile_size, W)
                y2 = min(y + tile_size, H)
                tile = image[y:y2, x:x2]
                if tile.size == 0:
                    continue
                # Per-channel entropy averaged
                tile_ent = 0.0
                for c in range(3):
                    hist, _ = np.histogram(
                        tile[:, :, c].ravel(), bins=num_bins, range=(0, 255)
                    )
                    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
                    tile_ent += scipy_entropy(hist + 1e-8) / 3.0
                entropies.append(tile_ent)

        if not entropies:
            return {"mean_entropy": 0.0, "std_entropy": 0.0,
                    "max_entropy": 0.0, "min_entropy": 0.0, "num_tiles": 0}

        return {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "max_entropy": float(np.max(entropies)),
            "min_entropy": float(np.min(entropies)),
            "num_tiles": len(entropies),
        }

    def compute_sparse_region_ratio(
        self,
        index: int,
        tile_size: int = 384,
        bg_threshold: float = 0.95,
    ) -> Dict[str, float]:
        """Compute the fraction of sparsely-populated tiles.

        A tile is "sparse" if the instance mask coverage is below
        (1 - bg_threshold).

        Args:
            index: Dataset index.
            tile_size: Analysis tile side length.
            bg_threshold: Ratio threshold for background classification.

        Returns:
            Dict with: sparse_ratio, dense_ratio, num_tiles_total,
                       num_tiles_sparse, num_tiles_dense.
        """
        info = self.get_image_info(self._image_ids[index])
        H, W = info.get("height", 1024), info.get("width", 1024)

        # Build a union mask
        masks, _ = self.load_instance_masks(index, H, W)
        if len(masks) > 0:
            union_mask = (masks.max(axis=0) > 0).astype(np.float32)
        else:
            union_mask = np.zeros((H, W), dtype=np.float32)

        sparse_count = 0
        dense_count = 0

        for y in range(0, H - tile_size // 2, tile_size // 2):
            for x in range(0, W - tile_size // 2, tile_size // 2):
                x2 = min(x + tile_size, W)
                y2 = min(y + tile_size, H)
                tile_mask = union_mask[y:y2, x:x2]
                coverage = tile_mask.mean()
                if coverage < (1.0 - bg_threshold):
                    sparse_count += 1
                else:
                    dense_count += 1

        total = sparse_count + dense_count
        if total == 0:
            return {"sparse_ratio": 0.0, "dense_ratio": 1.0,
                    "num_tiles_total": 0, "num_tiles_sparse": 0, "num_tiles_dense": 0}

        return {
            "sparse_ratio": sparse_count / total,
            "dense_ratio": dense_count / total,
            "num_tiles_total": total,
            "num_tiles_sparse": sparse_count,
            "num_tiles_dense": dense_count,
        }

    def compute_tile_object_stats(
        self,
        index: int,
        tile_sizes: List[int] = None,
    ) -> Dict[str, float]:
        """Compute per-tile object density statistics for multiple sizes.

        Args:
            index: Dataset index.
            tile_sizes: Tile sizes to analyze (default: [384, 768, 1536, 3072]).

        Returns:
            Dict with per-size density averages.
        """
        if tile_sizes is None:
            tile_sizes = [384, 768, 1536, 3072]

        info = self.get_image_info(self._image_ids[index])
        H, W = info.get("height", 1024), info.get("width", 1024)
        anns = self._load_annotations(index)

        stats = {}
        for ts in tile_sizes:
            if ts >= min(H, W):
                stats[str(ts)] = {"avg_objects_per_tile": 0.0, "num_tiles": 0}
                continue

            counts = []
            for y in range(0, H - ts + 1, ts):
                for x in range(0, W - ts + 1, ts):
                    n_objects = 0
                    for ann in anns:
                        bbox = ann.get("bbox", [0, 0, 0, 0])
                        bx, by, bw, bh = bbox
                        if (bx + bw > x and bx < x + ts and
                                by + bh > y and by < y + ts):
                            n_objects += 1
                    counts.append(n_objects)

            stats[str(ts)] = {
                "avg_objects_per_tile": float(np.mean(counts)) if counts else 0.0,
                "num_tiles": len(counts),
            }

        return stats

    # ── Tile Region Annotation ───────────────────────────────────

    def get_tile_annotations(
        self,
        index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> List[Dict[str, Any]]:
        """Get annotations that overlap a tile region.

        Args:
            index: Dataset index.
            x1, y1, x2, y2: Tile bounding box in pixel coordinates.

        Returns:
            List of COCO annotations with adjusted spatial coordinates.
        """
        anns = self._load_annotations(index)
        results = []
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            bx, by, bw, bh = bbox
            # Check overlap
            if (bx + bw > x1 and bx < x2 and by + bh > y1 and by < y2):
                # Clone and adjust coordinates to be tile-relative
                adjusted = dict(ann)
                adjusted["bbox"] = [
                    max(0, bx - x1),
                    max(0, by - y1),
                    min(bw, x2 - max(bx, x1)),
                    min(bh, y2 - max(by, y1)),
                ]
                results.append(adjusted)
        return results


# ── iSAID Dataset ────────────────────────────────────────────────────

ISAID_CATEGORIES = [
    {"id": 1, "name": "small_vehicle"},
    {"id": 2, "name": "large_vehicle"},
    {"id": 3, "name": "plane"},
    {"id": 4, "name": "storage_tank"},
    {"id": 5, "name": "ship"},
    {"id": 6, "name": "harbor"},
    {"id": 7, "name": "ground_track_field"},
    {"id": 8, "name": "soccer_ball_field"},
    {"id": 9, "name": "tennis_court"},
    {"id": 10, "name": "swimming_pool"},
    {"id": 11, "name": "road"},
    {"id": 12, "name": "basketball_court"},
    {"id": 13, "name": "bridge"},
    {"id": 14, "name": "helicopter"},
    {"id": 15, "name": "roundabout"},
]


@DATASET.register()
class ISAIDDataset(CocoDataset):
    """iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images.

    15 object categories. Resolution: 800×800 to 4000×4000 pixels.
    High-resolution remote sensing — ideal for AdaTile's adaptive tiling.

    Reference: Waqas Zamir et al., CVPR 2019 Workshop.
    """

    COCO_CATEGORIES = ISAID_CATEGORIES

    def __init__(
        self,
        root_dir: str = "datasets/iSAID",
        split: str = "train",
        transforms: Optional[callable] = None,
        tile_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            image_dir="images",
            anno_file=f"{root_dir}/annotations/{split}/instances_{split}.json",
            split=split,
            transforms=transforms,
            **kwargs,
        )
        # Default tile sizes tuned for aerial imagery
        self.preferred_tile_sizes = tile_sizes or [384, 768, 1536, 3072]

    def get_tile_scale_recommendations(
        self, index: int
    ) -> Dict[str, Any]:
        """Recommend tile sizes based on object scale distribution.

        Analyzes the size distribution of instances to suggest
        which tile sizes are most appropriate.

        Returns:
            Dict with per-tile-size coverage score.
        """
        anns = self._load_annotations(index)
        areas = []
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            areas.append(bbox[2] * bbox[3])

        if not areas:
            return {"tiny": 1.0, "small": 0.0, "medium": 0.0, "large": 0.0}

        areas = np.array(areas)
        info = self.get_image_info(self._image_ids[index])
        img_area = info.get("width", 1024) * info.get("height", 1024)

        # Relative object size thresholds
        tiny = np.sum(areas < img_area * 1e-4) / len(areas)
        small = np.sum((areas >= img_area * 1e-4) & (areas < img_area * 1e-2)) / len(areas)
        medium = np.sum((areas >= img_area * 1e-2) & (areas < img_area * 5e-2)) / len(areas)
        large = np.sum(areas >= img_area * 5e-2) / len(areas)

        return {
            "tiny_ratio": float(tiny),
            "small_ratio": float(small),
            "medium_ratio": float(medium),
            "large_ratio": float(large),
            # Recommended tile size per category
            "recommended_tile_for_tiny": 384,
            "recommended_tile_for_small": 768,
            "recommended_tile_for_medium": 1536,
            "recommended_tile_for_large": 3072,
        }


# ── LoveDA Dataset ───────────────────────────────────────────────────

LOVEDA_CATEGORIES = [
    {"id": 1, "name": "background"},
    {"id": 2, "name": "building"},
    {"id": 3, "name": "road"},
    {"id": 4, "name": "water"},
    {"id": 5, "name": "barren"},
    {"id": 6, "name": "forest"},
    {"id": 7, "name": "agriculture"},
]


@DATASET.register()
class LoveDADataset(CocoDataset):
    """LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive
    Semantic Segmentation. 7 land-cover categories. 1024×1024 pixels.

    Reference: Wang et al., NeurIPS 2021 Datasets and Benchmarks Track.
    """

    COCO_CATEGORIES = LOVEDA_CATEGORIES

    def __init__(
        self,
        root_dir: str = "datasets/LoveDA",
        split: str = "train",
        transforms: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            image_dir="images",
            anno_file=f"{root_dir}/annotations/{split}/instances_{split}.json",
            split=split,
            transforms=transforms,
            **kwargs,
        )

    def load_semantic_mask(
        self, index: int, height: Optional[int] = None, width: Optional[int] = None
    ) -> np.ndarray:
        """LoveDA is primarily a semantic (land-cover) dataset.

        Override to support both instance and semantic mask formats.
        """
        # Try to load from semantic mask files first
        info = self.get_image_info(self._image_ids[index])
        H = height or info.get("height", 1024)
        W = width or info.get("width", 1024)

        # Check for pre-computed semantic mask
        fname = Path(info.get("file_name", ""))
        mask_path = self.root_dir / "masks" / self.split / f"{fname.stem}.png"
        if mask_path.exists():
            with Image.open(mask_path) as im:
                return np.array(im, dtype=np.int32)

        # Fallback to instance-based semantic mask
        return super().load_semantic_mask(index, H, W)
