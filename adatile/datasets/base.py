"""Base dataset class for instance segmentation.

Provides:
    - COCO JSON loading with lazy indexing
    - Polygon / RLE → binary mask conversion
    - Object density map generation (from GT instances → [H/16, W/16] density)
    - Tile entropy and sparse-region statistics
    - Class distribution analysis
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract dataset for instance segmentation.

    All dataset backends must implement:
        - _load_image(index) → np.ndarray                    (H, W, 3) uint8 RGB
        - _load_annotations(index) → List[Dict]              COCO-style annotations
        - _load_category_mapping() → Dict[int, str]          category_id → name
    """

    def __init__(
        self,
        root_dir: str,
        image_dir: str = "images",
        anno_dir: str = "annotations",
        split: str = "train",
        transforms: Optional[callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir / split
        self.anno_dir = self.root_dir / anno_dir        # no split — filename already contains it
        self.split = split
        self.transforms = transforms
        self._image_ids: List[int] = []
        self._category_mapping: Dict[int, str] = {}
        self._image_info: Dict[int, Dict] = {}
        self._image_to_anns: Dict[int, List[Dict]] = {}

    # ── Abstract Methods ─────────────────────────────────────────

    @abstractmethod
    def _load_image(self, index: int) -> np.ndarray:
        """Load image as (H, W, 3) uint8 RGB numpy array."""

    @abstractmethod
    def _load_annotations(self, index: int) -> List[Dict[str, Any]]:
        """Load COCO-format annotations for the given index."""

    @abstractmethod
    def _load_category_mapping(self) -> Dict[int, str]:
        """Return {category_id: category_name}."""

    # ── Properties ───────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self._category_mapping)

    @property
    def image_ids(self) -> List[int]:
        return self._image_ids

    def get_image_info(self, image_id: int) -> Dict:
        return self._image_info.get(image_id, {})

    def get_category_name(self, category_id: int) -> str:
        return self._category_mapping.get(category_id, "unknown")

    # ── Indexing ─────────────────────────────────────────────────

    def __len__(self) -> int:
        # Trigger lazy loading of COCO data if not already loaded.
        # CocoDataset calls _build_indices() inside coco_data property.
        if hasattr(self, 'coco_data'):
            _ = self.coco_data
        return len(self._image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image = self._load_image(index)
        annotations = self._load_annotations(index)
        image_id = self._image_ids[index] if self._image_ids else index

        sample = {
            "image": image,
            "annotations": annotations,
            "image_id": image_id,
            "image_info": self.get_image_info(image_id),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    # ── Mask Conversion ──────────────────────────────────────────

    @staticmethod
    def polygon_to_mask(
        polygons: List[List[float]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Convert COCO polygon(s) to a binary mask.

        Args:
            polygons: List of flattened polygon coordinates,
                      e.g. [[x0,y0, x1,y1, ...]].
                      Multiple polygons are UNIONed.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Binary mask (H, W) as uint8 numpy array.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        try:
            from pycocotools.mask import frPyObjects, decode

            rles = frPyObjects(
                polygons, height, width
            )
            if isinstance(rles, dict):
                rles = [rles]
            decoded = decode(rles)
            if decoded.ndim == 3:
                mask = decoded.max(axis=2).astype(np.uint8)
            else:
                mask = decoded.astype(np.uint8)
        except ImportError:
            # Fallback: rasterize polygons manually
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                import cv2
                cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def rle_to_mask(rle: Dict[str, Any], height: int, width: int) -> np.ndarray:
        """Decode COCO RLE to binary mask."""
        try:
            from pycocotools.mask import decode
            return decode(rle).astype(np.uint8)
        except ImportError:
            raise ImportError(
                "pycocotools required for RLE decoding. "
                "Install: pip install pycocotools"
            )

    @classmethod
    def ann_to_mask(
        cls,
        ann: Dict[str, Any],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Convert a single COCO annotation to a binary mask.

        Handles both polygon and RLE segmentation formats.

        Args:
            ann: COCO annotation dict with "segmentation" and "iscrowd".
            height, width: Image dimensions.

        Returns:
            Binary mask (H, W) uint8.
        """
        seg = ann.get("segmentation", [])
        if not seg:
            return np.zeros((height, width), dtype=np.uint8)

        if isinstance(seg, dict):
            # RLE format
            return cls.rle_to_mask(seg, height, width)
        elif isinstance(seg, list):
            # Polygon format (list of lists of floats)
            return cls.polygon_to_mask(seg, height, width)
        return np.zeros((height, width), dtype=np.uint8)

    # ── Mask Loading ─────────────────────────────────────────────

    def load_instance_masks(
        self, index: int, height: Optional[int] = None, width: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Load all instance masks for an image.

        Args:
            index: Dataset index.
            height, width: Override image dimensions (auto-detected if None).

        Returns:
            masks:  (N_inst, H, W) uint8 binary masks.
            cat_ids: (N_inst,) category IDs.
        """
        info = self.get_image_info(self._image_ids[index])
        H = height or info.get("height", 1024)
        W = width or info.get("width", 1024)
        anns = self._load_annotations(index)

        masks = []
        cat_ids = []
        for ann in anns:
            m = self.ann_to_mask(ann, H, W)
            masks.append(m)
            cat_ids.append(ann.get("category_id", 0))

        if masks:
            return np.stack(masks, axis=0), np.array(cat_ids, dtype=np.int64)
        return np.empty((0, H, W), dtype=np.uint8), np.array([], dtype=np.int64)

    def load_semantic_mask(
        self, index: int, height: Optional[int] = None, width: Optional[int] = None
    ) -> np.ndarray:
        """Load a semantic mask where pixel value = class_id.

        Args:
            index: Dataset index.

        Returns:
            (H, W) int32 semantic mask.
        """
        masks, cat_ids = self.load_instance_masks(index, height, width)
        if len(masks) == 0:
            return np.zeros((1024, 1024), dtype=np.int32)

        semantic = np.zeros(masks.shape[1:], dtype=np.int32)
        for i, cid in enumerate(cat_ids):
            semantic[masks[i] > 0] = cid
        return semantic

    # ── Density Map Generation ───────────────────────────────────

    def generate_density_map(
        self,
        index: int,
        stride: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an object instance density map for an image.

        Density[y,x] = count of instances whose center falls in this grid cell.
        Normalized to [0, 1].

        Args:
            index: Dataset index.
            stride: Downsample factor (default 16 → feature map scale).
            height, width: Override image size.

        Returns:
            (H//stride, W//stride) float32 array in [0, 1].
        """
        info = self.get_image_info(self._image_ids[index])
        H = height or info.get("height", 1024)
        W = width or info.get("width", 1024)
        anns = self._load_annotations(index)

        H_ds = max(H // stride, 1)
        W_ds = max(W // stride, 1)
        density = np.zeros((H_ds, W_ds), dtype=np.float32)

        for ann in anns:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            x, y, bw, bh = bbox
            cx = x + bw / 2
            cy = y + bh / 2
            gx = int(cx / stride)
            gy = int(cy / stride)
            gx = max(0, min(gx, W_ds - 1))
            gy = max(0, min(gy, H_ds - 1))
            density[gy, gx] += 1.0

        # Normalize to [0, 1] across the image
        dmax = density.max()
        if dmax > 0:
            density /= dmax

        return density

    def generate_density_map_from_masks(
        self,
        masks: np.ndarray,
        stride: int = 16,
    ) -> np.ndarray:
        """Generate a density map from pre-loaded instance masks.

        Args:
            masks: (N, H, W) binary instance masks.

        Returns:
            (H//stride, W//stride) float32 density in [0, 1].
        """
        if len(masks) == 0:
            return np.zeros((1, 1), dtype=np.float32)

        H, W = masks.shape[1:]
        H_ds = max(H // stride, 1)
        W_ds = max(W // stride, 1)

        # Instance presence per downsampled cell
        density = np.zeros((H_ds, W_ds), dtype=np.float32)
        for i in range(len(masks)):
            m = masks[i]
            ds = m.reshape(H_ds, stride, W_ds, stride).max(axis=(1, 3))
            density += ds.astype(np.float32)

        dmax = density.max()
        if dmax > 0:
            density /= dmax
        return density

    def save_density_map(self, index: int, save_dir: str, stride: int = 16) -> str:
        """Generate and save density map to disk.

        Returns:
            Path to the saved .npy file.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_id = self._image_ids[index]
        density = self.generate_density_map(index, stride)
        path = save_dir / f"{image_id}_density.npy"
        np.save(path, density)
        return str(path)

    # ── Statistics ───────────────────────────────────────────────

    def get_instance_count(self, index: int) -> int:
        """Number of GT instances in an image."""
        return len(self._load_annotations(index))

    def get_image_area(self, index: int) -> int:
        """Image area (width × height)."""
        info = self.get_image_info(self._image_ids[index])
        return info.get("width", 1024) * info.get("height", 1024)

    def get_instance_density(self, index: int) -> float:
        """Instance density: count / area."""
        area = self.get_image_area(index)
        if area == 0:
            return 0.0
        return self.get_instance_count(index) / area

    def compute_class_distribution(self) -> Dict[int, int]:
        """Aggregate instance count per category across the dataset.

        Returns:
            {category_id: total_instance_count}.
        """
        dist = defaultdict(int)
        for idx in range(len(self)):
            for ann in self._load_annotations(idx):
                dist[ann.get("category_id", -1)] += 1
        return dict(dist)

    def compute_class_distribution_normalized(self) -> Dict[int, float]:
        """Return class distribution as ratios [0, 1]."""
        dist = self.compute_class_distribution()
        total = sum(dist.values())
        if total == 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def compute_image_stats(self) -> Dict[str, float]:
        """Aggregate statistics across all images.

        Returns:
            Dict with:
                avg_resolution_w, avg_resolution_h
                avg_instances, max_instances, min_instances
                avg_density, std_density
                total_images
        """
        areas = []
        counts = []
        densities = []
        for idx in range(len(self)):
            area = self.get_image_area(idx)
            count = self.get_instance_count(idx)
            areas.append(area)
            counts.append(count)
            densities.append(count / max(area, 1))

        return {
            "total_images": len(self),
            "avg_resolution_w": float(np.mean([self.get_image_info(iid).get("width", 0) for iid in self._image_ids])),
            "avg_resolution_h": float(np.mean([self.get_image_info(iid).get("height", 0) for iid in self._image_ids])),
            "avg_instances": float(np.mean(counts)) if counts else 0.0,
            "max_instances": float(np.max(counts)) if counts else 0.0,
            "min_instances": float(np.min(counts)) if counts else 0.0,
            "avg_density": float(np.mean(densities)) if densities else 0.0,
            "std_density": float(np.std(densities)) if densities else 0.0,
        }


# ── Few-Shot Split ──────────────────────────────────────────────────


class FewShotSplit:
    """Fixed few-shot split manager.

    Split JSON format:
    {
      "novel_classes": [3, 7, 11],
      "support_images": {"3": ["0001.png"], "7": ["0201.png"]},
      "query_images": ["0100.png", "0101.png", ...],
      "base_classes": [1, 2, 4, 5, 6, 8, 9, 10]
    }
    """

    def __init__(self, split_path: str):
        self.split_path = Path(split_path)
        if not self.split_path.exists():
            raise FileNotFoundError(f"Few-shot split not found: {split_path}")
        with open(self.split_path, "r") as f:
            self._data = json.load(f)

    @property
    def novel_classes(self) -> List[int]:
        """Classes used for few-shot evaluation."""
        return self._data["novel_classes"]

    @property
    def base_classes(self) -> List[int]:
        """Classes available during base training (optional)."""
        return self._data.get("base_classes", [])

    @property
    def support_images(self) -> Dict[str, List[str]]:
        """{class_id_str: [filename, ...]} for support set."""
        return self._data.get("support_images", {})

    @property
    def query_images(self) -> List[str]:
        """Query image filenames."""
        return self._data["query_images"]

    def get_support_for_class(self, class_id: int) -> List[str]:
        """Get support image filenames for a given class."""
        return self.support_images.get(str(class_id), self.support_images.get(class_id, []))

    def get_all_support_filenames(self) -> List[str]:
        """All unique support image filenames."""
        seen = set()
        result = []
        for fnames in self.support_images.values():
            for f in fnames:
                if f not in seen:
                    seen.add(f)
                    result.append(f)
        return result

    def to_dict(self) -> Dict:
        return dict(self._data)

    @classmethod
    def create_split(
        cls,
        novel_classes: List[int],
        support_images: Dict[str, List[str]],
        query_images: List[str],
        base_classes: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> FewShotSplit:
        """Create and optionally persist a new split."""
        data = {
            "novel_classes": novel_classes,
            "support_images": support_images,
            "query_images": query_images,
        }
        if base_classes is not None:
            data["base_classes"] = base_classes

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

        return cls(str(save_path) if save_path else "")
