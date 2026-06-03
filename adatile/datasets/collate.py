"""Batch collation functions for AdaTile-FastSAM.

Supports:
    - Standard COCO batch collation (images + annotations as lists)
    - Tile-aware collation (tile tensors + tile metadata per image)
    - Few-shot episodic collation (support + query separated)
    - Variable-size image batching with padding
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from adatile.core import TileInfo


def coco_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Standard COCO-format collate function.

    Stacks images to [B, 3, H, W] (must be same size).
    Annotations and metadata remain as lists.

    Args:
        batch: List of dataset __getitem__ results.

    Returns:
        Batched dict with keys: images, annotations, image_ids, image_infos.
    """
    images = torch.stack([
        _to_tensor(item["image"]).float()
        for item in batch
    ])

    return {
        "images": images,
        "annotations": [item["annotations"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_infos": [item.get("image_info", {}) for item in batch],
    }


def coco_collate_with_masks(
    batch: List[Dict[str, Any]],
    mask_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Collate COCO batch including instance masks.

    Includes pre-computed masks for direct loss computation.

    Args:
        batch: List of samples.
        mask_size: Optional (H, W) to resize all masks to.

    Returns:
        Batched dict with additional "gt_masks" and "gt_classes" fields.
    """
    result = coco_collate(batch)

    all_masks = []
    all_classes = []

    for item in batch:
        image = item["image"]
        H, W = image.shape[:2]
        anns = item["annotations"]

        img_masks = []
        img_classes = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            from pycocotools.mask import frPyObjects, decode
            rles = frPyObjects(seg, H, W)
            if isinstance(rles, dict):
                rles = [rles]
            m = decode(rles)
            if m.ndim == 3:
                m = m.max(axis=2)
            m = torch.from_numpy(m).float()
            if mask_size is not None:
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=mask_size,
                    mode="nearest",
                ).squeeze()
            img_masks.append(m)
            img_classes.append(ann.get("category_id", 0))

        if img_masks:
            all_masks.append(torch.stack(img_masks))
        else:
            all_masks.append(torch.empty(0, H, W, dtype=torch.float32))
        all_classes.append(torch.tensor(img_classes, dtype=torch.long))

    result["gt_masks"] = all_masks
    result["gt_classes"] = all_classes
    return result


def tile_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tile-aware collation: full images + associated tile metadata.

    Used with DynamicTileDataLoader.
    Images are kept at original size (may vary per sample).
    TileInfo lists are preserved for tile retrieval.

    Args:
        batch: List of samples, each potentially with tile_infos.

    Returns:
        Batched dict with: images (list of tensors if variable size),
        annotations, tile_infos (list of List[TileInfo]), image_ids.
    """
    images = [_to_tensor(item["image"]).float() for item in batch]

    return {
        "images": images,
        "annotations": [item["annotations"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_infos": [item.get("image_info", {}) for item in batch],
        "tile_infos": [item.get("tile_infos", []) for item in batch],
    }


def fewshot_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Few-shot episodic collation.

    Expects batch items with "is_support" flag.
    Separates support and query, builds the episode.

    Returns:
        Dict with:
            support_images: [S, 3, H, W]
            support_masks: [S, H, W]
            support_classes: [S]
            query_images: [Q, 3, H, W]
            query_annotations: List
            class_ids: List[int] (unique classes in this episode)
    """
    support_items = [item for item in batch if item.get("is_support", False)]
    query_items = [item for item in batch if not item.get("is_support", False)]

    def stack_images(items):
        if not items:
            return torch.empty(0)
        return torch.stack([_to_tensor(it["image"]).float() for it in items])

    support_images = stack_images(support_items)
    support_masks_list = []
    support_classes_list = []

    for it in support_items:
        # Support items should include a pre-loaded mask
        mask = it.get("mask")
        if mask is not None:
            support_masks_list.append(torch.from_numpy(mask).float())
        support_classes_list.append(it.get("category_id", 0))

    if support_masks_list:
        support_masks = torch.stack(support_masks_list)
    else:
        support_masks = torch.empty(0)

    query_images = stack_images(query_items)

    class_ids = list(set(support_classes_list))
    if not query_items:
        class_ids = list(set(
            it.get("category_id", 0) for it in support_items
        ))

    return {
        "support_images": support_images,
        "support_masks": support_masks,
        "support_classes": torch.tensor(support_classes_list, dtype=torch.long),
        "query_images": query_images,
        "query_annotations": [it.get("annotations", []) for it in query_items],
        "query_image_ids": [it.get("image_id") for it in query_items],
        "class_ids": class_ids,
    }


def pad_collate(
    batch: List[Dict[str, Any]],
    pad_value: float = 0.0,
    size_divisor: int = 32,
) -> Dict[str, Any]:
    """Collate with padding for variable-size images.

    Pads all images to the maximum dimensions in the batch,
    rounded up to the nearest multiple of size_divisor.

    Args:
        batch: List of samples.
        pad_value: Value for padding.
        size_divisor: Round up dimensions to this multiple.

    Returns:
        Batched dict with padded images (all same size).
    """
    images = [_to_tensor(item["image"]).float() for item in batch]
    annotations = [item["annotations"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    # Find max dims
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Round up
    max_h = ((max_h + size_divisor - 1) // size_divisor) * size_divisor
    max_w = ((max_w + size_divisor - 1) // size_divisor) * size_divisor

    padded = []
    for img in images:
        c, h, w = img.shape
        if h < max_h or w < max_w:
            padded_img = torch.full((c, max_h, max_w), pad_value, dtype=img.dtype)
            padded_img[:, :h, :w] = img
            padded.append(padded_img)
        else:
            padded.append(img)

    return {
        "images": torch.stack(padded),
        "annotations": annotations,
        "image_ids": image_ids,
        "image_infos": [item.get("image_info", {}) for item in batch],
        "original_sizes": [(img.shape[1], img.shape[2]) for img in images],
    }


def _to_tensor(image: np.ndarray) -> Tensor:
    """Convert HWC uint8 numpy image to CHW float32 tensor in [0, 1]."""
    if isinstance(image, Tensor):
        return image
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
