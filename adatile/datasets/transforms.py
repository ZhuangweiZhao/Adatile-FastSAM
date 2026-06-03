"""Augmentation pipeline for instance segmentation.

Built on albumentations. Supports:
    - Instance-aware transforms (mask-preserving)
    - Multi-scale training (dynamic resize)
    - COCO-format input/output conventions
    - Square padding for tile-friendly batching
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.core.composition import Compose


# ── Pre-registered Pipelines ──────────────────────────────────────


def build_train_pipeline(
    image_size: Tuple[int, int] = (1024, 1024),
    hflip_prob: float = 0.5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    scale_limit: float = 0.25,
    rotate_limit: int = 10,
) -> Compose:
    """Build a standard training augmentation pipeline.

    Order matters:
        1. Spatial transforms (flip, rotate, scale)
        2. Pixel-level transforms (brightness, contrast, noise)
        3. Normalization (optional, often done in model)

    All transforms preserve instance masks.
    """
    return Compose(
        [
            # Spatial
            A.HorizontalFlip(p=hflip_prob),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=0.5,
            ),
            # Resolution
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.Resize(
                height=image_size[0],
                width=image_size[1],
                p=1.0,
            ),
            # Pixel
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=64,
                max_width=64,
                fill_value=0,
                p=0.2,
            ),
        ],
        # bbox_params handled at dataset level
    )


def build_val_pipeline(
    image_size: Tuple[int, int] = (1024, 1024),
) -> Compose:
    """Build a validation pipeline (resize only, no augmentation)."""
    return Compose([
        A.Resize(
            height=image_size[0],
            width=image_size[1],
            p=1.0,
        ),
    ])


def build_tile_pipeline(
    tile_size: int = 768,
    padding: bool = True,
) -> Compose:
    """Build pipeline for tile preprocessing.

    If padding=True, smaller tiles are padded to tile_size×tile_size.
    """
    transforms = []
    if padding:
        transforms.append(
            A.PadIfNeeded(
                min_height=tile_size,
                min_width=tile_size,
                border_mode=0,
                p=1.0,
            )
        )
    transforms.append(A.Resize(tile_size, tile_size, p=1.0))
    return Compose(transforms)


# ── Transform Wrapper ─────────────────────────────────────────────


class InstanceSegTransform:
    """Wraps an albumentations Compose with COCO-format bbox handling.

    Converts between:
        - Input: {"image": HWC-uint8, "annotations": List[COCO-ann]}
        - Output: same format with transforms applied to both
                  image and segmentation masks + bboxes.

    Segmentation is converted to masks before transform and extracted back.
    """

    def __init__(
        self,
        transform: Compose,
        mask_key: str = "masks",
        bbox_format: str = "coco",
    ):
        self.transform = transform
        self.mask_key = mask_key
        self.bbox_format = bbox_format  # "coco" → [x,y,w,h] (albumentations format)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]  # (H, W, 3) uint8
        annotations = sample.get("annotations", [])

        H, W = image.shape[:2]

        # Convert COCO bboxes from [x,y,w,h] to [x_min,y_min,x_max,y_max]
        bboxes = []
        masks = []

        for ann in annotations:
            seg = ann.get("segmentation", [])
            bbox = ann.get("bbox")
            cat_id = ann.get("category_id", 0)

            # Build binary mask for this instance
            if isinstance(seg, list) and len(seg) > 0:
                from pycocotools.mask import frPyObjects, decode
                rles = frPyObjects(seg, H, W)
                if isinstance(rles, dict):
                    rles = [rles]
                m = decode(rles)
                if m.ndim == 3:
                    m = m.max(axis=2)
                masks.append(m.astype(np.uint8))
            elif bbox is not None:
                # Fallback: rasterize bbox
                x, y, bw, bh = bbox
                m = np.zeros((H, W), dtype=np.uint8)
                x1 = int(max(0, x))
                y1 = int(max(0, y))
                x2 = int(min(W, x + bw))
                y2 = int(min(H, y + bh))
                m[y1:y2, x1:x2] = 1
                masks.append(m)

            if bbox is not None:
                x, y, bw, bh = bbox
                bboxes.append([
                    x, y, min(x + bw, W), min(y + bh, H)
                ])

        # Stack masks for albumentations
        if masks:
            masks_array = np.stack(masks, axis=-1) if len(masks) > 1 else masks[0]
        else:
            masks_array = np.zeros((H, W, 1), dtype=np.uint8)

        # Apply transform
        transformed = self.transform(
            image=image,
            mask=masks_array if masks_array.ndim == 2 else masks_array,
            bboxes=bboxes if bboxes else None,
        )

        sample["image"] = transformed["image"]

        # Reconstruct annotations from transformed masks
        new_annotations = []
        if masks:
            new_masks = transformed["mask"]
            if new_masks.ndim == 2:
                new_masks = new_masks[:, :, np.newaxis]

            for i in range(new_masks.shape[-1] if new_masks.ndim == 3 else 1):
                m = new_masks[:, :, i] if new_masks.ndim == 3 else new_masks
                if m.max() == 0:
                    continue
                # Find bbox from mask
                rows = np.any(m, axis=1)
                cols = np.any(m, axis=0)
                if not (rows.any() and cols.any()):
                    continue
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                new_annotations.append({
                    "segmentation": [],  # Could convert mask back to polygon
                    "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
                    "category_id": annotations[i]["category_id"] if i < len(annotations) else 0,
                    "area": int((xmax - xmin) * (ymax - ymin)),
                    "iscrowd": 0,
                })

        sample["annotations"] = new_annotations
        return sample


# ── Registry Integration ───────────────────────────────────────────

from adatile.registry import TRANSFORM


@TRANSFORM.register()
def get_train_pipeline(
    image_size: Tuple[int, int] = (1024, 1024),
    **kwargs,
) -> InstanceSegTransform:
    """Registered training transform."""
    return InstanceSegTransform(build_train_pipeline(image_size=image_size, **kwargs))


@TRANSFORM.register()
def get_val_pipeline(
    image_size: Tuple[int, int] = (1024, 1024),
) -> InstanceSegTransform:
    """Registered validation transform."""
    return InstanceSegTransform(build_val_pipeline(image_size=image_size))
