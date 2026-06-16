"""
BSSegDataset v1.0 — Boundary-aware Segmentation Dataset.

Simple image-mask pair dataset for innovation verification.
Loads PNG images + binary masks, supports albumentations augmentations.

Metadata-only caching: stores paths and shapes, loads images on __getitem__.
"""

import os
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict
import torch


class BSSegDataset(Dataset):
    """Simple binary segmentation dataset from image-mask pairs.

    Directory layout:
        data_dir/
        ├── TrainDataset/images/*.png
        ├── TrainDataset/masks/*.png
        ├── ValDataset/images/*.png
        ├── ValDataset/masks/*.png
        ├── TestDataset/images/*.png
        └── TestDataset/masks/*.png

    Args:
        data_dir: Root data directory.
        split: "train", "val", or "test".
        image_size: (H, W) target size, None = keep original.
        max_samples: Max samples (-1 = all, sorted by name).
        transform: Optional albumentations Compose.
        file_filter: Optional list of filenames (without ext) to include.
        num_classes: 1 = binary (>128 → 1), >1 = class labels 0..K as-is.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Optional[Tuple[int, int]] = None,
        max_samples: int = -1,
        transform=None,
        file_filter: Optional[List[str]] = None,
        num_classes: int = 1,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.transform = transform
        self.num_classes = num_classes

        self.samples = self._load_samples()

        if file_filter is not None:
            filter_set = {os.path.splitext(f)[0] for f in file_filter}
            self.samples = [s for s in self.samples if s["name"] in filter_set]
        elif self.max_samples is not None and self.max_samples > 0:
            self.samples = self.samples[:self.max_samples]

    def _load_samples(self) -> List[Dict[str, str]]:
        split_map = {"train": "TrainDataset", "val": "ValDataset", "test": "TestDataset"}
        split_dir = split_map.get(self.split, "TrainDataset")
        images_dir = os.path.join(self.data_dir, split_dir, "images")
        masks_dir = os.path.join(self.data_dir, split_dir, "masks")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images dir not found: {images_dir}")

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        samples = []
        for fname in sorted(os.listdir(images_dir)):
            if os.path.splitext(fname.lower())[1] not in exts:
                continue
            img_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, fname)
            if os.path.exists(mask_path):
                samples.append({
                    "image": img_path,
                    "mask": mask_path,
                    "name": os.path.splitext(fname)[0],
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        if cv2 is not None:
            image = cv2.imread(sample["image"])
            if image is None:
                raise FileNotFoundError(f"Failed to load: {sample['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(sample["image"]).convert("RGB"))

        # Load mask (force 2D [H, W])
        if cv2 is not None:
            mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.array(Image.open(sample["mask"]).convert("L"))
        if mask is None:
            raise FileNotFoundError(f"Failed to load: {sample['mask']}")
        # Handle palette/indexed PNGs that cv2 might load as [H,W,1]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)

        if self.num_classes == 1:
            # Binary: threshold at 128
            mask = (mask > 128).astype(np.float32)
        else:
            # Multi-class: keep class labels as-is (0=bg, 1..K=classes)
            # Mask values are already class indices (0-14 for iSAID)
            mask = mask.astype(np.float32)
            mask = np.clip(mask, 0, self.num_classes - 1)

        # Resize
        if self.image_size is not None:
            H, W = self.image_size
            if cv2 is not None:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                image = np.array(Image.fromarray(image).resize((W, H), Image.BILINEAR))
                mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))

        # Augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # To tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

        return {
            "images": img_tensor,
            "masks": mask_tensor,
            "name": sample["name"],
        }
