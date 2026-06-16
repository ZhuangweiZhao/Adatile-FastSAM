"""
UniversalDataset — auto-detect dataset layout, uniform interface.

Supported layouts (auto-detected by directory structure):

  bsseg:    data_dir/TrainDataset/{images,masks}/*.png    (binary or class labels)
  loveda:   data_dir/Train/Train/{Urban,Rural}/images_png/*.png
                                {Urban,Rural}/masks_png/*.png
  flat:     data_dir/train/{images,masks}/*.png            (generic)
            data_dir/{images,masks}/*.png                  (single dir, no split)

Interface: {"images": [3,H,W] float32 [0,1], "masks": [H,W] int64 class labels, "name": str}
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image


def _read_image(path):
    """Read RGB image → [H,W,3] uint8. Uses PIL for unicode path support."""
    im = Image.open(str(path)).convert("RGB")
    return np.array(im)


def _read_mask(path, num_classes=1):
    """Read mask → [H,W] uint8 class labels.
    Binary (num_classes=1): threshold at 128.
    Multi-class (num_classes>1): keep raw values.
    Uses PIL for unicode path support.
    """
    m = np.array(Image.open(str(path)).convert("L"))
    if m.ndim == 3:
        m = m.squeeze(-1)
    if num_classes == 1:
        m = (m > 128).astype(np.uint8)
    else:
        m = m.astype(np.uint8)
        m = np.clip(m, 0, num_classes - 1)
    return m


def _detect_layout(data_dir: str):
    """Detect dataset layout from directory structure.

    Returns one of: "bsseg", "loveda", "flat"
    """
    root = Path(data_dir)
    if (root / "TrainDataset").is_dir() or (root / "ValDataset").is_dir():
        return "bsseg"
    if (root / "Train").is_dir() and (root / "Train" / "Train").is_dir():
        return "loveda"
    return "flat"


def _find_pairs_bsseg(data_dir, split):
    """BSDSeg layout."""
    split_map = {"train": "TrainDataset", "val": "ValDataset", "test": "TestDataset"}
    sdir = Path(data_dir) / split_map[split]
    img_dir = sdir / "images"
    mask_dir = sdir / "masks"
    if not img_dir.exists():
        return [], set()
    pairs = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            mf = mask_dir / f.name
            if mf.exists():
                pairs.append((str(f), str(mf), f.stem))
    return pairs, set()


def _find_pairs_loveda(data_dir, split):
    """LoveDA layout."""
    split_map = {"train": "Train", "val": "Val", "test": "Test"}
    sdir = Path(data_dir) / split_map[split] / split_map[split]
    if not sdir.exists():
        return [], set()
    pairs = []
    for scene_dir in sorted(sdir.iterdir()):
        if not scene_dir.is_dir():
            continue
        img_dir = scene_dir / "images_png"
        mask_dir = scene_dir / "masks_png"
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.glob("*.png")):
            mf = mask_dir / f.name
            if mf.exists():
                # Unique name: scene_filename
                name = f"{scene_dir.name}_{f.stem}"
                pairs.append((str(f), str(mf), name))
    return pairs, set()


def _find_pairs_flat(data_dir, split):
    """Generic flat layout: {split}/images/, or single images/ masks/ dir."""
    root = Path(data_dir)
    for candidate in [root / split / "images", root / "images"]:
        if candidate.exists():
            img_dir = candidate
            for sibling_name in ["masks", "mask", "labels", "gt"]:
                mask_dir = img_dir.parent / sibling_name
                if mask_dir.exists():
                    break
            else:
                mask_dir = img_dir.parent / "masks"
            if not mask_dir.exists():
                continue
            pairs = []
            for f in sorted(img_dir.glob("*")):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    mf = mask_dir / f.name
                    if mf.exists():
                        pairs.append((str(f), str(mf), f.stem))
            if pairs:
                return pairs, set()
    return [], set()


class UniversalDataset(Dataset):
    """Auto-detecting universal dataset.

    Usage:
        ds = UniversalDataset("datasets/loveda", split="train", num_classes=8)
        ds = UniversalDataset("datasets/isaid_binary", split="train", image_size=(640,640))
        ds = UniversalDataset("dataset", split="train")  # old BSDSeg, auto num_classes=1

    Args:
        data_dir: Root dataset directory.
        split: "train", "val", or "test".
        image_size: (H,W) target, None=keep original.
        num_classes: 1=binary, >1=multi-class. None=auto-detect from 10 sample masks.
        max_samples: Limit number of samples.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Optional[Tuple[int, int]] = None,
        num_classes: Optional[int] = None,
        max_samples: int = -1,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples

        # Detect layout and find pairs
        layout = _detect_layout(data_dir)
        finders = {"bsseg": _find_pairs_bsseg, "loveda": _find_pairs_loveda, "flat": _find_pairs_flat}
        pairs, _ = finders.get(layout, _find_pairs_flat)(data_dir, split)

        if not pairs:
            raise FileNotFoundError(
                f"No image-mask pairs found in {data_dir}/{split}. "
                f"Detected layout: {layout}")

        self.pairs = pairs[:max_samples] if max_samples > 0 else pairs

        # Auto-detect num_classes
        if num_classes is None:
            sample_masks = []
            for _, mp, _ in self.pairs[:min(50, len(self.pairs))]:
                m = _read_mask(mp, num_classes=999)  # 999 = passthrough, no clip
                sample_masks.append(m)
            all_vals = np.unique(np.concatenate([m.ravel() for m in sample_masks]))
            n_unique = len(all_vals)
            max_val = int(all_vals.max())
            if n_unique <= 2 or max_val <= 1:
                num_classes = 1  # binary (0/1 or 0/255 → threshold at 128)
            else:
                num_classes = max_val + 1  # multi-class: 0..max inclusive

        self.num_classes = num_classes
        self.layout = layout

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, name = self.pairs[idx]

        image = _read_image(img_path)
        mask = _read_mask(mask_path, self.num_classes)

        if self.image_size is not None:
            H, W = self.image_size
            if cv2 is not None:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                image = np.array(Image.fromarray(image).resize((W, H), Image.BILINEAR))
                mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))

        return {
            "images": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "masks": torch.from_numpy(mask).long(),
            "name": name,
        }
