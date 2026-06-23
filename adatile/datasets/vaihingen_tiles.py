"""Vaihingen Pre-cut Tile Dataset.

Reads pre-cut 1024x1024 tiles from ISPRS Vaihingen.
Images: TIFF RGB uint8. Masks: PNG class IDs.

Classes (dense semantic, fg_ratio ~60%):
  0: Impervious surfaces / background
  1: Building
  2: Low vegetation
  3: Tree
  4: Car
  5: Clutter (test only)
  6: Clutter variant (test only)

Key: dense semantic => NO empty tiles. Different from iSAID.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from adatile.logging import get_logger

logger = get_logger("vaihingen_tiles")

VAIHINGEN_CLASSES = {0:"impervious",1:"building",2:"low_veg",3:"tree",4:"car",5:"clutter",6:"clutter2"}

class VaihingenTileDataset(Dataset):
    CLASS_NAMES = VAIHINGEN_CLASSES

    def __init__(self, root_dir="data/Vaihingen", split="train", semantic=True):
        self.root = Path(root_dir)
        self.split = split
        self.semantic = semantic
        self._img_dir = self.root / split / "images_1024"
        self._mask_dir = self.root / split / "masks_1024"
        self._tiles = sorted(p.stem for p in self._img_dir.glob("*.tif"))
        if not self._tiles:
            raise FileNotFoundError(f"No TIFF tiles in {self._img_dir}")
        logger.log_info("dataset/vaihingen_init",
            f"VaihingenTile {split}: {len(self)} tiles")

    def __len__(self):
        return len(self._tiles)

    def __getitem__(self, index):
        stem = self._tiles[index]
        try:
            img = Image.open(self._img_dir / f"{stem}.tif")
            img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            mask = Image.open(self._mask_dir / f"{stem}.png")
            mask_np = np.array(mask)
        except (IOError, OSError, ValueError):
            logger.log_info("data/skip_corrupted", f"Skipping: {stem}")
            if not hasattr(self, "_skip_count"):
                self._skip_count = 0
            self._skip_count += 1
            if self._skip_count > min(100, len(self._tiles)):
                raise RuntimeError(f"Too many corrupted: {self._skip_count}")
            return self.__getitem__((index + 1) % len(self._tiles))

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        if self.semantic:
            mask_np = np.where(mask_np <= 6, mask_np, 0).astype(np.int64)
            mask_t = torch.from_numpy(mask_np)
        else:
            mask_t = torch.from_numpy((mask_np > 0).astype(np.float32))
        return {"image":img_t,"mask":mask_t,"image_id":stem,"image_size":tuple(img_t.shape[1:])}

    def fg_ratio(self, mask):
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        return float((mask > 0).sum() / max(mask.size, 1))
