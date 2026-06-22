"""
LoveDA Pre-cut Tile Dataset | LoveDA 预切 Tile 数据集.
=========================================================

读取预切 1024x1024 tile，ISPRS LoveDA 土地覆盖语义分割。
Reads pre-cut 1024x1024 tiles, ISPRS LoveDA land-cover semantic segmentation.

目录结构 | Directory structure:
    LoveDA/
    ├── Train/Train/
    │   ├── Rural/
    │   │   ├── images_png/  # RGB PNG
    │   │   └── masks_png/   # 7-class uint8 PNG
    │   └── Urban/
    │       ├── images_png/
    │       └── masks_png/
    ├── Val/Val/{Rural,Urban}/
    └── Test/Test/{Rural,Urban}/
        └── images_png/  (无标注 | no masks)

类别 | Classes (7-class dense semantic):
    0: Background   / 背景
    1: Building     / 建筑
    2: Road         / 道路
    3: Water        / 水体
    4: Barren       / 荒地
    5: Forest       / 森林
    6: Agriculture  / 农田

特点 | Characteristics:
    - 密集语义标注 (类似 Vaihingen, 不同于 iSAID) | Dense semantic (like Vaihingen)
    - SSI < 50 → Spatial Router 不适用 (B-02.5) | Router NOT applicable
    - Rural + Urban 域 | Rural + Urban domains
    - 土地覆盖 → fg_ratio 失效 → Contribution Routing 需要新的重要性定义
    - Land-cover → fg_ratio fails → contribution routing needs new importance def

用法 | Usage:
    >>> ds = LoveDATileDataset("data/LoveDA", split="train")
    >>> sample = ds[0]
    >>> sample["image"].shape  # [3, 1024, 1024]
    >>> sample["mask"].shape   # [1024, 1024]

API: 与 FastISAIDTileDataset 和 VaihingenTileDataset 完全一致
    {"image": [3,H,W] float32, "mask": [H,W] int64, "image_id": str}
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from adatile.logging import get_logger

logger = get_logger("loveda_tiles")

# ═══════════════════════════════════════════════════════════════════
# 类别定义 | Class Definitions
# ═══════════════════════════════════════════════════════════════════

LOVEDA_CLASSES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}

NUM_CLASSES = 7  # incl. background

# ═══════════════════════════════════════════════════════════════════
# 数据集类 | Dataset Class
# ═══════════════════════════════════════════════════════════════════

class LoveDATileDataset(Dataset):
    """
    LoveDA 预切 Tile 数据集 | LoveDA Pre-cut Tile Dataset.

    从 Rural + Urban 子目录加载所有 1024x1024 tile。
    Loads all 1024x1024 tiles from Rural + Urban subdirectories.

    Parameters
    ----------
    root_dir : str
        LoveDA 根目录 (含 Train/Val/Test) | Root directory.
    split : str
        "train", "val", or "test".
    semantic : bool
        True → 返回语义掩码 int64 | return semantic mask int64.
        False → 返回二值掩码 float32 | return binary mask float32.
    domains : tuple[str] | None
        加载的域 ("Rural", "Urban"). None → 全部 | None → all.
    """

    CLASS_NAMES = LOVEDA_CLASSES

    def __init__(
        self,
        root_dir: str = "data/LoveDA",
        split: str = "train",
        semantic: bool = True,
        domains: tuple[str, ...] | None = None,
    ):
        self.root = Path(root_dir)
        self.split = split.lower()
        self.semantic = semantic

        # LoveDA 使用 PascalCase 目录名 | LoveDA uses PascalCase dir names
        split_map = {"train": "Train", "val": "Val", "test": "Test"}
        split_dir = split_map.get(self.split, self.split)
        self._split_dir = self.root / split_dir / split_dir  # e.g. Train/Train

        if not self._split_dir.exists():
            raise FileNotFoundError(
                f"LoveDA split directory not found: {self._split_dir}"
            )

        # 收集域子目录 | Collect domain subdirectories
        if domains is None:
            domains = ("Rural", "Urban")
        self._domains = list(domains)

        # 扫描所有 tile 文件名 | Scan all tile filenames
        self._samples: list[dict] = []  # {domain, stem, img_path, mask_path}
        for domain in self._domains:
            img_dir = self._split_dir / domain / "images_png"
            mask_dir = self._split_dir / domain / "masks_png"

            if not img_dir.exists():
                logger.log_info("loveda/skip",
                               f"Domain {domain} not found for {split}, skipping")
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                sample = {
                    "domain": domain,
                    "stem": img_path.stem,
                    "img_path": str(img_path),
                    "mask_path": str(mask_dir / img_path.name) if mask_dir.exists() else None,
                }
                self._samples.append(sample)

        if not self._samples:
            raise FileNotFoundError(
                f"No PNG tiles found in {self._split_dir}. "
                f"Checked domains: {self._domains}"
            )

        logger.log_info(
            "dataset/loveda_init",
            f"LoveDATile {split}: {len(self)} tiles "
            f"(domains={self._domains}, semantic={semantic})",
        )

    def __len__(self) -> int:
        """返回 tile 总数 | Return total tile count."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        """
        加载单个 tile → {image, mask, image_id} | Load single tile.

        Returns:
            image:    [3, 1024, 1024] float32, 值域 [0,1] | values in [0,1]
            mask:     [1024, 1024] int64 (semantic) or float32 (binary)
            image_id: str  "{domain}/{stem}"
            domain:   str  "Rural" or "Urban"
        """
        sample = self._samples[index]

        # 加载图像 | Load image
        try:
            img = Image.open(sample["img_path"])
            img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        except (IOError, OSError, ValueError):
            logger.log_info("data/skip_corrupted",
                           f"Skipping corrupted image: {sample['img_path']}")
            return self.__getitem__((index + 1) % len(self))

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC → CHW

        # 加载掩码 | Load mask
        if sample["mask_path"] is None:
            # Test split: 无标注 | Test split: no labels
            mask_t = torch.zeros(img_t.shape[1:], dtype=torch.int64)
        else:
            try:
                mask = Image.open(sample["mask_path"])
                mask_np = np.array(mask).astype(np.uint8)  # 0-6
                # 截断到有效类别范围 | Clamp to valid class range
                mask_np = np.where(mask_np < NUM_CLASSES, mask_np, 0)
            except (IOError, OSError, ValueError):
                logger.log_info("data/skip_corrupted",
                               f"Skipping corrupted mask: {sample['mask_path']}")
                return self.__getitem__((index + 1) % len(self))

            if self.semantic:
                # 语义模式: int64 类别 ID | Semantic mode: int64 class IDs
                mask_t = torch.from_numpy(mask_np.astype(np.int64))
            else:
                # 二值模式: >0 = 前景 | Binary mode: >0 = foreground
                mask_t = torch.from_numpy((mask_np > 0).astype(np.float32))

        return {
            "image": img_t,
            "mask": mask_t,
            "image_id": f"{sample['domain']}/{sample['stem']}",
            "domain": sample["domain"],
        }

    def fg_ratio(self, mask: np.ndarray | torch.Tensor) -> float:
        """
        计算前景占比 | Compute foreground pixel fraction.

        LoveDA 是密集标注 → fg_ratio 接近 1.0 → 不适合做重要性代理。
        LoveDA is dense → fg_ratio ≈ 1.0 → poor importance proxy.
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        return float((mask > 0).sum() / max(mask.size, 1))

    @property
    def num_classes(self) -> int:
        """类别总数 (含背景) | Total classes including background."""
        return NUM_CLASSES

    @property
    def class_names(self) -> dict[int, str]:
        """类别 ID → 名称 | Class ID → name."""
        return dict(LOVEDA_CLASSES)


# ═══════════════════════════════════════════════════════════════════
# 快速验证 | Quick Validation
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LoveDATileDataset 快速验证")
    p.add_argument("--root", type=str, default="data/LoveDA")
    p.add_argument("--split", type=str, default="train")
    args = p.parse_args()

    for split in ["train", "val", "test"]:
        try:
            ds = LoveDATileDataset(root_dir=args.root, split=split, semantic=True)
            s = ds[0]
            domains = set(ds._samples[i]["domain"] for i in range(len(ds)))
            unique_classes = torch.unique(s["mask"]).tolist()
            print(f"[{split:5s}] {len(ds):5d} tiles  "
                  f"image={list(s['image'].shape)}  "
                  f"mask={list(s['mask'].shape)}  "
                  f"classes={unique_classes}  "
                  f"domains={len(domains)}")
        except Exception as e:
            print(f"[{split:5s}] ERROR: {type(e).__name__}: {e}")
