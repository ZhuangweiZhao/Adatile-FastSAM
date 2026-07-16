"""
NEU_Seg 数据集加载器 | NEU_Seg Dataset Loader.
================================================

NEU_Seg 工业缺陷分割数据集 (多类别) | Industrial Defect Segmentation Dataset (Multi-class).

数据集: 200×200 表面缺陷图块, 三类别 + 背景。
Dataset: 200×200 tiles, 3 defect classes + background.

类别编码 | Class Encoding:
    0 = Background (背景)
    1 = Inclusion (夹杂物)
    2 = Patch (斑块)
    3 = Scratch (划痕)

数据集统计 | Dataset Statistics:
    - training: 3,630 (Inclusion≈27%, Patch≈27%, Scratch≈27%, Mixed≈19%)
    - test:       840
    - 图像: 200×200 RGB JPG
    - 标注: 200×200 PNG, 像素值 {0,1,2,3}

目录结构 | Directory Structure:
    NEU_Seg/
    ├── images/
    │   ├── training/       # 3630 JPG images
    │   └── test/           # 840 JPG images
    └── annotations/
        ├── training/       # 3630 PNG masks (0/1/2/3)
        └── test/           # 840 PNG masks

用法 | Usage::

    from adatile.datasets.neu_seg import NEUSegDataset

    # 二值训练 (合并 {1,2,3}→1) | Binary training (merge {1,2,3}→1)
    ds = NEUSegDataset(root="data/NEU_Seg", split="train", binary=True)

    # 多类别训练 (保留 0/1/2/3) | Multi-class training (preserve 0/1/2/3)
    ds = NEUSegDataset(root="data/NEU_Seg", split="train", binary=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from adatile.datasets.base import BaseSegDataset
from adatile.logging import get_logger


# ═══════════════════════════════════════════════════════════════════
# NEUSegDataset — NEU_Seg 数据集 | NEU_Seg Dataset
# ═══════════════════════════════════════════════════════════════════

class NEUSegDataset(BaseSegDataset):
    """
    NEU_Seg 工业缺陷分割数据集 (多类别) | Industrial Defect Segmentation Dataset (Multi-class).

    200×200 航拍/表面缺陷图块, 三类别 + 背景。
    200×200 tiles, 3 defect classes + background.

    类别编码 | Class Encoding:
        0 = Background (背景)
        1 = Inclusion (夹杂物)
        2 = Patch (斑块)
        3 = Scratch (划痕)

    Parameters
    ----------
    root : str
        数据集根目录 | Dataset root (e.g. "data/NEU_Seg").
    split : str
        "train" → images/training/, "val" or "test" → images/test/.
    binary : bool
        True → 二值化 mask (FG>0→1), 用于二值分割训练。
        False → 保留 0/1/2/3 多类别标签。
    transforms : callable | None
        数据增强变换 | Optional data augmentation transforms.

    返回格式 | Return Format (binary=True)::

        {
            "image": Tensor[3, H, W],       float32 [0,1]
            "masks": Tensor[1, H, W],        二值 float32 {0, 1}
            "image_id": str,
            "image_size": (H, W),
        }

    返回格式 | Return Format (binary=False)::

        {
            "image": Tensor[3, H, W],       float32 [0,1]
            "masks": Tensor[1, H, W],        多类别 int64 {0, 1, 2, 3}
            "image_id": str,
            "image_size": (H, W),
        }
    """

    NUM_CLASSES = 4  # BG + Inclusion + Patch + Scratch
    CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

    _SPLIT_MAP = {"train": "training", "val": "test", "test": "test"}

    def __init__(
        self,
        root: str = "data/NEU_Seg",
        split: str = "train",
        binary: bool = False,
        transforms=None,
    ) -> None:
        super().__init__(root_dir=root, split=split, transforms=transforms)

        self.binary = binary
        self._root = Path(root)
        dir_name = self._SPLIT_MAP.get(split, split)

        self._img_dir = self._root / "images" / dir_name
        self._ann_dir = self._root / "annotations" / dir_name

        # ── 验证目录存在 | Validate Directories ──
        for d, name in [(self._img_dir, "images"), (self._ann_dir, "annotations")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"目录未找到 | Directory not found: {d}\n"
                    f"Expected: root/images/{dir_name}/ and root/annotations/{dir_name}/"
                )

        # ── 扫描文件列表 | Scan file list ──
        img_files = sorted([f.stem for f in self._img_dir.glob("*.jpg")])
        self._samples = [
            name for name in img_files
            if (self._ann_dir / f"{name}.png").exists()
        ]

        skipped = len(img_files) - len(self._samples)
        if skipped > 0:
            self.logger.log_warn(
                "dataset/missing_masks",
                f"Skipped {skipped} images without matching mask in {dir_name}"
            )

        # ── 类别统计 (惰性计算) | Class stats (lazy) ──
        self._class_stats_computed = False
        self._class_mask_counts: dict[int, int] = {}
        self._class_pixel_counts: dict[int, int] = {}

        mode_str = "binary" if binary else "multi-class (0/1/2/3)"
        self.logger.log_info(
            "dataset/neuseg_init",
            f"NEU_Seg ({dir_name}): {len(self)} samples, "
            f"num_classes={self.NUM_CLASSES}, image_size=200x200, mode={mode_str}"
        )

    # ── 抽象方法实现 | Abstract Method Implementations ──

    def _load_image(self, index: int) -> torch.Tensor:
        """加载并归一化图像 | Load and normalize image. JPG → RGB → float32 [0,1] → [C,H,W]."""
        name = self._samples[index]
        img_path = self._img_dir / f"{name}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像 | Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def _load_masks(self, index: int) -> torch.Tensor:
        """
        加载标注掩码 | Load annotation mask.

        PNG (0/1/2/3) → binary={0,1} 或 multi-class={0,1,2,3}.
        """
        name = self._samples[index]
        ann_path = self._ann_dir / f"{name}.png"
        mask = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"无法读取标注 | Cannot read annotation: {ann_path}")

        if self.binary:
            # 二值模式: FG>0 → 1
            binary_mask = (mask > 0).astype(np.float32)
            return torch.from_numpy(binary_mask).unsqueeze(0)
        else:
            # 多类别模式: 保留 0/1/2/3
            return torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

    def _load_image_id(self, index: int) -> str:
        """返回样本名 (无扩展名) | Return sample name (without extension)."""
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    # ── 公共属性 | Public Properties ──

    @property
    def num_classes(self) -> int:
        """类别总数 | Total classes: 2 (binary) or 4 (multi-class)."""
        return 2 if self.binary else self.NUM_CLASSES

    @property
    def class_names(self) -> list[str]:
        """类别名称 | Class names."""
        if self.binary:
            return ["background", "foreground"]
        return list(self.CLASS_NAMES)

    @property
    def sample_names(self) -> list[str]:
        """所有样本名列表 | List of all sample names."""
        return list(self._samples)

    # ── 类别统计 (惰性) | Class Stats (lazy) ──

    def _ensure_class_stats(self):
        """计算类别统计 | Compute class statistics."""
        if self._class_stats_computed:
            return
        for i in range(len(self)):
            mask = cv2.imread(str(self._ann_dir / f"{self._samples[i]}.png"), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            for v in np.unique(mask):
                v_int = int(v)
                self._class_mask_counts[v_int] = self._class_mask_counts.get(v_int, 0) + 1
                self._class_pixel_counts[v_int] = self._class_pixel_counts.get(v_int, 0) + int((mask == v_int).sum())
        self._class_stats_computed = True

    def get_class_stats(self) -> dict:
        """获取类别统计 | Get per-class statistics."""
        self._ensure_class_stats()
        total_px = sum(self._class_pixel_counts.values())
        return {
            self.CLASS_NAMES[k]: {
                "masks": self._class_mask_counts.get(k, 0),
                "pixels": self._class_pixel_counts.get(k, 0),
                "pixel_pct": round(100 * self._class_pixel_counts.get(k, 0) / max(total_px, 1), 2),
            }
            for k in range(self.NUM_CLASSES)
        }

    def get_fg_ratio(self, class_id: int | None = None) -> float:
        """
        获取前景占比 | Get FG ratio.

        :param class_id: 指定类别 (None=所有FG, 1=Inclusion, 2=Patch, 3=Scratch).
        """
        self._ensure_class_stats()
        total_px = sum(self._class_pixel_counts.values())
        if class_id is not None:
            return self._class_pixel_counts.get(class_id, 0) / max(total_px, 1)
        fg_px = sum(v for k, v in self._class_pixel_counts.items() if k > 0)
        return fg_px / max(total_px, 1)


# ═══════════════════════════════════════════════════════════════════
# Self-test | 自测
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "data/NEU_Seg"

    ds_path = Path(root_dir)
    if not ds_path.exists():
        print(f"\n[ERROR] Dataset root not found: {root_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"  NEUSegDataset — Self-Test")
    print(f"  Root:    {root_dir}")
    print("=" * 60)

    for split in ["train", "test"]:
        print(f"\n── {split} ──")
        try:
            # 二值模式
            ds_bin = NEUSegDataset(root=root_dir, split=split, binary=True)
            print(f"  Binary mode: {len(ds_bin)} samples, classes={ds_bin.class_names}")

            # 多类别模式
            ds_mc = NEUSegDataset(root=root_dir, split=split, binary=False)
            print(f"  Multi-class mode: {len(ds_mc)} samples, classes={ds_mc.class_names}")

            if len(ds_mc) > 0:
                sample = ds_mc[0]
                print(f"  Sample 0: image={list(sample['image'].shape)}, "
                      f"masks={list(sample['masks'].shape)}, "
                      f"id={sample['image_id']}, "
                      f"size={sample['image_size']}")
                print(f"    mask dtype={sample['masks'].dtype}, "
                      f"unique={sample['masks'].unique().tolist()}")

                # 全量统计
                fg_ratios = []
                for i in range(len(ds_bin)):
                    s = ds_bin[i]
                    fg_ratios.append(s["masks"].mean().item())
                arr = np.array(fg_ratios)
                print(f"    FG ratio: min={arr.min():.4f} max={arr.max():.4f} "
                      f"mean={arr.mean():.4f} median={np.median(arr):.4f} "
                      f"std={arr.std():.4f}")

            # 类别统计
            stats = ds_mc.get_class_stats()
            print(f"  Class distribution:")
            for cls_name, info in stats.items():
                print(f"    {cls_name}: {info['masks']} masks, {info['pixel_pct']}% pixels")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Done] Self-test complete.")
