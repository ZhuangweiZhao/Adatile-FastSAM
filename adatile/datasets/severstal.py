"""
Severstal 钢铁缺陷检测数据集 | Severstal Steel Defect Detection Dataset.
==========================================================================

Kaggle Severstal 钢铁表面缺陷检测竞赛数据集适配。
Adaptation of the Kaggle Severstal Steel Defect Detection competition dataset.

数据集: 1600×256 钢带表面图像, RLE 编码标注, 4 类缺陷。
Dataset: 1600×256 steel strip images, RLE-encoded annotations, 4 defect classes.

类别编码 | Class Encoding (multi-class mode):
    0 = Background (无缺陷 | no defect)
    1 = Class 1 (缺陷类型 1 | defect type 1)
    2 = Class 2 (缺陷类型 2 | defect type 2)
    3 = Class 3 (缺陷类型 3 | defect type 3)
    4 = Class 4 (缺陷类型 4 | defect type 4)

关键特性 | Key Features:
    - 12,568 训练图像 (6,666 有缺陷 + 5,902 无缺陷/干净)
    - 12,568 training images (6,666 with defects + 5,902 clean/defect-free)
    - RLE 编码采用 Fortran (列优先) 顺序! RLE uses Fortran (column-major) order!
    - 图像尺寸 1600×256 (宽×高), 自动 pad 到 32 的倍数
    - 支持二值模式 (FG→1) 和多类别模式 (5 类)

目录结构 | Directory Structure:
    severstal-steel-defect-detection/
    ├── train.csv              # RLE 标注 (ImageId, ClassId, EncodedPixels)
    ├── train_images/          # 12,568 JPG images
    ├── test_images/           # 5,506 JPG images (无标注)
    └── sample_submission.csv  # Kaggle 提交格式

用法 | Usage::

    from adatile.datasets.severstal import SeverstalDataset

    # 二值训练 (合并所有缺陷类→1) | Binary training (merge all defects→1)
    ds = SeverstalDataset(root="data/severstal-steel-defect-detection", split="train", binary=True)

    # 多类别训练 (5 类: 0=BG, 1-4=各缺陷类) | Multi-class training (5 classes)
    ds = SeverstalDataset(root="data/severstal-steel-defect-detection", split="train", binary=False)

注意 | Note:
    RLE 编码格式为 Fortran (列优先) 顺序: mask.reshape(H, W, order='F').
    这与标准 COCO RLE (行优先, order='C') 不同!
    The RLE format is Fortran (column-major) order: mask.reshape(H, W, order='F').
    This differs from standard COCO RLE (row-major, order='C')!
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

from adatile.datasets.base import BaseSegDataset
from adatile.logging import get_logger


# ═══════════════════════════════════════════════════════════════════
# RLE 解码 | RLE Decoding
# ═══════════════════════════════════════════════════════════════════

def _decode_rle(rle_str: str, shape: tuple[int, int] = (256, 1600)) -> np.ndarray:
    """
    解码 Severstal RLE 格式为二值掩码 | Decode Severstal RLE format to binary mask.

    Severstal 的 RLE 是 Fortran (列优先) 顺序: mask.reshape(H, W, order='F').
    RLE 格式: 成对的 (start_position, run_length), start 为 1-indexed.

    Severstal RLE is Fortran (column-major) order: mask.reshape(H, W, order='F').
    RLE format: pairs of (start_position, run_length), start is 1-indexed.

    :param rle_str: RLE 字符串, 如 "29102 12 29346 24 ..."
    :param shape: 掩码形状 (H, W) | Mask shape (H, W). Default (256, 1600).
    :return: np.ndarray [H, W] uint8, 二值 {0, 1}.
    """
    if pd.isna(rle_str) or str(rle_str).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    numbers = [int(x) for x in str(rle_str).split()]
    H, W = shape
    flat_mask = np.zeros(H * W, dtype=np.uint8)

    for i in range(0, len(numbers), 2):
        start = numbers[i] - 1  # 1-indexed → 0-indexed
        length = numbers[i + 1]  # run length in pixels
        if start + length > H * W:
            # 截断防止越界 | Truncate to prevent overflow
            length = H * W - start
            if length <= 0:
                continue
        flat_mask[start:start + length] = 1

    # Fortran (column-major) 重塑 | Fortran (column-major) reshape
    return flat_mask.reshape(shape, order="F").astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
# SeverstalDataset — 主数据集类 | Main Dataset Class
# ═══════════════════════════════════════════════════════════════════

class SeverstalDataset(BaseSegDataset):
    """
    Severstal 钢铁缺陷检测数据集 | Severstal Steel Defect Detection Dataset.

    1600×256 钢带表面图像, 4 类缺陷, RLE 标注。
    1600×256 steel strip images, 4 defect classes, RLE annotations.

    类别编码 | Class Encoding:
        0 = Background (背景/无缺陷 | no defect)
        1 = Defect Class 1 (缺陷类型 1)
        2 = Defect Class 2 (缺陷类型 2)
        3 = Defect Class 3 (缺陷类型 3)
        4 = Defect Class 4 (缺陷类型 4)

    Parameters
    ----------
    root : str
        数据集根目录 | Dataset root (e.g. "data/severstal-steel-defect-detection").
    split : str
        "train" → 80% 训练集, "val" or "test" → 20% 验证集.
    binary : bool
        True → 二值化 mask (FG>0→1), 2 类输出.
        False → 保留 0/1/2/3/4 多类别标签, 5 类输出.
    transforms : callable | None
        数据增强变换 | Optional data augmentation transforms.

    返回格式 | Return Format (binary=True)::

        {
            "image": Tensor[3, H, W],       float32 [0,1], H=288, W=1632
            "masks": Tensor[1, H, W],        二值 float32 {0, 1}
            "image_id": str,
            "image_size": (H, W),            原始尺寸 | original size (256, 1600)
        }

    返回格式 | Return Format (binary=False)::

        {
            "image": Tensor[3, H, W],       float32 [0,1], H=288, W=1632
            "masks": Tensor[1, H, W],        多类别 int64 {0, 1, 2, 3, 4}
            "image_id": str,
            "image_size": (H, W),            原始尺寸 | original size (256, 1600)
        }
    """

    NUM_CLASSES = 5  # BG + Class1 + Class2 + Class3 + Class4
    CLASS_NAMES = ["background", "Class1", "Class2", "Class3", "Class4"]

    # 原始图像尺寸 | Native image dimensions: (H, W)
    IMG_H = 256
    IMG_W = 1600

    def __init__(
        self,
        root: str = "data/severstal-steel-defect-detection",
        split: str = "train",
        binary: bool = False,
        transforms=None,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__(root_dir=root, split=split, transforms=transforms)

        self.binary = binary
        self._root = Path(root)
        self._img_dir = self._root / "train_images"
        self._csv_path = self._root / "train.csv"
        self._val_ratio = val_ratio
        self._seed = seed

        # ── 验证目录/文件存在 | Validate paths ──
        for p, name in [(self._img_dir, "train_images"),
                        (self._csv_path, "train.csv")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"路径未找到 | Path not found: {p}\n"
                    f"Expected: root/train_images/ and root/train.csv"
                )

        # ── 加载 CSV 标注 | Load CSV annotations ──
        self._df = pd.read_csv(str(self._csv_path))
        self.logger.log_info(
            "dataset/severstal_csv",
            f"Loaded {len(self._df)} annotation rows from train.csv"
        )

        # ── 构建图像列表 (包含无缺陷图像) | Build image list (including defect-free) ──
        # CSV 中的图像 (有缺陷) | Images in CSV (with defects)
        csv_images = set(self._df["ImageId"].unique())
        self.logger.log_info(
            "dataset/severstal_images",
            f"Images in CSV (with defects): {len(csv_images)}"
        )

        # 文件夹中所有图像 | All images in folder
        all_images = sorted([
            f.name for f in self._img_dir.glob("*.jpg")
        ])
        self.logger.log_info(
            "dataset/severstal_all",
            f"Total images in folder: {len(all_images)}"
        )

        # 无缺陷图像 = 文件夹有但 CSV 没有的 | Defect-free = in folder but not in CSV
        clean_images = sorted(list(set(all_images) - csv_images))
        self.logger.log_info(
            "dataset/severstal_clean",
            f"Defect-free (clean) images: {len(clean_images)}"
        )

        # ── 有缺陷图像的标注缓存 | Build annotations cache for defect images ──
        self._annotations: dict[str, dict[int, str]] = {}
        # 格式: {image_id: {class_id: rle_string}}
        for img_id in csv_images:
            self._annotations[img_id] = {}
        for _, row in self._df.iterrows():
            img_id = row["ImageId"]
            cls_id = int(row["ClassId"])
            rle = str(row["EncodedPixels"]) if pd.notna(row["EncodedPixels"]) else ""
            if rle:
                self._annotations[img_id][cls_id] = rle

        # ── 合并所有图像 + 划分 Train/Val Split ──
        # 分层划分: 有缺陷图像和无缺陷图像分别 shuffle
        rng = np.random.RandomState(seed)
        defect_images = sorted(list(csv_images))
        rng.shuffle(defect_images)
        rng.shuffle(clean_images)

        n_val_defect = max(1, int(len(defect_images) * val_ratio))
        n_val_clean = max(1, int(len(clean_images) * val_ratio))

        if split in ("train",):
            self._defect_samples = defect_images[n_val_defect:]
            self._clean_samples = clean_images[n_val_clean:]
        else:  # val / test
            self._defect_samples = defect_images[:n_val_defect]
            self._clean_samples = clean_images[:n_val_clean]

        # ── 构建完整样本列表 (有缺陷在前, 无缺陷在后) ──
        self._samples = (
            [(name, True) for name in self._defect_samples]
            + [(name, False) for name in self._clean_samples]
        )
        # samples: List[Tuple[str, bool]] — (image_name, has_defect)

        # ── 延迟类别统计 | Lazy class stats ──
        self._class_stats_computed = False
        self._class_mask_counts: dict[int, int] = {}
        self._class_pixel_counts: dict[int, int] = {}

        # ── 日志 | Log ──
        mode_str = "binary" if binary else "multi-class (0/1/2/3/4)"
        self.logger.log_info(
            "dataset/severstal_init",
            f"Severstal ({split}): {len(self)} samples "
            f"({len(self._defect_samples)} defective + {len(self._clean_samples)} clean), "
            f"num_classes={self.num_classes}, image_size={self.IMG_H}×{self.IMG_W}, "
            f"mode={mode_str}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 抽象方法实现 | Abstract Method Implementations
    # ═══════════════════════════════════════════════════════════════

    def _load_image(self, index: int) -> torch.Tensor:
        """
        加载并归一化图像 | Load and normalize image.
        JPG → RGB → float32 [0,1] → [C, H, W].
        自动 pad 到 32 的倍数 (FastSAM 要求).
        Auto-pad to multiple of 32 (FastSAM requirement).
        """
        name, _has_defect = self._samples[index]
        img_path = self._img_dir / name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像 | Cannot read image: {img_path}")
        # BGR → RGB → float32
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Pad to multiple of 32
        img = self._pad_image(img)
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def _load_masks(self, index: int) -> torch.Tensor:
        """
        加载标注掩码 | Load annotation mask.

        RLE → 多类别 (0/1/2/3/4) 或二值 (0/1).
        RLE → multi-class (0/1/2/3/4) or binary (0/1).

        无缺陷图像: 全 0 掩码 | Defect-free images: all-zero mask.
        """
        name, has_defect = self._samples[index]

        if not has_defect:
            # 无缺陷图像: 全背景 | Clean image: all background
            if self.binary:
                mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.float32)
            else:
                mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.int64)
        else:
            # 有缺陷图像: 解码 RLE | Defect image: decode RLE
            ann = self._annotations.get(name, {})
            if self.binary:
                # 二值模式: 所有缺陷类合并 → FG=1
                mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.float32)
                for cls_id, rle in ann.items():
                    cls_mask = _decode_rle(rle)
                    mask = np.maximum(mask, cls_mask.astype(np.float32))
            else:
                # 多类别模式: 按 ClassId 分配像素值
                # 冲突解决: 多个类重叠时, 取 RLE 面积更大的类
                # Conflict resolution: when classes overlap, prefer class with larger RLE area
                cls_masks: list[tuple[int, np.ndarray, int]] = []
                for cls_id, rle in ann.items():
                    m = _decode_rle(rle)
                    area = int(m.sum())
                    cls_masks.append((cls_id, m, area))
                # 按面积降序排序 (大面积优先)
                cls_masks.sort(key=lambda x: x[2], reverse=True)

                mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.int64)
                for cls_id, m, _area in cls_masks:
                    mask[m > 0] = cls_id  # 后到的不会覆盖先到的 (面积大的)

        # Pad to multiple of 32
        mask = self._pad_mask(mask)
        return torch.from_numpy(mask).unsqueeze(0)

    def _load_image_id(self, index: int) -> str:
        """返回样本名 (无扩展名) | Return sample name (without extension)."""
        name, _ = self._samples[index]
        return Path(name).stem

    def __len__(self) -> int:
        return len(self._samples)

    # ═══════════════════════════════════════════════════════════════
    # Padding 辅助 | Padding Helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _pad_image(img: np.ndarray) -> np.ndarray:
        """
        Pad 图像到 32 的倍数 | Pad image to multiple of 32.
        img: [H, W, C] float32.
        """
        H, W = img.shape[:2]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant", constant_values=0)
        return img

    @staticmethod
    def _pad_mask(mask: np.ndarray) -> np.ndarray:
        """
        Pad 掩码到 32 的倍数 | Pad mask to multiple of 32.
        mask: [H, W].
        """
        H, W = mask.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                         mode="constant", constant_values=0)
        return mask

    # ═══════════════════════════════════════════════════════════════
    # 公共属性 | Public Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def num_classes(self) -> int:
        """类别总数 | Total classes: 2 (binary) or 5 (multi-class)."""
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
        return [name for name, _ in self._samples]

    @property
    def native_size(self) -> tuple[int, int]:
        """原始图像尺寸 (H, W) | Native image size (H, W)."""
        return (self.IMG_H, self.IMG_W)

    def class_to_images(self, class_id: int) -> list[int]:
        """
        获取包含指定缺陷类的所有样本索引。
        Return all dataset indices containing a given defect class.

        遍历 _samples 和 _annotations, 找到包含指定 class_id 的图像，
        返回其在数据集中的索引列表（可用于 dataset[idx]）。

        Scans _samples and _annotations to find images containing the given
        class_id, returning their dataset indices (usable as dataset[idx]).

        :param class_id: 缺陷类 ID (1-4)。0 (background) 返回所有无缺陷图像。
        :return: 可用于 dataset[idx] 的索引列表。
        :raises ValueError: 如果 class_id 不在 [0, 1, 2, 3, 4] 范围内。
        """
        if class_id not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"class_id must be 0-4, got {class_id}. "
                f"0=clean images, 1-4=defect classes."
            )

        if class_id == 0:
            return [
                idx for idx, (_, has_defect) in enumerate(self._samples)
                if not has_defect
            ]

        result = []
        for idx, (name, has_defect) in enumerate(self._samples):
            if not has_defect:
                continue
            ann = self._annotations.get(name, {})
            if class_id in ann:
                result.append(idx)
        return result

    # ═══════════════════════════════════════════════════════════════
    # 类别统计 (惰性) | Class Stats (lazy)
    # ═══════════════════════════════════════════════════════════════

    def _ensure_class_stats(self):
        """计算类别统计 | Compute class statistics."""
        if self._class_stats_computed:
            return
        for idx in range(len(self)):
            name, has_defect = self._samples[idx]
            if not has_defect:
                # 无缺陷: 全背景
                self._class_mask_counts[0] = self._class_mask_counts.get(0, 0) + 1
                self._class_pixel_counts[0] = (
                    self._class_pixel_counts.get(0, 0) + self.IMG_H * self.IMG_W
                )
                continue
            ann = self._annotations.get(name, {})
            if not ann:
                self._class_mask_counts[0] = self._class_mask_counts.get(0, 0) + 1
                self._class_pixel_counts[0] = (
                    self._class_pixel_counts.get(0, 0) + self.IMG_H * self.IMG_W
                )
                continue

            # 有缺陷: 构建多类别掩码进行统计
            cls_masks: list[tuple[int, np.ndarray, int]] = []
            for cls_id, rle in ann.items():
                m = _decode_rle(rle)
                area = int(m.sum())
                cls_masks.append((cls_id, m, area))
            cls_masks.sort(key=lambda x: x[2], reverse=True)

            full_mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.int64)
            for cls_id, m, _area in cls_masks:
                full_mask[m > 0] = cls_id

            for v in np.unique(full_mask):
                v_int = int(v)
                self._class_mask_counts[v_int] = self._class_mask_counts.get(v_int, 0) + 1
                self._class_pixel_counts[v_int] = (
                    self._class_pixel_counts.get(v_int, 0) + int((full_mask == v_int).sum())
                )

        self._class_stats_computed = True

    def get_class_stats(self) -> dict:
        """获取类别统计 | Get per-class statistics."""
        self._ensure_class_stats()
        total_px = sum(self._class_pixel_counts.values())
        return {
            self.CLASS_NAMES[k]: {
                "masks": self._class_mask_counts.get(k, 0),
                "pixels": self._class_pixel_counts.get(k, 0),
                "pixel_pct": round(
                    100 * self._class_pixel_counts.get(k, 0) / max(total_px, 1), 4
                ),
            }
            for k in range(self.NUM_CLASSES)
        }

    def get_fg_ratio(self, class_id: int | None = None) -> float:
        """
        获取前景占比 | Get FG ratio.

        :param class_id: 指定类别 (None=所有FG, 1-4=各缺陷类).
        """
        self._ensure_class_stats()
        total_px = sum(self._class_pixel_counts.values())
        if class_id is not None:
            return self._class_pixel_counts.get(class_id, 0) / max(total_px, 1)
        fg_px = sum(v for k, v in self._class_pixel_counts.items() if k > 0)
        return fg_px / max(total_px, 1)


# ═══════════════════════════════════════════════════════════════════
# 少样本采样工具 | Few-Shot Sampling Utility
# ═══════════════════════════════════════════════════════════════════

def sample_k_shot_severstal(
    dataset: SeverstalDataset,
    k: int = 5,
    seed: int = 42,
    include_clean: bool = True,
    max_clean: int | None = None,
) -> list[int]:
    """
    从 Severstal 数据集中每类采样 K 张图，返回去重后的训练子集索引。
    Sample K images per defect class, return deduplicated training subset indices.

    对每类缺陷 (1-4) 随机采样最多 K 张包含该类的图，取并集去重。
    可选地加入无缺陷（干净）图像用于背景学习。

    For each defect class (1-4), randomly samples up to K images containing
    that class. Takes the union across all classes (deduplicated). Optionally
    includes clean (defect-free) images for background learning.

    :param dataset: SeverstalDataset 实例 (split="train")。
    :param k: 每类采样图像数 | Number of images to sample per class.
    :param seed: 随机种子 | Random seed for reproducibility.
    :param include_clean: 是否包含无缺陷图像 | Whether to include clean images.
    :param max_clean: 最多包含多少张无缺陷图 (None=全部) | Max clean images.
    :return: 排序后的数据集索引列表 | Sorted list of dataset indices.
    :raises ValueError: 如果 K 超过某类可用图像数。
    """
    rng = random.Random(seed)
    selected: set[int] = set()

    # ── 每类采样 K 张 | Sample K per class ──
    for cls_id in [1, 2, 3, 4]:
        candidates = dataset.class_to_images(cls_id)
        if k > len(candidates):
            raise ValueError(
                f"K={k} exceeds available images for Class {cls_id} "
                f"({len(candidates)} images). Reduce --k-shot or exclude this class."
            )
        n_pick = min(k, len(candidates))
        picked = rng.sample(candidates, n_pick)
        selected.update(picked)

    n_defect = len(selected)

    # ── 可选: 加入无缺陷图像 | Optional: add clean images ──
    n_clean = 0
    if include_clean:
        clean_candidates = dataset.class_to_images(0)
        if max_clean is not None:
            clean_candidates = rng.sample(
                clean_candidates, min(max_clean, len(clean_candidates))
            )
        selected.update(clean_candidates)
        n_clean = len(selected) - n_defect

    return sorted(selected)


# ═══════════════════════════════════════════════════════════════════
# Self-test | 自测
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "data/severstal-steel-defect-detection"

    ds_path = Path(root_dir)
    if not ds_path.exists():
        print(f"\n[ERROR] Dataset root not found: {root_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"  SeverstalDataset — Self-Test")
    print(f"  Root:    {root_dir}")
    print("=" * 60)

    for split in ["train", "val"]:
        print(f"\n── {split} ──")
        try:
            # 二值模式
            ds_bin = SeverstalDataset(root=root_dir, split=split, binary=True)
            print(f"  Binary mode: {len(ds_bin)} samples, classes={ds_bin.class_names}")

            # 多类别模式
            ds_mc = SeverstalDataset(root=root_dir, split=split, binary=False)
            print(f"  Multi-class mode: {len(ds_mc)} samples, classes={ds_mc.class_names}")

            if len(ds_mc) > 0:
                sample = ds_mc[0]
                print(f"  Sample 0: image={list(sample['image'].shape)}, "
                      f"masks={list(sample['masks'].shape)}, "
                      f"id={sample['image_id']}, "
                      f"size={sample['image_size']}")
                print(f"    mask dtype={sample['masks'].dtype}, "
                      f"unique={sample['masks'].unique().tolist()}")

                # 全量 FG ratio 统计
                fg_ratios = []
                for i in range(min(len(ds_bin), 200)):
                    s = ds_bin[i]
                    fg_ratios.append(s["masks"].mean().item())
                arr = np.array(fg_ratios)
                print(f"    FG ratio (first 200): min={arr.min():.4f} max={arr.max():.4f} "
                      f"mean={arr.mean():.4f} median={np.median(arr):.4f} "
                      f"std={arr.std():.4f}")

            # 类别统计
            stats = ds_mc.get_class_stats()
            print(f"  Class distribution:")
            for cls_name, info in stats.items():
                print(f"    {cls_name}: {info['masks']} masks, {info['pixel_pct']}% pixels")

            # 验证 train/val 无重叠
            if split == "train":
                train_names = set(ds_mc.sample_names)
            else:
                val_names = set(ds_mc.sample_names)

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # 检查 train/val 无重叠
    if "train_names" in dir() and "val_names" in dir():
        overlap = train_names & val_names
        print(f"\n  Train/Val overlap: {len(overlap)} images {'[WARNING!]' if overlap else '[OK]'}")

    print(f"\n[Done] Self-test complete.")
