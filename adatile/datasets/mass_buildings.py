"""
MassachusettsBuildingsDataset — 马萨诸塞州建筑语义分割数据集。
================================================================
Massachusetts Buildings semantic segmentation dataset.

二值分类：背景 (0) vs 建筑 (1)。| Binary: background (0) vs building (1).

数据集特性 | Dataset characteristics:
    - 图像：1500×1500 RGB PNG 航拍图 | Images: 1500×1500 RGB PNG aerial
    - 标注：RGB PNG (255,255,255=建筑, 0,0,0=背景) | Labels: RGB (255=building, 0=background)
    - 划分：train=137, val=4, test=10 | Splits: train=137, val=4, test=10
    - 数据源：Toronto 大学 Mass Buildings 数据集
      Source: University of Toronto Mass Buildings Dataset

目录结构 | Directory structure:
    Massachusetts_Buildings/
    ├── png/
    │   ├── train/           # 训练图像 | Training images
    │   ├── train_labels/    # 训练标注 | Training labels
    │   ├── val/             # 验证图像 | Validation images
    │   ├── val_labels/      # 验证标注 | Validation labels
    │   ├── test/            # 测试图像 | Test images
    │   └── test_labels/     # 测试标注 | Test labels
    ├── tiff/                # TIFF 格式副本（不使用）| TIFF copy (unused)
    ├── label_class_dict.csv # 类别映射 | Class mapping
    └── metadata.csv         # 元数据 | Metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from adatile.datasets.base import BaseSegDataset


class MassachusettsBuildingsDataset(BaseSegDataset):
    """
    马萨诸塞州建筑语义分割数据集 | Massachusetts Buildings semantic segmentation dataset.

    与 iSAID 的关键区别：
    - 语义分割（单通道 [1,H,W]），不是实例分割（多通道 [N,H,W]）
    - RGB 标注 → 自动转为二值 mask
    - 无 COCO JSON，直接使用目录中的 PNG 文件

    Key differences from iSAID:
    - Semantic segmentation (single [1,H,W]), not instance (multi [N,H,W])
    - RGB labels → auto-converted to binary mask
    - No COCO JSON, directly uses PNG files in directories

    ----------
    root_dir : str
        数据集根目录 | Dataset root directory.
    split : str
        数据划分 ("train", "val", "test") | Data split.
    tile_size : int | None
        瓦片尺寸。None = 全图 (1500×1500)。建议值 512 或 768。
        Tile size. None = full image. Recommended 512 or 768.
    tile_overlap : float
        瓦片重叠比例 (0.0 ~ 1.0) | Tile overlap ratio.
    transforms : callable | None
        数据增强变换 | Optional data augmentation transforms.
    """

    # 类别定义 | Class definitions
    NUM_CLASSES = 2
    CLASS_NAMES = ["background", "building"]

    def __init__(
        self,
        root_dir: str = "data/Massachusetts_Buildings",
        split: str = "train",
        tile_size: int | None = None,
        tile_overlap: float = 0.0,
        transforms=None,
    ) -> None:
        super().__init__(root_dir=root_dir, split=split, transforms=transforms)
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

        # 构建文件列表 | Build file list
        self._img_dir = self.root_dir / "png" / self.split
        self._label_dir = self.root_dir / "png" / f"{self.split}_labels"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"图像目录未找到 | Image dir not found: {self._img_dir}")
        if not self._label_dir.exists():
            raise FileNotFoundError(f"标注目录未找到 | Label dir not found: {self._label_dir}")

        # 收集所有图像文件名 | Collect all image filenames
        self._image_files = sorted([
            f.name for f in self._img_dir.glob("*.png")
        ])

        # 预计算 tile 索引（仅在 tile 模式）| Precompute tile index (tile mode only)
        self._tile_index: list[tuple[int, int, int, int]] = []  # (img_idx, x, y, tile_idx)
        if self._tile_size is not None:
            self._build_tile_index()

        # 日志 | Logging
        self.logger.log_info(
            "dataset/mass_buildings_init",
            f"Massachusetts Buildings {split}: {len(self)} samples, "
            f"{len(self._image_files)} images, "
            f"tile_size={tile_size}, "
            f"num_classes={self.NUM_CLASSES}",
        )

    # ── 抽象方法实现 | Abstract Method Implementations ────────

    def _load_image(self, index: int) -> torch.Tensor:
        """
        加载并归一化图像 | Load and normalize image.

        PNG → RGB → float32 [0, 1] → [C, H, W].
        """
        filename = self._image_files[index]
        img_path = self._img_dir / filename

        # PIL 读取 | PIL read
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, C], [0, 1]

        # [H, W, C] → [C, H, W]
        return torch.from_numpy(img_np).permute(2, 0, 1)

    def _load_masks(self, index: int) -> torch.Tensor:
        """
        加载并转换标注为二值 mask | Load and convert label to binary mask.

        RGB 标注 → 二值 mask:
            - 原标注: building=(255,255,255), background=(0,0,0)
            - Original: building=255, background=0
            - 转换：取 R 通道 > 128 → 1, 否则 → 0
            - Convert: R channel > 128 → 1, else → 0

        :return: torch.Tensor [1, H, W] float32, 二值 (0/1) | Binary (0/1). 语义分割：始终返回 [1, H, W]（单通道）。 Semantic: always returns [1, H, W] (single channel).
        :rtype: torch.Tensor
        """
        filename = self._image_files[index]
        label_path = self._label_dir / filename

        # 读取 RGB 标注 | Read RGB label
        label = Image.open(label_path).convert("RGB")
        label_np = np.array(label)  # [H, W, 3]

        # 取 R 通道（三个通道值相同）| Take R channel (all 3 channels are identical)
        # building: (255,255,255) → 255 → >128 → 1
        # background: (0,0,0) → 0 → >128 → 0
        binary_mask = (label_np[:, :, 0] > 128).astype(np.float32)

        # [H, W] → [1, H, W]
        return torch.from_numpy(binary_mask).unsqueeze(0)

    def _load_image_id(self, index: int) -> str:
        """
        返回图像 ID（文件名去掉扩展名）| Return image ID (filename without extension).
        例: "22678915_15.png" → "22678915_15"
        """
        return self._image_files[index].replace(".png", "")

    def __len__(self) -> int:
        """
        数据集样本数量 | Number of samples.
        Tile 模式：返回瓦片总数；否则返回图像数。
        Tile mode: returns total tiles; otherwise returns image count.
        """
        if self._tile_size is not None:
            return len(self._tile_index)
        return len(self._image_files)

    # ── Tile 支持 | Tile Support ──────────────────────────────

    def _build_tile_index(self) -> None:
        """
        构建瓦片索引 | Build tile index.

        将每张 1500×1500 图像按 tile_size 切分，
        记录每个瓦片的位置 (img_idx, x, y, tile_id)。
        Splits each 1500×1500 image by tile_size,
        records each tile position.
        """
        stride = int(self._tile_size * (1 - self._tile_overlap))

        for img_idx in range(len(self._image_files)):
            # 获取图像尺寸（通过 metadata 或固定 1500）| Get image size (from metadata or fixed 1500)
            h = w = 1500  # Massachusetts Buildings 固定尺寸 | fixed size
            tile_idx = 0
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    self._tile_index.append((img_idx, x, y, tile_idx))
                    tile_idx += 1

    def _extract_tile(
        self, image: torch.Tensor, masks: torch.Tensor, x: int, y: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从全图中提取瓦片 | Extract tile from full image.

        :param image: [C, H, W] 全图 | Full image.
        :type image: torch.Tensor

        :param masks: [1, H, W] 全尺寸掩码 | Full-size mask. x, y:  瓦片左上角坐标 | Tile top-left coordinates.
        :type masks: torch.Tensor

        :param x: 
        :type x: int

        :param y: 
        :type y: int

        :return: (tile_image [C, th, tw], tile_masks [1, th, tw])
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        ts = self._tile_size
        _, h, w = image.shape
        th = min(ts, h - y)
        tw = min(ts, w - x)

        tile_img = image[:, y:y + th, x:x + tw]
        tile_masks = masks[:, y:y + th, x:x + tw]
        return tile_img, tile_masks

    def __getitem__(self, index: int) -> dict:
        """
        获取样本（支持 tile 模式）| Get sample (supports tile mode).

        Tile 模式：通过 tile_index 映射到 (img_idx, x, y)。
        Non-tile 模式：直接使用父类实现。
        """
        if self._tile_size is not None:
            img_idx, x, y, _ = self._tile_index[index]

            # 加载全图 + 掩码 | Load full image + mask
            image = self._load_image(img_idx)
            masks = self._load_masks(img_idx)
            image_id = self._load_image_id(img_idx)

            # 提取瓦片 | Extract tile
            tile_img, tile_masks = self._extract_tile(image, masks, x, y)

            sample = {
                "image": tile_img,
                "masks": tile_masks,
                "image_id": image_id,
                "image_size": tuple(tile_img.shape[1:]),  # tile (H, W)
            }
            return self._apply_transforms(sample)

        # 非 tile 模式 | Non-tile mode
        return super().__getitem__(index)

    # ── 公共属性 | Public Properties ──────────────────────────

    @property
    def num_classes(self) -> int:
        """类别总数 | Total number of classes."""
        return self.NUM_CLASSES

    @property
    def class_names(self) -> list[str]:
        """类别名称列表 | List of class names."""
        return list(self.CLASS_NAMES)
