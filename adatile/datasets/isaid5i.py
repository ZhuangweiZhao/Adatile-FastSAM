#!/usr/bin/env python3
"""
标准 iSAID-5i 数据集适配器 | Standard iSAID-5i Dataset Adapter.
=============================================================

读取官方 iSAID-5i Few-shot Benchmark 数据格式，提供与现有训练框架兼容的接口。
Reads official iSAID-5i Few-shot Benchmark data format, provides API compatible
with existing training framework.

数据格式 | Data Format::

    iSAID-5i/iSAID/
    ├── label.xlsx                          # 类别名称映射 | Category name mapping
    ├── train/
    │   ├── images/                         # 256×256 RGB tiles
    │   ├── semantic_png/                   # 256×256 单通道 0-15 类别标签
    │   ├── semantic_mask/                  # RGB 彩色语义掩码 (可选用)
    │   ├── instance_mask/                  # RGB 实例掩码 (可选用)
    │   └── train_list/
    │       ├── split0_train.txt            # Fold 0: Base 类 tiles
    │       ├── split1_train.txt            # Fold 1: Base 类 tiles
    │       └── split2_train.txt            # Fold 2: Base 类 tiles
    └── val/
        ├── images/                         # 256×256 RGB tiles
        ├── semantic_png/                   # 256×256 单通道 0-15 类别标签
        ├── semantic_mask/                  # RGB 彩色语义掩码
        ├── instance_mask/                  # RGB 实例掩码
        └── val_list/
            ├── split0_val.txt              # Fold 0: Novel 类 val tiles
            ├── split1_val.txt              # Fold 1: Novel 类 val tiles
            └── split2_val.txt              # Fold 2: Novel 类 val tiles

Tile 命名格式 | Tile Naming::

    P{img_id}_{x1}_{y1}_{x2}_{y2}.png

标准 iSAID-5i 使用 256×256 tiles。FastSAM stride=16 → 仅 16×16 feature cells。
为充分利用 FastSAM 多尺度表示，我们保留 896×896 tile 切分策略，
但严格遵循官方 Fold 划分和类别映射。
Standard iSAID-5i uses 256×256 tiles. FastSAM stride=16 → only 16×16 cells.
To better exploit FastSAM multi-scale representation, we keep 896×896 tiling
while strictly following official Fold splits and category mappings.

用法 | Usage::

    from adatile.datasets.isaid5i import ISAID5iDataset

    ds = ISAID5iDataset("data/iSAID-5i/iSAID", split="train", fold=0)
    tiles_for_class_9 = ds.class_to_images(9)  # small_vehicle
    img = ds.load_image(tile_idx)
    mask = ds.render_class_mask(tile_idx, class_id=9)
"""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


class ISAID5iDataset:
    """
    标准 iSAID-5i 数据集适配器 | Standard iSAID-5i Dataset Adapter.

    提供 class_to_images / load_image / render_class_mask 三个必需方法，
    与 FewShotEpisodeDataset 和 PreCutTileAdapter 接口完全兼容。
    Provides three required methods compatible with FewShotEpisodeDataset API.

    Parameters
    ----------
    root : str
        iSAID-5i 数据根目录 (e.g. "data/iSAID-5i/iSAID").
    split : str
        "train" 或 "val".
    fold : int
        Fold ID (0/1/2). 控制使用哪个 split 文件。
    tile_size : int
        Tile 尺寸 (默认 256，标准 iSAID-5i).
    """

    def __init__(
        self,
        root: str = "data/iSAID-5i/iSAID",
        split: str = "train",
        fold: int = 0,
        tile_size: int = 256,
    ):
        import cv2
        self._cv2 = cv2

        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.tile_size = tile_size

        self._img_dir = self.root / split / "images"
        self._mask_dir = self.root / split / "semantic_png"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self._img_dir}")
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self._mask_dir}")

        # ── 加载 split 文件 | Load split file ──
        list_dir = self.root / split / f"{split}_list"
        list_file = list_dir / f"split{fold}_{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {list_file}\n"
                f"Available: {list(list_dir.glob('*.txt')) if list_dir.exists() else 'N/A'}"
            )

        with open(list_file) as f:
            self._tile_names = [line.strip() for line in f if line.strip()]

        # ── 构建 tile 索引 | Build tile index ──
        # 格式: P{img_id}_{x1}_{y1}_{x2}_{y2}_instance_color_RGB.png_XX
        # 标准 iSAID-5i 文件名后缀可能是 _instance_color_RGB.png_XX
        # 提取基础 tile 名用于匹配 images/ 中的文件
        self._tiles = []
        for name in self._tile_names:
            # 从 split 文件中提取干净的 tile 名
            # 例如: "P1092_1648_1904_824_1080_instance_color_RGB.png_04"
            #    → "P1092_1648_1904_824_1080"
            clean = self._clean_tile_name(name)
            if clean:
                self._tiles.append(clean)

        # ── 构建 class → tile indices 映射 | Build class → tile indices ──
        self._cls_to_tiles = defaultdict(list)
        self._index_built = False  # 防止重复构建 | Prevent repeated builds

        total_tiles = len(self._tiles)
        print(f"[ISAID5iDataset] {split} (fold={fold}): {total_tiles} tiles "
              f"({tile_size}px), split_file={list_file.name}")

    @staticmethod
    def _clean_tile_name(raw: str) -> str | None:
        """从 split 文件行提取干净的 tile 名 | Extract clean tile name."""
        # 移除后缀 (_instance_color_RGB.png_XX, _instance_id_RGB.png_XX 等)
        raw = raw.strip()
        # 尝试多种后缀模式
        for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
            idx = raw.find(suffix)
            if idx > 0:
                return raw[:idx]
        return raw.rsplit(".", 1)[0] if "." in raw else raw

    def class_to_images(self, class_id: int) -> list[int]:
        """
        class_id → tile index list.
        延迟构建: 首次调用时扫描所有 tile 的语义掩码，之后永久缓存。
        Lazy: scans semantic masks on first call, then cached permanently.
        """
        if not self._index_built:
            self._build_class_index()
            self._index_built = True
        return self._cls_to_tiles.get(class_id, [])

    def _build_class_index(self):
        """扫描所有 tile 的语义掩码，构建 class→tiles 映射。"""
        from tqdm import tqdm
        print(f"[ISAID5iDataset] Building class index for {self.split} fold={self.fold} "
              f"({len(self._tiles)} tiles)...")
        for i, tile_name in enumerate(tqdm(self._tiles, desc="  Indexing")):
            mask_path = self._mask_dir / f"{tile_name}.png"
            # 尝试 _instance_color_RGB.png 后缀
            if not mask_path.exists():
                mask_path = self._mask_dir / f"{tile_name}_instance_color_RGB.png"
            if mask_path.exists():
                mask = self._cv2.imread(str(mask_path), self._cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    if mask.ndim == 3:
                        # RGB mask → 取第一个通道 (所有通道值相同)
                        mask = mask[:, :, 0]
                    classes = set(np.unique(mask).tolist()) - {0}
                    for c in classes:
                        self._cls_to_tiles[int(c)].append(i)
        n_classes = len(self._cls_to_tiles)
        print(f"[ISAID5iDataset] Index done: {n_classes} classes with FG tiles")

    def _get_mask_path(self, tile_name: str) -> Path:
        """获取语义掩码路径 | Get semantic mask path."""
        # 标准 iSAID-5i 的 semantic_png 是单通道 PNG
        p = self._mask_dir / f"{tile_name}.png"
        if p.exists():
            return p
        p = self._mask_dir / f"{tile_name}_instance_color_RGB.png"
        if p.exists():
            return p
        return self._mask_dir / f"{tile_name}_instance_color_RGB.png"

    def load_image(self, tile_idx: int) -> torch.Tensor:
        """加载 tile 图像 → [3, H, W] float32."""
        tile_name = self._tiles[tile_idx]
        # 尝试 .png 和 .jpg
        img_path = self._img_dir / f"{tile_name}.png"
        if not img_path.exists():
            img_path = self._img_dir / f"{tile_name}_instance_color_RGB.png"
        img = self._cv2.imread(str(img_path), self._cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {tile_name} at {img_path}")
        img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def render_class_mask(self, tile_idx: int, class_id: int) -> torch.Tensor:
        """渲染指定类别的二值掩码 → [H, W] float32."""
        tile_name = self._tiles[tile_idx]
        mask_path = self._get_mask_path(tile_name)
        mask = self._cv2.imread(str(mask_path), self._cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {tile_name}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return torch.from_numpy((mask == class_id).astype(np.float32))

    def __len__(self) -> int:
        return len(self._tiles)

    @staticmethod
    def get_tile_size() -> int:
        return 256
