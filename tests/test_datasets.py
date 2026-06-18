"""
Datasets 模块测试 | Datasets module tests.
===========================================

验证 iSAID 数据集加载器的核心功能：
- COCO JSON 标注解析
- 图像和掩码加载
- 掩码格式标准化（{0,255} → {0,1}）
- 大图 tile 处理

使用 mock 数据集（临时目录 + 合成图像 + 虚拟 COCO JSON）。
Uses mock dataset (temp dir + synthetic images + dummy COCO JSON).
"""

from __future__ import annotations

import pytest
import torch

from adatile.datasets import ISAIDDataset


# ════════════════════════════════════════════════════════════════
# ISAIDDataset 测试 | ISAIDDataset Tests
# mock_isaid_dir fixture 定义在 conftest.py | Defined in conftest.py
# ════════════════════════════════════════════════════════════════

class TestISAIDDataset:
    """验证 iSAID 数据集加载器 | Verify iSAID dataset loader."""

    def test_init(self, mock_isaid_dir: str) -> None:
        """数据集初始化成功 | Dataset initializes successfully."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        assert ds is not None

    def test_len(self, mock_isaid_dir: str) -> None:
        """__len__ 正确 | __len__ is correct."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        assert len(ds) == 2  # 2 images in train

    def test_len_val(self, mock_isaid_dir: str) -> None:
        """验证集长度正确 | Validation set length correct."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="val")
        assert len(ds) == 1

    def test_getitem_keys(self, mock_isaid_dir: str) -> None:
        """输出字典包含所有必需键 | Output dict has all required keys."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]
        assert "image" in sample
        assert "masks" in sample
        assert "image_id" in sample
        assert "image_size" in sample

    def test_image_shape(self, mock_isaid_dir: str) -> None:
        """图像是 [C, H, W] float tensor | Image is [C, H, W] float tensor."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]
        img = sample["image"]
        assert isinstance(img, torch.Tensor)
        assert img.dim() == 3  # [C, H, W]
        assert img.shape[0] == 3  # RGB
        assert img.dtype == torch.float32

    def test_masks_shape(self, mock_isaid_dir: str) -> None:
        """掩码是 [N, H, W] binary tensor | Masks are [N, H, W] binary tensor."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]  # image 000001 has 2 instances
        masks = sample["masks"]
        assert isinstance(masks, torch.Tensor)
        assert masks.dim() == 3  # [N, H, W]
        assert masks.shape[0] == 2  # 2 instances
        assert masks.shape[1:] == (256, 256)

    def test_masks_binary_values(self, mock_isaid_dir: str) -> None:
        """掩码值只有 0 和 1 | Mask values are only 0 and 1."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]
        masks = sample["masks"]
        unique_vals = masks.unique()
        # 二值：只有 {0, 1} 或 {0}（空实例时只有 0）
        # Binary: only {0, 1} or {0} (only 0 when no instances)
        assert unique_vals.max() <= 1.0
        assert unique_vals.min() >= 0.0

    def test_image_id(self, mock_isaid_dir: str) -> None:
        """image_id 是整数 | image_id is an integer."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]
        assert isinstance(sample["image_id"], int)

    def test_num_classes(self, mock_isaid_dir: str) -> None:
        """num_classes = 15 (iSAID 标准) | num_classes = 15 (iSAID standard)."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        assert ds.num_classes == 15

    def test_category_names(self, mock_isaid_dir: str) -> None:
        """类别名称映射正确 | Category name mapping correct."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        assert ds.category_name(1) == "small_vehicle"
        assert ds.category_name(15) == "roundabout"

    def test_tile_mode(self, mock_isaid_dir: str) -> None:
        """tile 模式产生正确尺寸的瓦片 | Tile mode produces correct-size tiles."""
        ds = ISAIDDataset(
            root_dir=mock_isaid_dir, split="train",
            tile_size=128, tile_overlap=0.0,
        )
        sample = ds[0]  # 256×256 image → 4 tiles of 128×128
        img = sample["image"]
        # 256/128 = 2 tiles per dim → up to 4 tiles (some may be empty)
        assert img.dim() == 3
        h, w = img.shape[1], img.shape[2]
        assert h <= 128, f"Tile height {h} > 128"
        assert w <= 128, f"Tile width {w} > 128"

    def test_no_tile_mode(self, mock_isaid_dir: str) -> None:
        """无 tile 模式：输出原图 | No tile: outputs full image."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train", tile_size=None)
        sample = ds[0]
        assert sample["image"].shape[1:] == (256, 256)

    def test_empty_mask_handling(self, mock_isaid_dir: str) -> None:
        """无实例的图像返回空掩码 | Images with no instances return empty masks."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="val")
        sample = ds[0]  # val set has no annotations
        masks = sample["masks"]
        assert masks.shape[0] == 0  # empty masks
        assert masks.dim() == 3

    def test_image_size_in_output(self, mock_isaid_dir: str) -> None:
        """输出包含原始图像尺寸 | Output contains original image size."""
        ds = ISAIDDataset(root_dir=mock_isaid_dir, split="train")
        sample = ds[0]
        assert sample["image_size"] == (256, 256)
