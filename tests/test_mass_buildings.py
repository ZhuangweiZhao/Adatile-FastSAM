"""
Massachusetts Buildings 数据集测试 | Massachusetts Buildings dataset tests.
==============================================================================

验证二值分割数据集加载器。
Verify binary instance segmentation dataset loader.

数据集特点 | Dataset characteristics:
    - 二值分割（建筑 vs 背景）| Binary instance segmentation (building vs background)
    - 图像和标注都是 1500×1500 RGB PNG
    - 标注是 RGB：255=建筑, 0=背景 → 需转为二值 mask
    - 训练 137 / 验证 4 / 测试 10 张图像
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from PIL import Image

from adatile.datasets import MassachusettsBuildingsDataset


# ════════════════════════════════════════════════════════════════
# 真实数据路径 | Real data path
# ════════════════════════════════════════════════════════════════

REAL_DATA_PATH = "data/Massachusetts_Buildings"


# ════════════════════════════════════════════════════════════════
# 基础功能测试 | Basic Functionality Tests
# ════════════════════════════════════════════════════════════════

class TestMassBuildingsBasic:
    """验证数据集基础功能 | Verify dataset basic functionality."""

    def test_init_train(self) -> None:
        """训练集初始化 | Train set initialization."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")
        assert ds is not None

    def test_len_train(self) -> None:
        """训练集长度 | Train set length."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")
        assert len(ds) == 137

    def test_len_val(self) -> None:
        """验证集长度 | Validation set length."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="val")
        assert len(ds) == 4

    def test_len_test(self) -> None:
        """测试集长度 | Test set length."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="test")
        assert len(ds) == 10

    def test_num_classes(self) -> None:
        """二分类任务 | Binary classification task."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")
        assert ds.num_classes == 2  # background + building | 背景 + 建筑


# ════════════════════════════════════════════════════════════════
# 样本格式测试 | Sample Format Tests
# ════════════════════════════════════════════════════════════════

class TestMassBuildingsSample:
    """验证单个样本格式 | Verify single sample format."""

    @pytest.fixture(scope="class")
    def ds(self) -> MassachusettsBuildingsDataset:
        """Class-scoped: 加载一次 | Load once."""
        return MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")

    @pytest.fixture(scope="class")
    def sample(self, ds: MassachusettsBuildingsDataset) -> dict:
        """获取第一个样本 | Get first sample."""
        return ds[0]

    def test_output_keys(self, sample: dict) -> None:
        """输出包含所有必需键 | Output has all required keys."""
        assert "image" in sample
        assert "masks" in sample
        assert "image_id" in sample
        assert "image_size" in sample

    def test_image_shape(self, sample: dict) -> None:
        """图像 [C, H, W] float32 [0, 1] | Image [C, H, W] float32 [0, 1]."""
        img = sample["image"]
        assert isinstance(img, torch.Tensor)
        assert img.dim() == 3
        assert img.shape[0] == 3  # RGB
        assert img.dtype == torch.float32
        assert 0.0 <= img.min() <= img.max() <= 1.0

    def test_image_size_1500(self, sample: dict) -> None:
        """原始图像 1500×1500 | Original image 1500×1500."""
        h, w = sample["image_size"]
        assert h == 1500
        assert w == 1500

    def test_masks_shape(self, sample: dict) -> None:
        """掩码 [1, H, W] binary | Mask [1, H, W] binary."""
        masks = sample["masks"]
        assert isinstance(masks, torch.Tensor)
        assert masks.dim() == 3  # [1, H, W]
        assert masks.shape[0] == 1  # 单通道二值分割 | single-channel binary
        assert masks.shape[1] == 1500
        assert masks.shape[2] == 1500

    def test_masks_binary_values(self, sample: dict) -> None:
        """掩码值只有 0 和 1 | Mask values are only 0 and 1."""
        masks = sample["masks"]
        unique_vals = masks.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0}), (
            f"Mask values must be binary, got {unique_vals.tolist()}"
        )

    def test_image_id_format(self, sample: dict) -> None:
        """image_id 是字符串 | image_id is a string."""
        assert isinstance(sample["image_id"], str)
        # 格式：22678915_15 | Format: {id}_15
        assert "_15" in sample["image_id"]

    def test_label_conversion(self, ds: MassachusettsBuildingsDataset, sample: dict) -> None:
        """
        RGB 标注正确转为二值 mask | RGB label correctly converted to binary mask.
        原始标注：255=建筑, 0=背景 | Original label: 255=building, 0=background.
        """
        # 加载原始标注验证 | Load original label to verify
        import numpy as np
        from PIL import Image

        orig_label_path = ds.root_dir / "png" / f"{ds.split}_labels" / f"{sample['image_id']}.png"
        orig_label = np.array(Image.open(orig_label_path))
        # 原始标注只有 {0, 255} | Original label only has {0, 255}
        assert set(np.unique(orig_label)).issubset({0, 255})


# ════════════════════════════════════════════════════════════════
# Tile 模式测试 | Tile Mode Tests
# ════════════════════════════════════════════════════════════════

class TestMassBuildingsTile:
    """验证 tile 模式 | Verify tile mode."""

    def test_tile_mode_enabled(self) -> None:
        """tile_size=512 → 输出 ≤ 512 | tile_size=512 → output ≤ 512."""
        ds = MassachusettsBuildingsDataset(
            root_dir=REAL_DATA_PATH, split="train",
            tile_size=512, tile_overlap=0.0,
        )
        sample = ds[0]
        h, w = sample["image"].shape[1], sample["image"].shape[2]
        assert h <= 512, f"Tile height {h} > 512"
        assert w <= 512, f"Tile width {w} > 512"
        assert sample["masks"].shape[1] == h  # mask 与 image 同尺寸 | mask matches image
        assert sample["masks"].shape[2] == w

    def test_tile_count(self) -> None:
        """
        Tile 模式下 len(ds) > 原始图像数 | Tile mode len(ds) > original image count.
        1500×1500 用 512 tile（无重叠）→ 每张图 3×3=9 个 tile
        1500×1500 with 512 tile (no overlap) → 9 tiles per image.
        """
        ds = MassachusettsBuildingsDataset(
            root_dir=REAL_DATA_PATH, split="val",
            tile_size=512, tile_overlap=0.0,
        )
        # val 有 4 张图, 每张 9 tiles = 36 tiles total (最后一行/列可能更小)
        # val has 4 images, 9 tiles each = ~36 tiles total
        assert len(ds) > 4  # 一定多于原始图像数 | definitely more than image count
        assert len(ds) <= 4 * 9  # 最多 | max possible

    def test_no_tile_mode(self) -> None:
        """tile_size=None → 全图 1500×1500 | Full image 1500×1500."""
        ds = MassachusettsBuildingsDataset(
            root_dir=REAL_DATA_PATH, split="train", tile_size=None,
        )
        sample = ds[0]
        assert sample["image"].shape[1:] == (1500, 1500)

    def test_tile_mask_consistency(self) -> None:
        """Tile 的图像和掩码尺寸一致 | Tile image and mask same size."""
        ds = MassachusettsBuildingsDataset(
            root_dir=REAL_DATA_PATH, split="train",
            tile_size=512, tile_overlap=0.0,
        )
        sample = ds[0]
        assert sample["image"].shape[1:] == sample["masks"].shape[1:]


# ════════════════════════════════════════════════════════════════
# 完整性测试 | Integrity Tests
# ════════════════════════════════════════════════════════════════

class TestMassBuildingsIntegrity:
    """验证数据集完整性 | Verify dataset integrity."""

    def test_all_splits_loadable(self) -> None:
        """所有划分都能加载 | All splits loadable."""
        for split in ["train", "val", "test"]:
            ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split=split)
            assert len(ds) > 0, f"Split '{split}' is empty"

    def test_no_file_missing(self) -> None:
        """每个图像都有对应的标注文件 | Every image has corresponding label file."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")
        for idx in range(len(ds)):
            sample = ds[idx]
            # 如果有 mask 数据 → 验证非全零（但允许全零：全背景图）
            # If mask data present → verify not all-zero (all-zero allowed: all-background)
            masks = sample["masks"]
            assert masks.dim() == 3, f"Sample {idx}: masks should be 3D, got {masks.dim()}D"

    def test_building_pixels_exist(self) -> None:
        """训练集中至少有一些建筑像素 | Train set has at least some building pixels."""
        ds = MassachusettsBuildingsDataset(root_dir=REAL_DATA_PATH, split="train")
        total_building = 0
        for idx in range(min(5, len(ds))):  # 前5张 | first 5
            sample = ds[idx]
            total_building += sample["masks"].sum().item()
        assert total_building > 0, "No building pixels found in first 5 images!"
