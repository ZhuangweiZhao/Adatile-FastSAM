"""
分割数据集抽象基类 | Abstract base class for segmentation datasets.
=====================================================================

定义统一的数据样本格式，所有分割数据集都继承此类。
Defines unified data sample format; all segmentation datasets inherit from this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from adatile.logging import get_logger


class BaseSegDataset(Dataset, ABC):
    """
    分割数据集抽象基类 | Abstract base class for segmentation datasets.

    子类必须实现 | Subclasses MUST implement:
        - _load_image(index) → torch.Tensor  [C, H, W]
        - _load_masks(index) → torch.Tensor  [N, H, W] 二值实例掩码 | binary instance masks
        - _load_image_id(index) → int | str

    可选重写 | Optional override:
        - _apply_transforms(sample) → dict  — 数据增强 | data augmentation

    统一输出格式 | Unified output format:
        {
            "image": Tensor [C, H, W]      归一化图像 | Normalized image (float32, 0..1)
            "masks": Tensor [N, H, W]       二值实例掩码 | Binary instance masks
            "image_id": int                  图像标识符 | Image identifier
            "image_size": (H, W)             原始图像尺寸 | Original image size
        }
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms=None,
    ) -> None:
        """
        初始化数据集 | Initialize dataset.

        Args:
            root_dir:   数据集根目录 | Dataset root directory.
            split:      数据划分 ("train", "val", "test") | Data split.
            transforms: 数据增强变换（可选）| Optional data augmentation transforms.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.logger = get_logger(f"data.{split}")

        # 验证数据集目录存在 | Verify dataset directory exists
        if not self.root_dir.exists():
            self.logger.log_warn(
                "dataset/dir_missing",
                f"Dataset directory not found: {self.root_dir}. "
                f"Some operations may fail.",
            )

    @abstractmethod
    def _load_image(self, index: int) -> torch.Tensor:
        """
        加载并预处理图像 | Load and preprocess image.

        Args:
            index: 样本索引 | Sample index.

        Returns:
            torch.Tensor [C, H, W] float32, 值域 [0, 1] | Values in [0, 1].
        """
        ...

    @abstractmethod
    def _load_masks(self, index: int) -> torch.Tensor:
        """
        加载实例掩码 | Load instance masks.

        Args:
            index: 样本索引 | Sample index.

        Returns:
            torch.Tensor [N, H, W] float32, 二值 (0/1) | Binary (0/1).
            N 是实例数量。如果无实例，返回 shape [0, H, W] 的张量。
            N is number of instances. Returns [0, H, W] if no instances.
        """
        ...

    @abstractmethod
    def _load_image_id(self, index: int) -> int | str:
        """
        加载图像唯一标识符 | Load unique image identifier.

        Args:
            index: 样本索引 | Sample index.

        Returns:
            int | str: 图像 ID | Image identifier.
        """
        ...

    def _apply_transforms(self, sample: dict) -> dict:
        """
        应用数据增强（子类可重写）| Apply data augmentation (override in subclass).

        Args:
            sample: 原始样本字典 | Raw sample dict.

        Returns:
            dict: 增强后的样本 | Augmented sample.
        """
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __getitem__(self, index: int) -> dict:
        """
        获取样本 | Get sample.

        Returns:
            dict with keys: "image", "masks", "image_id", "image_size"
        """
        image = self._load_image(index)
        masks = self._load_masks(index)
        image_id = self._load_image_id(index)

        sample = {
            "image": image,
            "masks": masks,
            "image_id": image_id,
            "image_size": tuple(image.shape[1:]),  # (H, W)
        }

        return self._apply_transforms(sample)

    @abstractmethod
    def __len__(self) -> int:
        """数据集样本总数 | Total number of samples."""
        ...
