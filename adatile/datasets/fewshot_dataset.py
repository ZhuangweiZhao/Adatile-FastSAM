#!/usr/bin/env python3
"""
Few-Shot Instance Segmentation Episode Dataset | 少样本实例分割 Episode 数据集.

基于 ISAIDTileWrapper 提供 episodic training: 每次 __getitem__ 返回:
  - support_images: [K, 3, H, W]
  - support_masks:  [K, H, W]  (binary, per-class)
  - query_image:    [1, 3, H, W]
  - query_mask:     [H, W]

用法 | Usage::
    ds = FewShotEpisodeDataset(tile_dataset, fold=0, shot=1, split="train")
    batch = ds[0]  # {support_imgs, support_masks, query_img, query_mask, class_id}
"""

import numpy as np
import torch
from typing import Optional

from adatile.datasets.fewshot_split import get_novel_classes
from adatile.utils.label_mapping import ISAID_CATEGORIES


class FewShotEpisodeDataset:
    """
    Episode-based Few-Shot Dataset | 回合式少样本数据集.

    每次采样一个 Novel 类 → K 张 support + 1 张 query → 组成一个 episode.
    Samples one Novel class → K support images + 1 query image → one episode.

    Parameters
    ----------
    tile_dataset : ISAIDTileWrapper
        预包装的 tile 数据集 (train 或 val).
    fold : int
        Fold ID (0/1/2). Novel 类来自该 Fold, Base 类数据不用于 Novel 训练.
    shot : int
        Support 图像数量 (K-shot).
    split : str
        "train" (用 train 数据做 support 和 query) 或 "val" (用 val 数据).
    episodes_per_epoch : int
        每 epoch 采样的 episode 数量.
    seed : int
        随机种子.
    """

    def __init__(
        self,
        tile_dataset,
        fold: int = 0,
        shot: int = 1,
        split: str = "train",
        episodes_per_epoch: int = 200,
        seed: int = 42,
        crop_support: bool = True,
        crop_margin: float = 0.2,
        novel_classes: Optional[list[int]] = None,
        category_names: Optional[dict] = None,
    ):
        self.ds = tile_dataset
        self.fold = fold
        self.shot = shot
        self.split = split
        self.episodes_per_epoch = episodes_per_epoch
        self.crop_support = crop_support
        self.crop_margin = crop_margin  # bbox扩张比例 | bbox expansion ratio

        # 如果外部传入了 novel_classes，直接使用；否则从 fewshot_split 获取
        # If novel_classes provided externally, use them; otherwise get from fewshot_split
        if novel_classes is not None:
            self.novel_classes = list(novel_classes)
        else:
            self.novel_classes = get_novel_classes(fold)

        self.valid_novel = []
        for cid in self.novel_classes:
            n_images = len(tile_dataset.class_to_images(cid))
            if n_images >= shot:
                self.valid_novel.append(cid)

        self.rng = np.random.RandomState(seed)
        self.class_names = category_names if category_names is not None else ISAID_CATEGORIES

        if not self.valid_novel:
            raise ValueError(f"Fold {fold}: no Novel classes with >= {shot} images!")

        print(f"[FewShotEpisodeDataset] fold={fold}, shot={shot}, split={split}")
        print(f"  Novel: {[self.class_names.get(c, str(c)) for c in self.valid_novel]}")
        print(f"  Crop support: {crop_support} (margin={crop_margin})")

    def _roi_crop(self, img_tensor, mask_tensor):
        """ROI crop around mask with margin. | 围绕 GT mask 做 ROI 裁剪."""
        mask_np = mask_tensor.numpy()
        ys, xs = np.where(mask_np > 0)
        if len(ys) < 4:
            return img_tensor, mask_tensor  # too small, return as-is

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        h, w = y2 - y1, x2 - x1
        # Expand by margin
        dy, dx = int(h * self.crop_margin), int(w * self.crop_margin)
        y1 = max(0, y1 - dy)
        y2 = min(mask_tensor.shape[0], y2 + dy)
        x1 = max(0, x1 - dx)
        x2 = min(mask_tensor.shape[1], x2 + dx)

        # Crop + resize back to tile_size
        cropped_img = img_tensor[:, y1:y2, x1:x2]
        cropped_mask = mask_tensor[y1:y2, x1:x2]
        tile_size = img_tensor.shape[1]  # 896
        cropped_img = torch.nn.functional.interpolate(
            cropped_img.unsqueeze(0), size=(tile_size, tile_size),
            mode="bilinear", align_corners=False).squeeze(0)
        cropped_mask = torch.nn.functional.interpolate(
            cropped_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(tile_size, tile_size), mode="nearest").squeeze(0).squeeze(0).bool().float()
        return cropped_img, cropped_mask

    def __len__(self):
        return self.episodes_per_epoch

    def sample_episode(self):
        """
        .. deprecated::
            训练循环 (train_and_evaluate) 直接采样 indices 并通过 train_episode()
            构建 episode，不使用此方法。此方法保留仅用于兼容性和独立测试。
            The training loop directly samples indices and constructs episodes
            via train_episode(). This method is retained only for compatibility
            and standalone testing.

        采样一个 episode | Sample one episode.

        :return: dict with support_imgs, support_masks, query_img, query_mask, class_id
        """
        import warnings
        warnings.warn(
            "FewShotEpisodeDataset.sample_episode() is deprecated. "
            "The training loop samples episodes directly via train_episode(). "
            "This method is not used in standard training flow.",
            DeprecationWarning, stacklevel=2,
        )
        # 随机选一个 Novel 类 | Random Novel class
        cls_id = int(self.rng.choice(self.valid_novel))
        candidates = self.ds.class_to_images(cls_id)
        if len(candidates) < self.shot + 1:
            # fallback: 不够就全用 | not enough → use all
            indices = self.rng.choice(candidates, min(len(candidates), self.shot + 1), replace=False)
            support_idxs = indices[:self.shot] if len(indices) > self.shot else indices
            query_idx = int(indices[-1]) if len(indices) > self.shot else int(indices[0])
        else:
            indices = self.rng.choice(candidates, self.shot + 1, replace=False)
            support_idxs = indices[:self.shot]
            query_idx = int(indices[self.shot])

        # 加载 support (可选 ROI crop) | Load support (optional ROI crop)
        s_imgs, s_masks = [], []
        for si in support_idxs:
            img = self.ds.load_image(int(si))
            mask = self.ds.render_class_mask(int(si), cls_id)
            if self.crop_support and mask.sum() > 64:  # at least 64 FG pixels
                img, mask = self._roi_crop(img, mask)
            s_imgs.append(img)
            s_masks.append(mask)
        support_imgs = torch.stack(s_imgs)
        support_masks = torch.stack(s_masks)

        # 加载 query | Load query
        query_img = self.ds.load_image(int(query_idx)).unsqueeze(0)
        query_mask = self.ds.render_class_mask(int(query_idx), cls_id)

        return {
            "support_imgs": support_imgs,     # [K, 3, H, W]
            "support_masks": support_masks,   # [K, H, W]
            "query_img": query_img,           # [1, 3, H, W]
            "query_mask": query_mask,         # [H, W]
            "class_id": cls_id,
        }

    def __getitem__(self, idx: int):
        """采样一个 episode (忽略 idx, 随机采样) | Sample episode (idx ignored, random)."""
        return self.sample_episode()

    @property
    def class_to_images(self):
        """兼容现有代码: class_id → tile indices | Compat with existing code."""
        return self.ds.class_to_images

    def load_image(self, tile_idx: int) -> torch.Tensor:
        """兼容现有代码 | Compat with existing code."""
        return self.ds.load_image(tile_idx)

    def render_class_mask(self, tile_idx: int, class_id: int) -> torch.Tensor:
        """兼容现有代码 | Compat with existing code."""
        return self.ds.render_class_mask(tile_idx, class_id)
