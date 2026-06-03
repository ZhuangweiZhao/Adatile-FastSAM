"""Few-shot data loader.

Implements episodic few-shot data loading with support/query split.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from adatile.datasets.base import FewShotSplit


class FewShotDataLoader:
    """Episodic few-shot data loader.

    Each batch contains:
        - support_images: [N_way * K_shot, 3, H, W]
        - support_masks: [N_way * K_shot, H, W]
        - query_images: [N_way * N_query, 3, H, W]
        - class_ids: [N_way]
    """

    def __init__(
        self,
        dataset: Dataset,
        fewshot_split: FewShotSplit,
        n_way: int = 3,
        k_shot: int = 5,
        n_query: int = 5,
        num_episodes: int = 1000,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.split = fewshot_split
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes

        self._loader = DataLoader(
            dataset,
            batch_sampler=None,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_episode,
        )

    def _collate_episode(self, batch: List[Dict]) -> Dict[str, Tensor]:
        """Collate a support + query episode."""
        # Support
        support = [item for item in batch if item.get("is_support")]
        query = [item for item in batch if not item.get("is_support")]

        support_images = torch.stack([
            torch.from_numpy(s["image"]).permute(2, 0, 1).float() / 255.0
            for s in support
        ])
        support_masks = torch.stack([
            torch.from_numpy(s["mask"]).float()
            for s in support
        ])
        query_images = torch.stack([
            torch.from_numpy(q["image"]).permute(2, 0, 1).float() / 255.0
            for q in query
        ])

        class_ids = list(set(
            s["category_id"] for s in support
        ))

        return {
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_images,
            "class_ids": class_ids,
        }

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return self.num_episodes
