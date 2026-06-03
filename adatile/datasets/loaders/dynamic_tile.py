"""Dynamic tile data loader.

Loads images and their pre-cached tile variants.
Supports on-the-fly tile slicing when cache is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from adatile.core import TileInfo
from adatile.datasets.cache import TileCache


class DynamicTileDataLoader:
    """Wraps a standard DataLoader with tile-aware batching.

    During training:
        1. Load full images (standard DataLoader)
        2. Backbone extracts features
        3. Ada-SPM predicts importance
        4. Tokenizer references tile cache for fast region extraction

    This design preserves original images while enabling fast tile access.
    """

    def __init__(
        self,
        dataset: Dataset,
        tile_cache: Optional[TileCache] = None,
        tile_sizes: Optional[List[int]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.tile_cache = tile_cache
        self.tile_sizes = tile_sizes or [384, 768, 1536]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Tensor]:
        """Custom collate: stack images, collect annotations as lists."""
        images = torch.stack([
            torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 255.0
            for item in batch
        ])
        return {
            "images": images,
            "annotations": [item["annotations"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
            "image_infos": [item.get("image_info", {}) for item in batch],
        }

    def get_tiles(
        self,
        image_id: str,
        tile_infos: List[TileInfo],
        split: str = "train",
    ) -> Optional[List[Tuple[Tensor, TileInfo]]]:
        """Retrieve cached tiles for a list of TileInfo.

        Returns None for tiles not found in cache.
        """
        if self.tile_cache is None:
            return None

        tiles = []
        for info in tile_infos:
            try:
                tile, _ = self.tile_cache.load_tile(
                    image_id, info.tile_size,
                    info.x1, info.y1, info.x2, info.y2,
                    split,
                )
                tiles.append((tile, info))
            except FileNotFoundError:
                continue

        return tiles if tiles else None

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)
