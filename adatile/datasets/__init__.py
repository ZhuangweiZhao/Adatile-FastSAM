"""Dataset system for AdaTile-FastSAM.

Supports:
    - COCO-style instance segmentation
    - iSAID remote sensing (15 classes, up to 4000×4000)
    - LoveDA land-cover (7 classes, 1024×1024)
    - Few-shot split loading with episodic sampling
    - Adaptive tile cache (384, 768, 1536, 3072)
    - Density map generation and sparse statistics
    - Augmentation pipeline (albumentations)
    - Batch collation functions
    - Cache and metadata management
"""

from adatile.datasets.base import BaseDataset, FewShotSplit
from adatile.datasets.coco import CocoDataset, ISAIDDataset, LoveDADataset
from adatile.datasets.cache.tile_cache import TileCache
from adatile.datasets.cache import TileCache as TileCacheAlias
from adatile.datasets.loaders.dynamic_tile import DynamicTileDataLoader
from adatile.datasets.loaders.fewshot import FewShotDataLoader
from adatile.datasets.loaders import DynamicTileDataLoader as DTL, FewShotDataLoader as FSL
from adatile.datasets.samplers.fewshot_sampler import (
    FewShotEpisodicSampler,
    SparseSampler,
)
from adatile.datasets.transforms import (
    build_train_pipeline,
    build_val_pipeline,
    build_tile_pipeline,
    InstanceSegTransform,
)
from adatile.datasets.collate import (
    coco_collate,
    coco_collate_with_masks,
    tile_collate,
    fewshot_collate,
    pad_collate,
)
from adatile.datasets.metadata import (
    MetadataManager,
    DatasetMeta,
    ImageMeta,
    TileMeta,
)
from adatile.datasets.manager import CacheManager

__all__ = [
    # Core
    "BaseDataset",
    "CocoDataset",
    "ISAIDDataset",
    "LoveDADataset",
    # Few-shot
    "FewShotSplit",
    "FewShotEpisodicSampler",
    "SparseSampler",
    "FewShotDataLoader",
    # Cache
    "TileCache",
    "DynamicTileDataLoader",
    # Augmentation
    "build_train_pipeline",
    "build_val_pipeline",
    "build_tile_pipeline",
    "InstanceSegTransform",
    # Collate
    "coco_collate",
    "coco_collate_with_masks",
    "tile_collate",
    "fewshot_collate",
    "pad_collate",
    # Metadata
    "MetadataManager",
    "DatasetMeta",
    "ImageMeta",
    "TileMeta",
    "CacheManager",
]
