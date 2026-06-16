"""Dataset system for AdaTile-FastSAM.

Active datasets:
    - BSSegDataset: simple image-mask pairs (binary segmentation)
    - CocoDataset: COCO-style instance segmentation
    - ISAIDDataset: iSAID remote sensing

Data loading:
    - DynamicTileDataLoader: tile-aware batch loading
"""

from adatile.datasets.base import BaseDataset, FewShotSplit
from adatile.datasets.coco import CocoDataset, ISAIDDataset
from adatile.datasets.bsseg import BSSegDataset
from adatile.datasets.loaders.dynamic_tile import DynamicTileDataLoader

__all__ = [
    "BaseDataset",
    "FewShotSplit",
    "CocoDataset",
    "ISAIDDataset",
    "BSSegDataset",
    "DynamicTileDataLoader",
]
