"""
adatile.datasets — 数据集加载与预处理 | Data Loading & Preprocessing.
========================================================================

AdaTile 以实例分割为第一优先级。所有数据集优先支持实例分割模式。
AdaTile is instance-seg-first. All datasets prioritize instance segmentation mode.

导出 | Exports:
    BaseSegDataset               — 分割数据集抽象基类 | Segmentation dataset ABC
    ISAIDDataset                 — iSAID 航拍实例分割数据集 | iSAID aerial instance segmentation
    MassachusettsBuildingsDataset — 马萨诸塞州建筑二值分割数据集 | Mass Buildings binary segmentation
    ISAID_CATEGORIES             — iSAID 类别定义 | iSAID category definitions
"""

from adatile.datasets.base import BaseSegDataset
from adatile.datasets.isaid import ISAIDDataset, ISAID_CATEGORIES
from adatile.datasets.mass_buildings import MassachusettsBuildingsDataset

__all__ = [
    "BaseSegDataset",
    "ISAIDDataset",
    "MassachusettsBuildingsDataset",
    "ISAID_CATEGORIES",
]

from adatile.datasets.vaihingen_tiles import VaihingenTileDataset
from adatile.datasets.loveda_tiles import LoveDATileDataset
from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper
from adatile.datasets.p4_cache import P4Cache
