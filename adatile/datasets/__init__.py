"""
adatile.datasets — 数据集加载与预处理 | Data Loading & Preprocessing.
========================================================================

导出 | Exports:
    BaseSegDataset               — 分割数据集抽象基类 | Segmentation dataset ABC
    ISAIDDataset                 — iSAID 航拍实例分割数据集 | iSAID aerial instance segmentation
    MassachusettsBuildingsDataset — 马萨诸塞州建筑语义分割数据集 | Mass Buildings semantic segmentation
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
