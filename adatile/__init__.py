"""
AdaTile-FastSAM: Adaptive Sparse Tiling Framework
for High-Resolution Few-Shot Instance Segmentation.

Core Components:
- Adaptive Spatial Partition Module (Ada-SPM)
- Dynamic Token Router v2 (DTR-v2)
- Prototype-guided Segmentation
- Sparse Token Allocation
"""

__version__ = "0.1.0"
__author__ = ""

from adatile.registry import BACKBONE, SPARSE, TOKENIZER, ROUTER, DECODER, PROTOTYPE
from adatile.config import Config, get_default_config
