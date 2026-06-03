"""Dynamic tile tokenizer module.

Partitions high-resolution images into variable-size tiles
based on importance predictions from Ada-SPM.

Components:
    - TilePlanner: importance-guided tile allocation (cell-based + quadtree)
    - TokenGenerator: patch extraction, conv stem, 2D positional encoding
    - GlobalThumbnailBranch: downsampled full-image context branch
    - TileMerger: overlap-aware merging with NMS dedup and soft blending
"""

from adatile.registry import TOKENIZER

from adatile.tokenizer.base import (
    DynamicTileTokenizerImpl,
    UniformTileTokenizer,
)
from adatile.tokenizer.tile_planner import (
    TilePlanner, TilePlan, TileSpec, PlannerStats,
)
from adatile.tokenizer.token_generator import (
    TokenGenerator, PatchEmbed, PosEmbed2D,
)
from adatile.tokenizer.global_branch import (
    GlobalThumbnailBranch, TileMerger,
    gaussian_blend_kernel, soft_merge_overlap,
)


def build_tokenizer(name: str, **kwargs):
    """Factory: instantiate a registered tokenizer by name."""
    _aliases = {
        "dynamic_tile": "DynamicTileTokenizerImpl",
        "uniform_tile": "UniformTileTokenizer",
    }
    name = _aliases.get(name, name)
    return TOKENIZER.build(name, **kwargs)


__all__ = [
    "DynamicTileTokenizerImpl", "UniformTileTokenizer",
    "TilePlanner", "TilePlan", "TileSpec", "PlannerStats",
    "TokenGenerator", "PatchEmbed", "PosEmbed2D",
    "GlobalThumbnailBranch", "TileMerger",
    "gaussian_blend_kernel", "soft_merge_overlap",
    "build_tokenizer",
]
