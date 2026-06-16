"""Dynamic tile tokenizer module.

Partitions high-resolution images into variable-size tiles
based on importance predictions from Ada-SPM.

Components:
    - TilePlanner: importance-guided tile allocation (cell-based + quadtree)
    - TokenGenerator: patch extraction, conv stem, 2D positional encoding
"""

from adatile.registry import TOKENIZER

from adatile.tokenizer.base import DynamicTileTokenizerImpl
from adatile.tokenizer.tile_planner import (
    TilePlanner, TilePlan, TileSpec, PlannerStats,
)
from adatile.tokenizer.token_generator import (
    TokenGenerator, PatchEmbed, PosEmbed2D,
)


def build_tokenizer(name: str, **kwargs):
    """Factory: instantiate a registered tokenizer by name."""
    _aliases = {
        "dynamic_tile": "DynamicTileTokenizerImpl",
    }
    name = _aliases.get(name, name)
    return TOKENIZER.build(name, **kwargs)


__all__ = [
    "DynamicTileTokenizerImpl",
    "TilePlanner", "TilePlan", "TileSpec", "PlannerStats",
    "TokenGenerator", "PatchEmbed", "PosEmbed2D",
    "build_tokenizer",
]
