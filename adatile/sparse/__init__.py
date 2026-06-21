"""
adatile.sparse ? Spatial Sparsity Modules (Paper B).

Exports:
    DensityHead              ? Foreground density prediction head (~75K)
    EdgeHead                 ? Edge-aware head (ablation only)
    ForegroundDensityRouter  ? FDR mainline architecture
    DualStreamRouter         ? Density + Edge fusion (ablation only)
    TinyCNNRouter            ? Ultra-lightweight lower bound
"""

from adatile.sparse.spatial_router import (
    DensityHead,
    DualStreamRouter,
    EdgeHead,
    ForegroundDensityRouter,
    TinyCNNRouter,
)

__all__ = [
    "DensityHead",
    "DualStreamRouter",
    "EdgeHead",
    "ForegroundDensityRouter",
    "TinyCNNRouter",
]
