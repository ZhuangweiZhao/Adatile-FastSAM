"""
adatile.sparse — Spatial Sparsity Modules (Paper B).

Exports:
    DensityHead              — Foreground density prediction head (~75K)
    EdgeHead                 — Edge-aware head (ablation only)
    ForegroundDensityRouter  — FDR mainline architecture
    DualStreamRouter         — Density + Edge fusion (ablation only)
    TinyCNNRouter            — Ultra-lightweight lower bound
    ProtoCoeffPredictor      — Support prototype → proto mask coefficients (NEW)
"""

from adatile.sparse.spatial_router import (
    DensityHead,
    DualStreamRouter,
    EdgeHead,
    ForegroundDensityRouter,
    TinyCNNRouter,
)
from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor

__all__ = [
    "DensityHead",
    "DualStreamRouter",
    "EdgeHead",
    "ForegroundDensityRouter",
    "TinyCNNRouter",
    "ProtoCoeffPredictor",
]
