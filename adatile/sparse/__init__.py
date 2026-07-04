"""
adatile.sparse — Spatial Sparsity Modules.

v3 naming (recommended):
    SparsePerceptionModule (SPM)  — Unified sparse perception module
    ImportanceHead                — Foreground density prediction head (~75K)
    TileRouter                    — Tile selection via importance map
    EdgeHead                      — Edge-aware head (ablation only)
    TinySPM                       — Ultra-lightweight lower bound
    ProtoCoeffPredictor           — Support prototype → proto mask coefficients

v2 naming (backward-compatible aliases):
    ForegroundDensityRouter = SparsePerceptionModule
    DensityHead = ImportanceHead
    TinyCNNRouter = TinySPM
"""

# ── v3 exports (recommended) ──
from adatile.sparse.spm import (
    SparsePerceptionModule,
    ImportanceHead,
    TileRouter,
    EdgeHead,
    TinySPM,
)

# ── v2 backward-compatible aliases ──
from adatile.sparse.spm import (
    ForegroundDensityRouter,
    DensityHead,
    TinyCNNRouter,
)

# ── Legacy spatial_router exports (for old code that imports directly) ──
from adatile.sparse.spatial_router import (
    DualStreamRouter,
)

# ── Coefficient predictor ──
from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor

__all__ = [
    # v3 names
    "SparsePerceptionModule",
    "ImportanceHead",
    "TileRouter",
    "EdgeHead",
    "TinySPM",
    # v2 backward-compatible aliases
    "ForegroundDensityRouter",
    "DensityHead",
    "TinyCNNRouter",
    # Legacy
    "DualStreamRouter",
    # Coefficient predictor
    "ProtoCoeffPredictor",
]
