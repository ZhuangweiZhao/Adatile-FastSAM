"""Sparse importance prediction (Ada-SPM) module.

Adaptive Spatial Partition Module:
    - Multi-scale FPN fusion neck
    - Density head: S ∈ [0,1] per spatial location
    - Granularity head: T ∈ tile-size index per location
    - Optional transformer refinement (windowed self-attention)
    - Differentiable Gumbel-Softmax tile allocation

Variants:
    - AdaSPM: Standard (FPN + optional transformer + dual-head)
    - AdaSPMLite: Lightweight (no transformer, fewer channels)
    - AdaSPMFull: Maximum accuracy (transformer + larger hidden dim)
    - DensityOnlySPM: Ablation (density only, no granularity)
    - UniformImportance: Baseline (all ones)
"""

from adatile.registry import SPARSE
from adatile.sparse.fpn_fusion import MultiScaleFPNFusion, LightweightFPNFusion
from adatile.sparse.ada_spm import (
    AdaSPM,
    AdaSPMLite,
    AdaSPMFull,
    DensityOnlySPM,
    DensityHead,
    GranularityHead,
    SpatialTransformerRefine,
)
from adatile.sparse.base import UniformImportance


def build_sparse(name: str, **kwargs):
    """Factory: instantiate a registered sparse predictor by name.

    Available names:
        - "AdaSPM"          — standard implementation
        - "AdaSPMLite"      — lightweight variant
        - "AdaSPMFull"      — full variant with transformer
        - "DensityOnlySPM"  — ablation: density only
        - "UniformImportance" — baseline: uniform importance
        - "ada_spm"         — alias for AdaSPM

    Usage:
        model = build_sparse("AdaSPM", fusion_dim=256, num_tile_sizes=4)
    """
    if name == "ada_spm":
        name = "AdaSPM"
    return SPARSE.build(name, **kwargs)


__all__ = [
    "AdaSPM", "AdaSPMLite", "AdaSPMFull", "DensityOnlySPM", "UniformImportance",
    "MultiScaleFPNFusion", "LightweightFPNFusion",
    "DensityHead", "GranularityHead", "SpatialTransformerRefine",
    "build_sparse",
]
