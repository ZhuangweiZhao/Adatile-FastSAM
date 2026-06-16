"""Sparse perception modules.

Active:
    - LightSPM: lightweight importance predictor (Stage B/C)
    - AdaSPM: full Ada-SPM with FPN + transformer (legacy pipeline)

Legacy (for train.py compatibility):
    - CAT modules: adapter + prompt bridge
    - UniformImportance: ablation baseline
"""

from adatile.registry import SPARSE
from adatile.backbone.fpn import MultiScaleFPNFusion, LightweightFPNFusion
from adatile.sparse.light_spm import LightSPM
from adatile.sparse.ada_spm import (
    AdaSPM, AdaSPMLite, AdaSPMFull, DensityOnlySPM,
    DensityHead, GranularityHead, SpatialTransformerRefine,
)
from adatile.adaptation import (
    CATModule, CATToken, CATMaskHead, PromptBridge,
    FPNAdapter, MultiScaleFPNAdapters, SpatialTransformerAdapter,
)
try:
    from adatile.sparse.base import UniformImportance
except ImportError:
    UniformImportance = None


def build_sparse(name: str, **kwargs):
    """Factory: instantiate a registered sparse predictor by name."""
    if name == "ada_spm":
        name = "AdaSPM"
    return SPARSE.build(name, **kwargs)


__all__ = [
    "LightSPM",
    "AdaSPM", "AdaSPMLite", "AdaSPMFull", "DensityOnlySPM", "UniformImportance",
    "MultiScaleFPNFusion", "LightweightFPNFusion",
    "DensityHead", "GranularityHead", "SpatialTransformerRefine",
    "CATModule", "CATToken", "CATMaskHead", "PromptBridge",
    "FPNAdapter", "MultiScaleFPNAdapters", "SpatialTransformerAdapter",
    "build_sparse",
]
