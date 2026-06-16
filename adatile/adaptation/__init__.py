"""Domain adaptation modules for AdaTile-FastSAM.

CAT (Conditional Adaptive Tuning): Lightweight adapters + prompt bridge
for few-shot domain adaptation with minimal trainable parameters.

Reference: CAT-SAM (Xiao et al., 2024)
"""

from adatile.adaptation.cat_adapter import (
    CATModule,
    CATToken,
    CATMaskHead,
    PromptBridge,
    FPNAdapter,
    MultiScaleFPNAdapters,
    SpatialTransformerAdapter,
)

__all__ = [
    "CATModule",
    "CATToken",
    "CATMaskHead",
    "PromptBridge",
    "FPNAdapter",
    "MultiScaleFPNAdapters",
    "SpatialTransformerAdapter",
]
