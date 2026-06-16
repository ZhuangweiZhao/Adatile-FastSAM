"""Baseline sparse importance predictor (uniform) + backward-compat re-exports.

The main Ada-SPM implementations are in ada_spm.py.
This file keeps the uniform baseline and re-exports for compatibility.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from adatile.core import SparseImportancePredictor, SparsePrediction
from adatile.registry import SPARSE

# Re-export main implementations (restored from legacy/)
from adatile.sparse.ada_spm import (
    AdaSPM, AdaSPMLite, AdaSPMFull, DensityOnlySPM,
)


@SPARSE.register()
class UniformImportance(SparseImportancePredictor):
    """Uniform importance baseline — all regions treated equally.

    Used for ablation studies: comparing adaptive vs. uniform tiling.
    Outputs an all-ones map of the same spatial size as the finest feature.
    """

    def forward(self, features: Dict[str, Tensor]) -> SparsePrediction:
        finest_key = sorted(features.keys())[0]
        f = features[finest_key]
        importance = torch.ones(
            f.shape[0], 1, f.shape[2], f.shape[3],
            device=f.device, dtype=f.dtype,
        )
        return SparsePrediction(importance=importance, density=importance)

    def compute_loss(
        self,
        pred_importance: Tensor,
        target_density: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.tensor(0.0, device=pred_importance.device)
