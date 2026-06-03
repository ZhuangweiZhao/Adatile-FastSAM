"""Evaluation metrics for instance segmentation.

Includes:
    - COCO evaluator (bbox + mask AP)
    - Few-shot evaluator (mIoU, FB-IoU)
    - Sparse efficiency metrics
"""

from .metrics import (
    COCOEvaluator,
    FewShotEvaluator,
    SparseEfficiencyMetrics,
)

__all__ = [
    "COCOEvaluator",
    "FewShotEvaluator",
    "SparseEfficiencyMetrics",
]
