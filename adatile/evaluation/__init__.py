"""Evaluation metrics for instance segmentation."""

from .metrics import COCOEvaluator
from .sparse_eval import sparse_eval, split_support_query

__all__ = ["COCOEvaluator", "sparse_eval", "split_support_query"]
