"""Full segmentation pipeline module.

Combines backbone, sparse, tokenizer, routing, decoder, and prototype
into a complete instance segmentation model.

Loss functions and TrainingLoss are defined in base.py and imported
directly where needed (tools/train.py, adatile/engine/trainer.py).
"""

from adatile.segmentation.base import (
    AdaTileFastSAMPipeline,
    DiceLoss,
    FocalLoss,
    SegmentationLoss,
    TrainingLoss,
)

__all__ = [
    "AdaTileFastSAMPipeline",
    "DiceLoss",
    "FocalLoss",
    "SegmentationLoss",
    "TrainingLoss",
]
