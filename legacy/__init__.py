"""Legacy modules — NOT used by active Stage A/B/C pipeline.

These modules were part of the original COCO/iSAID training pipeline
(train.py + config-based). They are kept for reference but the active
pipeline uses:

  FastSAMHookBackbone → LightDecoder + LightSPM
  UnifiedLoss (tools/train_as_fastsam.py)

If you need the old pipeline (FPN + transformer + tokenizer + router):
  1. pip install timm  (dependency for TimmBackbone)
  2. Move files back to their original locations
"""

# This file intentionally left empty — legacy modules are standalone.
