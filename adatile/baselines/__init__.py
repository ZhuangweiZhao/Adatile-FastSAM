"""
经典 CNN 分割基线 | Classic CNN segmentation baselines.

论文对比实验用 | For paper baseline comparisons:
    - UNet (vanilla, from scratch)
    - DeepLabV3+ (via segmentation_models_pytorch, 训练脚本内延迟导入 |
      lazily imported in the training script)
"""

from adatile.baselines.unet import UNet

__all__ = ["UNet"]
