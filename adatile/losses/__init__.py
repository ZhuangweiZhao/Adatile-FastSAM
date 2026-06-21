"""
adatile.losses — 损失函数 | Loss Functions.
=============================================

Paper A & B 共享的损失函数 | Loss functions shared by Paper A and Paper B.

导出 | Exports:
    FocalLoss      — Focal loss (γ adjustable), for multi-class segmentation
    DiceLoss       — Per-foreground-class Dice loss
    CombinedLoss   — Focal + Dice combined loss
"""

from adatile.losses.seg_losses import FocalLoss, DiceLoss, CombinedLoss

__all__ = ["FocalLoss", "DiceLoss", "CombinedLoss"]
