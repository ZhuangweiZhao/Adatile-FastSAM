"""
DA-FRN — Defect-Aware Feature Rectification Network.
=====================================================
面向工业缺陷的基础视觉模型特征校正网络.
Feature rectification network for industrial defect segmentation.

将 Foundation Model (FastSAM) 的 SA-1B 特征空间
"翻译"为工业缺陷特征空间，不修改 Backbone 权重。
Translates SA-1B feature space into industrial defect feature space
without modifying backbone weights.

三个模块 | Three Modules:
    DCR — Defect-aware Channel Reweighting (通道注意力)
    FDE — Frequency-aware Defect Enhancement (频域增强)
    CDF — Cross-scale Defect Fusion (动态多尺度融合)

用法 | Usage::

    from adatile.rectify import DA_FRN

    frn = DA_FRN(p3_channels=960, p4_channels=1280)
    p3_rect, p4_rect = frn(p3_features, p4_features)
    # → 直接送入现有 Decoder | Feed directly to existing Decoder
"""

from adatile.rectify.dcr import DefectChannelReweighting as DCR
from adatile.rectify.fde import FreqDefectEnhance as FDE
from adatile.rectify.cdf import CrossScaleDefectFusion as CDF
from adatile.rectify.hdn import HeatmapDenoiser as HDN
from adatile.rectify.dafrn import DA_FRN

__all__ = ["DCR", "FDE", "CDF", "HDN", "DA_FRN"]
