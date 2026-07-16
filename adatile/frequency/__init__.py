"""
adatile.frequency — 频域增强模块 | Frequency-Domain Enhancement Modules.
=======================================================================

基于 DCT 的即插即用频域模块, 增强 FastSAM 对工业缺陷的多频段感知能力。
DCT-based plug-and-play modules for multi-spectral industrial defect perception.

模块 | Modules:
    DCTSpectralAttention   — 多频段通道注意力 (FcaNet 风格)
    MultiScaleSpectralAttention — P2/P3/P4 多尺度频域注意力
    FrequencyGuidedFusion  — 频谱能量驱动的动态融合权重
    dct_spectral_loss      — DCT 频谱一致性损失
    spectral_combined_loss — CE + Dice + Spectral 组合损失
"""

from adatile.frequency.spectral import (
    DCTSpectralAttention,
    MultiScaleSpectralAttention,
    FrequencyGuidedFusion,
    dct_spectral_loss,
    spectral_combined_loss,
)

__all__ = [
    "DCTSpectralAttention",
    "MultiScaleSpectralAttention",
    "FrequencyGuidedFusion",
    "dct_spectral_loss",
    "spectral_combined_loss",
]
