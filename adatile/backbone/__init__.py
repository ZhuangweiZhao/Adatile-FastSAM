"""
adatile.backbone — 特征提取骨架 | Feature extraction backbone.
===============================================================

基于 thirdLibrary/FastSAM 的 FastSAM 骨干网络，
通过前向钩子提取多尺度特征图 (P4, P8)。

FastSAM backbone based on thirdLibrary/FastSAM,
extracts multi-scale feature maps via forward hooks (P4, P8).

导出 | Exports:
    FastSAMBackbone  — 带钩子的 FastSAM 特征提取器 | FastSAM feature extractor with hooks
    build_backbone   — 工厂函数，按名称构建 | Factory function, build by name
"""

from adatile.backbone.fastsam_backbone import FastSAMBackbone, build_backbone
from adatile.backbone.feature_adapter import ResidualAdapter, MultiScaleAdapter
from adatile.backbone.catsam_adapter import (
    FFTPromptExtractor,
    CrossPromptGenerator,
    CATSAMAFewShotDecoder,
)

__all__ = [
    "FastSAMBackbone",
    "build_backbone",
    "ResidualAdapter",
    "MultiScaleAdapter",
    "FFTPromptExtractor",
    "CrossPromptGenerator",
    "CATSAMAFewShotDecoder",
]
