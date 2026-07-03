"""
adatile.decoder — 分割掩码解码器 | Segmentation mask decoder.
===============================================================

导出 | Exports:
    LinearProbe  — 1×1 Conv 线性探针 (E002) | E002: P4 only linear probe
    FusionProbe  — P4+P8 融合探针 (E003) | E003: P4+P8 fusion probe
    AdaptiveSparseDecoder — 统一自适应稀疏解码器 (NEW) | Unified adaptive sparse decoder
    ProtoOnlyDecoder — 纯 Proto 系数解码器 (NEW, ablation) | Proto-only decoder
"""

from adatile.decoder.linear_probe import LinearProbe
from adatile.decoder.fusion_probe import FusionProbe
from adatile.decoder.light_decoder import LightDecoder, LightDecoderP3, LightDecoderP3P4
from adatile.decoder.adaptive_sparse_decoder import (
    AdaptiveSparseDecoder,
    ProtoOnlyDecoder,
)

__all__ = [
    "LinearProbe",
    "FusionProbe",
    "LightDecoder",
    "LightDecoderP3",
    "LightDecoderP3P4",
    "AdaptiveSparseDecoder",
    "ProtoOnlyDecoder",
]
