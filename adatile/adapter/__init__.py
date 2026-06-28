"""
adatile.adapter — 轻量特征适配器 | Lightweight Feature Adapters
================================================================

CAT-SAM 思路迁移到卷积网络：在冻结的 FastSAM Encoder 中插入轻量 ConvAdapter，
实现遥感域适配而不破坏预训练特征。
CAT-SAM concept adapted for ConvNets: insert lightweight ConvAdapters into
frozen FastSAM Encoder for remote sensing domain adaptation.

模块 | Modules:
    ConvAdapter       — 通道注意力适配器 | Channel attention adapter
    MultiScaleAdapter — P3/P4/P8 多尺度适配 | Multi-scale adaptation
"""

from adatile.adapter.conv_adapter import ConvAdapter, MultiScaleAdapter

__all__ = ["ConvAdapter", "MultiScaleAdapter"]
