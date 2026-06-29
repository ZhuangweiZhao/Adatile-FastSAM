"""
Feature Adapter — 将 Detection Backbone 特征适配到 Few-Shot Correspondence.
================================================================================
Feature Adapter: adapt detection backbone features for few-shot correspondence.

FastSAM 的 P3/P4 特征是为 YOLO detection + SAM prompt decoder 训练的,
不是为 few-shot support-query matching 训练的。
FastSAM's P3/P4 features are trained for YOLO detection + SAM prompt decoder,
NOT for few-shot support-query matching.

本模块提供轻量级残差适配器, 插入到 P3/P4 特征之后, 通过少量可训练参数
将 detection-oriented features 重定向到 few-shot correspondence space.
This module provides lightweight residual adapters inserted after P3/P4 features,
redirecting detection-oriented features to few-shot correspondence space
with minimal trainable parameters.

Architecture | 架构:
    ResidualAdapter (per scale):
        x → 1×1(down) → BN → ReLU → 3×3(conv) → BN → ReLU → 1×1(up) → + x
        初始化: up 层权重为零 → adapter 初始行为 = identity
        Init: up layer weights zero → adapter starts as identity

    MultiScaleAdapter:
        P3 → ResidualAdapter(960, hidden) → P3'
        P4 → ResidualAdapter(1280, hidden) → P4'
        P8 → passthrough (不修改 | unchanged)

参数统计 | Param Count:
    hidden_dim=256: P3 adapter ~498K, P4 adapter ~590K, Total ~1.09M
    hidden_dim=128: P3 adapter ~125K, P4 adapter ~148K, Total ~273K

用法 | Usage::
    >>> from adatile.backbone.feature_adapter import MultiScaleAdapter
    >>> adapter = MultiScaleAdapter(hidden_dim=128)
    >>> backbone.set_adapters(adapter)
    >>> # 之后 backbone(x) 自动应用 adapter
    >>> # After that, backbone(x) auto-applies adapter
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """
    残差特征适配器 | Residual Feature Adapter.

    将 detection-oriented 特征重定向到 few-shot correspondence space,
    同时保留原始特征的残差连接确保训练稳定。
    Redirects detection-oriented features to few-shot correspondence space,
    with residual connection to preserve original features for training stability.

    Architecture | 架构:
        x → Conv1×1(down) → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1(up) → x + residual

    Key property | 关键特性:
        up 层权重初始化为零 → 初始行为是 identity → 训练从原始特征开始逐步偏离。
        Up layer weights initialized to zero → starts as identity → gradually deviates.

    Parameters | 参数:
        in_channels: 输入通道数 | Input channels (P3=960, P4=1280).
        hidden_dim: 瓶颈维度 | Bottleneck dimension (default 128 for lightweight).
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 1×1 降维 | 1×1 channel reduction
        self.down = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)

        # 3×3 空间变换 | 3×3 spatial transformation
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)

        # 1×1 升维 | 1×1 channel expansion
        self.up = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)

        # 归一化 | Normalization
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

        # 激活 | Activation
        self.act = nn.ReLU(inplace=True)

        # ── 初始化: up 层为零 → adapter 初始为 identity ──
        # Init: up layer zero → adapter starts as identity
        self._init_weights()

    def _init_weights(self):
        """初始化权重: up 层为零, 其余 Kaiming normal. | Init: up=zero, rest=Kaiming."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if name == "up":
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, H, W] 输入特征 | Input features.
        :return: [B, C, H, W] 适配后特征 | Adapted features.
        """
        residual = self.down(x)
        residual = self.act(self.norm1(residual))
        residual = self.conv(residual)
        residual = self.act(self.norm2(residual))
        residual = self.up(residual)
        return x + residual

    @property
    def num_params(self) -> int:
        """可训练参数数量 | Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleAdapter(nn.Module):
    """
    多尺度特征适配器 | Multi-Scale Feature Adapter.

    对 P3 和 P4 特征分别应用 ResidualAdapter, P8 原样通过。
    Applies ResidualAdapter to P3 and P4 features, P8 passes through unchanged.

    Parameters | 参数:
        feat_dim_p3: P3 通道数 | P3 channels (default 960).
        feat_dim_p4: P4 通道数 | P4 channels (default 1280).
        hidden_dim: 适配器瓶颈维度 | Adapter bottleneck dimension (default 128).
    """

    def __init__(
        self,
        feat_dim_p3: int = 960,
        feat_dim_p4: int = 1280,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feat_dim_p3 = feat_dim_p3
        self.feat_dim_p4 = feat_dim_p4
        self.hidden_dim = hidden_dim

        self.p3_adapter = ResidualAdapter(feat_dim_p3, hidden_dim)
        self.p4_adapter = ResidualAdapter(feat_dim_p4, hidden_dim)

        self._total_params = self.p3_adapter.num_params + self.p4_adapter.num_params

    def forward(
        self,
        p3: torch.Tensor | None = None,
        p4: torch.Tensor | None = None,
        p8: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        对多尺度特征应用适配器 | Apply adapters to multi-scale features.

        :param p3: [B, 960, H/8, W/8] P3 特征 | P3 features.
        :param p4: [B, 1280, H/16, W/16] P4 特征 | P4 features.
        :param p8: [B, C, H/32, W/32] P8 特征 (passthrough) | P8 features (passthrough).
        :return: dict with keys 'p3', 'p4', 'p8' (only for non-None inputs).
        """
        result: dict[str, torch.Tensor] = {}
        if p3 is not None:
            result["p3"] = self.p3_adapter(p3)
        if p4 is not None:
            result["p4"] = self.p4_adapter(p4)
        if p8 is not None:
            result["p8"] = p8  # passthrough | 原样通过
        return result

    @property
    def num_params(self) -> int:
        """总可训练参数数量 | Total number of trainable parameters."""
        return self._total_params

    def __repr__(self) -> str:
        return (
            f"MultiScaleAdapter("
            f"P3: {self.feat_dim_p3}→{self.hidden_dim}→{self.feat_dim_p3}, "
            f"P4: {self.feat_dim_p4}→{self.hidden_dim}→{self.feat_dim_p4}, "
            f"total_params={self.num_params:,})"
        )
