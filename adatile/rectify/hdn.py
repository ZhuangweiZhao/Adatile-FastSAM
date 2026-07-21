"""
HDN — Heatmap Denoiser | 热力图去噪器.
========================================

轻量级可学习热力图去噪器，用于抑制经典 CV 提取的 Gradient/FFT 热力图中的噪声。
Lightweight learnable heatmap denoiser that suppresses noise in classical-CV
extracted heatmaps (Sobel gradient, FFT high-frequency).

设计理念 | Design:
    经典 CV 方法 (Sobel/FFT) 能检测缺陷区域，但 min-max 归一化将所有噪声
    也拉伸到 [0,1]。HDN 学习一个空间门控图 (spatial gate)，选择性地抑制
    噪声区域，保留真实缺陷响应。
    Classical CV methods (Sobel/FFT) detect defect regions, but min-max
    normalization stretches all noise to [0,1] too. HDN learns a spatial
    gate map that selectively suppresses noise while preserving true defects.

架构 | Architecture:
    Input:  raw_heatmap [B, 1, H, W]  (e.g., Sobel gradient magnitude)
    Output: denoised [B, 1, H, W]     (same range as input, noise suppressed)

    Conv2d(1→16, 3, pad=1) → InstanceNorm2d → ReLU
    Conv2d(16→16, 3, pad=1) → InstanceNorm2d → ReLU
    Conv2d(16→1, 1) → Sigmoid           ← zero-init (identity at start)
    output = raw_heatmap × gate

参数量 | Params: ~2.5K (negligible)

用法 | Usage::

    from adatile.rectify.hdn import HeatmapDenoiser

    hdn = HeatmapDenoiser()
    clean = hdn(raw_gradient_heatmap)  # [B, 1, H, W] → [B, 1, H, W]
    gate = hdn.get_gate_map(raw_gradient_heatmap)  # for visualization

监督 | Supervision:
    BCE(denoised_heatmap, GT_binary_mask) — 辅助 loss
    权重建议 0.05-0.2, 不应主导主任务 loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapDenoiser(nn.Module):
    """
    空间门控热力图去噪器 | Spatial-Gated Heatmap Denoiser.

    用一个小 CNN 学习每个空间位置是否属于缺陷，输出门控图与原始
    热力图逐元素相乘，实现选择性噪声抑制。
    A small CNN learns whether each spatial location belongs to a defect,
    outputs a gate map that element-wise multiplies the raw heatmap for
    selective noise suppression.

    Parameters
    ----------
    in_channels : int
        输入热力图通道数 (默认 1) | Input heatmap channels (default 1).
    hidden_channels : int
        隐藏层通道数 | Hidden layer channels. Default 16.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # ── 特征提取 | Feature Extraction ──
        # 两层 3×3 Conv + IN → 感受野 5×5，足够捕获局部纹理 vs 噪声差异
        # Two 3×3 Conv + IN → receptive field 5×5, enough for local texture vs noise
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(hidden_channels, affine=True)

        # ── 门控预测 | Gate Prediction ──
        # 1×1 Conv → Sigmoid → per-pixel gate in [0,1]
        # 零初始化 → sigmoid(0)=0.5 → 初始时轻微衰减 (可快速学习)
        self.gate_conv = nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

        self._init_params_count()

    def _init_params_count(self) -> None:
        """记录参数量 | Log parameter count."""
        n = sum(p.numel() for p in self.parameters())
        from adatile.logging import get_logger
        get_logger("rectify.hdn").log_info(
            "hdn/init",
            f"HDN(in={self.in_channels}, hidden={self.hidden_channels}): "
            f"{n/1e3:.1f}K params",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        去噪前向传播 | Denoising forward pass.

        :param x: [B, C, H, W] 原始热力图 (e.g., Sobel gradient magnitude).
        :return: [B, C, H, W] 去噪后热力图 (同形状) | Denoised heatmap (same shape).
        """
        # ── 特征提取 | Feature Extraction ──
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)

        # ── 门控预测 + 应用 | Gate Prediction + Application ──
        gate = torch.sigmoid(self.gate_conv(h))  # [B, C, H, W], range [0, 1]
        return x * gate

    def get_gate_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取空间门控图 (用于可解释性) | Get spatial gate map (for interpretability).

        低值 → 噪声区域被抑制，高值 → 缺陷区域被保留。
        Low values → noise regions suppressed, high values → defect regions preserved.

        :param x: [B, C, H, W] 原始热力图 | Raw heatmap.
        :return: [B, C, H, W] 门控图 | Gate map, range [0, 1].
        """
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.relu(self.norm2(self.conv2(h)))
        return torch.sigmoid(self.gate_conv(h))

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"HDN(in={self.in_channels}, hidden={self.hidden_channels}, "
                f"params={n/1e3:.1f}K)")
