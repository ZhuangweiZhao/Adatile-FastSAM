"""
DCR — Defect-aware Channel Reweighting | 缺陷感知通道重加权.
=============================================================

SE-Net 风格的通道注意力，学习"哪些通道对工业缺陷敏感"。
SE-Net style channel attention, learns "which channels are sensitive to defects."

核心思想 | Core Idea:
    SA-1B 的通道编码的是自然图像语义 (物体轮廓、颜色、纹理)，
    工业缺陷需要的是高频纹理、微小边缘、异常区域。
    通过通道重加权，抑制"自然图像通道"，增强"缺陷通道"。
    SA-1B channels encode natural image semantics (boundaries, colors, textures),
    but industrial defects need high-freq textures, fine edges, anomaly regions.
    Channel reweighting suppresses "natural image channels" and boosts "defect channels."

与 LoRA 的对比 | vs LoRA:
    LoRA: 修改 backbone 权重 (ΔW ≈ BA) — 危险，可能灾难性遗忘。
    LoRA: modifies backbone weights (ΔW ≈ BA) — risky, catastrophic forgetting.
    DCR:  只乘通道权重 — 安全，backbone 完全不变。
    DCR:  only multiplies channel weights — safe, backbone untouched.

参数量 | Params: ~18K (for 1280 channels, reduction=16)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectChannelReweighting(nn.Module):
    """
    缺陷感知通道重加权 | Defect-aware Channel Reweighting.

    SE-Net 风格: GAP → FC → ReLU → FC → Sigmoid → channel-wise multiply.
    SE-Net style: GAP → FC → ReLU → FC → Sigmoid → channel-wise multiply.

    支持多输入尺度 (P3, P4 各自独立重加权).
    Supports multi-scale input (P3, P4 independently reweighted).

    Parameters
    ----------
    channels : int
        输入通道数 | Input channels (e.g., 960 for P3, 1280 for P4).
    reduction : int
        FC 压缩比 | FC reduction ratio (default 16).
    use_norm : bool
        是否在 squeeze 后加 LayerNorm | Add LayerNorm after squeeze (stabilizes training).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 8)

        # ── Squeeze: GAP + 可选 LayerNorm | Squeeze: GAP + optional LayerNorm ──
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(channels)

        # ── Excitation: FC → ReLU → FC → Sigmoid ──
        self.fc1 = nn.Linear(channels, mid_channels, bias=False)
        self.fc2 = nn.Linear(mid_channels, channels, bias=False)

        # ── 初始化 | Init: fc2 零初始化 → 初始状态 = identity (所有权重≈0.5) ──
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.weight)  # 零初始化 → sigmoid(0) = 0.5 (identity-ish)

        self._init_params_count()

    def _init_params_count(self) -> None:
        n = sum(p.numel() for p in self.parameters())
        from adatile.logging import get_logger
        get_logger("rectify.dcr").log_info(
            "dcr/init",
            f"DCR(ch={self.channels}, r={self.fc1.out_features}): {n/1e3:.1f}K params",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道重加权前向 | Channel reweighting forward.

        :param x: [B, C, H, W] 输入特征图 | Input feature map.
        :return: [B, C, H, W] 重加权后特征 (同形状) | Reweighted features (same shape).
        """
        B, C, H, W = x.shape

        # ── Squeeze: GAP → [B, C] ──
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]

        # ── 可选归一化 | Optional normalization ──
        if self.use_norm:
            gap = self.norm(gap)

        # ── Excitation: FC → ReLU → FC → Sigmoid ──
        attn = self.fc1(gap)                    # [B, C//r]
        attn = F.relu(attn, inplace=True)
        attn = self.fc2(attn)                   # [B, C]
        attn = torch.sigmoid(attn)              # [B, C] ∈ [0, 1]

        # ── 通道乘 | Channel-wise multiply ──
        return x * attn.view(B, C, 1, 1)

    def get_channel_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取通道权重 (用于可解释性分析) | Extract channel weights for interpretability.

        :param x: [B, C, H, W] 输入特征图 | Input feature map.
        :return: [B, C] 通道权重 ∈ [0, 1] | Channel weights.
        """
        B, C, _, _ = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)
        if self.use_norm:
            gap = self.norm(gap)
        attn = self.fc1(gap)
        attn = F.relu(attn, inplace=True)
        attn = torch.sigmoid(self.fc2(attn))
        return attn

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"DCR(ch={self.channels}, r={self.fc1.out_features}, params={n/1e3:.1f}K)"
