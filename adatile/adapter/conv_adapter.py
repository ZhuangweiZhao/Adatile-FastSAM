"""
ConvAdapter — 轻量卷积适配器 (CAT-SAM Adapter 迁移到 ConvNet)
==============================================================
Lightweight Convolutional Adapter — CAT-SAM Adapter concept for ConvNets.

CAT-SAM 在 ViT 的 Transformer Block 中插入 Adapter。YOLOv8 没有 Transformer Block，
但有 C2f Block (跨阶段特征融合)。本模块将其适配为卷积形式：
CAT-SAM inserts Adapters into ViT's Transformer Blocks. YOLOv8 has no Transformer
Blocks, but has C2f Blocks (cross-stage feature fusion). This module adapts
the concept for ConvNets.

设计方案 | Design:
    ConvAdapter: 通道注意力 + 残差连接, 插入到特征提取层之后.
    ConvAdapter: channel attention + residual, inserted after feature extraction.

同 CAT-SAM Adapter 的对应 | Correspondence with CAT-SAM Adapter:
    CAT-SAM:  ViT Block 内 FFN 后 → 降维 → 激活 → 升维 → 残差
    Ours:     YOLOv8 层后 → SE 通道注意力 → 残差 (更适合 ConvNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adatile.logging import get_logger

logger = get_logger("adapter")


class ConvAdapter(nn.Module):
    """
    通道注意力残差适配器 — 使冻结特征适应遥感域.
    Channel-attention residual adapter — adapts frozen features for remote sensing.

    在每个 YOLOv8 特征层输出后插入，学习遥感域特有的特征响应模式。
    Inserted after each YOLOv8 feature layer output, learns remote-sensing-specific
    feature response patterns.

    设计 | Design:
        SE-style 通道注意力:
        Input → GlobalAvgPool → Conv1×1(C→C/r) → ReLU → Conv1×1(C/r→C) → Sigmoid
        Output = Input × (1 + channel_attention)

    对应 CAT-SAM: FFN 后 adapter, 但去掉 Transformer 特有的 LayerNorm + GELU.
    CAT-SAM analog: post-FFN adapter, minus Transformer-specific LayerNorm + GELU.

    ----------
    in_channels : int
        输入通道数 | Input channels.
    reduction : int
        通道压缩比 | Channel reduction ratio (default 4).
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 16)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 零初始化最后一层 Conv 权重 | Zero-init last Conv weight
        # (bias 可能是 None | bias may be None)
        last_conv = self.excitation[-2]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

        n = sum(p.numel() for p in self.parameters())
        logger.log_info("adapter/init",
                        f"ConvAdapter: C={in_channels}, r={reduction}, "
                        f"{n:,} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 特征 [B, C, H, W] | Features.
        :type x: torch.Tensor
        :return: 适配后特征 [B, C, H, W] | Adapted features.
        :rtype: torch.Tensor
        """
        attn = self.excitation(self.squeeze(x))  # [B, C, 1, 1]
        return x * (1.0 + attn)  # 残差连接 | Residual


class MultiScaleAdapter(nn.Module):
    """
    多尺度 ConvAdapter 集合 — 分别适配 P3/P4/P8 三个尺度.
    Multi-scale ConvAdapter set — adapts P3/P4/P8 separately.

    每个尺度的语义信息不同（P3=低层纹理, P4=中层语义, P8=高层语义），
    因此需要独立的 Adapter 参数。
    Each scale encodes different semantics (P3=texture, P4=semantic, P8=high-level),
    so separate adapter parameters are needed.

    ----------
    p3_channels : int
        P3 层通道数 | P3 layer channels (default 640 for FastSAM-x).
    p4_channels : int
        P4 层通道数 | P4 layer channels (default 1280 for FastSAM-x).
    p8_channels : int
        P8 层通道数 | P8 layer channels (default 1280 for FastSAM-x).
    reduction : int
        通道压缩比 | Channel reduction ratio.
    """

    def __init__(
        self,
        p3_channels: int = 640,
        p4_channels: int = 1280,
        p8_channels: int = 1280,
        reduction: int = 4,
    ):
        super().__init__()
        self.p3_adapter = ConvAdapter(p3_channels, reduction) if p3_channels else None
        self.p4_adapter = ConvAdapter(p4_channels, reduction) if p4_channels else None
        self.p8_adapter = ConvAdapter(p8_channels, reduction) if p8_channels else None

        n_total = sum(
            sum(p.numel() for p in a.parameters())
            for a in [self.p3_adapter, self.p4_adapter, self.p8_adapter] if a is not None
        )
        logger.log_info("adapter/init",
                        f"MultiScaleAdapter: P3={p3_channels}, P4={p4_channels}, "
                        f"P8={p8_channels}, total={n_total:,} params")

    def forward(
        self,
        p3: torch.Tensor | None = None,
        p4: torch.Tensor | None = None,
        p8: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        :return: dict with "p3"/"p4"/"p8" keys (only for provided inputs).
        :rtype: dict[str, torch.Tensor]
        """
        result = {}
        if p3 is not None and self.p3_adapter is not None:
            result["p3"] = self.p3_adapter(p3)
        if p4 is not None and self.p4_adapter is not None:
            result["p4"] = self.p4_adapter(p4)
        if p8 is not None and self.p8_adapter is not None:
            result["p8"] = self.p8_adapter(p8)
        return result

    def get_trainable_params(self) -> list[nn.Parameter]:
        """返回所有可训练参数 | Return all trainable parameters."""
        params = []
        for a in [self.p3_adapter, self.p4_adapter, self.p8_adapter]:
            if a is not None:
                params.extend(a.parameters())
        return params
