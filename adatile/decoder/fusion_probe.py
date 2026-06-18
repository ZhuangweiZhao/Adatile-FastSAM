"""
FusionProbe — P4+P8 融合探针 | P4+P8 Fusion Probe Decoder.
==============================================================

E003 实验：验证深层语义（P8）是否对建筑分割有额外贡献。
E003 experiment: verify whether deeper semantics (P8) contribute to building segmentation.

结构 | Architecture (params ≈ 10K):

    P4 [B, 1280, H/16, W/16]          P8 [B, 1280, H/32, W/32]
     │                                 │
     │                           Upsample → [B, 1280, H/16, W/16]
     │                                 │
    Conv2d(1280, 4, 1)          Conv2d(1280, 4, 1)
     │                                 │
     └───────── Concat ────────────────┘
                 │
          Conv2d(8, 1, 1)
                 │
          Upsample → [B, 1, H, W]
                 │
          Sigmoid → Binary Mask

参数分解 | Parameter breakdown:
    P4 branch:  1280×4 + 4 = 5,124
    P8 branch:  1280×4 + 4 = 5,124
    Fusion:     8×1 + 1 = 9
    Total:      ≈ 10,257

实验逻辑 | Experimental logic:
    如果 Dice: 0.40 → 0.55+ → 深层语义有价值
    如果 Dice: 0.40 → 0.42  → 问题不在语义层次，在解码能力
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger


class FusionProbe(nn.Module):
    """
    P4+P8 融合探针解码器 | P4+P8 Fusion Probe Decoder.

    极简参数量的双分支融合头，用于验证深层语义贡献。
    Ultra-low-parameter dual-branch fusion head for verifying deep semantic contribution.

    Parameters
    ----------
    in_channels : int
        P4/P8 输入通道数 | P4/P8 input channels.
        FastSAM-x: 1280, FastSAM-s: 640.
    hidden_channels : int
        中间通道数（每分支压缩后）| Hidden channels (per branch after compression).
        默认 4 → 总参数 ~10K。| Default 4 → ~10K total params.
    """

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 4) -> None:
        super().__init__()
        self.logger = get_logger("decoder.fusion_probe")
        self._hidden = hidden_channels

        # P4 分支 | P4 branch: 1280 → hidden
        self.p4_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)

        # P8 分支 | P8 branch: 1280 → hidden
        self.p8_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)

        # 融合层 | Fusion layer: hidden*2 → 1
        self.fusion = nn.Conv2d(hidden_channels * 2, 1, kernel_size=1, bias=True)

        # 参数统计 | Parameter stats
        n_params = sum(p.numel() for p in self.parameters())
        p4_n = sum(p.numel() for p in self.p4_conv.parameters())
        p8_n = sum(p.numel() for p in self.p8_conv.parameters())
        fusion_n = sum(p.numel() for p in self.fusion.parameters())
        self.logger.log_info(
            "fusion_probe/init",
            f"FusionProbe: in={in_channels}, hidden={hidden_channels}, "
            f"params={n_params} (P4={p4_n}, P8={p8_n}, Fusion={fusion_n})",
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        Args:
            features: backbone 输出 {"p4": [B,1280,H/16,W/16], "p8": [B,1280,H/32,W/32]}

        Returns:
            torch.Tensor [B, 1, H/16, W/16] 概率图 | Probability map at stride 16.
        """
        p4 = features["p4"]  # [B, 1280, H/16, W/16]
        p8 = features["p8"]  # [B, 1280, H/32, W/32]

        # P4 分支：直接压缩 | P4 branch: direct compression
        p4_feat = self.p4_conv(p4)  # [B, hidden, H/16, W/16]

        # P8 分支：上采样到 P4 尺寸 + 压缩 | P8 branch: upsample to P4 size + compress
        p8_up = F.interpolate(p8, size=p4.shape[2:], mode="bilinear", align_corners=False)
        p8_feat = self.p8_conv(p8_up)  # [B, hidden, H/16, W/16]

        # 融合 | Fusion
        fused = torch.cat([p4_feat, p8_feat], dim=1)  # [B, hidden*2, H/16, W/16]
        logit = self.fusion(fused)  # [B, 1, H/16, W/16]

        # Sigmoid → 概率 | Probability
        return torch.sigmoid(logit)

    def predict(
        self, features: dict[str, torch.Tensor], target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        预测并上采样到目标尺寸 | Predict and upsample to target size.

        Args:
            features:    backbone 输出 | Backbone output.
            target_size: (H, W) 目标尺寸 | Target spatial size.

        Returns:
            torch.Tensor [B, 1, H, W] 二值 mask | Binary mask.
        """
        prob = self.forward(features)  # [B, 1, H/16, W/16]

        # 上采样到目标尺寸 | Upsample to target size
        prob_up = F.interpolate(
            prob, size=target_size, mode="bilinear", align_corners=False,
        )

        # 二值化 | Binarize
        return (prob_up > 0.5).float()


# ── 用于对比的单独分支 | For ablation comparison ──

class P4OnlyProbe(nn.Module):
    """
    仅 P4 探针（等效于 E002 但控制中间通道数）
    P4-only probe (equivalent to E002 but with controlled hidden channels).

    用于在相同 hidden_channels 下对比 P4 vs P4+P8。
    For comparing P4 vs P4+P8 at the same hidden channels.
    """

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 4) -> None:
        super().__init__()
        self.p4_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        p4 = features["p4"]
        x = self.p4_conv(p4)
        return torch.sigmoid(self.head(x))

    def predict(
        self, features: dict[str, torch.Tensor], target_size: tuple[int, int]
    ) -> torch.Tensor:
        prob = self.forward(features)
        prob_up = F.interpolate(
            prob, size=target_size, mode="bilinear", align_corners=False,
        )
        return (prob_up > 0.5).float()
