"""
边界细化模块 | Boundary Refinement Module.
============================================

Foundation Models (SAM/FastSAM) 在自然图像上训练，输出粗粒度 Mask。
工业缺陷需要像素级边界精度 → BoundaryRefiner 只修正边界区域。

Foundation Models produce coarse masks (SAM prior: large natural objects).
Industrial defects need pixel-level boundaries → BoundaryRefiner only corrects edges.

设计理念 | Design:
    Coarse Mask → BoundaryDet → boundary prob
    P2 features + coarse mask + boundary → RefineNet → residual
    Fine Mask = Coarse + residual × boundary  (非边界区保持不变)

参数: ~85K | Params: ~85K
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefiner(nn.Module):
    """
    边界细化模块 | Boundary Refinement Module.

    仅修正 mask 边界附近像素，其他区域保持粗 mask 不变。
    Only corrects pixels near mask boundaries; interior regions unchanged.

    输入 | Input:
        coarse_mask:  [B, C, H, W] softmax prob from decoder
        p2_features:  [B, C2, H, W] high-res features from FastSAM P2

    输出 | Output:
        refined_mask: [B, C, H, W] refined softmax prob
        boundary:     [B, 1, H, W] learned boundary attention map

    Parameters
    ----------
    feat_channels : int
        P2 特征通道数 (FastSAM-x: 160) | P2 feature channels.
    num_classes : int
        输出类别数 (NEU-Seg: 4) | Number of classes.
    hidden : int
        中间特征通道数 | Hidden feature channels.
    """

    def __init__(self, feat_channels: int = 160, num_classes: int = 4, hidden: int = 64):
        super().__init__()
        self.num_classes = num_classes

        # ── 边界检测器: coarse mask → boundary probability ──
        # Boundary detector: learns which pixels need refinement
        self.boundary_det = nn.Sequential(
            nn.Conv2d(num_classes, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── 细化网络: [P2 + coarse_mask + boundary] → residual ──
        # Refinement net: takes features + mask + boundary → per-class residual
        refine_in = feat_channels + num_classes + 1  # P2 + mask + boundary
        self.refine_net = nn.Sequential(
            nn.Conv2d(refine_in, hidden, kernel_size=1, bias=False),  # 1x1 降维
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化。Residual 网络最后一层零初始化 → 从 identity 开始。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 残差最后一层零初始化 → refined = coarse + 0 = coarse (初始不变)
        # Zero-init last residual layer → refined starts as coarse (no change)
        last_conv = self.refine_net[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, coarse_mask: torch.Tensor,
                p2_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward pass.

        :param coarse_mask: [B, C, H, W] softmax prob from decoder.
        :param p2_features: [B, C2, H, W] P2 features from backbone.
        :return:
            refined_mask: [B, C, H, W] refined softmax prob.
            boundary:     [B, 1, H, W] boundary attention map (for visualization).
        """
        # ── 1. 检测边界 | Detect boundaries ──
        boundary = self.boundary_det(coarse_mask)  # [B, 1, H, W]

        # ── 2. 拼接特征 | Concatenate features ──
        concat = torch.cat([coarse_mask, boundary, p2_features], dim=1)

        # ── 3. 计算残差 | Compute residual ──
        residual = self.refine_net(concat)  # [B, C, H, W]

        # ── 4. 仅边界应用残差 | Apply residual only at boundaries ──
        refined = coarse_mask + residual * boundary

        # 确保仍是合法概率 | Ensure valid probabilities
        refined = refined.clamp(0, 1)
        refined = refined / refined.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return refined, boundary


def boundary_aware_loss(
    pred: torch.Tensor,          # [B, C, H, W] refined softmax prob
    target: torch.Tensor,        # [B, H, W] int64 labels
    boundary: torch.Tensor,      # [B, 1, H, W] boundary attention
    boundary_weight: float = 2.0,  # boundary 区域 loss 权重
) -> torch.Tensor:
    """
    边界感知损失 — 边界像素获得更高权重 | Boundary-aware loss.

    在边界区域加权 CE loss，忽略区域（无边界）权重为 1。
    Boundary pixels get higher CE weight via boundary map.

    :param pred: [B, C, H, W] refined softmax probabilities.
    :param target: [B, H, W] integer class labels.
    :param boundary: [B, 1, H, W] boundary attention from BoundaryRefiner.
    :param boundary_weight: extra weight multiplier for boundary pixels.
    :return: weighted CE loss scalar.
    """
    B, C, H, W = pred.shape
    log_pred = torch.log(pred.clamp(1e-7, 1.0))

    # 逐像素 CE: ne ga ative log of the correct class
    nll = -log_pred.gather(1, target.unsqueeze(1)).squeeze(1)  # [B, H, W]

    # 空间权重: 边界区域放大 | Spatial weight: amplify boundary regions
    bw = boundary.squeeze(1)  # [B, H, W]
    spatial_weight = 1.0 + (boundary_weight - 1.0) * bw  # [1, boundary_weight]

    return (nll * spatial_weight).mean()
