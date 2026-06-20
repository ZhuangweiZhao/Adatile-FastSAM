"""
Spatial Importance Router — 基于前景密度预测 Tile 重要性
==========================================================

Paper B 核心模块。B-02.5 证明 Router 学习的是 objectness / instance density
（而非类别语义或边缘纹理）。本模块将这一发现固化为架构设计。

设计原则 | Design principles:
    1. 密度驱动 (density-driven): 监督信号 = fg_ratio, 而非边缘或纹理
    2. 类别无关 (category-agnostic): 学习"哪里有目标", 不学习"什么目标"
    3. 极致轻量 (ultra-lightweight): 相对分割 decoder 可忽略不计

架构 | Architecture:
    Light Backbone (MV3/EfficientViT/简并CNN)
         ↓
    DensityHead (Conv stack → density map)
         ↓
    Importance Map [B, 1, H/32, W/32]
         ↓
    Tile Pooling → Per-Tile Scores → Top-K Selection

消融 | Ablation (在 eval_b03 中完成):
    R0: MobileNetV3-Small + Simple Head  (B-02 基线)
    R1: Tiny CNN (3×Conv3×3)            (极轻量, 验证下界)
    R2: DensityRouter (正式版)           (主线)
    R3: DensityRouter + EdgeHead         (消融: 边缘是否额外有用)

用法 | Usage:
    >>> from adatile.sparse.spatial_router import ForegroundDensityRouter, TinyCNNRouter
    >>> fdr = ForegroundDensityRouter(in_channels=576)   # MV3 backbone 特征
    >>> fdr = ForegroundDensityRouter(in_channels=1280)  # FastSAM P8 特征
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger

logger = get_logger("spatial_router")


# ═══════════════════════════════════════════════════════════════════
# DensityHead — 前景密度预测 | Foreground Density Prediction
# ═══════════════════════════════════════════════════════════════════

class DensityHead(nn.Module):
    """
    密度预测头：从骨干特征预测每个空间位置的前景密度.
    Density head: predicts foreground density at each spatial location from backbone features.

    B-02.5 发现: 此头学习的是 objectness / instance density,
    而非类别语义 — 因此可在基类训练、新类泛化.
    B-02.5 finding: this head learns objectness / instance density,
    not class semantics — enabling base-class training, novel-class generalization.

    Parameters
    ----------
    in_channels : int
        输入特征通道数 | Input feature channels.
    mid_channels : int
        中间通道数 | Mid channels.
    """

    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        # 特征投影 + 深度可分离卷积 → 密度图
        # Feature projection + depthwise conv → density map
        self.head = nn.Sequential(
            # 1×1 降维 | Channel reduction
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3×3 深度可分离 → 空间上下文 | Depthwise conv → spatial context
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                     groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1×1 → 单通道密度 | 1×1 → single-channel density
            nn.Conv2d(mid_channels, 1, 1, bias=True),
            nn.Sigmoid(),  # [0, 1] 密度分数 | density score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 骨干特征 [B, C, H, W] | Backbone features.
        Returns:
            密度图 [B, 1, H, W] ∈ [0, 1] | Density map.
        """
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════
# EdgeHead — 边缘感知 (仅用于消融, 非主线) | Edge-Aware (ablation only)
# ═══════════════════════════════════════════════════════════════════

class EdgeHead(nn.Module):
    """
    边缘感知头 (消融用, 非主线架构).
    Edge-aware head (for ablation, not the main architecture).

    Sobel 卷积核初始化 → 对边缘/纹理敏感.
    Sobel kernel initialization → sensitive to edges/texture.
    """

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.project = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.edge_conv = nn.Conv2d(mid_channels, 1, 3, padding=1, bias=False)
        self._init_sobel()

    def _init_sobel(self) -> None:
        """Sobel 初始化 | Sobel initialization."""
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        mid = self.edge_conv.in_channels
        kernel = torch.zeros(1, mid, 3, 3)
        for c in range(mid):
            kernel[0, c] = sx if c % 2 == 0 else sy
        self.edge_conv.weight.data.copy_(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.edge_conv(F.relu(self.bn(self.project(x)), inplace=True)))


# ═══════════════════════════════════════════════════════════════════
# FDR — Foreground Density Router (主线) | Paper B Mainline
# ═══════════════════════════════════════════════════════════════════

class ForegroundDensityRouter(nn.Module):
    """
    前景密度路由器 (FDR) — Paper B 主线架构.
    Foreground Density Router (FDR) — Paper B main architecture.

    B-03 消融确立的设计:
        R0 (MV3 full)  → 1.48M, Spearman r=0.884 (上界)
        R2 (FDR only)  → 75K,  Spearman r=0.846 (主线, 20×压缩仅-4.3%)
        R2→R3 (+Edge)  → +0.009 Δr, Edge 几乎无贡献

    设计哲学 | Design philosophy:
        监督信号 = fg_ratio. FDR 学习"这个区域有多少前景目标" (foreground density),
        而非"这个区域有什么边缘"或"这个区域是什么类别".
        B-03 证明: Edge ≠ Importance. Edge 容易被背景结构欺骗.
        B-02.5 证明: FDR 学习的 objectness/instance density 是类别无关的.

    Parameters
    ----------
    in_channels : int
        特征通道数 | Feature channels (MV3=576, FastSAM P8=1280).
    mid_channels : int
        DensityHead 中间通道 | DensityHead mid channels.
    tile_size_feat : int
        每个 tile 对应的特征像素数 | Feature pixels per tile (stride 32 → 32).
    """

    def __init__(
        self,
        in_channels: int = 576,
        mid_channels: int = 128,
        tile_size_feat: int = 32,
    ):
        super().__init__()
        self.tile_size_feat = tile_size_feat
        self.density_head = DensityHead(in_channels, mid_channels)

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.log_info(
            "router/init",
            f"ForegroundDensityRouter (FDR): {n_params:,} params "
            f"({n_trainable:,} trainable) | "
            f"Density only — Pareto optimal vs R0 (20× smaller, only -4.3% Spearman r)",
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: 骨干特征 [B, C, H, W] | Backbone features.
        Returns:
            dict with "importance" density map [B, 1, H, W].
        """
        return {"importance": self.density_head(features)}

    def tile_scores(
        self, importance_map: torch.Tensor, n_ty: int, n_tx: int
    ) -> torch.Tensor:
        """
        重要性图 → tile 分数 (平均池化) | Importance map → tile scores (average pool).
        """
        B, _, hp, wp = importance_map.shape
        scores = []
        for b in range(B):
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0 = ty * self.tile_size_feat
                    y1 = min(y0 + self.tile_size_feat, hp)
                    x0 = tx * self.tile_size_feat
                    x1 = min(x0 + self.tile_size_feat, wp)
                    if y1 > y0 and x1 > x0:
                        scores.append(
                            importance_map[b, 0, y0:y1, x0:x1].mean()
                        )
        return torch.stack(scores) if scores else torch.zeros(0)

    def select_tiles(
        self, importance_map: torch.Tensor, n_ty: int, n_tx: int, k: float = 0.4
    ) -> torch.Tensor:
        """
        Top-K tile 硬选择 (推理用) | Hard Top-K tile selection (for inference).

        返回 mask [B, n_ty, n_tx] bool.
        """
        scores = self.tile_scores(importance_map, n_ty, n_tx)
        B = importance_map.shape[0]
        n_total = n_ty * n_tx
        scores_2d = scores.reshape(B, n_ty, n_tx)
        n_keep = max(1, int(n_total * k))
        mask = torch.zeros(B, n_ty, n_tx, dtype=torch.bool, device=scores.device)
        for b in range(B):
            _, top_idx = torch.topk(scores_2d[b].flatten(), n_keep)
            mask[b, top_idx // n_tx, top_idx % n_tx] = True
        return mask


# ═══════════════════════════════════════════════════════════════════
# DualStreamRouter — 消融专用 | Ablation only
# ═══════════════════════════════════════════════════════════════════

class DualStreamRouter(nn.Module):
    """
    双流路由器: Density + Edge → Fusion (仅用于消融 R3).
    Dual-stream: Density + Edge → Fusion (ablation R3 only).

    验证 Edge 信息是否在 Density 之上有额外贡献.
    Validates whether edge information contributes beyond density.
    """

    def __init__(self, in_channels: int = 576):
        super().__init__()
        self.density_head = DensityHead(in_channels, 128)
        self.edge_head = EdgeHead(in_channels, 64)

        # 融合: [B, 2, H, W] → [B, 1, H, W] | Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        edge = self.edge_head(features)
        density = self.density_head(features)
        fused = self.fusion(torch.cat([edge, density], dim=1))
        return {"importance": fused, "edge": edge, "density": density}


# ═══════════════════════════════════════════════════════════════════
# TinyCNNRouter — 极轻量下界 | Minimal lower bound
# ═══════════════════════════════════════════════════════════════════

class TinyCNNRouter(nn.Module):
    """
    极轻量 CNN Router — 验证"再小的模型也能学到重要性排序".
    Ultra-lightweight CNN Router — validates minimal model suffices for ranking.

    直接从原图 RGB 预测密度，不依赖任何预训练 backbone.
    Directly predicts density from raw RGB, no pretrained backbone.

    结构: RGB → Conv3×3×2 → Conv3×3×2 → Conv1×1 → Sigmoid
    参数: ~20K.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层: stride=2 → 降采样 + 特征提取 | stride=2 → downsample
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 第二层: stride=2 | stride=2
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第三层: stride=2 | stride=2 — 总下采样 8×
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第四层: stride=2 | stride=2 — 总下采样 16×
            nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输出: 1×1 → 密度图 (stride=16) | Output: 1×1 → density map (stride 16)
            nn.Conv2d(64, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self._stride = 16

        n = sum(p.numel() for p in self.parameters())
        logger.log_info("router/init", f"TinyCNNRouter: {n:,} params (stride={self._stride})")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 3, H, W] image → importance map [B, 1, H/16, W/16]."""
        return {"importance": self.net(x)}
