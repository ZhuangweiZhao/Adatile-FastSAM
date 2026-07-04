"""
SPM (Sparse Perception Module) — v3 统一稀疏感知模块
=======================================================

v2: ForegroundDensityRouter + DensityHead + Ada-SPM (两个名字, 职责重叠)
v3: SparsePerceptionModule + ImportanceHead + TileRouter (统一命名为 SPM)

核心功能: 从 backbone 特征预测 tile 重要性, 选择 Top-K tiles.
Core function: predict tile importance from backbone features, select Top-K tiles.

设计原则 | Design principles:
    1. 密度驱动 (density-driven): 监督信号 = fg_ratio, 而非边缘或纹理
    2. 类别无关 (category-agnostic): 学习"哪里有目标", 不学习"什么目标"
    3. 极致轻量 (ultra-lightweight): 仅 75K 参数, 可忽略不计

架构 | Architecture:
    Backbone Features (e.g. P8 [H/32, W/32, C])
         ↓
    ImportanceHead (Conv stack → importance map)
         ↓
    Importance Map [B, 1, H/32, W/32]
         ↓
    TileRouter → Per-Tile Scores → Top-K Selection
         ↓
    Selected Tile Indices → 送入 AdaptiveDecoder

v2→v3 命名映射 | v2→v3 Naming:
    ForegroundDensityRouter → SparsePerceptionModule (SPM)
    DensityHead            → ImportanceHead
    select_tiles()         → TileRouter.select()
    EdgeHead               → (保留, ablation only)
    TinyCNNRouter          → TinySPM (保留, lower-bound baseline)

用法 | Usage::
    >>> from adatile.sparse.spm import SparsePerceptionModule, ImportanceHead, TileRouter
    >>> spm = SparsePerceptionModule(in_channels=1280)  # FastSAM P8
    >>> importance = spm.predict_importance(p8_features)
    >>> tile_indices = spm.select_tiles(importance, k=0.4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger

logger = get_logger("spm")


# ═══════════════════════════════════════════════════════════════════
# ImportanceHead — 前景密度预测 | Foreground Density Prediction
# ═══════════════════════════════════════════════════════════════════

class ImportanceHead(nn.Module):
    """
    重要性预测头: 从 backbone 特征预测每个空间位置的前景密度.
    Importance prediction head: predicts foreground density at each spatial location.

    v2 名称: DensityHead
    v3 名称: ImportanceHead

    B-02.5 发现: 此头学习的是 objectness / instance density,
    而非类别语义 — 因此可在基类训练、新类泛化.
    B-02.5 finding: this head learns objectness / instance density,
    not class semantics — enabling base-class training, novel-class generalization.

    Parameters
    ----------
    in_channels : int
        输入通道数 | Input channels (e.g. 1280 for FastSAM P8).
    mid_channels : int
        中间通道数 | Middle channels.
    """

    def __init__(self, in_channels: int = 1280, mid_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, H, W] backbone features.
        :return: [B, 1, H, W] density/importance map (raw logits, no activation).
        """
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════
# EdgeHead — 边缘预测头 (消融对照) | Edge Head (Ablation Only)
# ═══════════════════════════════════════════════════════════════════

class EdgeHead(nn.Module):
    """
    边缘预测头 — 仅用于消融对照, 证明 Edge != Importance.
    Edge prediction head — ablation only: proves Edge != Importance.

    B-03 结论: +EdgeHead 仅带来 r=+0.009 提升, 不值得增加复杂度.
    B-03 conclusion: +EdgeHead adds only r=+0.009, not worth the complexity.
    """

    def __init__(self, in_channels: int = 1280, mid_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, H, W] backbone features.
        :return: [B, 1, H, W] edge prediction logits.
        """
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════
# TileRouter — Tile 选择器 | Tile Selector
# ═══════════════════════════════════════════════════════════════════

class TileRouter:
    """
    Tile 路由器: 根据重要性图选择 Top-K tiles.
    Tile router: select Top-K tiles based on importance map.

    将连续的重要性图离散化为 tile 选择决策.
    Discretizes continuous importance map into tile selection decisions.

    v2 名称: select_tiles() 函数
    v3 名称: TileRouter.select()
    """

    @staticmethod
    def select(
        importance: torch.Tensor,
        tile_size: int,
        k: float = 0.4,
        min_tiles: int = 1,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        根据重要性图选择 Top-K tiles | Select Top-K tiles by importance.

        :param importance: [B, 1, H, W] importance map (可以是 raw logits).
        :param tile_size: tile 尺寸 (特征图坐标系) | Tile size in feature map coordinates.
        :param k: 保留的 tile 比例 (0.0-1.0) 或绝对数量 (int) | Fraction or absolute count.
        :param min_tiles: 最少保留 tile 数 | Minimum tiles to keep.
        :return: (selected_tiles_mask, tile_coords) — mask [B, n_tiles_h, n_tiles_w], coords list.
        """
        B, C, H, W = importance.shape
        imp = importance.squeeze(1)  # [B, H, W]

        # 计算 tile grid 尺寸 | Compute tile grid dimensions
        n_tiles_h = H // tile_size
        n_tiles_w = W // tile_size
        if n_tiles_h == 0 or n_tiles_w == 0:
            # 特征图比 tile 小 → 全选 | Feature map smaller than tile → select all
            mask = torch.ones(B, 1, 1, dtype=torch.bool, device=importance.device)
            return mask, [(0, 0)]

        # 池化至 tile-level importance | Pool to tile-level importance
        tile_imp = F.adaptive_avg_pool2d(
            imp.unsqueeze(1), (n_tiles_h, n_tiles_w)
        ).squeeze(1)  # [B, n_tiles_h, n_tiles_w]

        # Top-K 选择 | Top-K selection
        n_total = n_tiles_h * n_tiles_w
        if isinstance(k, float):
            n_select = max(min_tiles, int(n_total * k))
        else:
            n_select = max(min_tiles, min(k, n_total))

        tile_imp_flat = tile_imp.view(B, -1)  # [B, n_total]
        _, topk_idx = torch.topk(tile_imp_flat, n_select, dim=1)  # [B, n_select]

        # 构建 mask | Build mask
        mask = torch.zeros(B, n_total, dtype=torch.bool, device=importance.device)
        mask.scatter_(1, topk_idx, True)
        mask = mask.view(B, n_tiles_h, n_tiles_w)

        # 构建坐标列表 | Build coordinate list
        coords = []
        for b in range(B):
            selected = mask[b].nonzero(as_tuple=False)
            coords.append([(int(r) * tile_size, int(c) * tile_size)
                           for r, c in selected])

        return mask, coords[0] if B == 1 else coords


# ═══════════════════════════════════════════════════════════════════
# SparsePerceptionModule (SPM) — 统一稀疏感知模块
# ═══════════════════════════════════════════════════════════════════

class SparsePerceptionModule(nn.Module):
    """
    稀疏感知模块 (SPM) — v3 统一模块.
    Sparse Perception Module — v3 unified module.

    整合 ImportanceHead + TileRouter, 替代 v2 的 FDR + Ada-SPM.
    Integrates ImportanceHead + TileRouter, replaces v2 FDR + Ada-SPM.

    v2 名称: ForegroundDensityRouter
    v3 名称: SparsePerceptionModule

    Parameters
    ----------
    in_channels : int
        输入通道数 | Input channels (default 1280 for FastSAM P8).
    mid_channels : int
        ImportanceHead 中间通道数 | Middle channels.
    """

    def __init__(self, in_channels: int = 1280, mid_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.importance_head = ImportanceHead(in_channels, mid_channels)
        self.router = TileRouter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测重要性图 | Predict importance map.

        :param x: [B, C, H, W] backbone features (e.g. P8).
        :return: [B, 1, H, W] importance map (raw logits).
        """
        return self.importance_head(x)

    def predict_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测重要性图 (带 sigmoid) | Predict importance map (with sigmoid).

        :param x: [B, C, H, W] backbone features.
        :return: [B, 1, H, W] importance map [0, 1].
        """
        return torch.sigmoid(self.forward(x))

    def select_tiles(
        self,
        x_or_importance: torch.Tensor,
        tile_size: int = 28,
        k: float = 0.4,
        min_tiles: int = 1,
        apply_sigmoid: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        端到端: 特征 → 重要性 → Top-K tiles.
        End-to-end: features → importance → Top-K tiles.

        :param x_or_importance: [B, C, H, W] features OR [B, 1, H, W] importance map.
        :param tile_size: tile 尺寸 (特征图坐标系) | Tile size in feature coords.
        :param k: 保留的 tile 比例 | Fraction of tiles to keep.
        :param min_tiles: 最少 tile 数 | Minimum tiles.
        :param apply_sigmoid: 是否对 importance 做 sigmoid.
        :return: (selected_mask, tile_coords).
        """
        if x_or_importance.shape[1] > 1:
            # 输入是 features → 先预测 importance
            importance = self.importance_head(x_or_importance)
        else:
            importance = x_or_importance

        if apply_sigmoid:
            importance = torch.sigmoid(importance)

        return self.router.select(importance, tile_size, k, min_tiles)

    @property
    def num_params(self) -> int:
        """可训练参数数量 | Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# TinySPM — 极轻量下界基线 | Ultra-Lightweight Lower-Bound Baseline
# ═══════════════════════════════════════════════════════════════════

class TinySPM(nn.Module):
    """
    极轻量 SPM — 仅 3×Conv3×3 + Pooling, 用于验证下界.
    Ultra-lightweight SPM — only 3×Conv3×3 + Pooling, for lower-bound verification.

    v2 名称: TinyCNNRouter
    v3 名称: TinySPM

    Parameters
    ----------
    in_channels : int
        输入通道数 | Input channels.
    tile_size : int
        输出分辨率 (将特征图 pool 到 tile_size × tile_size).
    """

    def __init__(self, in_channels: int = 1280, tile_size: int = 14):
        super().__init__()
        self.in_channels = in_channels
        self.tile_size = tile_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, H, W] features.
        :return: [B, 1, tile_size, tile_size] tile-level importance.
        """
        imp = self.conv(x)  # [B, 1, H, W]
        return F.adaptive_avg_pool2d(imp, (self.tile_size, self.tile_size))


# ═══════════════════════════════════════════════════════════════════
# 向后兼容别名 | Backward-Compatible Aliases
# ═══════════════════════════════════════════════════════════════════

ForegroundDensityRouter = SparsePerceptionModule
DensityHead = ImportanceHead
TinyCNNRouter = TinySPM
