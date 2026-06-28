#!/usr/bin/env python3
"""
B-01 Token-Level: 概念验证 — Oracle 知道一切时，Token 级稀疏性的理论上界
=====================================================================
Concept Verification: Token-Level Oracle Sparsity Upper Bound.

核心问题 | Core Question:
    在单个 tile 内部，FastSAM P4 特征图的 token (stride=16 cell)
    中，前 30% 的 token 能覆盖多少前景像素？

    Within a single tile, what % of foreground pixels can be captured
    by the top 30% of P4 tokens (stride=16 cells)?

动机 | Motivation:
    B-01 (tile 级) 证明了 tile 之间的稀疏性 — Top 30% tile 捕获 96.5% FG。
    但是 tile 内部的 token 是否也存在类似的稀疏性？

    B-01 (tile-level) proved inter-tile sparsity — Top 30% tiles capture 96.5% FG.
    Does intra-tile token-level sparsity also exist?

    如果 Top 30% token → 95%+ FG，那么 Token Routing 就是可行的。
    If Top 30% tokens → 95%+ FG, then Token Routing is viable.

实验设计 | Design:
    1. 采样 N 个 tile，每个 tile 通过 FastSAM backbone 提取 P4 特征
    2. GT mask 下采样到 P4 空间分辨率 → per-token fg_ratio (Oracle)
    3. 按 fg_ratio 排序所有 token → 累积前景曲线
    4. 可视化: 保留率曲线 + token 密度分布 + 单 tile 示例

零训练成本 | Zero Training Cost:
    不需要任何训练。只需要 frozen FastSAM backbone + GT mask。
    5 分钟内跑完，motivation 就立住了。

用法 | Usage::
    python tools/paper_b/eval_b01_token_sparsity.py
    python tools/paper_b/eval_b01_token_sparsity.py --tile-root data/iSAID5i_tiles/tile_896 --n-tiles 100
    python tools/paper_b/eval_b01_token_sparsity.py --tile-root data/iSAID_tiles --tile-size 1024
"""

import sys, argparse, json
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.utils.seed import set_seed

logger = get_logger("b01_token")

# ═══════════════════════════════════════════════════════════════════════════
# 配置常量 | Configuration Constants
# ═══════════════════════════════════════════════════════════════════════════

STRIDE = 16                              # FastSAM P4 步长 | P4 stride
K_LEVELS = [5, 10, 15, 20, 25, 30,      # Top-K% 档位 | Top-K% levels
            35, 40, 45, 50, 60, 70, 80, 90, 100]
FG_THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]  # FG 密度分箱阈值 | Density bin thresholds


def parse_args():
    p = argparse.ArgumentParser(
        description="Token-Level Oracle Sparsity — Concept Verification")
    p.add_argument("--tile-root", type=str,
                   default="data/iSAID5i_tiles/tile_896",
                   help="Tile 数据集根目录 | Tile dataset root")
    p.add_argument("--split", type=str, default="train",
                   help="数据划分 | Data split (train/val)")
    p.add_argument("--n-tiles", type=int, default=100,
                   help="采样 tile 数量 (0=全部) | Number of tiles to sample (0=all)")
    p.add_argument("--tile-size", type=int, default=0,
                   help="Tile 尺寸 (0=自动检测) | Tile size (0=auto-detect)")
    p.add_argument("--device", type=str, default="cuda",
                   help="设备 | Device (cuda/cpu)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed")
    p.add_argument("--output-dir", type=str,
                   default="runs/b01_token_sparsity",
                   help="输出目录 | Output directory")
    p.add_argument("--show-examples", type=int, default=4,
                   help="可视化示例 tile 数量 | Number of example tiles to visualize")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 核心计算 | Core Computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_token_fg_ratio(
    label: np.ndarray,
    h_tokens: int,
    w_tokens: int,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    从 GT label 计算 per-token 前景比例 (Oracle Importance Map)。
    Compute per-token foreground ratio from GT label (Oracle Importance Map).

    将 dense label mask 划分为 stride×stride 的网格，
    计算每个格子内前景像素的比例。
    Divides the dense label mask into stride×stride grid cells,
    computes the foreground pixel ratio within each cell.

    Parameters
    ----------
    label : np.ndarray [H, W] uint8
        密集类别标签 (0=背景, >0=前景类别)。
        Dense category label (0=background, >0=foreground classes).
    h_tokens, w_tokens : int
        Token 网格尺寸 (H/stride, W/stride)。| Token grid dimensions.
    stride : int
        步长 (默认 16 = FastSAM P4)。| Stride (default 16 = FastSAM P4).

    Returns
    -------
    fg_ratio : np.ndarray [h_tokens, w_tokens] float32
        Per-token 前景比例 [0, 1]。| Per-token foreground ratio [0, 1].
    """
    H, W = label.shape[:2]
    # 裁剪到 stride 对齐 | Crop to stride-aligned region
    h_crop = h_tokens * stride
    w_crop = w_tokens * stride
    label_crop = label[:h_crop, :w_crop]

    # 二值前景 | Binary foreground
    fg = (label_crop > 0).astype(np.float32)

    # 重塑为 [h_tokens, stride, w_tokens, stride] → mean over stride dims
    # Reshape to [h_tokens, stride, w_tokens, stride] → mean over stride dims
    fg_ratio = fg.reshape(h_tokens, stride, w_tokens, stride)
    fg_ratio = fg_ratio.transpose(0, 2, 1, 3)  # [h_tokens, w_tokens, stride, stride]
    fg_ratio = fg_ratio.reshape(h_tokens, w_tokens, -1).mean(axis=2)  # [h_tokens, w_tokens]

    return fg_ratio.astype(np.float32)


def analyze_single_tile(
    img_path: Path,
    label_path: Path,
    backbone: FastSAMBackbone,
    device: torch.device,
) -> dict:
    """
    分析单个 tile 的 token 级前景稀疏性。
    Analyze token-level foreground sparsity for a single tile.

    Returns dict with:
        - fg_ratio_map: [h_tokens, w_tokens] oracle importance map
        - sorted_fg: [N] flattened fg_ratios sorted descending
        - cum_fg: [N] cumulative FG (normalized to total FG)
        - total_fg_pixels: int
        - n_tokens: int
        - n_fg_tokens: int (tokens with any FG)
        - tile_name: str
        - p4_shape: tuple
    """
    # ── 加载图像 → backbone → P4 | Load image → backbone → P4 ──
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

    feats = backbone(img_t)
    p4 = feats["p4"]  # [1, 1280, H/16, W/16]
    _, _, h_t, w_t = p4.shape

    # ── 加载 label → per-token fg_ratio | Load label → per-token fg_ratio ──
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    fg_map = compute_token_fg_ratio(label, h_t, w_t, stride=STRIDE)  # [h_t, w_t]

    # ── 排序 + 累积 | Sort + cumulative ──
    fg_flat = fg_map.flatten()
    sorted_idx = np.argsort(fg_flat)[::-1]  # 降序 | Descending
    sorted_fg = fg_flat[sorted_idx]

    total_fg = sorted_fg.sum()
    cum_fg = np.cumsum(sorted_fg) / (total_fg + 1e-8)  # 归一化累积 | Normalized cumulative

    n_fg_tokens = int((fg_flat > 0).sum())

    return {
        "fg_ratio_map": fg_map,
        "sorted_fg": sorted_fg,
        "cum_fg": cum_fg,
        "total_fg_pixels": float(total_fg),
        "n_tokens": len(fg_flat),
        "n_fg_tokens": n_fg_tokens,
        "tile_name": img_path.stem,
        "p4_shape": (h_t, w_t),
        "fg_ratios_raw": fg_flat,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════════════

def make_visualization(
    all_results: list,
    k_levels: list,
    output_dir: Path,
    example_indices: list,
    tile_root: Path,
    split: str,
):
    """
    生成 6 面板可视化 | Generate 6-panel visualization.

    Panel 1: 累积 FG 保留率曲线 (主结果) | Cumulative FG retention curve (main)
    Panel 2: 边际收益 (递减规律) | Marginal gain (diminishing returns)
    Panel 3: Token FG 密度分布直方图 | Token FG density histogram
    Panel 4: 单 tile 示例 (原图 + GT + Oracle热力图 + Top-K mask)
    Panel 5: 关键数字摘要表 | Key numbers summary table
    Panel 6: 浪费分析 + 结论 | Waste analysis + conclusion
    """
    n_total_tokens = sum(r["n_tokens"] for r in all_results)
    n_fg_tokens = sum(r["n_fg_tokens"] for r in all_results)
    total_fg_val = sum(r["total_fg_pixels"] for r in all_results)

    # ── 聚合所有 token 的排序 fg_ratio | Aggregate sorted fg_ratios across all tiles ──
    # 由于不同 tile 的 token 数量可能不同，我们用"分位累积"策略:
    # 对每个 tile，计算每个 K% 的 FG 保留率，然后平均。
    # Per-tile percentile-based cumulative, then average across tiles.

    k_pcts = np.array(k_levels)
    n_tiles = len(all_results)

    # 每个 tile 在每个 K 的保留率 | Per-tile retention at each K
    per_tile_retention = np.zeros((n_tiles, len(k_levels)))
    # 全局排序聚合 (把所有 token 混在一起排序) | Global pooling (mix all tokens together)
    all_sorted_fg = np.concatenate([r["sorted_fg"] for r in all_results])
    all_total_fg = all_sorted_fg.sum()
    all_cum_fg = np.cumsum(all_sorted_fg) / (all_total_fg + 1e-8)
    n_all_tokens = len(all_sorted_fg)

    for i, r in enumerate(all_results):
        cum = r["cum_fg"]  # [N_tokens] normalized cumulative
        n_tok = r["n_tokens"]
        for j, k in enumerate(k_levels):
            n_keep = max(1, int(n_tok * k / 100))
            per_tile_retention[i, j] = cum[min(n_keep - 1, n_tok - 1)]

    mean_retention = per_tile_retention.mean(axis=0)
    std_retention = per_tile_retention.std(axis=0)

    # 全局排序的保留率 | Global sorted retention
    global_retention = np.zeros(len(k_levels))
    for j, k in enumerate(k_levels):
        n_keep = max(1, int(n_all_tokens * k / 100))
        global_retention[j] = all_cum_fg[min(n_keep - 1, n_all_tokens - 1)]

    # ═══════════════════════════════════════════════════════════════
    # Figure setup | 图表布局
    # ═══════════════════════════════════════════════════════════════
    # 全局: 4行 × 3列 (底部行用更大的 row)
    # Global: 4 rows × 3 cols main grid. Row 1 = example tiles (full-width)
    fig = plt.figure(figsize=(26, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.35,
                  height_ratios=[1.0, 1.0, 0.9, 0.9])

    # ═══ Panel 1: FG 保留率曲线 — 核心结果 | FG Retention Curve — core result ═══
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(k_pcts,
                    (mean_retention - std_retention) * 100,
                    (mean_retention + std_retention) * 100,
                    alpha=0.15, color="#27AE60")
    ax.plot(k_pcts, mean_retention * 100, "o-", color="#27AE60",
            linewidth=2.5, markersize=8, label="Per-tile mean ± std")
    ax.plot(k_pcts, global_retention * 100, "s--", color="#E74C3C",
            linewidth=2.0, markersize=6, alpha=0.7, label="Global (all tokens pooled)")

    # 参考线 | Reference lines
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=30, color="#E74C3C", linestyle="--", alpha=0.3, linewidth=1)

    # 标注关键点 | Annotate key points
    for k_target in [10, 20, 30, 40, 50]:
        idx = k_levels.index(k_target)
        ax.annotate(f"Top {k_target}%\n→ {mean_retention[idx]*100:.1f}% FG",
                    (k_target, mean_retention[idx] * 100),
                    textcoords="offset points", xytext=(5, -22),
                    fontsize=8, color="#2C3E50",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Tokens Kept (%)", fontsize=12)
    ax.set_ylabel("Foreground Retained (%)", fontsize=12)
    ax.set_title(f"Token-Level Oracle: FG Retention vs Token Budget\n"
                 f"({n_tiles} tiles, {n_all_tokens:,} tokens, stride={STRIDE})",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # ═══ Panel 2: 边际收益 | Marginal Gain — diminishing returns ═══
    ax = fig.add_subplot(gs[0, 1])
    marginal_gain = np.diff(mean_retention * 100)
    k_mid = (k_pcts[:-1] + k_pcts[1:]) / 2
    colors = ["#E74C3C" if i < 6 else "#3498DB" for i in range(len(marginal_gain))]
    bars = ax.bar(k_mid, marginal_gain, width=3.5, color=colors,
                  edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    # 标注边际收益转折点 | Mark diminishing returns threshold
    best_idx = np.argmax(marginal_gain)
    ax.annotate(f"Peak: +{marginal_gain[best_idx]:.1f}%\nper 5% tokens",
                (k_mid[best_idx], marginal_gain[best_idx]),
                textcoords="offset points", xytext=(0, 10),
                fontsize=9, ha="center", color="#E74C3C", fontweight="bold")
    ax.set_xlabel("Tokens Kept (%)", fontsize=11)
    ax.set_ylabel("Δ FG Retention per 5% Tokens", fontsize=11)
    ax.set_title("Marginal Information Gain\n"
                 f"(Red=first 30% high-gain, Blue=diminishing returns)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # ═══ Panel 3: Token FG 密度分布 | Token FG Density Distribution ═══
    ax = fig.add_subplot(gs[0, 2])
    all_fg_raw = np.concatenate([r["fg_ratios_raw"] for r in all_results])
    ax.hist(all_fg_raw, bins=100, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Sparse (<1%)")
    ax.axvline(x=0.05, color="orange", linestyle="--", linewidth=1.5, label="Meaningful (≥5%)")
    # 密度分箱统计 | Density bin stats
    bins_text = []
    for th in FG_THRESHOLDS[1:]:
        pct = (all_fg_raw >= th).mean() * 100
        bins_text.append(f"≥{th*100:.0f}%: {pct:.1f}% tokens")
    ax.text(0.98, 0.95, "\n".join(bins_text), transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Foreground Ratio per Token", fontsize=11)
    ax.set_ylabel("Number of Tokens", fontsize=11)
    ax.set_title(f"Token-Level FG Density Distribution\n"
                 f"({n_all_tokens:,} tokens, "
                 f"{n_fg_tokens/n_all_tokens*100:.1f}% have FG)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.2)

    # ═══ Panel 4 (Row 1, full width): Tile Examples | 单 tile 示例 ═══
    # 嵌套 GridSpec：4 行(示例) × 4 列(视图) | Nested GridSpec: 4 rows × 4 cols
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    n_examples = len(example_indices)
    gs_examples = GridSpecFromSubplotSpec(
        n_examples + 1, 4, subplot_spec=gs[1, :],
        hspace=0.15, wspace=0.08,
    )

    # 列标题 | Column headers
    col_titles = ["Image", "GT Foreground", "Oracle Heatmap", "Top-30% Mask"]
    for col_idx, ctitle in enumerate(col_titles):
        ax_hdr = fig.add_subplot(gs_examples[0, col_idx])
        ax_hdr.text(0.5, 0.5, ctitle, ha="center", va="center",
                    fontsize=10, fontweight="bold")
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])

    for ex_i, tile_idx in enumerate(example_indices):
        r = all_results[tile_idx]
        tile_name = r["tile_name"]
        fg_map = r["fg_ratio_map"]

        # 加载原图和 label | Load image and label
        img_path = tile_root / split / "images" / f"{tile_name}.png"
        label_path = tile_root / split / "labels" / f"{tile_name}_label.png"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        fg_binary = (label > 0).astype(np.uint8) * 255

        # Top-30% mask | Top-30% token mask
        n_keep = max(1, int(r["n_tokens"] * 0.30))
        fg_flat = fg_map.flatten()
        threshold = np.sort(fg_flat)[-n_keep]
        top30_mask = (fg_map >= max(threshold, 1e-8)).astype(np.uint8) * 255
        top30_mask_up = cv2.resize(top30_mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Oracle 热力图叠加 | Oracle heatmap overlay
        fg_heatmap = (fg_map * 255).astype(np.uint8)
        fg_heatmap_color = cv2.applyColorMap(fg_heatmap, cv2.COLORMAP_JET)
        fg_heatmap_up = cv2.resize(fg_heatmap_color, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(img, 0.5, fg_heatmap_up, 0.5, 0)

        # FG 比例短标签 | Short FG ratio label
        fg_density = r["total_fg_pixels"] / r["n_tokens"]
        label_text = f"{tile_name[:18]}\nFG dens={fg_density:.4f}"

        for col, (img_data, cmap_name) in enumerate([
            (img, None),
            (fg_binary, "gray"),
            (overlay, None),
            (top30_mask_up, "gray"),
        ]):
            ax = fig.add_subplot(gs_examples[ex_i + 1, col])
            if cmap_name:
                ax.imshow(img_data, cmap=cmap_name)
            else:
                ax.imshow(img_data)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label_text, fontsize=7, rotation=0,
                              labelpad=50, va="center")

    # ═══ Panel 5 (Row 2, Col 0): 关键数字摘要 | Key Numbers Summary ═══
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 收集关键数据 | Collect key numbers
    r20_idx = k_levels.index(20)
    r30_idx = k_levels.index(30)
    r40_idx = k_levels.index(40)
    r50_idx = k_levels.index(50)

    # 达到 90%/95%/99% FG 需要多少 token | Tokens needed to reach 90%/95%/99% FG
    milestones = {}
    for target in [90, 95, 99]:
        idx = np.searchsorted(global_retention, target / 100)
        n_needed = min(int(idx) + 1, n_all_tokens)
        milestones[target] = (n_needed, n_needed / n_all_tokens * 100)

    summary_lines = [
        "B-01 Token: Oracle Upper Bound",
        "=" * 40,
        "",
        f"Dataset: {n_tiles} tiles",
        f"  {n_all_tokens:,} P4 tokens (stride={STRIDE})",
        f"  {n_fg_tokens:,} tokens with FG "
        f"({n_fg_tokens/n_all_tokens*100:.1f}%)",
        f"  {total_fg_val:.0f} total FG (token-sum)",
        "",
        "Oracle Retention (per-tile mean):",
        f"  Top 20% tokens → {mean_retention[r20_idx]*100:.1f}% FG",
        f"  Top 30% tokens → {mean_retention[r30_idx]*100:.1f}% FG",
        f"  Top 40% tokens → {mean_retention[r40_idx]*100:.1f}% FG",
        f"  Top 50% tokens → {mean_retention[r50_idx]*100:.1f}% FG",
        "",
        "Oracle Retention (global pool):",
        f"  Top 20% tokens → {global_retention[r20_idx]*100:.1f}% FG",
        f"  Top 30% tokens → {global_retention[r30_idx]*100:.1f}% FG",
        f"  Top 40% tokens → {global_retention[r40_idx]*100:.1f}% FG",
        "",
        "Inflection Points (global):",
        f"  90% FG → Top {milestones[90][1]:.1f}% tokens",
        f"  95% FG → Top {milestones[95][1]:.1f}% tokens",
        f"  99% FG → Top {milestones[99][1]:.1f}% tokens",
    ]
    for i, line in enumerate(summary_lines):
        y_pos = 9.5 - i * 0.38
        if line.startswith("B-01"):
            ax.text(0.5, y_pos, line, fontsize=13, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("===="):
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        elif "Inflection" in line or "→" in line:
            if "90%" in line or "95%" in line or "99%" in line:
                ax.text(0.5, y_pos, line, fontsize=9, fontweight="bold",
                        fontfamily="monospace", va="top", color="#27AE60")
            else:
                ax.text(0.5, y_pos, line, fontsize=9,
                        fontfamily="monospace", va="top")
        else:
            ax.text(0.5, y_pos, line, fontsize=9,
                    fontfamily="monospace", va="top")

    # ═══ Panel 6: 浪费分析 + 与 Tile 级对比 | Waste Analysis + Comparison with Tile-Level ═══
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 空 token 分析 | Empty token analysis
    empty_tokens = (all_fg_raw == 0).sum()
    sparse_tokens = ((all_fg_raw > 0) & (all_fg_raw < 0.05)).sum()
    meaningful_tokens = (all_fg_raw >= 0.05).sum()

    # 浪费的 FG: 后 50% token 贡献了多少 | Wasted FG: what do bottom 50% tokens contribute?
    n_half = n_all_tokens // 2
    bottom_half_fg = all_sorted_fg[n_half:].sum()
    bottom_half_pct = bottom_half_fg / (all_total_fg + 1e-8) * 100

    waste_lines = [
        "Waste Analysis (Token-Level)",
        "=" * 40,
        "",
        "Token Categories:",
        f"  Empty (fg=0):    {empty_tokens:,} "
        f"({empty_tokens/n_all_tokens*100:.1f}%)",
        f"  Sparse (0-5%):   {sparse_tokens:,} "
        f"({sparse_tokens/n_all_tokens*100:.1f}%)",
        f"  Meaningful (≥5%): {meaningful_tokens:,} "
        f"({meaningful_tokens/n_all_tokens*100:.1f}%)",
        "",
        f"Bottom 50% tokens contribute",
        f"  only {bottom_half_pct:.1f}% of total FG",
        "",
        "Comparison with Tile-Level (B-01):",
        f"  Tile: Top 30% → 96.5% FG retained",
        f"  Token: Top 30% → {mean_retention[r30_idx]*100:.1f}% FG",
        "",
        "Key Insight:",
    ]

    # 动态分析 | Dynamic analysis
    token_vs_tile = mean_retention[r30_idx] * 100
    if token_vs_tile > 90:
        insight = ("  Token sparsity ≈ Tile sparsity\n"
                   "  → Routing at token level is\n"
                   "    as effective as tile level!")
    elif token_vs_tile > 75:
        insight = ("  Token sparsity < Tile sparsity\n"
                   "  → Token routing is viable but\n"
                   "    less efficient than tile routing.\n"
                   "    Consider hybrid approach.")
    else:
        insight = ("  Token sparsity ≪ Tile sparsity\n"
                   "  → Token routing alone insufficient.\n"
                   "    Must combine tile pre-filter + token.\n"
                   "    This IS the AdaTile motivation.")

    for i, line in enumerate(waste_lines + [insight]):
        y_pos = 9.5 - i * 0.38
        if line.startswith("Waste") or line.startswith("Key Insight"):
            ax.text(0.5, y_pos, line, fontsize=11, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("===="):
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        elif "→" in line:
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="#E74C3C" if "only" in line else "#27AE60")
        elif line.startswith("  Token sparsity"):
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top",
                    color="#27AE60" if "≈" in line else "#E67E22")
        else:
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace", va="top")

    # ═══ Panel 7 (右下): Per-tile 方差分析 | Per-tile variance analysis ═══
    ax = fig.add_subplot(gs[2, 2])
    # 展示 Top-30% 保留率的 tile 间分布 | Show Top-30% retention distribution across tiles
    retention_30 = per_tile_retention[:, r30_idx] * 100
    ax.hist(retention_30, bins=30, color="#8E44AD", alpha=0.7, edgecolor="white")
    ax.axvline(x=retention_30.mean(), color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Mean={retention_30.mean():.1f}%")
    ax.axvline(x=np.median(retention_30), color="#3498DB", linestyle="--", linewidth=2,
               label=f"Median={np.median(retention_30):.1f}%")
    ax.set_xlabel("FG Retention at Top-30% Tokens (%)", fontsize=10)
    ax.set_ylabel("Number of Tiles", fontsize=10)
    ax.set_title(f"Per-Tile Variance: Top-30% Token FG Retention\n"
                 f"(σ={retention_30.std():.1f}%, "
                 f"min={retention_30.min():.1f}%, max={retention_30.max():.1f}%)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # ── 保存 | Save ──
    fig.suptitle("B-01 Token: Oracle Top-K Token Selection — Token-Level Sparsity Upper Bound\n"
                 f"(stride={STRIDE}, {n_tiles} tiles, {n_all_tokens:,} P4 tokens)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "token_oracle_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"Saved → {output_dir / 'token_oracle_sparsity.png'}")

    return {
        "mean_retention": mean_retention,
        "std_retention": std_retention,
        "global_retention": global_retention,
        "per_tile_retention": per_tile_retention,
        "milestones": milestones,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tile_root = Path(args.tile_root)
    split = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logging ──
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "token_sparsity.jsonl")))

    logger.log_info("start", "=" * 60)
    logger.log_info("start", "B-01 Token: Oracle Token-Level Sparsity — Concept Verification")
    logger.log_info("start", "=" * 60)
    logger.log_info("config",
                    f"Tile root: {tile_root}, Split: {split}, "
                    f"Device: {device}, Stride: {STRIDE}")

    # ── 发现 tile 文件 | Discover tile files ──
    img_dir = tile_root / split / "images"
    label_dir = tile_root / split / "labels"
    if not img_dir.exists():
        logger.log_error("data", f"Image directory not found: {img_dir}")
        logger.log_info("data", "Expected structure: {tile_root}/{split}/images/*.png")
        logger.log_info("data", "Generate tiles first: python tools/data/prep_isaid5i_multisize.py")
        sys.exit(1)

    img_paths = sorted(img_dir.glob("*.png"))
    if args.n_tiles > 0 and len(img_paths) > args.n_tiles:
        rng = np.random.RandomState(args.seed)
        img_paths = rng.choice(img_paths, args.n_tiles, replace=False).tolist()

    logger.log_info("data", f"Processing {len(img_paths)} tiles")

    # ── 加载 Backbone | Load Backbone ──
    logger.log_info("backbone", "Loading frozen FastSAM backbone...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()
    backbone.eval()
    logger.log_info("backbone", "Backbone loaded and frozen.")

    # ── 逐 tile 分析 | Analyze tile by tile ──
    all_results = []
    for img_path in tqdm(img_paths, desc="Analyzing tiles", unit="tile"):
        tile_name = img_path.stem
        label_path = label_dir / f"{tile_name}_label.png"
        if not label_path.exists():
            logger.log_info("warn", f"Label missing for {tile_name}, skipping")
            continue

        try:
            result = analyze_single_tile(img_path, label_path, backbone, device)
            all_results.append(result)
        except Exception as e:
            logger.log_info("error", f"Failed on {tile_name}: {e}")
            continue

    n_processed = len(all_results)
    n_all_tokens = sum(r["n_tokens"] for r in all_results)
    n_fg_tokens = sum(r["n_fg_tokens"] for r in all_results)
    total_fg_val = sum(r["total_fg_pixels"] for r in all_results)

    logger.log_info("data", f"Processed {n_processed}/{len(img_paths)} tiles successfully")
    logger.log_info("data", f"  Total P4 tokens: {n_all_tokens:,}")
    logger.log_info("data", f"  Tokens with FG: {n_fg_tokens:,} ({n_fg_tokens/n_all_tokens*100:.1f}%)")
    logger.log_info("data", f"  Total FG (token-sum): {total_fg_val:.0f}")

    if n_processed == 0:
        logger.log_error("data", "No tiles processed successfully. Check data paths.")
        sys.exit(1)

    # ── 日志输出关键拐点 | Log key inflection points ──
    # 全局聚合 | Global aggregation
    all_sorted_fg = np.concatenate([r["sorted_fg"] for r in all_results])
    all_total_fg = all_sorted_fg.sum()
    all_cum_fg = np.cumsum(all_sorted_fg) / (all_total_fg + 1e-8)

    logger.log_info("results", "")
    logger.log_info("results", "Oracle Token-Level Retention (Global Pool):")
    logger.log_info("results", f"  {'Top-K%':>7}  {'Tokens':>10}  {'FG Retained':>12}")
    logger.log_info("results", f"  {'-'*34}")

    r30_idx = K_LEVELS.index(30)

    for k in K_LEVELS:
        n_keep = max(1, int(n_all_tokens * k / 100))
        fg_ret = all_cum_fg[min(n_keep - 1, n_all_tokens - 1)] * 100
        logger.log_info("results", f"  {k:>5}%   {n_keep:>10,}   {fg_ret:>11.2f}%")

    # 拐点 | Inflection points
    logger.log_info("results", "")
    logger.log_info("results", "Inflection Points (tokens needed for target FG%):")
    for target in [90, 95, 99]:
        idx = np.searchsorted(all_cum_fg, target / 100)
        n_needed = min(int(idx) + 1, n_all_tokens)
        token_pct = n_needed / n_all_tokens * 100
        logger.log_info("results",
                        f"  {target}% FG → Top {token_pct:.1f}% tokens ({n_needed:,}/{n_all_tokens:,})")

    # ── Per-tile 均值 | Per-tile mean ──
    k_pcts = np.array(K_LEVELS)
    per_tile_ret = np.zeros((n_processed, len(K_LEVELS)))
    for i, r in enumerate(all_results):
        cum = r["cum_fg"]
        n_tok = r["n_tokens"]
        for j, k in enumerate(K_LEVELS):
            n_keep = max(1, int(n_tok * k / 100))
            per_tile_ret[i, j] = cum[min(n_keep - 1, n_tok - 1)]
    mean_ret = per_tile_ret.mean(axis=0)

    r30_val = mean_ret[r30_idx] * 100
    logger.log_info("results", "")
    logger.log_info("results", f"Per-tile mean Top-30% FG retention: {r30_val:.1f}%")
    logger.log_info("results",
                    f"Top 30% tokens capture {r30_val:.1f}% FG → "
                    f"~{100-r30_val:.1f}% token computation safely saveable")

    # ── 可视化 | Visualization ──
    logger.log_info("viz", "Generating visualization...")
    # 选择示例 tile: 高/中/低密度各一个 + 随机 | Pick diverse examples
    fg_densities = [r["total_fg_pixels"] / r["n_tokens"] for r in all_results]
    sorted_by_density = np.argsort(fg_densities)
    n_ex = min(args.show_examples, n_processed)
    example_indices = []
    if n_ex >= 4:
        example_indices = [
            sorted_by_density[0],                          # 最稀疏 | Sparsest
            sorted_by_density[len(sorted_by_density) // 3],  # 低密度 | Low density
            sorted_by_density[len(sorted_by_density) * 2 // 3],  # 中密度 | Medium density
            sorted_by_density[-1],                          # 最密集 | Densest
        ]
    else:
        example_indices = sorted_by_density[:n_ex].tolist()

    viz_data = make_visualization(
        all_results, K_LEVELS, output_dir,
        example_indices, tile_root, split,
    )

    # ── 保存 JSON | Save JSON ──
    import datetime
    summary = {
        "experiment": "B-01 Token Oracle Sparsity",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "tile_root": str(tile_root),
            "split": split,
            "stride": STRIDE,
            "n_tiles_processed": n_processed,
        },
        "statistics": {
            "n_total_tokens": n_all_tokens,
            "n_fg_tokens": n_fg_tokens,
            "fg_token_ratio": round(n_fg_tokens / n_all_tokens, 4),
            "total_fg_token_sum": float(total_fg_val),
        },
        "retention": {
            "per_tile_mean": {str(k): round(float(v), 4)
                              for k, v in zip(K_LEVELS, viz_data["mean_retention"])},
            "per_tile_std": {str(k): round(float(v), 4)
                             for k, v in zip(K_LEVELS, viz_data["std_retention"])},
            "global_pool": {str(k): round(float(v), 4)
                            for k, v in zip(K_LEVELS, viz_data["global_retention"])},
        },
        "inflection_points": {
            str(target): {"tokens_needed": int(n), "token_pct": round(float(pct), 2)}
            for target, (n, pct) in viz_data["milestones"].items()
        },
    }

    json_path = output_dir / "token_sparsity_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log_info("output", f"Results saved → {json_path}")

    # ── 一句话结论 | One-line Conclusion ──
    logger.log_info("conclusion", "")
    logger.log_info("conclusion", "=" * 60)
    logger.log_info("conclusion",
                    f"Token-Level Oracle: Top 30% tokens → {r30_val:.1f}% FG retained.")
    logger.log_info("conclusion",
                    f"Token Routing is {'VIABLE' if r30_val > 80 else 'PARTIALLY VIABLE' if r30_val > 60 else 'INSUFFICIENT'} "
                    f"for intra-tile sparsity exploitation.")
    logger.log_info("conclusion",
                    f"Combined with Tile-Level (Top 30% tiles → 96.5% FG): "
                    f"dual sparsity ≈ {r30_val * 0.965:.1f}% FG with "
                    f"~{100 - 0.30 * 0.30 * 100:.0f}% compute reduction.")
    logger.log_info("conclusion", "=" * 60)


if __name__ == "__main__":
    main()
