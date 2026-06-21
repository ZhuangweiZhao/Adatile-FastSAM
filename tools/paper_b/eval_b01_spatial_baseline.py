#!/usr/bin/env python3
"""
Paper B-01: Spatial Sparsity Baseline — 多少 tile 是空的？
=============================================================

用已生成的 tile metadata 量化空间稀疏性的机会。

核心问题 | Core question:
    全量 iSAID tile 中, 有多少是"空背景"?
    如果用 Top-K tile 选择 (按 fg_ratio), 能省多少计算?

这是 Paper B 的动机实验 — 空间稀疏性的 E008-A。

Tile metadata 格式 (prep_isaid_tiles.py --steps 3 生成):
    {tile_name, img_id, tile_idx, fg_ratio, fg_pixels, total_pixels, class_distribution}

用法 | Usage:
    python tools/eval_b01_spatial_baseline.py
    python tools/eval_b01_spatial_baseline.py --metadata data/iSAID_tiles/metadata/train.json
"""

import sys, argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend

logger = get_logger("b01_spatial")
logger.add_backend(ConsoleBackend())  # 终端实时输出 | Real-time console output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=str, default="data/iSAID_tiles/metadata/train.json")
    p.add_argument("--output-dir", type=str, default="runs/b01_spatial_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    meta_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path) as f:
        tiles = json.load(f)

    n_total = len(tiles)
    fg_ratios = np.array([t["fg_ratio"] for t in tiles])
    fg_pixels = np.array([t["fg_pixels"] for t in tiles])

    # ── 统计 | Statistics ──
    # 三分类计数：空 (<1%), 稀疏 (1-5%), 有意义 (≥5%) | Three-way classification: empty, sparse, meaningful
    empty = int((fg_ratios < 0.01).sum())
    sparse = int(((fg_ratios >= 0.01) & (fg_ratios < 0.05)).sum())
    meaningful = int((fg_ratios >= 0.05).sum())

    logger.log_info("b01/stats",
                    f"Total tiles: {n_total:,}")
    logger.log_info("b01/stats",
                    f"Empty (<1% fg): {empty:,} ({empty/n_total*100:.1f}%)")
    logger.log_info("b01/stats",
                    f"Sparse (1-5% fg): {sparse:,} ({sparse/n_total*100:.1f}%)")
    logger.log_info("b01/stats",
                    f"Meaningful (≥5% fg): {meaningful:,} ({meaningful/n_total*100:.1f}%)")

    # ── Top-K tile selection simulation ──
    # 如果只处理 Top-K 的 tile (按 fg_ratio), 能捕获多少总前景像素?
    # If we only process Top-K tiles (by fg_ratio), how much total FG can we capture?
    sorted_idx = np.argsort(fg_ratios)[::-1]  # 降序 | Descending (highest FG first)
    cum_fg = np.cumsum(fg_pixels[sorted_idx])  # 累积前景像素 | Cumulative FG pixels
    total_fg = fg_pixels.sum()
    cum_fg_frac = cum_fg / (total_fg + 1e-8)  # 归一化累积曲线 | Normalized cumulative curve

    # 找关键拐点: 捕获 90%, 95%, 99% 前景需要多少 tile | Find inflection: tiles needed for target FG%
    ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    thresholds = {}
    for k in ks:
        idx = np.searchsorted(cum_fg_frac, k/100)  # 二分查找累积阈值 | Binary search for threshold
        n_tiles = int(idx) + 1
        tiles_pct = n_tiles / n_total * 100  # 换算为百分比 | Convert to percentage
        thresholds[k] = (n_tiles, tiles_pct)

    logger.log_info("b01/topk", "Top-K Tile Selection (by fg_ratio):")
    logger.log_info("b01/topk",
                    f"  {'FG Target':>10} {'Tiles':>10} {'Tile %':>10}")
    for k in ks:
        n, pct = thresholds[k]
        logger.log_info("b01/topk", f"  {k:>7}% FG   {n:>10,}   {pct:>9.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 可视化 | Visualization
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ═══ (1) fg_ratio 分布直方图 | Panel 1: FG ratio histogram — tile density distribution ═══
    ax = axes[0]
    ax.hist(fg_ratios, bins=100, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0.01, color="red", linestyle="--", label="Empty (<1%)")
    ax.axvline(x=0.05, color="orange", linestyle="--", label="Meaningful (≥5%)")
    ax.set_xlabel("Foreground Ratio per Tile", fontsize=11)
    ax.set_ylabel("Number of Tiles", fontsize=11)
    ax.set_title(f"Tile Foreground Distribution\n"
                 f"Empty={empty:,} ({empty/n_total*100:.1f}%), "
                 f"Meaningful={meaningful:,} ({meaningful/n_total*100:.1f}%)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # ═══ (2) 累积前景覆盖 vs 处理 Tile 比例 | Panel 2: Cumulative FG capture — budget vs reward ═══
    ax = axes[1]
    tile_pcts = np.linspace(0, 100, 200)
    n_tiles_at_pct = [max(1, int(n_total * p / 100)) for p in tile_pcts]
    fg_captured = [cum_fg_frac[min(nt-1, n_total-1)] * 100 for nt in n_tiles_at_pct]
    ax.plot(tile_pcts, fg_captured, color="tab:green", linewidth=2)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=99, color="gray", linestyle="--", alpha=0.5)
    for k in [90, 95, 99]:
        n, tpct = thresholds[k]
        ax.annotate(f"{k}% FG\n({tpct:.0f}% tiles)", (tpct, k),
                    textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("Tiles Processed (%)", fontsize=11)
    ax.set_ylabel("Foreground Captured (%)", fontsize=11)
    ax.set_title("Cumulative FG Capture vs Tile Budget", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ═══ (3) 每张源图的 tile 前景分布 | Panel 3: Per-image tile density scatter — inter-image variance ═══
    ax = axes[2]
    # 按 img_id 分组，收集每张源图的所有 tile fg_ratio | Group by img_id, collect per-image fg_ratio list
    img_fg = {}
    for t in tiles:
        img_id = t["img_id"]
        img_fg.setdefault(img_id, []).append(t["fg_ratio"])
    # 取每张图的极值与均值 | Compute per-image max/min/mean fg_ratio
    img_max_fg = [max(v) for v in img_fg.values()]  # 最密集 tile | Densest tile per image
    img_min_fg = [min(v) for v in img_fg.values()]  # 最稀疏 tile | Sparsest tile per image
    img_mean_fg = [np.mean(v) for v in img_fg.values()]  # 平均密度 | Mean density per image
    ax.scatter(img_mean_fg, img_max_fg, c="tab:red", alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Mean FG per Image", fontsize=11)
    ax.set_ylabel("Max FG per Image (densest tile)", fontsize=11)
    ax.set_title(f"Per-Image Tile Density\n"
                 f"{len(img_fg)} source images", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Paper B-01: Spatial Sparsity Baseline — iSAID Tile Foreground Analysis",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 结论 | Conclusion ──
    k90_tiles, k90_pct = thresholds[90]
    logger.log_info("b01/conclusion",
                    f"To capture 90% of foreground: process {k90_pct:.1f}% tiles")
    logger.log_info("b01/conclusion",
                    f"Meaningless tiles (empty+sparse): {empty+sparse:,} "
                    f"({(empty+sparse)/n_total*100:.1f}%)")
    logger.log_info("b01/conclusion",
                    f"→ Spatial sparsity opportunity: "
                    f"up to {(empty+sparse)/n_total*100:.1f}% computation reduction")

    logger.log_info("b01/output", f"Results saved: {output_dir}/")


if __name__ == "__main__":
    main()
