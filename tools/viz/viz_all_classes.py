#!/usr/bin/env python3
"""
全 15 类 iSAID 尺寸 + 前景覆盖率可视化 | All 15-class size + FG coverage visualization.

用法 | Usage:
    python tools/viz/viz_all_classes.py --src-root data/iSAID_processed
"""

import argparse, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adatile.utils.label_mapping import ISAID_CATEGORIES

# 箱线图采样上限: 超过此值则随机下采样以加速渲染 | Max samples for boxplot
_MAX_BOXPLOT_SAMPLES = 3000

# ═══════════════════════════════════════════════════════════════════
# Class colors — semantic grouping | 类别颜色 — 按语义分组
# ═══════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    # Vehicles — warm reds/oranges
    1:  "#E63946",   # small_vehicle     red
    2:  "#F4A261",   # large_vehicle     orange
    3:  "#E76F51",   # plane             coral
    5:  "#F72585",   # ship              pink
    # Infrastructure — greens/teals
    6:  "#2A9D8F",   # harbor            teal
    7:  "#264653",   # GTF               dark teal
    8:  "#287271",   # SBF               dark green
    9:  "#52B788",   # tennis court      light green
    # Civil — blues/purples
    11: "#457B9D",   # road              blue
    12: "#9B5DE5",   # basketball court  purple
    13: "#7209B7",   # bridge            deep purple
    15: "#4A4E69",   # roundabout        gray
    # Objects — yellows/browns
    4:  "#FFB703",   # storage_tank      amber
    10: "#FF9F1C",   # swimming_pool     orange-yellow
    14: "#D4A373",   # helicopter        tan
}

# Per-class display names (shorter for plots) | 每类显示名（图表用短名）
CLASS_NAMES = {k: v.replace("_", "\n") for k, v in ISAID_CATEGORIES.items()}


def collect_stats(src_root: str):
    """Collect bbox + tile FG stats per class. | 收集每类 bbox + tile FG 统计."""
    import json

    ann_path = Path(src_root) / "train" / "annotations" / "instances_train.json"
    print(f"Loading: {ann_path}")
    with open(ann_path) as f:
        data = json.load(f)
    print(f"  {len(data['images'])} images, {len(data['annotations'])} annotations")

    # BBox stats from COCO JSON | 从 COCO JSON 统计 bbox
    bbox_areas = defaultdict(list)
    bbox_sizes = defaultdict(list)
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid in ISAID_CATEGORIES:
            w, h = ann["bbox"][2], ann["bbox"][3]
            bbox_areas[cid].append(w * h)
            bbox_sizes[cid].append((w, h))

    # Tile FG stats — estimate from bbox areas | 从 bbox 面积估算 FG 覆盖率
    # 每张 tile 896x896=802816px, 假设每个实例完全独立占据一个 tile
    # Per tile 896x896=802816px, assume each instance fully occupies its tile
    tile_px = 896 * 896  # 802816
    tile_fg = defaultdict(list)
    tile_fg_ratio = defaultdict(list)

    for cid in sorted(ISAID_CATEGORIES):
        areas = bbox_areas.get(cid, [])
        # Sample up to 2000 instances for FG stats
        sample_areas = areas[:2000] if len(areas) > 2000 else areas
        for area in sample_areas:
            fg_px = min(area, tile_px)
            tile_fg[cid].append(fg_px)
            tile_fg_ratio[cid].append(fg_px / tile_px)

    n_tiles = 23621  # known value from previous runs
    return bbox_areas, bbox_sizes, tile_fg, tile_fg_ratio, n_tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/viz_all_classes.png")
    args = parser.parse_args()

    bbox_areas, bbox_sizes, tile_fg, tile_fg_ratio, n_tiles = collect_stats(args.src_root)

    cls_ids = sorted(ISAID_CATEGORIES.keys())
    n_cls = len(cls_ids)

    output_path = Path(args.output)
    out_dir = output_path.parent
    out_stem = output_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        p = out_dir / f"{out_stem}_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

    # ── Summary stats | 汇总统计 ──
    summary = {}
    for cid in cls_ids:
        areas = bbox_areas.get(cid, [0])
        fgs = tile_fg.get(cid, [0])
        fg_ratios = tile_fg_ratio.get(cid, [0])
        summary[cid] = {
            "name": ISAID_CATEGORIES[cid],
            "n_bbox": len(areas),
            "area_median": np.median(areas),
            "area_mean": np.mean(areas),
            "fg_median": np.median(fgs),
            "fg_ratio_median": np.median(fg_ratios) * 100,
            "n_tiles_with": len([a for a in fg_ratios if a > 0]),
        }

    # ── 共享 box 标签和颜色 | Shared box labels and colors ──
    box_colors, box_labels = [], []
    for cid in cls_ids:
        box_colors.append(CLASS_COLORS.get(cid, "#888888"))
        box_labels.append(ISAID_CATEGORIES[cid].replace("_", "\n"))

    def _sample_for_boxplot(arr_list):
        """下采样大数组以加速箱线图渲染 | Downsample large arrays for fast boxplot."""
        result = []
        for arr in arr_list:
            arr = np.asarray(arr)
            if len(arr) > _MAX_BOXPLOT_SAMPLES:
                rng = np.random.RandomState(42)
                arr = rng.choice(arr, _MAX_BOXPLOT_SAMPLES, replace=False)
            result.append(arr if len(arr) > 0 else [0])
        return result

    def _boxplot(ax, data, title, ylabel, yscale="log", hlines=None):
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, orientation="vertical",
                       showfliers=False,  # 跳过离群点渲染, 极大加速 | skip outlier dots
                       whis=[5, 95])       # 5-95 百分位胡须 | 5-95 percentile whiskers
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        if hlines:
            for y, ls, lbl in hlines:
                ax.axhline(y=y, color="gray", linestyle=ls, alpha=0.5, linewidth=0.8, label=lbl)
            ax.legend(fontsize=7, loc="upper right")

    # ── Fig 1: BBox Area Boxplot ──
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    bbox_sampled = _sample_for_boxplot([bbox_areas.get(c, [1]) for c in cls_ids])
    _boxplot(ax1, bbox_sampled, "Object Size Distribution (BBox Area per Class)",
            "BBox Area (px, log scale)",
            hlines=[(256, ":", "16x16 px (P4=1 pixel)")])
    _save(fig1, "01_bbox_area")

    # ── Fig 2: Instance Count Bar ──
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    counts = [summary[c]["n_bbox"] for c in cls_ids]
    bar_colors = [CLASS_COLORS.get(c, "#888") for c in cls_ids]
    bars = ax2.bar(range(n_cls), counts, color=bar_colors, alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("Instance Count (log)", fontsize=10)
    ax2.set_title("Instance Count per Class (Train Split)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(n_cls))
    ax2.set_xticklabels([ISAID_CATEGORIES[c].replace("_", "\n") for c in cls_ids],
                       rotation=45, ha="right", fontsize=7)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{cnt:,}", ha="center", fontsize=6, rotation=90)
    ax2.grid(True, alpha=0.3, axis="y")
    _save(fig2, "02_instance_count")

    # ── Fig 3: Tile FG Pixels Boxplot ──
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    fg_sampled = _sample_for_boxplot([
        np.array(tile_fg.get(c, [0]))[np.array(tile_fg.get(c, [0])) > 0] for c in cls_ids
    ])
    _boxplot(ax3, fg_sampled, "Foreground Pixels per Tile (896x896)",
            "FG Pixels per Tile (log scale)",
            hlines=[(802816 * 0.01, ":", "1% of tile (8028 px)")])
    _save(fig3, "03_tile_fg_pixels")

    # ── Fig 4: FG Coverage Ratio Boxplot ──
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ratio_sampled = _sample_for_boxplot([
        np.array(tile_fg_ratio.get(c, [0]))[np.array(tile_fg_ratio.get(c, [0])) > 0.0001] * 100
        for c in cls_ids
    ])
    _boxplot(ax4, ratio_sampled, "Foreground Coverage Ratio (% per Tile)",
            "FG Coverage Ratio (%)",
            yscale="linear",
            hlines=[(1.0, ":", "1% coverage"),
                   (5.0, "--", "5% (B-04 FG filter)")])
    _save(fig4, "04_fg_coverage_ratio")

    # ── Fig 5: Scatter: Area vs Coverage ──
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    for cid in cls_ids:
        s = summary[cid]
        ax5.scatter(s["area_median"], s["fg_ratio_median"], s=s["n_bbox"] / 500,
                   color=CLASS_COLORS.get(cid, "#888"), alpha=0.8, edgecolors="white",
                   linewidth=0.5)
        ax5.annotate(ISAID_CATEGORIES[cid].replace("_", "\n"),
                    (s["area_median"], s["fg_ratio_median"]),
                    xytext=(s["area_median"] * 0.05, s["fg_ratio_median"] * 0.02),
                    fontsize=7, alpha=0.9)
    ax5.set_xscale("log")
    ax5.set_xlabel("Median BBox Area (px)", fontsize=10)
    ax5.set_ylabel("Median FG Coverage (%)", fontsize=10)
    ax5.set_title("Object Size vs Tile Coverage", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax5.axvline(x=256, color="gray", linestyle=":", alpha=0.4)
    ax5.text(30, 8, "Small\nHigh cov\n(RARE)", fontsize=7, alpha=0.5, ha="center")
    ax5.text(3000, 8, "Large\nHigh cov\n(EASY)", fontsize=7, alpha=0.5, ha="center")
    ax5.text(30, 0.2, "Small\nLow cov\n(HARD)", fontsize=7, alpha=0.5, ha="center")
    ax5.text(3000, 0.2, "Large\nLow cov", fontsize=7, alpha=0.5, ha="center")
    _save(fig5, "05_size_vs_coverage")

    # ── Fig 6: Summary Table ──
    fig6, ax6 = plt.subplots(figsize=(14, 8))
    ax6.axis("off")
    headers = ["Class", "Instances", "Area\nMedian", "Area\nMean",
              "FG/Tile\nMedian", "FG\nRatio%", "Tiles\nw/class", "P4\nvis?"]
    rows = []
    for cid in cls_ids:
        s = summary[cid]
        p4_vis = "YES" if s["area_median"] >= 256 else "NO"
        rows.append([
            s["name"].replace("_", "\n"),
            f"{s['n_bbox']:,}",
            f"{s['area_median']:.0f}",
            f"{s['area_mean']:.0f}",
            f"{s['fg_median']:.0f}",
            f"{s['fg_ratio_median']:.2f}%",
            str(s["n_tiles_with"]),
            p4_vis,
        ])
    table = ax6.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.3)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#333333")
        table[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=8)
    p4_col = len(headers) - 1
    for i, cid in enumerate(cls_ids):
        color = CLASS_COLORS.get(cid, "#888")
        table[(i+1, 0)].set_facecolor(color + "44")
        table[(i+1, 0)].set_text_props(fontweight="bold", fontsize=8)
        if rows[i][-1] == "NO":
            table[(i+1, p4_col)].set_facecolor("#FFE5E5")
            table[(i+1, p4_col)].set_text_props(color="#CC0000", fontweight="bold", fontsize=8)
    ax6.set_title("iSAID 15-Class Statistics Summary", fontsize=14, fontweight="bold", pad=15)
    _save(fig6, "06_summary_table")

    # ══════════════════════════════════════════════════════════════
    # Console summary | 控制台输出
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'Class':<22} {'Inst':>7} {'Area_med':>9} {'Area_avg':>9} "
          f"{'FG_med':>8} {'FG%':>6} {'Tiles':>6} {'P4':>4}")
    print("-" * 80)
    for cid in cls_ids:
        s = summary[cid]
        p4 = "Y" if s["area_median"] >= 256 else "N"
        print(f"{s['name']:<22} {s['n_bbox']:>7,} {s['area_median']:>9.0f} {s['area_mean']:>9.0f} "
              f"{s['fg_median']:>8.0f} {s['fg_ratio_median']:>5.2f}% {s['n_tiles_with']:>6} {p4:>4}")

    # Count classes below P4 threshold
    p4_no = [s for cid, s in summary.items() if s["area_median"] < 256]
    print(f"\nClasses below P4 visibility threshold (median < 256 px^2): "
          f"{len(p4_no)}/15: {[s['name'] for s in p4_no]}")


if __name__ == "__main__":
    main()
