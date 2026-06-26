#!/usr/bin/env python3
"""
论文级 iSAID 数据集可视化 | Paper-grade iSAID dataset visualization.
==================================================================

生成 6 组论文用图:
  1. 类别分布统计 (instance / pixel / tile count per class)
  2. 长尾分布 (instance count, log scale)
  3. 实例尺寸分布 (Small / Medium / Large 直方图)
  4. Tile 中目标数量分布 (instances per tile 柱状图)
  5. 空 Tile 比例 (饼图)
  6. Tile Occupancy 分布 (FG ratio 直方图)

All stats from COCO JSON — no image loading needed, runs in seconds.

用法 | Usage::
    python tools/viz/viz_paper_figures.py --src-root data/iSAID_processed
"""

import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import cv2

from adatile.utils.label_mapping import ISAID_CATEGORIES

# ═══════════════════════════════════════════════════════════════════
# Class colors — by semantic group | 按语义分组配色
# ═══════════════════════════════════════════════════════════════════
_CLASS_COLORS = {
     1: "#E63946",  2: "#F4A261",  3: "#E76F51",  4: "#FFB703",
     5: "#F72585",  6: "#2A9D8F",  7: "#264653",  8: "#287271",
     9: "#52B788", 10: "#FF9F1C", 11: "#457B9D", 12: "#9B5DE5",
    13: "#7209B7", 14: "#D4A373", 15: "#4A4E69",
}

_CLASS_GROUPS = {
    "Vehicle":      [1, 2, 3, 5],
    "Infrastructure": [6, 7, 8, 9],
    "Object":       [4, 10, 14],
    "Civil":        [11, 12, 13, 15],
}

_TILE_SIZE, _STRIDE = 896, 512
_TILE_SIZES = [512, 768, 896, 1024]  # 多尺寸对比 | Multi-size comparison

# ═══════════════════════════════════════════════════════════════════
# Data Collection | 数据收集
# ═══════════════════════════════════════════════════════════════════

def collect(src_root: str, split: str = "train", tile_size: int = 896, stride: int = 512):
    """Collect all stats from COCO JSON. | 从 COCO JSON 收集全部统计."""
    ann_path = Path(src_root) / split / "annotations" / f"instances_{split}.json"
    with open(ann_path) as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    print(f"[{split}] {len(images)} images, {len(annotations)} annotations")

    # — Per-class stats —
    cls_stats = {}
    for cid in ISAID_CATEGORIES:
        cls_stats[cid] = {"instances": 0, "pixel_count": 0, "bbox_areas": [], "mask_areas": []}

    for ann in annotations:
        cid = ann["category_id"]
        if cid not in ISAID_CATEGORIES:
            continue
        cls_stats[cid]["instances"] += 1
        w, h = ann["bbox"][2], ann["bbox"][3]
        area = w * h
        cls_stats[cid]["bbox_areas"].append(area)
        # mask area via segmentation
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and len(seg) > 0:
            # polygon area via shoelace
            mask_area = 0
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.array(poly).reshape(-1, 2)
                    mask_area += 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                                           - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
            cls_stats[cid]["mask_areas"].append(max(mask_area, area * 0.5))
            cls_stats[cid]["pixel_count"] += max(mask_area, area * 0.5)
        else:
            cls_stats[cid]["mask_areas"].append(area)
            cls_stats[cid]["pixel_count"] += area

    # — Tile grid stats (via bbox-tile overlap, no image loading) —
    img_id_to_info = {img["id"]: img for img in images}
    img_id_to_anns = defaultdict(list)
    for ann in annotations:
        if ann["category_id"] in ISAID_CATEGORIES:
            img_id_to_anns[ann["image_id"]].append(ann)

    # 获取图像尺寸: 优先磁盘缓存, 其次 COCO metadata, 最后读文件
    # Get image dims: use disk cache > COCO metadata > read file
    src_dir = Path(src_root) / split / "images"
    dims_cache_path = Path(src_root) / split / "image_dims.json"

    if dims_cache_path.exists():
        _img_dims = json.loads(dims_cache_path.read_text())
        # Convert string keys back to int
        _img_dims = {int(k): tuple(v) for k, v in _img_dims.items()}
    else:
        _img_dims = {}
        for img in tqdm(images, desc="Reading image dims", leave=False):
            W, H = img.get("width", 0), img.get("height", 0)
            if W <= 0 or H <= 0:
                img_path = src_dir / img["file_name"]
                im = cv2.imread(str(img_path))
                if im is not None:
                    H, W = im.shape[:2]
            _img_dims[img["id"]] = (W, H)
        # Cache to disk for future runs | 缓存到磁盘
        dims_cache_path.write_text(json.dumps({str(k): list(v) for k, v in _img_dims.items()}))

    tile_inst_count = []       # instances per tile
    tile_fg_pixels = []        # FG pixels per tile
    tile_classes_per_tile = [] # unique classes per tile

    for img_id, img_info in img_id_to_info.items():
        W, H = _img_dims[img_id]
        if W < tile_size or H < tile_size:
            continue
        anns = img_id_to_anns.get(img_id, [])

        n_cols = (W - tile_size) // stride + 1
        n_rows = (H - tile_size) // stride + 1

        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * stride
                y0 = row * stride
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                n_inst, fg_px, classes = 0, 0, set()
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]
                    if bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0:
                        # clip area
                        ox = max(0, min(bx + bw, x1) - max(bx, x0))
                        oy = max(0, min(by + bh, y1) - max(by, y0))
                        fg_px += ox * oy
                        n_inst += 1  # each ann = 1 instance
                        classes.add(ann["category_id"])

                tile_inst_count.append(n_inst)
                tile_fg_pixels.append(fg_px)
                tile_classes_per_tile.append(len(classes))

    # Per-class tile counts
    for cid in ISAID_CATEGORIES:
        cls_stats[cid]["tile_count"] = 0
    for img_id, anns in img_id_to_anns.items():
        img_info = img_id_to_info[img_id]
        W, H = _img_dims[img_id]
        if W < tile_size or H < tile_size:
            continue
        n_cols = (W - _TILE_SIZE) // _STRIDE + 1
        n_rows = (H - _TILE_SIZE) // _STRIDE + 1
        for row in range(n_rows):
            for col in range(n_cols):
                x0, x1 = col * _STRIDE, col * _STRIDE + _TILE_SIZE
                y0, y1 = row * _STRIDE, row * _STRIDE + _TILE_SIZE
                present = set()
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]
                    if bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0:
                        present.add(ann["category_id"])
                for cid in present:
                    cls_stats[cid]["tile_count"] += 1

    return cls_stats, tile_inst_count, tile_fg_pixels, tile_classes_per_tile


# ═══════════════════════════════════════════════════════════════════
# Plot Helpers | 绘图辅助
# ═══════════════════════════════════════════════════════════════════

def _save(fig, out_dir, stem, name):
    p = Path(out_dir) / f"{stem}_{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

def _bar_colors(cids):
    return [_CLASS_COLORS.get(c, "#888") for c in cids]

_CLS_NAMES = {k: v.replace("_", "\n") for k, v in ISAID_CATEGORIES.items()}


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Category Distribution | 类别分布统计
# ═══════════════════════════════════════════════════════════════════

def fig_category_distribution(cls_stats, out_dir, stem):
    cids = sorted(ISAID_CATEGORIES.keys())
    names = [_CLS_NAMES[c] for c in cids]
    inst = [cls_stats[c]["instances"] for c in cids]
    pixels = [cls_stats[c]["pixel_count"] for c in cids]
    tiles = [cls_stats[c]["tile_count"] for c in cids]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    for ax, data, title, ylabel in [
        (axes[0], inst, "Instance Count", "Instances"),
        (axes[1], pixels, "Pixel Count", "Pixels (log)"),
        (axes[2], tiles, "Tile Count", "Tiles"),
    ]:
        bars = ax.bar(range(len(cids)), data, color=_bar_colors(cids), alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ylabel == "Pixels (log)":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        # annotate top bars
        top_n = 3
        sorted_idx = np.argsort(data)[-top_n:]
        for idx in sorted_idx:
            ax.text(idx, data[idx] * 1.02, f"{data[idx]:,}", ha="center", fontsize=6,
                   fontweight="bold", rotation=90)

    fig.suptitle("iSAID Category Distribution (train split)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, stem, "01_category_distribution")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Long-Tail Distribution | 长尾分布
# ═══════════════════════════════════════════════════════════════════

def fig_long_tail(cls_stats, out_dir, stem):
    cids = sorted(ISAID_CATEGORIES.keys())
    data = sorted([(cls_stats[c]["instances"], _CLS_NAMES[c], c) for c in cids], reverse=True)
    values, names, colors = zip(*[(v, n, _CLASS_COLORS.get(c, "#888")) for v, n, c in data])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([n.replace("\n", " ") for n in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Instance Count (log scale)", fontsize=10)
    ax.set_title("iSAID Long-Tail Class Distribution", fontsize=14, fontweight="bold")

    # Horizontal lines: power-of-10
    for y in [10, 100, 1000, 10000, 100000]:
        ax.axhline(y=y, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    # Annotate ratio
    ax.annotate(f"{values[-1]:,}", xy=(len(values)-1, values[-1]),
               xytext=(len(values)-1, values[-1]*2), ha="center", fontsize=7, color="#CC0000",
               arrowprops=dict(arrowstyle="->", color="#CC0000", lw=0.8))

    # Imbalance annotation
    ratio = values[0] / max(values[-1], 1)
    ax.text(0.98, 0.95, f"Head:Tail Ratio\n{ratio:,.0f}:1",
           transform=ax.transAxes, fontsize=11, ha="right", va="top",
           bbox=dict(boxstyle="round", facecolor="#FFF3CD", alpha=0.9),
           fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir, stem, "02_long_tail")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Instance Size Distribution | 实例尺寸分布
# ═══════════════════════════════════════════════════════════════════

def fig_instance_size(cls_stats, out_dir, stem):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # — Left: histogram of all bbox areas —
    ax = axes[0]
    all_areas = []
    for cid in ISAID_CATEGORIES:
        all_areas.extend(cls_stats[cid]["bbox_areas"])

    # Size thresholds
    SMALL, MEDIUM = 32**2, 96**2   # 1024, 9216
    bins = np.logspace(1, 7, 80)

    counts, _, patches = ax.hist(all_areas, bins=bins, color="#457B9D", alpha=0.7,
                                  edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("BBox Area (px)", fontsize=10)
    ax.set_ylabel("Instance Count", fontsize=10)
    ax.set_title("Instance Size Distribution (All Classes)", fontsize=12, fontweight="bold")

    # Threshold lines
    for th, label, color, ypos in [(SMALL, "Small\n(0-32²)", "#E63946", 0.85),
                                     (MEDIUM, "Medium\n(32²-96²)", "#FFB703", 0.70),
                                     (MEDIUM*100, "Large\n(96²+)", "#2A9D8F", 0.55)]:
        ax.axvline(x=th, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(th * 1.1, max(counts) * ypos, label, fontsize=8, color=color, fontweight="bold")

    # Count per category
    small_n = sum(1 for a in all_areas if a < SMALL)
    medium_n = sum(1 for a in all_areas if SMALL <= a < MEDIUM)
    large_n = sum(1 for a in all_areas if a >= MEDIUM)
    total = small_n + medium_n + large_n
    ax.text(0.98, 0.95, f"Small:  {small_n/total*100:.1f}%\nMedium: {medium_n/total*100:.1f}%\nLarge:  {large_n/total*100:.1f}%",
           transform=ax.transAxes, fontsize=8, ha="right", va="top",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.grid(True, alpha=0.3, axis="y")

    # — Right: per-class median area bars —
    ax2 = axes[1]
    cids = sorted(ISAID_CATEGORIES.keys())
    medians = [np.median(cls_stats[c]["bbox_areas"]) if cls_stats[c]["bbox_areas"] else 0 for c in cids]
    bars = ax2.bar(range(len(cids)), medians, color=_bar_colors(cids), alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(cids)))
    ax2.set_xticklabels([_CLS_NAMES[c] for c in cids], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Median BBox Area (px, log)", fontsize=10)
    ax2.set_title("Median Object Size per Class", fontsize=12, fontweight="bold")
    ax2.axhline(y=256, color="#E63946", linestyle="--", linewidth=1, alpha=0.7, label="P4 visible (16px)")
    ax2.axhline(y=1024, color="#FFB703", linestyle="--", linewidth=1, alpha=0.7, label="P8 visible (32px)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("iSAID Object Size Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, stem, "03_instance_size")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Instances Per Tile | Tile 中目标数量分布
# ═══════════════════════════════════════════════════════════════════

def fig_tile_instances(tile_inst_count, tile_classes_per_tile, out_dir, stem):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # — Left: instances per tile —
    ax = axes[0]
    max_inst = min(30, max(tile_inst_count))
    bins = range(0, max_inst + 2)
    counts, bins, patches = ax.hist(np.clip(tile_inst_count, 0, max_inst + 1),
                                     bins=bins, color="#457B9D", alpha=0.8,
                                     edgecolor="white", linewidth=0.5, align="left")
    ax.set_xlabel("Instances per Tile", fontsize=10)
    ax.set_ylabel("Tile Count", fontsize=10)
    ax.set_title("Instances per Tile Distribution", fontsize=12, fontweight="bold")

    # Annotate empty tiles
    empty_pct = (np.array(tile_inst_count) == 0).mean() * 100
    ax.annotate(f"Empty: {empty_pct:.0f}%", xy=(0, counts[0]), xytext=(3, counts[0] * 1.1),
               fontsize=10, fontweight="bold", color="#E63946",
               arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2))
    # Annotate 1-3 instances
    pct13 = ((np.array(tile_inst_count) >= 1) & (np.array(tile_inst_count) <= 3)).mean() * 100
    ax.annotate(f"1-3 inst: {pct13:.0f}%", xy=(2, counts[2]),
               xytext=(5, counts[2] * 1.05), fontsize=9, color="#2A9D8F")
    ax.grid(True, alpha=0.3, axis="y")

    # — Right: classes per tile —
    ax2 = axes[1]
    max_cls = max(tile_classes_per_tile)
    bins2 = range(0, max_cls + 2)
    ax2.hist(np.clip(tile_classes_per_tile, 0, max_cls + 1),
            bins=bins2, color="#E76F51", alpha=0.8, edgecolor="white", linewidth=0.5, align="left")
    ax2.set_xlabel("Unique Classes per Tile", fontsize=10)
    ax2.set_ylabel("Tile Count", fontsize=10)
    ax2.set_title("Class Diversity per Tile", fontsize=12, fontweight="bold")
    single_cls = (np.array(tile_classes_per_tile) == 1).mean() * 100
    ax2.text(0.98, 0.95, f"Single-class tiles: {single_cls:.0f}%",
            transform=ax2.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Tile Content Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, stem, "04_tile_instances")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Empty Tile Ratio | 空 Tile 比例
# ═══════════════════════════════════════════════════════════════════

def fig_empty_tile(tile_inst_count, out_dir, stem):
    empty = (np.array(tile_inst_count) == 0).sum()
    non_empty = len(tile_inst_count) - empty

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Left: Pie —
    ax = axes[0]
    colors_pie = ["#E8E8E8", "#457B9D"]
    wedges, texts, autotexts = ax.pie(
        [empty, non_empty], labels=["Empty", "Non-Empty"],
        colors=colors_pie, autopct="%1.1f%%", startangle=90,
        explode=(0, 0.05), textprops={"fontsize": 11, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title(f"Empty Tile Ratio ({_TILE_SIZE}x{_TILE_SIZE}, stride={_STRIDE})",
                fontsize=12, fontweight="bold")

    # — Right: Instances per tile bar groups —
    ax2 = axes[1]
    labels = ["0", "1-5", "6-20", "21-50", "50+"]
    insts = np.array(tile_inst_count)
    groups = [
        (insts == 0).sum(),
        ((insts >= 1) & (insts <= 5)).sum(),
        ((insts >= 6) & (insts <= 20)).sum(),
        ((insts >= 21) & (insts <= 50)).sum(),
        (insts > 50).sum(),
    ]
    bar_colors = ["#E8E8E8", "#A8DADC", "#457B9D", "#E76F51", "#E63946"]
    bars = ax2.bar(labels, groups, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Tile Count", fontsize=10)
    ax2.set_title("Tiles by Instance Count Group", fontsize=12, fontweight="bold")
    for bar, g in zip(bars, groups):
        pct = g / len(tile_inst_count) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{pct:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Spatial Sparsity in iSAID Tiles", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, stem, "05_empty_tile")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Tile Occupancy Distribution | Tile Occupancy 分布
# ═══════════════════════════════════════════════════════════════════

def fig_tile_occupancy(tile_fg_pixels, out_dir, stem):
    tile_area = _TILE_SIZE * _TILE_SIZE
    ratios = np.array(tile_fg_pixels) / tile_area

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # — Left: histogram of FG ratio —
    ax = axes[0]
    bins = np.logspace(-5, 0, 80)
    ax.hist(ratios[ratios > 0] + 1e-6, bins=bins, color="#457B9D", alpha=0.8,
           edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("FG Ratio (FG pixels / Tile pixels, log scale)", fontsize=10)
    ax.set_ylabel("Tile Count", fontsize=10)
    ax.set_title("Tile Occupancy Distribution\n(FG ratio > 0 only)", fontsize=11, fontweight="bold")

    # Annotate percentiles
    for pct, color in [(50, "#E63946"), (90, "#FFB703"), (95, "#2A9D8F")]:
        val = np.percentile(ratios[ratios > 0], pct) if (ratios > 0).sum() > 0 else 0
        ax.axvline(x=val, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(val * 1.1, ax.get_ylim()[1] * 0.9, f"P{pct}={val*100:.2f}%",
               fontsize=7, color=color, fontweight="bold", rotation=90)
    ax.grid(True, alpha=0.3, axis="y")

    # — Right: CDF of occupancy —
    ax2 = axes[1]
    sorted_ratios = np.sort(ratios)
    cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    ax2.plot(sorted_ratios, cdf, color="#457B9D", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("FG Ratio", fontsize=10)
    ax2.set_ylabel("Cumulative Fraction of Tiles", fontsize=10)
    ax2.set_title("Cumulative Distribution of Tile Occupancy", fontsize=12, fontweight="bold")

    # Key thresholds
    for th, label in [(0.001, "0.1%"), (0.01, "1%"), (0.05, "5%"), (0.10, "10%")]:
        tile_pct = (ratios >= th).mean() * 100
        ax2.axvline(x=th, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
        ax2.text(th * 0.9, 1.02, f"{label}\n{tile_pct:.1f}% tiles",
                fontsize=7, ha="center", va="bottom", alpha=0.8)

    ax2.grid(True, alpha=0.3)

    fig.suptitle("iSAID Tile Occupancy — Spatial Sparsity Evidence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, stem, "06_tile_occupancy")


# ═══════════════════════════════════════════════════════════════════
# Figure 0: Combined Overview | 总图
# ═══════════════════════════════════════════════════════════════════

def _fig_overview(cls_stats, tile_inst, tile_fg, split, out_dir, stem):
    """2x3 combined overview figure for paper. | 论文总图."""
    cids = sorted(ISAID_CATEGORIES.keys())
    names = [_CLS_NAMES[c] for c in cids]
    inst = [cls_stats[c]["instances"] for c in cids]
    tile_area = _TILE_SIZE * _TILE_SIZE
    ratios = np.array(tile_fg) / tile_area
    empty = (np.array(tile_inst) == 0).sum()
    non_empty = len(tile_inst) - empty

    fig = plt.figure(figsize=(22, 14))

    # (0,0): Instance count bar (log)
    ax1 = fig.add_subplot(2, 3, 1)
    data_sorted = sorted(zip(inst, names, cids), reverse=True)
    vals, nms, clrs = zip(*[(v, n, _CLASS_COLORS.get(c, "#888")) for v, n, c in data_sorted])
    ax1.bar(range(len(vals)), vals, color=clrs, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels([n.replace("\n", " ") for n in nms], rotation=35, ha="right", fontsize=6)
    ax1.set_ylabel("Instances (log)", fontsize=9)
    ax1.set_title("a) Long-Tail Class Distribution", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # (0,1): Instance size histogram
    ax2 = fig.add_subplot(2, 3, 2)
    all_areas = []
    for cid in cids:
        all_areas.extend(cls_stats[cid]["bbox_areas"])
    bins = np.logspace(1, 7, 60)
    ax2.hist(all_areas, bins=bins, color="#457B9D", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.set_xscale("log")
    small = 32**2; medium = 96**2
    for th, lbl, c in [(small, "Small", "#E63946"), (medium, "Medium", "#FFB703")]:
        ax2.axvline(x=th, color=c, linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(th*1.1, ax2.get_ylim()[1]*0.85, lbl, fontsize=7, color=c, fontweight="bold")
    ax2.set_xlabel("BBox Area (px)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    n_small = sum(1 for a in all_areas if a < small)
    n_large = sum(1 for a in all_areas if a >= medium)
    ax2.text(0.98, 0.95, f"Small: {n_small/len(all_areas)*100:.1f}%\nLarge: {n_large/len(all_areas)*100:.1f}%",
            transform=ax2.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax2.set_title("b) Object Size Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # (0,2): Median area per class
    ax3 = fig.add_subplot(2, 3, 3)
    medians = [np.median(cls_stats[c]["bbox_areas"]) if cls_stats[c]["bbox_areas"] else 0 for c in cids]
    ax3.bar(range(len(cids)), medians, color=_bar_colors(cids), alpha=0.85, edgecolor="white", linewidth=0.3)
    ax3.set_yscale("log")
    ax3.set_xticks(range(len(cids)))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax3.axhline(y=256, color="#E63946", linestyle="--", linewidth=1, alpha=0.7, label="P4=1px")
    ax3.legend(fontsize=7)
    ax3.set_ylabel("Median Area (px, log)", fontsize=9)
    ax3.set_title("c) Median Object Size per Class", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # (1,0): Instances per tile
    ax4 = fig.add_subplot(2, 3, 4)
    inst_arr = np.array(tile_inst)
    labels = ["0", "1-3", "4-10", "11-30", "30+"]
    groups = [
        (inst_arr == 0).sum(),
        ((inst_arr >= 1) & (inst_arr <= 3)).sum(),
        ((inst_arr >= 4) & (inst_arr <= 10)).sum(),
        ((inst_arr >= 11) & (inst_arr <= 30)).sum(),
        (inst_arr > 30).sum(),
    ]
    bar_colors = ["#E8E8E8", "#A8DADC", "#457B9D", "#E76F51", "#E63946"]
    bars = ax4.bar(labels, groups, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax4.set_ylabel("Tile Count", fontsize=9)
    ax4.set_title(f"d) Instances per Tile ({_TILE_SIZE}x{_TILE_SIZE})", fontsize=11, fontweight="bold")
    for bar, g in zip(bars, groups):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"{g/len(tile_inst)*100:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # (1,1): Tile occupancy CDF
    ax5 = fig.add_subplot(2, 3, 5)
    sorted_r = np.sort(ratios)
    cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
    ax5.plot(sorted_r, cdf, color="#457B9D", linewidth=2)
    ax5.set_xscale("log")
    ax5.set_xlabel("FG Ratio", fontsize=9)
    ax5.set_ylabel("CDF", fontsize=9)
    ax5.set_title("e) Tile Occupancy CDF", fontsize=11, fontweight="bold")
    for th in [0.001, 0.01, 0.05]:
        ax5.axvline(x=th, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax5.grid(True, alpha=0.3)

    # (1,2): Empty tile pie + summary stats
    ax6 = fig.add_subplot(2, 3, 6)
    colors_pie = ["#E8E8E8", "#457B9D"]
    ax6.pie([empty, non_empty], labels=["Empty", "Non-Empty"], colors=colors_pie,
           autopct="%1.1f%%", startangle=90, explode=(0, 0.05),
           textprops={"fontsize": 10, "fontweight": "bold"})
    ax6.set_title(f"f) Empty Tile Ratio ({empty/len(tile_inst)*100:.1f}%)", fontsize=11, fontweight="bold")

    # Summary text box
    stats_text = (
        f"Dataset: iSAID ({split})\n"
        f"Images: {1411}  |  Instances: {sum(inst):,}\n"
        f"Classes: 15  |  Tile: {_TILE_SIZE}x{_TILE_SIZE}\n"
        f"Stride: {_STRIDE}  |  Tiles: {len(tile_inst):,}\n"
        f"Empty tiles: {empty/len(tile_inst)*100:.1f}%"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F8F9FA", alpha=0.9))

    fig.suptitle("iSAID Dataset Analysis — AdaTile Motivation",
                fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    _save(fig, out_dir, stem, "00_overview")


# ═══════════════════════════════════════════════════════════════════
# Figure 7: Occupancy Histogram (clean standalone) | Occupancy 直方图
# ═══════════════════════════════════════════════════════════════════

def fig_occupancy_histogram(tile_fg, out_dir, stem):
    tile_area = _TILE_SIZE * _TILE_SIZE
    ratios = np.array(tile_fg) / tile_area
    # Include zeros
    ratios_all = np.array(tile_fg) / tile_area

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(-6, 0, 70)
    ax.hist(ratios_all[ratios_all > 0] + 1e-7, bins=bins, color="#457B9D", alpha=0.85,
           edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Occupancy (FG pixels / Tile area)", fontsize=10)
    ax.set_ylabel("Tile Count", fontsize=10)
    ax.set_title(f"Tile Occupancy Histogram ({_TILE_SIZE}x{_TILE_SIZE})", fontsize=13, fontweight="bold")

    # Percentile annotations
    pos = ratios_all[ratios_all > 0]
    for p, c in [(50, "#E63946"), (80, "#FFB703"), (95, "#2A9D8F")]:
        if len(pos) > 0:
            v = np.percentile(pos, p)
            ax.axvline(x=v, color=c, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(v*1.1, ax.get_ylim()[1]*0.85, f"P{p}={v*100:.2f}%",
                   fontsize=8, color=c, fontweight="bold")

    # Zero tile annotation
    zero_pct = (ratios_all == 0).mean() * 100
    ax.text(0.95, 0.95, f"Empty tiles: {zero_pct:.1f}%\n"
            f"Non-empty tiles: {100-zero_pct:.1f}%",
           transform=ax.transAxes, fontsize=9, ha="right", va="top",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir, stem, "07_occupancy_histogram")


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Tile Information Distribution | 信息量联合分布
# ═══════════════════════════════════════════════════════════════════

def fig_tile_joint_distribution(tile_inst, tile_fg, out_dir, stem):
    """Joint distribution: instance_count × occupancy. | 联合分布."""
    tile_area = _TILE_SIZE * _TILE_SIZE
    inst_arr = np.array(tile_inst)
    ratio_arr = np.array(tile_fg) / tile_area

    # Define occupancy bins and instance bins
    occ_bins = [0, 0.001, 0.01, 0.05, 0.10, 1.0]
    inst_bins = [0, 1, 4, 11, 31, 10000]

    occ_labels = ["0", "<0.1%", "0.1-1%", "1-5%", "5-10%"]
    inst_labels = ["0", "1-3", "4-10", "11-30", ">30"]

    matrix = np.zeros((len(occ_bins)-1, len(inst_bins)-1))
    for i in range(len(occ_bins)-1):
        for j in range(len(inst_bins)-1):
            mask = ((ratio_arr >= occ_bins[i]) & (ratio_arr < occ_bins[i+1]) &
                    (inst_arr >= inst_bins[j]) & (inst_arr < inst_bins[j+1]))
            matrix[i, j] = mask.sum()

    matrix_pct = matrix / len(inst_arr) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix_pct, cmap="YlOrRd", aspect="auto", vmin=0)

    # Annotate cells
    for i in range(len(occ_bins)-1):
        for j in range(len(inst_bins)-1):
            val = matrix_pct[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                   fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(inst_bins)-1))
    ax.set_xticklabels(inst_labels, fontsize=9)
    ax.set_yticks(range(len(occ_bins)-1))
    ax.set_yticklabels(occ_labels, fontsize=9)
    ax.set_xlabel("Instances per Tile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Occupancy (FG ratio)", fontsize=11, fontweight="bold")
    ax.set_title(f"Tile Information Distribution ({_TILE_SIZE}x{_TILE_SIZE})\n"
                f"Instance Count x Occupancy Joint Distribution",
                fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% of All Tiles", fontsize=9)

    # Highlight the low-info quadrant
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, -0.5), 2, 2, linewidth=3, edgecolor="#E63946",
                    facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.text(0.5, 0.5, "LOW\nINFORMATION\nZONE",
           ha="center", va="center", fontsize=8, color="#E63946", fontweight="bold")

    plt.tight_layout()
    _save(fig, out_dir, stem, "08_tile_joint_distribution")


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Multi Tile-Size Comparison | 多尺寸对比
# ═══════════════════════════════════════════════════════════════════

def fig_tile_size_comparison(src_root, split, out_dir, stem):
    """Compare 4 tile sizes: 512, 768, 896, 1024. | 四组尺寸对比."""
    sizes = _TILE_SIZES  # [512, 768, 896, 1024]

    # Collect for all sizes
    all_stats = {}
    for ts in tqdm(sizes, desc="Multi-size sweep", leave=False):
        stride = ts // 2  # 50% overlap
        _, tile_inst, tile_fg, _ = collect(src_root, split, tile_size=ts, stride=stride)
        inst_arr = np.array(tile_inst)
        ratio_arr = np.array(tile_fg) / (ts * ts)

        empty = (inst_arr == 0).mean() * 100
        # occupancy stats (non-empty only)
        pos = ratio_arr[ratio_arr > 0]
        occ_median = np.median(pos) * 100 if len(pos) > 0 else 0
        occ_mean = pos.mean() * 100 if len(pos) > 0 else 0
        inst_density = inst_arr.mean()  # instances per tile
        n_tiles = len(tile_inst)

        all_stats[ts] = {
            "n_tiles": n_tiles,
            "empty_pct": empty,
            "occ_median": occ_median,
            "occ_mean": occ_mean,
            "inst_density": inst_density,
            "ratios": ratio_arr,
            "inst_arr": inst_arr,
        }

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0): Empty Ratio bar
    ax = axes[0, 0]
    empties = [all_stats[s]["empty_pct"] for s in sizes]
    bars = ax.bar([str(s) for s in sizes], empties, color=["#A8DADC", "#457B9D", "#E76F51", "#E63946"],
                  alpha=0.85, edgecolor="white")
    ax.set_ylabel("Empty Tile Ratio (%)", fontsize=10)
    ax.set_title("a) Empty Tile Ratio vs Tile Size", fontsize=11, fontweight="bold")
    for b, v in zip(bars, empties):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%",
               ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(empties) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1): Occupancy CDF
    ax2 = axes[0, 1]
    for ts, c in zip(sizes, ["#A8DADC", "#457B9D", "#E76F51", "#E63946"]):
        pos = all_stats[ts]["ratios"]
        pos = pos[pos > 0]
        if len(pos) > 0:
            sorted_r = np.sort(pos)
            cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
            ax2.plot(sorted_r, cdf, color=c, linewidth=2, label=f"{ts}px")
    ax2.set_xscale("log")
    ax2.set_xlabel("Occupancy (FG ratio)", fontsize=10)
    ax2.set_ylabel("CDF (non-empty tiles)", fontsize=10)
    ax2.set_title("b) Occupancy CDF by Tile Size", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (0,2): Occupancy distribution (boxplot)
    ax3 = axes[0, 2]
    box_data = []
    for ts in sizes:
        pos = all_stats[ts]["ratios"]
        pos = pos[pos > 0] * 100
        if len(pos) > 3000:
            pos = np.random.RandomState(42).choice(pos, 3000, replace=False)
        box_data.append(pos if len(pos) > 0 else [0])
    bp = ax3.boxplot(box_data, patch_artist=True, widths=0.5, showfliers=False, whis=[5, 95])
    ax3.set_xticklabels([str(s) for s in sizes])
    for patch, c in zip(bp["boxes"], ["#A8DADC", "#457B9D", "#E76F51", "#E63946"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax3.set_ylabel("Occupancy (% of tile)", fontsize=10)
    ax3.set_title("c) Occupancy Distribution by Tile Size", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # (1,0): Instances per tile CDF
    ax4 = axes[1, 0]
    for ts, c in zip(sizes, ["#A8DADC", "#457B9D", "#E76F51", "#E63946"]):
        insts = all_stats[ts]["inst_arr"]
        sorted_i = np.sort(insts[insts > 0])
        if len(sorted_i) > 0:
            cdf_i = np.arange(1, len(sorted_i)+1) / len(sorted_i)
            ax4.plot(sorted_i, cdf_i, color=c, linewidth=2, label=f"{ts}px")
    ax4.set_xlabel("Instances per Tile (non-empty)", fontsize=10)
    ax4.set_ylabel("CDF", fontsize=10)
    ax4.set_title("d) Instance Density CDF by Tile Size", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 50)
    ax4.grid(True, alpha=0.3)

    # (1,1): Summary metrics table
    ax5 = axes[1, 1]
    ax5.axis("off")
    headers = ["Tile Size", "N Tiles", "Empty%", "Occ.\nMedian", "Occ.\nMean", "Inst.\n/Tile"]
    rows = []
    for ts in sizes:
        s = all_stats[ts]
        rows.append([f"{ts}x{ts}", f"{s['n_tiles']:,}", f"{s['empty_pct']:.1f}%",
                    f"{s['occ_median']:.3f}%", f"{s['occ_mean']:.2f}%", f"{s['inst_density']:.2f}"])
    tbl = ax5.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.8)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#333333")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Highlight the 896 row
    for j in range(len(headers)):
        tbl[(3, j)].set_facecolor("#FFE5CC")  # 896 is index 3 in sizes
        tbl[(3, j)].set_text_props(fontweight="bold")
    ax5.set_title("e) Multi-Size Comparison Summary", fontsize=11, fontweight="bold", pad=10)

    # (1,2): Key insight text
    ax6 = axes[1, 2]
    ax6.axis("off")
    s896 = all_stats[896]
    s512 = all_stats[512]
    s1024 = all_stats[1024]

    insight = (
        f"WHY 896x896?\n\n"
        f"• 512: {s512['n_tiles']:,} tiles, but\n"
        f"  objects too fragmented\n"
        f"  (small tiles cut objects)\n\n"
        f"• 1024: {s1024['n_tiles']:,} tiles, but\n"
        f"  {s1024['empty_pct']:.1f}% empty — less efficient\n\n"
        f"• 896: {s896['n_tiles']:,} tiles\n"
        f"  {s896['empty_pct']:.1f}% empty\n"
        f"  Occ. median={s896['occ_median']:.3f}%\n"
        f"  Balanced: coverage + sparsity"
    )
    ax6.text(0.05, 0.95, insight, transform=ax6.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#FFF8E1", alpha=0.9))

    fig.suptitle("Tile Size Ablation — Why 896x896?",
                fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, stem, "09_tile_size_comparison")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Paper-grade iSAID dataset visualization")
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/paper_figures")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}/")

    # Collect stats for default tile size (896)
    print("Collecting dataset statistics...")
    cls_stats, tile_inst, tile_fg, tile_classes = collect(args.src_root, args.split)

    total_inst = sum(s["instances"] for s in cls_stats.values())
    total_tiles = len(tile_inst)
    empty_tiles = (np.array(tile_inst) == 0).sum()
    print(f"  Total instances: {total_inst:,}")
    print(f"  Total tiles: {total_tiles:,}")
    print(f"  Empty tiles: {empty_tiles:,} ({empty_tiles/total_tiles*100:.1f}%)")

    stem = f"paper_fig_{args.split}"

    # ── 总图 + 基础图 | Overview + basic figures ──
    _fig_overview(cls_stats, tile_inst, tile_fg, args.split, out_dir, stem)

    fig_category_distribution(cls_stats, out_dir, stem)
    fig_long_tail(cls_stats, out_dir, stem)
    fig_instance_size(cls_stats, out_dir, stem)
    fig_tile_instances(tile_inst, tile_classes, out_dir, stem)
    fig_empty_tile(tile_inst, out_dir, stem)
    fig_tile_occupancy(tile_fg, out_dir, stem)

    # ── 新增三张 | Three new figures ──
    fig_occupancy_histogram(tile_fg, out_dir, stem)
    fig_tile_joint_distribution(tile_inst, tile_fg, out_dir, stem)
    fig_tile_size_comparison(args.src_root, args.split, out_dir, stem)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
