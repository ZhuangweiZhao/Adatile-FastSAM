#!/usr/bin/env python3
"""
可视化 iSAID 三类 (small_vehicle, storage_tank, ship) 的尺寸差异。
Visualize size differences across small_vehicle / storage_tank / ship.

用法 | Usage:
    python tools/viz/viz_class_sizes.py --src-root data/iSAID_processed
"""

import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 三类颜色 + 名称 | 3-class colors + names
TARGET = {
    1:  {"name": "small_vehicle",  "color": "#E63946", "display": "Small Vehicle"},
    4:  {"name": "storage_tank",   "color": "#2A9D8F", "display": "Storage Tank"},
    5:  {"name": "ship",           "color": "#457B9D", "display": "Ship"},
}


def collect_bbox_stats(src_root: str, split: str = "train"):
    """Collect bbox area statistics from COCO JSON per class. | 从 COCO JSON 收集 bbox 面积统计."""
    ann_path = Path(src_root) / split / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Cannot find annotation file: {ann_path}")

    with open(ann_path) as f:
        data = json.load(f)

    # Build image_id -> image info | 构建 image_id -> 图片信息
    images = {img["id"]: img for img in data["images"]}
    anns_by_cls = defaultdict(list)

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in TARGET:
            bbox = ann["bbox"]  # COCO format: [x, y, w, h]
            area_px = bbox[2] * bbox[3]  # pixel area | 像素面积
            anns_by_cls[cat_id].append({
                "area": area_px,
                "bbox": bbox,
                "image_id": ann["image_id"],
            })

    return anns_by_cls, images


def plot_class_samples(ax, anns_by_cls, images, cls_id, n_samples=4):
    """Plot sample masks + bboxes for one class. | 绘制某一类的样本 mask + bbox."""
    import cv2

    info = TARGET[cls_id]
    anns = anns_by_cls[cls_id]

    # Sort by area, pick extremes: smallest and largest | 按面积排序，取最小和最大
    sorted_anns = sorted(anns, key=lambda a: a["area"])
    indices = []
    if len(sorted_anns) >= n_samples:
        step = max(1, len(sorted_anns) // n_samples)
        indices = list(range(0, len(sorted_anns), step))[:n_samples]
    else:
        indices = list(range(len(sorted_anns)))

    for j, idx in enumerate(indices):
        ann = sorted_anns[idx]
        img_info = images.get(ann["image_id"])
        if img_info is None:
            continue

        img_path = Path(img_info.get("file_name", ""))
        if not img_path.is_absolute():
            img_path = Path(anns_by_cls.get("_src_root", ".")) / img_path

        # Skip if image not found — just show bbox stats | 如果图片找不到，只显示 bbox
        ax_sub = ax[j] if n_samples > 1 else ax
        bbox = ann["bbox"]

        # Draw a colored rectangle representing the bbox scale | 绘制代表 bbox 比例的彩色矩形
        max_size = max(bbox[2], bbox[3])
        scale = 1.0 / max(max_size, 1)  # normalize | 归一化
        rect = Rectangle((0.05, 0.1), bbox[2] * scale * 0.9, bbox[3] * scale * 0.9,
                         linewidth=2, edgecolor=info["color"], facecolor=info["color"] + "44")
        ax_sub.add_patch(rect)
        ax_sub.set_xlim(0, 1)
        ax_sub.set_ylim(0, 1)
        ax_sub.set_title(f"{bbox[0]:.0f}×{bbox[1]:.0f}  ({ann['area']:.0f}px^2)", fontsize=7)
        ax_sub.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualize class size differences")
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/viz_class_sizes.png")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    anns_by_cls, images = collect_bbox_stats(str(src_root), args.split)

    print(f"Collected annotations:")
    for cls_id in sorted(TARGET):
        n = len(anns_by_cls.get(cls_id, []))
        if n > 0:
            areas = [a["area"] for a in anns_by_cls[cls_id]]
            print(f"  Class {cls_id} ({TARGET[cls_id]['name']:>15}): {n:5d} instances, "
                  f"area: min={min(areas):.0f}  median={np.median(areas):.0f}  "
                  f"mean={np.mean(areas):.0f}  max={max(areas):.0f} px^2")

    # ══════════════════════════════════════════════════════════
    # Figure 1: Area Distribution + Sample Visualizations
    # ══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 10))

    # ── Row 1: Area distribution histogram | 面积分布直方图 ──
    ax_hist = fig.add_subplot(2, 3, (1, 3))
    bins = np.logspace(1, 7, 60)  # 10^1 to 10^7 px^2
    for cls_id in sorted(TARGET):
        areas = [a["area"] for a in anns_by_cls.get(cls_id, [])]
        if areas:
            ax_hist.hist(areas, bins=bins, alpha=0.6, label=TARGET[cls_id]["display"],
                        color=TARGET[cls_id]["color"], edgecolor="white", linewidth=0.3)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("BBox Area (px^2)", fontsize=11)
    ax_hist.set_ylabel("Instance Count", fontsize=11)
    ax_hist.set_title("Object Size Distribution (BBox Area)", fontsize=13, fontweight="bold")
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.3)

    # ── Row 2-3: Per-class bbox scale scatter | 每类 bbox 尺寸散点 + 统计 ──
    for i, cls_id in enumerate(sorted(TARGET)):
        ax = fig.add_subplot(2, 3, 4 + i)
        areas = np.array([a["area"] for a in anns_by_cls.get(cls_id, [])])
        bboxes = np.array([a["bbox"] for a in anns_by_cls.get(cls_id, [])])
        if len(areas) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TARGET[cls_id]["display"])
            continue

        widths = bboxes[:, 2]
        heights = bboxes[:, 3]

        scatter = ax.scatter(widths, heights, c=TARGET[cls_id]["color"], alpha=0.3,
                            s=np.clip(areas / 100, 1, 50), edgecolors="none")

        # Reference lines: aspect ratio 1:1, 2:1, 1:2
        max_dim = max(widths.max(), heights.max())
        ax.plot([0, max_dim], [0, max_dim], "k--", alpha=0.2, linewidth=0.5, label="1:1")
        ax.plot([0, max_dim/2], [0, max_dim], "k:", alpha=0.15, linewidth=0.5, label="1:2")

        ax.set_xlabel("Width (px)", fontsize=9)
        ax.set_ylabel("Height (px)", fontsize=9)
        ax.set_xlim(0, max_dim * 1.05)
        ax.set_ylim(0, max_dim * 1.05)

        # Statistics box | 统计信息框
        stats_text = (
            f"N={len(areas)}\n"
            f"median: {np.median(areas):.0f}px^2\n"
            f"mean: {np.mean(areas):.0f}px^2\n"
            f"w×h median: {np.median(widths):.0f}×{np.median(heights):.0f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        ax.set_title(TARGET[cls_id]["display"], fontsize=12, fontweight="bold",
                    color=TARGET[cls_id]["color"])
        ax.grid(True, alpha=0.2)

    plt.tight_layout(pad=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # ══════════════════════════════════════════════════════════
    # Figure 2: Summary comparison table | 汇总对比表
    # ══════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.axis("off")

    headers = ["Class", "Instances", "Area min", "Area median", "Area mean", "Area max",
               "W median", "H median", "Aspect Ratio"]
    rows = []
    for cls_id in sorted(TARGET):
        areas_arr = np.array([a["area"] for a in anns_by_cls.get(cls_id, [])])
        bboxes_arr = np.array([a["bbox"] for a in anns_by_cls.get(cls_id, [])])
        if len(areas_arr) == 0:
            rows.append([TARGET[cls_id]["display"]] + ["-"] * 8)
            continue
        w_med = np.median(bboxes_arr[:, 2])
        h_med = np.median(bboxes_arr[:, 3])
        ar = w_med / h_med if h_med > 0 else 0
        rows.append([
            TARGET[cls_id]["display"],
            f"{len(areas_arr):,}",
            f"{areas_arr.min():.0f}",
            f"{np.median(areas_arr):.0f}",
            f"{areas_arr.mean():.0f}",
            f"{areas_arr.max():.0f}",
            f"{w_med:.0f}",
            f"{h_med:.0f}",
            f"{ar:.2f}",
        ])

    table = ax2.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    # Color header row + class name cells
    for j, _ in enumerate(headers):
        table[(0, j)].set_facecolor("#333333")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, cls_id in enumerate(sorted(TARGET)):
        table[(i+1, 0)].set_facecolor(TARGET[cls_id]["color"] + "33")
        table[(i+1, 0)].set_text_props(fontweight="bold")

    ax2.set_title("iSAID Object Size Statistics (train split)", fontsize=14,
                 fontweight="bold", pad=20)

    table_path = str(output_path).replace(".png", "_table.png")
    fig2.savefig(table_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
