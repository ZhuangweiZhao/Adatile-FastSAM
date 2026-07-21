#!/usr/bin/env python3
"""
Severstal 数据集全面分析与可视化 | Severstal Dataset Comprehensive Analysis.
==============================================================================

对 Kaggle Severstal 钢铁缺陷检测数据集进行 8 维分析，生成论文级图表。
8-dimension analysis of the Kaggle Severstal steel defect dataset, generating paper-ready figures.

分析维度 | Analysis Dimensions:
    1. 数据集概览 — 图像数、类别分布、表格摘要
    2. 缺陷面积分布 — 每类缺陷的像素面积直方图
    3. 空间热力图 — 缺陷在钢带上的空间位置偏好
    4. 每类样本展示 — 4 类缺陷 + 干净样本的典型示例
    5. RLE 编码统计 — 每段游程长度、每缺陷段数的分布
    6. FG 占比分布 — 前景像素占比的直方图 (极端不平衡证据)
    7. 多类别共现分析 — 单类/双类/三类的图像统计
    8. 图像亮度/对比度 — 钢带表面亮度分布

用法 | Usage::

    python tools/diag/diag_severstal_analysis.py
    python tools/diag/diag_severstal_analysis.py --data-root data/severstal-steel-defect-detection
    python tools/diag/diag_severstal_analysis.py --output-dir analysis/severstal --num-samples 8
"""

from __future__ import annotations

import sys, argparse, os, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# ── CJK 字体配置 | CJK Font Configuration ──
_CJK_FONT = None
for _fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    for _f in fm.fontManager.ttflist:
        if _fname.lower() in _f.name.lower():
            _CJK_FONT = _f.name
            break
    if _CJK_FONT:
        break
if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.severstal import _decode_rle


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 256, 1600  # 原始图像尺寸 | Native image dimensions
CLASS_NAMES = {1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}
CLASS_COLORS = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#F39C12"}  # 红蓝绿橙


# ═══════════════════════════════════════════════════════════════════
# 数据加载 | Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_dataset(data_root: str) -> dict:
    """
    加载并解析 Severstal 数据集 | Load and parse Severstal dataset.

    Returns dict with:
        df: 原始 DataFrame
        images: 所有图像文件名列表
        defect_images: 有缺陷的图像集合
        clean_images: 无缺陷的图像集合 (在文件夹但不在 CSV 中)
        annotations: {image_id: {class_id: rle_string}}
        masks_cache: {image_id: {class_id: np.ndarray}}  (惰性填充)
    """
    root = Path(data_root)
    csv_path = root / "train.csv"
    img_dir = root / "train_images"

    print(f"Loading train.csv from {csv_path}...")
    df = pd.read_csv(str(csv_path))
    print(f"  {len(df)} rows loaded")

    # 所有图像
    all_images = sorted([f.name for f in img_dir.glob("*.jpg")])
    csv_images = set(df["ImageId"].unique())
    clean_images = sorted(list(set(all_images) - csv_images))

    # 构建标注字典
    annotations: dict[str, dict[int, str]] = defaultdict(dict)
    for _, row in df.iterrows():
        img_id = row["ImageId"]
        cls_id = int(row["ClassId"])
        rle = str(row["EncodedPixels"]) if pd.notna(row["EncodedPixels"]) else ""
        if rle:
            annotations[img_id][cls_id] = rle

    print(f"  Images: {len(all_images)} total ({len(csv_images)} defective + {len(clean_images)} clean)")
    print(f"  Defect annotations: {len(annotations)} images")

    return {
        "df": df,
        "all_images": all_images,
        "defect_images": sorted(csv_images),
        "clean_images": clean_images,
        "annotations": dict(annotations),
        "img_dir": img_dir,
    }


def get_mask(annotations: dict, img_id: str, cls_id: int,
             mask_cache: dict) -> np.ndarray:
    """获取指定图像和类别的二值掩码 (带缓存)."""
    cache_key = (img_id, cls_id)
    if cache_key in mask_cache:
        return mask_cache[cache_key]
    rle = annotations.get(img_id, {}).get(cls_id, "")
    mask = _decode_rle(rle) if rle else np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask_cache[cache_key] = mask
    return mask


# ═══════════════════════════════════════════════════════════════════
# 分析函数 | Analysis Functions
# ═══════════════════════════════════════════════════════════════════

# ── 1. 数据集概览表格 | Dataset Overview Table ──

def analyze_overview(data: dict) -> dict:
    """生成概览统计 | Generate overview statistics."""
    df = data["df"]
    annotations = data["annotations"]
    all_imgs = data["all_images"]
    defect_imgs = data["defect_images"]
    clean_imgs = data["clean_images"]

    stats = {
        "total_images": len(all_imgs),
        "defect_images": len(defect_imgs),
        "clean_images": len(clean_imgs),
        "defect_ratio": round(len(defect_imgs) / len(all_imgs) * 100, 1),
        "total_annotations": len(df),
        "num_classes": 4,
        "image_size": f"{IMG_W}×{IMG_H}",
        "image_size_pixels": IMG_H * IMG_W,
    }

    # Per-class stats
    per_class = {}
    for cls_id in range(1, 5):
        cls_df = df[df["ClassId"] == cls_id]
        n_images = cls_df["ImageId"].nunique()
        n_annotations = len(cls_df)
        per_class[f"class_{cls_id}"] = {
            "name": CLASS_NAMES[cls_id],
            "annotations": n_annotations,
            "images": n_images,
            "avg_per_image": round(n_annotations / max(n_images, 1), 2),
        }
    stats["per_class"] = per_class

    # Multi-class
    img_classes = defaultdict(set)
    for img_id, ann in annotations.items():
        img_classes[img_id] = set(ann.keys())
    n_class_dist = Counter(len(v) for v in img_classes.values())
    stats["multi_class_dist"] = {f"{k}_class": v for k, v in sorted(n_class_dist.items())}

    return stats


# ── 2. 缺陷面积分布 | Defect Area Distribution ──

def analyze_defect_areas(data: dict) -> dict:
    """计算每类缺陷的面积分布 | Compute defect area distribution per class."""
    annotations = data["annotations"]
    mask_cache = {}

    areas_per_class = {c: [] for c in range(1, 5)}
    total_pixels = IMG_H * IMG_W

    print("  Computing defect areas...")
    for img_id, ann in tqdm(annotations.items(), desc="Area calc"):
        for cls_id in ann:
            mask = get_mask(annotations, img_id, cls_id, mask_cache)
            area_px = int(mask.sum())
            areas_per_class[cls_id].append(area_px)

    stats = {}
    for cls_id in range(1, 5):
        areas = np.array(areas_per_class[cls_id])
        stats[f"class_{cls_id}"] = {
            "name": CLASS_NAMES[cls_id],
            "count": len(areas),
            "mean_px": round(float(areas.mean()), 1),
            "median_px": round(float(np.median(areas)), 1),
            "std_px": round(float(areas.std()), 1),
            "min_px": int(areas.min()),
            "max_px": int(areas.max()),
            "mean_pct": round(float(areas.mean() / total_pixels * 100), 4),
            "p25_px": round(float(np.percentile(areas, 25)), 1),
            "p75_px": round(float(np.percentile(areas, 75)), 1),
            "p95_px": round(float(np.percentile(areas, 95)), 1),
        }

    return {"per_class": stats, "raw_areas": areas_per_class}


# ── 3. 空间热力图 | Spatial Heatmap ──

def compute_spatial_heatmap(data: dict) -> np.ndarray:
    """计算所有缺陷的空间叠加热力图 | Compute spatial superposition heatmap."""
    annotations = data["annotations"]
    mask_cache = {}

    heatmap = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    per_class_heat = {c: np.zeros((IMG_H, IMG_W), dtype=np.float64) for c in range(1, 5)}

    print("  Computing spatial heatmaps...")
    for img_id, ann in tqdm(annotations.items(), desc="Heatmap", total=len(annotations)):
        for cls_id in ann:
            mask = get_mask(annotations, img_id, cls_id, mask_cache)
            heatmap += mask
            per_class_heat[cls_id] += mask

    # 归一化到 [0, 1]
    heatmap_norm = heatmap / max(heatmap.max(), 1)
    per_class_norm = {c: h / max(h.max(), 1) for c, h in per_class_heat.items()}

    return heatmap_norm, per_class_norm


# ── 4. RLE 编码统计 | RLE Encoding Statistics ──

def analyze_rle_stats(data: dict) -> dict:
    """分析 RLE 编码的段数和游程长度 | Analyze RLE segment count and run lengths."""
    annotations = data["annotations"]

    segments_per_class = {c: [] for c in range(1, 5)}  # 每个 mask 有多少段
    run_lengths_per_class = {c: [] for c in range(1, 5)}  # 每段多长

    print("  Analyzing RLE statistics...")
    for img_id, ann in tqdm(annotations.items(), desc="RLE stats"):
        for cls_id, rle_str in ann.items():
            numbers = [int(x) for x in rle_str.split()]
            n_segments = len(numbers) // 2
            segments_per_class[cls_id].append(n_segments)
            for i in range(0, len(numbers), 2):
                run_lengths_per_class[cls_id].append(numbers[i + 1])

    stats = {}
    for cls_id in range(1, 5):
        segs = np.array(segments_per_class[cls_id])
        runs = np.array(run_lengths_per_class[cls_id])
        stats[f"class_{cls_id}"] = {
            "name": CLASS_NAMES[cls_id],
            "segments_mean": round(float(segs.mean()), 1),
            "segments_median": round(float(np.median(segs)), 1),
            "segments_max": int(segs.max()),
            "run_length_mean": round(float(runs.mean()), 1),
            "run_length_median": round(float(np.median(runs)), 1),
            "run_length_max": int(runs.max()),
            "total_runs": len(runs),
        }

    return {"per_class": stats, "raw_segments": segments_per_class, "raw_runs": run_lengths_per_class}


# ── 5. FG 占比分布 | FG Ratio Distribution ──

def analyze_fg_ratios(data: dict) -> dict:
    """计算所有图像的 FG 占比分布 | Compute FG ratio distribution."""
    annotations = data["annotations"]
    mask_cache = {}

    total_pixels = IMG_H * IMG_W
    ratios_with_defect = []  # 有缺陷图像
    ratios_all = []  # 所有图像

    print("  Computing FG ratios...")
    for img_id, ann in tqdm(annotations.items(), desc="FG ratio"):
        combined = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for cls_id in ann:
            mask = get_mask(annotations, img_id, cls_id, mask_cache)
            combined = np.maximum(combined, mask)
        ratio = combined.sum() / total_pixels
        ratios_with_defect.append(ratio)
        ratios_all.append(ratio)

    # 无缺陷图像 (ratio = 0)
    n_clean = len(data["clean_images"])
    ratios_all.extend([0.0] * n_clean)

    ratios_with_defect = np.array(ratios_with_defect)
    ratios_all = np.array(ratios_all)

    return {
        "all_mean": round(float(ratios_all.mean()) * 100, 4),
        "all_median": round(float(np.median(ratios_all)) * 100, 4),
        "all_max": round(float(ratios_all.max()) * 100, 2),
        "with_defect_mean": round(float(ratios_with_defect.mean()) * 100, 4),
        "with_defect_median": round(float(np.median(ratios_with_defect)) * 100, 4),
        "zero_ratio_pct": round(float((ratios_all < 1e-8).mean()) * 100, 1),
        "raw": ratios_all,
    }


# ── 6. 图像亮度/对比度 | Image Brightness & Contrast ──

def analyze_image_stats(data: dict, max_samples: int = 2000) -> dict:
    """分析图像亮度和对比度分布."""
    img_dir = data["img_dir"]
    all_images = data["all_images"]

    # 采样
    rng = np.random.RandomState(42)
    if len(all_images) > max_samples:
        sampled = rng.choice(all_images, max_samples, replace=False)
    else:
        sampled = all_images

    means = []
    stds = []
    print(f"  Analyzing image brightness (sampled {len(sampled)} images)...")
    for fname in tqdm(sampled, desc="Brightness"):
        img = cv2.imread(str(img_dir / fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            means.append(float(img.mean()))
            stds.append(float(img.std()))

    means = np.array(means)
    stds = np.array(stds)

    return {
        "brightness_mean": round(float(means.mean()), 1),
        "brightness_std": round(float(means.std()), 1),
        "brightness_min": round(float(means.min()), 1),
        "brightness_max": round(float(means.max()), 1),
        "contrast_mean": round(float(stds.mean()), 1),
        "contrast_std": round(float(stds.std()), 1),
        "raw_means": means,
        "raw_stds": stds,
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def make_figure_overview(stats: dict, save_dir: Path):
    """图 1: 数据集概览 | Figure 1: Dataset Overview."""
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # ── 左: 有缺陷 vs 干净 饼图 ──
    ax1 = fig.add_subplot(gs[0])
    sizes = [stats["defect_images"], stats["clean_images"]]
    labels = [f"Defective\n({stats['defect_images']:,})",
              f"Clean\n({stats['clean_images']:,})"]
    colors = ["#E74C3C", "#95A5A6"]
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, explode=(0.02, 0),
            textprops={"fontsize": 11, "fontweight": "bold"})
    ax1.set_title("Image Composition\n图像构成", fontsize=13, fontweight="bold", pad=15)

    # ── 中: 类别分布柱状图 ──
    ax2 = fig.add_subplot(gs[1])
    pc = stats["per_class"]
    cls_names = [pc[f"class_{c}"]["name"] for c in range(1, 5)]
    cls_counts = [pc[f"class_{c}"]["annotations"] for c in range(1, 5)]
    cls_colors = [CLASS_COLORS[c] for c in range(1, 5)]
    bars = ax2.bar(cls_names, cls_counts, color=cls_colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, cls_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Annotation Count", fontsize=11)
    ax2.set_title("Class Distribution\n类别分布", fontsize=13, fontweight="bold", pad=10)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── 右: 多类别共现 ──
    ax3 = fig.add_subplot(gs[2])
    mc = stats["multi_class_dist"]
    mc_labels = [k.replace("_", " ") for k in mc.keys()]
    mc_vals = list(mc.values())
    bar_colors = ["#2C3E50", "#3498DB", "#9B59B6"]
    bars = ax3.barh(mc_labels, mc_vals, color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, mc_vals):
        ax3.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({val/stats['defect_images']*100:.1f}%)",
                va="center", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Image Count", fontsize=11)
    ax3.set_title("Multi-Class Co-occurrence\n多类别共现", fontsize=13, fontweight="bold", pad=10)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Severstal Dataset — Overview | 数据集概览",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_01_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_01_overview.png")


def make_figure_areas(area_stats: dict, save_dir: Path):
    """图 2: 缺陷面积分布 | Figure 2: Defect Area Distribution."""
    raw_areas = area_stats["raw_areas"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, cls_id in enumerate(range(1, 5)):
        ax = axes[idx]
        areas = np.array(raw_areas[cls_id])
        color = CLASS_COLORS[cls_id]

        # 对数直方图
        bins = np.logspace(np.log10(max(1, areas.min())), np.log10(areas.max()), 50)
        ax.hist(areas, bins=bins, color=color, alpha=0.75, edgecolor="black", linewidth=0.5)

        # 标注统计量
        ax.axvline(np.median(areas), color="red", linestyle="--", linewidth=2,
                  label=f"Median: {np.median(areas):,.0f} px")
        ax.axvline(np.mean(areas), color="blue", linestyle="--", linewidth=2,
                  label=f"Mean: {np.mean(areas):,.0f} px")

        ax.set_xscale("log")
        ax.set_xlabel("Defect Area (pixels) | 缺陷面积", fontsize=10)
        ax.set_ylabel("Frequency | 频次", fontsize=10)
        ax.set_title(f"{CLASS_NAMES[cls_id]} (n={len(areas)})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

        # 添加文本统计
        textstr = (f"Min: {areas.min():,}\nMax: {areas.max():,}\n"
                  f"P95: {np.percentile(areas, 95):,.0f}\n"
                  f"Pct of image: {areas.mean()/(IMG_H*IMG_W)*100:.4f}%")
        props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=props, family="monospace")

    fig.suptitle("Severstal — Defect Area Distribution | 缺陷面积分布",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_02_defect_areas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_02_defect_areas.png")


def make_figure_spatial(heatmap_all: np.ndarray, per_class_heat: dict, save_dir: Path):
    """图 3: 空间热力图 | Figure 3: Spatial Heatmap."""
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1])

    # ── 全局热力图 ──
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(heatmap_all, cmap="hot", aspect="auto")
    ax0.set_title("All Classes Combined\n全类别合并", fontsize=10, fontweight="bold")
    ax0.set_xlabel("Width (px) | 宽度", fontsize=9)
    ax0.set_ylabel("Height (px) | 高度", fontsize=9)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # ── 各类别热力图 ──
    for idx, cls_id in enumerate(range(1, 5)):
        ax = fig.add_subplot(gs[idx + 1])
        im = ax.imshow(per_class_heat[cls_id], cmap="hot", aspect="auto")
        ax.set_title(f"{CLASS_NAMES[cls_id]}", fontsize=10, fontweight="bold",
                    color=CLASS_COLORS[cls_id])
        ax.set_xlabel("Width (px)", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Height (px)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Severstal — Spatial Defect Distribution Heatmap | 缺陷空间分布热力图\n"
                "(Brighter = more defects at that location | 越亮 = 该位置缺陷越多)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_03_spatial_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_03_spatial_heatmap.png")


def make_figure_samples(data: dict, num_samples: int, save_dir: Path):
    """图 4: 每类典型样本 | Figure 4: Per-class Typical Samples."""
    annotations = data["annotations"]
    img_dir = data["img_dir"]
    mask_cache = {}

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(5, num_samples, figure=fig)

    # 每类找面积中位数附近的样本
    for row_idx, cls_id in enumerate(range(1, 5)):
        # 收集该类所有样本
        cls_samples = []
        for img_id, ann in annotations.items():
            if cls_id in ann:
                mask = get_mask(annotations, img_id, cls_id, mask_cache)
                area = int(mask.sum())
                cls_samples.append((img_id, area, mask))

        # 按面积排序，选等距样本
        cls_samples.sort(key=lambda x: x[1])
        n = len(cls_samples)
        if n <= num_samples:
            selected = cls_samples
        else:
            indices = np.linspace(0, n - 1, num_samples, dtype=int)
            selected = [cls_samples[i] for i in indices]

        for col_idx, (img_id, area, mask) in enumerate(selected):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # 加载图像
            img = cv2.imread(str(img_dir / img_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 叠加掩码
            overlay = img.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.4 + np.array(
                [231, 76, 60] if cls_id == 1 else
                [52, 152, 219] if cls_id == 2 else
                [46, 204, 113] if cls_id == 3 else
                [243, 156, 18]
            ).reshape(1, 1, 3) * 0.6

            ax.imshow(overlay.astype(np.uint8))
            ax.set_title(f"Area: {area:,} px", fontsize=8)
            ax.axis("off")

            # 行标签
            if col_idx == 0:
                ax.set_ylabel(f"{CLASS_NAMES[cls_id]}\n(n={n})", fontsize=10,
                            fontweight="bold", color=CLASS_COLORS[cls_id],
                            rotation=0, labelpad=50, va="center")

    # ── 第 5 行: 干净样本 ──
    clean_images = data["clean_images"]
    rng = np.random.RandomState(42)
    selected_clean = rng.choice(clean_images, min(num_samples, len(clean_images)), replace=False)
    for col_idx, img_id in enumerate(selected_clean):
        ax = fig.add_subplot(gs[4, col_idx])
        img = cv2.imread(str(img_dir / img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title("Clean | 无缺陷", fontsize=8)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(f"Clean\n(n={len(clean_images):,})", fontsize=10,
                        fontweight="bold", color="#95A5A6",
                        rotation=0, labelpad=50, va="center")

    fig.suptitle("Severstal — Per-Class Typical Samples | 每类典型样本\n"
                "(Columns: min → max defect area | 列: 缺陷面积从小到大)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_04_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_04_samples.png")


def make_figure_rle(rle_stats: dict, save_dir: Path):
    """图 5: RLE 编码统计 | Figure 5: RLE Encoding Statistics."""
    raw_segments = rle_stats["raw_segments"]
    raw_runs = rle_stats["raw_runs"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # ── 上排: 段数分布 ──
    for idx, cls_id in enumerate(range(1, 5)):
        ax = axes[0, idx]
        segs = np.array(raw_segments[cls_id])
        color = CLASS_COLORS[cls_id]
        ax.hist(segs, bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.median(segs), color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{CLASS_NAMES[cls_id]}\nSegments per Mask\n(median={np.median(segs):.0f})",
                    fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Num Segments", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # ── 下排: 游程长度分布 ──
    for idx, cls_id in enumerate(range(1, 5)):
        ax = axes[1, idx]
        runs = np.array(raw_runs[cls_id])
        color = CLASS_COLORS[cls_id]
        # log bins
        if runs.max() > 0:
            bins = np.logspace(0, np.log10(max(1, runs.max())), 50)
            ax.hist(runs, bins=bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xscale("log")
        ax.axvline(np.median(runs), color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{CLASS_NAMES[cls_id]}\nRun Length (px)\n(median={np.median(runs):.0f})",
                    fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Run Length (log scale)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Severstal — RLE Encoding Statistics | RLE 编码统计\n"
                "(Top: segments per mask | Bottom: run length per segment)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_05_rle_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_05_rle_stats.png")


def make_figure_fg_ratio(fg_stats: dict, save_dir: Path):
    """图 6: FG 占比分布 | Figure 6: FG Ratio Distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ratios = fg_stats["raw"]

    # ── 左: 直方图 (log scale) ──
    ax1 = axes[0]
    nonzero = ratios[ratios > 1e-8]
    zero_pct = fg_stats["zero_ratio_pct"]
    ax1.hist(nonzero * 100, bins=100, color="#E74C3C", alpha=0.7,
            edgecolor="black", linewidth=0.3)
    ax1.axvline(np.median(ratios) * 100, color="blue", linestyle="--", linewidth=2,
               label=f"Median: {np.median(ratios)*100:.4f}%")
    ax1.axvline(np.mean(ratios) * 100, color="green", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(ratios)*100:.4f}%")
    ax1.set_xlabel("FG Ratio (%) | 前景占比", fontsize=11)
    ax1.set_ylabel("Frequency | 频次", fontsize=11)
    ax1.set_title(f"FG Ratio Distribution (Defective Only)\n前景占比分布 (仅缺陷图)\n"
                 f"Clean images (0%): {zero_pct:.1f}% of total",
                 fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 右: CDF (累积分布) ──
    ax2 = axes[1]
    sorted_ratios = np.sort(ratios) * 100
    cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
    ax2.plot(sorted_ratios, cdf, color="#2C3E50", linewidth=2)
    ax2.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="50th percentile")
    ax2.axhline(90, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="90th percentile")
    # 标注
    p50 = np.percentile(ratios, 50) * 100
    p90 = np.percentile(ratios, 90) * 100
    p99 = np.percentile(ratios, 99) * 100
    ax2.axvline(p50, color="blue", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(p90, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(p99, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.annotate(f"P50={p50:.4f}%", xy=(p50, 50), fontsize=8, color="blue",
                rotation=90, va="bottom")
    ax2.annotate(f"P90={p90:.4f}%", xy=(p90, 50), fontsize=8, color="orange",
                rotation=90, va="bottom")
    ax2.annotate(f"P99={p99:.3f}%", xy=(p99, 50), fontsize=8, color="red",
                rotation=90, va="bottom")
    ax2.set_xlabel("FG Ratio (%) | 前景占比", fontsize=11)
    ax2.set_ylabel("Cumulative % | 累积百分比", fontsize=11)
    ax2.set_title("Cumulative Distribution (All Images)\n累积分布 (全部图像)",
                 fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Severstal — Foreground Ratio Analysis | 前景占比分析\n"
                f"(Extreme imbalance: mean FG = {ratios.mean()*100:.4f}%)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_06_fg_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_06_fg_ratio.png")


def make_figure_brightness(brightness_stats: dict, save_dir: Path):
    """图 7: 图像亮度/对比度分布 | Figure 7: Image Brightness & Contrast."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    means = brightness_stats["raw_means"]
    stds = brightness_stats["raw_stds"]

    # ── 左: 亮度直方图 ──
    ax1 = axes[0]
    ax1.hist(means, bins=40, color="#3498DB", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axvline(np.mean(means), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(means):.1f}")
    ax1.set_xlabel("Mean Pixel Value | 平均像素值", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title("Brightness Distribution\n亮度分布", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 中: 对比度直方图 ──
    ax2 = axes[1]
    ax2.hist(stds, bins=40, color="#2ECC71", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axvline(np.mean(stds), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(stds):.1f}")
    ax2.set_xlabel("Std Dev | 标准差", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("Contrast Distribution\n对比度分布", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── 右: 亮度 vs 对比度 散点图 ──
    ax3 = axes[2]
    # 采样以避免散点太密
    n_plot = min(1000, len(means))
    idx = np.random.RandomState(42).choice(len(means), n_plot, replace=False)
    ax3.scatter(means[idx], stds[idx], c="#E74C3C", alpha=0.4, s=8, edgecolors="none")
    ax3.set_xlabel("Mean Pixel Value | 平均像素值", fontsize=10)
    ax3.set_ylabel("Std Dev | 标准差", fontsize=10)
    ax3.set_title("Brightness vs Contrast\n亮度 vs 对比度", fontsize=11, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Severstal — Image Quality Statistics | 图像质量统计",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_07_brightness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_07_brightness.png")


def make_figure_summary(data: dict, stats: dict, area_stats: dict,
                       fg_stats: dict, rle_stats: dict, save_dir: Path):
    """图 8: 综合摘要大图 | Figure 8: Summary Figure."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── (0,0): 关键数字网格 ──
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    key_numbers = [
        ("Total Images\n总图像数", f"{stats['total_images']:,}"),
        ("Defective\n有缺陷", f"{stats['defect_images']:,} ({stats['defect_ratio']}%)"),
        ("Clean\n无缺陷", f"{stats['clean_images']:,} ({100-stats['defect_ratio']}%)"),
        ("Image Size\n图像尺寸", f"{IMG_W}×{IMG_H}"),
        ("Defect Classes\n缺陷类别", "4"),
        ("FG Ratio (mean)\n前景占比均值", f"{stats.get('fg_mean', 'N/A')}"),
    ]
    for i, (label, value) in enumerate(key_numbers):
        row, col = divmod(i, 2)
        y = 0.8 - row * 0.4
        x = 0.05 + col * 0.5
        ax0.text(x, y, label, transform=ax0.transAxes, fontsize=9, va="top",
                ha="left", color="gray", fontweight="bold")
        ax0.text(x, y - 0.12, value, transform=ax0.transAxes, fontsize=16, va="top",
                ha="left", color="#2C3E50", fontweight="bold")
    ax0.set_title("Key Numbers | 关键数字", fontsize=12, fontweight="bold", pad=10)

    # ── (0,1): 类别分布 ──
    ax1 = fig.add_subplot(gs[0, 1])
    pc = stats["per_class"]
    cls_names = [pc[f"class_{c}"]["name"] for c in range(1, 5)]
    cls_anns = [pc[f"class_{c}"]["annotations"] for c in range(1, 5)]
    cls_imgs = [pc[f"class_{c}"]["images"] for c in range(1, 5)]
    x = np.arange(4)
    w = 0.35
    bars1 = ax1.bar(x - w/2, cls_anns, w, label="Annotations", color="#E74C3C",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, cls_imgs, w, label="Images", color="#3498DB",
                   edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x, cls_names, fontsize=9)
    ax1.set_title("Per-Class Distribution\n每类分布", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── (0,2): 多类别共现 ──
    ax2 = fig.add_subplot(gs[0, 2])
    mc = stats["multi_class_dist"]
    mc_labels = [k.replace("_", " ").title() for k in mc.keys()]
    mc_vals = list(mc.values())
    ax2.bar(mc_labels, mc_vals, color=["#2C3E50", "#3498DB", "#9B59B6"],
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(mc_vals):
        ax2.text(i, v + 30, str(v), ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("Multi-Class Co-occurrence\n多类别共现", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── (1,0): 面积分布 (箱线图) ──
    ax3 = fig.add_subplot(gs[1, 0])
    area_data = [np.array(area_stats["raw_areas"][c]) for c in range(1, 5)]
    bp = ax3.boxplot(area_data, labels=[CLASS_NAMES[c] for c in range(1, 5)],
                     patch_artist=True, showfliers=False)
    for patch, c_id in zip(bp["boxes"], range(1, 5)):
        patch.set_facecolor(CLASS_COLORS[c_id])
        patch.set_alpha(0.6)
    ax3.set_ylabel("Area (pixels)", fontsize=9)
    ax3.set_title("Defect Area (Boxplot, w/o outliers)\n缺陷面积 (箱线图, 不含离群值)",
                 fontsize=10, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)

    # ── (1,1): FG 占比 ──
    ax4 = fig.add_subplot(gs[1, 1])
    fg_data = fg_stats["raw"] * 100
    nonzero = fg_data[fg_data > 1e-8]
    ax4.hist(nonzero, bins=80, color="#E74C3C", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax4.set_xlabel("FG Ratio (%)", fontsize=9)
    ax4.set_title(f"FG Ratio (mean={fg_data.mean():.4f}%)\n前景占比分布",
                 fontsize=10, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)

    # ── (1,2): RLE 段数 ──
    ax5 = fig.add_subplot(gs[1, 2])
    seg_data = [np.array(rle_stats["raw_segments"][c]) for c in range(1, 5)]
    bp2 = ax5.boxplot(seg_data, labels=[CLASS_NAMES[c] for c in range(1, 5)],
                      patch_artist=True, showfliers=False)
    for patch, c_id in zip(bp2["boxes"], range(1, 5)):
        patch.set_facecolor(CLASS_COLORS[c_id])
        patch.set_alpha(0.6)
    ax5.set_ylabel("Segments per Mask", fontsize=9)
    ax5.set_title("RLE Segments per Defect\n每缺陷 RLE 段数",
                 fontsize=10, fontweight="bold")
    ax5.spines[["top", "right"]].set_visible(False)

    # ── (2,:): 空间热力图 ──
    heatmap_all = data.get("heatmap_all")
    if heatmap_all is not None:
        # 沿宽度方向投影
        ax_bottom = fig.add_subplot(gs[2, :])
        w_profile = heatmap_all.mean(axis=0)  # 沿高度平均
        ax_bottom.fill_between(range(IMG_W), w_profile, alpha=0.6, color="#E74C3C")
        ax_bottom.plot(w_profile, color="#C0392B", linewidth=1)
        ax_bottom.set_xlabel("Width Position (px) | 宽度位置", fontsize=10)
        ax_bottom.set_ylabel("Defect Density | 缺陷密度", fontsize=10)
        ax_bottom.set_title("Defect Density Along Width (Averaged over Height)\n"
                           "沿宽度方向的缺陷密度 (高度方向取平均)",
                           fontsize=11, fontweight="bold")
        # 标注峰谷
        peak_x = np.argmax(w_profile)
        ax_bottom.axvline(peak_x, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax_bottom.annotate(f"Peak @ {peak_x}px", xy=(peak_x, w_profile[peak_x]),
                          fontsize=8, color="red")
        ax_bottom.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Severstal Dataset — Comprehensive Analysis | 综合分析摘要\n"
                f"Total: {stats['total_images']:,} images, {stats['defect_images']:,} "
                f"defective ({stats['defect_ratio']}%), {stats['clean_images']:,} clean",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "severstal_08_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: severstal_08_summary.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Severstal Dataset Analysis")
    parser.add_argument("--data-root", type=str,
                       default="data/severstal-steel-defect-detection")
    parser.add_argument("--output-dir", type=str, default="analysis/severstal")
    parser.add_argument("--num-samples", type=int, default=6,
                       help="每类展示的样本数 | Number of samples per class")
    parser.add_argument("--skip-heavy", action="store_true",
                       help="跳过耗时的分析 (亮度统计)")
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Severstal Dataset — Comprehensive Analysis")
    print(f"  Data:  {args.data_root}")
    print(f"  Output: {save_dir}")
    print("=" * 60)

    # ── Step 0: 加载数据 ──
    print("\n[1/7] Loading dataset...")
    data = load_dataset(args.data_root)

    # ── Step 1: 概览 ──
    print("\n[2/7] Overview statistics...")
    overview_stats = analyze_overview(data)

    # ── Step 2: 缺陷面积 ──
    print("\n[3/7] Defect area analysis...")
    area_stats = analyze_defect_areas(data)

    # ── Step 3: 空间热力图 ──
    print("\n[4/7] Spatial heatmap...")
    heatmap_all, per_class_heat = compute_spatial_heatmap(data)
    data["heatmap_all"] = heatmap_all  # for summary figure

    # ── Step 4: RLE 编码统计 ──
    print("\n[5/7] RLE encoding statistics...")
    rle_stats = analyze_rle_stats(data)

    # ── Step 5: FG 占比 ──
    print("\n[6/7] FG ratio analysis...")
    fg_stats = analyze_fg_ratios(data)
    overview_stats["fg_mean"] = f"{fg_stats['all_mean']}%"

    # ── Step 6: 亮度 ──
    if not args.skip_heavy:
        print("\n[7/7] Image brightness/contrast...")
        brightness_stats = analyze_image_stats(data)
    else:
        brightness_stats = None

    # ═══════════════════════════════════════════════════════════════
    # 可视化 | Visualization
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  Generating figures...")
    print("=" * 60)

    print("\n  Figure 1: Overview")
    make_figure_overview(overview_stats, save_dir)

    print("\n  Figure 2: Defect Areas")
    make_figure_areas(area_stats, save_dir)

    print("\n  Figure 3: Spatial Heatmap")
    make_figure_spatial(heatmap_all, per_class_heat, save_dir)

    print("\n  Figure 4: Per-Class Samples")
    make_figure_samples(data, args.num_samples, save_dir)

    print("\n  Figure 5: RLE Statistics")
    make_figure_rle(rle_stats, save_dir)

    print("\n  Figure 6: FG Ratio")
    make_figure_fg_ratio(fg_stats, save_dir)

    if brightness_stats is not None:
        print("\n  Figure 7: Brightness/Contrast")
        make_figure_brightness(brightness_stats, save_dir)

    print("\n  Figure 8: Summary")
    make_figure_summary(data, overview_stats, area_stats, fg_stats, rle_stats, save_dir)

    # ═══════════════════════════════════════════════════════════════
    # 保存 JSON 统计 | Save JSON Stats
    # ═══════════════════════════════════════════════════════════════

    stats_output = {
        "overview": {k: v for k, v in overview_stats.items() if k != "fg_mean"},
        "defect_areas": {k: v for k, v in area_stats["per_class"].items()},
        "rle": {k: v for k, v in rle_stats["per_class"].items()},
        "fg_ratio": {k: v for k, v in fg_stats.items() if k != "raw"},
    }
    if brightness_stats is not None:
        stats_output["brightness"] = {
            k: v for k, v in brightness_stats.items()
            if not k.startswith("raw_")
        }

    stats_path = save_dir / "severstal_analysis.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Stats saved: {stats_path}")

    # ── 打印关键发现 ──
    print("\n" + "=" * 60)
    print("  Key Findings | 关键发现")
    print("=" * 60)
    print(f"  数据集构成: {overview_stats['defect_images']:,} 有缺陷 + "
          f"{overview_stats['clean_images']:,} 无缺陷 = "
          f"{overview_stats['total_images']:,} 总计")
    print(f"  类别不平衡: Class3 占绝对主导 ({overview_stats['per_class']['class_3']['annotations']} 标注), "
          f"Class2 极稀少 ({overview_stats['per_class']['class_2']['annotations']} 标注)")
    print(f"  多类别重叠: {overview_stats['multi_class_dist'].get('1_class', 0):,} 单类, "
          f"{overview_stats['multi_class_dist'].get('2_class', 0):,} 双类")
    print(f"  极端 FG/BG 不平衡: FG 仅占 {fg_stats['all_mean']}% 像素")
    print(f"  缺陷面积: Class1 median={area_stats['per_class']['class_1']['median_px']:,.0f}px, "
          f"Class3 median={area_stats['per_class']['class_3']['median_px']:,.0f}px")
    print(f"  图像一致性: 所有图像均为 {IMG_W}×{IMG_H}, 无需 resize")
    print(f"\n  Figures saved to: {save_dir}")
    print(f"  Stats saved to: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
