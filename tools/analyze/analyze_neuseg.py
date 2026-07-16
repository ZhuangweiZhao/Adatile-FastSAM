#!/usr/bin/env python3
"""
NEU_Seg 数据集分析工具 | NEU_Seg Dataset Analysis Tool.
=========================================================

分析新版 NEU_Seg 数据集的统计特性，生成可视化报告。
Analyze statistical properties of the new NEU_Seg dataset, generate visualization report.

分析内容 | Analysis Items:
    1. 数据集概览 (样本数, 图像尺寸, splits)
    2. FG 占比分布 (histogram, per-split stats)
    3. 图像统计 (亮度, 对比度, 色彩分布)
    4. 标注质量 (连通域, 目标尺寸)
    5. Train/Test 分布对比

用法 | Usage::

    python tools/analyze/analyze_neuseg.py
    python tools/analyze/analyze_neuseg.py --data-root data/NEU_Seg --output runs/neuseg_analysis
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from adatile.datasets.neu_seg import NEUSegDataset


# ═══════════════════════════════════════════════════════════════════
# 分析函数 | Analysis Functions
# ═══════════════════════════════════════════════════════════════════

def analyze_fg_distribution(ds: NEUSegDataset, split_name: str) -> dict:
    """
    FG 占比分布分析 | FG ratio distribution analysis.

    :return: dict with ratios array, stats dict, histogram bins.
    """
    print(f"  Analyzing FG distribution for {split_name} ({len(ds)} samples)...")
    ratios = []
    for i in tqdm(range(len(ds)), desc=f"  FG ratio [{split_name}]", unit="sample"):
        ratios.append(ds[i]["masks"].mean().item())

    arr = np.array(ratios)
    stats = {
        "n": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "empty": int((arr < 1e-6).sum()),
        "sparse": int(((arr > 0) & (arr < 0.001)).sum()),
        "low": int(((arr >= 0.001) & (arr < 0.01)).sum()),
        "mid": int(((arr >= 0.01) & (arr < 0.1)).sum()),
        "high": int((arr >= 0.1).sum()),
    }
    return {"ratios": arr, "stats": stats}


def analyze_image_stats(ds: NEUSegDataset, split_name: str, max_samples: int = 500) -> dict:
    """
    图像亮度/对比度/色彩统计 | Image brightness/contrast/color statistics.

    :param max_samples: 最多分析的样本数 (全量太慢) | Max samples to analyze.
    """
    n = min(len(ds), max_samples)
    print(f"  Analyzing image stats for {split_name} ({n} samples)...")

    means_list, stds_list = [], []
    for i in tqdm(range(n), desc=f"  Image stats [{split_name}]", unit="img"):
        img = ds[i]["image"].numpy()  # [3, H, W]
        means_list.append(img.mean(axis=(1, 2)))  # per-channel mean
        stds_list.append(img.std(axis=(1, 2)))    # per-channel std

    means = np.array(means_list)  # [N, 3]
    stds = np.array(stds_list)    # [N, 3]
    channel_names = ["R", "G", "B"]

    return {
        "n": n,
        "brightness_mean": {ch: float(means[:, i].mean()) for i, ch in enumerate(channel_names)},
        "brightness_std": {ch: float(means[:, i].std()) for i, ch in enumerate(channel_names)},
        "contrast_mean": {ch: float(stds[:, i].mean()) for i, ch in enumerate(channel_names)},
        "contrast_std": {ch: float(stds[:, i].std()) for i, ch in enumerate(channel_names)},
        "overall_brightness": float(means.mean()),
        "overall_contrast": float(stds.mean()),
        "means_per_channel": means,
        "stds_per_channel": stds,
    }


def analyze_object_sizes(ds: NEUSegDataset, split_name: str, max_samples: int = 500) -> dict:
    """
    连通域/目标尺寸分析 | Connected component / object size analysis.

    统计每个 mask 中的独立连通域数量和面积。
    Count independent connected components and their areas per mask.
    """
    n = min(len(ds), max_samples)
    print(f"  Analyzing object sizes for {split_name} ({n} samples)...")

    all_areas = []      # 所有连通域面积 | All component areas
    n_objects_per_img = []  # 每张图的目标数 | Objects per image
    empty_imgs = 0

    for i in tqdm(range(n), desc=f"  Object size [{split_name}]", unit="img"):
        mask = ds[i]["masks"].numpy().squeeze(0)  # [H, W]
        mask_bin = (mask > 0.5).astype(np.uint8)

        if mask_bin.sum() == 0:
            empty_imgs += 1
            n_objects_per_img.append(0)
            continue

        # 连通域分析 | Connected component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8
        )
        # stats[0] 是背景 | stats[0] is background
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            all_areas.append(int(area))
        n_objects_per_img.append(num_labels - 1)

    areas_arr = np.array(all_areas, dtype=np.float64)
    nobj_arr = np.array(n_objects_per_img)

    return {
        "n_samples": n,
        "empty_images": empty_imgs,
        "empty_pct": 100 * empty_imgs / n,
        "total_objects": len(all_areas),
        "objects_per_image_mean": float(nobj_arr.mean()),
        "objects_per_image_median": float(np.median(nobj_arr)),
        "objects_per_image_max": int(nobj_arr.max()),
        "area_min_px": float(areas_arr.min()) if len(areas_arr) > 0 else 0,
        "area_max_px": float(areas_arr.max()) if len(areas_arr) > 0 else 0,
        "area_mean_px": float(areas_arr.mean()) if len(areas_arr) > 0 else 0,
        "area_median_px": float(np.median(areas_arr)) if len(areas_arr) > 0 else 0,
        "area_percentiles": {
            "p10": float(np.percentile(areas_arr, 10)) if len(areas_arr) > 0 else 0,
            "p25": float(np.percentile(areas_arr, 25)) if len(areas_arr) > 0 else 0,
            "p50": float(np.percentile(areas_arr, 50)) if len(areas_arr) > 0 else 0,
            "p75": float(np.percentile(areas_arr, 75)) if len(areas_arr) > 0 else 0,
            "p90": float(np.percentile(areas_arr, 90)) if len(areas_arr) > 0 else 0,
            "p95": float(np.percentile(areas_arr, 95)) if len(areas_arr) > 0 else 0,
            "p99": float(np.percentile(areas_arr, 99)) if len(areas_arr) > 0 else 0,
        },
        "all_areas": all_areas,
        "n_objects_per_img": n_objects_per_img,
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_dataset_report(
    train_fg: dict,
    test_fg: dict,
    train_img: dict,
    test_img: dict,
    train_obj: dict,
    test_obj: dict,
    output_dir: Path,
) -> None:
    """
    绘制综合分析报告 | Plot comprehensive analysis report.
    """
    print("\n  Generating visualization report...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: FG Distribution + Image Stats ──
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # (0,0): FG ratio histogram
    ax = fig.add_subplot(gs[0, 0])
    bins = np.logspace(-5, 0, 60)
    ax.hist(train_fg["ratios"] + 1e-6, bins=bins, alpha=0.6, label=f'Train (n={train_fg["stats"]["n"]})', color="steelblue", edgecolor="white")
    ax.hist(test_fg["ratios"] + 1e-6, bins=bins, alpha=0.6, label=f'Test (n={test_fg["stats"]["n"]})', color="coral", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("FG Ratio (log scale)")
    ax.set_ylabel("Sample Count")
    ax.set_title("FG Ratio Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1): FG ratio CDF
    ax = fig.add_subplot(gs[0, 1])
    sorted_train = np.sort(train_fg["ratios"])
    sorted_test = np.sort(test_fg["ratios"])
    ax.plot(sorted_train, np.linspace(0, 1, len(sorted_train)), color="steelblue", lw=2, label="Train")
    ax.plot(sorted_test, np.linspace(0, 1, len(sorted_test)), color="coral", lw=2, label="Test")
    ax.set_xscale("log")
    ax.set_xlabel("FG Ratio (log scale)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("FG Ratio CDF")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,2): FG category bar chart
    ax = fig.add_subplot(gs[0, 2])
    categories = ["Empty\n(FG=0)", "Sparse\n(<0.1%)", "Low\n(0.1-1%)", "Mid\n(1-10%)", "High\n(>10%)"]
    x = np.arange(len(categories))
    w = 0.35
    train_vals = [train_fg["stats"]["empty"], train_fg["stats"]["sparse"],
                  train_fg["stats"]["low"], train_fg["stats"]["mid"], train_fg["stats"]["high"]]
    test_vals = [test_fg["stats"]["empty"], test_fg["stats"]["sparse"],
                 test_fg["stats"]["low"], test_fg["stats"]["mid"], test_fg["stats"]["high"]]
    ax.bar(x - w/2, train_vals, w, label="Train", color="steelblue", edgecolor="white")
    ax.bar(x + w/2, test_vals, w, label="Test", color="coral", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Sample Count")
    ax.set_title("FG Density Categories")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0): Per-channel brightness
    ax = fig.add_subplot(gs[1, 0])
    ch_names = ["R", "G", "B"]
    for ds_img, label, color in [(train_img, "Train", "steelblue"), (test_img, "Test", "coral")]:
        for i, ch in enumerate(ch_names):
            vals = ds_img["means_per_channel"][:, i]
            ax.boxplot([vals], positions=[i + (0 if label == "Train" else 0.15)],
                       widths=0.12, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color="black", lw=1))
    ax.set_xticks(range(3))
    ax.set_xticklabels(ch_names)
    ax.set_ylabel("Mean Pixel Value [0,1]")
    ax.set_title("Per-Channel Brightness Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    # custom legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="steelblue", alpha=0.6), Patch(facecolor="coral", alpha=0.6)],
              ["Train", "Test"], fontsize=9)

    # (1,1): Object size histogram
    ax = fig.add_subplot(gs[1, 1])
    if len(train_obj["all_areas"]) > 0:
        train_areas = np.array(train_obj["all_areas"])
        test_areas = np.array(test_obj["all_areas"])
        bins = np.logspace(0, np.log10(max(train_areas.max(), test_areas.max()) + 1), 40)
        ax.hist(train_areas, bins=bins, alpha=0.6, label=f"Train ({len(train_areas)} objects)", color="steelblue", edgecolor="white")
        ax.hist(test_areas, bins=bins, alpha=0.6, label=f"Test ({len(test_areas)} objects)", color="coral", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Object Area (pixels, log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Connected Component Size Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,2): Objects per image
    ax = fig.add_subplot(gs[1, 2])
    max_obj = max(train_obj["objects_per_image_max"], test_obj["objects_per_image_max"])
    bins = np.arange(0, min(max_obj + 2, 30))
    ax.hist(train_obj["n_objects_per_img"], bins=bins, alpha=0.6, label="Train", color="steelblue", edgecolor="white")
    ax.hist(test_obj["n_objects_per_img"], bins=bins, alpha=0.6, label="Test", color="coral", edgecolor="white")
    ax.set_xlabel("Objects per Image")
    ax.set_ylabel("Image Count")
    ax.set_title("Objects per Image Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("NEU_Seg Dataset Analysis Report", fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(output_dir / "01_dataset_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {output_dir / '01_dataset_report.png'}")

    # ── Figure 2: Sample Montage ──
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    ds_train = NEUSegDataset(root=train_fg.get("_root", "data/NEU_Seg"), split="train")
    indices = np.linspace(0, len(ds_train) - 1, 15, dtype=int)

    for idx, ax in zip(indices, axes.flat):
        sample = ds_train[idx]
        img = sample["image"].permute(1, 2, 0).numpy()  # [H, W, 3]
        mask = sample["masks"].squeeze(0).numpy()        # [H, W]

        # 叠加显示 | Overlay display
        overlay = img.copy()
        overlay[mask > 0.5, :] = overlay[mask > 0.5, :] * 0.5 + np.array([1.0, 0.2, 0.2]) * 0.5
        ax.imshow(overlay)
        fg_pct = mask.mean() * 100
        ax.set_title(f"FG={fg_pct:.2f}%", fontsize=8)
        ax.axis("off")

    fig.suptitle("NEU_Seg — Training Samples (Red = FG Building)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "02_sample_montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {output_dir / '02_sample_montage.png'}")


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NEU_Seg 数据集分析 | NEU_Seg Dataset Analysis"
    )
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg",
                        help="数据集根目录 | Dataset root directory")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 | Output directory (default: runs/neuseg_analysis_<timestamp>)")
    parser.add_argument("--max-image-stats", type=int, default=500,
                        help="图像统计最大样本数 | Max samples for image stats")
    parser.add_argument("--max-object-stats", type=int, default=500,
                        help="目标分析最大样本数 | Max samples for object analysis")
    parser.add_argument("--no-viz", action="store_true",
                        help="跳过可视化 | Skip visualization")
    args = parser.parse_args()

    from datetime import datetime

    if args.output is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output = f"runs/neuseg_analysis_{ts}"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NEU_Seg Dataset Analysis")
    print(f"  Data:   {args.data_root}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── 数据集概览 | Dataset Overview ──
    print("\n── 1. Dataset Overview ──")
    ds_train = NEUSegDataset(root=args.data_root, split="train")
    ds_test = NEUSegDataset(root=args.data_root, split="test")

    overview = {
        "train": {"samples": len(ds_train), "image_size": "200×200"},
        "test": {"samples": len(ds_test), "image_size": "200×200"},
        "total": len(ds_train) + len(ds_test),
    }
    print(f"  Train: {overview['train']}")
    print(f"  Test:  {overview['test']}")
    print(f"  Total: {overview['total']}")

    # ── FG 分布 | FG Distribution ──
    print("\n── 2. FG Ratio Distribution ──")
    train_fg = analyze_fg_distribution(ds_train, "train")
    test_fg = analyze_fg_distribution(ds_test, "test")

    for name, result in [("Train", train_fg), ("Test", test_fg)]:
        s = result["stats"]
        print(f"  {name}: mean={s['mean']:.4f} median={s['median']:.4f} "
              f"min={s['min']:.4f} max={s['max']:.4f} std={s['std']:.4f}")
        print(f"         empty={s['empty']} sparse={s['sparse']} low={s['low']} "
              f"mid={s['mid']} high={s['high']}")

    # ── 图像统计 | Image Statistics ──
    print("\n── 3. Image Statistics ──")
    train_img = analyze_image_stats(ds_train, "train", max_samples=args.max_image_stats)
    test_img = analyze_image_stats(ds_test, "test", max_samples=args.max_image_stats)

    for name, result in [("Train", train_img), ("Test", test_img)]:
        print(f"  {name}: brightness={result['overall_brightness']:.4f} "
              f"contrast={result['overall_contrast']:.4f}")

    # ── 目标尺寸 | Object Sizes ──
    print("\n── 4. Object Size Analysis ──")
    train_obj = analyze_object_sizes(ds_train, "train", max_samples=args.max_object_stats)
    test_obj = analyze_object_sizes(ds_test, "test", max_samples=args.max_object_stats)

    for name, result in [("Train", train_obj), ("Test", test_obj)]:
        print(f"  {name}: {result['total_objects']} objects in {result['n_samples']} images, "
              f"{result['empty_images']} empty ({result['empty_pct']:.1f}%)")
        print(f"         objects/img: mean={result['objects_per_image_mean']:.1f} "
              f"median={result['objects_per_image_median']:.1f} "
              f"max={result['objects_per_image_max']}")
        print(f"         area (px): mean={result['area_mean_px']:.1f} "
              f"median={result['area_median_px']:.1f} "
              f"min={result['area_min_px']:.0f} max={result['area_max_px']:.0f}")
        if result["area_percentiles"]:
            p = result["area_percentiles"]
            print(f"         percentiles: p10={p['p10']:.0f} p25={p['p25']:.0f} "
                  f"p50={p['p50']:.0f} p75={p['p75']:.0f} p90={p['p90']:.0f} "
                  f"p95={p['p95']:.0f} p99={p['p99']:.0f}")

    # ── 可视化 | Visualization ──
    if not args.no_viz:
        print("\n── 5. Visualization ──")
        # inject root for montage
        train_fg["_root"] = args.data_root
        plot_dataset_report(
            train_fg, test_fg,
            train_img, test_img,
            train_obj, test_obj,
            out_dir,
        )

    # ── 保存 JSON 报告 | Save JSON Report ──
    import json
    report = {
        "dataset": "NEU_Seg",
        "image_size": "200×200",
        "overview": overview,
        "fg_distribution": {
            "train": train_fg["stats"],
            "test": test_fg["stats"],
        },
        "image_stats": {
            "train": {k: v for k, v in train_img.items()
                      if k not in ("means_per_channel", "stds_per_channel")},
            "test": {k: v for k, v in test_img.items()
                     if k not in ("means_per_channel", "stds_per_channel")},
        },
        "object_sizes": {
            "train": {k: v for k, v in train_obj.items()
                      if k not in ("all_areas", "n_objects_per_img")},
            "test": {k: v for k, v in test_obj.items()
                     if k not in ("all_areas", "n_objects_per_img")},
        },
    }

    report_file = out_dir / "analysis_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  [OK] Report saved: {report_file}")

    print(f"\n[Done] Analysis complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
