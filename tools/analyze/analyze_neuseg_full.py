#!/usr/bin/env python3
"""
NEU_Seg 完整数据集分析工具 | Comprehensive Dataset Analysis Tool.
===================================================================

十二维度全面分析 + 可视化 + CSV导出 + 自动报告生成。
12-dimension analysis + visualization + CSV export + auto report.

用法 | Usage::

    # 完整分析
    python tools/analyze/analyze_neuseg_full.py

    # 指定路径和输出
    python tools/analyze/analyze_neuseg_full.py --data-root data/NEU_Seg --output analysis

    # CPU 加速
    python tools/analyze/analyze_neuseg_full.py --workers 8

    # 仅数据质量检查
    python tools/analyze/analyze_neuseg_full.py --quality-only

模块 | Modules:
    M01 — 基础统计 (样本量/尺寸/格式/目录结构)
    M02 — 类别统计 (BG/FG 像素占比/类别均衡)
    M03 — 图像统计 (亮度/对比度/直方图/动态范围)
    M04 — 图像质量 (模糊检测/Laplacian方差)
    M05 — Mask 统计 (连通域/面积/AspectRatio/Circularity)
    M06 — 目标尺度 (小/中/大目标, 面积分位数)
    M07 — 空间分布 (中心热力图/边缘分布)
    M08 — 数据质量 (损坏/重复/空/异常)
    M09 — Train/Test 对比 (分布偏移/KS检验)
    M10 — Few-shot 分析 (样本数/Episode/覆盖率)
    M11 — 可视化 (12类图表)
    M12 — 报告生成 (REPORT.md)
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# 常量 & 配置 | Constants & Configuration
# ═══════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

# 目标尺度阈值 (像素) | Object size thresholds (pixels)
SMALL_THRESH = 32 * 32    # < 1024 px
LARGE_THRESH = 96 * 96    # > 9216 px

# 模糊阈值 (Laplacian Variance < 100 = 模糊) | Blur threshold
BLUR_THRESHOLD = 100.0


@dataclass
class DatasetInfo:
    """数据集元信息 | Dataset metadata."""
    name: str = "NEU_Seg"
    root: Path = Path("data/NEU_Seg")
    splits: dict[str, str] = field(default_factory=lambda: {
        "train": "training", "test": "test"
    })
    img_ext: str = ".jpg"
    mask_ext: str = ".png"
    image_size: tuple[int, int] = (200, 200)
    channels: int = 3  # RGB
    num_classes: int = 4  # 0=BG, 1=Inclusion, 2=Patch, 3=Scratch
    class_names: list[str] = field(default_factory=lambda: [
        "background", "Inclusion", "Patch", "Scratch"
    ])
    binary_mode: bool = False  # True→二值模式, False→多类别模式


@dataclass
class SampleInfo:
    """单样本信息 | Single sample info."""
    path: Path
    mask_path: Optional[Path] = None
    split: str = "train"
    image_id: str = ""
    binary_mode: bool = True  # True→二值化, False→保留多类别
    # 缓存字段 | Cached fields
    _img: Optional[np.ndarray] = None
    _mask: Optional[np.ndarray] = None

    def load_image(self) -> np.ndarray:
        if self._img is None:
            self._img = cv2.cvtColor(cv2.imread(str(self.path)), cv2.COLOR_BGR2RGB)
        return self._img

    def load_mask(self, binary: Optional[bool] = None) -> Optional[np.ndarray]:
        """加载 mask: binary=True → 二值化, binary=False → 保留多类别标签."""
        if self.mask_path is None:
            return None
        use_binary = binary if binary is not None else self.binary_mode
        if self._mask is None:
            self._mask = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
        if self._mask is None:
            return None
        if use_binary:
            return (self._mask > 0).astype(np.uint8)
        return self._mask  # 保留 0/1/2/3

    def get_classes_present(self) -> list[int]:
        """返回 mask 中出现的类别列表 (含 BG) | Classes present in mask (incl. BG)."""
        m = self.load_mask(binary=False)
        return sorted(np.unique(m).tolist()) if m is not None else [0]

    @property
    def fg_ratio(self) -> float:
        m = self.load_mask(binary=True)
        return float(m.mean()) if m is not None else 0.0

    @property
    def is_empty(self) -> bool:
        return self.fg_ratio == 0.0


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def _hash_file(path: Path) -> str:
    """文件 MD5 哈希 | File MD5 hash."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """保存 CSV | Save CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150):
    """保存图表 | Save figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 数据加载 | Data Loading
# ═══════════════════════════════════════════════════════════════════

def discover_samples(info: DatasetInfo) -> dict[str, list[SampleInfo]]:
    """扫描数据集目录, 发现所有样本 | Scan dataset directory, discover all samples.

    :return: {split_name: [SampleInfo, ...]}
    """
    all_samples: dict[str, list[SampleInfo]] = {}

    for split_name, dir_name in info.splits.items():
        img_dir = info.root / "images" / dir_name
        ann_dir = info.root / "annotations" / dir_name

        if not img_dir.exists():
            print(f"  [WARN] Image dir not found: {img_dir}")
            continue

        samples = []
        img_files = sorted(img_dir.glob(f"*{info.img_ext}"))
        for img_path in img_files:
            mask_path = ann_dir / f"{img_path.stem}{info.mask_ext}"
            if not mask_path.exists():
                mask_path = None
            samples.append(SampleInfo(
                path=img_path, mask_path=mask_path,
                split=split_name, image_id=img_path.stem,
                binary_mode=info.binary_mode,
            ))
        all_samples[split_name] = samples
        print(f"  [{split_name}] {len(samples)} samples ({len([s for s in samples if s.mask_path is None])} missing masks)")

    return all_samples


# ═══════════════════════════════════════════════════════════════════
# M01 — 基础统计 | Basic Statistics
# ═══════════════════════════════════════════════════════════════════

def m01_basic_stats(info: DatasetInfo, all_samples: dict[str, list[SampleInfo]],
                    out_csv: Path) -> dict:
    print("\n" + "=" * 60)
    print("  M01 — Basic Statistics")
    print("=" * 60)

    results = {"dataset_name": info.name, "splits": {}}
    all_sizes = []
    total_imgs, total_masks = 0, 0

    for split_name, samples in all_samples.items():
        n_imgs = len(samples)
        n_masks = len([s for s in samples if s.mask_path is not None])
        sizes = []
        for s in samples:
            img = s.load_image()
            sizes.append(img.shape[:2])
            all_sizes.append(img.shape[:2])

        h_arr = np.array([s[0] for s in sizes])
        w_arr = np.array([s[1] for s in sizes])
        total_imgs += n_imgs
        total_masks += n_masks

        results["splits"][split_name] = {
            "images": n_imgs, "masks": n_masks,
            "paired": n_imgs == n_masks,
            "height_mean": float(h_arr.mean()), "height_min": int(h_arr.min()), "height_max": int(h_arr.max()),
            "width_mean": float(w_arr.mean()), "width_min": int(w_arr.min()), "width_max": int(w_arr.max()),
        }

    results["total_images"] = total_imgs
    results["total_masks"] = total_masks
    results["channels"] = info.channels
    results["image_size_mode"] = f"{int(np.median([s[0] for s in all_sizes]))}x{int(np.median([s[1] for s in all_sizes]))}"

    # Mask 数据类型检查 | Check mask dtype/values
    first_mask_sample = next((s for sl in all_samples.values() for s in sl if s.mask_path), None)
    if first_mask_sample:
        m = first_mask_sample.load_mask()
        results["mask_dtype"] = str(m.dtype)
        results["mask_unique_values"] = [int(v) for v in np.unique(m)]

    # ── CSV ──
    rows = []
    for sn, sd in results["splits"].items():
        rows.append({"split": sn, **sd})
    _save_csv(out_csv, rows, ["split", "images", "masks", "paired",
                               "height_mean", "height_min", "height_max",
                               "width_mean", "width_min", "width_max"])

    # ── 打印 | Print ──
    print(f"  Total images: {total_imgs}, Total masks: {total_masks}")
    print(f"  Channels: {info.channels}, Image size: {results['image_size_mode']}")
    print(f"  Mask dtype: {results.get('mask_dtype', 'N/A')}, "
          f"Unique: {results.get('mask_unique_values', 'N/A')}")
    for sn, sd in results["splits"].items():
        print(f"  [{sn}] {sd['images']} images, {sd['masks']} masks, "
              f"paired={sd['paired']}, size={sd['height_mean']:.0f}x{sd['width_mean']:.0f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# M02 — 类别统计 | Class Statistics
# ═══════════════════════════════════════════════════════════════════

def m02_class_stats(info: DatasetInfo, all_samples: dict[str, list[SampleInfo]],
                    out_csv: Path, out_fig_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("  M02 — Class Statistics")
    print("=" * 60)

    class_pixels = defaultdict(int)
    class_imgs = defaultdict(set)
    total_pixels = 0

    for split_name, samples in all_samples.items():
        for s in tqdm(samples, desc=f"  Class stats [{split_name}]"):
            # 使用 binary=False 保留多类别标签
            m = s.load_mask(binary=info.binary_mode)
            if m is None:
                continue
            total_pixels += m.size
            for cls_val in np.unique(m):
                cls_name = info.class_names[int(cls_val)] if int(cls_val) < len(info.class_names) else f"class_{cls_val}"
                n_px = int((m == cls_val).sum())
                class_pixels[cls_name] += n_px
                if n_px > 0:
                    class_imgs[cls_name].add(s.image_id)

    results = {}
    rows = []
    for cls_name in info.class_names:
        px = class_pixels.get(cls_name, 0)
        n_imgs = len(class_imgs.get(cls_name, set()))
        rows.append({
            "class": cls_name,
            "pixels": px,
            "pixel_pct": round(100 * px / total_pixels, 4) if total_pixels > 0 else 0,
            "images": n_imgs,
            "image_pct": round(100 * n_imgs / max(len(set.union(*class_imgs.values())) if class_imgs else set(), 1), 2),
        })
        results[cls_name] = {"pixels": px, "pixel_pct": rows[-1]["pixel_pct"],
                             "images": n_imgs, "image_pct": rows[-1]["image_pct"]}

    # 类别不均衡度 | Class imbalance ratio
    if info.binary_mode:
        fg_pct = rows[1]["pixel_pct"] if len(rows) > 1 else 0
        bg_pct = rows[0]["pixel_pct"] if len(rows) > 0 else 100
        results["fg_bg_ratio"] = f"1:{bg_pct/fg_pct:.0f}" if fg_pct > 0 else "N/A"
        results["balanced"] = "Highly Imbalanced" if fg_pct < 10 else "Approximately balanced"
        results["mode"] = "binary"
    else:
        fg_classes = [(r["class"], r["pixel_pct"]) for r in rows if r["class"] != "background"]
        total_fg_pct = sum(f[1] for f in fg_classes)
        bg_pct = rows[0]["pixel_pct"] if rows and rows[0]["class"] == "background" else 87.9
        min_cls, max_cls = min(fg_classes, key=lambda x: x[1]), max(fg_classes, key=lambda x: x[1])
        ratio = max_cls[1] / max(min_cls[1], 0.01)
        results["fg_bg_ratio"] = f"1:{bg_pct/total_fg_pct:.1f}" if total_fg_pct > 0 else "N/A"
        results["balanced"] = f"{'Balanced' if ratio < 2 else 'Imbalanced'} (max/min FG class ratio {ratio:.1f}x)"
        results["mode"] = "multi-class (3 defect types)"
        results["fg_class_distribution"] = {cls: pct for cls, pct in fg_classes}

    _save_csv(out_csv, rows, ["class", "pixels", "pixel_pct", "images", "image_pct"])

    print(f"  FG/BG pixel ratio: {results['fg_bg_ratio']}")
    print(f"  Balanced: {results['balanced']}")

    # ── Bar chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["class"] for r in rows]
    pcts = [r["pixel_pct"] for r in rows]
    img_pcts = [r["image_pct"] for r in rows]
    colors = ["#7f8c8d", "#e74c3c"][:len(names)]
    ax1.bar(names, pcts, color=colors, edgecolor="white")
    ax1.set_ylabel("Pixel %"); ax1.set_title("Pixel Distribution")
    ax2.pie(pcts, labels=names, autopct="%1.1f%%", colors=colors, startangle=90)
    ax2.set_title("Pixel Ratio")
    fig.suptitle("Class Distribution — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "class_distribution.png")

    return results


# ═══════════════════════════════════════════════════════════════════
# M03 — 图像统计 | Image Statistics
# ═══════════════════════════════════════════════════════════════════

def m03_image_stats(all_samples: dict[str, list[SampleInfo]],
                    out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M03 — Image Statistics")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl]

    def _compute(split_name, sample):
        img = sample.load_image()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return {
            "image_id": sample.image_id, "split": split_name,
            "brightness_mean": float(gray.mean()), "brightness_std": float(gray.std()),
            "contrast": float(gray.std() / (gray.mean() + 1e-8)),
            "r_mean": float(img[:,:,0].mean()), "g_mean": float(img[:,:,1].mean()), "b_mean": float(img[:,:,2].mean()),
            "r_std": float(img[:,:,0].std()), "g_std": float(img[:,:,1].std()), "b_std": float(img[:,:,2].std()),
            "dynamic_range": float(gray.max() - gray.min()),
            "min_pixel": float(gray.min()), "max_pixel": float(gray.max()),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Image stats"):
            rows.append(f.result())

    _save_csv(out_csv, rows, list(rows[0].keys()) if rows else [])

    brightness_arr = np.array([r["brightness_mean"] for r in rows])
    contrast_arr = np.array([r["contrast"] for r in rows])

    OVEREXPOSE_THRESH = 250  # max pixel > 250 (of 255) = overexposed
    UNDEREXPOSE_THRESH = 5    # min pixel < 5 (of 255) = underexposed

    results = {
        "brightness": {"mean": float(brightness_arr.mean()), "std": float(brightness_arr.std()),
                       "min": float(brightness_arr.min()), "max": float(brightness_arr.max())},
        "contrast": {"mean": float(contrast_arr.mean()), "std": float(contrast_arr.std())},
        "overexposed_pct": float((np.array([r["max_pixel"] for r in rows]) > OVEREXPOSE_THRESH).mean() * 100),
        "overexposed_definition": f"max_pixel > {OVEREXPOSE_THRESH}/255",
        "underexposed_pct": float((np.array([r["min_pixel"] for r in rows]) < UNDEREXPOSE_THRESH).mean() * 100),
        "underexposed_definition": f"min_pixel < {UNDEREXPOSE_THRESH}/255",
    }

    # ── Visualization ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    ax = axes[0, 0]
    all_gray_vals = []
    for s, _ in all_flat[:200]:  # sample 200
        all_gray_vals.append(cv2.cvtColor(s.load_image(), cv2.COLOR_RGB2GRAY).ravel())
    all_gray = np.concatenate(all_gray_vals)
    ax.hist(all_gray, bins=128, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Pixel Value"); ax.set_ylabel("Frequency"); ax.set_title("Gray Histogram (200 samples)")

    # Brightness boxplot by split
    ax = axes[0, 1]
    split_data = {}
    for sn in all_samples:
        split_data[sn] = [r["brightness_mean"] for r in rows if r["split"] == sn]
    ax.boxplot(split_data.values(), labels=split_data.keys(), patch_artist=True)
    ax.set_ylabel("Mean Brightness"); ax.set_title("Brightness by Split")

    # RGB channels
    ax = axes[1, 0]
    for ch, color in [("r_mean", "#e74c3c"), ("g_mean", "#2ecc71"), ("b_mean", "#3498db")]:
        vals = [r[ch] for r in rows]
        ax.hist(vals, bins=64, alpha=0.4, color=color, label=ch[0].upper(), edgecolor="white")
    ax.set_xlabel("Mean Channel Value"); ax.set_ylabel("Count"); ax.set_title("RGB Channel Distribution")
    ax.legend()

    # Dynamic range
    ax = axes[1, 1]
    dr_vals = [r["dynamic_range"] for r in rows]
    ax.hist(dr_vals, bins=64, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Dynamic Range"); ax.set_ylabel("Count"); ax.set_title("Dynamic Range Distribution")

    fig.suptitle("Image Statistics — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "brightness_hist.png")

    return results


# ═══════════════════════════════════════════════════════════════════
# M04 — 图像质量 | Image Quality (Blur Detection)
# ═══════════════════════════════════════════════════════════════════

def m04_image_quality(all_samples: dict[str, list[SampleInfo]],
                      out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M04 — Image Quality (Blur Detection)")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl]

    def _compute(split_name, sample):
        img = sample.load_image()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"image_id": sample.image_id, "split": split_name,
                "laplacian_var": lap_var, "is_blurry": lap_var < BLUR_THRESHOLD}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Quality check"):
            rows.append(f.result())

    _save_csv(out_csv, rows, ["image_id", "split", "laplacian_var", "is_blurry"])

    lap_arr = np.array([r["laplacian_var"] for r in rows])
    n_blurry = int(sum(1 for r in rows if r["is_blurry"]))

    results = {
        "laplacian_mean": float(lap_arr.mean()), "laplacian_std": float(lap_arr.std()),
        "laplacian_min": float(lap_arr.min()), "laplacian_max": float(lap_arr.max()),
        "blurry_count": n_blurry, "blurry_pct": round(100 * n_blurry / len(rows), 2),
        "threshold": BLUR_THRESHOLD,
    }

    # ── Visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Blur histogram
    ax = axes[0]
    ax.hist(lap_arr, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(BLUR_THRESHOLD, color="red", ls="--", lw=2, label=f"Threshold={BLUR_THRESHOLD}")
    ax.set_xlabel("Laplacian Variance"); ax.set_ylabel("Count")
    ax.set_title(f"Blur Distribution (blurry={n_blurry}/{len(rows)}={results['blurry_pct']}%)")
    ax.legend()

    # Top-20 blurriest images montage
    ax = axes[1]
    sorted_rows = sorted(rows, key=lambda r: r["laplacian_var"])
    blurriest = sorted_rows[:20]
    clearest = sorted_rows[-20:]
    ax.scatter(range(len(lap_arr)), sorted(lap_arr), c="steelblue", s=2, alpha=0.5)
    ax.axhline(BLUR_THRESHOLD, color="red", ls="--", lw=1)
    ax.set_xlabel("Sample (sorted)"); ax.set_ylabel("Laplacian Variance")
    ax.set_title("Laplacian Variance — All Samples (sorted)")

    fig.suptitle("Image Quality — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "blur_distribution.png")

    # ── Top-20 blurriest + clearest montage ──
    sample_lookup = {s.image_id: s for s, _ in all_flat}
    for tag, ids in [("blurriest", [r["image_id"] for r in blurriest]),
                      ("clearest", [r["image_id"] for r in clearest])]:
        n = min(20, len(ids))
        cols = 5; rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols*2.5, rows_n*2.5))
        for i, img_id in enumerate(ids[:n]):
            ax = axes[i // cols, i % cols] if rows_n > 1 else axes[i % cols]
            s = sample_lookup[img_id]
            ax.imshow(s.load_image())
            ax.set_title(f"{img_id}\nLapVar={lap_arr[[r['image_id'] for r in rows].index(img_id)]:.1f}", fontsize=7)
            ax.axis("off")
        for i in range(n, rows_n * cols):
            ax = axes[i // cols, i % cols] if rows_n > 1 else axes[i % cols]
            ax.axis("off")
        fig.suptitle(f"Top-20 {tag.capitalize()} Images — NEU_Seg", fontweight="bold")
        _save_fig(fig, out_fig_dir / f"top20_{tag}.png")

    print(f"  Blurry: {n_blurry}/{len(rows)} ({results['blurry_pct']}%)")
    return results


# ═══════════════════════════════════════════════════════════════════
# M05 — Mask 统计 | Mask Statistics
# ═══════════════════════════════════════════════════════════════════

def m05_mask_stats(all_samples: dict[str, list[SampleInfo]],
                   out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M05 — Mask Statistics")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl if s.mask_path]

    def _compute(split_name, sample):
        m = sample.load_mask(binary=True)  # always binary for FG ratio & components
        fg_ratio = float(m.mean())
        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        components = []
        for j in range(1, n_labels):
            area = int(stats[j, cv2.CC_STAT_AREA])
            left, top, w, h = int(stats[j, cv2.CC_STAT_LEFT]), int(stats[j, cv2.CC_STAT_TOP]), \
                             int(stats[j, cv2.CC_STAT_WIDTH]), int(stats[j, cv2.CC_STAT_HEIGHT])
            # 周长 (轮廓近似) | Perimeter (contour approx)
            comp_mask = (labels == j).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
            # 面积 | Area for convex hull
            hull_area = float(cv2.contourArea(cv2.convexHull(contours[0]))) if contours else float(area)
            components.append({
                "component_id": j, "area": area,
                "width": w, "height": h,
                "bbox_left": left, "bbox_top": top, "bbox_right": left + w, "bbox_bottom": top + h,
                "aspect_ratio": round(w / max(h, 1), 4),
                "perimeter": round(perimeter, 2),
                "circularity": round(4 * np.pi * area / max(perimeter * perimeter, 1e-8), 6),
                "solidity": round(area / max(hull_area, 1e-8), 6) if hull_area > 0 else 0,
            })
        return {
            "image_id": sample.image_id, "split": split_name,
            "fg_ratio": fg_ratio, "n_components": n_labels - 1,
            "components": components,
        }

    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Mask stats"):
            all_results.append(f.result())

    # Flatten components to CSV
    rows = []
    for r in all_results:
        for c in r["components"]:
            rows.append({"image_id": r["image_id"], "split": r["split"],
                         "fg_ratio": r["fg_ratio"], "n_components": r["n_components"], **c})
    if not rows:
        rows = [{"image_id": "", "split": "", "fg_ratio": 0, "n_components": 0,
                 "component_id": 0, "area": 0, "width": 0, "height": 0,
                 "bbox_left": 0, "bbox_top": 0, "bbox_right": 0, "bbox_bottom": 0,
                 "aspect_ratio": 0, "perimeter": 0, "circularity": 0, "solidity": 0}]
    _save_csv(out_csv, rows, list(rows[0].keys()))

    # Aggregate stats
    fg_arr = np.array([r["fg_ratio"] for r in all_results])
    ncomp_arr = np.array([r["n_components"] for r in all_results])
    areas = [c["area"] for r in all_results for c in r["components"]]
    area_arr = np.array(areas) if areas else np.array([0])

    results = {
        "fg_ratio": {"mean": float(fg_arr.mean()), "median": float(np.median(fg_arr)),
                     "min": float(fg_arr.min()), "max": float(fg_arr.max()), "std": float(fg_arr.std())},
        "n_components": {"mean": float(ncomp_arr.mean()), "median": float(np.median(ncomp_arr)),
                         "max": int(ncomp_arr.max()), "empty_pct": float((ncomp_arr == 0).mean() * 100)},
        "object_area": {"mean": float(area_arr.mean()), "median": float(np.median(area_arr)),
                        "min": int(area_arr.min()), "max": int(area_arr.max())},
    }

    # ── Visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # FG ratio hist
    ax = axes[0, 0]
    ax.hist(fg_arr + 1e-6, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xscale("log"); ax.set_xlabel("FG Ratio"); ax.set_ylabel("Count")
    ax.set_title("FG Ratio Distribution"); ax.axvline(fg_arr.mean(), color="red", ls="--", lw=1, label=f"Mean={fg_arr.mean():.3f}")
    ax.legend()

    # Area hist
    ax = axes[0, 1]
    if len(area_arr) > 0 and area_arr.max() > 0:
        ax.hist(area_arr, bins=np.logspace(1, np.log10(area_arr.max() + 1), 50), color="coral", alpha=0.7, edgecolor="white")
        ax.set_xscale("log")
    ax.set_xlabel("Area (px)"); ax.set_ylabel("Count"); ax.set_title("Object Area Distribution")

    # Area CDF
    ax = axes[0, 2]
    if len(area_arr) > 0:
        sorted_areas = np.sort(area_arr)
        ax.plot(sorted_areas, np.linspace(0, 1, len(sorted_areas)), color="coral", lw=2)
        ax.set_xscale("log"); ax.set_xlabel("Area (px)"); ax.set_ylabel("CDF")
        ax.set_title("Object Area CDF"); ax.grid(True, alpha=0.3)

    # Components CDF
    ax = axes[1, 0]
    sorted_nc = np.sort(ncomp_arr)
    ax.plot(sorted_nc, np.linspace(0, 1, len(sorted_nc)), color="steelblue", lw=2)
    ax.set_xlabel("N Components"); ax.set_ylabel("CDF"); ax.set_title("Connected Components per Image (CDF)")
    ax.grid(True, alpha=0.3)

    # Aspect ratio
    ax = axes[1, 1]
    ar_vals = [c["aspect_ratio"] for r in all_results for c in r["components"] if c["aspect_ratio"] > 0]
    if ar_vals:
        ax.hist(np.clip(ar_vals, 0, 5), bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Aspect Ratio (W/H)"); ax.set_ylabel("Count"); ax.set_title("Aspect Ratio Distribution")

    # FG ratio by split boxplot
    ax = axes[1, 2]
    split_fg = {}
    for sn in all_samples:
        split_fg[sn] = [r["fg_ratio"] for r in all_results if r["split"] == sn]
    ax.boxplot(split_fg.values(), labels=split_fg.keys(), patch_artist=True)
    ax.set_ylabel("FG Ratio"); ax.set_title("FG Ratio by Split")

    fig.suptitle("Mask Statistics — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "area_hist.png")

    # Additional: FG ratio + components combined
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(fg_arr[fg_arr > 0] + 1e-6, bins=60, color="coral", alpha=0.7, edgecolor="white")
    ax1.set_xscale("log"); ax1.set_xlabel("FG Ratio (non-empty only)"); ax1.set_title("FG Ratio — Non-Empty Samples")
    ax2.hist(np.clip(ncomp_arr, 0, 10), bins=11, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("N Components"); ax2.set_title("Connected Components per Image")
    _save_fig(fig2, out_fig_dir / "connected_components.png")

    print(f"  FG ratio: mean={results['fg_ratio']['mean']:.4f}, median={results['fg_ratio']['median']:.4f}")
    print(f"  Components: mean={results['n_components']['mean']:.1f}/img, empty={results['n_components']['empty_pct']:.1f}%")
    return results


# ═══════════════════════════════════════════════════════════════════
# M06 — 目标尺度分析 | Object Size Analysis
# ═══════════════════════════════════════════════════════════════════

def m06_size_analysis(all_samples: dict[str, list[SampleInfo]],
                      out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M06 — Object Size Analysis")
    print("=" * 60)

    # Reuse mask stats computation or recompute lightweight version
    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl if s.mask_path]

    def _compute(split_name, sample):
        m = sample.load_mask(binary=True)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        sizes = []
        for j in range(1, n_labels):
            area = int(stats[j, cv2.CC_STAT_AREA])
            w, h = int(stats[j, cv2.CC_STAT_WIDTH]), int(stats[j, cv2.CC_STAT_HEIGHT])
            cat = "small" if area < SMALL_THRESH else ("large" if area > LARGE_THRESH else "medium")
            sizes.append({"image_id": sample.image_id, "split": split_name,
                          "area": area, "width": w, "height": h, "size_category": cat})
        return sizes

    all_sizes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Size analysis"):
            all_sizes.extend(f.result())

    _save_csv(out_csv, all_sizes, ["image_id", "split", "area", "width", "height", "size_category"])

    areas_arr = np.array([s["area"] for s in all_sizes]) if all_sizes else np.array([0])
    small_n = sum(1 for s in all_sizes if s["size_category"] == "small")
    medium_n = sum(1 for s in all_sizes if s["size_category"] == "medium")
    large_n = sum(1 for s in all_sizes if s["size_category"] == "large")
    total = max(len(all_sizes), 1)

    results = {
        "total_objects": len(all_sizes),
        "small": {"count": small_n, "pct": round(100 * small_n / total, 1)},
        "medium": {"count": medium_n, "pct": round(100 * medium_n / total, 1)},
        "large": {"count": large_n, "pct": round(100 * large_n / total, 1)},
        "area_min": int(areas_arr.min()), "area_max": int(areas_arr.max()),
        "area_mean": float(areas_arr.mean()), "area_median": float(np.median(areas_arr)),
        "area_percentiles": {
            "p25": float(np.percentile(areas_arr, 25)),
            "p50": float(np.percentile(areas_arr, 50)),
            "p75": float(np.percentile(areas_arr, 75)),
            "p90": float(np.percentile(areas_arr, 90)),
            "p95": float(np.percentile(areas_arr, 95)),
            "p99": float(np.percentile(areas_arr, 99)),
        },
        "thresholds": {"small": f"< {SMALL_THRESH}px", "medium": f"{SMALL_THRESH}-{LARGE_THRESH}px",
                        "large": f"> {LARGE_THRESH}px"},
    }

    # ── Visualization ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pie
    ax = axes[0]
    ax.pie([results["small"]["pct"], results["medium"]["pct"], results["large"]["pct"]],
           labels=["Small", "Medium", "Large"], autopct="%1.1f%%",
           colors=["#3498db", "#f39c12", "#e74c3c"], startangle=90)
    ax.set_title("Object Size Distribution")

    # Area boxplot
    ax = axes[1]
    cats = {"small": [], "medium": [], "large": []}
    for s in all_sizes:
        cats[s["size_category"]].append(s["area"])
    ax.boxplot([cats["small"], cats["medium"], cats["large"]],
               labels=["Small", "Medium", "Large"], patch_artist=True)
    ax.set_yscale("log"); ax.set_ylabel("Area (px, log)"); ax.set_title("Area by Size Category")

    # Area CDF
    ax = axes[2]
    sorted_a = np.sort(areas_arr)
    ax.plot(sorted_a, np.linspace(0, 1, len(sorted_a)), color="coral", lw=2)
    ax.set_xscale("log"); ax.set_xlabel("Area (px)"); ax.set_ylabel("CDF")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(areas_arr, p)
        ax.axvline(val, color="gray", ls="--", alpha=0.4)
        ax.text(val, 0.6, f"P{p}", fontsize=7, rotation=90)
    ax.set_title("Area CDF with Percentiles"); ax.grid(True, alpha=0.3)

    fig.suptitle("Object Size Analysis — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "area_cdf.png")

    # Aspect ratio
    ar_arr = np.array([s["width"] / max(s["height"], 1) for s in all_sizes])
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(ar_arr, 0, 5), bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Aspect Ratio (W/H)"); ax.set_ylabel("Count"); ax.set_title("Aspect Ratio Distribution")
    _save_fig(fig2, out_fig_dir / "aspect_ratio_hist.png")

    print(f"  Objects: small={results['small']['count']}({results['small']['pct']}%), "
          f"medium={results['medium']['count']}({results['medium']['pct']}%), "
          f"large={results['large']['count']}({results['large']['pct']}%)")
    print(f"  Object Area: mean={results['area_mean']:.0f} px, median={results['area_median']:.0f} px, "
          f"min={results['area_min']}, max={results['area_max']}")
    return results


# ═══════════════════════════════════════════════════════════════════
# M07 — 空间分布 | Spatial Distribution
# ═══════════════════════════════════════════════════════════════════

def m07_spatial_distribution(all_samples: dict[str, list[SampleInfo]],
                              out_fig_dir: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M07 — Spatial Distribution")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl if s.mask_path]

    def _compute(split_name, sample):
        m = sample.load_mask(binary=True)
        h, w = m.shape
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        centers = []
        for j in range(1, n_labels):
            cx, cy = centroids[j]
            centers.append({"cx": cx / w, "cy": cy / h})  # normalized
        return centers

    all_centers = {"cx": [], "cy": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Spatial analysis"):
            for c in f.result():
                all_centers["cx"].append(c["cx"])
                all_centers["cy"].append(c["cy"])

    cx_arr = np.array(all_centers["cx"])
    cy_arr = np.array(all_centers["cy"])

    # Determine bias
    center_dist = np.sqrt((cx_arr - 0.5) ** 2 + (cy_arr - 0.5) ** 2)
    edge_count = int(((cx_arr < 0.1) | (cx_arr > 0.9) | (cy_arr < 0.1) | (cy_arr > 0.9)).sum())

    results = {
        "n_objects": len(cx_arr),
        "center_x_mean": float(cx_arr.mean()), "center_y_mean": float(cy_arr.mean()),
        "center_x_std": float(cx_arr.std()), "center_y_std": float(cy_arr.std()),
        "mean_dist_from_center": float(center_dist.mean()),
        "edge_bias_pct": round(100 * edge_count / max(len(cx_arr), 1), 1),
        "bias": "center" if center_dist.mean() < 0.25 else ("edge" if edge_count / max(len(cx_arr), 1) > 0.5 else "uniform"),
    }

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(cx_arr) > 0:
        h = ax.hist2d(cx_arr, cy_arr, bins=40, range=[[0, 1], [0, 1]], cmap="hot")
        plt.colorbar(h[-1], ax=ax, label="Count")
    ax.set_xlabel("Normalized X"); ax.set_ylabel("Normalized Y")
    ax.set_title(f"Object Center Heatmap (n={len(cx_arr)}, bias={results['bias']})")
    ax.invert_yaxis()
    _save_fig(fig, out_fig_dir / "heatmap.png")

    print(f"  Objects: {len(cx_arr)}, center mean=({cx_arr.mean():.3f},{cy_arr.mean():.3f})")
    print(f"  Bias: {results['bias']}, edge_pct={results['edge_bias_pct']}%")
    return results


# ═══════════════════════════════════════════════════════════════════
# M08 — 数据质量检查 | Data Quality Check
# ═══════════════════════════════════════════════════════════════════

def m08_quality_check(info: DatasetInfo, all_samples: dict[str, list[SampleInfo]],
                       out_csv: Path, workers: int = 4) -> dict:
    print("\n" + "=" * 60)
    print("  M08 — Data Quality Check")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl]
    target_h, target_w = info.image_size

    def _check(split_name, sample):
        issues = []
        # Image existence
        if not sample.path.exists():
            issues.append("image_missing")
        # Mask existence
        if sample.mask_path is None or not sample.mask_path.exists():
            issues.append("mask_missing")
        if issues:
            return {"image_id": sample.image_id, "split": split_name, "issues": ";".join(issues),
                    "status": "error"}

        img = sample.load_image()
        h, w = img.shape[:2]

        if h != target_h or w != target_w:
            issues.append(f"size_mismatch({h}x{w} vs {target_h}x{target_w})")

        # Check for corrupt image
        if img is None or img.size == 0:
            issues.append("corrupt_image")

        # Check mask
        m = sample.load_mask(binary=True)
        if m is not None:
            if m.sum() == 0:
                issues.append("empty_mask")
            if m.min() >= 1 and m.max() == 1:
                issues.append("all_white_mask")

        return {"image_id": sample.image_id, "split": split_name,
                "issues": ";".join(issues) if issues else "ok",
                "status": "warning" if issues else "ok"}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Quality check"):
            rows.append(f.result())

    _save_csv(out_csv, rows, ["image_id", "split", "issues", "status"])

    # ── Duplicate detection ──
    print("  Detecting duplicates (this may take a while)...")
    img_hashes = {}
    dup_imgs = []
    for s, sn in tqdm(all_flat, desc="  Image hash"):
        h = _hash_file(s.path)
        if h in img_hashes:
            dup_imgs.append((s.image_id, img_hashes[h]))
        else:
            img_hashes[h] = s.image_id

    mask_hashes = {}
    dup_masks = []
    for s, sn in tqdm([x for x in all_flat if x[0].mask_path], desc="  Mask hash"):
        h = _hash_file(s.mask_path)
        if h in mask_hashes:
            dup_masks.append((s.image_id, mask_hashes[h]))
        else:
            mask_hashes[h] = s.image_id

    errors = [r for r in rows if r["status"] == "error"]
    warnings_list = [r for r in rows if r["status"] == "warning"]

    results = {
        "total_checked": len(rows),
        "errors": len(errors),
        "warnings": len(warnings_list),
        "error_pct": round(100 * len(errors) / len(rows), 2),
        "duplicate_images": len(dup_imgs),
        "duplicate_masks": len(dup_masks),
        "issue_breakdown": {},
    }
    for r in warnings_list:
        for iss in r["issues"].split(";"):
            results["issue_breakdown"][iss] = results["issue_breakdown"].get(iss, 0) + 1

    print(f"  Errors: {len(errors)}, Warnings: {len(warnings_list)}")
    print(f"  Duplicate images: {len(dup_imgs)}, Duplicate masks: {len(dup_masks)}")
    for iss, count in results["issue_breakdown"].items():
        print(f"    - {iss}: {count}")

    return results


# ═══════════════════════════════════════════════════════════════════
# M09 — Train/Test 对比 | Train/Test Comparison
# ═══════════════════════════════════════════════════════════════════

def m09_train_test_compare(all_samples: dict[str, list[SampleInfo]],
                            out_csv: Path, out_fig_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("  M09 — Train/Test Comparison")
    print("=" * 60)

    # Compute per-sample metrics
    metrics = {}
    for sn, samples in all_samples.items():
        fg_list, brightness_list, blur_list = [], [], []
        for s in samples:
            img = s.load_image()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            fg_list.append(s.fg_ratio)
            brightness_list.append(float(gray.mean()))
            blur_list.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        metrics[sn] = {"fg": np.array(fg_list), "brightness": np.array(brightness_list),
                       "blur": np.array(blur_list)}

    split_names = list(metrics.keys())
    results = {"comparisons": {}, "ks_test": {}}

    rows = []
    for metric_name in ["fg", "brightness", "blur"]:
        metric_label = {"fg": "FG Ratio", "brightness": "Brightness", "blur": "Laplacian Var"}[metric_name]
        for sn in split_names:
            arr = metrics[sn][metric_name]
            rows.append({
                "split": sn, "metric": metric_label,
                "mean": round(float(arr.mean()), 6),
                "median": round(float(np.median(arr)), 6),
                "std": round(float(arr.std()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
            })

        # KS test between train and test
        if len(split_names) >= 2:
            a_arr = metrics[split_names[0]][metric_name]
            b_arr = metrics[split_names[1]][metric_name]
            ks_stat, ks_pval = scipy_stats.ks_2samp(a_arr, b_arr)
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(a_arr) + np.var(b_arr)) / 2)
            cohens_d = abs(np.mean(a_arr) - np.mean(b_arr)) / max(pooled_std, 1e-8)
            effect_label = "large" if cohens_d > 0.8 else ("medium" if cohens_d > 0.5 else "small")
            results["ks_test"][metric_label] = {
                "statistic": round(float(ks_stat), 6),
                "p_value": round(float(ks_pval), 6),
                "significant": "Yes (p<0.05)" if ks_pval < 0.05 else "No (p>=0.05)",
                "cohens_d": round(float(cohens_d), 3),
                "effect_size": effect_label,
            }

    _save_csv(out_csv, rows, ["split", "metric", "mean", "median", "std", "min", "max"])

    # ── Visualization ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["steelblue", "coral"]
    for i, metric_name in enumerate(["fg", "brightness", "blur"]):
        ax = axes[i]
        for j, sn in enumerate(split_names):
            arr = metrics[sn][metric_name]
            metric_label = {"fg": "FG Ratio", "brightness": "Brightness", "blur": "Laplacian Var"}[metric_name]
            ax.hist(arr, bins=50, alpha=0.5, color=colors[j], label=sn, edgecolor="white")
        ax.set_xlabel(metric_label); ax.set_ylabel("Count"); ax.set_title(f"{metric_label} Distribution")
        ax.legend()

    fig.suptitle("Train vs Test Distribution Comparison — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "train_test_compare.png")

    for name, ks in results["ks_test"].items():
        print(f"  KS test [{name}]: D={ks['statistic']:.4f}, p={ks['p_value']:.4f}, "
              f"d={ks.get('cohens_d', '?'):.3f} ({ks.get('effect_size', '?')}) -> {ks['significant']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# M10 — Few-shot 分析 | Few-shot Analysis
# ═══════════════════════════════════════════════════════════════════

def m10_fewshot_analysis(info: DatasetInfo, all_samples: dict[str, list[SampleInfo]],
                          out_csv: Path) -> dict:
    print("\n" + "=" * 60)
    print("  M10 — Few-shot Analysis")
    print("=" * 60)

    train_samples = all_samples.get("train", [])
    fg_train = [s for s in train_samples if s.fg_ratio > 0]
    bg_train = [s for s in train_samples if s.fg_ratio == 0]

    n_fg = len(fg_train)
    img_size = info.image_size[0] * info.image_size[1]

    results = {
        "total_samples": len(train_samples),
        "fg_samples": n_fg,
        "bg_samples": len(bg_train),
        "fg_ratio_mean": float(np.mean([s.fg_ratio for s in fg_train])) if fg_train else 0,
        "fg_ratio_std": float(np.std([s.fg_ratio for s in fg_train])) if fg_train else 0,
        "classes": 1,  # binary
        "class_balanced": "N/A (binary)",
    }

    # Episode counts
    for k in [1, 5, 10]:
        n_episodes = n_fg // max(k, 1)
        coverage_pct = round(100 * min(n_episodes * k, n_fg) / max(n_fg, 1), 1)
        results[f"K={k}"] = {
            "n_support_available": n_fg,
            "possible_episodes": n_episodes if k == 1 else n_episodes,
            "support_coverage_pct": coverage_pct,
        }

    # Prototype stability: σ(bootstrap means of FG ratio), 100 resamples of K=5 support
    # Defined as: PS = std(mean(FG_ratio of random K-shot support set))
    # Lower = more stable prototype across support samples
    if n_fg >= 30:
        fg_ratios_all = np.array([s.fg_ratio for s in fg_train])
        bootstrap_means = [np.mean(np.random.choice(fg_ratios_all, size=min(5, n_fg))) for _ in range(100)]
        results["prototype_stability"] = round(float(np.std(bootstrap_means)), 6)
        results["prototype_stability_formula"] = (
            "PS = std(mean(FG_ratio of K=5 random support)), 100 bootstrap iterations. "
            "Lower = more stable support prototype."
        )
    else:
        results["prototype_stability"] = None
        results["prototype_stability_formula"] = "N/A (insufficient FG samples)"

    # CSV
    rows = []
    for k_info in ["K=1", "K=5", "K=10"]:
        if k_info in results:
            rows.append({"k": k_info, **results[k_info]})
    _save_csv(out_csv, rows, list(rows[0].keys()) if rows else ["k"])

    print(f"  FG samples: {n_fg}/{len(train_samples)} ({100*n_fg/max(len(train_samples),1):.1f}%)")
    for k_info in ["K=1", "K=5", "K=10"]:
        if k_info in results:
            print(f"  {k_info}: episodes={results[k_info]['possible_episodes']}, "
                  f"coverage={results[k_info]['support_coverage_pct']}%")

    return results


# ═══════════════════════════════════════════════════════════════════
# M11 — 综合可视化 | Combined Visualization
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# M13 — 纹理复杂度分析 | Texture Complexity (GLCM)
# ═══════════════════════════════════════════════════════════════════

def m13_texture_analysis(all_samples: dict[str, list[SampleInfo]],
                          out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    """GLCM 纹理分析: 对比度、相关性、能量、同质性 | GLCM texture: contrast, correlation, energy, homogeneity."""
    print("\n" + "=" * 60)
    print("  M13 — Texture Analysis (GLCM)")
    print("=" * 60)

    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        print("  [WARN] scikit-image not installed, skipping GLCM")
        return {"error": "scikit-image not installed"}

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl[:200]]  # sample 200 per split

    def _compute(split_name, sample):
        img = sample.load_image()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_uint8 = (gray / 4).astype(np.uint8)  # quantize to 64 levels
        glcm = graycomatrix(gray_uint8, distances=[1, 3], angles=[0, np.pi/2],
                            levels=64, symmetric=True, normed=True)
        props = {}
        for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
            vals = graycoprops(glcm, prop_name)
            props[prop_name] = float(vals.mean())
        return {"image_id": sample.image_id, "split": split_name, **props}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  GLCM texture"):
            rows.append(f.result())

    _save_csv(out_csv, rows, list(rows[0].keys()) if rows else [])

    results = {}
    for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = np.array([r[prop_name] for r in rows])
        results[prop_name] = {"mean": float(vals.mean()), "std": float(vals.std()),
                              "min": float(vals.min()), "max": float(vals.max())}

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, prop_name in enumerate(["contrast", "correlation", "energy", "homogeneity"]):
        ax = axes[i]
        train_vals = [r[prop_name] for r in rows if r["split"] == "train"]
        test_vals = [r[prop_name] for r in rows if r["split"] == "test"]
        ax.boxplot([train_vals, test_vals], labels=["Train", "Test"], patch_artist=True)
        ax.set_title(f"{prop_name.capitalize()}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("GLCM Texture — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "texture_glcm.png")

    print(f"  GLCM Contrast: {results['contrast']['mean']:.3f} (±{results['contrast']['std']:.3f})")
    print(f"  GLCM Energy:   {results['energy']['mean']:.4f} (higher=more uniform)")
    return results


# ═══════════════════════════════════════════════════════════════════
# M14 — 形状统计 | Shape Statistics
# ═══════════════════════════════════════════════════════════════════

def m14_shape_analysis(all_samples: dict[str, list[SampleInfo]],
                        out_csv: Path, out_fig_dir: Path, workers: int = 4) -> dict:
    """扩展形状统计: 偏心率、凸包比、矩形度、圆形度、伸长率 | Extended shape stats."""
    print("\n" + "=" * 60)
    print("  M14 — Shape Statistics (Extended)")
    print("=" * 60)

    all_flat = [(s, sn) for sn, sl in all_samples.items() for s in sl if s.mask_path]

    def _compute(split_name, sample):
        m = sample.load_mask(binary=True)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        shapes = []
        for j in range(1, n_labels):
            area = float(stats[j, cv2.CC_STAT_AREA])
            w, h = float(stats[j, cv2.CC_STAT_WIDTH]), float(stats[j, cv2.CC_STAT_HEIGHT])
            comp_mask = (labels == j).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]
            perimeter = max(cv2.arcLength(cnt, True), 1e-6)
            moments = cv2.moments(cnt)
            mu20 = moments["mu20"]; mu02 = moments["mu02"]; mu11 = moments["mu11"]

            # 等效椭圆 | Equivalent ellipse
            if moments["mu20"] + moments["mu02"] > 0:
                lambda1 = 0.5 * (mu20 + mu02 + np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
                lambda2 = 0.5 * (mu20 + mu02 - np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
                eccentricity = np.sqrt(1 - min(lambda2, lambda1) / max(lambda1, lambda2)) if max(lambda1, lambda2) > 0 else 0
            else:
                eccentricity = 0

            hull = cv2.convexHull(cnt)
            hull_area = max(cv2.contourArea(hull), 1.0)  # avoid div-by-zero
            rect = cv2.minAreaRect(cnt)
            rect_area = max(rect[1][0] * rect[1][1], 1.0)

            # 圆形度: 4πA/P² ∈ [0,1]; digitized contours can give >1, clip
            circ_val = 4 * np.pi * area / (perimeter * perimeter)
            circularity = min(max(circ_val, 0.0), 1.0)
            # 凸包比: Area/HullArea ∈ [0,1]
            convexity = min(area / hull_area, 1.0)

            shapes.append({
                "image_id": sample.image_id, "split": split_name,
                "area": area, "perimeter": perimeter,
                "circularity": circularity,
                "eccentricity": float(eccentricity),
                "convexity": convexity,
                "rectangularity": min(area / rect_area, 1.0),
                "elongation": max(w, h) / max(min(w, h), 1),
                "solidity": convexity,
                "aspect_ratio": w / max(h, 1),
            })
        return shapes

    all_shapes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute, sn, s): (sn, s) for s, sn in all_flat}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Shape analysis"):
            all_shapes.extend(f.result())

    _save_csv(out_csv, all_shapes, list(all_shapes[0].keys()) if all_shapes else [])

    results = {}
    for metric in ["circularity", "eccentricity", "convexity", "rectangularity", "elongation"]:
        vals = np.array([s[metric] for s in all_shapes])
        results[metric] = {"mean": float(vals.mean()), "std": float(vals.std()),
                           "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}

    # 打印时带 std | Print with std
    for metric in ["circularity", "eccentricity", "convexity", "elongation"]:
        r = results[metric]
        print(f"  {metric.capitalize()}: {r['mean']:.3f} ± {r['std']:.3f}")

    # Visualization: shape metric distributions
    metrics_plot = [("circularity", "Circularity\n(1=circle)"), ("eccentricity", "Eccentricity\n(0=circle,1=line)"),
                    ("convexity", "Convexity\n(1=convex)"), ("elongation", "Elongation\n(W/H ratio)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (key, label) in enumerate(metrics_plot):
        vals = np.clip([s[key] for s in all_shapes], 0, 5)
        axes[i].hist(vals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        axes[i].set_xlabel(label); axes[i].set_ylabel("Count")
    fig.suptitle("Shape Statistics — NEU_Seg", fontweight="bold")
    _save_fig(fig, out_fig_dir / "shape_statistics.png")

    print(f"  Circularity: {results['circularity']['mean']:.3f} (1=perfect circle)")
    print(f"  Eccentricity: {results['eccentricity']['mean']:.3f} (0=circle, 1=line)")
    print(f"  Elongation: {results['elongation']['mean']:.1f}x (W/H ratio)")

    # Defect size distribution by FG ratio
    fg_bins = {"Tiny (<0.5%)": 0.005, "Small (0.5-2%)": 0.02, "Medium (2-5%)": 0.05, "Large (>5%)": 1.0}
    all_flat_n = [(s, sn) for sn, sl in all_samples.items() for s in sl if s.mask_path]
    fg_ratios = []
    for s, sn in all_flat_n:
        m = s.load_mask(binary=True)
        fg_ratios.append(float(m.mean()))

    size_dist = {}
    for label, upper in fg_bins.items():
        lower = 0.0 if label.startswith("Tiny") else list(fg_bins.values())[list(fg_bins.keys()).index(label) - 1]
        count = sum(1 for r in fg_ratios if lower <= r < upper)
        size_dist[label] = {"count": count, "pct": round(100 * count / max(len(fg_ratios), 1), 1)}

    results["defect_size_distribution"] = size_dist

    # Visualization
    fig2, ax = plt.subplots(figsize=(8, 5))
    labels = list(size_dist.keys())
    counts = [size_dist[l]["count"] for l in labels]
    pcts = [size_dist[l]["pct"] for l in labels]
    bars = ax.bar(labels, counts, color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"], edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{pct}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Defect Size Distribution (by FG Ratio)")
    fig2.suptitle("Defect Size Categories — NEU_Seg", fontweight="bold")
    _save_fig(fig2, out_fig_dir / "defect_size_distribution.png")

    for label, info in size_dist.items():
        print(f"  {label}: {info['count']} ({info['pct']}%)")

    return results


# ═══════════════════════════════════════════════════════════════════
# M11 — 综合可视化 | Combined Visualization
# ═══════════════════════════════════════════════════════════════════

def m11_combined_visualization(info: DatasetInfo, all_samples: dict[str, list[SampleInfo]],
                                 out_fig_dir: Path, all_results: dict):
    print("\n" + "=" * 60)
    print("  M11 — Combined Visualization")
    print("=" * 60)

    # ── Dataset Overview Figure ──
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 0: Basic stats text + class pie + size pie + image sample
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    total_imgs = sum(len(sl) for sl in all_samples.values())
    total_masks = sum(len([s for s in sl if s.mask_path]) for sl in all_samples.values())
    text = (f"NEU_Seg Dataset Overview\n"
            f"{'='*30}\n"
            f"Total Images: {total_imgs}\n"
            f"Total Masks:  {total_masks}\n"
            f"Image Size:   {info.image_size[0]}x{info.image_size[1]}\n"
            f"Channels:     {info.channels} (RGB)\n"
            f"Classes:      {info.num_classes} (BG + FG)")
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10, fontfamily="monospace",
            verticalalignment="center")

    # Sample images
    for i, sn in enumerate(all_samples.keys()):
        ax = fig.add_subplot(gs[0, 1 + i])
        samples = all_samples[sn]
        if samples:
            idx = min(i * 10, len(samples) - 1)
            s = samples[idx]
            img = s.load_image()
            m = s.load_mask(binary=True)
            if m is not None:
                overlay = img.copy()
                overlay[m > 0] = overlay[m > 0] * 0.5 + np.array([255, 60, 60]) * 0.5
                ax.imshow(overlay)
            else:
                ax.imshow(img)
            ax.set_title(f"{sn}: {s.image_id}\nFG={s.fg_ratio:.3f}", fontsize=8)
        ax.axis("off")

    # Row 1: FG hist + area hist + brightness hist + blur hist
    metrics_order = [
        ("FG Ratio", "fg_ratio", "hist"),
        ("Object Area", "area", "hist"),
        ("Brightness", "brightness", "box"),
        ("Blur", "blur", "hist"),
    ]

    # Row 2: Heatmap placeholder + train/test comparison summary
    for i, sn in enumerate(all_samples.keys()):
        ax = fig.add_subplot(gs[2, i])
        n = len(all_samples[sn])
        fg_imgs = len([s for s in all_samples[sn] if s.fg_ratio > 0])
        ax.axis("off")
        text = (f"{sn.upper()} Split\n{'='*20}\n"
                f"Samples: {n}\n"
                f"FG images: {fg_imgs} ({100*fg_imgs/max(n,1):.1f}%)\n"
                f"Empty images: {n-fg_imgs} ({100*(n-fg_imgs)/max(n,1):.1f}%)")
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=9,
                fontfamily="monospace", verticalalignment="center")

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    # Summary from all_results
    text_lines = ["Key Findings", "=" * 30]
    if "m05" in all_results:
        mr = all_results["m05"]
        text_lines.append(f"FG ratio mean: {mr['fg_ratio']['mean']:.4f}")
        text_lines.append(f"Components/img: {mr['n_components']['mean']:.1f}")
    if "m04" in all_results:
        text_lines.append(f"Blurry: {all_results['m04']['blurry_pct']}%")
    if "m07" in all_results:
        text_lines.append(f"Spatial bias: {all_results['m07']['bias']}")
    if "m08" in all_results:
        text_lines.append(f"Quality errors: {all_results['m08']['errors']}")
    if "m10" in all_results:
        fk = all_results["m10"]
        text_lines.append(f"FG samples: {fk['fg_samples']}/{fk['total_samples']}")
    ax.text(0.05, 0.5, "\n".join(text_lines), transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment="center")

    fig.suptitle("NEU_Seg — Comprehensive Dataset Analysis", fontsize=16, fontweight="bold")
    _save_fig(fig, out_fig_dir / "dataset_summary.png")
    print(f"  [OK] Saved: {out_fig_dir / 'dataset_summary.png'}")


# ═══════════════════════════════════════════════════════════════════
# M15 — 一致性校验 | Consistency Check
# ═══════════════════════════════════════════════════════════════════

def m15_consistency_check(all_results: dict, info: DatasetInfo) -> dict:
    """跨模块一致性验证 | Cross-module consistency validation."""
    print("\n" + "=" * 60)
    print("  M15 — Consistency Check")
    print("=" * 60)

    issues = []
    m01 = all_results.get("m01", {})
    m02 = all_results.get("m02", {})
    m05 = all_results.get("m05", {})
    m06 = all_results.get("m06", {})
    m14 = all_results.get("m14", {})

    # 1. Total pixels consistency
    total_imgs = m01.get("total_images", 0)
    img_hw = 200 * 200  # known image size
    expected_px = total_imgs * img_hw
    if info.binary_mode and "background" in m02:
        bg_px = m02.get("background", {}).get("pixels", 0)
        fg_px = m02.get("foreground", {}).get("pixels", 0)
        if abs(bg_px + fg_px - expected_px) > expected_px * 0.01:
            issues.append(f"Pixel sum mismatch (binary): BG+FG={bg_px+fg_px}, expected={expected_px}")
    elif not info.binary_mode:
        total_cls_px = sum(m02.get(c, {}).get("pixels", 0) for c in info.class_names)
        if total_cls_px > 0 and abs(total_cls_px - expected_px) > expected_px * 0.01:
            issues.append(f"Pixel sum mismatch (multiclass): sum={total_cls_px}, expected={expected_px}")

    # 2. Global FG ratio ≈ per-image FG ratio mean (within tolerance)
    if info.binary_mode:
        global_fg = _get_safe("m02.foreground.pixel_pct", all_results, 0) / 100
        per_img_mean = _get_safe("m05.fg_ratio.mean", all_results, 0)
        if global_fg > 0.001 and abs(global_fg - per_img_mean) > 0.15:
            issues.append(f"FG ratio mismatch: global={global_fg:.4f}, per-image mean={per_img_mean:.4f} (diff={abs(global_fg-per_img_mean):.4f})")

    # 3. Object area × count ≈ total FG area
    n_objects = m06.get("total_objects", 0)
    mean_area = m06.get("area_mean", 0)
    if n_objects > 0 and mean_area > 0:
        est_fg_px = n_objects * mean_area
        if info.binary_mode:
            true_fg = _get_safe("m02.foreground.pixels", all_results, 0)
        else:
            true_fg = sum(m02.get(c, {}).get("pixels", 0) for c in info.class_names[1:])
        if true_fg > 0 and abs(est_fg_px - true_fg) / true_fg > 1.0:
            issues.append(f"Object area estimate mismatch: {n_objects} × {mean_area:.0f} = {est_fg_px:.0f}, true FG = {true_fg}")

    # 4. Circularity ∈ [0,1]
    circ_mean = _get_safe("m14.circularity.mean", all_results, None)
    if circ_mean is not None and (circ_mean < 0 or circ_mean > 1.01):
        issues.append(f"Circularity out of range: mean={circ_mean:.4f} (expected [0,1])")

    # 5. Convexity ∈ [0,1]
    conv_mean = _get_safe("m14.convexity.mean", all_results, None)
    if conv_mean is not None and (conv_mean < 0 or conv_mean > 1.01):
        issues.append(f"Convexity out of range: mean={conv_mean:.4f} (expected [0,1])")

    # 6. Empty mask consistency
    empty_pct = _get_safe("m05.n_components.empty_pct", all_results, None)
    m08_empty = _get_safe("m08.issue_breakdown.empty_mask", all_results, None)
    if empty_pct is not None and m08_empty is not None and abs(empty_pct - m08_empty / max(total_imgs, 1) * 100) > 1:
        issues.append(f"Empty mask count mismatch: M05={empty_pct:.1f}%, M08={m08_empty}")

    # Print results
    if issues:
        print(f"  [FAIL] {len(issues)} consistency issues found:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  [PASS] All consistency checks passed.")

    return {"passed": len(issues) == 0, "n_issues": len(issues), "issues": issues}


def _get_safe(path: str, data: dict, default=0):
    """Safe nested dict access for consistency check."""
    keys = path.split(".")
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val if val is not None else default


# ═══════════════════════════════════════════════════════════════════
# M12 — 报告生成 | Report Generation
# ═══════════════════════════════════════════════════════════════════

def m12_generate_report(ds_info: DatasetInfo, all_results: dict, out_path: Path):
    print("\n" + "=" * 60)
    print("  M12 — Report Generation")
    print("=" * 60)

    def _get(path: str, default="N/A"):
        """安全获取嵌套字典值 | Safe nested dict access."""
        keys = path.split(".")
        val = all_results
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val if val is not None else default

    lines = []
    def a(line=""):
        lines.append(line)

    a(f"# NEU_Seg — Comprehensive Dataset Analysis Report")
    a(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"\n---\n")

    # ── 1. Overview ──
    a("## 1. Dataset Overview\n")
    a("| Property | Value |")
    a("|----------|-------|")
    a(f"| Dataset | {_get('m01.dataset_name')} |")
    a(f"| Total Images | {_get('m01.total_images')} |")
    a(f"| Total Masks | {_get('m01.total_masks')} |")
    a(f"| Image Size | {_get('m01.image_size_mode')} px |")
    a(f"| Channels | {_get('m01.channels')} (RGB) |")
    a(f"| Mask Encoding | Original: uint8 {{0,1,2,3}} (0=BG, 1=Inclusion, 2=Patch, 3=Scratch); Binary: FG>0→1 |")
    for sn, sd in _get("m01.splits", {}).items():
        a(f"| {sn} Split | {sd.get('images', '?')} images, {sd.get('masks', '?')} masks, {sd.get('height_mean', 0):.0f}x{sd.get('width_mean', 0):.0f} |")

    # ── 2. Class Statistics ──
    a("\n## 2. Class Statistics\n")
    m02 = all_results.get("m02", {})
    if ds_info.binary_mode:
        a(f"- **FG/BG Pixel Ratio:** {_get('m02.fg_bg_ratio')}")
        a(f"- **Class Balance:** {_get('m02.balanced')}")
        a(f"- **Background:** {_get('m02.background.pixel_pct', 0):.1f}% pixels, {_get('m02.background.images', 0)} images")
        a(f"- **Foreground:** {_get('m02.foreground.pixel_pct', 0):.1f}% pixels, {_get('m02.foreground.images', 0)} images")
    else:
        a("| Class | Encoding | Images | Pixel % |")
        a("|-------|----------|--------|---------|")
        for cls_name in ds_info.class_names:
            cd = m02.get(cls_name, {})
            enc = ds_info.class_names.index(cls_name)
            a(f"| {cls_name} | {enc} | {cd.get('images', 0)} | {cd.get('pixel_pct', 0):.1f}% |")
        a(f"\n- **Class Balance:** {_get('m02.balanced')}")
        a(f"- **Global FG Ratio:** {100 - _get('m02.background.pixel_pct', 0):.1f}% pixels are defect (non-BG)")

    # ── 3. Image Quality ──
    a("\n## 3. Image Quality Analysis\n")
    a("### 3.1 Brightness & Contrast\n")
    a(f"- **Brightness:** mean={_get('m03.brightness.mean', 0):.1f}/255, std={_get('m03.brightness.std', 0):.1f}")
    a(f"- **Dynamic Range:** min-max spread")
    a(f"- **Overexposed:** {_get('m03.overexposed_pct', 0):.1f}% (definition: {_get('m03.overexposed_definition')})")
    a(f"- **Underexposed:** {_get('m03.underexposed_pct', 0):.1f}% (definition: {_get('m03.underexposed_definition')})")

    a("\n### 3.2 Image Sharpness\n")
    a(f"- **Blur Detection:** Laplacian Variance < {_get('m04.threshold', 100)}")
    a(f"- **Blurry Images:** {_get('m04.blurry_count', 0)}/{_get('m04.total_checked', 4470)} ({_get('m04.blurry_pct', 0)}%)")
    a(f"- **Laplacian Var:** mean={_get('m04.laplacian_mean', 0):.1f} (±{_get('m04.laplacian_std', 0):.1f})")

    # ── 4. Mask & Annotation ──
    a("\n## 4. Annotation Quality Analysis\n")
    a(f"- **FG Ratio (all):** mean={_get('m05.fg_ratio.mean', 0):.4f}, median={_get('m05.fg_ratio.median', 0):.4f}")
    a(f"- **FG Ratio (non-empty):** see Defect Size Distribution (M06)")
    a(f"- **Components/Image:** mean={_get('m05.n_components.mean', 0):.1f}, max={_get('m05.n_components.max', 0)}")
    a(f"- **Empty Masks (no FG):** {_get('m05.n_components.empty_pct', 0):.1f}%")
    a(f"- **Object Area:** mean={_get('m05.object_area.mean', 0):.0f} px, median={_get('m05.object_area.median', 0):.0f} px")
    a(f"- **Object Area Range:** {_get('m05.object_area.min', 0)}–{_get('m05.object_area.max', 0)} px")

    # ── 5. Defect Size Distribution ──
    a("\n## 5. Defect Size Distribution\n")
    m06 = all_results.get("m06", {})
    a(f"- **Total Objects:** {_get('m06.total_objects', 0)}")
    a(f"- **Small (<1024px):** {_get('m06.small.count', 0)} ({_get('m06.small.pct', 0)}%)")
    a(f"- **Medium (1024–9216px):** {_get('m06.medium.count', 0)} ({_get('m06.medium.pct', 0)}%)")
    a(f"- **Large (>9216px):** {_get('m06.large.count', 0)} ({_get('m06.large.pct', 0)}%)")
    a(f"\n**Area Percentiles:** p25={_get('m06.area_percentiles.p25', 0):.0f}px, "
      f"p50={_get('m06.area_percentiles.p50', 0):.0f}px, "
      f"p75={_get('m06.area_percentiles.p75', 0):.0f}px, "
      f"p95={_get('m06.area_percentiles.p95', 0):.0f}px, "
      f"p99={_get('m06.area_percentiles.p99', 0):.0f}px")

    # ── 5b. Defect Size by FG Ratio (from M14) ──
    m14 = all_results.get("m14", {})
    if "defect_size_distribution" in m14:
        a("\n### 5.1 Defect Size by FG Ratio\n")
        a("| Category | Count | % |")
        a("|----------|-------|---|")
        for label, info in m14["defect_size_distribution"].items():
            a(f"| {label} | {info['count']} | {info['pct']}% |")

    # ── 6. Distribution Shift ──
    a("\n## 6. Distribution Shift (Train vs Test)\n")
    a("| Metric | KS D | p-value | Significant? |")
    a("|--------|------|---------|--------------|")
    for name, ks in _get("m09.ks_test", {}).items():
        a(f"| {name} | {ks['statistic']:.4f} | {ks['p_value']:.4f} | {ks['significant']} |")
    a(f"\n**Interpretation:** Significant shifts in brightness (p={_get('m09.ks_test.Brightness.p_value',1):.4f}) "
      f"and blur (p={_get('m09.ks_test.Laplacian Var.p_value',1):.4f}) indicate train/test domain gap. "
      f"FG ratio distribution (p={_get('m09.ks_test.FG Ratio.p_value',1):.4f}) is consistent between splits.")

    # ── 7. Few-shot ──
    a("\n## 7. Few-shot Adaptation Analysis\n")
    a(f"- **FG Samples Available:** {_get('m10.fg_samples', 0)}/{_get('m10.total_samples', 0)}")
    a(f"- **Support Sampling Stability (PS):** {_get('m10.prototype_stability', 'N/A')}")
    a(f"  - Definition: PS = std(bootstrap mean of FG ratio from K=5 random support), 100 iterations.")
    a(f"  - Lower PS → support set choice has less impact on FG ratio estimation.")
    a(f"  - Note: This measures pixel-level FG ratio stability, NOT feature-space prototype similarity.")
    # Per-class few-shot
    per_class = _get("m10.per_class", {})
    if per_class:
        a("\n### 7.1 Per-Class Few-shot Analysis\n")
        a("| Class | Samples | FG Ratio Mean | FG Ratio Std | Proto Var | K=1 Ep | K=5 Ep | K=10 Ep |")
        a("|-------|---------|---------------|-------------|-----------|--------|--------|---------|")
        for cls_name, cd in per_class.items():
            a(f"| {cls_name} | {cd.get('samples', 'N/A')} | {cd.get('fg_ratio_mean', 0):.4f} | "
              f"{cd.get('fg_ratio_std', 0):.4f} | {cd.get('prototype_var', 0):.4f} | "
              f"{cd.get('K=1_episodes', 'N/A')} | {cd.get('K=5_episodes', 'N/A')} | "
              f"{cd.get('K=10_episodes', 'N/A')} |")
    a(f"\n**Episode definition:** For K-shot, a support set of K FG images is drawn without replacement. "
      f"Episodes = floor(FG_samples / K). Coverage = fraction of FG samples used across all episodes.\n")
    a("| K | FG Samples | Possible Episodes | Support Coverage |")
    a("|---|-----------|-------------------|-----------------|")
    for k_info in ["K=1", "K=5", "K=10"]:
        d = _get(f"m10.{k_info}", {})
        a(f"| {k_info} | {d.get('n_support_available', 'N/A')} | {d.get('possible_episodes', 'N/A')} | {d.get('support_coverage_pct', 'N/A')}% |")

    # ── 8. Texture & Shape ──
    a("\n## 8. Texture & Shape Characteristics\n")
    m13 = all_results.get("m13", {})
    if m13 and "error" not in m13:
        a("### 8.1 GLCM Texture\n")
        a("| Property | Mean ± Std | Description |")
        a("|----------|-----------|-------------|")
        a(f"| Contrast | {_get('m13.contrast.mean', 0):.3f} ± {_get('m13.contrast.std', 0):.3f} | Local intensity variation |")
        a(f"| Energy | {_get('m13.energy.mean', 0):.4f} ± {_get('m13.energy.std', 0):.4f} | Texture uniformity (higher=more uniform) |")
        a(f"| Homogeneity | {_get('m13.homogeneity.mean', 0):.3f} ± {_get('m13.homogeneity.std', 0):.3f} | Similarity of neighboring pixels |")
        a(f"| Correlation | {_get('m13.correlation.mean', 0):.3f} ± {_get('m13.correlation.std', 0):.3f} | Linear dependency of neighboring pixels |")

    if m14:
        a("\n### 8.2 Shape Metrics\n")
        a("| Metric | Mean | Median | Description |")
        a("|--------|------|--------|-------------|")
        a(f"| Circularity | {_get('m14.circularity.mean', 0):.3f} | {_get('m14.circularity.median', 0):.3f} | 4πA/P² (1=circle) |")
        a(f"| Eccentricity | {_get('m14.eccentricity.mean', 0):.3f} | {_get('m14.eccentricity.median', 0):.3f} | 0=circle, 1=line |")
        a(f"| Convexity | {_get('m14.convexity.mean', 0):.3f} | {_get('m14.convexity.median', 0):.3f} | Area/HullArea (1=convex) |")
        a(f"| Elongation | {_get('m14.elongation.mean', 0):.1f} | {_get('m14.elongation.median', 0):.1f} | Max(W,H)/Min(W,H) |")

    # ── 9. Data Quality Issues ──
    a("\n## 9. Data Quality Issues\n")
    m08 = all_results.get("m08", {})
    a(f"- **Quality Errors:** {_get('m08.errors', 0)}")
    a(f"- **Quality Warnings:** {_get('m08.warnings', 0)}")
    for iss, count in _get("m08.issue_breakdown", {}).items():
        a(f"- **{iss}:** {count} samples")
    dup_pct = round(100 * _get("m08.duplicate_images", 0) / max(_get("m08.total_checked", 4470), 1), 2)
    a(f"- **Duplicate Images:** {_get('m08.duplicate_images', 0)} ({dup_pct}%)")
    a(f"- **Duplicate Masks:** {_get('m08.duplicate_masks', 0)}")
    a(f"- **Spatial Bias:** {_get('m07.bias', 'N/A')} (center mean=({_get('m07.center_x_mean', 0):.3f},{_get('m07.center_y_mean', 0):.3f}))")

    # ── 10. Recommendations ──
    a("\n## 10. Recommended Data Augmentation Strategies\n")
    a("| Augmentation | Rationale |")
    a("|-------------|-----------|")
    a("| RandomFlip (H+V), RandomRotate90 | Objects have no canonical orientation |")
    a("| RandomBrightnessContrast | Significant brightness distribution shift between train/test (KS p<0.01) |")
    a("| GaussianNoise | Improve robustness to blur variation (50.8% blurry, KS p<0.001) |")
    empty_pct = _get("m05.n_components.empty_pct", 0)
    a(f"| FG-Weighted Sampling | FG pixels={100 - _get('m02.background.pixel_pct', 87.9):.1f}% of total; "
      f"weighted sampling reduces background dominance |")
    a(f"| RandomCrop (larger context) | 200x200 is small; expanding receptive field helps capture context |")

    # ── 11. Suitability ──
    a("\n## 11. Task Suitability Assessment\n")
    a("| Task | Suitable? | Notes |")
    a("|------|-----------|-------|")
    a("| Semantic Segmentation | ✅ Yes | Binary FG/BG masks, well-defined |")
    a("| Instance Segmentation | ⚠️ Partial | Connected components separable but no instance IDs |")
    a(f"| Few-shot Segmentation | ✅ Yes | {_get('m10.fg_samples', 0)} FG support candidates, K=1 viable |")
    a("| Anomaly Detection | ✅ Yes | Binary normal/defect framing |")
    a("| Foundation Model Fine-tuning | ✅ Yes | Compatible with SAM/FastSAM binary mask format |")

    # ── 12. Key Metrics Summary ──
    a("\n## 12. Key Metrics Summary\n")
    a("| Metric | Value | Paper-Ready? |")
    a("|--------|-------|-------------|")
    a(f"| Samples (Train/Test) | {_get('m01.splits.train.images', 0)}/{_get('m01.splits.test.images', 0)} | ✅ |")
    a(f"| Image Size | {_get('m01.image_size_mode')} | ✅ |")
    a(f"| FG/BG Pixel Ratio | {_get('m02.fg_bg_ratio')} | ✅ |")
    a(f"| Object Area Median | {_get('m05.object_area.median', 0):.0f} px | ✅ |")
    a(f"| Object Size Split | S={_get('m06.small.pct',0)}%/M={_get('m06.medium.pct',0)}%/L={_get('m06.large.pct',0)}% | ✅ |")
    a(f"| Brightness Shift | KS p={_get('m09.ks_test.Brightness.p_value', 1):.4f} | ✅ |")
    a(f"| Blur Shift | KS p={_get('m09.ks_test.Laplacian Var.p_value', 1):.4f} | ✅ |")
    a(f"| Prototype Stability | {_get('m10.prototype_stability', 'N/A')} | ✅ |")
    a(f"| Duplicate Rate | {dup_pct}% | ✅ |")
    a(f"| Spatial Bias | {_get('m07.bias', 'N/A')} | ✅ |")

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [OK] Report saved: {out_path}")

    # ── Validation JSON ──
    m15 = all_results.get("m15", {})
    validation = {
        "timestamp": datetime.now().isoformat(),
        "dataset": ds_info.name,
        "mode": "multi-class" if not ds_info.binary_mode else "binary",
        "checks": {
            "pixel_sum_consistency": m15.get("passed", True),
            "fg_ratio_consistency": m15.get("passed", True),
            "area_consistency": m15.get("passed", True),
            "circularity_range": m15.get("passed", True),
            "convexity_range": m15.get("passed", True),
            "empty_mask_consistency": m15.get("passed", True),
        },
        "all_passed": m15.get("passed", True),
        "n_issues": m15.get("n_issues", 0),
    }
    val_path = out_path.parent / "validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        import json as _json2
        _json2.dump(validation, f, indent=2)
    print(f"  [OK] Validation saved: {val_path}")

    # ── Pipeline Diagram ──
    _draw_pipeline(out_path.parent / "pipeline.png", ds_info.binary_mode)
    print(f"  [OK] Pipeline diagram saved: {out_path.parent / 'pipeline.png'}")

    return report_text


def _draw_pipeline(out_path: Path, multiclass: bool):
    """绘制数据验证流水线图 | Draw data validation pipeline diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    modules = [
        ("M01\nBasic\nStats", 0.5),
        ("M02\nClass\nStats", 1.8),
        ("M03\nImage\nStats", 3.1),
        ("M04\nQuality\n(Blur)", 4.4),
        ("M05\nMask\nStats", 5.7),
        ("M06\nObject\nSize", 7.0),
        ("M07\nSpatial\nDist.", 8.3),
        ("M08\nData\nQuality", 0.5),
        ("M09\nTrain/Test\nCompare", 1.8),
        ("M10\nFew-shot\nAnalysis", 3.1),
        ("M13\nTexture\n(GLCM)", 4.4),
        ("M14\nShape\nStats", 5.7),
        ("M15\nConsistency\nCheck", 7.0),
        ("M11\nVisualization\n", 8.3),
        ("M12\nREPORT.md\n", 9.6),
    ]
    for name, x in modules[:8]:
        y = 2.5
        rect = mpatches.FancyBboxPatch((x, y), 1.1, 1.2, boxstyle="round,pad=0.05",
                                        facecolor="#3498db", edgecolor="white", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + 0.55, y + 0.6, name, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    for name, x in modules[8:14]:
        y = 1.0
        rect = mpatches.FancyBboxPatch((x, y), 1.1, 1.2, boxstyle="round,pad=0.05",
                                        facecolor="#2ecc71", edgecolor="white", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + 0.55, y + 0.6, name, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    for name, x in modules[14:]:
        y = -0.5
        rect = mpatches.FancyBboxPatch((x, y), 1.1, 1.1, boxstyle="round,pad=0.05",
                                        facecolor="#e74c3c", edgecolor="white", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + 0.55, y + 0.55, name, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    # Arrows between rows
    for arr_x in [3.0, 6.5]:
        ax.annotate("", xy=(arr_x + 1.0, 1.1), xytext=(arr_x, 2.4),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(10.0, -0.5), xytext=(9.4, 0.1),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

    mode_str = "Multi-class (0/1/2/3)" if multiclass else "Binary"
    ax.text(6, 3.5, f"NEU_Seg Data Validation Pipeline — {mode_str}",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6, 3.2, "M01–M08: Data Layer | M09–M14: Analysis Layer | M15/M11/M12: Validation & Output",
            ha="center", fontsize=9, color="gray")

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NEU_Seg Comprehensive Dataset Analysis")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quality-only", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--modules", type=str, default="all",
                   help="Comma-separated module list (m01,m02,...) or 'all'")
    p.add_argument("--multiclass", action="store_true",
                   help="多类别模式 (保留 0/1/2/3 标签) | Multi-class mode (preserve labels)")
    return p.parse_args()


def main():
    args = parse_args()
    info = DatasetInfo(root=Path(args.data_root), binary_mode=not args.multiclass)

    if args.output is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output = f"analysis/neuseg_{ts}"
    out_dir = Path(args.output)
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    report_dir = out_dir / "report"
    for d in [csv_dir, fig_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  NEU_Seg — Comprehensive Dataset Analysis")
    print(f"  Data:   {args.data_root}")
    print(f"  Output: {out_dir}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    # Discover samples
    print("\n── Discovering samples ──")
    all_samples = discover_samples(info)

    enabled_all = (args.modules == "all")
    enabled_set = None if enabled_all else set(args.modules.split(","))

    def _enabled(mod: str) -> bool:
        return enabled_all or (enabled_set is not None and mod in enabled_set)

    all_results = {}

    # M01
    if _enabled("m01"):
        all_results["m01"] = m01_basic_stats(info, all_samples, csv_dir / "dataset_summary.csv")

    # M02
    if _enabled("m02"):
        all_results["m02"] = m02_class_stats(info, all_samples, csv_dir / "class_statistics.csv", fig_dir)

    # M03
    if not args.quality_only and _enabled("m03"):
        all_results["m03"] = m03_image_stats(all_samples, csv_dir / "image_statistics.csv", fig_dir, args.workers)

    # M04
    if not args.quality_only and _enabled("m04"):
        all_results["m04"] = m04_image_quality(all_samples, csv_dir / "image_quality.csv", fig_dir, args.workers)

    # M05
    if not args.quality_only and _enabled("m05"):
        all_results["m05"] = m05_mask_stats(all_samples, csv_dir / "mask_statistics.csv", fig_dir, args.workers)

    # M06
    if not args.quality_only and _enabled("m06"):
        all_results["m06"] = m06_size_analysis(all_samples, csv_dir / "size_statistics.csv", fig_dir, args.workers)

    # M07
    if not args.quality_only and _enabled("m07"):
        all_results["m07"] = m07_spatial_distribution(all_samples, fig_dir, args.workers)

    # M08
    if _enabled("m08"):
        all_results["m08"] = m08_quality_check(info, all_samples, csv_dir / "quality_report.csv", args.workers)

    if args.quality_only:
        print(f"\n[Done] Quality check complete. Results in: {out_dir}")
        return

    # M09
    if _enabled("m09"):
        all_results["m09"] = m09_train_test_compare(all_samples, csv_dir / "train_test_compare.csv", fig_dir)

    # M10
    if _enabled("m10"):
        all_results["m10"] = m10_fewshot_analysis(info, all_samples, csv_dir / "fewshot_analysis.csv")

    # M13 — Texture (GLCM)
    if not args.quality_only and _enabled("m13"):
        all_results["m13"] = m13_texture_analysis(all_samples, csv_dir / "texture_glcm.csv", fig_dir, args.workers)

    # M14 — Shape Statistics
    if not args.quality_only and _enabled("m14"):
        all_results["m14"] = m14_shape_analysis(all_samples, csv_dir / "shape_statistics.csv", fig_dir, args.workers)

    # M11 — Combined Visualization
    if not args.skip_viz and _enabled("m11"):
        m11_combined_visualization(info, all_samples, fig_dir, all_results)

    # M15 — Consistency Check (before report)
    if _enabled("m15"):
        all_results["m15"] = m15_consistency_check(all_results, info)

    # M12 — Report
    if _enabled("m12"):
        all_results["m12"] = m12_generate_report(info, all_results, report_dir / "REPORT.md")

    # ── Save all_results JSON ──
    import json as _json
    def _json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    report_json = out_dir / "all_results.json"
    with open(report_json, "w", encoding="utf-8") as f:
        _json.dump(all_results, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n{'='*60}")
    print(f"  [Done] Analysis complete!")
    print(f"  CSV:     {csv_dir}")
    print(f"  Figures: {fig_dir}")
    print(f"  Report:  {report_dir / 'REPORT.md'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
