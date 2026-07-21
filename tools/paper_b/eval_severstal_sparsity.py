#!/usr/bin/env python3
"""
Severstal 空间稀疏性分析 | Severstal Spatial Sparsity Analysis.
=============================================================

验证 AdaTile 在 Severstal 钢铁缺陷数据集上的核心假设:
    "少数 Tile 覆盖了绝大部分缺陷像素"

核心问题 | Core Questions:
    1. 不同 Tile 划分下，多少 Tile 是空的（无缺陷）？
    2. Oracle Top-K: 保留 Top-K% Tile 能覆盖多少缺陷像素？
    3. 理论计算量节省上界是多少？

实验设计 | Design:
    - 图像 256×1600 (H×W)，沿长度方向切 Tile
    - Tile 宽度档位: 50, 80, 100, 160, 200, 400, 800
    - 对每个档位: 虚拟切分 → 统计 fg_ratio → Oracle Top-K 曲线
    - 零训练成本 (Zero training cost): 仅需 GT mask，无需 backbone

与 iSAID 的关键差异 | Key Differences from iSAID:
    - 一维切分 (沿长度) vs 二维网格切分
    - 缺陷像素极度稀疏 (~0.1% vs ~5%)
    - 无缺陷图占 47% — 整张图 0 缺陷像素

用法 | Usage::
    python tools/paper_b/eval_severstal_sparsity.py
    python tools/paper_b/eval_severstal_sparsity.py --max-images 2000 --workers 8
"""

import sys, argparse, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend

logger = get_logger("severstal_sparsity")
logger.add_backend(ConsoleBackend())

# ═══════════════════════════════════════════════════════════════════════════
# 配置 | Configuration
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_H = 256   # Severstal 图像固定高度 | Fixed image height
IMAGE_W = 1600  # Severstal 图像固定宽度 | Fixed image width

# Tile 宽度档位 (高度固定 256) | Tile width levels (height fixed at 256)
TILE_WIDTHS = [50, 80, 100, 160, 200, 400, 800]

# Oracle Top-K 档位 | Oracle Top-K retention levels
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            60, 70, 80, 90, 100]

# Severstal 缺陷类名 | Severstal defect class names
CLASS_NAMES = {1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}


# ═══════════════════════════════════════════════════════════════════════════
# RLE 解码 | RLE Decoding
# ═══════════════════════════════════════════════════════════════════════════

def decode_rle(rle_str: str, shape: tuple = (256, 1600)) -> np.ndarray:
    """
    解码 Severstal RLE 格式为二值掩码 (Fortran 列优先).
    Decode Severstal RLE format to binary mask (Fortran column-major).

    :param rle_str: RLE 字符串, 如 "29102 12 29346 24 ..."
    :param shape: 掩码形状 (H, W). Default (256, 1600).
    :return: np.ndarray [H, W] uint8, 二值 {0, 1}.
    """
    if not rle_str or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)

    values = list(map(int, rle_str.strip().split()))
    for i in range(0, len(values), 2):
        start_pos = values[i] - 1          # 1-indexed → 0-indexed
        run_len = min(values[i + 1], h * w - start_pos)
        mask[start_pos:start_pos + run_len] = 1

    # Fortran (列优先) reshape | Fortran (column-major) reshape
    return mask.reshape(h, w, order="F")


def encode_rle(mask_flat: np.ndarray) -> str:
    """
    将展开的掩码编码为 RLE 字符串.
    Encode flattened binary mask to RLE string.
    """
    pixels = np.where(mask_flat == 1)[0]
    if len(pixels) == 0:
        return ""

    runs = []
    prev = pixels[0]
    run_start = prev
    run_len = 1
    for p in pixels[1:]:
        if p == prev + 1:
            run_len += 1
        else:
            runs.append(f"{run_start + 1} {run_len}")  # 1-indexed
            run_start = p
            run_len = 1
        prev = p
    runs.append(f"{run_start + 1} {run_len}")
    return " ".join(runs)


# ═══════════════════════════════════════════════════════════════════════════
# 单图分析 Worker | Single Image Analysis Worker
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_single_image(args_tuple: tuple) -> dict:
    """
    单图 worker: 解码掩码 → 虚拟切 Tile → 统计各宽度下的 fg_ratio.
    Single-image worker: decode mask → virtual tile cut → fg_ratio stats.

    与 iSAID B-01 的 _analyze_single_image 对应, 但:
    - 输入是 RLE 字符串列表 (非 COCO 标注)
    - 一维切分 (沿宽度 | along width), 非二维网格
    """
    (img_name, class_rles, tile_widths) = args_tuple

    h, w = IMAGE_H, IMAGE_W

    # ── 逐类解码 → 堆叠为 [H, W] 总缺陷掩码 | Per-class decode → stack to overall defect mask ──
    # 总掩码: 0=背景, N=defect class_id
    dense_mask = np.zeros((h, w), dtype=np.uint8)
    for cls_id, rle_str in class_rles.items():
        cls_mask = decode_rle(rle_str, shape=(h, w))
        dense_mask[cls_mask > 0] = cls_id

    # 二值前景 | Binary foreground
    fg_mask = (dense_mask > 0)
    total_fg_px = int(fg_mask.sum())

    results = {
        "img_name": img_name,
        "total_fg_pixels": total_fg_px,
        "has_defect": total_fg_px > 0,
        "tile_widths": {},
    }

    for tw in tile_widths:
        fg_ratios = []

        # 沿宽度方向切 Tile (高度固定 256) | Cut tiles along width (height fixed 256)
        for x in range(0, w, tw):
            tx = min(tw, w - x)  # 实际 tile 宽度 (处理边缘) | Actual tile width (handle boundary)
            tile_mask = fg_mask[:, x:x + tx]

            total_px = h * tx
            fg_px = int(tile_mask.sum())
            fg_ratio = fg_px / total_px if total_px > 0 else 0.0
            fg_ratios.append((fg_ratio, fg_px))

        results["tile_widths"][tw] = {
            "fg_ratios": fg_ratios,
            "n_tiles": len(fg_ratios),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 加载 Severstal 数据 | Load Severstal Data
# ═══════════════════════════════════════════════════════════════════════════

def load_severstal_data(csv_path: str, max_images: int = 0, seed: int = 42) -> list:
    """
    从 train.csv 加载 RLE 标注, 聚合为 (image_name → {class_id: rle_str}).
    同时扫描 train_images/ 目录, 将不在 CSV 中的图像标记为 clean (无缺陷).

    Load RLE annotations from train.csv, aggregate as (image_name → {class_id: rle_str}).
    Also scan train_images/ dir, marking images NOT in CSV as clean (defect-free).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["EncodedPixels"] = df["EncodedPixels"].fillna("")

    # 按图像聚合 | Aggregate by image
    img_to_rles = {}
    for _, row in df.iterrows():
        img_name = row["ImageId"]
        class_id = int(row["ClassId"])
        rle = str(row["EncodedPixels"]).strip()

        if img_name not in img_to_rles:
            img_to_rles[img_name] = {}
        if rle:
            img_to_rles[img_name][class_id] = rle

    # 扫描目录中的干净图像 (不在 CSV 中 = 无缺陷)
    # Scan directory for clean images (not in CSV = defect-free)
    img_dir = Path(csv_path).parent / "train_images"
    all_files = set(f.name for f in img_dir.glob("*.jpg"))
    csv_images = set(img_to_rles.keys())

    # 将未在 CSV 中的图像添加为空字典 (所有 Tile 都是空的)
    # Add images not in CSV with empty dict (all tiles are empty/defect-free)
    n_clean_added = 0
    for fname in sorted(all_files):
        if fname not in csv_images:
            img_to_rles[fname] = {}
            n_clean_added += 1

    logger.log_info("data",
                    f"CSV: {len(csv_images)} defect images, "
                    f"+{n_clean_added} clean images (not in CSV) "
                    f"= {len(img_to_rles)} total")

    # 构建可用列表 (所有图像都通过目录验证存在)
    available = [(img_name, class_rles) for img_name, class_rles in img_to_rles.items()]

    if max_images > 0 and len(available) > max_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(available), max_images, replace=False)
        available = [available[i] for i in idx]

    return available


# ═══════════════════════════════════════════════════════════════════════════
# 汇总统计 | Aggregate Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(all_img_results: list, tile_widths: list) -> dict:
    """
    汇总所有图像的 Tile 数据 → 各宽度统计.
    Aggregate tile data across all images → per-width statistics.
    """
    stats = {}
    for tw in tile_widths:
        # 收集该宽度下所有 tile 的 fg_ratio + fg_pixels
        all_fg_ratios = []
        all_fg_pixels = []
        for img_r in all_img_results:
            if tw in img_r["tile_widths"]:
                td = img_r["tile_widths"][tw]
                all_fg_ratios.extend([rp[0] for rp in td["fg_ratios"]])
                all_fg_pixels.extend([rp[1] for rp in td["fg_ratios"]])

        fg_arr = np.array(all_fg_ratios, dtype=np.float64)
        px_arr = np.array(all_fg_pixels, dtype=np.int64)
        n_total = len(fg_arr)

        # 空/稀疏/有意义 比例 (使用与缺陷检测相适应的阈值)
        # 缺陷像素极度稀疏 → 阈值下调 | Defect pixels are extremely sparse → lower thresholds
        # <0.01% = 空, 0.01%-0.1% = 稀疏, ≥0.1% = 有意义
        empty_ratio = float((fg_arr < 0.0001).mean())       # <0.01%
        sparse_ratio = float(((fg_arr >= 0.0001) & (fg_arr < 0.001)).mean())  # 0.01%-0.1%
        meaningful_ratio = float((fg_arr >= 0.001).mean())   # ≥0.1%

        # Oracle Top-K 累积曲线
        sorted_idx = np.argsort(fg_arr)[::-1]
        cum_fg = np.cumsum(px_arr[sorted_idx])
        total_fg = px_arr.sum()

        # 捕获 90%/95%/99% 前景所需的 Tile 比例
        fg_capture = {}
        for target_pct in [90, 95, 99]:
            if total_fg > 0:
                cum_frac = cum_fg / (total_fg + 1e-8)
                idx = np.searchsorted(cum_frac, target_pct / 100)
                n_needed = min(int(idx) + 1, n_total)
                tile_pct = n_needed / n_total * 100
            else:
                n_needed, tile_pct = 0, 0.0
            fg_capture[target_pct] = {
                "tiles_needed": int(n_needed),
                "tile_pct": round(float(tile_pct), 2),
            }

        # 前景像素分布 | FG pixel distribution
        if total_fg > 0:
            half_idx = n_total // 2
            bottom_half_fg = px_arr[sorted_idx[half_idx:]].sum()
            wasted_fg_pct = bottom_half_fg / total_fg * 100
        else:
            wasted_fg_pct = 0.0

        stats[tw] = {
            "n_tiles": int(n_total),
            "n_images": len(all_img_results),
            "avg_tiles_per_image": round(n_total / max(len(all_img_results), 1), 1),
            "empty_ratio": round(float(empty_ratio), 4),
            "sparse_ratio": round(float(sparse_ratio), 4),
            "meaningful_ratio": round(float(meaningful_ratio), 4),
            "total_fg_pixels": int(total_fg),
            "wasted_fg_bottom_half_pct": round(float(wasted_fg_pct), 2),
            "fg_capture": fg_capture,
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 全局缺陷统计 | Global Defect Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_defect_stats(all_img_results: list) -> dict:
    """
    计算全局缺陷统计: 类别分布、空图比例、每图缺陷面积.
    Global defect statistics: class distribution, clean ratio, defect area per image.
    """
    n_total = len(all_img_results)
    n_clean = sum(1 for r in all_img_results if not r["has_defect"])
    n_defect = n_total - n_clean

    fg_pixels = [r["total_fg_pixels"] for r in all_img_results]
    fg_nonzero = [p for p in fg_pixels if p > 0]

    return {
        "n_total_images": n_total,
        "n_clean_images": n_clean,
        "n_defect_images": n_defect,
        "clean_ratio": round(n_clean / n_total, 4),
        "total_fg_pixels": int(sum(fg_pixels)),
        "fg_pixels_median": float(np.median(fg_nonzero)) if fg_nonzero else 0,
        "fg_pixels_mean": float(np.mean(fg_nonzero)) if fg_nonzero else 0,
        "fg_pixels_max": int(max(fg_nonzero)) if fg_nonzero else 0,
        "fg_pixels_min": int(min(fg_nonzero)) if fg_nonzero else 0,
        "image_area": IMAGE_H * IMAGE_W,
        "fg_density_global": round(sum(fg_pixels) / (n_total * IMAGE_H * IMAGE_W), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════════════

def make_visualization(
    all_img_results: list,
    stats: dict,
    tile_widths: list,
    defect_stats: dict,
    output_dir: Path,
):
    """
    生成 6 面板可视化, 与 B-01 Oracle Top-K 格式对齐.
    Generate 6-panel visualization, aligned with B-01 Oracle Top-K format.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    tw_labels = [str(tw) for tw in tile_widths]

    # ═══ Panel 1: Oracle Top-K FG 保留率曲线 | Oracle Top-K FG Retention Curve ═══
    ax = axes[0, 0]
    colors = ["#E74C3C", "#E67E22", "#F39C12", "#27AE60", "#3498DB", "#8E44AD", "#2C3E50"]
    for tw, c in zip(tile_widths, colors):
        # 收集该宽度下所有 tile 的 fg_ratio + fg_pixels
        all_ratios = []
        all_pixels = []
        for r in all_img_results:
            if tw in r["tile_widths"]:
                td = r["tile_widths"][tw]
                all_ratios.extend([rp[0] for rp in td["fg_ratios"]])
                all_pixels.extend([rp[1] for rp in td["fg_ratios"]])

        fg_arr = np.array(all_ratios, dtype=np.float64)
        px_arr = np.array(all_pixels, dtype=np.int64)
        sorted_idx = np.argsort(fg_arr)[::-1]
        cum_fg = np.cumsum(px_arr[sorted_idx])
        total_fg = px_arr.sum()

        # 在每个 K% 点的保留率
        ret_vals = []
        for k in K_VALUES:
            n_keep = max(1, int(len(fg_arr) * k / 100))
            ret = cum_fg[min(n_keep - 1, len(cum_fg) - 1)] / (total_fg + 1e-8) * 100
            ret_vals.append(ret)
        ax.plot(K_VALUES, ret_vals, "o-", color=c, linewidth=2.0,
                markersize=5, label=f"W={tw} ({len(fg_arr)} tiles)")

    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=30, color="#E74C3C", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Tiles Kept (%)", fontsize=12)
    ax.set_ylabel("Defect Pixels Retained (%)", fontsize=12)
    ax.set_title(f"Oracle Top-K: Defect Retention vs Tile Budget\n"
                 f"({defect_stats['n_total_images']} images, Severstal Steel)",
                 fontsize=11)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # ═══ Panel 2: 各宽度下空 Tile 比例 | Empty Tile Ratio by Width ═══
    ax = axes[0, 1]
    x = np.arange(len(tile_widths))
    width = 0.25

    empty_vals = [stats[tw]["empty_ratio"] * 100 for tw in tile_widths]
    sparse_vals = [stats[tw]["sparse_ratio"] * 100 for tw in tile_widths]
    meaningful_vals = [stats[tw]["meaningful_ratio"] * 100 for tw in tile_widths]

    p1 = ax.bar(x - width, empty_vals, width, color="#E74C3C",
                label="Empty (<0.01% FG)", edgecolor="white", linewidth=0.5)
    p2 = ax.bar(x, sparse_vals, width, color="#F39C12",
                label="Sparse (0.01-0.1%)", edgecolor="white", linewidth=0.5)
    p3 = ax.bar(x + width, meaningful_vals, width, color="#27AE60",
                label="Meaningful (≥0.1%)", edgecolor="white", linewidth=0.5)

    # 标注数值
    for i, (ev, mv) in enumerate(zip(empty_vals, meaningful_vals)):
        ax.text(i - width, ev + 1, f"{ev:.0f}%", ha="center", fontsize=8, color="#E74C3C")
        ax.text(i + width, mv + 1, f"{mv:.0f}%", ha="center", fontsize=8, color="#27AE60")

    ax.set_xticks(x)
    ax.set_xticklabels([f"W={tw}" for tw in tile_widths], fontsize=10)
    ax.set_ylabel("% of Tiles", fontsize=11)
    ax.set_title("Tile Composition by Width\n"
                 f"(Thresholds adjusted for defect sparsity)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, 105)

    # ═══ Panel 3: 95% FG 捕获需要的 Tile 比例 | Tiles Needed for 95% FG ═══
    ax = axes[0, 2]
    capture_90 = [stats[tw]["fg_capture"][90]["tile_pct"] for tw in tile_widths]
    capture_95 = [stats[tw]["fg_capture"][95]["tile_pct"] for tw in tile_widths]
    capture_99 = [stats[tw]["fg_capture"][99]["tile_pct"] for tw in tile_widths]

    ax.plot(tw_labels, capture_90, "D-", color="#3498DB", linewidth=2,
            markersize=9, label="90% FG")
    ax.plot(tw_labels, capture_95, "o-", color="#E67E22", linewidth=2.5,
            markersize=10, label="95% FG")
    ax.plot(tw_labels, capture_99, "s-", color="#8E44AD", linewidth=2,
            markersize=9, label="99% FG")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("Tiles Needed (% of Total)", fontsize=11)
    ax.set_title("Top-K FG Capture vs Tile Width\n"
                 "(Lower = Better Sparsity)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ═══ Panel 4: Tile 数 vs 宽度 (计算量代理) | Tile Count vs Width (compute proxy) ═══
    ax = axes[1, 0]
    n_tiles_vals = [stats[tw]["n_tiles"] for tw in tile_widths]
    bars = ax.bar(x, n_tiles_vals, color="#3498DB", edgecolor="white", alpha=0.85)
    for i, v in enumerate(n_tiles_vals):
        ax.text(i, v + max(n_tiles_vals) * 0.02, f"{v:,}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={tw}" for tw in tile_widths], fontsize=10)
    ax.set_ylabel("Total Tiles", fontsize=11)
    ax.set_title("Total Tile Count vs Width\n"
                 f"({defect_stats['n_total_images']} images, 30% clean)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # ═══ Panel 5: 浪费分析 + 关键数字 | Waste Analysis + Key Numbers ═══
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ds = defect_stats
    # 找最佳宽度: 使 empty + capture 平衡 | Find best width: balance empty ratio + capture
    best_tw = tile_widths[0]
    best_score = 0
    for tw in tile_widths:
        s = stats[tw]
        score = s["empty_ratio"] * (100 - s["fg_capture"][95]["tile_pct"]) / 100
        if score > best_score:
            best_score = score
            best_tw = tw

    s_best = stats[best_tw]
    summary_lines = [
        "Severstal Spatial Sparsity",
        "=" * 42,
        "",
        f"Dataset: {ds['n_total_images']} images",
        f"  Clean (no defect): {ds['n_clean_images']} ({ds['clean_ratio']*100:.0f}%)",
        f"  With defect: {ds['n_defect_images']} ({100-ds['clean_ratio']*100:.0f}%)",
        f"  Total defect pixels: {ds['total_fg_pixels']:,}",
        f"  Global FG density: {ds['fg_density_global']*100:.4f}%",
        f"  Median FG px/defect img: {ds['fg_pixels_median']:.0f}",
        "",
        f"Best Tile Width: W={best_tw}",
        f"  Tiles/image: {s_best['avg_tiles_per_image']:.0f}",
        f"  Empty tiles: {s_best['empty_ratio']*100:.0f}%",
        f"  95% FG → Top {s_best['fg_capture'][95]['tile_pct']:.0f}% tiles",
        "",
        "Comparison with iSAID B-01:",
    ]

    r30_isaid_95fg = 55.0  # iSAID B-00: 1024px tile, 95% FG needs ~55% tiles
    # 估算 Severstal 的 95% FG 需要的 tile 比例 (在最佳宽度下)
    sev_95fg_tiles = s_best["fg_capture"][95]["tile_pct"]

    summary_lines += [
        f"  iSAID 1024: 95% FG → ~{r30_isaid_95fg:.0f}% tiles",
        f"  Severstal W={best_tw}: 95% FG → {sev_95fg_tiles:.0f}% tiles",
    ]

    if sev_95fg_tiles < r30_isaid_95fg:
        verdict = ("Verdict: Severstal MORE sparse\n"
                   f"  → AdaTile potentially MORE effective\n"
                   f"    than on iSAID!")
        color = "#27AE60"
    else:
        verdict = ("Verdict: Severstal LESS sparse\n"
                   f"  → AdaTile viable but needs\n"
                   f"    per-defect-class analysis")
        color = "#E67E22"

    for i, line in enumerate(summary_lines + [verdict]):
        y_pos = 9.5 - i * 0.35
        if line.startswith("Severstal"):
            ax.text(0.5, y_pos, line, fontsize=13, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("===="):
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        elif "Verdict" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top", color=color)
        elif "←" in line or "MORE" in line or "LESS" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top", color=color)
        else:
            ax.text(0.5, y_pos, line, fontsize=8.5,
                    fontfamily="monospace", va="top")

    # ═══ Panel 6: Per-image 缺陷像素直方图 | Per-image Defect Pixel Histogram ═══
    ax = axes[1, 2]
    fg_pixels = [r["total_fg_pixels"] for r in all_img_results]
    # 对数分箱: 0, 1-10, 11-100, 101-1000, 1001-10000, 10001+
    bins = [-0.5, 0.5, 10.5, 100.5, 1000.5, 10000.5, 50000.5]
    bin_labels = ["0\n(clean)", "1-10", "11-100", "101-1K", "1K-10K", "10K+"]
    bin_colors = ["#BDC3C7", "#3498DB", "#27AE60", "#F39C12", "#E74C3C", "#8E44AD"]

    counts, _ = np.histogram(fg_pixels, bins=bins)
    bars = ax.bar(range(len(bin_labels)), counts, color=bin_colors,
                  edgecolor="white", alpha=0.85)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c + max(counts) * 0.03, f"{c}\n({c/len(fg_pixels)*100:.1f}%)",
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_title(f"Defect Pixels per Image Distribution\n"
                 f"({defect_stats['n_total_images']} images, "
                 f"{defect_stats['n_clean_images']} clean)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # ── 保存 | Save ──
    fig.suptitle("Severstal Spatial Sparsity: Oracle Top-K Tile Selection\n"
                 f"({defect_stats['n_total_images']} images, 256×(50-800) tiles, "
                 f"FG density={defect_stats['fg_density_global']*100:.4f}%)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "severstal_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"best_tile_width": best_tw, "best_stats": s_best}


# ═══════════════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Severstal Spatial Sparsity — Oracle Top-K Analysis")
    p.add_argument("--data-root", type=str,
                   default="data/severstal-steel-defect-detection",
                   help="Severstal 数据集根目录")
    p.add_argument("--tile-widths", type=str,
                   default="50,80,100,160,200,400,800",
                   help="Tile 宽度列表 (逗号分隔)")
    p.add_argument("--max-images", type=int, default=0,
                   help="限制图片数 (0=全部 12568)")
    p.add_argument("--workers", type=int, default=1,
                   help="多进程数")
    p.add_argument("--output-dir", type=str,
                   default="runs/severstal_sparsity")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    tile_widths = [int(x.strip()) for x in args.tile_widths.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logging ──
    logger.add_backend(FileBackend(str(output_dir / "sparsity.jsonl")))

    csv_path = data_root / "train.csv"
    if not csv_path.exists():
        logger.log_info("error", f"train.csv not found: {csv_path}")
        sys.exit(1)

    # ── 加载数据 | Load Data ──
    logger.log_info("data", f"Loading Severstal annotations from {csv_path}...")
    samples = load_severstal_data(
        str(csv_path), max_images=args.max_images, seed=args.seed)
    logger.log_info("data", f"Loaded {len(samples)} images")

    # ── 构建任务 | Build Tasks ──
    tasks = [(img_name, class_rles, tile_widths)
             for img_name, class_rles in samples]

    # ── 处理 | Process ──
    logger.log_info("phase", f"Processing {len(tasks)} images × {len(tile_widths)} tile widths...")
    logger.log_info("phase", f"Tile widths: {tile_widths}")

    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_results = list(tqdm(
                ex.map(_analyze_single_image, tasks),
                total=len(tasks), desc="  Analyzing", unit="img"))
    else:
        all_results = [_analyze_single_image(t) for t in
                       tqdm(tasks, desc="  Analyzing", unit="img")]

    # ── 缺陷全局统计 | Global Defect Stats ──
    defect_stats = compute_defect_stats(all_results)

    logger.log_info("defect_stats", "")
    logger.log_info("defect_stats", "=" * 50)
    logger.log_info("defect_stats", "Severstal Global Defect Statistics")
    logger.log_info("defect_stats", "=" * 50)
    logger.log_info("defect_stats",
                    f"  Total images:       {defect_stats['n_total_images']:,}")
    logger.log_info("defect_stats",
                    f"  Clean (no defect):  {defect_stats['n_clean_images']:,} "
                    f"({defect_stats['clean_ratio']*100:.1f}%)")
    logger.log_info("defect_stats",
                    f"  With defect:        {defect_stats['n_defect_images']:,} "
                    f"({100-defect_stats['clean_ratio']*100:.1f}%)")
    logger.log_info("defect_stats",
                    f"  Total defect px:    {defect_stats['total_fg_pixels']:,}")
    logger.log_info("defect_stats",
                    f"  Global FG density:  {defect_stats['fg_density_global']*100:.4f}%")
    logger.log_info("defect_stats",
                    f"  Median FG px/img:   {defect_stats['fg_pixels_median']:.0f}")
    logger.log_info("defect_stats",
                    f"  Mean FG px/img:     {defect_stats['fg_pixels_mean']:.1f}")
    logger.log_info("defect_stats",
                    f"  Image area:         {defect_stats['image_area']:,} px")

    # ── 汇总统计 | Aggregate Stats ──
    stats = compute_stats(all_results, tile_widths)

    # ── 输出结果表 | Print Results Table ──
    header = (f"{'Width':>6}  {'Tiles':>8}  {'Tiles/Img':>9}  "
              f"{'Empty%':>7}  {'Sparse%':>8}  {'Meaningful%':>11}  "
              f"{'95%FG→TopK%':>11}  {'Bottom50%Waste':>15}")
    logger.log_info("results", "")
    logger.log_info("results", "Oracle Top-K Tile Analysis (Severstal)")
    logger.log_info("results", header)
    logger.log_info("results", "-" * 80)

    for tw in tile_widths:
        s = stats[tw]
        logger.log_info("results",
                        f"  {tw:>6}  {s['n_tiles']:>8,}  {s['avg_tiles_per_image']:>8.1f}  "
                        f"{s['empty_ratio']*100:>6.1f}%  {s['sparse_ratio']*100:>7.2f}%  "
                        f"{s['meaningful_ratio']*100:>10.2f}%  "
                        f"{s['fg_capture'][95]['tile_pct']:>8.1f}%  "
                        f"{s['wasted_fg_bottom_half_pct']:>13.1f}%")

    # ── 关键拐点 | Key Inflection Points ──
    logger.log_info("inflection", "")
    logger.log_info("inflection", "Key Inflection Points (95% Defect Pixels):")
    for tw in tile_widths:
        s = stats[tw]
        c95 = s["fg_capture"][95]
        save_pct = 100 - c95["tile_pct"]
        logger.log_info("inflection",
                        f"  W={tw}: Top {c95['tile_pct']:.1f}% tiles "
                        f"({c95['tiles_needed']}/{s['n_tiles']}) → 95% FG. "
                        f"Can save ~{save_pct:.0f}% compute.")

    # ── 可视化 | Visualization ──
    viz_data = make_visualization(
        all_results, stats, tile_widths, defect_stats, output_dir)

    # ── 保存 JSON | Save JSON ──
    import datetime
    summary = {
        "experiment": "Severstal Spatial Sparsity",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "data_root": str(data_root),
            "tile_widths": tile_widths,
            "n_images": len(all_results),
            "image_size": [IMAGE_H, IMAGE_W],
        },
        "defect_statistics": defect_stats,
        "oracle_topk": {},
    }
    for tw in tile_widths:
        s = stats[tw]
        summary["oracle_topk"][str(tw)] = {
            "n_tiles": s["n_tiles"],
            "avg_tiles_per_image": s["avg_tiles_per_image"],
            "empty_ratio": s["empty_ratio"],
            "sparse_ratio": s["sparse_ratio"],
            "meaningful_ratio": s["meaningful_ratio"],
            "total_fg_pixels": s["total_fg_pixels"],
            "wasted_fg_bottom_half_pct": s["wasted_fg_bottom_half_pct"],
            "fg_capture_90pct_tiles": s["fg_capture"][90]["tile_pct"],
            "fg_capture_95pct_tiles": s["fg_capture"][95]["tile_pct"],
            "fg_capture_99pct_tiles": s["fg_capture"][99]["tile_pct"],
        }

    json_path = output_dir / "severstal_sparsity_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.log_info("output", f"Results saved: {output_dir}/")
    logger.log_info("output", f"  - {json_path}")
    logger.log_info("output", f"  - {output_dir / 'severstal_sparsity.png'}")

    # ── 一句话结论 | One-line Conclusion ──
    best_tw = viz_data["best_tile_width"]
    s_best = stats[best_tw]
    logger.log_info("conclusion", "")
    logger.log_info("conclusion", "=" * 60)
    logger.log_info("conclusion",
                    f"Severstal Oracle: W={best_tw}, "
                    f"{s_best['empty_ratio']*100:.0f}% tiles empty (defect-free). "
                    f"Top {s_best['fg_capture'][95]['tile_pct']:.0f}% tiles → 95% FG. "
                    f"~{100 - s_best['fg_capture'][95]['tile_pct']:.0f}% compute safely saveable.")
    logger.log_info("conclusion",
                    f"iSAID comparison: iSAID 1024px → ~55% tiles for 95% FG.")
    logger.log_info("conclusion",
                    f"AdaTile Cross-Domain: Severstal {'MORE' if s_best['fg_capture'][95]['tile_pct'] < 55 else 'LESS'} "
                    f"sparse than iSAID.")
    logger.log_info("conclusion", "=" * 60)


if __name__ == "__main__":
    main()
