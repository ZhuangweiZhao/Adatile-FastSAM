#!/usr/bin/env python3
"""
B-01: Oracle Top-K Selection — GT 知道一切时，空间稀疏性的理论上界
====================================================================

B-00 证明 Spatial Sparsity 存在。B-01 回答核心问题：

    如果 Oracle 完美知道每个 tile 的前景含量,
    保留 Top-K% 的 tile, 能保留多少前景信息?

这是 Spatial Sparsity 的理论上界 (Upper Bound)。
任何可学习的 Router 都不可能超过这个数字。

实验设计 | Design:
    固定 tile_size=1024 (B-00 验证的理论最优)
    K ∈ [5%, 10%, 15%, ..., 100%]
    指标: FG 像素保留率, Per-class FG 保留率

核心叙事 | Core narrative:
    "Top 30% tiles → X% FG retained"
    → 大量空间计算可以安全丢弃

用法 | Usage:
    python tools/eval_b01_oracle_topk.py
    python tools/eval_b01_oracle_topk.py --max-images 500 --workers 8
"""

import sys, argparse, json
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

logger = get_logger("b01_oracle")
logger.add_backend(ConsoleBackend())  # 终端实时输出 | Real-time console output

# iSAID 类别名 | iSAID class names (standard ISAID_CATEGORIES)
from adatile.utils.label_mapping import _ID_TO_NAME as CLASS_NAMES

TILE_SIZE = 1024
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            60, 70, 80, 90, 100]
NUM_CLASSES = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="runs/b01_oracle_topk")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """渲染语义掩码 [H,W] uint8 (0-15) | Render semantic mask.
    直接使用 ann["category_id"]（映射已在预处理中完成 | mapping done in preprocessing）."""
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    # 遍历所有标注，逐实例渲染 | Iterate all annotations, render per instance
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0:
            continue
        seg = ann.get("segmentation", [])
        # 无分割区域 → 回退到 bbox 矩形填充 | No segmentation → fallback to bbox fill
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            sem[y1:y2, x1:x2] = cat_id
            continue
        # RLE 格式暂不支持 | RLE format not yet supported
        if isinstance(seg, dict):
            continue
        # 多边形渲染：统一包装为列表列表 | Polygon render: wrap into list-of-lists format
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            # 构建顶点数组并裁剪到图像边界 | Build vertex array and clip to image bounds
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
            cv2.fillPoly(sem, [pts], cat_id)  # OpenCV 多边形填充 | OpenCV polygon fill
    return sem


def _analyze_single_image(args_tuple: tuple) -> dict:
    """
    单图: 渲染mask → 虚拟切1024 tile → 统计每个tile的fg_ratio + 每类像素.
    Single image: render mask → virtual 1024 cut → per-tile fg stats.
    """
    (img_id, anns, h, w, ts) = args_tuple

    sem = render_semantic_mask(anns, h, w)

    # 虚拟切分 1024×1024 网格 | Virtual 1024×1024 grid cut
    tiles = []
    for y in range(0, h, ts):
        for x in range(0, w, ts):
            th, tw = min(ts, h - y), min(ts, w - x)  # 边缘 tile 尺寸调整 | Boundary tile size clamp
            tile_mask = sem[y:y+th, x:x+tw]

            total_px = th * tw
            fg_mask = (tile_mask > 0)
            fg_px = int(fg_mask.sum())
            fg_ratio = fg_px / total_px

            # 每类前景像素统计 | Per-class foreground pixel counting
            # 遍历 1..15 类，记录每类在该 tile 中的像素数（>0 才存储） | Iterate classes 1-15, store count if >0
            class_px = {}
            for c in range(1, NUM_CLASSES + 1):
                cnt = int((tile_mask == c).sum())
                if cnt > 0:
                    class_px[c] = cnt

            tiles.append({
                "y": y, "x": x,
                "fg_ratio": round(fg_ratio, 6),
                "fg_pixels": fg_px,
                "total_pixels": total_px,
                "class_pixels": class_px,
            })

    return {"img_id": img_id, "h": h, "w": w,
            "n_tiles": len(tiles), "tiles": tiles}


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    split = args.split
    ts = args.tile_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 COCO | Load COCO ──
    ann_file = src_root / split / "annotations" / f"instances_{split}.json"
    with open(ann_file) as f:
        coco = json.load(f)

    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    tasks = []
    for img_info in coco["images"]:
        anns = img_id_to_anns.get(img_info["id"], [])
        tasks.append((img_info["file_name"], anns,
                      img_info["height"], img_info["width"], ts))

    if args.max_images > 0:
        tasks = tasks[:args.max_images]

    logger.log_info("exp/start",
                    f"B-01 Oracle Top-K | {len(tasks)} images, tile={ts}")

    # ── 处理 | Process ──
    logger.log_info("b01/phase",
                    f"Oracle Top-K: {len(tasks)} images, tile_size={ts}")
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_results = list(tqdm(ex.map(_analyze_single_image, tasks),
                                    total=len(tasks), desc="  Virtual cut", unit="img"))
    else:
        all_results = [_analyze_single_image(t) for t in
                       tqdm(tasks, desc="  Virtual cut", unit="img")]

    # ── 汇总所有 tile | Collect all tiles ──
    all_tiles = []
    total_fg_pixels = 0
    total_class_pixels = {c: 0 for c in range(1, NUM_CLASSES + 1)}
    for img_r in all_results:
        for t in img_r["tiles"]:
            all_tiles.append(t)
            total_fg_pixels += t["fg_pixels"]
            for c, cnt in t["class_pixels"].items():
                total_class_pixels[c] += cnt

    n_total = len(all_tiles)
    fg_arr = np.array([t["fg_ratio"] for t in all_tiles], dtype=np.float64)
    px_arr = np.array([t["fg_pixels"] for t in all_tiles], dtype=np.int64)

    # 按 fg_ratio 降序排列 | Sort by fg_ratio descending
    sorted_idx = np.argsort(fg_arr)[::-1]

    logger.log_info("data",
                    f"Total tiles: {n_total:,}, "
                    f"Total FG pixels: {total_fg_pixels:,}, "
                    f"Avg tiles/img: {n_total/len(all_results):.1f}")

    # ═══════════════════════════════════════════════════════════════
    # Oracle Top-K 模拟 | Oracle Top-K Simulation
    # ═══════════════════════════════════════════════════════════════

    # 全局累积前景（按重要性降序） | Global cumulative FG (sorted by importance descending)
    cum_fg = np.cumsum(px_arr[sorted_idx])
    total_fg = px_arr.sum()

    # Per-class 累积前景曲线 | Per-class cumulative FG curves
    # 对 1..15 类分别构建累积数组，用于后续 per-class retention 计算 | Build per-class cumsum for retention calc
    class_cum = {c: np.cumsum(
        np.array([t["class_pixels"].get(c, 0) for t in all_tiles],
                 dtype=np.int64)[sorted_idx]
    ) for c in range(1, NUM_CLASSES + 1)}

    # ── Oracle Top-K 模拟 | Oracle Top-K simulation ──
    # 对每个 K% 档位，模拟 Oracle 选择 Top-K tile 后的前景保留率 | For each K%, compute FG retention
    oracle_results = {}
    for k in K_VALUES:
        n_keep = max(1, int(n_total * k / 100))  # 保留的 tile 数量 | Number of tiles kept
        fg_captured = cum_fg[n_keep - 1] if n_keep > 0 else 0  # 捕获的前景像素 | FG pixels captured
        fg_retention = fg_captured / (total_fg + 1e-8)  # 保留率 | Retention rate

        # Per-class 保留率计算 | Per-class retention calculation
        class_retention = {}
        for c in range(1, NUM_CLASSES + 1):
            total_c = total_class_pixels[c]
            if total_c > 0:
                captured_c = class_cum[c][n_keep - 1] if n_keep > 0 else 0
                class_retention[c] = captured_c / total_c
            else:
                class_retention[c] = 0.0

        oracle_results[k] = {
            "n_tiles_kept": n_keep,
            "tile_pct": k,
            "fg_retention": round(float(fg_retention), 6),
            "fg_captured": int(fg_captured),
            "class_retention": {c: round(float(r), 6)
                                for c, r in class_retention.items()},
        }

    # ── 日志记录结果表 | Log results table ──
    header = (f"{'Top-K%':>7}  {'Tiles Kept':>10}  "
              f"{'FG Retained%':>13}  {'Wasted FG%':>10}")
    logger.log_info("b01/table",
                    f"Oracle Top-K Selection (tile_size={ts}, "
                    f"{len(all_results)} images, {n_total:,} tiles)")
    logger.log_info("b01/table", header)

    for k in K_VALUES:
        r = oracle_results[k]
        logger.log_info("b01/table",
                        f"  {k:>6}%  {r['n_tiles_kept']:>10,}  "
                        f"{r['fg_retention']*100:>12.2f}%  "
                        f"{(1-r['fg_retention'])*100:>9.2f}%")

    # ── 关键拐点 (日志) | Key inflection points (logged) ──
    # 计算达到特定 FG 保留率需要的最小 tile 比例 | Compute min tile % needed for target retention
    milestones = [
        (90, "90% FG retained (minimal quality loss)"),
        (95, "95% FG retained (negligible quality loss)"),
        (99, "99% FG retained (near-perfect)"),
    ]
    logger.log_info("b01/inflection", "Key Inflection Points:")
    for target_pct, desc in milestones:
        # 二分查找：累积 FG 达到目标百分比的位置 | Binary search for cumulative FG threshold
        idx = np.searchsorted(cum_fg / (total_fg + 1e-8), target_pct / 100)
        n_needed = min(int(idx) + 1, n_total)
        tile_pct = n_needed / n_total * 100
        logger.log_info("b01/inflection",
                        f"  {target_pct}% FG → Top {tile_pct:.1f}% tiles "
                        f"({n_needed:,}/{n_total:,}) — {desc}")

    # ═══════════════════════════════════════════════════════════════
    # 可视化 | Visualization
    # ═══════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    k_pcts = np.array(K_VALUES)
    fg_ret = np.array([oracle_results[k]["fg_retention"] * 100 for k in K_VALUES])

    # ═══ (1) FG 保留率曲线 — 核心图 | Panel 1: FG Retention Curve — core result ═══
    ax = axes[0, 0]
    ax.fill_between(k_pcts, 0, fg_ret, alpha=0.2, color="#27AE60")
    ax.plot(k_pcts, fg_ret, "o-", color="#27AE60", linewidth=2.5, markersize=8)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=30, color="#E74C3C", linestyle="--", alpha=0.3, linewidth=1)

    # 标注关键点 | Annotate key points
    for k_target in [20, 30, 40, 50]:
        if k_target in oracle_results:
            r = oracle_results[k_target]
            ax.annotate(f"Top {k_target}%\n→ {r['fg_retention']*100:.1f}% FG",
                        (k_target, r["fg_retention"] * 100),
                        textcoords="offset points", xytext=(5, -20),
                        fontsize=8, color="#2C3E50",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Tiles Kept (%)", fontsize=12)
    ax.set_ylabel("Foreground Retained (%)", fontsize=12)
    ax.set_title("Oracle Top-K: FG Retention vs Tile Budget\n"
                 f"(tile={ts}, {len(all_results)} images, {n_total:,} tiles)",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # ═══ (2) 边际收益 | Panel 2: Marginal gain — diminishing returns visualization ═══
    ax = axes[0, 1]
    marginal_gain = np.diff(fg_ret)
    k_mid = (k_pcts[:-1] + k_pcts[1:]) / 2
    bars = ax.bar(k_mid, marginal_gain, width=3.5, color="#3498DB",
                  edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    # 高亮最高边际收益 | Highlight highest marginal gain
    best_idx = np.argmax(marginal_gain)
    ax.bar(k_mid[best_idx], marginal_gain[best_idx], width=3.5,
           color="#E74C3C", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Tiles Kept (%)", fontsize=11)
    ax.set_ylabel("Δ FG Retention per 5% Tiles", fontsize=11)
    ax.set_title("Marginal Information Gain\n"
                 f"(Diminishing returns after ~30-40%)", fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # ═══ (3) FG 保留率摘要表 | Panel 3: Summary table — key numbers at a glance ═══
    ax = axes[0, 2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 构建结论文本 | Build conclusion text
    r30 = oracle_results[30]
    r40 = oracle_results[40]
    r50 = oracle_results[50]
    r20 = oracle_results[20]

    summary_lines = [
        "B-01: Oracle Top-K Upper Bound",
        "=" * 32,
        "",
        f"Dataset: iSAID {split}",
        f"  {len(all_results)} images → {n_total:,} tiles",
        f"  {total_fg_pixels:,} total FG pixels",
        "",
        "Oracle Retention:",
        f"  Top 20% tiles → {r20['fg_retention']*100:.1f}% FG",
        f"  Top 30% tiles → {r30['fg_retention']*100:.1f}% FG",
        f"  Top 40% tiles → {r40['fg_retention']*100:.1f}% FG",
        f"  Top 50% tiles → {r50['fg_retention']*100:.1f}% FG",
        "",
        "Meaning:",
        f"  Retaining {r30['fg_retention']*100:.0f}% FG with",
        f"  only 30% computation is the",
        f"  THEORETICAL UPPER BOUND.",
        "",
        "Conclusion:",
        f"  Spatial Sparsity can save",
        f"  ≥{100-r30['fg_retention']*100:.0f}% computation",
        f"  with minimal quality loss.",
    ]
    for i, line in enumerate(summary_lines):
        y_pos = 9.5 - i * 0.42
        if line.startswith("B-01"):
            ax.text(0.5, y_pos, line, fontsize=13, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("="):
            ax.text(0.5, y_pos, line, fontsize=9,
                    fontfamily="monospace", va="top", color="gray")
        elif "UPPER BOUND" in line or "Conclusion" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top", color="#E74C3C")
        elif "≥" in line and "computation" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top", color="#27AE60")
        else:
            ax.text(0.5, y_pos, line, fontsize=9,
                    fontfamily="monospace", va="top")

    # ═══ (4) Per-class FG 保留率热力图 | Panel 4: Per-class retention heatmap — fairness check ═══
    ax = axes[1, 0]
    # 取有前景的类别 | Get classes with foreground
    active_classes = [c for c in range(1, NUM_CLASSES + 1)
                      if total_class_pixels[c] > 1000]
    n_active = len(active_classes)
    heatmap_data = np.zeros((n_active, len(K_VALUES)))
    for i, c in enumerate(active_classes):
        for j, k in enumerate(K_VALUES):
            heatmap_data[i, j] = oracle_results[k]["class_retention"][c] * 100

    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"{k}%" for k in K_VALUES], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_active))
    ax.set_yticklabels([f"c{c} {CLASS_NAMES.get(c,'?')}" for c in active_classes],
                       fontsize=8)
    ax.set_title(f"Per-Class FG Retention\n"
                 f"({n_active} active classes)", fontsize=10)
    plt.colorbar(im, ax=ax, label="FG Retained (%)", shrink=0.8)

    # ═══ (5) FG 密度分箱贡献 | Panel 5: Density bin contribution — where FG actually lives ═══
    ax = axes[1, 1]
    density_bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2),
                    (0.2, 0.5), (0.5, 1.0)]
    bin_names = ["<1%", "1-5%", "5-10%", "10-20%", "20-50%", ">50%"]
    bin_colors = ["#BDC3C7", "#F39C12", "#E67E22", "#E74C3C",
                  "#8E44AD", "#27AE60"]
    bin_tile_counts = []
    bin_fg_contrib = []
    for lo, hi in density_bins:
        mask = (fg_arr >= lo) & (fg_arr < hi)
        bin_tile_counts.append(mask.sum())
        bin_fg_contrib.append(px_arr[mask].sum() / (total_fg + 1e-8) * 100)

    ax_twin = ax.twinx()
    bars = ax.bar(range(len(bin_names)), bin_tile_counts, color=bin_colors,
                  edgecolor="white", alpha=0.8)
    ax_twin.plot(range(len(bin_names)), bin_fg_contrib, "D-",
                 color="#2C3E50", linewidth=2.5, markersize=10, zorder=5)
    ax.set_xticks(range(len(bin_names)))
    ax.set_xticklabels(bin_names, fontsize=9)
    ax.set_ylabel("Number of Tiles", fontsize=11, color="#7F8C8D")
    ax_twin.set_ylabel("FG Contribution (%)", fontsize=11, color="#2C3E50")
    ax.set_title("Tile Density Distribution\n"
                 f"(How much FG does each density bin contribute?)", fontsize=10)
    ax.grid(axis="y", alpha=0.15)

    # 标注 | Labels
    for i, (tc, fgc) in enumerate(zip(bin_tile_counts, bin_fg_contrib)):
        ax.text(i, tc + max(bin_tile_counts) * 0.03, f"{tc:,}",
                ha="center", fontsize=8, color="#2C3E50")

    # ═══ (6) 浪费分析 | Panel 6: Waste analysis — empty/sparse/bottom-half breakdown ═══
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 计算浪费 | Compute waste analysis statistics
    # 空 tile (fg<1%): 完全浪费 — 包含极少量 FG 但占用大量计算 | Empty tiles: waste without benefit
    empty_mask = fg_arr < 0.01
    n_empty = empty_mask.sum()
    empty_fg_pct = px_arr[empty_mask].sum() / (total_fg + 1e-8) * 100

    # 稀疏 tile (fg 1-5%): 部分浪费 — 低密度前景，效率低 | Sparse tiles: marginal contribution, low efficiency
    sparse_mask = (fg_arr >= 0.01) & (fg_arr < 0.05)
    n_sparse = sparse_mask.sum()
    sparse_fg_pct = px_arr[sparse_mask].sum() / (total_fg + 1e-8) * 100

    # Oracle 可跳过: 按 fg_ratio 降序后，后 50% tile 仅贡献了多少 FG | Bottom 50% by FG sorting
    bottom_half = sorted_idx[n_total // 2:]  # bottom 50% by fg
    bottom_fg = px_arr[bottom_half].sum()
    bottom_fg_pct = bottom_fg / (total_fg + 1e-8) * 100

    r30 = oracle_results[30]
    r40 = oracle_results[40]

    waste_lines = [
        "Waste Analysis (tile=1024)",
        "=" * 32,
        "",
        f"Total tiles: {n_total:,}",
        f"Total FG pixels: {total_fg_pixels:,}",
        "",
        "Tile Categories:",
        f"  Empty (<1% FG): {n_empty:,} ({n_empty/n_total*100:.0f}%)",
        f"    → contribute {empty_fg_pct:.1f}% of FG",
        f"  Sparse (1-5% FG): {n_sparse:,} ({n_sparse/n_total*100:.0f}%)",
        f"    → contribute {sparse_fg_pct:.1f}% of FG",
        "",
        "Bottom 50% tiles:",
        f"  contribute only {bottom_fg_pct:.1f}% of FG",
        "",
        "Oracle Top-K Summary:",
        f"  Top 30% → {r30['fg_retention']*100:.1f}% FG retained",
        f"    (safe: ≥{100-r30['fg_retention']*100:.0f}% compute saved)",
        f"  Top 40% → {r40['fg_retention']*100:.1f}% FG retained",
        f"    (conservative: ≥{100-r40['fg_retention']*100:.0f}% saved)",
    ]
    for i, line in enumerate(waste_lines):
        y_pos = 9.5 - i * 0.42
        if line.startswith("Waste Analysis"):
            ax.text(0.5, y_pos, line, fontsize=12, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("="):
            ax.text(0.5, y_pos, line, fontsize=9,
                    fontfamily="monospace", va="top", color="gray")
        elif "compute saved" in line.lower() or "≥" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontweight="bold",
                    fontfamily="monospace", va="top", color="#27AE60")
        else:
            ax.text(0.5, y_pos, line, fontsize=9,
                    fontfamily="monospace", va="top")

    fig.suptitle("B-01: Oracle Top-K Selection — Spatial Sparsity Upper Bound",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_topk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 保存数据 | Save data ──
    import datetime
    summary = {
        "experiment": "B-01 Oracle Top-K",
        "timestamp": datetime.datetime.now().isoformat(),
        "source": f"iSAID {split} ({len(all_results)} images)",
        "tile_size": ts,
        "n_tiles": n_total,
        "total_fg_pixels": total_fg_pixels,
        "K_values": K_VALUES,
        "oracle_results": oracle_results,
        "inflection_points": {},
    }
    for target_pct, desc in milestones:
        idx = np.searchsorted(cum_fg / (total_fg + 1e-8), target_pct / 100)
        n_needed = min(int(idx) + 1, n_total)
        summary["inflection_points"][str(target_pct)] = {
            "tiles_needed": n_needed,
            "tile_pct": round(n_needed / n_total * 100, 2),
            "description": desc,
        }

    with open(output_dir / "oracle_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b01/output",
                    f"Results saved: {output_dir}/")
    logger.log_info("b01/output",
                    f"  - {output_dir / 'oracle_topk.png'}")
    logger.log_info("b01/output",
                    f"  - {output_dir / 'oracle_results.json'}")

    # ── 一句话结论 (日志) | One-line conclusion (logged) ──
    r30 = oracle_results[30]
    compute_saved = 100 - 30  # 保留 30% tile = 节省 70% 计算 | Keep 30% = save 70%
    logger.log_info("b01/conclusion",
                    f"Oracle Upper Bound: Top 30% tiles → "
                    f"{r30['fg_retention']*100:.1f}% FG retained. "
                    f"Spatial sparsity can safely save "
                    f"{compute_saved:.0f}% computation "
                    f"(only {100-r30['fg_retention']*100:.1f}% FG lost).")


if __name__ == "__main__":
    main()
