#!/usr/bin/env python3
"""
诊断目标尺寸分布 | Diagnose Object Size Distribution.
=====================================================

统计 iSAID 全图 instance 标注中每类的 bbox 尺寸分布,
计算超过 896px tile 边长的比例, 输出论文可用表格.

Diagnoses whether 896×896 tile context is a fundamental bottleneck
for large objects in aerial imagery.

用法 | Usage::
    python tools/diag/diag_object_size.py

输出 | Output:
    runs/diag/object_size_distribution.csv
    console: paper-ready markdown table
"""

from __future__ import annotations

import sys, json, csv
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# ── 配置 | Configuration ──
ANNO_DIR = _PROJECT_ROOT / "data" / "iSAID_processed"
TILE_SIZE = 896  # 当前 tile 边长 | Current tile side length

CATEGORY_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

OUT_DIR = _PROJECT_ROOT / "runs" / "diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_annotations(split: str) -> list[dict]:
    """加载全图 COCO 标注 | Load full-image COCO annotations."""
    path = ANNO_DIR / split / "annotations" / f"instances_{split}.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 构建 image_id → (width, height) 映射 | Build image_id → (W, H) map
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    anns = []
    for a in data["annotations"]:
        cat_id = a.get("category_id", 0)
        bbox = a.get("bbox", [0, 0, 0, 0])  # COCO: [x, y, w, h]
        img_w, img_h = img_sizes.get(a.get("image_id", -1), (0, 0))
        anns.append({
            "cat_id": cat_id,
            "cat_name": CATEGORY_NAMES.get(cat_id, f"cls_{cat_id}"),
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "area": a.get("area", 0),
            "img_w": img_w,
            "img_h": img_h,
        })
    return anns


def main():
    print("=" * 72)
    print("  Object Size Distribution Diagnosis")
    print(f"  Tile Size: {TILE_SIZE}×{TILE_SIZE}")
    print("=" * 72)

    # ── 加载所有标注 | Load all annotations ──
    all_anns = []
    for split in ["train", "val"]:
        anns = load_annotations(split)
        print(f"  Loaded {split}: {len(anns)} instances")
        all_anns.extend(anns)
    print(f"  Total: {len(all_anns)} instances\n")

    # ── 按类别统计 | Per-class statistics ──
    by_class = defaultdict(list)
    for a in all_anns:
        by_class[a["cat_id"]].append(a)

    # ── 构建表格 | Build table ──
    rows = []
    for cat_id in sorted(by_class.keys()):
        anns = by_class[cat_id]
        name = CATEGORY_NAMES.get(cat_id, f"cls_{cat_id}")
        n = len(anns)

        widths = np.array([a["bbox_w"] for a in anns])
        heights = np.array([a["bbox_h"] for a in anns])
        areas = np.array([a["area"] for a in anns])
        max_dims = np.maximum(widths, heights)  # 较长边 | Longer side

        # ── 超过 896 的比例 | Proportion exceeding 896 ──
        pct_w_over = 100 * (widths > TILE_SIZE).mean()
        pct_h_over = 100 * (heights > TILE_SIZE).mean()
        pct_either_over = 100 * ((widths > TILE_SIZE) | (heights > TILE_SIZE)).mean()

        rows.append({
            "cat_id": cat_id,
            "name": name,
            "n": n,
            "mean_w": widths.mean(),
            "median_w": np.median(widths),
            "max_w": widths.max(),
            "mean_h": heights.mean(),
            "median_h": np.median(heights),
            "max_h": heights.max(),
            "mean_area": areas.mean(),
            "median_area": np.median(areas),
            "max_area": areas.max(),
            "pct_w_over": pct_w_over,
            "pct_h_over": pct_h_over,
            "pct_either_over": pct_either_over,
        })

    # ═══════════════════════════════════════════════════════════════
    # Console 输出: 论文可用 Markdown 表格 | Paper-ready Markdown table
    # ═══════════════════════════════════════════════════════════════
    header = (f"{'Class':<22s} {'N':>6s} {'Mean W':>8s} {'Mean H':>8s} "
              f"{'Median W':>8s} {'Median H':>8s} {'Max W':>8s} {'Max H':>8s} "
              f"{'W>896':>8s} {'H>896':>8s} {'Either>896':>10s} {'Risk':>6s}")
    sep = "─" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  Per-Class Bounding Box Size Distribution (Full Image)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    # 按 pct_either_over 降序排列 | Sort by pct_either_over descending
    rows_sorted = sorted(rows, key=lambda r: r["pct_either_over"], reverse=True)

    for r in rows_sorted:
        # 风险评级 | Risk level
        if r["pct_either_over"] > 50:
            risk = "HIGH"
        elif r["pct_either_over"] > 20:
            risk = "MED"
        elif r["pct_either_over"] > 5:
            risk = "LOW"
        else:
            risk = "OK"

        print(f"{r['name']:<22s} {r['n']:>6d} "
              f"{r['mean_w']:>8.1f} {r['mean_h']:>8.1f} "
              f"{r['median_w']:>8.1f} {r['median_h']:>8.1f} "
              f"{r['max_w']:>8.1f} {r['max_h']:>8.1f} "
              f"{r['pct_w_over']:>7.1f}% {r['pct_h_over']:>7.1f}% "
              f"{r['pct_either_over']:>9.1f}% {risk:>6s}")

    print(sep)

    # ── 汇总 | Summary ──
    all_widths = np.array([a["bbox_w"] for a in all_anns])
    all_heights = np.array([a["bbox_h"] for a in all_anns])
    n_total = len(all_anns)
    pct_total_w = 100 * (all_widths > TILE_SIZE).mean()
    pct_total_h = 100 * (all_heights > TILE_SIZE).mean()
    pct_total = 100 * ((all_widths > TILE_SIZE) | (all_heights > TILE_SIZE)).mean()

    print(f"{'OVERALL':<22s} {n_total:>6d} "
          f"{all_widths.mean():>8.1f} {all_heights.mean():>8.1f} "
          f"{np.median(all_widths):>8.1f} {np.median(all_heights):>8.1f} "
          f"{all_widths.max():>8.1f} {all_heights.max():>8.1f} "
          f"{pct_total_w:>7.1f}% {pct_total_h:>7.1f}% "
          f"{pct_total:>9.1f}%")

    # ── 关键判据 | Key verdict ──
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  KEY FINDING                                        ║")
    print(f"  ║  {pct_total:.1f}% of all instances ({int(pct_total * n_total / 100)}/{n_total})   ║")
    print(f"  ║  have bbox W or H > {TILE_SIZE}px                           ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    # ── 高风险类 + FT vs ZS 对比 | High-risk classes + FT vs ZS ──
    print(f"\n  High-risk classes (either>896 > 20%) vs FT performance (K=1 Adaptive):")
    ft_results = {
        # From eval_fewshot_adaptive_K1_ImageShot_0707_2212 (Top-100% Oracle)
        "small_vehicle": 0.2468,
        "large_vehicle": 0.4144,
        "plane": 0.3121,
        "storage_tank": 0.2194,
        "ship": 0.3489,
        "harbor": 0.4367,
        "ground_track_field": 0.2653,
        "soccer_ball_field": 0.1895,
        "tennis_court": 0.5639,
        "swimming_pool": 0.2298,
        "baseball_diamond": 0.1935,
        "basketball_court": 0.3297,
        "bridge": 0.0596,
        "helicopter": 0.2358,
        "roundabout": 0.2321,
    }
    zs_results = {
        "small_vehicle": 0.0675, "large_vehicle": 0.105, "plane": 0.2065,
        "storage_tank": 0.3275, "ship": 0.0984, "harbor": 0.0136,
        "ground_track_field": 0.5069, "soccer_ball_field": 0.5125,
        "tennis_court": 0.2926, "swimming_pool": 0.0252,
        "baseball_diamond": 0.3388, "basketball_court": 0.3203,
        "bridge": 0.0911, "helicopter": 0.1577, "roundabout": 0.5105,
    }

    for r in rows_sorted:
        if r["pct_either_over"] > 20:
            ft = ft_results.get(r["name"], 0)
            zs = zs_results.get(r["name"], 0)
            delta = ft - zs
            flag = "FT < ZS" if delta < -0.05 else ("FT > ZS" if delta > 0.05 else "~tie")
            print(f"    {r['name']:<22s}: {r['pct_either_over']:>5.1f}% >896  "
                  f"FT={ft:.3f}  ZS={zs:.3f}  Δ={delta:+.3f}  {flag}")

    # ═══════════════════════════════════════════════════════════════
    # CSV 输出 | CSV output
    # ═══════════════════════════════════════════════════════════════
    csv_path = OUT_DIR / "object_size_distribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "cat_id", "name", "n",
            "mean_w", "median_w", "max_w",
            "mean_h", "median_h", "max_h",
            "mean_area", "median_area", "max_area",
            "pct_w_over_896", "pct_h_over_896", "pct_either_over_896",
            "ft_k1_adaptive", "zs_baseline", "ft_minus_zs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            ft = ft_results.get(r["name"], 0)
            zs = zs_results.get(r["name"], 0)
            writer.writerow({
                "cat_id": r["cat_id"], "name": r["name"], "n": r["n"],
                "mean_w": round(r["mean_w"], 1), "median_w": round(r["median_w"], 1),
                "max_w": round(r["max_w"], 1),
                "mean_h": round(r["mean_h"], 1), "median_h": round(r["median_h"], 1),
                "max_h": round(r["max_h"], 1),
                "mean_area": round(r["mean_area"], 1),
                "median_area": round(r["median_area"], 1),
                "max_area": round(r["max_area"], 1),
                "pct_w_over_896": round(r["pct_w_over"], 1),
                "pct_h_over_896": round(r["pct_h_over"], 1),
                "pct_either_over_896": round(r["pct_either_over"], 1),
                "ft_k1_adaptive": ft,
                "zs_baseline": zs,
                "ft_minus_zs": round(ft - zs, 4),
            })
    print(f"\n  [OK] CSV → {csv_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
