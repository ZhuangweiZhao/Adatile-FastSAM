#!/usr/bin/env python3
"""
Tile-Mode Recall Ceiling via GT visibility analysis.
=====================================================
不依赖 FastSAM everything mode — 直接统计 GT 在 tile/P4/P3 下是否"可见"。

Counts what fraction of GT instances are resolvable at each feature level.
Pure resolution-based recall ceiling without running FastSAM.

用法 | Usage::
    python tools/instance/eval_tile_recall.py --src-root data/iSAID_processed
"""

import sys, argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm
from adatile.utils.label_mapping import ISAID_CATEGORIES, _ID_TO_NAME as ISAID_NAMES

# Feature stride for each level
STRIDES = {"p4": 16, "p3": 8}

# Visibility threshold: GT must occupy at least this many feature cells
MIN_FEATURE_CELLS = 1.0


def is_visible(bbox_area_px, stride):
    """Check if a bbox is visible at given stride. | 检查 bbox 在给定步长下是否可见."""
    # bbox width/height in feature cells
    side_px = np.sqrt(bbox_area_px)
    side_cells = side_px / stride
    return side_cells >= np.sqrt(MIN_FEATURE_CELLS)  # at least 1 cell on each side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--tile-stride", type=int, default=512)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    ann_path = src_root / args.split / "annotations" / f"instances_{args.split}.json"
    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    print(f"Split: {args.split}, {len(images)} images, {len(annotations)} annotations")

    # ── Compute image dimensions | 获取图像尺寸 ──
    # Try dims cache first
    dims_cache = src_root / args.split / "image_dims.json"
    if dims_cache.exists():
        img_dims = {int(k): tuple(v) for k, v in json.loads(dims_cache.read_text()).items()}
        print(f"Using cached image dims ({len(img_dims)} images)")
    else:
        import cv2
        img_dims = {}
        img_dir = src_root / args.split / "images"
        for img in tqdm(images, desc="Reading image dims"):
            im = cv2.imread(str(img_dir / img["file_name"]))
            if im is not None:
                img_dims[img["id"]] = (im.shape[1], im.shape[0])

    # ── Per-instance: estimate tile and feature visibility ──
    # 对每个 GT 实例，统计在不同 stride 下是否可见
    stats = {"total": 0}
    for stride_name, stride in STRIDES.items():
        stats[f"visible_{stride_name}"] = 0
    per_cls = defaultdict(lambda: {"total": 0, **{f"visible_{s}": 0 for s in STRIDES}})
    per_cls_full = defaultdict(lambda: {"total": 0, **{f"visible_{s}": 0 for s in STRIDES}})
    # 也统计全图 1024 resize 模式作为对比
    FULL_IMGSZ = 1024

    for ann in tqdm(annotations, desc="Checking visibility"):
        cat_id = ann["category_id"]
        if cat_id not in ISAID_NAMES:
            continue

        bbox = ann["bbox"]
        area = bbox[2] * bbox[3]
        if area <= 0:
            continue

        img_id = ann["image_id"]
        W, H = img_dims.get(img_id, (0, 0))
        if W <= 0 or H <= 0:
            continue

        # ── 全图 1024 resize 模式下的有效面积 | effective area after resize ──
        scale_full = FULL_IMGSZ / max(W, H)
        area_full = area * (scale_full ** 2)

        # ── Tile 模式：实例在原图中的面积直接保留 | tile preserves original area ──
        area_tile = area

        stats["total"] += 1
        per_cls[cat_id]["total"] += 1
        per_cls_full[cat_id]["total"] += 1

        for stride_name, stride in STRIDES.items():
            if is_visible(area_tile, stride):
                stats[f"visible_{stride_name}"] += 1
                per_cls[cat_id][f"visible_{stride_name}"] += 1

            if is_visible(area_full, stride):
                per_cls_full[cat_id][f"visible_{stride_name}"] += 1

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"  RESOLUTION-BASED RECALL CEILING (GT visibility analysis)")
    print(f"  Tile: {args.tile_size}x{args.tile_size}, stride={args.tile_stride}")
    print(f"  Full-image: resize to {FULL_IMGSZ}px")
    print(f"{'='*70}")
    print()

    print(f"  {'Category':<22} {'#GT':>6}  {'Tile P4':>8} {'Tile P3':>8}  {'Full P4':>8} {'Full P3':>8}")
    print(f"  {'-'*22} {'-'*6}  {'-'*8} {'-'*8}  {'-'*8} {'-'*8}")

    for cat_id in sorted(per_cls.keys()):
        name = ISAID_NAMES.get(cat_id, str(cat_id))
        n = per_cls[cat_id]["total"]
        tp4 = per_cls[cat_id]["visible_p4"] / max(n, 1) * 100
        tp3 = per_cls[cat_id]["visible_p3"] / max(n, 1) * 100
        fp4 = per_cls_full[cat_id]["visible_p4"] / max(n, 1) * 100
        fp3 = per_cls_full[cat_id]["visible_p3"] / max(n, 1) * 100
        print(f"  {name:<22} {n:>6}  {tp4:>7.1f}% {tp3:>7.1f}%  {fp4:>7.1f}% {fp3:>7.1f}%")

    # Totals
    n = stats["total"]
    print(f"  {'─'*60}")
    print(f"  {'OVERALL':<22} {n:>6}  {stats['visible_p4']/n*100:>7.1f}% "
          f"{stats['visible_p3']/n*100:>7.1f}%  "
          f"{sum(1 for a in annotations if a['category_id'] in ISAID_NAMES and is_visible(a['bbox'][2]*a['bbox'][3] * (FULL_IMGSZ/max(img_dims.get(a['image_id'],(1,1))[0], img_dims.get(a['image_id'],(1,1))[1]))**2, 16))/n*100:>7.1f}% "
          f"{sum(1 for a in annotations if a['category_id'] in ISAID_NAMES and is_visible(a['bbox'][2]*a['bbox'][3] * (FULL_IMGSZ/max(img_dims.get(a['image_id'],(1,1))[0], img_dims.get(a['image_id'],(1,1))[1]))**2, 8))/n*100:>7.1f}%")

    # Brief summary
    print(f"\n  Tile P4 ceiling: {stats['visible_p4']/n*100:.1f}%")
    print(f"  Tile P3 ceiling: {stats['visible_p3']/n*100:.1f}%")
    print(f"  Improvement from P4→P3: {stats['visible_p3']/n*100 - stats['visible_p4']/n*100:+.1f}pp")

    # Save
    out = {
        "tile_size": args.tile_size,
        "stride": args.tile_stride,
        "full_imgsz": FULL_IMGSZ,
        "total_gt": n,
        "tile_p4_visible_pct": round(stats["visible_p4"] / n * 100, 1),
        "tile_p3_visible_pct": round(stats["visible_p3"] / n * 100, 1),
        "per_class": {},
    }
    for cat_id in sorted(per_cls):
        name = ISAID_NAMES.get(cat_id, str(cat_id))
        out["per_class"][name] = {
            "n_gt": per_cls[cat_id]["total"],
            "tile_p4_pct": round(per_cls[cat_id]["visible_p4"] / max(per_cls[cat_id]["total"], 1) * 100, 1),
            "tile_p3_pct": round(per_cls[cat_id]["visible_p3"] / max(per_cls[cat_id]["total"], 1) * 100, 1),
        }

    out_path = Path("runs/tile_recall_ceiling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
