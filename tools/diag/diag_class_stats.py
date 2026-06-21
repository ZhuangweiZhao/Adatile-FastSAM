#!/usr/bin/env python3
"""
iSAID 类别统计 | iSAID Class Statistics
========================================

统计两个维度:
    1. COCO 标注: 每类实例数 + bbox 面积分布 (GT 真实分布)
    2. Tile: FG>5% tile 中每类像素数 + 出现次数 (训练数据分布)

输出: 类别不均衡分析 + 尺寸分布 P10/P50/P90

用法:
    python tools/diag_class_stats.py
    python tools/diag_class_stats.py --isaid-root data/iSAID_processed --tile-root data/iSAID_tiles
"""

import sys, json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


from adatile.utils.label_mapping import build_mapping, _ID_TO_NAME as ISAID_NAMES, VALID_CAT_IDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--isaid-root", type=str, default="data/iSAID_processed")
    p.add_argument("--tile-root", type=str, default="data/iSAID_tiles")
    return p.parse_args()


def analyze_coco(args):
    """COCO 标注: 每类实例数 + bbox 面积分布."""
    ann_file = Path(args.isaid_root) / "train" / "annotations" / "instances_train.json"
    if not ann_file.exists():
        print(f"  COCO annotations not found: {ann_file}")
        return None

    with open(ann_file) as f:
        coco = json.load(f)

    # Per-category stats
    cat_instances = Counter()
    cat_areas = {}
    cat_bbox_w = {}
    cat_bbox_h = {}

    # 使用 JSON 自身的 categories 构建映射（兼容旧/新两种 JSON）
    # Build mapping from JSON's own categories (compatible with old & new JSONs)
    mapping, _ = build_mapping(coco.get("categories", []))

    for ann in coco["annotations"]:
        raw_id = ann.get("category_id", 0)
        code_id = mapping.get(raw_id, raw_id if raw_id in VALID_CAT_IDS else 0)
        if code_id <= 0:
            continue

        bbox = ann["bbox"]  # [x, y, w, h]
        w, h = bbox[2], bbox[3]
        area = ann.get("area", w * h)

        cat_instances[code_id] += 1
        cat_areas.setdefault(code_id, []).append(area)
        cat_bbox_w.setdefault(code_id, []).append(w)
        cat_bbox_h.setdefault(code_id, []).append(h)

    print(f"\n{'='*70}")
    print(f"  COCO GT: Instance Count + Object Size Distribution")
    print(f"  Source: {ann_file}")
    print(f"  Total instances: {sum(cat_instances.values()):,}")
    print(f"{'='*70}")
    print(f"  {'Class':<20} {'Instances':>8} {'Area P10':>8} {'Area P50':>8} {'Area P90':>10} {'W P50':>7} {'H P50':>7}")
    print(f"  {'─'*70}")

    for c in sorted(cat_instances.keys()):
        name = ISAID_NAMES.get(c, f"class{c}")
        n = cat_instances[c]
        areas = np.array(cat_areas[c])
        ws = np.array(cat_bbox_w[c])
        hs = np.array(cat_bbox_h[c])
        print(f"  {name:<20} {n:>8,} {np.percentile(areas,10):>8.0f} {np.percentile(areas,50):>8.0f} {np.percentile(areas,90):>10.0f} {np.percentile(ws,50):>7.0f} {np.percentile(hs,50):>7.0f}")

    return cat_instances, cat_areas


def analyze_tiles(args):
    """Tile数据: FG>5% tile 中每类像素数 + 出现次数."""
    from PIL import Image

    mask_dir = Path(args.tile_root) / "masks" / "train"
    if not mask_dir.exists():
        print(f"  Tile masks not found: {mask_dir}")
        return None

    masks = sorted(mask_dir.glob("*.png"))
    total = len(masks)
    fg5_count = 0

    class_pixels = Counter()
    class_tile_count = Counter()
    fg_ratios = []

    print(f"\n  Scanning {total:,} tiles...")

    for png in tqdm(masks, desc="  Tiles"):
        mask = np.array(Image.open(png))
        fg_r = (mask > 0).sum() / mask.size
        fg_ratios.append(fg_r)

        if fg_r > 0.05:
            fg5_count += 1
            for c in range(1, 16):
                cp = int((mask == c).sum())
                if cp > 0:
                    class_pixels[c] += cp
                    class_tile_count[c] += 1

    fg_arr = np.array(fg_ratios)

    print(f"\n{'='*70}")
    print(f"  Tile Class Stats (FG>5% tiles)")
    print(f"  Total tiles: {total:,}")
    print(f"  FG>5% tiles: {fg5_count:,} ({fg5_count/total*100:.1f}%)")
    print(f"  FG ratio: mean={fg_arr.mean():.4f} median={np.median(fg_arr):.4f} max={fg_arr.max():.4f}")
    print(f"{'='*70}")
    print(f"  {'Class':<20} {'Pixels':>12} {'TileCount':>10} {'Px/Tile':>10} {'Px% of FG':>10}")
    print(f"  {'─'*70}")

    total_fg_px = sum(class_pixels.values())
    for c in sorted(class_pixels.keys()):
        name = ISAID_NAMES.get(c, f"class{c}")
        px = class_pixels[c]
        nt = class_tile_count[c]
        px_per_tile = px / max(nt, 1)
        px_pct = px / max(total_fg_px, 1) * 100
        print(f"  {name:<20} {px:>12,} {nt:>10,} {px_per_tile:>10.0f} {px_pct:>9.1f}%")

    return class_pixels, class_tile_count, fg_arr


def main():
    args = parse_args()

    # ── 1. COCO GT 统计 ──
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  iSAID Class Statistics — Training Data Analysis" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")

    coco_stats = analyze_coco(args)

    # ── 2. Tile 统计 ──
    tile_stats = analyze_tiles(args)

    # ── 3. 对比分析 ──
    if coco_stats and tile_stats:
        cat_instances, cat_areas = coco_stats
        class_pixels, class_tile_count, fg_arr = tile_stats

        print(f"\n{'='*70}")
        print(f"  Cross-Validation: COCO vs Tile")
        print(f"{'='*70}")
        print(f"  {'Class':<20} {'GT Inst':>8} {'Tile Px':>12} {'Tile Tiles':>10} {'Area P50':>8}")
        print(f"  {'─'*70}")

        for c in sorted(cat_instances.keys()):
            name = ISAID_NAMES.get(c, f"class{c}")
            inst = cat_instances[c]
            areas = np.array(cat_areas[c])
            px = class_pixels.get(c, 0)
            nt = class_tile_count.get(c, 0)
            print(f"  {name:<20} {inst:>8,} {px:>12,} {nt:>10,} {np.percentile(areas,50):>8.0f}")

        # ── 异常检测 ──
        print(f"\n{'='*70}")
        print(f"  Anomaly Detection")
        print(f"{'='*70}")

        for c in sorted(cat_instances.keys()):
            name = ISAID_NAMES.get(c, f"class{c}")
            inst = cat_instances[c]
            px = class_pixels.get(c, 0)
            nt = class_tile_count.get(c, 0)
            areas = np.array(cat_areas[c])
            median_area = np.percentile(areas, 50)

            warnings = []
            if inst < 200:
                warnings.append(f"rare instances ({inst})")
            if nt < 50:
                warnings.append(f"rare tile count ({nt})")
            if median_area < 50:
                warnings.append(f"tiny objects (P50={median_area:.0f}px²)")

            if warnings:
                print(f"  ⚠ {name}: {', '.join(warnings)}")

        # ── 推荐 ──
        print(f"\n{'='*70}")
        print(f"  Recommendation for B-04 Training")
        print(f"{'='*70}")
        rare_classes = [c for c in sorted(cat_instances.keys())
                       if cat_instances[c] < 200
                       or class_tile_count.get(c, 0) < 50]
        if rare_classes:
            names = [ISAID_NAMES.get(c) for c in rare_classes]
            print(f"  Rare classes: {names}")
            print(f"  → Consider oversampling tiles containing these classes")
            print(f"  → Or use class-balanced sampling in DataLoader")
        if tile_stats:
            _, _, fg_arr = tile_stats
            n = (fg_arr > 0.05).sum()
            print(f"  → Training on FG>5% tiles ({n:,}) = {n/len(fg_arr)*100:.1f}% of all tiles")

    print()


if __name__ == "__main__":
    main()
