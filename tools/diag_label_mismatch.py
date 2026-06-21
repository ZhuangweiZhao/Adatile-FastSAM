#!/usr/bin/env python3
"""
诊断: 检查 train vs val tile mask 的像素值分布是否一致
============================================================
Diagnosis: Check if train vs val tile mask pixel distributions are consistent.

如果 train 和 val 的 unique values 不同 → 类别映射不一致
如果同值的语义不同 → mapping 漂移

用法 | Usage:
    python tools/diag_label_mismatch.py --tile-root /root/autodl-tmp/iSAID_tiles --n-samples 50
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from collections import Counter
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid_tiles import FastISAIDTileDataset

# ISAID_CATEGORIES: 代码定义的类别 | Code-defined categories
ISAID_NAMES = {
    0: "background",
    1: "small_vehicle",
    2: "large_vehicle",
    3: "plane",
    4: "storage_tank",
    5: "ship",
    6: "harbor",
    7: "ground_track_field",
    8: "soccer_ball_field",
    9: "tennis_court",
    10: "swimming_pool",
    11: "road",
    12: "basketball_court",
    13: "bridge",
    14: "helicopter",
    15: "roundabout",
}


def analyze_split(tile_root, split, n_samples=50):
    """分析一个 split 的 mask 像素值分布 | Analyze mask pixel distribution for a split."""
    ds = FastISAIDTileDataset(tile_root, split=split, semantic=True)
    tiles = ds._tiles

    # 采样 | Sample
    if n_samples > 0 and len(tiles) > n_samples:
        import random
        random.seed(42)
        tiles = random.sample(tiles, n_samples)

    print(f"\n{'='*60}")
    print(f"  Split: {split} | Total tiles: {len(ds._tiles)} | Sampled: {len(tiles)}")
    print(f"{'='*60}")

    pixel_counts = Counter()
    fg_ratios = []
    class_per_tile = Counter()  # 每个类出现在多少个 tile | How many tiles each class appears in

    for fname in tqdm(tiles, desc=f"  Scanning {split}"):
        from PIL import Image
        mask = np.array(Image.open(ds._mask_dir / fname))
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counts[int(u)] += int(c)
            if u > 0:
                class_per_tile[int(u)] += 1
        fg_ratios.append((mask > 0).sum() / mask.size)

    print(f"\n  Pixel value distribution | 像素值分布:")
    print(f"  {'Value':<6} {'Name':<20} {'Pixel Count':>12} {'% of FG':>10} {'# Tiles':>8}")
    print(f"  {'-'*60}")
    total_fg = sum(c for v, c in pixel_counts.items() if v > 0)
    for v in sorted(pixel_counts.keys()):
        name = ISAID_NAMES.get(v, f"UNKNOWN_{v}")
        pct = pixel_counts[v] / total_fg * 100 if total_fg > 0 else 0
        n_tiles = class_per_tile.get(v, 0)
        flag = "⚠️ " if v not in ISAID_NAMES else "  "
        print(f"  {flag}{v:<5} {name:<20} {pixel_counts[v]:>12,} {pct:>9.2f}% {n_tiles:>8}")

    print(f"\n  FG ratio stats: mean={np.mean(fg_ratios):.4f}, "
          f"median={np.median(fg_ratios):.4f}, "
          f"min={np.min(fg_ratios):.4f}, max={np.max(fg_ratios):.4f}")

    unique_values = sorted(pixel_counts.keys())
    return unique_values, pixel_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--n-samples", type=int, default=50)
    args = p.parse_args()

    train_vals, train_counts = analyze_split(args.tile_root, "train", args.n_samples)
    val_vals, val_counts = analyze_split(args.tile_root, "val", args.n_samples)

    # ── 对比诊断 | Comparison diagnosis ──
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS | 诊断")
    print(f"{'='*60}")

    print(f"\n  Train unique values: {train_vals}")
    print(f"  Val unique values:   {val_vals}")

    only_train = set(train_vals) - set(val_vals)
    only_val = set(val_vals) - set(train_vals)

    if only_train:
        print(f"\n  ⚠️  ONLY in train: {only_train}")
    if only_val:
        print(f"\n  ⚠️  ONLY in val: {only_val}")

    if set(train_vals) == set(val_vals):
        print(f"\n  ✅ Train and val have the SAME set of unique values")
        print(f"     → Label encoding is CONSISTENT between splits")
    else:
        print(f"\n  ❌ TRAIN/VAL LABEL MISMATCH DETECTED!")
        print(f"     → Different pixel values in train vs val masks")
        print(f"     → This would explain val mIoU ≈ 0 despite train mIoU > 0.7")

    # 检查是否有超出 0-15 范围的值 | Check for out-of-range values
    all_vals = set(train_vals) | set(val_vals)
    out_of_range = [v for v in all_vals if v not in ISAID_NAMES]
    if out_of_range:
        print(f"\n  ⚠️  Out-of-range pixel values (not in 0-15): {out_of_range}")
        print(f"     → These would be treated as class 0 (background) by the model!")
        print(f"     → Model outputs 16 channels (0-15), cannot predict these values")

    # 建议 | Recommendation
    print(f"\n  RECOMMENDED FIX | 建议修复:")
    print(f"  1. 删除 prep_isaid_tiles.py 中的 ACTUAL_TO_CODE_ID 映射（第 95 行）")
    print(f"     因为 prep_isaid.py 已经做了映射，prep_isaid_tiles.py 不应再做一次")
    print(f"     OR: 在 render_semantic_mask 中直接用 ann['category_id'] 不映射")
    print(f"  2. 重新运行 prep_isaid_tiles.py --steps 1,2,3 --splits train,val")
    print(f"  3. 重新训练")


if __name__ == "__main__":
    main()
