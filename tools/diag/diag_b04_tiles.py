#!/usr/bin/env python3
"""
B-04 数据诊断 | Tile dataset diagnostics.
检查: mask 类别范围、前景占比、类别分布.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid_tiles import FastISAIDTileDataset


def diagnose(root: str):
    for split in ["train", "val"]:
        ds = FastISAIDTileDataset(root, split=split, semantic=True)
        n = len(ds)
        print(f"\n{'='*60}")
        print(f"  [{split.upper()}] {n} tiles")
        print(f"{'='*60}")

        all_vals = set()
        all_fg_ratios = []
        class_counts = Counter()
        empty_count = 0
        fg_count = 0

        n_sample = min(20, n)
        for i in range(n_sample):
            masks = ds[i]["mask"]
            vals = masks.unique().tolist()
            fg = (masks > 0).float().mean().item()
            all_vals.update(vals)
            all_fg_ratios.append(fg)
            if fg == 0:
                empty_count += 1
            else:
                fg_count += 1
            print(f"  [{i:4d}] unique={vals}  fg_ratio={fg:.4f}")

        # 全局采样统计 | Global sampling stats
        sample_size = min(2000, n)
        idxs = np.random.choice(n, sample_size, replace=False)
        for i in idxs:
            masks = ds[i]["mask"]
            all_vals.update(masks.unique().tolist())
            all_fg_ratios.append((masks > 0).float().mean().item())
            for c in masks.unique().tolist():
                class_counts[c] += (masks == c).sum().item()

        fg_arr = np.array(all_fg_ratios)
        print(f"\n  ── Summary ──")
        print(f"  All unique values: {sorted(all_vals)}")
        print(f"  Num classes seen:   {len(all_vals)} (expected 16: 0 bg + 15 fg)")
        print(f"  Foreground ratio:   mean={fg_arr.mean():.4f}  "
              f"median={np.median(fg_arr):.4f}  "
              f"max={fg_arr.max():.4f}")
        print(f"  Tiles w/ 0 FG:      {(fg_arr==0).sum()}/{len(fg_arr)} "
              f"({(fg_arr==0).mean()*100:.1f}%)")
        print(f"  Tiles w/ >1% FG:    {(fg_arr>0.01).sum()}/{len(fg_arr)} "
              f"({(fg_arr>0.01).mean()*100:.1f}%)")
        print(f"  Tiles w/ >5% FG:    {(fg_arr>0.05).sum()}/{len(fg_arr)} "
              f"({(fg_arr>0.05).mean()*100:.1f}%)")
        print(f"\n  Per-class pixel counts (sampled {sample_size} tiles):")
        for c in sorted(class_counts.keys()):
            label = "BG" if c == 0 else f"class{c}"
            print(f"    {label:<10} {class_counts[c]:>12,}")

        if 255 in all_vals:
            print(f"\n  ⚠️  WARNING: ignore_index=255 found in mask!")

        if len(all_vals) <= 1:
            print(f"\n  ❌ CRITICAL: Only {len(all_vals)} unique value(s) — dataset is broken!")
        elif len(all_vals) < 5:
            print(f"\n  ⚠️  WARNING: Only {len(all_vals)} unique values — very few classes")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, default="data/iSAID_tiles")
    args = p.parse_args()
    diagnose(args.tile_root)
