#!/usr/bin/env python3
"""
B-04 数据诊断 | Tile Dataset Diagnostics
===========================================

检查内容 | Checks performed:
    - mask 类别范围 (unique pixel values) | Mask category range (unique pixel values)
    - 前景占比分布 (foreground ratio)      | Foreground ratio distribution
    - 类别像素分布 (per-class pixel count) | Per-class pixel count distribution
    - 空 tile 比例 (empty tile ratio)      | Empty tile ratio
    - 255 ignore_index 检测               | 255 ignore_index detection

用途 | Purpose:
    验证数据集预处理是否正确，确定 FG>5% 过滤阈值是否合理。
    Verify dataset preprocessing correctness, determine if FG>5% filter threshold is appropriate.

用法 | Usage::
    python tools/diag/diag_b04_tiles.py --tile-root data/iSAID_tiles
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid_tiles import FastISAIDTileDataset


def diagnose(root: str):
    """
    对 train/val 分片执行数据质量诊断 | Run data quality diagnostics on train/val splits.

    :param root: Tile 数据集根目录 | Tile dataset root directory.
    :type root: str
    """
    for split in ["train", "val"]:
        # 加载数据集 | Load dataset
        ds = FastISAIDTileDataset(root, split=split, dense_labels=True)
        n = len(ds)
        print(f"\n{'='*60}")
        print(f"  [{split.upper()}] {n} tiles")
        print(f"{'='*60}")

        # 初始化统计容器 | Initialize stats containers
        all_vals = set()
        all_fg_ratios = []
        class_counts = Counter()
        empty_count = 0
        fg_count = 0

        # ═══ 小样本展示: 前 20 个 tile 详情 | Small sample: first 20 tiles detail ═══
        n_sample = min(20, n)
        for i in range(n_sample):
            masks = ds[i]["mask"]
            vals = masks.unique().tolist()
            fg = (masks > 0).float().mean().item()  # 前景占比 | Foreground ratio
            all_vals.update(vals)
            all_fg_ratios.append(fg)
            if fg == 0:
                empty_count += 1
            else:
                fg_count += 1
            print(f"  [{i:4d}] unique={vals}  fg_ratio={fg:.4f}")

        # 全局采样统计 (最多 2000 tile) | Global sampling stats (up to 2000 tiles)
        sample_size = min(2000, n)
        idxs = np.random.choice(n, sample_size, replace=False)
        for i in idxs:
            masks = ds[i]["mask"]
            all_vals.update(masks.unique().tolist())
            all_fg_ratios.append((masks > 0).float().mean().item())
            # 累计各类别像素数 | Accumulate per-class pixel counts
            for c in masks.unique().tolist():
                class_counts[c] += (masks == c).sum().item()

        # ═══ 汇总统计 | Summary Statistics ═══
        fg_arr = np.array(all_fg_ratios)
        print(f"\n  ── Summary | 汇总 ──")
        print(f"  All unique values: {sorted(all_vals)}")
        print(f"  Num classes seen:   {len(all_vals)} (expected 16: 0 bg + 15 fg)")
        print(f"  Foreground ratio:   mean={fg_arr.mean():.4f}  "
              f"median={np.median(fg_arr):.4f}  "
              f"max={fg_arr.max():.4f}")
        # 各级前景占比: 0%, >1%, >5% | FG ratio tiers: 0%, >1%, >5%
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

        # ═══ 数据质量警告 | Data Quality Warnings ═══
        if 255 in all_vals:
            print(f"\n  ⚠️  WARNING: ignore_index=255 found in mask!")
            print(f"     255 是 ignore_index，不应出现在实际 mask 中 | "
                  f"255 is ignore_index, should not appear in actual masks")

        if len(all_vals) <= 1:
            print(f"\n  ❌ CRITICAL: Only {len(all_vals)} unique value(s) — dataset is broken!")
            print(f"     可能原因: Step 1 未运行，所有 mask 全为 0 | "
                  f"Possible cause: Step 1 not run, all masks are 0")
        elif len(all_vals) < 5:
            print(f"\n  ⚠️  WARNING: Only {len(all_vals)} unique values — very few classes")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, default="data/iSAID_tiles")
    args = p.parse_args()
    diagnose(args.tile_root)
