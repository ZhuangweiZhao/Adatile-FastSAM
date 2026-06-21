#!/usr/bin/env python3
"""
快速检查: train vs val mask 像素值是否在同一语义空间
==========================================================
Quick check: do train and val masks use the same pixel values?

最简验证 | Simplest verification:
    1. 随便挑一个类别 (如 plane=3)
    2. 在 train 和 val 中各找几个包含该类的 tile
    3. 看 mask 中该类的像素值是否一致

用法 | Usage:
    python tools/diag_quick_check.py --tile-root /root/autodl-tmp/iSAID_tiles
"""

import sys, argparse, random
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

ISAID_NAMES = {
    0: "bg", 1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor", 7: "ground_track",
    8: "soccer", 9: "tennis", 10: "pool", 11: "road",
    12: "basketball", 13: "bridge", 14: "helicopter", 15: "roundabout",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, required=True)
    args = p.parse_args()

    root = Path(args.tile_root)
    random.seed(42)

    for split in ["train", "val"]:
        mask_dir = root / "masks" / split
        if not mask_dir.exists():
            print(f"❌ {mask_dir} NOT FOUND")
            continue

        files = sorted(mask_dir.glob("*.png"))
        n_sample = min(100, len(files))
        samples = random.sample(files, n_sample)

        # 统计所有出现过的像素值 + 每个值覆盖多少 tile
        value_counter = Counter()
        value_tiles = Counter()  # 每个值出现在多少个 tile

        for f in samples:
            m = np.array(Image.open(f))
            for v in np.unique(m):
                value_counter[int(v)] += 1
                value_tiles[int(v)] += 1

        # 同时采样几个具体 tile 的 unique
        show_n = min(5, n_sample)
        print(f"\n{'='*60}")
        print(f"  [{split.upper()}] {len(files)} total tiles, sampled {n_sample}")
        print(f"{'='*60}")

        for f in samples[:show_n]:
            m = np.array(Image.open(f))
            vals = sorted(np.unique(m).tolist())
            fg_pct = (m > 0).sum() / m.size * 100
            names = [f"{v}({ISAID_NAMES.get(v, '?')})" for v in vals if v > 0]
            print(f"  {f.name}: values={vals} fg={fg_pct:.1f}% → {names}")

        print(f"\n  Value distribution across {n_sample} tiles:")
        print(f"  {'Val':<5} {'Name':<18} {'#Tiles':>8} {'Expected?':>10}")
        print(f"  {'-'*50}")
        for v in sorted(value_tiles.keys()):
            name = ISAID_NAMES.get(v, f"???_{v}")
            flag = "" if v in ISAID_NAMES else " ⚠️ UNKNOWN"
            print(f"  {v:<5} {name:<18} {value_tiles[v]:>8}{flag}")

    # ── 对比 | Comparison ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON | 对比")
    print(f"{'='*60}")

    train_vals = set()
    val_vals = set()
    for split in ["train", "val"]:
        mask_dir = root / "masks" / split
        files = sorted(mask_dir.glob("*.png"))
        samples = random.sample(files, min(200, len(files)))
        for f in samples:
            m = np.array(Image.open(f))
            train_vals if split == "train" else val_vals
            (train_vals if split == "train" else val_vals).update(np.unique(m).tolist())

    train_only = train_vals - val_vals
    val_only = val_vals - train_vals
    common = train_vals & val_vals

    print(f"  Common values: {sorted(common)}")
    print(f"  Train-only:     {sorted(train_only)}")
    print(f"  Val-only:       {sorted(val_only)}")

    if train_only or val_only:
        print(f"\n  ❌ LABEL SPACE MISMATCH!")
        print(f"     Train and val have DIFFERENT pixel value sets.")
        print(f"     This is the root cause of val mIoU ≈ 0.")
    else:
        print(f"\n  ✅ Label space CONSISTENT between train and val.")

    # ── 时间戳检查 | Timestamp check ──
    print(f"\n{'='*60}")
    print(f"  TIMESTAMP CHECK | 时间戳检查")
    print(f"{'='*60}")
    for split in ["train", "val"]:
        mask_dir = root / "masks" / split
        if mask_dir.exists():
            files = list(mask_dir.glob("*.png"))
            if files:
                mtimes = [f.stat().st_mtime for f in files[:100]]
                import datetime
                newest = datetime.datetime.fromtimestamp(max(mtimes))
                oldest = datetime.datetime.fromtimestamp(min(mtimes))
                print(f"  {split}: {len(files)} tiles, "
                      f"oldest={oldest.strftime('%Y-%m-%d %H:%M')}, "
                      f"newest={newest.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
