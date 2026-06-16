#!/usr/bin/env python
"""Convert LoveDA → BSSegDataset-compatible layout.

Source:  数据集/LoveDA/{Train,Val}/{Train,Val}/{Urban,Rural}/
           ├── images_png/*.png
           └── masks_png/*.png  (class labels 1-7, no bg)

Output:  datasets/loveda/
           ├── TrainDataset/images/*.png
           ├── TrainDataset/masks/*.png
           ├── ValDataset/images/*.png
           └── ValDataset/masks/*.png

Masks preserve class labels (no background=0 means the label IDs stay as-is).
num_classes = max label + 1 (likely 8, with 0 as bg for images with fewer classes)

Run:
    python tools/convert_loveda.py
"""

import argparse, shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/数据集/LoveDA")
    p.add_argument("--out", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/loveda")
    p.add_argument("--workers", type=int, default=min(8, cpu_count()))
    return p.parse_args()


def copy_pair(args):
    """Copy one image-mask pair with unique naming."""
    img_path, mask_path, img_dir, mask_dir, idx = args
    try:
        # Unique name: urban_1366.png, rural_1.png
        scene = img_path.parent.parent.name  # Urban or Rural
        fname = f"{scene}_{img_path.stem}.png"
        out_img = img_dir / fname
        out_mask = mask_dir / fname

        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        if not out_mask.exists():
            shutil.copy2(mask_path, out_mask)
        return 1, fname, "ok"
    except Exception as e:
        return -1, str(img_path), str(e)


def main():
    args = parse_args()
    src = Path(args.src)

    split_map = {
        "train": ["Train/Train/Urban", "Train/Train/Rural"],
        "val":   ["Val/Val/Urban",   "Val/Val/Rural"],
    }
    out_map = {"train": "TrainDataset", "val": "ValDataset"}

    for split, subdirs in split_map.items():
        out_split = out_map[split]
        out_img = Path(args.out) / out_split / "images"
        out_mask = Path(args.out) / out_split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        tasks = []
        for sub in subdirs:
            img_dir = src / sub / "images_png"
            mask_dir = src / sub / "masks_png"
            if not img_dir.exists():
                print(f"  SKIP: {img_dir} not found")
                continue
            for img_f in sorted(img_dir.glob("*.png")):
                mask_f = mask_dir / img_f.name
                if mask_f.exists():
                    tasks.append((img_f, mask_f, out_img, out_mask, len(tasks)))

        print(f"[{split}] {len(tasks)} pairs to copy")

        done, errors = 0, 0
        with Pool(args.workers) as pool:
            for status, fname, msg in pool.imap_unordered(copy_pair, tasks, chunksize=100):
                if status == 1:
                    done += 1
                else:
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR: {fname}: {msg}")
                if (done + errors) % 500 == 0:
                    print(f"  ... {done}/{len(tasks)}")

        print(f"  [{split}] {done} copied, {errors} errors")

    # Count unique mask values
    print("\n=== Class stats ===")
    import numpy as np
    from PIL import Image
    all_vals = set()
    for mf in Path(args.out).rglob("masks/*.png"):
        vals = np.unique(np.array(Image.open(mf)))
        all_vals.update(vals.tolist())
    all_vals = sorted(all_vals)
    print(f"Unique mask values across all images: {all_vals}")
    print(f"num_classes = {max(all_vals) + 1} (including bg=0)")
    print(f"\nRun ablation with: --dataset datasets/loveda --num-classes {max(all_vals)}")


if __name__ == "__main__":
    main()
