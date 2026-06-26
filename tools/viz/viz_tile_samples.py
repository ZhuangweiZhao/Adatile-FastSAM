#!/usr/bin/env python3
"""
在 tile 样本上可视化三类 (small_vehicle, storage_tank, ship) 的 mask 叠加。
Visualize 3-class masks overlaid on tile samples.

用法 | Usage:
    python tools/viz/viz_tile_samples.py --src-root data/iSAID_processed
"""

import argparse, sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper
from tools.instance.eval_c02a_fastsam_fewshot import ISAIDInstanceDataset


TARGET = {
    1:  {"name": "small_vehicle",  "color": (0.902, 0.212, 0.212)},   # red
    4:  {"name": "storage_tank",   "color": (0.165, 0.616, 0.561)},   # teal
    5:  {"name": "ship",           "color": (0.271, 0.482, 0.616)},   # blue
}


def overlay_mask(image, mask, color, alpha=0.5):
    """Overlay binary mask on image with given color. | 在图像上叠加彩色 mask."""
    img = image.astype(np.float32) / 255.0
    mask_bool = mask > 0
    overlay = img.copy()
    for c in range(3):
        overlay[:, :, c][mask_bool] = (
            overlay[:, :, c][mask_bool] * (1 - alpha) + color[c] * alpha
        )
    return (overlay * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/viz_tile_samples.png")
    parser.add_argument("--n-per-class", type=int, default=4,
                       help="每类样本数 | Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset with tile wrapper
    train_ds = ISAIDInstanceDataset(args.src_root, split="train")
    val_ds = ISAIDInstanceDataset(args.src_root, split="val")
    train_tiles = ISAIDTileWrapper(train_ds, tile_size=896, stride=512)
    val_tiles = ISAIDTileWrapper(val_ds, tile_size=896, stride=512)

    rng = np.random.RandomState(args.seed)

    # ══════════════════════════════════════════════════════════
    # Figure: 3 rows (classes) × N columns (samples)
    # ══════════════════════════════════════════════════════════
    n_cls = len(TARGET)
    n_per = args.n_per_class
    fig, axes = plt.subplots(n_cls, n_per, figsize=(n_per * 3.5, n_cls * 3.5))
    if n_per == 1:
        axes = axes.reshape(-1, 1)

    cls_stats = {}

    for row, cls_id in enumerate(sorted(TARGET)):
        info = TARGET[cls_id]
        candidates = train_tiles.class_to_images(cls_id)
        print(f"Class {cls_id} ({info['name']}): {len(candidates)} tiles with this class")

        # Pick random tiles that have sizable mask area
        good_samples = []
        for _ in range(min(200, len(candidates))):
            idx = int(rng.choice(candidates))
            mask = train_tiles.render_class_mask(idx, cls_id).numpy()
            fg_ratio = (mask > 0).sum() / mask.size
            if fg_ratio > 0.001:  # at least some foreground
                good_samples.append((idx, mask, fg_ratio))

        # Sort by fg_ratio to show diversity: small, medium, large coverage
        good_samples.sort(key=lambda x: x[2])

        if len(good_samples) >= n_per:
            step = max(1, len(good_samples) // n_per)
            selected = [good_samples[i] for i in range(0, len(good_samples), step)][:n_per]
        else:
            selected = good_samples[:n_per]

        areas_list = []
        for col, (idx, mask, fg_ratio) in enumerate(selected):
            img = train_tiles.load_image(idx).numpy().transpose(1, 2, 0)  # [H,W,3]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)

            # Overlay mask
            result = overlay_mask(img, mask, info["color"], alpha=0.4)

            ax = axes[row, col]
            ax.imshow(result)
            n_fg = (mask > 0).sum()
            areas_list.append(n_fg)
            ax.set_title(f"fg={n_fg}px ({fg_ratio*100:.2f}%)", fontsize=8)
            ax.axis("off")

        # Row label
        axes[row, 0].set_ylabel(info["name"].replace("_", "\n"), fontsize=11,
                                fontweight="bold", rotation=0, ha="right", va="center",
                                labelpad=30)

        cls_stats[cls_id] = {
            "n_tiles": len(candidates),
            "fg_pixels": [int((train_tiles.render_class_mask(i, cls_id).numpy() > 0).sum())
                         for i in candidates[:500]],  # sample 500 for speed
        }
        avg_fg = np.mean(cls_stats[cls_id]["fg_pixels"])
        print(f"  avg fg pixels: {avg_fg:.0f} px per tile (over {min(500, len(candidates))} sampled)")

    fig.suptitle("iSAID Tile Samples — small_vehicle / storage_tank / ship",
                fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # ══════════════════════════════════════════════════════════
    # Figure 2: FG pixel histogram per class
    # ══════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for cls_id in sorted(TARGET):
        fg_px = cls_stats[cls_id]["fg_pixels"]
        if fg_px:
            ax2.hist(fg_px, bins=50, alpha=0.5, label=TARGET[cls_id]["name"],
                    color=TARGET[cls_id]["color"])
    ax2.set_xlabel("Foreground pixels per tile", fontsize=11)
    ax2.set_ylabel("Tile count", fontsize=11)
    ax2.set_title("Foreground Pixel Distribution per Tile (896x896)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    hist_path = str(output_path).replace(".png", "_hist.png")
    fig2.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {hist_path}")


if __name__ == "__main__":
    main()
