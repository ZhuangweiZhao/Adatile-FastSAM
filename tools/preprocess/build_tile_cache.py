#!/usr/bin/env python
"""Pre-compute tile cache for all images in a dataset.

Usage:
    python tools/preprocess/build_tile_cache.py \\
        --dataset datasets/iSAID \\
        --tile-sizes 384 768 1536 \\
        --cache-dir datasets/iSAID/tiles \\
        --split train val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute tile cache for AdaTile-FastSAM"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--tile-sizes", "-t", type=int, nargs="+",
        default=[384, 768, 1536],
        help="Tile side lengths.",
    )
    parser.add_argument(
        "--strides", "-s", type=int, nargs="+",
        default=None,
        help="Per-size strides (default: tile_size // 2).",
    )
    parser.add_argument(
        "--cache-dir", "-o", type=str, required=True,
        help="Output directory for tile cache.",
    )
    parser.add_argument(
        "--split", nargs="+", default=["train", "val"],
        help="Data splits to process.",
    )
    parser.add_argument(
        "--density-maps", action="store_true",
        help="Also pre-compute density maps.",
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=4,
        help="Parallel workers.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-cached tiles.",
    )
    return parser.parse_args()


def precompute_tile_cache(
    dataset_root: Path,
    tile_sizes: List[int],
    strides: List[int],
    cache_dir: Path,
    splits: List[str],
    compute_density: bool = False,
    num_workers: int = 4,
    resume: bool = True,
) -> None:
    """Pre-compute and cache tiles for an entire dataset.

    This is the primary entry point for offline tile cache generation.
    Run this once before starting training to avoid on-the-fly slicing overhead.
    """
    from adatile.datasets.cache import TileCache
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    cache = TileCache(str(cache_dir))

    for split in splits:
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist, skipping.")
            continue

        image_paths = sorted(image_dir.glob("*"))
        print(f"Processing {len(image_paths)} images in {split} split...")

        def process_image(img_path: Path) -> int:
            image_id = img_path.stem
            try:
                infos = cache.precompute_tiles(
                    str(img_path),
                    image_id,
                    tile_sizes,
                    strides,
                    split=split,
                )
                return len(infos)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return 0

        total_tiles = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, p): p for p in image_paths}
            for future in tqdm(as_completed(futures), total=len(futures)):
                total_tiles += future.result()

        print(f"  → {total_tiles} tiles cached.")

    # Save cache statistics
    stats = cache.get_cache_stats()
    stats_path = cache_dir / "cache_meta" / "tile_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nCache statistics:")
    for k, v in sorted(stats.items()):
        if "ratio" in k:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    cache_dir = Path(args.cache_dir)
    strides = args.strides or [ts // 2 for ts in args.tile_sizes]

    precompute_tile_cache(
        dataset_root=dataset_root,
        tile_sizes=args.tile_sizes,
        strides=strides,
        cache_dir=cache_dir,
        splits=args.split,
        compute_density=args.density_maps,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
