#!/usr/bin/env python3
"""
多尺寸 Tile 生成器 (iSAID-5i 官方 Fold 兼容) | Multi-Size Tile Generator.
=============================================================

从原始 iSAID 大图按指定尺寸切 tile，生成 PreCutTileAdapter 兼容的 metadata，
配合 --dataset isaid5i 使用官方 Fold + 类别定义。

用法 | Usage::

    # 生成 256/384/512/640/768/896 六种尺寸
    python tools/data/prep_isaid5i_multisize.py \
        --src-root /root/autodl-tmp/iSAID_processed \
        --dst-root /root/autodl-tmp/iSAID5i_tiles \
        --sizes 256,384,512,640,768,896 \
        --stride-ratio 0.57 --workers 8

输出结构 | Output Structure::

    iSAID5i_tiles/
    ├── tile_256/
    │   ├── train/
    │   │   ├── images/         # 256×256 RGB tiles
    │   │   ├── labels/         # 256×256 单通道 0-15 dense labels
    │   │   └── metadata.json   # {tiles: [{tile_name, classes}], class_to_tiles: {str→[int]}}
    │   └── val/
    │       └── ...
    ├── tile_384/
    │   └── ...
    └── tile_896/
        └── ...
"""

import sys, argparse, json, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Size Tile Generator for iSAID-5i")
    p.add_argument("--src-root", type=str, required=True,
                   help="iSAID COCO JSON root (same as --src-root for fastsam protocol)")
    p.add_argument("--dst-root", type=str, required=True,
                   help="Output root directory")
    p.add_argument("--sizes", type=str, default="256,384,512,640,768,896",
                   help="Comma-separated tile sizes to generate")
    p.add_argument("--stride-ratio", type=float, default=0.57,
                   help="Stride as fraction of tile size (default 0.57 → ~57% overlap)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers for tile generation")
    p.add_argument("--splits", type=str, default="train,val",
                   help="Splits to process (train,val)")
    p.add_argument("--metadata-only", action="store_true",
                   help="Only generate metadata.json, skip image/label saving")
    return p.parse_args()


def load_isaid_coco(src_root: str, split: str) -> dict:
    """加载 iSAID COCO JSON 并构建 image_id → (path, W, H, annotations) 映射."""
    ann_path = Path(src_root) / split / "annotations" / f"instances_{split}.json"
    with open(ann_path) as f:
        coco = json.load(f)

    img_dir = Path(src_root) / split / "images"
    img_map = {img["id"]: {
        "file_name": img["file_name"],
        "width": 0,
        "height": 0,
        "annotations": [],
    } for img in coco["images"]}

    # 从实际图像文件读取尺寸 | Read dimensions from actual image files
    print(f"  Reading image dimensions for {len(img_map)} images...")
    for img_id, info in tqdm(list(img_map.items()), desc="  Reading dims"):
        img_path = img_dir / info["file_name"]
        im = cv2.imread(str(img_path))
        if im is not None:
            info["height"], info["width"] = im.shape[:2]

    # 过滤掉无法读取的图像 | Filter unreadable images
    img_map = {k: v for k, v in img_map.items() if v["width"] > 0}
    print(f"  Valid images: {len(img_map)}")

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_map:
            img_map[img_id]["annotations"].append(ann)

    return img_map, img_dir


def render_dense_mask(img_info: dict) -> np.ndarray:
    """
    从 COCO annotations 渲染 dense label mask (0-15).
    使用标准 iSAID-5i 类别 ID (1-15).
    """
    H, W = img_info["height"], img_info["width"]
    mask = np.zeros((H, W), dtype=np.uint8)

    # 加载标准 iSAID-5i 类别映射: 我们的内部 ID → 标准 ID
    from adatile.utils.label_mapping import ISAID5I_CATEGORIES, _NAME_TO_ID

    # 我们的内部 ID → 标准 iSAID-5i ID 的映射
    # 内部: 1=small_vehicle, 2=large_vehicle, ...
    # 标准: 1=ship, 2=storage_tank, ...
    from adatile.utils.label_mapping import ISAID_CATEGORIES

    # 构建: 内部名称 → 标准 ID
    internal_name_to_std = {}
    for std_id, std_name in ISAID5I_CATEGORIES.items():
        internal_name_to_std[std_name] = std_id

    for ann in img_info["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in ISAID_CATEGORIES:
            continue
        internal_name = ISAID_CATEGORIES[cat_id]
        std_id = internal_name_to_std.get(internal_name)
        if std_id is None:
            continue

        segmentation = ann.get("segmentation", [])
        if not segmentation:
            continue

        for seg in segmentation:
            if len(seg) < 6:
                continue
            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly], std_id)

    return mask


def generate_tiles(img_info: dict, img_path: Path, tile_size: int, stride: int,
                   split: str) -> list[dict]:
    """对单张图像切 tile，返回 tile metadata 列表."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    H, W = img.shape[:2]
    # Render dense mask
    mask = render_dense_mask(img_info)

    tiles = []
    tile_idx = 0
    img_stem = Path(img_info["file_name"]).stem

    for y0 in range(0, H - tile_size + 1, stride):
        for x0 in range(0, W - tile_size + 1, stride):
            y1, x1 = y0 + tile_size, x0 + tile_size

            # Crop image + mask
            tile_img = img[y0:y1, x0:x1]
            tile_mask = mask[y0:y1, x0:x1]

            # Check: any FG pixels?
            classes = set(np.unique(tile_mask).tolist()) - {0}
            if not classes:
                continue  # skip all-background tiles

            tiles.append({
                "img_stem": img_stem,
                "x0": x0, "y0": y0,
                "tile_name": f"{img_stem}__x{x0}_y{y0}",
                "classes": sorted(classes),
                "img_idx": -1,  # will be set by caller
                "tile_idx": -1,
            })

    return tiles


def save_tile_data(tile_info: dict, img: np.ndarray, mask: np.ndarray,
                   img_dir: Path, label_dir: Path):
    """保存单个 tile 的图像和标签到磁盘."""
    tile_name = tile_info["tile_name"]
    cv2.imwrite(str(img_dir / f"{tile_name}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(label_dir / f"{tile_name}_label.png"), mask)


def main():
    args = parse_args()
    tile_sizes = [int(s.strip()) for s in args.sizes.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]

    print(f"[MultiSizeTileGen] Sizes: {tile_sizes}")
    print(f"[MultiSizeTileGen] Stride ratio: {args.stride_ratio}")
    print(f"[MultiSizeTileGen] Splits: {splits}")

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"{'='*60}")

        img_map, img_dir = load_isaid_coco(args.src_root, split)
        print(f"  Loaded {len(img_map)} images")

        for tile_size in tile_sizes:
            stride = max(1, int(tile_size * args.stride_ratio))
            out_dir = Path(args.dst_root) / f"tile_{tile_size}" / split
            out_img_dir = out_dir / "images"
            out_label_dir = out_dir / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  --- Tile size: {tile_size}px, stride: {stride}px ---")

            all_tiles = []
            for img_id, img_info in tqdm(img_map.items(), desc=f"    Tiling {tile_size}px"):
                img_path = img_dir / img_info["file_name"]
                tiles = generate_tiles(img_info, img_path, tile_size, stride, split)
                for t in tiles:
                    t["img_idx"] = img_id
                all_tiles.extend(tiles)

            # Assign tile_idx
            for i, t in enumerate(all_tiles):
                t["tile_idx"] = i

            # Build class_to_tiles
            class_to_tiles = defaultdict(list)
            for t in all_tiles:
                for c in t["classes"]:
                    class_to_tiles[str(c)].append(t["tile_idx"])

            # Save metadata
            metadata = {
                "tile_size": tile_size,
                "stride": stride,
                "overlap": 1.0 - stride / tile_size,
                "split": split,
                "n_tiles": len(all_tiles),
                "tiles": all_tiles,
                "class_to_tiles": dict(class_to_tiles),
            }
            meta_path = out_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"    Saved {len(all_tiles)} tiles → {meta_path}")

            # Save tile images + labels
            if args.metadata_only:
                print(f"    Skipping tile image save (--metadata-only)")
            else:
                print(f"    Generating tile images + labels (resume-safe)...")
                img_cache = {}
                saved, skipped = 0, 0
                for t in tqdm(all_tiles, desc="    Saving tiles"):
                    tile_name = t["tile_name"]
                    img_file = out_img_dir / f"{tile_name}.png"
                    label_file = out_label_dir / f"{tile_name}_label.png"
                    # 断点续传: 跳过已存在的 tile | Resume: skip existing tiles
                    if img_file.exists() and label_file.exists():
                        skipped += 1
                        continue

                    img_idx = t["img_idx"]
                    if img_idx not in img_cache:
                        img_path = img_dir / img_map[img_idx]["file_name"]
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        mask = render_dense_mask(img_map[img_idx])
                        img_cache[img_idx] = (img, mask)

                    img, mask = img_cache[img_idx]
                    y0, x0 = t["y0"], t["x0"]
                    tile_img = img[y0:y0+tile_size, x0:x0+tile_size]
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                    tile_mask = mask[y0:y0+tile_size, x0:x0+tile_size]
                    save_tile_data(t, tile_img, tile_mask, out_img_dir, out_label_dir)
                    saved += 1
                print(f"    Saved {saved} new tiles, skipped {skipped} existing")

    print(f"\n{'='*60}")
    print(f"  Done! Output: {args.dst_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
