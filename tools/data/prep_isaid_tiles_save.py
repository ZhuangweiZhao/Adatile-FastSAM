#!/usr/bin/env python3
"""
预切分 iSAID tile 并保存到磁盘 (多进程加速) | Pre-cut iSAID tiles with multiprocessing.
===================================================================

将全图 → 滑窗 tile (896x896) → PNG 文件，文件名直接编码来源信息。
Uses multiprocessing + bbox-based class detection for speed.

加速策略：
  1. 多进程并行处理图像 (--workers)
  2. bbox-tile 相交检测代替 np.unique 扫描
  3. 低 PNG 压缩级别

用法 | Usage::
    python tools/data/prep_isaid_tiles_save.py \
        --src-root data/iSAID_processed \
        --dst-root data/iSAID_tiles_precut \
        --splits train,val \
        --workers 4
"""

import argparse, json, sys, os, time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.utils.label_mapping import ISAID_CATEGORIES

# PNG 写入参数: 压缩级别 1 (最快) | PNG write params: compression level 1 (fastest)
_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def _build_label_map(anns, H, W):
    """从 COCO annotations 构建多类 label map."""
    label_map = np.zeros((H, W), dtype=np.uint8)
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id not in ISAID_CATEGORIES:
            continue
        seg = ann.get("segmentation")
        if not seg or len(seg) == 0:
            continue
        if isinstance(seg, list):
            polys = []
            for poly_pts in seg:
                if len(poly_pts) < 6:
                    continue
                pts = np.array(poly_pts).reshape(-1, 2).astype(np.int32)
                polys.append(pts)
            if polys:
                cv2.fillPoly(label_map, polys, cat_id)
    return label_map


def _tile_classes_from_bbox(anns, x0, y0, tile_size, cls_set):
    """用 bbox 相交检测 tile 内有哪些类 — 比 np.unique 快 100×.
    Detect classes in tile via bbox overlap — 100x faster than np.unique."""
    x1, y1 = x0 + tile_size, y0 + tile_size
    present = set()
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        # 检查 bbox 与 tile 是否相交 | Check bbox-tile overlap
        if bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0:
            cid = ann["category_id"]
            if cid in cls_set:
                present.add(cid)
    return sorted(present)


def _process_single_image(args):
    """处理单张图像 (worker 函数) | Process single image (worker function)."""
    img_idx, img_info, anns, src_dir, dst_dir, tile_size, stride = args

    img_path = os.path.join(src_dir, "images", img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        return None, f"unreadable: {img_path}"

    H, W = img.shape[:2]
    if H < tile_size or W < tile_size:
        return None, f"too small: {W}x{H}"

    # Build label map
    label_map = _build_label_map(anns, H, W)
    img_stem = Path(img_info["file_name"]).stem

    img_subdir = os.path.join(dst_dir, "images")
    label_subdir = os.path.join(dst_dir, "labels")
    os.makedirs(img_subdir, exist_ok=True)
    os.makedirs(label_subdir, exist_ok=True)

    cls_set = set(ISAID_CATEGORIES.keys())
    tiles_meta = []

    n_cols = (W - tile_size) // stride + 1
    n_rows = (H - tile_size) // stride + 1

    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * stride
            y0 = row * stride
            x1, y1 = x0 + tile_size, y0 + tile_size

            tile_name = f"{img_stem}__x{x0}_y{y0}"

            # Save tile image
            tile_img = img[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(img_subdir, f"{tile_name}.png"),
                       tile_img, _PNG_PARAMS)

            # Save label map
            tile_label = label_map[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(label_subdir, f"{tile_name}_label.png"),
                       tile_label, _PNG_PARAMS)

            # 用 bbox 检测类存在 (比 np.unique 快) | Detect classes via bbox (faster)
            present = _tile_classes_from_bbox(anns, x0, y0, tile_size, cls_set)

            tiles_meta.append({
                "img_idx": img_idx,
                "img_file": img_info["file_name"],
                "img_stem": img_stem,
                "x0": x0, "y0": y0,
                "tile_name": tile_name,
                "classes": present,
            })

    return tiles_meta, None


def cut_and_save(src_root: str, dst_root: str, split: str,
                 tile_size: int = 896, stride: int = 512,
                 n_workers: int = 1):
    """切分并保存一个 split 的所有 tile (多进程)."""
    src_dir = Path(src_root) / split
    dst_dir = Path(dst_root) / split
    os.makedirs(dst_dir / "images", exist_ok=True)
    os.makedirs(dst_dir / "labels", exist_ok=True)

    # — Load COCO JSON —
    ann_path = src_dir / "annotations" / f"instances_{split}.json"
    with open(ann_path) as f:
        data = json.load(f)
    n_images = len(data["images"])
    print(f"[{split}] {n_images} images, {len(data['annotations'])} annotations, "
          f"{n_workers} workers")

    # — Per-image annotation index —
    img_idx_to_anns = defaultdict(list)
    img_id_to_idx = {img["id"]: i for i, img in enumerate(data["images"])}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_id_to_idx:
            img_idx_to_anns[img_id_to_idx[img_id]].append(ann)

    # — 预先过滤掉无标注的小图 | Pre-filter images without useful annotations —
    valid_images = []
    for img_idx in range(n_images):
        img_info = data["images"][img_idx]
        anns = img_idx_to_anns.get(img_idx, [])
        valid_anns = [a for a in anns if a["category_id"] in ISAID_CATEGORIES]
        if valid_anns:
            valid_images.append((img_idx, img_info, valid_anns))

    print(f"[{split}] {len(valid_images)}/{n_images} images have valid annotations "
          f"({n_images - len(valid_images)} skipped as empty)")

    # — 多进程处理 | Multiprocessing —
    work_items = [(img_idx, img_info, anns, str(src_dir), str(dst_dir), tile_size, stride)
                  for img_idx, img_info, anns in valid_images]

    all_tiles = []
    t0 = time.perf_counter()

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_image, work_items, chunksize=8),
                total=len(work_items), desc=f"[{split}] Cutting"
            ))
    else:
        results = []
        for item in tqdm(work_items, desc=f"[{split}] Cutting"):
            results.append(_process_single_image(item))

    # — 合并结果 + 分配 tile_idx | Merge results + assign tile_idx —
    errors = []
    for tiles_meta, err in results:
        if err:
            errors.append(err)
            continue
        if tiles_meta:
            all_tiles.extend(tiles_meta)

    dt = time.perf_counter() - t0
    print(f"[{split}] {len(all_tiles)} tiles in {dt:.0f}s ({dt/60:.1f} min, "
          f"{len(all_tiles)/max(dt,1):.0f} tiles/s)")
    if errors:
        print(f"[{split}] {len(errors)} errors: {errors[:5]}...")

    # — 构建 class→tiles 索引 | Build class→tiles index —
    class_to_tiles = {str(k): [] for k in ISAID_CATEGORIES}
    for i, t in enumerate(all_tiles):
        t["tile_idx"] = i
        for cid in t["classes"]:
            class_to_tiles[str(cid)].append(i)

    # — 保存 metadata | Save metadata —
    metadata = {
        "tile_size": tile_size, "stride": stride,
        "overlap": tile_size - stride, "split": split,
        "n_tiles": len(all_tiles),
        "tiles": all_tiles,
        "class_to_tiles": class_to_tiles,
    }
    meta_path = dst_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    print(f"[{split}] Metadata → {meta_path}")

    # Class distribution
    for cid in sorted(ISAID_CATEGORIES):
        n = len(class_to_tiles.get(str(cid), []))
        if n > 0:
            print(f"  Class {cid:2d} ({ISAID_CATEGORIES[cid]:<20}): {n:5d} tiles")


def main():
    parser = argparse.ArgumentParser(description="Pre-cut iSAID tiles (multiprocessing)")
    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--dst-root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0,
                       help="进程数 (0=CPU核数, 1=单进程) | Worker count")
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else cpu_count()
    print(f"Using {n_workers} workers (CPU cores: {cpu_count()})")

    for split in args.splits.split(","):
        cut_and_save(args.src_root, args.dst_root, split.strip(),
                    args.tile_size, args.stride, n_workers)

    print(f"\nDone. Tiles saved to {args.dst_root}/")


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
