#!/usr/bin/env python3
"""
iSAID 离线切 Tile + 元数据 | Offline Tile Preprocessing + Metadata
===================================================================

三步流水线 | Three-step pipeline:

    Step 1: 生成全图语义掩码 (一次性, 只跑一次)
            render_semantic_mask() → P0001_mask.png
            后续切 tile 不再渲染 COCO。
    Step 2: 图像 + 掩码 → 切 1024×1024 tile + 保存
            可选 --format jpg (训练用, 更快更小)。
    Step 3: 生成 tile metadata JSON
            {name, img_id, x, y, building_ratio, class_distribution}

优化 | Optimizations:
    - uint8 全程 (不转 float32 再转回来)
    - cv2.imwrite (比 PIL 快 1.5~3×)
    - O(1) filename→id 查找
    - 多进程并行

输出 | Output (data/iSAID_tiles/):
    ├── masks_full/         # Step1: 全图语义掩码 {img_id}_mask.png
    ├── images/
    │   ├── train/          # Step2: tile images
    │   ├── val/
    │   └── test/
    ├── masks/
    │   ├── train/          # Step2: tile masks
    │   ├── val/
    │   └── test/
    └── metadata/
        ├── train.json      # Step3: tile metadata
        ├── val.json
        └── test.json

用法 | Usage:
    python tools/prep_isaid_tiles.py --max-images 50          # 快速测试
    python tools/prep_isaid_tiles.py --steps 1                 # 只渲染全图 mask
    python tools/prep_isaid_tiles.py --steps 2                 # 只切 tile
    python tools/prep_isaid_tiles.py --steps 3                 # 只生成 metadata
    python tools/prep_isaid_tiles.py --format jpg --workers 8  # 全量, JPG, 8 进程
"""

import sys, argparse, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="iSAID Offline Tile Preprocessing")
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--dst-root", type=str, default="data/iSAID_tiles")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--format", type=str, default="png",
                   choices=["png", "jpg"],
                   help="Tile 图像格式 | Tile image format (jpg=faster+smaller)")
    p.add_argument("--steps", type=str, default="1,2,3",
                   help="执行步骤 | Steps to run: 1=mask, 2=tile, 3=metadata")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """
    渲染语义掩码 [H, W] uint8 (值 0-15) | Render semantic mask.

    直接使用 ann["category_id"]。映射已在 prep_isaid.py fix_annotations() 中完成。
    Uses ann["category_id"] directly. Mapping is done by prep_isaid.py fix_annotations().

    优化: 使用 cv2.fillPoly, 避免 PIL 逐多边形开销。
    Optimized: uses cv2.fillPoly, avoids PIL per-polygon overhead.
    """
    sem = np.zeros((h, w), dtype=np.uint8)
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    for ann in annotations:
        # 直接使用 category_id, 不再二次映射。
        # prep_isaid.py fix_annotations() 已将原始 ID 映射为 ISAID ID (0-15)。
        # Use category_id directly, no double-mapping.
        # prep_isaid.py fix_annotations() already mapped original → ISAID ID (0-15).
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0 or cat_id > 15:
            continue

        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            sem[y1:y2, x1:x2] = cat_id
            continue
        if isinstance(seg, dict):
            continue

        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)

            if _has_cv2:
                cv2.fillPoly(sem, [pts], cat_id)
            else:
                # PIL fallback (slower)
                from PIL import Image, ImageDraw
                mask_pil = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask_pil).polygon(
                    [(int(p[0]), int(p[1])) for p in pts.reshape(-1, 2)],
                    fill=cat_id)
                sem = np.where(np.array(mask_pil) > 0, cat_id, sem)

    return sem


# ═══════════════════════════════════════════════════════════════════
# Step 1: 生成全图语义掩码 | Generate full-image semantic masks
# ═══════════════════════════════════════════════════════════════════

def _step1_worker(args_tuple: tuple) -> dict:
    """单图 worker: 渲染 + 保存全图 mask (独立进程 | separate process)."""
    from PIL import Image
    import cv2

    (img_path, anns, mask_path, split) = args_tuple

    if os.path.exists(mask_path):
        return {"img_path": img_path, "status": "skip"}

    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]

    if split != "test" and anns:
        sem = render_semantic_mask(anns, H, W)
    else:
        sem = np.zeros((H, W), dtype=np.uint8)

    cv2.imwrite(mask_path, sem)
    return {"img_path": img_path, "status": "rendered"}


def step1_render_masks(src_root: Path, dst_root: Path, splits: list,
                       max_images: int, workers: int, dry_run: bool):
    """Step 1: 生成全图语义掩码."""
    print("\n" + "=" * 60)
    print("  Step 1: Render Full-Image Semantic Masks")
    print("=" * 60)

    for split in splits:
        img_dir = src_root / split / "images"
        ann_file = src_root / split / "annotations" / f"instances_{split}.json"
        mask_dir = dst_root / "masks_full"
        mask_dir.mkdir(parents=True, exist_ok=True)

        # O(1) 查找: filename → annotations | O(1) lookup
        filename_to_anns = {}
        if ann_file.exists():
            with open(ann_file) as f:
                coco = json.load(f)
            # filename → image_id | O(N)
            fname_to_img_id = {img["file_name"]: img["id"] for img in coco["images"]}
            # image_id → annotations | O(N)
            img_id_to_anns = {}
            for ann in coco["annotations"]:
                img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
            # filename → annotations | O(1) lookup
            for fname, img_id in fname_to_img_id.items():
                filename_to_anns[fname] = img_id_to_anns.get(img_id, [])

        pngs = sorted(img_dir.glob("*.png"))
        if max_images > 0:
            pngs = pngs[:max_images]

        # 构建任务 | Build tasks
        tasks = []
        for png in pngs:
            mask_path = str(mask_dir / f"{png.stem}_mask.png")
            tasks.append((str(png), filename_to_anns.get(png.name, []),
                         mask_path, split))

        existing = sum(1 for t in tasks if os.path.exists(t[2]))
        need = len(tasks) - existing
        print(f"  [{split.upper()}] {len(tasks)} images ({existing} exist, {need} to render)")

        if dry_run or need == 0:
            continue

        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(tqdm(ex.map(_step1_worker, tasks),
                                    total=len(tasks), desc=f"  [{split}] Rendering",
                                    unit="img"))
        else:
            results = [tqdm(None, ...)]  # placeholder
            results = [_step1_worker(t) for t in tqdm(tasks, desc=f"  [{split}] Rendering", unit="img")]

        rendered = sum(1 for r in results if r["status"] == "rendered")
        print(f"    → {rendered} rendered, {len(results) - rendered} skipped")


# ═══════════════════════════════════════════════════════════════════
# Step 2: 切 Tile | Cut tiles
# ═══════════════════════════════════════════════════════════════════

def _step2_worker(args_tuple: tuple) -> dict:
    """单图 worker: 加载图像+mask → 切 tile → 保存 (独立进程)."""
    from PIL import Image
    import cv2

    (img_path, mask_full_path, img_id, img_out_dir, mask_out_dir,
     ts, img_fmt) = args_tuple

    # 读取 (uint8 全程 | uint8 throughout) | Read images as uint8
    img = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3] uint8
    if os.path.exists(mask_full_path):
        mask_full = np.array(Image.open(mask_full_path))  # [H, W] uint8
    else:
        mask_full = np.zeros(img.shape[:2], dtype=np.uint8)

    H, W = img.shape[:2]
    tile_idx = 0
    saved, skipped = 0, 0

    for y in range(0, H, ts):
        for x in range(0, W, ts):
            tile_name = f"{img_id}_t{tile_idx:03d}.{img_fmt}"
            tile_img_path = os.path.join(img_out_dir, tile_name)

            if os.path.exists(tile_img_path):
                skipped += 1
                tile_idx += 1
                continue

            th, tw = min(ts, H - y), min(ts, W - x)

            # 提取 (uint8 不转 float32) | Extract (keep uint8)
            tile_img = img[y:y+th, x:x+tw]     # [th, tw, 3] uint8
            tile_mask = mask_full[y:y+th, x:x+tw]  # [th, tw] uint8

            # 边界补零 | Pad edges
            if th < ts or tw < ts:
                padded = np.zeros((ts, ts, 3), dtype=np.uint8)
                padded[:th, :tw] = tile_img
                tile_img = padded
                padded_m = np.zeros((ts, ts), dtype=np.uint8)
                padded_m[:th, :tw] = tile_mask
                tile_mask = padded_m

            # 保存 (cv2 比 PIL 快 1.5~3×) | Save (cv2 faster than PIL)
            if img_fmt == "jpg":
                cv2.imwrite(tile_img_path, tile_img[:, :, ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(tile_img_path, tile_img[:, :, ::-1])

            mask_path = os.path.join(mask_out_dir, f"{img_id}_t{tile_idx:03d}.png")
            cv2.imwrite(mask_path, tile_mask)

            saved += 1
            tile_idx += 1

    return {"img_id": img_id, "tiles": tile_idx, "saved": saved, "skipped": skipped}


def step2_cut_tiles(src_root: Path, dst_root: Path, splits: list,
                    max_images: int, workers: int, ts: int, img_fmt: str,
                    dry_run: bool):
    """Step 2: 切 Tile."""
    print("\n" + "=" * 60)
    print(f"  Step 2: Cut Tiles ({ts}×{ts}, format={img_fmt})")
    print("=" * 60)

    for split in splits:
        img_dir = src_root / split / "images"
        mask_full_dir = dst_root / "masks_full"
        img_out = dst_root / "images" / split
        mask_out = dst_root / "masks" / split
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        pngs = sorted(img_dir.glob("*.png"))
        if max_images > 0:
            pngs = pngs[:max_images]

        tasks = []
        for png in pngs:
            img_id = png.stem
            mask_full_path = str(mask_full_dir / f"{img_id}_mask.png")
            tasks.append((str(png), mask_full_path, img_id,
                         str(img_out), str(mask_out), ts, img_fmt))

        existing = len(list(img_out.glob(f"*.{img_fmt}")))
        print(f"  [{split.upper()}] {len(tasks)} images ({existing} tiles exist)")

        if dry_run or not tasks:
            continue

        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(tqdm(ex.map(_step2_worker, tasks),
                                    total=len(tasks),
                                    desc=f"  [{split}] Cutting tiles",
                                    unit="img"))
        else:
            results = [_step2_worker(t) for t in
                       tqdm(tasks, desc=f"  [{split}] Cutting tiles", unit="img")]

        saved = sum(r["saved"] for r in results)
        skipped = sum(r["skipped"] for r in results)
        print(f"    → {saved} new + {skipped} skipped ({existing + saved} total)")


# ═══════════════════════════════════════════════════════════════════
# Step 3: 生成 Tile Metadata | Generate Tile Metadata
# ═══════════════════════════════════════════════════════════════════

def step3_metadata(dst_root: Path, splits: list, max_images: int, dry_run: bool):
    """Step 3: 生成 tile metadata JSON."""
    print("\n" + "=" * 60)
    print("  Step 3: Generate Tile Metadata")
    print("=" * 60)

    from PIL import Image

    meta_dir = dst_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        mask_dir = dst_root / "masks" / split
        if not mask_dir.exists():
            print(f"  [{split.upper()}] No masks dir, skipping")
            continue

        tiles = sorted(mask_dir.glob("*.png"))
        if max_images > 0:
            # 从 tile 名推断 image_id 并限制 | Limit by unique image_ids
            seen = set()
            limited = []
            for t in tiles:
                # tile name: P0000_t003.png → img_id = P0000
                img_id = t.stem.rsplit("_t", 1)[0]
                seen.add(img_id)
                if len(seen) <= max_images:
                    limited.append(t)
                elif img_id in seen:
                    limited.append(t)
            tiles = limited

        if dry_run:
            print(f"  [{split.upper()}] {len(tiles)} tiles (dry-run)")
            continue

        metadata = []
        for tile_path in tqdm(tiles, desc=f"  [{split}] Metadata", unit="tile"):
            mask = np.array(Image.open(tile_path))  # [1024, 1024] uint8

            # 统计 | Statistics
            total_px = mask.size
            fg_px = int((mask > 0).sum())
            fg_ratio = fg_px / total_px

            # 类别分布 | Class distribution
            class_counts = {}
            for c in range(1, 16):
                cnt = int((mask == c).sum())
                if cnt > 0:
                    class_counts[int(c)] = cnt

            # 解析文件名 | Parse filename
            stem = tile_path.stem  # "P0000_t003"
            parts = stem.rsplit("_t", 1)
            img_id = parts[0]
            tile_idx = int(parts[1])

            metadata.append({
                "tile_name": tile_path.name,
                "img_id": img_id,
                "tile_idx": tile_idx,
                "fg_ratio": round(fg_ratio, 4),
                "fg_pixels": fg_px,
                "total_pixels": total_px,
                "class_distribution": class_counts,
            })

        meta_path = meta_dir / f"{split}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        n_fg = sum(1 for m in metadata if m["fg_ratio"] > 0.05)
        print(f"    → {len(metadata)} tiles, {n_fg} with fg_ratio>5% → {meta_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    src = Path(args.src_root)
    dst = Path(args.dst_root)
    splits = [s.strip() for s in args.splits.split(",")]
    steps = [int(s.strip()) for s in args.steps.split(",")]

    # 确保 cv2 可用 | Ensure cv2 is available
    try:
        import cv2
    except ImportError:
        print("ERROR: cv2 not installed. Run: pip install opencv-python")
        sys.exit(1)

    print("=" * 70)
    print("  iSAID Offline Tile Preprocessing + Metadata")
    print(f"  Source:  {src}")
    print(f"  Output:  {dst}")
    print(f"  Steps:   {steps}")
    print(f"  Workers: {args.workers}")
    print(f"  Format:  {args.format}")
    print("=" * 70)

    if 1 in steps:
        step1_render_masks(src, dst, splits, args.max_images,
                          args.workers, args.dry_run)

    if 2 in steps:
        step2_cut_tiles(src, dst, splits, args.max_images,
                       args.workers, args.tile_size, args.format,
                       args.dry_run)

    if 3 in steps:
        step3_metadata(dst, splits, args.max_images, args.dry_run)

    if not args.dry_run:
        print(f"\n  ✅ All steps done! Output: {dst}/")


if __name__ == "__main__":
    main()
