#!/usr/bin/env python
"""Convert iSAID COCO → binary mask PNGs (optimized).

Optimizations:
  1. cv2.fillPoly (C++) instead of PIL ImageDraw (Python loops)
  2. pycocotools C-optimized RLE decode
  3. Multiprocess per-image parallelization
  4. Skip already-converted images (resumable)
  5. Lazy image dimension detection

Usage:
    python tools/convert_isaid_binary.py --class "Small_Vehicle" --output datasets/isaid_binary --workers 8
"""

import argparse, os, json, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pycocotools import mask as cocomask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--isaid", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/iSAID")
    p.add_argument("--output", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_binary")
    p.add_argument("--class", dest="class_name", type=str, default="Small_Vehicle")
    p.add_argument("--min-instances", type=int, default=1)
    p.add_argument("--workers", type=int, default=min(8, cpu_count()))
    return p.parse_args()


def poly_to_mask_cv2(polygons, h, w):
    """cv2.fillPoly — much faster than PIL ImageDraw."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        # clip to image bounds
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def rle_to_mask_cv2(rle, h, w):
    """Decode COCO RLE via pycocotools."""
    if isinstance(rle, dict) and 'counts' in rle:
        m = cocomask.decode(rle)
        return (m * 255).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def anns_to_mask(anns, h, w):
    """All annotations → one binary mask. Uses cv2 if available, pycocotools for RLE."""
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        seg = ann.get('segmentation')
        if seg is None:
            continue

        if isinstance(seg, list):
            # Polygon
            if len(seg) == 0:
                continue
            if cv2 is not None:
                m = poly_to_mask_cv2(seg, h, w)
            else:
                # PIL fallback
                from PIL import Image, ImageDraw
                im = Image.new('L', (w, h), 0)
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    ImageDraw.Draw(im).polygon(pts, outline=255, fill=255)
                m = np.array(im)
        elif HAS_COCO:
            m = rle_to_mask_cv2(seg, h, w)
        else:
            continue

        if m is not None:
            np.maximum(mask, m, out=mask)

    return mask


def process_image(args_tuple):
    """Process ONE image — called by multiprocessing pool."""
    img_id, anns, info, src_path, out_img_path, out_mask_path = args_tuple

    # Skip if already done
    if out_img_path.exists() and out_mask_path.exists():
        if out_mask_path.stat().st_size > 0:
            return 0, info.get('file_name', str(img_id)), "skipped"

    try:
        # Get image dimensions
        if src_path.exists():
            if cv2 is not None:
                im = cv2.imread(str(src_path))
                if im is not None:
                    h, w = im.shape[:2]
                else:
                    # Read with PIL to get size, then convert
                    from PIL import Image
                    im = Image.open(src_path)
                    w, h = im.size
                    im = np.array(im.convert("RGB"))
            else:
                from PIL import Image
                im = Image.open(src_path)
                w, h = im.size
                im = np.array(im.convert("RGB"))
        else:
            return -1, info.get('file_name', str(img_id)), "src not found"

        # Generate mask
        mask = anns_to_mask(anns, h, w)

        # Save
        if cv2 is not None:
            # cv2 saves RGB as BGR unless we convert
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), im_bgr)
            cv2.imwrite(str(out_mask_path), mask)
        else:
            from PIL import Image
            Image.fromarray(im).save(out_img_path)
            Image.fromarray(mask).save(out_mask_path)

        pct = (mask > 0).sum() / mask.size * 100
        return 1, info.get('file_name', str(img_id)), f"{len(anns)} inst, {pct:.1f}% fg"
    except Exception as e:
        return -1, info.get('file_name', str(img_id)), str(e)


def main():
    args = parse_args()
    print(f"Workers: {args.workers}, Backend: {'cv2' if cv2 else 'PIL'}, "
          f"COCO: {'pycocotools' if HAS_COCO else 'manual'}")

    class_id = None

    for split in ["train", "val"]:
        anno_path = Path(args.isaid) / "annotations" / split / f"instances_{split}.json"
        if not anno_path.exists():
            continue

        print(f"\n[{split}] Loading annotations...")
        with open(anno_path) as f:
            data = json.load(f)

        # Find class ID
        if class_id is None:
            for cat in data["categories"]:
                if cat["name"] == args.class_name:
                    class_id = cat["id"]
                    break
            if class_id is None:
                print(f"Class '{args.class_name}' not found in {list(c['name'] for c in data['categories'])}")
                sys.exit(1)
            print(f"Class: {args.class_name} (ID={class_id})")

        # Group annotations by image_id for this class
        class_anns = {}
        for ann in data["annotations"]:
            if ann["category_id"] == class_id:
                iid = ann["image_id"]
                class_anns.setdefault(iid, []).append(ann)

        # Build image info, detect dimensions lazily
        img_info = {}
        for img in data["images"]:
            img_info[img["id"]] = {
                "file_name": img["file_name"],
                "path": Path(args.isaid) / "images" / split / img["file_name"],
            }

        # Output dirs
        split_map = {"train": "TrainDataset", "val": "ValDataset"}
        out_split = split_map.get(split, split)
        out_img_dir = Path(args.output) / out_split / "images"
        out_mask_dir = Path(args.output) / out_split / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # Build task list (skip if < min_instances)
        tasks = []
        for img_id, anns in class_anns.items():
            if len(anns) < args.min_instances:
                continue
            info = img_info.get(img_id)
            if info is None:
                continue
            tasks.append((
                img_id, anns, info,
                info["path"],
                out_img_dir / info["file_name"],
                out_mask_dir / info["file_name"],
            ))

        print(f"  {len(tasks)} images to process")

        # Parallel process with progress bar
        converted, skipped, errors = 0, 0, 0
        pool = Pool(args.workers)
        results = pool.imap_unordered(process_image, tasks, chunksize=max(1, len(tasks) // args.workers // 4))

        pbar = tqdm(total=len(tasks), desc=f"[{split}]", unit="img", dynamic_ncols=True) if tqdm else None

        for status, fname, msg in results:
            if status == 1:
                converted += 1
            elif status == 0:
                skipped += 1
            else:
                errors += 1
            if pbar:
                pbar.set_postfix({"new": converted, "skip": skipped, "err": errors}, refresh=False)
                pbar.update(1)
            elif (converted + skipped + errors) % 200 == 0:
                print(f"  ... {converted + skipped + errors}/{len(tasks)} ({converted} new, {skipped} skipped)")

        pool.close(); pool.join()
        if pbar:
            pbar.close()

        print(f"  [{split}] Done: {converted} new, {skipped} skipped, {errors} errors ({len(tasks)} total)")


if __name__ == "__main__":
    main()
