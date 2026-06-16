#!/usr/bin/env python
"""iSAID + DOTA dataset builder.

Step 1: Extract DOTA-v1.0 images from zips → flat directory (one-time)
Step 2: Rasterize COCO polygons → class-label masks (0=bg, 1-15=class)

Output: 数据集/isaid_dota/
  DOTA_images/           ← all extracted RGB images (flat)
  train/images/  masks/  ← class-label PNGs
  val/images/    masks/

Usage:
  python tools/build_isaid_dota.py                         # full build (15-class)
  python tools/build_isaid_dota.py --mode binary           # binary masks
  python tools/build_isaid_dota.py --extract-only          # just unzip
  python tools/build_isaid_dota.py --max-images 10         # test
"""

import argparse, os, sys, shutil, json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "数据集"


def parse_args():
    p = argparse.ArgumentParser(description="Build iSAID+DOTA dataset")
    p.add_argument("--dota-zip-dirs", type=str, nargs="+",
                   default=[
                       str(DATA_ROOT / "DOTA" / "train" / "images"),
                       str(DATA_ROOT / "DOTA" / "val" / "images"),
                       str(DATA_ROOT / "DOTA" / "test" / "images"),
                   ],
                   help="DOTA image zip directories")
    p.add_argument("--isaid-dir", type=str, default=str(DATA_ROOT / "iSAID"))
    p.add_argument("--output-dir", type=str, default=str(DATA_ROOT / "isaid_dota"))
    p.add_argument("--extract-only", action="store_true",
                   help="Only extract zips, skip mask pairing")
    p.add_argument("--mode", type=str, default="multiclass",
                   choices=["binary", "multiclass"],
                   help="binary=fg/bg, multiclass=15-class labels")
    p.add_argument("--force", action="store_true",
                   help="Regenerate all masks (ignore existing)")
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def extract_zips(zip_sources, out_dir):
    """Extract all PNGs from multiple zip source directories → out_dir (flat).

    zip_sources: list of directory paths containing *.zip files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for zip_dir in zip_sources:
        zip_dir = Path(zip_dir)
        if not zip_dir.exists():
            continue
        zips = sorted(f for f in os.listdir(str(zip_dir)) if f.endswith('.zip'))
        for zname in zips:
            zpath = zip_dir / zname
            with zipfile.ZipFile(str(zpath)) as zf:
                members = [n for n in zf.namelist() if n.endswith('.png')]
                for i, name in enumerate(members):
                    fname = os.path.basename(name)
                    dst = out_dir / fname
                    if not dst.exists():
                        with zf.open(name) as src:
                            with open(dst, 'wb') as f:
                                shutil.copyfileobj(src, f)
                    total += 1
                    if i == 0 or (i + 1) % 100 == 0 or (i + 1) == len(members):
                        print(f"  {zip_dir.name}/{zname}: [{i+1}/{len(members)}]")

    print(f"  Total extracted: {total} images → {out_dir}")
    return total


def main():
    args = parse_args()
    img_cache = Path(args.output_dir) / "DOTA_images"

    # ═══ Step 1: Extract zips ═════════════════════════════════
    print("=== Step 1: Extract DOTA zips ===")
    if not img_cache.exists() or len(list(img_cache.glob("*.png"))) < 1869:
        extract_zips(args.dota_zip_dirs, img_cache)
    else:
        n = len(list(img_cache.glob("*.png")))
        print(f"  Already extracted: {n} images (skip)")

    if args.extract_only:
        print("Done (extract only).")
        return

    # ═══ Step 2: Build masks from COCO JSON ═══════════════════
    # NOTE: iSAID train vs val use DIFFERENT category_id→name mappings.
    # We normalize: train IDs are canonical, val IDs are remapped.
    print(f"\n=== Step 2: Build masks (mode={args.mode}) ===")
    isaid_dir = Path(args.isaid_dir)

    splits = [
        ("train", isaid_dir / "TrainData" / "train",
         isaid_dir / "TrainData" / "train" / "Annotations" / "iSAID_train.json"),
        ("val",   isaid_dir / "ValidationData" / "val",
         isaid_dir / "ValidationData" / "val" / "Annotations" / "iSAID_val.json"),
    ]

    # Build canonical name→id from train (class order is fixed by train JSON)
    with open(str(splits[0][2])) as f:
        train_coco = json.load(f)
    name_to_id = {c["name"]: c["id"] for c in train_coco["categories"]}
    num_classes = len(name_to_id) + 1  # +1 for background
    print(f"  Canonical classes ({num_classes - 1}):")
    for c in train_coco["categories"]:
        print(f"    id={c['id']:2d} {c['name']}")
    print(f"  num_classes = {num_classes} (including bg=0)")

    total = 0

    for split_name, split_dir, json_path in splits:
        if not json_path.exists():
            print(f"  ⚠️ Skip {split_name}: no {json_path}")
            continue

        print(f"\n  Loading {split_name} annotations...")
        with open(str(json_path)) as f:
            coco = json.load(f)

        # local cat_id → canonical cat_id (via name matching)
        local_cats = {c["id"]: c["name"] for c in coco["categories"]}
        id_map = {lid: name_to_id[name] for lid, name in local_cats.items()
                  if name in name_to_id}
        unmapped = set(local_cats.keys()) - set(id_map.keys())
        if unmapped:
            print(f"  ⚠️ Unmapped: {unmapped}")
        print(f"  Remap: {len(id_map)}/{len(local_cats)} mapped")

        img_info = {img["id"]: img for img in coco["images"]}

        # image_id → [(polygon, canonical_cat_id)]
        anno_by_image = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            cat = id_map.get(ann["category_id"], 0)
            if cat == 0:
                continue
            if iid not in anno_by_image:
                anno_by_image[iid] = []
            anno_by_image[iid].append((ann["segmentation"], cat))

        scenes = sorted(img_info.keys())
        if args.max_images:
            scenes = scenes[:args.max_images]
        n = len(scenes)

        out_img = Path(args.output_dir) / split_name / "images"
        out_mask = Path(args.output_dir) / split_name / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        ok = miss = 0

        for i, image_id in enumerate(scenes):
            info = img_info[image_id]
            scene = info["file_name"].replace(".png", "")

            src_img = img_cache / f"{scene}.png"
            if not src_img.exists():
                miss += 1
                continue

            dst_img = out_img / f"{scene}.png"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            dst_mask = out_mask / f"{scene}.png"
            if dst_mask.exists() and not args.force:
                ok += 1
                continue

            W = info.get("width") or Image.open(str(dst_img)).size[0]
            H = info.get("height") or Image.open(str(dst_img)).size[1]
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)

            fill_val = 255 if args.mode == "binary" else None
            for segs, cat_id in anno_by_image.get(image_id, []):
                for poly in segs:
                    pts = [(poly[j], poly[j+1]) for j in range(0, len(poly), 2)]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=fill_val if fill_val else cat_id)

            mask.save(str(dst_mask))
            ok += 1

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"  {split_name}: [{i+1}/{n}] ok={ok} miss={miss}")

        total += ok
        print(f"  {split_name} done: {ok} pairs")

    print(f"\n  All masks: canonical IDs 1-{num_classes-1}, bg=0")

    # ═══ Summary ═══════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"Complete: {total} image-mask pairs")
    for s in ["train", "val"]:
        d = Path(args.output_dir) / s / "images"
        if d.exists():
            print(f"  {s}/images: {len(list(d.glob('*.png')))}")
            print(f"  {s}/masks:  {len(list((Path(args.output_dir)/s/'masks').glob('*.png')))}")


if __name__ == "__main__":
    main()
