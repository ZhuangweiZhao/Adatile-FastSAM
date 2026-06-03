#!/usr/bin/env python
"""Set up dataset directory structure expected by AdaTile-FastSAM configs.

Expected structure (by CocoDataset / ISAIDDataset):
    datasets/iSAID/
        images/{train,val,test}/   ← image files
        annotations/{train,val}/   ← COCO JSON annotations
    datasets/COCO/
        images/{train2017,val2017}/
        annotations/               ← instances_train2017.json, etc.
    datasets/LoveDA/
        images/{train,val,test}/
        annotations/{train,val}/

Actual data lives in: 数据集/{iSAID,LoveDA,COCO}/

Usage:
    python tools/setup_datasets.py
    python tools/setup_datasets.py --check   # verify structure only
"""

import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SRC = PROJECT_ROOT / "数据集"
DATA_DST = PROJECT_ROOT / "datasets"


def create_symlink_or_copy(src: Path, dst: Path):
    """Create symlink if possible, otherwise copy (Windows fallback)."""
    if dst.exists():
        return  # already exists

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try symlink first (requires Developer Mode on Windows)
        if sys.platform == "win32":
            os.symlink(str(src), str(dst), target_is_directory=True)
        else:
            os.symlink(str(src), str(dst))
        print(f"  symlink: {dst.name} -> {src}")
    except OSError:
        # Fallback: create directory junction on Windows, or copy
        if sys.platform == "win32":
            try:
                import subprocess
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    check=True, capture_output=True,
                )
                print(f"  junction: {dst.name} -> {src}")
            except Exception:
                print(f"  WARNING: Cannot create link for {dst.name}. Copying...")
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                print(f"  copied: {dst.name}")
        else:
            print(f"  WARNING: Cannot symlink. Copying {dst.name}...")
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            print(f"  copied: {dst.name}")


def setup_isaid():
    """Set up iSAID dataset structure.

    Source: 数据集/iSAID/
      TrainData/train/Annotations/iSAID_train.json
      TrainData/train/Instance_masks/images/images/*.png
      ValidationData/val/Annotations/iSAID_val.json
      ValidationData/val/Instance_masks/images/images/*.png
      TestData/TestData/Part1-*/images/*.png

    Target: datasets/iSAID/
      annotations/train/instances_train.json
      annotations/val/instances_val.json
      images/train/*.png
      images/val/*.png
      images/test/*.png
    """
    src = DATA_SRC / "iSAID"
    dst = DATA_DST / "iSAID"
    print("\n── Setting up iSAID ──")

    # Annotations
    ann_train_src = src / "TrainData" / "train" / "Annotations" / "iSAID_train.json"
    ann_val_src = src / "ValidationData" / "val" / "Annotations" / "iSAID_val.json"
    ann_train_dst = dst / "annotations" / "train" / "instances_train.json"
    ann_val_dst = dst / "annotations" / "val" / "instances_val.json"

    ann_train_dst.parent.mkdir(parents=True, exist_ok=True)
    ann_val_dst.parent.mkdir(parents=True, exist_ok=True)

    if ann_train_src.exists():
        shutil.copy2(str(ann_train_src), str(ann_train_dst))
        print(f"  annotations/train/instances_train.json [OK]")
    else:
        print(f"  WARNING: {ann_train_src} not found")

    if ann_val_src.exists():
        shutil.copy2(str(ann_val_src), str(ann_val_dst))
        print(f"  annotations/val/instances_val.json [OK]")

    # Images — use symlinks to avoid duplication
    # Train images: Instance_masks/images/images/
    train_img_src = src / "TrainData" / "train" / "Instance_masks" / "images" / "images"
    train_img_dst = dst / "images" / "train"

    if train_img_src.exists():
        # Create symlinks for each image
        train_img_dst.mkdir(parents=True, exist_ok=True)
        pngs = list(train_img_src.glob("*.png"))
        linked = 0
        for png in pngs:
            link_dst = train_img_dst / png.name
            if not link_dst.exists():
                try:
                    os.symlink(str(png), str(link_dst))
                    linked += 1
                except OSError:
                    shutil.copy2(str(png), str(link_dst))
                    linked += 1
        print(f"  images/train/: {len(pngs)} images ({linked} linked/copied) [OK]")
    else:
        print(f"  WARNING: train images not found at {train_img_src}")

    # Val images: Instance_masks/images/images/
    val_img_src = src / "ValidationData" / "val" / "Instance_masks" / "images" / "images"
    val_img_dst = dst / "images" / "val"

    if val_img_src.exists():
        val_img_dst.mkdir(parents=True, exist_ok=True)
        pngs = list(val_img_src.glob("*.png"))
        linked = 0
        for png in pngs:
            link_dst = val_img_dst / png.name
            if not link_dst.exists():
                try:
                    os.symlink(str(png), str(link_dst))
                    linked += 1
                except OSError:
                    shutil.copy2(str(png), str(link_dst))
                    linked += 1
        print(f"  images/val/: {len(pngs)} images ({linked} linked/copied) [OK]")
    else:
        print(f"  WARNING: val images not found at {val_img_src}")

    # Test images: TestData/TestData/Part1-*/images/
    test_img_dst = dst / "images" / "test"
    test_img_dst.mkdir(parents=True, exist_ok=True)
    test_total = 0
    for part_dir in (src / "TestData" / "TestData").glob("Part1-*"):
        for png in (part_dir / "images").glob("*.png"):
            link_dst = test_img_dst / png.name
            if not link_dst.exists():
                try:
                    os.symlink(str(png), str(link_dst))
                except OSError:
                    pass  # skip if can't symlink
            test_total += 1
    print(f"  images/test/: {test_total} images [OK]")


def setup_loveda():
    """Set up LoveDA dataset structure.

    Source: 数据集/LoveDA/
      Train/Train/{Rural,Urban}/images_png/*.png
      Train/Train/{Rural,Urban}/masks_png/*.png
      Val/Val/{Rural,Urban}/images_png/*.png
      Test/Test/{Rural,Urban}/images_png/*.png

    Target: datasets/LoveDA/
      images/{train,val,test}/*.png
      masks/{train,val}/*.png
    """
    src = DATA_SRC / "LoveDA"
    dst = DATA_DST / "LoveDA"
    print("\n── Setting up LoveDA ──")

    splits = {
        "train": src / "Train" / "Train",
        "val": src / "Val" / "Val",
        "test": src / "Test" / "Test",
    }

    total = 0
    for split_name, split_dir in splits.items():
        img_dst = dst / "images" / split_name
        mask_dst = dst / "masks" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)

        for domain in ["Rural", "Urban"]:
            domain_dir = split_dir / domain
            if not domain_dir.exists():
                continue

            # Link images
            img_src_dir = domain_dir / "images_png"
            if img_src_dir.exists():
                for png in img_src_dir.glob("*.png"):
                    link_dst = img_dst / f"{domain}_{png.name}"
                    if not link_dst.exists():
                        try:
                            os.symlink(str(png), str(link_dst))
                            total += 1
                        except OSError:
                            pass

            # Link masks
            mask_src_dir = domain_dir / "masks_png"
            if mask_src_dir.exists():
                mask_dst.mkdir(parents=True, exist_ok=True)
                for png in mask_src_dir.glob("*.png"):
                    link_dst = mask_dst / f"{domain}_{png.name}"
                    if not link_dst.exists():
                        try:
                            os.symlink(str(png), str(link_dst))
                        except OSError:
                            pass

    print(f"  images/: {total} images linked [OK]")
    print(f"  Note: LoveDA needs COCO-format annotations. Use tools/generate_coco_annotations.py")


def setup_coco():
    """Set up COCO dataset structure.

    Source: 数据集/COCO/
      train2017/train2017/*.jpg (double-nested from zip)
      val2017/val2017/*.jpg
      annotations_trainval2017.zip

    Target: datasets/COCO/
      images/train2017/*.jpg
      images/val2017/*.jpg
      annotations/instances_train2017.json
      annotations/instances_val2017.json
    """
    src = DATA_SRC / "COCO"
    dst = DATA_DST / "COCO"
    print("\n── Setting up COCO ──")

    # Train images (fix double-nesting)
    train_src = src / "train2017" / "train2017"
    train_dst = dst / "images" / "train2017"
    if train_src.exists():
        train_dst.mkdir(parents=True, exist_ok=True)
        jpgs = list(train_src.glob("*.jpg"))
        linked = 0
        for jpg in jpgs:
            link_dst = train_dst / jpg.name
            if not link_dst.exists():
                try:
                    os.symlink(str(jpg), str(link_dst))
                    linked += 1
                except OSError:
                    shutil.copy2(str(jpg), str(link_dst))
                    linked += 1
        print(f"  images/train2017/: {len(jpgs)} images ({linked} linked/copied) [OK]")

    # Val images
    val_src = src / "val2017" / "val2017"
    val_dst = dst / "images" / "val2017"
    if val_src.exists():
        val_dst.mkdir(parents=True, exist_ok=True)
        jpgs = list(val_src.glob("*.jpg"))
        linked = 0
        for jpg in jpgs:
            link_dst = val_dst / jpg.name
            if not link_dst.exists():
                try:
                    os.symlink(str(jpg), str(link_dst))
                    linked += 1
                except OSError:
                    shutil.copy2(str(jpg), str(link_dst))
                    linked += 1
        print(f"  images/val2017/: {len(jpgs)} images ({linked} linked/copied) [OK]")

    # Annotations — need to extract from zip
    ann_zip = src / "annotations_trainval2017.zip"
    if ann_zip.exists():
        import zipfile
        ann_dst = dst / "annotations"
        ann_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(ann_zip)) as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    zf.extract(name, str(ann_dst))
        print(f"  annotations/ extracted [OK]")
    else:
        print(f"  WARNING: {ann_zip} not found")


def verify_structure():
    """Verify the dataset structure is correctly set up."""
    print("\n── Verifying Dataset Structure ──\n")

    checks = [
        ("iSAID train annotations", DATA_DST / "iSAID" / "annotations" / "train" / "instances_train.json"),
        ("iSAID val annotations", DATA_DST / "iSAID" / "annotations" / "val" / "instances_val.json"),
        ("iSAID train images", DATA_DST / "iSAID" / "images" / "train"),
        ("iSAID val images", DATA_DST / "iSAID" / "images" / "val"),
        ("COCO train images", DATA_DST / "COCO" / "images" / "train2017"),
        ("COCO val images", DATA_DST / "COCO" / "images" / "val2017"),
        ("LoveDA train images", DATA_DST / "LoveDA" / "images" / "train"),
    ]

    all_ok = True
    for name, path in checks:
        exists = path.exists()
        if exists and path.is_dir():
            count = len(list(path.glob("*")))
            status = f"[OK] {count} items"
        elif exists:
            status = "[OK]"
        else:
            status = "[MISSING]"
            all_ok = False
        print(f"  [{status}] {name}")

    if all_ok:
        print("\n[OK] All datasets ready!")
    else:
        print("\n⚠️  Some datasets are incomplete. Run: python tools/setup_datasets.py")
    return all_ok


def clean_empty_dirs():
    """Remove empty leftover directories from old structure."""
    for subdir in ["builders", "cache", "common", "loaders", "stats"]:
        d = DATA_DST / subdir
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            print(f"  cleaned: datasets/{subdir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up dataset directory structure")
    parser.add_argument("--check", action="store_true", help="Only verify structure")
    parser.add_argument("--dataset", type=str, choices=["isaid", "coco", "loveda", "all"],
                        default="all", help="Which dataset to set up")
    args = parser.parse_args()

    clean_empty_dirs()

    if args.check:
        verify_structure()
        return

    DATA_DST.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("isaid", "all"):
        setup_isaid()
    if args.dataset in ("coco", "all"):
        setup_coco()
    if args.dataset in ("loveda", "all"):
        setup_loveda()

    print()
    verify_structure()


if __name__ == "__main__":
    main()
