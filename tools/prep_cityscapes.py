#!/usr/bin/env python3
"""
Cityscapes → iSAID Tiles 格式转换 | Cityscapes → iSAID Tile Format

将 Cityscapes gtFine 彩色标签转换为 class-index PNG, 切 tile, 输出 FastISAIDTileDataset 兼容格式。

Cityscapes 类别映射 | Class mapping (19 train + void=255):
    void→255, road→0, sidewalk→1, building→2, wall→3, fence→4,
    pole→5, traffic_light→6, traffic_sign→7, vegetation→8, terrain→9,
    sky→10, person→11, rider→12, car→13, truck→14, bus→15,
    train→16, motorcycle→17, bicycle→18

输入 | Input: /root/autodl-pub/cityscapes/
    leftImg8bit/{train,val}/*/*.png
    gtFine/{train,val}/*/*_gtFine_labelIds.png

输出 | Output: {dst}/images/{split}/  +  {dst}/masks/{split}/
    tile 格式, 1024×1024 PNG

用法 | Usage:
    python tools/prep_cityscapes.py --src /root/autodl-pub/cityscapes --dst /root/autodl-tmp/cityscapes_tiles
"""

import sys, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


# Cityscapes labelIds → trainIds (19 类 + ignore=255)
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
_LABEL_TO_TRAIN = np.array([
    255, 255, 255, 255, 255, 255, 255,    # 0-6
    0, 1, 255, 255, 2, 3, 4,              # 7-13 (road, sidewalk, building, wall, fence)
    255, 255, 255, 5, 255, 6, 7,          # 14-20 (pole, tlight, tsign)
    8, 9, 10, 11, 12, 13, 14, 15,        # 21-28 (vegetation, terrain, sky, person, rider, car, truck, bus)
    255, 255, 16, 17, 18,                 # 29-33 (train, motorcycle, bicycle)
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default="/root/autodl-pub/cityscapes")
    p.add_argument("--dst", type=str, default="/root/autodl-tmp/cityscapes_tiles")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def convert_and_save(src_img: Path, src_label: Path, img_out_dir: Path,
                     mask_out_dir: Path, prefix: str, ts: int) -> int:
    """转换单张图 → tile + 保存 | Convert one image → tiles + save."""
    img = np.array(Image.open(src_img).convert("RGB"))  # [H, W, 3]
    label = np.array(Image.open(src_label))              # [H, W], labelIds

    # 映射 labelId → trainId | Map labelId → trainId
    label_flat = label.flatten()
    train_flat = _LABEL_TO_TRAIN[label_flat.clip(0, 33)]
    train = train_flat.reshape(label.shape)  # [H, W], trainId

    H, W = img.shape[:2]
    saved = 0

    for y in range(0, H, ts):
        for x in range(0, W, ts):
            tile_name = f"{prefix}_t{y//ts:02d}_{x//ts:02d}.png"
            if (img_out_dir / tile_name).exists():
                continue

            th, tw = min(ts, H-y), min(ts, W-x)
            tile_img = img[y:y+th, x:x+tw]
            tile_mask = train[y:y+th, x:x+tw]

            # Pad edges
            if th < ts or tw < ts:
                pi = np.zeros((ts, ts, 3), dtype=np.uint8)
                pi[:th, :tw] = tile_img; tile_img = pi
                pm = np.full((ts, ts), 255, dtype=np.uint8)
                pm[:th, :tw] = tile_mask; tile_mask = pm

            Image.fromarray(tile_img).save(img_out_dir / tile_name)
            Image.fromarray(tile_mask).save(mask_out_dir / tile_name)
            saved += 1
    return saved


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    ts = args.tile_size

    print("=" * 60)
    print("  Cityscapes → Tile Format")
    print(f"  Source: {src}  →  Output: {dst}")
    print("=" * 60)

    for split in ["train", "val"]:
        img_dir = src / "leftImg8bit" / split
        label_root = src / "gtFine" / split
        img_out = dst / "images" / split
        mask_out = dst / "masks" / split
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        # 找所有图像-标签对 | Find all image-label pairs
        pairs = []
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir(): continue
            city = city_dir.name
            for img_path in sorted(city_dir.glob("*.png")):
                # leftImg8bit: {city}_000001_000019_leftImg8bit.png
                # gtFine:      {city}_000001_000019_gtFine_labelIds.png
                name = img_path.stem.replace("_leftImg8bit", "")
                label_path = label_root / city / f"{name}_gtFine_labelIds.png"
                if label_path.exists():
                    pairs.append((img_path, label_path, f"{city}_{name[:20]}"))

        existing = len(list(img_out.glob("*.png")))
        print(f"\n  [{split.upper()}] {len(pairs)} images ({existing} tiles exist)")

        total = 0
        for img_p, lbl_p, prefix in tqdm(pairs, desc=f"  [{split}] Converting"):
            total += convert_and_save(img_p, lbl_p, img_out, mask_out, prefix, ts)

        print(f"    → {total} tiles")

    print(f"\n  ✅ Done! Output: {dst}/")
    print(f"  Use: FastISAIDTileDataset(root_dir='{dst}', split='train', semantic=True)")


if __name__ == "__main__":
    main()
