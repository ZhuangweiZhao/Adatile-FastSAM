#!/usr/bin/env python3
"""
Cityscapes → Tile 格式, 自动解压 + 进度条 + 多进程
=====================================================

Cityscapes → FastISAIDTileDataset 兼容格式, 自动处理 zip 解压。

Cityscapes 类别 | Classes: 19 train + void=255
Cityscapes 图像 | Images: 2048×1024, ~3000 train / 500 val
每张图切 2 个 1024×1024 tile (2048/1024=2)

输出: {dst}/images/{split}/ + {dst}/masks/{split}/

用法 | Usage::
    python tools/prep_cityscapes.py
    python tools/prep_cityscapes.py --src /root/autodl-pub/cityscapes --dst /root/autodl-tmp/cityscapes_tiles
"""

import sys, argparse, zipfile, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


# labelId → trainId 查找表 | labelId → trainId lookup table
# https://github.com/mcordts/cityscapesScripts
_LABEL_TO_TRAIN = np.array([
    255,255,255,255,255,255,255,   0,1,255,255,   2,3,4,
    255,255,255,   5,255,   6,7,   8,9,10,
    11,12,13,14,15,255,255,  16,17,18
], dtype=np.uint8)


def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Cityscapes → Tile Format | Cityscapes 转瓦片格式")
    p.add_argument("--src", type=str, default="/root/autodl-pub/cityscapes",
                   help="Cityscapes 原始数据目录 | Cityscapes raw data directory")
    p.add_argument("--dst", type=str, default="/root/autodl-tmp/cityscapes_tiles",
                   help="瓦片输出目录 | Tile output directory")
    p.add_argument("--tile-size", type=int, default=1024,
                   help="瓦片尺寸 (像素) | Tile size in pixels")
    p.add_argument("--workers", type=int, default=8,
                   help="并行进程数 | Number of parallel workers")
    return p.parse_args()


def extract_zips(src: Path, tmp_dir: Path):
    """解压 Cityscapes zip 到临时目录 | Extract Cityscapes zips to tmp directory (src may be read-only).

    解压 leftImg8bit (图像) 和 gtFine (标签) 两个 zip 包。
    增量：已解压则跳过，不重复解压。
    Extracts leftImg8bit (images) and gtFine (labels) zip files.
    Incremental: skips if already extracted to avoid redundant work.
    """
    import shutil
    for fname in ["leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"]:
        zp = src / fname
        if not zp.exists():
            continue
        # 检查是否已解压到 tmp | Check if already extracted to tmp
        if fname.startswith("leftImg8bit"):
            if (tmp_dir / "leftImg8bit" / "train").exists():
                print(f"  {fname} already extracted, skipping | 已解压，跳过")
                continue
        else:
            if (tmp_dir / "gtFine" / "train").exists():
                print(f"  {fname} already extracted, skipping | 已解压，跳过")
                continue

        with zipfile.ZipFile(zp) as zf:
            total = len(zf.namelist())
            print(f"  Extracting {fname} ({total} files)... | 解压中...")
            for m in tqdm(zf.namelist(), desc=f"  {fname.split('_')[0]}", unit="f"):
                zf.extract(m, tmp_dir)


def _process_single(args_tuple: tuple) -> dict:
    """单图 worker: 转换 labelId→trainId → 切 tile → 保存 (独立进程) | Single-image worker: convert labelId→trainId → cut tiles → save (separate process)."""
    src_img, src_label, img_out_dir, mask_out_dir, prefix, ts = args_tuple

    import cv2
    from PIL import Image

    # 跳过已处理 (增量) | Skip if already done (incremental)
    first_tile = f"{prefix}_t00_00.png"
    if os.path.exists(os.path.join(img_out_dir, first_tile)):
        return {"status": "skip", "saved": 0}

    # 读图 (uint8 全程，不转 float32) | Read image (uint8 throughout, never float32)
    img = cv2.imread(src_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读标签 + 映射 labelId→trainId (34→19 类 + 255=ignore) | Read label + map labelId→trainId (34→19 classes + 255=ignore)
    label = cv2.imread(src_label, cv2.IMREAD_UNCHANGED)
    train = _LABEL_TO_TRAIN[label.clip(0, 33)]  # [H, W], trainId

    H, W = img.shape[:2]
    saved = 0

    for y in range(0, H, ts):
        for x in range(0, W, ts):
            tile_name = f"{prefix}_t{y//ts:02d}_{x//ts:02d}.png"
            tile_img_path = os.path.join(img_out_dir, tile_name)
            tile_mask_path = os.path.join(mask_out_dir, tile_name)

            if os.path.exists(tile_img_path):
                continue

            th, tw = min(ts, H-y), min(ts, W-x)
            tile_img = img[y:y+th, x:x+tw]
            tile_mask = train[y:y+th, x:x+tw]

            # 边界补零 | Pad edges to ts×ts
            if th < ts or tw < ts:
                pi = np.zeros((ts, ts, 3), dtype=np.uint8)
                pi[:th, :tw] = tile_img; tile_img = pi
                pm = np.full((ts, ts), 255, dtype=np.uint8)  # 255=ignore
                pm[:th, :tw] = tile_mask; tile_mask = pm

            # cv2 保存 (比 PIL 快 1.5~3×) | Save via cv2 (1.5~3× faster than PIL)
            cv2.imwrite(tile_img_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(tile_mask_path, tile_mask)
            saved += 1

    return {"status": "done", "saved": saved}


# ═══════════════════════════════════════════════════════════════════
# Main | 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    """主入口：解压 → 构建任务 → 多进程切 tile | Main entry: extract → build tasks → multiprocess tile cutting."""
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    ts = args.tile_size

    print("=" * 60)
    print("  Cityscapes → Tile Format | Cityscapes 转瓦片格式")
    print(f"  Source: {src}  →  Output: {dst}")
    print("=" * 60)

    # ── Step 0: 自动解压到临时目录 | Auto-extract zips to tmp ──
    # Cityscapes 原始数据通常是只读的 zip 包，需先解压
    # Cityscapes raw data is typically read-only zips — extract first
    tmp_src = dst / "_extracted"
    tmp_src.mkdir(parents=True, exist_ok=True)
    extract_zips(src, tmp_src)

    # ── Step 1: 构建任务列表 (train + val) | Build task list (train + val) ──
    for split in ["train", "val"]:
        img_root = tmp_src / "leftImg8bit" / split
        label_root = tmp_src / "gtFine" / split
        img_out = dst / "images" / split
        mask_out = dst / "masks" / split
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        # 收集图像-标签对 (按城市子目录遍历) | Collect image-label pairs (iterate over city subdirectories)
        tasks = []
        for city_dir in sorted(img_root.iterdir()) if img_root.exists() else []:
            if not city_dir.is_dir(): continue
            city = city_dir.name
            for img_path in sorted(city_dir.glob("*.png")):
                # 匹配对应的 gtFine 标签文件 | Match corresponding gtFine label file
                name = img_path.stem.replace("_leftImg8bit", "")
                label_path = label_root / city / f"{name}_gtFine_labelIds.png"
                if label_path.exists():
                    prefix = f"{city}_{name[:20]}"  # 用于 tile 命名 | for tile naming
                    tasks.append((str(img_path), str(label_path),
                                 str(img_out), str(mask_out), prefix, ts))

        if not tasks:
            print(f"\n  [{split.upper()}] No images found (did you unzip?) | 未找到图像 (是否已解压?)")
            continue

        # Cityscapes 图像 2048×1024, tile=1024 → 每张图 2 个 tile
        # Cityscapes images 2048×1024, tile=1024 → 2 tiles per image
        existing = len(list(img_out.glob("*.png")))
        print(f"\n  [{split.upper()}] {len(tasks)} images → "
              f"~{len(tasks)*2} tiles ({existing} tiles exist)")

        # ── Step 2: 多进程处理 | Multiprocess tile conversion ──
        if args.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(tqdm(ex.map(_process_single, tasks),
                                    total=len(tasks),
                                    desc=f"  [{split}] Converting",
                                    unit="img"))
        else:
            results = [_process_single(t) for t in
                       tqdm(tasks, desc=f"  [{split}] Converting", unit="img")]

        total_saved = sum(r.get("saved", 0) for r in results)
        total_skip = sum(1 for r in results if r["status"] == "skip")
        print(f"    → {total_saved} new tiles, {total_skip} images skipped | {total_saved} 新瓦片, {total_skip} 图像已跳过")

    # ── 完成 | Done ──
    print(f"\n  ✅ Done! Output: {dst}/ | 完成！输出: {dst}/")
    print(f"  Use: FastISAIDTileDataset(root_dir='{dst}', split='train', dense_labels=True)")


if __name__ == "__main__":
    main()
