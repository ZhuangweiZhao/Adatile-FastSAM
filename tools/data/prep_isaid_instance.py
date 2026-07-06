#!/usr/bin/env python3
"""
iSAID Instance Few-Shot Split 数据预处理 | Instance Few-Shot Split Preprocessing.
===================================================================================

从 iSAID COCO 全图生成 896×896 tile + Base/Novel 3-Fold 定义 + COCO JSON 标注。
Generates 896×896 tiles + Base/Novel 3-Fold definitions + COCO JSON annotations
from iSAID COCO full images.

三步流水线 | Three-step pipeline::

    Step 1: 生成 instances JSON (修正好的全图 COCO, 这一步已由 prep_isaid.py 完成)
            → data/iSAID_processed/{split}/annotations/instances_{split}.json
    Step 2: 切 tile + 生成 tile-level COCO JSON
            → data/iSAID_instance_fewshot/images/{split}/*.png
            → data/iSAID_instance_fewshot/annotations/instances_{split}.json
    Step 3: 生成 Fold 定义 + 统计
            → data/iSAID_instance_fewshot/folds/fold_{0,1,2}.json
            → data/iSAID_instance_fewshot/stats/class_distribution.json

输出结构 | Output structure::

    data/iSAID_instance_fewshot/
    ├── images/
    │   ├── train/           # 896×896 tile PNG images
    │   └── val/             # 896×896 tile PNG images
    ├── annotations/
    │   ├── instances_train.json   # COCO JSON (tile-level annotations)
    │   └── instances_val.json     # COCO JSON (tile-level annotations)
    ├── folds/
    │   ├── fold_0.json      # Base/Novel class IDs per fold
    │   ├── fold_1.json
    │   └── fold_2.json
    └── stats/
        └── class_distribution.json  # Per-class instance/tile/pixel stats

类别 ID 体系 | Category ID System::

    使用标准 ``adatile.utils.label_mapping.ISAID_CATEGORIES`` (1-15)。
    这是 prep_isaid.py fix_annotations() 输出的 ID 体系。
    Uses standard ISAID_CATEGORIES (1-15) from label_mapping.py.
    This is the ID system output by prep_isaid.py fix_annotations().

用法 | Usage::

    python tools/data/prep_isaid_instance.py                          # 全量
    python tools/data/prep_isaid_instance.py --max-images 20          # 快速测试
    python tools/data/prep_isaid_instance.py --steps 2,3              # 只切 tile + metadata
    python tools/data/prep_isaid_instance.py --tile-size 896 --stride 640           # 默认配置
    python tools/data/prep_isaid_instance.py --src-root data/iSAID-few --no-folds     # 处理采样子集，跳过 fold
"""

import sys, argparse, json, os, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict

# Fix Windows GBK encoding for special characters
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

# 标准 ISAID 15 类别 | Standard ISAID 15 categories (from label_mapping.py)
ISAID_CATEGORIES = {
    1: "small_vehicle",
    2: "large_vehicle",
    3: "plane",
    4: "storage_tank",
    5: "ship",
    6: "harbor",
    7: "ground_track_field",
    8: "soccer_ball_field",
    9: "tennis_court",
    10: "swimming_pool",
    11: "baseball_diamond",
    12: "basketball_court",
    13: "bridge",
    14: "helicopter",
    15: "roundabout",
}

# 3-Fold Base/Novel 划分 | 3-Fold Base/Novel split definitions
# 与 iSAID-5i 使用相同的类别分组, 但使用标准 ISAID_CATEGORIES ID
# Same class grouping as iSAID-5i, but using standard ISAID_CATEGORIES IDs
ISAID_INSTANCE_FOLDS = {
    0: {
        "base": [5, 4, 11, 9, 7, 13, 2, 14, 8, 3],
        # ship, storage_tank, baseball_diamond, tennis_court, ground_track_field,
        # bridge, large_vehicle, helicopter, soccer_ball_field, plane
        "novel": [1, 6, 10, 12, 15],
        # small_vehicle, harbor, swimming_pool, basketball_court, roundabout
    },
    1: {
        "base": [5, 4, 11, 12, 1, 14, 10, 15, 8, 6],
        # ship, storage_tank, baseball_diamond, basketball_court, small_vehicle,
        # helicopter, swimming_pool, roundabout, soccer_ball_field, harbor
        "novel": [9, 7, 13, 2, 3],
        # tennis_court, ground_track_field, bridge, large_vehicle, plane
    },
    2: {
        "base": [9, 12, 7, 13, 2, 1, 10, 15, 3, 6],
        # tennis_court, basketball_court, ground_track_field, bridge, large_vehicle,
        # small_vehicle, swimming_pool, roundabout, plane, harbor
        "novel": [5, 4, 11, 14, 8],
        # ship, storage_tank, baseball_diamond, helicopter, soccer_ball_field
    },
}

# 类别元数据 (用于 COCO JSON) | Category metadata (for COCO JSON)
ISAID_CATEGORIES_COCO = [
    {"id": 1, "name": "small_vehicle", "supercategory": "vehicle"},
    {"id": 2, "name": "large_vehicle", "supercategory": "vehicle"},
    {"id": 3, "name": "plane", "supercategory": "vehicle"},
    {"id": 4, "name": "storage_tank", "supercategory": "infrastructure"},
    {"id": 5, "name": "ship", "supercategory": "vehicle"},
    {"id": 6, "name": "harbor", "supercategory": "infrastructure"},
    {"id": 7, "name": "ground_track_field", "supercategory": "sports"},
    {"id": 8, "name": "soccer_ball_field", "supercategory": "sports"},
    {"id": 9, "name": "tennis_court", "supercategory": "sports"},
    {"id": 10, "name": "swimming_pool", "supercategory": "infrastructure"},
    {"id": 11, "name": "baseball_diamond", "supercategory": "sports"},
    {"id": 12, "name": "basketball_court", "supercategory": "sports"},
    {"id": 13, "name": "bridge", "supercategory": "infrastructure"},
    {"id": 14, "name": "helicopter", "supercategory": "vehicle"},
    {"id": 15, "name": "roundabout", "supercategory": "infrastructure"},
]

MIN_AREA = 16         # 最小实例面积 (像素²) | Minimum instance area (px²)
MIN_FG_RATIO = 0.01   # tile 最小 FG 占比 | Minimum FG ratio per tile
NUM_WORKERS = 8       # 并行进程数 | Number of parallel workers


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="iSAID Instance Few-Shot Split 数据预处理 | Preprocessing"
    )
    p.add_argument("--src-root", type=str, default="data/iSAID_processed",
                   help="iSAID 处理后数据目录 | iSAID processed data directory")
    p.add_argument("--dst-root", type=str, default="data/iSAID_instance_fewshot",
                   help="输出目录 | Output directory")
    p.add_argument("--tile-size", type=int, default=896,
                   help="Tile 尺寸 (像素) | Tile size in pixels")
    p.add_argument("--stride", type=int, default=640,
                   help="滑动窗口步长 (像素) | Sliding window stride in pixels")
    p.add_argument("--max-images", type=int, default=0,
                   help="最大处理图像数 (0=全部, 调试用) | Max images (0=all)")
    p.add_argument("--splits", type=str, default="train,val",
                   help="处理的 split 列表 | Splits to process")
    p.add_argument("--steps", type=str, default="2,3",
                   help="执行步骤 | Steps: 2=tile, 3=metadata")
    p.add_argument("--workers", type=int, default=NUM_WORKERS,
                   help="并行进程数 | Number of parallel workers")
    p.add_argument("--no-folds", action="store_true",
                   help="跳过 Fold 定义生成，仅生成统计 | Skip fold generation, only generate stats")
    p.add_argument("--dry-run", action="store_true",
                   help="只检查不执行 | Only check, don't execute")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Step 2: 切 Tile + 生成 COCO JSON | Cut Tiles + Generate COCO JSON
# ═══════════════════════════════════════════════════════════════════

def _render_polygon_mask(poly: list, h: int, w: int) -> np.ndarray:
    """
    渲染单个 polygon 为二值掩码 | Render single polygon to binary mask.

    :param poly: [x1, y1, x2, y2, ...] 绝对坐标 | Absolute coordinates.
    :param h: 画布高度 | Canvas height.
    :param w: 画布宽度 | Canvas width.
    :return: [h, w] uint8 binary mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) < 6:
        return mask
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    # 裁剪到画布边界 | Clip to canvas boundary
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _extract_contours(mask: np.ndarray) -> list[list[float]]:
    """
    从二值掩码提取 polygon 轮廓 | Extract polygon contours from binary mask.

    :param mask: [h, w] uint8 binary mask.
    :return: List of polygons (each [x1,y1,x2,y2,...]).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # 简化 polygon: 保留所有点但防止过多 | Simplify: keep all but cap
        pts = cnt.squeeze(1).astype(float)
        if pts.ndim == 1:
            continue  # 退化情况 | Degenerate case
        # 如果点数太多 ( > 100), 用 Douglas-Peucker 简化
        # If too many points (> 100), simplify with Douglas-Peucker
        if len(pts) > 100:
            epsilon = 1.0  # 1px tolerance
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            pts = approx.squeeze(1).astype(float)
            if pts.ndim == 1:
                continue
        polygons.append(pts.flatten().tolist())
    return polygons


def _compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    从掩码计算 bbox [x, y, w, h] | Compute bbox [x, y, w, h] from mask.

    :param mask: [h, w] uint8 binary mask.
    :return: (x, y, w, h).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return (x, y, w, h)


def _process_single_image(args_tuple: tuple) -> dict:
    """
    处理单张全图: 切 tile + 生成 tile-level 标注 | Process single full image: cut tiles + generate tile-level annotations.

    :param args_tuple: (img_info, anns, img_dir, tile_size, stride, output_img_dir, split, global_tile_id_start)
    :return: {"tile_images": [...], "tile_annotations": [...], "stats": {...}}
    """
    (img_info, anns, img_dir, tile_size, stride,
     output_img_dir, split, global_tile_id_start) = args_tuple

    img_id_orig = img_info["id"]
    img_name = img_info.get("file_name", f"{img_id_orig}.png")
    img_path = img_dir / img_name
    if not img_path.exists():
        # 尝试其他可能的名称 | Try other possible names
        for alt in [f"{img_id_orig}.png", f"{img_id_orig}.jpg"]:
            if (img_dir / alt).exists():
                img_path = img_dir / alt
                break
        else:
            return {"status": "skip", "reason": f"Image not found: {img_name}"}

    # ── 加载图像 | Load image ──
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "skip", "reason": f"Cannot read: {img_path}"}
    H, W = img.shape[:2]

    tile_images = []
    tile_annotations = []
    total_instances = 0
    total_empty = 0
    total_tiles = 0

    # ── 切片循环 | Tile loop ──
    for y0 in range(0, H - tile_size + 1, stride) if H >= tile_size else [0]:
        for x0 in range(0, W - tile_size + 1, stride) if W >= tile_size else [0]:
            # 处理最后不完整的 tile | Handle incomplete edge tiles
            if H - y0 < tile_size:
                y0 = max(0, H - tile_size)
            if W - x0 < tile_size:
                x0 = max(0, W - tile_size)

            total_tiles += 1
            tile_id = global_tile_id_start + total_tiles
            tile_name = f"{Path(img_name).stem}_t{total_tiles:04d}.png"

            # ── 提取 tile 图像 | Extract tile image ──
            tile_img = img[y0:y0 + tile_size, x0:x0 + tile_size]
            tile_h, tile_w = tile_img.shape[:2]

            # 补零填充至标准 tile_size (处理图像边缘) | Zero-pad to standard tile_size (handle image edges)
            pad_bottom = tile_size - tile_h
            pad_right = tile_size - tile_w
            if pad_bottom > 0 or pad_right > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                tile_h, tile_w = tile_img.shape[:2]

            # ── 查找重叠标注 | Find overlapping annotations ──
            tile_anns = []
            tile_fg_area = 0

            for ann in anns:
                cat_id = ann.get("category_id", 0)
                if cat_id < 1 or cat_id > 15:
                    continue

                # 检查 bbox 重叠 | Check bbox overlap
                bbox = ann.get("bbox", [0, 0, 0, 0])
                ax, ay, aw, ah = bbox
                # 快速跳过: bbox 不相交 | Quick skip: bbox no overlap
                if (ax + aw <= x0 or ax >= x0 + tile_size or
                    ay + ah <= y0 or ay >= y0 + tile_size):
                    continue

                # ── 渲染 instance mask → tile 坐标系 | Render instance mask → tile coordinates ──
                seg = ann.get("segmentation", [])
                if not seg:
                    # 回退到 bbox | Fallback to bbox
                    bx = max(0, ax - x0)
                    by = max(0, ay - y0)
                    bw = min(aw, tile_w - bx)
                    bh = min(ah, tile_h - by)
                    if bw <= 0 or bh <= 0:
                        continue
                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                    tile_mask[by:by + bh, bx:bx + bw] = 1
                elif isinstance(seg, list):
                    if not seg:
                        continue
                    # 处理 polygon | Process polygon
                    if isinstance(seg[0], list):
                        polys = seg
                    elif isinstance(seg[0], (int, float)):
                        polys = [seg]
                    else:
                        continue

                    # 在 tile 画布上渲染所有 polygon | Render all polygons on tile canvas
                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                    for poly in polys:
                        if len(poly) < 6:
                            continue
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                        # 平移坐标系: absolute → tile-relative
                        # Translate coordinate system: absolute → tile-relative
                        pts[:, :, 0] -= x0
                        pts[:, :, 1] -= y0
                        # 裁剪 | Clip
                        pts[:, :, 0] = np.clip(pts[:, :, 0], 0, tile_w - 1)
                        pts[:, :, 1] = np.clip(pts[:, :, 1], 0, tile_h - 1)
                        cv2.fillPoly(tile_mask, [pts], 1)
                elif isinstance(seg, dict):
                    # RLE 格式 — 暂不支持, 回退到 bbox | RLE format — not yet supported, fallback to bbox
                    bx = max(0, int(ax) - x0)
                    by = max(0, int(ay) - y0)
                    bw = min(int(aw), tile_w - bx)
                    bh = min(int(ah), tile_h - by)
                    if bw <= 0 or bh <= 0:
                        continue
                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                    tile_mask[by:by + bh, bx:bx + bw] = 1
                else:
                    continue

                area_in_tile = int(tile_mask.sum())
                if area_in_tile < MIN_AREA:
                    continue

                tile_fg_area += area_in_tile

                # ── 提取 polygon 轮廓 (COCO 格式) | Extract polygon contours (COCO format) ──
                polygons = _extract_contours(tile_mask)
                if not polygons:
                    continue

                # ── 计算 tile-relative bbox | Compute tile-relative bbox ──
                tx, ty, tw, th = _compute_bbox(tile_mask)
                if tw == 0 or th == 0:
                    continue

                tile_anns.append({
                    "category_id": cat_id,
                    "segmentation": polygons,
                    "bbox": [tx, ty, tw, th],
                    "area": area_in_tile,
                    "iscrowd": ann.get("iscrowd", 0),
                    "orig_image_id": img_id_orig,
                })

            # ── 保存 tile 图像 | Save tile image ──
            tile_img_path = output_img_dir / tile_name
            cv2.imwrite(str(tile_img_path), tile_img)

            # ── 记录 tile image 条目 | Record tile image entry ──
            tile_images.append({
                "id": tile_id,
                "file_name": tile_name,
                "width": tile_w,
                "height": tile_h,
                "orig_image_id": img_id_orig,
                "orig_x": x0,
                "orig_y": y0,
            })

            # ── 记录 tile annotations (带 tile image id) | Record tile annotations (with tile image id) ──
            for ta in tile_anns:
                ta["image_id"] = tile_id
                ta["id"] = len(tile_annotations) + 1  # global unique annotation id
                tile_annotations.append(ta)

            total_instances += len(tile_anns)
            if len(tile_anns) == 0:
                total_empty += 1

    return {
        "status": "ok",
        "tile_images": tile_images,
        "tile_annotations": tile_annotations,
        "stats": {
            "orig_image_id": img_id_orig,
            "orig_size": [H, W],
            "n_tiles": total_tiles,
            "n_instances": total_instances,
            "n_empty": total_empty,
        },
    }


def step2_cut_tiles(args) -> dict:
    """
    Step 2: 切 tile + 生成 tile-level COCO JSON | Cut tiles + generate tile-level COCO JSON.

    :return: {"status": "ok", "tile_count": N, "annotation_count": M}
    """
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    tile_size = args.tile_size
    stride = args.stride
    splits = [s.strip() for s in args.splits.split(",")]

    print(f"\n{'='*60}")
    print(f"  Step 2: Cut Tiles + Generate COCO JSON")
    print(f"  Tile size: {tile_size}px, Stride: {stride}px")
    print(f"  Splits: {splits}")
    print(f"{'='*60}")

    total_stats = {}

    for split in splits:
        print(f"\n── Processing {split} ──")

        # ── 路径设置 | Path setup ──
        img_dir = src_root / split / "images"
        anno_path = src_root / split / "annotations" / f"instances_{split}.json"
        output_img_dir = dst_root / "images" / split
        output_anno_dir = dst_root / "annotations"

        if not anno_path.exists():
            print(f"  ⚠️  Annotation not found: {anno_path}, skipping...")
            continue
        if not img_dir.exists():
            print(f"  ⚠️  Image directory not found: {img_dir}, skipping...")
            continue

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_anno_dir.mkdir(parents=True, exist_ok=True)

        # ── 加载 COCO JSON | Load COCO JSON ──
        with open(anno_path) as f:
            coco_data = json.load(f)

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])

        # 按 image_id 索引标注 | Index annotations by image_id
        ann_by_image: dict[int, list] = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann["image_id"]].append(ann)

        if args.max_images > 0:
            images = images[:args.max_images]

        print(f"  {len(images)} images, {len(annotations)} annotations")

        # ── 并行处理 | Parallel processing ──
        global_tile_id = 0
        all_tile_images = []
        all_tile_annotations = []
        annotation_id_counter = 1
        split_stats = []

        # 为每张图分配 tile ID 起始值 | Assign tile ID start value per image
        tasks = []
        for img_info in images:
            img_id = img_info["id"]
            img_anns = ann_by_image.get(img_id, [])
            tasks.append((
                img_info, img_anns, img_dir, tile_size, stride,
                output_img_dir, split, global_tile_id,
            ))
            # 预估算 tile 数 | Estimate tile count
            h, w = img_info.get("height", 4000), img_info.get("width", 4000)
            n_tiles_x = max(1, (w - tile_size) // stride + 1) if w >= tile_size else 1
            n_tiles_y = max(1, (h - tile_size) // stride + 1) if h >= tile_size else 1
            global_tile_id += n_tiles_x * n_tiles_y

        # 串行处理 (GPU 无关, 用单进程避免 GIL 问题)
        # Serial processing (no GPU, single process to avoid GIL issues)
        n_workers = min(args.workers, len(tasks))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(tqdm(
                    ex.map(_process_single_image, tasks),
                    total=len(tasks),
                    desc=f"  Cutting tiles ({split})",
                    unit="img",
                ))
        else:
            results = []
            for task in tqdm(tasks, desc=f"  Cutting tiles ({split})", unit="img"):
                results.append(_process_single_image(task))

        # ── 收集结果 | Collect results ──
        for result in results:
            if result["status"] != "ok":
                print(f"  ⚠️  {result.get('reason', 'unknown error')}")
                continue

            for tile_img in result["tile_images"]:
                all_tile_images.append(tile_img)

            for tile_ann in result["tile_annotations"]:
                tile_ann["id"] = annotation_id_counter
                annotation_id_counter += 1
                all_tile_annotations.append(tile_ann)

            split_stats.append(result["stats"])

        # ── 写入 COCO JSON | Write COCO JSON ──
        coco_output = {
            "images": all_tile_images,
            "annotations": all_tile_annotations,
            "categories": ISAID_CATEGORIES_COCO,
        }

        output_anno_path = output_anno_dir / f"instances_{split}.json"
        with open(output_anno_path, "w") as f:
            json.dump(coco_output, f)
        print(f"  ✅ Written: {output_anno_path}")
        print(f"     {len(all_tile_images)} tiles, {len(all_tile_annotations)} instances")

        # ── 统计 | Statistics ──
        n_total_tiles = sum(s["n_tiles"] for s in split_stats)
        n_total_inst = sum(s["n_instances"] for s in split_stats)
        n_total_empty = sum(s["n_empty"] for s in split_stats)
        print(f"     Empty tiles: {n_total_empty}/{n_total_tiles} "
              f"({100 * n_total_empty / max(1, n_total_tiles):.1f}%)")

        total_stats[split] = {
            "n_images": len(results),
            "n_tiles": n_total_tiles,
            "n_instances": n_total_inst,
            "n_empty": n_total_empty,
        }

    print(f"\n  ✅ Step 2 complete.")
    for split, stats in total_stats.items():
        print(f"     {split}: {stats['n_tiles']} tiles, {stats['n_instances']} instances")

    return {"status": "ok", "stats": total_stats}


# ═══════════════════════════════════════════════════════════════════
# Step 3: 生成 Fold 定义 + 统计 | Generate Fold Definitions + Stats
# ═══════════════════════════════════════════════════════════════════

def step3_metadata(args) -> dict:
    """
    Step 3: 生成 per-class 统计 (+ 可选 Fold 定义) | Generate per-class stats (+ optional Fold definitions).

    :return: {"status": "ok"}
    """
    dst_root = Path(args.dst_root)
    splits = [s.strip() for s in args.splits.split(",")]
    skip_folds = getattr(args, "no_folds", False)

    print(f"\n{'='*60}")
    print(f"  Step 3: Generate Statistics" + (" (no folds)" if skip_folds else " + Fold Definitions"))
    print(f"{'='*60}")

    # ── 生成 Fold 定义 (可选) | Generate Fold Definitions (optional) ──
    if not skip_folds:
        folds_dir = dst_root / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)

        for fold_id, fold_data in ISAID_INSTANCE_FOLDS.items():
            fold_path = folds_dir / f"fold_{fold_id}.json"
            fold_output = {
                "fold": fold_id,
                "base": sorted(fold_data["base"]),
                "novel": sorted(fold_data["novel"]),
                "base_names": {str(cid): ISAID_CATEGORIES[cid] for cid in fold_data["base"]},
                "novel_names": {str(cid): ISAID_CATEGORIES[cid] for cid in fold_data["novel"]},
                "n_base": len(fold_data["base"]),
                "n_novel": len(fold_data["novel"]),
            }
            with open(fold_path, "w") as f:
                json.dump(fold_output, f, indent=2, ensure_ascii=False)
            print(f"  [OK] Fold {fold_id}: Base={fold_output['n_base']}, "
                  f"Novel={fold_output['n_novel']} -> {fold_path}")
    else:
        print(f"  [SKIP] Fold generation disabled (--no-folds)")

    # ── 统计 per-class instance/tile 分布 | Per-class instance/tile distribution ──
    stats_dir = dst_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    class_stats = {}
    for split in splits:
        anno_path = dst_root / "annotations" / f"instances_{split}.json"
        if not anno_path.exists():
            print(f"  ⚠️  {anno_path} not found, skipping stats for {split}")
            continue

        with open(anno_path) as f:
            coco_data = json.load(f)

        tiles = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])

        # Per-class annotation count + area stats
        class_count: dict[int, int] = defaultdict(int)
        class_area: dict[int, list] = defaultdict(list)
        class_tiles: dict[int, set] = defaultdict(set)

        for ann in annotations:
            cat_id = ann["category_id"]
            class_count[cat_id] += 1
            class_area[cat_id].append(ann.get("area", 0))
            class_tiles[cat_id].add(ann["image_id"])

        per_class = {}
        for cat_id in range(1, 16):
            cat_name = ISAID_CATEGORIES[cat_id]
            areas = class_area.get(cat_id, [])
            per_class[cat_name] = {
                "id": cat_id,
                "n_instances": class_count.get(cat_id, 0),
                "n_tiles": len(class_tiles.get(cat_id, set())),
                "total_area_px": sum(areas),
                "mean_area_px": float(np.mean(areas)) if areas else 0.0,
                "median_area_px": float(np.median(areas)) if areas else 0.0,
                "min_area_px": int(np.min(areas)) if areas else 0,
                "max_area_px": int(np.max(areas)) if areas else 0,
            }

        n_total_tiles = len(tiles)
        n_fg_tiles = len(set(ann["image_id"] for ann in annotations))
        n_empty_tiles = n_total_tiles - n_fg_tiles

        class_stats[split] = {
            "n_tiles": n_total_tiles,
            "n_fg_tiles": n_fg_tiles,
            "n_empty_tiles": n_empty_tiles,
            "empty_ratio": n_empty_tiles / max(1, n_total_tiles),
            "n_annotations": len(annotations),
            "per_class": per_class,
        }

    # ── 写入统计 JSON | Write stats JSON ──
    stats_path = stats_dir / "class_distribution.json"
    stats_output = {
        "tile_config": {
            "tile_size": args.tile_size,
            "stride": args.stride,
        },
        "categories": ISAID_CATEGORIES,
        "splits": class_stats,
    }
    if not skip_folds:
        stats_output["folds"] = {
            str(fold_id): {
                "base": sorted(fold_data["base"]),
                "novel": sorted(fold_data["novel"]),
            }
            for fold_id, fold_data in ISAID_INSTANCE_FOLDS.items()
        }
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Statistics: {stats_path}")

    # ── 打印摘要 | Print Summary ──
    print(f"\n  {'─'*70}")
    print(f"  Per-Class Instance Distribution (train split):")
    print(f"  {'─'*70}")
    if "train" in class_stats:
        pc = class_stats["train"]["per_class"]
        for cat_name in ISAID_CATEGORIES.values():
            info = pc.get(cat_name, {})
            print(f"  {cat_name:20s} (id={info.get('id', '?'):>2}): "
                  f"{info.get('n_instances', 0):>6d} instances, "
                  f"{info.get('n_tiles', 0):>5d} tiles, "
                  f"mean_area={info.get('mean_area_px', 0):.0f}px²")

    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    print("=" * 70)
    print("  iSAID Instance Few-Shot Split — Preprocessing")
    print(f"  Source:     {args.src_root}")
    print(f"  Output:     {args.dst_root}")
    print(f"  Tile:       {args.tile_size}px  Stride: {args.stride}px")
    print(f"  Splits:     {args.splits}")
    print(f"  Steps:      {steps}")
    if args.max_images > 0:
        print(f"  Max images: {args.max_images}")
    if args.dry_run:
        print("  MODE:       DRY-RUN")
    print("=" * 70)

    # ── 预检查 | Pre-check ──
    src_root = Path(args.src_root)
    if not src_root.exists():
        print(f"\n❌ Source root not found: {src_root}")
        print("   Run tools/data/prep_isaid.py first to generate iSAID_processed/")
        return

    if args.dry_run:
        print("\n  Dry-run complete. Remove --dry-run to execute.")
        return

    # ── Step 2: Cut Tiles ──
    if "2" in steps:
        result = step2_cut_tiles(args)
        if result["status"] != "ok":
            print(f"  ❌ Step 2 failed!")
            return

    # ── Step 3: Metadata ──
    if "3" in steps:
        result = step3_metadata(args)
        if result["status"] != "ok":
            print(f"  ❌ Step 3 failed!")
            return

    print(f"\n{'='*70}")
    print(f"  ✅ All steps complete!")
    print(f"  Output: {args.dst_root}")
    print(f"\n  Use: ISAIDInstanceFewShotDataset(root='{args.dst_root}', split='train', fold=0)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
