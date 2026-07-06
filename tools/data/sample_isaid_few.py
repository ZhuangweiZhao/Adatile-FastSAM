#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
 sample_isaid_few.py — iSAID 1% Few-Shot 子集抽取工具
 iSAID 1% Few-Shot Subset Sampler

 功能: 从 iSAID_processed 数据集的每个子文件夹中随机抽取 1% 样本，
       使用固定种子保证可复现，生成 iSAID-few 数据集。

 Usage:
   python tools/data/sample_isaid_few.py \
       --seed 42 \
       --input E:/A_postgraduate_stude/AdaTile-FastSAM/data/iSAID_processed \
       --output E:/A_postgraduate_stude/AdaTile-FastSAM/data/iSAID-few \
       --percent 1

 输出结构 / Output Structure:
   iSAID-few/
   ├── train/
   │   ├── images/              # 1% sampled images
   │   ├── annotations/
   │   │   └── instances_train.json   # filtered COCO JSON
   │   └── image_dims.json            # filtered dims
   ├── val/
   │   ├── images/
   │   └── annotations/
   │       └── instances_val.json
   └── test/
       └── images/

============================================================================
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


# ==============================================================================
# 命令行参数解析 / CLI Argument Parser
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="iSAID 1% Few-Shot Subset Sampler — 从 iSAID_processed 抽取 1% 子集"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 / Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--input", type=str,
        default="E:/A_postgraduate_stude/AdaTile-FastSAM/data/iSAID_processed",
        help="输入数据集路径 / Input dataset path (iSAID_processed)"
    )
    parser.add_argument(
        "--output", type=str,
        default="E:/A_postgraduate_stude/AdaTile-FastSAM/data/iSAID-few",
        help="输出数据集路径 / Output dataset path (iSAID-few)"
    )
    parser.add_argument(
        "--percent", type=float, default=1.0,
        help="采样百分比 / Sampling percentage (default: 1.0)"
    )
    return parser.parse_args()


# ==============================================================================
# 核心逻辑 / Core Logic
# ==============================================================================

def sample_images(image_dir: Path, percent: float, seed: int):
    """
    从 images 目录随机抽样 / Randomly sample images from a directory.

    Args:
        image_dir: 图片目录路径
        percent: 采样百分比 (0-100)
        seed: 随机种子

    Returns:
        selected: 被选中的文件名列表 (排序后) / Sorted list of selected filenames
    """
    all_images = sorted([
        f.name for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ])

    n_total = len(all_images)
    n_sample = max(1, int(n_total * percent / 100.0))

    rng = random.Random(seed)
    selected = sorted(rng.sample(all_images, n_sample))

    return selected, n_total, n_sample


def filter_coco_json(
    src_json: Path, dst_json: Path, selected_filenames: set
):
    """
    过滤 COCO JSON，只保留被选中图片的标注 / Filter COCO JSON to only keep selected images.

    Args:
        src_json: 源 COCO JSON 路径
        dst_json: 目标 JSON 路径
        selected_filenames: 被选中的图片文件名集合
    """
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 建立 file_name → image_id 映射 / Build file_name → image_id mapping
    file_to_id = {}
    for img in data["images"]:
        file_to_id[img["file_name"]] = img["id"]

    # 获取被选中的 image_id 集合 / Get selected image_id set
    selected_ids = set()
    for fname in selected_filenames:
        if fname in file_to_id:
            selected_ids.add(file_to_id[fname])

    # 过滤 images 和 annotations / Filter images and annotations
    filtered_images = [
        img for img in data["images"]
        if img["file_name"] in selected_filenames
    ]
    filtered_annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] in selected_ids
    ]

    # 重新分配 annotation id (保持连续) / Reassign annotation ids (keep continuous)
    for i, ann in enumerate(filtered_annotations):
        ann["id"] = i + 1

    out_data = {
        "images": filtered_images,
        "categories": data["categories"],  # 保留全部类别 / Keep all categories
        "annotations": filtered_annotations,
    }

    os.makedirs(dst_json.parent, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    return len(filtered_images), len(filtered_annotations)


def filter_image_dims(
    src_dims: Path, dst_dims: Path, selected_filenames: set
):
    """
    过滤 image_dims.json，只保留被选中图片 / Filter image_dims.json to keep only selected images.

    image_dims 格式: { "image_id": [height, width], ... }

    需要注意的是，image_dims 的 key 可能是数字 ID 或文件名，
    这里我们通过 COCO JSON 的 file_name → image_id 映射来过滤。
    如果没有 COCO JSON，则保留全部。

    Args:
        src_dims: 源 image_dims.json 路径
        dst_dims: 目标 image_dims.json 路径
        selected_filenames: 被选中的图片文件名集合
    """
    with open(src_dims, "r", encoding="utf-8") as f:
        dims = json.load(f)

    # image_dims.json 的 key 是数字字符串 (image_id)
    # COCO JSON images 列表中的 id 就是这些 key
    # 我们需要借助 COCO JSON 的 file_name → id 映射来过滤
    # 先尝试从同目录的 annotations 中找到对应关系

    # 简化方案: 如果 key 是数字, 尝试从 annotations 中来映射
    # 如果找不到 annotations 或者映射失败, 保留全部 dims
    # (image_dims.json 本身很小, 全保留也没问题)

    if not dims:
        return 0, 0

    # 判断 key 的类型: 是否为数字 ID
    first_key = next(iter(dims.keys()))
    if first_key.isdigit():
        # key 是 image_id (数字), key 数量应该等于 images 数量
        # 无法直接映射, 全保留 (文件很小)
        os.makedirs(dst_dims.parent, exist_ok=True)
        with open(dst_dims, "w", encoding="utf-8") as f:
            json.dump(dims, f, ensure_ascii=False, indent=2)
        return len(dims), len(dims)
    else:
        # key 是文件名, 直接过滤
        filtered = {k: v for k, v in dims.items() if k in selected_filenames}
        os.makedirs(dst_dims.parent, exist_ok=True)
        with open(dst_dims, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        return len(filtered), len(dims)


def process_split(
    src_split: Path,
    dst_split: Path,
    percent: float,
    seed: int,
    split_name: str,
):
    """
    处理一个 split (train/val/test) / Process one split.

    Args:
        src_split: 源 split 目录
        dst_split: 目标 split 目录
        percent: 采样百分比
        seed: 随机种子
        split_name: split 名称 (用于日志)

    Returns:
        stats: 统计信息字典 / Statistics dict
    """
    print(f"\n{'='*60}")
    print(f"  处理 {split_name} / Processing {split_name}")
    print(f"{'='*60}")

    stats = {"split": split_name}
    src_images = src_split / "images"

    if not src_images.exists():
        print(f"  [WARN] Skip: {src_images} not found")
        return stats

    # 1. 抽样图片 / Sample images
    selected, n_total, n_sample = sample_images(src_images, percent, seed)
    stats["total_images"] = n_total
    stats["sampled_images"] = n_sample
    print(f"  图片 / Images: {n_total} → {n_sample} ({percent}%, seed={seed})")

    # 2. 复制图片 / Copy images
    dst_images = dst_split / "images"
    os.makedirs(dst_images, exist_ok=True)
    for fname in selected:
        shutil.copy2(src_images / fname, dst_images / fname)
    print(f"  [OK] Copied {n_sample} images to: {dst_images}")

    selected_set = set(selected)

    # 3. 过滤 COCO JSON / Filter COCO JSON
    src_ann_dir = src_split / "annotations"
    if src_ann_dir.exists():
        for ann_file in src_ann_dir.iterdir():
            if ann_file.suffix == ".json" and ann_file.name.startswith("instances"):
                dst_ann = dst_split / "annotations" / ann_file.name
                n_img, n_ann = filter_coco_json(ann_file, dst_ann, selected_set)
                stats["annotation_file"] = ann_file.name
                stats["filtered_ann_images"] = n_img
                stats["filtered_annotations"] = n_ann
                print(f"  标注 / Annotations: {ann_file.name} → {n_img} imgs, {n_ann} anns")
            # 跳过 .bak 文件 / Skip backup files
    else:
        print(f"  [INFO] No annotations directory")

    # 4. 过滤 image_dims.json / Filter image_dims.json
    src_dims = src_split / "image_dims.json"
    if src_dims.exists():
        dst_dims = dst_split / "image_dims.json"
        n_kept, n_orig = filter_image_dims(src_dims, dst_dims, selected_set)
        stats["dims_kept"] = n_kept
        stats["dims_original"] = n_orig
        print(f"  尺寸 / Dims: {n_kept}/{n_orig} 条目保留")

    return stats


# ==============================================================================
# 主流程 / Main
# ==============================================================================

def main():
    args = parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    seed = args.seed
    percent = args.percent

    # 验证输入 / Validate input
    if not src_root.exists():
        print(f"[ERROR] Input path not found: {src_root}")
        sys.exit(1)

    if percent <= 0 or percent > 100:
        print(f"[ERROR] Invalid percentage: {percent} (must be 0 < p <= 100)")
        sys.exit(1)

    # 检测可用的 splits / Detect available splits
    available_splits = [
        d.name for d in src_root.iterdir()
        if d.is_dir() and d.name in {"train", "val", "test"}
    ]
    print(f"[Splits] Detected: {available_splits}")
    print(f"[Seed] Random seed: {seed}")
    print(f"[Ratio] Sampling ratio: {percent}%")
    print(f"[Output] Path: {dst_root}")

    # 清理或创建输出目录 / Clean or create output directory
    if dst_root.exists():
        print(f"\n[WARN] Output exists, will be cleared: {dst_root}")
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    # 处理每个 split / Process each split
    all_stats = []
    for split_name in ["train", "val", "test"]:
        src_split = src_root / split_name
        if not src_split.exists():
            continue
        dst_split = dst_root / split_name

        # train/val/test 使用不同的 seed 偏移，确保抽样独立
        # Each split gets a different seed offset for independent sampling
        split_seed = seed + {"train": 0, "val": 1000, "test": 2000}[split_name]
        stats = process_split(src_split, dst_split, percent, split_seed, split_name)
        all_stats.append(stats)

    # ==========================================================================
    # 汇总报告 / Summary Report
    # ==========================================================================
    print(f"\n\n{'='*60}")
    print(f"  == SUMMARY REPORT ==")
    print(f"{'='*60}")
    print(f"  种子 / Seed:       {seed}")
    print(f"  百分比 / Percent:  {percent}%")
    print(f"  源路径 / Source:   {src_root}")
    print(f"  输出路径 / Output: {dst_root}")
    print(f"  {'─'*50}")

    total_imgs = 0
    total_anns = 0

    for s in all_stats:
        split_name = s.get("split", "?")
        n_total = s.get("total_images", 0)
        n_sampled = s.get("sampled_images", 0)
        n_anns = s.get("filtered_annotations", 0)
        total_imgs += n_sampled
        total_anns += n_anns

        print(f"\n  [{split_name}]")
        print(f"    图片 / Images:      {n_sampled:5d} / {n_total:5d}  ({n_sampled/max(n_total,1)*100:.1f}%)")
        if "annotation_file" in s:
            print(f"    标注 / Annotations: {n_anns:6d}")
            print(f"    标注文件 / Ann file: {s['annotation_file']}")
        if "dims_kept" in s:
            print(f"    尺寸 / Dims:        {s['dims_kept']}/{s['dims_original']}")

    print(f"\n  {'─'*50}")
    print(f"  == Total:")
    print(f"     总图片 / Total images:      {total_imgs}")
    print(f"     总标注 / Total annotations: {total_anns}")
    print(f"     总类别 / Total categories:  15 (保留全部 / all kept)")
    print(f"{'='*60}\n")

    # 保存汇总到文件 / Save summary to file
    summary_path = dst_root / "sampling_summary.json"
    summary = {
        "seed": seed,
        "percent": percent,
        "source": str(src_root),
        "output": str(dst_root),
        "splits": all_stats,
        "total_images": total_imgs,
        "total_annotations": total_anns,
        "total_categories": 15,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
