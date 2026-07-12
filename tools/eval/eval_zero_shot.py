#!/usr/bin/env python3
"""
V3-01: FastSAM Zero-Shot 实例分割 Baseline (896² tiles).
===========================================================

基于 v3 Instance Few-Shot Split (896² tiles), 对 FastSAM 做 zero-shot COCO AP 评估。
使用 tile 直接推理 (不 resize), 建立 v3 新基线。

Zero-shot COCO AP baseline on v3 Instance Few-Shot Split (896² tiles).
Uses tile-direct inference (no resize), establishes v3 baseline.

与 D-00 (v2) 的关键区别 | Key differences from D-00 (v2):
    - v2: 全图 resize→1024px 推理, 小目标被压缩消失
    - v3: 896² tile 直接推理, 小目标保持原始分辨率
    - v2: iSAID_processed COCO JSON (全图)
    - v3: iSAID_instance_fewshot COCO JSON (tile-level)

用法 | Usage::

    # 本地快速测试 (100 tiles)
    python tools/eval/eval_zero_shot.py --max-tiles 100

    # 全量评估 (后台)
    nohup python tools/eval/eval_zero_shot.py \
        --device cuda --max-tiles 0 \
        > /root/autodl-tmp/v3_01_zero_shot.log 2>&1 &
"""

from __future__ import annotations

import sys, argparse, json, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import cv2
import numpy as np
from tqdm import tqdm

import torch
from pycocotools.mask import encode as rle_encode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="V3-01: FastSAM Zero-Shot on v3 Instance Few-Shot Split"
    )
    p.add_argument("--data-root", type=str, default="data/iSAID_instance_fewshot",
                   help="v3 数据根目录 | v3 data root")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conf", type=float, default=0.25,
                   help="FastSAM 置信度阈值 | Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7,
                   help="FastSAM NMS IoU 阈值 | NMS IoU threshold")
    p.add_argument("--max-tiles", type=int, default=0,
                   help="最大 tile 数 (0=全部) | Max tiles (0=all)")
    p.add_argument("--class-agnostic", action="store_true", default=True,
                   help="Class-agnostic 评估 (默认) | Class-agnostic evaluation (default)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 辅助函数 | Helpers
# ═══════════════════════════════════════════════════════════════════

def mask_to_rle(mask: np.ndarray) -> dict:
    """
    二值 mask → COCO RLE 格式 (counts 为 UTF-8 字符串).
    Binary mask → COCO RLE format (counts as UTF-8 string).
    """
    mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
    rle = rle_encode(mask_uint8)
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    """二值 mask → COCO bbox [x, y, w, h]."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return [float(x), float(y), float(w), float(h)]


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── 输出目录 | Output ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/v3_01_zero_shot_{args.split}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("v3_01_zero_shot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "eval.jsonl")))
    logger.log_info("config", f"V3-01 Zero-Shot: split={args.split}, "
                    f"conf={args.conf}, iou={args.iou}, max_tiles={args.max_tiles}")

    # ── 加载 FastSAM | Load FastSAM ──
    logger.log_info("model", "Loading FastSAM-x...")
    from fastsam import FastSAM
    fastsam_weights = str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt")
    model = FastSAM(fastsam_weights)
    device = torch.device(args.device)
    model.model.to(device)
    logger.log_info("model", f"FastSAM loaded on {device}")

    # ── 加载 GT | Load Ground Truth ──
    gt_path = Path(args.data_root) / "annotations" / f"instances_{args.split}.json"
    if not gt_path.exists():
        logger.log_info("error", f"GT not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_images = gt_data["images"]
    gt_annotations = gt_data["annotations"]

    # 按 image_id 索引 GT annotations | Index GT annotations by image_id
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in gt_annotations:
        gt_by_image[ann["image_id"]].append(ann)

    img_dir = Path(args.data_root) / "images" / args.split

    # ── 限制 tile 数 | Limit tiles ──
    if args.max_tiles > 0:
        gt_images = gt_images[:args.max_tiles]
        logger.log_info("config", f"Limited to {len(gt_images)} tiles")

    logger.log_info("config", f"Evaluating {len(gt_images)} tiles...")

    # ── 推理循环 | Inference Loop ──
    all_predictions: list[dict] = []
    pred_id = 0
    tiles_processed = 0
    tiles_with_preds = 0
    total_pred_masks = 0

    pbar = tqdm(gt_images, desc="Inference", unit="tile")
    for img_info in pbar:
        tile_id = img_info["id"]
        tile_name = img_info["file_name"]
        tile_h, tile_w = img_info["height"], img_info["width"]

        # ── 加载 tile 图像 | Load tile image ──
        img_path = img_dir / tile_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── FastSAM predict | FastSAM inference ──
        try:
            results = model(
                source=img_rgb,
                device=device,
                conf=args.conf,
                iou=args.iou,
                retina_masks=True,
                imgsz=max(tile_h, tile_w),
                verbose=False,
            )
        except Exception as e:
            logger.log_info("error", f"Inference error on tile {tile_id}: {e}")
            continue

        tiles_processed += 1

        # ── 提取 instance masks → COCO prediction | Extract masks → COCO prediction ──
        if results and len(results) > 0:
            result = results[0]  # 单张图的结果 | Single image result
            masks = result.masks
            if masks is not None and len(masks.data) > 0:
                tiles_with_preds += 1
                # FastSAM Masks 对象没有 conf 属性，统一用默认 score
                # FastSAM Masks object has no conf attribute, use default score
                scores = getattr(masks, "conf", None)
                if scores is None:
                    scores = [1.0] * len(masks.data)
                for mask_tensor, score in zip(masks.data, scores):
                    mask_np = mask_tensor.cpu().numpy().astype(bool)

                    # 确保 mask 尺寸与 tile 一致 | Ensure mask matches tile size
                    if mask_np.shape[0] != tile_h or mask_np.shape[1] != tile_w:
                        mask_np_resized = np.zeros((tile_h, tile_w), dtype=bool)
                        hh = min(mask_np.shape[0], tile_h)
                        ww = min(mask_np.shape[1], tile_w)
                        mask_np_resized[:hh, :ww] = mask_np[:hh, :ww]
                        mask_np = mask_np_resized

                    area = int(mask_np.sum())
                    if area < 16:  # 过滤极小噪声 | Filter tiny noise
                        continue

                    rle = mask_to_rle(mask_np)
                    bbox = mask_to_bbox(mask_np)

                    all_predictions.append({
                        "id": pred_id,
                        "image_id": tile_id,
                        "category_id": 1,  # class-agnostic
                        "segmentation": rle,
                        "bbox": bbox,
                        "score": float(score) if score is not None else args.conf,
                        "area": area,
                    })
                    pred_id += 1
                    total_pred_masks += 1

        # 更新进度条 | Update progress bar
        pbar.set_postfix({
            "preds": total_pred_masks,
            "tiles_w_pred": tiles_with_preds,
        })

    logger.log_info("inference", f"Processed {tiles_processed}/{len(gt_images)} tiles, "
                    f"{total_pred_masks} masks from {tiles_with_preds} tiles")

    # ── 保存 predictions | Save predictions ──
    pred_path = out_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    logger.log_info("save", f"Saved {len(all_predictions)} predictions → {pred_path}")

    # ── 准备 GT for class-agnostic eval | Prepare GT for class-agnostic eval ──
    gt_agnostic = {
        "images": gt_data["images"],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    for ann in gt_data["annotations"]:
        gt_agnostic["annotations"].append({
            **ann,
            "category_id": 1,  # remap to class-agnostic
        })

    gt_agnostic_path = out_dir / "gt_agnostic.json"
    with open(gt_agnostic_path, "w") as f:
        json.dump(gt_agnostic, f)

    # ── COCO AP 评估 | COCO AP Evaluation ──
    logger.log_info("eval", "Running COCO AP evaluation...")
    coco_gt = COCO(str(gt_agnostic_path))

    if len(all_predictions) == 0:
        logger.log_info("eval", "WARNING: 0 predictions! All AP = 0.0")
        coco_stats = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                      "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
                      "AR_max1": 0.0, "AR_max10": 0.0, "AR_max100": 0.0,
                      "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0}
    else:
        coco_pred = coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_stats = {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
            "AR_max1": float(coco_eval.stats[6]),
            "AR_max10": float(coco_eval.stats[7]),
            "AR_max100": float(coco_eval.stats[8]),
        }
        # 额外获取 AR per area | Extra AR per area
        coco_eval_area = COCOeval(coco_gt, coco_pred, iouType="segm")
        coco_eval_area.params.areaRng = [[0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        coco_eval_area.params.areaRngLbl = ["small", "medium", "large"]
        # 使用 maxDets=100 的 AR
        coco_eval_area.evaluate()
        coco_eval_area.accumulate()
        # AR at maxDets=100 (index 2 in the stats array per area)
        # stats array: AP, AP50, AP75, AP_small, AP_medium, AP_large, AR_max1, AR_max10, AR_max100, AR_small, AR_medium, AR_large
        coco_stats["AR_small"] = float(coco_eval_area.stats[9]) if len(coco_eval_area.stats) > 9 else 0.0
        coco_stats["AR_medium"] = float(coco_eval_area.stats[10]) if len(coco_eval_area.stats) > 10 else 0.0
        coco_stats["AR_large"] = float(coco_eval_area.stats[11]) if len(coco_eval_area.stats) > 11 else 0.0

    # ── 记录结果 | Log Results ──
    logger.log_metric("AP", coco_stats["AP"])
    logger.log_metric("AP50", coco_stats["AP50"])
    logger.log_metric("AP75", coco_stats["AP75"])
    logger.log_metric("AP_small", coco_stats["AP_small"])
    logger.log_metric("AP_medium", coco_stats["AP_medium"])
    logger.log_metric("AP_large", coco_stats["AP_large"])
    logger.log_metric("AR_max100", coco_stats["AR_max100"])
    logger.log_metric("AR_large", coco_stats.get("AR_large", 0.0))

    # ── 保存结果 JSON | Save Results JSON ──
    results = {
        "config": {
            "data_root": args.data_root,
            "split": args.split,
            "conf": args.conf,
            "iou": args.iou,
            "class_agnostic": args.class_agnostic,
        },
        "stats": coco_stats,
        "counts": {
            "n_tiles_total": len(gt_images),
            "n_tiles_processed": tiles_processed,
            "n_tiles_with_preds": tiles_with_preds,
            "n_predictions": len(all_predictions),
            "n_gt_annotations": len(gt_data["annotations"]),
        },
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── 打印摘要 | Print Summary ──
    print(f"\n{'='*60}")
    print(f"  V3-01 Zero-Shot Baseline — Results")
    print(f"  Split: {args.split}, Tiles: {tiles_processed}")
    print(f"  Predictions: {len(all_predictions)}")
    print(f"{'='*60}")
    print(f"  AP           (IoU=0.50:0.95): {coco_stats['AP']:.4f}")
    print(f"  AP50         (IoU=0.50):      {coco_stats['AP50']:.4f}")
    print(f"  AP75         (IoU=0.75):      {coco_stats['AP75']:.4f}")
    print(f"  AP_small     (<32²):           {coco_stats['AP_small']:.4f}")
    print(f"  AP_medium    (32²-96²):        {coco_stats['AP_medium']:.4f}")
    print(f"  AP_large     (>=96²):          {coco_stats['AP_large']:.4f}")
    print(f"  AR_max100:                      {coco_stats['AR_max100']:.4f}")
    print(f"  AR_large@100:                   {coco_stats.get('AR_large', 0.0):.4f}")
    print(f"{'='*60}")
    print(f"  Output: {out_dir}")

    logger.log_info("done", f"Results saved to {out_dir}")


if __name__ == "__main__":
    from tools._deprecated_guard import require_legacy_optin
    require_legacy_optin(__file__)  # DEPRECATED: pre-V3 protocol (see EVALUATION_PROTOCOL_V3.md §12.4)
    main()
