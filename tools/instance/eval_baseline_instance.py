#!/usr/bin/env python3
"""
FastSAM Zero-Shot 实例分割 Baseline | FastSAM Zero-Shot Instance Segmentation Baseline.
==========================================================================================

使用 FastSAM 原生 predict 模式评估 iSAID 数据集上的实例分割性能，
建立 COCO AP 上界参考。

Evaluates FastSAM's native predict mode on iSAID for instance segmentation,
establishing COCO AP upper bound reference.

这是 few-shot fine-tuning 的对比基线：如果 FastSAM 全模型能达到 AP=X，
few-shot adaptation 的目标就是接近 X。

用法 | Usage::

    python tools/instance/eval_baseline_instance.py \
        --data-root data/iSAID_processed \
        --split val \
        --device cuda
"""

from __future__ import annotations

import sys, argparse, json, os
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.metrics.coco_eval import COCOInstanceEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="FastSAM Zero-Shot 实例分割 Baseline")

    p.add_argument("--data-root", type=str, default="data/iSAID_processed",
                   help="iSAID COCO 数据根目录 | iSAID COCO data root")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conf", type=float, default=0.25,
                   help="置信度阈值 | Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU 阈值 | NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="推理图像尺寸 | Inference image size")
    p.add_argument("--max-images", type=int, default=0,
                   help="最大评估图像数 (0=全部) | Max images to evaluate (0=all)")
    p.add_argument("--output-dir", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # ── 导入 FastSAM | Import FastSAM ──
    from fastsam import FastSAM

    # ── 输出目录 | Output ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/baseline_fastsam_zero_shot_{args.split}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("baseline")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "eval.jsonl")))

    logger.log_info("config", f"FastSAM Zero-Shot Baseline: split={args.split}, "
                    f"conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    # ── GT 标注路径 | GT Annotation Path ──
    gt_path = Path(args.data_root) / args.split / "annotations" / f"instances_{args.split}.json"
    if not gt_path.exists():
        logger.log_info("error", f"GT annotation not found: {gt_path}")
        sys.exit(1)

    # ── 加载 FastSAM | Load FastSAM ──
    logger.log_info("model", "Loading FastSAM...")
    model = FastSAM(str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "FastSAM-x.pt"))
    device = torch.device(args.device)
    model.model.to(device)

    # ── 获取图像列表 | Get image list ──
    import json
    with open(gt_path) as f:
        coco_data = json.load(f)
    images = coco_data["images"]
    if args.max_images > 0:
        images = images[:args.max_images]

    logger.log_info("data", f"Evaluating {len(images)} images")

    # ── 初始化 COCO 评估器 | Initialize COCO Evaluator ──
    evaluator = COCOInstanceEvaluator(str(gt_path), iouType="segm")

    img_dir = Path(args.data_root) / args.split / "images"

    for img_info in tqdm(images, desc="Evaluating"):
        image_id = img_info["id"]
        img_name = img_info.get("file_name", f"{image_id}.png")
        img_path = img_dir / img_name
        if not img_path.exists():
            img_path = img_dir / f"{image_id}.png"
            if not img_path.exists():
                continue

        # ── FastSAM 推理 | FastSAM Inference ──
        try:
            results = model.predict(
                source=str(img_path),
                device=device,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                retina_masks=True,
            )
        except Exception as e:
            logger.log_info("error", f"Image {image_id}: prediction failed: {e}")
            continue

        if results is None or len(results) == 0 or results[0].masks is None:
            continue

        result = results[0]
        masks = result.masks.data  # [N, H, W] binary
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            conf_val = float(boxes.conf[i])
            cls_id = int(boxes.cls[i]) if boxes.cls is not None else 1

            # ── 类别映射 | Category Mapping ──
            # FastSAM 是 class-agnostic (SA-1B 训练)，所有检测为 cls=0。
            # iSAID 有 15 个类别 (1-15)。zero-shot 类别匹配无法做到。
            # 策略: 将所有 FastSAM 检测统一映射到 iSAID 类别，
            # 最乐观的评估方式是对每个预测尝试所有 15 个类别。
            # 但 COCOeval 的匹配规则是: 每个 pred 只能匹配一个 GT instance。
            # 如果 15 个类别共享同一组预测，则 GT 匹配会被稀释。
            #
            # 保守做法: 将预测归为类别 1，仅与 GT cls=1 比较。
            # 这低估了 FastSAM 的 class-agnostic segment-anything 能力，
            # 但这是 COCO AP 框架下唯一语义正确的做法。
            # FastSAM is class-agnostic (SA-1B). iSAID has 15 classes (1-15).
            # Conservative: map all preds to cls=1, match only GT cls=1.
            # This undercounts FastSAM's capability but is the only
            # semantically valid approach under per-class COCO AP.
            #
            # 同时，对于 class-agnostic 评估，我们也输出 proxy AP
            # (所有 GT 类合并为 class=1 的简化评估)。
            # Also output proxy AP (all GT classes merged to cls=1).
            cls_id = 1  # 所有检测映射到 class 1 | Map all detections to class 1

            mask = masks[i].cpu().numpy()
            if mask.sum() < 16:
                continue

            bbox_xywh = boxes.xywh[i].cpu().tolist() if hasattr(boxes, 'xywh') else None

            evaluator.add_prediction(
                image_id=image_id,
                category_id=1,  # FastSAM class-agnostic → all cls=1
                mask=mask,
                score=conf_val,
            )

    # ── 运行 COCO 评估 | Run COCO Evaluation ──
    logger.log_info("eval", "=" * 60)
    logger.log_info("eval", "Class-Agnostic Evaluation (all GT+preds → cls=1)")
    logger.log_info("eval", "Measures FastSAM's segment-anything capability.")
    logger.log_info("eval", "")
    segm_results = evaluator.evaluate_class_agnostic(verbose=True)

    # ── 保存结果 | Save Results ──
    final_report = {
        "experiment": "FastSAM Zero-Shot Instance Segmentation Baseline",
        "note": "Class-agnostic evaluation (all GT+preds remapped to cls=1). Measures segment-anything capability.",
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "n_images": len(images),
        "segmentation": segm_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.log_info("done", f"Results saved to {out_dir / 'results.json'}")
    logger.log_info("done",
        f"Class-Agnostic AP={segm_results['AP']:.4f}, AP50={segm_results['AP50']:.4f}, "
        f"AP75={segm_results['AP75']:.4f}")


if __name__ == "__main__":
    main()
