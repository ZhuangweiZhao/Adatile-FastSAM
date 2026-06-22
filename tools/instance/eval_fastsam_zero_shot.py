#!/usr/bin/env python3
"""
C-01: FastSAM Zero-Shot Instance Segmentation Evaluation (Optimized)
=====================================================================

FastSAM predict mode on iSAID — evaluates mask quality (class-agnostic AP).

优化内容 | Optimizations:
  - 完整 AP50/AP75 计算 (PR curve + 11-point interpolation)
  - Mask 分辨率对齐 (FastSAM imgsz vs GT 原图)
  - Per-class 分解 (matched pred inherits GT class)
  - 高效 GT 渲染 (per-image, not per-instance)
  - 采样可视化 (5 张对比图)

用法 | Usage:
    python tools/instance/eval_fastsam_zero_shot.py \
        --src-root data/iSAID_processed --split val \
        --n-images 50 --output-dir runs/c01_isaid
"""

import sys, argparse, json, datetime, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

# Monkey-patch: bypass pandas dependency in ultralytics (only needed for export)
# pandas is only used in export_formats() which is irrelevant for inference
try:
    import pandas  # noqa: F401
except ImportError:
    import types, importlib.machinery
    _mock_pd = types.ModuleType('pandas')
    _mock_pd.__spec__ = importlib.machinery.ModuleSpec('pandas', None)
    _mock_pd.__version__ = '2.0.0'
    sys.modules['pandas'] = _mock_pd

    # Replace export_formats in autobackend (where it's actually called)
    # Must do BEFORE any ultralytics module loads _model_type
    class _DummySuffixes:
        Suffix = ['.pt', '.onnx', '.engine', '.xml', '.tflite', '.pb',
                   '.torchscript', '.mlmodel', '.mlpackage', '.h5',
                   '.tflite', '.edgetpu', '.tfjs', '.saved_model',
                   '.coreml', '.trt', '.paddle', '.ncnn']

    _dummy_export_formats = lambda: _DummySuffixes()

    # Patch both exporter module (source) and autobackend (consumer)
    import ultralytics.nn.autobackend as _ab_module
    _ab_module.export_formats = _dummy_export_formats

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.label_mapping import _ID_TO_NAME as ISAID_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou-thresh", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--output-dir", type=str, default="runs/c01_isaid")
    p.add_argument("--device", type=str,
                   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    return p.parse_args()


def render_gt_mask(ann, H, W):
    """Render single GT instance mask at full resolution."""
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    seg = ann.get("segmentation", [])
    if seg and not isinstance(seg, dict):
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, W - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, H - 1)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
    else:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, bw, bh = bbox
        mask[int(max(0, y)):int(min(H, y+bh)),
             int(max(0, x)):int(min(W, x+bw))] = 1
    return mask.astype(bool)


def compute_iou(mask_a, mask_b):
    """Binary mask IoU."""
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_ap(preds, gts, iou_thresh):
    """
    Compute AP at given IoU threshold (class-agnostic per-image).
    preds: [{"mask": bool_arr, "score": float}, ...]
    gts:   lazy-rendered [{"ann": dict, "cat_id": int, "H": int, "W": int}]
    """
    # Lazy GT mask renderer (to avoid OOM on 200+ instance images)
    # GT mask ???????????? mask
    _gt_mask_cache = {}

    def _get_gt_mask(gi):
        """Render and cache GT mask for instance gi."""
        if gi not in _gt_mask_cache:
            gt = gts[gi]
            _gt_mask_cache[gi] = render_gt_mask(gt["ann"], gt["H"], gt["W"])
        return _gt_mask_cache[gi]

    if not preds or not gts:
        return 0.0, {}

    preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    matched_gt = set()
    tp, fp = [], []
    per_cls_tp = {}
    n_gt = len(gts)

    for pred in preds:
        best_iou, best_gt = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred["mask"], _get_gt_mask(gi))
            if iou > best_iou:
                best_iou, best_gt = iou, gi

        if best_iou >= iou_thresh:
            tp.append(1); fp.append(0)
            matched_gt.add(best_gt)
            cat = gts[best_gt]["cat_id"]
            per_cls_tp[cat] = per_cls_tp.get(cat, 0) + 1
        else:
            tp.append(0); fp.append(1)

    # 11-point interpolated AP
    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    recall = tp_cum / max(n_gt, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask_r = recall >= t
        if np.any(mask_r):
            ap += float(np.max(precision[mask_r]))
    ap /= 11.0

    # Per-class AP (approximate)
    per_cls_ap = {}
    n_cls_map = {}
    for g in gts:
        n_cls_map[g["cat_id"]] = n_cls_map.get(g["cat_id"], 0) + 1
    for cat, n_cls_gt in n_cls_map.items():
        per_cls_ap[cat] = per_cls_tp.get(cat, 0) / max(n_cls_gt, 1)

    return ap, per_cls_ap


def resize_masks_to_original(masks_tensor, H_orig, W_orig, chunk_size=10):
    """
    Resize FastSAM masks from processing resolution to original image resolution.
    Uses CPU to avoid GPU OOM on high-res images (4000x4000+).
    """
    import torch
    import torch.nn.functional as F
    if masks_tensor is None or len(masks_tensor) == 0:
        return []
    h_mask, w_mask = masks_tensor.shape[1], masks_tensor.shape[2]
    if h_mask == H_orig and w_mask == W_orig:
        return [m.cpu().numpy().astype(bool) for m in masks_tensor]

    # Move to CPU for large upsampling | 搬到 CPU 避免 GPU OOM
    masks_np = masks_tensor.cpu().numpy()  # [N, h, w]
    import cv2
    result = []
    for i in range(len(masks_np)):
        mask_up = cv2.resize(masks_np[i].astype(np.float32), (W_orig, H_orig),
                             interpolation=cv2.INTER_LINEAR)
        result.append(mask_up > 0.5)
    return result


def main():
    args = parse_args()
    from fastsam import FastSAM

    src_root = Path(args.src_root)
    split = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("c01")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "c01.jsonl")))

    # Load GT annotations
    ann_file = src_root / split / "annotations" / f"instances_{split}.json"
    with open(ann_file) as f:
        coco = json.load(f)

    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    images = [img for img in coco["images"]
              if img_id_to_anns.get(img["id"])][:args.n_images]
    n_gt_total = sum(len(img_id_to_anns.get(img["id"], [])) for img in images)
    logger.log_info("c01/data", f"{len(images)} images, {n_gt_total} GT instances")

    # Load FastSAM
    logger.log_info("c01/model", f"Loading FastSAM-x (conf={args.conf}, iou={args.iou_thresh})")
    model = FastSAM(str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"))

    # Evaluation
    all_ap50, all_ap75 = [], []
    per_class_ap50 = {}  # cat_id -> [ap values]
    per_class_ap75 = {}
    per_class_gt_count = {}  # cat_id -> n_gt
    sample_images = []

    img_dir = src_root / split / "images"
    logger.log_info("c01/eval", f"Starting eval on {len(images)} images...")

    for img_info in tqdm(images, desc="  FastSAM eval"):
        img_path = str(img_dir / img_info["file_name"])
        if not Path(img_path).exists():
            continue

        from PIL import Image as PILImage
        with PILImage.open(img_path) as pil_img:
            W_orig, H_orig = pil_img.size

        # Check valid GT annotations (lazy mask rendering later)
        # Check valid GT annotations (lazy mask rendering later)
        gt_check = [a for a in img_id_to_anns.get(img_info['id'], [])
                     if a['category_id'] in ISAID_NAMES]
        if not gt_check:
            continue

        # FastSAM inference
        try:
            results = model(source=img_path, device=args.device,
                           imgsz=args.imgsz, conf=args.conf,
                           iou=args.iou_thresh, mode="predict", max_det=500)
        except Exception as e:
            logger.log_warn("c01/error", f"FastSAM failed on {img_info.get('file_name','?')}: {e}")
            continue

        if results is None or len(results) == 0:
            continue

        r = results[0]
        if not hasattr(r, 'masks') or r.masks is None:
            continue

        # Extract and resize FastSAM masks to original resolution
        # FastSAM ?? resize ? imgsz?mask ? imgsz ???
        # FastSAM internally resizes to imgsz=1024, upscale masks for accurate IoU
        masks_tensor = r.masks.data  # [N, h_imgsz, w_imgsz]
        mask_list = resize_masks_to_original(masks_tensor, H_orig, W_orig)

        # GT instances (lazy mask rendering to avoid OOM)
        # FastSAM internally resizes to imgsz=1024, upscale masks for accurate IoU
        gt_anns_raw = img_id_to_anns.get(img_info["id"], [])
        gts = []
        for ann in gt_anns_raw:
            cat_id = ann["category_id"]
            if cat_id not in ISAID_NAMES:
                continue
            gts.append({"ann": ann, "cat_id": cat_id, "H": H_orig, "W": W_orig})
            per_class_gt_count[cat_id] = per_class_gt_count.get(cat_id, 0) + 1

        if not gts:
            continue

        scores = r.boxes.conf.cpu().numpy() if hasattr(r, 'boxes') and r.boxes is not None else np.ones(len(mask_list))

        # Build pred list at FastSAM resolution
        preds = [{"mask": m, "score": float(s)}
                 for m, s in zip(mask_list, scores)]

        # Compute AP
        ap50, per_cls_ap50_i = compute_ap(preds, gts, 0.50)
        ap75, per_cls_ap75_i = compute_ap(preds, gts, 0.75)

        all_ap50.append(ap50)
        all_ap75.append(ap75)

        for cat_id, ap_val in per_cls_ap50_i.items():
            per_class_ap50.setdefault(cat_id, []).append(ap_val)
        for cat_id, ap_val in per_cls_ap75_i.items():
            per_class_ap75.setdefault(cat_id, []).append(ap_val)

        logger.log_info("c01/per_img",
                       f"  {img_info.get('file_name','?')}: "
                       f"{len(preds)} preds {len(gts)} GT -> AP50={ap50:.4f} AP75={ap75:.4f}")

        # Collect first 5 images for visualization
        if len(sample_images) < 5:
            sample_images.append({
                "name": img_info.get("file_name", "?"),
                "preds": preds,
                "gts": gts,
                "H": H_orig, "W": W_orig,
            })

    # Summary
    logger.log_info("c01/summary", f"\n{'='*70}")
    logger.log_info("c01/summary", f"  FastSAM Zero-Shot on iSAID ({split})")
    logger.log_info("c01/summary", f"  Images: {len(images)} | GT instances: {n_gt_total}")
    logger.log_info("c01/summary", f"  conf={args.conf} iou_thresh={args.iou_thresh} imgsz={args.imgsz}")
    logger.log_info("c01/summary", f"{'='*70}")
    logger.log_info("c01/summary",
                   f"  mAP@50 = {np.mean(all_ap50)*100:.2f}% (+-{np.std(all_ap50)*100:.2f}%) [{len(all_ap50)} imgs]")
    logger.log_info("c01/summary",
                   f"  mAP@75 = {np.mean(all_ap75)*100:.2f}% (+-{np.std(all_ap75)*100:.2f}%) [{len(all_ap75)} imgs]")

    # Per-class
    logger.log_info("c01/per_class", "\n  Per-class AP@50:")
    for cat_id in sorted(per_class_ap50.keys()):
        vals = per_class_ap50[cat_id]
        name = ISAID_NAMES.get(cat_id, f"cls_{cat_id}")
        n_gt_cls = per_class_gt_count.get(cat_id, 0)
        logger.log_info("c01/per_class",
                       f"    {name:<20s} AP50={np.mean(vals)*100:.1f}% (n_gt={n_gt_cls})")

    # Visualization
    if sample_images:
        logger.log_info("c01/viz",f"  Saving {len(sample_images)} sample visualizations...")
        for si, sample in enumerate(sample_images):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            H, W = sample["H"], sample["W"]

            # GT overlay
            ax = axes[0]
            gt_canvas = np.zeros((H, W, 3), dtype=np.float32)
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            for gi, gt in enumerate(sample["gts"]):
                c = colors[gi % 20, :3]
                gt_mask = render_gt_mask(gt["ann"], gt["H"], gt["W"])
                gt_canvas[gt_mask, 0] = c[0]
                gt_canvas[gt_mask, 1] = c[1]
                gt_canvas[gt_mask, 2] = c[2]
            ax.imshow(gt_canvas)
            ax.set_title(f"GT ({len(sample['gts'])} instances)")
            ax.axis("off")

            # Pred overlay
            ax = axes[1]
            pred_canvas = np.zeros((H, W, 3), dtype=np.float32)
            for pi, pred in enumerate(sorted(sample["preds"], key=lambda p: -p["score"])[:50]):
                c = colors[pi % 20, :3]
                m = pred["mask"]
                pred_canvas[m, 0] = c[0]
                pred_canvas[m, 1] = c[1]
                pred_canvas[m, 2] = c[2]
            ax.imshow(pred_canvas)
            ax.set_title(f"FastSAM Preds ({len(sample['preds'])} masks, top 50 shown)")
            ax.axis("off")

            # Overlap
            ax = axes[2]
            overlap = np.zeros((H, W, 3), dtype=np.float32)
            gt_union = np.zeros((H, W), dtype=bool)
            pred_union = np.zeros((H, W), dtype=bool)
            for gt in sample["gts"]:
                gt_mask = render_gt_mask(gt["ann"], gt["H"], gt["W"])
                gt_union |= gt_mask
            for pred in sample["preds"]:
                pred_union |= pred["mask"]
            overlap[gt_union & ~pred_union] = [1, 0, 0]  # GT only: red
            overlap[~gt_union & pred_union] = [0, 0, 1]  # Pred only: blue
            overlap[gt_union & pred_union] = [0, 1, 0]   # Overlap: green
            ax.imshow(overlap)
            ax.set_title("Overlap (red=GT only, blue=pred only, green=overlap)")
            ax.axis("off")

            plt.suptitle(f"FastSAM Zero-Shot on {sample['name']}")
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{si+1}_{sample['name']}.png", dpi=100,
                       bbox_inches="tight")
            plt.close()
        logger.log_info("c01/viz", f"  Visualizations saved to {output_dir}/")

    # Save JSON report
    summary = {
        "dataset": "iSAID",
        "split": split,
        "n_images": len(images),
        "n_gt_total": n_gt_total,
        "config": {"conf": args.conf, "iou_thresh": args.iou_thresh, "imgsz": args.imgsz},
        "mAP50": float(np.mean(all_ap50)) if all_ap50 else 0.0,
        "mAP75": float(np.mean(all_ap75)) if all_ap75 else 0.0,
        "mAP50_std": float(np.std(all_ap50)) if all_ap50 else 0.0,
        "per_class_AP50": {str(k): float(np.mean(v)) for k, v in per_class_ap50.items()},
    }
    with open(output_dir / "fastsam_zero_shot.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("c01/done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
