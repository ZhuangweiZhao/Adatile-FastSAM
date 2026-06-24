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

用法 | Usage::
    python tools/instance/eval_fastsam_zero_shot.py \
        --src-root data/iSAID_processed --split val \
        --n-images 50 --output-dir runs/c01_isaid
"""

import sys, argparse, json, datetime, os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))
GREEN = '\033[92m'
RESET = '\033[0m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
DEBUG = False  # 设为 False 关闭调试输出 | Set False to disable debug output


def _dbg(tag: str, *args):
    """统一调试输出 | Unified debug output."""
    if DEBUG:
        print(f"{CYAN}[DEBUG {tag}]{RESET}", *args)

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
    """
    解析命令行参数 | Parse command-line arguments.

    :return: Parsed argument namespace
    :rtype: argparse.Namespace
    """
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
    p.add_argument("--max-det", type=int, default=100,
                   help="FastSAM 最大检测数 | max detections per image (default 100)")
    return p.parse_args()


def render_gt_mask(ann, H, W):
    """
    Render single GT instance mask at full resolution | 渲染单个 GT 实例掩码。

    优化 | Optimization:
        - 所有 polygon 合并为一次 fillPoly 调用，减少 Python→C 开销
          Batched fillPoly — single C call for all polygons, eliminates per-poly overhead.
        - RLE 格式直接解码，比 polygon 填充快 5-10×
          RLE direct decode, 5-10× faster than polygon fill.

    :param ann: COCO annotation dict with "segmentation" or "bbox" key
    :type ann: dict
    :param H: Image height in pixels
    :type H: int
    :param W: Image width in pixels
    :type W: int
    :return: Boolean mask array of shape (H, W)
    :rtype: numpy.ndarray
    """
    import cv2

    seg = ann.get("segmentation", [])

    # ── RLE 格式：直接解码 | RLE format: direct decode ──
    if isinstance(seg, dict):
        # COCO RLE: {"size": [H, W], "counts": "..."} 或 {"counts": [...]}
        # COCO RLE: {"size": [H, W], "counts": "..."} or {"counts": [...]}
        try:
            from pycocotools import mask as coco_mask
            rle = coco_mask.frPyObjects(seg, H, W)
            if isinstance(rle, list):
                rle = coco_mask.merge(rle)
            m = coco_mask.decode(rle).astype(bool)
            _dbg("render_gt_mask", f"RLE decode: shape={m.shape}, fg={m.sum()}")
            return m
        except ImportError:
            pass  # fallthrough to bbox | 回退到 bbox

    mask = np.zeros((H, W), dtype=np.uint8)

    # ── Polygon 格式：批量收集 → 一次 fillPoly | Batch collect → single fillPoly ──
    if seg and not isinstance(seg, dict):
        # 归一化：COCO polygon 可能是 [[x1,y1,x2,y2,...]] 或 [[[x1,y1],[x2,y2],...]]
        # COCO spec: seg is list[list[float]] (flattened) or list[list[list[float]]]
        polys = seg if isinstance(seg[0], list) else [seg]

        # 批量收集所有有效 polygon 的 numpy 数组 | batch-collect valid polygon arrays
        batch_polys = []
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            # clip 到图像边界 | clip to image bounds
            np.clip(pts[:, :, 0], 0, W - 1, out=pts[:, :, 0])
            np.clip(pts[:, :, 1], 0, H - 1, out=pts[:, :, 1])
            if len(pts) >= 3:
                batch_polys.append(pts)

        if batch_polys:
            _dbg("render_gt_mask", f"polygon batch: {len(batch_polys)} polys → single fillPoly, image={W}x{H}")
            cv2.fillPoly(mask, batch_polys, 1)  # 一次 C 调用 | single C call
    else:
        # ── Bbox 回退 | Bbox fallback ──
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, bw, bh = bbox
        _dbg("render_gt_mask", f"bbox fallback: x={x:.0f} y={y:.0f} w={bw:.0f} h={bh:.0f}, image={W}x{H}")
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W, x + bw)), int(min(H, y + bh))
        mask[y1:y2, x1:x2] = 1

    fg_px = mask.sum()
    _dbg("render_gt_mask", f"  → mask fg_pixels={fg_px} ({fg_px/(H*W)*100:.2f}%)")
    return mask.astype(bool)


def compute_iou(mask_a, mask_b):
    """
    Binary mask IoU | 二值掩码 IoU。

    :param mask_a: First boolean mask
    :type mask_a: numpy.ndarray
    :param mask_b: Second boolean mask
    :type mask_b: numpy.ndarray
    :return: IoU score in range [0, 1]
    :rtype: float
    """
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    iou = float(inter / union) if union > 0 else 0.0
    if DEBUG and iou > 0.3:  # 只打印有意义的匹配 | only log meaningful overlaps
        _dbg("compute_iou", f"inter={inter} union={union} iou={iou:.4f}")
    return iou


def compute_ap(preds, gts, iou_thresh):
    """
    Compute AP at given IoU threshold (class-agnostic per-image) | 计算给定 IoU 阈值的 AP。

    :param preds: List of dicts with "mask" (bool array) and "score" (float) keys
    :type preds: list[dict]
    :param gts: List of dicts with "ann" (COCO annotation), "cat_id" (int), "H" (int), "W" (int)
    :type gts: list[dict]
    :param iou_thresh: IoU threshold for matching
    :type iou_thresh: float
    :return: Tuple of (AP score, per-class AP dict mapping cat_id → AP)
    :rtype: tuple[float, dict[int, float]]
    """
    # 直接读取预渲染的 mask（main 中已渲染好）| use pre-rendered mask from main
    def _get_gt_mask(gi):
        """Return pre-rendered GT mask (rendered once in main)."""
        return gts[gi]["mask"]

    _dbg("compute_ap", f"START: {len(preds)} preds, {len(gts)} GTs, iou_thresh={iou_thresh}")
    if DEBUG:
        # 打印 GT 类别分布 | print GT class distribution
        gt_cls_counts = {}
        for g in gts:
            gt_cls_counts[g["cat_id"]] = gt_cls_counts.get(g["cat_id"], 0) + 1
        _dbg("compute_ap", f"  GT class distribution: {dict(sorted(gt_cls_counts.items()))}")

    if not preds or not gts:
        return 0.0, {}

    preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    if DEBUG:
        top_scores = [f"{p['score']:.4f}" for p in preds[:5]]
        _dbg("compute_ap", f"  top-5 pred scores: {top_scores}")
    tp, fp = [], []
    per_cls_match_rate = {}
    n_gt = len(gts)

    # ── 向量化 IoU @ 1024 尺度，仅堆叠 GT mask ──
    # Vectorized IoU @ 1024 scale, GT stack only (~100-200 MB, released after return)
    h, w = gts[0]["mask"].shape
    gt_masks = np.zeros((n_gt, h, w), dtype=bool)
    gt_areas = np.zeros(n_gt, dtype=np.int64)
    for gi, gt in enumerate(gts):
        m = gt["mask"]
        gt_masks[gi] = m
        gt_areas[gi] = m.sum()

    matched_mask = np.zeros(n_gt, dtype=bool)

    for pi, pred in enumerate(preds):
        pm = pred["mask"]
        p_area = pm.sum()

        # 一次 numpy 调用完成与所有 GT 的 IoU | single numpy call for IoU vs ALL GTs
        inter = (gt_masks & pm).sum(axis=(1, 2)).astype(np.float64)
        union = gt_areas.astype(np.float64) + float(p_area) - inter
        ious = np.zeros(n_gt, dtype=np.float64)
        valid = union > 0
        ious[valid] = inter[valid] / union[valid]
        ious[matched_mask] = -1.0

        best_gi = int(ious.argmax())
        best_iou = float(ious[best_gi])

        if best_iou >= iou_thresh:
            tp.append(1); fp.append(0)
            matched_mask[best_gi] = True
            cat = gts[best_gi]["cat_id"]
            per_cls_match_rate[cat] = per_cls_match_rate.get(cat, 0) + 1
            cat_name = ISAID_NAMES.get(cat, f"cls_{cat}")
            if pi < 20:
                _dbg("compute_ap",
                     f"  pred[{pi}] score={pred['score']:.4f} → "
                     f"GT[{best_gi}] cat={cat_name}({cat}) "
                     f"best_iou={best_iou:.3f} ✓ TP")
        else:
            tp.append(0); fp.append(1)
            if pi < 20:
                _dbg("compute_ap",
                     f"  pred[{pi}] score={pred['score']:.4f} → "
                     f"best_iou={best_iou:.3f} < {iou_thresh} ✗ FP")

    # 11-point interpolated AP
    n_matched = int(matched_mask.sum())
    _dbg("compute_ap", f"  Matching done: {sum(tp)} TP / {sum(fp)} FP, "
         f"{n_matched}/{n_gt} GTs matched, "
         f"{n_gt - n_matched} unmatched")
    if DEBUG:
        unmatched = [gi for gi in range(n_gt) if not matched_mask[gi]]
        if unmatched:
            unmatched_cats = {}
            for gi in unmatched:
                c = gts[gi]["cat_id"]
                unmatched_cats[c] = unmatched_cats.get(c, 0) + 1
            _dbg("compute_ap", f"  Unmatched GTs by class: "
                 f"{ {ISAID_NAMES.get(c,f'cls_{c}'): n for c, n in sorted(unmatched_cats.items())} }")

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

    # Per-class match rate (TP/GT = Recall, NOT true AP) | 每类匹配率
    n_cls_map = {}
    for g in gts:
        n_cls_map[g["cat_id"]] = n_cls_map.get(g["cat_id"], 0) + 1
    for cat, n_cls_gt in n_cls_map.items():
        per_cls_match_rate[cat] = per_cls_match_rate.get(cat, 0) / max(n_cls_gt, 1)

    _dbg("compute_ap", f"END: AP={ap:.4f}, matched={n_matched}/{n_gt} GTs, "
         f"TP={sum(tp)}, FP={sum(fp)}")
    if DEBUG:
        for cat, rate in sorted(per_cls_match_rate.items()):
            n_cls = n_cls_map.get(cat, 0)
            name = ISAID_NAMES.get(cat, f"cls_{cat}")
            _dbg("compute_ap", f"  per-class: {name}(cat={cat}) GT={n_cls} match_rate={rate:.4f}")

    return ap, per_cls_match_rate



def compute_recall(preds, gts, iou_thresh):
    """
    Compute class-agnostic recall | 计算类别无关的召回率。

    For each GT: is there at least one pred mask with IoU >= threshold?
    This is the RECALL CEILING for any downstream classifier.

    :param preds: List of dicts with "mask" (bool array) and "score" (float) keys
    :type preds: list[dict]
    :param gts: List of dicts with "ann", "cat_id", "H", "W" keys
    :type gts: list[dict]
    :param iou_thresh: IoU threshold for matching
    :type iou_thresh: float
    :return: Tuple of (overall_recall, per_cls_recall dict, matched_gt set)
    :rtype: tuple[float, dict[int, float], set[int]]
    """
    _dbg("compute_recall", f"START: {len(preds)} preds, {len(gts)} GTs, iou_thresh={iou_thresh}")

    if not preds or not gts:
        return 0.0, {}, set()

    # ── Recall @ 1024 尺度：每 GT 找最佳 pred，不堆叠 ──
    # Recall @ 1024 scale: per-GT best pred match, no mass stacking
    matched_gt = set()
    for gi in range(len(gts)):
        gt_m = gts[gi]["mask"]
        gt_a = gt_m.sum()

        best_iou = 0.0
        best_pi = -1
        for pi, pred in enumerate(preds):
            pm = pred["mask"]
            inter = (gt_m & pm).sum()
            union = gt_a + int(pm.sum()) - inter
            iou = float(inter / union) if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_pi = iou, pi

        if best_iou >= iou_thresh:
            matched_gt.add(gi)
            if gi < 10:
                cat = gts[gi]["cat_id"]
                cat_name = ISAID_NAMES.get(cat, f"cls_{cat}")
                _dbg("compute_recall",
                     f"  GT[{gi}] cat={cat_name}({cat}) → "
                     f"pred[{best_pi}] best_iou={best_iou:.3f} ✓ matched")
        else:
            if gi < 10 or best_iou == 0.0:
                cat = gts[gi]["cat_id"]
                cat_name = ISAID_NAMES.get(cat, f"cls_{cat}")
                _dbg("compute_recall",
                     f"  GT[{gi}] cat={cat_name}({cat}) → "
                     f"best_iou={best_iou:.3f} < {iou_thresh} ✗ unmatched")

    overall_recall = len(matched_gt) / max(len(gts), 1)

    # Per-class recall
    per_cls_recall = {}
    n_cls_map = {}
    for g in gts:
        n_cls_map[g["cat_id"]] = n_cls_map.get(g["cat_id"], 0) + 1
    for cat, n_gt_cls in n_cls_map.items():
        matched_cls = sum(1 for gi in matched_gt if gts[gi]["cat_id"] == cat)
        per_cls_recall[cat] = matched_cls / max(n_gt_cls, 1)

    _dbg("compute_recall", f"END: overall_recall={overall_recall:.4f} ({len(matched_gt)}/{len(gts)} GTs matched)")
    if DEBUG:
        for cat, rec_val in sorted(per_cls_recall.items()):
            n_cls = n_cls_map.get(cat, 0)
            matched_cls = sum(1 for gi in matched_gt if gts[gi]["cat_id"] == cat)
            name = ISAID_NAMES.get(cat, f"cls_{cat}")
            _dbg("compute_recall", f"  per-class: {name}(cat={cat}) GT={n_cls} matched={matched_cls} Recall={rec_val:.4f}")

    return overall_recall, per_cls_recall, matched_gt


def main():
    """
    C-01 主入口：FastSAM Zero-Shot 实例分割评估 | C-01 main entry: FastSAM zero-shot evaluation.

    评估 FastSAM predict mode 在 iSAID 上的 mask 质量 (class-agnostic AP/Recall)。
    """
    args = parse_args()
    _dbg("main", f"Config: src_root={args.src_root}, split={args.split}, "
         f"n_images={args.n_images}, conf={args.conf}, iou_thresh={args.iou_thresh}, "
         f"imgsz={args.imgsz}, device={args.device}")
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
    _dbg("main", f"COCO loaded: {len(coco['images'])} total images, "
         f"{len(coco['annotations'])} total annotations, "
         f"{len(coco['categories'])} categories")
    _dbg("main", f"Selected: {len(images)} images (capped at --n-images={args.n_images}), "
         f"{n_gt_total} GT instances")
    if DEBUG:
        # 打印类别映射 | print category mapping
        for cat in sorted(coco.get("categories", []), key=lambda c: c["id"]):
            name = ISAID_NAMES.get(cat["id"], "???")
            _dbg("main", f"  category: id={cat['id']} name={cat.get('name','?')} → ISAID={name}")

    # Load FastSAM
    logger.log_info("c01/model", f"Loading FastSAM-x (conf={args.conf}, iou={args.iou_thresh})")
    model = FastSAM(str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"))

    # Evaluation
    all_ap50, all_ap75 = [], []
    all_recall50, all_recall75 = [], []
    per_class_match50 = {}  # cat_id -> [match_rate values] — NOT true AP, per-image TP/GT
    per_class_match75 = {}
    per_class_rec50 = {}  # cat_id -> [recall values]
    per_class_rec75 = {}
    per_class_gt_count = {}  # cat_id -> n_gt
    sample_images = []

    img_dir = src_root / split / "images"
    logger.log_info("c01/eval", f"Starting eval on {len(images)} images...")

    for img_info in tqdm(images, desc="FastSAM eval"):
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
                           iou=args.iou_thresh, mode="predict", max_det=args.max_det)
        except Exception as e:
            logger.log_warn("c01/error", f"FastSAM failed on {img_info.get('file_name','?')}: {e}")
            continue

        if results is None or len(results) == 0:
            continue

        r = results[0]
        if not hasattr(r, 'masks') or r.masks is None:
            _dbg("main", f"  {img_info.get('file_name','?')}: NO masks in FastSAM result — skipping")
            continue

        # Pred masks 保持 FastSAM 原生分辨率 (1024) | keep preds at FastSAM native resolution
        masks_tensor = r.masks.data  # [N, 1024, 1024] — NO upscale to 4000!
        _dbg("main", f"--- {img_info.get('file_name','?')} ---")
        _dbg("main", f"  original={W_orig}x{H_orig}, FastSAM imgsz={args.imgsz}")
        _dbg("main", f"  masks_tensor.shape={masks_tensor.shape}, "
             f"dtype={masks_tensor.dtype}, device={masks_tensor.device}")
        if hasattr(r, 'boxes') and r.boxes is not None:
            _dbg("main", f"  boxes: {len(r.boxes)} detections, "
                 f"conf range=[{r.boxes.conf.min():.3f}, {r.boxes.conf.max():.3f}]")

        # Pred masks → CPU numpy (keep at FastSAM native resolution, may be non-square)
        # pred mask 保持 FastSAM 原生分辨率（可能非正方形）
        mask_list = [m.cpu().numpy().astype(bool) for m in masks_tensor]
        pred_h, pred_w = masks_tensor.shape[1], masks_tensor.shape[2]  # 如 1024×736
        _dbg("main", f"  pred mask native shape: {pred_h}x{pred_w}")

        # GT instances: 全分辨率渲染 → 缩放到 pred 分辨率用于 IoU
        # GT: render at full res → downsample to pred resolution for IoU
        gt_anns_raw = img_id_to_anns.get(img_info["id"], [])
        gts = []
        gts_fullres = []  # 保留全分辨率用于可视化 | keep full-res for visualization
        for ann in gt_anns_raw:
            cat_id = ann["category_id"]
            if cat_id not in ISAID_NAMES:
                continue
            mask_full = render_gt_mask(ann, H_orig, W_orig)  # 全分辨率渲染 | full-res render
            # 缩放到 pred 原生分辨率 | downsample to pred native resolution
            mask_ds = cv2.resize(mask_full.astype(np.float32),
                                 (pred_w, pred_h),
                                 interpolation=cv2.INTER_LINEAR) > 0.5
            gts.append({"mask": mask_ds, "cat_id": cat_id})
            gts_fullres.append({"ann": ann, "cat_id": cat_id, "H": H_orig, "W": W_orig, "mask": mask_full})
            per_class_gt_count[cat_id] = per_class_gt_count.get(cat_id, 0) + 1

        if not gts:
            continue

        scores = r.boxes.conf.cpu().numpy() if hasattr(r, 'boxes') and r.boxes is not None else np.ones(len(mask_list))

        # Build pred list (all at 1024) | pred 全在 1024 尺度
        preds = [{"mask": m, "score": float(s)}
                 for m, s in zip(mask_list, scores)]
        top5_scores = [f"{s:.3f}" for s in scores[:5]]
        _dbg("main", f"  built: {len(preds)} preds (top5_scores={top5_scores}), "
             f"{len(gts)} valid GTs in ISAID_NAMES")
        if DEBUG:
            gt_cls = {}
            for g in gts:
                gt_cls[g["cat_id"]] = gt_cls.get(g["cat_id"], 0) + 1
            _dbg("main", f"  GT classes: {dict(sorted(gt_cls.items()))}")

        # Compute AP and Recall
        ap50, per_cls_ap50_i = compute_ap(preds, gts, 0.50)
        ap75, per_cls_ap75_i = compute_ap(preds, gts, 0.75)
        rec50, per_cls_rec50_i, _ = compute_recall(preds, gts, 0.50)
        rec75, per_cls_rec75_i, _ = compute_recall(preds, gts, 0.75)

        all_ap50.append(ap50)
        all_ap75.append(ap75)
        all_recall50.append(rec50)
        all_recall75.append(rec75)

        for cat_id, rate in per_cls_ap50_i.items():
            per_class_match50.setdefault(cat_id, []).append(rate)
        for cat_id, rate in per_cls_ap75_i.items():
            per_class_match75.setdefault(cat_id, []).append(rate)
        for cat_id, rec_val in per_cls_rec50_i.items():
            per_class_rec50.setdefault(cat_id, []).append(rec_val)
        for cat_id, rec_val in per_cls_rec75_i.items():
            per_class_rec75.setdefault(cat_id, []).append(rec_val)

        logger.log_info("c01/per_img",
                       f"  {img_info.get('file_name','?')}: "
                       f"{len(preds)} preds {len(gts)} GT -> "
                       f"R@50={rec50:.3f} R@75={rec75:.3f} AP50={ap50:.4f}")

        # Collect first 5 images for visualization (use full-res GTs) | 用全分辨率 GT 做可视化
        if len(sample_images) < 5:
            sample_images.append({
                "name": img_info.get("file_name", "?"),
                "preds": mask_list,  # raw bool arrays at 1024
                "gts": gts_fullres,   # full-res masks for viz
                "H": H_orig, "W": W_orig,
            })

    # Summary — RECALL as primary metric (few-shot ceiling analysis)
    logger.log_info("c01/summary", "\n" + "=" * 70)
    logger.log_info("c01/summary", f"  FastSAM Zero-Shot Proposal Quality -- iSAID ({split})")
    logger.log_info("c01/summary", f"  Images: {len(images)} | GT instances: {n_gt_total}")
    logger.log_info("c01/summary", f"  conf={args.conf} iou_thresh={args.iou_thresh} imgsz={args.imgsz} max_det={args.max_det}")
    logger.log_info("c01/summary", "  NOTE: AP is per-image averaged (approximate, NOT COCO-standard global PR)")
    logger.log_info("c01/summary", "=" * 70)
    logger.log_info("c01/summary",
                   f"  mRecall@50 = {np.mean(all_recall50)*100:.1f}% (+-{np.std(all_recall50)*100:.1f}%) "
                   f"[Recall Ceiling: max possible downstream AP]")
    logger.log_info("c01/summary",
                   f"  mRecall@75 = {np.mean(all_recall75)*100:.1f}% (+-{np.std(all_recall75)*100:.1f}%)")
    if all_ap50:
        logger.log_info("c01/summary",
                   f"  Approx AP@50 = {np.mean(all_ap50)*100:.1f}% (gap to Recall: {np.mean(all_ap50)/max(np.mean(all_recall50),1e-8):.1f}x)")
    if all_ap75:
        logger.log_info("c01/summary",
                   f"  Approx AP@75 = {np.mean(all_ap75)*100:.1f}%")

    # Per-class Recall (primary) + Match Rate
    logger.log_info("c01/per_class", "\n  Per-class Recall@50 (Recall Ceiling) + Match Rate:")
    logger.log_info("c01/per_class",
                   f'    {"Class":<20s} {"#GT":>5} {"Rec@50":>7} {"Match@50":>8} {"Rec@75":>7}')
    logger.log_info("c01/per_class", "    " + "-"*60)
    for cat_id in sorted(per_class_rec50.keys()):
        rec50_cls = np.mean(per_class_rec50[cat_id])
        rec75_cls = np.mean(per_class_rec75.get(cat_id, [0])) if cat_id in per_class_rec75 else 0.0
        match50_cls = np.mean(per_class_match50[cat_id]) if cat_id in per_class_match50 else 0.0
        name = ISAID_NAMES.get(cat_id, f"cls_{cat_id}")
        n_gt_cls = per_class_gt_count.get(cat_id, 0)
        logger.log_info("c01/per_class",
                       f"    {name:<20s} {n_gt_cls:>5} {rec50_cls*100:>6.1f}% "
                       f"{match50_cls*100:>7.1f}% {rec75_cls*100:>6.1f}%")


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
                gt_mask = gt["mask"]  # 预渲染的 mask | pre-rendered
                gt_canvas[gt_mask, 0] = c[0]
                gt_canvas[gt_mask, 1] = c[1]
                gt_canvas[gt_mask, 2] = c[2]
            ax.imshow(gt_canvas)
            ax.set_title(f"GT ({len(sample['gts'])} instances)")
            ax.axis("off")

            # Pred overlay (preds are raw 1024 masks → upsample to full res for viz)
            ax = axes[1]
            pred_canvas = np.zeros((H, W, 3), dtype=np.float32)
            for pi, pm in enumerate(sample["preds"][:50]):
                c = colors[pi % 20, :3]
                # Upsample 1024 → full res for visualization | 上采样到全分辨率用于可视化
                pm_full = cv2.resize(pm.astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_LINEAR) > 0.5
                pred_canvas[pm_full, 0] = c[0]
                pred_canvas[pm_full, 1] = c[1]
                pred_canvas[pm_full, 2] = c[2]
            ax.imshow(pred_canvas)
            ax.set_title(f"FastSAM Preds ({len(sample['preds'])} masks, top 50 shown)")
            ax.axis("off")

            # Overlap
            ax = axes[2]
            overlap = np.zeros((H, W, 3), dtype=np.float32)
            gt_union = np.zeros((H, W), dtype=bool)
            pred_union = np.zeros((H, W), dtype=bool)
            for gt in sample["gts"]:
                gt_union |= gt["mask"]  # 预渲染的 mask | pre-rendered
            for pm in sample["preds"]:
                pm_full = cv2.resize(pm.astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_LINEAR) > 0.5
                pred_union |= pm_full
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
        "config": {"conf": args.conf, "iou_thresh": args.iou_thresh, "imgsz": args.imgsz, "max_det": args.max_det},
        "note": "AP is per-image averaged (approximate, NOT COCO global PR)",
        # Primary: Recall (proposal quality, upper bound for any downstream method)
        "mRecall50": float(np.mean(all_recall50)) if all_recall50 else 0.0,
        "mRecall50_std": float(np.std(all_recall50)) if all_recall50 else 0.0,
        "mRecall75": float(np.mean(all_recall75)) if all_recall75 else 0.0,
        # Secondary: Approx AP (per-image avg, NOT COCO standard)
        "approx_AP50": float(np.mean(all_ap50)) if all_ap50 else 0.0,
        "approx_AP50_std": float(np.std(all_ap50)) if all_ap50 else 0.0,
        "approx_AP75": float(np.mean(all_ap75)) if all_ap75 else 0.0,
        # Per-class (recall + match rate)
        "per_class_Recall50": {str(k): float(np.mean(v)) for k, v in per_class_rec50.items()},
        "per_class_Recall75": {str(k): float(np.mean(v)) for k, v in per_class_rec75.items()},
        "per_class_MatchRate50": {str(k): float(np.mean(v)) for k, v in per_class_match50.items()},
        "per_class_gt_count": {str(k): v for k, v in per_class_gt_count.items()},
    }
    with open(output_dir / "fastsam_zero_shot.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("c01/done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
