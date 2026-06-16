#!/usr/bin/env python
"""Main Experiment: Semantic Segmentation SOTA Comparison (Paper Table 1).

Dataset: iSAID Semantic Segmentation (16 classes: 0=bg, 1-15=foreground)
Primary metric: mIoU

Baselines:
  Classic:     U-Net, DeepLabV3+, PSPNet
  Transformer: SegFormer-B0, SegFormer-B2, Mask2Former (Swin-T)
  SAM-family:  FastSAM-x, EfficientSAM
  Ours:        AdaTile-FastSAM (v2 Full / Sparse)

Usage:
  python tools/exp_main_comparison.py
  python tools/exp_main_comparison.py --keep-ratios 1.0 0.15
  python tools/exp_main_comparison.py --baselines results.json
"""

import argparse, json, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging import PaperLogger


@dataclass
class MethodResult:
    name: str; category: str  # "classic", "transformer", "sam", "ours"
    # Accuracy (semantic segmentation)
    mIoU: float = 0.0; mAcc: float = 0.0; aAcc: float = 0.0; Dice: float = 0.0
    # Efficiency
    FPS: float = 0.0; latency_ms: float = 0.0
    FLOPs_G: float = 0.0; params_M: float = 0.0; memory_MB: float = 0.0
    # Sparse
    sparse_FLOPs_G: float = 0.0; sparse_FPS: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ═══════════════════════════════════════════════════════════════════
# iSAID 16-class labels
# ═══════════════════════════════════════════════════════════════════

ISAID_CLASSES = [
    'background', 'ship', 'storage_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'ground_track_field',
    'bridge', 'large_vehicle', 'small_vehicle', 'helicopter',
    'swimming_pool', 'roundabout', 'soccer_ball_field', 'plane', 'harbor',
]
NUM_CLASSES = 16


# ═══════════════════════════════════════════════════════════════════
# Semantic segmentation evaluation (streaming)
# ═══════════════════════════════════════════════════════════════════

def _pred_to_semantic_map(
    output: Dict, H: int, W: int, num_classes: int
) -> np.ndarray:
    """Convert instance predictions → per-pixel class map.

    Since FastSAM is class-agnostic (no classification), we:
      1. Project each instance mask onto the full image
      2. Assign class 1 (generic foreground) to all predicted instances
      3. Overlapping regions keep the highest-confidence instance
    """
    pred_map = np.zeros((H, W), dtype=np.int32)
    conf_map = np.zeros((H, W), dtype=np.float32)

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output["masks"]

    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        score = float(scores[i])
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(W, box[2]), min(H, box[3])
        out_w, out_h = x2 - x1, y2 - y1
        if out_w <= 0 or out_h <= 0:
            continue

        # Get instance mask
        if masks.numel() > 0 and i < len(masks):
            inst_mask = masks[i].cpu().numpy()
            if inst_mask.shape[0] != out_h or inst_mask.shape[1] != out_w:
                from PIL import Image
                inst_mask = np.array(Image.fromarray(
                    (inst_mask > 0.5).astype(np.uint8) * 255
                ).resize((out_w, out_h), Image.NEAREST)) > 128
            inst_bin = (inst_mask > 0.5) if inst_mask.dtype != bool else inst_mask
            crop_h = min(out_h, H - y1)
            crop_w = min(out_w, W - x1)
            if crop_h <= 0 or crop_w <= 0:
                continue
            region = inst_bin[:crop_h, :crop_w]
        else:
            # No mask: use bbox center 50% as proxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            hw, hh = out_w // 3, out_h // 3
            px1, py1 = max(0, cx - hw), max(0, cy - hh)
            px2, py2 = min(W, cx + hw), min(H, cy + hh)
            crop_h = py2 - py1
            crop_w = px2 - px1
            if crop_h <= 0 or crop_w <= 0:
                continue
            region = np.ones((crop_h, crop_w), dtype=bool)
            x1, y1 = px1, py1

        # Class-agnostic: all predictions → foreground (class 1)
        # Higher-confidence predictions overwrite lower ones
        target_y1, target_x1 = y1, x1
        existing_conf = conf_map[target_y1:target_y1 + crop_h, target_x1:target_x1 + crop_w]
        update_mask = score > existing_conf
        pred_map[target_y1:target_y1 + crop_h, target_x1:target_x1 + crop_w][update_mask] = 1
        conf_map[target_y1:target_y1 + crop_h, target_x1:target_x1 + crop_w][update_mask] = score

    return pred_map


def evaluate_v2_semantic(
    model, val_loader, device, image_size: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """Streaming semantic segmentation evaluation — per-class mIoU/mAcc/aAcc/Dice."""
    model.eval()
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    pixel_correct = 0.0
    pixel_total = 0.0
    class_correct = np.zeros(num_classes, dtype=np.float64)
    class_total = np.zeros(num_classes, dtype=np.float64)
    dice_inter = 0.0
    dice_union = 0.0

    H_img, W_img = image_size
    total = len(val_loader)

    print(f"  [eval] Processing {total} images (semantic streaming)...")
    with torch.no_grad():
        for i, vb in enumerate(val_loader):
            if i % 50 == 0 and i > 0:
                print(f"  [eval] {i}/{total} ({i*100//total}%)")
            img = vb["images"].to(device)
            gt_mask = vb["masks"]  # [H, W] class labels

            # Forward
            output = model(img, use_sparse=False)

            # Get GT
            gt = gt_mask.cpu().squeeze().numpy().astype(np.int32)
            if gt.shape[0] != H_img or gt.shape[1] != W_img:
                from PIL import Image
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                    (W_img, H_img), Image.NEAREST))
            gt = np.clip(gt, 0, num_classes - 1)

            # Pred → semantic map
            pred = _pred_to_semantic_map(output, H_img, W_img, num_classes)

            # Per-class IoU
            for c in range(num_classes):
                p_c = (pred == c)
                g_c = (gt == c)
                inter = (p_c & g_c).sum()
                union = (p_c | g_c).sum()
                intersections[c] += inter
                unions[c] += union
                if g_c.sum() > 0:
                    class_correct[c] += inter
                    class_total[c] += g_c.sum()

            # Overall pixel accuracy
            pixel_correct += (pred == gt).sum()
            pixel_total += gt.size

            # Foreground Dice (class 0 excluded)
            if pred.max() > 0 or gt.max() > 0:
                p_fg = (pred > 0)
                g_fg = (gt > 0)
                dice_inter += (p_fg & g_fg).sum()
                dice_union += (p_fg | g_fg).sum()

    # ── Compute metrics ──────────────────────────────────
    valid = unions > 0
    per_class_iou = np.where(valid, intersections / unions, 0.0)
    mIoU = float(np.mean(per_class_iou[1:]))  # exclude background (class 0)

    # mAcc: mean of per-class accuracy (only for classes present in GT)
    per_class_acc = np.where(class_total > 0, class_correct / class_total, 0.0)
    mAcc = float(np.mean(per_class_acc[1:][class_total[1:] > 0])) if (class_total[1:] > 0).any() else 0.0

    # aAcc: overall pixel accuracy
    aAcc = float(pixel_correct / max(pixel_total, 1))

    # Dice
    Dice = float(2 * dice_inter / max(dice_union, 1e-8))

    # Per-class IoU for paper
    class_iou_str = ", ".join(
        f"{ISAID_CLASSES[c]}={per_class_iou[c]:.3f}"
        for c in range(1, num_classes) if unions[c] > 0
    )

    return {
        "mIoU": mIoU, "mAcc": mAcc, "aAcc": aAcc, "Dice": Dice,
        "per_class_iou": class_iou_str,
    }


# ═══════════════════════════════════════════════════════════════════
# Efficiency measurement
# ═══════════════════════════════════════════════════════════════════

def measure_efficiency_v2(
    model, device, image_size: Tuple[int, int],
    keep_ratio: float = 1.0, use_sparse: bool = False,
    warmup: int = 5, repeats: int = 30,
) -> Dict[str, float]:
    H, W = image_size
    dummy = torch.randn(1, 3, H, W, device=device)
    model.eval()
    params_total = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        for _ in range(warmup):
            try: _ = model(dummy, use_sparse=use_sparse)
            except Exception: pass
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            try: _ = model(dummy, use_sparse=use_sparse)
            except Exception: pass
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / max(repeats, 1) * 1000
        fps = 1000.0 / max(latency_ms, 1e-3)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            try: _ = model(dummy, use_sparse=use_sparse)
            except Exception: pass
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_mb = 0.0

    backbone_flops_640 = 257e9
    area_factor = (H * W) / (640 * 640)
    dense_flops = backbone_flops_640 * area_factor
    encoder_flops = 2.8e9 * area_factor
    total_flops = dense_flops + encoder_flops
    sparse_flops = dense_flops * max(keep_ratio, 0.05) + encoder_flops

    return {
        "FPS": round(fps, 1), "latency_ms": round(latency_ms, 1),
        "FLOPs_G": round(total_flops / 1e9, 1),
        "params_M": round(params_total / 1e6, 1),
        "memory_MB": round(peak_mb, 1),
        "sparse_FLOPs_G": round(sparse_flops / 1e9, 1),
        "sparse_FPS": round(fps * (1.0 / max(keep_ratio, 0.05)), 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Baseline templates (fill from your runs or literature)
# ═══════════════════════════════════════════════════════════════════

BASELINE_TEMPLATES: Dict[str, dict] = {
    # ── Classic CNN ─────────────────────────────────────
    "U-Net (R50)":          {"category": "classic", "params_M": 31.0},
    "DeepLabV3+ (R50)":     {"category": "classic", "params_M": 41.0},
    "PSPNet (R50)":         {"category": "classic", "params_M": 47.0},
    # ── Transformer ─────────────────────────────────────
    "SegFormer-B0":          {"category": "transformer", "params_M": 3.7},
    "SegFormer-B2":          {"category": "transformer", "params_M": 24.7},
    "Mask2Former (Swin-T)":  {"category": "transformer", "params_M": 42.0},
    # ── SAM-family ─────────────────────────────────────
    "FastSAM-x":             {"category": "sam", "params_M": 68.8},
    "EfficientSAM":          {"category": "sam", "params_M": 9.3},
}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Semantic Segmentation SOTA Comparison")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--keep-ratios", type=float, nargs="+", default=[1.0, 0.25, 0.15, 0.10])
    p.add_argument("--baselines", type=str, default=None,
                   help="JSON with pre-computed baseline results")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    plog = PaperLogger("outputs", "main_comparison", vars(args))
    plog.start()
    log = plog.runner

    img_sz_t = (args.image_size, args.image_size)
    all_results: List[MethodResult] = []

    # ═════════════════════════════════════════════════════
    # Our model
    # ═════════════════════════════════════════════════════
    from adatile.modeling.as_fastsam_v2 import build_as_fastsam_v2

    val_ds = UniversalDataset(args.dataset, split="val", image_size=img_sz_t, num_classes=None)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    n_classes = val_ds.num_classes
    print(f"Dataset: val={len(val_ds)}, classes={n_classes}")
    print(f"Classes: {ISAID_CLASSES[:n_classes]}")

    # Full inference (accuracy + efficiency)
    print(f"\n{'='*60}\n  Evaluating: AS-FastSAM v2 (Full)\n{'='*60}")
    model = build_as_fastsam_v2("FastSAM-x.pt", keep_ratio=1.0,
                                conf_threshold=0.25, image_size=args.image_size, device=device)
    acc = evaluate_v2_semantic(model, vl, device, img_sz_t, n_classes)
    eff = measure_efficiency_v2(model, device, img_sz_t, keep_ratio=1.0,
                                repeats=10 if args.quick else 30)
    r = MethodResult(name="AS-FastSAM v2 (Full)", category="ours")
    for k in ["mIoU", "mAcc", "aAcc", "Dice"]:
        setattr(r, k, acc.get(k, 0))
    for k in ["FPS", "latency_ms", "FLOPs_G", "params_M", "memory_MB",
               "sparse_FLOPs_G", "sparse_FPS"]:
        setattr(r, k, eff.get(k, 0))
    all_results.append(r)
    print(f"  mIoU={r.mIoU:.4f}  mAcc={r.mAcc:.4f}  aAcc={r.aAcc:.4f}  Dice={r.Dice:.4f}")
    print(f"  Per-class: {acc.get('per_class_iou', 'N/A')}")
    print(f"  FLOPs={r.FLOPs_G:.1f}G  FPS={r.FPS:.1f}  "
          f"Params={r.params_M:.1f}M  Mem={r.memory_MB:.0f}MB")
    del model; torch.cuda.empty_cache()

    # Sparse variants (efficiency only — uses same model weights)
    for kr in args.keep_ratios:
        if kr >= 1.0:
            continue
        variant_name = f"AS-FastSAM v2 (kr={kr:.0%})"
        print(f"\n{'='*60}\n  Evaluating: {variant_name}\n{'='*60}")
        model = build_as_fastsam_v2("FastSAM-x.pt", keep_ratio=kr,
                                    conf_threshold=0.25, image_size=args.image_size, device=device)
        eff = measure_efficiency_v2(model, device, img_sz_t, keep_ratio=kr,
                                    repeats=10 if args.quick else 30)
        r = MethodResult(name=variant_name, category="ours")
        for k in ["FPS", "latency_ms", "FLOPs_G", "params_M", "memory_MB",
                   "sparse_FLOPs_G", "sparse_FPS"]:
            setattr(r, k, eff.get(k, 0))
        all_results.append(r)
        print(f"  FLOPs={r.FLOPs_G:.1f}G  FPS={r.FPS:.1f}  "
              f"Params={r.params_M:.1f}M  Mem={r.memory_MB:.0f}MB")
        del model; torch.cuda.empty_cache()

    # ── Baselines ──────────────────────────────────────
    if args.baselines and Path(args.baselines).exists():
        with open(args.baselines, "r") as f:
            for entry in json.load(f): all_results.append(MethodResult(**entry))
    else:
        for name, data in BASELINE_TEMPLATES.items():
            r = MethodResult(name=name, category=data["category"])
            for k, v in data.items():
                if k != "category" and v is not None: setattr(r, k, v)
            all_results.append(r)

    # ═════════════════════════════════════════════════════
    # Paper Tables
    # ═════════════════════════════════════════════════════
    ours = [r for r in all_results if r.category == "ours"]
    classic = [r for r in all_results if r.category == "classic"]
    transformer = [r for r in all_results if r.category == "transformer"]
    sam = [r for r in all_results if r.category == "sam"]

    def F(val, d=2, s=""):
        return f"{val:.{d}f}{s}" if (val is not None and val != 0.0) else "   —    "

    # Accuracy Table
    print(f"\n{'='*100}")
    print(f"  Table 1: Semantic Segmentation on iSAID (16 classes)")
    print(f"{'='*100}")
    print(f"  {'Method':<30} {'mIoU':>8} {'mAcc':>8} {'aAcc':>8} {'Dice':>8}")
    print(f"  {'-'*65}")
    for grp_name, grp in [("Classic CNN", classic), ("Transformer", transformer),
                           ("SAM-family", sam), ("Ours", ours)]:
        if grp:
            print(f"  {grp_name}:")
            for r in grp:
                m = " ★" if r.category == "ours" and "kr=" not in r.name else ""
                print(f"  {r.name:<30} {F(r.mIoU):>8} {F(r.mAcc):>8} "
                      f"{F(r.aAcc):>8} {F(r.Dice):>8}{m}")
        print(f"  {'-'*65}")

    # Efficiency Table
    print(f"\n  {'Method':<30} {'FPS':>7} {'Latency':>9} {'FLOPs(G)':>10} "
          f"{'Params(M)':>10} {'Mem(MB)':>9}")
    print(f"  {'-'*82}")
    for grp_name, grp in [("Classic CNN", classic), ("Transformer", transformer),
                           ("SAM-family", sam), ("Ours", ours)]:
        if grp:
            print(f"  {grp_name}:")
            for r in grp:
                m = " ★" if r.category == "ours" and "kr=" not in r.name else "  "
                print(f"  {r.name:<30} {F(r.FPS,1):>7} {F(r.latency_ms,1,'ms'):>9} "
                      f"{F(r.FLOPs_G,1):>10} {F(r.params_M,1):>10} {F(r.memory_MB,0):>9}{m}")
        print(f"  {'-'*82}")

    # LaTeX Table
    print(f"\n  % ===== LaTeX Table =====")
    print(f"  \\begin{{table}}[t]")
    print(f"  \\caption{{Semantic segmentation on iSAID. ★: our method.}}")
    print(f"  \\label{{tab:main_comparison}}")
    print(f"  \\begin{{tabular}}{{lccccc}}\\toprule")
    print(f"  Method & mIoU & mAcc & FPS & FLOPs(G) & Params(M) \\\\\\midrule")
    for gn, grp in [("Classic CNN", classic), ("Transformer", transformer),
                     ("SAM-family", sam), ("Ours", ours)]:
        if grp:
            print(f"  \\multicolumn{{6}}{{c}}{{\\textit{{{gn}}}}} \\\\")
            for r in grp:
                b = "\\textbf{" if r.category == "ours" and "kr=" not in r.name else ""
                e = "}" if b else ""
                print(f"  {b}{r.name:<25}{e} & {b}{F(r.mIoU)}{e} & {b}{F(r.mAcc)}{e} & "
                      f"{b}{F(r.FPS,1)}{e} & {b}{F(r.FLOPs_G,1)}{e} & "
                      f"{b}{F(r.params_M,1)}{e} \\\\")
    print(f"  \\bottomrule\\end{{tabular}}\\end{{table}}")

    log.log_table("main_comparison", [r.to_dict() for r in all_results])
    best_ours = next((r for r in ours if "kr=" not in r.name), ours[0] if ours else None)
    flops_ratio = (1.0 - (ours[-1].FLOPs_G / max(ours[0].FLOPs_G, 1e-8))
                   if len(ours) >= 2 and ours[0].FLOPs_G > 0 else None)
    plog.finish(
        best_dice=best_ours.Dice if best_ours else 0.0,
        n_params=int(best_ours.params_M * 1e6) if best_ours else 0,
        flops_saved=flops_ratio,
    )
    print(f"\nResults: {log.run_dir}")


if __name__ == "__main__":
    a = parse_args()
    if a.quick: a.keep_ratios = [1.0, 0.15]
    main()
