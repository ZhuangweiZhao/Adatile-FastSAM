#!/usr/bin/env python3
"""
缺陷 Prompt → FastSAM 评估 | Defect Prompt → FastSAM Evaluation.
=================================================================

端到端评估：经典 CV 提取缺陷 Prompt → FastSAM prompt 模式 → 分割 Mask → mIoU。
End-to-end eval: classical CV defect prompts → FastSAM prompt mode → masks → mIoU.

核心假设 | Core Hypothesis:
    FastSAM 不需要训练也能分割缺陷——它缺的是正确的 Prompt。
    FastSAM doesn't need training to segment defects — it just needs the right prompts.
    经典 CV (FFT/梯度/纹理) 提供无训练的、物理可解释的 Prompt。
    Classical CV (FFT/gradient/texture) provides training-free, physically interpretable prompts.

用法 | Usage::

    python tools/eval/eval_defect_prompt.py \
        --data-root data/NEU_Seg --fastsam-weight FastSAM-x.pt \
        --sources fft,gradient,texture --max-boxes 12
"""

from __future__ import annotations

import sys, argparse, json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm
import cv2

import torch

from adatile.datasets.neu_seg import NEUSegDataset
from adatile.prompt.defect_prompts import DefectPromptExtractor, DefectHeatmaps

NUM_CLASSES = 4
CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]


# ═══════════════════════════════════════════════════════════════════
# FastSAM Prompt 推理 | FastSAM Prompt Inference
# ═══════════════════════════════════════════════════════════════════

class FastSAMPromptEvaluator:
    """
    使用 FastSAM prompt API 进行缺陷分割评估。
    FastSAM prompt-based defect segmentation evaluator.

    Parameters
    ----------
    model_path : str
        FastSAM 权重路径 | Path to FastSAM weights.
    device : str
        设备 | Device (cuda/cpu).
    imgsz : int
        FastSAM 推理尺寸 | FastSAM inference size.
    conf : float
        Everything mode 置信度阈值 | Confidence threshold for everything mode.
    iou : float
        Everything mode NMS IoU 阈值 | NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str = "FastSAM-x.pt",
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.15,
        iou: float = 0.5,
    ):
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        # 导入 FastSAM | Import FastSAM
        from fastsam import FastSAM, FastSAMPrompt
        self.FastSAM = FastSAM
        self.FastSAMPrompt = FastSAMPrompt

        print(f"Loading FastSAM from {model_path}...")
        self.model = FastSAM(model_path)

    @torch.no_grad()
    def segment_with_boxes(
        self,
        image_np: np.ndarray,
        boxes: np.ndarray,           # [N, 4] xyxy
    ) -> np.ndarray | None:
        """
        FastSAM everything → box_prompt → 合并 mask.
        FastSAM everything → box_prompt → merged masks.

        :param image_np: [H, W, 3] RGB image, uint8 [0,255].
        :param boxes: [N, 4] xyxy boxes in image coordinates.
        :return: [H, W] binary mask (defect=1, BG=0), or None if no masks selected.
        """
        H, W = image_np.shape[:2]

        # ── FastSAM everything mode ──
        everything_results = self.model(
            image_np,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
        )

        if everything_results is None or len(everything_results) == 0:
            return None

        # ── FastSAMPrompt: 用 box 选择相关 mask ──
        prompt = self.FastSAMPrompt(image_np, everything_results, device=self.device)

        if boxes is None or len(boxes) == 0:
            # 没有 prompt → 退回到 everything 最高置信度 mask
            all_masks = prompt.everything_prompt()
            if all_masks is None or len(all_masks) == 0:
                return None
            return self._merge_masks(all_masks.cpu().numpy(), H, W)

        # 用 box prompt 选择 mask | Select masks using box prompts
        selected = prompt.box_prompt(bboxes=boxes.tolist())
        if selected is None or len(selected) == 0:
            return None

        return self._merge_masks(selected, H, W)

    def _merge_masks(self, masks: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        合并多个二值 mask → 单个二值 mask (逻辑 OR).
        Merge multiple binary masks → single binary mask (logical OR).

        :param masks: [N, h, w] binary masks at FastSAM resolution.
        :param H: 目标高度 | Target height.
        :param W: 目标宽度 | Target width.
        :return: [H, W] merged binary mask.
        """
        if len(masks) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # 缩放到原图尺寸 | Resize to original size
        merged = np.zeros((H, W), dtype=np.float32)
        for mask in masks:
            mask_resized = cv2.resize(
                mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
            )
            merged = np.maximum(merged, mask_resized)

        return (merged >= 0.5).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    """
    计算 per-class IoU + mIoU.
    Compute per-class IoU + mIoU.

    :param pred: [H, W] predicted class labels.
    :param gt: [H, W] ground truth class labels.
    :return: dict with 'mIoU', 'per_class', 'binary_defect_iou'.
    """
    per_class = {}
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        per_class[CLASS_NAMES[c]] = float(inter / union) if union > 0 else float("nan")

    valid = [v for v in per_class.values() if not (v != v)]
    miou = float(np.mean(valid)) if valid else 0.0

    # Binary defect IoU (non-BG classes vs BG)
    pred_defect = (pred > 0).astype(np.uint8)
    gt_defect = (gt > 0).astype(np.uint8)
    inter_b = (pred_defect & gt_defect).sum()
    union_b = (pred_defect | gt_defect).sum()
    binary_iou = float(inter_b / union_b) if union_b > 0 else 0.0

    # Defect recall: what fraction of GT defect pixels were found
    gt_defect_count = gt_defect.sum()
    defect_recall = float(inter_b / gt_defect_count) if gt_defect_count > 0 else 0.0

    # Defect precision: what fraction of predicted defect pixels are correct
    pred_defect_count = pred_defect.sum()
    defect_precision = float(inter_b / pred_defect_count) if pred_defect_count > 0 else 0.0

    return {
        "mIoU": round(miou, 6),
        "per_class": {k: round(v, 6) for k, v in per_class.items()},
        "binary_defect_iou": round(binary_iou, 6),
        "defect_recall": round(defect_recall, 6),
        "defect_precision": round(defect_precision, 6),
    }


def assign_multiclass(pred_binary: np.ndarray, image: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    将二值 defect mask 转换为多类预测 (简单启发式).
    Convert binary defect mask to multi-class prediction (simple heuristic).

    由于 classical CV 无法区分缺陷类型, 使用颜色/强度启发式:
    Since classical CV can't distinguish defect types, use color/intensity heuristics:
    - 更亮的缺陷 → Scratch (高反射)
    - 更暗的缺陷 → Inclusion (夹杂物)
    - 其他 → Patch

    注: 这是粗略近似, 主要用于 per-class IoU 参考。
    Note: This is a rough approximation, mainly for per-class IoU reference.
    """
    result = np.zeros_like(pred_binary)

    if pred_binary.sum() == 0:
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image

    defect_pixels = pred_binary > 0
    defect_gray = gray[defect_pixels]

    if len(defect_gray) == 0:
        return result

    mean_val = defect_gray.mean()
    std_val = defect_gray.std()

    # 简单启发式 | Simple heuristic
    bright_mask = pred_binary & (gray > mean_val + 0.5 * std_val)   # Scratch (bright scratches)
    dark_mask = pred_binary & (gray < mean_val - 0.5 * std_val)     # Inclusion (dark spots)
    mid_mask = pred_binary & (~bright_mask) & (~dark_mask)           # Patch (irregular)

    result[bright_mask] = 3  # Scratch
    result[dark_mask] = 1     # Inclusion
    result[mid_mask] = 2      # Patch

    return result


# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Defect Prompt → FastSAM Evaluation")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--fastsam-weight", type=str,
                   default="thirdLibrary/FastSAM/weights/FastSAM-x.pt")
    p.add_argument("--sources", type=str, default="fft,gradient,texture",
                   help="逗号分隔的 prompt 源 | comma-separated prompt sources")
    p.add_argument("--max-boxes", type=int, default=12)
    p.add_argument("--fft-cutoff", type=float, default=0.08)
    p.add_argument("--conf", type=float, default=0.15,
                   help="FastSAM everything 置信度 | everything confidence threshold")
    p.add_argument("--iou", type=float, default=0.5,
                   help="FastSAM everything NMS IoU | NMS IoU threshold")
    p.add_argument("--max-samples", type=int, default=0,
                   help="最多评估样本数 (0=全部) | max samples to eval (0=all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-vis", type=int, default=0,
                   help="保存 N 张可视化样本 | save N visualization samples")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = args.device
    sources = tuple(s.strip() for s in args.sources.split(","))

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"runs/defect_prompt_eval_{datetime.now().strftime('%m%d_%H%M')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"Test samples: {len(dataset)}")

    # ── Extractors ──
    prompt_extractor = DefectPromptExtractor(
        fft_cutoff=args.fft_cutoff,
        max_boxes=args.max_boxes,
    )
    fastsam_eval = FastSAMPromptEvaluator(
        model_path=args.fastsam_weight,
        device=device,
        conf=args.conf,
        iou=args.iou,
    )

    # ── Evaluation ──
    per_class_inter = np.zeros(NUM_CLASSES)
    per_class_union = np.zeros(NUM_CLASSES)
    binary_inter = 0.0
    binary_union = 0.0
    total_defect_pixels = 0
    found_defect_pixels = 0

    results = []
    indices = list(range(len(dataset)))
    if args.max_samples > 0:
        indices = indices[:args.max_samples]

    vis_samples = []

    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        image_tensor = sample["image"]           # [3, H, W] float32 [0,1]
        gt_mask = sample["masks"].squeeze(0).numpy().astype(np.int64)  # [H, W]

        # ── Tensor → numpy image ──
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3] uint8

        # ── Step 1: Classical CV → Prompt Boxes ──
        boxes, heatmaps = prompt_extractor(
            image_np, sources=sources, return_heatmaps=True
        )

        # ── Step 2: FastSAM box_prompt → Mask ──
        pred_binary = fastsam_eval.segment_with_boxes(image_np, boxes)

        if pred_binary is None:
            pred_binary = np.zeros(gt_mask.shape, dtype=np.uint8)

        # ── Step 3: Multi-class assignment (heuristic) ──
        pred_multiclass = assign_multiclass(pred_binary, image_np)

        # ── Step 4: Compute metrics ──
        for c in range(NUM_CLASSES):
            pc = (pred_multiclass == c)
            gc = (gt_mask == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

        gt_defect = (gt_mask > 0).astype(np.uint8)
        pred_defect = (pred_binary > 0).astype(np.uint8)
        binary_inter += (pred_defect & gt_defect).sum()
        binary_union += (pred_defect | gt_defect).sum()
        total_defect_pixels += gt_defect.sum()
        found_defect_pixels += (pred_defect & gt_defect).sum()

        results.append({
            "idx": idx,
            "num_boxes": len(boxes),
            "gt_defect_pct": float(gt_defect.mean()),
            "pred_defect_pct": float(pred_defect.mean()),
        })

        # ── 可视化采样 | Visualization samples ──
        if args.save_vis > 0 and len(vis_samples) < args.save_vis:
            if gt_defect.sum() > 0:  # 优先保存有缺陷的样本
                vis_samples.append({
                    "idx": idx,
                    "image": image_np,
                    "gt": gt_mask,
                    "pred_binary": pred_binary,
                    "pred_multiclass": pred_multiclass,
                    "boxes": boxes.copy() if len(boxes) > 0 else None,
                    "heatmaps": heatmaps,
                })

    # ── Aggregate ──
    binary_iou = binary_inter / max(binary_union, 1)
    defect_recall = found_defect_pixels / max(total_defect_pixels, 1)

    per_class_iou = {}
    for c in range(NUM_CLASSES):
        inter = per_class_inter[c]
        union = per_class_union[c]
        per_class_iou[CLASS_NAMES[c]] = round(float(inter / union), 6) if union > 0 else float("nan")

    valid = [v for v in per_class_iou.values() if not (v != v)]
    miou = round(float(np.mean(valid)), 6) if valid else 0.0

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  Defect Prompt → FastSAM Evaluation")
    print(f"  Sources: {sources}, Max Boxes: {args.max_boxes}")
    print(f"  Samples: {len(results)}")
    print(f"{'='*60}")
    print(f"  Binary Defect IoU:  {binary_iou:.4f}")
    print(f"  Defect Recall:      {defect_recall:.4f}")
    print(f"  Multi-class mIoU:   {miou:.4f}")
    print(f"  Per-class IoU:")
    for name, iou in per_class_iou.items():
        print(f"    {name:12s}: {iou:.4f}")

    avg_boxes = np.mean([r["num_boxes"] for r in results])
    print(f"\n  Avg boxes/image: {avg_boxes:.1f}")

    # ── Save results ──
    summary = {
        "binary_defect_iou": binary_iou,
        "defect_recall": defect_recall,
        "multi_class_mIoU": miou,
        "per_class_IoU": per_class_iou,
        "avg_boxes_per_image": avg_boxes,
        "num_samples": len(results),
        "args": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")

    # ── 保存可视化 | Save visualizations ──
    if vis_samples:
        save_visualizations(vis_samples, out_dir)
        print(f"Visualizations saved to {out_dir}")


def save_visualizations(samples: list, out_dir: Path):
    """保存对比可视化 | Save comparison visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CLASS_COLORS = [
        [0, 0, 0],           # BG — black
        [1.0, 0.3, 0.3],     # Inclusion — red
        [0.3, 0.6, 1.0],     # Patch — blue
        [0.3, 1.0, 0.3],     # Scratch — green
    ]

    def _build_lbl(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(NUM_CLASSES):
            rgb[mask == c] = CLASS_COLORS[c]
        return rgb

    n = len(samples)
    n_cols = 5  # Image, GT, Pred Binary, Pred Multi, Heatmap
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.5, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, s in enumerate(samples):
        img = s["image"]
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Image", fontsize=8)

        axes[row, 1].imshow(_build_lbl(s["gt"]))
        axes[row, 1].set_title("GT", fontsize=8)

        axes[row, 2].imshow(s["pred_binary"], cmap="gray")
        axes[row, 2].set_title("Pred (Binary)", fontsize=8)

        axes[row, 3].imshow(_build_lbl(s["pred_multiclass"]))
        axes[row, 3].set_title("Pred (Multi)", fontsize=8)

        # Heatmap + boxes overlay
        hm = s["heatmaps"].fused if s["heatmaps"] else np.zeros_like(s["gt"], dtype=np.float32)
        axes[row, 4].imshow(img.mean(axis=-1), cmap="gray")
        axes[row, 4].imshow(hm, cmap="hot", alpha=0.5)
        if s["boxes"] is not None and len(s["boxes"]) > 0:
            for box in s["boxes"]:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color="lime", linewidth=1)
                axes[row, 4].add_patch(rect)
        axes[row, 4].set_title("Heatmap + Boxes", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Defect Prompt → FastSAM", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "viz_samples.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
