#!/usr/bin/env python3
"""
Neu_seg 评估脚本 | Neu_seg Evaluation Script.
===============================================

加载训练好的 checkpoint, 在 Neu_seg 测试集上计算多类别分割指标 (mIoU / per-class IoU / Pixel Accuracy)。
Load trained checkpoint, compute multi-class segmentation metrics on Neu_seg test set.

用法 | Usage::

    # 基础评估
    python tools/eval/eval_neuseg.py --checkpoint runs/neuseg_xxx/best_model.pt

    # 保存可视化
    python tools/eval/eval_neuseg.py --checkpoint runs/neuseg_xxx/best_model.pt --save-vis
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

import torch
import torch.nn.functional as F

from adatile.backbone import FastSAMBackbone
from adatile.decoder.adaptive_sparse_decoder import (
    AdaptiveSparseDecoder, ProtoOnlyDecoder, ProtoOnlyDecoderP3P4,
)
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4, PureDecoderP2P3P4
from adatile.adapter import MultiScaleAdapter
from adatile.frequency import MultiScaleSpectralAttention, FrequencyGuidedFusion
from adatile.datasets.neu_seg import NEUSegDataset

# 类别名称 | Class Names
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]
NUM_CLASSES = 4


def pad_to_32(image, mask=None):
    """Pad image to multiple of 32 (FastSAM backbone requirement)."""
    if image.dim() == 4:
        H, W = image.shape[2], image.shape[3]
    else:
        H, W = image.shape[1], image.shape[2]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return image, mask, (H, W)
    pad_dims = (0, pad_w, 0, pad_h)
    image_padded = F.pad(image, pad_dims, mode='constant', value=0)
    mask_padded = F.pad(mask, pad_dims, mode='constant', value=0) if mask is not None else None
    return image_padded, mask_padded, (H, W)


# ═══════════════════════════════════════════════════════════════════
# Decoder 前向传播 | Decoder Forward
# ═══════════════════════════════════════════════════════════════════

def _decoder_forward(decoder, feats, support_cache, freq_fusion=None):
    """统一的 decoder 前向传播 | Unified decoder forward."""
    if isinstance(decoder, PureDecoderP2P3P4):
        p2 = feats.get("p2")
        if p2 is None:
            return None
        return decoder(p2, feats["p3"], feats["p4"], freq_fusion=freq_fusion)
    elif isinstance(decoder, PureDecoderP3P4):
        return decoder(feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoder):
        return decoder(feats["p4"])
    elif isinstance(decoder, ProtoOnlyDecoderP3P4):
        proto = feats.get("proto")
        if proto is None:
            return None
        sp3, sp4 = support_cache
        return decoder(proto, sp3, sp4)
    elif isinstance(decoder, ProtoOnlyDecoder):
        proto = feats.get("proto")
        if proto is None:
            return None
        return decoder(proto, support_cache)
    else:  # Adaptive
        proto = feats.get("proto")
        if proto is None:
            return None
        return decoder(feats["p4"], proto, support_cache)


# ═══════════════════════════════════════════════════════════════════
# 评估核心 | Evaluation Core
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_neuseg(
    decoder: torch.nn.Module,
    backbone: FastSAMBackbone,
    support_cache,
    dataset: NEUSegDataset,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    adapter: torch.nn.Module | None = None,
    spectral_attn: torch.nn.Module | None = None,
    freq_fusion: torch.nn.Module | None = None,
) -> dict:
    """
    在 Neu_seg 数据集上评估多类别分割 | Evaluate multi-class segmentation on Neu_seg dataset.

    :param decoder: decoder module.
    :param backbone: FastSAM backbone.
    :param support_cache: prototype tensor or (p3_proto, p4_proto) tuple.
    :param dataset: validation/test dataset.
    :param device: compute device.
    :param num_classes: total classes (default 4: BG + Inclusion + Patch + Scratch).
    :return: evaluation metrics dict.
    """
    decoder.eval()

    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)
    all_correct = 0
    all_total = 0
    per_sample = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].squeeze(0)  # [H, W]
        H, W = gt_mask.shape

        # ── Pad to multiple of 32 ──
        img, _, _ = pad_to_32(img)

        # ── Backbone ──
        feats = backbone(img, extract_proto=True)

        # ── Adapter (CAT-SAM style) | 特征域适配 ──
        if adapter is not None:
            adapted = adapter(p3=feats.get("p3"), p4=feats.get("p4"), p8=feats.get("p8"))
            feats.update(adapted)

        # ── DCT Spectral Attention | 频域注意力 ──
        if spectral_attn is not None:
            spec_feats = spectral_attn(
                p2=feats.get("p2"), p3=feats.get("p3"), p4=feats.get("p4"),
            )
            feats.update(spec_feats)

        # ── Decoder ──
        pred_prob = _decoder_forward(decoder, feats, support_cache,
                                     freq_fusion=freq_fusion)
        if pred_prob is None:
            continue

        # ── 上采样 | Upsample → [C, H, W] ──
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0)  # [C, H, W]

        # ── 计算指标 | Compute Metrics ──
        pred_class = torch.argmax(pred_full, dim=0)  # [H, W]
        gt_class = gt_mask.long().to(device)          # [H, W]

        all_correct += (pred_class == gt_class).sum().item()
        all_total += gt_class.numel()

        sample_iou = 0.0
        n_classes_present = 0
        for c in range(num_classes):
            pred_c = (pred_class == c)
            gt_c = (gt_class == c)
            inter = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            per_class_inter[c] += inter
            per_class_union[c] += union
            if gt_c.sum() > 0 and union > 0:
                sample_iou += (inter / union).item()
                n_classes_present += 1

        sample_miou = sample_iou / max(n_classes_present, 1)
        per_sample.append({
            "image_id": sample["image_id"],
            "mIoU": round(sample_miou, 6),
            "pixel_acc": round((pred_class == gt_class).float().mean().item(), 6),
        })

    # ── 全局指标 | Global Metrics ──
    per_class_iou = {}
    valid_ious = []
    for c in range(num_classes):
        inter = per_class_inter[c].item()
        union = per_class_union[c].item()
        iou_c = inter / union if union > 0 else float("nan")
        per_class_iou[CLASS_NAMES[c]] = round(iou_c, 6)
        if iou_c == iou_c:  # not NaN
            valid_ious.append(iou_c)

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc = all_correct / all_total if all_total > 0 else 0.0

    sample_mious = [s["mIoU"] for s in per_sample]
    return {
        "mIoU": round(miou, 6),
        "pixel_accuracy": round(pixel_acc, 6),
        "per_class_IoU": per_class_iou,
        "n_samples": len(per_sample),
        "sample_mIoU_mean": round(float(np.mean(sample_mious)), 6) if sample_mious else 0.0,
        "sample_mIoU_median": round(float(np.median(sample_mious)), 6) if sample_mious else 0.0,
        "per_sample": per_sample,
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_visualizations(
    decoder: torch.nn.Module,
    backbone: FastSAMBackbone,
    support_cache,
    dataset: NEUSegDataset,
    device: torch.device,
    output_dir: Path,
    num_classes: int = NUM_CLASSES,
    max_samples: int = 5,
    adapter: torch.nn.Module | None = None,
    spectral_attn: torch.nn.Module | None = None,
    freq_fusion: torch.nn.Module | None = None,
):
    """
    保存验证集图像的多类别可视化结果 | Save multi-class visualization for val images.

    输出: 原图 + GT 类别色 + Pred 类别色 + 差异标注
    Output: original + GT color-coded + Pred color-coded + diff

    :param output_dir: 输出目录 | Output directory.
    :param max_samples: 最多保存样本数 | Max samples to save.
    """
    import cv2

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── 多类别颜色映射 | Multi-class color map ──
    class_colors = {
        0: (128, 128, 128),   # BG: gray
        1: (255, 0, 0),       # Inclusion: red
        2: (0, 255, 0),       # Patch: green
        3: (0, 0, 255),       # Scratch: blue
    }

    for idx in range(min(max_samples, len(dataset))):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].squeeze(0)  # [H, W]
        H, W = gt_mask.shape

        # ── Pad to multiple of 32 ──
        img, _, _ = pad_to_32(img)

        feats = backbone(img, extract_proto=True)

        # ── Adapter (CAT-SAM style) | 特征域适配 ──
        if adapter is not None:
            adapted = adapter(p3=feats.get("p3"), p4=feats.get("p4"), p8=feats.get("p8"))
            feats.update(adapted)

        # ── DCT Spectral Attention | 频域注意力 ──
        if spectral_attn is not None:
            spec_feats = spectral_attn(
                p2=feats.get("p2"), p3=feats.get("p3"), p4=feats.get("p4"),
            )
            feats.update(spec_feats)

        pred_prob = _decoder_forward(decoder, feats, support_cache,
                                     freq_fusion=freq_fusion)
        if pred_prob is None:
            continue

        # ── 上采样 | Upsample → [C, H, W] ──
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0).cpu()  # [C, H, W]

        # ── 加载原图 | Load original image ──
        img_np = (sample["image"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ── 多类别: color-coded GT + Pred side-by-side ──
        gt_class = gt_mask.long().cpu().numpy()
        pred_class = torch.argmax(pred_full, dim=0).cpu().numpy()

        gt_colored = np.zeros((H, W, 3), dtype=np.uint8)
        pred_colored = np.zeros((H, W, 3), dtype=np.uint8)
        for c, color in class_colors.items():
            gt_colored[gt_class == c] = color
            pred_colored[pred_class == c] = color

        # Side-by-side: GT | Pred | Difference
        diff = np.zeros((H, W, 3), dtype=np.uint8)
        correct = (gt_class == pred_class)
        diff[correct] = (128, 128, 128)      # correct → gray
        diff[~correct] = (0, 0, 255)          # wrong → red

        combined = np.hstack([gt_colored, pred_colored, diff])
        cv2.imwrite(str(vis_dir / f"{sample['image_id']}_multiclass.png"),
                   cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"  [OK] Saved {min(max_samples, len(dataset))} visualizations to {vis_dir}")


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Neu_seg Multi-class Evaluation"
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="模型 checkpoint 路径 | Path to model checkpoint")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg",
                   help="数据根目录 | Data root")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", type=str, default=None,
                   choices=["fastsam-x", "fastsam-s"],
                   help="Backbone 模型 (默认从 checkpoint 读取) | Backbone model (default: read from checkpoint)")
    p.add_argument("--save-vis", action="store_true",
                   help="保存可视化结果 | Save visualization results")
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 (默认与 checkpoint 同目录) | Output dir")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── 输出目录 | Output Dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ckpt_path.parent / "eval_neuseg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 Checkpoint | Load Checkpoint ──
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    args_ckpt = ckpt.get("args", {})
    decoder_type = args_ckpt.get("decoder_type", "adaptive")
    proto_source = args_ckpt.get("proto_source", "p4")
    backbone_name = args.backbone or args_ckpt.get("backbone", "fastsam-x")

    print(f"  Backbone: {backbone_name}")
    print(f"  Decoder: {decoder_type}, proto_source: {proto_source}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    # 新 ckpt 记录 best_mIoU；旧 ckpt 仅有 best_Dice/Dice (向后兼容)
    # New ckpts record best_mIoU; old ones only best_Dice/Dice (backward compat)
    print(f"  Best mIoU (train-time): "
          f"{ckpt.get('best_mIoU', ckpt.get('mIoU', 'N/A'))}")
    print(f"  Best Dice (train-time): {ckpt.get('best_Dice', ckpt.get('Dice', 'N/A'))}")

    # ── 加载数据集 (始终多类别) | Load Dataset (always multi-class) ──
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"  Val samples: {len(val_ds)} (format=NEU_Seg, multi-class)")

    # ── 构建模型 | Build Model ──
    checkpoint_path = f"thirdLibrary/FastSAM/weights/FastSAM-{backbone_name.split('-')[-1]}.pt"
    backbone = FastSAMBackbone(freeze_backbone=True, checkpoint=checkpoint_path).to(device)
    backbone.eval()

    # ── 自动探测通道数 | Auto-detect channel counts ──
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        backbone(dummy, extract_proto=False)
    ch = backbone.channels if all(v > 0 for v in backbone.channels.values()) else \
         {"p2": 160, "p3": 960, "p4": 1280, "p8": 1280}
    print(f"  Auto-detected channels: {ch}")

    if decoder_type == "proto_only":
        decoder = ProtoOnlyDecoder(proto_dim=32, feat_dim=ch["p4"]).to(device)
    elif decoder_type == "proto_only_p3p4":
        decoder = ProtoOnlyDecoderP3P4(proto_dim=32, p3_dim=ch["p3"], p4_dim=ch["p4"]).to(device)
    elif decoder_type == "pure":
        decoder = PureDecoder(in_channels=ch["p4"], out_channels=NUM_CLASSES).to(device)
    elif decoder_type == "pure_p3p4":
        decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"],
                                  out_channels=NUM_CLASSES).to(device)
    elif decoder_type == "pure_p2p3p4":
        decoder = PureDecoderP2P3P4(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=NUM_CLASSES, mid_channels=128,
        ).to(device)
    else:
        decoder = AdaptiveSparseDecoder(
            in_channels=ch["p4"], proto_dim=32, use_fdr=False,
            out_channels=NUM_CLASSES,
        ).to(device)

    # 加载 decoder 权重 | Load decoder weights
    decoder_state = ckpt.get("decoder_state_dict", {})
    if decoder_state:
        try:
            decoder.load_state_dict(decoder_state, strict=False)
            print(f"  Loaded decoder weights: {len(decoder_state)} keys")
        except Exception as e:
            print(f"  [WARN] Failed to load decoder state: {e}")
    else:
        print("  [WARN] No decoder_state_dict in checkpoint!")

    # ── CAT-SAM Adapter | 特征域适配器 ──
    adapter = None
    adapter_state = ckpt.get("adapter_state_dict")
    if adapter_state is not None:
        adapter = MultiScaleAdapter(
            p3_channels=ch["p3"], p4_channels=ch["p4"], p8_channels=ch["p8"], reduction=4,
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        print(f"  Loaded MultiScaleAdapter: {len(adapter_state)} keys")

    # ── DCT Spectral Attention | 频域注意力 ──
    spectral_attn = None
    spectral_state = ckpt.get("spectral_attn_state_dict")
    if spectral_state is not None:
        spectral_attn = MultiScaleSpectralAttention(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            reduction=4, n_freq=16,
        ).to(device)
        spectral_attn.load_state_dict(spectral_state)
        spectral_attn.eval()
        print(f"  Loaded MultiScaleSpectralAttention: {len(spectral_state)} keys")

    # ── Frequency Guided Fusion | 频率引导融合 ──
    freq_fusion = None
    ff_state = ckpt.get("freq_fusion_state_dict")
    if ff_state is not None:
        freq_fusion = FrequencyGuidedFusion(mid_channels=128, patch_size=8).to(device)
        freq_fusion.load_state_dict(ff_state)
        freq_fusion.eval()
        print(f"  Loaded FrequencyGuidedFusion: {len(ff_state)} keys")

    # 加载 support prototype | Load support prototype
    if decoder_type == "proto_only_p3p4":
        support_proto_p3 = ckpt.get("support_proto_p3",
            torch.zeros(960, device=device))
        support_proto_p4 = ckpt.get("support_proto_p4",
            torch.zeros(1280, device=device))
        support_cache = (support_proto_p3.to(device), support_proto_p4.to(device))
        print(f"  Support proto (dual) |p3|={support_proto_p3.norm().item():.4f}"
              f"  |p4|={support_proto_p4.norm().item():.4f}")
    else:
        support_proto = ckpt.get("support_proto")
        if support_proto is None:
            support_proto = torch.zeros(1280, device=device)
        elif not isinstance(support_proto, torch.Tensor):
            support_proto = torch.zeros(1280, device=device)
        else:
            support_proto = support_proto.to(device)
        support_cache = support_proto
        if isinstance(support_proto, torch.Tensor):
            print(f"  Support proto |p|={support_proto.norm().item():.4f}")

    # ── 评估 | Evaluate ──
    print(f"\n{'='*60}")
    print(f"  Multi-class Evaluation")
    print(f"{'='*60}")

    result = evaluate_neuseg(
        decoder, backbone, support_cache, val_ds, device,
        num_classes=NUM_CLASSES, adapter=adapter,
        spectral_attn=spectral_attn, freq_fusion=freq_fusion,
    )

    # ── 打印结果 | Print Results ──
    print(f"  mIoU:          {result['mIoU']:.4f}")
    print(f"  Pixel Accuracy: {result['pixel_accuracy']:.4f}")
    print(f"  Sample mIoU:   mean={result['sample_mIoU_mean']:.4f} "
          f"median={result['sample_mIoU_median']:.4f}")
    print(f"  Per-class IoU:")
    for cls_name, iou_c in result["per_class_IoU"].items():
        print(f"    {cls_name:>12s}: {iou_c:.4f}")

    # ── 保存结果 | Save Results ──
    results_file = out_dir / "eval_results.json"
    eval_output = {
        "checkpoint": str(ckpt_path),
        "decoder_type": decoder_type,
        "use_adapter": adapter is not None,
        "num_classes": NUM_CLASSES,
        "data_root": args.data_root,
        "n_val_samples": len(val_ds),
        "results": result,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_file, "w") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_file}")

    # ── 可视化 | Visualization ──
    if args.save_vis:
        print(f"\n  Saving visualizations...")
        save_visualizations(
            decoder, backbone, support_cache, val_ds, device,
            output_dir=out_dir,
            num_classes=NUM_CLASSES,
            max_samples=5,
            adapter=adapter,
            spectral_attn=spectral_attn,
            freq_fusion=freq_fusion,
        )

    print(f"\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
