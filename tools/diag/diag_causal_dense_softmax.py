#!/usr/bin/env python3
"""
Causal Experiment: Dense Softmax → Frozen Decoder | 因果实验.

================================================================================
THE ONLY CAUSAL EXPERIMENT | 唯一的因果实验:
================================================================================

    问题: Cross-Attention vs Dense Softmax, 差异来自 Attention 还是 Decoder?

    对照组 (Control):   Cross-Attention + Frozen Decoder  → baseline IoU
    实验组 (Treatment):  Dense Softmax  + Frozen Decoder  → oracle IoU

    保持 Decoder 权重完全不变。
    Keep Decoder weights IDENTICAL between conditions.

    如果 Δ(Treatment - Control) 很大 → Attention 是瓶颈
    如果 Δ(Treatment - Control) 很小 → Decoder 是瓶颈

================================================================================
METHOD | 方法:
================================================================================

    Control (Cross-Attention):
        fuse(P3,P4) → CrossAttn(Q=fuse, K/V=proj(tokens)) → resadd → upsample → mask

    Treatment (Dense Softmax):
        fuse(P3,P4) → DenseSoftmax(cos_sim(raw_P4, raw_tokens)) → attend(proj(tokens))
                    → resadd → upsample → mask     ← SAME Decoder weights!

    关键: Dense Softmax 使用 ALL support FG tokens (不采样), 在 RAW P4 特征上计算
          余弦相似度, 与 Dense Matching 诊断一致。

USAGE | 用法:
    python tools/diag/diag_causal_dense_softmax.py \
        --checkpoint runs/fewshot_f0_k5_0629_1639/decoder_sparsesupport_5shot_best.pt \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 5 --device cuda

Author: 2026-06-29
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adatile.backbone.fastsam_backbone import build_backbone
from adatile.utils.seed import set_seed
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS
from tools.instance.eval_c04_full_fewshot import build_decoder, binary_iou
from tools.train.train_fewshot import PreCutTileAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Dense Softmax Forward | Dense Softmax 前向
# ═══════════════════════════════════════════════════════════════════════════════

def dense_softmax_forward(decoder, query_p3, query_p4, support_tokens_raw, target_size):
    """
    Decoder forward using Dense Softmax instead of Cross-Attention.
    使用 Dense Softmax 替代 Cross-Attention 的 Decoder 前向。

    Replaces the trained Cross-Attention with zero-training cosine-similarity matching.
    ALL decoder weights (proj_p3, proj_p4, gate, token_proj, upsample, mask_head)
    are frozen and identical to the control condition.

    Key differences from decoder.forward():
        - NO random sampling → uses ALL support FG tokens
        - Attention = softmax(cos_sim(raw_P4, raw_tokens)) — NOT learned QKV
        - Attention computed on RAW P4 features (1280-dim), not projected (256-dim)
        - Value = token_proj(tokens) — same projection as trained decoder

    :param decoder: trained SparseSupportCrossAttnDecoder
    :param query_p3: [1, 960, H_p3, W_p3]
    :param query_p4: [1, 1280, H_p4, W_p4]
    :param support_tokens_raw: [N, 1280] ALL support FG pixel vectors
    :param target_size: (H, W) for output mask
    :return: logit [1, 1, H, W]
    """
    N = support_tokens_raw.shape[0]
    B, C_p3, H_p3, W_p3 = query_p3.shape

    # ── Step 1: P3+P4 fusion (IDENTICAL to trained decoder) ──
    f3 = decoder.proj_p3(query_p3)  # [1, 256, H_p3, W_p3]
    f4 = decoder.proj_p4(F.interpolate(
        query_p4, size=(H_p3, W_p3), mode="bilinear", align_corners=False
    ))  # [1, 256, H_p3, W_p3]

    proto_cond = support_tokens_raw.mean(dim=0)  # [1280]
    alpha = decoder.proto_gate_mlp(proto_cond)  # [256]
    fused = alpha[None, :, None, None] * f3 + (1 - alpha)[None, :, None, None] * f4

    # ── Step 2: Dense Softmax attention (REPLACES Cross-Attention) ──
    # Project support tokens → V (same trained projection)
    v_tokens = decoder.token_proj(support_tokens_raw)  # [N, 256]

    # Dense Softmax: cosine similarity on RAW P4 features
    # q_raw: [1, 1280, H_p4, W_p4] → [1, H_p4*W_p4, 1280]
    _, C_p4, H_p4, W_p4 = query_p4.shape
    q_raw_flat = query_p4.reshape(1, C_p4, -1).permute(0, 2, 1)  # [1, H_p4*W_p4, 1280]
    q_norm = F.normalize(q_raw_flat, dim=-1)
    s_norm = F.normalize(support_tokens_raw, dim=-1)  # [N, 1280]

    # Cosine similarity → softmax (temperature=1.0, same as Dense Matching diagnosis)
    cos_sim = q_norm @ s_norm.T  # [1, H_p4*W_p4, N]
    attn = cos_sim.softmax(dim=-1)  # [1, H_p4*W_p4, N]

    # Attend: aggregate V tokens by Dense Softmax weights
    attended = attn @ v_tokens  # [1, H_p4*W_p4, 256]
    attended = attended.permute(0, 2, 1).reshape(1, 256, H_p4, W_p4)  # [1, 256, H_p4, W_p4]

    # Upsample attended features to P3 resolution (to match Cross-Attention output)
    attended_up = F.interpolate(
        attended, size=(H_p3, W_p3), mode="bilinear", align_corners=False
    )  # [1, 256, H_p3, W_p3]

    # Residual add (same as Cross-Attention: q + attended)
    x = fused + attended_up

    # ── Step 3: Upsample decoder (IDENTICAL to trained decoder) ──
    x = decoder.up1(x)
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = decoder.up2(x)
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = decoder.up3(x)
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    x = decoder.mask_head(x)

    if target_size is not None:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment | 主实验
# ═══════════════════════════════════════════════════════════════════════════════

def run_causal_experiment(checkpoint_path, backbone, decoder, train_ds, val_ds,
                           target_classes, novel_ids, shot, device_str, n_eps=10):
    """
    For each episode, evaluate BOTH:
      A. Control: trained Cross-Attention
      B. Treatment: Dense Softmax (same decoder weights)

    Compare per-class IoU between conditions.
    """
    device = torch.device(device_str)
    rng = np.random.RandomState(42)

    # ── Load checkpoint ──
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    else:
        decoder.load_state_dict(ckpt)
    decoder.to(device)
    decoder.eval()
    print(f"Decoder loaded. {sum(p.numel() for p in decoder.parameters()):,} params")

    # ── Pre-sample episodes ──
    class_to_train = {c: train_ds.class_to_images(c) for c in target_classes}
    class_to_val = {c: val_ds.class_to_images(c) for c in target_classes}

    episodes = []
    for cls_id in sorted(target_classes):
        train_cands = class_to_train.get(cls_id, [])
        val_cands = class_to_val.get(cls_id, [])
        if len(train_cands) < shot or len(val_cands) < 1:
            continue
        for _ in range(n_eps):
            s_idxs = rng.choice(train_cands, shot, replace=False).tolist()
            q_idx = int(rng.choice(val_cands))
            episodes.append((cls_id, s_idxs, q_idx))

    print(f"Episodes: {len(episodes)} "
          f"({len(target_classes)} classes × ~{n_eps} eps, shot={shot})")

    # ── Per-condition results ──
    control_ious = defaultdict(list)    # Cross-Attention
    treatment_ious = defaultdict(list)  # Dense Softmax
    n_skipped = 0

    for ep_idx, (cls_id, s_idxs, q_idx) in enumerate(
        tqdm(episodes, desc="Causal experiment")
    ):
        cls_name = target_classes[cls_id]

        # ── Support ──
        support_imgs = torch.stack(
            [train_ds.load_image(si) for si in s_idxs]
        ).to(device)
        support_masks = [
            train_ds.render_class_mask(si, cls_id).to(device)
            for si in s_idxs
        ]

        # ── Query ──
        query_img = val_ds.load_image(q_idx).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(q_idx, cls_id).to(device)

        with torch.no_grad():
            # Backbone
            s_feats = backbone(support_imgs)
            q_feats = backbone(query_img)

            # Collect ALL support FG tokens (raw P4, no sampling)
            fg_tokens = []
            for i in range(len(support_imgs)):
                m = support_masks[i]
                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=s_feats["p4"].shape[2:], mode="nearest"
                ).squeeze() > 0.5
                if m_resized.sum() >= 4:
                    fg_tokens.append(
                        s_feats["p4"][i][:, m_resized].permute(1, 0)
                    )

        if not fg_tokens:
            n_skipped += 1
            continue

        all_tokens = torch.cat(fg_tokens, dim=0)  # [N_total, 1280]
        if all_tokens.shape[0] < 4:
            n_skipped += 1
            continue

        tsize = tuple(query_mask.shape)

        with torch.no_grad():
            # ── Control: Cross-Attention ──
            logit_ctrl = decoder(
                q_feats["p3"], q_feats["p4"], all_tokens,
                target_size=tsize,
            )

            # ── Treatment: Dense Softmax ──
            logit_trt = dense_softmax_forward(
                decoder, q_feats["p3"], q_feats["p4"], all_tokens,
                target_size=tsize,
            )

        # Compute IoU
        pred_ctrl = (logit_ctrl.squeeze(0).squeeze(0).cpu() > 0)
        pred_trt = (logit_trt.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)

        control_ious[cls_id].append(binary_iou(pred_ctrl, gt))
        treatment_ious[cls_id].append(binary_iou(pred_trt, gt))

    # ── Aggregate ──
    per_class = {}
    for cls_id in sorted(set(list(control_ious.keys()) + list(treatment_ious.keys()))):
        ctrl_vals = control_ious.get(cls_id, [])
        trt_vals = treatment_ious.get(cls_id, [])
        per_class[cls_id] = {
            "name": target_classes[cls_id],
            "control_mean": float(np.mean(ctrl_vals)) if ctrl_vals else 0.0,
            "control_std": float(np.std(ctrl_vals)) if ctrl_vals else 0.0,
            "treatment_mean": float(np.mean(trt_vals)) if trt_vals else 0.0,
            "treatment_std": float(np.std(trt_vals)) if trt_vals else 0.0,
            "delta": float(np.mean(trt_vals) - np.mean(ctrl_vals)) if trt_vals and ctrl_vals else 0.0,
            "n_ctrl": len(ctrl_vals),
            "n_trt": len(trt_vals),
        }

    # Base vs Novel split
    base_ctrl = [per_class[c]["control_mean"] for c in per_class if c not in novel_ids]
    base_trt = [per_class[c]["treatment_mean"] for c in per_class if c not in novel_ids]
    novel_ctrl = [per_class[c]["control_mean"] for c in per_class if c in novel_ids]
    novel_trt = [per_class[c]["treatment_mean"] for c in per_class if c in novel_ids]

    summary = {
        "base": {
            "control_miou": float(np.mean(base_ctrl)) if base_ctrl else 0.0,
            "treatment_miou": float(np.mean(base_trt)) if base_trt else 0.0,
            "delta": float(np.mean(base_trt) - np.mean(base_ctrl)) if base_ctrl and base_trt else 0.0,
        },
        "novel": {
            "control_miou": float(np.mean(novel_ctrl)) if novel_ctrl else 0.0,
            "treatment_miou": float(np.mean(novel_trt)) if novel_trt else 0.0,
            "delta": float(np.mean(novel_trt) - np.mean(novel_ctrl)) if novel_ctrl and novel_trt else 0.0,
        },
    }

    return {"per_class": per_class, "summary": summary, "n_episodes": len(episodes),
            "n_skipped": n_skipped}


def print_report(results: dict, target_classes: dict, novel_ids: list):
    """Print comparison report."""
    pc = results["per_class"]
    s = results["summary"]

    print(f"\n{'='*80}")
    print(f"  CAUSAL EXPERIMENT: Dense Softmax → Frozen Decoder")
    print(f"  因果实验: Dense Softmax → 冻结 Decoder")
    print(f"{'='*80}")

    print(f"\n  ── Per-Class Comparison | 按类别对比 ──")
    print(f"  {'Class':<22} {'Control':>8} {'DenseSM':>8} {'Δ':>8} {'Gain%':>8} {'Verdict':>12}")
    print(f"  {'-'*72}")

    for cls_id in sorted(pc.keys()):
        info = pc[cls_id]
        is_novel = "★N" if cls_id in novel_ids else ""
        gain_pct = (info["delta"] / max(info["control_mean"], 1e-6)) * 100
        if abs(gain_pct) < 10:
            verdict = "~ Decoder"
        elif gain_pct > 0:
            verdict = "↑ Attention"
        else:
            verdict = "↓ Attention"
        print(f"  {info['name']:<22} {info['control_mean']:>7.4f} {info['treatment_mean']:>7.4f} "
              f"{info['delta']:>+7.4f} {gain_pct:>+7.0f}% {verdict:>12} {is_novel}")

    print(f"\n  ── Base vs Novel | 基类 vs 新类 ──")
    for split in ["base", "novel"]:
        ss = s[split]
        gain_pct = (ss["delta"] / max(ss["control_miou"], 1e-6)) * 100
        print(f"  {split:<10} Control={ss['control_miou']:.4f}  "
              f"DenseSM={ss['treatment_miou']:.4f}  "
              f"Δ={ss['delta']:+.4f}  ({gain_pct:+.0f}%)")

    # ── Interpretation ──
    novel_gain = s["novel"]["delta"]
    novel_gain_pct = (novel_gain / max(s["novel"]["control_miou"], 1e-6)) * 100

    print(f"\n  ── Verdict | 判决 ──")
    if novel_gain_pct > 30:
        print(f"  ★★★ Attention IS the bottleneck (Novel Δ={novel_gain_pct:+.0f}%).")
        print(f"      Dense Softmax 显著优于 Cross-Attention.")
        print(f"      → 需要改进 Attention: Dense Distillation / Attention Supervision")
    elif novel_gain_pct > 10:
        print(f"  ★★☆  Attention is a PARTIAL bottleneck (Novel Δ={novel_gain_pct:+.0f}%).")
        print(f"      → 同时改进 Attention + Decoder")
    else:
        print(f"  ★☆☆ Decoder IS the bottleneck (Novel Δ={novel_gain_pct:+.0f}%).")
        print(f"      Dense Softmax 几乎没有提升，问题在 Decoder 或 Optimization.")
        print(f"      → 需要改进 Decoder 训练目标 (Contrastive / Consistency)")

    print(f"\n  N={results['n_episodes']} episodes, {results['n_skipped']} skipped")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Causal Experiment: Dense Softmax → Frozen Decoder | 因果实验"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tile-root", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-eps", type=int, default=10,
                        help="Episodes per class")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)

    # ── Datasets ──
    print("Building datasets...")
    train_ds = PreCutTileAdapter(args.tile_root, "train")
    val_ds = PreCutTileAdapter(args.tile_root, "val")

    # ── Classes ──
    fold_info = ISAID5I_FOLDS[args.fold]
    base_ids = fold_info["base"]
    novel_ids = fold_info["novel"]
    all_classes = base_ids + novel_ids
    target_classes = {cid: ISAID5I_CATEGORIES[cid] for cid in all_classes
                      if cid in ISAID5I_CATEGORIES}

    print(f"Base: {[target_classes[c] for c in base_ids if c in target_classes]}")
    print(f"Novel: {[target_classes[c] for c in novel_ids if c in target_classes]}")

    # ── Model ──
    device_t = torch.device(args.device)
    backbone = build_backbone("FastSAM-x").to(device_t)
    decoder = build_decoder(method="sparsesupport", feature_level="p3p4")

    # ── Run ──
    results = run_causal_experiment(
        checkpoint_path=args.checkpoint,
        backbone=backbone,
        decoder=decoder,
        train_ds=train_ds,
        val_ds=val_ds,
        target_classes=target_classes,
        novel_ids=novel_ids,
        shot=args.shot,
        device_str=args.device,
        n_eps=args.n_eps,
    )

    print_report(results, target_classes, novel_ids)

    # ── Save ──
    if args.output:
        out_dir = args.output
    else:
        out_dir = os.path.join(str(ROOT), "runs", "diag_causal_dense_softmax",
                               time.strftime("%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "causal_results.json")

    # Clean for JSON serialization
    serializable = {
        "config": {"checkpoint": args.checkpoint, "shot": args.shot, "fold": args.fold},
        "summary": results["summary"],
        "per_class": {str(k): v for k, v in results["per_class"].items()},
        "n_episodes": results["n_episodes"],
        "n_skipped": results["n_skipped"],
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
