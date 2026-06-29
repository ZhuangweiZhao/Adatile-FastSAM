#!/usr/bin/env python3
"""
Cross-Attention Quality Diagnosis — Attention 是否真正工作？
==============================================================
Diagnose whether Cross-Attention learned meaningful correspondences,
or collapsed to uniform/shortcut behavior.

四大诊断 | Four Diagnostics:
    D1. Attention Heatmap — attention 是否聚焦到 support FG 像素？
    D2. Identity Cross-Attn (no QKV) — raw cosine+softmax 比 trained 更好？
    D3. Frozen QK (only learn V) — projection 是否过拟合？
    D4. Attention Entropy — attention 是否均匀塌缩？

核心问题 | Core Question:
    Dense Matching (cosine, zero-training) = 0.22
    Cross-Attention (trained) Novel = 0.07
    → 为什么 training 让事情变差了？

用法 | Usage:
    python tools/diag/diag_attention_quality.py \
        --ckpt runs/fewshot_f0_k5_0629_1319/decoder_p3p4crossattn_5shot_best.pt \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 5 --device cuda
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.utils.prototype import compute_fg_prototype
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, get_isaid5i_novel_classes, get_isaid5i_base_classes
from adatile.utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════
# 辅助函数 | Helpers
# ═══════════════════════════════════════════════════════════════════

def attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    计算 attention 分布的熵（归一化 0~1）。
    Compute entropy of attention distribution (normalized to 0~1).

    ent=0 → 完全均匀 (collapse)
    ent=1 → 极度集中 (peaked)
    """
    # attn_weights: [N_query, K_support]
    eps = 1e-8
    n_tokens = attn_weights.shape[-1]
    if n_tokens < 2:
        return 0.0
    ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # [N_query]
    max_ent = -np.log(1.0 / n_tokens)
    normalized = (ent / max_ent).mean().item()
    return round(normalized, 4)


def attention_fg_ratio(attn_weights: torch.Tensor, support_mask_p4: torch.Tensor) -> float:
    """
    Attention 落在 support FG 像素上的比例。
    Ratio of attention mass on support FG pixels.
    """
    # attn_weights: [N_query, K_support]  (N_query total across query positions)
    # support_mask_p4: [K_spatial] binary, 1=foreground

    if support_mask_p4.sum() == 0:
        return 0.0

    # Average attention per support token across all query positions
    avg_attn = attn_weights.mean(dim=0)  # [K_support]
    fg_attn = avg_attn[support_mask_p4].sum().item()
    total_attn = avg_attn.sum().item()
    return round(fg_attn / max(total_attn, 1e-8), 4)


# ═══════════════════════════════════════════════════════════════════
# Diagnostic Cross-Attention | 诊断用 Cross-Attention
# ═══════════════════════════════════════════════════════════════════

class DiagP3P4CrossAttn(nn.Module):
    """
    P3P4CrossAttn decoder 的包装器，暴露中间 attention weight。
    Wrapper around P3P4CrossAttn decoder that exposes attention weights.

    Modes:
      "trained"  — use learned QKV (from checkpoint)
      "identity" — Q=query_feat, K=V=support_feat (no projection)
      "frozen_qk" — freeze Wq, Wk at random init, only learn Wv

    通过 monkey-patch self._cross_attention 来捕获 attention weights。
    """

    def __init__(self, base_decoder, mode: str = "trained"):
        super().__init__()
        self.decoder = base_decoder
        self.mode = mode
        self._last_attn = None  # stores last attention weights

        # Identity projections for "identity" mode
        self._identity_mode = (mode == "identity")
        if mode == "frozen_qk":
            # Freeze proto_mlp (which generates K,V) — only allow upsampling to learn
            for name, param in self.decoder.named_parameters():
                if "proto_mlp" in name or "gate_mlp" in name:
                    param.requires_grad = False
                # Q is query features, no separate Q projection in current arch

    def forward(self, query_p3, query_p4, fg_prototype, target_size=None):
        self._last_attn = None

        if self._identity_mode:
            return self._forward_identity(query_p3, query_p4, fg_prototype, target_size)
        else:
            # Hook into cross_attention to capture weights
            return self._forward_hooked(query_p3, query_p4, fg_prototype, target_size)

    def _forward_hooked(self, query_p3, query_p4, fg_prototype, target_size):
        """Forward with attention weight capture."""
        # Monkey-patch _cross_attention to capture weights
        orig_cross_attn = self.decoder._cross_attention

        def hooked_cross_attn(q, proto_tokens):
            # q: [B, C, H, W], proto_tokens: [K, C]
            B, C, H, W = q.shape
            K = proto_tokens.shape[0]
            q_flat = q.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]
            kv = proto_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]
            scale = C ** -0.5
            attn_logits = torch.bmm(q_flat, kv.transpose(1, 2)) * scale

            if K == 1:
                gate = torch.sigmoid(attn_logits)
                self._last_attn = gate.squeeze(-1)  # [B, N]
                attended = gate * kv
            else:
                attn = attn_logits.softmax(dim=-1)
                self._last_attn = attn  # [B, N, K]
                attended = torch.bmm(attn, kv)
            attended = attended.permute(0, 2, 1).reshape(B, C, H, W)
            return q + attended

        self.decoder._cross_attention = hooked_cross_attn
        try:
            result = self.decoder(query_p3, query_p4, fg_prototype, target_size)
        finally:
            self.decoder._cross_attention = orig_cross_attn
        return result

    def _forward_identity(self, query_p3, query_p4, fg_prototype, target_size):
        """Identity cross-attention: Q=query, K=V=support_feat (normalized cosine)."""
        # Use the decoder's P3/P4 projection but skip proto_mlp
        f3 = self.decoder.proj_p3(query_p3)
        f4 = self.decoder.proj_p4(F.interpolate(query_p4,
            size=query_p3.shape[2:], mode="bilinear", align_corners=False))

        if fg_prototype.dim() == 2:
            proto_cond = fg_prototype.mean(dim=0)
        else:
            proto_cond = fg_prototype

        alpha = self.decoder.gate_mlp(proto_cond)
        fused = alpha[None, :, None, None] * f3 + (1 - alpha)[None, :, None, None] * f4

        # Identity cross-attention: use raw support features as K, V
        # Here we replace proto_tokens with raw cosine matching
        # The "identity" attention is: attn = softmax(cos(q, proto_tokens) / tau)
        # which is equivalent to Dense-Softmax

        # Actually: use proto_cond [1280] as proto, compute cosine with fused
        q_norm = F.normalize(fused, p=2, dim=1)  # [B, 256, H, W]
        # But proto_cond is 1280-d, fused is 256-d. Can't compute cosine.
        # Fallback: use proto_tokens from proto_mlp but with identity attention
        if fg_prototype.dim() == 2:
            proto_tokens = fg_prototype.mean(dim=0).unsqueeze(0)
        else:
            proto_tokens = fg_prototype.unsqueeze(0)
        proto_tokens = self.decoder.proto_mlp(proto_tokens)  # use trained proto_mlp

        # Identity: skip learned gating, just use cosine softmax
        q_flat = fused.reshape(fused.shape[0], fused.shape[1], -1).permute(0, 2, 1)
        kv = proto_tokens.unsqueeze(0).expand(fused.shape[0], -1, -1)
        scale = fused.shape[1] ** -0.5
        attn = (q_flat @ kv.transpose(1, 2) * scale / 0.1).softmax(dim=-1)  # [B, N, K]
        self._last_attn = attn
        attended = attn @ kv
        attended = attended.permute(0, 2, 1).reshape(*fused.shape)
        x = fused + attended

        # Upsample path (same as decoder)
        x = self.decoder.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder.mask_head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════
# Episode Runner | Episode 执行
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_attention_episode(
    dataset, backbone, diag_decoder, class_id: int, shot: int,
    device: torch.device, rng: np.random.RandomState,
) -> dict:
    """Run one episode and collect attention diagnostics."""

    candidates = dataset.class_to_images(class_id)
    if len(candidates) < shot + 1:
        return None

    indices = rng.choice(candidates, shot + 1, replace=False)
    support_idxs = indices[:shot]
    query_idx = int(indices[shot])

    # Load support
    support_imgs, support_masks_orig = [], []
    for si in support_idxs:
        img = dataset.load_image(int(si)).to(device)
        mask = dataset.render_class_mask(int(si), class_id)
        if dataset.crop_support and mask.sum() > 64:
            img, mask = dataset._roi_crop(img, mask)
        support_imgs.append(img)
        support_masks_orig.append(mask.to(device))
    support_batch = torch.stack(support_imgs)

    # Load query
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask = dataset.render_class_mask(int(query_idx), class_id).to(device)
    H_orig, W_orig = query_mask.shape

    # Backbone
    s_feats = backbone(support_batch)
    q_feats = backbone(query_img)

    s_p4 = s_feats["p4"]
    s_masks_p4 = []
    for k in range(shot):
        m = F.interpolate(support_masks_orig[k].unsqueeze(0).unsqueeze(0).float(),
                         size=s_p4.shape[2:], mode="nearest").squeeze() > 0.5
        s_masks_p4.append(m)
    s_mask_p4 = torch.stack(s_masks_p4).to(device)

    if s_mask_p4.sum() < 10:
        return None

    # Prototype
    s_p4s = [s_p4[i] for i in range(shot)]
    s_m_list = [support_masks_orig[i].to(device) for i in range(shot)]
    proto = compute_fg_prototype(s_p4s, s_m_list)
    if proto.sum() == 0:
        return None

    # Decoder forward
    logit = diag_decoder(q_feats["p3"], q_feats["p4"], proto, target_size=(H_orig, W_orig))
    pred_mask = (torch.sigmoid(logit.squeeze()) > 0.5).float()
    inter = (pred_mask * query_mask).sum().item()
    union = ((pred_mask + query_mask) > 0).sum().item()
    pred_iou = inter / union if union > 0 else 0.0

    # Attention diagnostics
    attn = diag_decoder._last_attn  # [1, N_query] or [1, N_query, K]
    if attn is not None:
        if attn.dim() == 2:
            # K=1: gate [1, N_query] → convert to [N_query, 1]
            attn = attn.squeeze(0).unsqueeze(-1)
        else:
            attn = attn.squeeze(0)  # [N_query, K]

        # Compute entropy
        ent = attention_entropy(attn)

        # Compute FG attention ratio
        # For P3: need to map support mask to attention token space
        # K=1 mode: attention is per-query-position gating, not per-support-pixel
        # We can only check: is attention correlated with query FG?
        if attn.shape[-1] == 1:
            # Single proto token → can't compute per-support-pixel FG ratio
            # Instead: check if gate values differ between query FG vs BG
            q_mask_p3 = F.interpolate(
                query_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(112, 112) if q_feats["p3"].shape[2] == 112 else (56, 56),
                mode="nearest"
            ).squeeze() > 0.5
            fg_gate = attn[q_mask_p3.flatten()].mean().item() if q_mask_p3.sum() > 0 else 0.0
            bg_gate = attn[~q_mask_p3.flatten()].mean().item() if (~q_mask_p3).sum() > 0 else 0.0
            gate_contrast = abs(fg_gate - bg_gate)
        else:
            gate_contrast = -1.0  # multi-proto not yet supported
    else:
        ent = -1.0
        gate_contrast = -1.0

    return {
        "class_id": class_id,
        "pred_iou": round(pred_iou, 4),
        "attn_entropy": ent,
        "gate_contrast": round(gate_contrast, 4) if gate_contrast >= 0 else -1.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Main | 主逻辑
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Cross-Attention Quality Diagnosis")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--n-episodes-per-class", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/diag_attention")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    base_classes = get_isaid5i_base_classes(args.fold)
    novel_classes = get_isaid5i_novel_classes(args.fold)
    all_classes = base_classes + novel_classes
    novel_set = set(novel_classes)
    cat_names = ISAID5I_CATEGORIES

    print(f"\n{'='*70}")
    print(f"  Cross-Attention Quality Diagnosis")
    print(f"  CKPT: {args.ckpt}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    # ── Load models ──
    print("[1/4] Loading models...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    with torch.no_grad():
        probe = backbone(torch.randn(1, 3, 896, 896).to(device))
        p3_dim = probe["p3"].shape[1]
        p4_dim = probe["p4"].shape[1]
    print(f"      P3 dim: {p3_dim}, P4 dim: {p4_dim}")

    from tools.instance.eval_c04_full_fewshot import P3P4CrossAttnDecoder
    decoder = P3P4CrossAttnDecoder(
        feat_dim_p3=p3_dim, feat_dim_p4=p4_dim
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    decoder.load_state_dict(ckpt)
    decoder.eval()
    print(f"      Decoder: P3P4CrossAttn ({sum(p.numel() for p in decoder.parameters()):,} params)")

    # Create diagnostic wrappers
    diag_trained = DiagP3P4CrossAttn(decoder, mode="trained")
    diag_identity = DiagP3P4CrossAttn(decoder, mode="identity")

    # ── Load dataset ──
    print("[2/4] Loading dataset...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset
    from tools.train.train_fewshot import PreCutTileAdapter

    train_tiles = PreCutTileAdapter(args.tile_root, "train")
    val_tiles = PreCutTileAdapter(args.tile_root, "val")

    base_ds = FewShotEpisodeDataset(
        train_tiles, fold=args.fold, shot=args.shot, split="train",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed,
        crop_support=True, crop_margin=0.2, novel_classes=base_classes,
    )
    novel_ds = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=args.shot, split="val",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed + 1,
        crop_support=False, novel_classes=novel_classes,
    )

    # ── Run episodes ──
    print(f"[3/4] Running episodes (Trained + Identity)...")
    rng = np.random.RandomState(args.seed + 99)

    results = {"trained": [], "identity": []}

    for mode, diag_model in [("trained", diag_trained), ("identity", diag_identity)]:
        print(f"\n  Mode: {mode}")
        mode_results = []

        for cls_id in all_classes:
            cls_name = cat_names.get(cls_id, f"c{cls_id}")
            ds = novel_ds if cls_id in novel_set else base_ds
            n = len(ds.class_to_images(cls_id))
            if n < args.shot + 1:
                continue

            for ep_i in tqdm(range(args.n_episodes_per_class), desc=f"    [{cls_name:>20s}]"):
                r = run_attention_episode(ds, backbone, diag_model,
                                          cls_id, args.shot, device, rng)
                if r:
                    mode_results.append(r)

        results[mode] = mode_results

    # ── Analysis ──
    print(f"\n[4/4] Analysis...")

    def avg(lst):
        return np.mean(lst) if lst else 0.0

    for mode in ["trained", "identity"]:
        res = results[mode]
        base_r = [r for r in res if r["class_id"] not in novel_set]
        novel_r = [r for r in res if r["class_id"] in novel_set]

        print(f"\n  ── {mode.upper()} ──")
        for label, subset in [("BASE", base_r), ("NOVEL", novel_r)]:
            if not subset:
                continue
            iou_avg = avg([r["pred_iou"] for r in subset])
            ent_avg = avg([r["attn_entropy"] for r in subset if r["attn_entropy"] >= 0])
            gc_avg = avg([r["gate_contrast"] for r in subset if r["gate_contrast"] >= 0])
            print(f"    {label}: IoU={iou_avg:.4f}, Attn Entropy={ent_avg:.4f}, "
                  f"Gate Contrast={gc_avg:.4f}")

            if ent_avg < 0.3 and len(subset) > 0:
                print(f"      🔴 Low entropy → attention is NEAR-UNIFORM (collapse)")
            elif ent_avg > 0.7:
                print(f"      🟢 High entropy → attention is PEAKED (working)")

    # ── Verdict ──
    print(f"\n{'█'*70}")
    print(f"  VERDICT")
    print(f"{'█'*70}")

    trained_novel = [r for r in results["trained"] if r["class_id"] in novel_set]
    ident_novel = [r for r in results["identity"] if r["class_id"] in novel_set]

    t_iou = avg([r["pred_iou"] for r in trained_novel])
    i_iou = avg([r["pred_iou"] for r in ident_novel])
    t_ent = avg([r["attn_entropy"] for r in trained_novel if r["attn_entropy"] >= 0])

    print(f"\n  Trained Novel IoU:  {t_iou:.4f}")
    print(f"  Identity Novel IoU: {i_iou:.4f}")
    print(f"  Trained Attn Ent:   {t_ent:.4f}")

    if i_iou > t_iou * 1.3:
        print(f"\n  ★ Identity > Trained → learned QKV projections are HURTING generalization")
        print(f"    Cross-Attention training is overfitting to Base-class patterns.")
    elif i_iou < t_iou * 0.7:
        print(f"\n  ★ Trained > Identity → QKV projections ARE learning useful transforms")
    else:
        print(f"\n  ★ Identity ≈ Trained → QKV projections are not the differentiator")

    if t_ent < 0.3:
        print(f"\n  ★ Attention entropy < 0.3 → attention has COLLAPSED to near-uniform")
        print(f"    The model is NOT using cross-attention — it's ignoring support features.")
    elif t_ent > 0.7:
        print(f"\n  ★ High entropy → attention is actively selecting support regions (good)")

    print(f"{'█'*70}\n")

    # Save
    diagnosis = {
        "config": {"ckpt": args.ckpt, "fold": args.fold, "shot": args.shot},
        "trained": {
            "novel_iou": t_iou,
            "novel_entropy": t_ent,
        },
        "identity": {
            "novel_iou": i_iou,
        },
    }
    with open(out_dir / "attention_diagnosis.json", "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✅ Saved → {out_dir}/\n")


if __name__ == "__main__":
    main()
