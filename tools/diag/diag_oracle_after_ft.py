#!/usr/bin/env python3
"""
Dense Oracle Before/After Partial Fine-tune Comparison.
=========================================================
比较冻结 backbone vs partial-FT backbone 的 Dense Oracle P3 上限。

用法 | Usage:
    python tools/diag/diag_oracle_after_ft.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --checkpoint runs/fewshot_f0_k5_0629_2049/decoder_sparsesupport_5shot_best.pt \
        --fold 0 --device cuda
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, get_isaid5i_novel_classes
from adatile.utils.seed import set_seed
from tools.diag.diag_dense_matching import dense_matching, prototype_matching, compute_best_iou
from tools.train.train_fewshot import PreCutTileAdapter
from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Partial-FT checkpoint. None → measure frozen baseline.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--n-eps-per-class", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def measure_dense_oracle(backbone, dataset, class_id, shot, device, rng):
    """Single-episode Dense Oracle P3 measurement."""
    candidates = dataset.class_to_images(class_id)
    if len(candidates) < shot + 1:
        return None

    indices = rng.choice(candidates, shot + 1, replace=False)
    support_idxs = indices[:shot]
    query_idx = int(indices[shot])

    # Load support
    support_imgs, support_masks = [], []
    for si in support_idxs:
        img = dataset.load_image(int(si)).to(device)
        mask = dataset.render_class_mask(int(si), class_id).to(device)
        if dataset.crop_support and mask.sum() > 64:
            img, mask = dataset._roi_crop(img, mask)
        support_imgs.append(img)
        support_masks.append(mask)

    # Load query
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask = dataset.render_class_mask(int(query_idx), class_id).to(device)
    H_orig, W_orig = query_mask.shape

    # Backbone
    s_feats = backbone(torch.stack(support_imgs))
    q_feats = backbone(query_img)

    s_p3, s_p4 = s_feats["p3"], s_feats["p4"]
    q_p3, q_p4 = q_feats["p3"], q_feats["p4"]

    # Resize masks
    def resize_mask(mask, size):
        return F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                            size=size, mode="nearest").squeeze() > 0.5

    results = {"class_id": class_id}

    for level, s_fm, q_fm in [("p4", s_p4, q_p4), ("p3", s_p3, q_p3)]:
        s_mask_l = torch.stack([resize_mask(m, s_fm.shape[2:]) for m in support_masks]).to(device)
        q_mask_l = resize_mask(query_mask, q_fm.shape[2:])

        if s_mask_l.sum() < 10 or q_mask_l.sum() < 4:
            continue

        # Dense Softmax matching
        dense_map = dense_matching(s_fm, s_mask_l, q_fm, strategy="softmax")
        dense_full = F.interpolate(dense_map, size=(H_orig, W_orig),
                                   mode="bilinear", align_corners=False).squeeze()
        results[f"dense_{level}"] = compute_best_iou(dense_full, query_mask)["best_iou"]

        # Oracle: GT-guided max-cosine (OFCM)
        s_norm = F.normalize(s_fm, p=2, dim=1)
        q_norm = F.normalize(q_fm, p=2, dim=1)

        # Collect all support FG vectors
        s_fg_list = []
        for k in range(shot):
            if (s_mask_l[k] > 0.5).sum() > 0:
                s_fg_list.append(s_norm[k, :, s_mask_l[k] > 0.5])
        if not s_fg_list:
            continue
        s_fg_all = torch.cat(s_fg_list, dim=1)  # [C, N_total]

        # For each query FG pixel, find max cosine with any support FG pixel
        q_fg_vecs = q_norm[0, :, q_mask_l].permute(1, 0)  # [N_q_fg, C]
        oracle_map = torch.zeros(1, q_fm.shape[2] * q_fm.shape[3], device=device)
        q_mask_flat = q_mask_l.flatten()
        for q_start in range(0, q_fg_vecs.shape[0], 128):
            q_end = min(q_start + 128, q_fg_vecs.shape[0])
            sim_chunk = s_fg_all.T @ q_fg_vecs[q_start:q_end].T  # [N_total, chunk]
            fg_indices = torch.where(q_mask_flat)[0][q_start:q_end]
            oracle_map[0, fg_indices] = sim_chunk.max(dim=0)[0]
        oracle_map = oracle_map.reshape(1, 1, q_fm.shape[2], q_fm.shape[3])
        oracle_full = F.interpolate(oracle_map, size=(H_orig, W_orig),
                                    mode="bilinear", align_corners=False).squeeze()
        results[f"oracle_{level}"] = compute_best_iou(oracle_full, query_mask)["best_iou"]

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    novel_classes = get_isaid5i_novel_classes(args.fold)
    cat_names = ISAID5I_CATEGORIES

    # ── Load backbone ──
    print("[1] Loading backbone...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    # ── Load partial-FT checkpoint if provided ──
    if args.checkpoint:
        print(f"[2] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Restore backbone params (keys prefixed with "backbone.")
        backbone_params = {}
        for k, v in ckpt.items():
            if k.startswith("backbone."):
                idx = int(k[len("backbone."):])
                backbone_params[idx] = v
        if backbone_params:
            n_restored = 0
            for i, param in enumerate(backbone.model.model.parameters()):
                if i in backbone_params:
                    param.data.copy_(backbone_params[i])
                    n_restored += 1
            print(f"  Restored {n_restored:,} backbone params")
            # 标记 backbone 为非冻结以便 forward 使用 grad
            backbone._freeze_backbone = False
        else:
            print("  WARNING: No backbone params found in checkpoint!")
    else:
        print("[2] No checkpoint → measuring FROZEN baseline")

    # ── Load dataset ──
    print("[3] Loading dataset...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset

    train_tiles = PreCutTileAdapter(args.tile_root, "train")
    val_tiles = PreCutTileAdapter(args.tile_root, "val")

    novel_ds = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=args.shot, split="val",
        episodes_per_epoch=args.n_eps_per_class, seed=args.seed + 1,
        crop_support=False, novel_classes=novel_classes,
    )

    rng = np.random.RandomState(args.seed + 99)

    # ── Measure ──
    print(f"\n[4] Measuring Dense Oracle on {len(novel_classes)} Novel classes...")
    results = []
    per_class = defaultdict(list)

    for cls_id in novel_classes:
        cls_name = cat_names.get(cls_id, f"c{cls_id}")
        n_candidates = len(novel_ds.class_to_images(cls_id))
        if n_candidates < args.shot + 1:
            print(f"  [{cls_name}]: insufficient candidates ({n_candidates}), skip")
            continue

        for ep_i in tqdm(range(args.n_eps_per_class), desc=f"  [{cls_name:>20s}]"):
            r = measure_dense_oracle(backbone, novel_ds, cls_id, args.shot, device, rng)
            if r:
                results.append(r)
                per_class[cls_id].append(r)

    if not results:
        print("ERROR: No valid results!")
        return

    # ── Report ──
    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    print(f"\n{'='*65}")
    print(f"  DENSE ORACLE — {'Partial FT' if args.checkpoint else 'FROZEN'}")
    print(f"  Fold={args.fold}, Shot={args.shot}, Eps/class={args.n_eps_per_class}")
    print(f"{'='*65}")

    print(f"\n  {'Level':<10s} {'Dense':>8s} {'Oracle':>8s} {'Gap':>8s}")
    print(f"  {'─'*38}")
    for level in ["p4", "p3"]:
        d_key = f"dense_{level}"
        o_key = f"oracle_{level}"
        d_vals = [r[d_key] for r in results if d_key in r]
        o_vals = [r[o_key] for r in results if o_key in r]
        if d_vals and o_vals:
            d_avg = avg(d_vals)
            o_avg = avg(o_vals)
            print(f"  {level:<10s} {d_avg:>8.4f} {o_avg:>8.4f} {o_avg-d_avg:>+8.4f}")

    # Per-class
    print(f"\n  {'Class':<22s} {'Dense P4':>8s} {'Oracle P4':>8s} {'Dense P3':>8s} {'Oracle P3':>8s}")
    print(f"  {'─'*60}")
    for cls_id in sorted(per_class.keys()):
        items = per_class[cls_id]
        dp4 = avg([r.get("dense_p4", 0) for r in items])
        op4 = avg([r.get("oracle_p4", 0) for r in items])
        dp3 = avg([r.get("dense_p3", 0) for r in items])
        op3 = avg([r.get("oracle_p3", 0) for r in items])
        print(f"  {cat_names.get(cls_id, f'c{cls_id}'):<22s} {dp4:>8.4f} {op4:>8.4f} {dp3:>8.4f} {op3:>8.4f}")

    # Summary
    print(f"\n  {'─'*65}")
    print(f"  SUMMARY")
    dp4_avg = avg([r.get("dense_p4", 0) for r in results])
    op4_avg = avg([r.get("oracle_p4", 0) for r in results])
    dp3_avg = avg([r.get("dense_p3", 0) for r in results])
    op3_avg = avg([r.get("oracle_p3", 0) for r in results])
    print(f"  Dense P4 = {dp4_avg:.4f}  |  Oracle P4 = {op4_avg:.4f}  (gap={op4_avg-dp4_avg:+.4f})")
    print(f"  Dense P3 = {dp3_avg:.4f}  |  Oracle P3 = {op3_avg:.4f}  (gap={op3_avg-dp3_avg:+.4f})")
    print(f"  Oracle P4→P3 improvement: {op3_avg-op4_avg:+.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
