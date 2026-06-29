#!/usr/bin/env python3
"""
Fused Feature Space Diagnosis | 融合特征空间诊断.

================================================================================
THE LAST DIAGNOSIS | 最后一个诊断:
================================================================================

    "Cross-Attention 输出 (fused feature) 是否比原始 P4 更可分？"

    回答这个问题后，诊断阶段关闭，全力转向方法。

    对比三个特征空间:
        1. Pre-attn P4  — 原始 backbone 输出 (基准, 已知 Sil≈0.02)
        2. Post-CrossAttn — 训练后 Cross-Attention 输出 (核心问题)
        3. Post-DenseSM  — 零训练 Dense Softmax 输出 (参考上限)

    诊断逻辑:
        Post-CrossAttn ≈ Pre-attn P4  → 特征未改善 → Metric Learning
        Post-CrossAttn ≈ Post-DenseSM → 特征已改善 → Decoder/训练策略
        Post-CrossAttn 居中            → 方向对但不够 → Dense Distillation

USAGE | 用法:
    python tools/diag/diag_fused_feature.py \
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
from tools.instance.eval_c04_full_fewshot import build_decoder
from tools.train.train_fewshot import PreCutTileAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction | 特征提取
# ═══════════════════════════════════════════════════════════════════════════════

def dense_softmax_fused(decoder, query_p3, query_p4, support_tokens_raw):
    """
    计算 Dense Softmax 的 Cross-Attention 输出 (fused feature).
    Compute Dense Softmax fused feature (after attention, before upsample).

    Replaces learned Cross-Attention with zero-training cosine-similarity matching,
    using the SAME projection layers (proj_p3, proj_p4, gate, token_proj).
    """
    N = support_tokens_raw.shape[0]
    B, _, H_p3, W_p3 = query_p3.shape

    # P3+P4 fusion (same as trained)
    f3 = decoder.proj_p3(query_p3)
    f4 = decoder.proj_p4(F.interpolate(
        query_p4, size=(H_p3, W_p3), mode="bilinear", align_corners=False
    ))
    proto_cond = support_tokens_raw.mean(dim=0)
    alpha = decoder.proto_gate_mlp(proto_cond)
    fused_q = alpha[None, :, None, None] * f3 + (1 - alpha)[None, :, None, None] * f4

    # Project support tokens → V (same trained projection)
    v_tokens = decoder.token_proj(support_tokens_raw)  # [N, 256]

    # Dense Softmax: cosine similarity on RAW P4 features
    _, C_p4, H_p4, W_p4 = query_p4.shape
    q_raw_flat = query_p4.reshape(1, C_p4, -1).permute(0, 2, 1)  # [1, N_q, 1280]
    q_norm = F.normalize(q_raw_flat, dim=-1)
    s_norm = F.normalize(support_tokens_raw, dim=-1)  # [N, 1280]
    cos_sim = q_norm @ s_norm.T  # [1, N_q, N]
    attn = cos_sim.softmax(dim=-1)

    # Attend and reshape
    attended = attn @ v_tokens  # [1, N_q, 256]
    attended = attended.permute(0, 2, 1).reshape(1, 256, H_p4, W_p4)
    attended_up = F.interpolate(
        attended, size=(H_p3, W_p3), mode="bilinear", align_corners=False
    )
    return fused_q + attended_up  # [1, 256, H_p3, W_p3]


def extract_fg_features(feat_map, query_mask, target_size, max_per_class=5000):
    """
    从特征图中提取 FG 像素的特征向量。
    Extract FG pixel feature vectors from a feature map.

    :param feat_map: [1, C, H_f, W_f] feature map at some resolution
    :param query_mask: [H, W] GT binary mask at original resolution
    :param target_size: (H_f, W_f) target spatial size to resize mask to
    :param max_per_class: max FG pixels to collect (subsample if more)
    :return: [N_fg, C] tensor, or None if no FG pixels
    """
    # Resize mask to feature map resolution
    mask_resized = F.interpolate(
        query_mask.unsqueeze(0).unsqueeze(0).float(),
        size=target_size, mode="nearest"
    ).squeeze() > 0.5  # [H_f, W_f]

    if mask_resized.sum() < 4:
        return None

    C = feat_map.shape[1]
    fg_vecs = feat_map[0, :, mask_resized].permute(1, 0)  # [N_fg, C]

    # Subsample if too many
    if fg_vecs.shape[0] > max_per_class:
        indices = torch.randperm(fg_vecs.shape[0], device=fg_vecs.device)[:max_per_class]
        fg_vecs = fg_vecs[indices]

    return fg_vecs


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Space Metrics | 特征空间指标
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_metrics(features_dict, class_labels):
    """
    计算 Silhouette, DB Index, intra/inter distances.
    Compute Silhouette, Davies-Bouldin Index, intra/inter distances.

    :param features_dict: {class_name: np.ndarray [N, C]} — L2-normalized features
    :param class_labels: list of class names (ordered)
    :return: dict of metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    # Flatten into single matrix + labels
    all_feats = []
    all_labels = []
    for i, cls_name in enumerate(class_labels):
        if cls_name not in features_dict or features_dict[cls_name].shape[0] < 2:
            continue
        feats = features_dict[cls_name]
        all_feats.append(feats)
        all_labels.append(np.full(feats.shape[0], i, dtype=np.int32))

    if len(all_feats) < 2:
        return {"silhouette": 0.0, "db_index": 0.0, "n_classes": len(all_feats),
                "n_samples": 0, "error": "Insufficient classes"}

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    n_classes = len(all_feats)
    n_samples = X.shape[0]

    # Silhouette (higher = better separated)
    try:
        sil = float(silhouette_score(X, y, metric='cosine', random_state=42,
                                      sample_size=min(3000, n_samples)))
    except Exception:
        sil = 0.0

    # Davies-Bouldin (lower = better separated)
    try:
        db = float(davies_bouldin_score(X, y))
    except Exception:
        db = 999.0

    # ── Intra-class cosine similarity ──
    intra_sims = []
    for feats in all_feats:
        if feats.shape[0] < 2:
            continue
        # Random pairs for efficiency
        n = min(feats.shape[0], 200)
        idx = np.random.choice(feats.shape[0], n, replace=False)
        f = feats[idx]
        sim_matrix = f @ f.T  # cosine sim (already normalized)
        # Upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(n, k=1)
        intra_sims.extend(sim_matrix[triu_idx].tolist())

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    intra_std = float(np.std(intra_sims)) if intra_sims else 0.0

    # ── Inter-class cosine similarity ──
    inter_sims = []
    for i in range(len(all_feats)):
        for j in range(i + 1, len(all_feats)):
            # Centroids
            ci = all_feats[i].mean(axis=0)
            cj = all_feats[j].mean(axis=0)
            inter_sims.append(float(ci @ cj))

    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    inter_std = float(np.std(inter_sims)) if inter_sims else 0.0

    return {
        "silhouette": sil,
        "db_index": db,
        "n_classes": n_classes,
        "n_samples": n_samples,
        "intra_cos_mean": intra_mean,
        "intra_cos_std": intra_std,
        "inter_cos_mean": inter_mean,
        "inter_cos_std": inter_std,
        "separation_ratio": float(inter_mean / max(intra_mean, 1e-8)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# t-SNE Visualization | t-SNE 可视化
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tsne(features_dict, class_labels, title, output_path, n_per_class=300):
    """
    Generate t-SNE visualization for feature space.
    生成 t-SNE 可视化.

    :param features_dict: {class_name: np.ndarray [N, C]}
    :param class_labels: list of class names (ordered)
    :param title: plot title
    :param output_path: path to save PNG
    :param n_per_class: max samples per class for t-SNE
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Subsample
    sampled_feats = []
    sampled_labels = []
    for i, cls_name in enumerate(class_labels):
        if cls_name not in features_dict:
            continue
        feats = features_dict[cls_name]
        n = min(feats.shape[0], n_per_class)
        if n < 2:
            continue
        idx = np.random.choice(feats.shape[0], n, replace=False)
        sampled_feats.append(feats[idx])
        sampled_labels.extend([cls_name] * n)

    if len(sampled_feats) < 2:
        print(f"  ⚠ Not enough data for t-SNE: {title}")
        return

    X = np.concatenate(sampled_feats, axis=0)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0] - 1),
                random_state=42, max_iter=1000, metric='cosine')
    X_2d = tsne.fit_transform(X)

    # Plot
    unique_labels = sorted(set(sampled_labels))
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, label in enumerate(unique_labels):
        mask = np.array(sampled_labels) == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap(i)], label=label,
                   alpha=0.6, s=8, edgecolors='none')
    ax.legend(markerscale=3, fontsize=8, loc='lower left', ncol=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Diagnosis | 主诊断
# ═══════════════════════════════════════════════════════════════════════════════

def run_fused_feature_diagnosis(checkpoint_path, backbone, decoder, train_ds, val_ds,
                                 target_classes, novel_ids, shot, device_str,
                                 n_eps=10, max_fg=5000):
    """
    采集三个特征空间的 FG 像素特征, 计算指标 + t-SNE.
    Collect FG pixel features from 3 feature spaces, compute metrics + t-SNE.
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

    # ── Storage for collected features ──
    # pre_p4[class_name] = list of [N, 1280] tensors
    # post_crossattn[class_name] = list of [N, 256] tensors
    # post_densesm[class_name] = list of [N, 256] tensors
    pre_p4 = defaultdict(list)
    post_crossattn = defaultdict(list)
    post_densesm = defaultdict(list)

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

    # ── Collect features per episode ──
    n_skipped = 0
    total_fg = defaultdict(int)

    for ep_idx, (cls_id, s_idxs, q_idx) in enumerate(
        tqdm(episodes, desc="Collecting features")
    ):
        cls_name = target_classes[cls_id]

        # Support
        support_imgs = torch.stack(
            [train_ds.load_image(si) for si in s_idxs]
        ).to(device)
        support_masks = [
            train_ds.render_class_mask(si, cls_id).to(device)
            for si in s_idxs
        ]

        # Query
        query_img = val_ds.load_image(q_idx).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(q_idx, cls_id).to(device)

        with torch.no_grad():
            s_feats = backbone(support_imgs)
            q_feats = backbone(query_img)

            # Collect ALL support FG tokens (raw P4)
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

        with torch.no_grad():
            # ── 1. Pre-attn P4 features ──
            p4_feat = q_feats["p4"]  # [1, 1280, H_p4, W_p4]
            p4_fg = extract_fg_features(
                p4_feat, query_mask,
                target_size=p4_feat.shape[2:], max_per_class=max_fg
            )

            # ── 2. Post-CrossAttn fused features ──
            logit, fused_crossattn = decoder(
                q_feats["p3"], q_feats["p4"], all_tokens,
                target_size=tuple(query_mask.shape),
                return_fused=True,
            )
            crossattn_fg = extract_fg_features(
                fused_crossattn, query_mask,
                target_size=fused_crossattn.shape[2:], max_per_class=max_fg
            )

            # ── 3. Post-DenseSM fused features ──
            fused_densesm = dense_softmax_fused(
                decoder, q_feats["p3"], q_feats["p4"], all_tokens
            )
            densesm_fg = extract_fg_features(
                fused_densesm, query_mask,
                target_size=fused_densesm.shape[2:], max_per_class=max_fg
            )

        # Store (move to CPU, normalize)
        if p4_fg is not None and p4_fg.shape[0] >= 4:
            pre_p4[cls_name].append(F.normalize(p4_fg, dim=-1).cpu().numpy())
            total_fg[cls_name] += p4_fg.shape[0]

        if crossattn_fg is not None and crossattn_fg.shape[0] >= 4:
            post_crossattn[cls_name].append(F.normalize(crossattn_fg, dim=-1).cpu().numpy())
        else:
            post_crossattn[cls_name].append(None)  # placeholder

        if densesm_fg is not None and densesm_fg.shape[0] >= 4:
            post_densesm[cls_name].append(F.normalize(densesm_fg, dim=-1).cpu().numpy())
        else:
            post_densesm[cls_name].append(None)

    # ── Merge per-class features ──
    def merge_features(feat_dict):
        """Concatenate all episodes for each class."""
        merged = {}
        for cls_name, feat_list in feat_dict.items():
            valid = [f for f in feat_list if f is not None and f.shape[0] >= 4]
            if valid:
                merged[cls_name] = np.concatenate(valid, axis=0)
            else:
                merged[cls_name] = np.zeros((0, 1))  # empty placeholder
        return merged

    pre_p4_merged = merge_features(pre_p4)
    post_crossattn_merged = merge_features(post_crossattn)
    post_densesm_merged = merge_features(post_densesm)

    print(f"\nFeature collection done. {n_skipped} episodes skipped.")
    for cls_name in sorted(pre_p4_merged.keys()):
        print(f"  {cls_name:<22}: pre_p4={pre_p4_merged[cls_name].shape[0]:>6d}  "
              f"post_ca={post_crossattn_merged[cls_name].shape[0]:>6d}  "
              f"post_ds={post_densesm_merged[cls_name].shape[0]:>6d}")

    # ── Compute metrics ──
    class_list = sorted(pre_p4_merged.keys())

    print(f"\n{'='*70}")
    print(f"  FEATURE SPACE METRICS | 特征空间指标")
    print(f"{'='*70}")

    def compute_and_print(merged_dict, space_name):
        metrics = compute_feature_metrics(merged_dict, class_list)
        print(f"\n  ── {space_name} ──")
        print(f"    Silhouette (cosine)  = {metrics['silhouette']:.4f}  "
              f"(higher=better separated)")
        print(f"    Davies-Bouldin Index  = {metrics['db_index']:.2f}  "
              f"(lower=better separated)")
        print(f"    Intra-class cos_sim   = {metrics['intra_cos_mean']:.4f} ± "
              f"{metrics['intra_cos_std']:.4f}")
        print(f"    Inter-class cos_sim   = {metrics['inter_cos_mean']:.4f} ± "
              f"{metrics['inter_cos_std']:.4f}")
        print(f"    Separation Ratio      = {metrics['separation_ratio']:.3f}  "
              f"(inter/intra, higher=better)")
        print(f"    N_classes={metrics['n_classes']}, N_samples={metrics['n_samples']}")
        return metrics

    m_pre = compute_and_print(pre_p4_merged, "Pre-attn P4 (1280-dim)")
    m_ca = compute_and_print(post_crossattn_merged, "Post-CrossAttn (256-dim)")
    m_ds = compute_and_print(post_densesm_merged, "Post-DenseSM (256-dim)")

    # ── Base vs Novel breakdown ──
    novel_names = {target_classes[c] for c in novel_ids if c in target_classes}

    def compute_split(merged_dict, split_name, split_names):
        subset = {k: v for k, v in merged_dict.items() if k in split_names}
        subset_classes = sorted(subset.keys())
        if len(subset_classes) < 2:
            return None
        return compute_feature_metrics(subset, subset_classes)

    for split_name, split_names in [("Base", set(class_list) - novel_names),
                                     ("Novel", novel_names)]:
        m_base = compute_split(post_crossattn_merged, f"Post-CrossAttn {split_name}",
                               split_names)
        if m_base:
            print(f"\n  ── Post-CrossAttn [{split_name}] ──")
            print(f"    Sil={m_base['silhouette']:.4f}  DB={m_base['db_index']:.2f}  "
                  f"SepRatio={m_base['separation_ratio']:.3f}")

    # ── Generate t-SNE ──
    out_dir = os.path.join(str(ROOT), "runs", "diag_fused_feature",
                           time.strftime("%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  t-SNE VISUALIZATION | t-SNE 可视化")
    print(f"{'='*70}")

    generate_tsne(
        pre_p4_merged, class_list,
        f"Pre-attn P4 Features (Sil={m_pre['silhouette']:.3f})",
        os.path.join(out_dir, "tsne_pre_p4.png")
    )
    generate_tsne(
        post_crossattn_merged, class_list,
        f"Post-CrossAttn Features (Sil={m_ca['silhouette']:.3f})",
        os.path.join(out_dir, "tsne_post_crossattn.png")
    )
    generate_tsne(
        post_densesm_merged, class_list,
        f"Post-DenseSM Features (Sil={m_ds['silhouette']:.3f})",
        os.path.join(out_dir, "tsne_post_densesm.png")
    )

    # ── Save metrics ──
    results = {
        "config": {"checkpoint": checkpoint_path, "shot": shot, "fold": args.fold},
        "pre_p4": m_pre,
        "post_crossattn": m_ca,
        "post_densesm": m_ds,
    }
    # Clean for JSON
    for m in [m_pre, m_ca, m_ds]:
        for k in list(m.keys()):
            if isinstance(m[k], (np.floating,)):
                m[k] = float(m[k])
            elif isinstance(m[k], (np.integer,)):
                m[k] = int(m[k])

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics saved → {metrics_path}")
    print(f"Plots saved → {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fused Feature Space Diagnosis | 融合特征空间诊断"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tile-root", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-eps", type=int, default=10,
                        help="Episodes per class for feature collection")
    args = parser.parse_args()

    set_seed(42)

    print("Building datasets...")
    train_ds = PreCutTileAdapter(args.tile_root, "train")
    val_ds = PreCutTileAdapter(args.tile_root, "val")

    fold_info = ISAID5I_FOLDS[args.fold]
    base_ids = fold_info["base"]
    novel_ids = fold_info["novel"]
    all_classes = base_ids + novel_ids
    target_classes = {cid: ISAID5I_CATEGORIES[cid] for cid in all_classes
                      if cid in ISAID5I_CATEGORIES}

    print(f"Base ({len(base_ids)}): "
          f"{[target_classes[c] for c in base_ids if c in target_classes]}")
    print(f"Novel ({len(novel_ids)}): "
          f"{[target_classes[c] for c in novel_ids if c in target_classes]}")

    device_t = torch.device(args.device)
    backbone = build_backbone("FastSAM-x").to(device_t)
    decoder = build_decoder(method="sparsesupport", feature_level="p3p4")

    run_fused_feature_diagnosis(
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


if __name__ == "__main__":
    main()
