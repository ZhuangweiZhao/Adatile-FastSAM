#!/usr/bin/env python3
"""
SparseSupport Cross-Attention Quality Diagnosis.
稀疏支持交叉注意力质量诊断.

================================================================================
PURPOSE | 目的:
================================================================================

统计已训练的 SparseSupportCrossAttnDecoder 的 Cross-Attention 质量，
回答核心问题:

    "Cross-Attention 学会 correspondence 了吗？
     瓶颈在 Attention 还是在 Decoder？"

Measures on trained checkpoints (no training, inference only):
    ⚠ 在已训练 checkpoint 上测量 (仅推理, 不训练):
    1. Attention Entropy — 注意力是软融合(distributed)还是退化为一选一(one-hot)?
    2. Oracle Similarity — 训练后的 attention 与 Dense Softmax (特征余弦相似度) 的 KL/相关性
    3. Effective Token Count — 实际参与了多少个 support token?
    4. Attention-FG Consistency — 同类 query FG pixel 的 attention 是否一致?
    5. Per-Class Breakdown — 注意力质量是否与最终 IoU 相关?

KEY DIAGNOSTIC LOGIC | 核心诊断逻辑:
================================================================================

    IF   attention entropy ≈ 0 (one-hot, collapsed)
    THEN attention IS the bottleneck → need Attention Supervision

    IF   attention entropy > 0.3 (distributed, soft)
         AND trained attention ≈ oracle attention
    THEN attention is fine → bottleneck is Decoder/Optimization

    IF   attention quality correlates with per-class IoU
    THEN attention quality drives final performance

    IF   attention quality is flat across classes but IoU varies widely
    THEN decoder/optimization is the bottleneck

REFERENCE | 参考:
    - Dense Softmax diagnosis: Novel 15.2% (zero-training, cosine matching)
    - KMeans Proto CrossAttn: Entropy=0.000 (one-hot collapse)
    - SparseSupport 5-shot: Base SWA=30.94%, Novel=8.71%

USAGE | 用法:
    python tools/diag/diag_attention_sparse.py \
        --checkpoint runs/fewshot_f0_k5_0629_1639/best_model.pt \
        --dataset isaid5i \
        --src-root /root/Adatile-FastSAM/data/iSAID_processed \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 5 \
        --device cuda

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

# ── Project imports | 项目导入 ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adatile.backbone.fastsam_backbone import build_backbone
from adatile.utils.seed import set_seed
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS
from tools.instance.eval_c04_full_fewshot import (
    SparseSupportCrossAttnDecoder,
    build_decoder,
    _extract_support_tokens,
    binary_iou,
)
from tools.train.train_fewshot import PreCutTileAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Metrics | 注意力指标
# ═══════════════════════════════════════════════════════════════════════════════

def compute_attention_entropy(attn_map: torch.Tensor, eps: float = 1e-8):
    """
    计算归一化注意力熵 | Compute normalized attention entropy.

    H_norm = -sum(p * log(p)) / log(K)

    :param attn_map: [H, W, K] attention weights (already softmaxed)
    :return: {mean, std, per_pixel[H,W]} dict
    """
    K = attn_map.shape[-1]
    if K <= 1:
        return {"mean": 0.0, "std": 0.0, "per_pixel": torch.zeros_like(attn_map[..., 0])}

    # H = -sum(p * log(p)) / log(K) ∈ [0, 1]
    log_p = (attn_map + eps).log()
    entropy = -(attn_map * log_p).sum(dim=-1)  # [H, W]
    entropy_norm = entropy / np.log(K)

    return {
        "mean": float(entropy_norm.mean()),
        "std": float(entropy_norm.std()),
        "per_pixel": entropy_norm,
    }


def compute_effective_tokens(attn_map: torch.Tensor, eps: float = 1e-8):
    """
    计算有效 token 数 | Compute effective number of tokens.

    eff = 1 / sum(p^2)  (inverse Simpson index / inverse Herfindahl)

    :param attn_map: [H, W, K]
    :return: {mean, std} dict
    """
    K = attn_map.shape[-1]
    # Inverse Simpson index: 1 / sum(p_i^2)
    simpson = 1.0 / (attn_map ** 2).sum(dim=-1).clamp(min=eps)  # [H, W]
    # Normalize to [0, 1] by dividing by K
    eff_norm = simpson / K

    return {
        "mean": float(eff_norm.mean()),
        "std": float(eff_norm.std()),
        "raw_mean": float(simpson.mean()),
        "raw_std": float(simpson.std()),
        "K": K,
    }


def compute_oracle_attention(query_feat_raw, support_tokens_raw, temperature: float = 1.0):
    """
    计算 Oracle Attention: Dense Softmax (cosine similarity + softmax).
    Compute Oracle Attention via Dense Softmax matching.

    这是零训练的、纯粹基于特征的匹配 — 与 Dense Matching 诊断一致。
    Zero-training, purely feature-based matching — consistent with Dense Matching diagnosis.

    :param query_feat_raw: [1, C, H, W] raw backbone features (P4, 1280-dim)
    :param support_tokens_raw: [N, C] raw backbone features for FG pixels
    :param temperature: softmax temperature (default 1.0)
    :return: [H, W, N] oracle attention map
    """
    C = query_feat_raw.shape[1]
    H, W = query_feat_raw.shape[2], query_feat_raw.shape[3]

    # L2-normalize for cosine similarity
    q_flat = query_feat_raw.reshape(1, C, -1).permute(0, 2, 1)  # [1, H*W, C]
    q_norm = F.normalize(q_flat, dim=-1)
    s_norm = F.normalize(support_tokens_raw, dim=-1)  # [N, C]

    # Cosine similarity → softmax
    sim = (q_norm @ s_norm.T) / temperature  # [1, H*W, N]
    oracle_attn = sim.softmax(dim=-1)  # [1, H*W, N]

    return oracle_attn.reshape(H, W, -1)  # [H, W, N]


def compute_kl_divergence(trained_attn: torch.Tensor, oracle_attn: torch.Tensor,
                          eps: float = 1e-8):
    """
    KL(trained || oracle) — how much does trained attention diverge from oracle?
    KL(训练注意力 || Oracle注意力) — 训练后的注意力偏离 Oracle 多远?

    :param trained_attn: [H, W, K]
    :param oracle_attn: [H, W, K]
    :return: {mean, std} dict
    """
    # Clamp to avoid log(0)
    t = trained_attn.clamp(min=eps)
    o = oracle_attn.clamp(min=eps)

    kl = (t * (t.log() - o.log())).sum(dim=-1)  # [H, W]

    return {
        "mean": float(kl.mean()),
        "std": float(kl.std()),
    }


def compute_rank_correlation(trained_attn: torch.Tensor, oracle_attn: torch.Tensor):
    """
    Spearman rank correlation between trained and oracle attention weights.
    训练注意力与 Oracle 注意力的 Spearman 秩相关系数.

    对每个 query pixel, 计算两种 attention 在 K 个 token 上的排序相关性。
    高相关性 → 训练后的 attention 保留了特征的拓扑结构。
    Low correlation → training distorted the feature correspondence.

    :param trained_attn: [H, W, K]
    :param oracle_attn: [H, W, K]
    :return: {mean, std} dict
    """
    H, W, K = trained_attn.shape

    # Spearman: rank correlation of the K values for each pixel
    # Efficient approximation: Pearson correlation of ranks
    def spearman_per_pixel(x, y):
        """x, y: [K]"""
        # Get ranks
        x_rank = x.argsort().argsort().float()
        y_rank = y.argsort().argsort().float()
        # Pearson r of ranks
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()
        denom = (x_centered.norm() * y_centered.norm()).clamp(min=1e-8)
        return (x_centered * y_centered).sum() / denom

    correlations = []
    for h in range(H):
        for w in range(W):
            r = spearman_per_pixel(trained_attn[h, w], oracle_attn[h, w])
            correlations.append(float(r))

    correlations = np.array(correlations)

    return {
        "mean": float(np.mean(correlations)),
        "std": float(np.std(correlations)),
        "median": float(np.median(correlations)),
    }


def compute_topk_overlap(trained_attn: torch.Tensor, oracle_attn: torch.Tensor,
                         top_k: int = 10):
    """
    Top-K overlap: 训练注意力的 Top-K token 中有多少也在 Oracle 的 Top-K 中?
    Top-K overlap: fraction of trained Top-K tokens also in Oracle Top-K.

    :param trained_attn: [H, W, K]
    :param oracle_attn: [H, W, K]
    :param top_k: K for Top-K comparison
    :return: {mean, std} dict
    """
    H, W, K = trained_attn.shape
    actual_k = min(top_k, K)

    trained_top = trained_attn.topk(actual_k, dim=-1).indices  # [H, W, K]
    oracle_top = oracle_attn.topk(actual_k, dim=-1).indices    # [H, W, K]

    overlaps = []
    for h in range(H):
        for w in range(W):
            overlap = len(set(trained_top[h, w].tolist()) & set(oracle_top[h, w].tolist()))
            overlaps.append(overlap / actual_k)

    overlaps = np.array(overlaps)

    return {
        "mean": float(np.mean(overlaps)),
        "std": float(np.std(overlaps)),
        "top_k": actual_k,
    }


def compute_attention_consistency(attn_map: torch.Tensor, query_mask: torch.Tensor,
                                   window: int = 4):
    """
    空间一致性: 相邻 query FG pixel 的 attention 分布是否相似?
    Spatial consistency: do adjacent query FG pixels have similar attention?

    用滑动窗口内的 attention 分布方差来衡量。
    Measures variance of attention within a sliding window.

    :param attn_map: [H, W, K]
    :param query_mask: [H, W] bool — FG pixels
    :param window: window size for local consistency
    :return: {fg_consistency, bg_consistency} dict
    """
    H, W, K = attn_map.shape

    # AveragePool to get local mean attention
    # F.pad then pool; crop to original size (padding can cause +1 on odd dims)
    pad = window // 2
    attn_4d = attn_map.permute(2, 0, 1).unsqueeze(0)  # [1, K, H, W]
    attn_padded = F.pad(attn_4d, (pad, pad, pad, pad), mode='reflect')
    local_mean = F.avg_pool2d(attn_padded, kernel_size=window, stride=1)
    local_mean = local_mean[:, :, :H, :W]  # crop to original size
    local_mean = local_mean.squeeze(0).permute(1, 2, 0)  # [H, W, K]

    # Local variance: ||attn - local_mean||^2
    local_var = ((attn_map - local_mean) ** 2).sum(dim=-1)  # [H, W]

    fg_mask = query_mask > 0
    bg_mask = ~fg_mask

    result = {
        "fg_mean": float(local_var[fg_mask].mean()) if fg_mask.any() else 0.0,
        "fg_std": float(local_var[fg_mask].std()) if fg_mask.any() else 0.0,
        "bg_mean": float(local_var[bg_mask].mean()) if bg_mask.any() else 0.0,
        "bg_std": float(local_var[bg_mask].std()) if bg_mask.any() else 0.0,
    }
    return result


def compute_fg_attention_peakiness(attn_map: torch.Tensor, query_mask: torch.Tensor):
    """
    FG 像素的注意力峰值度 | FG pixel attention peakiness.

    peakiness = max(attn) / mean(attn) — 衡量注意力是否集中于少数 token.
    过高 → 退化为一选一 (one-hot collapse).
    适中 → 软融合 (soft matching).

    :return: {mean, std} dict
    """
    fg_mask = query_mask > 0
    if not fg_mask.any():
        return {"mean": 0.0, "std": 0.0}

    fg_attn = attn_map[fg_mask]  # [N_FG, K]
    peakiness = fg_attn.max(dim=-1).values / fg_attn.mean(dim=-1).clamp(min=1e-8)

    return {
        "mean": float(peakiness.mean()),
        "std": float(peakiness.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Diagnosis Pipeline | 主诊断流程
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_attention(checkpoint_path: str, backbone, decoder, train_ds, val_ds,
                       target_classes: dict, shot: int, device: str,
                       n_episodes_per_class: int = 10, seed: int = 42):
    """
    对已训练模型进行完整的 Attention 诊断。
    Run complete attention diagnosis on a trained model.

    For each episode:
        1. Extract support FG tokens (raw P4 features)
        2. Backbone forward on query → raw P4 features
        3. Decoder forward with return_attn=True → trained attention + prediction
        4. Compute Oracle (Dense Softmax) attention
        5. Compute all metrics: entropy, effective tokens, KL, rank corr, top-K overlap
        6. Record per-class IoU for correlation analysis

    Returns all metrics organized by class and episode.
    """
    rng = np.random.RandomState(seed)
    device_t = torch.device(device)

    # ── Load checkpoint | 加载模型权重 ──
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)

    # Handle both formats: raw state_dict vs wrapped {"decoder": ..., "epoch": ...}
    if isinstance(ckpt, dict) and "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
        print(f"Checkpoint loaded (wrapped). Epoch={ckpt.get('epoch', '?')}, "
              f"val_mIoU={ckpt.get('val_miou', '?')}")
    else:
        decoder.load_state_dict(ckpt)
        print(f"Checkpoint loaded (raw state_dict). Keys: {len(ckpt)} params")

    # ── Pre-sample episodes | 预采样 episode ──
    class_to_train = {c: train_ds.class_to_images(c) for c in target_classes}
    class_to_val = {c: val_ds.class_to_images(c) for c in target_classes}

    episodes = []
    for cls_id in sorted(target_classes):
        train_cands = class_to_train.get(cls_id, [])
        val_cands = class_to_val.get(cls_id, [])
        if len(train_cands) < shot or len(val_cands) < 1:
            print(f"  ⚠ Class {cls_id} ({target_classes[cls_id]}): insufficient data, skip")
            continue
        for _ in range(n_episodes_per_class):
            s_idxs = rng.choice(train_cands, shot, replace=False).tolist()
            q_idx = int(rng.choice(val_cands))
            episodes.append((cls_id, s_idxs, q_idx))

    print(f"Episodes: {len(episodes)} ({len(target_classes)} classes × "
          f"~{n_episodes_per_class} eps)")

    # ── Per-episode metrics collection | 逐 episode 收集指标 ──
    all_results = []

    for ep_idx, (cls_id, s_idxs, q_idx) in enumerate(
        tqdm(episodes, desc="Diagnosing")
    ):
        cls_name = target_classes[cls_id]

        # ── Support: load + extract FG tokens ──
        support_imgs = torch.stack(
            [train_ds.load_image(si) for si in s_idxs]
        ).to(device_t)
        support_masks = [
            train_ds.render_class_mask(si, cls_id).to(device_t)
            for si in s_idxs
        ]

        # Extract raw P4 features and FG pixel tokens
        with torch.no_grad():
            s_feats = backbone(support_imgs)

            # Collect FG pixel features (raw, before any projection)
            fg_tokens_raw = []  # raw P4 features
            fg_tokens_info = []  # (support_img_idx, pixel_count)
            for i in range(len(support_imgs)):
                m = support_masks[i]
                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=s_feats["p4"].shape[2:], mode="nearest"
                ).squeeze() > 0.5
                if m_resized.sum() >= 4:
                    tokens_i = s_feats["p4"][i][:, m_resized].permute(1, 0)  # [N_i, 1280]
                    fg_tokens_raw.append(tokens_i)
                    fg_tokens_info.append((i, tokens_i.shape[0]))

        if not fg_tokens_raw:
            continue

        all_fg_tokens = torch.cat(fg_tokens_raw, dim=0)  # [N_total, 1280]
        if all_fg_tokens.shape[0] < 4:
            continue

        # ── Query: backbone + decoder forward ──
        query_img = val_ds.load_image(q_idx).unsqueeze(0).to(device_t)
        query_mask = val_ds.render_class_mask(q_idx, cls_id).to(device_t)

        with torch.no_grad():
            q_feats = backbone(query_img)

            # Ensure support tokens are on the same device as decoder
            # 确保 support tokens 与 decoder 在同一设备上
            decoder_device = next(decoder.parameters()).device
            tokens_dev = all_fg_tokens.to(decoder_device)

            # Decoder forward with attention extraction
            logit, trained_attn = decoder(
                q_feats["p3"].to(decoder_device),
                q_feats["p4"].to(decoder_device),
                tokens_dev,
                target_size=tuple(query_mask.shape),
                return_attn=True,
            )

        # ── Oracle Attention (Dense Softmax) ──
        oracle_attn = compute_oracle_attention(
            q_feats["p4"], all_fg_tokens
        )  # [H_p4, W_p4, N_total]

        # ── Align resolutions for comparison ──
        # trained_attn is at P3 resolution (H/8, W/8)
        # oracle_attn is at P4 resolution (H/16, W/16)
        # oracle_attn needs to be upsampled to trained_attn's spatial resolution
        #   for per-pixel comparison
        # But they operate on different K dimensions (trained has K≤128 after sampling,
        # oracle has N_total). For comparison, we take the same sampled tokens.

        # Get the sampled token indices (after _sample_tokens):
        # The decoder samples internally — we need to know which tokens were selected.
        # Solution: re-run the sampling deterministically (same randperm seed won't work
        # since we're in no_grad). Instead, compute oracle attention only over the
        # K tokens that were actually selected in decoder's forward.
        #
        # ACTUALLY: Let's compute metrics on trained attention in its own space,
        # and oracle attention in its own space, then compare the statistics
        # rather than pixel-aligned KL.

        # ── Compute metrics | 计算指标 ──
        pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)
        iou = binary_iou(pred, gt)

        # Trained attention metrics (on P3 resolution)
        t_attn = trained_attn[0]  # [H_p3, W_p3, K≤128]
        t_H, t_W, t_K = t_attn.shape

        # Resize query mask to P3 resolution for FG/BG masking
        # Keep on same device as t_attn to avoid CPU/CUDA mismatch
        q_mask_p3 = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(t_H, t_W), mode="nearest"
        ).squeeze() > 0.5
        q_mask_p3 = q_mask_p3.to(t_attn.device)

        # 1. Entropy
        entropy_metrics = compute_attention_entropy(t_attn)

        # 2. Effective tokens
        eff_metrics = compute_effective_tokens(t_attn)

        # 3. Peakiness
        peak_metrics = compute_fg_attention_peakiness(t_attn, q_mask_p3)

        # 4. Spatial consistency
        consistency_metrics = compute_attention_consistency(t_attn, q_mask_p3)

        # 5. Oracle attention + comparison
        o_attn = oracle_attn  # [H_p4, W_p4, N]
        o_H, o_W, o_N = o_attn.shape

        # Resize oracle to P3 resolution for comparison
        o_attn_p3 = F.interpolate(
            o_attn.permute(2, 0, 1).unsqueeze(0),  # [1, N, H_p4, W_p4]
            size=(t_H, t_W), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # [H_p3, W_p3, N]

        # Since trained and oracle have different K dimensions (K≤128 vs N_total),
        # we compute intrinsic quality metrics separately and compare.
        # The key comparison is:
        #   - Trained entropy vs Oracle entropy → did training collapse or preserve diversity?
        #   - Per-class patterns → does attention quality predict IoU?

        oracle_entropy_metrics = compute_attention_entropy(o_attn_p3)
        oracle_eff_metrics = compute_effective_tokens(o_attn_p3)
        oracle_peak_metrics = compute_fg_attention_peakiness(
            o_attn_p3, q_mask_p3.to(o_attn_p3.device))

        # ── Record results | 记录结果 ──
        result = {
            "ep": ep_idx,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "iou": float(iou),
            "n_support_tokens": int(all_fg_tokens.shape[0]),
            "n_sampled_tokens": t_K,
            # Trained attention
            "trained": {
                "entropy": entropy_metrics,
                "effective_tokens": eff_metrics,
                "peakiness": peak_metrics,
                "consistency": consistency_metrics,
            },
            # Oracle (Dense Softmax) attention
            "oracle": {
                "entropy": oracle_entropy_metrics,
                "effective_tokens": oracle_eff_metrics,
                "peakiness": oracle_peak_metrics,
            },
        }
        all_results.append(result)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation & Reporting | 聚合与报告
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: list, target_classes: dict,
                       novel_classes: list = None):
    """聚合诊断结果, 按类 + Base/Novel 分组. | Aggregate results by class and Base/Novel."""

    # Per-class aggregation
    per_class = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        c = r["cls_name"]
        per_class[c]["iou"].append(r["iou"])
        per_class[c]["trained_entropy"].append(r["trained"]["entropy"]["mean"])
        per_class[c]["oracle_entropy"].append(r["oracle"]["entropy"]["mean"])
        per_class[c]["trained_eff"].append(r["trained"]["effective_tokens"]["mean"])
        per_class[c]["oracle_eff"].append(r["oracle"]["effective_tokens"]["mean"])
        per_class[c]["trained_peak"].append(r["trained"]["peakiness"]["mean"])
        per_class[c]["oracle_peak"].append(r["oracle"]["peakiness"]["mean"])

    # Compute per-class means
    per_class_summary = {}
    for c, metrics in sorted(per_class.items()):
        per_class_summary[c] = {
            "iou": float(np.mean(metrics["iou"])),
            "iou_std": float(np.std(metrics["iou"])),
            "trained_entropy": float(np.mean(metrics["trained_entropy"])),
            "oracle_entropy": float(np.mean(metrics["oracle_entropy"])),
            "entropy_ratio": float(np.mean(metrics["trained_entropy"]) /
                                   np.mean(metrics["oracle_entropy"]).clip(min=1e-8)),
            "trained_eff": float(np.mean(metrics["trained_eff"])),
            "oracle_eff": float(np.mean(metrics["oracle_eff"])),
            "trained_peak": float(np.mean(metrics["trained_peak"])),
            "oracle_peak": float(np.mean(metrics["oracle_peak"])),
            "n_eps": len(metrics["iou"]),
        }

    # Global means
    all_entropy_t = [r["trained"]["entropy"]["mean"] for r in all_results]
    all_entropy_o = [r["oracle"]["entropy"]["mean"] for r in all_results]
    all_eff_t = [r["trained"]["effective_tokens"]["mean"] for r in all_results]
    all_eff_o = [r["oracle"]["effective_tokens"]["mean"] for r in all_results]
    all_peak_t = [r["trained"]["peakiness"]["mean"] for r in all_results]
    all_peak_o = [r["oracle"]["peakiness"]["mean"] for r in all_results]

    global_summary = {
        "n_episodes": len(all_results),
        "trained_entropy_mean": float(np.mean(all_entropy_t)),
        "trained_entropy_std": float(np.std(all_entropy_t)),
        "oracle_entropy_mean": float(np.mean(all_entropy_o)),
        "oracle_entropy_std": float(np.std(all_entropy_o)),
        "trained_eff_mean": float(np.mean(all_eff_t)),
        "trained_eff_std": float(np.std(all_eff_t)),
        "oracle_eff_mean": float(np.mean(all_eff_o)),
        "oracle_eff_std": float(np.std(all_eff_o)),
        "trained_peak_mean": float(np.mean(all_peak_t)),
        "trained_peak_std": float(np.std(all_peak_t)),
        "oracle_peak_mean": float(np.mean(all_peak_o)),
        "oracle_peak_std": float(np.std(all_peak_o)),
    }

    # Base vs Novel split
    base_vs_novel = None
    if novel_classes:
        novel_names = {novel_classes.get(c, f"class_{c}") for c in novel_classes}
        base_metrics = defaultdict(list)
        novel_metrics = defaultdict(list)
        for c, summary in per_class_summary.items():
            target = novel_metrics if c in novel_names else base_metrics
            for k, v in summary.items():
                target[k].append(v)
        base_vs_novel = {
            "base": {k: float(np.mean(v)) for k, v in base_metrics.items()
                     if k not in ("n_eps",)},
            "novel": {k: float(np.mean(v)) for k, v in novel_metrics.items()
                      if k not in ("n_eps",)},
        }

    return {
        "global": global_summary,
        "per_class": per_class_summary,
        "base_vs_novel": base_vs_novel,
    }


def print_report(aggregated: dict, target_classes: dict, novel_classes: list = None):
    """打印人类可读的诊断报告. | Print human-readable diagnosis report."""

    print(f"\n{'='*70}")
    print(f"  ATTENTION DIAGNOSIS REPORT | 注意力诊断报告")
    print(f"{'='*70}")

    g = aggregated["global"]
    print(f"\n  ── Global Metrics | 全局指标 (N={g['n_episodes']} episodes) ──")
    print(f"  {'Metric':<35} {'Trained':>10} {'Oracle':>10} {'Ratio':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Entropy (norm)':<35} {g['trained_entropy_mean']:>10.4f} "
          f"{g['oracle_entropy_mean']:>10.4f} "
          f"{g['trained_entropy_mean']/max(g['oracle_entropy_mean'],1e-8):>10.3f}")
    print(f"  {'Effective Tokens (norm)':<35} {g['trained_eff_mean']:>10.4f} "
          f"{g['oracle_eff_mean']:>10.4f} "
          f"{g['trained_eff_mean']/max(g['oracle_eff_mean'],1e-8):>10.3f}")
    print(f"  {'Peakiness':<35} {g['trained_peak_mean']:>10.2f} "
          f"{g['oracle_peak_mean']:>10.2f} "
          f"{g['trained_peak_mean']/max(g['oracle_peak_mean'],1e-8):>10.3f}")

    print(f"\n  ── Per-Class Breakdown | 按类别分解 ──")
    print(f"  {'Class':<22} {'IoU':>8} {'T_Ent':>7} {'O_Ent':>7} "
          f"{'E_Ratio':>7} {'T_Eff':>7} {'O_Eff':>7} {'T_Peak':>7} {'O_Peak':>7}")
    print(f"  {'-'*85}")

    pc = aggregated["per_class"]
    for cls_name in sorted(pc.keys()):
        s = pc[cls_name]
        is_novel = ""
        if novel_classes:
            novel_names = {novel_classes.get(c, f"class_{c}") for c in novel_classes}
            if cls_name in novel_names:
                is_novel = " ★N"
        print(f"  {cls_name:<22} {s['iou']:>7.4f} {s['trained_entropy']:>7.4f} "
              f"{s['oracle_entropy']:>7.4f} {s['entropy_ratio']:>7.3f} "
              f"{s['trained_eff']:>7.4f} {s['oracle_eff']:>7.4f} "
              f"{s['trained_peak']:>7.1f} {s['oracle_peak']:>7.1f}{is_novel}")

    # Base vs Novel summary
    if aggregated["base_vs_novel"]:
        bn = aggregated["base_vs_novel"]
        print(f"\n  ── Base vs Novel Summary | 基类 vs 新类 ──")
        print(f"  {'Split':<10} {'IoU':>8} {'T_Ent':>7} {'O_Ent':>7} "
              f"{'E_Ratio':>7} {'T_Eff':>7} {'O_Eff':>7}")
        print(f"  {'-'*62}")
        for split in ["base", "novel"]:
            s = bn[split]
            print(f"  {split:<10} {s['iou']:>8.4f} {s['trained_entropy']:>7.4f} "
                  f"{s['oracle_entropy']:>7.4f} {s['entropy_ratio']:>7.3f} "
                  f"{s['trained_eff']:>7.4f} {s['oracle_eff']:>7.4f}")

    # ── Diagnostic interpretation | 诊断解读 ──
    print(f"\n  ── Interpretation | 诊断解读 ──")

    # Entropy check
    if g["trained_entropy_mean"] < 0.05:
        print(f"  ⚠ CRITICAL: Trained attention entropy={g['trained_entropy_mean']:.4f} — "
              f"ONE-HOT COLLAPSE!")
        print(f"    Cross-Attention 退化为一选一，与 KMeans Proto 问题相同。")
        print(f"    → Attention IS the bottleneck. 需要 Attention Supervision.")
    elif g["trained_entropy_mean"] < 0.2:
        print(f"  ⚠ WARNING: Trained attention entropy={g['trained_entropy_mean']:.4f} — "
              f"moderately peaked.")
        print(f"    Cross-Attention 有一定软融合能力但偏硬。")
        print(f"    → Attention may be a partial bottleneck.")
    else:
        print(f"  ✓ OK: Trained attention entropy={g['trained_entropy_mean']:.4f} — "
              f"distributed attention.")
        print(f"    Cross-Attention 在做软融合。")
        print(f"    → Attention is NOT the bottleneck. 查 Decoder/Optimization.")

    # Entropy ratio (trained vs oracle)
    e_ratio = g["trained_entropy_mean"] / max(g["oracle_entropy_mean"], 1e-8)
    if e_ratio < 0.3:
        print(f"  ⚠ Entropy ratio trained/oracle={e_ratio:.3f} — training COLLAPSED attention!")
        print(f"    Oracle (Dense Softmax) has much richer attention distribution.")
        print(f"    Training is destroying the feature correspondence.")
    elif e_ratio < 0.7:
        print(f"  ~ Entropy ratio trained/oracle={e_ratio:.3f} — training moderately "
              f"sharpened attention.")
    else:
        print(f"  ✓ Entropy ratio trained/oracle={e_ratio:.3f} — training preserved "
              f"attention diversity.")

    # IoU-Entropy correlation
    ious = [pc[c]["iou"] for c in pc]
    ents = [pc[c]["trained_entropy"] for c in pc]
    if len(ious) > 2:
        corr = np.corrcoef(ious, ents)[0, 1]
        print(f"\n  ── IoU vs Attention Quality Correlation | IoU 与注意力质量的相关性 ──")
        print(f"  Pearson r(IoU, Trained Entropy) = {corr:+.3f}")
        if abs(corr) > 0.5:
            print(f"  → 强相关: 注意力质量直接影响最终 IoU。Attention IS a bottleneck.")
        elif abs(corr) > 0.3:
            print(f"  → 中等相关: 注意力质量部分影响 IoU。")
        else:
            print(f"  → 弱相关: 注意力质量不是 IoU 差异的主要解释变量。")
            print(f"    → Bottleneck is likely in Decoder or Optimization, NOT Attention.")

    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI | 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SparseSupport Cross-Attention Quality Diagnosis | 注意力质量诊断"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="isaid5i",
                        help="Dataset name")
    parser.add_argument("--src-root", type=str, required=True,
                        help="Path to iSAID_processed directory")
    parser.add_argument("--tile-root", type=str, required=True,
                        help="Path to tile directory")
    parser.add_argument("--tile-size", type=int, default=896,
                        help="Tile size")
    parser.add_argument("--fold", type=int, default=0,
                        help="iSAID-5i fold (0-3)")
    parser.add_argument("--shot", type=int, default=5,
                        help="K-shot setting")
    parser.add_argument("--feature-level", type=str, default="p3p4",
                        choices=["p4", "p3p4"],
                        help="Feature level for decoder")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Episodes per class for diagnosis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results JSON")
    args = parser.parse_args()

    set_seed(42)

    # ── Build datasets | 构建数据集 ──
    print("Building datasets...")
    # Use PreCutTileAdapter — same as train_fewshot.py
    train_ds = PreCutTileAdapter(args.tile_root, "train")
    val_ds = PreCutTileAdapter(args.tile_root, "val")

    # ── Build model | 构建模型 ──
    print("Building model...")
    device_t = torch.device(args.device)
    backbone = build_backbone("FastSAM-x").to(device_t)
    decoder = build_decoder(
        method="sparsesupport",
        feature_level="p3p4",
    )

    # ── Target classes | 目标类别 ──
    # Use iSAID-5i official folds
    fold_info = ISAID5I_FOLDS[args.fold]
    base_ids = fold_info["base"]
    novel_ids = fold_info["novel"]

    # All 15 classes (Base + Novel)
    all_classes = base_ids + novel_ids
    target_classes = {cid: ISAID5I_CATEGORIES[cid] for cid in all_classes
                      if cid in ISAID5I_CATEGORIES}
    novel_classes = {cid: ISAID5I_CATEGORIES[cid] for cid in novel_ids
                     if cid in ISAID5I_CATEGORIES}

    print(f"\nTarget classes: {len(target_classes)}")
    print(f"Novel classes: {len(novel_classes)}: "
          f"{[novel_classes[c] for c in sorted(novel_classes)]}")

    # ── Run diagnosis | 运行诊断 ──
    all_results = diagnose_attention(
        checkpoint_path=args.checkpoint,
        backbone=backbone,
        decoder=decoder,
        train_ds=train_ds,
        val_ds=val_ds,
        target_classes=target_classes,
        shot=args.shot,
        device=args.device,
        n_episodes_per_class=args.n_episodes,
    )

    # ── Aggregate & Report | 聚合 & 报告 ──
    aggregated = aggregate_results(all_results, target_classes, novel_classes)
    print_report(aggregated, target_classes, novel_classes)

    # ── Save results | 保存结果 ──
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "attention_diagnosis.json")
    else:
        out_dir = os.path.join(str(ROOT), "runs", "diag_attention_sparse",
                               time.strftime("%m%d_%H%M"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "attention_diagnosis.json")

    # Convert to serializable format
    serializable = {
        "config": {
            "checkpoint": args.checkpoint,
            "shot": args.shot,
            "fold": args.fold,
            "n_episodes_per_class": args.n_episodes,
            "feature_level": args.feature_level,
        },
        "global": aggregated["global"],
        "per_class": aggregated["per_class"],
        "base_vs_novel": aggregated["base_vs_novel"],
        # Raw per-episode data (for later detailed analysis)
        "raw_episodes": [
            {k: v for k, v in r.items() if k != "trained" and k != "oracle"}
            for r in all_results
        ],
    }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
