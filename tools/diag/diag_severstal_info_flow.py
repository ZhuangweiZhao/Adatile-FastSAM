#!/usr/bin/env python3
"""
Decoder 信息流审计 | Decoder Information Flow Audit.
======================================================

逐级追踪 Prototype → Coeff → ProtoMask → Refined → Final 的信号衰减。
Trace signal decay stage-by-stage through the decoder pipeline.

回答 | Answers:
    Q1: ProtoCoeffPredictor 是否为不同类别产生不同的 coefficient？
        → coeff cross-class cosine, coeff std, PCA
    Q2: Coefficient 差异是否传导到 Proto Mask？
        → proto_mask cross-class cosine / IoU
    Q3: P4 Refinement 的贡献占比多少？是否覆盖了 Proto Mask？
        → |refined| / |proto_mask| ratio, ablation (proto-only vs P4-only vs full)
    Q4: 信号在哪一级消失？
        → 逐级 Δ 表，精确定位瓶颈

用法 | Usage::

    python tools/diag/diag_severstal_info_flow.py \
        --checkpoint runs/.../best_model.pt \
        --data-root /root/autodl-tmp/severstal-steel-defect-detection \
        --k-shot 3 --num-queries 30 --device cuda

输出 | Output:
    diag_output/severstal_info_flow_{ts}/
    ├── info_flow_report.json
    ├── coeff_pca.png              # 32-d coeff PCA (per class)
    ├── stage_cosine_decay.png     # 逐级余弦相似度衰减曲线
    ├── branch_contribution.png    # Proto vs P4 分支贡献比
    └── proto_mask_per_class.png   # 每类 proto_mask 统计
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from adatile.backbone import FastSAMBackbone
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.datasets.severstal import SeverstalDataset

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 256, 1600
DEFECT_CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}
CLASS_COLORS = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#F39C12"}


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def get_class_mask(sample: dict, class_id: int) -> torch.Tensor:
    mask = sample["masks"].squeeze(0)
    if mask.max() > 1:
        return (mask == class_id).float()
    return mask.float()


@torch.no_grad()
def compute_prototype(backbone, images, masks, device):
    """从 support images 计算 prototype (同训练逻辑)."""
    N = images.shape[0]
    feats_list = []
    for i in range(N):
        img = images[i:i + 1].to(device)
        mask = masks[i].to(device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        feats = backbone(img)
        p4 = feats["p4"]
        _, _, H_p4, W_p4 = p4.shape
        mask_p4 = F.interpolate(
            mask.unsqueeze(0).float(), size=(H_p4, W_p4), mode="nearest"
        ).squeeze(0)
        fg_area = mask_p4.sum()
        if fg_area > 0:
            proto = (p4.squeeze(0) * mask_p4).sum(dim=(1, 2)) / (fg_area + 1e-8)
            feats_list.append(proto)
    if not feats_list:
        return torch.zeros(1280, device=device)
    proto = torch.stack(feats_list).mean(dim=0)
    return F.normalize(proto, dim=0, p=2)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 | Cosine similarity."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def cross_class_cosine(per_class_data: dict[int, np.ndarray]) -> float:
    """
    计算跨类平均余弦相似度 (越低越好 = 越能区分).
    Compute mean cross-class cosine similarity (lower = better discrimination).

    per_class_data: {cls_id: np.ndarray [N, dim]}.
    :return: mean cosine similarity across all class pairs.
    """
    class_ids = sorted(per_class_data.keys())
    sims = []
    for i, ci in enumerate(class_ids):
        for cj in class_ids[i + 1:]:
            # Mean of all sample pairs between ci and cj
            ai = per_class_data[ci]  # [Ni, D]
            aj = per_class_data[cj]  # [Nj, D]
            # Efficient: compute mean vector per class → cosine
            mi = ai.mean(axis=0)
            mj = aj.mean(axis=0)
            sims.append(cos_sim(mi, mj))
    return float(np.mean(sims)) if sims else 1.0


def within_class_similarity(per_class_data: dict[int, np.ndarray]) -> float:
    """
    计算类内平均成对余弦相似度 (越高越好 = 越聚集).
    Compute mean within-class pairwise cosine similarity (higher = tighter clustering).

    :return: mean cosine similarity within each class.
    """
    sims_all = []
    for cls_id, data in per_class_data.items():
        N = data.shape[0]
        if N < 2:
            continue
        for i in range(N):
            for j in range(i + 1, N):
                sims_all.append(cos_sim(data[i], data[j]))
    return float(np.mean(sims_all)) if sims_all else 0.0


# ═══════════════════════════════════════════════════════════════════
# Decoder 分步前向 | Instrumented Step-by-Step Forward
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def instrumented_forward(
    decoder: AdaptiveSparseDecoder,
    p4_features: torch.Tensor,      # [1, 1280, H/16, W/16]
    proto_masks: torch.Tensor,       # [32, H/4, W/4]
    support_proto: torch.Tensor,     # [1280]
    device: torch.device,
) -> dict:
    """
    分步执行 decoder forward，捕获所有中间激活。
    Step through decoder forward, capturing all intermediate activations.

    :return: {
        "coeffs": [32],           # ProtoCoeffPredictor 输出
        "proto_mask": [H/4, W/4], # sigmoid(coeffs@proto)
        "refined_up": [H/4, W/4], # P4 refinement (upsampled)
        "final_mask": [H/4, W/4], # sigmoid(proto_mask + refined_up)
        "proto_branch_norm": float,   # |proto_mask|_mean
        "refined_branch_norm": float, # |refined_up|_mean
        "contribution_ratio": float,  # |refined| / (|proto| + |refined|)
    }
    """
    # ── 输入标准化 (同 forward) ──
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)
    if support_proto.dim() == 2:
        support_proto = support_proto.squeeze(0)

    proto_masks = decoder._normalize_proto(proto_masks)

    # ── Step 1: Coeff + Proto Mask ──
    coeffs = decoder.coeff_predictor(support_proto.unsqueeze(0))  # [1, 32]
    proto_mask = decoder.coeff_predictor.generate_mask(
        coeffs, proto_masks
    )  # [1, H/4, W/4]

    # ── Step 2: P4 Refinement ──
    feat_proj = decoder.feat_proj(p4_features)       # [1, 256, H/16, W/16]
    feat_refined = decoder.feat_refine(feat_proj)     # [1, 64, H/16, W/16]
    refined_logit = decoder.mask_head(feat_refined)   # [1, 1, H/16, W/16]
    refined_up = F.interpolate(
        refined_logit, size=proto_mask.shape[1:],
        mode="bilinear", align_corners=False,
    ).squeeze(1)  # [1, H/4, W/4]

    # ── Step 3: Fusion ──
    final_logit = refined_up.squeeze(0) + proto_mask.squeeze(0)  # [H/4, W/4]
    final_mask = torch.sigmoid(final_logit)

    # ── 贡献比 | Contribution Ratio ──
    proto_norm = proto_mask.squeeze(0).abs().mean().item()
    refined_norm = refined_up.squeeze(0).abs().mean().item()
    total_norm = proto_norm + refined_norm + 1e-8
    refined_ratio = refined_norm / total_norm

    return {
        "coeffs": coeffs.squeeze(0).cpu().numpy(),          # [32]
        "proto_mask": proto_mask.squeeze(0).cpu().numpy(),   # [H/4, W/4]
        "refined_up": refined_up.squeeze(0).cpu().numpy(),   # [H/4, W/4]
        "final_mask": final_mask.cpu().numpy(),               # [H/4, W/4]
        "proto_branch_norm": proto_norm,
        "refined_branch_norm": refined_norm,
        "contribution_ratio": refined_ratio,
    }


# ═══════════════════════════════════════════════════════════════════
# Ablation: 比较 proto-only / P4-only / full 的 mask 差异
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def ablation_forward(
    decoder: AdaptiveSparseDecoder,
    p4_features: torch.Tensor,
    proto_masks: torch.Tensor,
    support_proto: torch.Tensor,
) -> dict:
    """
    三种模式对比 | Three-mode comparison:
    - full:       proto_mask + refined_up → sigmoid
    - proto_only: proto_mask only → sigmoid (P4 ablated)
    - p4_only:    refined_up only → sigmoid (Proto ablated)
    """
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)
    if support_proto.dim() == 2:
        support_proto = support_proto.squeeze(0)

    proto_masks = decoder._normalize_proto(proto_masks)

    # ── proto_only ──
    coeffs = decoder.coeff_predictor(support_proto.unsqueeze(0))
    proto_mask = decoder.coeff_predictor.generate_mask(coeffs, proto_masks)
    proto_only = torch.sigmoid(proto_mask.squeeze(0))

    # ── p4_only ──
    feat_proj = decoder.feat_proj(p4_features)
    feat_refined = decoder.feat_refine(feat_proj)
    refined_logit = decoder.mask_head(feat_refined)
    refined_up = F.interpolate(
        refined_logit, size=proto_mask.shape[1:],
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)
    p4_only = torch.sigmoid(refined_up)

    # ── full ──
    full = torch.sigmoid(refined_up + proto_mask.squeeze(0))

    return {
        "proto_only": proto_only.cpu().numpy(),
        "p4_only": p4_only.cpu().numpy(),
        "full": full.cpu().numpy(),
    }


# ═══════════════════════════════════════════════════════════════════
# 主审计函数 | Main Audit Function
# ═══════════════════════════════════════════════════════════════════

def run_info_flow_audit(
    backbone: FastSAMBackbone,
    decoder: AdaptiveSparseDecoder,
    dataset: SeverstalDataset,
    device: torch.device,
    k_shot: int = 3,
    num_queries: int = 30,
    seed: int = 42,
) -> dict:
    """
    执行完整的 Decoder 信息流审计。
    Run complete Decoder Information Flow Audit.

    Returns a dict with per-stage metrics and raw data for visualization.
    """
    rng = random.Random(seed)
    decoder.eval()
    backbone.eval()

    # ── 每类数据容器 | Per-class data containers ──
    # Prototype level
    support_protos: dict[int, np.ndarray] = {}  # {cls: [1280]}
    # Coeff level
    per_class_coeffs: dict[int, list[np.ndarray]] = defaultdict(list)  # {cls: [[32], ...]}
    # Proto Mask level
    per_class_proto_masks: dict[int, list[np.ndarray]] = defaultdict(list)  # {cls: [[H/4,W/4], ...]}
    # Refined level
    per_class_refined: dict[int, list[np.ndarray]] = defaultdict(list)
    # Final Mask level
    per_class_final: dict[int, list[np.ndarray]] = defaultdict(list)
    # Contribution ratios
    per_class_contrib: dict[int, list[float]] = defaultdict(list)
    # Ablation masks (proto_only, p4_only, full)
    per_class_ablation: dict[int, list[dict]] = defaultdict(list)

    H4, W4 = IMG_H // 4, IMG_W // 4  # 64, 400

    for cls_id in DEFECT_CLASSES:
        pool = dataset.class_to_images(cls_id)
        n_needed = k_shot + num_queries

        if len(pool) < n_needed:
            print(f"  ⚠ Class {cls_id}: only {len(pool)} images, using all.")
            sampled = pool[:]
        else:
            sampled = rng.sample(pool, n_needed)

        support_indices = sampled[:k_shot]
        query_indices = sampled[k_shot:k_shot + num_queries]

        # ── Support Prototype ──
        s_imgs, s_masks = [], []
        for idx in support_indices:
            s = dataset[idx]
            s_imgs.append(s["image"])
            s_masks.append(get_class_mask(s, cls_id))
        sp = compute_prototype(
            backbone, torch.stack(s_imgs), torch.stack(s_masks), device
        )
        support_protos[cls_id] = sp.cpu().numpy()

        # ── Per-query instrumented forward ──
        for idx in tqdm(query_indices, desc=f"  Class {cls_id}", leave=False):
            q = dataset[idx]
            q_img = q["image"].unsqueeze(0).to(device)  # [1, 3, 256, 1600]
            q_mask_full = get_class_mask(q, cls_id)       # [256, 1600] binary

            # Backbone forward
            feats = backbone(q_img, extract_proto=True)
            p4 = feats["p4"]         # [1, 1280, H/16, W/16]
            proto_masks = feats["proto"]  # [1, 32, H/4, W/4]

            # Instrumented decoder forward
            result = instrumented_forward(
                decoder, p4, proto_masks, sp, device,
            )

            per_class_coeffs[cls_id].append(result["coeffs"])
            per_class_proto_masks[cls_id].append(result["proto_mask"])
            per_class_refined[cls_id].append(result["refined_up"])
            per_class_final[cls_id].append(result["final_mask"])
            per_class_contrib[cls_id].append(result["contribution_ratio"])

            # Ablation
            abl = ablation_forward(decoder, p4, proto_masks, sp)
            per_class_ablation[cls_id].append(abl)

    # ── 转换为 numpy | Convert to numpy ──
    def stack_dict(d, cls_ids):
        return {c: np.stack(d[c]) for c in cls_ids}

    coeffs_data = stack_dict(per_class_coeffs, DEFECT_CLASSES)
    proto_mask_data = stack_dict(per_class_proto_masks, DEFECT_CLASSES)
    refined_data = stack_dict(per_class_refined, DEFECT_CLASSES)
    final_data = stack_dict(per_class_final, DEFECT_CLASSES)
    contrib_data = {c: np.array(per_class_contrib[c]) for c in DEFECT_CLASSES}
    ablation_data = {c: per_class_ablation[c] for c in DEFECT_CLASSES}

    # ═══════════════════════════════════════════════════════════════
    # Stage 0: Prototype — 类间余弦相似度
    # ═══════════════════════════════════════════════════════════════
    proto_cross_cos = cross_class_cosine(
        {c: support_protos[c][np.newaxis, :] for c in DEFECT_CLASSES}
    )
    proto_within_sim = within_class_similarity(
        {c: support_protos[c][np.newaxis, :] for c in DEFECT_CLASSES}
    )

    # ═══════════════════════════════════════════════════════════════
    # Stage 1: Coeff — 类间余弦相似度 + std
    # ═══════════════════════════════════════════════════════════════
    coeff_cross_cos = cross_class_cosine(coeffs_data)
    coeff_within_sim = within_class_similarity(coeffs_data)
    coeff_per_class_std = {
        c: float(np.mean(np.std(coeffs_data[c], axis=0)))
        for c in DEFECT_CLASSES
    }

    # Coeff PCA (合并所有 query)
    all_coeffs = np.concatenate([coeffs_data[c] for c in DEFECT_CLASSES], axis=0)
    all_coeff_labels = np.concatenate([
        np.full(coeffs_data[c].shape[0], c) for c in DEFECT_CLASSES
    ])
    if all_coeffs.shape[0] >= 4:
        pca = PCA(n_components=2)
        coeff_pca_2d = pca.fit_transform(all_coeffs)
        coeff_pca_var = pca.explained_variance_ratio_.tolist()
    else:
        coeff_pca_2d = np.zeros((all_coeffs.shape[0], 2))
        coeff_pca_var = [0, 0]

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: Proto Mask — 类间相似度 (spatial)
    # ═══════════════════════════════════════════════════════════════
    # 展平为向量计算 | Flatten to vectors for similarity
    proto_mask_flat = {
        c: proto_mask_data[c].reshape(proto_mask_data[c].shape[0], -1)
        for c in DEFECT_CLASSES
    }
    proto_mask_cross_cos = cross_class_cosine(proto_mask_flat)
    proto_mask_within_sim = within_class_similarity(proto_mask_flat)

    # 类间 IoU (取每类平均 mask → 二值化 → IoU)
    proto_mask_mean = {c: proto_mask_data[c].mean(axis=0) for c in DEFECT_CLASSES}
    proto_mask_cross_iou = 0.0
    iou_pairs = 0
    for i, ci in enumerate(DEFECT_CLASSES):
        for cj in DEFECT_CLASSES[i + 1:]:
            a = proto_mask_mean[ci] > 0.5
            b = proto_mask_mean[cj] > 0.5
            inter = (a & b).sum()
            union = (a | b).sum()
            if union > 0:
                proto_mask_cross_iou += float(inter / union)
                iou_pairs += 1
    proto_mask_cross_iou = proto_mask_cross_iou / max(iou_pairs, 1)

    # ═══════════════════════════════════════════════════════════════
    # Stage 3: P4 Refined — 类间相似度
    # ═══════════════════════════════════════════════════════════════
    refined_flat = {
        c: refined_data[c].reshape(refined_data[c].shape[0], -1)
        for c in DEFECT_CLASSES
    }
    refined_cross_cos = cross_class_cosine(refined_flat)
    refined_within_sim = within_class_similarity(refined_flat)

    # ═══════════════════════════════════════════════════════════════
    # Stage 4: Final Mask — 类间相似度
    # ═══════════════════════════════════════════════════════════════
    final_flat = {
        c: final_data[c].reshape(final_data[c].shape[0], -1)
        for c in DEFECT_CLASSES
    }
    final_cross_cos = cross_class_cosine(final_flat)
    final_within_sim = within_class_similarity(final_flat)

    # ═══════════════════════════════════════════════════════════════
    # Branch Contribution | 分支贡献
    # ═══════════════════════════════════════════════════════════════
    contrib_mean = {c: float(np.mean(contrib_data[c])) for c in DEFECT_CLASSES}
    contrib_global = float(np.mean(list(contrib_mean.values())))

    # ═══════════════════════════════════════════════════════════════
    # Ablation: mask diversity (类间差异)
    # ═══════════════════════════════════════════════════════════════
    ablation_cross_cos = {}
    for mode in ["proto_only", "p4_only", "full"]:
        mode_data = {}
        for cls_id in DEFECT_CLASSES:
            masks = np.stack([abl[mode] for abl in ablation_data[cls_id]])
            mode_data[cls_id] = masks.reshape(masks.shape[0], -1)
        ablation_cross_cos[mode] = cross_class_cosine(mode_data)

    return {
        # Per-class support prototypes
        "support_protos": {c: support_protos[c].tolist() for c in DEFECT_CLASSES},
        # Stage metrics
        "stage_metrics": {
            "prototype": {
                "cross_class_cosine": proto_cross_cos,
                "within_class_similarity": proto_within_sim,
                "discrimination": 1.0 - proto_cross_cos,
            },
            "coeff": {
                "cross_class_cosine": coeff_cross_cos,
                "within_class_similarity": coeff_within_sim,
                "per_class_std": coeff_per_class_std,
                "pca_variance_ratio": coeff_pca_var,
            },
            "proto_mask": {
                "cross_class_cosine": proto_mask_cross_cos,
                "within_class_similarity": proto_mask_within_sim,
                "cross_class_iou": proto_mask_cross_iou,
            },
            "refined": {
                "cross_class_cosine": refined_cross_cos,
                "within_class_similarity": refined_within_sim,
            },
            "final": {
                "cross_class_cosine": final_cross_cos,
                "within_class_similarity": final_within_sim,
            },
        },
        # Signal decay curve
        "signal_decay": {
            "stages": ["Prototype", "Coeff", "ProtoMask", "Refined", "Final"],
            "cross_class_cosine": [
                proto_cross_cos, coeff_cross_cos, proto_mask_cross_cos,
                refined_cross_cos, final_cross_cos,
            ],
            "discrimination": [
                1.0 - x for x in [
                    proto_cross_cos, coeff_cross_cos, proto_mask_cross_cos,
                    refined_cross_cos, final_cross_cos,
                ]
            ],
        },
        # Branch contribution
        "branch_contribution": {
            "per_class": contrib_mean,
            "global_mean": contrib_global,
        },
        # Ablation
        "ablation": ablation_cross_cos,
        # Raw data for visualization
        "raw": {
            "coeffs_2d": coeff_pca_2d.tolist(),
            "coeff_labels": all_coeff_labels.tolist(),
            "coeffs_per_class": {c: coeffs_data[c].tolist() for c in DEFECT_CLASSES},
            "proto_mask_mean": {c: proto_mask_mean[c].tolist() for c in DEFECT_CLASSES},
            "contrib_per_class": contrib_mean,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Backbone 加载 (支持 LoRA)
# ═══════════════════════════════════════════════════════════════════

def load_backbone_and_decoder(checkpoint_path: str, device: torch.device, logger):
    """加载 backbone + decoder (含 LoRA 恢复)."""
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt",
    ).to(device)
    backbone.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    lora_rank = ckpt.get("lora_rank", 0)

    if lora_rank > 0 and "lora_state_dict" in ckpt:
        backbone.apply_conv_lora(rank=lora_rank, alpha=1.0)
        backbone.model.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        logger(f"Backbone: +ConvLoRA r={lora_rank} restored")
    else:
        logger("Backbone: Frozen (no LoRA)")

    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=True)

    # ── Decoder ──
    decoder = AdaptiveSparseDecoder(
        in_channels=1280, proto_dim=32, hidden_dim=256,
        use_fdr=False, normalize_proto="none", out_channels=1,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    logger(f"Decoder: {sum(p.numel() for p in decoder.parameters())/1e3:.1f}K params restored")

    return backbone, decoder, lora_rank


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════

def plot_signal_decay(audit: dict, output_path: Path, label: str):
    """信号衰减曲线 | Signal Decay Curve: Discrimination (1-cos) per stage."""
    decay = audit["signal_decay"]
    stages = decay["stages"]
    disc = decay["discrimination"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stages))
    bars = ax.bar(x, disc, color=["#2ECC71", "#3498DB", "#F39C12", "#E74C3C", "#9B59B6"],
                  edgecolor="black", linewidth=0.5)

    # 标注值
    for bar, val in zip(bars, disc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Discrimination (1 - Cross-Class Cosine)", fontsize=12)
    ax.set_title(f"Decoder Information Flow: Signal Decay ({label})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(disc) * 1.3 if max(disc) > 0 else 0.1)
    ax.grid(axis="y", alpha=0.3)

    # 添加信息损失标注
    for i in range(1, len(disc)):
        drop = disc[i - 1] - disc[i]
        if drop > 0.01:
            ax.annotate(f"-{drop:.3f}",
                        xy=(i - 0.5, (disc[i - 1] + disc[i]) / 2),
                        fontsize=8, color="red",
                        ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Signal decay plot: {output_path}")


def plot_coeff_pca(audit: dict, output_path: Path, label: str):
    """Coeff PCA 可视化 | Coeff PCA Visualization."""
    raw = audit["raw"]
    pts = np.array(raw["coeffs_2d"])
    labels = np.array(raw["coeff_labels"])

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_id in DEFECT_CLASSES:
        mask = labels == cls_id
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1],
                      c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                      s=50, alpha=0.7, edgecolors="black", linewidth=0.5)

    pca_var = audit["stage_metrics"]["coeff"]["pca_variance_ratio"]
    ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"ProtoCoeffPredictor Output PCA ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Coeff PCA plot: {output_path}")


def plot_branch_contribution(audit: dict, output_path: Path, label: str):
    """分支贡献比 | Branch Contribution Ratio."""
    contrib = audit["branch_contribution"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Per-class bar
    ax = axes[0]
    classes = [CLASS_NAMES[c] for c in DEFECT_CLASSES]
    vals = [contrib["per_class"].get(c, 0) for c in DEFECT_CLASSES]
    colors = [CLASS_COLORS[c] for c in DEFECT_CLASSES]
    bars = ax.bar(classes, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("P4 Refinement Contribution Ratio", fontsize=11)
    ax.set_title("Per-Class P4 Contribution", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # (b) Proto vs P4 pie
    ax = axes[1]
    proto_pct = 1.0 - contrib["global_mean"]
    p4_pct = contrib["global_mean"]
    wedges, texts, autotexts = ax.pie(
        [proto_pct, p4_pct],
        labels=["Proto Branch", "P4 Branch"],
        autopct="%1.1f%%",
        colors=["#3498DB", "#E74C3C"],
        explode=(0, 0.05),
        startangle=90,
    )
    ax.set_title(f"Global Contribution Split ({label})", fontsize=12, fontweight="bold")

    fig.suptitle(f"Proto vs P4 Branch Contribution ({label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Branch contribution plot: {output_path}")


def plot_proto_mask_grid(audit: dict, output_path: Path, label: str):
    """每类平均 proto_mask 网格图 | Per-class mean proto_mask grid."""
    raw = audit["raw"]
    proto_mean = raw["proto_mask_mean"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, cls_id in enumerate(DEFECT_CLASSES):
        ax = axes[i]
        mask = np.array(proto_mean[str(cls_id)]).reshape(IMG_H // 4, IMG_W // 4)
        im = ax.imshow(mask, cmap="hot", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{CLASS_NAMES[cls_id]}\nMean Proto Mask", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-Class Mean Proto Mask ({label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Proto mask grid: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# 报告打印 | Report Printing
# ═══════════════════════════════════════════════════════════════════

def print_audit_report(audit: dict, label: str):
    """打印审计报告 | Print audit report."""
    sm = audit["stage_metrics"]
    decay = audit["signal_decay"]
    bc = audit["branch_contribution"]
    abl = audit["ablation"]

    print(f"\n{'═'*80}")
    print(f"  Decoder Information Flow Audit — {label}")
    print(f"{'═'*80}")

    # ── Signal Decay Table ──
    print(f"\n  ┌─ STAGE-BY-STAGE SIGNAL DECAY {'─'*45}")
    print(f"  │ {'Stage':<14} {'Cross-Class Cos':<18} {'Discrimination':<16} {'Δ Drop':<12}")
    print(f"  │ {'─'*60}")
    stages = decay["stages"]
    cos_vals = decay["cross_class_cosine"]
    disc_vals = decay["discrimination"]
    for i, (stage, cos_v, disc_v) in enumerate(zip(stages, cos_vals, disc_vals)):
        drop = disc_vals[i - 1] - disc_v if i > 0 else 0
        drop_str = f"-{drop:.4f}" if drop > 0.001 else "—"
        marker = " ⚠ DROP" if drop > 0.02 else ""
        print(f"  │ {stage:<14} {cos_v:<18.4f} {disc_v:<16.4f} {drop_str:<12}{marker}")
    print(f"  └{'─'*60}")

    # ── Coeff Details ──
    print(f"\n  ┌─ COEFF PREDICTOR DETAILS {'─'*48}")
    coeff_m = sm["coeff"]
    print(f"  │ Cross-Class Cosine:  {coeff_m['cross_class_cosine']:.4f}")
    print(f"  │ Within-Class Sim:    {coeff_m['within_class_similarity']:.4f}")
    print(f"  │ PCA Var Ratio:       PC1={coeff_m['pca_variance_ratio'][0]*100:.1f}%  "
          f"PC2={coeff_m['pca_variance_ratio'][1]*100:.1f}%")
    for cls_id in DEFECT_CLASSES:
        std = coeff_m["per_class_std"].get(str(cls_id), coeff_m["per_class_std"].get(cls_id, 0))
        print(f"  │ Coeff Std ({CLASS_NAMES[cls_id]}): {std:.4f}")
    print(f"  └{'─'*60}")

    # ── Branch Contribution ──
    print(f"\n  ┌─ BRANCH CONTRIBUTION {'─'*51}")
    for cls_id in DEFECT_CLASSES:
        pct = bc["per_class"].get(str(cls_id), bc["per_class"].get(cls_id, 0))
        bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
        print(f"  │ {CLASS_NAMES[cls_id]:<10} P4={pct:.3f}  Proto={1-pct:.3f}  [{bar}]")
    print(f"  │ {'─'*50}")
    print(f"  │ GLOBAL: P4 branch = {bc['global_mean']*100:.1f}%, Proto branch = {(1-bc['global_mean'])*100:.1f}%")
    print(f"  └{'─'*60}")

    # ── Ablation ──
    print(f"\n  ┌─ ABLATION: CROSS-CLASS COSINE (lower=better) {'─'*31}")
    for mode in ["proto_only", "p4_only", "full"]:
        val = abl[mode]
        print(f"  │ {mode:<15} {val:.4f}")
    print(f"  └{'─'*60}")

    # ── Verdict ──
    print(f"\n  ┌─ BOTTLENECK VERDICT {'─'*51}")
    # Find the stage with the largest discrimination drop
    max_drop_stage = ""
    max_drop_val = 0
    for i in range(1, len(disc_vals)):
        drop = disc_vals[i - 1] - disc_vals[i]
        if drop > max_drop_val:
            max_drop_val = drop
            max_drop_stage = stages[i]

    if max_drop_val > 0.02:
        print(f"  │ ⚠ LARGEST DROP: {max_drop_stage} (Δdisc = -{max_drop_val:.4f})")
        if max_drop_stage == "Coeff":
            print(f"  │ → ProtoCoeffPredictor is the BOTTLENECK.")
            print(f"  │ → Class-specific prototype info is LOST in the MLP.")
            print(f"  │ → Fix: Cross-Attention or FiLM instead of MLP.")
        elif max_drop_stage == "ProtoMask":
            print(f"  │ → Proto Basis is the BOTTLENECK.")
            print(f"  │ → Different coeffs produce similar masks (basis redundancy).")
            print(f"  │ → Fix: Regularize proto basis orthogonality.")
        elif max_drop_stage == "Refined":
            print(f"  │ → P4 Refinement is the BOTTLENECK.")
            print(f"  │ → P4 branch produces similar output regardless of class.")
            print(f"  │ → Fix: Condition P4 refinement on prototype (FiLM).")
        elif max_drop_stage == "Final":
            print(f"  │ → Fusion step dilutes discrimination.")
            print(f"  │ → Fix: Adjust fusion weight or use gating.")
    else:
        print(f"  │ ➡ NO single bottleneck — discrimination is uniformly low at all stages.")

    if bc["global_mean"] > 0.7:
        print(f"  │")
        print(f"  │ ⚠ P4 BRANCH DOMINATES ({bc['global_mean']*100:.0f}% contribution).")
        print(f"  │ → Prototype pathway is effectively BYPASSED.")
        print(f"  │ → Performance gain (if any) comes from P4, NOT prototype.")
    print(f"  └{'─'*60}")

    print(f"\n{'═'*80}\n")


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Decoder Information Flow Audit — Severstal"
    )
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="训练 checkpoint 路径")
    p.add_argument("--label", type=str, default=None,
                   help="显示标签 (默认从 checkpoint 路径推断)")
    p.add_argument("--k-shot", type=int, default=3)
    p.add_argument("--num-queries", type=int, default=30,
                   help="每类 Query 图像数 (default: 30)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 输出目录 ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"diag_output/severstal_info_flow_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

    label = args.label or Path(args.checkpoint).parent.parent.name
    log(f"Output: {out_dir}")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Label: {label}")

    # ── 数据 ──
    train_ds = SeverstalDataset(root=args.data_root, split="train", binary=False, seed=args.seed)
    log(f"Dataset: {len(train_ds)} images")

    # ── 加载模型 ──
    log("Loading model...")
    backbone, decoder, lora_rank = load_backbone_and_decoder(args.checkpoint, device, log)

    # ── 执行审计 ──
    log("Running information flow audit...")
    audit = run_info_flow_audit(
        backbone, decoder, train_ds, device,
        k_shot=args.k_shot, num_queries=args.num_queries, seed=args.seed,
    )

    # ── 打印报告 ──
    print_audit_report(audit, label)

    # ── 可视化 ──
    log("Generating visualizations...")
    plot_signal_decay(audit, out_dir / "signal_decay.png", label)
    plot_coeff_pca(audit, out_dir / "coeff_pca.png", label)
    plot_branch_contribution(audit, out_dir / "branch_contribution.png", label)
    plot_proto_mask_grid(audit, out_dir / "proto_mask_grid.png", label)

    # ── 保存 JSON ──
    audit_json = {
        "timestamp": datetime.now().isoformat(),
        "config": {"checkpoint": args.checkpoint, "label": label,
                   "k_shot": args.k_shot, "num_queries": args.num_queries,
                   "lora_rank": lora_rank},
        "stage_metrics": audit["stage_metrics"],
        "signal_decay": audit["signal_decay"],
        "branch_contribution": audit["branch_contribution"],
        "ablation": audit["ablation"],
    }
    json_path = out_dir / "info_flow_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False, default=str)
    log(f"✓ Report saved: {json_path}")


if __name__ == "__main__":
    main()
