#!/usr/bin/env python3
"""
C-04: Full-Category Few-Shot Instance Segmentation on iSAID
============================================================
全类别少样本实例分割 —— 验证 FastSAM + Decoder 的 Few-Shot 上限。
Full-category few-shot instance segmentation — validate FastSAM + Decoder upper bound.

Phase D 核心实验 | Core Experiment:
    回答: "FastSAM + Decoder 的 Few-Shot 上限到底在哪里？"
    Answer: "Where is the upper bound of FastSAM + Decoder few-shot?"

实验设计 | Design:
    - 冻结 FastSAM → P4 特征 | Frozen FastSAM → P4 features
    - Support: K-shot images → per-class prototype (masked mean P4)
    - Query: P4 → Decoder(prototype) → binary mask
    - 全部 15 类 (ISAID_CATEGORIES) | All 15 classes
    - K = 1, 3, 5 shot, 200 eval episodes/shot
    - Episodic training: 30-50 epochs, 200 episodes/epoch

Decoder 变体 | Decoder Variants:
    - 'baseline': ProtoRefineDecoder (~10K params) — Proto → cosine_sim → Refine CNN
    - 'film':     FiLMFewShotDecoder (~1.1M params) — Proto → FiLM → InstanceDecoder
    - 'crossattn': CATFewShotDecoder (~1.1M params) — Proto → CrossAttn → InstanceDecoder (C-03 同款)
    - 'contrastive': ContrastiveProtoDecoder (~1.3M params) — Proto → Projection → Contrastive

改进 | Improvements over C-02B/C-03:
    1. 全 15 类 (vs 3 类) | All 15 classes (vs 3)
    2. 稀有类过采样 | Rare class oversampling
    3. Warmup + Cosine LR | Warmup + cosine schedule
    4. Gradient clipping | 梯度裁剪
    5. Per-class validation (30 eps/class) | 逐类验证
    6. Best checkpoint per class | 逐类最佳检查点
    7. 完整对比表 (C-02A/B/C baseline) | Full comparison table

用法 | Usage::
    # 快速测试 (3 类, 5 epochs)
    python tools/instance/eval_c04_full_fewshot.py \
        --src-root data/iSAID_processed --device cuda \
        --shots 1 --epochs 5 --episodes-per-epoch 50 \
        --decoder baseline --classes 1,4,5

    # 完整实验 (15 类, 30 epochs)
    python tools/instance/eval_c04_full_fewshot.py \
        --src-root data/iSAID_processed --device cuda \
        --shots 1,3,5 --epochs 30 --episodes-per-epoch 200 \
        --decoder baseline,film

    # 仅评估（不训练）
    python tools/instance/eval_c04_full_fewshot.py \
        --src-root data/iSAID_processed --device cuda \
        --shots 1,3,5 --eval-only --checkpoint runs/c04/decoder_1shot_best.pt
"""

import sys, argparse, time, json, warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.utils.env import get_env_info
from adatile.backbone import FastSAMBackbone
from adatile.utils.label_mapping import ISAID_CATEGORIES
from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper

# ── 复用 C-02A 的数据集 | Reuse C-02A dataset ──
from tools.instance.eval_c02a_fastsam_fewshot import (
    ISAIDInstanceDataset,
)

from adatile.utils.prototype import compute_fg_prototype

# ── 复用 C-03 的 Cross-Attention Decoder 和 Multi-Prototype ──
# Reuse C-03 Cross-Attention Decoder + Multi-Prototype
from tools.instance.eval_c03_catsam_fewshot import (
    CATFewShotDecoder,
    compute_multi_prototype,
)

# ═══════════════════════════════════════════════════════════════════
# 全部 15 个 ISAID 类别 | All 15 ISAID Categories
# ═══════════════════════════════════════════════════════════════════

ALL_ISAID_CLASSES: Dict[int, str] = dict(ISAID_CATEGORIES)

# 类别分组 (用于分析) | Class groups (for analysis)
CLASS_GROUPS = {
    "vehicle":  [1, 2, 3, 5],       # small_vehicle, large_vehicle, plane, ship
    "infra":    [6, 7, 8, 9, 11, 12, 13, 15],  # harbor, GTF, SBF, tennis, road, basketball, bridge, roundabout
    "object":   [4, 10, 14],         # storage_tank, swimming_pool, helicopter
}
# ═══════════════════════════════════════════════════════════════════
# Prototype Computation (shared) | 原型计算（共享）
# ═══════════════════════════════════════════════════════════════════
# Decoder Variants | 解码器变体
# ═══════════════════════════════════════════════════════════════════

class ProtoRefineDecoder(nn.Module):
    """
    Baseline Decoder | Proto → cosine_sim → Refine CNN → mask.
    与 C-02B 相同 | Same as C-02B. ~10K params.
    """

    _logger = get_logger("ProtoRefineDecoder")

    def __init__(self, feat_dim: int = 1280, temperature: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
        )
        n = sum(p.numel() for p in self.parameters())
        ProtoRefineDecoder._logger.log_info("init", f"ProtoRefineDecoder: {n:,} trainable params")

    def forward(self, query_p4, fg_prototype, target_size=None):
        q_norm = F.normalize(query_p4, dim=1, p=2)  # [B, C, H, W]

        if fg_prototype.dim() == 2:
            # Multi-proto [K, C]: max-similarity across K prototypes
            p_norm = F.normalize(fg_prototype, dim=1, p=2)  # [K, C]
            sim = torch.einsum('bchw,kc->bkhw', q_norm, p_norm)  # [B, K, H, W]
            sim = sim.max(dim=1, keepdim=True)[0]  # [B, 1, H, W] — max across K
        else:
            # Single proto [C]: standard cosine similarity
            p_norm = F.normalize(fg_prototype, dim=0, p=2)
            sim = (q_norm * p_norm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        sim = sim / self.temperature
        x = self.refine(sim)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear",
                            align_corners=False)
        return x


class FiLMFewShotDecoder(nn.Module):
    """
    FiLM-conditioned InstanceDecoder.
    C-04 FiLM 变体 | C-04 FiLM variant. ~1.1M params.
    """

    _logger = get_logger("FiLMFewShotDecoder")

    def __init__(self, feat_dim: int = 1280):
        super().__init__()
        self.feat_dim = feat_dim

        # FiLM Generator | proto → γ, β
        self.film_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )

        # InstanceDecoder-style upsample path
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1, bias=True)

        n_film = sum(p.numel() for p in self.film_mlp.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        FiLMFewShotDecoder._logger.log_info(
            "init", f"FiLMFewShotDecoder: {n_total:,} params (FiLM: {n_film:,})"
        )

    def forward(self, query_p4, fg_prototype, target_size=None):
        x = self.proj(query_p4)

        # Multi-proto [K, C]: 取平均用于 FiLM conditioning
        # Multi-proto [K, C]: average for FiLM conditioning
        if fg_prototype.dim() == 2:
            proto_for_film = fg_prototype.mean(dim=0)  # [K, C] → [C]
        else:
            proto_for_film = fg_prototype

        # FiLM condition injection
        if proto_for_film is not None and proto_for_film.abs().sum() > 1e-8:
            film_out = self.film_mlp(proto_for_film)
            gamma, beta = film_out.chunk(2, dim=0)
            x = gamma[None, :, None, None] * x + beta[None, :, None, None]
        else:
            warnings.warn("FiLMFewShotDecoder: empty prototype, FiLM skipped. "
                         "Output is unconditioned.", RuntimeWarning)

        # Upsample path: H/16 → H/4 → H/2 → H
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.mask_head(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class P3P4FiLMFusionDecoder(nn.Module):
    """
    Prototype-Guided Adaptive P3+P4 Fusion Decoder.
    原型引导的自适应 P3+P4 融合解码器.

    Key: proto → gate → α·P3 + (1-α)·P4↑ (channel-wise, learned per class)
    Replaces naive concat+1×1 with prototype-conditioned gating.
    """

    _logger = get_logger("P3P4FiLMFusionDecoder")

    def __init__(self, feat_dim_p3: int = 640, feat_dim_p4: int = 1280,
                 fusion_dim: int = 256, proto_dim: int = 1280):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.proto_dim = proto_dim

        # Align P3 and P4 channels to fusion_dim
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(feat_dim_p3, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim), nn.ReLU(inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(feat_dim_p4, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim), nn.ReLU(inplace=True),
        )

        # Proto → Gate: generates per-channel fusion weight α ∈ [0,1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(proto_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, fusion_dim),
            nn.Sigmoid(),  # α ∈ [0,1] per channel
        )

        # FiLM Generator: proto → γ, β for fused features
        self.film_mlp = nn.Sequential(
            nn.Linear(proto_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fusion_dim * 2),
        )

        # Utilization Module: predict which features are under-utilized
        self.util_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, fusion_dim, 1, bias=False),
            nn.Sigmoid(),  # per-channel utilization weight 0~1
        )

        # Upsample path
        self.up1 = nn.Sequential(
            nn.Conv2d(fusion_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1, bias=True)

        n_gate = sum(p.numel() for p in self.gate_mlp.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        P3P4FiLMFusionDecoder._logger.log_info(
            "init", f"P3P4FiLMFusionDecoder (adaptive gate): {n_total:,} params "
            f"(Gate MLP: {n_gate:,})"
        )

    def forward(self, query_p3, query_p4, fg_prototype, target_size=None):
        # Project to same channel space
        f3 = self.proj_p3(query_p3)  # [B, fusion_dim, H/8, W/8]
        f4 = self.proj_p4(F.interpolate(query_p4, size=query_p3.shape[2:],
                          mode="bilinear", align_corners=False))

        # Multi-proto [K, C]: 取平均用于 gate/FiLM
        if fg_prototype.dim() == 2:
            proto_for_cond = fg_prototype.mean(dim=0)
        else:
            proto_for_cond = fg_prototype

        # Prototype-guided gated fusion: α·P3 + (1-α)·P4
        if proto_for_cond is not None and proto_for_cond.abs().sum() > 1e-8:
            alpha = self.gate_mlp(proto_for_cond)  # [fusion_dim]
            fused = alpha[None, :, None, None] * f3 + (1 - alpha)[None, :, None, None] * f4
        else:
            fused = 0.5 * f3 + 0.5 * f4  # fallback: equal weight

        # ── Utilization Module | 表示利用率模块 ──
        util_map = self.util_conv(fused)  # [B, 256, H/8, W/8], 0~1
        self.util_mean = util_map.mean()  # scalar for supervision
        fused = fused * util_map

        # FiLM modulation
        if proto_for_cond is not None and proto_for_cond.abs().sum() > 1e-8:
            film_out = self.film_mlp(proto_for_cond)
            gamma, beta = film_out.chunk(2, dim=0)
            fused = gamma[None, :, None, None] * fused + beta[None, :, None, None]

        # Upsample: H/8 → H/4 → H/2 → H
        x = self.up1(fused)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.mask_head(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class ContrastiveProtoDecoder(nn.Module):
    """
    Contrastive Prototype Decoder | 对比原型解码器.

    核心创新 | Core Innovation (Phase C 预备):
        Proto → Projection MLP → contrastive-friendly space
        Query P4 → Projection → cosine_sim(projected_proto) → Refine CNN → mask

    ~1.3M params (projection ~394K + decoder ~900K).
    """

    _logger = get_logger("ContrastiveProtoDecoder")

    def __init__(self, feat_dim: int = 1280, proj_dim: int = 256,
                 temperature: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.temperature = temperature

        # 投影到 contrastive-friendly 空间 (输出保持 feat_dim 以便与原始原型维度对齐)
        # Project to contrastive-friendly space (output feat_dim for alignment with original proto dim)
        self.proto_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, feat_dim),
        )

        self.query_proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, feat_dim, 1, bias=False),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
        )

        n_proj = sum(p.numel() for p in self.proto_proj.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        ContrastiveProtoDecoder._logger.log_info(
            "init", f"ContrastiveProtoDecoder: {n_total:,} params (Proj: {n_proj:,})"
        )

    def project_prototype(self, proto: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proto_proj(proto), dim=0, p=2)

    def forward(self, query_p4, fg_prototype, target_size=None):
        q_proj = self.query_proj(query_p4)
        q_norm = F.normalize(q_proj, dim=1, p=2)
        p_proj = self.project_prototype(fg_prototype)
        sim = (q_norm * p_proj.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        sim = sim / self.temperature
        x = self.refine(sim)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear",
                            align_corners=False)
        return x


def build_decoder(method: str, **kwargs) -> nn.Module:
    """
    Decoder factory | 解码器工厂.

    :param method: 'baseline' | 'film' | 'crossattn' | 'contrastive'
    :param kwargs: passed to decoder constructor (e.g. num_prototypes, feat_dim)
    """
    if method == "baseline":
        return ProtoRefineDecoder(
            feat_dim=kwargs.get("feat_dim", 1280),
            temperature=kwargs.get("temperature", 0.1))
    elif method == "film":
        return FiLMFewShotDecoder(
            feat_dim=kwargs.get("feat_dim", 1280))
    elif method == "crossattn":
        return CATFewShotDecoder(**kwargs)
    elif method == "contrastive":
        return ContrastiveProtoDecoder(
            feat_dim=kwargs.get("feat_dim", 1280),
            proj_dim=kwargs.get("proj_dim", 256),
            temperature=kwargs.get("temperature", 0.1))
    elif method == "p3p4film":
        return P3P4FiLMFusionDecoder(
            feat_dim_p3=kwargs.get("feat_dim_p3", 640),
            feat_dim_p4=kwargs.get("feat_dim_p4", 1280),
            fusion_dim=kwargs.get("fusion_dim", 256),
            proto_dim=kwargs.get("feat_dim_p4", 1280))
    else:
        raise ValueError(f"Unknown decoder: {method}")


# ═══════════════════════════════════════════════════════════════════
# Binary IoU | 二值 IoU
# ═══════════════════════════════════════════════════════════════════

def _decoder_forward(decoder, backbone, query_img, fg_prototype, target_size,
                     feature_level="p4"):
    """统一的 decoder 前向调用, 处理单层和 p3p4 融合 | Unified decoder forward."""
    feats = backbone(query_img)
    if feature_level == "p3p4":
        return decoder(feats["p3"], feats["p4"], fg_prototype, target_size=target_size)
    else:
        return decoder(feats[feature_level], fg_prototype, target_size=target_size)


def _extract_prototype(backbone, support_imgs, support_masks, feature_level, num_proto):
    """提取 prototype, 处理单层和 p3p4 | Extract prototype for single or fusion.

    当 num_proto > 1 时，返回 [K, C] 矩阵（K 个独立 prototype）。
    Decoder 端用 max-similarity 跨 prototype 聚合。
    When num_proto > 1, returns [K, C] matrix (K independent prototypes).
    Decoder uses max-similarity across prototypes for aggregation.
    """
    feats = backbone(support_imgs)
    if feature_level == "p3p4":
        support_feats_list = [feats["p4"][i] for i in range(len(support_masks))]
    else:
        support_feats_list = [feats[feature_level][i] for i in range(len(support_masks))]

    if num_proto > 1:
        proto = compute_multi_prototype(support_feats_list, support_masks,
                                        num_prototypes=num_proto)
        # 保留 [K, C] 矩阵 — 不再取平均！
        # Keep [K, C] matrix — do NOT average!
    else:
        proto = compute_fg_prototype(support_feats_list, support_masks)
    return proto


def _extract_feats_for_proto(backbone, support_imgs, feature_level):
    """从 backbone 提取用于 prototype 的特征列表 | Extract feature list for proto."""
    feats = backbone(support_imgs)
    if feature_level == "p3p4":
        return [feats["p4"][i] for i in range(len(support_imgs))]
    return [feats[feature_level][i] for i in range(len(support_imgs))]


def query_aware_prototype(query_feat, static_proto, blend=0.5):
    """
    Query-aware dynamic prototype via cross-attention.
    Query (e.g. P3, fine spatial) guides Proto (e.g. P4, strong semantics).

    static_proto [C_p] → query_feat [C_q,H,W] → dim align → attention → dynamic [C_p].
    """
    if query_feat.dim() == 4:
        query_feat = query_feat.squeeze(0)
    C_q, H, W = query_feat.shape
    C_p = static_proto.shape[0]

    # Align: pad query to proto dim | query 补齐到 proto 维度
    if C_q != C_p:
        C = C_p
        query_feat = F.pad(query_feat, (0, 0, 0, 0, 0, C_p - C_q))  # [C_p, H, W]
    else:
        C = C_q
    static_p = static_proto  # don't modify original

    # Per-location cosine similarity → attention → weighted dynamic proto
    q_flat = F.normalize(query_feat.reshape(C, -1).T, dim=1)  # [HW, C]
    p_norm = F.normalize(static_p, dim=0)  # [C]
    attn = (q_flat @ p_norm).softmax(dim=0)  # [HW]
    dynamic_p = (query_feat.reshape(C, -1) * attn).sum(dim=1)  # [C]
    return (1 - blend) * F.normalize(static_p, dim=0, p=2) + blend * F.normalize(dynamic_p, dim=0, p=2)


def binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / union if union > 0 else 0.0


def binary_recall_precision(pred: torch.Tensor, gt: torch.Tensor):
    """Compute pixel-level recall, precision, dice. | 像素级召回率/精确率/Dice."""
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return recall, precision, dice


# ═══════════════════════════════════════════════════════════════════
# Focal + Dice Loss | 损失函数
# ═══════════════════════════════════════════════════════════════════

def focal_dice_loss(logit: torch.Tensor, target: torch.Tensor,
                    gamma: float = 5.0, ce_weight: float = 0.5,
                    dice_weight: float = 0.5, gap_weight: float = 1.0):
    """Focal (gamma=5) + Dice + FG-presence loss.

    修复 "全背景崩溃":
    - pos_weight: 按 BG/FG 比例平衡 BCE, 防止背景主导梯度
    - fg_presence: 惩罚预测全背景 (sigmoid max < 0.5), 仅在前景存在时生效

    Fixes "all-background collapse":
    - pos_weight: balances BCE by BG/FG ratio
    - fg_presence: penalizes all-background predictions when GT has foreground
    """
    # pos_weight: FG/BG ratio for BCE balance
    n_fg = target.sum().clamp(min=1)
    n_bg = target.numel() - n_fg
    pos_weight = (n_bg / n_fg).clamp(max=100.0)  # cap at 100:1

    ce = F.binary_cross_entropy_with_logits(
        logit, target, reduction="none",
        pos_weight=pos_weight.expand_as(target))
    focal = ((1 - torch.exp(-ce)) ** gamma * ce).mean()

    prob = torch.sigmoid(logit)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum() + 1e-8
    dice = 1.0 - (2 * inter / union)

    # FG presence: 如果 GT 有前景但 pred 全 <0.5, 额外惩罚
    # Penalize "all-background" when GT has foreground pixels
    fg_presence = 0.0
    if n_fg > 10:
        pred_max = prob.max()
        if pred_max < 0.5:
            fg_presence = (0.5 - pred_max) ** 2

    loss = gap_weight * (ce_weight * focal + dice_weight * dice + 0.1 * fg_presence)
    return loss, {"focal": focal.item(), "dice": dice.item(), "fg_presence": fg_presence}

# ═══════════════════════════════════════════════════════════════════
# Episodic Training | 回合式训练
# ═══════════════════════════════════════════════════════════════════

def train_episode(decoder, backbone, support_idxs, query_idx,
                  train_ds, query_class, device, opt,
                  scaler=None, grad_clip=1.0, use_amp=False,
                  feature_level="p4", use_dynamic_proto=False,
                  gap_weight=1.0):
    """Single training episode: Support -> proto -> query -> loss -> backward.

    Support and Query BOTH come from train_ds (standard Few-shot protocol).
    Returns: (loss: float, timing: dict) or (None, None) if empty prototype.
    """
    t0 = time.perf_counter()

    # Support — image loading + mask rendering (from train_ds)
    support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
    support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                     for si in support_idxs]
    t_load_support = time.perf_counter()

    # Query — image loading + mask rendering (also from train_ds)
    query_img = train_ds.load_image(query_idx).unsqueeze(0).to(device)
    query_mask = train_ds.render_class_mask(query_idx, query_class)
    query_mask = query_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    t_load_query = time.perf_counter()

    # Forward - support + query backbone (no grad on frozen backbone)
    with torch.no_grad():
        num_proto = getattr(decoder, 'num_prototypes', 1)
        fg_proto = _extract_prototype(backbone, support_imgs, support_masks,
                                      feature_level, num_proto)
        t_proto = time.perf_counter()

        # Check empty prototype
        if fg_proto.dim() == 1 and fg_proto.sum() == 0:
            return None, None
        if fg_proto.dim() == 2 and fg_proto.sum() == 0:
            return None, None

        # ── Query-aware Dynamic Prototype | 查询感知动态原型 ──
        if use_dynamic_proto:
            query_feat_proto = backbone(query_img)
            if feature_level == "p3p4":
                # P3 (spatial) + P4 (semantic) → concat → model learns which scale
                p3_f = query_feat_proto["p3"].squeeze(0)  # [960, H/8, W/8]
                p4_f = query_feat_proto["p4"]  # [1, 1280, H/16, W/16]
                p4_f = F.interpolate(p4_f, size=p3_f.shape[1:],
                                    mode="bilinear", align_corners=False).squeeze(0)
                qf = torch.cat([p3_f, p4_f], dim=0)  # [2240, H/8, W/8]
            else:
                qf = query_feat_proto[feature_level]
            fg_proto = query_aware_prototype(qf, fg_proto)

    t_backbone_support = time.perf_counter()

    # Forward - query decoder (grad only through decoder)
    tsize = tuple(query_mask.shape[2:])
    if use_amp:
        with torch.amp.autocast(device.type):
            logit = _decoder_forward(decoder, backbone, query_img, fg_proto,
                                     tsize, feature_level)
            loss, components = focal_dice_loss(logit, query_mask, gap_weight=gap_weight)
    else:
        logit = _decoder_forward(decoder, backbone, query_img, fg_proto,
                                 tsize, feature_level)
        loss, components = focal_dice_loss(logit, query_mask, gap_weight=gap_weight)
    t_decoder = time.perf_counter()

    opt.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        opt.step()
    t_backward = time.perf_counter()

    timing = {
        "load_support": t_load_support - t0,
        "load_query": t_load_query - t_load_support,
        "backbone_proto": t_proto - t_load_query,
        "decoder_bw": t_backward - t_proto,
    }
    return loss.item(), timing


@torch.no_grad()
def validate_all_classes_batched(decoder, backbone, train_ds, val_ds,
                                  target_classes, shot, device, rng,
                                  n_val_per_class=10, batch_size=32,
                                  feature_level="p4"):
    """
    批量验证所有类 | Batched validation for all classes.

    一次 backbone forward 处理所有验证 episode, 替代逐类逐 episode 的 N×2 次 backbone forward。
    One backbone forward for all validation episodes, replacing N×2 sequential forwards.

    加速比 | Speedup: ~5-10× for validation phase (90→1 backbone forwards per epoch).
    """
    num_proto = getattr(decoder, 'num_prototypes', 1)
    t_start = time.perf_counter()

    # Phase 1: 预采样所有 episode 的 (support_idxs, query_idx, class)
    # Pre-sample all episodes: (support_idxs, query_idx, class)
    episodes = []  # list of (support_idxs: list[int], query_idx: int, cls: int)
    for cls_id in sorted(target_classes):
        train_candidates = train_ds.class_to_images(cls_id)
        val_candidates = val_ds.class_to_images(cls_id)
        if len(train_candidates) < shot or not val_candidates:
            continue
        for _ in range(n_val_per_class):
            s_idxs = rng.choice(train_candidates, shot, replace=False).tolist()
            q_idx = int(rng.choice(val_candidates))
            episodes.append((s_idxs, q_idx, cls_id))

    if not episodes:
        return ({c: 0.0 for c in target_classes},
                {"sample": 0, "load_backbone": 0, "decoder": 0, "n_images": 0, "n_episodes": 0})

    t_phase1_done = time.perf_counter()

    # Phase 2: 收集 train/val unique tile indices + 加载图像 + backbone forward
    # Collect train/val unique tile indices (separate index spaces!) + backbone forward
    train_tiles = set()
    val_tiles = set()
    for s_idxs, q_idx, _ in episodes:
        train_tiles.update(s_idxs)
        val_tiles.add(q_idx)

    # 分别加载 train/val 图像并过 backbone | Load train/val images separately
    tile_to_p4 = {}  # (ds, tile_idx) -> P4 tensor, ds: 'train' or 'val'
    n_images_total = 0

    for ds_tag, ds, tile_set in [("train", train_ds, train_tiles),
                                  ("val", val_ds, val_tiles)]:
        unique_tiles = sorted(tile_set)
        for batch_start in range(0, len(unique_tiles), batch_size):
            batch_tiles = unique_tiles[batch_start:batch_start + batch_size]
            batch_imgs = torch.stack([ds.load_image(ti) for ti in batch_tiles]).to(device)
            batch_feats = backbone(batch_imgs)
            for i, ti in enumerate(batch_tiles):
                if feature_level == "p3p4":
                    # 移到 CPU — 200+ tile 的 P3+P4 GPU 特征会撑爆 6GB 显存
                    # Move to CPU — 200+ tiles × P3+P4 = 10GB+ GPU memory
                    tile_to_p4[(ds_tag, ti)] = (
                        batch_feats["p3"][i].cpu(),
                        batch_feats["p4"][i].cpu(),
                    )
                else:
                    tile_to_p4[(ds_tag, ti)] = batch_feats[feature_level][i].cpu()
            n_images_total += len(batch_tiles)
    t_phase2_done = time.perf_counter()

    # Phase 3: 计算 prototypes + decoder forward + IoU
    # Compute prototypes + decoder forward + IoU per episode
    per_cls_ious = {c: [] for c in target_classes}

    # ── Mask cache: 避免同一 tile 被不同 class 重复读盘 | Avoid repeated disk reads ──
    mask_cache: dict[tuple, torch.Tensor] = {}  # (ds_tag, tile_idx) → dense_label

    def get_mask_cached(ds, ds_tag, tile_idx, cls_id):
        """从缓存获取指定类的二值掩码 | Get class-specific binary mask from cache."""
        key = (ds_tag, tile_idx)
        if key not in mask_cache:
            # 从 wrapped dataset 加载 dense label | Load dense label from wrapped dataset
            # FewShotEpisodeDataset wraps underlying adapter at .ds
            inner = getattr(ds, 'ds', ds)
            tile_info = inner._tiles[tile_idx]
            fname = f"{tile_info['tile_name']}_label.png"
            raw = inner._cv2.imread(str(inner._mask_dir / fname), inner._cv2.IMREAD_UNCHANGED)
            mask_cache[key] = torch.from_numpy(raw).long()
        return (mask_cache[key] == cls_id).float().to(device)

    for batch_start in range(0, len(episodes), batch_size):
        batch_eps = episodes[batch_start:batch_start + batch_size]

        # 收集 batch 内所有 query P4 | Collect query P4s within batch
        query_p4s, query_masks, query_classes = [], [], []
        proto_list = []

        for s_idxs, q_idx, cls_id in batch_eps:
            # Support → prototype (indices from train_ds)
            # For p3p4: use P4 features for prototype (stronger semantics)
            raw_s = [tile_to_p4[("train", si)] for si in s_idxs]
            if feature_level == "p3p4":
                s_p4s = [r[1].to(device) for r in raw_s]  # P4 from (p3, p4) tuple, CPU→GPU
            else:
                s_p4s = [r.to(device) for r in raw_s]  # CPU→GPU
            s_masks = [get_mask_cached(train_ds, "train", si, cls_id) for si in s_idxs]

            if num_proto > 1:
                proto = compute_multi_prototype(s_p4s, s_masks, num_prototypes=num_proto)
                if (proto.dim() == 1 and proto.sum() == 0) or \
                   (proto.dim() == 2 and proto.sum() == 0):
                    continue
            else:
                proto = compute_fg_prototype(s_p4s, s_masks)
                if proto.sum() == 0:
                    continue

            # Query (indices from val_ds) — features cached on CPU, move to GPU
            q_p4 = tile_to_p4[("val", q_idx)]
            if feature_level == "p3p4":
                q_p4 = (q_p4[0].to(device), q_p4[1].to(device))
            else:
                q_p4 = q_p4.to(device)
            q_mask = get_mask_cached(val_ds, "val", q_idx, cls_id)

            proto_list.append(proto)
            query_p4s.append(q_p4)
            query_masks.append(q_mask)
            query_classes.append(cls_id)

        if not query_p4s:
            continue

        # Decoder — process one at a time since prototypes differ
        for proto, q_p4, q_mask, cls_id in zip(proto_list, query_p4s, query_masks, query_classes):
            tsize = tuple(q_mask.shape)
            if feature_level == "p3p4":
                q_p3, q_p4_single = q_p4
                logit = decoder(q_p3.unsqueeze(0), q_p4_single.unsqueeze(0),
                              proto, target_size=tsize)
            else:
                logit = decoder(q_p4.unsqueeze(0), proto, target_size=tsize)
            pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
            gt = (q_mask.cpu() > 0)
            per_cls_ious[cls_id].append(binary_iou(pred, gt))

    t_phase3_done = time.perf_counter()

    timing = {
        "sample": t_phase1_done - t_start,
        "load_backbone": t_phase2_done - t_phase1_done,
        "decoder": t_phase3_done - t_phase2_done,
        "total": t_phase3_done - t_start,
        "n_images": n_images_total,
        "n_episodes": len(episodes),
    }

    return ({c: float(np.mean(per_cls_ious[c])) if per_cls_ious[c] else 0.0
             for c in target_classes}, timing)


@torch.no_grad()
def validate_episode(decoder, backbone, train_ds, val_ds, query_class,
                     shot, device, rng, n_val=30, feature_level="p4"):
    """Validation: fixed episodes per class (legacy, use validate_all_classes_batched)."""
    train_candidates = train_ds.class_to_images(query_class)
    val_candidates = val_ds.class_to_images(query_class)
    if len(train_candidates) < shot or not val_candidates:
        return 0.0

    num_proto = getattr(decoder, 'num_prototypes', 1)

    ious = []
    n_skipped = 0
    for _ in range(n_val):
        support_idxs = rng.choice(train_candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_p4s = _extract_feats_for_proto(backbone, support_imgs, feature_level)
        if num_proto > 1:
            fg_proto = compute_multi_prototype(support_p4s, support_masks,
                                                num_prototypes=num_proto)
            if (fg_proto.dim() == 1 and fg_proto.sum() == 0) or \
               (fg_proto.dim() == 2 and fg_proto.sum() == 0):
                n_skipped += 1
                continue
        else:
            fg_proto = compute_fg_prototype(support_p4s, support_masks)
            if fg_proto.sum() == 0:
                n_skipped += 1
                continue

        logit = _decoder_forward(decoder, backbone, query_img, fg_proto,
                       target_size=tuple(query_mask.shape),
                       feature_level=feature_level)
        pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)
        ious.append(binary_iou(pred, gt))

    if n_skipped == n_val:
        return 0.0  # 全部空 proto: 返回 0.0 而非 NaN, 让调用方感知到低值
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate_full(decoder, backbone, train_ds, val_ds, device,
                  shot, n_episodes, target_classes, logger, tag,
                  feature_level="p4"):
    """
    标准 FSS Episodic Evaluation | Standard FSS Episodic Evaluation.
    ================================================================

    与 PANet/PFENet/HSNet/Matcher 协议完全一致 | Consistent with standard FSS protocol:
        - 每类等量 episode (n_episodes 自动均分)
          Equal episodes per class (n_episodes auto-distributed)
        - mIoU = mean(per_class_mean_IoU), 而非 mean(all_episode_IoUs)
          mIoU = mean(per-class means), NOT mean of all episode IoUs
        - Support 从 train_ds, Query 从 val_ds
          Support from train_ds, Query from val_ds

    :return: dict with miou_mean (standard), miou_std, per_class_iou, ...
    """
    num_proto = getattr(decoder, 'num_prototypes', 1)
    class_to_images = {c: train_ds.class_to_images(c) for c in target_classes}
    rng = np.random.RandomState(42)
    per_cls_ious = defaultdict(list)
    per_cls_recall = defaultdict(list)
    per_cls_precision = defaultdict(list)
    per_cls_dice = defaultdict(list)
    t0 = time.perf_counter()

    # ── 确定有效类 + 每类 episode 数 | Determine valid classes + episodes per class ──
    valid_classes = [c for c in target_classes if len(class_to_images.get(c, [])) >= shot
                     and len(val_ds.class_to_images(c)) >= 1]
    if not valid_classes:
        logger.log_warn(f"{tag}/eval", "No valid classes for evaluation!")
        return {
            "miou_mean": 0.0, "miou_std": 0.0, "mrecall_mean": 0.0,
            "n_valid": 0, "per_class_iou": {}, "per_class_recall": {},
            "group_stats": {}, "time_s": 0.0,
        }

    n_per_class = max(1, n_episodes // len(valid_classes))
    total_episodes = n_per_class * len(valid_classes)
    logger.log_info(f"{tag}/eval",
                    f"Evaluating {len(valid_classes)} classes × {n_per_class} eps "
                    f"= {total_episodes} episodes (shot={shot})")

    # ── 为每类每个 episode 预采样 (support_idxs, query_idx) | Pre-sample per class ──
    # 预采样确保可复现: 固定 seed rng | Pre-sample for reproducibility: fixed seed rng
    all_episodes: list[tuple[int, list[int], int]] = []  # (cls_id, [s_idxs], q_idx)
    skipped_empty = 0
    for cls_id in valid_classes:
        candidates = class_to_images[cls_id]
        val_candidates = val_ds.class_to_images(cls_id)
        for _ in range(n_per_class):
            if len(candidates) < shot or not val_candidates:
                skipped_empty += 1
                continue
            s_idxs = rng.choice(candidates, shot, replace=False).tolist()
            q_idx = int(rng.choice(val_candidates))
            all_episodes.append((cls_id, s_idxs, q_idx))

    if skipped_empty > 0:
        logger.log_warn(f"{tag}/eval",
                        f"Skipped {skipped_empty} episodes due to insufficient "
                        f"support/query tiles")

    # ── Episode-by-episode evaluation | 逐 episode 评估 ──
    log_every = max(10, len(all_episodes) // 10)
    for ep_idx, (query_class, support_idxs, qi) in enumerate(
        tqdm(all_episodes, desc=f"  {shot}-shot eval")
    ):
        # ── Support: image + mask → backbone → prototype ──
        support_imgs = torch.stack(
            [train_ds.load_image(si) for si in support_idxs]
        ).to(device)
        support_masks = [
            train_ds.render_class_mask(si, query_class).to(device)
            for si in support_idxs
        ]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_p4s = _extract_feats_for_proto(backbone, support_imgs, feature_level)
        if num_proto > 1:
            fg_proto = compute_multi_prototype(
                support_p4s, support_masks, num_prototypes=num_proto
            )
            if (fg_proto.dim() == 1 and fg_proto.sum() == 0) or \
               (fg_proto.dim() == 2 and fg_proto.sum() == 0):
                continue
        else:
            fg_proto = compute_fg_prototype(support_p4s, support_masks)
            if fg_proto.sum() == 0:
                continue

        # ── Query: backbone → decoder(proto) → mask ──
        logit = _decoder_forward(
            decoder, backbone, query_img, fg_proto,
            target_size=tuple(query_mask.shape),
            feature_level=feature_level,
        )
        pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)
        iou = binary_iou(pred, gt)
        per_cls_ious[query_class].append(iou)

        # 累积 recall/precision/dice | accumulate for macro averaging
        rec, prec, dice = binary_recall_precision(pred, gt)
        per_cls_recall[query_class].append(rec)
        per_cls_precision[query_class].append(prec)
        per_cls_dice[query_class].append(dice)

        if (ep_idx + 1) % log_every == 0:
            running_cls = {c: float(np.mean(per_cls_ious[c]))
                          for c in per_cls_ious if per_cls_ious[c]}
            running_miou = float(np.mean(list(running_cls.values()))) if running_cls else 0.0
            logger.log_info(f"{tag}/progress",
                          f"  Ep {ep_idx+1}/{len(all_episodes)}: running mIoU={running_miou:.4f}")

    dt = time.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════
    # 标准 FSS mIoU: mean(per_class_mean_IoU) — 每类等权
    # Standard FSS mIoU: mean of per-class means — equal weight per class
    # ═══════════════════════════════════════════════════════════════════
    per_cls_avg = {}
    per_cls_recall_avg = {}
    for c in sorted(per_cls_ious.keys()):
        avg = float(np.mean(per_cls_ious[c]))
        rec_avg = float(np.mean(per_cls_recall[c])) if per_cls_recall[c] else 0.0
        prec_avg = float(np.mean(per_cls_precision[c])) if per_cls_precision[c] else 0.0
        dice_avg = float(np.mean(per_cls_dice[c])) if per_cls_dice[c] else 0.0
        cls_name = target_classes.get(c, f"class_{c}")
        per_cls_avg[str(c)] = avg
        per_cls_recall_avg[str(c)] = rec_avg
        logger.log_info(f"{tag}/per_cls",
                       f"  {cls_name:<20} IoU={avg:.4f}  Rec={rec_avg:.4f}  "
                       f"Dice={dice_avg:.4f}  ({len(per_cls_ious[c])} eps)")

    # 标准 mIoU = mean(per_class_means) | Standard mIoU = mean(per-class means)
    per_class_means = [per_cls_avg[str(c)] for c in valid_classes if str(c) in per_cls_avg]
    miou_mean = float(np.mean(per_class_means)) if per_class_means else 0.0
    miou_std = float(np.std(per_class_means)) if per_class_means else 0.0  # std across classes

    # 全局统计 (保留兼容) | Global stats (kept for compatibility)
    all_ious_flat = [iou for ious in per_cls_ious.values() for iou in ious]

    # Group stats | 分组统计
    group_stats = {}
    for group_name, group_classes in CLASS_GROUPS.items():
        group_means = []
        for c in group_classes:
            if c in per_cls_ious and per_cls_ious[c]:
                group_means.append(float(np.mean(per_cls_ious[c])))
        group_stats[group_name] = {
            "miou": float(np.mean(group_means)) if group_means else 0.0,
            "n_classes": len(group_means),
        }

    # Recall — mean of per-class means
    per_class_recalls = [per_cls_recall_avg[str(c)] for c in valid_classes
                         if str(c) in per_cls_recall_avg]
    mrecall = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0

    for group_name, stats in group_stats.items():
        logger.log_info(f"{tag}/groups",
                       f"  [{group_name}] mIoU={stats['miou']:.4f} ({stats['n_classes']} classes)")

    logger.log_info(f"{tag}/done",
                   f"  {shot}-shot: mIoU={miou_mean:.4f} (std={miou_std:.4f})  "
                   f"mRecall={mrecall:.4f}  "
                   f"({len(all_ious_flat)} eps × {len(valid_classes)} classes, {dt:.0f}s)")

    result = {
        "miou_mean": miou_mean,          # ← 标准 FSS: mean(per_class_means)
        "miou_std": miou_std,            # ← std across per-class means
        "mrecall_mean": mrecall,
        "n_valid": len(all_ious_flat),
        "n_classes_evaluated": len(valid_classes),
        "n_episodes_per_class": n_per_class,
        "per_class_iou": per_cls_avg,
        "per_class_recall": per_cls_recall_avg,
        "group_stats": group_stats,
        "time_s": dt,
    }
    return result

# ═══════════════════════════════════════════════════════════════════
# Timing Helper | 计时工具
# ═══════════════════════════════════════════════════════════════════

def _log_timing_summary(logger, tag, timing_accum, dt_total, n_epochs):
    """打印计时总结 | Print timing breakdown summary."""
    logger.log_info(f"{tag}/timing", f"\n{'='*60}")
    logger.log_info(f"{tag}/timing", f"  TIMING BREAKDOWN ({n_epochs} epochs, {dt_total:.0f}s total)")
    logger.log_info(f"{tag}/timing", f"{'='*60}")

    # Train sub-steps (per-epoch average)
    logger.log_info(f"{tag}/timing", f"  ── Train per epoch (avg) ──")
    train_keys = [k for k in sorted(timing_accum) if k.startswith("train_")]
    for k in train_keys:
        step_name = k.replace("train_", "  ")
        avg_s = timing_accum[k] / n_epochs
        pct = timing_accum[k] / dt_total * 100 if dt_total > 0 else 0
        logger.log_info(f"{tag}/timing", f"  {step_name:<25s}: {avg_s:7.1f}s/epoch ({pct:5.1f}%)")

    # Val sub-steps (per-epoch average)
    val_keys = [k for k in sorted(timing_accum) if k.startswith("val_")]
    if val_keys:
        logger.log_info(f"{tag}/timing", f"  ── Val per epoch (avg) ──")
        for k in val_keys:
            step_name = k.replace("val_", "  ")
            avg_s = timing_accum[k] / n_epochs
            pct = timing_accum[k] / dt_total * 100 if dt_total > 0 else 0
            logger.log_info(f"{tag}/timing", f"  {step_name:<25s}: {avg_s:7.1f}s/epoch ({pct:5.1f}%)")

    # Totals
    total_train = sum(timing_accum[k] for k in train_keys)
    total_val = sum(timing_accum[k] for k in val_keys)
    logger.log_info(f"{tag}/timing", f"  ── Totals ──")
    logger.log_info(f"{tag}/timing", f"  {'train_total':<25s}: {total_train:7.0f}s ({total_train/dt_total*100:5.1f}%)")
    logger.log_info(f"{tag}/timing", f"  {'val_total':<25s}: {total_val:7.0f}s ({total_val/dt_total*100:5.1f}%)")
    logger.log_info(f"{tag}/timing", f"  {'TOTAL':<25s}: {dt_total:7.0f}s")


# ═══════════════════════════════════════════════════════════════════
# Oracle Analysis & Learning Curves | Oracle分析与学习曲线
# ═══════════════════════════════════════════════════════════════════

def _log_oracle_gap(logger, tag, oracle_per_cls, final_per_cls, target_classes, best_miou):
    """报告Oracle差距: 每类最佳epoch vs mIoU最佳epoch | Oracle gap analysis."""
    logger.log_info(f"{tag}/oracle", f"\n{'='*60}")
    logger.log_info(f"{tag}/oracle", "  ORACLE ANALYSIS — Per-Class Best vs mIoU-Best")
    logger.log_info(f"{tag}/oracle", f"{'='*60}")
    logger.log_info(f"{tag}/oracle",
                   f"  {'Class':<22} {'Oracle':>8} {'@mIoU-best':>8} {'Gap':>8} {'Best Ep':>8}")
    logger.log_info(f"{tag}/oracle", f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    oracle_miou = 0.0
    n_cls = 0
    biases = []
    for cls_id in sorted(oracle_per_cls.keys()):
        cls_name = target_classes.get(cls_id, f"class_{cls_id}")
        oracle_val = oracle_per_cls[cls_id]
        final_val = final_per_cls.get(cls_id, 0.0)
        gap = oracle_val - final_val
        oracle_miou += oracle_val
        n_cls += 1
        biases.append((cls_name, gap))
        logger.log_info(f"{tag}/oracle",
                       f"  {cls_name:<22} {oracle_val*100:>7.2f}% {final_val*100:>7.2f}% "
                       f"{gap*100:>+7.2f}% {'':>8}")

    oracle_miou /= max(n_cls, 1)
    bias_gap = oracle_miou - best_miou
    logger.log_info(f"{tag}/oracle", f"  {'─'*54}")
    logger.log_info(f"{tag}/oracle",
                   f"  {'Oracle mIoU':<22} {oracle_miou*100:>7.2f}%  "
                   f"(+{bias_gap*100:.2f}% vs best)")
    logger.log_info(f"{tag}/oracle", f"  {'Best mIoU':<22} {best_miou*100:>7.2f}%")
    logger.log_info(f"{tag}/oracle",
                   f"  {'Optimization Bias':<22} {bias_gap*100:>7.2f}%  "
                   f"← per-class gap due to mIoU selection")

    # Biggest victims
    biases.sort(key=lambda x: -x[1])
    logger.log_info(f"{tag}/oracle", f"  Top victims of mIoU selection bias:")
    for name, gap in biases[:3]:
        logger.log_info(f"{tag}/oracle", f"    {name}: oracle better by {gap*100:+.1f}%")

    return oracle_miou, bias_gap


def _plot_learning_curves(metrics_path, output_dir, decoder_type, shot, target_classes):
    """绘制每类学习曲线 | Plot per-class learning curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import json as _json
    with open(metrics_path) as f:
        lines = f.readlines()
    if not lines:
        return
    epochs_data = [_json.loads(l) for l in lines]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # — Left: Per-class IoU curves —
    ax = axes[0]
    colors = plt.cm.tab10.colors
    for i, cls_id in enumerate(sorted(target_classes.keys())):
        vals = [e["per_cls"].get(str(cls_id), 0) * 100 for e in epochs_data]
        if max(vals) < 0.1:
            continue
        name = target_classes[cls_id].replace("_", " ")
        color = colors[i % len(colors)]
        ax.plot(range(1, len(vals) + 1), vals, color=color, linewidth=1.2, alpha=0.8, label=name)
        # Mark best epoch
        best_i = max(range(len(vals)), key=lambda j: vals[j])
        ax.scatter(best_i + 1, vals[best_i], color=color, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation IoU (%)", fontsize=10)
    ax.set_title(f"Per-Class Learning Curves ({decoder_type} {shot}-shot)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    # — Right: mIoU + Optimization Bias —
    ax2 = axes[1]
    mious = [e["val_miou"] * 100 for e in epochs_data]
    ax2.plot(range(1, len(mious) + 1), mious, color="black", linewidth=2, label="mIoU (best)")
    ax2.scatter(max(range(len(mious)), key=lambda j: mious[j]) + 1,
               max(mious), color="black", s=60, zorder=5, marker="*",
               edgecolors="white", linewidth=1)

    # Highlight the gap: mark the epoch range where small classes peaked vs mIoU peaked
    best_miou_ep = max(range(len(mious)), key=lambda j: mious[j]) + 1
    ax2.axvline(x=best_miou_ep, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label=f"mIoU best (E{best_miou_ep})")

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("mIoU (%)", fontsize=10)
    ax2.set_title(f"mIoU vs Optimization Bias ({decoder_type} {shot}-shot)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Learning Dynamics Analysis — {decoder_type} {shot}-shot",
                fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = Path(output_dir) / f"learning_curves_{decoder_type}_{shot}shot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop | 主训练循环
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(decoder, backbone, train_ds, val_ds, device,
                       shot, target_classes, args, logger, output_dir,
                       decoder_type: str = "unknown",
                       feature_level: str = "p4"):
    """Full training + evaluation pipeline."""
    tag = f"c04/{decoder_type}/{shot}shot"
    n_classes = len(target_classes)
    logger.log_info(f"{tag}/start",
                   f"\n{'='*60}\n"
                   f"  C-04 [{decoder_type}] {shot}-Shot -- {n_classes} classes\n"
                   f"{'='*60}")

    # Optimizer & Scheduler
    # beta2=0.95 适合少样本: 有效记忆 ~20 步 (vs 默认 0.999 的 ~1000 步)
    # beta2=0.95 for few-shot: effective memory ~20 steps (vs default 0.999 ~1000 steps)
    # 优化器参数: decoder + 可选的 backbone LoRA
    # Optimizer params: decoder + optional backbone LoRA
    opt_params = list(decoder.parameters())
    lora_params = [p for p in backbone.parameters() if p.requires_grad and p is not decoder]
    lora_params = [p for p in lora_params if p not in set(opt_params)]
    if lora_params:
        opt_params += lora_params
        logger.log_info(f"{tag}/opt",
                       f"Optimizing {sum(p.numel() for p in opt_params):,} params "
                       f"(+{sum(p.numel() for p in lora_params):,} LoRA)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    # warmup_epochs >= epochs 时 T_max=0 会导致 ZeroDivisionError，回退到纯 Cosine
    # When warmup_epochs >= epochs, T_max=0 causes ZeroDivisionError → fallback to pure Cosine
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                         total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs,
                                   eta_min=args.lr * 0.01)
        sch = SequentialLR(opt, schedulers=[warmup, cosine],
                          milestones=[args.warmup_epochs])
    else:
        if args.warmup_epochs >= args.epochs and args.warmup_epochs > 0:
            logger.log_info(f"{tag}/sched",
                           f"warmup ({args.warmup_epochs}) >= epochs ({args.epochs}), "
                           f"skipping warmup to avoid ZeroDivisionError")
        sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Static RUR gap weights | 固定表示差距权重 ──
    # Fixed per-class weights from pre-computed ceiling (not per-epoch dynamic)
    gap_weights_fixed = {c: 1.0 for c in target_classes}
    if getattr(args, 'rur_ceiling', None):
        try:
            with open(args.rur_ceiling) as f:
                rur_ceiling_data = json.load(f)
            for cid in target_classes:
                name = target_classes[cid]
                ceil_pct = rur_ceiling_data['per_class'].get(name, {}).get('tile_p3_pct', 100)
                # gap = ceiling/100 → larger gap = higher weight (capped at 2.0)
                gap = max(0, 100 - ceil_pct) / 100
                gap_weights_fixed[cid] = min(2.0, 1.0 + gap * 1.5)
            logger.log_info(f"{tag}/loss",
                           f"Static RUR weights: " + ", ".join(
                               f"{target_classes[c][:7]}={gap_weights_fixed[c]:.2f}"
                               for c in sorted(target_classes) if gap_weights_fixed[c] > 1.01))
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            logger.log_info(f"{tag}/loss",
                           f"RUR ceiling file not found or invalid, using uniform weights")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Class sampling weights (rare class oversampling)
    # 排除零样本类（如 iSAID 的 road 在 train 中可能为 0）| Exclude zero-sample classes
    class_image_counts = {c: len(train_ds.class_to_images(c)) for c in target_classes}
    valid_classes = {c: cnt for c, cnt in class_image_counts.items() if cnt >= shot}
    zero_shot_classes = {c: cnt for c, cnt in class_image_counts.items() if cnt < shot}

    if zero_shot_classes:
        logger.log_info(f"{tag}/classes",
                       f"  !! EXCLUDED (insufficient samples for {shot}-shot): "
                       + ", ".join(f"{c}({target_classes[c]}={cnt})"
                                   for c, cnt in sorted(zero_shot_classes.items())))

    total_images = sum(valid_classes.values())
    class_weights = {c: total_images / cnt for c, cnt in valid_classes.items()}
    weight_sum = sum(class_weights.values())
    class_probs = {c: w / weight_sum for c, w in class_weights.items()}

    logger.log_info(f"{tag}/classes", f"  Training on {len(valid_classes)}/{len(target_classes)} classes "
                   f"(shot={shot}, excluded {len(zero_shot_classes)} with <{shot} samples)")

    logger.log_info(f"{tag}/classes", "  Class distribution (train images):")
    for c in sorted(target_classes):
        name = target_classes[c]
        cnt = class_image_counts[c]
        if c in zero_shot_classes:
            marker = " !! SKIP (0 imgs)"
            prob_str = "---"
        else:
            prob = class_probs[c]
            marker = " !! RARE" if cnt < 20 else ""
            prob_str = f"{prob:.3f}"
        logger.log_info(f"{tag}/classes",
                       f"    {c:2d} {name:<20} {cnt:4d} imgs  p={prob_str}{marker}")

    # ── EMA Model | 指数移动平均 ──
    # EMA stabilizes few-shot training by smoothing weight updates across episodes
    ema_decay = getattr(args, 'ema_decay', 0.999)
    ema_state = {k: v.clone() for k, v in decoder.state_dict().items()} if ema_decay > 0 else None

    # ── SWA config | 随机权重平均 ──
    swa_start = getattr(args, 'swa_start_epoch', 0)
    if swa_start <= 0:
        swa_start = max(1, int(args.epochs * 0.6))  # auto: start at 60% of training
    swa_state = None
    swa_n = 0
    logger.log_info(f"{tag}/train",
                   f"EMA decay={ema_decay}, SWA start=E{swa_start}")

    # Training State
    best_val_miou = 0.0
    best_epoch = 1
    best_state = None
    best_ema_miou = 0.0
    best_ema_state = None
    best_per_cls_state = {}
    best_per_cls_iou = defaultdict(float)
    metrics_path = output_dir / f"decoder_{decoder_type}_{shot}shot_metrics.jsonl"
    rng = np.random.RandomState(args.seed)
    rng_val = np.random.RandomState(args.seed + 1)

    class_list = list(valid_classes.keys())
    class_sample_probs = [class_probs[c] for c in class_list]

    t0 = time.perf_counter()
    timing_accum = defaultdict(float)

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.perf_counter()
        decoder.train()
        total_loss, n_eps, n_empty = 0.0, 0, 0

        pbar = tqdm(range(args.episodes_per_epoch),
                    desc=f"  E{epoch}/{args.epochs}", leave=False)
        for _ in pbar:
            query_class = int(rng.choice(class_list, p=class_sample_probs))
            candidates = train_ds.class_to_images(query_class)
            if len(candidates) < shot + 1:
                continue

            indices = rng.choice(candidates, shot + 1, replace=False)
            support_idxs = indices[:shot].tolist()
            qi = int(indices[shot])  # Query also from train

            loss, ep_timing = train_episode(decoder, backbone, support_idxs, qi,
                                train_ds, query_class, device, opt,
                                scaler=scaler, grad_clip=args.grad_clip,
                                use_amp=use_amp, feature_level=feature_level,
                                use_dynamic_proto=args.use_dynamic_proto,
                                gap_weight=gap_weights_fixed.get(query_class, 1.0))
            if loss is not None:
                total_loss += loss
                n_eps += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                for k, v in ep_timing.items():
                    timing_accum[f"train_{k}"] += v
            else:
                n_empty += 1

        t_train_done = time.perf_counter()
        sch.step()
        avg_loss = total_loss / max(n_eps, 1)

        # 空原型率过高告警 | Warn if empty prototype rate is high
        if n_empty > args.episodes_per_epoch * 0.1:
            logger.log_warn(f"{tag}/train",
                          f"E{epoch}: {n_empty}/{args.episodes_per_epoch} episodes had empty prototype "
                          f"({n_empty/args.episodes_per_epoch*100:.1f}%) — check data/mask rendering")

        # Per-class validation — batched: one backbone forward for all episodes
        # 批量验证: 所有 episode 共享一次 backbone forward
        decoder.eval()
        per_cls_val, val_timing = validate_all_classes_batched(
            decoder, backbone, train_ds, val_ds,
            valid_classes, shot, device, rng_val,
            n_val_per_class=args.val_episodes_per_class,
            batch_size=args.val_batch_size,
            feature_level=feature_level)
        for k, v in val_timing.items():
            timing_accum[f"val_{k}"] += v
        for cls_id in zero_shot_classes:
            per_cls_val[cls_id] = 0.0

        mval = float(np.mean(list(per_cls_val.values())))
        t_epoch_done = time.perf_counter()

        # ── EMA Update | 指数移动平均更新 ──
        if ema_state is not None:
            alpha = min(1.0 - 1.0 / (epoch + 1), ema_decay)
            for k in ema_state:
                ema_state[k] = alpha * ema_state[k] + (1 - alpha) * decoder.state_dict()[k]

        # ── SWA Update | 随机权重平均 ──
        if epoch >= swa_start:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in decoder.state_dict().items()}
                swa_n = 1
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * swa_n + decoder.state_dict()[k]) / (swa_n + 1)
                swa_n += 1

        # Per-class best saving
        for cls_id, val_iou in per_cls_val.items():
            if val_iou > best_per_cls_iou[cls_id]:
                best_per_cls_iou[cls_id] = val_iou
                best_per_cls_state[cls_id] = {k: v.clone() for k, v in decoder.state_dict().items()}

        # Timing for this epoch
        t_epoch_train = t_train_done - t_epoch_start
        t_epoch_val = t_epoch_done - t_train_done
        t_epoch_total = t_epoch_done - t_epoch_start

        # Log
        cls_str = ", ".join(f"{target_classes[c][:7]}={per_cls_val[c]:.3f}"
                          for c in sorted(target_classes))
        logger.log_info(f"{tag}/train",
                       f"E{epoch:3d}/{args.epochs} loss={avg_loss:.4f} "
                       f"val_mIoU={mval:.4f} lr={sch.get_last_lr()[0]:.2e} "
                       f"train={t_epoch_train:.1f}s val={t_epoch_val:.1f}s "
                       f"({cls_str})")

        epoch_metrics = {
            "epoch": epoch, "loss": round(avg_loss, 6),
            "val_miou": round(mval, 6),
            "lr": round(sch.get_last_lr()[0], 8),
            "per_cls": {str(k): round(v, 6) for k, v in per_cls_val.items()},
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(epoch_metrics) + "\n")
            mf.flush()

        if mval > best_val_miou:
            best_val_miou = mval
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            torch.save(best_state, str(output_dir / f"decoder_{decoder_type}_{shot}shot_best.pt"))
            best_epoch = epoch

        # ── Early Stopping | 早停 ──
        patience = getattr(args, 'early_stop_patience', 0)
        if patience > 0:
            epochs_since_best = epoch - best_epoch
            if epochs_since_best >= patience:
                logger.log_info(f"{tag}/train",
                               f"Early stop at E{epoch}: no improvement for {patience} epochs "
                               f"(best_val={best_val_miou:.4f} @E{best_epoch})")
                break

    dt_train = time.perf_counter() - t0
    logger.log_info(f"{tag}/best",
                   f"Best overall val mIoU={best_val_miou:.4f} ({dt_train:.0f}s training)")

    # ── 计时总结 | Timing Summary ──
    _log_timing_summary(logger, tag, timing_accum, dt_train, args.epochs)

    # ── EMA Evaluation | EMA 模型评估 ──
    ema_result = None
    if ema_state is not None:
        ema_backup = {k: v.clone() for k, v in decoder.state_dict().items()}
        decoder.load_state_dict(ema_state)
        decoder.eval()
        logger.log_info(f"{tag}/ema", "Evaluating EMA model...")
        ema_result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                                   shot, args.eval_episodes, target_classes,
                                   logger, f"{tag}/ema",
                                   feature_level=feature_level)
        decoder.load_state_dict(ema_backup)
        logger.log_info(f"{tag}/ema",
                       f"EMA mIoU={ema_result['miou_mean']*100:.2f}% "
                       f"(vs best_val={best_val_miou*100:.2f}%, "
                       f"delta={ema_result['miou_mean']-best_val_miou:+4f})")

    # ── SWA Evaluation | SWA 模型评估 ──
    swa_result = None
    if swa_state is not None and swa_n >= 3:
        swa_backup = {k: v.clone() for k, v in decoder.state_dict().items()}
        decoder.load_state_dict(swa_state)
        decoder.eval()
        logger.log_info(f"{tag}/swa", f"Evaluating SWA model (n={swa_n} epochs)...")
        swa_result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                                   shot, args.eval_episodes, target_classes,
                                   logger, f"{tag}/swa",
                                   feature_level=feature_level)
        decoder.load_state_dict(swa_backup)
        logger.log_info(f"{tag}/swa",
                       f"SWA mIoU={swa_result['miou_mean']*100:.2f}% "
                       f"(vs best_val={best_val_miou*100:.2f}%, "
                       f"delta={swa_result['miou_mean']-best_val_miou:+4f})")

    # Save per-class best states + oracle evaluation
    # 保存每类最佳状态 + Oracle评估
    oracle_per_cls = {}  # cls_id -> best val IoU ever achieved
    for cls_id, state in best_per_cls_state.items():
        cls_name = target_classes[cls_id]
        torch.save(state, str(output_dir / f"decoder_{decoder_type}_{shot}shot_best_c{cls_id}_{cls_name}.pt"))
        oracle_per_cls[cls_id] = best_per_cls_iou[cls_id]

    # ── Oracle Analysis: 报告 Optimization Bias | 优化偏差分析 ──
    oracle_miou, bias_gap = _log_oracle_gap(logger, tag, oracle_per_cls, per_cls_val,
                                             target_classes, best_val_miou)

    # ── Learning Curves Plot | 学习曲线图 ──
    _plot_learning_curves(metrics_path, output_dir, decoder_type, shot, target_classes)

    # Restore best overall
    if best_state:
        decoder.load_state_dict(best_state)

    # Full Evaluation
    logger.log_info(f"{tag}/eval", "Evaluating best model...")
    result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                           shot, args.eval_episodes, target_classes,
                           logger, f"{tag}/eval",
                           feature_level=feature_level)

    result["best_val_miou"] = best_val_miou
    result["per_cls_best"] = dict(best_per_cls_iou)
    result["oracle_miou"] = oracle_miou
    result["optimization_bias"] = bias_gap
    result["train_time_s"] = dt_train
    result["class_image_counts"] = class_image_counts
    if ema_result is not None:
        result["ema_miou"] = ema_result["miou_mean"]
    if swa_result is not None:
        result["swa_miou"] = swa_result["miou_mean"]

    return decoder, result, best_val_miou

# ═══════════════════════════════════════════════════════════════════
# Non-Parametric Evaluation (C-02A baseline) | 非参数评估
# ═══════════════════════════════════════════════════════════════════

class NonParametricMatcher(nn.Module):
    """C-02A style non-parametric prototype matcher."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_p4, fg_prototype, target_size=None):
        q_norm = F.normalize(query_p4, dim=1, p=2)
        p_norm = F.normalize(fg_prototype, dim=0, p=2)  # true cosine similarity
        sim = (q_norm * p_norm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        logit = sim / self.temperature
        if target_size is not None:
            logit = F.interpolate(logit, size=target_size, mode="bilinear",
                                  align_corners=False)
        return logit


# ═══════════════════════════════════════════════════════════════════
# Main | 主函数
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="C-04: Full-Category Few-Shot on iSAID")
    p.add_argument("--src-root", type=str, required=True,
                   help="iSAID 数据根目录 (e.g. data/iSAID_processed, /root/autodl-tmp/iSAID_processed)")
    p.add_argument("--classes", type=str, default="all",
                   help="Comma-separated class IDs, or 'all' for all 15")
    p.add_argument("--shots", type=str, default="1,3,5")
    p.add_argument("--decoder", type=str, default="baseline",
                   help="baseline, film, contrastive. Comma-separated OK.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--episodes-per-epoch", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--val-episodes-per-class", type=int, default=10,
                   help="每类每 epoch 验证 episode 数 (默认 10, 30ep 已足够稳定)")
    p.add_argument("--val-batch-size", type=int, default=0,
                   help="批量验证时每批处理的图像数 (0=auto: 6GB→4, 12GB→12, 24GB→32)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--non-parametric", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/c04_full_fewshot")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-prototypes", type=int, default=1,
                   help="多原型数量 (K-means聚类, 1=mean pool. 适用所有decoder)")
    p.add_argument("--tile", action="store_true",
                   help="Tile 模式: 切分全图为 896x896 tiles (stride=512, overlap=384)")
    p.add_argument("--tile-size", type=int, default=896)
    p.add_argument("--tile-stride", type=int, default=512)
    p.add_argument("--tile-cache-size", type=int, default=0,
                   help="Tile wrapper 原图 LRU 缓存大小 (0=auto: GPU VRAM/2GB, 默认 32)")
    p.add_argument("--feature-level", type=str, default="p4",
                   choices=["p3", "p4", "p8", "p3p4"],
                   help="Backbone 特征层 | p3(stride=8), p4(stride=16), p8(stride=32), p3p4(fusion)")
    p.add_argument("--use-dynamic-proto", action="store_true",
                   help="Query-aware dynamic prototype (cross-attn query×proto)")
    p.add_argument("--rur-ceiling", type=str, default=None,
                   help="RUR ceiling JSON path for gap-aware loss weighting")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Parse classes
    if args.classes == "all":
        target_classes = dict(ALL_ISAID_CLASSES)
    else:
        ids = [int(x.strip()) for x in args.classes.split(",")]
        target_classes = {cid: ALL_ISAID_CLASSES[cid] for cid in ids if cid in ALL_ISAID_CLASSES}

    shots = [int(x.strip()) for x in args.shots.split(",")]
    decoder_types = [x.strip() for x in args.decoder.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("c04")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "c04.jsonl")))

    logger.log_info("c04/config",
                   f"C-04 Full-Category Few-Shot | {len(target_classes)} classes, "
                   f"shots={shots}, decoders={decoder_types}")
    logger.log_info("c04/config",
                   f"Training: {args.epochs} epochs x {args.episodes_per_epoch} episodes "
                   f"(warmup={args.warmup_epochs}, grad_clip={args.grad_clip}, AMP={args.amp})")

    # Auto-detect tile cache size from GPU VRAM | 根据显存自动选择缓存大小
    if args.tile and args.tile_cache_size == 0:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            args.tile_cache_size = max(16, int(vram_gb / 2))  # 1 cache slot ≈ 2GB full image
            logger.log_info("c04/config",
                           f"Auto tile-cache-size={args.tile_cache_size} (VRAM={vram_gb:.1f}GB)")
        else:
            args.tile_cache_size = 16
            logger.log_info("c04/config", "Auto tile-cache-size=16 (CPU mode)")

    # Auto val batch size from VRAM | 根据显存自动选择验证批大小
    if args.val_batch_size == 0:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            # p3p4 stores 2× features per tile → halve batch size
            mem_factor = 0.5 if args.feature_level == "p3p4" else 1.0
            if vram_gb <= 6:
                args.val_batch_size = max(2, int(4 * mem_factor))
            elif vram_gb <= 12:
                args.val_batch_size = max(4, int(12 * mem_factor))
            else:
                args.val_batch_size = max(8, int(32 * mem_factor))
            logger.log_info("c04/config",
                           f"Auto val-batch-size={args.val_batch_size} (VRAM={vram_gb:.1f}GB"
                           f"{', p3p4 halved' if mem_factor<1 else ''})")
        else:
            args.val_batch_size = 2

    # Load data
    train_ds = ISAIDInstanceDataset(args.src_root, split="train")
    val_ds = ISAIDInstanceDataset(args.src_root, split="val")

    use_tile = args.tile
    if use_tile:
        train_ds = ISAIDTileWrapper(train_ds, tile_size=args.tile_size,
                                     stride=args.tile_stride)
        train_ds._cache_max = args.tile_cache_size
        val_ds = ISAIDTileWrapper(val_ds, tile_size=args.tile_size,
                                   stride=args.tile_stride)
        val_ds._cache_max = args.tile_cache_size
        logger.log_info("c04/data",
                       f"iSAID Tile: {len(train_ds)} train tiles, {len(val_ds)} val tiles "
                       f"({args.tile_size}x{args.tile_size}, stride={args.tile_stride}, "
                       f"LRU cache={args.tile_cache_size})")
    else:
        logger.log_info("c04/data",
                       f"iSAID: {len(train_ds)} train, {len(val_ds)} val images")

    for c in sorted(target_classes):
        n_train = len(train_ds.class_to_images(c))
        n_val = len(val_ds.class_to_images(c))
        logger.log_info("c04/data",
                       f"  Class {c:2d} ({target_classes[c]:<20}): "
                       f"{n_train:4d} train, {n_val:4d} val images")

    # Backbone (frozen, shared)
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()
    logger.log_info("c04/model", f"FastSAM backbone (frozen) on {device}")

    # 特征层级 + 自动检测通道数 | Feature level + auto-detect feat_dim
    feature_level = args.feature_level
    with torch.no_grad():
        dummy = torch.randn(1, 3, 896, 896).to(device)
        probe = backbone(dummy)
        if feature_level == "p3p4":
            feat_dim_p3 = probe["p3"].shape[1]
            feat_dim_p4 = probe["p4"].shape[1]
            feat_dim = feat_dim_p3  # for logging; actual dims passed to decoder
            logger.log_info("c04/model",
                           f"Feature level: p3p4 fusion, p3_dim={feat_dim_p3}, p4_dim={feat_dim_p4}")
        else:
            feat_dim = probe[feature_level].shape[1]
            logger.log_info("c04/model",
                           f"Feature level: {feature_level}, feat_dim={feat_dim}")

    # Non-Parametric Baseline
    if args.non_parametric:
        logger.log_info("c04/mode", "Non-parametric baseline (C-02A style)")
        model = NonParametricMatcher().to(device)
        model.eval()
        nonparam_results = {}
        for shot in shots:
            tag = f"c04/nonparam/{shot}shot"
            result = evaluate_full(model, backbone, train_ds, val_ds, device,
                                   shot, args.eval_episodes, target_classes,
                                   logger, tag, feature_level=feature_level)
            nonparam_results[f"{shot}shot"] = result
            logger.log_metric(f"c04/nonparam_miou_{shot}shot", result["miou_mean"],
                            tags=["c04", "nonparam", f"{shot}shot"])
        summary = {
            "experiment": "C-04 Non-Parametric Baseline",
            "decoder": "non_parametric",
            "target_classes": {str(k): v for k, v in target_classes.items()},
            "timestamp": datetime.now().isoformat(),
            "results": {k: {"miou_mean": v["miou_mean"], "miou_std": v["miou_std"],
                           "n_valid": v["n_valid"], "per_class_iou": v["per_class_iou"]}
                       for k, v in nonparam_results.items()},
        }
        with open(output_dir / "c04_nonparam_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.log_info("done", f"Non-parametric results saved to {output_dir}/")
        return

    # Training + Evaluation per decoder per shot
    all_results = {}

    for decoder_type in decoder_types:
        logger.log_info(f"c04/{decoder_type}/header",
                       f"\n{'='*60}\n  Decoder: {decoder_type}\n{'='*60}")
        all_results[decoder_type] = {}

        for shot in shots:
            # 当 feature_level=p3p4 时, 自动使用 p3p4film 解码器
            actual_decoder = decoder_type
            if feature_level == "p3p4" and decoder_type == "film":
                actual_decoder = "p3p4film"
            decoder_kwargs = {"feat_dim": feat_dim}
            if feature_level == "p3p4":
                decoder_kwargs["feat_dim_p3"] = feat_dim_p3
                decoder_kwargs["feat_dim_p4"] = feat_dim_p4
            # Multi-prototype: set on decoder for proto extraction, works for all decoders
            decoder_kwargs["num_prototypes"] = args.num_prototypes
            decoder = build_decoder(actual_decoder, **decoder_kwargs).to(device)
            n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            logger.log_info(f"c04/{decoder_type}/model",
                           f"{decoder_type} decoder: {n_params:,} trainable params")

            if args.eval_only:
                ckpt_path = args.checkpoint or str(
                    output_dir / f"decoder_{decoder_type}_{shot}shot_best.pt")
                if Path(ckpt_path).exists():
                    decoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                    logger.log_info(f"c04/{decoder_type}/load", f"Loaded: {ckpt_path}")
                else:
                    logger.log_warn(f"c04/{decoder_type}/load",
                                   f"Checkpoint not found: {ckpt_path}")
                decoder.eval()
                result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                                       shot, args.eval_episodes, target_classes,
                                       logger, f"c04/{decoder_type}/{shot}shot/eval",
                                       feature_level=feature_level)
                all_results[decoder_type][f"{shot}shot"] = result
            else:
                # ── 跳过已完成实验 | Skip already-completed experiments ──
                metrics_path = output_dir / f"decoder_{decoder_type}_{shot}shot_metrics.jsonl"
                best_path = output_dir / f"decoder_{decoder_type}_{shot}shot_best.pt"
                skip = False
                if metrics_path.exists():
                    try:
                        lines = open(metrics_path).readlines()
                        if lines:
                            last = json.loads(lines[-1])
                            if last.get("epoch", 0) >= args.epochs:
                                best_skip = max(json.loads(l)["val_miou"] for l in lines)
                                logger.log_info(f"c04/{decoder_type}/{shot}shot/skip",
                                               f"Already complete (epoch {last['epoch']}/{args.epochs}), "
                                               f"best_val_mIoU={best_skip:.4f}. Skipping.")
                                skip = True
                    except Exception:
                        pass
                if skip:
                    if best_path.exists():
                        decoder.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                    decoder.eval()
                    result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                                           shot, args.eval_episodes, target_classes,
                                           logger, f"c04/{decoder_type}/{shot}shot/eval",
                                           feature_level=feature_level)
                    all_results[decoder_type][f"{shot}shot"] = result
                    # 扫描所有 epoch 取最佳 | Scan all epochs for best val_mIoU
                    best_val = max(json.loads(l)["val_miou"] for l in lines)
                    result["best_val_miou"] = best_val
                else:
                    decoder, result, best_val = train_and_evaluate(
                        decoder, backbone, train_ds, val_ds, device,
                        shot, target_classes, args, logger, output_dir,
                        decoder_type=decoder_type,
                        feature_level=feature_level)
                    all_results[decoder_type][f"{shot}shot"] = result
                logger.log_metric(f"c04/{decoder_type}_miou_{shot}shot",
                                result["miou_mean"],
                                tags=["c04", decoder_type, f"{shot}shot", "trained"])
                logger.log_metric(f"c04/{decoder_type}_bestval_{shot}shot",
                                best_val,
                                tags=["c04", decoder_type, f"{shot}shot", "val"])

    # Final Summary
    logger.log_info("c04/summary", f"\n{'='*90}")
    logger.log_info("c04/summary",
                   f"  C-04: Full-Category Few-Shot -- {len(target_classes)} classes")
    logger.log_info("c04/summary", "="*90)

    short_classes = sorted(target_classes.keys())[:8]
    header = f"  {'Decoder':<14} {'Shot':<8} {'mIoU':>10} {'mRec':>10} {'+-std':>8}"
    for c in short_classes:
        header += f" {target_classes[c][:6]:>8}"
    if len(target_classes) > 8:
        header += f" {'...':>8}"
    header += f" {'vehicle':>10} {'infra':>10} {'object':>10}"
    logger.log_info("c04/summary", header)
    sep = f"  {'-'*14} {'-'*8} {'-'*10} {'-'*8}"
    sep += f" {'-'*8}" * min(len(target_classes), 8)
    if len(target_classes) > 8:
        sep += f" {'-'*8}"
    sep += f" {'-'*10}" * 3
    logger.log_info("c04/summary", sep)

    for decoder_type in decoder_types:
        for shot in shots:
            key = f"{shot}shot"
            if key not in all_results.get(decoder_type, {}):
                continue
            r = all_results[decoder_type][key]
            mrec = r.get('mrecall_mean', 0) * 100
            line = f"  {decoder_type:<14} {shot:<8} {r['miou_mean']*100:>9.2f}% {mrec:>9.2f}% {r['miou_std']*100:>7.2f}%"
            for c in short_classes:
                pc = r["per_class_iou"].get(str(c), 0.0)
                line += f" {pc*100:>7.2f}%"
            if len(target_classes) > 8:
                line += f" {'...':>8}"
            for group in ["vehicle", "infra", "object"]:
                gs = r.get("group_stats", {}).get(group, {})
                line += f" {gs.get('miou', 0)*100:>9.2f}%"
            logger.log_info("c04/summary", line)

    # SES
    for decoder_type in decoder_types:
        res = all_results.get(decoder_type, {})
        if "1shot" in res and "5shot" in res:
            miou_1 = res["1shot"]["miou_mean"]
            miou_5 = res["5shot"]["miou_mean"]
            if miou_5 > 0:
                ses = miou_1 / miou_5
                ses_note = ""
                if ses > 1.0:
                    ses_note = " (WARN: 1-shot > 5-shot — 5-shot may be undertrained or noisy)"
                elif ses < 0.5:
                    ses_note = " (WARN: 1-shot << 5-shot — few-shot adaptation weak)"
                logger.log_info("c04/ses",
                               f"  [{decoder_type}] SES(1-shot/5-shot) = {ses:.3f} "
                               f"-> 1-shot retains {ses*100:.0f}% of 5-shot mIoU{ses_note}")

    # Save results + environment info | 保存结果 + 环境信息
    summary = {
        "experiment": "C-04 Full-Category Few-Shot Instance Segmentation",
        "dataset": "iSAID",
        "target_classes": {str(k): v for k, v in target_classes.items()},
        "timestamp": datetime.now().isoformat(),
        "environment": get_env_info(),
        "shots": shots,
        "decoders": decoder_types,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "lr": args.lr,
        "results": {},
    }
    for decoder_type in decoder_types:
        summary["results"][decoder_type] = {}
        for shot in shots:
            key = f"{shot}shot"
            if key in all_results.get(decoder_type, {}):
                r = all_results[decoder_type][key]
                summary["results"][decoder_type][key] = {
                    "miou_mean": r["miou_mean"],
                    "miou_std": r["miou_std"],
                    "mrecall_mean": r.get("mrecall_mean", 0.0),
                    "n_valid": r["n_valid"],
                    "per_class_iou": r["per_class_iou"],
                    "per_class_recall": r.get("per_class_recall", {}),
                    "group_stats": r.get("group_stats", {}),
                    "best_val_miou": r.get("best_val_miou", 0.0),
                    "oracle_miou": r.get("oracle_miou", 0.0),
                    "optimization_bias": r.get("optimization_bias", 0.0),
                    "train_time_s": r.get("train_time_s", 0.0),
                }

    with open(output_dir / "c04_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.log_info("done", f"Results saved to {output_dir}/c04_results.json")


if __name__ == "__main__":
    main()
