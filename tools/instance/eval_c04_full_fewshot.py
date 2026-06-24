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

import sys, argparse, time, json, datetime, warnings
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
from adatile.backbone import FastSAMBackbone
from adatile.utils.label_mapping import ISAID_CATEGORIES

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
        print(f"[ProtoRefineDecoder] Trainable: {n:,}")

    def forward(self, query_p4, fg_prototype, target_size=None):
        q_norm = F.normalize(query_p4, dim=1, p=2)
        p_norm = F.normalize(fg_prototype, dim=0, p=2)  # true cosine similarity
        sim = (q_norm * p_norm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        sim = sim / self.temperature
        x = self.refine(sim)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear",
                            align_corners=False)
        return x


class FiLMFewShotDecoder(nn.Module):
    """
    CAT-SAM Lite: FiLM-conditioned InstanceDecoder.
    C-04 FiLM 变体 | C-04 FiLM variant. ~1.1M params.
    """

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
        print(f"[FiLMFewShotDecoder] Total: {n_total:,} (FiLM: {n_film:,})")

    def forward(self, query_p4, fg_prototype, target_size=None):
        x = self.proj(query_p4)

        # FiLM condition injection
        if fg_prototype is not None and fg_prototype.abs().sum() > 1e-8:
            film_out = self.film_mlp(fg_prototype)
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


class ContrastiveProtoDecoder(nn.Module):
    """
    Contrastive Prototype Decoder | 对比原型解码器.
    
    核心创新 | Core Innovation (Phase C 预备):
        Proto → Projection MLP → contrastive-friendly space
        Query P4 → Projection → cosine_sim(projected_proto) → Refine CNN → mask
    
    ~1.3M params (projection ~394K + decoder ~900K).
    """

    def __init__(self, feat_dim: int = 1280, proj_dim: int = 256,
                 temperature: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.temperature = temperature

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
        print(f"[ContrastiveProtoDecoder] Total: {n_total:,} (Proj: {n_proj:,})")

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
    :param kwargs: passed to decoder constructor (e.g. num_prototypes for crossattn)
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
    else:
        raise ValueError(f"Unknown decoder: {method}")


# ═══════════════════════════════════════════════════════════════════
# Binary IoU | 二值 IoU
# ═══════════════════════════════════════════════════════════════════

def binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Focal + Dice Loss | 损失函数
# ═══════════════════════════════════════════════════════════════════

def focal_dice_loss(logit: torch.Tensor, target: torch.Tensor,
                    gamma: float = 5.0, ce_weight: float = 0.5,
                    dice_weight: float = 0.5):
    """Focal (gamma=5) + Dice loss for binary segmentation."""
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    focal = ((1 - torch.exp(-ce)) ** gamma * ce).mean()
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum() + 1e-8
    dice = 1.0 - (2 * inter / union)
    loss = ce_weight * focal + dice_weight * dice
    return loss, {"focal": focal.item(), "dice": dice.item()}

# ═══════════════════════════════════════════════════════════════════
# Episodic Training | 回合式训练
# ═══════════════════════════════════════════════════════════════════

def train_episode(decoder, backbone, support_idxs, query_idx,
                  train_ds, val_ds, query_class, device, opt,
                  scaler=None, grad_clip=1.0, use_amp=False):
    """Single training episode: Support -> proto -> query -> loss -> backward."""
    # Support
    support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
    support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                     for si in support_idxs]

    # Query
    query_img = val_ds.load_image(query_idx).unsqueeze(0).to(device)
    query_mask = val_ds.render_class_mask(query_idx, query_class)
    query_mask = query_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    # Forward - support + query backbone (no grad on frozen backbone)
    with torch.no_grad():
        support_feats = backbone(support_imgs)
        support_p4s = [support_feats["p4"][i] for i in range(len(support_idxs))]
        num_proto = getattr(decoder, 'num_prototypes', 1)
        if num_proto > 1:
            fg_proto = compute_multi_prototype(support_p4s, support_masks,
                                                num_prototypes=num_proto)
        else:
            fg_proto = compute_fg_prototype(support_p4s, support_masks)

        # Check empty prototype
        if fg_proto.dim() == 1 and fg_proto.sum() == 0:
            return None
        if fg_proto.dim() == 2 and fg_proto.sum() == 0:
            return None

        query_p4 = backbone(query_img)["p4"]

    # Forward - query decoder (grad only through decoder)
    if use_amp:
        with torch.amp.autocast('cuda'):
            logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape[2:]))
            loss, components = focal_dice_loss(logit, query_mask)
    else:
        logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape[2:]))
        loss, components = focal_dice_loss(logit, query_mask)

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

    return loss.item()


@torch.no_grad()
def validate_episode(decoder, backbone, train_ds, val_ds, query_class,
                     shot, device, rng, n_val=30):
    """Validation: fixed episodes per class."""
    train_candidates = train_ds.class_to_images(query_class)
    val_candidates = val_ds.class_to_images(query_class)
    if len(train_candidates) < shot or not val_candidates:
        return 0.0

    num_proto = getattr(decoder, 'num_prototypes', 1)

    ious = []
    for _ in range(n_val):
        support_idxs = rng.choice(train_candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_p4s = [backbone(support_imgs)["p4"][i] for i in range(len(support_idxs))]
        if num_proto > 1:
            fg_proto = compute_multi_prototype(support_p4s, support_masks,
                                                num_prototypes=num_proto)
            if (fg_proto.dim() == 1 and fg_proto.sum() == 0) or \
               (fg_proto.dim() == 2 and fg_proto.sum() == 0):
                continue
        else:
            fg_proto = compute_fg_prototype(support_p4s, support_masks)
            if fg_proto.sum() == 0:
                continue

        logit = decoder(backbone(query_img)["p4"], fg_proto,
                       target_size=tuple(query_mask.shape))
        pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)
        ious.append(binary_iou(pred, gt))

    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate_full(decoder, backbone, train_ds, val_ds, device,
                  shot, n_episodes, target_classes, logger, tag):
    """Full evaluation post-training with per-class + group breakdown."""
    num_proto = getattr(decoder, 'num_prototypes', 1)
    class_to_images = {c: train_ds.class_to_images(c) for c in target_classes}
    rng = np.random.RandomState(42)
    all_ious, per_cls_ious = [], defaultdict(list)
    t0 = time.perf_counter()
    log_every = max(10, n_episodes // 10)

    for ep in tqdm(range(n_episodes), desc=f"  {shot}-shot eval"):
        valid_classes = [c for c in target_classes if len(class_to_images[c]) >= shot]
        if not valid_classes:
            continue
        query_class = int(rng.choice(valid_classes))

        candidates = class_to_images[query_class]
        val_candidates = val_ds.class_to_images(query_class)
        if not val_candidates:
            continue

        support_idxs = rng.choice(candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_p4s = [backbone(support_imgs)["p4"][i] for i in range(len(support_idxs))]
        if num_proto > 1:
            fg_proto = compute_multi_prototype(support_p4s, support_masks,
                                                num_prototypes=num_proto)
            if (fg_proto.dim() == 1 and fg_proto.sum() == 0) or \
               (fg_proto.dim() == 2 and fg_proto.sum() == 0):
                continue
        else:
            fg_proto = compute_fg_prototype(support_p4s, support_masks)
            if fg_proto.sum() == 0:
                continue

        logit = decoder(backbone(query_img)["p4"], fg_proto,
                       target_size=tuple(query_mask.shape))
        pred = (logit.squeeze(0).squeeze(0).cpu() > 0)
        gt = (query_mask.cpu() > 0)
        iou = binary_iou(pred, gt)
        all_ious.append(iou)
        per_cls_ious[query_class].append(iou)

        if (ep + 1) % log_every == 0 and all_ious:
            running = float(np.mean(all_ious[-log_every:]))
            total = float(np.mean(all_ious))
            logger.log_info(f"{tag}/progress",
                           f"  Ep {ep+1}/{n_episodes}: running={running:.4f} total={total:.4f}")

    dt = time.perf_counter() - t0

    # Per-class + Group stats
    per_cls_avg = {}
    group_stats = {}
    for group_name, group_classes in CLASS_GROUPS.items():
        group_ious = []
        for c in group_classes:
            if c in per_cls_ious:
                group_ious.extend(per_cls_ious[c])
        group_stats[group_name] = {
            "miou": float(np.mean(group_ious)) if group_ious else 0.0,
            "n": len(group_ious),
        }

    for c in sorted(per_cls_ious.keys()):
        avg = float(np.mean(per_cls_ious[c]))
        cls_name = target_classes.get(c, f"class_{c}")
        per_cls_avg[str(c)] = avg
        logger.log_info(f"{tag}/per_cls",
                       f"  {cls_name:<20} IoU={avg:.4f} ({len(per_cls_ious[c])} eps)")

    for group_name, stats in group_stats.items():
        logger.log_info(f"{tag}/groups",
                       f"  [{group_name}] mIoU={stats['miou']:.4f} ({stats['n']} eps)")

    result = {
        "miou_mean": float(np.mean(all_ious)) if all_ious else 0.0,
        "miou_std": float(np.std(all_ious)) if all_ious else 0.0,
        "n_valid": len(all_ious),
        "per_class_iou": per_cls_avg,
        "group_stats": group_stats,
        "time_s": dt,
    }
    logger.log_info(f"{tag}/done",
                   f"  {shot}-shot: mIoU={result['miou_mean']:.4f} "
                   f"({len(all_ious)} eps, {dt:.0f}s, {dt/max(len(all_ious),1):.2f}s/ep)")
    return result

# ═══════════════════════════════════════════════════════════════════
# Main Training Loop | 主训练循环
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(decoder, backbone, train_ds, val_ds, device,
                       shot, target_classes, args, logger, output_dir,
                       decoder_type: str = "unknown"):
    """Full training + evaluation pipeline."""
    tag = f"c04/{decoder_type}/{shot}shot"
    n_classes = len(target_classes)
    logger.log_info(f"{tag}/start",
                   f"\n{'='*60}\n"
                   f"  C-04 [{decoder_type}] {shot}-Shot -- {n_classes} classes\n"
                   f"{'='*60}")

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    if args.warmup_epochs > 0:
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                         total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs,
                                   eta_min=args.lr * 0.01)
        sch = SequentialLR(opt, schedulers=[warmup, cosine],
                          milestones=[args.warmup_epochs])
    else:
        sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

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

    # Training State
    best_val_miou = 0.0
    best_state = None
    best_per_cls_state = {}
    best_per_cls_iou = defaultdict(float)
    metrics_path = output_dir / f"decoder_{decoder_type}_{shot}shot_metrics.jsonl"
    rng = np.random.RandomState(args.seed)
    rng_val = np.random.RandomState(args.seed + 1)  # 独立 rng: validation 不影响 training 随机序列

    class_list = list(valid_classes.keys())
    class_sample_probs = [class_probs[c] for c in class_list]

    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss, n_eps = 0.0, 0

        pbar = tqdm(range(args.episodes_per_epoch),
                    desc=f"  E{epoch}/{args.epochs}", leave=False)
        for _ in pbar:
            query_class = int(rng.choice(class_list, p=class_sample_probs))
            candidates = train_ds.class_to_images(query_class)
            val_candidates = val_ds.class_to_images(query_class)
            if len(candidates) < shot or not val_candidates:
                continue

            support_idxs = rng.choice(candidates, shot, replace=False)
            qi = int(rng.choice(val_candidates))

            loss = train_episode(decoder, backbone, support_idxs, qi,
                                train_ds, val_ds, query_class, device, opt,
                                scaler=scaler, grad_clip=args.grad_clip,
                                use_amp=use_amp)
            if loss is not None:
                total_loss += loss
                n_eps += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})

        sch.step()
        avg_loss = total_loss / max(n_eps, 1)

        # Per-class validation (skip zero-shot classes)
        decoder.eval()
        per_cls_val = {}
        for cls_id in valid_classes:
            per_cls_val[cls_id] = validate_episode(
                decoder, backbone, train_ds, val_ds,
                cls_id, shot, device, rng_val, n_val=args.val_episodes_per_class)
        for cls_id in zero_shot_classes:
            per_cls_val[cls_id] = 0.0

        mval = float(np.mean(list(per_cls_val.values())))

        # Per-class best saving
        for cls_id, val_iou in per_cls_val.items():
            if val_iou > best_per_cls_iou[cls_id]:
                best_per_cls_iou[cls_id] = val_iou
                best_per_cls_state[cls_id] = {k: v.clone() for k, v in decoder.state_dict().items()}

        # Log
        cls_str = ", ".join(f"{target_classes[c][:8]}={per_cls_val[c]:.4f}"
                          for c in sorted(target_classes)[:8])
        if n_classes > 8:
            cls_str += f" ... ({n_classes-8} more)"
        logger.log_info(f"{tag}/train",
                       f"E{epoch:3d}/{args.epochs} loss={avg_loss:.4f} "
                       f"val_mIoU={mval:.4f} lr={sch.get_last_lr()[0]:.2e} "
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

    dt_train = time.perf_counter() - t0
    logger.log_info(f"{tag}/best",
                   f"Best overall val mIoU={best_val_miou:.4f} ({dt_train:.0f}s training)")

    # Save per-class best states
    for cls_id, state in best_per_cls_state.items():
        cls_name = target_classes[cls_id]
        torch.save(state, str(output_dir / f"decoder_{decoder_type}_{shot}shot_best_c{cls_id}_{cls_name}.pt"))

    # Restore best overall
    if best_state:
        decoder.load_state_dict(best_state)

    # Full Evaluation
    logger.log_info(f"{tag}/eval", "Evaluating best model...")
    result = evaluate_full(decoder, backbone, train_ds, val_ds, device,
                           shot, args.eval_episodes, target_classes,
                           logger, f"{tag}/eval")

    result["best_val_miou"] = best_val_miou
    result["per_cls_best"] = dict(best_per_cls_iou)
    result["train_time_s"] = dt_train
    result["class_image_counts"] = class_image_counts

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
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--classes", type=str, default="all",
                   help="Comma-separated class IDs, or 'all' for all 15")
    p.add_argument("--shots", type=str, default="1,3,5")
    p.add_argument("--decoder", type=str, default="baseline",
                   help="baseline, film, contrastive. Comma-separated OK.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--episodes-per-epoch", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--val-episodes-per-class", type=int, default=30)
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
                   help="多原型数量 (仅 crossattn decoder, 默认 1=mean pool)")
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

    # Load data
    train_ds = ISAIDInstanceDataset(args.src_root, split="train")
    val_ds = ISAIDInstanceDataset(args.src_root, split="val")
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
                                   logger, tag)
            nonparam_results[f"{shot}shot"] = result
            logger.log_metric(f"c04/nonparam_miou_{shot}shot", result["miou_mean"],
                            tags=["c04", "nonparam", f"{shot}shot"])
        summary = {
            "experiment": "C-04 Non-Parametric Baseline",
            "decoder": "non_parametric",
            "target_classes": {str(k): v for k, v in target_classes.items()},
            "timestamp": datetime.datetime.now().isoformat(),
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
            decoder_kwargs = {}
            if decoder_type == "crossattn":
                decoder_kwargs["num_prototypes"] = args.num_prototypes
            decoder = build_decoder(decoder_type, **decoder_kwargs).to(device)
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
                                       logger, f"c04/{decoder_type}/{shot}shot/eval")
                all_results[decoder_type][f"{shot}shot"] = result
            else:
                decoder, result, best_val = train_and_evaluate(
                    decoder, backbone, train_ds, val_ds, device,
                    shot, target_classes, args, logger, output_dir,
                    decoder_type=decoder_type)
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
    header = f"  {'Decoder':<14} {'Shot':<8} {'mIoU':>10} {'+-std':>8}"
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
            line = f"  {decoder_type:<14} {shot:<8} {r['miou_mean']*100:>9.2f}% {r['miou_std']*100:>7.2f}%"
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
                logger.log_info("c04/ses",
                               f"  [{decoder_type}] SES(1-shot/5-shot) = {ses:.3f} "
                               f"-> 1-shot retains {ses*100:.0f}% of 5-shot mIoU")

    # Save results
    summary = {
        "experiment": "C-04 Full-Category Few-Shot Instance Segmentation",
        "dataset": "iSAID",
        "target_classes": {str(k): v for k, v in target_classes.items()},
        "timestamp": datetime.datetime.now().isoformat(),
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
                    "n_valid": r["n_valid"],
                    "per_class_iou": r["per_class_iou"],
                    "group_stats": r.get("group_stats", {}),
                    "best_val_miou": r.get("best_val_miou", 0.0),
                    "train_time_s": r.get("train_time_s", 0.0),
                }

    with open(output_dir / "c04_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.log_info("done", f"Results saved to {output_dir}/c04_results.json")


if __name__ == "__main__":
    main()
