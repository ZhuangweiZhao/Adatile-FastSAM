#!/usr/bin/env python3
"""
V3-06: Novel 类 K-Shot 微调 | Novel Class K-Shot Fine-Tuning.
==============================================================

v3 协议 Phase 3 Step 2: 在 Novel 5 类上 K-shot 微调预训练的 Decoder。
v3 Protocol Phase 3 Step 2: K-shot fine-tune pre-trained Decoder on Novel 5 classes.

训练策略 | Training Strategy:
    加载 Base 预训练 Decoder → 每个 Novel 类选 K 张 support tiles
    → 微调 Decoder (降低 lr) → COCO AP 评估
    Load Base pre-trained Decoder → select K support tiles per Novel class
    → Fine-tune Decoder (reduced lr) → COCO AP evaluation

核心设计 | Core Design:
    - Support tiles 固定 (sample_k_shot, 非随机采样), 模拟真实 few-shot 场景
    - Support tiles fixed (not random per step), simulating real few-shot scenario
    - 可选: 冻结部分 Decoder (coeff_predictor 冻结, 仅微调 refinement)
    - COCO AP 为主要评估指标 (非 mIoU)

K-Shot 缩放 | K-Shot Scaling:
    K=1/3/5/10 → AP vs K 曲线 → SSI-1=88.6% 饱和点

与 v2 train_fewshot.py 的区别 | Differences from v2:
    - v2: Episode-based FSS meta-learning (iSAID-5i, 256²)
    - v3: Base→Novel fine-tune, fixed support set (896², COCO format)
    - v3: 加载 Base 预训练权重, 不是从头训练
    - v3: COCO AP 为主指标, 不是 mIoU

用法 | Usage::

    # K=5 微调 (标准)
    python tools/train/train_fewshot.py --fold 0 --k-shot 5 \
        --checkpoint runs/v3_05_base_pretrain_F0_Adaptive_XXXX/best_model.pt

    # K=1 微调 (极限)
    python tools/train/train_fewshot.py --fold 0 --k-shot 1 \
        --checkpoint runs/v3_05_base_pretrain_F0_Adaptive_XXXX/best_model.pt \
        --epochs 30 --lr 5e-6

    # Overfit 测试 (验证 pipeline)
    python tools/train/train_fewshot.py --fold 0 --k-shot 10 --epochs 10 \
        --overfit --classes 1
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.sparse import SparsePerceptionModule
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder, ProtoOnlyDecoder
from adatile.datasets.isaid_instance_fewshot import (
    ISAIDInstanceFewShotDataset,
    sample_k_shot,
    ISAID_CATEGORIES as ISAID_CAT_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = "data/iSAID_instance_fewshot"


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def focal_loss(pred, target, gamma=5.0, eps=1e-4):
    """Focal Loss (binary) — γ=5.0 for remote sensing extreme FG/BG imbalance."""
    pred = torch.clamp(pred, eps, 1.0 - eps)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    pt = pred * target + (1 - pred) * (1 - target)
    return ((1 - pt) ** gamma * bce).mean()


def dice_loss(pred, target, smooth=1e-6):
    """Dice Loss (binary)."""
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def combined_loss(pred, target, alpha=0.5, focal_gamma=5.0):
    """组合损失: alpha * Focal + (1-alpha) * Dice."""
    fl = focal_loss(pred, target, gamma=focal_gamma)
    dl = dice_loss(pred, target)
    return alpha * fl + (1 - alpha) * dl, {"focal": fl.item(), "dice": dl.item()}


# ═══════════════════════════════════════════════════════════════════
# Support Prototype 计算 | Support Prototype Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_support_prototype(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,       # [K, 3, H, W]
    support_masks: torch.Tensor,        # [K, H, W] binary per-class
    device: torch.device,
) -> torch.Tensor:
    """
    从 K 张 support 图像计算 FG prototype (L2-normalized).
    Compute FG prototype from K support images (L2-normalized).

    :return: [1280] prototype vector.
    """
    K = support_images.shape[0]
    prototypes = []

    for i in range(K):
        img = support_images[i:i + 1].to(device)
        mask = support_masks[i:i + 1].to(device)
        feats = backbone(img)
        p4 = feats["p4"]
        mask_ds = F.interpolate(
            mask.unsqueeze(0).float(), size=p4.shape[2:], mode="nearest"
        ).squeeze(1)
        fg = mask_ds > 0.5
        if fg.sum() > 0:
            proto = p4[:, :, fg.squeeze(0)].mean(dim=-1).squeeze(0)
            prototypes.append(proto)

    if not prototypes:
        return torch.zeros(1280, device=device)
    proto = torch.stack(prototypes).mean(dim=0)
    return F.normalize(proto, dim=0)


# ═══════════════════════════════════════════════════════════════════
# K-Shot Support Cache | 构建固定 Support Set
# ═══════════════════════════════════════════════════════════════════

def build_support_cache(
    dataset: ISAIDInstanceFewShotDataset,
    class_ids: list[int],
    k_shot: int,
    backbone: FastSAMBackbone,
    device: torch.device,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """
    为每个 Novel 类预计算 support prototype (固定 K-shot).
    Pre-compute support prototype per Novel class (fixed K-shot).

    在微调开始前调用一次，之后 prototype 不变（模拟真实 few-shot: support 固定）。
    Called once before fine-tuning. Prototypes remain fixed (real few-shot: fixed support).

    :return: {class_id: prototype_tensor [1280]}.
    """
    k_indices = sample_k_shot(dataset, k=k_shot, seed=seed, per_class=True)

    class_to_support: dict[int, list[int]] = defaultdict(list)
    for idx in k_indices:
        sample = dataset[idx]
        for inst in sample["instances"]:
            cls_id = inst["category_id"]
            if cls_id in class_ids and idx not in class_to_support[cls_id]:
                class_to_support[cls_id].append(idx)

    cache: dict[int, torch.Tensor] = {}
    print(f"\n[SupportCache] Building K={k_shot} support prototypes:")

    for cls_id in sorted(class_ids):
        indices = class_to_support.get(cls_id, [])
        if len(indices) == 0:
            print(f"  ⚠ Class {cls_id} ({ISAID_CAT_NAMES.get(cls_id, '?')}): 0 support tiles!")
            continue

        support_imgs, support_masks = [], []
        for idx in indices[:k_shot]:
            sample = dataset[idx]
            support_imgs.append(sample["image"])
            H, W = sample["image"].shape[1:]
            s_mask = torch.zeros(H, W, dtype=torch.float32)
            for inst in sample["instances"]:
                if inst["category_id"] == cls_id:
                    s_mask = torch.logical_or(s_mask, inst["mask"]).float()
            support_masks.append(s_mask)

        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)
        proto = compute_support_prototype(backbone, support_imgs, support_masks, device)
        cache[cls_id] = proto
        print(f"  ✓ Class {cls_id:2d} ({ISAID_CAT_NAMES.get(cls_id, '?'):20s}): "
              f"{len(indices[:k_shot])} support tiles → |proto|={proto.norm().item():.3f}")

    print(f"  Total: {len(cache)}/{len(class_ids)} classes cached\n")
    return cache


# ═══════════════════════════════════════════════════════════════════
# 训练一步 (Few-Shot) | One Training Step (Few-Shot)
# ═══════════════════════════════════════════════════════════════════

def train_step_fewshot(
    decoder: AdaptiveSparseDecoder | ProtoOnlyDecoder,
    backbone: FastSAMBackbone,
    spm: SparsePerceptionModule | None,
    support_cache: dict[int, torch.Tensor],
    query_img: torch.Tensor,             # [1, 3, H, W]
    query_instances: list[torch.Tensor], # list of [H, W] per-instance binary masks
    class_id: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_spm: bool = False,
) -> tuple[float, dict]:
    """
    训练一步 (per-instance, 固定 support prototype).
    Train step (per-instance, fixed support prototype).

    v3 修复: per-instance mask 训练替代 union mask 训练。
    每步对 query tile 中 class_id 类的每个实例独立 forward + loss，
    loss 平均后 backward。弥合语义→实例分割鸿沟。
    v3 fix: per-instance mask training replaces union mask training.
    Each instance in the query tile gets independent forward + loss,
    averaged then backward. Bridges the semantic→instance gap.

    :return: (loss_value, metrics_dict).
    """
    decoder.train()

    # ── 0. 跳过无实例的 tile | Skip tiles with no instances ──
    if len(query_instances) == 0:
        return 0.0, {"loss": 0.0, "focal": 0.0, "dice": 0.0,
                     "pred_mean": 0.0, "class_id": class_id, "n_instances": 0}

    H, W = query_instances[0].shape
    N = len(query_instances)

    # ── 1. 预计算的 support prototype | Pre-computed support prototype ──
    support_proto = support_cache[class_id].to(device)

    # ── 2. Query → Backbone (共享于所有实例) ──
    #        Query → Backbone (shared across all instances)
    feats = backbone(query_img.to(device), extract_proto=True)
    p4 = feats["p4"]
    proto_masks = feats.get("proto")
    if proto_masks is None:
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id, "n_instances": N}

    # ── 3. SPM ──
    spm_map = None
    if use_spm and spm is not None:
        spm_map = spm(feats["p8"])

    # ── 4. Per-Instance 训练循环 | Per-Instance Training Loop ──
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    total_pred_mean = 0.0
    valid_count = 0

    for inst_mask in query_instances:
        # ── Decoder → Mask ──
        if isinstance(decoder, ProtoOnlyDecoder):
            pred_mask = decoder(proto_masks, support_proto)
        else:
            pred_mask = decoder(p4, proto_masks, support_proto, spm_map)
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.squeeze(0)

        pred_full = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)

        # ── Per-instance loss ──
        loss_i, loss_dict = combined_loss(pred_full, inst_mask.float().to(device))

        if torch.isnan(loss_i) or torch.isinf(loss_i):
            continue

        total_loss += loss_i
        total_focal += loss_dict["focal"]
        total_dice += loss_dict["dice"]
        total_pred_mean += pred_full.mean().item()
        valid_count += 1

    if valid_count == 0:
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id, "n_instances": N}

    loss = total_loss / valid_count

    # ── NaN 安全 | NaN Safety ──
    if torch.isnan(loss) or torch.isinf(loss):
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id, "n_instances": N}

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

    grad_nan = False
    for name, param in decoder.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            grad_nan = True
            break
    if grad_nan:
        optimizer.zero_grad()
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id, "n_instances": N}

    optimizer.step()
    return loss.item(), {
        "loss": loss.item(), "focal": total_focal / valid_count,
        "dice": total_dice / valid_count,
        "pred_mean": total_pred_mean / valid_count,
        "class_id": class_id, "n_instances": N,
    }


# ═══════════════════════════════════════════════════════════════════
# COCO AP 评估 | COCO AP Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_coco_ap(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    spm: SparsePerceptionModule | None,
    support_cache: dict[int, torch.Tensor],
    dataset: ISAIDInstanceFewShotDataset,
    class_ids: list[int],
    device: torch.device,
    gt_anno_path: str,
    use_spm: bool = False,
    max_samples_per_class: int = 0,
) -> dict:
    """
    COCO AP 评估 (v3 主指标) | COCO AP Evaluation (v3 primary metric).

    对每个 Novel 类: prototype → val tiles → per-class mask
    → connected components → per-instance masks → COCOeval.
    For each Novel class: prototype → val tiles → mask → CC → instances → COCOeval.

    :param dataset: val split dataset (mode should include Novel classes).
    :param class_ids: Novel 类 ID | Novel class IDs.
    :param gt_anno_path: COCO GT JSON path.
    :param max_samples_per_class: 每类最多 tile 数 (0=全部).
    :return: {"AP": float, "AP50": float, "AP75": float, "n_predictions": int, ...}.
    """
    from adatile.metrics.coco_eval import COCOInstanceEvaluator
    import cv2

    decoder.eval()
    evaluator = COCOInstanceEvaluator(gt_anno_path, iouType="segm")
    total_preds = 0

    for cls_id in tqdm(class_ids, desc="Eval COCO AP", leave=False):
        if cls_id not in support_cache:
            continue
        support_proto = support_cache[cls_id].to(device)
        tile_indices = dataset.class_to_tiles(cls_id)

        if max_samples_per_class > 0 and len(tile_indices) > max_samples_per_class:
            tile_indices = random.Random(42).sample(tile_indices, max_samples_per_class)

        for idx in tqdm(tile_indices, desc=f"  cls{cls_id}", leave=False):
            sample = dataset[idx]
            tile_id = sample["tile_id"]
            H, W = sample["image"].shape[1:]

            query_img = sample["image"].unsqueeze(0).to(device)
            feats = backbone(query_img, extract_proto=True)
            p4 = feats["p4"]
            proto_masks = feats.get("proto")
            if proto_masks is None:
                continue

            spm_map = None
            if use_spm and spm is not None:
                spm_map = spm(feats["p8"])

            if isinstance(decoder, ProtoOnlyDecoder):
                pred_prob = decoder(proto_masks, support_proto)
            else:
                pred_prob = decoder(p4, proto_masks, support_proto, spm_map)
            if pred_prob.dim() == 3:
                pred_prob = pred_prob.squeeze(0)

            pred_full = F.interpolate(
                pred_prob.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)

            # 连通分量分解为 per-instance masks | Connected components → per-instance
            pred_bin = (pred_full > 0.5).cpu().numpy().astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(pred_bin, connectivity=8)

            for label_id in range(1, num_labels):
                inst_mask = (labels == label_id)
                area = inst_mask.sum()
                if area < 16:
                    continue
                score = float(pred_full.cpu().numpy()[inst_mask].mean())
                evaluator.add_prediction(
                    image_id=tile_id, category_id=cls_id,
                    mask=inst_mask, score=score,
                )
                total_preds += 1

    if total_preds == 0:
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "APS": 0.0, "APM": 0.0, "APL": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
            "n_predictions": 0,
        }

    coco_result = evaluator.evaluate()
    return {
        "AP": round(float(coco_result.get("AP", 0.0)), 6),
        "AP50": round(float(coco_result.get("AP50", 0.0)), 6),
        "AP75": round(float(coco_result.get("AP75", 0.0)), 6),
        "APS": round(float(coco_result.get("AP_small", 0.0)), 6),
        "APM": round(float(coco_result.get("AP_medium", 0.0)), 6),
        "APL": round(float(coco_result.get("AP_large", 0.0)), 6),
        "AR1": round(float(coco_result.get("AR_max1", 0.0)), 6),
        "AR10": round(float(coco_result.get("AR_max10", 0.0)), 6),
        "AR100": round(float(coco_result.get("AR_max100", 0.0)), 6),
        "n_predictions": total_preds,
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="V3-06: Novel Class K-Shot Fine-Tuning"
    )
    # 数据 | Data
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    # 模型 | Model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Base 预训练 checkpoint 路径 | Base pre-training checkpoint path")
    p.add_argument("--decoder-type", type=str, default="adaptive",
                   choices=["adaptive", "proto_only"])
    # Few-Shot
    p.add_argument("--k-shot", type=int, default=5,
                   help="每类 support tile 数 | Support tiles per class")
    p.add_argument("--classes", type=str, default="novel")
    # 训练 | Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="微调学习率 (比 base 低 10×) | Fine-tune LR (10× lower than base)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--freeze-coeff", action="store_true",
                   help="冻结 coefficient predictor (仅微调 refinement)")
    p.add_argument("--use-spm", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    # Overfit
    p.add_argument("--overfit", action="store_true")
    # 硬件 | Hardware
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # 输出 | Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-max-samples", type=int, default=50,
                   help="COCO AP 评估每类最多 tile 数 (0=全部)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 输出目录 | Output ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/v3_06_fewshot_F{args.fold}_K{args.k_shot}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_fewshot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── 加载 Checkpoint | Load Checkpoint ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.log_info("error", f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    logger.log_info("checkpoint", f"Loaded: {ckpt_path}")
    logger.log_info("checkpoint", f"  Epoch: {checkpoint.get('epoch', '?')}, "
                    f"mIoU: {checkpoint.get('miou', checkpoint.get('best_miou', '?'))}")

    # ── 类别配置 | Class Config ──
    fold_path = Path(args.data_root) / "folds" / f"fold_{args.fold}.json"
    if fold_path.exists():
        with open(fold_path) as f:
            fold_data = json.load(f)
        base_ids = fold_data["base"]
        novel_ids = fold_data["novel"]
    else:
        from adatile.datasets.isaid_instance_fewshot import DEFAULT_FOLDS
        fd = DEFAULT_FOLDS[args.fold]
        base_ids = fd["base"]
        novel_ids = fd["novel"]

    if args.classes == "novel":
        train_class_ids = novel_ids
    else:
        train_class_ids = [int(c.strip()) for c in args.classes.split(",")]

    novel_names = [ISAID_CAT_NAMES.get(c, f"cls{c}") for c in train_class_ids]
    logger.log_info("config", f"Fold {args.fold}: Fine-tuning on {len(train_class_ids)} Novel classes: {novel_names}")
    logger.log_info("config", f"K={args.k_shot}, epochs={args.epochs}, lr={args.lr}, "
                    f"freeze_coeff={args.freeze_coeff}, overfit={args.overfit}")

    # ── 数据集 | Datasets ──
    train_ds = ISAIDInstanceFewShotDataset(
        root=args.data_root, split="train", fold=args.fold, mode="novel",
    )
    val_ds = ISAIDInstanceFewShotDataset(
        root=args.data_root, split="val", fold=args.fold, mode="novel",
    )
    logger.log_info("data", f"Train (novel): {train_ds.tile_count} tiles, "
                    f"Val (novel): {val_ds.tile_count} tiles")

    # ── 模型 | Models ──
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_backbone).to(device)
    backbone.eval()

    if args.decoder_type == "proto_only":
        decoder = ProtoOnlyDecoder(proto_dim=32, feat_dim=1280).to(device)
    else:
        decoder = AdaptiveSparseDecoder(
            in_channels=1280, proto_dim=32, use_fdr=args.use_spm,
        ).to(device)

    # 加载预训练权重 | Load pre-trained weights
    decoder_state = checkpoint.get("decoder_state_dict", checkpoint.get("model_state_dict", {}))
    if decoder_state:
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        if missing:
            logger.log_info("checkpoint", f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.log_info("checkpoint", f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.log_info("checkpoint", "Decoder weights loaded ✓")
    else:
        logger.log_info("error", "No decoder_state_dict found in checkpoint!")
        sys.exit(1)

    # ── 可选: 冻结 Coefficient Predictor | Optional: Freeze Coeff ──
    if args.freeze_coeff and hasattr(decoder, 'coeff_predictor'):
        for p in decoder.coeff_predictor.parameters():
            p.requires_grad = False
        n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        logger.log_info("model", f"Coefficient predictor frozen → {n_trainable:,} trainable params")
    else:
        n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        logger.log_info("model", f"All decoder params trainable: {n_trainable:,}")

    spm = None
    if args.use_spm:
        spm = SparsePerceptionModule(in_channels=1280).to(device)
        spm.eval()

    # ── 构建 Support Cache | Build Support Cache ──
    support_cache = build_support_cache(
        train_ds, train_class_ids, args.k_shot, backbone, device, seed=args.seed,
    )

    # ── 优化器 | Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch,
    )

    # ── Overfit 模式 | Overfit Mode ──
    overfit_sample = None
    if args.overfit:
        for cls_id in train_class_ids:
            tiles = train_ds.class_to_tiles(cls_id)
            if tiles:
                overfit_sample = train_ds[tiles[0]]
                break
        if overfit_sample and overfit_sample["instances"]:
            oc = overfit_sample["instances"][0]["category_id"]
            logger.log_info("overfit", f"Fixed query tile (class={oc}, "
                            f"{len(overfit_sample['instances'])} instances)")

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Starting K={args.k_shot} Few-Shot Fine-Tuning")
    logger.log_info("train", f"{'='*60}")

    best_ap = 0.0
    global_step = 0
    nan_skip_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses, epoch_dice_vals = [], []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            cls_id = random.choice(train_class_ids)
            if cls_id not in support_cache:
                continue

            if overfit_sample is not None:
                q_sample = overfit_sample
                if not any(i["category_id"] == cls_id for i in q_sample["instances"]):
                    continue
            else:
                tile_indices = train_ds.class_to_tiles(cls_id)
                if not tile_indices:
                    continue
                q_sample = train_ds[random.choice(tile_indices)]

            query_img = q_sample["image"].unsqueeze(0)
            # v3 修复: per-instance mask 替代 union mask
            # v3 fix: per-instance masks instead of union mask
            q_instances = []
            for inst in q_sample["instances"]:
                if inst["category_id"] == cls_id:
                    q_instances.append(inst["mask"].float())

            if len(q_instances) == 0:
                continue

            loss_val, metrics = train_step_fewshot(
                decoder, backbone, spm, support_cache,
                query_img, q_instances, cls_id,
                optimizer, device, use_spm=args.use_spm,
            )

            if loss_val >= 999.0:
                nan_skip_count += 1
                continue

            scheduler.step()
            global_step += 1
            epoch_losses.append(loss_val)
            epoch_dice_vals.append(metrics["dice"])

            if epoch_losses:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "dice": f"{np.mean(epoch_dice_vals[-50:]):.4f}",
                })

        # ── Epoch 汇总 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice_vals) if epoch_dice_vals else 0.0
        logger.log_info("epoch",
            f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} dice={avg_dice:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_skip_count}")
        logger.log_metric("loss", avg_loss, step=epoch, tags=["fewshot_train"])
        logger.log_metric("dice", avg_dice, step=epoch, tags=["fewshot_train"])

        # ── COCO AP 评估 | COCO AP Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"COCO AP Evaluation @ Epoch {epoch}")
            gt_path = Path(args.data_root) / "annotations" / "instances_val.json"
            if gt_path.exists():
                eval_result = evaluate_coco_ap(
                    decoder, backbone, spm, support_cache,
                    val_ds, train_class_ids, device,
                    gt_anno_path=str(gt_path),
                    use_spm=args.use_spm,
                    max_samples_per_class=args.eval_max_samples,
                )
                ap = eval_result["AP"]
                ap50 = eval_result["AP50"]
                ap75 = eval_result["AP75"]
                aps = eval_result["APS"]
                apm = eval_result["APM"]
                apl = eval_result["APL"]
                ar1 = eval_result["AR1"]
                ar100 = eval_result["AR100"]

                logger.log_info("eval",
                    f"  AP={ap:.4f}  AP50={ap50:.4f}  AP75={ap75:.4f}  "
                    f"APS={aps:.4f}  APM={apm:.4f}  APL={apl:.4f}  "
                    f"AR1={ar1:.4f}  AR100={ar100:.4f}  "
                    f"n_preds={eval_result['n_predictions']}  best_AP={best_ap:.4f}"
                )
                for key in ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR100"]:
                    logger.log_metric(key, eval_result[key], step=epoch, tags=["fewshot_eval"])

                # ── Per-class AP | Per-Class AP ──
                per_class = {}
                for cls_id in train_class_ids:
                    if cls_id not in support_cache:
                        continue
                    cls_result = evaluate_coco_ap(
                        decoder, backbone, spm, support_cache,
                        val_ds, [cls_id], device,
                        gt_anno_path=str(gt_path),
                        use_spm=args.use_spm,
                        max_samples_per_class=args.eval_max_samples,
                    )
                    per_class[cls_id] = {"AP": cls_result["AP"], "AP50": cls_result["AP50"]}
                cls_ap_str = "  ".join(
                    f"{ISAID_CAT_NAMES.get(c, str(c))}={per_class[c]['AP']:.3f}"
                    for c in sorted(per_class.keys()) if c in per_class
                )
                logger.log_info("eval", f"  Per-Class: {cls_ap_str}")

                # ── 保存最佳 | Save best by AP ──
                if ap > best_ap:
                    best_ap = ap
                    torch.save({
                        "epoch": epoch, "global_step": global_step,
                        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "support_cache": {k: v.clone() for k, v in support_cache.items()},
                        "AP": ap, "AP50": ap50, "AP75": ap75,
                        "APS": aps, "APM": apm, "APL": apl,
                        "per_class_AP": per_class,
                        "k_shot": args.k_shot, "novel_ids": train_class_ids,
                        "args": vars(args),
                    }, str(out_dir / "best_model.pt"))
                    logger.log_info("eval", f"  ✓ New best: AP={best_ap:.4f}")
            else:
                logger.log_info("eval", f"  ⚠ GT not found: {gt_path}")

    # ── 最终保存 | Final Save ──
    torch.save({
        "epoch": args.epochs, "global_step": global_step,
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "support_cache": {k: v.clone() for k, v in support_cache.items()},
        "best_ap": best_ap, "k_shot": args.k_shot,
        "novel_ids": train_class_ids, "args": vars(args),
    }, str(out_dir / "last_model.pt"))

    results = {
        "experiment": "V3-06 Few-Shot Fine-Tuning",
        "fold": args.fold, "k_shot": args.k_shot,
        "decoder_type": args.decoder_type,
        "freeze_coeff": args.freeze_coeff,
        "epochs": args.epochs,
        "best_AP": round(best_ap, 6),
        "nan_skip_count": nan_skip_count,
        "novel_classes": {c: ISAID_CAT_NAMES.get(c, str(c)) for c in train_class_ids},
        "base_checkpoint": str(ckpt_path),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  V3-06 K-Shot Fine-Tuning — Complete")
    print(f"  Fold: {args.fold}, K={args.k_shot}, Decoder: {args.decoder_type}")
    print(f"  Best AP: {best_ap:.4f}")
    print(f"  NaN skips: {nan_skip_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    logger.log_info("done", f"Output: {out_dir}")
    logger.log_info("done", f"Best AP: {best_ap:.4f}")


if __name__ == "__main__":
    main()
