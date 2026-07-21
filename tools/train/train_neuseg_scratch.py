#!/usr/bin/env python3
"""
AdaTile-FastSAM 从头训练 — NEU_Seg 数据集 | From-Scratch Training on NEU_Seg.
==============================================================================

使用 FastSAM backbone (YOLOv8) + PureDecoder 系列解码器，在 NEU_Seg 上从头训练，
与 SegNeXt (train_segnext.py) 公平对比。

Train FastSAM backbone + PureDecoder family from scratch on NEU_Seg,
for fair comparison with SegNeXt (train_segnext.py).

核心变化 (vs train_neuseg.py 少样本训练) | Core Changes (vs few-shot):
    1. Backbone 全部解冻，从头训练 | Backbone fully unfrozen, trained from scratch
    2. 移除 support prototype 机制 | Remove support prototype mechanism
    3. 批量训练 (bs ≥ 1) | Batch training (bs ≥ 1)
    4. PureDecoder 系列 (无需 prototype) | PureDecoder family (no prototype needed)

4 类工业缺陷分割 | 4-class industrial defect segmentation:
    BG(0) + Inclusion(1) + Patch(2) + Scratch(3)

用法 | Usage::

    # FastSAM-x + PureDecoderP2P3P4 (默认, 推荐)
    python tools/train/train_neuseg_scratch.py --epochs 200

    # FastSAM-s + PureDecoderP3P4
    python tools/train/train_neuseg_scratch.py --backbone fastsam-s --decoder-type pure_p3p4 --epochs 200

    # FastSAM-x + PureDecoder (P4 only, 最轻量)
    python tools/train/train_neuseg_scratch.py --decoder-type pure --epochs 200

    # 快速验证
    python tools/train/train_neuseg_scratch.py --epochs 10 --steps-per-epoch 100 --device cpu

对比 SegNeXt | Compare with SegNeXt::

    python tools/train/train_segnext.py --model-size tiny --epochs 200
    python tools/train/train_neuseg_scratch.py --backbone fastsam-s --decoder-type pure_p3p4 --epochs 200
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
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
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4, PureDecoderP2P3P4
from adatile.datasets.neu_seg import NEUSegDataset


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = "data/NEU_Seg"
NUM_CLASSES = 4  # BG + Inclusion + Patch + Scratch
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

# 支持的 decoder | Supported decoders
SUPPORTED_DECODERS = {"pure", "pure_p3p4", "pure_p2p3p4"}


# ═══════════════════════════════════════════════════════════════════
# 损失函数 (与 SegNeXt 一致，公平对比) | Loss Functions (same as SegNeXt)
# ═══════════════════════════════════════════════════════════════════

def multi_class_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                          smooth: float = 1e-5) -> torch.Tensor:
    """
    多类别 Dice 损失 | Multi-class Dice Loss.

    :param pred: [B, C, H, W] softmax 概率 | softmax probabilities.
    :param target: [B, H, W] 类别索引 (long) | class index (long).
    :return: scalar Dice loss.
    """
    B, C, H, W = pred.shape
    target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  ce_weight: torch.Tensor | None = None,
                  ce_alpha: float = 0.5, dice_alpha: float = 0.5) -> dict:
    """
    CE + Dice 组合损失 (与 SegNeXt 完全一致) | Combined CE + Dice loss (identical to SegNeXt).

    :return: dict with "loss", "ce", "dice" for logging.
    """
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight)
    dice = multi_class_dice_loss(pred, target)
    loss = ce_alpha * ce + dice_alpha * dice
    return {"loss": loss, "ce": ce.item(), "dice": dice.item()}


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def pad_to_32(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 [B, C, H, W] 张量 pad 到 32 的倍数 (FastSAM backbone 要求).
    Pad [B, C, H, W] tensor to multiple of 32 (FastSAM backbone requirement).
    """
    H, W = tensor.shape[2], tensor.shape[3]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)


# ═══════════════════════════════════════════════════════════════════
# 评估 (与 SegNeXt 一致) | Evaluation (same as SegNeXt)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_scratch(
    backbone: FastSAMBackbone,
    decoder: nn.Module,
    val_ds: NEUSegDataset,
    device: torch.device,
) -> dict:
    """
    评估 AdaTile-FastSAM 从头训练模型 | Evaluate AdaTile-FastSAM from-scratch model.

    与 evaluate_segnext 使用相同逻辑，保证公平对比。
    Same logic as evaluate_segnext for fair comparison.

    :return: dict with mIoU, per_class IoU, Dice, pixel_accuracy.
    """
    backbone.eval()
    decoder.eval()

    per_class_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    per_class_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_correct = 0
    total_pixels = 0
    per_sample_ious = []

    for idx in range(len(val_ds)):
        try:
            sample = val_ds[idx]
        except (ValueError, OSError, FileNotFoundError):
            continue

        img = sample["image"].unsqueeze(0).to(device)     # [1, 3, H, W]
        gt_raw = sample["masks"]
        if isinstance(gt_raw, torch.Tensor):
            gt_raw = gt_raw.numpy()
        gt = gt_raw.astype(np.int64)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)                             # [1, H, W] → [H, W]
        H_orig, W_orig = gt.shape

        # ── Pad + Backbone + Decoder ──
        img_padded = pad_to_32(img)
        feats = backbone(img_padded, extract_proto=False)

        pred = _decoder_forward(decoder, feats)
        if pred is None:
            continue
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # [1, C, H/4, W/4]

        # ── 上采样到原图 | Upsample to original resolution ──
        pred_up = F.interpolate(pred, size=(H_orig, W_orig),
                                mode="bilinear", align_corners=False)
        pred_cls = pred_up.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]

        # ── Per-class IoU | 逐类别 IoU ──
        for c in range(NUM_CLASSES):
            pred_c = (pred_cls == c)
            gt_c = (gt == c)
            per_class_intersection[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()

        total_correct += (pred_cls == gt).sum()
        total_pixels += gt.size

        # ── Per-sample mIoU | 逐样本 mIoU ──
        sample_ious = []
        for c in range(NUM_CLASSES):
            pred_c = (pred_cls == c)
            gt_c = (gt == c)
            inter = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            if union > 0:
                sample_ious.append(inter / union)
        if sample_ious:
            per_sample_ious.append(np.mean(sample_ious))

    # ── 汇总指标 | Aggregate Metrics ──
    ious = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        union = per_class_union[c]
        ious[c] = per_class_intersection[c] / max(union, 1)

    mIoU = float(np.mean(ious))
    pixel_acc = total_correct / max(total_pixels, 1)
    dice = float(2 * per_class_intersection.sum() /
                 max(per_class_intersection.sum() + per_class_union.sum(), 1))

    return {
        "mIoU": round(mIoU, 6),
        "Dice": round(dice, 6),
        "pixel_accuracy": round(float(pixel_acc), 6),
        "per_class_IoU": {name: round(float(ious[i]), 4)
                          for i, name in enumerate(CLASS_NAMES)},
        "sample_mIoU_mean": round(float(np.mean(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "sample_mIoU_median": round(float(np.median(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "n_evaluated": len(per_sample_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# 模型构建 | Model Construction
# ═══════════════════════════════════════════════════════════════════

def _probe_channels(backbone: FastSAMBackbone, device: torch.device) -> dict[str, int]:
    """
    触发 backbone 探测，返回所有层的通道数 | Trigger backbone probing, return channel counts.

    送一张 dummy 224×224 图像走一遍 backbone，触发布局探测。
    Feed a dummy 224×224 image through backbone to trigger stride probing.
    """
    dummy = torch.randn(1, 3, 224, 224, device=device)
    backbone(dummy, extract_proto=False)
    return backbone.channels  # {"p2": int, "p3": int, "p4": int, "p8": int}


def build_decoder_scratch(decoder_type: str, ch: dict[str, int],
                          logger) -> nn.Module:
    """
    构建 PureDecoder (无需 prototype) | Build PureDecoder (no prototype needed).

    :param decoder_type: "pure", "pure_p3p4", "pure_p2p3p4".
    :param ch: backbone 通道数 | Backbone channel counts {"p2","p3","p4","p8"}.
    :param logger: logger instance.
    :return: decoder module.
    """
    out_channels = NUM_CLASSES

    if decoder_type == "pure":
        decoder = PureDecoder(in_channels=ch["p4"], out_channels=out_channels)
    elif decoder_type == "pure_p3p4":
        decoder = PureDecoderP3P4(
            p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=out_channels,
        )
    elif decoder_type == "pure_p2p3p4":
        decoder = PureDecoderP2P3P4(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=out_channels,
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    params = sum(p.numel() for p in decoder.parameters())
    logger.log_info("model",
        f"Decoder: {decoder.__class__.__name__}, "
        f"{params/1e3:.1f}K params, out_channels={out_channels}")
    return decoder


def _decoder_forward(decoder: nn.Module, feats: dict) -> torch.Tensor | None:
    """
    统一的 decoder 前向传播 | Unified decoder forward.

    根据 decoder 类型自动选择正确的特征组合。
    Auto-selects correct feature combination based on decoder type.

    :return: [1, C, H/4, W/4] softmax probs, or None if required features missing.
    """
    if isinstance(decoder, PureDecoderP2P3P4):
        p2 = feats.get("p2")
        if p2 is None:
            return None
        return decoder(p2, feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoderP3P4):
        return decoder(feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoder):
        return decoder(feats["p4"])
    else:
        return None


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AdaTile-FastSAM From-Scratch Training on NEU_Seg"
    )

    # ── 模型 | Model ──
    p.add_argument("--backbone", type=str, default="fastsam-x",
                   choices=["fastsam-x", "fastsam-s"],
                   help="Backbone 模型: fastsam-x (YOLOv8x/68M) / fastsam-s (YOLOv8s/~14M)")
    p.add_argument("--decoder-type", type=str, default="pure_p2p3p4",
                   choices=["pure", "pure_p3p4", "pure_p2p3p4"],
                   help="Decoder 类型 | Decoder type")
    p.add_argument("--pretrained-backbone", type=str, default=None,
                   help="Backbone 预训练权重路径 (None=从头训练) | Pretrained backbone weights")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4,
                   help="批次大小 | Batch size")

    # ── 损失 | Loss ──
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced", "inverse"],
                   help="类别权重策略 | Class weight strategy")

    # ── 优化器 | Optimizer ──
    p.add_argument("--lr", type=float, default=6e-4,
                   help="学习率 | Learning rate (与 SegNeXt 一致)")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="最小学习率 | Minimum LR for cosine schedule")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="权重衰减 | Weight decay")
    p.add_argument("--warmup-iters", type=int, default=500,
                   help="warmup 迭代数 | Warmup iterations")

    # ── 硬件 | Hardware ──
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10,
                   help="每 N epochs 评估一次 | Evaluate every N epochs")
    p.add_argument("--output-dir", type=str, default=None)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 输出目录 | Output Directory ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        dec_short = {"pure": "Pure", "pure_p3p4": "PureP3P4",
                     "pure_p2p3p4": "PureP2P3P4"}[args.decoder_type]
        pretrained_tag = "_pt" if args.pretrained_backbone else ""
        args.output_dir = f"runs/adatile_scratch_{dec_short}_{args.backbone}{pretrained_tag}_NEUSeg_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_neuseg_scratch")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    logger.log_info("config", f"AdaTile-FastSAM From-Scratch Training on NEU_Seg")
    logger.log_info("config", f"Backbone: {args.backbone}, Decoder: {args.decoder_type}")
    logger.log_info("config", f"Pretrained backbone: {args.pretrained_backbone or 'None (from scratch)'}")
    logger.log_info("config", f"Training: {args.epochs} epochs × "
                    f"{args.steps_per_epoch} steps, bs={args.batch_size}, lr={args.lr}")
    logger.log_info("config", f"Device: {args.device}, Seed: {args.seed}")

    # ── 数据集 | Datasets ──
    logger.log_info("data", "Loading NEU_Seg datasets...")
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 类别权重 | Class Weights ──
    ce_weight = None
    if args.class_weights != "none":
        if hasattr(train_ds, 'get_class_stats'):
            stats = train_ds.get_class_stats()
            pixel_counts = [stats[cn]["pixels"] for cn in CLASS_NAMES]
            total = sum(pixel_counts)
            if args.class_weights == "balanced":
                raw_weights = [total / max(p, 1) for p in pixel_counts]
                mean_w = sum(raw_weights) / len(raw_weights)
                weights = [w / mean_w for w in raw_weights]
            elif args.class_weights == "inverse":
                freqs = [p / total for p in pixel_counts]
                raw_weights = [1.0 / max(f, 1e-6)**0.5 for f in freqs]
                mean_w = sum(raw_weights) / len(raw_weights)
                weights = [w / mean_w for w in raw_weights]
            ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            logger.log_info("data", f"Class weights ({args.class_weights}): "
                            f"{dict(zip(CLASS_NAMES, weights))}")

    # ── 模型 | Models ──
    logger.log_info("model", f"Building FastSAM backbone ({args.backbone})...")

    # Backbone: 使用预训练权重或从头初始化
    # Backbone: use pretrained weights or from scratch
    if args.pretrained_backbone:
        checkpoint_path = args.pretrained_backbone
    else:
        # 从头训练: 仍然加载 FastSAM 官方权重文件（YOLOv8 结构需要）
        # 但设置 freeze_backbone=False 让所有参数可训练
        # From scratch: still need the weight file for YOLOv8 structure,
        # but set freeze_backbone=False to make all params trainable
        checkpoint_path = f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt"

    backbone = FastSAMBackbone(
        freeze_backbone=False,  # ⚠️ 关键: 全部解冻，从头训练 | Key: unfreeze all
        checkpoint=checkpoint_path,
    ).to(device)

    # ── 自动探测通道数 | Auto-detect channels ──
    ch = _probe_channels(backbone, device)
    logger.log_info("model", f"Backbone channels: {ch}")

    # ── Decoder | 解码器 ──
    decoder = build_decoder_scratch(args.decoder_type, ch, logger).to(device)

    # ⚠️ FastSAMBackbone 包装了非 nn.Module 的 FastSAM 类，
    # 实际 YOLO 参数在 backbone.model.model 中
    # FastSAMBackbone wraps non-nn.Module FastSAM class,
    # actual YOLO params are in backbone.model.model
    _yolo_params = list(backbone.model.model.parameters())
    n_backbone = sum(p.numel() for p in _yolo_params)
    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in _yolo_params if p.requires_grad) + n_decoder
    logger.log_info("model",
        f"Backbone: {n_backbone/1e6:.3f}M, Decoder: {n_decoder/1e3:.1f}K, "
        f"Total trainable: {n_trainable/1e6:.3f}M")

    # ── 优化器 (与 SegNeXt 一致: AdamW + poly schedule) | Optimizer (same as SegNeXt) ──
    optim_params = _yolo_params + list(decoder.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Poly schedule: lr = lr0 * (1 - iter/max_iter)^power
    max_iters = args.epochs * args.steps_per_epoch

    def poly_lambda(current_iter):
        if current_iter < args.warmup_iters:
            return current_iter / max(args.warmup_iters, 1) * 1.0
        return (1 - (current_iter - args.warmup_iters) /
                max(max_iters - args.warmup_iters, 1)) ** 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lambda)

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train",
        f"Starting from-scratch training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    logger.log_info("train",
        f"Backbone: {'pretrained' if args.pretrained_backbone else 'from scratch (SA-1B weights as init)'}")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0
    best_dice = 0.0
    best_epoch = 0
    global_step = 0
    nan_skip_count = 0

    n_train = len(train_ds)
    indices_pool = list(range(n_train))

    # ⚠️ FastSAM backbone 永远保持 eval 模式（YOLOv8 Detect head 限制）
    # Backbone stays in eval mode permanently (YOLOv8 Detect head constraint)
    # requires_grad 已在构造函数中通过 _unfreeze() 设置为 True
    # requires_grad was set to True by _unfreeze() in constructor

    for epoch in range(1, args.epochs + 1):
        # backbone 不能调用 train() — 已通过 requires_grad 控制参数训练
        # backbone.train() is FORBIDDEN — param training controlled via requires_grad
        decoder.train()

        epoch_losses = []
        epoch_ces = []
        epoch_dices = []

        pbar = tqdm(range(args.steps_per_epoch),
                    desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            # ── 批量采样 | Batch Sampling (与 SegNeXt 一致) ──
            batch_idxs = random.sample(indices_pool, min(args.batch_size, n_train))

            imgs_list, masks_list = [], []
            for idx in batch_idxs:
                try:
                    s = train_ds[idx]
                except (ValueError, OSError, FileNotFoundError):
                    continue
                imgs_list.append(s["image"])
                m = s["masks"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m.copy()).long()
                else:
                    m = m.long()
                # Ensure [H, W] (no channel dim) | 确保 [H, W]
                if m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                masks_list.append(m)

            if len(imgs_list) == 0:
                continue

            imgs = torch.stack(imgs_list).to(device)          # [B, 3, H, W]
            gts = torch.stack(masks_list).to(device)           # [B, H, W]
            if gts.dim() == 4 and gts.shape[1] == 1:
                gts = gts.squeeze(1)                          # [B, 1, H, W] → [B, H, W]

            H_orig, W_orig = gts.shape[1], gts.shape[2]

            # ── Pad to 32 倍数 | Pad to multiple of 32 ──
            imgs_padded = pad_to_32(imgs)

            # ── 前向传播 | Forward ──
            feats = backbone(imgs_padded, extract_proto=False)

            pred = _decoder_forward(decoder, feats)
            if pred is None:
                continue

            if pred.dim() == 3:
                pred = pred.unsqueeze(0)  # [C, H/4, W/4] → [1, C, H/4, W/4]

            # ── 上采样到原图尺寸 | Upsample to original resolution ──
            pred_up = F.interpolate(pred, size=(H_orig, W_orig),
                                    mode="bilinear", align_corners=False)

            # ── 损失计算 | Loss Computation ──
            loss_dict = combined_loss(pred_up, gts, ce_weight=ce_weight)

            if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
                nan_skip_count += 1
                optimizer.zero_grad()
                continue

            epoch_losses.append(loss_dict["loss"].item())
            epoch_ces.append(loss_dict["ce"])
            epoch_dices.append(loss_dict["dice"])

            # ── 反向传播 | Backward ──
            optimizer.zero_grad()
            loss_dict["loss"].backward()

            # 梯度裁剪 | Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(backbone.model.model.parameters()) + list(decoder.parameters()),
                max_norm=5.0,
            )

            # 梯度 NaN 检测 | Gradient NaN detection
            grad_nan = False
            for name, param in list(backbone.model.model.named_parameters()) + list(decoder.named_parameters()):
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grad_nan = True
                        break
            if grad_nan:
                optimizer.zero_grad()
                nan_skip_count += 1
                continue

            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.set_postfix(
                loss=f"{np.mean(epoch_losses[-10:]):.4f}",
                ce=f"{np.mean(epoch_ces[-10:]):.4f}",
                dice=f"{np.mean(epoch_dices[-10:]):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ── 日志记录 | Logging ──
            if global_step % 20 == 0:
                logger.log_metric("loss", loss_dict["loss"].item(), step=global_step,
                                  tags=["neuseg_scratch_train"])
                logger.log_metric("ce", loss_dict["ce"], step=global_step,
                                  tags=["neuseg_scratch_train"])
                logger.log_metric("dice", loss_dict["dice"], step=global_step,
                                  tags=["neuseg_scratch_train"])

        # ── Epoch 总结 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_ce = np.mean(epoch_ces) if epoch_ces else 0.0
        avg_dice = np.mean(epoch_dices) if epoch_dices else 0.0
        logger.log_info("epoch",
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} ce={avg_ce:.4f} dice={avg_dice:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | NaN={nan_skip_count}")

        logger.log_metric("epoch_loss", avg_loss, step=epoch, tags=["neuseg_scratch"])
        logger.log_metric("epoch_ce", avg_ce, step=epoch, tags=["neuseg_scratch"])
        logger.log_metric("epoch_dice", avg_dice, step=epoch, tags=["neuseg_scratch"])

        # ── 评估 | Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*60}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")

            metrics = evaluate_scratch(backbone, decoder, val_ds, device)

            miou = metrics["mIoU"]
            logger.log_info("eval",
                f"  mIoU={miou:.4f}  dice={metrics['Dice']:.4f}  "
                f"px_acc={metrics['pixel_accuracy']:.4f}  "
                f"n={metrics['n_evaluated']}  "
                f"best_mIoU={max(best_miou, miou):.4f} (epoch {best_epoch})")
            for name in CLASS_NAMES:
                logger.log_info("eval",
                    f"    {name:>12s}: IoU={metrics['per_class_IoU'][name]:.4f}")

            logger.log_metric("mIoU", metrics["mIoU"], step=epoch,
                              tags=["neuseg_scratch_eval"])
            logger.log_metric("Dice", metrics["Dice"], step=epoch,
                              tags=["neuseg_scratch_eval"])

            # ── 保存最佳模型 (按 mIoU) | Save Best Model (by mIoU) ──
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                best_dice = metrics["Dice"]
                best_epoch = epoch

                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "backbone_type": args.backbone,
                    "decoder_type": args.decoder_type,
                    "backbone_state_dict": {k: v.clone() for k, v
                                            in backbone.state_dict().items()},
                    "decoder_state_dict": {k: v.clone() for k, v
                                           in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                    "num_classes": NUM_CLASSES,
                    "backbone_channels": ch,
                }
                torch.save(checkpoint, str(out_dir / "best_model.pt"))
                logger.log_info("eval",
                    f"  ✓ New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── 最终保存 | Final Save ──
    final_checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "backbone_type": args.backbone,
        "decoder_type": args.decoder_type,
        "backbone_state_dict": {k: v.clone() for k, v
                                in backbone.state_dict().items()},
        "decoder_state_dict": {k: v.clone() for k, v
                               in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou,
        "best_Dice": best_dice,
        "best_epoch": best_epoch,
        "args": vars(args),
        "num_classes": NUM_CLASSES,
        "backbone_channels": ch,
        "nan_skip_count": nan_skip_count,
    }
    torch.save(final_checkpoint, str(out_dir / "last_model.pt"))

    # ── 保存结果 JSON | Save Results JSON ──
    results = {
        "experiment": "AdaTile-FastSAM From-Scratch NEU_Seg",
        "backbone": args.backbone,
        "decoder_type": args.decoder_type,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "best_mIoU": round(best_miou, 6),
        "best_Dice": round(best_dice, 6),
        "best_epoch": best_epoch,
        "nan_skip_count": nan_skip_count,
        "backbone_params": n_backbone,
        "decoder_params": n_decoder,
        "trainable_params": n_trainable,
        "pretrained_backbone": args.pretrained_backbone,
        "class_weights": args.class_weights,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 完成 | Done ──
    print()
    print("=" * 60)
    print(f"  AdaTile-FastSAM From-Scratch NEU_Seg Training -- Complete")
    print(f"  Backbone: {args.backbone}, Decoder: {args.decoder_type}")
    print(f"  Epochs: {args.epochs}, Steps/epoch: {args.steps_per_epoch}")
    print(f"  Best mIoU: {best_miou:.4f} (Dice: {best_dice:.4f}) @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_skip_count}")
    print(f"  Trainable params: {n_trainable/1e6:.2f}M")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    return out_dir, best_miou


if __name__ == "__main__":
    main()
