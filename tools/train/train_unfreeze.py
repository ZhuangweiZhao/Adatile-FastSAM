#!/usr/bin/env python3
"""
Backbone 解冻测试 | Backbone Unfreeze Test.
=============================================

系统性测试：逐层解冻 FastSAM backbone 的最后 N 层，测量 mIoU vs 可训练参数。
Systematic test: unfreeze last N layers of FastSAM backbone, measure mIoU vs params.

核心问题 | Key Question:
    FastSAM 的 5 点差距 (0.7763 vs 0.8264 DeepLabV3+) 是:
    (a) 软上限 — 冻结权重域不适应，解冻可补救 → unfreeze helps
    (b) 硬上限 — YOLOv8 架构不适合缺陷分割 → unfreeze doesn't help

实验设计 | Experiment Design:
    --unfreeze-layers 0  → 完全冻结 (baseline, 0.7631)
    --unfreeze-layers 2  → 解冻最后 2 层
    --unfreeze-layers 4  → 解冻最后 4 层
    --unfreeze-layers 8  → 解冻最后 8 层
    --unfreeze-layers -1 → 解冻全部 neck (P3/P4/P5 形成区)
    --unfreeze-layers -2 → 解冻全部 (backbone + neck)

用法 | Usage::

    # 逐层解冻测试 (在 AutoDL 跑)
    for N in 0 2 4 8; do
        python tools/train/train_unfreeze.py --epochs 200 --unfreeze-layers $N
    done
"""

from __future__ import annotations

import sys, argparse, random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.datasets.neu_seg import NEUSegDataset

# ═══════════════════════════════════════════════════════════════════
NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]


# ═══════════════════════════════════════════════════════════════════
# 增强 + 损失 + 评估 | Aug + Loss + Eval
# ═══════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """极简增强: flip + rotate + brightness + noise."""
    def __init__(self, p_flip=0.5, p_rotate=0.5, brightness=0.2, contrast=0.2, noise_std=0.02):
        self.p_flip = p_flip; self.p_rotate = p_rotate
        self.brightness = brightness; self.contrast = contrast; self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-1]); mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-2]); mask = torch.flip(mask, dims=[-2])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1]); mask = torch.rot90(mask, k, dims=[-2, -1])
        if torch.rand(1).item() < 0.7:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            m = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - m) * c + m, 0.0, 1.0)
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)
        return image, mask


def multiclass_dice_loss(pred, target, smooth=1e-6, ignore_bg=True):
    C = pred.shape[1]; dice_sum = 0.0; count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c]; target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum(); union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth); count += 1
    if count == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


@torch.no_grad()
def evaluate(decoder, backbone, dataset, device, max_samples=0):
    decoder.eval()
    per_class_inter = torch.zeros(NUM_CLASSES, device=device)
    per_class_union = torch.zeros(NUM_CLASSES, device=device)
    indices = list(range(len(dataset)))
    if max_samples > 0: indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).to(device).long()
        H, W = gt.shape
        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        feats = backbone(img, extract_proto=False)
        pred = decoder(feats["p3"], feats["p4"])
        pred_full = F.interpolate(pred.unsqueeze(0), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)
        for c in range(NUM_CLASSES):
            pc = (pred_class == c); gc = (gt == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()
    per_class_iou = {}
    for c in range(NUM_CLASSES):
        inter = per_class_inter[c].item(); union = per_class_union[c].item()
        per_class_iou[CLASS_NAMES[c]] = round(inter / union, 6) if union > 0 else float("nan")
    valid = [v for v in per_class_iou.values() if not (v != v)]
    return {"mIoU": round(float(np.mean(valid)), 6) if valid else 0.0, "per_class_IoU": per_class_iou}


# ═══════════════════════════════════════════════════════════════════
# Backbone 解冻 | Backbone Unfreeze
# ═══════════════════════════════════════════════════════════════════

def unfreeze_backbone_layers(backbone: FastSAMBackbone, num_layers: int) -> int:
    """
    解冻 YOLOv8 backbone 的最后 N 层 | Unfreeze last N layers of YOLOv8 backbone.

    YOLOv8 结构 (FastSAM-x):
        model.0  - model.9   : Backbone (CSPDarkNet)
        model.10 - model.21  : Neck (FPN + PAN, P3/P4/P5 形成)
        model.22             : Detect head (不用于分割 | not used for segmentation)

    :param backbone: FastSAMBackbone 实例.
    :param num_layers: 解冻层数 | number of layers to unfreeze.
            > 0: 解冻最后 N 层 | unfreeze last N layers
            -1:  解冻全部 Neck (model.10 - model.21)
            -2:  解冻全部 Backbone + Neck
    :return: 解冻的参数量 | number of unfrozen parameters.
    """
    # YOLOv8 内部结构 | YOLOv8 internal structure:
    #   backbone.model → FastSAM wrapper
    #   backbone.model.model → YOLO model (has named_parameters)
    #   backbone.model.model.model → nn.Sequential of layers
    yolo_model = backbone.model.model
    sequential = yolo_model.model
    total_layers = len(sequential)

    # ── 确定解冻范围 | Determine unfreeze range ──
    if num_layers == -1:
        # 全部 Neck | All Neck
        unfreeze_range = list(range(10, min(22, total_layers)))
        tag = "all-neck"
    elif num_layers == -2:
        # 全部 Backbone + Neck | All Backbone + Neck
        unfreeze_range = list(range(min(22, total_layers)))
        tag = "all"
    elif num_layers > 0:
        # 最后 N 层 | Last N layers
        # 从后往前数，排除 Detect head (最后一层)
        start = max(0, total_layers - 1 - num_layers)  # -1 排除 detect head
        unfreeze_range = list(range(start, total_layers - 1))
        tag = f"last{num_layers}"
    else:
        tag = "none"
        unfreeze_range = []

    # ── 首先冻结全部 backbone | First freeze everything ──
    for param in yolo_model.parameters():
        param.requires_grad = False

    # ── 解冻指定层 | Unfreeze specified layers ──
    unfrozen_params = 0
    unfrozen_layer_names = []
    for idx in unfreeze_range:
        if idx < total_layers:
            layer = sequential[idx]
            layer_name = layer.__class__.__name__
            for name, param in layer.named_parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()
            unfrozen_layer_names.append(f"{idx}({layer_name})")

    print(f"  Unfreeze [{tag}]: {len(unfrozen_layer_names)} layers, "
          f"{unfrozen_params/1e6:.2f}M params trainable in backbone")
    print(f"  Layers: {unfrozen_layer_names}")

    return unfrozen_params, tag


# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Backbone Unfreeze Test")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--unfreeze-layers", type=int, default=0,
                   help="解冻最后 N 层: >0=N层, -1=全部Neck, -2=全部Backbone+Neck")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--class-weights", type=str, default="none", choices=["none", "balanced"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", type=str, default="fastsam-x", choices=["fastsam-x", "fastsam-s"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── Output dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        uf_tag = f"UF{args.unfreeze_layers}" if args.unfreeze_layers != 0 else "Frozen"
        args.output_dir = f"runs/neuseg_{uf_tag}_{ts}"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_unfreeze")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Unfreeze: layers={args.unfreeze_layers}, lr={args.lr}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── Data ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pc = [stats[cn]["pixels"] for cn in CLASS_NAMES]; total = sum(pc)
        raw = [total / max(p, 1) for p in pc]; mean_w = sum(raw) / len(raw)
        ce_weight = torch.tensor([w / mean_w for w in raw], dtype=torch.float32, device=device)
    augment = None if args.no_augment else BasicAugmentation()
    logger.log_info("data", f"Aug: {'none' if args.no_augment else 'basic'}")

    # ── Backbone ──
    backbone = FastSAMBackbone(freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt").to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    # ── 解冻 Backbone 层 | Unfreeze Backbone Layers ──
    backbone_unfrozen = 0
    uf_tag = "Frozen"
    if args.unfreeze_layers != 0:
        backbone_unfrozen, uf_tag = unfreeze_backbone_layers(backbone, args.unfreeze_layers)
        logger.log_info("model", f"Backbone unfrozen: {backbone_unfrozen/1e6:.2f}M params ({uf_tag})")

    # ── Decoder ──
    decoder = PureDecoderP3P4(
        p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES,
    ).to(device)

    dec_p = sum(p.numel() for p in decoder.parameters())
    total_p = dec_p + backbone_unfrozen
    logger.log_info("model", f"Decoder: {dec_p/1e3:.1f}K | Backbone UF: {backbone_unfrozen/1e3:.1f}K | Total: {total_p/1e3:.1f}K ({total_p:,})")
    logger.log_info("model", f"Backbone P3: {ch['p3']}ch, P4: {ch['p4']}ch")

    # ── Optimizer ──
    optim_params = list(decoder.parameters())
    if args.unfreeze_layers != 0:
        optim_params += [p for p in backbone.model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * args.steps_per_epoch)

    # ── Training ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Unfreeze: {uf_tag} ({args.unfreeze_layers} layers)")
    logger.log_info("train", f"Trainable params: {total_p:,}")
    logger.log_info("train", f"{'='*60}")
    best_miou = 0.0; best_epoch = 0; global_step = 0; nan_count = 0
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []; epoch_ce = []; epoch_dice = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            q_idx = rng.randint(0, len(train_ds) - 1)
            try: sample = train_ds[q_idx]
            except: continue
            img = sample["image"]; mask = sample["masks"]; H, W = mask.shape[1:]
            if (mask > 0).sum() < 1.0: continue
            if augment: img, mask = augment(img, mask)

            img_dev = img.unsqueeze(0).to(device); mask_dev = mask.to(device)
            pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img_dev = F.pad(img_dev, (0, pad_w, 0, pad_h), mode='constant', value=0)

            decoder.train()
            if args.unfreeze_layers != 0:
                backbone.model.model.train()  # backbone 需要 train mode (有 BN)
            feats = backbone(img_dev, extract_proto=False)

            pred_prob = decoder(feats["p3"], feats["p4"])

            pred_full = F.interpolate(pred_prob.unsqueeze(0), size=(H, W),
                                      mode="bilinear", align_corners=False).squeeze(0)
            target = mask_dev.squeeze(0).long()

            log_pred = torch.log(pred_full.unsqueeze(0).clamp(1e-7, 1))
            ce = F.nll_loss(log_pred, target.unsqueeze(0), weight=ce_weight, reduction='mean')
            dice = multiclass_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))
            loss_val = 0.5 * ce + 0.5 * dice

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1; continue

            optimizer.zero_grad()
            loss_val.backward()

            grad_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in optim_params
            )
            if grad_nan: optimizer.zero_grad(); nan_count += 1; continue

            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step(); scheduler.step(); global_step += 1

            epoch_losses.append(loss_val.item()); epoch_ce.append(ce.item()); epoch_dice.append(dice.item())

            if epoch_losses:
                postfix = {"loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                           "ce": f"{np.mean(epoch_ce[-50:]):.4f}",
                           "dice": f"{np.mean(epoch_dice[-50:]):.4f}"}
                pbar.set_postfix(postfix)

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}"
        if epoch_ce: log_msg += f" ce={np.mean(epoch_ce):.4f}"
        if epoch_dice: log_msg += f" dice={np.mean(epoch_dice):.4f}"
        log_msg += f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}"
        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["unfreeze"])

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")
            result = evaluate(decoder, backbone, val_ds, device)
            miou = result["mIoU"]
            logger.log_info("eval", f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["unfreeze"])
            if miou > best_miou:
                best_miou = miou; best_epoch = epoch
                ckpt = {"epoch": epoch, "global_step": global_step,
                        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                        "backbone_state_dict": {k: v.clone() for k, v in backbone.model.model.state_dict().items()
                                                if v.requires_grad},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "mIoU": miou, "args": vars(args), "num_classes": NUM_CLASSES}
                torch.save(ckpt, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # Final save
    final = {"epoch": args.epochs, "global_step": global_step,
             "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
             "optimizer_state_dict": optimizer.state_dict(),
             "best_mIoU": best_miou, "best_epoch": best_epoch,
             "args": vars(args), "num_classes": NUM_CLASSES, "nan_skip_count": nan_count}
    torch.save(final, str(out_dir / "last_model.pt"))

    print(f"\n{'='*60}")
    print(f"  Unfreeze Training — Complete")
    print(f"  Config: {uf_tag}, Trainable: {total_p:,}")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__": main()
