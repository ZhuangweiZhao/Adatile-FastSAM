#!/usr/bin/env python3
"""
极简 LoRA 训练脚本 | Minimal LoRA Training Script.
====================================================

FastSAM (冻结) + ConvLoRA + PureDecoder → NEU-Seg 4 类分割。
FastSAM (frozen) + ConvLoRA + PureDecoder → NEU-Seg 4-class segmentation.

唯一创新点: Backbone ConvLoRA — SA-1B → 工业纹理的领域适配只需 ~200K 参数。
Single innovation: Backbone ConvLoRA — domain adaptation from SA-1B to industrial textures with ~200K params.

无 Adapter / Spectral / Frequency / Boundary / Lovász — 干净消融.
No Adapter / Spectral / Frequency / Boundary / Lovász — clean ablation.

用法 | Usage::

    # 主力实验 | Main experiment
    python tools/train/train_neuseg_lora.py --lora-rank 4 --epochs 50

    # LoRA 消融 | LoRA ablation
    python tools/train/train_neuseg_lora.py --lora-rank 2 --epochs 50
    python tools/train/train_neuseg_lora.py --lora-rank 8 --epochs 50

    # 纯基线 (无 LoRA) | Pure baseline (no LoRA)
    python tools/train/train_neuseg_lora.py --no-lora --epochs 50
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
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.datasets.neu_seg import NEUSegDataset

# ═══════════════════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = 4  # BG + Inclusion + Patch + Scratch
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]


# ═══════════════════════════════════════════════════════════════════════════════
# 基础数据增强 (仅空间变换 + 光度) | Basic Augmentation (spatial + photometric only)
# ═══════════════════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """
    极简增强: 翻转 + 旋转 + 亮度对比度 + 噪声。
    Minimal augmentation: flip + rotate + brightness/contrast + noise.
    不含 CLAHE/Gamma/MotionBlur/GaussianBlur — 保持干净。
    No CLAHE/Gamma/MotionBlur/GaussianBlur — keep it clean.
    """

    def __init__(self, p_flip: float = 0.5, p_rotate: float = 0.5,
                 brightness: float = 0.2, contrast: float = 0.2, noise_std: float = 0.02):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param image: [3, H, W] float32 in [0, 1].
        :param mask: [1, H, W] int64.
        :return: augmented (image, mask).
        """
        # 空间变换 (图像+掩码同步) | Spatial transforms (synced)
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])

        # 光度变换 (仅图像) | Photometric (image only)
        if torch.rand(1).item() < 0.7:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - mean_val) * c + mean_val, 0.0, 1.0)

        # 噪声 | Noise
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)

        return image, mask


# ═══════════════════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def multiclass_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                         smooth: float = 1e-6, ignore_bg: bool = True) -> torch.Tensor:
    """
    多类别 Dice Loss | Multi-class Dice Loss.
    对每个前景类别计算二值 Dice，取平均。
    Compute binary Dice per foreground class, average.

    :param pred: [B, C, H, W] softmax probabilities.
    :param target: [B, H, W] int64 class labels.
    :return: 1 - mean(Dice over FG classes).
    """
    C = pred.shape[1]
    dice_sum = 0.0
    count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c, :, :]
        target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  ce_weight: torch.Tensor | None = None,
                  ce_alpha: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    """
    CE + Dice 组合损失 | Combined CE + Dice loss.
    :return: (total_loss, {"ce": float, "dice": float}).
    """
    log_pred = torch.log(pred + 1e-7)
    # CE with optional class weights
    nll = F.nll_loss(log_pred, target, weight=ce_weight, reduction='mean')
    # Multi-class Dice
    dice = multiclass_dice_loss(pred, target)
    total = ce_alpha * nll + (1 - ce_alpha) * dice
    return total, {"ce": nll.item(), "dice": dice.item()}


# ═══════════════════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(decoder: nn.Module, backbone: FastSAMBackbone,
             dataset: NEUSegDataset, device: torch.device,
             max_samples: int = 0) -> dict:
    """
    极简多类别评估 → per-class IoU + mIoU。
    Minimal multi-class evaluation → per-class IoU + mIoU.
    """
    decoder.eval()

    per_class_inter = torch.zeros(NUM_CLASSES, device=device)
    per_class_union = torch.zeros(NUM_CLASSES, device=device)

    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).to(device).long()
        H, W = gt.shape

        # Pad to 32
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        feats = backbone(img, extract_proto=True)
        pred_prob = decoder(feats["p3"], feats["p4"])  # [C, H/4, W/4]

        # 上采样到原图 | Upsample to original
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)

        for c in range(NUM_CLASSES):
            pred_c = (pred_class == c)
            gt_c = (gt == c)
            per_class_inter[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()

    per_class_iou = {}
    for c in range(NUM_CLASSES):
        inter = per_class_inter[c].item()
        union = per_class_union[c].item()
        per_class_iou[CLASS_NAMES[c]] = round(inter / union, 6) if union > 0 else float("nan")

    valid = [v for v in per_class_iou.values() if not (v != v)]
    miou = float(np.mean(valid)) if valid else 0.0

    return {"mIoU": round(miou, 6), "per_class_IoU": per_class_iou}


# ═══════════════════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Minimal LoRA Training for NEU-Seg")

    # 数据 | Data
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    # 训练 | Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=4,
                   help="LoRA rank (0 = --no-lora 等效)")
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--no-lora", action="store_true",
                   help="禁用 LoRA → 纯基线 | Disable LoRA → pure baseline")
    # 数据增强 | Augmentation
    p.add_argument("--no-augment", action="store_true",
                   help="禁用数据增强 | Disable augmentation")
    # 类别权重 | Class weights
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced"])
    # 硬件 | Hardware
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", type=str, default="fastsam-x",
                   choices=["fastsam-x", "fastsam-s"])
    # 输出 | Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    rank = 0 if args.no_lora else args.lora_rank  # --no-lora 覆盖 rank

    # ── 输出目录 | Output Directory ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        backbone_short = "X" if args.backbone == "fastsam-x" else "S"
        lora_tag = f"LoRA_r{rank}" if rank > 0 else "NoLoRA"
        aug_tag = "_noAug" if args.no_augment else ""
        args.output_dir = f"runs/neuseg_{lora_tag}_{backbone_short}{aug_tag}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_neuseg_lora")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Minimal LoRA Training — rank={rank}, backbone={args.backbone}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── 数据集 | Datasets ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 类别权重 | Class Weights ──
    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pixel_counts = [stats[cn]["pixels"] for cn in CLASS_NAMES]
        total = sum(pixel_counts)
        raw = [total / max(p, 1) for p in pixel_counts]
        mean_w = sum(raw) / len(raw)
        weights = [w / mean_w for w in raw]
        ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.log_info("data", f"Class weights (balanced): {dict(zip(CLASS_NAMES, weights))}")

    # ── 数据增强 | Augmentation ──
    augment = None if args.no_augment else BasicAugmentation()
    logger.log_info("data", f"Augmentation: {'disabled' if args.no_augment else 'basic (flip/rot/brightness/noise)'}")

    # ── 模型 | Models ──
    logger.log_info("model", f"Building FastSAMBackbone ({args.backbone})...")
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()

    # ── ConvLoRA 注入 | Inject ConvLoRA ──
    if rank > 0:
        lora_params = backbone.apply_conv_lora(rank=rank, alpha=args.lora_alpha)
        logger.log_info("model",
            f"✓ ConvLoRA injected: rank={rank}, alpha={args.lora_alpha}, +{lora_params:,} params ({lora_params/1e3:.1f}K)")
    else:
        logger.log_info("model", "ConvLoRA disabled — pure frozen backbone baseline")

    # ── 通道探测 | Channel probing ──
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        backbone(dummy, extract_proto=False)
    ch = backbone.channels
    logger.log_info("model", f"Backbone channels: {ch}")

    # ── Decoder | Decoder ──
    decoder = PureDecoderP3P4(
        p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES,
    ).to(device)

    dec_params = sum(p.numel() for p in decoder.parameters())
    total_trainable = dec_params
    if rank > 0:
        total_trainable += lora_params
    logger.log_info("model",
        f"PureDecoderP3P4: {dec_params/1e3:.1f}K params | "
        f"Total trainable: {total_trainable/1e3:.1f}K ({total_trainable:,})")

    # ── 优化器 | Optimizer ──
    optim_params = list(decoder.parameters())
    if rank > 0:
        optim_params += backbone.get_lora_parameters()
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch,
    )

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    logger.log_info("train", f"Rank: {rank}, LR: {args.lr}, Device: {args.device}")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0
    best_epoch = 0
    global_step = 0
    rng = random.Random(args.seed)
    nan_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_ce_vals = []
        epoch_dice_vals = []

        pbar = tqdm(range(args.steps_per_epoch),
                    desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            # ── 随机采样 | Random sample ──
            q_idx = rng.randint(0, len(train_ds) - 1)
            try:
                sample = train_ds[q_idx]
            except (ValueError, OSError, FileNotFoundError):
                continue

            img = sample["image"]
            mask = sample["masks"]
            H, W = mask.shape[1:]

            if (mask > 0).sum() < 1.0:
                continue  # 跳过无缺陷样本 | Skip empty

            # ── 增强 | Augmentation ──
            if augment:
                img, mask = augment(img, mask)

            img_dev = img.unsqueeze(0).to(device)
            mask_dev = mask.to(device)

            # ── Pad to 32 ──
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img_dev = F.pad(img_dev, (0, pad_w, 0, pad_h), mode='constant', value=0)

            # ── Forward ──
            decoder.train()
            feats = backbone(img_dev, extract_proto=True)
            pred_prob = decoder(feats["p3"], feats["p4"])  # [C, H/4, W/4]

            # ── 上采样 + Loss | Upsample + Loss ──
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0)  # [C, H, W]

            target = mask_dev.squeeze(0).long()  # [H, W]

            loss_val, loss_dict = combined_loss(
                pred_full.unsqueeze(0), target.unsqueeze(0), ce_weight=ce_weight,
            )

            # ── NaN 保护 | NaN guard ──
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1
                continue

            optimizer.zero_grad()
            loss_val.backward()

            # 梯度 NaN 检测 | Gradient NaN check
            grad_nan = False
            for name, param in list(decoder.named_parameters()):
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    grad_nan = True
                    break
            if rank > 0 and not grad_nan:
                for param in backbone.get_lora_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        grad_nan = True
                        break
            if grad_nan:
                optimizer.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss_val.item())
            epoch_ce_vals.append(loss_dict["ce"])
            epoch_dice_vals.append(loss_dict["dice"])

            if epoch_losses:
                postfix = {
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "ce": f"{np.mean(epoch_ce_vals[-50:]):.4f}",
                    "dice": f"{np.mean(epoch_dice_vals[-50:]):.4f}",
                }
                pbar.set_postfix(postfix)

        # ── Epoch 汇总 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}"
        if epoch_ce_vals:
            log_msg += f" ce={np.mean(epoch_ce_vals):.4f}"
        if epoch_dice_vals:
            log_msg += f" dice={np.mean(epoch_dice_vals):.4f}"
        log_msg += f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}"
        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["neuseg_lora"])

        # ── 评估 | Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")

            result = evaluate(decoder, backbone, val_ds, device)
            miou = result["mIoU"]

            logger.log_info("eval",
                f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["neuseg_lora"])

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                checkpoint = {
                    "epoch": epoch, "global_step": global_step,
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mIoU": miou, "args": vars(args), "num_classes": NUM_CLASSES,
                }
                if rank > 0:
                    lora_weights = {k: v.clone() for k, v in backbone.model.model.state_dict().items()
                                    if any(x in k for x in ["lora_down", "lora_up"])}
                    checkpoint["lora_state_dict"] = lora_weights
                torch.save(checkpoint, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  ✓ New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── 最终保存 | Final Save ──
    final = {
        "epoch": args.epochs, "global_step": global_step,
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou, "best_epoch": best_epoch,
        "args": vars(args), "num_classes": NUM_CLASSES, "nan_skip_count": nan_count,
    }
    if rank > 0:
        lora_weights = {k: v.clone() for k, v in backbone.model.model.state_dict().items()
                        if any(x in k for x in ["lora_down", "lora_up"])}
        final["lora_state_dict"] = lora_weights
    torch.save(final, str(out_dir / "last_model.pt"))

    # ── 摘要 | Summary ──
    print(f"\n{'='*60}")
    print(f"  Minimal LoRA Training — Complete")
    print(f"  LoRA rank: {rank}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
