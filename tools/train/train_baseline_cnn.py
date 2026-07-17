#!/usr/bin/env python3
"""
经典 CNN 基线训练 — NEU_Seg | Classic CNN Baseline Training — NEU_Seg.
======================================================================

统一训练入口: UNet (from scratch) / DeepLabV3+ (ImageNet 预训练 encoder)。
Unified trainer: UNet (from scratch) / DeepLabV3+ (ImageNet-pretrained encoder).

训练管线与 train_segnext.py **完全同协议** (直接复用其组件):
Pipeline shares the EXACT protocol with train_segnext.py (reuses its components):
    - OfficialAugment (RandomResize 0.5~2.0 → RandomCrop → HFlip → PhotoMetric → Pad)
    - ImageNet 归一化 | ImageNet normalization
    - CE loss (ignore_index=255)
    - linear warmup 1500 iters → poly power=1.0
    - 40k iters (200 epochs × 200 steps) @ bs16, 按 Val mIoU 保存最佳

依赖 | Dependencies:
    UNet:        无额外依赖 | no extra deps
    DeepLabV3+:  pip install segmentation-models-pytorch

用法 | Usage::

    # UNet (from scratch)
    python tools/train/train_baseline_cnn.py --arch unet

    # DeepLabV3+ (ResNet-50, ImageNet)
    python tools/train/train_baseline_cnn.py --arch deeplabv3plus --encoder resnet50

    # 仅评估已有 checkpoint | Eval-only on an existing checkpoint
    python tools/train/train_baseline_cnn.py --eval-only --checkpoint runs/.../best_model.pt
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 同目录导入 train_segnext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.baselines import UNet
from adatile.datasets.neu_seg import NEUSegDataset

# ── 复用官方配方组件 (保证与 SegNeXt 基线同协议) ──
# Reuse official-recipe components (protocol-identical to the SegNeXt baseline)
from train_segnext import (
    IGNORE_INDEX, OfficialAugment, normalize_img, compute_loss,
)

NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

# 输入 pad 尺寸: 200 不是 32 的倍数, UNet/DeepLabV3+ 均需对齐下采样
# Pad size: 200 is not divisible by 32; both UNet and DeepLabV3+ need alignment
PAD_SIZE = 224

# 各架构默认学习率 | Per-arch default learning rates
ARCH_DEFAULT_LR = {
    "unet": 1e-3,           # from scratch → 较大 lr | larger lr
    "deeplabv3plus": 1e-4,  # ImageNet encoder 微调 | fine-tune
}


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utilities
# ═══════════════════════════════════════════════════════════════════

def pad_batch(imgs: torch.Tensor, gts: torch.Tensor | None = None,
              size: int = PAD_SIZE):
    """
    右下 pad 到 size² (img→0, mask→255=ignore) | Bottom-right pad to size².

    :param imgs: [B, 3, H, W]. :param gts: [B, H, W] or None.
    """
    pad_h, pad_w = size - imgs.shape[-2], size - imgs.shape[-1]
    if pad_h <= 0 and pad_w <= 0:
        return imgs, gts
    imgs = F.pad(imgs, (0, pad_w, 0, pad_h), value=0.0)
    if gts is not None:
        gts = F.pad(gts, (0, pad_w, 0, pad_h), value=IGNORE_INDEX)
    return imgs, gts


def build_model(arch: str, encoder: str, encoder_weights: str | None,
                device: torch.device) -> nn.Module:
    """构建基线模型 | Build baseline model."""
    if arch == "unet":
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=64)
    elif arch == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            print("[ERROR] DeepLabV3+ 需要 segmentation_models_pytorch:\n"
                  "        pip install segmentation-models-pytorch")
            sys.exit(1)
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,   # "imagenet" or None
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unknown arch: {arch!r}")
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════
# 评估 (与 evaluate_segnext 同口径) | Evaluation (same protocol)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_baseline(model: nn.Module, dataset: NEUSegDataset,
                      device: torch.device, img_norm: str = "imagenet",
                      max_samples: int = 0) -> dict:
    """
    数据集级 per-class IoU 累积 (与 eval_segnext/eval_neuseg 同口径)。
    Dataset-level per-class IoU accumulation (same as eval_segnext/eval_neuseg).

    :param max_samples: 最多评估样本数, 0=全部 (仅冒烟测试用) | 0=all (smoke tests only).
    """
    model.eval()

    inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_correct, total_pixels = 0, 0
    per_sample_ious = []

    n_eval = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    for idx in tqdm(range(n_eval), desc="Eval", leave=False):
        try:
            sample = dataset[idx]
        except (ValueError, OSError, FileNotFoundError):
            continue

        img = sample["image"].unsqueeze(0).to(device)      # [1, 3, H, W]
        gt_raw = sample["masks"]
        if isinstance(gt_raw, torch.Tensor):
            gt_raw = gt_raw.numpy()
        gt = gt_raw.astype(np.int64)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)                              # [H, W]
        H, W = gt.shape

        img, _ = pad_batch(img)
        img = normalize_img(img, img_norm)
        logits = model(img)[:, :, :H, :W]                   # 裁回原尺寸 | crop back
        pred_cls = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        sample_ious = []
        for c in range(NUM_CLASSES):
            p, g = (pred_cls == c), (gt == c)
            i, u = (p & g).sum(), (p | g).sum()
            inter[c] += i
            union[c] += u
            if u > 0:
                sample_ious.append(i / u)
        if sample_ious:
            per_sample_ious.append(np.mean(sample_ious))

        total_correct += (pred_cls == gt).sum()
        total_pixels += gt.size

    ious = inter / np.maximum(union, 1)
    # Macro Dice: 逐类 2I/(I+U) ≡ 2TP/(2TP+FP+FN) (全仓库统一定义)
    # Macro Dice: per-class 2I/(I+U) — unified definition repo-wide
    dices = 2 * inter / np.maximum(inter + union, 1)

    return {
        "mIoU": round(float(ious.mean()), 6),
        "Dice": round(float(dices.mean()), 6),
        "pixel_accuracy": round(float(total_correct / max(total_pixels, 1)), 6),
        "per_class_IoU": {n: round(float(ious[i]), 4)
                          for i, n in enumerate(CLASS_NAMES)},
        "per_class_Dice": {n: round(float(dices[i]), 4)
                           for i, n in enumerate(CLASS_NAMES)},
        "sample_mIoU_mean": round(float(np.mean(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "sample_mIoU_median": round(float(np.median(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "n_evaluated": len(per_sample_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Classic CNN Baselines on NEU_Seg")

    # ── 模型 | Model ──
    p.add_argument("--arch", type=str, default="unet",
                   choices=["unet", "deeplabv3plus"],
                   help="基线架构 | Baseline architecture")
    p.add_argument("--encoder", type=str, default="resnet50",
                   help="smp encoder 名 (仅 deeplabv3plus) | smp encoder name")
    p.add_argument("--encoder-weights", type=str, default="imagenet",
                   choices=["imagenet", "none"],
                   help="encoder 预训练 (仅 deeplabv3plus) | Encoder pretraining")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")

    # ── 训练 (与 SegNeXt 基线同预算) | Training (same budget as SegNeXt) ──
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="官方增广管线 | Official augmentation pipeline")
    p.add_argument("--img-norm", type=str, default="imagenet",
                   choices=["imagenet", "unit"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "ce_dice"])

    # ── 优化器 | Optimizer ──
    p.add_argument("--lr", type=float, default=None,
                   help="默认按架构: unet=1e-3, deeplabv3plus=1e-4 | Per-arch default")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=1500)
    p.add_argument("--min-lr", type=float, default=0.0)

    # ── 评估模式 | Eval-only mode ──
    p.add_argument("--eval-only", action="store_true",
                   help="仅评估 --checkpoint, 不训练 | Evaluate a checkpoint only")
    p.add_argument("--checkpoint", type=str, default=None)

    # ── 硬件 / 输出 | Hardware / Output ──
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--max-eval-samples", type=int, default=0,
                   help="每次评估的最大样本数, 0=全部 (仅冒烟测试用) | 0=all (smoke only)")
    p.add_argument("--output-dir", type=str, default=None)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 仅评估模式 | Eval-only Mode
# ═══════════════════════════════════════════════════════════════════

def run_eval_only(args) -> None:
    """加载 checkpoint → 完整评估 → eval_results.json。"""
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    t_args = ckpt.get("args", {})
    arch = t_args.get("arch", args.arch)
    encoder = t_args.get("encoder", args.encoder)
    img_norm = t_args.get("img_norm", "imagenet")
    print(f"  Arch: {arch}" + (f" ({encoder})" if arch == "deeplabv3plus" else ""))
    print(f"  Epoch: {ckpt.get('epoch', '?')}, img_norm: {img_norm}")

    # 权重已在 ckpt 中, encoder 无需再下载 ImageNet 预训练
    # Weights are in the ckpt; no need to re-download ImageNet pretraining
    model = build_model(arch, encoder, None, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded model weights: {len(ckpt['model_state_dict'])} keys")

    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"  Val samples: {len(val_ds)}")

    metrics = evaluate_baseline(model, val_ds, device, img_norm=img_norm,
                                max_samples=args.max_eval_samples)

    print(f"\n  mIoU:           {metrics['mIoU']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Sample mIoU:    mean={metrics['sample_mIoU_mean']:.4f} "
          f"median={metrics['sample_mIoU_median']:.4f}")
    print(f"  Per-class IoU:")
    for name in CLASS_NAMES:
        print(f"      {name:>12s}: {metrics['per_class_IoU'][name]:.4f}")

    out_dir = ckpt_path.parent / "eval_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.update({"checkpoint": str(ckpt_path), "arch": arch,
                    "encoder": encoder if arch == "deeplabv3plus" else None,
                    "img_norm": img_norm})
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_dir / 'eval_results.json'}")


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.eval_only:
        if not args.checkpoint:
            print("[ERROR] --eval-only 需要 --checkpoint")
            sys.exit(1)
        run_eval_only(args)
        return

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.lr is None:
        args.lr = ARCH_DEFAULT_LR[args.arch]
    enc_w = None if (args.arch == "unet" or args.encoder_weights == "none") \
        else args.encoder_weights

    # ── 输出目录 | Output Directory ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = args.arch if args.arch == "unet" else f"{args.arch}_{args.encoder}"
        args.output_dir = f"runs/baseline_{tag}_NEUSeg_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_baseline_cnn")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Arch: {args.arch}"
                    + (f" (encoder={args.encoder}, weights={enc_w})"
                       if args.arch == "deeplabv3plus" else " (from scratch)"))
    logger.log_info("config", f"Training: {args.epochs}×{args.steps_per_epoch} steps, "
                    f"bs={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")
    logger.log_info("config", f"Recipe: loss={args.loss}, augment={args.augment}, "
                    f"img_norm={args.img_norm}, warmup={args.warmup_iters}")

    # ── 数据集 | Datasets ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 模型 | Model ──
    model = build_model(args.arch, args.encoder, enc_w, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.log_info("model", f"{args.arch}: {n_params/1e6:.2f}M params (all trainable)")

    # ── 优化器 + 调度 (warmup→poly, 与 SegNeXt 基线一致) | Optimizer + schedule ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    max_iters = args.epochs * args.steps_per_epoch
    min_factor = args.min_lr / max(args.lr, 1e-12)

    def poly_lambda(current_iter):
        if current_iter < args.warmup_iters:
            return current_iter / max(args.warmup_iters, 1)
        factor = (1 - (current_iter - args.warmup_iters) /
                  max(max_iters - args.warmup_iters, 1)) ** 1.0
        return max(factor, min_factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lambda)

    augment = OfficialAugment(crop_size=200) if args.augment else None

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"Starting: {args.epochs} epochs × "
                    f"{args.steps_per_epoch} steps = {max_iters} iters")

    best_miou, best_dice, best_epoch = 0.0, 0.0, 0
    global_step, nan_skip_count = 0, 0
    n_train = len(train_ds)
    indices_pool = list(range(n_train))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses, epoch_ces = [], []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}")
        for _ in pbar:
            # ── 批量采样 + 增广 | Batch sampling + augmentation ──
            batch_idxs = random.sample(indices_pool, min(args.batch_size, n_train))
            imgs_list, masks_list = [], []
            for idx in batch_idxs:
                try:
                    s = train_ds[idx]
                except (ValueError, OSError, FileNotFoundError):
                    continue
                img_t = s["image"]                          # [3, H, W] float [0,1]
                m = s["masks"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m.copy()).long()
                else:
                    m = m.long()
                if m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)                        # [H, W]

                if augment is not None:
                    img_np = img_t.permute(1, 2, 0).numpy() * 255.0
                    img_np, m_np = augment(img_np, m.numpy())
                    img_t = torch.from_numpy(
                        np.ascontiguousarray(img_np / 255.0)).permute(2, 0, 1).float()
                    m = torch.from_numpy(m_np).long()

                imgs_list.append(img_t)
                masks_list.append(m)

            if not imgs_list:
                continue

            imgs = torch.stack(imgs_list).to(device)        # [B, 3, 200, 200]
            gts = torch.stack(masks_list).to(device)        # [B, 200, 200]
            imgs, gts = pad_batch(imgs, gts)                # → 224² (mask pad=255)
            imgs = normalize_img(imgs, args.img_norm)

            # ── 前向 + 损失 | Forward + loss ──
            logits = model(imgs)                            # [B, C, 224, 224]
            pred = F.softmax(logits, dim=1)
            loss_dict = compute_loss(pred, gts, loss_type=args.loss)

            if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
                nan_skip_count += 1
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss_dict["loss"].item())
            epoch_ces.append(loss_dict["ce"])
            pbar.set_postfix(loss=f"{np.mean(epoch_losses[-10:]):.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if global_step % 20 == 0:
                logger.log_metric("loss", loss_dict["loss"].item(),
                                  step=global_step, tags=["baseline_train"])

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        logger.log_info("epoch",
            f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | NaN={nan_skip_count}")

        # ── 评估 + 按 mIoU 保存最佳 | Eval + save best by mIoU ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_baseline(model, val_ds, device,
                                        img_norm=args.img_norm,
                                        max_samples=args.max_eval_samples)
            miou = metrics["mIoU"]
            logger.log_info("eval",
                f"  mIoU={miou:.4f}  dice={metrics['Dice']:.4f}  "
                f"best_mIoU={max(best_miou, miou):.4f} (epoch {best_epoch})")
            for name in CLASS_NAMES:
                logger.log_info("eval",
                    f"    {name:>12s}: IoU={metrics['per_class_IoU'][name]:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["baseline_eval"])

            if miou > best_miou:
                best_miou, best_dice, best_epoch = miou, metrics["Dice"], epoch
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "arch": args.arch,
                    "model_state_dict": {k: v.clone() for k, v
                                         in model.state_dict().items()},
                    "metrics": metrics,
                    "args": vars(args),
                    "num_classes": NUM_CLASSES,
                }, str(out_dir / "best_model.pt"))
                logger.log_info("eval",
                    f"  ✓ New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── 最终保存 | Final save ──
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "arch": args.arch,
        "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "best_mIoU": best_miou,
        "best_epoch": best_epoch,
        "args": vars(args),
        "num_classes": NUM_CLASSES,
    }, str(out_dir / "last_model.pt"))

    results = {
        "experiment": f"NEU_Seg Baseline: {args.arch}",
        "arch": args.arch,
        "encoder": args.encoder if args.arch == "deeplabv3plus" else None,
        "encoder_weights": enc_w,
        "params": n_params,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "augment": args.augment,
        "img_norm": args.img_norm,
        "best_mIoU": round(best_miou, 6),
        "best_Dice": round(best_dice, 6),
        "best_epoch": best_epoch,
        "nan_skip_count": nan_skip_count,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  {args.arch} NEU_Seg Baseline -- Complete")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
