#!/usr/bin/env python3
"""
DefectPrompt 训练脚本 | DefectPrompt Training Script.
======================================================

极简训练脚本 — 只测一个创新: DefectPrompt Generator.
Minimal training script — tests ONE innovation: DefectPrompt Generator.

消融设计 | Ablation Design:
    --no-prompt          → PureDecoderP3P4 (baseline, 同 NoLoRA | same as NoLoRA)
    (default)            → PromptDecoderP3P4 (DefectPrompt, K=5)
    --num-prompts K      → 调整 prompt 数量 | tune prompt count
    --lora-rank 2        → DefectPrompt + LoRA

用法 | Usage::

    # 纯 Prompt 基线 | Pure Prompt baseline
    python tools/train/train_defect_prompt.py --epochs 200

    # 消融: 无 Prompt | Ablation: no prompt (= NoLoRA baseline)
    python tools/train/train_defect_prompt.py --no-prompt --epochs 200

    # Prompt + LoRA | Prompt + LoRA
    python tools/train/train_defect_prompt.py --lora-rank 2 --epochs 200
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
from adatile.decoder.prompt_decoder import PromptDecoderP3P4
from adatile.prompt.defect_prompt import prompt_diversity_loss
from adatile.datasets.neu_seg import NEUSegDataset

# ═══════════════════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]


# ═══════════════════════════════════════════════════════════════════════════════
# 基础增强 + 损失 + 评估 | Basic Aug + Loss + Eval
# ═══════════════════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """极简增强: 翻转 + 旋转 + 亮度 + 噪声. No CLAHE/Gamma/Blur."""

    def __init__(self, p_flip: float = 0.5, p_rotate: float = 0.5,
                 brightness: float = 0.2, contrast: float = 0.2, noise_std: float = 0.02):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
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


def multiclass_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                         smooth: float = 1e-6, ignore_bg: bool = True) -> torch.Tensor:
    C = pred.shape[1]
    dice_sum = 0.0; count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c, :, :]; target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth); count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  ce_weight: torch.Tensor | None = None,
                  ce_alpha: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight, reduction='mean')
    dice = multiclass_dice_loss(pred, target)
    return ce_alpha * ce + (1 - ce_alpha) * dice, {"ce": ce.item(), "dice": dice.item()}


@torch.no_grad()
def evaluate(decoder: nn.Module, backbone: FastSAMBackbone,
             dataset: NEUSegDataset, device: torch.device,
             use_prompt: bool = True, max_samples: int = 0) -> dict:
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

        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        feats = backbone(img, extract_proto=True)

        if use_prompt:
            pred_prob, _heatmaps = decoder(feats["p2"], feats["p3"], feats["p4"])
        else:
            pred_prob = decoder(feats["p3"], feats["p4"])

        pred_full = F.interpolate(
            pred_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)

        for c in range(NUM_CLASSES):
            pred_c = (pred_class == c); gt_c = (gt == c)
            per_class_inter[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()

    per_class_iou = {}
    for c in range(NUM_CLASSES):
        inter = per_class_inter[c].item(); union = per_class_union[c].item()
        per_class_iou[CLASS_NAMES[c]] = round(inter / union, 6) if union > 0 else float("nan")
    valid = [v for v in per_class_iou.values() if not (v != v)]
    miou = float(np.mean(valid)) if valid else 0.0
    return {"mIoU": round(miou, 6), "per_class_IoU": per_class_iou}


# ═══════════════════════════════════════════════════════════════════════════════
# 参数解析 | Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DefectPrompt Training")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Prompt
    p.add_argument("--no-prompt", action="store_true",
                   help="禁用 Prompt → PureDecoderP3P4 基线 | Disable prompt → PureDecoderP3P4 baseline")
    p.add_argument("--num-prompts", type=int, default=5,
                   help="Prompt 数量 K | Number of defect prompts")
    p.add_argument("--diversity-weight", type=float, default=0.1,
                   help="Prompt 多样性损失权重 (0=禁用) | Prompt diversity loss weight (0=disabled)")
    # LoRA (可选叠加) | LoRA (optional)
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank (0=禁用)")
    p.add_argument("--lora-alpha", type=float, default=1.0)
    # Augmentation
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--class-weights", type=str, default="none", choices=["none", "balanced"])
    # Hardware
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", type=str, default="fastsam-x", choices=["fastsam-x", "fastsam-s"])
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
    use_prompt = not args.no_prompt

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = "NoPrompt" if not use_prompt else f"DPG_K{args.num_prompts}"
        if args.lora_rank > 0:
            tag += f"_LoRA{args.lora_rank}"
        if args.no_augment:
            tag += "_noAug"
        args.output_dir = f"runs/neuseg_{tag}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_defect_prompt")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"DefectPrompt Training: prompt={use_prompt}, K={args.num_prompts}, LoRA rank={args.lora_rank}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── Data ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pixel_counts = [stats[cn]["pixels"] for cn in CLASS_NAMES]
        total = sum(pixel_counts)
        raw = [total / max(p, 1) for p in pixel_counts]
        mean_w = sum(raw) / len(raw)
        ce_weight = torch.tensor([w / mean_w for w in raw], dtype=torch.float32, device=device)
        logger.log_info("data", f"Class weights (balanced): {dict(zip(CLASS_NAMES, [round(w,2) for w in ce_weight.tolist()]))}")

    augment = None if args.no_augment else BasicAugmentation()
    logger.log_info("data", f"Augmentation: {'none' if args.no_augment else 'basic'}")

    # ── Models ──
    logger.log_info("model", f"Building FastSAMBackbone ({args.backbone})...")
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()

    # LoRA (optional)
    lora_params = 0
    if args.lora_rank > 0:
        lora_params = backbone.apply_conv_lora(rank=args.lora_rank, alpha=args.lora_alpha)
        logger.log_info("model", f"ConvLoRA: rank={args.lora_rank}, +{lora_params:,} params")

    # Channel probing
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels
    logger.log_info("model", f"Channels: {ch}")

    # Decoder
    if use_prompt:
        decoder = PromptDecoderP3P4(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=NUM_CLASSES, prompt_dim=256, num_prompts=args.num_prompts,
        ).to(device)
    else:
        decoder = PureDecoderP3P4(
            p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES,
        ).to(device)

    dec_params = sum(p.numel() for p in decoder.parameters())
    prompt_params = sum(p.numel() for p in decoder.prompt_gen.parameters()) if use_prompt else 0
    total_trainable = dec_params + lora_params
    logger.log_info("model",
        f"Decoder: {dec_params/1e3:.1f}K (prompt: {prompt_params/1e3:.1f}K) | "
        f"Total trainable: {total_trainable/1e3:.1f}K ({total_trainable:,})")

    # ── Optimizer ──
    optim_params = list(decoder.parameters())
    if args.lora_rank > 0:
        optim_params += backbone.get_lora_parameters()
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch,
    )

    # ── Training ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Training: {args.epochs} epochs x {args.steps_per_epoch} steps")
    logger.log_info("train", f"Prompt: {use_prompt} (K={args.num_prompts}), LoRA: {args.lora_rank}")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0; best_epoch = 0; global_step = 0; nan_count = 0
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []; epoch_ce_vals = []; epoch_dice_vals = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            q_idx = rng.randint(0, len(train_ds) - 1)
            try:
                sample = train_ds[q_idx]
            except (ValueError, OSError, FileNotFoundError):
                continue

            img = sample["image"]; mask = sample["masks"]
            H, W = mask.shape[1:]
            if (mask > 0).sum() < 1.0:
                continue

            if augment:
                img, mask = augment(img, mask)

            img_dev = img.unsqueeze(0).to(device); mask_dev = mask.to(device)
            pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img_dev = F.pad(img_dev, (0, pad_w, 0, pad_h), mode='constant', value=0)

            decoder.train()
            feats = backbone(img_dev, extract_proto=True)

            heatmaps = None
            if use_prompt:
                pred_prob, heatmaps = decoder(feats["p2"], feats["p3"], feats["p4"])
            else:
                pred_prob = decoder(feats["p3"], feats["p4"])

            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0)
            target = mask_dev.squeeze(0).long()

            loss_val, loss_dict = combined_loss(
                pred_full.unsqueeze(0), target.unsqueeze(0), ce_weight=ce_weight,
            )

            # 多样性损失 (促使 K 个 prompt 关注不同区域)
            # Diversity loss (encourages K prompts to attend to different regions)
            if use_prompt and args.diversity_weight > 0 and heatmaps is not None:
                div_loss = prompt_diversity_loss(heatmaps.unsqueeze(0) if heatmaps.dim() == 3 else heatmaps)
                loss_val = loss_val + args.diversity_weight * div_loss
                loss_dict["diversity"] = div_loss.item()

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1; continue

            optimizer.zero_grad()
            loss_val.backward()

            grad_nan = False
            for name, param in list(decoder.named_parameters()):
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    grad_nan = True; break
            if args.lora_rank > 0 and not grad_nan:
                for param in backbone.get_lora_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        grad_nan = True; break
            if grad_nan:
                optimizer.zero_grad(); nan_count += 1; continue

            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step(); scheduler.step(); global_step += 1

            epoch_losses.append(loss_val.item())
            epoch_ce_vals.append(loss_dict["ce"]); epoch_dice_vals.append(loss_dict["dice"])

            if epoch_losses:
                pbar.set_postfix({"loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                                  "ce": f"{np.mean(epoch_ce_vals[-50:]):.4f}",
                                  "dice": f"{np.mean(epoch_dice_vals[-50:]):.4f}"})

        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}"
        if epoch_ce_vals: log_msg += f" ce={np.mean(epoch_ce_vals):.4f}"
        if epoch_dice_vals: log_msg += f" dice={np.mean(epoch_dice_vals):.4f}"
        log_msg += f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}"
        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["defect_prompt"])

        # Eval
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")
            result = evaluate(decoder, backbone, val_ds, device, use_prompt=use_prompt)
            miou = result["mIoU"]
            logger.log_info("eval", f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["defect_prompt"])

            if miou > best_miou:
                best_miou = miou; best_epoch = epoch
                ckpt = {
                    "epoch": epoch, "global_step": global_step,
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mIoU": miou, "args": vars(args), "num_classes": NUM_CLASSES,
                }
                if args.lora_rank > 0:
                    ckpt["lora_state_dict"] = {k: v.clone() for k, v in backbone.model.model.state_dict().items()
                                               if any(x in k for x in ["lora_down", "lora_up"])}
                torch.save(ckpt, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # Final save
    final = {
        "epoch": args.epochs, "global_step": global_step,
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou, "best_epoch": best_epoch,
        "args": vars(args), "num_classes": NUM_CLASSES, "nan_skip_count": nan_count,
    }
    if args.lora_rank > 0:
        final["lora_state_dict"] = {k: v.clone() for k, v in backbone.model.model.state_dict().items()
                                    if any(x in k for x in ["lora_down", "lora_up"])}
    torch.save(final, str(out_dir / "last_model.pt"))

    print(f"\n{'='*60}")
    print(f"  DefectPrompt Training — Complete")
    print(f"  Prompt: {use_prompt} (K={args.num_prompts}), LoRA: {args.lora_rank}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
