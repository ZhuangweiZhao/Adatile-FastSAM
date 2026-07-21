#!/usr/bin/env python3
"""
少样本训练 — Severstal 钢铁缺陷检测 | Few-Shot Training — Severstal Steel Defect.
==================================================================================

K-shot per-class 采样 → 训练 → 全量验证集评估。支持多 K 值 × 多 seed 扫参。
K-shot per-class sampling → train → evaluate on full validation set.
Supports multi-K × multi-seed sweep.

每类缺陷 (Class 1-4) 各采样 K 张图像，取并集作为训练子集。
可选加入无缺陷（干净）图像用于背景学习。

For each defect class (1-4), K images are sampled, unioned as the training subset.
Clean (defect-free) images are optionally included for background learning.

用法 | Usage::

    # K=1, 单种子快速验证 | Quick test: K=1, single seed
    python tools/train/train_severstal_fewshot.py --k-shot 1 --seeds 42 --epochs 5

    # 完整扫参: K=1/3/5/10/20 × 3 seeds | Full sweep
    python tools/train/train_severstal_fewshot.py \
        --k-shot 1 3 5 10 20 --seeds 42 123 456 \
        --epochs 100 --batch-size 8 --device cuda

    # 二值模式 + 指定干净样本数 | Binary mode + capped clean samples
    python tools/train/train_severstal_fewshot.py \
        --k-shot 5 --binary --clean-samples 50
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.rectify import DA_FRN
from adatile.rectify.hdn import HeatmapDenoiser
from adatile.datasets.severstal import SeverstalDataset, sample_k_shot_severstal


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

MULTI_NUM_CLASSES = 5
MULTI_CLASS_NAMES = ["background", "Class1", "Class2", "Class3", "Class4"]
BINARY_NUM_CLASSES = 2
BINARY_CLASS_NAMES = ["background", "foreground"]

IMG_H, IMG_W = 256, 1600  # Severstal native (multiples of 32)


# ═══════════════════════════════════════════════════════════════════
# 数据增强 | Data Augmentation (from train_dafrn_severstal.py)
# ═══════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """极简增强: flip + rotate + brightness + noise."""

    def __init__(self, p_flip=0.5, p_rotate=0.3, brightness=0.2, contrast=0.2, noise_std=0.02):
        self.p_hflip = p_flip
        self.p_vflip = p_flip * 0.5
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        if torch.rand(1).item() < 0.7:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            m_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - m_val) * c + m_val, 0.0, 1.0)
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions (from train_dafrn_severstal.py)
# ═══════════════════════════════════════════════════════════════════

def multiclass_dice_loss(pred, target, smooth=1e-6, ignore_bg=True):
    """多类别 Dice 损失 | Multi-class Dice loss."""
    C = pred.shape[1]
    dice_sum = 0.0
    count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


def binary_dice_loss(pred, target, smooth=1e-6):
    """二值 Dice 损失 | Binary Dice loss."""
    pred_fg = pred[:, 1]
    target_fg = (target > 0).float()
    if target_fg.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    inter = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation (from train_dafrn_severstal.py)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(decoder, frn, backbone, dataset, device, num_classes, class_names,
             max_samples=0):
    """分割评估: per-class IoU + mIoU | Segmentation evaluation."""
    decoder.eval()
    backbone.eval()
    if frn is not None:
        frn.eval()

    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)

    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).to(device).long()
        H, W = gt.shape

        feats = backbone(img, extract_proto=False)
        p3, p4 = feats["p3"], feats["p4"]

        if frn is not None:
            p3, p4 = frn(p3, p4)

        pred_prob = decoder(p3, p4)
        pred_full = F.interpolate(pred_prob.unsqueeze(0), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)

        for c in range(num_classes):
            pc = (pred_class == c)
            gc = (gt == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

    per_class_iou = {}
    for c in range(num_classes):
        inter = per_class_inter[c].item()
        union = per_class_union[c].item()
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class_iou[name] = round(inter / union, 6) if union > 0 else float("nan")

    valid = [v for v in per_class_iou.values() if not (v != v)]
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "per_class_IoU": per_class_iou,
    }


# ═══════════════════════════════════════════════════════════════════
# HDN 辅助 | HDN Helpers (from train_dafrn_severstal.py)
# ═══════════════════════════════════════════════════════════════════

def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def _sobel_gradient(gray: torch.Tensor) -> torch.Tensor:
    device = gray.device
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_y)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    B = mag.shape[0]
    for b in range(B):
        m = mag[b].flatten()
        p_low = torch.quantile(m, 0.02)
        p_high = torch.quantile(m, 0.98)
        if p_high > p_low:
            mag[b] = torch.clamp(mag[b], p_low, p_high)
            mag[b] = (mag[b] - p_low) / (p_high - p_low + 1e-8)
    return mag


# ═══════════════════════════════════════════════════════════════════
# 命令行参数 | CLI Arguments
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Few-Shot Training — Severstal Steel Defect Detection"
    )

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/severstal-steel-defect-detection")
    p.add_argument("--binary", action="store_true",
                   help="二值模式 (FG/BG)。默认多类别 (5 类)。")
    p.add_argument("--no-augment", action="store_true")

    # ── 少样本 | Few-Shot ──
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                   help="每类采样 K 张图，可多个值 (default: 1 3 5 10 20)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="随机种子，可多个值 (default: 42 123 456)")
    p.add_argument("--clean-samples", type=int, default=None,
                   help="最多包含多少张无缺陷图 (None=全部, 0=不含)")
    p.add_argument("--epochs", type=int, default=100,
                   help="每个 (K, seed) 的训练轮数 (default: 100)")

    # ── 模型 | Model ──
    p.add_argument("--backbone", type=str, default="fastsam-x",
                   choices=["fastsam-x", "fastsam-s"])
    p.add_argument("--dcr", dest="enable_dcr", action="store_true", default=True,
                   help="启用 DCR (Defect-aware Channel Reweighting)")
    p.add_argument("--no-dcr", dest="enable_dcr", action="store_false")
    p.add_argument("--fde", dest="enable_fde", action="store_true", default=True,
                   help="启用 FDE (Frequency-aware Defect Enhancement)")
    p.add_argument("--no-fde", dest="enable_fde", action="store_false")
    p.add_argument("--cdf", dest="enable_cdf", action="store_true", default=True,
                   help="启用 CDF (Cross-scale Defect Fusion)")
    p.add_argument("--no-cdf", dest="enable_cdf", action="store_false")
    p.add_argument("--dcr-reduction", type=int, default=16)
    p.add_argument("--fde-alpha", type=float, default=0.5)
    p.add_argument("--cdf-hidden", type=int, default=64)
    p.add_argument("--heatmap-denoise", action="store_true",
                   help="启用 HDN 热力图去噪")
    p.add_argument("--hdn-weight", type=float, default=0.1)

    # ── 训练 | Training ──
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (default: 1). B>1 时逐样本梯度累积。")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-frn", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weights", type=str, default="balanced",
                   choices=["none", "balanced"],
                   help="类别权重 (推荐 balanced 应对极端 FG/BG 不平衡)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=10,
                   help="每 N 轮评估一次 (default: 10)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 单次 (K, seed) 训练 | Single (K, seed) Training Run
# ═══════════════════════════════════════════════════════════════════

def train_one_run(
    train_indices: list[int],
    train_ds: SeverstalDataset,
    val_ds: SeverstalDataset,
    backbone: FastSAMBackbone,
    device: torch.device,
    args: argparse.Namespace,
    logger,
    run_dir: Path,
    k: int,
    seed: int,
    class_names: list[str],
    num_classes: int,
    augment,
) -> dict:
    """
    在 K-shot 子集上训练一次 | Train on K-shot subset for one (K, seed).

    :return: {"k": int, "seed": int, "n_defect": int, "n_clean": int,
              "best_mIoU": float, "best_epoch": int, "per_class_IoU": dict}
    """
    set_seed(seed)

    # ── 构建训练子集 DataLoader | Build subset DataLoader ──
    subset = Subset(train_ds, train_indices)
    train_loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 统计子集构成 | Log subset composition ──
    n_defect = sum(1 for idx in train_indices
                   if train_ds._samples[idx][1])  # has_defect
    n_clean = len(train_indices) - n_defect
    logger.log_info("fewshot/run",
                    f"K={k} seed={seed}: {n_defect} defect + {n_clean} clean = "
                    f"{len(train_indices)} train samples, "
                    f"{len(train_loader)} batches/epoch")

    # ── 类别权重 | Class weights ──
    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pc = [stats.get(cn, {}).get("pixels", 1) for cn in class_names]
        total = sum(pc)
        if total > 0:
            raw = [total / max(p, 0.01) for p in pc]
            mean_w = sum(raw) / len(raw)
            ce_weight = torch.tensor([w / mean_w for w in raw],
                                     dtype=torch.float32, device=device)

    # ── 模型 (重新初始化 FRN + Decoder) | Model (re-init FRN + Decoder) ──
    ch = backbone.channels
    frn = DA_FRN(
        p3_channels=ch["p3"], p4_channels=ch["p4"],
        dcr_reduction=args.dcr_reduction,
        fde_kernel_sizes=(3, 7, 15), fde_alpha_init=args.fde_alpha,
        cdf_hidden=args.cdf_hidden,
        enable_dcr=args.enable_dcr, enable_fde=args.enable_fde,
        enable_cdf=args.enable_cdf,
    ).to(device)
    decoder = PureDecoderP3P4(
        p3_channels=ch["p3"], p4_channels=ch["p4"],
        out_channels=num_classes,
    ).to(device)
    hdn = HeatmapDenoiser(in_channels=1).to(device) if args.heatmap_denoise else None

    # ── Optimizer ──
    frn_lr = args.lr_frn if args.lr_frn is not None else args.lr
    optim_params = [
        {"params": decoder.parameters(), "lr": args.lr},
        {"params": frn.parameters(), "lr": frn_lr},
    ]
    if hdn is not None:
        optim_params.append({"params": hdn.parameters(), "lr": frn_lr})
    optimizer = torch.optim.AdamW(optim_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))

    # ── 训练循环 | Training Loop ──
    global_step = 0
    best_miou = 0.0
    best_epoch = 0
    best_per_class = {}
    nan_count = 0

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        frn.train()
        epoch_losses, epoch_ce, epoch_dice = [], [], []

        pbar = tqdm(train_loader, desc=f"K{k}_S{seed} E{epoch:3d}/{args.epochs}",
                    unit="batch", leave=False)
        for batch in pbar:
            img_batch = batch["image"]
            mask_batch = batch["masks"].squeeze(1)
            B = img_batch.shape[0]

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_ce_sum = 0.0
            batch_dice_sum = 0.0
            valid_samples = 0

            for b_idx in range(B):
                img = img_batch[b_idx]
                mask = mask_batch[b_idx]

                if augment:
                    img, mask = augment(img, mask)

                H, W = mask.shape
                img_dev = img.unsqueeze(0).to(device)
                mask_dev = mask.unsqueeze(0).to(device)

                # Forward
                feats = backbone(img_dev, extract_proto=False)
                p3, p4 = feats["p3"], feats["p4"]
                p3_r, p4_r = frn(p3, p4)
                pred_prob = decoder(p3_r, p4_r)

                pred_full = F.interpolate(pred_prob.unsqueeze(0), size=(H, W),
                                          mode="bilinear", align_corners=False).squeeze(0)
                target = mask_dev.squeeze(0).long()

                # HDN
                hdn_loss = torch.tensor(0.0, device=device)
                if hdn is not None:
                    gray = _rgb_to_gray(img_dev)
                    raw_grad = _sobel_gradient(gray)
                    denoised = hdn(raw_grad)
                    gt_binary = (target > 0).float().unsqueeze(0).unsqueeze(0)
                    denoised_resized = F.interpolate(
                        denoised, size=(H, W), mode="bilinear", align_corners=False)
                    hdn_loss = F.binary_cross_entropy(
                        denoised_resized.clamp(1e-7, 1 - 1e-7), gt_binary, reduction='mean')

                # Loss
                log_pred = torch.log(pred_full.unsqueeze(0).clamp(1e-7, 1))
                ce = F.nll_loss(log_pred, target.unsqueeze(0), weight=ce_weight,
                               reduction='mean')

                if args.binary:
                    dice = binary_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))
                else:
                    dice = multiclass_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))

                loss_val = (0.5 * ce + 0.5 * dice + args.hdn_weight * hdn_loss) / B

                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    nan_count += 1
                    continue

                loss_val.backward()
                batch_loss += loss_val.item() * B
                batch_ce_sum += ce.item()
                batch_dice_sum += dice.item()
                valid_samples += 1

            if valid_samples == 0:
                continue

            # Gradient check + clip + step
            all_params = list(decoder.parameters()) + list(frn.parameters())
            if hdn is not None:
                all_params += list(hdn.parameters())
            grad_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in all_params
            )
            if grad_nan:
                optimizer.zero_grad()
                nan_count += valid_samples
                continue

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(batch_loss / valid_samples)
            epoch_ce.append(batch_ce_sum / valid_samples)
            epoch_dice.append(batch_dice_sum / valid_samples)

            if epoch_losses:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses[-20:]):.4f}",
                    "dice": f"{np.mean(epoch_dice[-20:]):.4f}",
                })

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # ── Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            result = evaluate(decoder, frn, backbone, val_ds, device,
                             num_classes, class_names)
            miou = result["mIoU"]
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                best_per_class = result["per_class_IoU"]
                ckpt = {
                    "epoch": epoch, "global_step": global_step,
                    "frn_state_dict": {k: v.clone() for k, v in frn.state_dict().items()},
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "mIoU": miou, "per_class_IoU": best_per_class,
                    "args": vars(args), "k": k, "seed": seed,
                    "num_classes": num_classes, "class_names": class_names,
                }
                if hdn is not None:
                    ckpt["hdn_state_dict"] = {
                        k: v.clone() for k, v in hdn.state_dict().items()}
                torch.save(ckpt, str(run_dir / "best_model.pt"))

    logger.log_info("fewshot/run_done",
                    f"K={k} S={seed}: best mIoU={best_miou:.4f} @ epoch {best_epoch}, "
                    f"NaN={nan_count}")

    return {
        "k": k, "seed": seed,
        "n_defect": n_defect, "n_clean": n_clean,
        "n_total": len(train_indices),
        "best_mIoU": best_miou, "best_epoch": best_epoch,
        "per_class_IoU": best_per_class,
        "nan_count": nan_count,
    }


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(42)  # global seed for output dir, etc.
    device = torch.device(args.device)

    # ── 模式 | Mode ──
    if args.binary:
        NUM_CLASSES = BINARY_NUM_CLASSES
        CLASS_NAMES = BINARY_CLASS_NAMES
        mode_str = "binary"
    else:
        NUM_CLASSES = MULTI_NUM_CLASSES
        CLASS_NAMES = MULTI_CLASS_NAMES
        mode_str = "multi"

    # ── 模块标签 | Module tag ──
    modules = []
    if args.enable_dcr: modules.append("DCR")
    if args.enable_fde: modules.append("FDE")
    if args.enable_cdf: modules.append("CDF")
    if args.heatmap_denoise: modules.append("HDN")
    module_tag = "+".join(modules) if modules else "Baseline"

    # ── 输出目录 | Output dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/severstal_fewshot_{module_tag}_{mode_str}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_severstal_fewshot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Severstal Few-Shot | K={args.k_shot} | seeds={args.seeds}")
    logger.log_info("config", f"DA-FRN: {module_tag} | mode={mode_str}")
    logger.log_info("config", f"Epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")
    logger.log_info("config", f"Class weights={args.class_weights} | clean_samples={args.clean_samples}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── 数据 | Data ──
    train_ds = SeverstalDataset(root=args.data_root, split="train",
                                binary=args.binary, seed=42)
    val_ds = SeverstalDataset(root=args.data_root, split="val",
                              binary=args.binary, seed=42)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}, Mode: {mode_str}")
    logger.log_info("data", f"Per-class images: "
                    f"C1={len(train_ds.class_to_images(1))}, "
                    f"C2={len(train_ds.class_to_images(2))}, "
                    f"C3={len(train_ds.class_to_images(3))}, "
                    f"C4={len(train_ds.class_to_images(4))}")

    # ── Backbone (frozen, shared across all runs) ──
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=False)
    ch = backbone.channels
    logger.log_info("model", f"Backbone: {args.backbone}, P3={ch['p3']}ch, P4={ch['p4']}ch")

    # ── Augmentation ──
    augment = None if args.no_augment else BasicAugmentation()

    # ── 计算验证 K 值合法性 | Validate K values ──
    for k in args.k_shot:
        for cls_id in [1, 2, 3, 4]:
            n_avail = len(train_ds.class_to_images(cls_id))
            if k > n_avail:
                logger.log_info("warning",
                                f"K={k} > Class {cls_id} available images ({n_avail}). "
                                f"Will sample all {n_avail}.")

    # ── 扫参: K × Seed | Sweep: K × Seed ──
    runs = [(k, s) for k in args.k_shot for s in args.seeds]
    all_results = []
    total_runs = len(runs)

    logger.log_info("sweep", f"{'='*60}")
    logger.log_info("sweep", f"Starting sweep: {total_runs} runs ({len(args.k_shot)} K × {len(args.seeds)} seeds)")
    logger.log_info("sweep", f"{'='*60}")

    for run_idx, (k, seed) in enumerate(runs):
        logger.log_info("sweep", f"[{run_idx+1}/{total_runs}] K={k}, seed={seed}")

        # 采样 | Sample
        try:
            fewshot_indices = sample_k_shot_severstal(
                train_ds, k=k, seed=seed,
                include_clean=(args.clean_samples is None or args.clean_samples > 0),
                max_clean=args.clean_samples,
            )
        except ValueError as e:
            logger.log_info("sweep", f"  SKIP: {e}")
            continue

        # 子目录 | Run subdirectory
        run_dir = out_dir / f"K{k}_S{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 训练 | Train
        result = train_one_run(
            train_indices=fewshot_indices,
            train_ds=train_ds, val_ds=val_ds,
            backbone=backbone, device=device,
            args=args, logger=logger, run_dir=run_dir,
            k=k, seed=seed,
            class_names=CLASS_NAMES, num_classes=NUM_CLASSES,
            augment=augment,
        )
        all_results.append(result)
        logger.log_metric(f"miou_K{k}_S{seed}", result["best_mIoU"],
                          step=0, tags=["fewshot"])

    # ── 汇总 | Summary ──
    if not all_results:
        logger.log_info("done", "No results to report.")
        return

    # 按 K 聚合 | Aggregate by K
    summary_by_k = {}
    for k in args.k_shot:
        k_results = [r for r in all_results if r["k"] == k]
        if not k_results:
            continue
        mious = [r["best_mIoU"] for r in k_results]
        summary_by_k[str(k)] = {
            "mean_mIoU": round(float(np.mean(mious)), 4),
            "std_mIoU": round(float(np.std(mious)), 4) if len(mious) > 1 else 0.0,
            "min_mIoU": round(float(np.min(mious)), 4),
            "max_mIoU": round(float(np.max(mious)), 4),
            "n_runs": len(k_results),
        }

    # ── 打印汇总表 | Print Summary Table ──
    header = (f"{'K':>3}  {'Seed':>4}  {'#Defect':>7}  {'#Clean':>6}  "
              f"{'mIoU':>8}  {'BG':>8}  {'C1':>8}  {'C2':>8}  {'C3':>8}  {'C4':>8}  {'Epoch':>6}")
    sep = "-" * len(header.expandtabs())

    print(f"\n{'='*len(header.expandtabs())}")
    print(f"  Severstal Few-Shot Results — DA-FRN ({module_tag})")
    print(f"{'='*len(header.expandtabs())}")
    print(header)
    print(sep)

    for r in all_results:
        p = r["per_class_IoU"]
        print(f"{r['k']:3d}  {r['seed']:4d}  {r['n_defect']:7d}  {r['n_clean']:6d}  "
              f"{r['best_mIoU']:8.4f}  "
              f"{p.get('background', float('nan')):8.4f}  "
              f"{p.get('Class1', p.get('foreground', float('nan'))):8.4f}  "
              f"{p.get('Class2', float('nan')):8.4f}  "
              f"{p.get('Class3', float('nan')):8.4f}  "
              f"{p.get('Class4', float('nan')):8.4f}  "
              f"{r['best_epoch']:6d}")

    # 均值行 | Mean rows
    print(sep)
    for k_str, s in summary_by_k.items():
        print(f"{k_str:>3}  {'mean':>4}  {'-':>7}  {'-':>6}  "
              f"{s['mean_mIoU']:8.4f}  {'±'+str(s['std_mIoU']):>13}")
    print(f"{'='*len(header.expandtabs())}\n")

    # ── 保存 results.json | Save results.json ──
    results_json = {
        "experiment": "Severstal Few-Shot Training",
        "timestamp": datetime.now().isoformat(),
        "module_tag": module_tag,
        "mode": mode_str,
        "config": vars(args),
        "runs": all_results,
        "summary_by_k": summary_by_k,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
    logger.log_info("done", f"Results saved to {json_path}")
    logger.log_info("done", f"Best per K: {json.dumps(summary_by_k, indent=2)}")

    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
