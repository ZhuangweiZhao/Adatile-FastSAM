#!/usr/bin/env python3
"""
B-02 v2: Token-Level Learnability — FDR on FastSAM P4 (对标 eval_fdr_token_fss.py)
=====================================================================================
Token-Level Learnability Study — Can FDR learn per-token FG density from P4 features?

B-02 原版: MobileNetV3 + tile-level importance prediction (Spearman r=0.889).
B-02 v2:    FastSAM P4 + token-level FDR (ForegroundDensityRouter, 165K params).
            Adapted to match eval_fdr_token_fss.py architecture.

实验设计 | Design:
    Backbone: Frozen FastSAM → P4 features [B, 1280, H/16, W/16]
    Model:    FDR (DensityHead: Conv1×1→DWConv3×3→Conv1×1→Sigmoid)
    GT:       Per-token fg_ratio from label mask (stride=16)
    Loss:     MSE(pred_density, gt_fg_ratio)
    Metrics:  Spearman r, Pearson r, Oracle vs Learned FG Retention, IDG

对应 eval_fdr_token_fss.py: TokenFGRatioDataset + ForegroundDensityRouter + MSE training.
对标 B-02 原版: 同样的评估协议 (Pearson/Spearman/Retention/IDG), 但 token 级.

用法 | Usage::
    python tools/paper_b/eval_b02_learnability_v2.py \
        --tile-root data/iSAID5i_tiles/tile_896 \
        --epochs 50 --batch-size 16 --device cuda
"""

import sys, argparse, json, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.sparse.spatial_router import ForegroundDensityRouter

logger = get_logger("b02_v2")

STRIDE = 16  # FastSAM P4 stride


def parse_args():
    p = argparse.ArgumentParser(
        description="B-02 v2: Token-Level Learnability — FDR on FastSAM P4")
    p.add_argument("--tile-root", type=str, default="data/iSAID5i_tiles/tile_896")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/b02_learnability_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: Token-Level FG Ratio (same as train_spm.py / eval_fdr_token_fss.py)
# ═══════════════════════════════════════════════════════════════════════════

class TokenFGRatioDataset(Dataset):
    """Tile → per-token fg_ratio at P4 stride."""

    def __init__(self, tile_root: str, split: str = "train", stride: int = STRIDE):
        self.tile_root = Path(tile_root)
        self.split = split
        self.stride = stride
        img_dir = self.tile_root / split / "images"
        label_dir = self.tile_root / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {label_dir}")
        self.img_paths = sorted(img_dir.glob("*.png"))
        self.label_dir = label_dir
        # 过滤 FG>0 的 tile | Filter to tiles with FG
        self._valid_indices = []
        for i, img_path in enumerate(self.img_paths):
            lp = self.label_dir / f"{img_path.stem}_label.png"
            if lp.exists():
                label = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
                if label is not None and (label > 0).any():
                    self._valid_indices.append(i)
        logger.log_info("data",
                        f"TokenFGRatioDataset [{split}]: "
                        f"{len(self._valid_indices)}/{len(self.img_paths)} tiles with FG")

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        real_idx = self._valid_indices[idx]
        img_path = self.img_paths[real_idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

        label_path = self.label_dir / f"{img_path.stem}_label.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        H, W = label.shape[:2]
        fg = (label > 0).astype(np.float32)
        h_t, w_t = H // self.stride, W // self.stride
        fg = fg[:h_t * self.stride, :w_t * self.stride]
        fg_ratio = (fg.reshape(h_t, self.stride, w_t, self.stride)
                      .transpose(0, 2, 1, 3)
                      .reshape(h_t, w_t, -1)
                      .mean(axis=2))
        fg_t = torch.from_numpy(fg_ratio).unsqueeze(0).float()
        return img_t, fg_t


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation (对标 eval_fdr_token_fss.py train_fdr)
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(fdr, backbone, loader, opt, device, scaler=None):
    """训练一个 epoch | Train one epoch."""
    fdr.train()
    backbone.eval()
    total_loss, n = 0.0, 0
    criterion = nn.MSELoss()

    pbar = tqdm(loader, desc="B-02 [train]", unit="batch")
    for imgs, fg_gt in pbar:
        imgs, fg_gt = imgs.to(device), fg_gt.to(device)
        with torch.no_grad():
            p4 = backbone(imgs)["p4"]
        imp = fdr(p4)["importance"]
        if imp.shape[2:] != fg_gt.shape[2:]:
            fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                  mode="bilinear", align_corners=False)
        loss = criterion(imp, fg_gt)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(fdr, backbone, loader, device):
    """
    对标 B-02 原版 evaluate_model: 收集所有 token 的 pred/gt/fg → 计算指标.
    Collects all token-level pred/gt → Pearson/Spearman + Oracle vs Learned FG retention.
    """
    fdr.eval()
    backbone.eval()

    all_pred, all_gt, all_fg_px = [], [], []

    for imgs, fg_gts in tqdm(loader, desc="B-02 [eval]", unit="batch"):
        imgs, fg_gts = imgs.to(device), fg_gts.to(device)
        p4 = backbone(imgs)["p4"]
        imp = fdr(p4)["importance"]

        # Per-sample collection | 逐样本收集
        for b in range(imgs.shape[0]):
            gt = fg_gts[b, 0]  # [h_t, w_t]
            pred = imp[b, 0]    # [h_t, w_t]

            if pred.shape != gt.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0),
                    size=gt.shape, mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)

            gt_flat = gt.flatten()
            pred_flat = pred.flatten()
            # FG像素: gt val本身 = fg_ratio, 乘以token面积 = FG像素数估计值
            # FG pixels: gt value = fg_ratio, token has stride*stride pixels
            fg_px = gt_flat * (STRIDE * STRIDE)

            all_pred.append(pred_flat.cpu())
            all_gt.append(gt_flat.cpu())
            all_fg_px.append(fg_px.cpu())

    # 拼接所有 token | Concatenate all tokens
    pred_all = torch.cat(all_pred).numpy()
    gt_all = torch.cat(all_gt).numpy()
    fg_all = torch.cat(all_fg_px).numpy()

    # Spearman / Pearson (对标 B-02) | Spearman / Pearson (match B-02)
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pred_all, gt_all)
    sr, _ = spearmanr(pred_all, gt_all)

    # Oracle vs Learned FG retention (按重要性排序后累积FG)
    # Oracle: sort by GT fg_ratio descending
    # Learned: sort by predicted importance descending
    oracle_ord = np.argsort(gt_all)[::-1]
    learned_ord = np.argsort(pred_all)[::-1]

    ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    oracle_r, learned_r = {}, {}
    for k in ks:
        n = max(1, int(len(gt_all) * k / 100))
        oracle_r[k] = float(fg_all[oracle_ord[:n]].sum() / max(fg_all.sum(), 1))
        learned_r[k] = float(fg_all[learned_ord[:n]].sum() / max(fg_all.sum(), 1))

    # Per-tile mean retention (更贴近真实路由场景)
    # Per-tile mean retention (closer to real routing scenario)
    # 对每个 tile 独立计算 retention，然后平均
    per_tile_oracle = {k: [] for k in ks}
    per_tile_learned = {k: [] for k in ks}
    offset = 0
    for pred_t, gt_t, fg_t in zip(all_pred, all_gt, all_fg_px):
        n_tokens = len(fg_t)
        if fg_t.sum() == 0:
            offset += n_tokens
            continue
        o_ord = np.argsort(gt_t.numpy())[::-1]
        l_ord = np.argsort(pred_t.numpy())[::-1]
        for k in ks:
            nk = max(1, int(n_tokens * k / 100))
            per_tile_oracle[k].append(float(fg_t[o_ord[:nk]].sum() / fg_t.sum()))
            per_tile_learned[k].append(float(fg_t[l_ord[:nk]].sum() / fg_t.sum()))

    per_tile_mean_oracle = {k: float(np.mean(v)) if v else 0.0 for k, v in per_tile_oracle.items()}
    per_tile_mean_learned = {k: float(np.mean(v)) if v else 0.0 for k, v in per_tile_learned.items()}

    return {
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "global_oracle_retention": oracle_r,
        "global_learned_retention": learned_r,
        "per_tile_oracle_retention": per_tile_mean_oracle,
        "per_tile_learned_retention": per_tile_mean_learned,
        "n_tokens": len(gt_all),
        "n_tiles": len(all_pred),
        "pred_all": pred_all,
        "gt_all": gt_all,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (对标 B-02 原版 6-panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results, all_losses, args, n_trainable, out_dir):
    """对标 B-02 原版的 6-panel 可视化 | Match B-02 original 6-panel visualization."""
    ks = sorted(results["global_oracle_retention"].keys())
    ofg = [results["global_oracle_retention"][k] * 100 for k in ks]
    lfg = [results["global_learned_retention"][k] * 100 for k in ks]
    per_tile_ofg = [results["per_tile_oracle_retention"][k] * 100 for k in ks]
    per_tile_lfg = [results["per_tile_learned_retention"][k] * 100 for k in ks]
    sr = results["spearman_r"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    # Panel (0,0): FG Retention Curve (Per-Tile Mean) — 核心结果
    ax = axes[0, 0]
    ax.fill_between(ks, per_tile_lfg, per_tile_ofg, alpha=0.08, color="gray")
    ax.plot(ks, per_tile_ofg, "o-", color="#E74C3C", lw=2.5, ms=7, label="Oracle")
    ax.plot(ks, per_tile_lfg, "s-", color="#3498DB", lw=2.5, ms=7,
            label=f"Learned (r={sr:.3f})")
    ax.axvline(30, color="gray", ls="--", alpha=0.3)
    ax.axvline(40, color="gray", ls="--", alpha=0.3)
    ax.set(xlabel="Tokens Kept (%)", ylabel="FG Retained (%)",
           title="Per-Tile Mean: Oracle vs Learned FG Retention",
           xlim=(0, 105), ylim=(0, 105))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel (0,1): Training Loss
    ax = axes[0, 1]
    ax.plot(range(1, len(all_losses) + 1), all_losses, "o-", color="#27AE60", lw=2)
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title="Training Loss")
    ax.grid(alpha=0.2)

    # Panel (0,2): Oracle - Learned Gap
    ax = axes[0, 2]
    gap = [o - l for o, l in zip(per_tile_ofg, per_tile_lfg)]
    bar_colors = ["#E74C3C" if g > 5 else "#F39C12" if g > 2 else "#27AE60" for g in gap]
    ax.bar(ks, gap, width=3, color=bar_colors, edgecolor="white", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Tokens Kept (%)", ylabel="Oracle - Learned Gap (%)",
           title=f"Retention Gap (mean={np.mean(gap):.2f}%)")
    ax.grid(axis="y", alpha=0.2)

    # Panel (1,0): Pred vs GT Scatter
    ax = axes[1, 0]
    n_sample = min(5000, len(results["pred_all"]))
    idx = np.random.RandomState(42).choice(len(results["pred_all"]), n_sample, replace=False)
    ax.scatter(results["gt_all"][idx], results["pred_all"][idx],
               alpha=0.15, s=3, c="#3498DB", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set(xlabel="GT fg_ratio", ylabel="Predicted Importance",
           title=f"Pred vs GT (n={n_sample} tokens)")
    ax.grid(alpha=0.2)

    # Panel (1,1): IDG Comparison
    ax = axes[1, 1]
    oracle_idg = {k: results["per_tile_oracle_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    learned_idg = {k: results["per_tile_learned_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    xp = np.arange(4)
    w = 0.3
    for i, k in enumerate([20, 30, 40, 50]):
        ax.bar(i - w / 2, oracle_idg[k], w, color="#E74C3C", ec="white",
               alpha=0.85, label="Oracle" if i == 0 else "")
        ax.bar(i + w / 2, learned_idg[k], w, color="#3498DB", ec="white",
               alpha=0.85, label="Learned" if i == 0 else "")
    ax.set_xticks(xp)
    ax.set_xticklabels(["Top20%", "Top30%", "Top40%", "Top50%"])
    ax.set(ylabel="IDG (FG Retention / Token Fraction)",
           title=f"IDG: Oracle={oracle_idg[40]:.2f}x vs Learned={learned_idg[40]:.2f}x")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Panel (1,2): 文本摘要 + 实验信息
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Verdict
    if sr > 0.6:
        verdict = "LEARNABLE"
        v_color = "#27AE60"
    elif sr > 0.3:
        verdict = "PARTIALLY LEARNABLE"
        v_color = "#F39C12"
    else:
        verdict = "HARD"
        v_color = "#E74C3C"

    l40 = results["per_tile_learned_retention"][40] * 100
    o40 = results["per_tile_oracle_retention"][40] * 100
    best_k = max(ks, key=lambda k: results["per_tile_learned_retention"][k] - k / 100)
    best_l = results["per_tile_learned_retention"][best_k] * 100

    lines = [
        "B-02 v2: Token-Level Learnability",
        "=" * 38,
        "",
        f"Backbone: FastSAM P4 (frozen)",
        f"Model: FDR DensityHead (165K)",
        f"Train: {results['n_tiles']} tiles x {args.epochs} ep",
        f"Eval tokens: {results['n_tokens']:,}",
        "",
        f"Pearson r  = {results['pearson_r']:.4f}",
        f"Spearman r = {sr:.4f}",
        f"Oracle Top40: {o40:.1f}% FG",
        f"Learned Top40: {l40:.1f}% FG",
        f"Learned Top{best_k}%: {best_l:.1f}% FG (best)",
        "",
        f"Oracle IDG@40: {oracle_idg[40]:.2f}x",
        f"Learned IDG@40: {learned_idg[40]:.2f}x",
        "",
        f"VERDICT: {verdict}",
    ]
    for i, line in enumerate(lines):
        y = 9.5 - i * 0.38
        if line.startswith("B-02"):
            ax.text(0.5, y, line, fontsize=12, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("===="):
            ax.text(0.5, y, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        elif "VERDICT" in line:
            ax.text(0.5, y, line, fontsize=11, fontweight="bold",
                    fontfamily="monospace", va="top", color=v_color)
        elif "→" in line or "Spearman" in line or "Pearson" in line:
            ax.text(0.5, y, line, fontsize=9, fontfamily="monospace",
                    va="top")
        else:
            ax.text(0.5, y, line, fontsize=9, fontfamily="monospace", va="top")

    fig.suptitle("B-02 v2: Token-Level Learnability — Can FDR Learn Per-Token FG Density from FastSAM P4?",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "learnability_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"Saved → {out_dir / 'learnability_v2.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "b02_v2.jsonl")))

    logger.log_info("config", "=" * 60)
    logger.log_info("config", "B-02 v2: Token-Level Learnability — FDR on FastSAM P4")
    logger.log_info("config", "=" * 60)
    logger.log_info("config",
                    f"Tile root: {args.tile_root}, Epochs: {args.epochs}, "
                    f"Batch: {args.batch_size}, LR: {args.lr}, Device: {device}")

    # ── Data ──
    train_ds = TokenFGRatioDataset(args.tile_root, args.train_split)
    val_ds = TokenFGRatioDataset(args.tile_root, args.val_split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ── Backbone ──
    logger.log_info("model", "Loading frozen FastSAM backbone...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()
    backbone.eval()

    # ── FDR (对标 eval_fdr_token_fss.py) ──
    fdr = ForegroundDensityRouter(in_channels=1280, mid_channels=128)
    fdr.train().to(device)
    n_params = sum(p.numel() for p in fdr.parameters())
    n_trainable = sum(p.numel() for p in fdr.parameters() if p.requires_grad)
    logger.log_info("model", f"FDR params: {n_params:,} ({n_trainable:,} trainable)")

    # ── Optimizer ──
    opt = torch.optim.AdamW(fdr.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    # ── Training ──
    logger.log_info("train", f"Training {args.epochs} epochs...")
    all_losses = []
    best_sr = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(fdr, backbone, train_loader, opt, device, scaler)
        sch.step()
        all_losses.append(avg_loss)

        # 每 5 epoch 做一次评估 | Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            results = evaluate_model(fdr, backbone, val_loader, device)
            sr = results["spearman_r"]
            pr = results["pearson_r"]
            logger.log_info("epoch",
                            f"E{epoch:4d} | loss={avg_loss:.5f} | "
                            f"Pearson r={pr:.4f} | Spearman r={sr:.4f} | "
                            f"N_tokens={results['n_tokens']:,}")

            if sr > best_sr:
                best_sr = sr
                best_epoch = epoch
                torch.save({
                    "epoch": epoch, "spearman_r": sr,
                    "model_state_dict": fdr.state_dict(),
                }, out_dir / "fdr_best.pt")
        else:
            logger.log_info("epoch", f"E{epoch:4d} | loss={avg_loss:.5f}")

    # ── Final Evaluation ──
    logger.log_info("eval", "Final evaluation on validation set...")
    results = evaluate_model(fdr, backbone, val_loader, device)

    sr = results["spearman_r"]
    pr = results["pearson_r"]

    # ── Log Results (对标 B-02 原版输出格式) ──
    logger.log_info("results", "")
    logger.log_info("results", "=" * 60)
    logger.log_info("results", "B-02 v2 Results: Token-Level Learnability")
    logger.log_info("results", "=" * 60)
    logger.log_info("results",
                    f"Pearson r  = {pr:.4f}")
    logger.log_info("results",
                    f"Spearman r = {sr:.4f}")
    logger.log_info("results",
                    f"N tokens   = {results['n_tokens']:,} (from {results['n_tiles']} tiles)")

    # Retention table
    logger.log_info("results", "")
    logger.log_info("results",
                    f"  {'K':>6s} | {'Oracle':>10s} | {'Learned':>10s} | {'Gap':>7s}")
    logger.log_info("results", "  " + "-" * 39)
    for k in [20, 30, 40, 50]:
        o = results["per_tile_oracle_retention"][k] * 100
        l = results["per_tile_learned_retention"][k] * 100
        logger.log_info("results",
                        f"  {k:>5}% | {o:>9.2f}% | {l:>9.2f}% | {o - l:>6.2f}%")

    # IDG
    oracle_idg = {k: results["per_tile_oracle_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    learned_idg = {k: results["per_tile_learned_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    logger.log_info("results",
                    f"Oracle IDG@40={oracle_idg[40]:.2f}x, "
                    f"Learned IDG@40={learned_idg[40]:.2f}x")

    # Verdict
    if sr > 0.6:
        verdict = "LEARNABLE"
    elif sr > 0.3:
        verdict = "PARTIALLY LEARNABLE"
    else:
        verdict = "HARD"
    logger.log_info("results",
                    f"VERDICT: {verdict}  (Spearman r={sr:.4f}, best E{best_epoch})")

    # Compare with B-02 original
    logger.log_info("results", "")
    logger.log_info("results", "Comparison with B-02 Original:")
    logger.log_info("results", "  B-02 (MV3):        Spearman r=0.889, 1.48M params")
    logger.log_info("results", f"  B-02 v2 (FastSAM):  Spearman r={sr:.4f}, {n_trainable:,} params")
    logger.log_info("results",
                    f"  Delta: r={sr - 0.889:+.4f}, "
                    f"params ratio={(1.48e6 / n_trainable):.1f}x smaller")

    # ── Visualization ──
    plot_results(results, all_losses, args, n_trainable, out_dir)

    # ── Save JSON ──
    summary = {
        "experiment": "B-02 v2: Token-Level Learnability — FDR on FastSAM P4",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": {
            "pearson_r": pr,
            "spearman_r": sr,
            "per_tile_oracle_retention": results["per_tile_oracle_retention"],
            "per_tile_learned_retention": results["per_tile_learned_retention"],
            "global_oracle_retention": results["global_oracle_retention"],
            "global_learned_retention": results["global_learned_retention"],
            "oracle_idg_40": oracle_idg[40],
            "learned_idg_40": learned_idg[40],
            "n_tokens": results["n_tokens"],
            "n_tiles": results["n_tiles"],
        },
        "verdict": verdict,
        "comparison": {
            "b02_original_spearman": 0.889,
            "b02_original_params": 1480000,
            "v2_spearman": sr,
            "v2_params": n_trainable,
        },
    }
    with open(out_dir / "learnability_v2_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log_info("output", f"Results saved → {out_dir}/")

    logger.log_info("done", "B-02 v2 complete!")


if __name__ == "__main__":
    main()
