#!/usr/bin/env python3
"""
B-02 v2: Tile-Level Learnability — MobileNetV3 predicts per-tile fg_ratio
==========================================================================
Can a lightweight MobileNetV3 learn to predict tile importance from tile images?

B-02 原版: 整图 resize → 虚拟切 1024×1024 tile → MV3 → per-tile fg_ratio.
B-02 v2:   直接用预切 896×896 tile → MV3 → per-tile fg_ratio.
           适配到 eval_fdr_token_fss.py 的数据格式 (iSAID5i_tiles/tile_896).

实验设计 | Design:
    Backbone: MobileNetV3-Small (pretrained, frozen features, 1.48M)
    Head:     Conv1×1 → DWConv3×3 → Conv1×1 → Sigmoid → GlobalAvgPool → score
    Input:    896×896 tile image → resize 224×224
    GT:       Single fg_ratio per tile (from label mask)
    Loss:     MSE(pred_score, gt_fg_ratio)
    Metrics:  Spearman r (tile ranking), FG Retention curves, IDG

B-02 原版参考值: Spearman r=0.889, 1.48M params.

用法 | Usage::
    python tools/paper_b/eval_b02_learnability_v2.py \
        --tile-root data/iSAID5i_tiles/tile_896 \
        --epochs 30 --batch-size 32 --device cuda
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
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed

logger = get_logger("b02_v2")


def parse_args():
    p = argparse.ArgumentParser(
        description="B-02 v2: Tile-Level Learnability — MobileNetV3 on pre-cut tiles")
    p.add_argument("--tile-root", type=str, default="data/iSAID5i_tiles/tile_896")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224,
                   help="MobileNetV3 输入尺寸 | Input size for MV3")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/b02_learnability_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tiles", type=int, default=0,
                   help="限制训练 tile 数量 (0=全部) | Limit training tiles (0=all)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: Tile → single fg_ratio (tile-level, 对标 eval_fdr_token_fss.py 数据)
# ═══════════════════════════════════════════════════════════════════════════

class TileFGRatioDataset(Dataset):
    """
    预切 tile → 单个 fg_ratio 标签.
    Pre-cut tile → single fg_ratio label.

    对标 B-02 原版: 每 tile 一个 GT fg_ratio, 用于 tile 级重要性排序.
    Match B-02 original: one GT fg_ratio per tile, for tile-level importance ranking.
    """

    def __init__(self, tile_root: str, split: str = "train", image_size: int = 224):
        self.tile_root = Path(tile_root)
        self.split = split
        self.image_size = image_size

        img_dir = self.tile_root / split / "images"
        label_dir = self.tile_root / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        self.img_paths = sorted(img_dir.glob("*.png"))
        self.label_dir = label_dir

        # 预计算所有 tile 的 fg_ratio | Precompute all tile fg_ratios
        self.valid_indices = []
        self.fg_ratios = []

        for i, img_path in enumerate(self.img_paths):
            lp = self.label_dir / f"{img_path.stem}_label.png"
            if lp.exists():
                label = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
                if label is not None:
                    fg_ratio = (label > 0).sum() / label.size
                    self.valid_indices.append(i)
                    self.fg_ratios.append(float(fg_ratio))

        n_all = len(self.img_paths)
        n_fg = sum(1 for f in self.fg_ratios if f > 0)
        logger.log_info("data",
                        f"TileFGRatioDataset [{split}]: {len(self)} tiles "
                        f"({n_fg} with FG, {len(self)-n_fg} empty) | "
                        f"fg_ratio mean={np.mean(self.fg_ratios):.4f}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.img_paths[real_idx]

        # 加载图像, resize 到 MV3 输入尺寸 | Load image, resize to MV3 input size
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, S, S]

        fg_ratio = torch.tensor(self.fg_ratios[idx], dtype=torch.float32)
        return img_t, fg_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Model: MobileNetV3 + Density Head (对标 B-02 原版 MobileNetSpatialRouter)
# ═══════════════════════════════════════════════════════════════════════════

class MobileNetTileScorer(nn.Module):
    """
    MobileNetV3-Small → 单个 tile 重要性得分.
    MobileNetV3-Small → single tile importance score.

    对标 B-02 原版 MobileNetSpatialRouter, 但输出是 1 个 scalar 而非 feature map.
    简化: MV3 全局特征 → 小 MLP → fg_ratio ∈ [0,1].
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        # MobileNetV3-Small features: stride=32, 576 channels
        mnet = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.backbone = mnet.features  # [B, 576, 7, 7] for 224×224 input

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Global pooling → MLP → scalar score
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, 576, 1, 1]
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.log_info("model",
                        f"MobileNetTileScorer: {n_total:,} params "
                        f"({n_trainable:,} trainable, backbone frozen={freeze_backbone})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, 3, S, S] tile images.
        :return: [B] predicted fg_ratio scores.
        """
        feat = self.backbone(x)       # [B, 576, S/32, S/32]
        feat = self.pool(feat)        # [B, 576, 1, 1]
        feat = feat.flatten(1)        # [B, 576]
        return self.head(feat).squeeze(-1)  # [B]


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, device, scaler=None):
    """训练一个 epoch | Train one epoch."""
    model.train()
    total_loss, n = 0.0, 0
    criterion = nn.MSELoss()

    pbar = tqdm(loader, desc="B-02 [train]", unit="batch")
    for imgs, fg_gt in pbar:
        imgs, fg_gt = imgs.to(device), fg_gt.to(device)
        pred = model(imgs)
        loss = criterion(pred, fg_gt)
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
def evaluate_model(model, loader, device):
    """对标 B-02 原版 evaluate_model: 收集所有 tile 的 pred/gt."""
    model.eval()
    all_pred, all_gt, all_fg_px = [], [], []

    for imgs, fg_gts in tqdm(loader, desc="B-02 [eval]", unit="batch"):
        imgs = imgs.to(device)
        pred = model(imgs)
        all_pred.append(pred.cpu())
        all_gt.append(fg_gts)
        # fg_pixels: fg_ratio * tile_area (896*896)
        all_fg_px.append(fg_gts * (896 * 896))

    pred_all = torch.cat(all_pred).numpy()
    gt_all = torch.cat(all_gt).numpy()
    fg_all = torch.cat(all_fg_px).numpy()

    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pred_all, gt_all)
    sr, _ = spearmanr(pred_all, gt_all)

    # Oracle vs Learned FG retention (tile 级排序)
    oracle_ord = np.argsort(gt_all)[::-1]
    learned_ord = np.argsort(pred_all)[::-1]

    ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    oracle_r, learned_r = {}, {}
    for k in ks:
        n = max(1, int(len(gt_all) * k / 100))
        oracle_r[k] = float(fg_all[oracle_ord[:n]].sum() / max(fg_all.sum(), 1))
        learned_r[k] = float(fg_all[learned_ord[:n]].sum() / max(fg_all.sum(), 1))

    return {
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "oracle_retention": oracle_r,
        "learned_retention": learned_r,
        "n_tiles": len(gt_all),
        "pred_all": pred_all,
        "gt_all": gt_all,
        "fg_all": fg_all,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (对标 B-02 原版 6-panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results, all_losses, args, n_trainable, out_dir):
    """对标 B-02 原版 6-panel."""
    ks = sorted(results["oracle_retention"].keys())
    ofg = [results["oracle_retention"][k] * 100 for k in ks]
    lfg = [results["learned_retention"][k] * 100 for k in ks]
    sr = results["spearman_r"]
    pr = results["pearson_r"]

    oracle_idg = {k: results["oracle_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    learned_idg = {k: results["learned_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    # Panel (0,0): FG Retention Curve
    ax = axes[0, 0]
    ax.fill_between(ks, lfg, ofg, alpha=0.08, color="gray")
    ax.plot(ks, ofg, "o-", color="#E74C3C", lw=2.5, ms=7, label="Oracle")
    ax.plot(ks, lfg, "s-", color="#3498DB", lw=2.5, ms=7,
            label=f"Learned (r={sr:.3f})")
    ax.axvline(30, color="gray", ls="--", alpha=0.3)
    ax.axvline(40, color="gray", ls="--", alpha=0.3)
    ax.set(xlabel="Tiles Kept (%)", ylabel="FG Retained (%)",
           title="Oracle vs Learned Tile-Level FG Retention",
           xlim=(0, 105), ylim=(0, 105))
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Panel (0,1): Training Loss
    ax = axes[0, 1]
    ax.plot(range(1, len(all_losses) + 1), all_losses, "o-", color="#27AE60", lw=2)
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title="Training Loss"); ax.grid(alpha=0.2)

    # Panel (0,2): Oracle - Learned Gap
    ax = axes[0, 2]
    gap = [o - l for o, l in zip(ofg, lfg)]
    ax.bar(ks, gap, width=3, color="#E67E22", edgecolor="white", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Tiles Kept (%)", ylabel="Oracle - Learned Gap (%)",
           title=f"Retention Gap (mean={np.mean(gap):.2f}%)"); ax.grid(axis="y", alpha=0.2)

    # Panel (1,0): Pred vs GT Scatter
    ax = axes[1, 0]
    n_sample = min(3000, len(results["pred_all"]))
    idx = np.random.RandomState(42).choice(len(results["pred_all"]), n_sample, replace=False)
    ax.scatter(results["gt_all"][idx], results["pred_all"][idx],
               alpha=0.2, s=5, c="#3498DB", edgecolors="none")
    ax.plot([0, max(results["gt_all"][idx])], [0, max(results["gt_all"][idx])],
            "k--", alpha=0.3, lw=1)
    ax.set(xlabel="GT fg_ratio", ylabel="Predicted Score",
           title=f"Pred vs GT ({n_sample} tiles)\nPearson={pr:.4f}, Spearman={sr:.4f}")
    ax.grid(alpha=0.2)

    # Panel (1,1): IDG
    ax = axes[1, 1]
    xp = np.arange(4); w = 0.3
    for i, k in enumerate([20, 30, 40, 50]):
        ax.bar(i - w / 2, oracle_idg[k], w, color="#E74C3C", ec="white",
               alpha=0.85, label="Oracle" if i == 0 else "")
        ax.bar(i + w / 2, learned_idg[k], w, color="#3498DB", ec="white",
               alpha=0.85, label="Learned" if i == 0 else "")
    ax.set_xticks(xp); ax.set_xticklabels(["Top20%","Top30%","Top40%","Top50%"])
    ax.set(ylabel="IDG (FG Retention / Tile Fraction)",
           title=f"Oracle IDG@40={oracle_idg[40]:.2f}x, Learned={learned_idg[40]:.2f}x")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)

    # Panel (1,2): Summary
    ax = axes[1, 2]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    if sr > 0.6:
        verdict, vc = "LEARNABLE", "#27AE60"
    elif sr > 0.3:
        verdict, vc = "PARTIALLY LEARNABLE", "#F39C12"
    else:
        verdict, vc = "HARD", "#E74C3C"

    lines = [
        "B-02 v2: Tile-Level Learnability",
        "=" * 35, "",
        f"Backbone: MobileNetV3-Small",
        f"Trainable: {n_trainable:,} params",
        f"Train: {results['n_tiles']} tiles x {args.epochs} ep",
        f"Image: {args.image_size}x{args.image_size}",
        "", f"Pearson r  = {pr:.4f}",
        f"Spearman r = {sr:.4f}",
        f"Oracle Top40: {results['oracle_retention'][40]*100:.1f}% FG",
        f"Learned Top40: {results['learned_retention'][40]*100:.1f}% FG",
        "", f"Oracle IDG@40: {oracle_idg[40]:.2f}x",
        f"Learned IDG@40: {learned_idg[40]:.2f}x",
        "", f"VERDICT: {verdict}",
    ]
    for i, line in enumerate(lines):
        y = 9.5 - i * 0.42
        if "B-02" in line and i == 0:
            ax.text(0.5, y, line, fontsize=12, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif "===" in line:
            ax.text(0.5, y, line, fontsize=9, fontfamily="monospace", va="top", color="gray")
        elif "VERDICT" in line:
            ax.text(0.5, y, line, fontsize=11, fontweight="bold",
                    fontfamily="monospace", va="top", color=vc)
        else:
            ax.text(0.5, y, line, fontsize=9, fontfamily="monospace", va="top")

    fig.suptitle("B-02 v2: Tile-Level Learnability — MobileNetV3 predicts per-tile fg_ratio",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "learnability_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"Saved -> {out_dir / 'learnability_v2.png'}")


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
    logger.log_info("config", "B-02 v2: Tile-Level Learnability — MobileNetV3 on pre-cut tiles")
    logger.log_info("config", "=" * 60)
    logger.log_info("config",
                    f"Tile root: {args.tile_root}, Epochs: {args.epochs}, "
                    f"Batch: {args.batch_size}, LR: {args.lr}, Device: {device}")

    # ── Data ──
    train_ds = TileFGRatioDataset(args.tile_root, "train", args.image_size)
    val_ds = TileFGRatioDataset(args.tile_root, "val", args.image_size)

    if args.max_tiles > 0 and len(train_ds) > args.max_tiles:
        rng = np.random.RandomState(args.seed)
        train_ds.valid_indices = rng.choice(
            train_ds.valid_indices, args.max_tiles, replace=False).tolist()
        train_ds.fg_ratios = [train_ds.fg_ratios[i] for i in train_ds.valid_indices]
        logger.log_info("data", f"Limited to {args.max_tiles} training tiles")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ── Model ──
    model = MobileNetTileScorer(pretrained=True, freeze_backbone=True).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_info("model", f"Total={n_total:,}, Trainable={n_trainable:,}")

    # ── Optimizer ──
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    # ── Training ──
    logger.log_info("train", f"Training {args.epochs} epochs...")
    all_losses = []

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, opt, device, scaler)
        sch.step()
        all_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            results = evaluate_model(model, val_loader, device)
            sr, pr = results["spearman_r"], results["pearson_r"]
            logger.log_info("epoch",
                            f"E{epoch:4d} | loss={avg_loss:.5f} | "
                            f"Pearson r={pr:.4f} | Spearman r={sr:.4f} | "
                            f"N_tiles={results['n_tiles']:,}")
        else:
            logger.log_info("epoch", f"E{epoch:4d} | loss={avg_loss:.5f}")

    # ── Final Evaluation ──
    results = evaluate_model(model, val_loader, device)
    sr, pr = results["spearman_r"], results["pearson_r"]

    # ── Report ──
    oracle_idg = {k: results["oracle_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}
    learned_idg = {k: results["learned_retention"][k] / (k / 100) for k in [20, 30, 40, 50]}

    logger.log_info("results", "")
    logger.log_info("results", "=" * 60)
    logger.log_info("results", "B-02 v2 Results: Tile-Level Learnability (MobileNetV3)")
    logger.log_info("results", "=" * 60)
    logger.log_info("results", f"Pearson r  = {pr:.4f}")
    logger.log_info("results", f"Spearman r = {sr:.4f}")
    logger.log_info("results", f"N tiles    = {results['n_tiles']:,}")

    logger.log_info("results", "")
    logger.log_info("results", f"  {'K':>6s} | {'Oracle':>10s} | {'Learned':>10s} | {'Gap':>7s}")
    logger.log_info("results", "  " + "-" * 39)
    for k in [20, 30, 40, 50]:
        o = results["oracle_retention"][k] * 100
        l = results["learned_retention"][k] * 100
        logger.log_info("results", f"  {k:>5}% | {o:>9.2f}% | {l:>9.2f}% | {o-l:>6.2f}%")

    logger.log_info("results",
                    f"Oracle IDG@40={oracle_idg[40]:.2f}x, Learned IDG@40={learned_idg[40]:.2f}x")

    if sr > 0.6:
        verdict = "LEARNABLE"
    elif sr > 0.3:
        verdict = "PARTIALLY LEARNABLE"
    else:
        verdict = "HARD"
    logger.log_info("results", f"VERDICT: {verdict} (Spearman r={sr:.4f})")

    # Compare with B-02 original
    logger.log_info("results", "")
    logger.log_info("results", "Comparison:")
    logger.log_info("results", "  B-02 original:  r=0.889, 1.48M params, original images")
    logger.log_info("results", f"  B-02 v2:        r={sr:.4f}, {n_trainable:,} params, pre-cut tiles")

    # ── Visualization ──
    plot_results(results, all_losses, args, n_trainable, out_dir)

    # ── Save JSON ──
    summary = {
        "experiment": "B-02 v2: Tile-Level Learnability (MobileNetV3 on pre-cut tiles)",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": {
            "pearson_r": pr, "spearman_r": sr,
            "oracle_retention": results["oracle_retention"],
            "learned_retention": results["learned_retention"],
            "oracle_idg_40": oracle_idg[40],
            "learned_idg_40": learned_idg[40],
            "n_tiles": results["n_tiles"],
        },
        "verdict": verdict,
    }
    with open(out_dir / "learnability_v2_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log_info("output", f"Results saved -> {out_dir}/")
    logger.log_info("done", "B-02 v2 complete!")


if __name__ == "__main__":
    main()
