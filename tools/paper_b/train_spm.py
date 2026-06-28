#!/usr/bin/env python3
"""
SPM / FDR 训练 — Token 级前景密度路由器
==========================================
Train Token-Level Foreground Density Router (SPM Phase 2).

训练流程 | Training Pipeline:
    1. 冻结 FastSAM backbone → 提取 P4 特征 [B, 1280, H/16, W/16]
    2. FDR (DensityHead, 75K params) → 预测 per-token 密度 [B, 1, H/16, W/16]
    3. GT 监督信号: 从 label mask 计算的 per-token fg_ratio
    4. 损失: MSE(预测密度, GT密度)

评估指标 | Evaluation Metrics:
    - Spearman r: 预测与 GT 的排序相关性 (B-02 核心指标, r=0.889)
    - Val MSE: 回归精度
    - Per-tile 可视化: 预测热力图 vs GT 热力图

设计原则 | Design:
    - 类别无关: FDR 学习 objectness, 不学习 class semantics
    - 密度驱动: 监督信号 = fg_ratio, 而非边缘或纹理
    - 极致轻量: 75K 参数, 相对 decoder 可忽略不计

用法 | Usage::
    python tools/paper_b/train_spm.py \
        --tile-root data/iSAID5i_tiles/tile_896 \
        --epochs 50 --batch-size 16 --device cuda

    # 服务器 | On server:
    python tools/paper_b/train_spm.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --epochs 100 --batch-size 32 --device cuda \
        --output-dir runs/spm_training
"""

import sys, argparse, json, time
from pathlib import Path
from collections import defaultdict

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

logger = get_logger("spm_train")

# ═══════════════════════════════════════════════════════════════════════════
# 数据集: Tile → Per-Token FG Ratio | Dataset: Tile → Per-Token FG Ratio
# ═══════════════════════════════════════════════════════════════════════════

class TokenFGRatioDataset(Dataset):
    """
    从 tile 图像提取 per-token fg_ratio GT (P4 stride=16)。
    Extracts per-token foreground ratio GT from tile images at P4 resolution (stride=16).

    每张 tile 的 label mask 下采样到 P4 空间分辨率 (H/16 × W/16)，
    计算每个 token 格内的前景像素比例作为回归目标。
    Each tile's label mask is downsampled to P4 spatial resolution,
    computing the foreground pixel ratio within each token cell as the regression target.

    Parameters
    ----------
    tile_root : str
        Tile 数据集根目录 (含 {split}/images/ 和 {split}/labels/)。
    split : str
        数据划分 (train/val)。
    stride : int
        P4 步长 (默认 16)。
    """

    def __init__(self, tile_root: str, split: str = "train", stride: int = 16):
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

        # 检查有效的 tile (至少有一个前景像素) | Filter to valid tiles (at least 1 FG pixel)
        self._valid_indices = []
        for i, img_path in enumerate(self.img_paths):
            label_path = self.label_dir / f"{img_path.stem}_label.png"
            if label_path.exists():
                label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
                if label is not None and (label > 0).any():
                    self._valid_indices.append(i)

        logger.log_info("data",
                        f"TokenFGRatioDataset [{split}]: {len(self._valid_indices)}/"
                        f"{len(self.img_paths)} tiles with FG (stride={stride})")

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        """返回 (image [3,H,W], fg_ratio [1, H/stride, W/stride])。"""
        real_idx = self._valid_indices[idx]
        img_path = self.img_paths[real_idx]

        # ── 加载图像 | Load image ──
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]

        # ── 加载 label → per-token fg_ratio | Load label → per-token fg_ratio ──
        label_path = self.label_dir / f"{img_path.stem}_label.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        H, W = label.shape[:2]
        fg = (label > 0).astype(np.float32)

        # 裁剪到 stride 对齐 | Crop to stride-aligned region
        h_t, w_t = H // self.stride, W // self.stride
        fg = fg[:h_t * self.stride, :w_t * self.stride]

        # 重塑 → per-token 均值 | Reshape → per-token mean
        fg_ratio = (fg.reshape(h_t, self.stride, w_t, self.stride)
                      .transpose(0, 2, 1, 3)
                      .reshape(h_t, w_t, -1)
                      .mean(axis=2))

        fg_t = torch.from_numpy(fg_ratio).unsqueeze(0).float()  # [1, H/16, W/16]
        return img_t, fg_t


# ═══════════════════════════════════════════════════════════════════════════
# 评估: Spearman 相关系数 | Evaluation: Spearman Rank Correlation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_spearman(preds: list, gts: list) -> float:
    """
    计算预测重要性图与 GT 密度图的 Spearman 排序相关系数。
    Compute Spearman rank correlation between predicted importance and GT density.

    这是 B-02 的核心指标: r 衡量 FDR 学习"排序"的能力，
    而非精确值重建 — 因为路由只需要 Top-K 顺序。
    This is B-02's core metric: r measures FDR's ranking ability,
    not exact value reconstruction — routing only needs Top-K ordering.

    Parameters
    ----------
    preds : list of torch.Tensor
        预测的重要性图列表 (每个形状 [1, H, W])。
    gts : list of torch.Tensor
        GT fg_ratio 图列表。

    Returns
    -------
    float
        Spearman r ∈ [-1, 1]。
    """
    from scipy.stats import spearmanr
    if not preds or not gts:
        return 0.0
    pred_flat = torch.cat([p.flatten().cpu() for p in preds]).numpy()
    gt_flat = torch.cat([g.flatten().cpu() for g in gts]).numpy()
    r, _ = spearmanr(pred_flat, gt_flat)
    return float(r)


@torch.no_grad()
def compute_metrics(preds: list, gts: list) -> dict:
    """
    计算多个评估指标 | Compute multiple evaluation metrics.

    Returns dict with:
        spearman_r: 排序相关性 | rank correlation
        mse: 均方误差 | mean squared error
        mae: 平均绝对误差 | mean absolute error
        pred_mean: 预测均值 | prediction mean (用于检测坍缩 | for collapse detection)
        gt_mean: GT 均值 | ground truth mean
    """
    if not preds or not gts:
        return {"spearman_r": 0.0, "mse": 0.0, "mae": 0.0,
                "pred_mean": 0.0, "gt_mean": 0.0}

    pred_cat = torch.cat([p.flatten().cpu() for p in preds])
    gt_cat = torch.cat([g.flatten().cpu() for g in gts])

    mse = F.mse_loss(pred_cat, gt_cat).item()
    mae = F.l1_loss(pred_cat, gt_cat).item()
    spearman_r = compute_spearman(preds, gts)

    return {
        "spearman_r": spearman_r,
        "mse": mse,
        "mae": mae,
        "pred_mean": pred_cat.mean().item(),
        "gt_mean": gt_cat.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 训练 | Training
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(
    fdr: nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> dict:
    """训练一个 epoch | Train one epoch."""
    fdr.train()
    backbone.eval()

    total_loss = 0.0
    all_preds, all_gts = [], []

    pbar = tqdm(loader, desc="SPM [train]", unit="batch")
    for imgs, fg_gt in pbar:
        imgs = imgs.to(device)
        fg_gt = fg_gt.to(device)

        # ── 冻结 backbone → P4 特征 | Frozen backbone → P4 features ──
        with torch.no_grad():
            p4 = backbone(imgs)["p4"]  # [B, 1280, H/16, W/16]

        # ── FDR → 重要性图 | FDR → importance map ──
        imp = fdr(p4)["importance"]  # [B, 1, H/16, W/16]

        # ── 对齐分辨率 (如果 backbone 和 label 分辨率不同) | Align resolution ──
        if imp.shape[2:] != fg_gt.shape[2:]:
            fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                  mode="bilinear", align_corners=False)

        # ── 损失 | Loss ──
        loss = criterion(imp, fg_gt)

        # ── 反向传播 | Backward ──
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(imp.detach())
        all_gts.append(fg_gt.detach())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_gts)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate_epoch(
    fdr: nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """验证一个 epoch | Validate one epoch."""
    fdr.eval()
    backbone.eval()

    total_loss = 0.0
    all_preds, all_gts = [], []

    for imgs, fg_gt in tqdm(loader, desc="SPM [val]", unit="batch"):
        imgs = imgs.to(device)
        fg_gt = fg_gt.to(device)

        p4 = backbone(imgs)["p4"]
        imp = fdr(p4)["importance"]

        if imp.shape[2:] != fg_gt.shape[2:]:
            fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                  mode="bilinear", align_corners=False)

        loss = criterion(imp, fg_gt)
        total_loss += loss.item()
        all_preds.append(imp.detach())
        all_gts.append(fg_gt.detach())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_gts)
    metrics["loss"] = avg_loss
    return metrics


def train_spm(args, backbone: FastSAMBackbone, device: torch.device, out_dir: Path):
    """
    SPM/FDR 完整训练流程 | Complete SPM/FDR training pipeline.

    Returns
    -------
    fdr : ForegroundDensityRouter
        训练好的 FDR 模型 | Trained FDR model.
    metrics_history : list[dict]
        每个 epoch 的训练/验证指标 | Per-epoch train/val metrics.
    """
    logger.log_info("train", "=" * 70)
    logger.log_info("train", "SPM/FDR Training — Token-Level Foreground Density Router")
    logger.log_info("train", "=" * 70)

    # ── 数据集 | Datasets ──
    train_ds = TokenFGRatioDataset(args.tile_root, "train", stride=16)
    val_ds = TokenFGRatioDataset(args.tile_root, "val", stride=16)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    logger.log_info("data", f"Train tiles: {len(train_ds):,} (FG>0 filtered)")
    logger.log_info("data", f"Val tiles:   {len(val_ds):,} (FG>0 filtered)")
    logger.log_info("data", f"Batch size: {args.batch_size}, Workers: {args.workers}")

    # ── 模型 | Model ──
    fdr = ForegroundDensityRouter(
        in_channels=1280,   # FastSAM P4 channels
        mid_channels=128,   # DensityHead mid channels
        tile_size_feat=1,   # Token-level (no tile pooling)
    )
    fdr.train().to(device)

    n_params = sum(p.numel() for p in fdr.parameters())
    n_trainable = sum(p.numel() for p in fdr.parameters() if p.requires_grad)
    logger.log_info("model", f"FDR params: {n_params:,} ({n_trainable:,} trainable)")

    # ── 优化器 + 调度器 | Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        fdr.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── 损失 | Loss ──
    criterion = nn.MSELoss()

    # ── AMP | Mixed precision ──
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    # ── 训练循环 | Training Loop ──
    best_val_r = -1.0
    best_epoch = 0
    metrics_history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── 训练 | Train ──
        train_m = train_epoch(fdr, backbone, train_loader, optimizer, criterion,
                              device, scaler)

        # ── 验证 | Validate ──
        val_m = validate_epoch(fdr, backbone, val_loader, criterion, device)

        scheduler.step()

        # ── 记录 | Log ──
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log_info(
            "epoch",
            f"E{epoch:4d} | "
            f"train: loss={train_m['loss']:.5f} MSE={train_m['mse']:.5f} "
            f"r={train_m['spearman_r']:.4f} | "
            f"val: loss={val_m['loss']:.5f} MSE={val_m['mse']:.5f} "
            f"r={val_m['spearman_r']:.4f} | "
            f"lr={lr_now:.2e}",
        )

        # 重要性均值检查 (检测坍缩 | Collapse detection)
        logger.log_info(
            "density",
            f"  imp_mean: train={train_m['pred_mean']:.4f} val={val_m['pred_mean']:.4f} "
            f"vs GT={val_m['gt_mean']:.4f}",
        )

        metrics_history.append({
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
            "lr": lr_now,
        })

        # ── 保存最佳 | Save Best ──
        if val_m["spearman_r"] > best_val_r:
            best_val_r = val_m["spearman_r"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "val_metrics": val_m,
                "train_metrics": train_m,
                "model_state_dict": fdr.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
            }, out_dir / "spm_best.pt")
            logger.log_info("save", f"  ✓ Best model saved (r={best_val_r:.4f})")

        # ── 定期保存 | Periodic Save ──
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "val_metrics": val_m,
                "model_state_dict": fdr.state_dict(),
            }, out_dir / f"spm_e{epoch:04d}.pt")

    elapsed = time.time() - t_start
    logger.log_info("done", f"Training complete in {elapsed/60:.1f} min")
    logger.log_info("done", f"Best: E{best_epoch}, val r={best_val_r:.4f}")

    return fdr, metrics_history


# ═══════════════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_spm(
    fdr: nn.Module,
    backbone: FastSAMBackbone,
    val_loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    n_examples: int = 6,
):
    """
    生成 SPM 训练可视化 | Generate SPM training visualization.

    六个面板 | Six panels:
        1-2: 训练曲线 (Loss + Spearman r)
        3-4: 预测 vs GT 散点图 (per-token)
        5-6: 示例 tile 对比 (原图 + GT密度 + 预测密度 + 误差图)
    """
    logger.log_info("viz", "Generating SPM visualization...")

    # ── 收集验证集样本 | Collect validation samples ──
    fdr.eval()
    backbone.eval()

    samples = []
    for imgs, fg_gt in val_loader:
        imgs = imgs.to(device)
        fg_gt = fg_gt.to(device)
        p4 = backbone(imgs)["p4"]
        imp = fdr(p4)["importance"]
        if imp.shape[2:] != fg_gt.shape[2:]:
            fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                  mode="bilinear", align_corners=False)
        for i in range(min(imgs.shape[0], n_examples - len(samples))):
            samples.append({
                "image": imgs[i].cpu(),
                "gt": fg_gt[i, 0].cpu().numpy(),
                "pred": imp[i, 0].cpu().numpy(),
            })
        if len(samples) >= n_examples:
            break

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

    for idx in range(min(n_examples, len(samples))):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        s = samples[idx]

        # 原图 | Original image
        img_np = s["image"].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # GT 密度热力图 | GT density heatmap
        gt_heat = s["gt"]
        gt_heat_up = cv2.resize(gt_heat, (img_np.shape[1], img_np.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        gt_color = cv2.applyColorMap((gt_heat_up * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 预测密度热力图 | Predicted density heatmap
        pred_heat = s["pred"]
        pred_heat_up = cv2.resize(pred_heat, (img_np.shape[1], img_np.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        pred_color = cv2.applyColorMap((pred_heat_up * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 叠加显示: 原图 + GT overlay + Pred overlay | Side-by-side: GT vs Pred
        # 左侧: GT (绿色系) | Left: GT (green)
        # 右侧: Pred (红色系) | Right: Pred (red)
        gt_overlay = cv2.addWeighted(img_np, 0.4, gt_color.astype(np.float32) / 255.0, 0.6, 0)
        pred_overlay = cv2.addWeighted(img_np, 0.4, pred_color.astype(np.float32) / 255.0, 0.6, 0)

        # 拼接: GT 在左, Pred 在右 | Concatenate: GT left, Pred right
        combined = np.hstack([gt_overlay, pred_overlay])

        # 添加分界线 | Add divider line
        h, w = combined.shape[:2]
        mid_x = w // 2
        combined[:, mid_x:mid_x+2] = [1, 1, 1]

        ax.imshow(combined)
        ax.set_title(f"Tile {idx+1}\nLeft=GT | Right=Pred", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余 subplot | Hide unused subplots
    for idx in range(len(samples), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("SPM/FDR: Predicted vs GT Foreground Density\n"
                 "(Left=GT Heatmap, Right=FDR Prediction)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "spm_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"  Saved → {out_dir / 'spm_predictions.png'}")


def plot_metrics(metrics_history: list, out_dir: Path):
    """
    绘制训练曲线 | Plot training curves.

    Four panels:
        1. Loss (train + val)
        2. Spearman r (train + val)
        3. Importance mean (train + val vs GT)
        4. Learning rate
    """
    epochs = [m["epoch"] for m in metrics_history]
    train_loss = [m["train"]["loss"] for m in metrics_history]
    val_loss = [m["val"]["loss"] for m in metrics_history]
    train_r = [m["train"]["spearman_r"] for m in metrics_history]
    val_r = [m["val"]["spearman_r"] for m in metrics_history]
    train_imp = [m["train"]["pred_mean"] for m in metrics_history]
    val_imp = [m["val"]["pred_mean"] for m in metrics_history]
    gt_mean = metrics_history[0]["val"]["gt_mean"]
    lrs = [m["lr"] for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Train", alpha=0.7)
    ax.plot(epochs, val_loss, "r-", label="Val", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spearman r — 核心指标 | Core metric
    ax = axes[0, 1]
    ax.plot(epochs, train_r, "b-", label="Train", alpha=0.7)
    ax.plot(epochs, val_r, "r-", label="Val", linewidth=2.0)
    ax.axhline(y=0.889, color="green", linestyle="--", alpha=0.5,
               label="B-02 ref (MV3, r=0.889)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    best_r = max(val_r)
    best_e = epochs[val_r.index(best_r)]
    ax.annotate(f"Best: r={best_r:.4f} (E{best_e})",
                (best_e, best_r), textcoords="offset points",
                xytext=(0, 10), fontsize=10, ha="center",
                color="red", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman r")
    ax.set_title("Spearman Rank Correlation (↑ = Better Ranking)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    # Panel 3: Importance mean — 坍缩检测 | Collapse detection
    ax = axes[1, 0]
    ax.plot(epochs, train_imp, "b-", label="Train imp_mean", alpha=0.7)
    ax.plot(epochs, val_imp, "r-", label="Val imp_mean", alpha=0.7)
    ax.axhline(y=gt_mean, color="green", linestyle="--", alpha=0.5,
               label=f"GT mean={gt_mean:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Importance")
    ax.set_title("Importance Mean (Collapse Detection)\n"
                 "If → 0: importance collapse; If ≫ GT: over-estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, lrs, "k-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Cosine)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("SPM/FDR Training Curves", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "spm_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"  Saved → {out_dir / 'spm_metrics.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI & Main | 命令行入口
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SPM/FDR Training — Token-Level Foreground Density Router")

    # 数据 | Data
    p.add_argument("--tile-root", type=str, default="data/iSAID5i_tiles/tile_896",
                   help="Tile 数据集根目录 | Tile dataset root")
    p.add_argument("--output-dir", type=str, default="runs/spm_training",
                   help="输出目录 | Output directory")

    # 训练 | Training
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数 | Training epochs")
    p.add_argument("--batch-size", type=int, default=16,
                   help="批次大小 | Batch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率 | Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="权重衰减 | Weight decay")
    p.add_argument("--workers", type=int, default=4,
                   help="数据加载线程数 | DataLoader workers")

    # 硬件 | Hardware
    p.add_argument("--device", type=str, default="cuda",
                   help="设备 | Device (cuda/cpu)")
    p.add_argument("--amp", action="store_true",
                   help="启用自动混合精度 | Enable AMP")

    # 其他 | Other
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed")
    p.add_argument("--save-every", type=int, default=10,
                   help="定期保存间隔 (epoch) | Periodic save interval")
    p.add_argument("--skip-viz", action="store_true",
                   help="跳过可视化 | Skip visualization")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logging ──
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "spm_training.jsonl")))

    logger.log_info("config", f"Device: {device}, AMP: {args.amp}")
    logger.log_info("config", f"Epochs: {args.epochs}, Batch: {args.batch_size}, "
                    f"LR: {args.lr}, WD: {args.weight_decay}")
    logger.log_info("config", f"Tile root: {args.tile_root}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── 保存配置 | Save config ──
    with open(out_dir / "config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    # ── 加载 Backbone | Load Backbone ──
    logger.log_info("backbone", "Loading frozen FastSAM backbone...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()
    backbone.eval()
    logger.log_info("backbone", "Backbone loaded and frozen.")

    # ── 训练 SPM | Train SPM ──
    fdr, metrics_history = train_spm(args, backbone, device, out_dir)

    # ── 可视化 | Visualization ──
    if not args.skip_viz:
        logger.log_info("viz", "Generating visualizations...")
        plot_metrics(metrics_history, out_dir)

        # 用验证集生成预测可视化 | Prediction visualization on val set
        val_ds = TokenFGRatioDataset(args.tile_root, "val", stride=16)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0,  # 单进程避免 CUDA fork 问题 | Single process to avoid CUDA fork
        )
        visualize_spm(fdr, backbone, val_loader, out_dir, device, n_examples=6)

    # ── 保存最终结果 | Save Final Results ──
    best_r = max(m["val"]["spearman_r"] for m in metrics_history)
    best_epoch = max(
        range(len(metrics_history)),
        key=lambda i: metrics_history[i]["val"]["spearman_r"],
    )
    final_val = metrics_history[-1]["val"]

    summary = {
        "experiment": "SPM/FDR Training",
        "best_epoch": best_epoch + 1,
        "best_spearman_r": round(best_r, 4),
        "final_val_metrics": final_val,
        "n_params": sum(p.numel() for p in fdr.parameters()),
        "total_epochs": args.epochs,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 结论 | Conclusion ──
    logger.log_info("done", "")
    logger.log_info("done", "=" * 70)
    logger.log_info("done", f"SPM/FDR Training Complete!")
    logger.log_info("done", f"  Best val Spearman r: {best_r:.4f} (Epoch {best_epoch + 1})")
    logger.log_info("done", f"  Final val MSE:       {final_val['mse']:.5f}")
    logger.log_info("done", f"  Checkpoint:          {out_dir / 'spm_best.pt'}")
    logger.log_info("done", f"  Metrics plot:        {out_dir / 'spm_metrics.png'}")

    # ── B-02 对标 | Benchmark against B-02 ──
    b02_reference = 0.889  # MV3 backbone reference
    gap = b02_reference - best_r
    if gap < 0.05:
        logger.log_info("done",
                        f"  ✓ Within 5% of B-02 MV3 reference (r={b02_reference}, gap={gap:.4f})")
    else:
        logger.log_info("done",
                        f"  ⚠ Gap to B-02 MV3 reference: {gap:.4f} "
                        f"(FastSAM P4 features, no separate backbone)")

    # ── 下一阶段指令 | Next Phase Instructions ──
    logger.log_info("done", "")
    logger.log_info("done", "Next: Run Token Routing Verification (Phase 3)")
    logger.log_info("done", "  python tools/paper_b/eval_fdr_token_fss.py \\")
    logger.log_info("done", f"    --tile-root {args.tile_root} \\")
    logger.log_info("done", "    --fold 0 --shot 1 \\")
    logger.log_info("done", f"    --fdr-ckpt {out_dir / 'spm_best.pt'} \\")
    logger.log_info("done", "    --decoder-ckpt <path/to/decoder.pt> \\")
    logger.log_info("done", "    --skip-fdr-train --device cuda")
    logger.log_info("done", "=" * 70)


if __name__ == "__main__":
    main()
