#!/usr/bin/env python3
"""
B-02.5: Generalization Study — 空间重要性是否类别无关且可迁移？
================================================================

B-02 证明：模型能学会预测 Tile 重要性 (Spearman r=0.889)。
B-02.5 回答：这个能力是类别特定的还是类别无关的？

三个实验 | Three experiments:

    A: iSAID train → iSAID val           (B-02 已完成, 同分布基线)
    B: iSAID 部分类 → iSAID 未见类         (类别迁移 | Class generalization)
    C: iSAID → Cityscapes                 (跨数据集迁移 | Cross-dataset transfer)

核心假设 | Core hypothesis:
    Spatial importance is category-agnostic — the router learns visual
    complexity/texture patterns, not class-specific semantics.
    → 支持 Few-Shot 路线: Router 不需要见过所有类别就能工作.

实验 B 设计 | Experiment B design:
    - 从 iSAID 15 类中 hold out N 类
    - Train: 仅包含训练类的 tile
    - Test: 仅包含 hold-out 类的 tile
    - 如果 Spearman r 仍然 >0.7 → 类别无关

实验 C 设计 | Experiment C design:
    - Train: iSAID (航拍)
    - Test: Cityscapes (街景) — 完全不同的域、类别、分辨率
    - 如果 Spearman r 仍然 >0.6 → 跨域可迁移

用法 | Usage::
    # Experiment B (类别迁移)
    python tools/eval_b02_5_generalization.py --exp B --holdout-classes 5 --epochs 20

    # Experiment C (跨数据集)
    python tools/eval_b02_5_generalization.py --exp C --cityscapes-root data/cityscapes_tiles

    # Experiment A+B+C 全跑
    python tools/eval_b02_5_generalization.py --exp ALL
"""

import sys, argparse, json, datetime, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed

logger = get_logger("b02_5_gen")
logger.add_backend(ConsoleBackend())  # 终端实时输出 | Real-time console output

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

# iSAID 类别名 | iSAID class names (standard ISAID_CATEGORIES)
from adatile.utils.label_mapping import _ID_TO_NAME as ISAID_CLASSES

TILE_SIZE = 1024
IMAGE_SIZE = 2048  # 固定 resize 尺寸 | Fixed resize size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default="B",
                   choices=["A", "B", "C", "ALL"],
                   help="实验选择 | Experiment: A=同分布 B=类别迁移 C=跨数据集 ALL=全部")
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--train-images", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--holdout-classes", type=int, default=5,
                   help="Hold-out 类别数 | Number of classes to hold out")
    p.add_argument("--test-dataset", type=str, default="cityscapes",
                   choices=["cityscapes", "loveda"],
                   help="跨数据集测试集 | Cross-dataset test set")
    p.add_argument("--cityscapes-root", type=str, default="data/cityscapes_tiles",
                   help="Cityscapes 预处理目录 | Cityscapes tile root")
    p.add_argument("--loveda-root", type=str, default="data/LoveDA",
                   help="LoveDA 根目录 | LoveDA root dir")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="runs/b02_5_generalization")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 模型 | Model (same as B-02)
# ═══════════════════════════════════════════════════════════════════

class MobileNetSpatialRouter(nn.Module):
    """MobileNetV3-Small → Importance Map [B,1,H/32,W/32]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        mnet = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.backbone = mnet.features
        self.head = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ═══════════════════════════════════════════════════════════════════
# 数据准备 | Data preparation
# ═══════════════════════════════════════════════════════════════════

def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """渲染语义掩码 | Render semantic mask [H, W] uint8 (0-15)."""
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0:
            continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = bbox
            x1, y1, y2, x2 = max(0, int(x)), max(0, int(y)), min(h, int(y+bh)), min(w, int(x+bw))
            sem[y1:y2, x1:x2] = cat_id
            continue
        if isinstance(seg, dict):
            continue
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
            cv2.fillPoly(sem, [pts], cat_id)
    return sem


def _preprocess_worker(args_tuple):
    """预处理 worker: 渲染 mask → resize → 保存 .npz | Preprocess worker: render mask → resize → save .npz."""
    img_id, img_path, anns, image_size, cache_dir = args_tuple
    try:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        # 渲染语义分割掩码 | Render semantic mask
        mask = render_semantic_mask(anns, H, W)

        # Resize: 保持宽高比缩放到固定尺寸 | Resize: keep aspect ratio, scale to fixed square
        scale = image_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        img = np.array(Image.fromarray(img).resize((new_W, new_H), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((new_W, new_H), Image.NEAREST))

        # Pad: 右侧/底部补零到方形 | Zero-pad to square
        pad_h, pad_w = image_size - new_H, image_size - new_W
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        # 归一化 + CHW 转换 | Normalize + CHW conversion
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        H2, W2 = mask.shape
        # 计算每个维度 tile 数量 | Compute tile counts per dimension
        n_ty = (H2 + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W2 + TILE_SIZE - 1) // TILE_SIZE
        tile_scores = np.zeros((n_ty, n_tx), dtype=np.float32)  # fg_ratio 网格 | fg_ratio grid
        fg_pixels = np.zeros(n_ty * n_tx, dtype=np.int64)       # 总 FG 像素 | total FG pixels
        # Per-class FG per tile: 用于类别过滤评估 | For class-filtered evaluation
        class_fg = np.zeros((n_ty * n_tx, 16), dtype=np.int64)
        idx = 0
        # 双循环遍历所有 tile | Double loop over all tiles
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H2)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W2)
                tile_mask = mask[y0:y1, x0:x1]
                total_px = (y1 - y0) * (x1 - x0)
                # 前景占比 = fg_pixels / total_pixels | FG ratio
                fg_px = int((tile_mask > 0).sum())
                tile_scores[ty, tx] = fg_px / max(total_px, 1)
                fg_pixels[idx] = fg_px
                # 统计每类 FG 像素 (class 1-15) | Count FG pixels per class
                for c in range(1, 16):
                    class_fg[idx, c] = int((tile_mask == c).sum())
                idx += 1

        out_path = os.path.join(cache_dir, f"{img_id}.npz")
        np.savez_compressed(out_path, image=img, tile_scores=tile_scores,
                            fg_pixels=fg_pixels, class_fg=class_fg,
                            n_ty=n_ty, n_tx=n_tx)
        return out_path
    except Exception:
        return None


class CachedDataset(Dataset):
    """从 .npz 加载 | Load from .npz cache."""

    def __init__(self, npz_paths: list[str]):
        self.paths = npz_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 从 .npz 延迟加载: 避免 DataLoader pickle 大数组 | Lazy load from .npz: avoid pickling large arrays
        data = np.load(self.paths[idx])
        return (torch.from_numpy(data["image"]),         # [3,S,S] 归一化图像 | normalized image
                torch.from_numpy(data["tile_scores"]),    # [n_ty,n_tx] GT fg_ratio 网格 | GT fg_ratio grid
                torch.from_numpy(data["fg_pixels"]),      # [n_tiles] 前景像素数 | FG pixel count
                torch.from_numpy(data["class_fg"]),       # [n_tiles,16] 每类 FG 像素 | per-class FG pixels
                int(data["n_ty"]), int(data["n_tx"]))     # tile 网格尺寸 | tile grid dimensions


def preprocess_to_cache(images: list, image_size: int, cache_dir: str) -> list[str]:
    """多进程预处理 → 返回 .npz 路径列表."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)
    tasks = [(img_id, img_path, anns, image_size, cache_dir)
             for img_id, img_path, anns, _ in images]
    n_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(_preprocess_worker, tasks),
                       total=len(tasks), desc="  Preprocess", unit="img"):
            if r is not None:
                paths.append(r)
    return paths


# ═══════════════════════════════════════════════════════════════════
# 训练 & 评估 (复用 B-02 逻辑) | Training & Evaluation (reuse B-02)
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, device):
    """训练一个 epoch | Train one epoch."""
    model.train()
    total_loss, n = 0.0, 0
    for imgs, gt_scores, fg_px, class_fg, n_ty_arr, n_tx_arr in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        # 前向传播: 图像 → 重要性图 | Forward: image → importance map
        imp_maps = model(imgs)
        _, _, hp, wp = imp_maps.shape

        batch_loss = 0.0
        # 逐样本: 每个样本可能不同的 tile 数量 | Per-sample: varying tile counts per image
        for b in range(B):
            gt = gt_scores[b].to(device)
            n_ty, n_tx = int(n_ty_arr[b]), int(n_tx_arr[b])
            pred_list, gt_list = [], []
            # 固定 stride=32: MV3 backbone 下采样 32× | Fixed stride=32: MV3 downsamples 32×
            for ty in range(min(n_ty, hp // 32)):
                for tx in range(min(n_tx, wp // 32)):
                    y0, y1 = ty * 32, min(ty * 32 + 32, hp)
                    x0, x1 = tx * 32, min(tx * 32 + 32, wp)
                    if y1 > y0 and x1 > x0 and ty < gt.shape[0] and tx < gt.shape[1]:
                        pred_list.append(imp_maps[b, 0, y0:y1, x0:x1].mean())
                        gt_list.append(gt[ty, tx])
            if pred_list:
                batch_loss += F.mse_loss(torch.stack(pred_list), torch.stack(gt_list))

        # 梯度更新 | Gradient update
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        total_loss += batch_loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, device, class_filter=None):
    """
    评估 | Evaluate.
    class_filter: 如果指定, 只统计包含这些类别的 tile | If set, only count tiles with these classes.
    """
    model.eval()
    all_pred, all_gt, all_fg = [], [], []

    for imgs, gt_scores, fg_px, class_fg, n_ty_arr, n_tx_arr in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        # 前向传播 → 重要性图 | Forward → importance map
        imp_maps = model(imgs)
        _, _, hp, wp = imp_maps.shape

        for b in range(B):
            gt = gt_scores[b].numpy()
            fg = fg_px[b].numpy()
            cf = class_fg[b].numpy()  # 每类 FG 像素矩阵 | per-class FG pixel matrix
            n_ty, n_tx = int(n_ty_arr[b]), int(n_tx_arr[b])

            idx = 0
            for ty in range(min(n_ty, hp // 32)):
                for tx in range(min(n_tx, wp // 32)):
                    y0, y1 = ty * 32, min(ty * 32 + 32, hp)
                    x0, x1 = tx * 32, min(tx * 32 + 32, wp)
                    if y1 > y0 and x1 > x0 and ty < gt.shape[0] and tx < gt.shape[1]:
                        # 类别过滤: 如果指定了 class_filter, 只保留包含目标类别的 tile | Class filter: if set, keep only tiles containing target classes
                        if class_filter is not None:
                            # 检查 tile 是否包含至少一个指定类别的像素 | Check if tile contains at least one pixel of target classes
                            has_target_class = any(cf[idx, c] > 0 for c in class_filter)
                            if not has_target_class:
                                idx += 1
                                continue
                        all_pred.append(imp_maps[b, 0, y0:y1, x0:x1].mean().item())
                        all_gt.append(float(gt[ty, tx]))
                        all_fg.append(int(fg[idx]))
                        idx += 1

    if len(all_pred) < 50:
        return None  # 样本不足 | Insufficient samples

    pred_all = np.array(all_pred)
    gt_all = np.array(all_gt)
    fg_all = np.array(all_fg)

    # 计算 Pearson (线性相关) 和 Spearman (排序相关) | Compute Pearson and Spearman correlation
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pred_all, gt_all)
    sr, _ = spearmanr(pred_all, gt_all)

    # 按真实密度/预测得分排序 → 计算 FG 保留率 | Sort by true density / predicted score → compute FG retention
    oracle_ord = np.argsort(gt_all)[::-1]
    learned_ord = np.argsort(pred_all)[::-1]

    # 对每个 Top-K% 计算保留的 FG 占比 | For each Top-K%, compute fraction of FG retained
    oracle_r, learned_r = {}, {}
    for k in [20, 30, 40, 50]:
        n = max(1, int(len(gt_all) * k / 100))
        oracle_r[k] = float(fg_all[oracle_ord[:n]].sum() / max(fg_all.sum(), 1))
        learned_r[k] = float(fg_all[learned_ord[:n]].sum() / max(fg_all.sum(), 1))

    return {
        "pearson_r": float(pr), "spearman_r": float(sr),
        "oracle_retention": oracle_r, "learned_retention": learned_r,
        "n_tiles": len(gt_all),
    }


def print_results(results, label, logger_tag):
    """打印结果到日志 | Print results to log."""
    if results is None:
        logger.log_info(logger_tag, f"{label}: INSUFFICIENT DATA")
        return
    # 输出 Pearson/Spearman 相关性 + tile 数量 | Output correlation + tile count
    logger.log_info(logger_tag,
                    f"{label}: Pearson r={results['pearson_r']:.4f}  "
                    f"Spearman r={results['spearman_r']:.4f}  "
                    f"N={results['n_tiles']:,}")
    # 输出 Top-K FG 保留率表格 | Output Top-K FG retention table
    logger.log_info(logger_tag,
                    f"  {'K%':<7} {'Oracle':>9} {'Learned':>9} {'Gap':>7}")
    for k in [20, 30, 40, 50]:
        o = results["oracle_retention"][k] * 100
        l = results["learned_retention"][k] * 100
        logger.log_info(logger_tag,
                        f"  {k:>5}%  {o:>8.2f}%  {l:>8.2f}%  {o - l:>6.2f}%")


# ═══════════════════════════════════════════════════════════════════
# Experiment A: 同分布 | Same distribution (baseline)
# ═══════════════════════════════════════════════════════════════════

def run_exp_a(args, all_images, cache_dir):
    """iSAID train → iSAID val (同分布基线) | Same-distribution baseline."""
    logger.log_info("b02_5/expA", "=" * 50)
    logger.log_info("b02_5/expA", "Experiment A: iSAID → iSAID val (same-distribution)")

    # Train/eval 随机划分 | Random train/eval split
    np.random.seed(args.seed)
    perm = np.random.permutation(len(all_images))
    n_train = min(args.train_images, len(all_images) - 30)
    train_imgs = [all_images[i] for i in perm[:n_train]]
    eval_imgs = [all_images[i] for i in perm[n_train:min(n_train + 50, len(all_images))]]

    # 预处理 + 缓存 | Preprocess + cache
    train_cache = os.path.join(cache_dir, "A_train")
    eval_cache = os.path.join(cache_dir, "A_eval")
    train_paths = preprocess_to_cache(train_imgs, IMAGE_SIZE, train_cache)
    eval_paths = preprocess_to_cache(eval_imgs, IMAGE_SIZE, eval_cache)

    train_ds = CachedDataset(train_paths)
    eval_ds = CachedDataset(eval_paths)

    # DataLoader: 训练集 shuffle + pin_memory (GPU) | Train: shuffle + pin_memory for GPU
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(args.device == "cuda"),
                              persistent_workers=(args.num_workers > 0))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(args.device == "cuda"),
                             persistent_workers=(args.num_workers > 0))

    # 模型: MV3 backbone (frozen) + Conv Head (trainable)
    model = MobileNetSpatialRouter(pretrained=True).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # 训练循环 | Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, args.device)
        sch.step()
        if epoch % 5 == 0 or epoch == 1:
            logger.log_info("b02_5/expA", f"E{epoch}/{args.epochs} loss={loss:.4f}")

    # 评估 + 输出 | Evaluate + print
    results = evaluate_model(model, eval_loader, args.device)
    print_results(results, "Exp A: same-dist", "b02_5/expA")
    return results, model


# ═══════════════════════════════════════════════════════════════════
# Experiment B: 类别迁移 | Class generalization
# ═══════════════════════════════════════════════════════════════════

def run_exp_b(args, all_images, cache_dir):
    """训练集部分类别 → 测试集未见类别 | Train on seen classes, test on unseen."""
    logger.log_info("b02_5/expB", "=" * 50)
    logger.log_info("b02_5/expB",
                    f"Experiment B: Class Generalization "
                    f"(holdout {args.holdout_classes} classes)")

    # 随机选择 hold-out 类别 (15 类中随机挑 N 类) | Randomly select hold-out classes from 15
    np.random.seed(args.seed)
    all_classes = list(range(1, 16))
    holdout_classes = sorted(np.random.choice(all_classes, size=args.holdout_classes,
                                              replace=False).tolist())
    train_classes = [c for c in all_classes if c not in holdout_classes]

    holdout_names = [ISAID_CLASSES[c] for c in holdout_classes]
    logger.log_info("b02_5/expB",
                    f"Hold-out classes: {holdout_classes} ({holdout_names})")
    logger.log_info("b02_5/expB",
                    f"Train classes: {train_classes}")

    # 分组图片: 图像级使用全部标注图 → tile 级过滤在评估时做
    # Image-level: use all annotated images → tile-level filtering at evaluation
    np.random.seed(args.seed)
    perm = np.random.permutation(len(all_images))

    train_imgs, test_imgs = [], []
    for i in perm:
        _, _, anns, _ = all_images[i]
        # 简单策略: 有标注的图全部用于训练 (tile 级过滤在评估时做)
        # Simple: all annotated images for training (filter at tile level)
        train_imgs.append(all_images[i])
        if len(train_imgs) >= args.train_images:
            break

    # 测试: 取后面的 50 张图 | Test: take next 50 images
    test_start = min(args.train_images + 30, len(all_images) - 20)
    for i in perm[test_start:test_start + 50]:
        test_imgs.append(all_images[i])

    # 预处理 + 缓存 | Preprocess + cache
    train_cache = os.path.join(cache_dir, "B_train")
    test_cache = os.path.join(cache_dir, "B_test")
    train_paths = preprocess_to_cache(train_imgs, IMAGE_SIZE, train_cache)
    test_paths = preprocess_to_cache(test_imgs, IMAGE_SIZE, test_cache)

    train_ds = CachedDataset(train_paths)
    test_ds = CachedDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(args.device == "cuda"),
                              persistent_workers=(args.num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(args.device == "cuda"),
                             persistent_workers=(args.num_workers > 0))

    # 模型 + 优化器 | Model + optimizer
    model = MobileNetSpatialRouter(pretrained=True).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # 训练循环 | Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, args.device)
        sch.step()
        if epoch % 5 == 0 or epoch == 1:
            logger.log_info("b02_5/expB", f"E{epoch}/{args.epochs} loss={loss:.4f}")

    # ── 三类评估: B1=全部, B2=仅 holdout 类, B3=仅训练类(对照) ──
    # ── Three-way evaluation: B1=all tiles, B2=holdout only, B3=train classes (control) ──
    # B1: 全部 tile (与 B-02 相同) | All tiles (same as B-02)
    r_all = evaluate_model(model, test_loader, args.device, class_filter=None)
    print_results(r_all, "Exp B1: all tiles", "b02_5/expB")

    # B2: 仅 hold-out 类 tile — 关键指标: 如果 r 仍高 → 类别无关 | Only holdout-class tiles — key: if r still high → category-agnostic
    r_holdout = evaluate_model(model, test_loader, args.device,
                               class_filter=holdout_classes)
    print_results(r_holdout, f"Exp B2: holdout classes {holdout_classes}", "b02_5/expB")

    # B3: 仅训练类 tile (作为对照) — 期望 r 与 B-02 基线一致 | Only train-class tiles (control) — expect r similar to B-02 baseline
    r_train_cls = evaluate_model(model, test_loader, args.device,
                                 class_filter=train_classes)
    print_results(r_train_cls, "Exp B3: train classes (control)", "b02_5/expB")

    return {
        "all": r_all, "holdout": r_holdout, "train_cls": r_train_cls,
        "holdout_classes": holdout_classes, "train_classes": train_classes,
    }, model


# ═══════════════════════════════════════════════════════════════════
# Experiment C: 跨数据集 | Cross-dataset transfer
# ═══════════════════════════════════════════════════════════════════

def load_cityscapes_images(cityscapes_root: str) -> list:
    """
    加载 Cityscapes tile 图片列表 | Load Cityscapes tile image list.
    Cityscapes tile 格式: images/train/*.png, masks/train/*.png.
    从 mask 计算 fg_ratio (0=bg, 1-18=前景, 255=ignore).
    """
    root = Path(cityscapes_root)
    # Cityscapes tile 目录结构 | Cityscapes tile directory structure
    img_dir = root / "images" / "train"
    mask_dir = root / "masks" / "train"

    if not img_dir.exists():
        logger.log_info("b02_5/expC",
                        f"Cityscapes not found at {cityscapes_root}. "
                        f"Run prep_cityscapes.py first.")
        return []

    # 匹配图像和 mask 文件 | Match image and mask files
    images = []
    for png in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / png.name
        if mask_path.exists():
            images.append((png.name, str(png), [], IMAGE_SIZE))
    return images


def _preprocess_cityscapes_worker(args_tuple):
    """Cityscapes 预处理 worker | Cityscapes preprocessing worker."""
    img_id, img_path, _, image_size, cache_dir = args_tuple
    try:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # 读 Cityscapes mask (trainId: 0=bg, 1-18=class, 255=ignore)
        # Read Cityscapes mask
        mask_path = img_path.replace("images", "masks")
        mask = np.array(Image.open(mask_path))

        # 255→0: 将 ignore 标签视为背景 (fg_ratio 计算用) | Treat ignore label as background for fg_ratio
        mask_clean = np.where(mask == 255, 0, mask)

        # Resize + Pad | Resize + pad
        scale = image_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        img = np.array(Image.fromarray(img).resize((new_W, new_H), Image.BILINEAR))
        mask_clean = np.array(Image.fromarray(mask_clean.astype(np.uint8))
                              .resize((new_W, new_H), Image.NEAREST))

        pad_h, pad_w = image_size - new_H, image_size - new_W
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask_clean = np.pad(mask_clean, ((0, pad_h), (0, pad_w)), mode="constant")

        # 归一化 + CHW 转换 | Normalize + CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        # 计算 tile fg_ratio (Cityscapes: fg = classes 1-18) | Compute tile fg_ratio
        H2, W2 = mask_clean.shape
        n_ty = (H2 + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W2 + TILE_SIZE - 1) // TILE_SIZE
        tile_scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        fg_pixels = np.zeros(n_ty * n_tx, dtype=np.int64)
        class_fg = np.zeros((n_ty * n_tx, 20), dtype=np.int64)  # 0-19 classes
        idx = 0
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H2)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W2)
                tile_mask = mask_clean[y0:y1, x0:x1]
                total_px = (y1 - y0) * (x1 - x0)
                fg_px = int((tile_mask > 0).sum())
                tile_scores[ty, tx] = fg_px / max(total_px, 1)
                fg_pixels[idx] = fg_px
                idx += 1

        out_path = os.path.join(cache_dir, f"{img_id}.npz")
        np.savez_compressed(out_path, image=img, tile_scores=tile_scores,
                            fg_pixels=fg_pixels, class_fg=class_fg,
                            n_ty=n_ty, n_tx=n_tx)
        return out_path
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# LoveDA 数据处理 | LoveDA data handling
# ═══════════════════════════════════════════════════════════════════

def load_loveda_images(loveda_root: str) -> list:
    """
    加载 LoveDA 训练集 (带 mask) | Load LoveDA training set (with masks).
    LoveDA 结构: {root}/Train/Train/Rural/images_png/ + masks_png/
    Mask 值: 0=bg, 1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agriculture
    """
    root = Path(loveda_root)
    # LoveDA 目录结构 (Rural 场景) | LoveDA directory structure (Rural scene)
    img_dir = root / "Train" / "Train" / "Rural" / "images_png"
    mask_dir = root / "Train" / "Train" / "Rural" / "masks_png"

    if not img_dir.exists() or not mask_dir.exists():
        logger.log_info("b02_5/loveda",
                        f"LoveDA not found at {loveda_root}. "
                        f"Expected: {img_dir} and {mask_dir}")
        return []

    # 匹配图像和 mask 文件 | Match image and mask files
    images = []
    for png in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / png.name
        if mask_path.exists():
            images.append((png.stem, str(png), [], IMAGE_SIZE))
    return images


def _preprocess_loveda_worker(args_tuple):
    """
    LoveDA 预处理 worker | LoveDA preprocessing worker.
    LoveDA 已经是 1024×1024, mask 是 uint8 class IDs.
    LoveDA native images are 1024×1024, mask is uint8 class IDs.
    """
    img_id, img_path, _, image_size, cache_dir = args_tuple

    try:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # 读 LoveDA mask (uint8 class IDs: 0=bg, 1-7=classes)
        # Read LoveDA mask
        mask_path = img_path.replace("images_png", "masks_png")
        mask = np.array(Image.open(mask_path))

        # Resize 到固定尺寸 | Resize to fixed size
        scale = image_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        img = np.array(Image.fromarray(img).resize((new_W, new_H), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((new_W, new_H), Image.NEAREST))

        # Pad 到方形 | Pad to square
        pad_h, pad_w = image_size - new_H, image_size - new_W
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        # 归一化 + CHW | Normalize + CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        # 计算 tile fg_ratio (LoveDA: fg = classes 1-7) | Compute tile fg_ratio
        H2, W2 = mask.shape
        n_ty = (H2 + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W2 + TILE_SIZE - 1) // TILE_SIZE
        tile_scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        fg_pixels = np.zeros(n_ty * n_tx, dtype=np.int64)
        class_fg = np.zeros((n_ty * n_tx, 8), dtype=np.int64)  # LoveDA: 0-7 classes
        idx = 0
        # 遍历所有 tile | Iterate all tiles
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H2)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W2)
                tile_mask = mask[y0:y1, x0:x1]
                total_px = (y1 - y0) * (x1 - x0)
                fg_px = int((tile_mask > 0).sum())
                tile_scores[ty, tx] = fg_px / max(total_px, 1)
                fg_pixels[idx] = fg_px
                idx += 1

        out_path = os.path.join(cache_dir, f"{img_id}.npz")
        np.savez_compressed(out_path, image=img, tile_scores=tile_scores,
                            fg_pixels=fg_pixels, class_fg=class_fg,
                            n_ty=n_ty, n_tx=n_tx)
        return out_path
    except Exception:
        return None


def run_exp_c(args, isaid_all_images, cache_dir):
    """
    iSAID → 跨数据集 (Cityscapes 或 LoveDA) | iSAID → cross-dataset (Cityscapes or LoveDA).
    训练: iSAID 航拍; 测试: Cityscapes/LoveDA (不同域/类别/分辨率).
    Train: iSAID aerial; Test: Cityscapes/LoveDA (different domain/classes/resolution).
    """
    test_dataset = args.test_dataset

    logger.log_info("b02_5/expC", "=" * 50)
    logger.log_info("b02_5/expC",
                    f"Experiment C: iSAID → {test_dataset} (cross-dataset)")

    # ── 加载目标数据集测试图像 | Load target dataset test images ──
    if test_dataset == "cityscapes":
        target_images = load_cityscapes_images(args.cityscapes_root)
        preprocess_fn = _preprocess_cityscapes_worker
    else:  # loveda
        target_images = load_loveda_images(args.loveda_root)
        preprocess_fn = _preprocess_loveda_worker

    if not target_images:
        logger.log_info("b02_5/expC",
                        f"{test_dataset} not available, skipping")
        return None, None

    logger.log_info("b02_5/expC",
                    f"{test_dataset} test: {len(target_images)} images")

    # ── 训练: iSAID (与 Exp A/B 相同) | Train: iSAID (same as Exp A/B) ──
    np.random.seed(args.seed)
    perm = np.random.permutation(len(isaid_all_images))
    n_train = min(args.train_images, len(isaid_all_images) - 20)
    train_imgs = [isaid_all_images[i] for i in perm[:n_train]]

    # 预处理 iSAID 训练数据 | Preprocess iSAID training data
    train_cache = os.path.join(cache_dir, "C_train_isaid")
    train_paths = preprocess_to_cache(train_imgs, IMAGE_SIZE, train_cache)
    train_ds = CachedDataset(train_paths)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(args.device == "cuda"),
                              persistent_workers=(args.num_workers > 0))

    # ── 预处理目标测试集 (使用数据集专属的 worker) | Preprocess target test set (dataset-specific worker) ──
    test_cache = os.path.join(cache_dir, f"C_test_{test_dataset}")
    os.makedirs(test_cache, exist_ok=True)
    tasks = [(img_id, img_path, anns, IMAGE_SIZE, test_cache)
             for img_id, img_path, anns, _ in target_images]
    from concurrent.futures import ProcessPoolExecutor
    n_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    test_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(preprocess_fn, tasks),
                       total=len(tasks),
                       desc=f"  {test_dataset} preprocess", unit="img"):
            if r is not None:
                test_paths.append(r)

    logger.log_info("b02_5/expC",
                    f"{test_dataset} cached: {len(test_paths)}/{len(target_images)}")

    test_ds = CachedDataset(test_paths)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(args.device == "cuda"),
                             persistent_workers=(args.num_workers > 0))

    # ── 训练 (仅在 iSAID 上) + 跨域测试 | Train (iSAID only) + cross-domain test ──
    model = MobileNetSpatialRouter(pretrained=True).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # 训练循环: 只用 iSAID 训练 | Training loop: iSAID training only
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, args.device)
        sch.step()
        if epoch % 5 == 0 or epoch == 1:
            logger.log_info("b02_5/expC", f"E{epoch}/{args.epochs} loss={loss:.4f}")

    # 在目标数据集上测试 (零样本迁移) | Test on target dataset (zero-shot transfer)
    results = evaluate_model(model, test_loader, args.device, class_filter=None)
    print_results(results, f"Exp C: iSAID → {test_dataset}", "b02_5/expC")
    return results, model


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = args.cache_dir or str(output_dir / "cache")

    # ── 加载 iSAID COCO 标注 | Load iSAID COCO annotations ──
    src_root = Path(args.src_root)
    ann_file = src_root / "train" / "annotations" / "instances_train.json"
    if not ann_file.exists():
        logger.log_info("b02_5/error", f"iSAID annotations not found: {ann_file}")
        sys.exit(1)

    with open(ann_file) as f:
        coco = json.load(f)

    # 建立 image_id → annotations 索引 | Build image_id → annotations index
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_dir = src_root / "train" / "images"
    all_images = []
    for img_info in coco["images"]:
        anns = img_id_to_anns.get(img_info["id"], [])
        if not anns:
            continue
        img_path = str(img_dir / img_info["file_name"])
        if Path(img_path).exists():
            all_images.append((img_info["file_name"], img_path, anns, IMAGE_SIZE))

    logger.log_info("b02_5/data", f"iSAID images with annotations: {len(all_images)}")

    # ── 运行实验: A=同分布, B=类别迁移, C=跨数据集 | Run experiments: A=same-dist, B=class gen, C=cross-dataset ──
    all_results = {}

    if args.exp in ("A", "ALL"):
        r_a, _ = run_exp_a(args, all_images, os.path.join(cache_root, "A"))
        all_results["A_same_dist"] = r_a

    if args.exp in ("B", "ALL"):
        r_b, _ = run_exp_b(args, all_images, os.path.join(cache_root, "B"))
        all_results["B_class_gen"] = r_b

    if args.exp in ("C", "ALL"):
        r_c, _ = run_exp_c(args, all_images, os.path.join(cache_root, "C"))
        if r_c is not None:
            all_results["C_cross_dataset"] = r_c

    # ── 汇总所有实验结果 | Summarize all experiment results ──
    logger.log_info("b02_5/summary", "=" * 50)
    logger.log_info("b02_5/summary", "B-02.5 Generalization Study — Summary")
    logger.log_info("b02_5/summary", "=" * 50)

    conclusion_lines = []
    for name, r in all_results.items():
        if r is None:
            continue
        if isinstance(r, dict) and "spearman_r" in r:
            # Exp A / Exp C 的简单结构 | Simple structure for Exp A / Exp C
            sr = r["spearman_r"]
            gap40 = (r["oracle_retention"][40] - r["learned_retention"][40]) * 100
            logger.log_info("b02_5/summary",
                            f"  {name}: Spearman r={sr:.4f}  "
                            f"Top40 Gap={gap40:.2f}%  N={r['n_tiles']:,}")
            conclusion_lines.append((name, sr, gap40))
        elif isinstance(r, dict) and "holdout" in r:
            # Exp B 的特殊结构: all/holdout/train_cls 三个子结果 | Exp B nested: three sub-results
            for sub_name in ["all", "holdout", "train_cls"]:
                if r.get(sub_name):
                    sr = r[sub_name]["spearman_r"]
                    gap40 = (r[sub_name]["oracle_retention"][40] -
                             r[sub_name]["learned_retention"][40]) * 100
                    logger.log_info("b02_5/summary",
                                    f"  {name}/{sub_name}: Spearman r={sr:.4f}  "
                                    f"Top40 Gap={gap40:.2f}%")

    # ── 保存 | Save ──
    def convert_for_json(obj):
        """递归转换 numpy 类型 | Recursively convert numpy types."""
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        "experiment": "B-02.5 Generalization Study",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "results": convert_for_json(all_results),
    }
    with open(output_dir / "generalization_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b02_5/output", f"Saved: {output_dir}/")


if __name__ == "__main__":
    main()
