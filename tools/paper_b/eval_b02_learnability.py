#!/usr/bin/env python3
"""
B-02: Learnability Study — 模型能否学会预测 Tile 重要性？
===========================================================

B-00 证明 Spatial Sparsity 存在。B-01 证明 Oracle 能做到 Top40%→96.5% FG。
B-02 回答核心问题：模型能学到吗？

实验设计 | Design:
    骨干: MobileNetV3-Small
    Head:  Conv → Importance Map → Tile Scores
    训练:  固定尺寸 + batch + DataLoader 多线程 + 磁盘缓存
    GT:   每个 tile 的 fg_ratio
    Loss:  MSE(pred_score, gt_fg_ratio)

用法 | Usage:
    python tools/eval_b02_learnability.py
    python tools/eval_b02_learnability.py --train-images 500 --epochs 30 --batch-size 8
"""

import sys, argparse, json, datetime, os, tempfile, shutil
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

logger = get_logger("b02_learnability")
logger.add_backend(ConsoleBackend())  # 终端实时输出 | Real-time console output

TILE_SIZE = 1024
BACKBONE_STRIDE = 32
TILE_FEAT = TILE_SIZE // BACKBONE_STRIDE  # 32 feature pixels per tile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--train-images", type=int, default=300)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers. 0=no multiprocessing")
    p.add_argument("--image-size", type=int, default=2048,
                   help="固定 resize 尺寸 | Fixed square resize")
    p.add_argument("--cache-dir", type=str, default="",
                   help="预处理缓存目录 | Preprocess cache dir")
    p.add_argument("--output-dir", type=str, default="runs/b02_learnability")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """渲染语义掩码 [H,W] uint8 (0-15) | Render semantic mask.
    直接使用 ann["category_id"]（映射已在预处理中完成 | mapping done in preprocessing）."""
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
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
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


# ═══════════════════════════════════════════════════════════════════
# 预处理 + 磁盘缓存 | Preprocess + disk cache
# ═══════════════════════════════════════════════════════════════════

def _preprocess_worker(args_tuple):
    """独立进程: 渲染 mask → resize → 保存到缓存目录."""
    img_id, img_path, anns, image_size, cache_dir = args_tuple

    try:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        mask = render_semantic_mask(anns, H, W)

        # Resize
        scale = image_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        img = np.array(Image.fromarray(img).resize((new_W, new_H), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((new_W, new_H), Image.NEAREST))

        # Pad to square
        pad_h = image_size - new_H
        pad_w = image_size - new_W
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        # 转为 float32 [0,1] → [3, H, W] | Convert to float32 → CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # [3, S, S]

        # GT tile scores + per-tile fg pixels | Compute GT
        H2, W2 = mask.shape
        n_ty = (H2 + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W2 + TILE_SIZE - 1) // TILE_SIZE
        tile_scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        fg_pixels = np.zeros(n_ty * n_tx, dtype=np.int64)
        idx = 0
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

        # 保存为 .npz (压缩) | Save as .npz (compressed)
        out_path = os.path.join(cache_dir, f"{img_id}.npz")
        np.savez_compressed(out_path, image=img, tile_scores=tile_scores,
                            fg_pixels=fg_pixels)
        return out_path

    except Exception:
        return None


def preprocess_dataset(images: list, image_size: int, cache_dir: str) -> list[str]:
    """多进程预处理 → 返回 .npz 文件路径列表."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)

    tasks = [(img_id, img_path, anns, image_size, cache_dir)
             for img_id, img_path, anns, _ in images]

    n_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    logger.log_info("b02/preprocess",
                    f"Preprocessing {len(tasks)} images ({n_workers} workers)...")

    paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(_preprocess_worker, tasks),
                       total=len(tasks), desc="  Preprocess", unit="img"):
            if r is not None:
                paths.append(r)

    logger.log_info("b02/preprocess",
                    f"→ {len(paths)}/{len(tasks)} cached to {cache_dir}")
    return paths


class CachedSpatialDataset(Dataset):
    """
    从磁盘 .npz 文件按需加载，避免 pickle 大量数据.
    Loads from .npz files on-demand, avoids pickling large arrays.
    """

    def __init__(self, npz_paths: list[str]):
        self.paths = npz_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        img = torch.from_numpy(data["image"])           # [3, S, S] float32
        scores = torch.from_numpy(data["tile_scores"])  # [n_ty, n_tx] float32
        fg = torch.from_numpy(data["fg_pixels"])        # [n_tiles] int64
        return img, scores, fg


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class MobileNetSpatialRouter(nn.Module):
    """MobileNetV3-Small → Importance Map [B,1,H/32,W/32]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        mnet = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.backbone = mnet.features  # stride=32, 576 channels

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
        return self.head(self.backbone(x))  # [B, 576, H/32, W/32] → [B, 1, H/32, W/32]


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, device):
    """训练一个 epoch (批处理)."""
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, gt_scores, fg_px in pbar:
        imgs = imgs.to(device)  # [B, 3, S, S]
        B = imgs.shape[0]

        imp_maps = model(imgs)  # [B, 1, S/32, S/32]
        _, _, hp, wp = imp_maps.shape

        batch_loss = 0.0
        for b in range(B):
            gt = gt_scores[b].to(device)
            n_ty, n_tx = gt.shape

            pred_list, gt_list = [], []
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty * TILE_FEAT, min(ty * TILE_FEAT + TILE_FEAT, hp)
                    x0, x1 = tx * TILE_FEAT, min(tx * TILE_FEAT + TILE_FEAT, wp)
                    if y1 > y0 and x1 > x0:
                        pred_list.append(imp_maps[b, 0, y0:y1, x0:x1].mean())
                        gt_list.append(gt[ty, tx])

            if pred_list:
                batch_loss += F.mse_loss(torch.stack(pred_list), torch.stack(gt_list))

        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        total_loss += batch_loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """收集所有 tile pred/gt → Oracle vs Learned."""
    model.eval()
    all_pred, all_gt, all_fg = [], [], []

    for imgs, gt_scores, fg_px in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        imp_maps = model(imgs)
        _, _, hp, wp = imp_maps.shape

        for b in range(B):
            gt = gt_scores[b].numpy()
            fg = fg_px[b].numpy()
            n_ty, n_tx = gt.shape
            idx = 0
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty * TILE_FEAT, min(ty * TILE_FEAT + TILE_FEAT, hp)
                    x0, x1 = tx * TILE_FEAT, min(tx * TILE_FEAT + TILE_FEAT, wp)
                    if y1 > y0 and x1 > x0:
                        all_pred.append(imp_maps[b, 0, y0:y1, x0:x1].mean().item())
                        all_gt.append(float(gt[ty, tx]))
                        all_fg.append(int(fg[idx]))
                        idx += 1

    pred_all = np.array(all_pred)
    gt_all = np.array(all_gt)
    fg_all = np.array(all_fg)

    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pred_all, gt_all)
    sr, _ = spearmanr(pred_all, gt_all)

    oracle_ord = np.argsort(gt_all)[::-1]
    learned_ord = np.argsort(pred_all)[::-1]

    oracle_r, learned_r = {}, {}
    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]:
        n = max(1, int(len(gt_all) * k / 100))
        oracle_r[k] = float(fg_all[oracle_ord[:n]].sum() / max(fg_all.sum(), 1))
        learned_r[k] = float(fg_all[learned_ord[:n]].sum() / max(fg_all.sum(), 1))

    return {
        "pearson_r": float(pr), "spearman_r": float(sr),
        "oracle_retention": oracle_r, "learned_retention": learned_r,
        "n_tiles": len(gt_all),
    }


def main():
    args = parse_args()
    device = args.device
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 缓存目录 | Cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = str(output_dir / "cache")

    logger.log_info("exp/start",
                    f"B-02 | MobileNetV3-Small | train={args.train_images} "
                    f"epochs={args.epochs} batch={args.batch_size} "
                    f"size={args.image_size} device={device}")

    # ── 加载 COCO ──
    src_root = Path(args.src_root)
    ann_file = src_root / "train" / "annotations" / "instances_train.json"
    with open(ann_file) as f:
        coco = json.load(f)

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
            all_images.append((img_info["file_name"], img_path, anns, args.image_size))

    logger.log_info("data", f"Total: {len(all_images)} images with annotations")

    # ── Train/Eval split ──
    np.random.seed(args.seed)
    perm = np.random.permutation(len(all_images))
    n_train = min(args.train_images, len(all_images) - 30)
    train_imgs = [all_images[i] for i in perm[:n_train]]
    eval_imgs = [all_images[i] for i in perm[n_train:min(n_train + 50, len(all_images))]]

    logger.log_info("data/split", f"Train={len(train_imgs)}, Eval={len(eval_imgs)}")

    # ── 预处理 + 缓存 | Preprocess + cache ──
    logger.log_info("b02/config",
                    f"B-02: Spatial Router Learnability | "
                    f"Backbone: MobileNetV3-Small | Image size: {args.image_size}")
    logger.log_info("b02/config",
                    f"Train: {len(train_imgs)} | Eval: {len(eval_imgs)} | "
                    f"Batch: {args.batch_size} | Workers: {args.num_workers} | "
                    f"Cache: {cache_dir}")

    train_cache = os.path.join(cache_dir, "train")
    eval_cache = os.path.join(cache_dir, "eval")
    train_paths = preprocess_dataset(train_imgs, args.image_size, train_cache)
    eval_paths = preprocess_dataset(eval_imgs, args.image_size, eval_cache)

    train_ds = CachedSpatialDataset(train_paths)
    eval_ds = CachedSpatialDataset(eval_paths)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=False)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=False)

    # ── 模型 (日志) | Model (logged) ──
    model = MobileNetSpatialRouter(pretrained=True).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_info("b02/model", f"Total={n_total:,} params, Trainable={n_trainable:,}")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    # ── 训练 (日志) | Training (logged) ──
    logger.log_info("b02/train", f"Training {args.epochs} epochs...")
    all_losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, opt, device)
        sch.step()
        all_losses.append(avg_loss)
        logger.log_metric("b02/train_loss", avg_loss, step=epoch, tags=["b02"])
        if epoch % 5 == 0 or epoch == 1:
            logger.log_info("b02/train",
                            f"E{epoch:2d}/{args.epochs}  loss={avg_loss:.4f}")

    # ── 评估 (日志) | Evaluation (logged) ──
    logger.log_info("b02/eval", "Evaluating on hold-out set...")
    results = evaluate_model(model, eval_loader, device)

    # ── 结果 (日志) | Results (logged) ──
    logger.log_info("b02/results",
                    f"Pearson r={results['pearson_r']:.4f}  "
                    f"Spearman r={results['spearman_r']:.4f}  "
                    f"N_tiles={results['n_tiles']:,}")
    logger.log_info("b02/results",
                    f"  {'K%':<7} {'Oracle':>9} {'Learned':>9} {'Gap':>7}")
    for k in [20, 30, 40, 50]:
        o, l = results["oracle_retention"][k]*100, results["learned_retention"][k]*100
        logger.log_info("b02/results",
                        f"  {k:>5}%  {o:>8.2f}%  {l:>8.2f}%  {o-l:>6.2f}%")

    oracle_idg = {k: results["oracle_retention"][k]/(k/100) for k in [20, 30, 40, 50]}
    learned_idg = {k: results["learned_retention"][k]/(k/100) for k in [20, 30, 40, 50]}
    logger.log_info("b02/idg",
                    f"Oracle Top40 IDG={oracle_idg[40]:.2f}x  "
                    f"Learned Top40 IDG={learned_idg[40]:.2f}x")

    sr = results["spearman_r"]
    verdict = ("LEARNABLE" if sr > 0.6 else
               "PARTIALLY LEARNABLE" if sr > 0.3 else "HARD")
    logger.log_info("b02/verdict",
                    f"VERDICT: {verdict}  (Spearman r={sr:.4f})")

    # ── Save & Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    ks = sorted(results["oracle_retention"].keys())
    ofg = [results["oracle_retention"][k]*100 for k in ks]
    lfg = [results["learned_retention"][k]*100 for k in ks]

    ax = axes[0, 0]
    ax.plot(ks, ofg, "o-", color="#E74C3C", lw=2.5, ms=7, label="Oracle")
    ax.plot(ks, lfg, "s-", color="#3498DB", lw=2.5, ms=7,
            label=f"Learned (r={sr:.3f})")
    ax.fill_between(ks, lfg, ofg, alpha=0.08, color="gray")
    ax.axvline(30, color="gray", ls="--", alpha=0.3)
    ax.axvline(40, color="gray", ls="--", alpha=0.3)
    ax.set(xlabel="Tiles Kept (%)", ylabel="FG Retained (%)",
           title="Oracle vs Learned FG Retention", xlim=(0, 105), ylim=(0, 105))
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(range(1, len(all_losses)+1), all_losses, "o-", color="#27AE60", lw=2)
    ax.set(xlabel="Epoch", ylabel="MSE", title="Training Loss"); ax.grid(alpha=0.2)

    ax = axes[0, 2]
    gap = [o-l for o, l in zip(ofg, lfg)]
    ax.bar(ks, gap, width=3, color="#E67E22", edgecolor="white", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Tiles Kept (%)", ylabel="Oracle - Learned Gap (%)",
           title=f"Gap (mean={np.mean(gap):.2f}%)"); ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    xp = np.arange(4); w = 0.3
    for i, k in enumerate([20, 30, 40, 50]):
        ax.bar(i-w/2, oracle_idg[k], w, color="#E74C3C", ec="white",
               alpha=0.85, label="Oracle" if i==0 else "")
        ax.bar(i+w/2, learned_idg[k], w, color="#3498DB", ec="white",
               alpha=0.85, label="Learned" if i==0 else "")
    ax.set_xticks(xp); ax.set_xticklabels(["Top20%","Top30%","Top40%","Top50%"])
    ax.set(ylabel="IDG", title=f"IDG: Oracle={oracle_idg[40]:.2f}x "
           f"vs Learned={learned_idg[40]:.2f}x")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    ax.text(0.5, 0.5, f"Pearson r = {results['pearson_r']:.4f}\n"
            f"Spearman r = {results['spearman_r']:.4f}\n\n"
            f"Pred vs GT tile score\n"
            f"({results['n_tiles']:,} eval tiles)",
            transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.axis("off")

    ax = axes[1, 2]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    lines = [
        "B-02 Learnability", "="*30, "",
        f"Backbone: MobileNetV3-Small",
        f"Params: {n_trainable:,} trainable",
        f"Train: {len(train_ds)} imgs × {args.epochs} ep",
        f"Image size: {args.image_size} × batch={args.batch_size}",
        "",
        f"Pearson r  = {results['pearson_r']:.4f}",
        f"Spearman r = {sr:.4f}",
        f"Oracle Top40: {results['oracle_retention'][40]*100:.1f}% FG",
        f"Learned Top40: {results['learned_retention'][40]*100:.1f}% FG",
        "", f"VERDICT: {verdict}",
    ]
    for i, line in enumerate(lines):
        y = 9.5 - i*0.42
        c = ("#27AE60" if ("VERDICT" in line and "LEARNABLE" in line) else
             "#E74C3C" if "VERDICT" in line else None)
        kw = ({"fontsize": 13, "fontweight": "bold"} if line.startswith("B-02") else
              {"fontsize": 9, "color": "gray"} if line.startswith("=") else
              {"fontsize": 10, "fontweight": "bold", "color": c} if c else
              {"fontsize": 9})
        ax.text(0.5, y, line, fontfamily="monospace", va="top", **kw)

    fig.suptitle("B-02: Can MobileNetV3 Learn Tile Importance?",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "learnability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "experiment": "B-02 Learnability Study",
        "backbone": "MobileNetV3-Small",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "results": {
            "pearson_r": results["pearson_r"],
            "spearman_r": results["spearman_r"],
            "oracle_retention": results["oracle_retention"],
            "learned_retention": results["learned_retention"],
            "oracle_idg": oracle_idg,
            "learned_idg": learned_idg,
        },
        "verdict": verdict,
    }
    with open(output_dir / "learnability_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b02/output", f"Results saved: {output_dir}/")


if __name__ == "__main__":
    main()
