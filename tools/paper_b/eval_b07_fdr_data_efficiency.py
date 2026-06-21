#!/usr/bin/env python3
"""
B-07: FDR Sample Efficiency — 路由器需要多少标注数据？
=========================================================

Paper B 核心原创模块是 FDR。B-02 已证明可学习 (Spearman r=0.889)，
B-02.5 已证明类别无关。B-07 回答第三个维度：数据效率。

实验 | Experiment:
    Data%: 100 / 50 / 20 / 10 / 5 / 1
    Seeds: 3 per fraction (消除抽样方差)
    Eval:  固定 100 张全图

指标 | Metrics:
    1. Spearman r          — 排序质量
    2. FG Recall@40%       — 连接实际: Top40% Pred tiles 捕获的前景比例
    3. FDR-SES             — r(5%) / r(100%) 数据效率评分

用法 | Usage:
    python tools/paper_b/eval_b07_fdr_data_efficiency.py \
        --src-root data/iSAID_processed \
        --data-fractions 100,50,20,10,5,1 \
        --epochs 20 --seeds 3 \
        --output-dir runs/b07_fdr_efficiency
"""

import sys, argparse, json, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.sparse.spatial_router import DensityHead

TILE_SIZE = 1024
IMAGE_SIZE = 2048
MV3_STRIDE = 32


# ═══════════════════════════════════════════════════
# FDR Model
# ═══════════════════════════════════════════════════

class FDRModel(nn.Module):
    """Frozen MV3-Small + DensityHead (75K trainable)."""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.density_head = DensityHead(576, 128)

    def forward(self, x):
        return self.density_head(self.backbone(x))


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def render_semantic_mask(annotations, h, w):
    """Render semantic mask [H,W] uint8 from COCO annotations."""
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0: continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            sem[max(0, int(bbox[1])):min(h, int(bbox[1]+bbox[3])),
                max(0, int(bbox[0])):min(w, int(bbox[0]+bbox[2]))] = cat_id
            continue
        if isinstance(seg, dict): continue
        for poly in (seg if isinstance(seg[0], list) else [seg]):
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w-1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h-1)
            cv2.fillPoly(sem, [pts], int(cat_id))
    return sem


def compute_gt_tile_scores(mask, tile_size=TILE_SIZE):
    """Compute GT fg_ratio per tile at native resolution."""
    H, W = mask.shape
    n_ty = (H + tile_size - 1) // tile_size
    n_tx = (W + tile_size - 1) // tile_size
    scores = np.zeros((n_ty, n_tx), dtype=np.float32)
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * tile_size, min(ty * tile_size + tile_size, H)
            x0, x1 = tx * tile_size, min(tx * tile_size + tile_size, W)
            tm = mask[y0:y1, x0:x1]
            total = (y1 - y0) * (x1 - x0)
            scores[ty, tx] = (tm > 0).sum() / max(total, 1)
    return scores


# ═══════════════════════════════════════════════════
# Dataset (pre-cached GT + on-the-fly image load)
# ═══════════════════════════════════════════════════

class CachedFDRDataset(Dataset):
    """Loads image from disk, GT scores from cache, resizes to IMAGE_SIZE."""

    def __init__(self, image_items, cache_dir, target_size=IMAGE_SIZE):
        from PIL import Image
        self.items = image_items
        self.cache_dir = cache_dir
        self.target_size = target_size
        self.stride = MV3_STRIDE

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        img_id, img_path, _anns = self.items[idx]

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # Resize
        scale = self.target_size / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        img_r = np.array(Image.fromarray(img).resize((nW, nH), Image.BILINEAR))

        # Pad to 32
        ph = (self.stride - nH % self.stride) % self.stride
        pw = (self.stride - nW % self.stride) % self.stride
        if ph > 0 or pw > 0:
            img_r = np.pad(img_r, ((0, ph), (0, pw), (0, 0)), mode="constant")

        img_t = torch.from_numpy(img_r.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1)

        cached = np.load(self.cache_dir / f"{img_id}.npz")
        gt_scores = torch.from_numpy(cached["gt_scores"])
        n_ty, n_tx = int(cached["n_ty"]), int(cached["n_tx"])

        return {
            "image": img_t,
            "gt_scores": gt_scores,
            "n_ty": n_ty,
            "n_tx": n_tx,
        }


# ═══════════════════════════════════════════════════
# Precompute GT cache
# ═══════════════════════════════════════════════════

def build_cache(image_items, cache_dir):
    """Precompute GT tile scores for all images (once)."""
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    for img_id, img_path, anns in tqdm(image_items, desc="  Cache GT"):
        cache_path = cache_dir / f"{img_id}.npz"
        if cache_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        mask = render_semantic_mask(anns, H, W)
        gt_scores = compute_gt_tile_scores(mask)
        np.savez_compressed(cache_path, gt_scores=gt_scores,
                            n_ty=gt_scores.shape[0], n_tx=gt_scores.shape[1])


# ═══════════════════════════════════════════════════
# FDR Training (single run)
# ═══════════════════════════════════════════════════

def train_fdr(model, train_loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            gt_scores = batch["gt_scores"].to(device)

            imp = model(images)
            _, _, Hf, Wf = imp.shape

            losses = []
            for i in range(images.size(0)):
                n_ty_i = int(batch["n_ty"][i])
                n_tx_i = int(batch["n_tx"][i])
                imp_i = imp[i, 0]

                preds = []
                for ty in range(n_ty_i):
                    for tx in range(n_tx_i):
                        y0 = int(ty * Hf / n_ty_i)
                        y1 = int((ty + 1) * Hf / n_ty_i)
                        x0 = int(tx * Wf / n_tx_i)
                        x1 = int((tx + 1) * Wf / n_tx_i)
                        y0, y1 = max(0, min(y0, Hf-1)), max(y0+1, min(y1, Hf))
                        x0, x1 = max(0, min(x0, Wf-1)), max(x0+1, min(x1, Wf))
                        preds.append(imp_i[y0:y1, x0:x1].mean())

                pred_t = torch.stack(preds).reshape(n_ty_i, n_tx_i)
                gt_i = gt_scores[i, :n_ty_i, :n_tx_i]
                losses.append(F.mse_loss(pred_t, gt_i))

            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss


# ═══════════════════════════════════════════════════
# Evaluation: Spearman r + FG Recall@K
# ═══════════════════════════════════════════════════

@torch.no_grad()
def evaluate_fdr(model, eval_loader, device):
    """Compute Spearman r and FG Recall@40%."""
    model.eval()
    all_preds, all_gts, all_fg = [], [], []

    for batch in eval_loader:
        images = batch["image"].to(device)
        gt_scores = batch["gt_scores"]

        imp = model(images)
        _, _, Hf, Wf = imp.shape

        for i in range(images.size(0)):
            n_ty_i = int(batch["n_ty"][i])
            n_tx_i = int(batch["n_tx"][i])
            imp_i = imp[i, 0].cpu().numpy()
            gt_i = gt_scores[i, :n_ty_i, :n_tx_i].numpy()

            preds = np.zeros((n_ty_i, n_tx_i), dtype=np.float32)
            for ty in range(n_ty_i):
                for tx in range(n_tx_i):
                    y0 = int(ty * Hf / n_ty_i)
                    y1 = int((ty+1) * Hf / n_ty_i)
                    x0 = int(tx * Wf / n_tx_i)
                    x1 = int((tx+1) * Wf / n_tx_i)
                    y0, y1 = max(0, min(y0, Hf-1)), max(y0+1, min(y1, Hf))
                    x0, x1 = max(0, min(x0, Wf-1)), max(x0+1, min(x1, Wf))
                    preds[ty, tx] = imp_i[y0:y1, x0:x1].mean()

            all_preds.extend(preds.flatten().tolist())
            all_gts.extend(gt_i.flatten().tolist())
            all_fg.extend(gt_i.flatten().tolist())  # fg_ratio as FG pixel proxy

    sr, _ = spearmanr(all_preds, all_gts)

    # FG Recall@40%: Top40% pred tiles → fraction of total FG captured
    sorted_idx = np.argsort(all_preds)[::-1]
    n_top40 = max(1, int(len(sorted_idx) * 0.4))
    fg_captured = np.array(all_fg)[sorted_idx[:n_top40]].sum()
    total_fg = np.array(all_fg).sum()
    fg_recall_40 = fg_captured / max(total_fg, 1e-8)

    return float(sr), float(fg_recall_40)


# ═══════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════

def plot_results(data_fractions, results, output_path):
    """Dual-panel: Spearman r + FG Recall@40% vs Data%."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    xs = np.arange(len(data_fractions))
    labels = [f"{d}%" for d in data_fractions]

    # Panel A: Spearman r
    ax = axes[0]
    means = [results[d]["sr_mean"] for d in data_fractions]
    stds = [results[d]["sr_std"] for d in data_fractions]
    ax.errorbar(xs, means, yerr=stds, marker="o", color="#3498DB",
                linewidth=2.5, markersize=10, capsize=5, label="Spearman r")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Spearman r", fontsize=12)
    ax.set_xlabel("Training Data (%)", fontsize=12)
    ax.set_title("FDR Ranking Quality vs Data\nFDR 排序质量 vs 训练数据量", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    for i, m in enumerate(means):
        ax.annotate(f"{m:.3f}", (xs[i], m + stds[i] + 0.02),
                    ha="center", fontsize=9, color="#3498DB")

    # Panel B: FG Recall@40%
    ax = axes[1]
    means_fg = [results[d]["fg_recall_mean"] * 100 for d in data_fractions]
    stds_fg = [results[d]["fg_recall_std"] * 100 for d in data_fractions]
    ax.errorbar(xs, means_fg, yerr=stds_fg, marker="s", color="#E67E22",
                linewidth=2.5, markersize=10, capsize=5, label="FG Recall@40%")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("FG Recall@40% (%)", fontsize=12)
    ax.set_xlabel("Training Data (%)", fontsize=12)
    ax.set_title("Foreground Coverage vs Data\nTop40% Pred Tiles — 前景保留率", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    for i, m in enumerate(means_fg):
        ax.annotate(f"{m:.1f}%", (xs[i], m + stds_fg[i] + 1),
                    ha="center", fontsize=9, color="#E67E22")

    fig.suptitle("B-07: FDR Sample Efficiency — How Much Data Does the Router Need?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "fdr_data_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--data-fractions", type=str, default="100,50,20,10,5,1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="runs/b07_fdr_efficiency")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    data_fractions = [int(x.strip()) for x in args.data_fractions.split(",")]
    base_seed = 42

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b07")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b07.jsonl")))

    logger.log_info("b07/start",
                    f"B-07 FDR Sample Efficiency: {data_fractions} data% × {args.seeds} seeds")

    # ── 1. Load Data | 加载数据 ──
    src_root = Path(args.src_root)
    with open(src_root / "train" / "annotations" / "instances_train.json") as f:
        coco = json.load(f)

    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)

    img_dir = src_root / "train" / "images"
    all_images = []
    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if img_path.exists():
            anns = img_id_to_anns.get(img_info["id"], [])
            if anns:
                all_images.append((img_info["file_name"], str(img_path), anns))

    logger.log_info("b07/data", f"Total annotated images: {len(all_images)}")

    # Shuffle + split: first 200 train pool, next 100 fixed eval
    rng = np.random.RandomState(base_seed)
    rng.shuffle(all_images)
    train_pool = all_images[:200]
    eval_images = all_images[200:300]
    logger.log_info("b07/data", f"Train pool: {len(train_pool)}, Eval: {len(eval_images)}")

    # ── 2. Build GT Cache | 构建缓存 ──
    cache_dir = output_dir / "cache"
    build_cache(train_pool + eval_images, cache_dir)

    # ── 3. Per data% × seed: train + eval | 逐比例训练评估 ──
    all_results = {}

    for data_pct in data_fractions:
        sr_list, fg_list = [], []
        n_train = max(1, int(len(train_pool) * data_pct / 100))

        for seed_idx in range(args.seeds):
            seed = base_seed + seed_idx
            set_seed(seed)

            # 采样训练集 | Sample training set
            samp_rng = np.random.RandomState(seed)
            idxs = samp_rng.choice(len(train_pool), n_train, replace=False)
            train_items = [train_pool[i] for i in idxs]

            # DataLoader
            train_ds = CachedFDRDataset(train_items, cache_dir)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=False)
            eval_ds = CachedFDRDataset(eval_images, cache_dir)
            eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=min(2, args.num_workers),
                                     pin_memory=True)

            # 训练 | Train
            model = FDRModel().to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model, best_loss = train_fdr(model, train_loader, device, args.epochs, args.lr)

            # 评估 | Evaluate
            sr, fg40 = evaluate_fdr(model, eval_loader, device)
            sr_list.append(sr)
            fg_list.append(fg40)

            logger.log_info("b07/run",
                            f"Data={data_pct:>3}% seed={seed_idx} "
                            f"n_train={n_train} loss={best_loss:.5f} "
                            f"Spearman={sr:.4f} FG@40={fg40*100:.1f}%")

        all_results[data_pct] = {
            "sr_mean": float(np.mean(sr_list)),
            "sr_std": float(np.std(sr_list)),
            "fg_recall_mean": float(np.mean(fg_list)),
            "fg_recall_std": float(np.std(fg_list)),
            "n_train_images": n_train,
            "seeds": sr_list,
        }
        logger.log_info("b07/agg",
                        f"Data={data_pct:>3}% → Spearman={all_results[data_pct]['sr_mean']:.4f}±"
                        f"{all_results[data_pct]['sr_std']:.4f}, "
                        f"FG@40={all_results[data_pct]['fg_recall_mean']*100:.1f}%")

    # ── 4. FDR-SES | 数据效率评分 ──
    sr_100 = all_results[100]["sr_mean"]
    sr_5 = all_results[5]["sr_mean"]
    fdr_ses = sr_5 / max(sr_100, 1e-8)

    logger.log_info("b07/ses",
                    f"FDR-SES = r(5%)/r(100%) = {sr_5:.4f}/{sr_100:.4f} = {fdr_ses:.3f}")
    logger.log_info("b07/ses",
                    f"→ 5% data retains {fdr_ses*100:.1f}% of full-data ranking quality")

    # ── 5. Plot | 绘图 ──
    plot_results(data_fractions, all_results, output_dir)

    # ── 6. Summary | 汇总保存 ──
    summary = {
        "experiment": "B-07 FDR Sample Efficiency",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "data_fractions": data_fractions,
            "seeds_per_fraction": args.seeds,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_pool_size": len(train_pool),
            "eval_set_size": len(eval_images),
        },
        "FDR_SES": fdr_ses,
        "results": {
            str(d): {
                "n_train_images": all_results[d]["n_train_images"],
                "spearman_r_mean": all_results[d]["sr_mean"],
                "spearman_r_std": all_results[d]["sr_std"],
                "fg_recall_40_mean": all_results[d]["fg_recall_mean"],
                "fg_recall_40_std": all_results[d]["fg_recall_std"],
                "per_seed_spearman": all_results[d]["seeds"],
            } for d in data_fractions
        },
    }

    with open(output_dir / "fdr_data_efficiency.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── 7. Conclusion | 结论 ──
    logger.log_info("b07/done",
                    f"\n{'='*65}\n"
                    f"  B-07 CONCLUSION\n"
                    f"  {'='*65}")
    logger.log_info("b07/done",
                    f"  {'Data%':<8} {'Spearman r':>12} {'FG@40':>10}")
    logger.log_info("b07/done",
                    f"  {'─'*8} {'─'*12} {'─'*10}")
    for d in data_fractions:
        logger.log_info("b07/done",
                        f"  {d:>5}%   {all_results[d]['sr_mean']:>10.4f}±"
                        f"{all_results[d]['sr_std']:.4f}   "
                        f"{all_results[d]['fg_recall_mean']*100:>8.1f}%")
    logger.log_info("b07/done",
                    f"\n  FDR-SES = {fdr_ses:.3f} "
                    f"(5% data → {fdr_ses*100:.1f}% ranking quality)")
    logger.log_info("b07/done",
                    f"  Results: {output_dir}/")
    logger.log_info("b07/done",
                    f"  Server: tail -f runs/b07_fdr_efficiency/b07.jsonl  |  grep b07/run")


if __name__ == "__main__":
    main()
