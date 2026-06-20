#!/usr/bin/env python3
"""
B-03: Spatial Importance Router — 架构设计与消融
===================================================

B-02.5 证明：Router 学习的是 objectness / instance density，而非边缘或类别语义。
B-03 围绕"密度预测"这一核心来设计 Router，并消融各组件的贡献。

消融层次 | Ablation hierarchy:

    R0  MobileNetV3 + Simple Head   B-02 基线 (~1.5M params)
    R1  Tiny CNN (4×Conv)           极轻量下界 (~20K params) — "多小还能用?"
    R2  DensityRouter               **主线** — MV3 + DensityHead
    R3  DensityRouter + EdgeHead    消融验证 — 边缘是否在密度之上有额外贡献?

核心理念 | Core philosophy:
    监督信号 = fg_ratio → Router 应该预测"前景密度"，而非"边缘"或"纹理".
    R3 的存在只是为了向审稿人证明：Edge 信号已被密度隐式捕获, 无需显式建模.

用法 | Usage:
    python tools/eval_b03_router_architecture.py
    python tools/eval_b03_router_architecture.py --routers R0,R2,R3 --epochs 20
    python tools/eval_b03_router_architecture.py --routers R1 --epochs 10
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend
from adatile.utils.seed import set_seed
from adatile.sparse.spatial_router import (
    ForegroundDensityRouter, DualStreamRouter, TinyCNNRouter, DensityHead,
)

logger = get_logger("b03_router")
logger.add_backend(ConsoleBackend())

ACTUAL_TO_CODE_ID = {
    1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 10, 7: 6, 8: 9,
    9: 7, 10: 8, 11: 11, 12: 13, 13: 12, 14: 15, 15: 14,
}
TILE_SIZE = 1024
IMAGE_SIZE = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--train-images", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--routers", type=str, default="R0,R1,R2,R3",
                   help="Router 变体 | Router variants")
    p.add_argument("--cache-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="runs/b03_router_arch")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# R0: MV3 + Simple Head (B-02 基线)
# ═══════════════════════════════════════════════════════════════════

class MV3BaselineRouter(nn.Module):
    """R0: MobileNetV3-Small + Conv Head — B-02 baseline (~1.5M params)."""

    def __init__(self):
        super().__init__()
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features  # stride=32, 576ch
        self.head = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True), nn.Sigmoid(),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return {"importance": self.head(self.backbone(x))}


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def render_semantic_mask(annotations, h, w):
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ACTUAL_TO_CODE_ID.get(ann.get("category_id", 0), 0)
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
            cv2.fillPoly(sem, [pts], cat_id)
    return sem


def _preprocess_worker(args_tuple):
    img_id, img_path, anns, image_size, cache_dir = args_tuple
    try:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        mask = render_semantic_mask(anns, H, W)
        scale = image_size / max(H, W)
        nH, nW = int(H*scale), int(W*scale)
        img = np.array(Image.fromarray(img).resize((nW, nH), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((nW, nH), Image.NEAREST))
        ph, pw = image_size-nH, image_size-nW
        if ph>0 or pw>0:
            img = np.pad(img, ((0,ph),(0,pw),(0,0)), mode="constant")
            mask = np.pad(mask, ((0,ph),(0,pw)), mode="constant")
        img = np.transpose(img.astype(np.float32)/255.0, (2,0,1))
        H2, W2 = mask.shape
        n_ty, n_tx = (H2+TILE_SIZE-1)//TILE_SIZE, (W2+TILE_SIZE-1)//TILE_SIZE
        ts = np.zeros((n_ty, n_tx), dtype=np.float32)
        fp = np.zeros(n_ty*n_tx, dtype=np.int64)
        idx = 0
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty*TILE_SIZE, min(ty*TILE_SIZE+TILE_SIZE, H2)
                x0, x1 = tx*TILE_SIZE, min(tx*TILE_SIZE+TILE_SIZE, W2)
                tm = mask[y0:y1, x0:x1]
                tp = (y1-y0)*(x1-x0)
                fp[idx] = int((tm>0).sum())
                ts[ty, tx] = fp[idx]/max(tp,1)
                idx += 1
        out = os.path.join(cache_dir, f"{img_id}.npz")
        np.savez_compressed(out, image=img, tile_scores=ts, fg_pixels=fp,
                            n_ty=n_ty, n_tx=n_tx)
        return out
    except Exception:
        return None


class CachedDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        return (torch.from_numpy(d["image"]), torch.from_numpy(d["tile_scores"]),
                torch.from_numpy(d["fg_pixels"]), int(d["n_ty"]), int(d["n_tx"]))


def preprocess_to_cache(images, cache_dir):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)
    tasks = [(img_id, img_path, anns, IMAGE_SIZE, cache_dir)
             for img_id, img_path, anns, _ in images]
    nw = min(8, max(1, (os.cpu_count() or 4)-2))
    paths = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for r in tqdm(ex.map(_preprocess_worker, tasks),
                       total=len(tasks), desc="  Preprocess", unit="img"):
            if r is not None: paths.append(r)
    return paths


# ═══════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, gt_scores, fg_px, n_ty_arr, n_tx_arr in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        outputs = model(imgs)
        imp = outputs["importance"]
        _, _, hp, wp = imp.shape

        batch_loss = 0.0
        STRIDE = 32 if hp >= 32 else 16  # MV3 → stride 32, TinyCNN → stride 16
        TILE_F = TILE_SIZE // STRIDE
        for b in range(B):
            gt = gt_scores[b].to(device)
            n_ty, n_tx = int(n_ty_arr[b]), int(n_tx_arr[b])
            preds, gts = [], []
            for ty in range(min(n_ty, hp//TILE_F)):
                for tx in range(min(n_tx, wp//TILE_F)):
                    y0, y1 = ty*TILE_F, min(ty*TILE_F+TILE_F, hp)
                    x0, x1 = tx*TILE_F, min(tx*TILE_F+TILE_F, wp)
                    if y1>y0 and x1>x0 and ty<gt.shape[0] and tx<gt.shape[1]:
                        preds.append(imp[b, 0, y0:y1, x0:x1].mean())
                        gts.append(gt[ty, tx])
            if preds:
                batch_loss += F.mse_loss(torch.stack(preds), torch.stack(gts))
        opt.zero_grad(); batch_loss.backward(); opt.step()
        total_loss += batch_loss.item(); n += 1
    return total_loss/max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gt, all_fg = [], [], []
    for imgs, gt_scores, fg_px, n_ty_arr, n_tx_arr in tqdm(loader, desc="  Eval", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        outputs = model(imgs)
        imp = outputs["importance"]
        _, _, hp, wp = imp.shape
        STRIDE = 32 if hp >= 32 else 16
        TILE_F = TILE_SIZE // STRIDE

        for b in range(B):
            gt = gt_scores[b].numpy(); fg = fg_px[b].numpy()
            n_ty, n_tx = int(n_ty_arr[b]), int(n_tx_arr[b])
            idx = 0
            for ty in range(min(n_ty, hp//TILE_F)):
                for tx in range(min(n_tx, wp//TILE_F)):
                    y0, y1 = ty*TILE_F, min(ty*TILE_F+TILE_F, hp)
                    x0, x1 = tx*TILE_F, min(tx*TILE_F+TILE_F, wp)
                    if y1>y0 and x1>x0 and ty<gt.shape[0] and tx<gt.shape[1]:
                        all_pred.append(imp[b,0,y0:y1,x0:x1].mean().item())
                        all_gt.append(float(gt[ty,tx]))
                        all_fg.append(int(fg[idx]))
                        idx += 1

    if len(all_pred) < 50: return None
    pa, ga, fa = np.array(all_pred), np.array(all_gt), np.array(all_fg)
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pa, ga); sr, _ = spearmanr(pa, ga)
    oo = np.argsort(ga)[::-1]; lo = np.argsort(pa)[::-1]
    oracle_r, learned_r = {}, {}
    for k in [20, 30, 40, 50]:
        n = max(1, int(len(ga)*k//100))
        oracle_r[k] = float(fa[oo[:n]].sum()/max(fa.sum(),1))
        learned_r[k] = float(fa[lo[:n]].sum()/max(fa.sum(),1))
    return {"pearson_r": float(pr), "spearman_r": float(sr),
            "oracle_retention": oracle_r, "learned_retention": learned_r, "n_tiles": len(ga)}


def run_exp(name, router, train_loader, eval_loader, args, device):
    logger.log_info("b03/exp", f"{'='*40}")
    logger.log_info("b03/exp", f"Training {name}...")
    n_p = sum(p.numel() for p in router.parameters() if p.requires_grad)
    logger.log_info("b03/exp", f"  Trainable: {n_p:,}")

    router = router.to(device)
    opt = torch.optim.Adam([p for p in router.parameters() if p.requires_grad], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(router, train_loader, opt, device)
        sch.step()
        if epoch%10==0 or epoch==1:
            logger.log_info("b03/exp", f"  {name} E{epoch}/{args.epochs} loss={loss:.4f}")

    results = evaluate(router, eval_loader, device)
    if results:
        sr = results["spearman_r"]
        gap40 = (results["oracle_retention"][40]-results["learned_retention"][40])*100
        logger.log_info("b03/result",
                        f"{name}: Spearman r={sr:.4f}  "
                        f"Top40 Oracle={results['oracle_retention'][40]*100:.1f}%  "
                        f"Learned={results['learned_retention'][40]*100:.1f}%  "
                        f"Gap={gap40:.2f}%  Params={n_p:,}")
        return {"name": name, "params": n_p, **results}
    return None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir or str(output_dir/"cache")
    router_names = [r.strip() for r in args.routers.split(",")]

    logger.log_info("b03/start",
                    f"B-03 Router Architecture | routers={router_names} "
                    f"train={args.train_images} epochs={args.epochs}")

    # ── Data ──
    src_root = Path(args.src_root)
    with open(src_root/"train"/"annotations"/"instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    img_dir = src_root/"train"/"images"
    all_images = []
    for img_info in coco["images"]:
        anns = img_id_to_anns.get(img_info["id"], [])
        if anns and (img_dir/img_info["file_name"]).exists():
            all_images.append((img_info["file_name"], str(img_dir/img_info["file_name"]),
                              anns, IMAGE_SIZE))

    np.random.seed(args.seed)
    perm = np.random.permutation(len(all_images))
    n_train = min(args.train_images, len(all_images)-30)
    train_imgs = [all_images[i] for i in perm[:n_train]]
    eval_imgs = [all_images[i] for i in perm[n_train:min(n_train+30, len(all_images))]]
    logger.log_info("b03/data", f"Train={len(train_imgs)} Eval={len(eval_imgs)}")

    train_paths = preprocess_to_cache(train_imgs, os.path.join(cache_root, "train"))
    eval_paths = preprocess_to_cache(eval_imgs, os.path.join(cache_root, "eval"))
    train_loader = DataLoader(CachedDataset(train_paths), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device=="cuda"), persistent_workers=(args.num_workers>0))
    eval_loader = DataLoader(CachedDataset(eval_paths), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device=="cuda"), persistent_workers=(args.num_workers>0))

    # ── Router Ablation ──
    all_results = []

    for rname in router_names:
        if rname == "R0":
            router = MV3BaselineRouter()
        elif rname == "R1":
            router = TinyCNNRouter()
        elif rname == "R2":
            # 主线 FDR: MV3 backbone + ForegroundDensityRouter
            mnet = models.mobilenet_v3_small(weights="DEFAULT")
            backbone = mnet.features
            for p in backbone.parameters(): p.requires_grad = False
            router = nn.Sequential(backbone, ForegroundDensityRouter(576, 128))
        elif rname == "R3":
            mnet = models.mobilenet_v3_small(weights="DEFAULT")
            backbone = mnet.features
            for p in backbone.parameters(): p.requires_grad = False
            router = nn.Sequential(backbone, DualStreamRouter(576))
        else:
            continue

        r = run_exp(rname, router, train_loader, eval_loader, args, device)
        if r: all_results.append(r)

    # ── Summary ──
    logger.log_info("b03/summary", f"{'='*60}")
    logger.log_info("b03/summary", "B-03 Router Architecture Ablation — Summary")
    logger.log_info("b03/summary",
                    "  Density = learns objectness/instance density (supervised by fg_ratio)")
    logger.log_info("b03/summary",
                    "  Edge    = Sobel-initialized edge-aware stream (ablation only)")
    logger.log_info("b03/summary",
                    f"  {'Name':<6} {'Params':>10} {'Spearman r':>11} {'Top40 Gap':>10} {'Role':<20}")
    role_map = {"R0": "Baseline (B-02)", "R1": "Lower bound", "R2": "**MAIN LINE**",
                "R3": "Ablation (edge)"}
    for r in all_results:
        gap = (r["oracle_retention"][40]-r["learned_retention"][40])*100
        logger.log_info("b03/summary",
                        f"  {r['name']:<6} {r['params']:>10,} {r['spearman_r']:>11.4f} {gap:>9.2f}% {role_map.get(r['name'],''):<20}")

    # ── Plot ──
    if all_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        names = [r["name"] for r in all_results]
        srs = [r["spearman_r"] for r in all_results]
        colors = ["#3498DB", "#95A5A6", "#27AE60", "#E67E22"]

        ax = axes[0]
        ax.bar(names, srs, color=colors[:len(names)], edgecolor="white")
        ax.axhline(y=0.889, color="gray", ls="--", alpha=0.5, label="B-02 MV3 baseline")
        ax.set_ylabel("Spearman r"); ax.set_title("Ranking Quality"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.2)
        for i, (n, v) in enumerate(zip(names, srs)):
            ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax = axes[1]
        params = [r["params"] for r in all_results]
        ax.bar(names, params, color=colors[:len(names)], edgecolor="white")
        ax.set_ylabel("Trainable Params"); ax.set_title("Model Size"); ax.grid(axis="y", alpha=0.2)
        ax.set_yscale("log")
        for i, (n, v) in enumerate(zip(names, params)):
            ax.text(i, v*1.2, f"{v:,}", ha="center", fontsize=8)

        ax = axes[2]
        gaps = [(r["oracle_retention"][40]-r["learned_retention"][40])*100 for r in all_results]
        ax.bar(names, gaps, color=colors[:len(names)], edgecolor="white")
        ax.set_ylabel("Top40 Gap (%)"); ax.set_title("Oracle-Learned Gap (lower=better)"); ax.grid(axis="y", alpha=0.2)

        r2_gap = next((g for n, g in zip(names, gaps) if n == "R2"), None)
        if r2_gap:
            ax.axhline(y=r2_gap, color="#27AE60", ls="--", alpha=0.4, lw=1)

        fig.suptitle("B-03: Spatial Importance Router — Density vs Edge",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir/"router_ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Save ──
    summary = {"experiment": "B-03 Router Architecture Ablation",
               "design_principle": "Density over Edge — supervised by fg_ratio, learns objectness",
               "timestamp": datetime.datetime.now().isoformat(), "config": vars(args),
               "results": all_results}
    with open(output_dir/"router_ablation.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b03/output", f"Saved: {output_dir}/")


if __name__ == "__main__":
    main()
