#!/usr/bin/env python3
"""
Token Routing Verification — Oracle / Random / SPM 三层验证.
=============================================================

验证假设 | Hypothesis:
    FastSAM Decoder 不需要处理全部 Token — 只处理一部分重要 Token
    也能保持大部分 Few-shot 分割性能。

实验逻辑 | Experiment Logic (Phase 0→1→2):
    Phase 0 (Oracle):  GT mask → per-token fg_ratio → Top-K  → 验证理论上界
    Phase 1 (Random): 随机 importance → Top-K                → 验证下界
    Phase 2 (SPM):    训练 FDR → importance → Top-K          → 验证方法有效性

    理想梯度: Oracle ≥ SPM ≫ Random
    - Oracle ≈ 100%: Token 冗余存在，Decoder 可以剪
    - Random ≪ Oracle: 不是随便删都行
    - SPM → Oracle: 学到的 importance 逼近最优

输出指标 | Output Metrics (per K level):
    Keep% | FG Recall% | FG Precision% | mIoU | Relative Retention%

用法 | Usage::

    # 完整实验
    python tools/paper_b/eval_fdr_token_fss.py \
        --tile-root data/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 1 \
        --decoder-ckpt runs/fewshot_.../decoder_p3p4film_1shot_best.pt \
        --device cuda
"""

import sys, argparse, json
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "train"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "instance"))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS
from adatile.utils.prototype import compute_fg_prototype
from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.sparse.spatial_router import ForegroundDensityRouter

logger = get_logger("token_routing")


# ═══════════════════════════════════════════════════════════════════════
# Token-Level FG Ratio Dataset | Token 级 FG 密度数据集
# ═══════════════════════════════════════════════════════════════════════

class TokenFGRatioDataset(torch.utils.data.Dataset):
    """从 tile 图像提取 per-token fg_ratio GT (P4 stride=16)."""

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
        logger.log_info("data", f"TokenFGRatioDataset: {len(self.img_paths)} tiles, {split}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

        label_path = self.label_dir / f"{img_path.stem}_label.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        H, W = label.shape[:2]
        fg = (label > 0).astype(np.float32)
        h_t, w_t = H // self.stride, W // self.stride
        fg = fg[:h_t * self.stride, :w_t * self.stride]
        fg_ratio = fg.reshape(h_t, self.stride, w_t, self.stride)
        fg_ratio = fg_ratio.transpose(0, 2, 1, 3).reshape(h_t, w_t, -1).mean(axis=2)
        fg_t = torch.from_numpy(fg_ratio).unsqueeze(0).float()
        return img_t, fg_t


# ═══════════════════════════════════════════════════════════════════════
# Train Token-Level FDR | 训练 Token 级 FDR
# ═══════════════════════════════════════════════════════════════════════

def train_fdr(args, backbone, device, out_dir):
    """Train token-level FDR: P4 → importance, supervised by per-token fg_ratio (MSE)."""
    logger.log_info("fdr", "=" * 60)
    logger.log_info("fdr", "Training Token-Level FDR (fg_ratio → MSE)")
    logger.log_info("fdr", "=" * 60)

    train_ds = TokenFGRatioDataset(args.tile_root, "train", stride=16)
    val_ds = TokenFGRatioDataset(args.tile_root, "val", stride=16)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    fdr = ForegroundDensityRouter(in_channels=1280, mid_channels=128, tile_size_feat=1)
    fdr.train().to(device)
    logger.log_info("fdr", f"FDR params: {sum(p.numel() for p in fdr.parameters()):,}")

    opt = torch.optim.AdamW(fdr.parameters(), lr=args.lr_fdr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.fdr_epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    best_val = float("inf")

    for epoch in range(1, args.fdr_epochs + 1):
        fdr.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"FDR E{epoch}/{args.fdr_epochs} [train]")
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
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)

        fdr.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, fg_gt in val_loader:
                imgs, fg_gt = imgs.to(device), fg_gt.to(device)
                p4 = backbone(imgs)["p4"]
                imp = fdr(p4)["importance"]
                if imp.shape[2:] != fg_gt.shape[2:]:
                    fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                          mode="bilinear", align_corners=False)
                val_loss += criterion(imp, fg_gt).item()
        avg_val = val_loss / len(val_loader)

        logger.log_info("fdr", f"E{epoch:3d} | train={avg_train:.6f} | val={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"epoch": epoch, "val_loss": avg_val,
                        "model_state_dict": fdr.state_dict()},
                       out_dir / "fdr_token_best.pt")

        sch.step()

    # Load best
    ckpt = torch.load(out_dir / "fdr_token_best.pt", map_location=device)
    fdr.load_state_dict(ckpt["model_state_dict"])
    logger.log_info("fdr", f"Loaded best FDR (E{ckpt['epoch']}, val={ckpt['val_loss']:.6f})")

    # Spearman diagnostic
    fdr.eval()
    preds_all, gt_all = [], []
    with torch.no_grad():
        for imgs, fg_gt in val_loader:
            imgs = imgs.to(device)
            p4 = backbone(imgs)["p4"]
            imp = fdr(p4)["importance"]
            if imp.shape[2:] != fg_gt.shape[2:]:
                fg_gt = F.interpolate(fg_gt, size=imp.shape[2:],
                                      mode="bilinear", align_corners=False)
            preds_all.append(imp.flatten().cpu())
            gt_all.append(fg_gt.flatten().cpu())
    preds_all = torch.cat(preds_all).numpy()
    gt_all = torch.cat(gt_all).numpy()
    from scipy.stats import spearmanr
    r, _ = spearmanr(preds_all, gt_all)
    logger.log_info("fdr", f"Spearman r = {r:.4f} (token-level, val set)")
    return fdr


# ═══════════════════════════════════════════════════════════════════════
# Load Decoder from Checkpoint | 从 Checkpoint 加载 Decoder
# ═══════════════════════════════════════════════════════════════════════

def load_decoder(ckpt_path, device, num_proto=1):
    """从 train_fewshot.py checkpoint 重建并加载 decoder."""
    from eval_c04_full_fewshot import P3P4FiLMFusionDecoder, FiLMFewShotDecoder

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # 推断 decoder 类型 | Infer decoder type
    if "gate_mlp.0.weight" in state:
        decoder = P3P4FiLMFusionDecoder(
            feat_dim_p3=960, feat_dim_p4=1280, fusion_dim=256, proto_dim=1280,
        )
    elif "film_mlp.0.weight" in state:
        w = state.get("proj.0.weight")
        decoder = FiLMFewShotDecoder(
            feat_dim=w.shape[1] if w is not None else 1280)
    else:
        raise ValueError(
            f"Cannot infer decoder type from state_dict keys: "
            f"{[k for k in list(state.keys())[:5]]}")

    decoder.load_state_dict(state, strict=False)
    decoder.to(device).eval()
    if num_proto > 1:
        decoder.num_prototypes = num_proto
    n = sum(p.numel() for p in decoder.parameters())
    logger.log_info("decoder", f"Loaded {type(decoder).__name__}: {n:,} params")
    return decoder


# ═══════════════════════════════════════════════════════════════════════
# Generate Importance Maps | 生成三种 Importance Map
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_oracle_importance(query_img, query_mask, backbone, device, stride=16):
    """
    Oracle importance: fg_ratio from GT mask → per-token [0,1].
    上界 | Upper bound: 完美知道哪些 token 有前景.
    """
    # 将 GT mask 下采样到 P4 分辨率 | Downsample GT mask to P4 resolution
    if isinstance(query_mask, torch.Tensor):
        mask_np = query_mask.cpu().numpy()
    else:
        mask_np = query_mask
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    H, W = mask_np.shape[:2]
    h_t, w_t = H // stride, W // stride
    mask_crop = mask_np[:h_t * stride, :w_t * stride]
    fg = (mask_crop > 0).astype(np.float32)
    fg_ratio = fg.reshape(h_t, stride, w_t, stride)
    fg_ratio = fg_ratio.transpose(0, 2, 1, 3).reshape(h_t, w_t, -1).mean(axis=2)
    return torch.from_numpy(fg_ratio).unsqueeze(0).unsqueeze(0).float().to(device)


@torch.no_grad()
def get_random_importance(query_p4, device):
    """Random importance: 均匀随机 [0,1], 完全无信息. 下界 | Lower bound."""
    return torch.rand(1, 1, query_p4.shape[2], query_p4.shape[3], device=device)


@torch.no_grad()
def get_spm_importance(fdr, query_p4):
    """SPM/FDR importance: 学习到的密度预测. 方法 | Method."""
    return fdr(query_p4)["importance"]


# ═══════════════════════════════════════════════════════════════════════
# Top-K Mask from Importance | 从 Importance 生成 Top-K Mask
# ═══════════════════════════════════════════════════════════════════════

def topk_mask(importance_map, k, device):
    """
    importance_map: [1, 1, H, W] → binary mask [1, 1, H, W] with Top-K True.
    k: float in (0, 1], fraction to keep.
    """
    n_total = importance_map.numel()
    n_keep = max(1, int(n_total * k))
    imp_flat = importance_map.flatten()
    _, top_idx = torch.topk(imp_flat, n_keep)
    mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    mask[top_idx] = True
    return mask.reshape(importance_map.shape)


# ═══════════════════════════════════════════════════════════════════════
# Full Routing Evaluation | 完整路由评估 (Oracle + Random + SPM)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_all_routing(fdr, decoder, backbone, train_ds, val_ds,
                         novel_class_ids, shot, device, rng,
                         k_levels=(1.0, 0.7, 0.5, 0.3),
                         feature_level="p3p4", num_proto=1,
                         n_val_per_class=30):
    """
    三种路由策略统一评估 | Unified evaluation across 3 routing strategies.

    For each (class, episode, K, strategy):
      1. Support → prototype
      2. Query → P3/P4 features
      3. strategy(importance) → Top-K mask
      4. masked features → decoder → IoU + FG Recall/Precision

    Returns:
      results[strategy][K] = {
        "miou": float,
        "fg_recall": float,   # % of GT FG tokens retained by Top-K
        "fg_precision": float, # % of Top-K tokens that are FG
        "per_class": {cls_id: mean_iou},
      }
    """
    from eval_c03_catsam_fewshot import compute_multi_prototype

    STRATEGIES = ["oracle", "random", "spm"]

    # 初始化 | Initialize
    results = {s: {k: {"ious": [], "fg_recalls": [], "fg_precisions": []}
                   for k in k_levels}
               for s in STRATEGIES}
    for s in STRATEGIES:
        for k in k_levels:
            results[s][k]["per_cls"] = defaultdict(list)

    fdr.eval()
    decoder.eval()
    backbone.eval()

    for cls_id in sorted(novel_class_ids):
        train_cand = train_ds.class_to_images(cls_id)
        val_cand = val_ds.class_to_images(cls_id)
        if len(train_cand) < shot or not val_cand:
            continue

        cls_name = ISAID5I_CATEGORIES.get(cls_id, f"cls{cls_id}")
        pbar = tqdm(range(n_val_per_class),
                    desc=f"Eval cls {cls_id} ({cls_name})")
        for ep_idx in pbar:
            s_idxs = rng.choice(train_cand, shot, replace=False).tolist()
            q_idx = int(rng.choice(val_cand))

            # ── Support → prototype ──
            s_imgs = torch.stack([train_ds.load_image(si) for si in s_idxs]).to(device)
            s_masks = [train_ds.render_class_mask(si, cls_id).to(device)
                       for si in s_idxs]
            s_feats = backbone(s_imgs)
            s_p4s = [s_feats["p4"][i] for i in range(len(s_masks))]
            if num_proto > 1:
                proto = compute_multi_prototype(s_p4s, s_masks, num_prototypes=num_proto)
                if proto.dim() == 2 and proto.sum() == 0:
                    continue
            else:
                proto = compute_fg_prototype(s_p4s, s_masks)
                if proto.sum() == 0:
                    continue

            # ── Query → features + GT mask ──
            q_img = val_ds.load_image(q_idx).unsqueeze(0).to(device)
            q_mask = val_ds.render_class_mask(q_idx, cls_id).to(device)
            q_feats = backbone(q_img)

            if feature_level == "p3p4":
                q_p3, q_p4 = q_feats["p3"], q_feats["p4"]
                feat_shape = q_p4
            else:
                feat_shape = q_feats[feature_level]
            tsize = tuple(q_mask.shape)

            # ── 生成三种 importance | Generate 3 importance maps ──
            imp_oracle = get_oracle_importance(q_img, q_mask, backbone, device)
            imp_random = get_random_importance(feat_shape, device)
            if fdr is not None:
                imp_spm = get_spm_importance(fdr, feat_shape)
                # Resize oracle to match SPM resolution if needed
                if imp_oracle.shape[2:] != imp_spm.shape[2:]:
                    imp_oracle = F.interpolate(
                        imp_oracle, size=imp_spm.shape[2:],
                        mode="bilinear", align_corners=False)
                if imp_random.shape[2:] != imp_spm.shape[2:]:
                    imp_random = F.interpolate(
                        imp_random, size=imp_spm.shape[2:],
                        mode="bilinear", align_corners=False)
            else:
                imp_spm = None  # no FDR available

            imp_maps = {
                "oracle": imp_oracle,
                "random": imp_random,
                "spm": imp_spm,
            }

            # ── 为每个 K 和策略计算 ──
            for k in k_levels:
                if k >= 1.0:
                    # No routing baseline — 对所有策略相同 | Same for all
                    if feature_level == "p3p4":
                        logit = decoder(q_p3, q_p4, proto, target_size=tsize)
                    else:
                        logit = decoder(feat_shape, proto, target_size=tsize)

                    prob = torch.sigmoid(logit)
                    pred = (prob > 0.5).float()
                    intersection = (pred * q_mask).sum()
                    union = (pred + q_mask).clamp(0, 1).sum()
                    iou = (intersection / union.clamp(min=1)).item()

                    # FG Recall @ K=100% = 100% by definition
                    for s in STRATEGIES:
                        results[s][k]["ious"].append(iou)
                        results[s][k]["fg_recalls"].append(1.0)
                        results[s][k]["fg_precisions"].append(
                            q_mask.sum().item() / q_mask.numel())
                        results[s][k]["per_cls"][cls_id].append(iou)
                else:
                    for s_name, imp in imp_maps.items():
                        if imp is None:
                            continue

                        mask = topk_mask(imp, k, device)  # [1, 1, h, w]

                        # ── FG Recall: % of GT FG tokens retained ──
                        # GT FG tokens: oracle importance (fg_ratio) > 0
                        gt_fg_tokens = (imp_oracle > 0).float()  # [1, 1, h, w]
                        n_gt_fg = gt_fg_tokens.sum().clamp(min=1)
                        fg_retained = (mask.float() * gt_fg_tokens).sum()
                        fg_recall = (fg_retained / n_gt_fg).item()
                        results[s_name][k]["fg_recalls"].append(fg_recall)

                        # ── FG Precision: % of selected tokens that are FG ──
                        n_selected = mask.sum().clamp(min=1)
                        fg_precision = (fg_retained / n_selected).item()
                        results[s_name][k]["fg_precisions"].append(fg_precision)

                        # ── Decoder forward with mask ──
                        if feature_level == "p3p4":
                            q_p4_m = q_p4 * mask.float()
                            h3, w3 = q_p3.shape[2], q_p3.shape[3]
                            mask_p3 = F.interpolate(mask.float(), size=(h3, w3),
                                                    mode="nearest")
                            q_p3_m = q_p3 * mask_p3
                            logit = decoder(q_p3_m, q_p4_m, proto, target_size=tsize)
                        else:
                            mask_up = F.interpolate(mask.float(),
                                                    size=feat_shape.shape[2:],
                                                    mode="nearest")
                            logit = decoder(feat_shape * mask_up, proto,
                                            target_size=tsize)

                        prob = torch.sigmoid(logit)
                        pred = (prob > 0.5).float()
                        intersection = (pred * q_mask).sum()
                        union = (pred + q_mask).clamp(0, 1).sum()
                        iou = (intersection / union.clamp(min=1)).item()
                        results[s_name][k]["ious"].append(iou)
                        results[s_name][k]["per_cls"][cls_id].append(iou)

    # ── Aggregate ──
    summary = {}
    for s_name in STRATEGIES:
        summary[s_name] = {}
        for k in sorted(k_levels):
            r = results[s_name][k]
            per_cls = {}
            for cls_id in sorted(novel_class_ids):
                ious = r["per_cls"].get(cls_id, [])
                per_cls[cls_id] = float(np.mean(ious)) if ious else 0.0
            summary[s_name][k] = {
                "miou": float(np.mean(r["ious"])) if r["ious"] else 0.0,
                "fg_recall": float(np.mean(r["fg_recalls"])) if r["fg_recalls"] else 0.0,
                "fg_precision": float(np.mean(r["fg_precisions"])) if r["fg_precisions"] else 0.0,
                "per_class": per_cls,
            }
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Main | 主函数
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Token Routing Verification (Oracle→Random→SPM)")
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1, choices=[1, 3, 5])
    p.add_argument("--fdr-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-fdr", type=float, default=1e-3)
    p.add_argument("--decoder-ckpt", type=str, default="")
    p.add_argument("--num-prototypes", type=int, default=1)
    p.add_argument("--k-levels", type=str, default="0.3,0.5,0.7,1.0")
    p.add_argument("--skip-fdr-train", action="store_true")
    p.add_argument("--fdr-ckpt", type=str, default="")
    p.add_argument("--skip-routing-eval", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if not args.output_dir:
        args.output_dir = f"runs/token_routing/f{args.fold}_k{args.shot}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "token_routing.jsonl")))

    logger.log_info("config", f"Fold={args.fold}, Shot={args.shot}, "
                    f"K={args.k_levels}, Device={device}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── Backbone ──
    logger.log_info("backbone", "Loading frozen FastSAM backbone...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()

    # ── Data ──
    from train_fewshot import PreCutTileAdapter
    train_ds = PreCutTileAdapter(args.tile_root, "train")
    val_ds = PreCutTileAdapter(args.tile_root, "val")
    fold_info = ISAID5I_FOLDS[args.fold]
    novel_ids = fold_info["novel"]
    novel_names = [ISAID5I_CATEGORIES[c] for c in novel_ids]
    logger.log_info("data", f"Novel classes: {novel_names}")

    # ── FDR ──
    fdr = None
    if args.skip_fdr_train and args.fdr_ckpt:
        logger.log_info("fdr", f"Loading FDR from {args.fdr_ckpt}")
        fdr = ForegroundDensityRouter(in_channels=1280, mid_channels=128)
        ckpt = torch.load(args.fdr_ckpt, map_location=device)
        fdr.load_state_dict(ckpt["model_state_dict"])
        fdr.to(device).eval()
    elif not args.skip_fdr_train:
        fdr = train_fdr(args, backbone, device, out_dir)

    # ── Decoder ──
    if not args.decoder_ckpt:
        logger.log_error("decoder", "--decoder-ckpt is required!")
        logger.log_info("decoder",
                        "Train decoder first:\n"
                        "  python tools/train/train_fewshot.py "
                        "--dataset isaid5i --tile-root <...> --fold 0 --shot 1 "
                        "--decoder film --feature-level p3p4 --device cuda")
        sys.exit(1)

    feature_level = "p3p4"
    decoder = load_decoder(args.decoder_ckpt, device, num_proto=args.num_prototypes)

    # ── Routing Evaluation ──
    if args.skip_routing_eval:
        logger.log_info("eval", "Skipping routing evaluation (--skip-routing-eval)")
        return

    k_levels = [float(x.strip()) for x in args.k_levels.split(",")]
    rng = np.random.RandomState(args.seed + 999)

    logger.log_info("eval", f"Phase 0→1→2: Oracle → Random → SPM, K ∈ {k_levels}")
    summary = evaluate_all_routing(
        fdr, decoder, backbone, train_ds, val_ds,
        novel_ids, args.shot, device, rng,
        k_levels=k_levels,
        feature_level=feature_level,
        num_proto=args.num_prototypes,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Report | 报告
    # ═══════════════════════════════════════════════════════════════════

    baseline_miou = summary.get("oracle", {}).get(1.0, {}).get("miou", 0.0)
    if baseline_miou == 0.0:
        baseline_miou = summary.get("random", {}).get(1.0, {}).get("miou", 0.0)

    for s_name in ["oracle", "random", "spm"]:
        if s_name not in summary or not summary[s_name]:
            continue
        s = summary[s_name]
        label = {"oracle": "Phase 0: Oracle (Upper Bound)",
                 "random": "Phase 1: Random (Lower Bound)",
                 "spm": "Phase 2: SPM/FDR (Method)"}[s_name]

        logger.log_info("report", "")
        logger.log_info("report", "=" * 80)
        logger.log_info("report", label)
        logger.log_info("report", "=" * 80)

        # Header
        header = f"{'K':>6s} | {'mIoU':>7s} | {'FG Recall':>9s} | {'FG Prec':>8s} | {'Ret.%':>6s}"
        for c in novel_ids:
            header += f" | {ISAID5I_CATEGORIES[c][:6]:>6s}"
        logger.log_info("report", header)
        logger.log_info("report", "-" * len(header))

        for k in sorted(k_levels):
            if k not in s:
                continue
            r = s[k]
            ret = r["miou"] / baseline_miou * 100 if baseline_miou > 0 else 0
            line = (f"{k*100:5.0f}% | {r['miou']:7.4f} | "
                    f"{r['fg_recall']*100:8.1f}% | {r['fg_precision']*100:7.1f}% | "
                    f"{ret:5.1f}%")
            for cls_id in sorted(novel_ids):
                line += f" | {r['per_class'].get(cls_id, 0.0):6.4f}"
            logger.log_info("report", line)

    # ── Summary comparison | 三层对比 ──
    logger.log_info("report", "")
    logger.log_info("report", "=" * 80)
    logger.log_info("report", "Cross-Strategy Comparison | 三层对比")
    logger.log_info("report", "=" * 80)

    k_ref = 0.7  # 以 K=70% 为主要比较点
    logger.log_info("report", f"At K={k_ref*100:.0f}%:")
    for s_name in ["oracle", "random", "spm"]:
        if s_name not in summary or k_ref not in summary[s_name]:
            continue
        r = summary[s_name][k_ref]
        ret = r["miou"] / baseline_miou * 100 if baseline_miou > 0 else 0
        logger.log_info("report",
                        f"  {s_name:>6s}: mIoU={r['miou']:.4f}, "
                        f"FG Recall={r['fg_recall']*100:.1f}%, "
                        f"Retention={ret:.1f}%")

    # ── Gradient check | 梯度检查 ──
    logger.log_info("report", "")
    logger.log_info("report", "Gradient Check | 梯度验证:")
    logger.log_info("report", "  Ideal: Oracle ≥ SPM ≫ Random")
    if k_ref in summary.get("oracle", {}) and k_ref in summary.get("random", {}):
        o = summary["oracle"][k_ref]["miou"]
        r = summary["random"][k_ref]["miou"]
        if k_ref in summary.get("spm", {}):
            s = summary["spm"][k_ref]["miou"]
            oracle_spm_gap = (o - s) / o * 100 if o > 0 else 0
            spm_random_gap = (s - r) / o * 100 if o > 0 else 0
            logger.log_info("report",
                            f"  Oracle={o:.4f}  SPM={s:.4f}  Random={r:.4f}")
            logger.log_info("report",
                            f"  Oracle→SPM gap: {oracle_spm_gap:.1f}% (越小越好)")
            logger.log_info("report",
                            f"  SPM→Random gap: {spm_random_gap:.1f}% (越大越好)")
            if oracle_spm_gap < 5 and spm_random_gap > 10:
                logger.log_info("report",
                               "  ✓ Gradient confirmed: SPM close to Oracle, far from Random")
            else:
                logger.log_info("report",
                               "  ⚠ Gradient weak — check FDR training quality")
        else:
            logger.log_info("report",
                            f"  Oracle={o:.4f}  Random={r:.4f}  "
                            f"(SPM not available — run FDR training)")

    # Save
    result_path = out_dir / "routing_results.json"
    with open(result_path, "w") as f:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "summary": {s: {str(k): v for k, v in ss.items()} for s, ss in summary.items()},
        }, f, indent=2)
    logger.log_info("report", f"\nResults saved → {result_path}")
    logger.log_info("done", "Token Routing Verification Complete.")


if __name__ == "__main__":
    main()
