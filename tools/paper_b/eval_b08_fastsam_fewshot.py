#!/usr/bin/env python3
"""
B-08: FastSAM-FSS — FastSAM for Few-Shot Dense Segmentation
==============================================================

回答最基础的问题：冻结的 FastSAM Backbone 能否支持少样本密集分割？
Answer the foundational question: can a frozen FastSAM backbone support few-shot dense segmentation?

作为实例分割项目的辅助验证，在密集土地覆盖数据集上测试 few-shot 泛化能力。
Auxiliary validation for the instance-seg project: testing few-shot generalization on dense land-cover datasets.

实验设计 | Design:
    - 冻结 FastSAM → P4 特征 | Frozen FastSAM → P4 features
    - Support: K-shot images → per-class prototype (mean P4 feature)
    - Query: P4 → cosine similarity with prototypes → LightDecoder → mask
    - 数据集: LoveDA (7-class dense land-cover) or Vaihingen (6-class dense land-cover)
    - K = 1, 3, 5 shot

对比基线 | Baselines:
    - FastSAM + LightDecoder (main)
    - FastSAM + LinearProbe (1×1 conv only, no spatial reasoning)
    - ResNet50 + LightDecoder (ablation: is FastSAM better than RN50?)

用法 | Usage::
    python tools/paper_b/eval_b08_fastsam_fewshot.py \
        --tile-root data/LoveDA --dataset loveda \
        --shots 1,3,5 --episodes 200 --output-dir runs/b08_loveda
"""

import sys, argparse, json, datetime, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder

DATASET_CONFIGS = {
    "loveda":    {"num_classes": 7},
    "vaihingen": {"num_classes": 7},
    "isaid":     {"num_classes": 16},
}


def load_dataset(tile_root, dataset_name):
    """加载 tile 数据集，自动检测 train/val."""
    val_split = "val" if ((Path(tile_root) / "val").exists() or
                          (Path(tile_root) / "images" / "val").exists()) else "test"
    if dataset_name == "isaid":
        from adatile.datasets.isaid_tiles import FastISAIDTileDataset
        return (FastISAIDTileDataset(tile_root, split="train", dense_labels=True),
                FastISAIDTileDataset(tile_root, split=val_split, dense_labels=True))
    elif dataset_name == "vaihingen":
        from adatile.datasets.vaihingen_tiles import VaihingenTileDataset
        return (VaihingenTileDataset(tile_root, split="train", dense_labels=True),
                VaihingenTileDataset(tile_root, split=val_split, dense_labels=True))
    elif dataset_name == "loveda":
        from adatile.datasets.loveda_tiles import LoveDATileDataset
        return (LoveDATileDataset(tile_root, split="train", dense_labels=True),
                LoveDATileDataset(tile_root, split=val_split, dense_labels=True))
    raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_miou(pred, gt, num_classes):
    """Per-class mIoU (foreground only)."""
    miou_v, valid = 0.0, 0
    per_cls = {}
    for c in range(1, num_classes):
        pc = (pred == c); tc = (gt == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0:
            per_cls[c] = float(inter / union)
            miou_v += per_cls[c]; valid += 1
    return miou_v / max(valid, 1), per_cls


class FewShotDecoder(nn.Module):
    """
    非参数少样本解码器 | Non-Parametric Few-Shot Decoder.

    Support → per-class prototype (mean P4 feature)
    Query P4 → cosine similarity → softmax → mask
    零可训参数，纯特征质量测试 | Zero trainable params, pure feature quality test.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def compute_prototypes(self, p4_features, masks, num_classes):
        """从 support 特征计算 per-class prototype."""
        # ??: 3D??? shape[0]=C, shape[1]=H, shape[2]=W
        # p4_features[0] is [C, H, W] (3D after batch dim squeeze)
        feat_dim = p4_features[0].shape[0]
        p4_h, p4_w = p4_features[0].shape[1], p4_features[0].shape[2]
        prototypes = torch.zeros(num_classes, feat_dim,
                                device=p4_features[0].device)
        for c in range(1, num_classes):
            class_feats = []
            for i in range(len(p4_features)):
                # Downsample mask to P4 spatial size | 将 mask 下采样到 P4 空间尺寸
                m = masks[i]
                if m.dim() == 3:
                    m = m.squeeze(0)
                mask_orig = (m == c).float()  # [H_orig, W_orig]
                mask_4d = mask_orig.unsqueeze(0).unsqueeze(0)  # [1, 1, H_orig, W_orig]
                mask_p4 = F.interpolate(mask_4d, size=(p4_h, p4_w),
                                        mode="nearest").squeeze(0)  # [1, H_p4, W_p4]
                if mask_p4.sum() > 0:
                    weighted = (p4_features[i] * mask_p4).sum(dim=(1, 2)) / mask_p4.sum()
                    class_feats.append(weighted)
            if class_feats:
                prototypes[c] = torch.stack(class_feats).mean(dim=0)
        return F.normalize(prototypes, dim=1, p=2)

    def forward(self, query_p4, prototypes, num_classes, target_size=None):
        """
        Query P4 → cosine similarity with prototypes → softmax → mask.
        """
        q_norm = F.normalize(query_p4, dim=1, p=2)
        sim_maps = []
        for c in range(num_classes):
            if prototypes[c].sum() == 0:
                sim_maps.append(torch.zeros_like(q_norm[:, :1]))
            else:
                sim = (q_norm * prototypes[c].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
                sim_maps.append(sim / self.temperature)
        logit = torch.cat(sim_maps, dim=1)  # [1, C, H_p4, W_p4]
        if target_size is not None:
            logit = F.interpolate(logit, size=target_size, mode="bilinear",
                                  align_corners=False)
        return logit


class EpisodicDecoder(nn.Module):
    """
    回合式训练解码器 | Episodic Training Decoder.

    FewShotDecoder (prototype matching) + 可训练 refinement head.
    Non-parametric prototype computation + trainable refinement layers.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.prototype_computer = FewShotDecoder()
        # 轻量 refinement head | Lightweight refinement on similarity maps
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1, bias=True),
        )

    def compute_prototypes(self, p4_features, masks, num_classes):
        return self.prototype_computer.compute_prototypes(p4_features, masks, num_classes)

    def forward(self, query_p4, prototypes, num_classes, target_size=None):
        sim = self.prototype_computer(query_p4, prototypes, num_classes)
        x = self.refine(sim)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x




def run_episode(support_samples, query_sample, model, backbone, num_classes, device):
    """单次 episode | Single episode (non-parametric prototype matching)."""
    # Batch support images → one forward pass | 批量处理 support 图像
    support_imgs = torch.stack([s[0] for s in support_samples]).to(device)
    support_feats = backbone(support_imgs)
    support_p4s = [support_feats["p4"][i] for i in range(len(support_samples))]
    support_masks = [m.to(device) for _, m in support_samples]

    prototypes = model.compute_prototypes(support_p4s, support_masks, num_classes)
    query_img, query_gt = query_sample
    feats = backbone(query_img.unsqueeze(0).to(device))
    logit = model(feats["p4"], prototypes, num_classes, target_size=query_gt.shape)

    pred = logit.argmax(dim=1).cpu().numpy()[0]
    miou, per_cls = compute_miou(pred, query_gt.cpu().numpy(), num_classes)
    return miou, per_cls


@torch.no_grad()
def evaluate_fewshot(model, backbone, train_ds, val_ds, num_classes, device,
                     shot, n_episodes, logger, tag):
    """Run episodic evaluation | 运行回合式评估."""
    all_mious = []
    per_cls_collect = defaultdict(list)
    rng = np.random.RandomState(42)

    # Build class index from mask files (cached across shots) | 从 mask 文件建索引
    if not hasattr(evaluate_fewshot, "_class_cache"):
        evaluate_fewshot._class_cache = {}
    cache_key = id(train_ds)
    if cache_key not in evaluate_fewshot._class_cache:
        logger.log_info(f"{tag}/index",
                        f"  Building class index from {len(train_ds)} mask files...")
        t0 = time.perf_counter()
        from PIL import Image
        class_to_indices = defaultdict(list)
        for idx in tqdm(range(len(train_ds)), desc="  Indexing", leave=False):
            sample = train_ds._samples[idx]
            mask = np.array(Image.open(sample["mask_path"]))
            for c in np.unique(mask):
                if c > 0:
                    class_to_indices[int(c)].append(idx)
        evaluate_fewshot._class_cache[cache_key] = dict(class_to_indices)
        dt = time.perf_counter() - t0
        logger.log_info(f"{tag}/index", f"  Class index built in {dt:.0f}s (cached for all shots)")
    class_to_indices = evaluate_fewshot._class_cache[cache_key]

    logger.log_info(f"{tag}/classes",
                    f"  Class sizes: " +
                    ", ".join(f"c{c}={len(class_to_indices[c])}"
                             for c in sorted(class_to_indices.keys())))

    val_indices = list(range(len(val_ds)))
    t0 = time.perf_counter()
    log_every = 10  # 每 10 次 episode 输出一次 | log every 10 episodes

    for ep in tqdm(range(n_episodes), desc=f"  {shot}-shot"):
        query_class = rng.choice(list(class_to_indices.keys()))
        candidates = class_to_indices[query_class]
        if len(candidates) < shot:
            continue

        support_idxs = rng.choice(candidates, shot, replace=False)
        support_samples = [train_ds[si] for si in support_idxs]
        support_samples = [(s["image"], s["mask"]) for s in support_samples]

        query_idx = rng.choice(val_indices)
        q = val_ds[query_idx]
        query_sample = (q["image"], q["mask"])

        try:
            miou, per_cls = run_episode(support_samples, query_sample, model,
                                        backbone, num_classes, device)
            all_mious.append(miou)
            for c, v in per_cls.items():
                per_cls_collect[c].append(v)
        except RuntimeError as e:
            if "out of memory" in str(e) or "OOM" in str(e):
                logger.log_warn(f"{tag}/oom",
                               f"CUDA OOM at ep {ep+1}, skipping. Try --device cpu or smaller batch.")
                torch.cuda.empty_cache()
            else:
                raise  # 非 OOM 错误直接抛出 | non-OOM errors should propagate
        except Exception as e:
            logger.log_warn(f"{tag}/warn", f"Ep {ep+1}: {type(e).__name__}: {e}")

        # 中间聚合日志 | Intermediate aggregation log
        if (ep + 1) % log_every == 0 and all_mious:
            running_mean = float(np.mean(all_mious[-log_every:]))
            total_mean = float(np.mean(all_mious))
            logger.log_info(f"{tag}/progress",
                            f"  Ep {ep+1}/{n_episodes}: "
                            f"running_mIoU={running_mean:.4f} "
                            f"total_mIoU={total_mean:.4f} "
                            f"n_valid={len(all_mious)}")

    dt = time.perf_counter() - t0

    # Per-class breakdown | 逐类分解
    per_cls_avg = {}
    logger.log_info(f"{tag}/per_cls", f"  Per-class IoU ({shot}-shot):")
    for c in sorted(per_cls_collect.keys()):
        avg = float(np.mean(per_cls_collect[c]))
        per_cls_avg[str(c)] = avg
        logger.log_info(f"{tag}/per_cls", f"    Class {c}: {avg:.4f} ({len(per_cls_collect[c])} episodes)")

    result = {
        "miou_mean": float(np.mean(all_mious)) if all_mious else 0.0,
        "miou_std": float(np.std(all_mious)) if all_mious else 0.0,
        "n_valid_episodes": len(all_mious),
        "per_class_iou": per_cls_avg,
    }
    logger.log_info(f"{tag}/done",
                    f"  {shot}-shot done: {len(all_mious)} episodes in {dt:.0f}s "
                    f"({dt/max(len(all_mious),1):.2f}s/ep)")
    return result


# ═══════════════════════════════════════════════════
# Episodic Training | 回合式训练
# ═══════════════════════════════════════════════════

def train_episodic(model, backbone, train_ds, val_ds, num_classes, device,
                   shot, n_episodes_per_epoch, epochs, lr,
                   class_to_indices, val_indices, logger, output_dir):
    """
    跨 episode 训练 LightDecoder | Train LightDecoder across episodes.

    Each episode: K support → prototypes → query P4 + sim_maps → decoder → CE+Dice loss
    Decoder learns to decode prototype similarity into accurate masks.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_miou, best_state = 0.0, None
    rng = np.random.RandomState(42)
    metrics_path = Path(output_dir) / f"episodic_{shot}shot_metrics.jsonl"

    logger.log_info("episodic/train",
                    f"Episodic training: {shot}-shot, {epochs} epochs, "
                    f"{n_episodes_per_epoch} episodes/epoch")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for ep in tqdm(range(n_episodes_per_epoch), desc=f"  E{epoch}/{epochs}",
                       leave=False):
            # Sample episode | 采样 episode
            query_class = rng.choice(list(class_to_indices.keys()))
            candidates = class_to_indices[query_class]
            if len(candidates) < shot:
                continue

            support_idxs = rng.choice(candidates, shot, replace=False)
            support_imgs = torch.stack([train_ds[si]["image"] for si in support_idxs]).to(device)
            support_masks_raw = [train_ds[si]["mask"].to(device) for si in support_idxs]

            query_idx = rng.choice(val_indices)
            q = val_ds[query_idx]
            query_img = q["image"].unsqueeze(0).to(device)
            query_gt = q["mask"].unsqueeze(0).to(device)  # [1, H, W]

            # Forward | 前向
            support_feats = backbone(support_imgs)
            support_p4s = [support_feats["p4"][i] for i in range(len(support_idxs))]
            prototypes = model.compute_prototypes(support_p4s, support_masks_raw, num_classes)

            query_feats = backbone(query_img)
            logit = model(query_feats["p4"], prototypes, num_classes,
                         target_size=tuple(query_gt.shape[1:]))

            # Loss: Focal + Dice | 损失
            ce = F.cross_entropy(logit, query_gt, ignore_index=255, reduction="none")
            focal_loss = ((1 - torch.exp(-ce)) ** 5.0 * ce).mean()

            probs = F.softmax(logit, dim=1)
            dice_sum, vd = 0.0, 0
            for c in range(1, num_classes):
                p_c = probs[:, c]; t_c = (query_gt == c).float()
                inter = (p_c * t_c).sum()
                union = p_c.sum() + t_c.sum() + 1e-8
                if t_c.sum() > 0: dice_sum += (2*inter/union); vd += 1
            dice_loss = 1.0 - (dice_sum / max(vd, 1))

            loss = 0.5 * focal_loss + 0.5 * dice_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)

        # Validation | 验证 (固定 100 episodes)
        model.eval()
        val_mious = []
        with torch.no_grad():
            for _ in range(100):
                query_class = rng.choice(list(class_to_indices.keys()))
                candidates = class_to_indices[query_class]
                if len(candidates) < shot: continue
                support_idxs = rng.choice(candidates, shot, replace=False)
                support_imgs_v = torch.stack([train_ds[si]["image"] for si in support_idxs]).to(device)
                support_masks_v = [train_ds[si]["mask"].to(device) for si in support_idxs]
                qv = val_ds[rng.choice(val_indices)]

                sf = backbone(support_imgs_v)
                sp4s = [sf["p4"][i] for i in range(len(support_idxs))]
                protos = model.compute_prototypes(sp4s, support_masks_v, num_classes)
                qf = backbone(qv["image"].unsqueeze(0).to(device))
                gt_v = qv["mask"].squeeze() if qv["mask"].dim() == 3 else qv["mask"]
                logit_v = model(qf["p4"], protos, num_classes, target_size=tuple(gt_v.shape))
                pred_v = logit_v.argmax(dim=1).cpu().numpy()[0]
                miou_v, _ = compute_miou(pred_v, gt_v.cpu().numpy(), num_classes)
                val_mious.append(miou_v)

        val_miou = float(np.mean(val_mious))
        logger.log_info(f"episodic/{shot}shot",
                        f"E{epoch:2d}/{epochs} loss={avg_loss:.4f} val={val_miou:.4f} "
                        f"lr={sch.get_last_lr()[0]:.6f}")

        epoch_metrics = {"epoch": epoch, "loss": round(avg_loss, 6),
                         "val_miou": round(val_miou, 6)}
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(epoch_metrics) + "\n"); mf.flush()

        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(Path(output_dir) / f"decoder_{shot}shot_best.pt"))

    if best_state:
        model.load_state_dict(best_state)
    logger.log_info(f"episodic/{shot}shot",
                    f"Best val mIoU={best_miou:.4f} | saved decoder_{shot}shot_best.pt")
    return model, best_miou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="loveda",
                   choices=["loveda", "vaihingen", "isaid"])
    p.add_argument("--shots", type=str, default="1,3,5")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--train", action="store_true",
                   help="Train LightDecoder with episodic training (否则 non-parametric)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Episodic training epochs (only with --train)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", type=str, default="runs/b08_fewshot")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    num_classes = DATASET_CONFIGS[args.dataset]["num_classes"]
    shots = [int(x.strip()) for x in args.shots.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b08")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b08.jsonl")))

    # Load data
    train_ds, val_ds = load_dataset(args.tile_root, args.dataset)
    logger.log_info("b08/data", f"Dataset: {args.dataset} | {num_classes-1} FG classes + BG")
    logger.log_info("b08/data", f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
    # Unique classes in train
    train_classes = set()
    for idx in range(len(train_ds)):
        train_classes.update(torch.unique(train_ds[idx]["mask"]).tolist())
    logger.log_info("b08/data", f"Train classes present: {sorted(train_classes)}")

    # Load backbone (shared across all models)
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    logger.log_info("b08/model", "FastSAM backbone loaded (frozen) | 0 trainable params")
    logger.log_info("b08/model", "Method: Prototype Matching + Cosine Similarity (non-parametric)")
    logger.log_info("b08/config",
                    f"Shots: {shots} | Episodes per shot: {args.episodes} | Seed: {args.seed}")

    all_results = {}

    if args.train:
        # ═══ Episodic Training Mode | 回合式训练 ═══
        logger.log_info("b08/mode", "Training mode: EpisodicDecoder (prototype + refine head)")
        # Build class index from mask files (lazy, no full image load) | 从 mask 文件建索引
        logger.log_info("b08/cache", f"Building class index from {len(train_ds)} mask files...")
        t0 = time.perf_counter()
        from PIL import Image
        class_to_indices = defaultdict(list)
        for idx in tqdm(range(len(train_ds)), desc="  Indexing", leave=False):
            sample = train_ds._samples[idx]
            mask = np.array(Image.open(sample["mask_path"]))
            for c in np.unique(mask):
                if c > 0:
                    class_to_indices[int(c)].append(idx)
        val_indices = list(range(len(val_ds)))
        dt = time.perf_counter() - t0
        logger.log_info("b08/cache",
                        f"  Class index built in {dt:.0f}s (direct file reads, no RAM overhead)")

        for shot in shots:
            tag = f"b08/{shot}shot"
            logger.log_info(f"{tag}/start",
                            f"\n{'─'*50}\n  Episodic Training: {shot}-Shot\n{'─'*50}")
            t0 = time.perf_counter()

            model = EpisodicDecoder(num_classes).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log_info(f"{tag}/model", f"  Trainable params: {n_params:,}")

            model, best_miou = train_episodic(
                model, backbone, train_ds, val_ds, num_classes, device,
                shot, args.episodes, args.epochs, args.lr,
                class_to_indices, val_indices, logger, args.output_dir
            )
            dt = time.perf_counter() - t0

            # Evaluate trained model on val episodes | 评估训练后模型
            logger.log_info(f"{tag}/eval", "  Evaluating trained model...")
            eval_result = evaluate_fewshot(model, backbone, train_ds, val_ds,
                                          num_classes, device, shot,
                                          args.episodes, logger, f"{tag}/eval")
            all_results[f"{shot}shot"] = eval_result
            logger.log_info(f"{tag}/result",
                            f"  {shot}-shot TRAINED: mIoU={eval_result['miou_mean']:.4f}±"
                            f"{eval_result['miou_std']:.4f} "
                            f"(best val={best_miou:.4f}, {dt:.0f}s)")
            logger.log_metric(f"b08/miou_{shot}shot_trained", eval_result["miou_mean"],
                             tags=["b08", args.dataset, f"{shot}shot", "trained"])
    else:
        # ═══ Non-Parametric Mode (original) | 非参数模式 ═══
        model = FewShotDecoder().to(device)
        model.eval()

        for shot in shots:
            tag = f"b08/{shot}shot"
            logger.log_info(f"{tag}/start",
                            f"\n{'─'*50}\n  {shot}-Shot Evaluation ({args.episodes} episodes)\n{'─'*50}")
            t0 = time.perf_counter()
            result = evaluate_fewshot(model, backbone, train_ds, val_ds, num_classes,
                                      device, shot, args.episodes, logger, tag)
            dt = time.perf_counter() - t0
            all_results[f"{shot}shot"] = result

            logger.log_info(f"{tag}/result",
                            f"  {shot}-shot NON-PARAM: mIoU={result['miou_mean']:.4f}±"
                            f"{result['miou_std']:.4f} "
                            f"({result['n_valid_episodes']} episodes, {dt:.0f}s)")
            logger.log_metric(f"b08/miou_{shot}shot", result["miou_mean"],
                             tags=["b08", args.dataset, f"{shot}shot"])

    # ═══ Final Summary ═══
    logger.log_info("b08/summary", "\n" + "=" * 65)
    logger.log_info("b08/summary",
                    f"  FastSAM Few-Shot — {args.dataset.upper()} | Non-Parametric")
    logger.log_info("b08/summary", "=" * 65)
    # Header
    header = f"  {'Shot':<8} {'mIoU':>10} {'±std':>8}"
    # Get all class keys from results
    all_classes = set()
    for r in all_results.values():
        all_classes.update(r["per_class_iou"].keys())
    for c in sorted(all_classes, key=int):
        header += f" {'c'+c:>8}"
    logger.log_info("b08/summary", header)
    logger.log_info("b08/summary", f"  {'─'*8} {'─'*10} {'─'*8}" + f" {'─'*8}" * len(all_classes))

    for shot in shots:
        r = all_results[f"{shot}shot"]
        line = f"  {shot:<8} {r['miou_mean']*100:>9.2f}% {r['miou_std']*100:>7.2f}%"
        for c in sorted(all_classes, key=int):
            pc = r["per_class_iou"].get(c, 0.0)
            line += f" {pc*100:>7.2f}%"
        logger.log_info("b08/summary", line)

    # SES | 少样本效率评分
    if 1 in shots and 5 in shots:
        miou_1 = all_results["1shot"]["miou_mean"]
        miou_5 = all_results["5shot"]["miou_mean"]
        ses = miou_1 / max(miou_5, 1e-8)
        logger.log_info("b08/ses",
                        f"  SES(1-shot/5-shot) = {ses:.3f} → 1-shot retains {ses*100:.1f}% of 5-shot mIoU")

    # Save
    summary = {
        "experiment": "B-08 FastSAM Few-Shot" + (" (episodic training)" if args.train else " (non-parametric)"),
        "dataset": args.dataset,
        "timestamp": datetime.datetime.now().isoformat(),
        "num_classes": num_classes,
        "shots": shots,
        "episodes_per_shot": args.episodes,
        "method": "episodic_training" if args.train else "prototype_matching",
        "trainable_params": 0 if not args.train else
            sum(p.numel() for p in EpisodicDecoder(num_classes).parameters() if p.requires_grad),
        "epochs": args.epochs if args.train else 0,
        "results": {k: {"miou_mean": v["miou_mean"], "miou_std": v["miou_std"],
                        "n_episodes": v["n_valid_episodes"]}
                    for k, v in all_results.items()},
    }
    with open(output_dir / "fewshot_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
