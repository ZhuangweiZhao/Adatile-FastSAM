#!/usr/bin/env python3
"""
E011-T: Tile Size Ablation — Tile Size 如何影响 Proto Collapse
================================================================

核心假设 | Core hypothesis:
    Tile 太小 → 目标碎片化, 语义缺失 | object fragmentation
    Tile 太大 → 背景占比过高, Proto 坍缩到背景 | background dominance
    存在 Sweet Spot: 语义上下文 vs 前景占比的最优平衡

实验 | Experiment:
    固定 ProtoHead (N=16, D=128, C=15), 固定训练配置
    变量 | variable: tile size ∈ {256, 384, 512, 768, 1024}
    测量 | measure:
        - mIoU (分割质量 | segmentation quality)
        - BG-Dominant Proto count (Proto 坍缩程度 | collapse severity)
        - Proto Entropy (Proto 类别多样性 | class diversity)
        - Foreground Ratio (每个 tile 平均前景占比 | avg fg fraction)

用法 | Usage:
    python tools/eval_e011t_tile_ablation.py --max-images 5
    python tools/eval_e011t_tile_ablation.py --tile-sizes 256,512,1024 --max-images 10
"""

import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.logging import get_logger
from adatile.utils.seed import set_seed

logger = get_logger("e011t_tile")


def parse_args():
    p = argparse.ArgumentParser(description="E011-T: Tile Size Ablation")
    p.add_argument("--tile-sizes", type=str, default="256,384,512,768,1024",
                   help="待测试的 tile 尺寸列表")
    p.add_argument("--epochs", type=int, default=30,
                   help="每个尺寸的训练轮数 | epochs per tile size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-images", type=int, default=5,
                   help="限制源图片数 (保证不同 tile size 看到相同图像内容)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ProtoHead (同 E010 | same as E010)
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(nn.Module):
    """
    多类别 ProtoHead | Multi-Class Proto Head.

    P4 → Conv(1280→D)→ReLU → CosineSim(N protos) → Conv(N→C)→logit.
    """

    def __init__(self, in_channels=1280, embed_dim=128, n_protos=16, num_classes=15):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        # 特征投影 | Feature projection: 1280 → D
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 可学习原型向量 | Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)

        # 多类别分割头 | Multi-class segmentation head
        self.head = nn.Conv2d(n_protos, num_classes, kernel_size=1, bias=True)

    def forward(self, p4, temperature=0.1):
        """Returns: embedding [B,D,H,W], sim_maps [B,N,H,W], logit [B,C,H,W]."""
        emb = self.project(p4)
        emb_n = F.normalize(emb, dim=1, p=2)
        p_n = F.normalize(self.prototypes, dim=1, p=2)
        sim = torch.einsum("bdhw,nd->bnhw", emb_n, p_n) / temperature
        return emb, sim, self.head(sim)

    def get_hard_assignment(self, p4, temperature=0.01):
        """
        每个像素的 Winner Proto 索引 | Winner-take-all proto index per pixel.

        Returns:
            [B, H/16, W/16] int64, 值 ∈ [0, N-1]
        """
        _, sim, _ = self.forward(p4, temperature)
        return sim.argmax(dim=1)


# ═══════════════════════════════════════════════════════════════════
# 分析 | Analysis — 单尺寸 Proto 表现评估
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_tile_size(model, backbone, val_ds, device, num_classes, n_protos,
                      tile_size, temperature=0.1):
    """
    分析单个 tile size 下的 Proto 表现 | Analyze Proto behavior at one tile size.

    返回 | Returns:
        miou:            分割 mIoU
        n_bg_protos:     背景主导的 Proto 数量 (>50% 像素分配到 class 0)
        proto_entropy:   Proto 类别分布的平均熵 (越高 → Proto 越多样)
        fg_ratio:        平均前景像素占比
        pct:             [N, C] 每个 Proto 的类别分布矩阵
    """
    model.eval()
    # 累积统计 | Accumulated statistics
    proto_class = np.zeros((n_protos, num_classes))  # 每个 Proto 分配到各类的像素数
    total_fg_ratio = 0.0
    mious = []
    n_samples = 0

    for idx in range(len(val_ds)):
        s = val_ds[idx]
        img = s["image"].unsqueeze(0).to(device)
        tgt = s["mask"].to(device)  # [H, W], class IDs
        H, W = tgt.shape

        # 前景占比 | Foreground pixel ratio
        fg_ratio = (tgt > 0).float().mean().item()
        total_fg_ratio += fg_ratio

        # 前向 | Forward pass
        feats = backbone(img)
        hard = model.get_hard_assignment(feats["p4"])  # [1, H/16, W/16]

        # ── mIoU 计算 | mIoU computation ──
        _, _, logit = model(feats["p4"], temperature=temperature)
        logit_up = F.interpolate(logit, size=(H, W), mode="bilinear",
                                 align_corners=False)
        pred = logit_up.argmax(dim=1)  # [1, H, W]

        miou_val = 0.0
        valid_cls = 0
        for c in range(num_classes):
            pred_c = (pred == c)
            tgt_c = (tgt.unsqueeze(0) == c)
            inter = (pred_c & tgt_c).sum().float()
            union = (pred_c | tgt_c).sum().float()
            if union > 0:
                miou_val += (inter + 1e-8) / union
                valid_cls += 1
        if valid_cls > 0:
            mious.append((miou_val / valid_cls).item())

        # ── Proto-Category 亲和力 | Proto-category affinity ──

        # ── Proto-Category 亲和力 | Proto-category affinity ──
        # 下采样 target 到 Proto 分配分辨率 | Downsample target to proto resolution
        H_emb, W_emb = hard.shape[1], hard.shape[2]
        tgt_down = F.interpolate(
            tgt.unsqueeze(0).unsqueeze(0).float(),
            size=(H_emb, W_emb), mode="nearest"
        ).squeeze().long()  # [H_emb, W_emb]

        for p in range(n_protos):
            mask_p = (hard.squeeze(0) == p)
            n_px = mask_p.sum().item()
            if n_px > 50:  # 至少 50 像素才统计 | at least 50 pixels
                for c in range(num_classes):
                    proto_class[p, c] += (tgt_down[mask_p] == c).sum().item()
        n_samples += 1

    # ── 汇总 | Aggregate ──
    miou_mean = float(np.mean(mious)) if mious else 0.0
    fg_mean = total_fg_ratio / max(n_samples, 1)

    # 归一化为占比 | Normalize to proportions
    row_sums = proto_class.sum(axis=1, keepdims=True) + 1e-8
    pct = proto_class / row_sums  # [N, C]

    # 背景主导 Proto 数 | Number of background-dominant protos
    n_bg = int((pct[:, 0] > 0.5).sum())

    # Proto 类别熵 | Proto category entropy
    # 高熵 = Proto 负责多种类别 (好 → 多样性)
    # 低熵 = Proto 坍缩到单一类别
    entropies = []
    for p in range(n_protos):
        dist = pct[p] + 1e-8
        dist = dist / dist.sum()
        ent = -(dist * np.log(dist)).sum()
        entropies.append(ent)
    # 排除死 Proto (总像素数 < 100)
    active_ent = [e for p, e in enumerate(entropies) if proto_class[p].sum() > 100]
    proto_ent_mean = float(np.mean(active_ent)) if active_ent else 0.0

    return {
        "miou": miou_mean,
        "n_bg_protos": n_bg,
        "proto_entropy": proto_ent_mean,
        "fg_ratio": fg_mean,
        "pct": pct,
    }


# ═══════════════════════════════════════════════════════════════════
# 训练 | Training — 单 Tile Size
# ═══════════════════════════════════════════════════════════════════

def train_proto(proto_head, backbone, train_ds, args, device, tile_size):
    """
    训练 ProtoHead (单 tile size) | Train ProtoHead for one tile size.

    Args:
        tile_size: 当前 tile 尺寸 (仅用于日志标记)
    """
    proto_head.train()
    opt = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    logger.log_info(f"train/tile{tile_size}/start",
                    f"tile={tile_size} epochs={args.epochs} "
                    f"tiles={len(train_ds)} lr={args.lr}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        proto_head.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [{tile_size}] E{epoch}/{args.epochs}", leave=False)
        for idx in pbar:
            s = train_ds[idx]
            img = s["image"].unsqueeze(0).to(device)
            tgt = s["mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                feats = backbone(img)

            # 前向 + 损失 | Forward + loss
            _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tgt.shape[1:],
                                     mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logit_up, tgt, ignore_index=255)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{total_loss / n:.4f}"})

        sch.step()
        avg_loss = total_loss / max(n, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        # 每 10 epoch 或首末 epoch 记录 | Log every 10 epochs + first/last
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            logger.log_loss(f"ce/tile{tile_size}", avg_loss, step=epoch)
            logger.log_info(f"train/tile{tile_size}",
                            f"E{epoch:2d}/{args.epochs} loss={avg_loss:.4f}")

    return proto_head


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device
    tile_sizes = [int(x.strip()) for x in args.tile_sizes.split(",")]

    logger.log_info("exp/start",
                    f"E011-T Tile Size Ablation | "
                    f"sizes={tile_sizes} n_protos={args.n_protos} "
                    f"max_images={args.max_images} epochs={args.epochs}")
    logger.log_info("exp/config",
                    f"embed_dim={args.embed_dim} lr={args.lr} "
                    f"seed={args.seed} device={device}")

    # 固定随机种子 | Fix random seed
    set_seed(args.seed)

    # ── 实验管理 | Experiment management ──
    exp_id = generate_exp_id(name="e011t_tile")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="iSAID_tiles",
                              dataset_root="data/iSAID_tiles")
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    # 覆盖默认 image_size 为实际 tile 尺寸列表 | Override with actual tile sizes
    recorder.logger.log_info("config/tile_sizes", f"{args.tile_sizes}")
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)
    logger.log_info("exp/output", f"Results: {output_path}")

    # ── 共享 Backbone | Shared backbone ──
    logger.log_info("model/backbone", "Frozen FastSAM backbone loaded")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    all_results = {}

    # ── 逐尺寸训练 + 分析 | Train + analyze per tile size ──
    for ts in tile_sizes:
        logger.log_info("phase", f"Tile size = {ts}×{ts}")

        # 确定数据目录 | Determine data directory
        data_dir = "data/iSAID_tiles" if ts == 1024 else f"data/iSAID_tiles_{ts}"

        # 加载数据 | Load pre-cut tiles
        try:
            train_ds = FastISAIDTileDataset(root_dir=data_dir, split="train",
                                            semantic=True)
            val_ds = FastISAIDTileDataset(root_dir=data_dir, split="val",
                                          semantic=True)
        except FileNotFoundError:
            logger.log_info(f"data/tile{ts}/missing",
                            f"Data not found: {data_dir}. "
                            f"Run prep_isaid_tiles.py --tile-size {ts}")
            continue

        # 按源图片数限制 | Limit by source image count (fair comparison)
        if args.max_images > 0:
            def _filter_by_images(tiles, n_images):
                """保留前 N 张源图的全部 tile | Keep tiles from first N source images."""
                seen = set()
                filtered = []
                for t in tiles:
                    img_id = t.rsplit("_t", 1)[0]  # "P0000_t003.png" → "P0000"
                    if img_id not in seen:
                        seen.add(img_id)
                    if len(seen) <= n_images:
                        filtered.append(t)
                    else:
                        break
                return filtered

            train_ds._tiles = _filter_by_images(train_ds._tiles, args.max_images)
            val_n = max(1, args.max_images // 2)
            val_ds._tiles = _filter_by_images(val_ds._tiles, val_n)

        logger.log_info(f"data/tile{ts}",
                        f"{args.max_images} imgs → "
                        f"{len(train_ds)} train tiles, {len(val_ds)} val tiles")

        # 训练 | Train ProtoHead
        set_seed(args.seed)
        set_seed(args.seed)

        proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                                n_protos=args.n_protos,
                                num_classes=args.num_classes).to(device)
        n_params = sum(p.numel() for p in proto_head.parameters())
        logger.log_info(f"model/tile{ts}",
                        f"ProtoHead {n_params:,} params (N={args.n_protos}, D={args.embed_dim})")

        train_proto(proto_head, backbone, train_ds, args, device, ts)

        # 分析 | Analyze proto behavior
        r = analyze_tile_size(proto_head, backbone, val_ds, device,
                              args.num_classes, args.n_protos, ts,
                              temperature=args.temperature)
        all_results[ts] = r

        logger.log_metric(f"tile{ts}/miou", r["miou"], tags=["e011t", str(ts)])
        logger.log_metric(f"tile{ts}/bg_protos", int(r["n_bg_protos"]),
                          tags=["e011t", str(ts)])
        logger.log_metric(f"tile{ts}/proto_entropy", r["proto_entropy"],
                          tags=["e011t", str(ts)])
        logger.log_metric(f"tile{ts}/fg_ratio", r["fg_ratio"],
                          tags=["e011t", str(ts)])
        logger.log_info(f"result/tile{ts}",
                        f"mIoU={r['miou']:.4f} "
                        f"BG-Proto={r['n_bg_protos']}/{args.n_protos} "
                        f"ProtoEnt={r['proto_entropy']:.3f} "
                        f"FG%={r['fg_ratio']:.1%}")

    # ── 汇总表 | Summary table ──
    if not all_results:
        logger.log_info("error", "No results. Generate tiles first with prep_isaid_tiles.py")
        return

    logger.log_info("summary", "=" * 50)
    header = f"  {'Size':>10} {'mIoU':>8} {'BG-Proto':>10} {'ProtoEnt':>9} {'FG%':>8}"
    logger.log_info("summary/header", header)
    for ts in tile_sizes:
        if ts not in all_results:
            continue
        r = all_results[ts]
        line = (f"  {ts:>4}×{ts:<4} {r['miou']:>8.4f} "
                f"{r['n_bg_protos']:>5}/{args.n_protos}{'':>4} "
                f"{r['proto_entropy']:>8.4f} {r['fg_ratio']:>7.1%}")
        logger.log_info(f"summary/tile{ts}", line)

    # ── 最佳分析 | Best analysis ──
    best_ts = max(all_results, key=lambda ts: all_results[ts]["miou"])
    min_bg_ts = min(all_results, key=lambda ts: all_results[ts]["n_bg_protos"])
    logger.log_info("summary/best",
                    f"Best mIoU: {best_ts} ({all_results[best_ts]['miou']:.4f})")
    logger.log_info("summary/min_bg",
                    f"Fewest BG protos: {min_bg_ts} "
                    f"({all_results[min_bg_ts]['n_bg_protos']}/{args.n_protos})")
    logger.log_info("summary/sweet_spot",
                    f"Sweet spot: where mIoU high AND BG-Proto low "
                    f"(likely between {min_bg_ts} and {best_ts})")

    # ── 记录到 Recorder | Record to ExperimentRecorder ──
    for ts in tile_sizes:
        if ts not in all_results:
            continue
        recorder.record_metric(f"tile{ts}_miou",
                               float(all_results[ts]["miou"]), tags=["e011t"])
        recorder.record_metric(f"tile{ts}_bg",
                               int(all_results[ts]["n_bg_protos"]), tags=["e011t"])
        recorder.record_metric(f"tile{ts}_entropy",
                               float(all_results[ts]["proto_entropy"]), tags=["e011t"])

    recorder.finalize()
    recorder.close()
    logger.log_info("exp/done", f"Results: {output_path}/")


if __name__ == "__main__":
    main()
