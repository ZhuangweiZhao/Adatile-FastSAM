#!/usr/bin/env python3
"""
E011-T: Tile Size Ablation — Tile Size 如何影响 Proto Collapse
================================================================

核心假设 | Core hypothesis:
    Tile 太小 → 目标碎片化, 语义缺失
    Tile 太大 → 背景占比过高, Proto 坍缩到背景
    存在一个 sweet spot: 语义上下文 vs 前景占比的最优平衡

实验 | Experiment:
    固定 ProtoHead (N=16, D=128, C=15), 固定训练配置
    变量: tile size ∈ {256, 384, 512, 768, 1024}
    测量:
        - mIoU (分割质量)
        - BG-Dominant Proto count (Proto 坍缩程度)
        - Proto Entropy (Proto 多样性)
        - Foreground Ratio (每个 tile 平均前景占比)

预期 | Expected:
    Tile    mIoU    BG-Proto    FG%     ProtoEnt
    256     低      少          高      高       ← 碎片化, 但 Proto 多样
    384     ↑       ↑          ↓       ↓
    512     最优    中间        平衡    平衡    ← Sweet spot
    768     ↓       ↓          ↓       ↓
    1024    ↓       多(12/16)   低      低       ← Proto 坍缩

用法 | Usage:
    # Step 1: 生成多尺寸 tile (一次性)
    for ts in 256 384 512 768; do
        python tools/prep_isaid_tiles.py --tile-size $ts \
            --dst-root data/iSAID_tiles_$ts --max-images 50
    done
    # 1024 已存在: data/iSAID_tiles/

    # Step 2: 运行消融
    python tools/eval_e011t_tile_ablation.py --tile-sizes 256,384,512,768,1024 --max-images 5
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

logger = get_logger("e011t_tile")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-sizes", type=str, default="256,384,512,768,1024")
    p.add_argument("--epochs", type=int, default=30)
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
# ProtoHead (same as E010)
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(nn.Module):
    def __init__(self, in_channels=1280, embed_dim=128, n_protos=16, num_classes=15):
        super().__init__()
        self.n_protos = n_protos
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False), nn.ReLU(inplace=True))
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.head = nn.Conv2d(n_protos, num_classes, 1, bias=True)

    def forward(self, p4, temperature=0.1):
        emb = self.project(p4)
        emb_n = F.normalize(emb, dim=1, p=2)
        p_n = F.normalize(self.prototypes, dim=1, p=2)
        sim = torch.einsum("bdhw,nd->bnhw", emb_n, p_n) / temperature
        return emb, sim, self.head(sim)

    def get_hard_assignment(self, p4, temperature=0.01):
        _, sim, _ = self.forward(p4, temperature)
        return sim.argmax(dim=1)


# ═══════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_tile_size(model, backbone, val_ds, device, num_classes, n_protos, tile_size):
    """
    分析单个 tile size 下的 Proto 表现.
    返回: mIoU, BG-dominant count, proto entropy, avg fg ratio
    """
    model.eval()
    proto_class = np.zeros((n_protos, num_classes))
    total_fg_ratio = 0.0
    mious = []
    n_samples = 0

    for idx in range(len(val_ds)):
        s = val_ds[idx]
        img = s["image"].unsqueeze(0).to(device)
        tgt = s["mask"].to(device)  # [H, W]
        H, W = tgt.shape

        # 统计前景占比 | Foreground ratio
        fg_ratio = (tgt > 0).float().mean().item()
        total_fg_ratio += fg_ratio

        # 前向 | Forward
        feats = backbone(img)
        hard = model.get_hard_assignment(feats["p4"])  # [1, H/16, W/16]

        # mIoU
        _, _, logit = model(feats["p4"], temperature=args.temperature)
        logit_up = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        pred = logit_up.argmax(dim=1)
        for c in range(num_classes):
            pc = (pred == c)
            tc = (tgt.unsqueeze(0) == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            if union > 0:
                proto_class[:, c].fill(0)  # placeholder, real miou below
        miou_val = 0
        valid = 0
        for c in range(num_classes):
            pc = (pred == c)
            tc = (tgt.unsqueeze(0) == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            if union > 0:
                miou_val += (inter + 1e-8) / (union + 1e-8)
                valid += 1
        mious.append(miou_val / max(valid, 1))

        # Proto-class affinity
        H_emb, W_emb = hard.shape[1], hard.shape[2]
        tgt_down = F.interpolate(tgt.unsqueeze(0).unsqueeze(0).float(),
                                 size=(H_emb, W_emb), mode="nearest").squeeze().long()
        for p in range(n_protos):
            mask_p = (hard.squeeze(0) == p)
            if mask_p.sum() > 50:
                for c in range(num_classes):
                    proto_class[p, c] += (tgt_down[mask_p] == c).sum().item()
        n_samples += 1

    # ── 汇总 | Aggregate ──
    miou_mean = float(np.mean(mious))
    fg_mean = total_fg_ratio / max(n_samples, 1)
    pct = proto_class / (proto_class.sum(axis=1, keepdims=True) + 1e-8)
    n_bg = int((pct[:, 0] > 0.5).sum())

    # Proto 熵 (类别分布的多样性) | Proto entropy (diversity of class distribution)
    # 高熵 = Proto 负责多种类别 (好), 低熵 = Proto 坍缩到单一类别 (可能坏)
    entropies = []
    for p in range(n_protos):
        dist = pct[p] + 1e-8
        dist = dist / dist.sum()
        ent = -(dist * np.log(dist)).sum()
        entropies.append(ent)
    # 排除死 Proto (全零分布)
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
# Training
# ═══════════════════════════════════════════════════════════════════

def train_proto(proto_head, backbone, train_ds, args, device, tile_size):
    """训练 ProtoHead | Train ProtoHead for one tile size."""
    proto_head.train()
    opt = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(1, args.epochs + 1):
        proto_head.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [{tile_size}] E{epoch}/{args.epochs}", leave=False)
        for idx in pbar:
            s = train_ds[idx]
            img = s["image"].unsqueeze(0).to(device)
            tgt = s["mask"].unsqueeze(0).to(device)
            with torch.no_grad(): feats = backbone(img)
            _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tgt.shape[1:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logit_up, tgt, ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
            pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
        sch.step()
        if epoch % 10 == 0 or epoch == 1:
            logger.log_info(f"tile{tile_size}/train",
                            f"E{epoch:2d} loss={total_loss/n:.4f} tiles={n}")
    return proto_head


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device
    tile_sizes = [int(x.strip()) for x in args.tile_sizes.split(",")]

    print("=" * 70)
    print("  E011-T: Tile Size Ablation")
    print(f"  Sizes: {tile_sizes}")
    print("=" * 70)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    exp_id = generate_exp_id(name="e011t_tile")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="iSAID_tiles", dataset_root="data/iSAID_tiles")
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    all_results = {}

    for ts in tile_sizes:
        print(f"\n{'─'*50}")
        print(f"  Tile Size = {ts}×{ts}")
        print(f"{'─'*50}")

        # 确定数据目录 | Determine data directory
        if ts == 1024:
            data_dir = "data/iSAID_tiles"
        else:
            data_dir = f"data/iSAID_tiles_{ts}"

        # 加载数据 | Load data
        try:
            train_ds = FastISAIDTileDataset(root_dir=data_dir, split="train", semantic=True)
            val_ds = FastISAIDTileDataset(root_dir=data_dir, split="val", semantic=True)
        except FileNotFoundError:
            print(f"    ⚠️  Data not found: {data_dir}")
            print(f"    Run: python tools/prep_isaid_tiles.py --tile-size {ts} --dst-root {data_dir} --max-images 50")
            continue

        # 按源图片数限制 | Limit by source image count (fair across tile sizes)
        if args.max_images > 0:
            def _filter_by_images(tiles, n_images):
                seen = set()
                filtered = []
                for t in tiles:
                    img_id = t.rsplit("_t", 1)[0]  # "P0000_t003.png" → "P0000"
                    seen.add(img_id)
                    if len(seen) <= n_images:
                        filtered.append(t)
                    elif img_id not in seen:
                        break
                return filtered
            train_ds._tiles = _filter_by_images(train_ds._tiles, args.max_images)
            val_ds._tiles = _filter_by_images(val_ds._tiles, max(1, args.max_images // 2))
            print(f"    → {args.max_images} imgs: {len(train_ds._tiles)}T train, {len(val_ds._tiles)}T val")
        logger.log_info(f"tile{ts}", f"Train={len(train_ds)}, Val={len(val_ds)}")

        # 训练 | Train
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                                n_protos=args.n_protos, num_classes=args.num_classes).to(device)
        n_params = sum(p.numel() for p in proto_head.parameters())
        print(f"    ProtoHead: {n_params:,} params")
        train_proto(proto_head, backbone, train_ds, args, device, ts)

        # 分析 | Analyze
        r = analyze_tile_size(proto_head, backbone, val_ds, device,
                              args.num_classes, args.n_protos, ts)
        all_results[ts] = r
        logger.log_metric(f"tile{ts}/miou", r["miou"], tags=["e011t", str(ts)])
        logger.log_metric(f"tile{ts}/bg_protos", r["n_bg_protos"], tags=["e011t", str(ts)])
        logger.log_info(f"tile{ts}/result",
                        f"mIoU={r['miou']:.4f} BG={r['n_bg_protos']}/{args.n_protos} "
                        f"Ent={r['proto_entropy']:.3f} FG={r['fg_ratio']:.1%}")

    # ── Summary Table ──
    if not all_results:
        print("\nNo results. Generate tiles first.")
        return

    print(f"\n{'=' * 70}")
    print(f"  E011-T Results | Tile Size Ablation")
    print(f"  {'=' * 70}")
    print(f"  {'Size':>10} {'mIoU':>8} {'BG-Proto':>10} {'ProtoEnt':>9} {'FG%':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*9} {'─'*8}")
    for ts in tile_sizes:
        if ts not in all_results: continue
        r = all_results[ts]
        print(f"  {ts:>4}×{ts:<4} {r['miou']:>8.4f} {r['n_bg_protos']:>5}/{args.n_protos}"
              f"{'':>4} {r['proto_entropy']:>8.4f} {r['fg_ratio']:>7.1%}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*9} {'─'*8}")

    # ── 最佳 | Best ──
    best_ts = max(all_results, key=lambda ts: all_results[ts]["miou"])
    print(f"\n  Best mIoU: tile_size={best_ts} ({all_results[best_ts]['miou']:.4f})")
    min_bg_ts = min(all_results, key=lambda ts: all_results[ts]["n_bg_protos"])
    print(f"  Fewest BG protos: tile_size={min_bg_ts} ({all_results[min_bg_ts]['n_bg_protos']}/{args.n_protos})")
    print(f"  → Sweet spot likely where mIoU high AND BG-Proto low")

    for ts in tile_sizes:
        if ts in all_results:
            recorder.record_metric(f"tile{ts}_miou", all_results[ts]["miou"], tags=["e011t"])
            recorder.record_metric(f"tile{ts}_bg", all_results[ts]["n_bg_protos"], tags=["e011t"])
    recorder.finalize(); recorder.close()
    print(f"\n  Results: {output_path}/")


if __name__ == "__main__":
    main()
