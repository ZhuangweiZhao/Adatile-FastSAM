#!/usr/bin/env python3
"""
E011-U: Proto Capacity Scaling — N Proto 对 iSAID 多分类的影响
================================================================

E011-T 的补充实验: 固定 tile=1024, 扫描 N ∈ {2,4,8,16,32,64}

问题 | Question:
    Proto Collapse 是因为 Proto 数量不足 (Capacity Bottleneck)?
    还是 Proto 分配机制的问题 (Winner-Take-All)?

    如果 16→32 提升明显 → Capacity Bottleneck 成立
    如果 16→64 几乎不提升 → 不是容量问题

指标 | Metrics (per N):
    - mIoU
    - Effective Proto Number (1/Σp_i²)
    - BG-Dominant Proto count
    - Proto Entropy

用法 | Usage::
    python tools/eval_e011u_proto_capacity.py --max-tiles 2000 --epochs 30
"""

import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.logging import get_logger
from adatile.utils.seed import set_seed

logger = get_logger("e011u_capacity")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/iSAID_tiles")
    p.add_argument("--n-list", type=str, default="2,4,8,16,32,64")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tiles", type=int, default=2000)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e011u_capacity")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class ProtoHead(nn.Module):
    """
    多类别 ProtoHead | Multi-Class Proto Head (同 E010/E011-T 架构).

    P4 → Conv(1280→D)→ReLU → CosineSim(N protos) → Conv(N→C)→logit.
    用于 E011-U: 扫描 N ∈ {2,4,8,16,32,64}, 分析 Proto 容量瓶颈。
    """

    def __init__(self, in_channels=1280, embed_dim=128, n_protos=16, num_classes=15):
        super().__init__()
        self.n_protos = n_protos
        # 特征投影 | Feature projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False), nn.ReLU(inplace=True))
        # 可学习原型向量 | Learnable prototype vectors [N, D]
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        # 多类别分割头 | Multi-class segmentation head: N → C
        self.head = nn.Conv2d(n_protos, num_classes, 1, bias=True)

    def forward(self, p4, temperature=0.1):
        """
        前向传播 | Forward pass.

        :return: embedding: [B, D, H, W] 低维嵌入 sim_maps:  [B, N, H, W] proto 相似度图 logit:     [B, C, H, W] 多类别 logit
        """
        emb = self.project(p4)
        emb_n = F.normalize(emb, dim=1, p=2)
        p_n = F.normalize(self.prototypes, dim=1, p=2)
        sim = torch.einsum("bdhw,nd->bnhw", emb_n, p_n) / temperature
        return emb, sim, self.head(sim)


def train_proto(proto_head, backbone, train_ds, args, device):
    """
    训练 ProtoHead | Train ProtoHead.

    使用多类别交叉熵 (ignore_index=255) 训练。
    Validation 在 train subset 上快速评估 (节省全量 val 时间)。

    :return: best_miou: 最佳验证 mIoU
    """
    proto_head.train()
    opt = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)
    best_miou, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        proto_head.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [N={proto_head.n_protos}] E{epoch}/{args.epochs}",
                    leave=False)
        for idx in pbar:
            s = train_ds[idx]
            img = s["image"].unsqueeze(0).to(device)
            tgt = s["mask"].unsqueeze(0).to(device)
            with torch.no_grad(): feats = backbone(img)
            _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tgt.shape[1:],
                                     mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logit_up, tgt, ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
            pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
        sch.step()

        # Val mIoU
        proto_head.eval()
        mious = []
        with torch.no_grad():
            for idx in range(min(50, len(train_ds)//4)):
                s = train_ds[idx]  # quick val on train subset
                img = s["image"].unsqueeze(0).to(device)
                tgt = s["mask"].unsqueeze(0).to(device)
                feats = backbone(img)
                _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tgt.shape[1:],
                                         mode="bilinear", align_corners=False)
                pred = logit_up.argmax(dim=1)
                miou_v, valid = 0.0, 0
                for c in range(args.num_classes):
                    pc = (pred == c); tc = (tgt == c)
                    inter = (pc & tc).sum().float()
                    union = (pc | tc).sum().float()
                    if union > 0:
                        miou_v += (inter + 1e-8) / union; valid += 1
                if valid > 0: mious.append((miou_v / valid).item())
        miou_mean = float(np.mean(mious)) if mious else 0.0

        if miou_mean > best_miou:
            best_miou = miou_mean
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.log_info(f"train/N{proto_head.n_protos}",
                            f"E{epoch:2d} loss={total_loss/n:.4f} mIoU={miou_mean:.4f}")

    if best_state: proto_head.load_state_dict(best_state)
    return best_miou


@torch.no_grad()
def analyze_proto(proto_head, backbone, val_ds, device, args):
    """
    分析 Proto 使用效率 | Analyze proto utilization efficiency.

    Measures five dimensions:
      - mIoU:          分割质量 | segmentation quality
      - Eff N:         有效 Proto 数 (1/Σp_i², 逆 Herfindahl)
      - n_bg:          背景主导 Proto 数 (>50% 像素分配到 class 0)
      - n_active:      活跃 Proto 数 (总像素 >100)
      - entropy:       平均 Proto 类别熵

    :return: dict with keys: miou, eff_n, n_bg, n_active, entropy
    """
    proto_head.eval()
    n_p = proto_head.n_protos
    proto_class = np.zeros((n_p, args.num_classes))
    mious = []

    for idx in range(len(val_ds)):
        s = val_ds[idx]
        img = s["image"].unsqueeze(0).to(device)
        tgt = s["mask"].to(device)
        H, W = tgt.shape
        feats = backbone(img)

        # mIoU
        _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
        logit_up = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        pred = logit_up.argmax(dim=1)
        miou_v, valid = 0.0, 0
        for c in range(args.num_classes):
            pc = (pred == c); tc = (tgt.unsqueeze(0) == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            if union > 0: miou_v += (inter+1e-8)/union; valid += 1
        if valid > 0: mious.append((miou_v/valid).item())

        # Proto-class affinity (winner-take-all)
        hard = proto_head(feats["p4"])[1].argmax(dim=1)
        H_emb, W_emb = hard.shape[1], hard.shape[2]
        tgt_down = F.interpolate(tgt.unsqueeze(0).unsqueeze(0).float(),
                                 size=(H_emb, W_emb), mode="nearest").squeeze().long()
        for p in range(n_p):
            mask_p = (hard.squeeze(0) == p)
            if mask_p.sum() > 50:
                for c in range(args.num_classes):
                    proto_class[p, c] += (tgt_down[mask_p] == c).sum().item()

    miou = float(np.mean(mious))
    row_sums = proto_class.sum(axis=1, keepdims=True) + 1e-8
    pct = proto_class / row_sums  # [N, C] 每个 Proto 的类别分布 | per-proto class distribution

    # 有效 Proto 数: 1 / Σp_i² (逆 Herfindahl-Hirschman 指数)
    # Eff N: 1 / Σp_i² — inverse Herfindahl index. 衡量使用集中度.
    # 值接近 N → 均匀使用; 值小 → 少数 Proto 主导.
    usage = proto_class.sum(axis=1)  # 每个 Proto 的总像素 | total pixels per proto
    usage_p = usage / (usage.sum() + 1e-8)  # 归一化使用率 | normalized usage ratio
    eff_n = 1.0 / (usage_p**2).sum()

    # 背景主导 Proto: >50% 像素为背景 (class 0)
    n_bg = int((pct[:, 0] > 0.5).sum())
    # 活跃 Proto: 总像素超过 100 (有统计意义)
    n_active = int((usage > 100).sum())

    # Proto 类别熵 (越高 → Proto 越多样化)
    # Proto category entropy (higher → more diverse per-proto class assignment)
    ents = []
    for p in range(n_p):
        dist = pct[p] + 1e-8; dist /= dist.sum()
        ents.append(-(dist * np.log(dist)).sum())  # Shannon entropy
    active_ents = [e for p, e in enumerate(ents) if usage[p] > 100]
    ent_mean = float(np.mean(active_ents)) if active_ents else 0.0

    return {"miou": miou, "eff_n": eff_n, "n_bg": n_bg,
            "n_active": n_active, "entropy": ent_mean}


def main():
    args = parse_args()
    device = args.device
    n_list = [int(x.strip()) for x in args.n_list.split(",")]

    logger.log_info("exp/start",
                    f"E011-U Proto Capacity Scaling | N={n_list} "
                    f"C={args.num_classes} max_tiles={args.max_tiles}")

    set_seed(args.seed)

    exp_id = generate_exp_id(name=args.name)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="iSAID_tiles", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    train_ds = FastISAIDTileDataset(root_dir=args.data_root, split="train", dense_labels=True)
    val_ds = FastISAIDTileDataset(root_dir=args.data_root, split="val", dense_labels=True)
    if args.max_tiles > 0:
        train_ds._tiles = train_ds._tiles[:args.max_tiles]
        val_ds._tiles = val_ds._tiles[:max(1, args.max_tiles // 4)]
    logger.log_info("data", f"Train={len(train_ds)}, Val={len(val_ds)}")

    results = {}
    for n in n_list:
        logger.log_info("phase", f"Training N={n}")

        set_seed(args.seed)
        proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                                n_protos=n, num_classes=args.num_classes).to(device)
        n_params = sum(p.numel() for p in proto_head.parameters())
        logger.log_info(f"model/N{n}", f"{n_params:,} params")

        train_proto(proto_head, backbone, train_ds, args, device)
        r = analyze_proto(proto_head, backbone, val_ds, device, args)
        results[n] = r

        logger.log_info(f"result/N{n}",
                        f"mIoU={r['miou']:.4f} EffN={r['eff_n']:.1f} "
                        f"Active={r['n_active']}/{n} BG={r['n_bg']}/{n} "
                        f"Ent={r['entropy']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  E011-U: Proto Capacity Scaling")
    print(f"  {'N':<6} {'mIoU':>8} {'Eff N':>7} {'Active':>8} {'BG':>6} {'Entropy':>8}")
    for n in n_list:
        r = results[n]
        print(f"  N={n:<4} {r['miou']:>8.4f} {r['eff_n']:>7.1f} "
              f"{r['n_active']:>5}/{n:<3} {r['n_bg']:>4}/{n:<3} {r['entropy']:>8.3f}")

    for n in n_list:
        r = results[n]
        recorder.record_metric(f"N{n}_miou", r["miou"], tags=["e011u"])
        recorder.record_metric(f"N{n}_eff_n", r["eff_n"], tags=["e011u"])
    recorder.finalize(); recorder.close()
    logger.log_info("exp/done", f"Results: {output_path}/")


if __name__ == "__main__":
    main()
