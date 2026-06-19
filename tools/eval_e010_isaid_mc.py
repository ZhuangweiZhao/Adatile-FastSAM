#!/usr/bin/env python3
"""
E010: iSAID 多分类 Proto vs Embedding 公平对照 | Fair Head-to-Head
====================================================================

E007-B 在 iSAID 15 类场景的复现。验证 Proto 机制是否在
多类别遥感分割中同样有效。

设计 | Design (同 E007-B, 固定变量):
    - 相同 Backbone (Frozen FastSAM P4)
    - 相同 embedding 维度 (128)
    - 相同训练配置 (lr, epochs, CosineLR, seed)
    - 近乎相同参数量 (Δ=256, 0.15%)
    - 仅 Head 机制不同:
        EmbeddingHead: 128→Conv(128→16)→logit         (165,904 params)
        ProtoHead:     128→CosineSim(16 protos)→Conv(16→16)→logit (166,160)

指标 | Metrics:
    - mIoU (主要)
    - Per-class IoU
    - Pixel Accuracy
    - Silhouette Score (embedding 空间聚类质量)

假设 | Hypothesis:
    H0: Proto 在多类别遥感中也保持或提升 mIoU
    H1: Proto 的 Silhouette 显著高于 Embedding (表示空间结构化)
    H2: 16 proto 自发形成类别语义分化

用法 | Usage:
    python tools/eval_e010_isaid_mc.py --max-tiles 200
    python tools/eval_e010_isaid_mc.py --n-protos 16 --epochs 30
"""

from __future__ import annotations
import sys, argparse, time
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

logger = get_logger("e010_isaid_mc")


def parse_args():
    p = argparse.ArgumentParser(description="E010: iSAID Multi-Class Proto vs Embedding")
    p.add_argument("--data-root", type=str, default="data/iSAID_tiles")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--max-tiles", type=int, default=200,
                   help="限制 tile 数 (0=全部) | Limit tiles for quick test")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e010_isaid_mc")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Heads (与 E007-B 相同的设计, 仅输出通道改为 15 类)
# ═══════════════════════════════════════════════════════════════════

class EmbeddingHead(nn.Module):
    """
    嵌入头 (Baseline) | Embedding Head.

    P4 → Conv(1280→D)→ReLU → Conv(D→C)→logit.
    无 Proto 约束。| No Proto constraint.
    """

    def __init__(self, in_channels=1280, embed_dim=128, num_classes=15):
        super().__init__()
        self.embed_dim = embed_dim
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, 1, bias=True)
        n = sum(p.numel() for p in self.parameters())
        logger.log_info("model/embed_init",
                        f"EmbeddingHead: {n:,} params, D={embed_dim}, C={num_classes}")

    def forward(self, p4):
        embedding = self.project(p4)
        logit = self.head(embedding)
        return embedding, logit


class ProtoHead(nn.Module):
    """
    原型头 | Proto Head.

    P4 → Conv(1280→D)→ReLU → CosineSim(N protos) → Conv(N→C)→logit.
    结构约束: 分割必须通过 N 个全局原型向量的 cosine similarity。
    """

    def __init__(self, in_channels=1280, embed_dim=128, n_protos=16, num_classes=15):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos
        self.num_classes = num_classes

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.head = nn.Conv2d(n_protos, num_classes, 1, bias=True)

        n = sum(p.numel() for p in self.parameters())
        logger.log_info("model/proto_init",
                        f"ProtoHead: {n:,} params, D={embed_dim}, "
                        f"N={n_protos}, C={num_classes}")

    def forward(self, p4, temperature=0.1):
        embedding = self.project(p4)
        emb_n = F.normalize(embedding, dim=1, p=2)
        proto_n = F.normalize(self.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_n, proto_n) / temperature
        logit = self.head(sim_maps)
        return embedding, sim_maps, logit

    def get_hard_assignment(self, p4, temperature=0.01):
        _, sim_maps, _ = self.forward(p4, temperature)
        return sim_maps.argmax(dim=1)  # [B, H, W]


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int) -> dict:
    """
    计算 mIoU + per-class IoU + pixel accuracy.
    pred:  [B, H, W] or [B, C, H, W]
    target: [B, H, W]
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    results = {}
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        iou = (inter + 1e-8) / (union + 1e-8)
        ious.append(iou.item())

    results["miou"] = float(np.mean(ious))
    results["pixel_acc"] = (pred == target).float().mean().item()
    return results


@torch.no_grad()
def compute_silhouette(embedding: torch.Tensor, target: torch.Tensor,
                       n_samples: int = 3000) -> float:
    """
    多类别 Silhouette Score | Multi-class Silhouette on embedding space.
    embedding: [1, D, H, W], target: [1, H, W] class labels.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return float("nan")

    D = embedding.shape[1]
    emb_sq = embedding.squeeze(0)  # [D, H_emb, W_emb] (stride 16)
    tgt = target.squeeze(0)        # [H_tgt, W_tgt]   (full resolution)

    # 下采样 target 到 embedding 分辨率 | Downsample target to embedding resolution
    tgt_down = F.interpolate(
        tgt.unsqueeze(0).unsqueeze(0).float(),
        size=(emb_sq.shape[1], emb_sq.shape[2]), mode="nearest"
    ).squeeze().long()  # [H_emb, W_emb]

    n_total = tgt_down.numel()
    idx = torch.randperm(n_total)[:n_samples]
    feats = emb_sq.reshape(D, -1).T[idx].cpu().numpy()
    labels = tgt_down.flatten()[idx].cpu().numpy()

    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return float(silhouette_score(feats, labels))


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_head(head, backbone, train_ds, val_ds, args, device, recorder,
               head_name: str, is_proto: bool) -> dict:
    """训练一个 Head 变体 | Train one head variant (同 E007-B 配方)."""
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    best_miou, best_state = 0.0, None
    best_epoch = 0

    logger.log_info("train/start",
                    f"[{head_name}] {args.epochs} epochs, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [{head_name}] {epoch}/{args.epochs}", leave=False)
        for idx in pbar:
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                features = backbone(image)
            p4 = features["p4"]

            if is_proto:
                _, _, logit = head(p4, temperature=args.temperature)
            else:
                _, logit = head(p4)

            logit_up = F.interpolate(logit, size=target.shape[1:],
                                     mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logit_up, target, ignore_index=255)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── Val ──
        head.eval()
        mious, accs = [], []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                target = sample["mask"].unsqueeze(0).to(device)

                features = backbone(image)
                p4 = features["p4"]
                if is_proto:
                    _, _, logit = head(p4, temperature=args.temperature)
                else:
                    _, logit = head(p4)

                logit_up = F.interpolate(logit, size=target.shape[1:],
                                         mode="bilinear", align_corners=False)
                pred = logit_up.argmax(dim=1)
                m = compute_metrics(pred, target, args.num_classes)
                mious.append(m["miou"])
                accs.append(m["pixel_acc"])

        miou = float(np.mean(mious))
        acc = float(np.mean(accs))
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch

        marker = " *" if is_best else ""
        logger.log_metric(f"miou/{head_name.lower()}", miou, step=epoch, tags=[head_name])
        logger.log_info("train/epoch",
                        f"[{head_name}] E{epoch:2d} loss={avg_loss:.4f} "
                        f"mIoU={miou:.4f} Acc={acc:.4f}{marker}")

        recorder.record_metric("loss", avg_loss, step=epoch, tags=[head_name, "train"])
        recorder.record_metric("miou", miou, step=epoch, tags=[head_name, "val"])

    if best_state is not None:
        head.load_state_dict(best_state)
    logger.log_info("train/best",
                    f"[{head_name}] Best mIoU={best_miou:.4f} (epoch {best_epoch})")
    return {"best_miou": best_miou, "best_epoch": best_epoch, "best_state": best_state}


# ═══════════════════════════════════════════════════════════════════
# Proto Semantic Analysis
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_proto_semantics(proto_head, backbone, val_ds, device, num_classes):
    """
    分析每个 Proto 的类别倾向 | Analyze per-proto class affinity.
    每个 proto 被分配到的像素中, 各类别占比。
    """
    proto_head.eval()
    n_protos = proto_head.n_protos
    proto_class_counts = np.zeros((n_protos, num_classes))

    for idx in range(min(20, len(val_ds))):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].to(device)

        features = backbone(image)
        hard = proto_head.get_hard_assignment(features["p4"])  # [1, H/16, W/16]

        # 上采样 target 到 P4 分辨率 | Downsample target to P4 resolution
        tgt_down = F.interpolate(
            target.unsqueeze(0).unsqueeze(0).float(),
            size=(hard.shape[1], hard.shape[2]), mode="nearest"
        ).squeeze().long()

        for p in range(n_protos):
            mask_p = (hard.squeeze(0) == p)
            if mask_p.sum() > 0:
                for c in range(num_classes):
                    proto_class_counts[p, c] += (tgt_down[mask_p] == c).sum().item()

    # 归一化到 [0,1] | Normalize
    row_sums = proto_class_counts.sum(axis=1, keepdims=True) + 1e-8
    proto_class_pct = proto_class_counts / row_sums
    return proto_class_pct, proto_class_counts


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print(f"  E010: iSAID Multi-Class Proto vs Embedding")
    print(f"  N={args.n_protos}, D={args.embed_dim}, C={args.num_classes}")
    print(f"  E007-B 多类别复现 | E007-B multi-class replication")
    print("=" * 70)

    set_seed(args.seed)

    exp_id = generate_exp_id(name=args.name)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="iSAID_tiles", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Backbone ──
    logger.log_info("phase", "[1] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── Data ──
    logger.log_info("phase", "[2] iSAID Tiles (pre-cut)")
    train_ds = FastISAIDTileDataset(root_dir=args.data_root, split="train", semantic=True)
    val_ds = FastISAIDTileDataset(root_dir=args.data_root, split="val", semantic=True)
    if args.max_tiles > 0:
        train_ds._tiles = train_ds._tiles[:args.max_tiles]
        val_ds._tiles = val_ds._tiles[:max(1, args.max_tiles // 4)]
    logger.log_info("data", f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles")

    # ── A: EmbeddingHead ──
    logger.log_info("phase", "[3A] EmbeddingHead (Baseline)")
    set_seed(args.seed)
    embed_head = EmbeddingHead(in_channels=1280, embed_dim=args.embed_dim,
                                num_classes=args.num_classes).to(device)
    embed_params = sum(p.numel() for p in embed_head.parameters())

    t0 = time.time()
    embed_results = train_head(embed_head, backbone, train_ds, val_ds,
                               args, device, recorder, "Embed", is_proto=False)
    embed_time = time.time() - t0

    # ── Embedding Silhouette ──
    embed_head.eval()
    embed_sils = []
    with torch.no_grad():
        for idx in range(min(8, len(val_ds))):
            sample = val_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["mask"].unsqueeze(0).to(device)
            features = backbone(image)
            emb, _ = embed_head(features["p4"])
            s = compute_silhouette(emb, target)
            if not np.isnan(s): embed_sils.append(s)
    embed_sil = float(np.mean(embed_sils)) if embed_sils else float("nan")

    # ── B: ProtoHead ──
    logger.log_info("phase", "[3B] ProtoHead (Treatment)")
    set_seed(args.seed)
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos,
                            num_classes=args.num_classes).to(device)
    proto_params = sum(p.numel() for p in proto_head.parameters())

    t0 = time.time()
    proto_results = train_head(proto_head, backbone, train_ds, val_ds,
                               args, device, recorder, "Proto", is_proto=True)
    proto_time = time.time() - t0

    # ── Proto Silhouette ──
    proto_sils = []
    with torch.no_grad():
        for idx in range(min(8, len(val_ds))):
            sample = val_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["mask"].unsqueeze(0).to(device)
            features = backbone(image)
            emb, _, _ = proto_head(features["p4"], temperature=args.temperature)
            s = compute_silhouette(emb, target)
            if not np.isnan(s): proto_sils.append(s)
    proto_sil = float(np.mean(proto_sils)) if proto_sils else float("nan")

    # ── Proto Semantic Analysis ──
    proto_cls_pct, _ = analyze_proto_semantics(
        proto_head, backbone, val_ds, device, args.num_classes
    )

    # ── Summary ──
    delta_miou = proto_results["best_miou"] - embed_results["best_miou"]
    delta_sil = proto_sil - embed_sil
    delta_params = proto_params - embed_params

    print(f"\n{'=' * 70}")
    print(f"  E010 Results | iSAID Proto vs Embedding")
    print(f"  {'=' * 70}")
    print(f"  {'Metric':<30} {'Embedding':>12} {'Proto':>12} {'Δ':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Parameters':<30} {embed_params:>12,} {proto_params:>12,} "
          f"{delta_params:>+12,} ({delta_params/embed_params*100:.1f}%)")
    print(f"  {'Best mIoU':<30} {embed_results['best_miou']:>12.4f} "
          f"{proto_results['best_miou']:>12.4f} {delta_miou:>+12.4f}")
    print(f"  {'Silhouette':<30} {embed_sil:>12.4f} "
          f"{proto_sil:>12.4f} {delta_sil:>+12.4f}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    # ── Per-class IoU comparison (on best models) ──
    print(f"\n  Per-Class IoU (Proto):")
    # Find dominant class per proto
    dominant = proto_cls_pct.argmax(axis=1)
    for p in range(args.n_protos):
        top3 = proto_cls_pct[p].argsort()[-3:][::-1]
        info = ", ".join(f"c{t}({proto_cls_pct[p,t]:.0%})" for t in top3 if proto_cls_pct[p,t] > 0.05)
        if info:
            print(f"    P{p:2d}: {info}")

    # ── Verdict ──
    print(f"\n  {'─'*60}")
    if abs(delta_miou) < 0.015 and delta_sil > 0.01:
        print(f"  ✅ PROTO HYPOTHESIS SUPPORTED (Multi-Class)")
        print(f"     mIoU 持平 (|Δ|={abs(delta_miou):.3f})")
        print(f"     Silhouette 显著更高 (Δ=+{delta_sil:.3f})")
        print(f"     → Proto 在多类别遥感中同样有效")
        verdict = "proto_supported_mc"
    elif delta_miou > 0.015:
        print(f"  ✅ PROTO WINS (Multi-Class)")
        print(f"     mIoU +{delta_miou:.4f}, Silhouette Δ={delta_sil:+.3f}")
        verdict = "proto_wins_mc"
    elif abs(delta_miou) < 0.015 and abs(delta_sil) <= 0.01:
        print(f"  △ NO SIGNIFICANT DIFFERENCE")
        print(f"     → Proto 在多类别中无显著优势")
        verdict = "no_difference_mc"
    else:
        verdict = "mixed_mc"
    print(f"  {'─'*60}")

    recorder.record_metric("embed_miou", embed_results["best_miou"], tags=["e010", "embed"])
    recorder.record_metric("proto_miou", proto_results["best_miou"], tags=["e010", "proto"])
    recorder.record_metric("delta_miou", delta_miou, tags=["e010"])
    recorder.record_metric("delta_silhouette", delta_sil, tags=["e010"])
    recorder.logger.log_info("e010/verdict", verdict, tags=["e010", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results: {output_path}/")


if __name__ == "__main__":
    main()
