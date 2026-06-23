#!/usr/bin/env python3
"""
E011: SPM Router on iSAID — 类别感知 Proto 路由
===================================================

核心问题 | Core question:
    SPM Router 能否缓解 iSAID 上的 Proto Collapse (12/16→背景)?

叙事升级 | Narrative upgrade:
    E010: 多类别下 Proto Winner-Take-All → 12/16 Proto 坍缩到背景
    E011: SPM Router 学习类别感知路由 → Proto 恢复类别分工

    如果 After SPM: 背景 Proto 从 12 降到 6 → SPM 不仅是稀疏化, 而是 Proto 修复器

两阶段设计 | Two-stage design (同 E009):
    Stage 1: 训练 ProtoHead (normal, same as E007-B)
             → 观察 Proto Collapse (E010 已完成)
    Stage 2: 冻结 Proto Dictionary, 只训练 SPM Router
             → Router 学习每个位置选择哪些 Proto

用法 | Usage::
    python tools/eval_e011_spm_isaid.py --max-tiles 200
"""

from __future__ import annotations
import sys, argparse, time, glob as _glob
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

logger = get_logger("e011_spm_isaid")


def parse_args():
    p = argparse.ArgumentParser(description="E011: SPM Router on iSAID")
    p.add_argument("--data-root", type=str, default="data/iSAID_tiles")
    p.add_argument("--epochs-s1", type=int, default=30)
    p.add_argument("--epochs-s2", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-router", type=float, default=3e-4)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--router-k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--proto-checkpoint", type=str, default=None)
    p.add_argument("--max-tiles", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e011_spm_isaid")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ProtoHead (Stage 1, same as E010)
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(nn.Module):
    """
    多类别 ProtoHead (Stage 1) | Multi-Class Proto Head (Stage 1).

    P4 → Conv(1280→D)→ReLU → CosineSim(N protos) → Conv(N→C)→logit.
    训练完成后冻结, 供 Stage 2 的 SPM Router 使用。
    After training, frozen and used by Stage 2 SPM Router.
    """

    def __init__(self, in_channels=1280, embed_dim=128, n_protos=16, num_classes=15):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos
        self.num_classes = num_classes
        # 特征投影 | Feature projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 可学习原型向量 | Learnable prototype vectors [N, D]
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        # 多类别分割头 | Multi-class segmentation head: N → C
        self.head = nn.Conv2d(n_protos, num_classes, 1, bias=True)

    def forward(self, p4, temperature=0.1):
        """
        标准前向 | Standard forward pass.

        :return: embedding: [B, D, H, W] 低维嵌入 | low-dim embedding sim_maps:  [B, N, H, W] proto 相似度图 | proto similarity maps logit:     [B, C, H, W] 多类别 logit | multi-class logit
        """
        embedding = self.project(p4)
        emb_n = F.normalize(embedding, dim=1, p=2)
        proto_n = F.normalize(self.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_n, proto_n) / temperature
        logit = self.head(sim_maps)
        return embedding, sim_maps, logit

    def get_hard_assignment(self, p4, temperature=0.01):
        """
        硬分配: 每个像素的 Winner Proto 索引 | Hard winner-take-all assignment.

        :return: [B, H/16, W/16] int64, 值 ∈ [0, N-1]
        """
        _, sim_maps, _ = self.forward(p4, temperature)
        return sim_maps.argmax(dim=1)


# ═══════════════════════════════════════════════════════════════════
# SPMHead = Frozen Proto + Trainable SPM Router (Stage 2)
# ═══════════════════════════════════════════════════════════════════

class SPMHead(nn.Module):
    """
    冻结 Proto + 可训练 Router | Frozen Proto Dictionary + Trainable Router.

    Router 架构: Conv(128→64, 3×3)→ReLU→Conv(64→N, 1×1)
    3×3 conv 提供空间上下文感知路由 | Spatial context-aware routing.
    """

    def __init__(self, proto_head: ProtoHead, n_protos: int, router_k: int = 4):
        super().__init__()
        self.proto_head = proto_head
        self.n_protos = n_protos
        self.router_k = router_k

        for p in self.proto_head.parameters():
            p.requires_grad = False

        D = proto_head.embed_dim
        mid = max(32, D // 2)
        self.router = nn.Sequential(
            nn.Conv2d(D, mid, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_protos, 1, bias=True),
        )

        n_proto = sum(p.numel() for p in self.proto_head.parameters())
        n_router = sum(p.numel() for p in self.router.parameters())
        logger.log_info("model/spm_init",
                        f"SPMHead: {n_proto + n_router:,} params "
                        f"(Proto={n_proto:,} frozen, Router={n_router:,} trainable, K={router_k})")

    @torch.no_grad()
    def _get_sim_maps(self, p4, temperature=0.1):
        embedding = self.proto_head.project(p4)
        emb_n = F.normalize(embedding, dim=1, p=2)
        proto_n = F.normalize(self.proto_head.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_n, proto_n) / temperature
        return embedding, sim_maps

    def forward_full(self, p4, temperature=0.1):
        emb, sim = self._get_sim_maps(p4, temperature)
        logit = self.proto_head.head(sim)
        return logit, sim, emb

    def forward_routed(self, p4, temperature=0.1, k=None):
        if k is None:
            k = self.router_k
        emb, sim = self._get_sim_maps(p4, temperature)
        B, N, H, W = sim.shape

        if k >= N:
            return self.proto_head.head(sim), sim, emb, None

        router_logits = self.router(emb)

        if self.training:
            # Straight-Through Estimator
            r_flat = router_logits.permute(0, 2, 3, 1).reshape(-1, N)
            _, topk_idx = r_flat.topk(k, dim=1)
            m_hard = torch.zeros_like(r_flat).scatter_(1, topk_idx, 1.0)
            m_hard = m_hard.reshape(B, H, W, N).permute(0, 3, 1, 2)
            m_soft = F.softmax(router_logits, dim=1)
            mask = m_hard - m_soft.detach() + m_soft
        else:
            r_flat = router_logits.permute(0, 2, 3, 1).reshape(-1, N)
            _, topk_idx = r_flat.topk(k, dim=1)
            m = torch.zeros_like(r_flat).scatter_(1, topk_idx, 1.0)
            mask = m.reshape(B, H, W, N).permute(0, 3, 1, 2)

        sim_s = sim * mask
        logit = self.proto_head.head(sim_s)
        return logit, sim, emb, router_logits


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_miou(pred, target, num_classes):
    """
    计算 mIoU | Compute mean IoU.

    :param pred: [B, H, W] or [B, C, H, W] 预测 | prediction

    :param target: [B, H, W] GT 类别标签 | GT class labels

    :param num_classes: 类别数 | number of classes

    :return: float: mIoU averaged over valid classes (union > 0).
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    ious = [(target == c).float().sum() for c in range(num_classes)]
    miou = 0.0
    valid = 0
    for c in range(num_classes):
        pc = (pred == c)
        tc = (target == c)
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        if union > 0:
            miou += (inter + 1e-8) / union
            valid += 1
    return float(miou / max(valid, 1))


@torch.no_grad()
def analyze_proto_utilization(model, backbone, val_ds, device, num_classes, n_protos,
                              is_spm: bool, router_k: int = 4):
    """
    分析 Proto 类别亲和力 (SPM 前后对比) | Per-proto class affinity (before/after SPM).

    两种模式 | Two modes:
      - is_spm=False: Proto only — winner-take-all (每个像素选最相似的 Proto)
      - is_spm=True:  SPM — Top-K Router 选择 (每个像素可激活多个 Proto)

    :return: pct:         [N, C] 归一化类别占比 | normalized class proportions proto_class: [N, C] 原始计数 | raw counts n_active:    活跃 Proto 数 (总像素 > 100) | active protos n_bg:        背景主导 Proto 数 (>50% 为 class 0) | BG-dominant protos
    """
    model.eval()
    proto_class = np.zeros((n_protos, num_classes))

    # 采样最多 20 张验证图 | Sample at most 20 val images
    for idx in range(min(20, len(val_ds))):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].to(device)  # [H, W] class labels

        features = backbone(image)
        p4 = features["p4"]

        if is_spm:
            # SPM 模式: Router 选择 Top-K Proto (每个像素可激活多个 Proto)
            # SPM mode: Router selects Top-K protos (multi-proto per pixel)
            _, sim, emb, rl = model.forward_routed(p4, k=router_k)
            # 构建激活矩阵: 哪些 Proto 在每个像素被选中
            # Build activation matrix: which protos are selected per pixel
            r_flat = rl.permute(0, 2, 3, 1).reshape(-1, n_protos)
            _, topk_idx = r_flat.topk(router_k, dim=1)
            H_emb, W_emb = rl.shape[2], rl.shape[3]
            proto_active = torch.zeros(n_protos, H_emb * W_emb, device=device)
            for k_i in range(router_k):
                active_p = topk_idx[:, k_i]  # [H_emb*W_emb] — proto indices per pixel
                proto_active[active_p, torch.arange(H_emb * W_emb, device=device)] = 1.0
        else:
            # Proto only 模式: Winner-Take-All (每个像素只选一个 Proto)
            # Proto only mode: Winner-take-all (each pixel picks exactly one proto)
            hard = model.get_hard_assignment(p4)
            H_emb, W_emb = hard.shape[1], hard.shape[2]
            proto_active = F.one_hot(hard.squeeze(0).flatten(), num_classes=n_protos).float().T

        # 下采样 target 到 Proto 分配分辨率 | Downsample target to proto assignment resolution
        tgt_down = F.interpolate(
            target.unsqueeze(0).unsqueeze(0).float(),
            size=(H_emb, W_emb), mode="nearest"
        ).squeeze().long().flatten()

        # 累积每个 Proto 的类别计数 | Accumulate per-proto class counts
        for p in range(n_protos):
            mask_p = proto_active[p] > 0.5  # 该 Proto 激活的像素 | pixels where proto p is active
            if mask_p.sum() > 50:  # 至少 50 像素才统计 | require at least 50 pixels
                for c in range(num_classes):
                    proto_class[p, c] += (tgt_down[mask_p] == c).sum().item()

    # 归一化 | Normalize
    pct = proto_class / (proto_class.sum(axis=1, keepdims=True) + 1e-8)
    # 活跃 Proto 数 (总像素 >100, 有统计意义 | statistically meaningful)
    n_active = (proto_class.sum(axis=1) > 100).sum()
    # 背景主导 Proto 数 (>50% 像素为 class 0 | >50% pixels are background)
    n_bg = (pct[:, 0] > 0.5).sum()
    return pct, proto_class, n_active, n_bg


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_stage1(proto_head, backbone, train_ds, val_ds, args, device, recorder):
    """
    Stage 1: 标准 ProtoHead 训练 | Stage 1: Train ProtoHead normally.

    使用所有 N 个 Proto 进行训练。收敛后 Proto Dictionary 将被冻结。
    Trained with all N protos active. Proto Dictionary frozen after convergence.

    :return: best_miou: Best validation mIoU.
    """
    proto_head.train()
    opt = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s1, eta_min=1e-6)
    best_miou, best_state = 0.0, None

    for epoch in range(1, args.epochs_s1 + 1):
        proto_head.train()
        total_loss, n = 0.0, 0
        for idx in tqdm(range(len(train_ds)), desc=f"  [S1] {epoch}/{args.epochs_s1}", leave=False):
            s = train_ds[idx]
            img = s["image"].unsqueeze(0).to(device)
            tgt = s["mask"].unsqueeze(0).to(device)
            with torch.no_grad():
                feats = backbone(img)
            _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tgt.shape[1:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logit_up, tgt, ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        sch.step()

        # Val
        proto_head.eval()
        mious = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                s = val_ds[idx]
                img = s["image"].unsqueeze(0).to(device)
                tgt = s["mask"].unsqueeze(0).to(device)
                feats = backbone(img)
                _, _, logit = proto_head(feats["p4"], temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tgt.shape[1:], mode="bilinear", align_corners=False)
                mious.append(compute_miou(logit_up.argmax(dim=1), tgt, args.num_classes))
        miou = float(np.mean(mious))
        if miou > best_miou:
            best_miou = miou
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}
        print(f"    [S1] E{epoch:2d} loss={total_loss/n:.4f} mIoU={miou:.4f}{' *' if miou >= best_miou else ''}")
        recorder.record_metric("s1_miou", miou, step=epoch, tags=["s1"])

    if best_state: proto_head.load_state_dict(best_state)
    logger.log_info("s1/best", f"Stage1 best mIoU={best_miou:.4f}")
    return best_miou


def train_stage2(spm_head, backbone, train_ds, val_ds, args, device, recorder):
    """
    Stage 2: 只训练 Router (冻结 Proto) | Stage 2: Train Router only (frozen Proto).

    Router 学习类别感知路由, 减少背景 Proto 坍缩。
    Router learns category-aware routing to reduce background proto collapse.

    Key: BCE + 熵正则化 — BCE 驱动 Router 选择有助分割的 Proto;
         熵项防止 Router 坍缩到单一 Proto。
         BCE + entropy regularization — BCE drives useful proto selection;
         entropy term prevents collapse to a single proto.

    :return: best_miou: Best validation mIoU with SPM routing.
    """
    spm_head.train()
    opt = torch.optim.Adam(spm_head.router.parameters(), lr=args.lr_router)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s2,
                                                      eta_min=args.lr_router * 0.01)
    best_miou, best_state = 0.0, None
    k = args.router_k

    for epoch in range(1, args.epochs_s2 + 1):
        spm_head.train()
        total_loss, total_ent, n = 0.0, 0.0, 0
        for idx in tqdm(range(len(train_ds)), desc=f"  [S2] {epoch}/{args.epochs_s2}", leave=False):
            s = train_ds[idx]
            img = s["image"].unsqueeze(0).to(device)
            tgt = s["mask"].unsqueeze(0).to(device)
            with torch.no_grad():
                feats = backbone(img)

            # 路由前向 | Routed forward
            logit, _, _, rl = spm_head.forward_routed(feats["p4"], k=k, temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tgt.shape[1:], mode="bilinear", align_corners=False)
            bce = F.cross_entropy(logit_up, tgt, ignore_index=255)
            # 熵正则化: 防止 Router 坍缩到单一 Proto
            # Entropy regularization: prevent router collapse to a single proto
            if rl is not None:
                probs = F.softmax(rl, dim=1)  # [B, N, H, W] → softmax over N
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()  # mean entropy per pixel
                loss = bce + args.entropy_weight * ent
                total_ent += ent.item()
            else:
                loss = bce
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        sch.step()

        # Val
        spm_head.eval()
        mious = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                s = val_ds[idx]
                img = s["image"].unsqueeze(0).to(device)
                tgt = s["mask"].unsqueeze(0).to(device)
                feats = backbone(img)
                logit, _, _, _ = spm_head.forward_routed(feats["p4"], k=k, temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tgt.shape[1:], mode="bilinear", align_corners=False)
                mious.append(compute_miou(logit_up.argmax(dim=1), tgt, args.num_classes))
        miou = float(np.mean(mious))
        if miou > best_miou:
            best_miou = miou
            best_state = {k: v.clone() for k, v in spm_head.router.state_dict().items()}
        print(f"    [S2] E{epoch:2d} loss={total_loss/n:.4f} ent={total_ent/n:.4f} "
              f"mIoU={miou:.4f}{' *' if miou >= best_miou else ''}")
        recorder.record_metric("s2_miou", miou, step=epoch, tags=["s2"])

    if best_state: spm_head.router.load_state_dict(best_state)
    logger.log_info("s2/best", f"Stage2 best mIoU={best_miou:.4f}")
    return best_miou


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print("  E011: SPM Router on iSAID — Category-Aware Proto Routing")
    print("  N={}, K={}, C={}".format(args.n_protos, args.router_k, args.num_classes))
    print("=" * 70)

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

    train_ds = FastISAIDTileDataset(root_dir=args.data_root, split="train", semantic=True)
    val_ds = FastISAIDTileDataset(root_dir=args.data_root, split="val", semantic=True)
    if args.max_tiles > 0:
        train_ds._tiles = train_ds._tiles[:args.max_tiles]
        val_ds._tiles = val_ds._tiles[:max(1, args.max_tiles // 4)]
    logger.log_info("data", f"Train={len(train_ds)}, Val={len(val_ds)} tiles")

    # ── Stage 1: ProtoHead ──
    logger.log_info("phase", "[Stage 1] ProtoHead training")
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos,
                            num_classes=args.num_classes).to(device)

    if args.proto_checkpoint:
        ckpt = args.proto_checkpoint
        if "*" in ckpt: ckpt = _glob.glob(ckpt)[0]
        proto_head.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        logger.log_info("s1", f"Loaded ProtoHead from {ckpt}")
    else:
        train_stage1(proto_head, backbone, train_ds, val_ds, args, device, recorder)
        torch.save(proto_head.state_dict(), output_path / "proto_head_s1.pt")

    # ── BEFORE: Proto utilization (no SPM) ──
    logger.log_info("phase", "[Analysis] Proto utilization BEFORE SPM")
    pct_before, _, n_active_before, n_bg_before = analyze_proto_utilization(
        proto_head, backbone, val_ds, device, args.num_classes, args.n_protos,
        is_spm=False
    )
    logger.log_info("before", f"Active={int(n_active_before)}/{args.n_protos}, "
                    f"BG-dominant={int(n_bg_before)}/{args.n_protos}")

    # ── Stage 2: SPM Router ──
    logger.log_info("phase", f"[Stage 2] SPM Router training (K={args.router_k})")
    spm_head = SPMHead(proto_head, n_protos=args.n_protos, router_k=args.router_k).to(device)
    train_stage2(spm_head, backbone, train_ds, val_ds, args, device, recorder)
    torch.save({"proto": proto_head.state_dict(), "router": spm_head.router.state_dict()},
               output_path / "spm_head_s2.pt")

    # ── AFTER: Proto utilization (with SPM) ──
    logger.log_info("phase", "[Analysis] Proto utilization AFTER SPM")
    pct_after, _, n_active_after, n_bg_after = analyze_proto_utilization(
        spm_head, backbone, val_ds, device, args.num_classes, args.n_protos,
        is_spm=True, router_k=args.router_k
    )
    logger.log_info("after", f"Active={int(n_active_after)}/{args.n_protos}, "
                    f"BG-dominant={int(n_bg_after)}/{args.n_protos}")

    # ── Summary ──
    delta_bg = n_bg_before - n_bg_after
    print(f"\n{'=' * 70}")
    print(f"  E011 Results | Proto Collapse Before vs After SPM")
    print(f"  {'=' * 70}")
    print(f"  {'Metric':<30} {'Before (Proto)':>15} {'After (SPM)':>15}")
    print(f"  {'─'*30} {'─'*15} {'─'*15}")
    print(f"  {'Active Protos':<30} {n_active_before:>14}/{args.n_protos} "
          f"{n_active_after:>14}/{args.n_protos}")
    print(f"  {'BG-Dominant Protos':<30} {n_bg_before:>14}/{args.n_protos} "
          f"{n_bg_after:>14}/{args.n_protos} (Δ={delta_bg:+d})")

    # 逐 Proto 前后对比 | Per-proto before/after comparison
    # 显示每个 Proto 在 SPM 前后的主要类别变化
    # Show each proto's dominant class change before vs after SPM
    for p in range(args.n_protos):
        top_b = pct_before[p].argsort()[-2:][::-1]  # 前 2 类别 (Before)
        top_a = pct_after[p].argsort()[-2:][::-1]   # 前 2 类别 (After)
        b_str = "/".join(f"c{t}({pct_before[p,t]:.0%})" for t in top_b if pct_before[p,t] > 0.05)
        a_str = "/".join(f"c{t}({pct_after[p,t]:.0%})" for t in top_a if pct_after[p,t] > 0.05)
        bg_before = "BG" if pct_before[p, 0] > 0.5 else "  "
        bg_after = "BG" if pct_after[p, 0] > 0.5 else "  "
        arrow = "→" if bg_before != bg_after else " "  # 显示状态转移 | show state transition
        if b_str or a_str:
            print(f"    P{p:2d}: [{bg_before}] {b_str:<20s} {arrow} [{bg_after}] {a_str}")

    print(f"\n  {'─'*60}")
    if delta_bg >= 3:
        print(f"  ✅ SPM REDUCES PROTO COLLAPSE")
        print(f"     BG protos: {n_bg_before} → {n_bg_after} (Δ={delta_bg})")
        print(f"     → SPM 学到了类别感知路由, 修复了 Proto 坍缩")
        verdict = "spm_reduces_collapse"
    elif delta_bg >= 1:
        print(f"  △ MILD IMPROVEMENT (Δ={delta_bg})")
        verdict = "mild_improvement"
    else:
        print(f"  → No reduction in BG-dominant protos")
        verdict = "no_reduction"
    print(f"  {'─'*60}")

    recorder.record_metric("bg_before", int(n_bg_before), tags=["e011"])
    recorder.record_metric("bg_after", int(n_bg_after), tags=["e011"])
    recorder.record_metric("delta_bg", int(delta_bg), tags=["e011"])
    recorder.logger.log_info("e011/verdict", verdict)
    recorder.finalize(); recorder.close()
    print(f"\n  Results: {output_path}/")


if __name__ == "__main__":
    main()
