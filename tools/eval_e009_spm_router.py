#!/usr/bin/env python3
"""
E009: SPM Router — 两阶段学习式路由 | Two-Stage Learned Routing
==================================================================

核心问题 | Core question:
    学习式 Proto 路由 (SPM Router) 是否优于固定 |w·sim| 路由?

两阶段设计 | Two-Stage Design:
    Stage 1: 训练 ProtoHead (full, 同 E007-B)
             P4 → Embedding → CosineSim(N protos) → Head(N→1) → logit
             Dice ≈ 0.466
    Stage 2: 冻结 Proto Dictionary (project + prototypes + head)
             只训练 Router

             Embedding (frozen)
                 ├── CosineSim(N protos) → sim_maps (frozen)
                 └── Router (trainable) → routing_logits
                          ↓
                 Straight-Through Top-K → sparse sim_maps
                          ↓
                 Head (frozen) → logit

    变量唯一 | Only variable: routing mechanism.

三种路由模式对比 | Three routing modes compared:
    Learned (SPM):   Router(embedding) → Top-K
    Fixed (|w·sim|): |head_w · sim| → Top-K
    Sim (raw):       sim → Top-K (reference)

用法 | Usage:
    python tools/eval_e009_spm_router.py
    python tools/eval_e009_spm_router.py --proto-checkpoint path/to/proto_head.pt
"""

from __future__ import annotations
import sys, argparse, glob as _glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice
from adatile.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="E009: Two-Stage SPM Router")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs-s1", type=int, default=30,
                   help="Stage 1 epochs (ProtoHead training)")
    p.add_argument("--epochs-s2", type=int, default=20,
                   help="Stage 2 epochs (Router training)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-router", type=float, default=3e-4,
                   help="LR for Router (Stage 2)")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--router-k", type=int, default=4,
                   help="K for learned router (sweet spot)")
    p.add_argument("--router-arch", type=str, default="conv3x3",
                   choices=["conv3x3", "conv1x1", "mlp"],
                   help="Router architecture: conv3x3 (spatial context), "
                        "conv1x1 (per-pixel linear), mlp (per-pixel 2-layer)")
    p.add_argument("--compare-ks", type=str, default="2,3,4,6",
                   help="Comma-separated K values for comparison")
    p.add_argument("--entropy-weight", type=float, default=0.05,
                   help="Entropy regularization weight")
    p.add_argument("--proto-checkpoint", type=str, default=None,
                   help="Pre-trained ProtoHead checkpoint (skip Stage 1)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e009_spm")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ProtoHead (Stage 1 — same as E007-B)
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(nn.Module):
    """原型头 (Stage 1: fully trained, then frozen in Stage 2)."""

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128, n_protos: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.head = nn.Conv2d(n_protos, 1, kernel_size=1, bias=True)

    def forward(self, p4: torch.Tensor, temperature: float = 0.1):
        """Returns: embedding, sim_maps, logit."""
        embedding = self.project(p4)
        emb_norm = F.normalize(embedding, dim=1, p=2)
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm) / temperature
        logit = self.head(sim_maps)
        return embedding, sim_maps, logit


# ═══════════════════════════════════════════════════════════════════
# SPMHead = Frozen ProtoHead + Trainable Router (Stage 2)
# ═══════════════════════════════════════════════════════════════════

class SPMHead(nn.Module):
    """
    SPM Head: Frozen Proto Dictionary + Learnable Router.

    ProtoHead (project, prototypes, head) is frozen after Stage 1.
    Router is the ONLY trainable component in Stage 2.
    """

    def __init__(self, proto_head: ProtoHead, n_protos: int, router_k: int = 4,
                 router_arch: str = "conv3x3"):
        super().__init__()
        self.proto_head = proto_head
        self.n_protos = n_protos
        self.router_k = router_k
        self.router_arch = router_arch

        # Freeze Proto Dictionary | 冻结 Proto 字典
        for p in self.proto_head.parameters():
            p.requires_grad = False

        # Router: architecture variants
        embed_dim = proto_head.embed_dim
        mid_dim = max(32, embed_dim // 2)

        if router_arch == "conv3x3":
            # 3×3 Conv: spatial context aware (current best)
            self.router = nn.Sequential(
                nn.Conv2d(embed_dim, mid_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, n_protos, kernel_size=1, bias=True),
            )
        elif router_arch == "mlp":
            # MLP: 2-layer per-pixel, no spatial context
            self.router = nn.Sequential(
                nn.Conv2d(embed_dim, mid_dim, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, n_protos, kernel_size=1, bias=True),
            )
        elif router_arch == "conv1x1":
            # 1×1 Conv: single linear layer per-pixel
            self.router = nn.Conv2d(embed_dim, n_protos, kernel_size=1, bias=True)
        else:
            raise ValueError(f"Unknown router_arch: {router_arch}")

        self._report_params()

    def _report_params(self):
        n_proto = sum(p.numel() for p in self.proto_head.parameters())
        n_router = sum(p.numel() for p in self.router.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPMHead: {n_proto + n_router:,} total")
        print(f"    Proto Dict (frozen): {n_proto:,}")
        print(f"    Router (trainable):  {n_router:,} ({n_trainable:,})")

    @torch.no_grad()
    def _get_sim_maps(self, p4: torch.Tensor, temperature: float):
        """Get sim_maps from frozen ProtoHead."""
        embedding = self.proto_head.project(p4)
        emb_norm = F.normalize(embedding, dim=1, p=2)
        proto_norm = F.normalize(self.proto_head.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm) / temperature
        return embedding, sim_maps

    def forward_full(self, p4: torch.Tensor, temperature: float = 0.1):
        """Full forward (no routing) — reference."""
        embedding, sim_maps = self._get_sim_maps(p4, temperature)
        logit = self.proto_head.head(sim_maps)
        return logit, sim_maps, embedding

    def forward_routed(self, p4: torch.Tensor, temperature: float = 0.1,
                       mode: str = "learned", k: int = None):
        """
        路由前向 | Routed forward.

        Args:
            mode: "learned" = SPM Router, "fixed" = |w·sim|, "sim" = raw sim
            k:    override router_k (for multi-K comparison)
        """
        if k is None:
            k = self.router_k
        embedding, sim_maps = self._get_sim_maps(p4, temperature)
        B, N, H, W = sim_maps.shape

        if k >= N:
            logit = self.proto_head.head(sim_maps)
            return logit, sim_maps, embedding, None

        # ── Build routing mask ──
        if mode == "learned":
            router_logits = self.router(embedding)  # [B, N, H, W]

            if self.training:
                # Straight-Through Estimator
                router_flat = router_logits.permute(0, 2, 3, 1).reshape(-1, N)
                _, topk_idx = router_flat.topk(k, dim=1)
                mask_hard_flat = torch.zeros_like(router_flat).scatter_(1, topk_idx, 1.0)
                mask_hard = mask_hard_flat.reshape(B, H, W, N).permute(0, 3, 1, 2)
                mask_soft = F.softmax(router_logits, dim=1)
                mask = mask_hard - mask_soft.detach() + mask_soft
            else:
                # Hard Top-K at inference
                router_flat = router_logits.permute(0, 2, 3, 1).reshape(-1, N)
                _, topk_idx = router_flat.topk(k, dim=1)
                mask_flat = torch.zeros_like(router_flat).scatter_(1, topk_idx, 1.0)
                mask = mask_flat.reshape(B, H, W, N).permute(0, 3, 1, 2)

        elif mode == "fixed":
            head_w = self.proto_head.head.weight.squeeze()
            sim_flat = sim_maps.permute(0, 2, 3, 1).reshape(-1, N)
            importance = (sim_flat * head_w.unsqueeze(0)).abs()
            _, topk_idx = importance.topk(k, dim=1)
            mask_flat = torch.zeros_like(sim_flat).scatter_(1, topk_idx, 1.0)
            mask = mask_flat.reshape(B, H, W, N).permute(0, 3, 1, 2)
            router_logits = None

        elif mode == "sim":
            sim_flat = sim_maps.permute(0, 2, 3, 1).reshape(-1, N)
            _, topk_idx = sim_flat.topk(k, dim=1)
            mask_flat = torch.zeros_like(sim_flat).scatter_(1, topk_idx, 1.0)
            mask = mask_flat.reshape(B, H, W, N).permute(0, 3, 1, 2)
            router_logits = None

        else:
            raise ValueError(f"Unknown mode: {mode}")

        sim_sparse = sim_maps * mask
        logit = self.proto_head.head(sim_sparse)

        return logit, sim_maps, embedding, router_logits


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Train ProtoHead
# ═══════════════════════════════════════════════════════════════════

def train_stage1(proto_head, backbone, train_ds, val_ds, args, device, recorder):
    """训练 ProtoHead (同 E007-B 配置)."""
    proto_head.train()
    optimizer = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_s1, eta_min=1e-6)
    best_dice, best_state = 0.0, None

    print(f"  [S1] Training {args.epochs_s1} epochs (FULL proto, lr={args.lr})...")
    for epoch in range(1, args.epochs_s1 + 1):
        proto_head.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)), desc=f"  [S1] Epoch {epoch}/{args.epochs_s1}",
                    leave=False)
        for idx in pbar:
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["masks"].to(device)
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            elif gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(0).squeeze(0)
            with torch.no_grad():
                features = backbone(image)
            _, _, logit = proto_head(features["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})
        scheduler.step()

        proto_head.eval()
        dices = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(0).squeeze(0)
                features = backbone(image)
                _, _, logit = proto_head(features["p4"], temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                if pred.dim() == 2: pred = pred.unsqueeze(0)
                gm = gt_mask
                if gm.dim() == 2: gm = gm.unsqueeze(0)
                dices.append(compute_dice(pred, gm).item())

        dice_mean = float(np.mean(dices))
        is_best = dice_mean > best_dice
        if is_best:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}
        marker = " *" if is_best else ""
        print(f"    [S1] Epoch {epoch:2d}/{args.epochs_s1}  "
              f"loss={total_loss/len(train_ds):.4f}  Dice={dice_mean:.4f}{marker}")
        recorder.record_metric("s1/dice", dice_mean, step=epoch, phase="val")

    if best_state is not None:
        proto_head.load_state_dict(best_state)
    print(f"  [S1] Best Dice: {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Train Router (frozen ProtoHead)
# ═══════════════════════════════════════════════════════════════════

def train_stage2(spm_head, backbone, train_ds, val_ds, args, device, recorder):
    """训练 Router (Proto Dictionary 已冻结)."""
    spm_head.train()  # only router is trainable

    # Only router params | 只优化 Router 参数
    optimizer = torch.optim.Adam(spm_head.router.parameters(), lr=args.lr_router)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_s2, eta_min=args.lr_router * 0.01)
    best_dice, best_state = 0.0, None

    k = args.router_k
    ew = args.entropy_weight

    print(f"  [S2] Training {args.epochs_s2} epochs "
          f"(Router K={k}/{args.n_protos}, lr={args.lr_router}, entropy_w={ew})...")

    for epoch in range(1, args.epochs_s2 + 1):
        spm_head.train()
        total_bce, total_ent = 0.0, 0.0
        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [S2] Epoch {epoch}/{args.epochs_s2}", leave=False)
        for idx in pbar:
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["masks"].to(device)
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            elif gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(0).squeeze(0)
            with torch.no_grad():
                features = backbone(image)

            logit, _, _, router_logits = spm_head.forward_routed(
                features["p4"], temperature=args.temperature, mode="learned", k=k)
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            bce = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)

            # Entropy regularization (prevents router collapse)
            if router_logits is not None:
                probs = F.softmax(router_logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                ent_val = entropy.item()
                loss = bce + ew * entropy
            else:
                ent_val = 0.0
                loss = bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_bce += bce.item()
            total_ent += ent_val
            avg_loss = (total_bce + ew * total_ent) / (idx + 1)
            pbar.set_postfix({"bce": f"{total_bce/(idx+1):.4f}",
                              "ent": f"{ent_val:.4f}",
                              "loss": f"{avg_loss:.4f}"})
        scheduler.step()

        # ── Val ──
        spm_head.eval()
        dice_learned, dice_fixed, dice_full = [], [], []
        ent_vals = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(0).squeeze(0)
                features = backbone(image)
                p4 = features["p4"]

                # Full
                logit_f, _, _ = spm_head.forward_full(p4, temperature=args.temperature)
                # Learned
                logit_l, _, _, rl = spm_head.forward_routed(
                    p4, temperature=args.temperature, mode="learned", k=k)
                # Fixed
                logit_x, _, _, _ = spm_head.forward_routed(
                    p4, temperature=args.temperature, mode="fixed", k=k)

                for logit, dlist in [(logit_f, dice_full), (logit_l, dice_learned),
                                     (logit_x, dice_fixed)]:
                    logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                             mode="bilinear", align_corners=False)
                    pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                    if pred.dim() == 2: pred = pred.unsqueeze(0)
                    gm = gt_mask
                    if gm.dim() == 2: gm = gm.unsqueeze(0)
                    dlist.append(compute_dice(pred, gm).item())

                if rl is not None:
                    probs = F.softmax(rl, dim=1)
                    ent_vals.append(
                        -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item())

        dl = float(np.mean(dice_learned))
        df = float(np.mean(dice_fixed))
        dfull = float(np.mean(dice_full))
        ent_m = float(np.mean(ent_vals)) if ent_vals else 0.0

        is_best = dl > best_dice
        if is_best:
            best_dice = dl
            best_state = {k: v.clone() for k, v in spm_head.router.state_dict().items()}
        marker = " *" if is_best else ""
        print(f"    [S2] Epoch {epoch:2d}/{args.epochs_s2}  "
              f"bce={total_bce/len(train_ds):.4f}  "
              f"Learned={dl:.4f}  Fixed={df:.4f}  Full={dfull:.4f}  "
              f"ent={ent_m:.4f}{marker}")

        recorder.record_metric("s2/dice_learned", dl, step=epoch, phase="val")
        recorder.record_metric("s2/dice_fixed", df, step=epoch, phase="val")
        recorder.record_metric("s2/dice_full", dfull, step=epoch, phase="val")
        recorder.record_metric("s2/entropy", ent_m, step=epoch, phase="val")

    if best_state is not None:
        spm_head.router.load_state_dict(best_state)
    print(f"  [S2] Best Dice (learned K={k}): {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Comprehensive Comparison
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compare_routing(spm_head, backbone, val_ds, device, args):
    """三模式 × 多K 对比 | Three modes × multi-K comparison."""
    spm_head.eval()
    compare_ks = [int(x.strip()) for x in args.compare_ks.split(",")]
    all_ks = sorted(set(compare_ks + [args.router_k, args.n_protos]))

    modes = {"Learned": "learned", "Fixed": "fixed", "Sim": "sim"}
    results = {m: {k: [] for k in all_ks} for m in modes}
    agreement_vals = []

    print(f"\n  Comparing routing on {len(val_ds)} val images (K={all_ks})...")
    for idx in tqdm(range(len(val_ds)), desc="  Routing comparison"):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)
        features = backbone(image)
        p4 = features["p4"]

        # Full reference
        logit_full, sim_maps, _ = spm_head.forward_full(p4, temperature=args.temperature)

        for mode_name, mode in modes.items():
            for k in all_ks:
                if k >= args.n_protos:
                    logit_k = logit_full
                else:
                    logit_k, _, _, router_logits = spm_head.forward_routed(
                        p4, temperature=args.temperature, mode=mode, k=k)

                    # Agreement: Learned vs Fixed at router_k
                    if (mode == "learned" and router_logits is not None
                            and k == args.router_k):
                        router_flat = router_logits.permute(0, 2, 3, 1).reshape(-1, args.n_protos)
                        _, l_topk = router_flat.topk(k, dim=1)
                        head_w = spm_head.proto_head.head.weight.squeeze()
                        sim_flat = sim_maps.permute(0, 2, 3, 1).reshape(-1, args.n_protos)
                        importance = (sim_flat * head_w.unsqueeze(0)).abs()
                        _, f_topk = importance.topk(k, dim=1)
                        agree = sum(1 for i in range(l_topk.shape[0])
                                   for j in range(k) if l_topk[i, j] in f_topk[i])
                        agreement_vals.append(agree / (l_topk.shape[0] * k))

                # Dice
                logit_up = F.interpolate(logit_k, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                if pred.dim() == 2: pred = pred.unsqueeze(0)
                gm = gt_mask
                if gm.dim() == 2: gm = gm.unsqueeze(0)
                results[mode_name][k].append(compute_dice(pred, gm).item())

    agg = {}
    for mode_name in modes:
        agg[mode_name] = {}
        for k in all_ks:
            vals = results[mode_name][k]
            agg[mode_name][k] = {"dice_mean": float(np.mean(vals)),
                                 "dice_std": float(np.std(vals))}

    agree_m = float(np.mean(agreement_vals)) if agreement_vals else 0.0
    agree_s = float(np.std(agreement_vals)) if agreement_vals else 0.0
    return agg, agree_m, agree_s


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(agg, agreement_mean, agreement_std, args, output_path):
    """Learned vs Fixed vs Sim 对比图."""
    all_ks = sorted(list(agg["Learned"].keys()))
    colors = {"Learned": "tab:green", "Fixed": "tab:blue", "Sim": "tab:red"}
    markers = {"Learned": "D", "Fixed": "o", "Sim": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for mode_name in ["Learned", "Fixed", "Sim"]:
        d_means = [agg[mode_name][k]["dice_mean"] for k in all_ks]
        d_stds = [agg[mode_name][k]["dice_std"] for k in all_ks]
        ax.errorbar(all_ks, d_means, yerr=d_stds, marker=markers[mode_name],
                    capsize=3, color=colors[mode_name], linewidth=2, markersize=8,
                    label=mode_name)
    ax.set_xlabel("K (Protos per pixel)", fontsize=11)
    ax.set_ylabel("Dice (val)", fontsize=11)
    ax.set_title("Routing Strategy Comparison (Two-Stage)\n"
                 f"Agreement={agreement_mean:.1%} ± {agreement_std:.1%}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ks_plot = [k for k in all_ks if k != args.n_protos]
    l_deltas = [agg["Learned"][k]["dice_mean"] - agg["Learned"][args.n_protos]["dice_mean"]
                for k in ks_plot]
    f_deltas = [agg["Fixed"][k]["dice_mean"] - agg["Fixed"][args.n_protos]["dice_mean"]
                for k in ks_plot]
    x = np.arange(len(ks_plot))
    w = 0.3
    ax.bar(x - w/2, l_deltas, w, color="tab:green", label="Learned (SPM)")
    ax.bar(x + w/2, f_deltas, w, color="tab:blue", label="Fixed (|w·sim|)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks_plot])
    ax.set_ylabel("ΔDice from Full", fontsize=11)
    ax.set_title("Deviation from Full Performance", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E009: Two-Stage SPM Router — K={args.router_k} learned",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "routing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path / 'routing_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print(f"  E009: Two-Stage SPM Router")
    print(f"  Stage 1: Train ProtoHead ({args.epochs_s1} epochs)")
    print(f"  Stage 2: Train Router K={args.router_k} arch={args.router_arch} "
          f"({args.epochs_s2} epochs, frozen Proto)")
    print("=" * 70)

    set_seed(args.seed)

    exp_id = generate_exp_id(name=f"{args.name}_k{args.router_k}")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings",
                              dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Backbone ──
    print("\n[0] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── Data ──
    print("\n[1] Load Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Stage 1: ProtoHead ──
    print(f"\n[2] Stage 1: ProtoHead ({args.embed_dim}D, {args.n_protos} protos)")
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)

    if args.proto_checkpoint:
        ckpt = args.proto_checkpoint
        if "*" in ckpt:
            matches = _glob.glob(ckpt)
            ckpt = matches[0] if matches else ckpt
        proto_head.load_state_dict(torch.load(ckpt, map_location=device,
                                               weights_only=True))
        print(f"  ✅ Loaded ProtoHead from {ckpt}, skipping Stage 1.")
    else:
        s1_dice = train_stage1(proto_head, backbone, train_ds, val_ds, args, device, recorder)
        ckpt = output_path / "proto_head_s1.pt"
        torch.save(proto_head.state_dict(), ckpt)
        print(f"  [S1] Checkpoint: {ckpt}")
        recorder.record_metric("s1/best_dice", s1_dice, phase="val", tags=["s1"])

    # ── Stage 2: SPMHead (frozen Proto + trainable Router) ──
    print(f"\n[3] Stage 2: SPMHead (frozen Proto + Router K={args.router_k}, "
          f"arch={args.router_arch})")
    spm_head = SPMHead(proto_head, n_protos=args.n_protos, router_k=args.router_k,
                        router_arch=args.router_arch).to(device)

    s2_dice = train_stage2(spm_head, backbone, train_ds, val_ds, args, device, recorder)
    ckpt2 = output_path / "spm_head_s2.pt"
    torch.save({"proto_head": proto_head.state_dict(),
                "router": spm_head.router.state_dict(),
                "router_arch": args.router_arch}, ckpt2)
    print(f"  [S2] Checkpoint: {ckpt2}")

    # ── Comparison ──
    print(f"\n[4] Routing Strategy Comparison")
    agg, agree_m, agree_s = compare_routing(spm_head, backbone, val_ds, device, args)

    # ── Plot ──
    print(f"\n[5] Visualization")
    plot_comparison(agg, agree_m, agree_s, args, output_path)

    # ── Summary ──
    all_ks = sorted(list(agg["Learned"].keys()))
    full_dice = agg["Learned"][args.n_protos]["dice_mean"]

    print(f"\n{'=' * 70}")
    print(f"  E009 结果 | Results: Two-Stage SPM Router")
    print(f"  {'=' * 70}")
    print(f"  {'K':<8} {'Learned':>10} {'Fixed':>10} {'Δ(L-F)':>10} "
          f"{'Sim':>10} {'Full':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for k in all_ks:
        ld = agg["Learned"][k]["dice_mean"]
        fd = agg["Fixed"][k]["dice_mean"]
        sd = agg["Sim"][k]["dice_mean"]
        delta = ld - fd
        marker = " ← train" if k == args.router_k else ""
        print(f"  K={k:<6} {ld:>10.4f} {fd:>10.4f} {delta:>+10.4f} "
              f"{sd:>10.4f} {full_dice:>10.4f}{marker}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    print(f"\n  Router-Head Agreement: {agree_m:.1%} ± {agree_s:.1%}")

    # Verdict
    train_k = args.router_k
    ld_at_k = agg["Learned"][train_k]["dice_mean"]
    fd_at_k = agg["Fixed"][train_k]["dice_mean"]
    delta = ld_at_k - fd_at_k

    print(f"\n  {'─'*60}")
    if delta > 0.005:
        print(f"  ✅ LEARNED ROUTING WINS")
        print(f"     K={train_k}: Learned={ld_at_k:.4f} > Fixed={fd_at_k:.4f} (Δ=+{delta:.4f})")
        print(f"     → SPM Router 优于固定 |w·sim| 路由")
        verdict = "learned_wins"
    elif abs(delta) <= 0.005:
        print(f"  △ LEARNED ≈ FIXED")
        print(f"     K={train_k}: Learned={ld_at_k:.4f} ≈ Fixed={fd_at_k:.4f}")
        print(f"     → Router 匹配但不超越固定路由")
        verdict = "tied"
    else:
        print(f"  → FIXED ROUTING BETTER (Δ={delta:+.4f})")
        verdict = "fixed_wins"
    print(f"  {'─'*60}")

    for mode_name in agg:
        tag = mode_name.lower()
        for k in all_ks:
            recorder.record_metric(f"routing/{tag}_k{k}",
                                   agg[mode_name][k]["dice_mean"],
                                   phase="val", tags=["e009", tag, f"k{k}"])
    recorder.record_metric("routing/agreement", agree_m,
                           phase="val", tags=["e009", "summary"])
    recorder.logger.log_info("e009/verdict", verdict, tags=["e009", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {output_path}/")

    return agg


if __name__ == "__main__":
    main()
