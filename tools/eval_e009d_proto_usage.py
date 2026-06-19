#!/usr/bin/env python3
"""
E009-D: Proto Usage Analysis — 多少个 Proto 够用？
====================================================

核心问题 | Core question:
    模型真正使用了多少个 Proto？是否存在"有效 Proto 数"上限？

实验设计 | Experiment:
    1. 用 N ∈ {2, 4, 6, 8, 12} 训练 ProtoHead (E007-B recipe)
    2. 对每个 N 统计:
       - Dice (val)
       - Per-proto 激活频率 (winner-take-all)
       - Per-proto |w·sim| 贡献分布
       - Head 权重 L2 范数分布
       - Prototype 互余弦相似度 (冗余度)
       - Dominance / Entropy

假设 | Hypotheses:
    H1: Dice 在 N≥4 后饱和 (4 个 proto 已足够)
    H2: 大 N 下部分 Proto "坍缩" (贡献 → 0, 从未被激活)
    H3: Proto 间出现高冗余 (cosine similarity → 1)

发现 | Expected finding:
    Proto Sparsity: 模型自然趋向使用少数 Proto
    与 Spatial Sparsity (E008) 形成 Dual Sparsity 范式

用法 | Usage:
    python tools/eval_e009d_proto_usage.py
    python tools/eval_e009d_proto_usage.py --n-list "2,4,6,8,12" --epochs 30
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice
from adatile.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--dataset", type=str, default="mass_buildings",
                   choices=["mass_buildings", "isaid"])
    p.add_argument("--n-list", type=str, default="2,4,6,8,12")
    p.add_argument("--num-classes", type=int, default=15,
                   help="类别数 (仅 isaid) | num classes (isaid only)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e009d_usage")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ProtoHead
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(torch.nn.Module):
    def __init__(self, in_channels=1280, embed_dim=128, n_protos=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            torch.nn.ReLU(inplace=True),
        )
        self.prototypes = torch.nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.head = torch.nn.Conv2d(n_protos, 1, 1, bias=True)

    def forward(self, p4, temperature=0.1):
        embedding = self.project(p4)
        emb_norm = F.normalize(embedding, dim=1, p=2)
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm) / temperature
        logit = self.head(sim_maps)
        return embedding, sim_maps, logit

    def get_winner_map(self, p4, temperature=0.1):
        """每个像素的 Winner Proto 索引 | Winner proto index per pixel."""
        _, sim_maps, _ = self.forward(p4, temperature)
        return sim_maps.argmax(dim=1)  # [B, H, W]

    def get_proto_stats(self, val_ds, backbone, device, temperature=0.1):
        """
        统计 Proto 使用情况 | Compute per-proto usage statistics.

        Returns:
            winner_freq:    [N] — 每个 proto 是 winner 的频率
            energy_frac:    [N] — 每个 proto 的 |w·sim| 能量占比
            head_norm:      [N] — 每个 proto 的 head weight 的 L2 范数
            inter_cos:      [N,N] — proto 间 cosine similarity 矩阵
        """
        self.eval()
        n_protos = self.n_protos
        winner_counts = np.zeros(n_protos)
        energy_acc = np.zeros(n_protos)
        total_pixels = 0

        with torch.no_grad():
            for idx in range(min(20, len(val_ds))):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    features = backbone(image)

                winner = self.get_winner_map(features["p4"], temperature)
                embedding, sim_maps, _ = self.forward(features["p4"], temperature)

                # Winner frequency
                for p in range(n_protos):
                    winner_counts[p] += (winner == p).sum().item()
                total_pixels += winner.numel()

                # Energy: |w·sim| per proto
                head_w = self.head.weight.squeeze().detach()  # [N]
                sim_flat = sim_maps.squeeze(0).reshape(n_protos, -1)  # [N, HW]
                energy = (sim_flat * head_w.unsqueeze(1)).abs().sum(dim=1)  # [N]
                energy_acc += energy.cpu().numpy()

        winner_freq = winner_counts / total_pixels
        energy_frac = energy_acc / (energy_acc.sum() + 1e-8)

        # Head weight L2 norm
        head_w = self.head.weight.squeeze()
        head_norm = head_w.detach().abs().cpu().numpy()

        # Proto 间 cosine similarity
        proto_n = F.normalize(self.prototypes.detach(), dim=1, p=2)
        inter_cos = (proto_n @ proto_n.T).cpu().numpy()

        return winner_freq, energy_frac, head_norm, inter_cos


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_proto_head(proto_head, backbone, train_ds, val_ds, args, device, recorder):
    """训练 ProtoHead (同 E007-B)."""
    proto_head.train()
    optimizer = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        proto_head.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/{args.epochs}",
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
        scheduler.step()

        proto_head.eval()
        dices = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                if gt_mask.dim() == 3: gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4: gt_mask = gt_mask.squeeze(0).squeeze(0)
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
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}
        marker = " *" if dice_mean >= best_dice else ""
        print(f"    N={proto_head.n_protos} Epoch {epoch:2d}  "
              f"loss={total_loss/len(train_ds):.4f}  Dice={dice_mean:.4f}{marker}")

        recorder.record_metric(f"dice_n{proto_head.n_protos}", dice_mean,
                               step=epoch, phase="val")

    if best_state is not None:
        proto_head.load_state_dict(best_state)
    print(f"  N={proto_head.n_protos} Best Dice: {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_usage_analysis(results, output_path):
    """综合可视化 | Comprehensive visualization."""
    n_values = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (1) Dice vs N
    ax = axes[0, 0]
    dices = [results[n]["dice"] for n in n_values]
    ax.plot(n_values, dices, marker="D", color="tab:blue", linewidth=2, markersize=10)
    ax.axhline(y=max(dices), color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Protos N", fontsize=11)
    ax.set_ylabel("Best Dice (val)", fontsize=11)
    ax.set_title("Dice vs Proto Count\n"
                 f"N=2:{dices[0]:.4f} → N={n_values[-1]}:{dices[-1]:.4f}", fontsize=10)
    ax.grid(True, alpha=0.3)

    # (2) Winner Frequency distribution
    ax = axes[0, 1]
    for i, n in enumerate(n_values):
        freq = results[n]["winner_freq"]
        ax.plot(range(1, n+1), sorted(freq, reverse=True),
                marker="o", color=colors[i], linewidth=2, label=f"N={n}")
    ax.set_xlabel("Proto Rank (by frequency)", fontsize=11)
    ax.set_ylabel("Winner Frequency", fontsize=11)
    ax.set_title("Proto Usage Distribution\n(sorted, per N)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Dominance (top-1 energy fraction) vs N
    ax = axes[0, 2]
    dominance = [results[n]["energy_frac"][0] for n in n_values]
    # Effective proto count: 1 / sum(p_i^2) (inverse Herfindahl)
    eff_n = []
    for n in n_values:
        ef = results[n]["energy_frac"]
        eff_n.append(1.0 / (ef**2).sum() if (ef**2).sum() > 0 else 1)
    ax.plot(n_values, dominance, marker="s", color="tab:orange", linewidth=2,
            markersize=8, label="Top-1 Energy%")
    ax.plot(n_values, eff_n, marker="^", color="tab:green", linewidth=2,
            markersize=8, label="Effective N")
    ax.plot(n_values, n_values, color="gray", linestyle="--", alpha=0.3, label="N (ideal)")
    ax.set_xlabel("Number of Protos N", fontsize=11)
    ax.set_ylabel("Count / Fraction", fontsize=11)
    ax.set_title("Proto Concentration\n"
                 f"Dominance: {dominance[0]:.1%}→{dominance[-1]:.1%}  |  "
                 f"Eff N(max)={max(eff_n):.1f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Energy distribution (per proto |w·sim| fraction) for largest N
    ax = axes[1, 0]
    max_n = max(n_values)
    ef = results[max_n]["energy_frac"]
    bar_colors = ["tab:red" if v > 0.05 else "tab:blue" for v in ef]
    ax.bar(range(1, max_n+1), ef, color=bar_colors)
    ax.axhline(y=1/max_n, color="gray", linestyle="--", alpha=0.5,
               label=f"Uniform ({1/max_n:.1%})")
    ax.set_xlabel("Proto Index", fontsize=11)
    ax.set_ylabel("|w·sim| Energy Fraction", fontsize=11)
    ax.set_title(f"Energy Distribution (N={max_n})\n"
                 f"Active: {(ef>0.02).sum()}/{max_n}  |  "
                 f"Min/Avg={ef.min():.3f}/{ef.mean():.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (5) Head weight norm per proto (largest N)
    ax = axes[1, 1]
    hn = results[max_n]["head_norm"]
    ax.bar(range(1, max_n+1), hn, color=bar_colors)
    ax.set_xlabel("Proto Index", fontsize=11)
    ax.set_ylabel("|Head Weight|", fontsize=11)
    ax.set_title(f"Head Weight Magnitude (N={max_n})\n"
                 f"Max={hn.max():.3f} Min={hn.min():.3f} Ratio={hn.max()/max(hn.min(),1e-8):.1f}×",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    # (6) Proto cosine similarity matrix (largest N)
    ax = axes[1, 2]
    cm = results[max_n]["inter_cos"]
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(max_n))
    ax.set_yticks(range(max_n))
    ax.set_xticklabels([f"P{i}" for i in range(max_n)], fontsize=7)
    ax.set_yticklabels([f"P{i}" for i in range(max_n)], fontsize=7)
    off_diag = cm[~np.eye(max_n, dtype=bool)]
    ax.set_title(f"Proto Cosine Similarity (N={max_n})\n"
                 f"Max off-diag: {off_diag.max():.3f}  Mean: {off_diag.mean():.3f}",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("E009-D: Proto Usage Analysis — How Many Protos Are Enough?",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path / "proto_usage_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device
    n_list = [int(x.strip()) for x in args.n_list.split(",")]

    print("=" * 70)
    print(f"  E009-D: Proto Usage Analysis")
    print(f"  N = {n_list}")
    print(f"  Core: How many protos does the model actually use?")
    print("=" * 70)

    set_seed(args.seed)

    exp_id = generate_exp_id(name=args.name)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings",
                              dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    results = {}
    for n in n_list:
        print(f"\n{'─'*60}")
        print(f"  Training ProtoHead N={n}")
        print(f"{'─'*60}")

        set_seed(args.seed)
        set_seed(args.seed)

        proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                                n_protos=n).to(device)
        n_params = sum(p.numel() for p in proto_head.parameters())
        print(f"  Params: {n_params:,}")

        best_dice = train_proto_head(proto_head, backbone, train_ds, val_ds,
                                     args, device, recorder)
        winner_freq, energy_frac, head_norm, inter_cos = proto_head.get_proto_stats(
            val_ds, backbone, device, args.temperature
        )

        results[n] = {
            "dice": best_dice,
            "winner_freq": winner_freq,
            "energy_frac": energy_frac,
            "head_norm": head_norm,
            "inter_cos": inter_cos,
            "n_params": n_params,
        }

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  E009-D 结果 | Results: Proto Usage Analysis")
    print(f"  {'=' * 70}")
    print(f"  {'N':<6} {'Dice':>10} {'Params':>10} {'Eff N':>8} "
          f"{'Top-1%':>8} {'Active':>8} {'Max Cos':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for n in n_list:
        r = results[n]
        ef = r["energy_frac"]
        eff_n = 1.0 / (ef**2).sum() if (ef**2).sum() > 0 else 1.0
        n_active = (ef > 0.02).sum()
        max_cos = r["inter_cos"][~np.eye(n, dtype=bool)].max()
        print(f"  N={n:<4} {r['dice']:>10.4f} {r['n_params']:>10,} "
              f"{eff_n:>7.1f} {ef[0]:>7.1%} {n_active:>7} {max_cos:>7.3f}")

    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    # ── Key finding ──
    dices = [results[n]["dice"] for n in n_list]
    best_n = n_list[np.argmax(dices)]
    n4_dice = results.get(4, {}).get("dice", 0)
    n_best_dice = results.get(best_n, {}).get("dice", 0)
    dice_drop = max(dices) - n4_dice if 4 in n_list else 0

    print(f"\n  {'─'*60}")
    if 4 in n_list and dice_drop < 0.01:
        print(f"  ✅ FOUR PROTOS ARE SUFFICIENT")
        print(f"     N=4 Dice: {n4_dice:.4f} ≈ Best N={best_n}: {n_best_dice:.4f}")
        print(f"     ΔDice from best: {dice_drop:.4f}")
        print(f"     → Proto Sparsity 成立, 与 Spatial Sparsity 形成 Dual Sparsity")
        verdict = "four_sufficient"
    elif 4 in n_list and dice_drop < 0.02:
        print(f"  △ N=4 CLOSE TO BEST")
        print(f"     → 边际收益递减, 4 proto 是 sweet spot")
        verdict = "four_close"
    else:
        print(f"  → Saturation point needs further investigation")
        verdict = "needs_investigation"
    print(f"  {'─'*60}")

    # ── Plot ──
    plot_usage_analysis(results, output_path)
    print(f"\n  Plots saved to: {output_path}/")

    for n in n_list:
        r = results[n]
        recorder.record_metric(f"usage/dice_n{n}", r["dice"], phase="val", tags=["e009d"])
        recorder.record_metric(f"usage/eff_n{n}",
                               1.0 / (r["energy_frac"]**2).sum(),
                               phase="val", tags=["e009d"])
    recorder.logger.log_info("e009d/verdict", verdict, tags=["e009d", "summary"])
    recorder.finalize()
    recorder.close()

    return results


if __name__ == "__main__":
    main()
