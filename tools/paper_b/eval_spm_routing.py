#!/usr/bin/env python3
"""
Phase 3: SPM Token Routing 验证 — 不需要 Decoder 的三线对比
=============================================================
Token Routing Verification (Decoder-Free) — Oracle vs Random vs SPM.

核心问题 | Core Question:
    SPM/FDR 学到的 Top-K token mask 和 Oracle (GT) Top-K mask 有多接近？
    How close is the SPM-learned Top-K token mask to the Oracle (GT) Top-K mask?

实验设计 | Design:
    Oracle:  GT fg_ratio → Top-K token mask        (理论上界 | Upper bound)
    Random:  均匀随机 → Top-K token mask            (下界 | Lower bound)
    SPM:     FDR 预测 → Top-K token mask            (方法 | Method under test)
    Gradient: Oracle ≥ SPM ≫ Random

零依赖 | Zero Dependencies:
    不需要 Decoder, 不需要 Few-Shot, 不需要训练。
    只需: FastSAM backbone (frozen) + FDR checkpoint + val tiles.

运行时间 | Runtime:
    ~2 min for 5169 val tiles on GPU.

输出指标 | Output Metrics (per K level):
    Strategy   | Oracle-IoU | FG Recall | FG Precision | Overlap@Oracle
    Oracle     | 1.000      | X%        | X%           | 100%
    SPM        | X.XXX      | X%        | X%           | X%
    Random     | X.XXX      | X%        | X%           | X%

用法 | Usage::
    python tools/paper_b/eval_spm_routing.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fdr-ckpt runs/spm_training/spm_best.pt \
        --device cuda
"""

import sys, argparse, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.sparse.spatial_router import ForegroundDensityRouter

logger = get_logger("spm_routing")

# ═══════════════════════════════════════════════════════════════════════════
# 配置 | Configuration
# ═══════════════════════════════════════════════════════════════════════════

K_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
STRIDE = 16  # FastSAM P4 stride


def parse_args():
    p = argparse.ArgumentParser(
        description="SPM Token Routing Verification — Decoder-Free 3-Way Comparison")
    p.add_argument("--tile-root", type=str, default="data/iSAID5i_tiles/tile_896")
    p.add_argument("--fdr-ckpt", type=str, required=True,
                   help="FDR checkpoint path (spm_best.pt)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/spm_routing")
    p.add_argument("--n-viz-tiles", type=int, default=8,
                   help="可视化 tile 数量 | Number of tiles to visualize")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Token FG Ratio 数据集 (与 train_spm.py 相同) | Dataset (same as train_spm.py)
# ═══════════════════════════════════════════════════════════════════════════

from torch.utils.data import Dataset, DataLoader


class TokenFGRatioDataset(Dataset):
    """Tile → per-token fg_ratio GT at P4 stride."""

    def __init__(self, tile_root: str, split: str = "val", stride: int = 16):
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
        self._valid_indices = []
        for i, img_path in enumerate(self.img_paths):
            label_path = self.label_dir / f"{img_path.stem}_label.png"
            if label_path.exists():
                label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
                if label is not None and (label > 0).any():
                    self._valid_indices.append(i)
        logger.log_info("data",
                        f"TokenFGRatioDataset [{split}]: "
                        f"{len(self._valid_indices)}/{len(self.img_paths)} tiles with FG")

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        real_idx = self._valid_indices[idx]
        img_path = self.img_paths[real_idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        label_path = self.label_dir / f"{img_path.stem}_label.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        H, W = label.shape[:2]
        fg = (label > 0).astype(np.float32)
        h_t, w_t = H // self.stride, W // self.stride
        fg = fg[:h_t * self.stride, :w_t * self.stride]
        fg_ratio = (fg.reshape(h_t, self.stride, w_t, self.stride)
                      .transpose(0, 2, 1, 3)
                      .reshape(h_t, w_t, -1)
                      .mean(axis=2))
        fg_t = torch.from_numpy(fg_ratio).unsqueeze(0).float()
        return img_t, fg_t, str(img_path.stem)


# ═══════════════════════════════════════════════════════════════════════════
# Top-K Mask 生成 | Top-K Mask Generation
# ═══════════════════════════════════════════════════════════════════════════

def topk_mask(importance: torch.Tensor, k: float) -> torch.Tensor:
    """
    importance: [H, W] or [1, H, W] → binary mask with Top-K tokens = True.
    k: fraction in (0, 1].
    Returns: [H, W] bool.
    """
    if importance.dim() == 3:
        importance = importance.squeeze(0)
    n_total = importance.numel()
    n_keep = max(1, int(n_total * k))
    _, top_idx = torch.topk(importance.flatten(), n_keep)
    mask = torch.zeros(n_total, dtype=torch.bool, device=importance.device)
    mask[top_idx] = True
    return mask.reshape(importance.shape)


def get_oracle_importance(fg_gt: torch.Tensor) -> torch.Tensor:
    """Oracle = GT fg_ratio."""
    return fg_gt.squeeze(0)  # [H, W]


def get_random_importance(shape: tuple, device: torch.device) -> torch.Tensor:
    """Random = uniform [0, 1]."""
    return torch.rand(shape, device=device)


def get_spm_importance(fdr: torch.nn.Module, p4: torch.Tensor) -> torch.Tensor:
    """SPM = FDR prediction."""
    return fdr(p4)["importance"].squeeze(0).squeeze(0)  # [H, W]


# ═══════════════════════════════════════════════════════════════════════════
# 核心评估 | Core Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_routing(
    fdr: torch.nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    三线对比评估 | Three-way comparison evaluation.

    对每个 tile:
      1. Backbone → P4 features
      2. 生成 3 种 importance map: Oracle (GT), Random, SPM (FDR)
      3. 对每个 K: 生成 Top-K mask → 计算 overlap 指标

    Returns nested dict:
      results[strategy][K] = {
        "fg_recall": float,      # GT FG tokens retained / total GT FG tokens
        "fg_precision": float,   # GT FG tokens in Top-K / K tokens
        "overlap_oracle": float, # |strategy ∩ Oracle| / |Oracle|
      }
    """
    STRATEGIES = ["oracle", "random", "spm"]
    # 初始化累加器 | Initialize accumulators
    accum = {s: {k: {"fg_recall": [], "fg_precision": [], "overlap_oracle": []}
                  for k in K_LEVELS}
             for s in STRATEGIES}

    # 额外: per-tile 详细记录 (用于方差分析) | Per-tile detail for variance analysis
    per_tile = []

    fdr.eval()
    backbone.eval()

    for imgs, fg_gts, names in tqdm(loader, desc="Routing eval", unit="batch"):
        imgs = imgs.to(device)
        fg_gts = fg_gts.to(device)

        # ── Backbone → P4 ──
        p4s = backbone(imgs)["p4"]  # [B, 1280, H/16, W/16]

        for i in range(imgs.shape[0]):
            fg_gt = fg_gts[i]       # [1, H, W]
            p4 = p4s[i:i+1]         # [1, 1280, H, W]
            tile_name = names[i]

            # ── 对齐分辨率 | Align resolution ──
            imp_oracle = get_oracle_importance(fg_gt)  # [h_t, w_t]
            imp_random = get_random_importance(imp_oracle.shape, device)
            imp_spm = get_spm_importance(fdr, p4)

            # SPM 输出可能与 GT 分辨率不同 → 对齐 | Align SPM to GT resolution
            if imp_spm.shape != imp_oracle.shape:
                imp_spm = F.interpolate(
                    imp_spm.unsqueeze(0).unsqueeze(0),
                    size=imp_oracle.shape, mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)

            # ── Oracle 参考集: GT FG tokens | Oracle reference: tokens with any FG ──
            oracle_fg_tokens = (imp_oracle > 0)  # [h_t, w_t] bool — "true FG tokens"
            n_oracle_fg = oracle_fg_tokens.sum().item()

            if n_oracle_fg == 0:
                continue  # 理论上不应该出现 (已过滤 FG>0 tile)

            # ── 策略 importance maps | Strategy importance maps ──
            imp_maps = {
                "oracle": imp_oracle,
                "random": imp_random,
                "spm": imp_spm,
            }

            # ── 每个 K | Per K ──
            for k in K_LEVELS:
                oracle_mask = topk_mask(imp_oracle, k)  # Oracle Top-K

                for s_name in STRATEGIES:
                    s_mask = topk_mask(imp_maps[s_name], k)  # Strategy Top-K

                    # 1. FG Recall: 多少 GT FG token 被选中 | How many GT FG tokens selected
                    fg_recalled = (s_mask & oracle_fg_tokens).sum().item()
                    fg_recall = fg_recalled / n_oracle_fg

                    # 2. FG Precision: 选中的 token 有多少是 FG | How many selected are FG
                    n_selected = s_mask.sum().item()
                    fg_precision = fg_recalled / max(n_selected, 1)

                    # 3. Overlap with Oracle: |strategy ∩ Oracle| / |Oracle|
                    overlap = (s_mask & oracle_mask).sum().item() / max(oracle_mask.sum().item(), 1)

                    accum[s_name][k]["fg_recall"].append(fg_recall)
                    accum[s_name][k]["fg_precision"].append(fg_precision)
                    accum[s_name][k]["overlap_oracle"].append(overlap)

            # Per-tile 记录 (仅 K=0.3) | Per-tile record (K=0.3 only)
            k_ref = 0.3
            per_tile.append({
                "name": tile_name,
                "n_tokens": imp_oracle.numel(),
                "n_oracle_fg": n_oracle_fg,
                "spm_overlap": accum["spm"][k_ref]["overlap_oracle"][-1],
                "random_overlap": accum["random"][k_ref]["overlap_oracle"][-1],
            })

    # ── 聚合 | Aggregate ──
    summary = {}
    for s_name in STRATEGIES:
        summary[s_name] = {}
        for k in K_LEVELS:
            summary[s_name][k] = {
                "fg_recall": float(np.mean(accum[s_name][k]["fg_recall"])),
                "fg_precision": float(np.mean(accum[s_name][k]["fg_precision"])),
                "overlap_oracle": float(np.mean(accum[s_name][k]["overlap_oracle"])),
            }

    return summary, per_tile


# ═══════════════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_routing_results(
    summary: dict,
    per_tile: list,
    out_dir: Path,
    k_levels: list,
):
    """6-panel visualization."""
    strategies = ["oracle", "random", "spm"]
    colors = {"oracle": "#27AE60", "random": "#E74C3C", "spm": "#3498DB"}
    labels = {"oracle": "Oracle (Upper Bound)", "random": "Random (Lower Bound)",
              "spm": "SPM/FDR (Method)"}

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    k_pcts = np.array([k * 100 for k in k_levels])

    # ═══ Panel 1: Overlap with Oracle — 核心指标 | Core metric ═══
    ax = fig.add_subplot(gs[0, 0])
    for s_name in strategies:
        overlap_vals = [summary[s_name][k]["overlap_oracle"] * 100 for k in k_levels]
        ls = "-" if s_name != "random" else "--"
        lw = 2.5 if s_name == "spm" else 1.5
        alpha = 1.0 if s_name == "spm" else 0.6
        ax.plot(k_pcts, overlap_vals, f"{ls}o", color=colors[s_name],
                linewidth=lw, markersize=5 if s_name != "spm" else 7,
                alpha=alpha, label=labels[s_name])

    # SPM vs Random gap 标注 | Annotate gap
    k30_idx = k_levels.index(0.3)
    spm_30 = summary["spm"][0.3]["overlap_oracle"] * 100
    rnd_30 = summary["random"][0.3]["overlap_oracle"] * 100
    gap = spm_30 - rnd_30
    ax.annotate(f"SPM-Random gap\n= {gap:.1f}% at K=30%",
                (30, (spm_30 + rnd_30) / 2),
                textcoords="offset points", xytext=(30, -10),
                fontsize=10, ha="center", color="#3498DB", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#3498DB", lw=1.2))

    ax.set_xlabel("Tokens Kept (%)", fontsize=11)
    ax.set_ylabel("Overlap with Oracle Top-K (%)", fontsize=11)
    ax.set_title("Token Mask Overlap with Oracle\n"
                 "(↑ = closer to Oracle = better routing)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 105)

    # ═══ Panel 2: FG Recall per strategy | Per-strategy FG Recall ═══
    ax = fig.add_subplot(gs[0, 1])
    for s_name in strategies:
        recall_vals = [summary[s_name][k]["fg_recall"] * 100 for k in k_levels]
        ls = "-" if s_name != "random" else "--"
        lw = 2.5 if s_name == "spm" else 1.5
        ax.plot(k_pcts, recall_vals, f"{ls}s", color=colors[s_name],
                linewidth=lw, markersize=4, alpha=1.0 if s_name == "spm" else 0.6,
                label=labels[s_name])
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Tokens Kept (%)", fontsize=11)
    ax.set_ylabel("FG Tokens Recalled (%)", fontsize=11)
    ax.set_title("GT Foreground Token Recall\n"
                 "(What % of FG tokens are in Top-K?)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 105)

    # ═══ Panel 3: SPM/Oracle Overlap Ratio — 相对效率 | Relative Efficiency ═══
    ax = fig.add_subplot(gs[0, 2])
    # 计算 SPM 达到了 Oracle 的多少比例 | SPM efficiency vs Oracle
    for s_name in ["spm", "random"]:
        ratio_vals = []
        for k in k_levels:
            spm_ov = summary[s_name][k]["overlap_oracle"]
            ora_ov = summary["oracle"][k]["overlap_oracle"]
            ratio_vals.append(spm_ov / max(ora_ov, 1e-8) * 100)
        ls = "-" if s_name == "spm" else "--"
        ax.plot(k_pcts, ratio_vals, f"{ls}D", color=colors[s_name],
                linewidth=2.0, markersize=6, label=labels[s_name])
    ax.axhline(y=100, color="#27AE60", linestyle=":", alpha=0.5, label="Oracle (=100%)")
    ax.set_xlabel("Tokens Kept (%)", fontsize=11)
    ax.set_ylabel("Relative Efficiency (% of Oracle)", fontsize=11)
    ax.set_title("Routing Efficiency vs Oracle\n"
                 f"(SPM achieves {summary['spm'][0.3]['overlap_oracle']/max(summary['oracle'][0.3]['overlap_oracle'],1e-8)*100:.0f}% "
                 f"of Oracle at K=30%)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ═══ Panel 4: 单 tile 示例 | Single tile examples ═══
    # 选 SPM-Oracle 重叠最好/最差的 tile | Pick best/worst tiles by SPM-Oracle overlap
    tiles_sorted = sorted(per_tile, key=lambda x: x["spm_overlap"])
    demo_tiles = [tiles_sorted[0], tiles_sorted[len(tiles_sorted) // 2], tiles_sorted[-1]]
    demo_labels = ["Worst SPM", "Median SPM", "Best SPM"]

    for row_idx, (t, dlbl) in enumerate(zip(demo_tiles, demo_labels)):
        ax = fig.add_subplot(gs[1, row_idx])
        # 加载可视化 | Load for visualization
        tile_root = Path(args.tile_root) if 'args' in dir() else None
        ax.text(0.5, 0.5,
                f"{dlbl}\n{t['name'][:20]}\n"
                f"SPM overlap={t['spm_overlap']*100:.1f}%\n"
                f"Random overlap={t['random_overlap']*100:.1f}%\n"
                f"FG tokens={t['n_oracle_fg']}/{t['n_tokens']}",
                ha="center", va="center", fontsize=9, fontfamily="monospace",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{dlbl} Tile", fontsize=10)
        # Color border
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["spm"] if "SPM" in dlbl else "gray")
            spine.set_linewidth(2)

    # ═══ Panel 5: Per-Tile SPM Overlap 分布 | Distribution of SPM Overlap ═══
    ax = fig.add_subplot(gs[2, 0])
    spm_overlaps = [t["spm_overlap"] * 100 for t in per_tile]
    rnd_overlaps = [t["random_overlap"] * 100 for t in per_tile]
    bins = np.linspace(0, 100, 40)
    ax.hist(spm_overlaps, bins=bins, alpha=0.6, color=colors["spm"], label=f"SPM (μ={np.mean(spm_overlaps):.1f}%)")
    ax.hist(rnd_overlaps, bins=bins, alpha=0.4, color=colors["random"], label=f"Random (μ={np.mean(rnd_overlaps):.1f}%)")
    ax.set_xlabel("Overlap with Oracle Top-30% (%)", fontsize=11)
    ax.set_ylabel("Number of Tiles", fontsize=11)
    ax.set_title(f"Per-Tile SPM vs Oracle Overlap Distribution\n"
                 f"(K=30%, {len(per_tile)} tiles)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # ═══ Panel 6: FG Density vs SPM Quality | FG density vs routing quality ═══
    ax = fig.add_subplot(gs[2, 1])
    fg_densities = [t["n_oracle_fg"] / t["n_tokens"] * 100 for t in per_tile]
    ax.scatter(fg_densities, spm_overlaps, c=colors["spm"], alpha=0.3, s=8)
    # 趋势线 | Trend line
    if len(fg_densities) > 1:
        z = np.polyfit(fg_densities, spm_overlaps, 1)
        x_line = np.linspace(min(fg_densities), max(fg_densities), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="#E74C3C", linewidth=1.5,
                label=f"Trend (slope={z[0]:.2f})")
    ax.set_xlabel("Tile FG Density (%)", fontsize=11)
    ax.set_ylabel("SPM Overlap with Oracle (%)", fontsize=11)
    ax.set_title("Does SPM Work Better on Dense Tiles?\n"
                 f"r={np.corrcoef(fg_densities, spm_overlaps)[0,1]:.3f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ═══ Panel 7 (右下): 摘要表 | Summary Table ═══
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    k_show = [0.10, 0.20, 0.30, 0.40, 0.50]
    lines = ["SPM Routing Verification", "=" * 35, ""]
    lines.append(f"{'K':>6s}  {'Oracle':>8s}  {'SPM':>8s}  {'Random':>8s}  {'SPM Gain':>9s}")
    lines.append("  " + "-" * 42)

    for k in k_show:
        o = summary["oracle"][k]["overlap_oracle"] * 100
        s = summary["spm"][k]["overlap_oracle"] * 100
        r = summary["random"][k]["overlap_oracle"] * 100
        gain = s - r
        lines.append(f"  {k*100:4.0f}%  {o:7.1f}%  {s:7.1f}%  {r:7.1f}%  {gain:+8.1f}%")

    lines.append("")
    spm_eff = summary["spm"][0.3]["overlap_oracle"] / max(summary["oracle"][0.3]["overlap_oracle"], 1e-8) * 100
    spm_vs_rnd = summary["spm"][0.3]["overlap_oracle"] / max(summary["random"][0.3]["overlap_oracle"], 1e-8)
    lines.append(f"SPM Efficiency @30%: {spm_eff:.0f}% of Oracle")
    lines.append(f"SPM vs Random @30%:  {spm_vs_rnd:.1f}× better")

    for i, line in enumerate(lines):
        y_pos = 9.5 - i * 0.4
        if line.startswith("SPM Routing"):
            ax.text(0.5, y_pos, line, fontsize=13, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif "Gain" in line and "%" in line:
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace", va="top",
                    fontweight="bold")
        elif "×" in line or "Efficiency" in line:
            ax.text(0.5, y_pos, line, fontsize=10, fontfamily="monospace", va="top",
                    color="#27AE60" if "×" in line else "#3498DB", fontweight="bold")
        elif line.startswith("==="):
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        else:
            ax.text(0.5, y_pos, line, fontsize=9, fontfamily="monospace", va="top")

    fig.suptitle("Phase 3: Token Routing Verification — Oracle vs Random vs SPM\n"
                 "(Decoder-Free, Overlap-based Evaluation)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "routing_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"Saved → {out_dir / 'routing_overlap.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 可视化: Tile 级对比 | Visualization: Per-Tile Comparison
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_tile_examples(
    fdr: torch.nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    n_examples: int = 4,
):
    """
    可视化示例 tile: 原图 + GT FG + Oracle Heatmap + SPM Heatmap + SPM Top-30% Mask
    """
    logger.log_info("viz", "Generating tile example visualizations...")
    fdr.eval()
    backbone.eval()

    samples = []
    for imgs, fg_gts, names in loader:
        imgs = imgs.to(device)
        fg_gts = fg_gts.to(device)
        p4s = backbone(imgs)["p4"]
        for i in range(imgs.shape[0]):
            if len(samples) >= n_examples:
                break
            fg_gt = fg_gts[i]
            p4 = p4s[i:i+1]
            imp_oracle = get_oracle_importance(fg_gt)
            imp_spm = get_spm_importance(fdr, p4)
            if imp_spm.shape != imp_oracle.shape:
                imp_spm = F.interpolate(
                    imp_spm.unsqueeze(0).unsqueeze(0),
                    size=imp_oracle.shape, mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
            # 选有代表性: 中等 FG 密度 | Pick medium density
            n_fg = (imp_oracle > 0).sum().item()
            if 5 < (n_fg / imp_oracle.numel() * 100) < 30:
                samples.append({
                    "image": imgs[i].cpu(),
                    "gt_fg": (fg_gt.squeeze(0) > 0).cpu().numpy(),
                    "oracle_heat": imp_oracle.cpu().numpy(),
                    "spm_heat": imp_spm.cpu().numpy(),
                    "name": names[i],
                })
        if len(samples) >= n_examples:
            break

    if not samples:
        logger.log_info("viz", "  No suitable examples found (density filter too strict)")
        return

    n_ex = len(samples)
    fig, axes = plt.subplots(n_ex, 5, figsize=(22, 4.5 * n_ex))
    if n_ex == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Image", "GT FG Mask", "Oracle Heatmap", "SPM Heatmap", "SPM Top-30%"]

    for row_idx, s in enumerate(samples):
        img_np = s["image"].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Oracle heatmap upsampled
        ora_up = cv2.resize(s["oracle_heat"], (img_np.shape[1], img_np.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        ora_color = cv2.applyColorMap((ora_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ora_overlay = cv2.addWeighted(img_np, 0.4, ora_color.astype(np.float32) / 255.0, 0.6, 0)

        # SPM heatmap upsampled
        spm_up_raw = s["spm_heat"]
        spm_up_raw = (spm_up_raw - spm_up_raw.min()) / max(spm_up_raw.max() - spm_up_raw.min(), 1e-8)
        spm_up = cv2.resize(spm_up_raw, (img_np.shape[1], img_np.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        spm_color = cv2.applyColorMap((spm_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        spm_overlay = cv2.addWeighted(img_np, 0.4, spm_color.astype(np.float32) / 255.0, 0.6, 0)

        # SPM Top-30% mask (upsampled)
        spm_mask = topk_mask(torch.from_numpy(s["spm_heat"]), 0.30).numpy().astype(np.uint8) * 255
        spm_mask_up = cv2.resize(spm_mask, (img_np.shape[1], img_np.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        data_list = [img_np, s["gt_fg"], ora_overlay, spm_overlay, spm_mask_up]
        for col_idx, (ctitle, img_data) in enumerate(zip(col_titles, data_list)):
            ax = axes[row_idx, col_idx]
            cmap = "gray" if col_idx in [1, 4] else None
            ax.imshow(img_data, cmap=cmap)
            if row_idx == 0:
                ax.set_title(ctitle, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                fg_pct = s["gt_fg"].mean() * 100
                ax.set_ylabel(f"{s['name'][:18]}\nFG={fg_pct:.1f}%",
                              fontsize=7, rotation=0, labelpad=45, va="center")

    fig.suptitle("SPM vs Oracle: Per-Tile Importance Map Comparison", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "tile_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("viz", f"Saved → {out_dir / 'tile_examples.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main | 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global args
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logging ──
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "routing_eval.jsonl")))

    logger.log_info("config", f"Device: {device}, Tile root: {args.tile_root}")
    logger.log_info("config", f"FDR checkpoint: {args.fdr_ckpt}")
    logger.log_info("config", f"K levels: {K_LEVELS}")

    # ── Backbone ──
    logger.log_info("backbone", "Loading frozen FastSAM backbone...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()
    backbone.eval()

    # ── FDR | Load SPM ──
    logger.log_info("fdr", f"Loading FDR from {args.fdr_ckpt}")
    fdr = ForegroundDensityRouter(in_channels=1280, mid_channels=128)
    ckpt = torch.load(args.fdr_ckpt, map_location=device)
    fdr.load_state_dict(ckpt["model_state_dict"])
    fdr.to(device).eval()
    n_params = sum(p.numel() for p in fdr.parameters())
    logger.log_info("fdr",
                    f"FDR loaded: E{ckpt.get('epoch', '?')}, "
                    f"val r={ckpt.get('val_metrics', {}).get('spearman_r', '?'):.4f}, "
                    f"{n_params:,} params")

    # ── Data ──
    val_ds = TokenFGRatioDataset(args.tile_root, "val", stride=16)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── 评估 | Evaluate ──
    logger.log_info("eval", "=" * 70)
    logger.log_info("eval", "Phase 3: Oracle vs Random vs SPM — Token Overlap Evaluation")
    logger.log_info("eval", "=" * 70)

    summary, per_tile = evaluate_routing(fdr, backbone, val_loader, device)

    # ── 报告 | Report ──
    logger.log_info("report", "")
    logger.log_info("report", "=" * 80)
    logger.log_info("report", "Routing Verification Results")
    logger.log_info("report", "=" * 80)

    # 表头 | Table header
    header = (f"{'K':>6s} | {'Oracle':>10s} | {'SPM':>10s} | {'Random':>10s} | "
              f"{'SPM Gain':>9s} | {'SPM Eff':>8s}")
    logger.log_info("report", header)
    logger.log_info("report", "-" * len(header))

    for k in K_LEVELS:
        o = summary["oracle"][k]["overlap_oracle"] * 100
        s = summary["spm"][k]["overlap_oracle"] * 100
        r = summary["random"][k]["overlap_oracle"] * 100
        gain = s - r
        eff = s / max(o, 1e-8) * 100
        marker = " ←" if k == 0.3 else ""
        logger.log_info("report",
                        f"  {k*100:4.0f}% | {o:9.1f}% | {s:9.1f}% | {r:9.1f}% | "
                        f"{gain:+8.1f}% | {eff:7.0f}%{marker}")

    # ── 关键结论 | Key conclusions ──
    k_ref = 0.3
    spm_overlap = summary["spm"][k_ref]["overlap_oracle"] * 100
    rnd_overlap = summary["random"][k_ref]["overlap_oracle"] * 100
    spm_eff = spm_overlap / max(summary["oracle"][k_ref]["overlap_oracle"] * 100, 1e-8) * 100
    spm_vs_rnd = spm_overlap / max(rnd_overlap, 1e-8)

    spm_recall = summary["spm"][k_ref]["fg_recall"] * 100
    ora_recall = summary["oracle"][k_ref]["fg_recall"] * 100

    logger.log_info("report", "")
    logger.log_info("report", f"Key Results at K={k_ref*100:.0f}%:")
    logger.log_info("report", f"  Oracle Top-{k_ref*100:.0f}% overlap with self: "
                    f"{summary['oracle'][k_ref]['overlap_oracle']*100:.1f}% (by definition)")
    logger.log_info("report", f"  SPM overlap with Oracle:  {spm_overlap:.1f}% "
                    f"({spm_eff:.0f}% of Oracle, {spm_vs_rnd:.1f}× Random)")
    logger.log_info("report", f"  Random overlap with Oracle: {rnd_overlap:.1f}%")
    logger.log_info("report", f"  SPM FG Recall: {spm_recall:.1f}% vs Oracle: {ora_recall:.1f}%")

    # ── Gradient Check | 梯度验证 ──
    logger.log_info("report", "")
    logger.log_info("report", "Gradient Check | 梯度验证:")
    if spm_overlap > rnd_overlap * 2:
        logger.log_info("report",
                        f"  ✓ SPM ≫ Random: {spm_vs_rnd:.1f}× better → "
                        f"FDR has learned meaningful density ranking")
    elif spm_overlap > rnd_overlap * 1.3:
        logger.log_info("report",
                        f"  △ SPM > Random: {spm_vs_rnd:.1f}× → "
                        f"FDR provides moderate improvement over random")
    else:
        logger.log_info("report",
                        f"  ✗ SPM ≈ Random: {spm_vs_rnd:.1f}× → "
                        f"FDR not learning useful ranking (check training quality)")

    if spm_eff > 70:
        logger.log_info("report",
                        f"  ✓ SPM is {spm_eff:.0f}% as good as Oracle → "
                        f"Token routing is practical!")
    elif spm_eff > 40:
        logger.log_info("report",
                        f"  △ SPM is {spm_eff:.0f}% as good as Oracle → "
                        f"Moderate routing quality, room for improvement")
    else:
        logger.log_info("report",
                        f"  ✗ SPM is only {spm_eff:.0f}% as good as Oracle → "
                        f"Token routing needs better importance predictor")

    # ── 可视化 | Visualization ──
    logger.log_info("viz", "Generating visualizations...")
    plot_routing_results(summary, per_tile, out_dir, K_LEVELS)

    # Per-tile examples
    vis_loader = DataLoader(
        val_ds, batch_size=1, shuffle=True,
        num_workers=0,
    )
    visualize_tile_examples(fdr, backbone, vis_loader, out_dir, device,
                            n_examples=args.n_viz_tiles)

    # ── 保存 JSON | Save JSON ──
    import datetime
    result = {
        "experiment": "Phase 3: Token Routing Verification (Decoder-Free)",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "tile_root": str(args.tile_root),
            "fdr_ckpt": str(args.fdr_ckpt),
            "n_val_tiles": len(val_ds),
        },
        "summary": {
            s_name: {str(k): {
                "fg_recall": round(v["fg_recall"], 4),
                "fg_precision": round(v["fg_precision"], 4),
                "overlap_oracle": round(v["overlap_oracle"], 4),
            } for k, v in s_data.items()}
            for s_name, s_data in summary.items()
        },
        "key_results_at_30pct": {
            "spm_overlap_oracle": round(spm_overlap, 2),
            "random_overlap_oracle": round(rnd_overlap, 2),
            "spm_vs_random_ratio": round(spm_vs_rnd, 2),
            "spm_efficiency_pct": round(spm_eff, 1),
            "spm_fg_recall": round(spm_recall, 2),
            "oracle_fg_recall": round(ora_recall, 2),
        },
    }

    json_path = out_dir / "routing_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.log_info("output", f"Results saved → {json_path}")

    logger.log_info("done", "=" * 70)
    logger.log_info("done", "Phase 3 Decoder-Free Routing Verification Complete!")
    logger.log_info("done", "=" * 70)


if __name__ == "__main__":
    main()
