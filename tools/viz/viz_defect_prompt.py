#!/usr/bin/env python3
"""
DefectPrompt 可视化 | DefectPrompt Visualization.
==================================================

观察 DPG 的 K 个 heatmap 学会了关注哪些区域。
Visualize what regions each of the K DPG heatmaps learned to attend to.

用法 | Usage::

    # 查看最佳 checkpoint 的 heatmap
    python tools/viz/viz_defect_prompt.py \\
        --checkpoint runs/neuseg_DPG_K5_0717_1437/best_model.pt \\
        --num-samples 6

    # 对比两个 checkpoint
    python tools/viz/viz_defect_prompt.py \\
        --checkpoint runs/neuseg_DPG_K5_0717_1437/best_model.pt \\
        --checkpoint-b runs/neuseg_DPG_K5_0717_1500/best_model.pt \\
        --num-samples 4

输出 | Output:
    runs/<exp>/viz_heatmaps.png — 每行: 原图 | GT | Pred | K 个 heatmap
"""

from __future__ import annotations

import sys, argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from adatile.backbone import FastSAMBackbone
from adatile.decoder.prompt_decoder import PromptDecoderP3P4
from adatile.datasets.neu_seg import NEUSegDataset

NUM_CLASSES = 4
CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]
CLASS_COLORS = [
    [0, 0, 0],        # BG — 黑
    [1.0, 0.3, 0.3],  # Inclusion — 红
    [0.3, 0.6, 1.0],  # Patch — 蓝
    [0.3, 1.0, 0.3],  # Scratch — 绿
]


def build_label_image(mask: np.ndarray) -> np.ndarray:
    """整数标签 mask → RGB 图像 | Integer label mask → RGB image."""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


@torch.no_grad()
def infer_one(decoder, backbone, image, device):
    """
    单张推理 → pred, heatmaps | Single inference → pred, heatmaps.
    :return: pred_class [H, W], heatmaps [K, H_p2, W_p2], p2_feat for the image
    """
    H, W = image.shape[1:]
    img = image.unsqueeze(0).to(device)

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

    feats = backbone(img, extract_proto=True)
    pred_prob, heatmaps = decoder(feats["p2"], feats["p3"], feats["p4"])
    # pred_prob: [C, H/4, W/4], heatmaps: [K, H_p2, W_p2]

    pred_full = F.interpolate(
        pred_prob.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(0)
    pred_class = torch.argmax(pred_full, dim=0).cpu().numpy()

    heatmaps_np = heatmaps.cpu().numpy()  # [K, H_p2, W_p2]
    return pred_class, heatmaps_np


def plot_samples(checkpoint_path: str, output_path: str,
                 dataset: NEUSegDataset, device: torch.device,
                 num_samples: int = 6, seed: int = 42):
    """
    绘制多张样本的 heatmap 可视化。
    Plot heatmap visualization for multiple samples.
    """
    # ── 加载模型 | Load model ──
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    num_prompts = args.get("num_prompts", 5)
    ch = {"p2": 160, "p3": 960, "p4": 1280}

    backbone = FastSAMBackbone(freeze_backbone=True).to(device)
    backbone.eval()
    # probe channels
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    decoder = PromptDecoderP3P4(
        p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
        out_channels=NUM_CLASSES, prompt_dim=256, num_prompts=num_prompts,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()

    K = num_prompts
    epoch = ckpt.get("epoch", "?")

    # ── 选取样本 | Select samples ──
    rng = np.random.RandomState(seed)
    # 优先选有缺陷的样本 | Prefer samples with defects
    defect_indices = []
    normal_indices = []
    for i in range(len(dataset)):
        s = dataset[i]
        if (s["masks"] > 0).sum() > 0:
            defect_indices.append(i)
        else:
            normal_indices.append(i)

    selected = list(rng.choice(defect_indices, size=min(num_samples, len(defect_indices)), replace=False))
    # 加 1 个无缺陷样本看 heatmap 分布 | Add 1 defect-free sample
    if normal_indices and len(selected) < num_samples + 1:
        selected.append(rng.choice(normal_indices))

    # ── 绘图: 每行 = 原图 | GT | Pred | Heatmap×K ──
    n_cols = 3 + K  # image, GT, pred, K heatmaps
    n_rows = len(selected)

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.15, hspace=0.2)

    col_titles = ["Image", "GT", "Pred"] + [f"Prompt {k}" for k in range(K)]

    for row_idx, ds_idx in enumerate(tqdm(selected, desc="Inferring")):
        sample = dataset[ds_idx]
        image = sample["image"]  # [3, H, W]
        gt = sample["masks"].squeeze(0).numpy()  # [H, W]

        pred_class, heatmaps = infer_one(decoder, backbone, image, device)

        # 原图: [3, H, W] → [H, W, 3]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # ── 列 0: 原图 | Image ──
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(img_np)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Image", fontsize=9, fontweight="bold")

        # ── 列 1: GT ──
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(build_label_image(gt))
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("GT", fontsize=9, fontweight="bold")

        # ── 列 2: Pred ──
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(build_label_image(pred_class))
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Pred", fontsize=9, fontweight="bold")

        # ── 列 3+: K 个 heatmap ──
        for k in range(K):
            ax = fig.add_subplot(gs[row_idx, 3 + k])
            hm = heatmaps[k]  # [H_p2, W_p2]
            # 显示为彩色热力图，叠加在原图上
            hm_resized = F.interpolate(
                torch.from_numpy(hm).unsqueeze(0).unsqueeze(0),
                size=img_np.shape[:2], mode="bilinear", align_corners=False,
            ).squeeze().numpy()
            # Normalize to [0, 1] for visualization
            hm_min, hm_max = hm_resized.min(), hm_resized.max()
            if hm_max - hm_min > 1e-8:
                hm_resized = (hm_resized - hm_min) / (hm_max - hm_min)

            # Overlay heatmap on grayscale image
            img_gray = 0.5 * img_np.mean(axis=-1)  # [H, W]
            ax.imshow(img_gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(hm_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Prompt {k}", fontsize=9, fontweight="bold")

    # ── 总标题 | Suptitle ──
    fig.suptitle(
        f"DefectPrompt Heatmaps — Epoch {epoch} — K={K}\n"
        f"Checkpoint: {Path(checkpoint_path).parent.name}/{Path(checkpoint_path).name}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_heatmap_stats(checkpoint_path: str, output_path: str,
                       dataset: NEUSegDataset, device: torch.device,
                       max_samples: int = 50, seed: int = 42):
    """
    绘制 heatmap 统计: 每个 prompt 的激活分布 + per-class 响应偏好。
    Plot heatmap statistics: activation distribution per prompt + per-class preference.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    num_prompts = args.get("num_prompts", 5)

    backbone = FastSAMBackbone(freeze_backbone=True).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels
    decoder = PromptDecoderP3P4(
        p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
        out_channels=NUM_CLASSES, prompt_dim=256, num_prompts=num_prompts,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    K = num_prompts

    # 收集每个 prompt 在每个类别区域的激活值
    # Collect per-prompt activation on each class region
    per_class_activation = {c: [[] for _ in range(K)] for c in range(1, NUM_CLASSES)}  # skip BG
    prompt_entropy = []  # 每个样本的 prompt 熵 | per-sample prompt entropy

    indices = list(range(len(dataset)))
    rng = np.random.RandomState(seed)
    selected = rng.choice(indices, size=min(max_samples, len(indices)), replace=False)

    for ds_idx in tqdm(selected, desc="Collecting stats", leave=False):
        sample = dataset[ds_idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).numpy()
        H, W = gt.shape

        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        feats = backbone(img, extract_proto=True)
        _pred, heatmaps = decoder(feats["p2"], feats["p3"], feats["p4"])
        # heatmaps: [K, H_p2, W_p2] — raw logits (before spatial softmax)

        # 对每个 heatmap 做 spatial softmax 得到注意力分布
        hm_flat = heatmaps.flatten(1)  # [K, H*W]
        attn = hm_flat.softmax(dim=-1)  # [K, H*W]
        attn = attn.reshape(K, heatmaps.shape[1], heatmaps.shape[2])  # [K, H_p2, W_p2]

        # 计算熵: H = -sum(p * log p) per prompt
        hm_probs = hm_flat.softmax(dim=-1) + 1e-10  # [K, H*W]
        ent = -(hm_probs * hm_probs.log()).sum(dim=-1).mean().item()  # 平均熵
        max_ent = np.log(H * W)  # 最大可能熵 (均匀分布)
        prompt_entropy.append(ent / max_ent)  # 归一化

        # Per-class activation: gt 上采样到 heatmap 分辨率
        gt_resized = F.interpolate(
            torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float(),
            size=attn.shape[1:], mode="nearest",
        ).squeeze().numpy()

        for c in range(1, NUM_CLASSES):
            c_mask = (gt_resized == c)
            if c_mask.sum() > 0:
                for k in range(K):
                    per_class_activation[c][k].append(attn[k][c_mask].mean().item())

    # ── 绘图 | Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Per-prompt per-class mean activation
    ax = axes[0]
    x = np.arange(K)
    width = 0.25
    colors = ["#ff6666", "#6699ff", "#66cc66"]
    for ci, c in enumerate(range(1, NUM_CLASSES)):
        means = [np.mean(per_class_activation[c][k]) if per_class_activation[c][k] else 0 for k in range(K)]
        ax.bar(x + ci * width, means, width, color=colors[ci], alpha=0.8, label=CLASS_NAMES[c])
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Per-Class Activation per Prompt")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k) for k in range(K)])
    ax.legend(fontsize=8)

    # (b) 熵分布 | Entropy distribution
    ax = axes[1]
    ax.hist(prompt_entropy, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(prompt_entropy), color="red", linestyle="--", label=f"Mean={np.mean(prompt_entropy):.3f}")
    ax.set_xlabel("Normalized Entropy (1=uniform)")
    ax.set_title("Prompt Attention Entropy Distribution")
    ax.legend(fontsize=8)

    # (c) 每个 prompt 的平均激活强度 | Mean activation per prompt
    ax = axes[2]
    all_acts = []
    for k in range(K):
        k_acts = []
        for c in range(1, NUM_CLASSES):
            k_acts.extend(per_class_activation[c][k])
        all_acts.append(np.mean(k_acts) if k_acts else 0)
    bars = ax.bar(range(K), all_acts, color=["#ff6666", "#6699ff", "#66cc66", "#ffcc00", "#cc66ff"][:K])
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Overall Activation per Prompt")
    for bar, val in zip(bars, all_acts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", fontsize=8)

    fig.suptitle(f"DefectPrompt Statistics — K={K}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DefectPrompt heatmaps")
    p.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    p.add_argument("--checkpoint-b", type=str, default=None,
                   help="可选第二个 checkpoint 用于对比 | Optional second checkpoint for comparison")
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--num-samples", type=int, default=6, help="可视化样本数")
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 (默认从 checkpoint 路径推导)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 输出目录 | Output dir
    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.output_dir = str(ckpt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据集 | Dataset
    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"Dataset: {len(dataset)} test samples")

    # ── 热力图可视化 | Heatmap visualization ──
    out_heatmaps = out_dir / "viz_heatmaps.png"
    plot_samples(args.checkpoint, str(out_heatmaps), dataset, device,
                 num_samples=args.num_samples, seed=args.seed)

    # ── 统计分析 | Statistical analysis ──
    out_stats = out_dir / "viz_prompt_stats.png"
    plot_heatmap_stats(args.checkpoint, str(out_stats), dataset, device, seed=args.seed)

    # ── 第二 checkpoint 对比 | Second checkpoint comparison ──
    if args.checkpoint_b:
        out_heatmaps_b = out_dir / "viz_heatmaps_b.png"
        plot_samples(args.checkpoint_b, str(out_heatmaps_b), dataset, device,
                     num_samples=args.num_samples, seed=args.seed)
        out_stats_b = out_dir / "viz_prompt_stats_b.png"
        plot_heatmap_stats(args.checkpoint_b, str(out_stats_b), dataset, device, seed=args.seed)

    print("\nDone. Outputs:")
    print(f"  {out_heatmaps}")
    print(f"  {out_stats}")
    if args.checkpoint_b:
        print(f"  {out_heatmaps_b}")
        print(f"  {out_stats_b}")


if __name__ == "__main__":
    main()
