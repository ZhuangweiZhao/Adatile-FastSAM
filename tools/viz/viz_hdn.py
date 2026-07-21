#!/usr/bin/env python3
"""
HDN (Heatmap Denoiser) Visualization | HDN 热力图去噪可视化.
=============================================================

可视化 HeatmapDenoiser 的去噪效果：对比原始 Sobel 梯度 vs 去噪后热力图，
以及空间门控图。
Visualize HeatmapDenoiser: compare raw Sobel gradient vs denoised heatmap,
and the spatial gate map.

用法 | Usage::

    # 仅看 Sobel 梯度 (不加载模型)
    python tools/viz/viz_hdn.py --data-root data/NEU_Seg

    # 加载训练好的 HDN checkpoint
    python tools/viz/viz_hdn.py --data-root data/NEU_Seg \\
        --checkpoint runs/neuseg_DAFRN_HDN_XXXX/best_model.pt
"""

from __future__ import annotations

import argparse, sys, glob, os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_proj_root))
sys.path.insert(0, str(_proj_root / "thirdLibrary" / "FastSAM"))

from adatile.rectify.hdn import HeatmapDenoiser


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """RGB → Gray: 0.299R + 0.587G + 0.114B. img: [H, W, 3] uint8 → [H, W] float32."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)


def sobel_gradient_cv(gray: np.ndarray) -> np.ndarray:
    """
    经典 CV Sobel 梯度 (鲁棒归一化) | Classical CV Sobel with robust normalization.
    """
    gray_u8 = (gray * 255).astype(np.uint8)
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Robust percentile normalize
    p_low, p_high = np.percentile(mag, [2, 98])
    if p_high > p_low:
        mag = np.clip(mag, p_low, p_high)
        mag = (mag - p_low) / (p_high - p_low + 1e-8)
    return mag.astype(np.float32)


def sobel_gradient_torch(gray_np: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    PyTorch Sobel 梯度 (与训练时一致) | PyTorch Sobel (consistent with training).
    """
    gray_t = torch.from_numpy(gray_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='reflect'), sobel_y)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze()
    # Robust percentile normalize
    m = mag.flatten()
    p_low = torch.quantile(m, 0.02)
    p_high = torch.quantile(m, 0.98)
    if p_high > p_low:
        mag = torch.clamp(mag, p_low, p_high)
        mag = (mag - p_low) / (p_high - p_low + 1e-8)
    return mag.cpu().numpy()


def load_hdn(checkpoint_path: str, device: str = "cpu") -> HeatmapDenoiser:
    """Load HDN from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hdn = HeatmapDenoiser(in_channels=1)
    if "hdn_state_dict" in ckpt:
        hdn.load_state_dict(ckpt["hdn_state_dict"])
    else:
        # Try keys with 'hdn.' prefix
        hdn_state = {k.replace("hdn.", ""): v for k, v in ckpt.items()
                     if k.startswith("hdn.")}
        if hdn_state:
            hdn.load_state_dict(hdn_state)
        else:
            raise KeyError("No hdn_state_dict found in checkpoint. "
                          f"Keys: {list(ckpt.keys())}")
    hdn.to(device)
    hdn.eval()
    return hdn


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def visualize(
    img_path: str,
    gt_path: str | None,
    hdn: HeatmapDenoiser | None,
    device: str,
    save_dir: Path,
    sample_name: str,
):
    """Generate 6-panel visualization for one sample."""
    # Load image
    img = np.array(Image.open(img_path).convert("RGB"))
    H_orig, W_orig = img.shape[:2]

    # Resize to 896×896 for backbone compatibility
    img_resized = cv2.resize(img, (896, 896))
    gray = rgb_to_gray(img_resized)

    # ── Sobel gradient (CV + PyTorch) ──
    grad_cv = sobel_gradient_cv(gray)
    grad_torch = sobel_gradient_torch(gray, device)

    # ── HDN denoising ──
    if hdn is not None:
        grad_t = torch.from_numpy(grad_torch).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised = hdn(grad_t).squeeze().cpu().numpy()
            gate = hdn.get_gate_map(grad_t).squeeze().cpu().numpy()
        has_hdn = True
    else:
        denoised = None
        gate = None
        has_hdn = False

    # ── GT mask ──
    if gt_path and os.path.exists(gt_path):
        gt_img = np.array(Image.open(gt_path))
        if gt_img.ndim == 3:
            gt_img = gt_img[:, :, 0] if gt_img.shape[2] >= 1 else gt_img
        gt_resized = cv2.resize(gt_img, (896, 896), interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_resized > 0).astype(np.float32)
    else:
        gt_binary = None

    # ── Plot ──
    n_cols = 7 if has_hdn else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))

    titles = ["Original", "GT Mask"]
    images = [img_resized, gt_binary]

    titles.append("Sobel CV (robust)")
    images.append(grad_cv)
    titles.append("Sobel Torch")
    images.append(grad_torch)

    if has_hdn:
        titles.append("HDN Denoised")
        images.append(denoised)
        titles.append("Gate Map")
        images.append(gate)
        # Overlay: denoised heatmap on image
        overlay = img_resized.astype(np.float32) / 255.0 * 0.5
        denoised_rgb = np.stack([denoised, np.zeros_like(denoised), np.zeros_like(denoised)], -1)
        overlay = np.clip(overlay + denoised_rgb * 0.5, 0, 1)
        titles.append("Overlay (red=defect)")
        images.append(overlay)

    for i, (ax, title, im) in enumerate(zip(axes, titles, images)):
        if im is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        elif im.ndim == 2:
            vmin, vmax = (0, 1)
            cmap = "hot" if i >= 2 else "gray"
            ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # RGB image: normalize uint8 [0,255] → float [0,1]
            im_disp = im.astype(np.float32) / 255.0 if im.max() > 1.0 else im
            ax.imshow(np.clip(im_disp, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"HDN Analysis: {sample_name}\n"
                f"{'Trained HDN' if has_hdn else 'Untrained (zero-init, gate=0.5)'}",
                fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = save_dir / f"hdn_{sample_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Print stats ──
    print(f"  {sample_name}:")
    print(f"    Sobel CV:    mean={grad_cv.mean():.4f}, std={grad_cv.std():.4f}")
    if has_hdn:
        print(f"    Denoised:    mean={denoised.mean():.4f}, std={denoised.std():.4f}")
        print(f"    Gate:        mean={gate.mean():.4f}, "
              f"suppressed(<0.3)={np.mean(gate<0.3)*100:.1f}%, "
              f"active(>0.7)={np.mean(gate>0.7)*100:.1f}%")
        if gt_binary is not None:
            # Compute overlap metrics
            gt_defect = gt_binary > 0.5
            if gt_defect.any():
                denoised_bin = denoised > 0.5
                tp = (denoised_bin & gt_defect).sum()
                fp = (denoised_bin & ~gt_defect).sum()
                fn = (~denoised_bin & gt_defect).sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                print(f"    Precision:   {precision:.4f}, Recall: {recall:.4f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HDN Visualization")
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="HDN checkpoint path (skip for untrained)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="analysis/hdn")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-indices", type=str, default=None,
                       help="Comma-separated sample indices, e.g. '0,5,10'")
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load HDN (optional) ──
    hdn = None
    if args.checkpoint:
        print(f"Loading HDN from: {args.checkpoint}")
        hdn = load_hdn(args.checkpoint, args.device)
        print(f"  {hdn}")

    # ── Find images ──
    img_dir = Path(args.data_root) / "images" / "training"
    img_files = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not img_files:
        img_dir = Path(args.data_root) / "images"
        img_files = sorted(glob.glob(str(img_dir / "*.jpg")))

    if not img_files:
        print(f"No images found in {args.data_root}")
        return

    # ── Find GT masks ──
    gt_dir = Path(args.data_root) / "annotations" / "training"
    if not gt_dir.exists():
        gt_dir = Path(args.data_root) / "annotations"

    # ── Select samples ──
    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",")]
    else:
        step = max(1, len(img_files) // args.num_samples)
        indices = list(range(0, len(img_files), step))[:args.num_samples]

    print(f"Processing {len(indices)} samples...")

    for idx in indices:
        img_path = img_files[idx]
        stem = Path(img_path).stem

        # Find corresponding GT
        gt_path = gt_dir / f"{stem}.png"
        if not gt_path.exists():
            gt_path = None

        try:
            visualize(img_path, str(gt_path) if gt_path else None,
                     hdn, args.device, save_dir, stem)
        except Exception as e:
            print(f"  Error on {stem}: {e}")

    print(f"\nDone. Figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
