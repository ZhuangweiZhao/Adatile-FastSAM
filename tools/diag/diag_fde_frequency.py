#!/usr/bin/env python3
"""
FDE 频域诊断 | FDE Frequency Domain Diagnostic.
=================================================

诊断 FFT 版 FDE 在频域是否真正学习了任何调制模式.
Diagnose whether FFT-based FDE actually learned any modulation in frequency domain.

检查内容 | What We Check:
    1. 幅度谱调制量 (|X_out| - |X_in|) 在频域的分布
       Magnitude modulation (|X_out| - |X_in|) distribution in frequency domain
    2. 方位角平均功率谱 (1D radial power spectrum) — 低频 vs 高频差异
       Azimuthally-averaged power spectrum — low-freq vs high-freq difference
    3. 学到的径向剖面 vs 初始化剖面
       Learned radial profile vs initialization profile
    4. 逐通道频率调制 (每个通道的高频/低频比变化)
       Per-channel frequency modulation (HF/LF ratio change per channel)

如果 FDE 没学到任何东西, 我们将看到:
    - 调制量在整个频域均匀分布 (≈1.0 everywhere)
    - 学习剖面 = 初始化剖面 (几乎重叠)
    - 高频/低频比不变

用法 | Usage::

    python tools/diag/diag_fde_frequency.py \
        --checkpoint runs/neuseg_DAFRN_DCR+FDE+CDF_0717_2338/best_model.pt \
        --num-samples 20 --device cuda
"""

from __future__ import annotations

import sys, argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from adatile.backbone import FastSAMBackbone
from adatile.datasets.neu_seg import NEUSegDataset


# ═══════════════════════════════════════════════════════════════════
# 旧 FFT 版 FDE (内联, 用于加载旧 checkpoint 做诊断)
# Old FFT-based FDE (inlined for loading old checkpoint for diagnosis)
# ═══════════════════════════════════════════════════════════════════

class _OldFFT_FDE(nn.Module):
    """FFT 版 FDE (诊断用 | For diagnosis only)."""
    def __init__(self, channels, num_bins=32):
        super().__init__()
        self.channels = channels
        self.num_bins = num_bins
        self.alpha = nn.Parameter(torch.tensor(0.15))
        profile_init = torch.linspace(0.05, 0.8, num_bins)
        self.profile = nn.Parameter(profile_init)
        self.channel_scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def _build_radial_mask(self, H_freq, W_freq, device, dtype):
        y = torch.arange(H_freq, device=device, dtype=dtype)
        x = torch.arange(W_freq, device=device, dtype=dtype)
        y_norm = y / max(H_freq - 1, 1)
        x_norm = x / max(W_freq - 1, 1)
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        dist = torch.sqrt(yy ** 2 + xx ** 2) / math.sqrt(2)
        r_indices = dist * (self.num_bins - 1)
        profile_c = F.softplus(self.profile, beta=5.0)
        idx_floor = torch.floor(r_indices).long().clamp(0, self.num_bins - 1)
        idx_ceil = torch.ceil(r_indices).long().clamp(0, self.num_bins - 1)
        frac = (r_indices - idx_floor.float()).clamp(0.0, 1.0)
        w_floor = profile_c[idx_floor]
        w_ceil = profile_c[idx_ceil]
        weight = w_floor * (1 - frac) + w_ceil * frac
        return weight.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x.float(), norm="ortho")
        H_freq, W_freq = X.shape[2], X.shape[3]
        radial_mask = self._build_radial_mask(H_freq, W_freq, x.device, x.dtype)
        mag = torch.abs(X)
        phase = torch.angle(X)
        mask = radial_mask * self.channel_scale
        alpha_c = torch.clamp(self.alpha, 0.0, 2.0)
        modulated_mag = mag * (1.0 + alpha_c * mask)
        X_mod = modulated_mag * torch.exp(1j * phase)
        x_freq = torch.fft.irfft2(X_mod, s=(H, W), norm="ortho")
        return x + 0.5 * (x_freq - x)

    def get_profile(self):
        return F.softplus(self.profile, beta=5.0)


def load_model(checkpoint_path: str, device: torch.device):
    """加载旧 checkpoint 的 Backbone + DCR + FDE(FFT版) | Load old checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args_dict = ckpt.get("args", {})

    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    # 构建 DCR (SE-style channel attention) | Build DCR
    from adatile.rectify.dcr import DefectChannelReweighting
    dcr_p3 = DefectChannelReweighting(ch["p3"]).to(device) if args_dict.get("enable_dcr", True) else None
    dcr_p4 = DefectChannelReweighting(ch["p4"]).to(device) if args_dict.get("enable_dcr", True) else None

    # 构建旧 FFT FDE | Build old FFT FDE
    fde_p3 = _OldFFT_FDE(ch["p3"]).to(device) if args_dict.get("enable_fde", True) else None
    fde_p4 = _OldFFT_FDE(ch["p4"]).to(device) if args_dict.get("enable_fde", True) else None

    # 加载 state dict (只加载存在的 key) | Load state dict (only existing keys)
    frn_sd = ckpt["frn_state_dict"]
    if dcr_p3 is not None:
        dcr_p3.load_state_dict({k.replace("dcr_p3.", ""): v for k, v in frn_sd.items() if k.startswith("dcr_p3.")})
    if dcr_p4 is not None:
        dcr_p4.load_state_dict({k.replace("dcr_p4.", ""): v for k, v in frn_sd.items() if k.startswith("dcr_p4.")})
    if fde_p3 is not None:
        fde_p3.load_state_dict({k.replace("fde_p3.", ""): v for k, v in frn_sd.items() if k.startswith("fde_p3.")})
    if fde_p4 is not None:
        fde_p4.load_state_dict({k.replace("fde_p4.", ""): v for k, v in frn_sd.items() if k.startswith("fde_p4.")})

    dcr_p3.eval(); dcr_p4.eval(); fde_p3.eval(); fde_p4.eval()

    return backbone, dcr_p3, dcr_p4, fde_p3, fde_p4, ckpt


@torch.no_grad()
def collect_fft_spectra(fde_module, features_list: list[torch.Tensor]) -> dict:
    """
    收集 FDE 前后的 FFT 幅度谱 | Collect FFT magnitude spectra before/after FDE.

    对每个特征图:
        1. FFT → 幅度谱 |X|
        2. 方位角平均 → 1D 功率谱 profile(r)
        3. 逐频率 bin 的调制比 = |X_after| / |X_before|

    :param fde_module: FreqDefectEnhance instance (FFT version).
    :param features_list: list of [C, H, W] feature tensors.
    :return: dict with spectra data.
    """
    num_bins = fde_module.num_bins
    all_profiles_before = []  # [num_samples, num_bins]
    all_profiles_after = []
    all_modulation_maps = []  # [num_samples, H_freq, W_freq]

    for feat in tqdm(features_list, desc="Collecting FFT spectra", leave=False):
        x = feat.unsqueeze(0)  # [1, C, H, W]
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        # ── FFT before FDE ──
        X_before = torch.fft.rfft2(x.float(), norm="ortho")  # [1, C, H, W//2+1]
        mag_before = torch.abs(X_before).squeeze(0)  # [C, H, W//2+1]

        # ── FDE forward ──
        x_after = fde_module(x)

        # ── FFT after FDE ──
        X_after = torch.fft.rfft2(x_after.float(), norm="ortho")  # [1, C, H, W//2+1]
        mag_after = torch.abs(X_after).squeeze(0)  # [C, H, W//2+1]

        # ── 方位角平均 → 1D 功率谱 | Azimuthal average → 1D power spectrum ──
        H_freq, W_freq = mag_before.shape[1], mag_before.shape[2]

        # 距离图 (重用 FDE 的 _build_radial_mask 逻辑)
        y = torch.arange(H_freq, device=x.device, dtype=torch.float32)
        x_coord = torch.arange(W_freq, device=x.device, dtype=torch.float32)
        y_norm = y / max(H_freq - 1, 1)
        x_norm = x_coord / max(W_freq - 1, 1)
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        dist = torch.sqrt(yy ** 2 + xx ** 2) / math.sqrt(2)  # [H_freq, W_freq], r ∈ [0, 1]

        # 将 r ∈ [0, 1] 映射到 [0, num_bins-1]
        bin_idx = torch.clamp(
            (dist * (num_bins - 1)).long(), 0, num_bins - 1
        )  # [H_freq, W_freq]

        # Per-bin mean magnitude (averaged over channels + spatial locations in each bin)
        profile_before = torch.zeros(num_bins, device=x.device)
        profile_after = torch.zeros(num_bins, device=x.device)
        for b in range(num_bins):
            mask_b = (bin_idx == b)
            if mask_b.sum() > 0:
                profile_before[b] = mag_before[:, mask_b].mean()
                profile_after[b] = mag_after[:, mask_b].mean()

        all_profiles_before.append(profile_before.cpu().numpy())
        all_profiles_after.append(profile_after.cpu().numpy())

        # ── 频率调制图 (magnitude ratio) | Frequency modulation map ──
        # Averaged over channels
        modulation = (mag_after / (mag_before + 1e-8)).mean(dim=0)  # [H_freq, W_freq]
        all_modulation_maps.append(modulation.cpu().numpy())

    return {
        "profiles_before": np.array(all_profiles_before),       # [N, num_bins]
        "profiles_after": np.array(all_profiles_after),          # [N, num_bins]
        "modulation_maps": np.array(all_modulation_maps),        # [N, H_freq, W_freq]
        "num_bins": num_bins,
    }


@torch.no_grad()
def collect_per_channel_modulation(fde_module, features_list: list[torch.Tensor]) -> dict:
    """
    逐通道频率调制分析 | Per-channel frequency modulation analysis.

    对每个通道计算: HF_ratio_after / HF_ratio_before
    HF_ratio = mean(mag[high-freq bins]) / mean(mag[low-freq bins])

    如果 FDE 增强了高频, HF_ratio_after > HF_ratio_before.
    """
    num_bins = fde_module.num_bins
    # 低频 = bins 0..num_bins//3, 高频 = bins 2*num_bins//3..
    lf_bins = range(0, num_bins // 3)
    hf_bins = range(2 * num_bins // 3, num_bins)

    all_hf_ratios_before = []
    all_hf_ratios_after = []

    for feat in tqdm(features_list, desc="Per-channel modulation", leave=False):
        x = feat.unsqueeze(0)  # [1, C, H, W]
        C = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        H_freq, W_freq = H, W // 2 + 1

        # Distance bins
        y = torch.arange(H_freq, device=x.device, dtype=torch.float32)
        xc = torch.arange(W_freq, device=x.device, dtype=torch.float32)
        y_norm = y / max(H_freq - 1, 1)
        x_norm = xc / max(W_freq - 1, 1)
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        dist = torch.sqrt(yy ** 2 + xx ** 2) / math.sqrt(2)
        bin_idx = torch.clamp((dist * (num_bins - 1)).long(), 0, num_bins - 1)

        lf_mask = torch.zeros_like(dist, dtype=torch.bool)
        hf_mask = torch.zeros_like(dist, dtype=torch.bool)
        for b in lf_bins: lf_mask |= (bin_idx == b)
        for b in hf_bins: hf_mask |= (bin_idx == b)

        # Before
        X_before = torch.fft.rfft2(x.float(), norm="ortho")
        mag_b = torch.abs(X_before).squeeze(0)  # [C, H_freq, W_freq]
        lf_b = mag_b[:, lf_mask].mean(dim=1)  # [C]
        hf_b = mag_b[:, hf_mask].mean(dim=1)  # [C]
        ratio_b = hf_b / (lf_b + 1e-8)

        # After
        x_after = fde_module(x)
        X_after = torch.fft.rfft2(x_after.float(), norm="ortho")
        mag_a = torch.abs(X_after).squeeze(0)
        lf_a = mag_a[:, lf_mask].mean(dim=1)
        hf_a = mag_a[:, hf_mask].mean(dim=1)
        ratio_a = hf_a / (lf_a + 1e-8)

        all_hf_ratios_before.append(ratio_b.cpu().numpy())
        all_hf_ratios_after.append(ratio_a.cpu().numpy())

    return {
        "hf_ratio_before": np.array(all_hf_ratios_before),  # [N, C]
        "hf_ratio_after": np.array(all_hf_ratios_after),     # [N, C]
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_radial_power_spectrum(spectra_p3: dict, spectra_p4: dict,
                                fde_p3, fde_p4, save_path: Path):
    """方位角平均功率谱 | Azimuthally-averaged power spectrum."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    r = np.linspace(0, 1, spectra_p3["num_bins"])

    for col, (name, spectra, fde) in enumerate([
        ("P3 (stride 8)", spectra_p3, fde_p3),
        ("P4 (stride 16)", spectra_p4, fde_p4),
    ]):
        # ── Row 0: Power spectrum before vs after ──
        before_mean = spectra["profiles_before"].mean(axis=0)
        before_std = spectra["profiles_before"].std(axis=0)
        after_mean = spectra["profiles_after"].mean(axis=0)
        after_std = spectra["profiles_after"].std(axis=0)

        axes[0, col].plot(r, before_mean, color="#3498DB", lw=2, label="Before FDE")
        axes[0, col].fill_between(r, before_mean - before_std, before_mean + before_std,
                                  color="#3498DB", alpha=0.15)
        axes[0, col].plot(r, after_mean, color="#E74C3C", lw=2, label="After FDE")
        axes[0, col].fill_between(r, after_mean - after_std, after_mean + after_std,
                                  color="#E74C3C", alpha=0.15)
        axes[0, col].set_title(f"{name} — Radial Power Spectrum\n"
                               f"(azimuthal avg over {spectra['profiles_before'].shape[0]} samples)",
                               fontsize=10, fontweight="bold")
        axes[0, col].set_xlabel("Normalized Distance from DC (0=LF, 1=HF)")
        axes[0, col].set_ylabel("Mean Magnitude")
        axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3)

        # ── Row 1: Modulation ratio (after/before) vs r ──
        ratio = after_mean / (before_mean + 1e-8)
        axes[1, col].plot(r, ratio, color="#27AE60", lw=2.5, marker="o", markersize=3)
        axes[1, col].axhline(y=1.0, color="gray", ls="--", lw=1, alpha=0.5,
                             label="No modulation (ratio=1)")
        axes[1, col].fill_between(r, 1.0, ratio, alpha=0.2, color="#27AE60")
        axes[1, col].set_title(f"{name} — Frequency Modulation Ratio\n"
                               f"(|X_after| / |X_before| per frequency bin)\n"
                               f"Mean ratio={ratio.mean():.4f}, HF/LF ratio={ratio[-5:].mean()/ratio[:5].mean():.3f}",
                               fontsize=10, fontweight="bold")
        axes[1, col].set_xlabel("Normalized Distance from DC")
        axes[1, col].set_ylabel("Modulation Ratio (>1=boost, <1=suppress)")
        axes[1, col].legend(fontsize=8); axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_ylim(0.95, 1.05)

        # ── Annotate learned alpha ──
        alpha_val = fde.alpha.item() if fde is not None else "N/A"
        axes[1, col].text(0.95, 0.05, f"Learned α={alpha_val}",
                          transform=axes[1, col].transAxes, fontsize=9, ha="right",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("FDE Frequency Domain Analysis — Does FFT-based FDE actually modulate frequencies?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Power spectrum plot saved → {save_path}")


def plot_2d_modulation_map(spectra_p3: dict, spectra_p4: dict, save_path: Path):
    """2D 频率调制热力图 | 2D frequency modulation heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, (name, spectra) in enumerate([
        ("P3", spectra_p3), ("P4", spectra_p4),
    ]):
        # Average modulation map across samples
        mod_map = spectra["modulation_maps"].mean(axis=0)  # [H_freq, W_freq]
        H_freq, W_freq = mod_map.shape
        vmin = mod_map.min()
        vmax = mod_map.max()
        # Center at 1.0
        vlim = max(abs(vmin - 1), abs(vmax - 1))
        vmin = 1.0 - vlim
        vmax = 1.0 + vlim

        im = axes[col].imshow(mod_map, cmap="RdBu_r", origin="lower",
                               aspect="auto", vmin=vmin, vmax=vmax)
        axes[col].set_title(f"{name} — 2D Modulation Map\n"
                            f"|X_after| / |X_before| (avg over channels + samples)\n"
                            f"Range: [{mod_map.min():.4f}, {mod_map.max():.4f}]",
                            fontsize=10, fontweight="bold")
        axes[col].set_xlabel("Frequency X (rfft: 0=DC left, max freq right)")
        axes[col].set_ylabel("Frequency Y (0=DC bottom)")
        plt.colorbar(im, ax=axes[col], label="Modulation Ratio", shrink=0.85)

        # Annotate DC (bottom-left) and HF (top-right)
        axes[col].annotate("DC", xy=(0, 0), xytext=(W_freq*0.1, H_freq*0.1),
                           fontsize=9, color="black",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        axes[col].annotate("HF", xy=(W_freq-1, H_freq-1),
                           xytext=(W_freq*0.7, H_freq*0.8),
                           fontsize=9, color="black",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.suptitle("FDE — 2D Frequency Modulation Map (|X_after| / |X_before|)\n"
                 "Red=Suppress, Blue=Boost. If all white → nothing learned.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2D modulation map saved → {save_path}")


def plot_per_channel_hf_ratio(hf_data_p3: dict, hf_data_p4: dict, save_path: Path):
    """逐通道高频/低频比变化 | Per-channel HF/LF ratio change."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (name, hf_data, color) in enumerate([
        ("P3 (960 channels)", hf_data_p3, "#E74C3C"),
        ("P4 (1280 channels)", hf_data_p4, "#2980B9"),
    ]):
        ratio_before = hf_data["hf_ratio_before"].mean(axis=0)  # [C]
        ratio_after = hf_data["hf_ratio_after"].mean(axis=0)    # [C]
        ratio_change = ratio_after / (ratio_before + 1e-8)       # [C]

        axes[col].hist(ratio_change, bins=60, color=color, alpha=0.7,
                       edgecolor="white", lw=0.2)
        axes[col].axvline(x=1.0, color="gray", ls="--", lw=1.2, alpha=0.6,
                          label="No change (1.0)")
        axes[col].axvline(x=ratio_change.mean(), color="darkred", ls="-", lw=1.5,
                          label=f"Mean={ratio_change.mean():.4f}")

        # Check if any channel deviates from 1.0 significantly
        max_dev = max(abs(ratio_change.max() - 1), abs(ratio_change.min() - 1))
        axes[col].set_title(f"{name} — Per-Channel HF/LF Ratio Change\n"
                            f"Mean={ratio_change.mean():.4f}, "
                            f"MaxDev={max_dev:.4f}, "
                            f"{'✓ Learned' if max_dev > 0.02 else '⚠ Dead (all ≈1.0)'}",
                            fontsize=10, fontweight="bold")
        axes[col].set_xlabel("HF/LF Ratio Change (after/before)\n>1=more HF, <1=less HF")
        axes[col].set_ylabel("Num Channels")
        axes[col].legend(fontsize=8); axes[col].grid(True, alpha=0.3)

    plt.suptitle("FDE — Per-Channel High-Freq / Low-Freq Ratio Change\n"
                 "If all channels stay near 1.0 → FDE produces no frequency modulation.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-channel HF ratio plot saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FDE Frequency Domain Diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained DA-FRN checkpoint (FFT version)")
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if args.output_dir is None:
        out_dir = ckpt_path.parent / "diag_fde"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Load model ──
    print("Loading model...")
    backbone, dcr_p3, dcr_p4, fde_p3, fde_p4, ckpt = load_model(args.checkpoint, device)
    print(f"  mIoU: {ckpt.get('mIoU', 'N/A')}")
    print(f"  FDE P3 alpha: {fde_p3.alpha.item():.4f}" if fde_p3 else "  FDE P3: disabled")
    print(f"  FDE P4 alpha: {fde_p4.alpha.item():.4f}" if fde_p4 else "  FDE P4: disabled")

    # ── Load data ──
    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)),
                               replace=False)

    # ── Collect P3 and P4 features ──
    print(f"Collecting features from {len(indices)} samples...")
    p3_features, p4_features = [], []
    for idx in tqdm(indices, desc="Extracting features"):
        img = dataset[idx]["image"].unsqueeze(0).to(device)
        H, W = img.shape[2], img.shape[3]
        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        feats = backbone(img, extract_proto=False)
        # Only pass through DCR (FDE is the target of diagnosis)
        p3 = feats["p3"].squeeze(0)  # [C, H/8, W/8]
        p4 = feats["p4"].squeeze(0)  # [C, H/16, W/16]
        if dcr_p3 is not None:
            p3 = dcr_p3(p3.unsqueeze(0)).squeeze(0)
            p4 = dcr_p4(p4.unsqueeze(0)).squeeze(0)
        p3_features.append(p3)
        p4_features.append(p4)

    # ── 1. Collect FFT spectra ──
    print("\n=== 1. FFT Power Spectrum Analysis ===")
    print("  P3...")
    spectra_p3 = collect_fft_spectra(fde_p3, p3_features)
    print("  P4...")
    spectra_p4 = collect_fft_spectra(fde_p4, p4_features)

    plot_radial_power_spectrum(spectra_p3, spectra_p4, fde_p3, fde_p4,
                               out_dir / "radial_power_spectrum.png")

    # ── 2. 2D Modulation Map ──
    print("\n=== 2. 2D Frequency Modulation Map ===")
    plot_2d_modulation_map(spectra_p3, spectra_p4,
                           out_dir / "modulation_2d_map.png")

    # ── 3. Per-channel HF/LF ratio ──
    print("\n=== 3. Per-Channel HF/LF Ratio ===")
    hf_data_p3 = collect_per_channel_modulation(fde_p3, p3_features)
    hf_data_p4 = collect_per_channel_modulation(fde_p4, p4_features)

    plot_per_channel_hf_ratio(hf_data_p3, hf_data_p4,
                              out_dir / "per_channel_hf_ratio.png")

    # ── 4. Summary ──
    print(f"\n{'='*60}")
    print(f"  FDE Frequency Domain Diagnostic — Complete")
    if fde_p3 is not None:
        print(f"  FDE P3 alpha: {fde_p3.alpha.item():.6f}")
        p = fde_p3.get_profile()
        print(f"  FDE P3 profile: DC={p[0].item():.4f}, HF={p[-1].item():.4f}, "
              f"ratio={p[-1].item()/p[0].item():.2f}x")
    if fde_p4 is not None:
        print(f"  FDE P4 alpha: {fde_p4.alpha.item():.6f}")
        p = fde_p4.get_profile()
        print(f"  FDE P4 profile: DC={p[0].item():.4f}, HF={p[-1].item():.4f}, "
              f"ratio={p[-1].item()/p[0].item():.2f}x")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
