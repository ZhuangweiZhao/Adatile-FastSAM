"""Diagnose NaN in Ada-SPM forward pass.

Run: python tools/debug_nan.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.config import Config
from adatile.sparse.fpn_fusion import MultiScaleFPNFusion
from adatile.sparse.ada_spm import AdaSPM, SpatialTransformerRefine, DensityHead


def check_nan(tensor, name):
    """Return True if NaN found, with diagnostic info."""
    if tensor is None:
        print(f"  {name}: None")
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"NaN={has_nan} Inf={has_inf} "
              f"min={tensor[~torch.isnan(tensor)].min().item():.4f} "
              f"max={tensor[~torch.isnan(tensor)].max().item():.4f}")
    else:
        print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"min={tensor.min().item():.4f} max={tensor.max().item():.4f} OK")
    return has_nan or has_inf


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AMP enabled: {torch.cuda.is_available()}")
    print()

    # ── Test 1: FPN fusion in fp32 ─────────────────────────────────
    print("=" * 60)
    print("Test 1: FPN Fusion (fp32)")
    print("=" * 60)
    fpn = MultiScaleFPNFusion(out_dim=256).to(device)
    dummy_features = {
        "p2": torch.randn(1, 256, 128, 128, device=device),
        "p3": torch.randn(1, 512, 64, 64, device=device),
        "p4": torch.randn(1, 1024, 32, 32, device=device),
        "p5": torch.randn(1, 2048, 16, 16, device=device),
    }
    with torch.no_grad():
        fused, pyramid = fpn(dummy_features)
    check_nan(fused, "FPN fused output (fp32)")
    print()

    # ── Test 2: FPN fusion in fp16 ─────────────────────────────────
    print("=" * 60)
    print("Test 2: FPN Fusion (fp16)")
    print("=" * 60)
    fpn_fp16 = MultiScaleFPNFusion(out_dim=256).to(device)
    dummy_features_fp16 = {k: v.half() for k, v in dummy_features.items()}
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            fused_fp16, _ = fpn_fp16(dummy_features_fp16)
    check_nan(fused_fp16, "FPN fused output (fp16)")
    print()

    # ── Test 3: SpatialTransformerRefine in fp32 ───────────────────
    print("=" * 60)
    print("Test 3: SpatialTransformerRefine (fp32)")
    print("=" * 60)
    transformer = SpatialTransformerRefine(dim=256, num_heads=4, window_size=8).to(device)
    x = torch.randn(1, 256, 16, 16, device=device)  # 512/32 = 16
    with torch.no_grad():
        out = transformer(x)
    check_nan(out, "Transformer output (fp32)")
    print()

    # ── Test 4: SpatialTransformerRefine in fp16 ───────────────────
    print("=" * 60)
    print("Test 4: SpatialTransformerRefine (fp16)")
    print("=" * 60)
    transformer_fp16 = SpatialTransformerRefine(dim=256, num_heads=4, window_size=8).to(device)
    x_fp16 = torch.randn(1, 256, 16, 16, device=device).half()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out_fp16 = transformer_fp16(x_fp16)
    check_nan(out_fp16, "Transformer output (fp16)")
    print()

    # ── Test 5: DensityHead in fp32 ────────────────────────────────
    print("=" * 60)
    print("Test 5: DensityHead (fp32)")
    print("=" * 60)
    dhead = DensityHead(in_dim=256, hidden_dim=128).to(device)
    with torch.no_grad():
        density = dhead(x)
    check_nan(density, "Density output (fp32)")
    print()

    # ── Test 6: DensityHead in fp16 ────────────────────────────────
    print("=" * 60)
    print("Test 6: DensityHead (fp16)")
    print("=" * 60)
    dhead_fp16 = DensityHead(in_dim=256, hidden_dim=128).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            density_fp16 = dhead_fp16(x_fp16)
    check_nan(density_fp16, "Density output (fp16)")
    print()

    # ── Test 7: Full AdaSPM in fp32 ────────────────────────────────
    print("=" * 60)
    print("Test 7: Full AdaSPM forward (fp32)")
    print("=" * 60)
    adaspm = AdaSPM(
        in_channels_list=[256, 512, 1024, 2048],
        fusion_dim=256,
        hidden_dim=128,
        num_tile_sizes=4,
        use_transformer=True,
    ).to(device)
    with torch.no_grad():
        output = adaspm(dummy_features)
    check_nan(output.importance, "importance (fp32)")
    check_nan(output.density, "density (fp32)")
    check_nan(output.granularity_soft, "granularity_soft (fp32)")
    print()

    # ── Test 8: Full AdaSPM in fp16 ────────────────────────────────
    print("=" * 60)
    print("Test 8: Full AdaSPM forward (fp16) [CRITICAL TEST]")
    print("=" * 60)
    adaspm_fp16 = AdaSPM(
        in_channels_list=[256, 512, 1024, 2048],
        fusion_dim=256,
        hidden_dim=128,
        num_tile_sizes=4,
        use_transformer=True,
    ).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output_fp16 = adaspm_fp16(dummy_features_fp16)
    has_nan_imp = check_nan(output_fp16.importance, "importance (fp16)")
    has_nan_den = check_nan(output_fp16.density, "density (fp16)")
    has_nan_gra = check_nan(output_fp16.granularity_soft, "granularity_soft (fp16)")

    if has_nan_imp or has_nan_den or has_nan_gra:
        print()
        print("⚠️  NaN DETECTED in fp16 forward pass!")
        print("   Root cause: AMP autocast + GroupNorm/Attention instability")
    else:
        print()
        print("✓ No NaN in fp16 forward pass with random weights")
        print("  NaN may appear during training with real gradients")

    # ── Test 9: NaN in importance → planner behavior ───────────────
    print()
    print("=" * 60)
    print("Test 9: NaN importance → planner behavior")
    print("=" * 60)
    nan_importance = torch.full((1, 1, 64, 64), float('nan'))
    print(f"  NaN > 0.15: {(float('nan') > 0.15)}")
    print(f"  NaN < 0.15: {(float('nan') < 0.15)}")
    print(f"  NaN < 0.075: {(float('nan') < 0.075)}")
    print("  → NaN comparisons ALWAYS return False")
    print("  → All cells fall to 'else' branch (high-importance path)")
    print("  → This is why tiles ARE generated despite NaN importance")


if __name__ == "__main__":
    main()
