"""Model component builder — instantiates backbone, decoder, SPM.

Extracted from tools/train_as_fastsam.py to serve as the single source
of truth for model construction across all experiment scripts.

Usage:
    from adatile.engine.builder import build_components, collect_params, save_checkpoint
    backbone, decoder, spm = build_components(args, device, num_classes=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from adatile.backbone.fastsam_hook import FastSAMHookBackbone
from adatile.decoder.light_decoder import LightDecoder
from adatile.sparse.light_spm import LightSPM


def build_backbone(
    weight_file: str = "FastSAM-x.pt",
    out_channels: int = 128,
    unfreeze_layers: int = 2,
    image_size: Optional[int] = None,
    device: torch.device = None,
    output_levels: Optional[Dict[str, int]] = None,
) -> FastSAMHookBackbone:
    """Build FastSAM hook backbone.

    Args:
        weight_file: Path to FastSAM checkpoint.
        out_channels: Unified projection channels for outputs.
        unfreeze_layers: Number of backbone layers to unfreeze (0=all frozen).
        image_size: Expected input size (None=keep original).
        device: Target device.
        output_levels: Dict mapping name→layer_index, or config name
            ("default", "fpn2", "fpn4"). Default: "default" (P8, P4).
            Use "fpn4" for AdaSPM variants needing 4 FPN levels.

    Returns:
        FastSAMHookBackbone instance on the target device.
    """
    img_sz = None if (image_size is None or image_size == 0) else image_size
    backbone = FastSAMHookBackbone(
        weight_file,
        out_channels=out_channels,
        unfreeze_layers=unfreeze_layers,
        image_size=img_sz,
        output_levels=output_levels,
    )
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def build_decoder(
    in_channels: int = 128,
    decoder_channels: int = 64,
    num_classes: int = 1,
    device: torch.device = None,
) -> LightDecoder:
    """Build LightDecoder for binary/multi-class segmentation.

    Args:
        in_channels: Feature channels from backbone.
        decoder_channels: Internal decoder channels.
        num_classes: 1 for binary, >1 for multi-class.
        device: Target device.

    Returns:
        LightDecoder instance on the target device.
    """
    decoder = LightDecoder(in_channels, decoder_channels, num_classes)
    if device is not None:
        decoder = decoder.to(device)
    return decoder


def build_spm(
    spm_type: str = "light",
    in_channels: int = 128,
    device: torch.device = None,
    **kwargs,
) -> Optional[nn.Module]:
    """Build SPM (Sparse Perception Module).

    Args:
        spm_type: "light" (default 3-conv), "lite" (AdaSPM-Lite, 128ch FPN),
                  "full" (AdaSPM-Full, 256ch FPN + transformer),
                  "density_only" (no granularity head), or None (no SPM).
        in_channels: Feature channels for LightSPM.
        device: Target device.
        **kwargs: Passed to the SPM constructor (fusion_dim, hidden_dim,
                  use_transformer, lightweight, dropout, etc.).

    Returns:
        SPM module instance or None.
    """
    if spm_type is None:
        return None

    if spm_type == "light":
        spm = LightSPM(in_channels=in_channels, **kwargs)
    elif spm_type in ("lite", "full", "density_only"):
        from adatile.sparse.ada_spm import AdaSPMLite, AdaSPMFull, DensityOnlySPM
        variant_map = {
            "lite": AdaSPMLite,
            "full": AdaSPMFull,
            "density_only": DensityOnlySPM,
        }
        spm_cls = variant_map[spm_type]
        spm = spm_cls(**kwargs)
    else:
        raise ValueError(
            f"Unknown spm_type: {spm_type}. "
            f"Choose from: light, lite, full, density_only, None"
        )

    if device is not None:
        spm = spm.to(device)
    return spm


def build_components(
    args,
    device: torch.device,
    num_classes: int = 1,
) -> Tuple[FastSAMHookBackbone, LightDecoder, Optional[nn.Module]]:
    """Build backbone, decoder, SPM from an args namespace.

    This is the convenience wrapper that most experiment scripts use.

    Args:
        args: Namespace with attributes:
            - image_size, unfreeze_layers: Backbone config.
            - use_spm: Whether to build an SPM.
            - spm_type: "light" (default), "lite", "full", "density_only".
            - output_levels: Optional dict for backbone (default: auto-based on spm_type).
        device: Target device.
        num_classes: Number of segmentation classes.

    Returns:
        (backbone, decoder, spm) tuple. spm is None if args.use_spm is False.
    """
    img_sz = None if args.image_size == 0 else args.image_size
    spm_type = getattr(args, "spm_type", "light")
    use_spm = getattr(args, "use_spm", False)

    # Auto-select output levels based on SPM type
    output_levels = getattr(args, "output_levels", None)
    if output_levels is None:
        if use_spm and spm_type in ("lite", "full", "density_only"):
            # AdaSPM variants need 4 FPN levels
            output_levels = "fpn4"
        else:
            output_levels = "default"

    backbone = build_backbone(
        "FastSAM-x.pt",
        out_channels=128,
        unfreeze_layers=getattr(args, "unfreeze_layers", 2),
        image_size=img_sz,
        device=device,
        output_levels=output_levels,
    )
    decoder = build_decoder(128, 64, num_classes, device=device)

    spm = None
    if use_spm:
        if spm_type == "light":
            spm = build_spm(spm_type="light", in_channels=128, device=device)
        else:
            # AdaSPM variants: detect channels from backbone output levels
            # after first forward pass, or pre-configure based on known FastSAM-x channels
            # FastSAM-x (YOLOv8): P3≈64, P4≈128, P5≈256, P8≈128
            # But actual channels depend on the specific layer — lazy-detect via FPN
            ada_kwargs = {}
            if spm_type == "lite":
                ada_kwargs = {"fusion_dim": 128, "hidden_dim": 64,
                              "use_transformer": False, "lightweight": True, "dropout": 0.0}
            elif spm_type == "full":
                ada_kwargs = {"fusion_dim": 256, "hidden_dim": 256,
                              "use_transformer": True, "lightweight": False, "dropout": 0.1}
            elif spm_type == "density_only":
                ada_kwargs = {"fusion_dim": 256, "hidden_dim": 128,
                              "use_transformer": True}
            spm = build_spm(spm_type=spm_type, device=device, **ada_kwargs)

    return backbone, decoder, spm


def collect_params(
    backbone: nn.Module,
    decoder: nn.Module,
    spm: Optional[nn.Module] = None,
    loss_fn: Optional[nn.Module] = None,
) -> list:
    """Collect all trainable parameters.

    Args:
        backbone: Backbone module.
        decoder: Decoder module.
        spm: Optional SPM module.
        loss_fn: Optional loss module (e.g., LearnableBudget parameters).

    Returns:
        List of parameters with requires_grad=True.
    """
    params = [
        p for p in list(backbone.parameters()) + list(decoder.parameters())
        if p.requires_grad
    ]
    if spm is not None:
        params += [p for p in spm.parameters() if p.requires_grad]
    if loss_fn is not None:
        params += list(loss_fn.parameters())
    return params


def save_checkpoint(
    backbone: nn.Module,
    decoder: nn.Module,
    spm: Optional[nn.Module],
    path: Path,
    dice: float,
) -> None:
    """Save model checkpoint.

    Args:
        backbone: Backbone module.
        decoder: Decoder module.
        spm: Optional SPM module.
        path: Save path.
        dice: Best validation Dice score.
    """
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "decoder": decoder.state_dict(),
            "spm": spm.state_dict() if spm is not None else {},
            "dice": dice,
        },
        str(path),
    )
