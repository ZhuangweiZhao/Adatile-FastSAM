#!/usr/bin/env python
"""Training entry point for AdaTile-FastSAM.

Usage:
    # From YAML config
    python tools/train.py --config configs/isaid.yaml

    # From Python config module
    python tools/train.py --config configs.isaid.get_isaid_config

    # Override specific values
    python tools/train.py --config configs/default.py -o train.epochs=100 data.batch_size=4

    # Distributed training
    torchrun --nproc_per_node=4 tools/train.py --config configs/isaid.yaml --distributed
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdaTile-FastSAM Training"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs.default.get_default_config",
        help="Config path (.yaml) or Python module path (module.function).",
    )
    parser.add_argument(
        "--override", "-o",
        type=str,
        nargs="*",
        default=[],
        help="Override config values, e.g. train.epochs=100 data.batch_size=4",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Resume from checkpoint directory.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def load_config(config_spec: str) -> "Config":
    """Load config from YAML file or Python module."""
    from adatile.config import Config

    if config_spec.endswith((".yaml", ".yml")):
        return Config.from_yaml(config_spec)

    if config_spec.endswith(".json"):
        return Config.from_json(config_spec)

    # Try as Python module path
    last_error = None
    try:
        parts = config_spec.rsplit(".", 1)
        if len(parts) == 2:
            module_name, func_name = parts
            module = importlib.import_module(module_name)
            config_fn = getattr(module, func_name)
            return config_fn()
    except (ImportError, AttributeError) as e:
        last_error = e

    raise ValueError(
        f"Cannot load config from: {config_spec}. "
        f"Last error: {last_error}"
    ) from last_error


def apply_overrides(cfg: "Config", overrides: list[str]) -> "Config":
    """Apply dot-separated config overrides.

    Example: "train.epochs=100" → cfg.train.epochs = 100
    """
    for override in overrides:
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the parent object
        obj = cfg
        for key in keys[:-1]:
            obj = getattr(obj, key)

        # Set value with type inference
        attr = keys[-1]
        current = getattr(obj, attr)
        if isinstance(current, bool):
            setattr(obj, attr, value.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(obj, attr, int(value))
        elif isinstance(current, float):
            setattr(obj, attr, float(value))
        elif isinstance(current, list):
            setattr(obj, attr, eval(value))
        else:
            setattr(obj, attr, value)

    return cfg


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # Resume
    if args.resume:
        cfg.train.resume_from = args.resume

    # Distributed
    if args.distributed:
        from adatile.utils import init_distributed
        cfg.train.distributed = True
        cfg.train.local_rank = args.local_rank
        init_distributed(local_rank=args.local_rank)

    # Build model
    from adatile.modeling import build_adatile_fastsam
    model = build_adatile_fastsam(cfg)

    # Build dataset
    from adatile.datasets import CocoDataset
    from adatile.datasets.loaders import DynamicTileDataLoader

    train_dataset = CocoDataset(
        root_dir=cfg.data.root_dir,
        split="train",
    )
    train_loader = DynamicTileDataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    val_dataset = CocoDataset(
        root_dir=cfg.data.root_dir,
        split="val",
    )
    val_loader = DynamicTileDataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    # Build optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.max_steps - cfg.train.warmup_steps,
    )

    # Build loss function — tuned weights to prevent importance collapse:
    # - density_weight=0.5: push importance toward GT object locations
    # - sparsity_weight=0.002: gentle nudge, won't collapse importance
    # - routing_weight=0.1: encourage router to actually skip tokens
    from adatile.segmentation.base import TrainingLoss
    loss_fn = TrainingLoss(
        mask_weight=1.0,
        density_weight=0.5,
        sparsity_weight=0.002,
        routing_weight=0.1,
        match_iou_threshold=0.3,
    )

    # Build trainer
    from adatile.engine import Trainer
    from adatile.engine.hooks import (
        LRSchedulerHook,
        LoggingHook,
        CheckpointHook,
        EvalHook,
        TensorBoardHook,
        DiagnosticsHook,
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    # Register hooks
    trainer.register_hook(LRSchedulerHook())
    trainer.register_hook(LoggingHook(log_interval=cfg.train.log_interval))
    trainer.register_hook(DiagnosticsHook(
        output_dir=cfg.output_dir,
        log_interval=cfg.train.log_interval,
    ))
    trainer.register_hook(CheckpointHook(save_interval=cfg.train.save_interval))
    trainer.register_hook(EvalHook(eval_interval=cfg.train.eval_interval))
    if cfg.train.tensorboard_dir:
        trainer.register_hook(TensorBoardHook(log_dir=cfg.train.tensorboard_dir))

    # Train
    trainer.train()


if __name__ == "__main__":
    import torch
    main()
