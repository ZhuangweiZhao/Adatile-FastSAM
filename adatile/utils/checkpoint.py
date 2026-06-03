"""Checkpoint management: save, load, resume training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    epoch: int,
    save_path: str,
    extra: Optional[Dict[str, Any]] = None,
    keep_last_n: int = 5,
) -> str:
    """Save a training checkpoint.

    Args:
        model: Model to save (state_dict only).
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        step: Global training step.
        epoch: Current epoch.
        save_path: Full path to .pt file.
        extra: Additional metadata (metrics, config, etc.).
        keep_last_n: Keep only the N most recent checkpoints.

    Returns:
        Path where checkpoint was saved.
    """
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

    # Cleanup old checkpoints
    ckpt_dir = save_path.parent
    pattern = save_path.stem.rsplit("_", 1)[0]  # strip step number
    old_ckpts = sorted(ckpt_dir.glob(f"{pattern}_*.pt"))
    for old in old_ckpts[:-keep_last_n]:
        old.unlink()

    return str(save_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint, restoring model/optimizer/scheduler states.

    Args:
        path: Path to .pt checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Target device.
        strict: Whether to require exact key match.

    Returns:
        Dict with step, epoch, and any extra metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return metadata
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        **{k: v for k, v in checkpoint.items()
           if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")},
    }


class CheckpointManager:
    """Manages checkpoint saving/loading with automatic cleanup.

    Usage:
        manager = CheckpointManager("checkpoints/exp1", keep_last_n=5)

        # Save
        manager.save(model, optimizer, scheduler, step=1000, epoch=5, metrics={"iou": 0.72})

        # Load best
        info = manager.load_best(model, optimizer)
    """

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_metric: float = -float("inf")
        self.best_path: Optional[str] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        key_metric: str = "iou",
        tag: str = "checkpoint",
    ) -> str:
        """Save a regular checkpoint."""
        path = self.checkpoint_dir / f"{tag}_step{step:07d}.pt"
        return save_checkpoint(
            model, optimizer, scheduler, step, epoch,
            str(path),
            extra={"metrics": metrics or {}},
            keep_last_n=self.keep_last_n,
        )

    def save_best(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        key_metric: str = "iou",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        mode: str = "max",
    ) -> bool:
        """Save a checkpoint if it achieves a new best metric.

        Args:
            mode: "max" (higher is better) or "min" (lower is better).

        Returns:
            True if this was a new best.
        """
        value = metrics.get(key_metric, 0.0)
        is_best = (
            (mode == "max" and value > self.best_metric) or
            (mode == "min" and value < self.best_metric)
        )

        if is_best:
            self.best_metric = value
            self.best_path = str(self.checkpoint_dir / "best_model.pt")
            save_checkpoint(
                model, optimizer, scheduler, step, epoch,
                self.best_path,
                extra={"metrics": metrics, "best_metric": value},
                keep_last_n=1,
            )
        return is_best

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        if self.best_path is None:
            raise FileNotFoundError("No best checkpoint saved yet.")
        return load_checkpoint(self.best_path, model, optimizer, device=device)

    def load_last(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {self.checkpoint_dir}")
        return load_checkpoint(str(ckpts[-1]), model, optimizer, device=device)
