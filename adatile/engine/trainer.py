"""Core training loop with hook system, mixed precision, distributed support.

The Trainer orchestrates the full training lifecycle:
    - Model forward/backward
    - Optimizer step + gradient clipping
    - Mixed precision (FP16/BF16) via torch.cuda.amp
    - Distributed data parallel
    - Hook callbacks at every lifecycle point
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from adatile.config import Config
from adatile.utils import (
    AverageMeter,
    CheckpointManager,
    ProgressMeter,
    get_logger,
    get_rank,
    get_world_size,
    is_distributed,
    reduce_tensor,
    synchronize,
)
from adatile.engine.hooks import HookBase
from adatile.evaluation import COCOEvaluator, FewShotEvaluator


class Trainer:
    """Generic training loop with hooks, mixed precision, and distributed support.

    Usage:
        trainer = Trainer(cfg, model, train_loader, optimizer, scheduler)
        trainer.register_hook(LoggingHook(log_interval=50))
        trainer.register_hook(CheckpointHook(save_interval=5000))
        trainer.train()

    Architecture inspired by Detectron2's TrainerBase.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Mixed precision
        self.use_amp = cfg.train.mixed_precision in ("fp16", "bf16")
        self.amp_dtype = (
            torch.bfloat16 if cfg.train.mixed_precision == "bf16" else torch.float16
        )
        self.scaler = GradScaler(enabled=(cfg.train.mixed_precision == "fp16"))

        # Distributed
        self.distributed = is_distributed()
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[cfg.train.local_rank],
                find_unused_parameters=False,
            )

        # State
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.max_steps: int = cfg.train.max_steps
        self.max_epochs: int = cfg.train.epochs
        self.gradient_accumulation_steps: int = cfg.train.gradient_accumulation_steps
        self.max_grad_norm: float = cfg.train.max_grad_norm
        self.log_interval: int = cfg.train.log_interval

        # Logging & Checkpointing
        self.logger = get_logger()
        self.checkpoint_manager = CheckpointManager(cfg.train.checkpoint_dir)
        self.meters: List[AverageMeter] = []
        self._init_meters()

        # Hooks
        self._hooks: List[HookBase] = []

        # Resume
        self._resume_from = cfg.train.resume_from

        # Evaluator
        self.evaluator: Optional[Any] = None
        if val_loader is not None:
            self.evaluator = COCOEvaluator(cfg)

    def _init_meters(self) -> None:
        """Initialize metric trackers."""
        self.meters = [
            AverageMeter("loss"),
            AverageMeter("loss_mask"),
            AverageMeter("loss_density"),
            AverageMeter("loss_sparse"),
            AverageMeter("loss_routing"),
            AverageMeter("lr"),
        ]

    @property
    def hooks(self) -> List[HookBase]:
        return self._hooks

    def register_hook(self, hook: HookBase) -> None:
        """Register a training hook."""
        hook.trainer = self
        self._hooks.append(hook)

    def train(self) -> None:
        """Main training loop."""
        self.logger.info(f"Starting training: {self.max_epochs} epochs, {self.max_steps} max steps")
        self.logger.info(f"Device: {self.device}, AMP: {self.use_amp}, Distributed: {self.distributed}")

        # Resume if specified
        if self._resume_from:
            self._resume(self._resume_from)

        self._call_hooks("before_train")

        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                self._call_hooks("before_epoch")
                self._train_epoch()
                self._call_hooks("after_epoch")

                if self.global_step >= self.max_steps:
                    break
        finally:
            self._call_hooks("after_train")
            self.logger.info("Training complete.")

    def _train_epoch(self) -> None:
        """Run one epoch of training."""
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            self._call_hooks("before_step")

            batch = self._to_device(batch)
            loss_dict = self._forward_backward(batch)

            # Accumulate gradient
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    # Detect NaN grads BEFORE clipping (for debugging)
                    nan_params = []
                    for name, p in self.model.named_parameters():
                        if p.grad is not None and torch.isnan(p.grad).any():
                            nan_params.append(name)
                    if nan_params:
                        self.logger.error(
                            "[NaN-GRAD] NaN gradient in %d params: %s...",
                            len(nan_params), ", ".join(nan_params[:5]),
                        )
                    # Clip with lower max_norm (1.0) for extra safety
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), min(self.max_grad_norm, 1.0)
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update meters
            for key, meter in zip(
                ["loss", "loss_mask", "loss_density", "loss_sparse", "loss_routing"],
                self.meters[:5],
            ):
                val = loss_dict.get(key, 0.0)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                meter.update(val)

            self.meters[-1].update(self.optimizer.param_groups[0]["lr"])

            self.global_step += 1
            self._call_hooks("after_step")

            if self.global_step >= self.max_steps:
                break

    def _forward_backward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass + backward pass with AMP."""
        amp_context = autocast(dtype=self.amp_dtype) if self.use_amp else nullcontext()

        with amp_context:
            images = batch.get("images", batch.get("image"))
            # Handle few-shot batch
            support_images = batch.get("support_images")
            support_masks = batch.get("support_masks")

            output, aux = self.model(
                images,
                support_images=support_images,
                support_masks=support_masks,
            )

            # Collect diagnostics from pipeline aux output
            self._collect_diagnostics(output, aux)

            # Compute loss
            if self.loss_fn is not None:
                loss_dict = self.loss_fn(output, batch, aux)
            else:
                # Default loss from auxiliary signals
                planning_loss = aux.get("planning_alignment_loss") if aux else None
                if planning_loss is not None:
                    loss = planning_loss
                else:
                    loss = torch.tensor(0.0, device=self.device)
                # Add output regularization
                if output is not None and output.scores.numel() > 0:
                    loss = loss + output.scores.mean() * 0.001
                loss_dict = {"loss": loss}

        # Backward with scaler
        loss = loss_dict.get("loss", loss_dict.get("loss_mask", 0.0))
        self.scaler.scale(loss).backward()

        # Reduce losses across GPUs
        if self.distributed:
            reduced = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    reduced[k] = reduce_tensor(v.detach())
            return reduced

        return {k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        if self.val_loader is None or self.evaluator is None:
            return {}

        self.model.eval()
        self._call_hooks("before_eval")

        for batch in self.val_loader:
            batch = self._to_device(batch)
            images = batch.get("images", batch.get("image"))
            output, _ = self.model(images)
            self.evaluator.process(output, batch)

        metrics = self.evaluator.evaluate()
        self._call_hooks("after_eval")

        # Synchronize metrics
        if self.distributed:
            synced = {}
            for k, v in metrics.items():
                t = torch.tensor(v, device=self.device)
                synced[k] = reduce_tensor(t).item()
            metrics = synced

        return metrics

    def _collect_diagnostics(self, output, aux: Dict) -> None:
        """Feed pipeline stats into DiagnosticsHook for logging."""
        for hook in self._hooks:
            if hook.__class__.__name__ == "DiagnosticsHook":
                dc = hook.dc
                mem = hook.mem
                oom = hook.oom

                # Tile stats — aux["planner_stats"] is a TilePlan, which wraps PlannerStats.
                # TilePlan.planner_stats is the actual PlannerStats with cells_with_tiles etc.
                tile_plan = aux.get("planner_stats") if aux else None
                if tile_plan is not None:
                    # Navigate: TilePlan → PlannerStats (nested)
                    actual_stats = getattr(tile_plan, 'planner_stats', None)
                    if actual_stats is not None:
                        num_tiles = getattr(actual_stats, 'cells_with_tiles', 0)
                        skip_ratio = getattr(actual_stats, 'skip_ratio', 0.0)
                    else:
                        # Fallback: read from TilePlan directly
                        num_tiles = getattr(tile_plan, 'active_tiles', 0)
                        skip_ratio = getattr(tile_plan, 'skip_ratio', 0.0)
                    # Build size distribution from actual specs
                    specs = getattr(tile_plan, 'specs', [])
                    size_dist: dict = {}
                    for s in specs:
                        sz = getattr(s, 'tile_size', 0)
                        size_dist[sz] = size_dist.get(sz, 0) + 1
                    dc.log_tiles(num_tiles, size_dist, skip_ratio)

                # Token stats
                routed = aux.get("routed_tokens") if aux else None
                skipped = aux.get("skipped_indices") if aux else None
                if routed is not None:
                    n_skip = len(skipped) if skipped is not None else 0
                    n_total = routed.shape[0] + n_skip if routed.dim() >= 1 else 0
                    dc.log_tokens(n_total, routed.shape[0] if routed.dim() >= 1 else 0,
                                  max(num_tiles, 1))

                # Router stats
                routing_weights = aux.get("routing_weights") if aux else None
                if routing_weights is not None and routing_weights.numel() > 0:
                    # Infer level distribution from routing weights (NaN-safe)
                    w = routing_weights.squeeze(-1).float()
                    w = torch.nan_to_num(w, nan=0.5)
                    skip_r = float((w < 0.1).float().mean())
                    full_r = float((w > 0.7).float().mean())
                    mid_r = max(0.0, 1.0 - skip_r - full_r)
                    dc.log_router(
                        {0: skip_r, 1: mid_r * 0.5, 2: mid_r * 0.5, 3: full_r},
                        mean_weight=float(w.mean()),
                    )

                # Decoder stats
                if output is not None:
                    proto_shape = f"[{output.masks.shape[1]},{output.masks.shape[2]}]" if output.masks.dim() >= 3 else "[]"
                    dc.log_decoder(
                        num_instances=output.masks.shape[0] if output.masks.dim() >= 3 else 0,
                        proto_shape=proto_shape,
                        mask_shape=proto_shape,
                        mean_score=float(output.scores.mean()) if output.scores.numel() > 0 else 0,
                    )
                    # OOM guard check
                    if output.masks.dim() >= 3:
                        oom.check_mask_allocation(
                            output.masks.shape[0],
                            output.masks.shape[1],
                            output.masks.shape[2],
                        )

                # Memory snapshot
                mem.log("forward")
                break  # only feed first DiagnosticsHook

    def _resume(self, path: str) -> None:
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming from {path}")
        info = self.checkpoint_manager.load_last(
            self.model, self.optimizer, device=str(self.device)
        )
        self.global_step = info.get("step", 0)
        self.current_epoch = info.get("epoch", 0)
        self.logger.info(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the training device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, list):
                out[k] = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in v
                ]
            else:
                out[k] = v
        return out

    def _call_hooks(self, event: str) -> None:
        """Trigger all registered hooks for an event."""
        for hook in self._hooks:
            getattr(hook, event, lambda: None)()
