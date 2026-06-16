"""ExperimentRunner — reusable training loop for all experiment scripts.

Encapsulates the common training pattern found across 5 scripts:
    backbone(img) → decoder(feats) → spm(feats) → loss_fn(logits, gt, imp)
    → backward → clip_grad → opt.step → sch.step → zero_grad

Supports three training modes:
    1. Epoch-based: nested for-epoch/for-batch loops (train_as_fastsam.py)
    2. Episodic: per-epoch support/query split (exp_fewshot.py)
    3. Step-based: flat step loop with iterator recycling (ablation_*.py)

Usage:
    runner = ExperimentRunner(device, max_steps=500, lr=1e-3)
    runner.setup(backbone, decoder, spm, loss_fn, train_loader, val_loader)
    results = runner.run()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.utils.early_stop import EarlyStopping

_log = logging.getLogger(__name__)


class ExperimentRunner:
    """Reusable training loop with optimizer, scheduler, early stopping.

    The runner owns the training loop mechanics but NOT model construction
    or data loading. Callers pass built models and DataLoaders.

    Args:
        device: torch device.
        max_steps: Maximum global steps.
        lr: Learning rate.
        weight_decay: Optimizer weight decay.
        patience: Early stopping patience (epochs/steps).
        mode: Early stopping mode ("max" for Dice/IoU, "min" for loss).
        log_interval: Steps between validation logs.
        clip_grad_norm: Max gradient norm (0 = no clipping).
        use_amp: Enable automatic mixed precision.
    """

    def __init__(
        self,
        device: torch.device,
        max_steps: int = 500,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        mode: str = "max",
        log_interval: int = None,
        clip_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        self.device = device
        self.max_steps = max_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.mode = mode
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp

        # Auto-compute log interval
        self.log_interval = log_interval or max(1, max_steps // 20)

        # Set during setup()
        self.backbone: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.spm: Optional[nn.Module] = None
        self.loss_fn: Optional[nn.Module] = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.stopper: Optional[EarlyStopping] = None

        # State
        self.global_step: int = 0
        self.best_metric: float = 0.0 if mode == "max" else float("inf")
        self.best_step: int = 0
        self.history: List[Dict[str, float]] = []

        # Callbacks
        self._on_step_callbacks: List[Callable] = []
        self._on_eval_callbacks: List[Callable] = []
        self._on_best_callbacks: List[Callable] = []

    # ── Setup ────────────────────────────────────────────────────────

    def setup(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        spm: Optional[nn.Module],
        loss_fn: nn.Module,
        train_loader,
        val_loader,
    ) -> None:
        """Configure models, optimizer, scheduler, early stopping.

        Call this once before run(). Builds the optimizer over all
        trainable parameters from backbone, decoder, spm, and loss_fn.
        """
        self.backbone = backbone
        self.decoder = decoder
        self.spm = spm
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Collect trainable parameters
        params = []
        for mod in [backbone, decoder]:
            if mod is not None:
                params += [p for p in mod.parameters() if p.requires_grad]
        if spm is not None:
            params += [p for p in spm.parameters() if p.requires_grad]
        if loss_fn is not None:
            params += list(loss_fn.parameters())

        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_steps
        )
        self.stopper = EarlyStopping(patience=self.patience, mode=self.mode)

    # ── Callbacks ────────────────────────────────────────────────────

    def on_step(self, fn: Callable) -> Callable:
        """Register a callback called after each training step.

        fn(step, train_metrics) is called.
        """
        self._on_step_callbacks.append(fn)
        return fn

    def on_eval(self, fn: Callable) -> Callable:
        """Register a callback called after each validation.

        fn(step, val_metrics) is called.
        """
        self._on_eval_callbacks.append(fn)
        return fn

    def on_best(self, fn: Callable) -> Callable:
        """Register a callback called when a new best model is found.

        fn(step, best_metric) is called.
        """
        self._on_best_callbacks.append(fn)
        return fn

    # ── Training loop modes ──────────────────────────────────────────

    def run_epoch_based(
        self,
        epochs: int = 500,
        batch_size: int = 4,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Standard epoch-based training (train_as_fastsam.py pattern).

        Args:
            epochs: Maximum number of epochs.
            batch_size: Batch size for DataLoader.
            seed: Random seed for reproducibility.

        Returns:
            Dict with best_val_metric, best_step, steps_completed.
        """
        torch.manual_seed(seed)

        self.decoder.train()
        if self.spm is not None:
            self.spm.train()

        dl = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        for epoch in range(epochs):
            for batch in dl:
                train_metrics = self._train_step(batch)
                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    should_stop = self._eval_and_log()
                    if should_stop:
                        return self._finalize()

                if self.global_step >= self.max_steps:
                    return self._finalize()

        return self._finalize()

    def run_episodic(
        self,
        dataset,
        n_shot: int,
        epochs: int = 500,
        batch_size: int = 1,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Episodic training with per-epoch support/query split (exp_fewshot.py pattern).

        Each epoch: randomly split dataset → support (n_shot) + query (rest).
        Train on query subset only.

        Args:
            dataset: Full training dataset with __len__.
            n_shot: Number of support samples per episode.
            epochs: Maximum number of epochs.
            batch_size: Batch size for query DataLoader.
            seed: Base random seed.

        Returns:
            Dict with best_val_metric, best_step, steps_completed.
        """
        from adatile.evaluation.sparse_eval import split_support_query

        torch.manual_seed(seed)

        self.decoder.train()
        if self.spm is not None:
            self.spm.train()

        for epoch in range(epochs):
            # Per-epoch support/query split
            s_idx, q_idx = split_support_query(dataset, n_shot, seed * 1000 + epoch)
            q_subset = torch.utils.data.Subset(dataset, q_idx)
            dl = torch.utils.data.DataLoader(
                q_subset, batch_size=batch_size, shuffle=True, num_workers=0
            )

            for batch in dl:
                train_metrics = self._train_step(batch)
                self.global_step += 1

                if self.global_step % self.log_interval == 0 or self.global_step == 1:
                    should_stop = self._eval_and_log()
                    if should_stop:
                        return self._finalize()

                if self.global_step >= self.max_steps:
                    return self._finalize()

        return self._finalize()

    def run_step_based(self, seed: int = 0) -> Dict[str, float]:
        """Flat step-based training with iterator recycling (ablation_*.py pattern).

        No concept of epochs — runs exactly max_steps steps, recycling
        the DataLoader iterator.

        Args:
            seed: Random seed.

        Returns:
            Dict with best_val_metric, best_step, steps_completed.
        """
        torch.manual_seed(seed)

        self.decoder.train()
        if self.spm is not None:
            self.spm.train()

        train_iter = iter(self.train_loader)

        for step in range(self.max_steps):
            # Iterator recycling
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            train_metrics = self._train_step(batch)
            self.global_step += 1

        # Final evaluation
        self._eval_and_log()
        return self._finalize()

    def run(self, **kwargs) -> Dict[str, float]:
        """Default training mode: epoch-based with keyword dispatch.

        Pass ``mode="epoch"``, ``mode="episodic"``, or ``mode="step"``.
        Additional kwargs are forwarded to the specific run_* method.
        """
        mode = kwargs.pop("mode", "epoch")
        if mode == "episodic":
            return self.run_episodic(**kwargs)
        elif mode == "step":
            return self.run_step_based(**kwargs)
        else:
            return self.run_epoch_based(**kwargs)

    # ── Core training step ───────────────────────────────────────────

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step: forward → loss → backward → update.

        Args:
            batch: DataLoader batch with "images" and "masks" keys.

        Returns:
            Dict with loss, iou, dice, imp_mean, coverage (train metrics).
        """
        img = batch["images"].to(self.device)
        gt = batch["masks"].to(self.device)

        # Forward
        feats = self.backbone(img)
        lgs = self.decoder(features=feats)
        imp = self.spm(feats) if self.spm is not None else None

        loss, metrics = self.loss_fn(lgs, gt, imp)

        # Backward
        loss.backward()
        if self.clip_grad_norm > 0:
            params = [p for p in self._all_params() if p.requires_grad and p.grad is not None]
            if params:
                nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        # Collect metrics
        train_metrics = {
            "loss": loss.item(),
            "iou": metrics.get("iou", 0),
            "dice": metrics.get("dice", 0),
            "imp_mean": metrics.get("imp_mean", 0),
            "coverage": metrics.get("coverage", 0),
        }
        return train_metrics

    # ── Evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation on the full val_loader.

        Returns:
            Dict with val_iou, val_dice (and imp_mean, coverage if SPM enabled).
        """
        self.decoder.eval()
        if self.spm is not None:
            self.spm.eval()

        ious, dices, covers, imps = [], [], [], []

        for vb in self.val_loader:
            vi = vb["images"].to(self.device)
            vg = vb["masks"].to(self.device)
            feats = self.backbone(vi)
            lgs = self.decoder(features=feats)
            imp = self.spm(feats) if self.spm is not None else None
            _, vm = self.loss_fn(lgs, vg, imp)
            ious.append(vm.get("iou", 0))
            dices.append(vm.get("dice", 0))
            if "coverage" in vm:
                covers.append(vm["coverage"])
            if "imp_mean" in vm:
                imps.append(vm["imp_mean"])

        self.decoder.train()
        if self.spm is not None:
            self.spm.train()

        return {
            "val_iou": float(np.mean(ious)),
            "val_dice": float(np.mean(dices)),
            "val_coverage": float(np.mean(covers)) if covers else 0.0,
            "val_imp_mean": float(np.mean(imps)) if imps else 0.0,
        }

    def validate_single_batch(self) -> Dict[str, float]:
        """Fast single-batch validation (train_as_fastsam.py pattern).

        Returns:
            Dict with val_iou, val_dice.
        """
        self.decoder.eval()
        if self.spm is not None:
            self.spm.eval()

        with torch.no_grad():
            vb = next(iter(self.val_loader))
            vi = vb["images"].to(self.device)
            vg = vb["masks"].to(self.device)
            feats = self.backbone(vi)
            lgs = self.decoder(features=feats)
            imp = self.spm(feats) if self.spm is not None else None
            _, vm = self.loss_fn(lgs, vg, imp)

        self.decoder.train()
        if self.spm is not None:
            self.spm.train()

        return {"val_iou": vm.get("iou", 0), "val_dice": vm.get("dice", 0)}

    # ── Internal helpers ─────────────────────────────────────────────

    def _all_params(self):
        """Yield all trainable parameters from all modules."""
        for mod in [self.backbone, self.decoder, self.spm, self.loss_fn]:
            if mod is not None:
                yield from mod.parameters()

    def _eval_and_log(self) -> bool:
        """Run validation and check early stopping.

        Returns:
            True if early stopping triggered.
        """
        val_metrics = self.evaluate()
        val_metric = val_metrics["val_dice"] if self.mode == "max" else val_metrics.get("val_loss", 0)

        # Check for new best
        is_better = (
            val_metric > self.best_metric if self.mode == "max"
            else val_metric < self.best_metric
        )
        if is_better:
            self.best_metric = val_metric
            self.best_step = self.global_step
            for cb in self._on_best_callbacks:
                cb(self.global_step, self.best_metric)

        # Fire callbacks
        for cb in self._on_eval_callbacks:
            cb(self.global_step, val_metrics)

        return self.stopper.step(val_metric)

    def _finalize(self) -> Dict[str, float]:
        """Return final results dict."""
        return {
            "best_val_metric": self.best_metric,
            "best_step": self.best_step,
            "steps_completed": self.global_step,
        }


class EpisodicRunner(ExperimentRunner):
    """Convenience subclass: always runs episodic training."""

    def run(
        self,
        dataset,
        n_shot: int,
        epochs: int = 500,
        batch_size: int = 1,
        seed: int = 0,
        **kwargs,
    ) -> Dict[str, float]:
        return self.run_episodic(
            dataset=dataset,
            n_shot=n_shot,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )


class StepRunner(ExperimentRunner):
    """Convenience subclass: always runs step-based training."""

    def run(self, seed: int = 0, **kwargs) -> Dict[str, float]:
        return self.run_step_based(seed=seed)
