"""ExperimentHook — non-invasive Trainer hook for experiment logging.

Integrates ExperimentLogger into the Trainer lifecycle without
modifying Trainer internals. Handles all logging, memory tracking,
exception capture, TensorBoard, and checkpoint management.

Usage:
    logger = ExperimentLogger("outputs")
    logger.start(cfg)
    trainer.register_hook(ExperimentHook(logger, log_interval=50))
"""

from __future__ import annotations

import atexit
import os
import signal
import traceback
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from adatile.engine.hooks import HookBase
from adatile.logging.experiment_logger import ExperimentLogger


class ExperimentHook(HookBase):
    """Hook that wires ExperimentLogger into the Trainer lifecycle.

    Non-invasive — all logging logic lives here, not in Trainer.
    Handles: config save, CSV metrics, GPU memory tracking,
    TensorBoard logging, exception capture, graceful shutdown.

    Args:
        exp_logger: ExperimentLogger instance (call exp_logger.start(cfg) before training).
        log_interval: Log metrics/memory every N steps.
        log_memory: Enable GPU memory tracking.
        log_tensorboard: Enable TensorBoard logging.
        extra_system_info: Optional dict of extra key-value pairs for system_info.txt.
    """

    def __init__(
        self,
        exp_logger: ExperimentLogger,
        log_interval: int = 50,
        log_memory: bool = True,
        log_tensorboard: bool = True,
        extra_system_info: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._el = exp_logger
        self._interval = log_interval
        self._log_memory = log_memory and torch.cuda.is_available()
        self._log_tb = log_tensorboard
        self._extra_info = extra_system_info or {}
        self._tb: Optional[SummaryWriter] = None
        self._best_loss: float = float("inf")
        self._best_step: int = 0
        self._best_epoch: int = 0
        self._started: bool = False

        # Register signal handlers for graceful crash logging
        self._register_crash_handlers()

    # ── Lifecycle hooks ──────────────────────────────────────────────

    def before_train(self):
        """Initialize experiment directory and TensorBoard."""
        if self._el.run_dir is None:
            # Start wasn't called externally — auto-start
            cfg = getattr(self.trainer, "cfg", None)
            if cfg is not None:
                self._el.start(cfg, self._extra_info)
                # Also save config in checkpoint dir for backward compat
                cfg.train.checkpoint_dir = str(self._el.checkpoint_dir)
                cfg.output_dir = str(self._el.run_dir)

        if self._log_tb:
            self._tb = SummaryWriter(str(self._el.tensorboard_dir))

        self._started = True
        self._el.logger.info("Training started. Logs: %s", self._el.run_dir)

    def after_train(self):
        """Write summary and close all loggers."""
        self._el.finish(
            best_loss=self._best_loss,
            best_epoch=self._best_epoch,
            best_step=self._best_step,
        )
        if self._tb:
            self._tb.close()
            self._tb = None

    def before_epoch(self):
        pass

    def after_epoch(self):
        self._el.logger.info("Epoch %d complete", self.trainer.current_epoch)

    def before_step(self):
        pass

    def after_step(self):
        step = self.trainer.global_step
        if step % self._interval != 0:
            return

        # ── Log metrics to CSV ──────────────────────────────────
        metrics = self._collect_metrics()
        self._el.log_metrics(step, metrics)

        # ── Log GPU memory ──────────────────────────────────────
        if self._log_memory:
            self._el.log_memory(step)

        # ── TensorBoard ─────────────────────────────────────────
        if self._tb:
            self._write_tensorboard(step, metrics)

        # ── Track best loss ─────────────────────────────────────
        loss = metrics.get("loss", float("inf"))
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_step = step
            self._best_epoch = self.trainer.current_epoch

    # ── Private ─────────────────────────────────────────────────────

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect metrics from Trainer's meters."""
        m = {}
        for meter in self.trainer.meters:
            try:
                m[meter.name] = float(meter.avg)
            except (ValueError, TypeError, AttributeError):
                m[meter.name] = 0.0
        return m

    def _write_tensorboard(self, step: int, metrics: Dict[str, float]):
        """Write metrics to TensorBoard."""
        if self._tb is None:
            return
        for k, v in metrics.items():
            self._tb.add_scalar(f"train/{k}", v, step)

        # Additional router/tile stats from pipeline aux
        # (These would come from diagnostics — placeholder for hook extension)
        try:
            self._tb.add_scalar("system/lr", metrics.get("lr", 0), step)
        except Exception:
            pass

    # ── Crash handlers ──────────────────────────────────────────────

    def _register_crash_handlers(self):
        """Register signal handlers to log exceptions on crash."""

        def _crash_handler(signum, frame):
            msg = f"Received signal {signum} ({signal.Signals(signum).name})"
            try:
                self._el.logger.error(msg)
                # Save partial summary
                self._el.finish(best_loss=self._best_loss)
            except Exception:
                print(msg)
            # Re-raise original signal
            signal.signal(signum, signal.SIG_DFL)
            if hasattr(os, "kill"):
                import os as _os
                _os.kill(_os.getpid(), signum)

        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, _crash_handler)
            except (ValueError, OSError):
                pass  # Not available on this platform

    def log_exception(self, exc: Exception):
        """Log an exception with full traceback."""
        self._el.log_exception(exc)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def best_step(self) -> int:
        return self._best_step
