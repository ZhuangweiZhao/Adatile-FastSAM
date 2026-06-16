"""Experiment Logger — orchestrates all logging subsystems.

Creates timestamped experiment directory with:
    config.yaml, train.log, metrics.csv, memory.log,
    system_info.txt, exception.log, summary.txt
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


class ExperimentLogger:
    """Orchestrates experiment logging.

    Creates an isolated directory for each run and coordinates
    all logging subsystems (CSV, memory, system info, exceptions).

    Usage:
        logger = ExperimentLogger("outputs")
        logger.start(cfg)  # creates dir, saves config, system info
        logger.log_metrics(step=100, loss=0.5, ...)
        logger.log_memory(step=100)
        logger.finish(best_loss=0.1637)
    """

    def __init__(self, root_dir: str = "outputs"):
        self.root_dir = Path(root_dir)
        self.run_dir: Optional[Path] = None
        self._logger: Optional[logging.Logger] = None
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[Any] = None
        self._memory_file: Optional[Any] = None
        self._train_start_time: Optional[float] = None
        self._peak_gpu_mb: float = 0.0
        self._step_count: int = 0
        self._best_loss: float = float("inf")
        self._best_step: int = 0
        self._best_epoch: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self, cfg: Any, extra_info: Optional[Dict[str, str]] = None) -> Path:
        """Initialize experiment directory and all logging subsystems.

        Args:
            cfg: Config object with to_dict() method.
            extra_info: Optional extra key-value pairs for system_info.txt.

        Returns:
            Path to the experiment directory.
        """
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = getattr(cfg, "experiment_name", "adatile")
        self.run_dir = self.root_dir / f"{exp_name}_{timestamp}"
        self._ensure_dirs()

        # ── Save config ────────────────────────────────────────────
        self._save_config(cfg)

        # ── Setup file logger ──────────────────────────────────────
        self._setup_logger()

        # ── Setup CSV ──────────────────────────────────────────────
        self._setup_csv()

        # ── Setup memory log ───────────────────────────────────────
        self._setup_memory_log()

        # ── System info ────────────────────────────────────────────
        self._save_system_info(extra_info)

        # ── Record start time ──────────────────────────────────────
        self._train_start_time = time.time()
        self._peak_gpu_mb = 0.0

        self._logger.info("=" * 60)
        self._logger.info("Experiment started: %s", self.run_dir.name)
        self._logger.info("=" * 60)

        return self.run_dir

    def finish(
        self,
        best_loss: Optional[float] = None,
        best_epoch: int = 0,
        best_step: int = 0,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Write experiment summary and close all log files."""
        if best_loss is not None:
            self._best_loss = best_loss
            self._best_epoch = best_epoch
            self._best_step = best_step

        elapsed = time.time() - (self._train_start_time or time.time())
        self._write_summary(elapsed, extra_metrics)
        self._close_files()
        if self._logger:
            self._logger.info("Experiment finished. Run dir: %s", self.run_dir)

    # ── Metric Logging ────────────────────────────────────────────────

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Write a row of metrics to CSV and log to console.

        Args:
            step: Global training step.
            metrics: Dict of metric_name → float value.
        """
        self._step_count = step
        if self._csv_writer is None:
            return

        row = {"step": step}
        row.update(metrics)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_memory(self, step: int):
        """Record GPU memory snapshot to memory.log."""
        if self._memory_file is None or not torch.cuda.is_available():
            return

        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        self._peak_gpu_mb = max(self._peak_gpu_mb, peak)

        self._memory_file.write(f"{step},{alloc:.1f},{reserved:.1f},{peak:.1f}\n")
        self._memory_file.flush()

    def log_exception(self, exc: Exception):
        """Log exception with full traceback to exception.log."""
        tb_str = traceback.format_exc()

        # Console + train.log
        if self._logger:
            self._logger.error("Exception caught:\n%s", tb_str)

        # exception.log
        exc_path = self.run_dir / "exception.log" if self.run_dir else None
        if exc_path:
            with open(exc_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {type(exc).__name__}: {exc}\n")
                f.write(tb_str)
                f.write("\n" + "-" * 60 + "\n\n")

        # OOM-specific report
        if "out of memory" in str(exc).lower() or "OOM" in str(exc).upper():
            self._save_oom_report()

    # ── Private helpers ────────────────────────────────────────────────

    def _ensure_dirs(self):
        dirs = ["logs", "checkpoints", "tensorboard"]
        for d in dirs:
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)

    def _save_config(self, cfg):
        config_path = self.run_dir / "config.yaml"
        try:
            d = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
            # Convert non-serializable values
            d = json.loads(json.dumps(d, default=str))
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"[ExperimentLogger] Failed to save config: {e}")

    def _setup_logger(self):
        self._logger = logging.getLogger(f"exp.{self.run_dir.name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # File handler — train.log
        fh = logging.FileHandler(self.run_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        ))
        self._logger.addHandler(ch)

        self._logger.propagate = False

    def _setup_csv(self):
        csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["step", "loss", "loss_mask", "loss_density",
                        "loss_sparse", "loss_routing", "lr",
                        "iou", "dice", "importance_mean",
                        "router_skip_pct", "router_full_pct",
                        "token_keep_ratio", "token_reduction",
                        "n_tiles", "n_instances"],
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _setup_memory_log(self):
        mem_path = self.run_dir / "memory.log"
        self._memory_file = open(mem_path, "w", encoding="utf-8")
        self._memory_file.write("step,alloc_mb,reserved_mb,peak_mb\n")

    def _save_system_info(self, extra_info=None):
        info_path = self.run_dir / "system_info.txt"
        lines = []

        # GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                lines.append(f"GPU[{i}]: {name} ({total:.1f} GB)")
        else:
            lines.append("GPU: None (CPU only)")

        # CUDA, PyTorch, Python
        lines.append(f"CUDA: {torch.version.cuda or 'N/A'}")
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"Python: {sys.version.split()[0]}")

        # CPU, RAM
        import platform
        lines.append(f"CPU: {platform.processor() or 'Unknown'}")
        lines.append(f"OS: {platform.system()} {platform.release()}")
        try:
            import psutil
            ram = psutil.virtual_memory()
            lines.append(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
        except ImportError:
            pass

        # Git commit
        try:
            import subprocess
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()[:8]
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            lines.append(f"Git: {branch} @ {commit}")
        except Exception:
            lines.append("Git: N/A")

        # Extra
        if extra_info:
            for k, v in extra_info.items():
                lines.append(f"{k}: {v}")

        with open(info_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _save_oom_report(self):
        oom_path = self.run_dir / "oom_report.txt"
        with open(oom_path, "w", encoding="utf-8") as f:
            f.write(f"OOM Report — {datetime.now().isoformat()}\n")
            f.write(f"Step at crash: {self._step_count}\n\n")
            if torch.cuda.is_available():
                f.write(torch.cuda.memory_summary())
            f.write("\n")
            f.write(f"Peak GPU during run: {self._peak_gpu_mb:.0f} MB\n")

    def _write_summary(self, elapsed_seconds: float, extra_metrics=None):
        summary_path = self.run_dir / "summary.txt"
        h = int(elapsed_seconds // 3600)
        m = int((elapsed_seconds % 3600) // 60)
        s = int(elapsed_seconds % 60)

        lines = [
            f"Experiment: {self.run_dir.name}",
            f"Completed:  {datetime.now().isoformat()}",
            f"Duration:   {h}h{m:02d}m{s:02d}s",
            f"Steps:      {self._step_count}",
            f"Best Loss:  {self._best_loss:.6f}",
            f"Best Epoch: {self._best_epoch}",
            f"Best Step:  {self._best_step}",
            f"Peak GPU:   {self._peak_gpu_mb:.0f} MB",
        ]
        if extra_metrics:
            for k, v in extra_metrics.items():
                lines.append(f"{k}: {v}")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _close_files(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        if self._memory_file:
            self._memory_file.close()
            self._memory_file = None

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints" if self.run_dir else Path("checkpoints")

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard" if self.run_dir else Path("tensorboard")

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger("exp")
        return self._logger
