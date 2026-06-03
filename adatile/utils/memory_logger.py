"""GPU memory tracker with per-stage logging.

Records allocated/reserved/peak memory at each pipeline stage.
Outputs to log file and optionally TensorBoard.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("adatile.memory")


@dataclass
class MemorySnapshot:
    """Single-point GPU memory state."""
    stage: str
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    step: int = 0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "allocated_mb": round(self.allocated_mb, 1),
            "reserved_mb": round(self.reserved_mb, 1),
            "max_allocated_mb": round(self.max_allocated_mb, 1),
            "max_reserved_mb": round(self.max_reserved_mb, 1),
            "step": self.step,
        }


class MemoryLogger:
    """Per-stage GPU memory tracker.

    Usage:
        mem = MemoryLogger("outputs/logs/memory.log")
        mem.log("backbone")
        # ... backbone forward ...
        mem.log("adaspm")
        # ... Ada-SPM forward ...
        mem.log("decoder")
        mem.save_csv()
    """

    STAGES = [
        "backbone", "adaspm", "tokenizer", "router", "decoder", "loss",
    ]

    def __init__(self, log_path: str = "outputs/logs/memory.log",
                 csv_path: str = "outputs/logs/memory.csv",
                 warn_threshold_mb: float = 5000):
        self.log_path = Path(log_path)
        self.csv_path = Path(csv_path)
        self.warn_threshold_mb = warn_threshold_mb
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._snapshots: List[MemorySnapshot] = []
        self._step = 0
        self._peak_total_mb = 0.0
        self._enabled = torch.cuda.is_available()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_step(self, step: int):
        self._step = step

    def log(self, stage: str) -> MemorySnapshot:
        """Record current GPU memory state for a pipeline stage."""
        if not self._enabled:
            return MemorySnapshot(stage=stage, timestamp=time.time(),
                                  allocated_mb=0, reserved_mb=0,
                                  max_allocated_mb=0, max_reserved_mb=0)

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)

        snap = MemorySnapshot(
            stage=stage,
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_alloc,
            max_reserved_mb=max_resv,
            step=self._step,
        )
        self._snapshots.append(snap)
        self._peak_total_mb = max(self._peak_total_mb, max_alloc)

        # Per-stage line
        logger.info(
            "[Memory] stage=%-12s alloc=%6.0fMB reserved=%6.0fMB peak=%6.0fMB",
            stage, allocated, reserved, max_alloc,
        )

        # Warning if approaching OOM
        if allocated > self.warn_threshold_mb:
            logger.warning(
                "[Memory] ⚠ HIGH USAGE: %s = %.0f MB (threshold %.0f MB)",
                stage, allocated, self.warn_threshold_mb,
            )

        return snap

    def get_delta(self, before: str, after: str) -> float:
        """Memory delta between two logged stages."""
        snap_before = None
        snap_after = None
        for s in reversed(self._snapshots):
            if s.stage == after and snap_after is None:
                snap_after = s
            if s.stage == before and snap_before is None:
                snap_before = s
            if snap_before and snap_after:
                break

        if snap_before and snap_after:
            return snap_after.allocated_mb - snap_before.allocated_mb
        return 0.0

    def get_peak(self) -> float:
        return self._peak_total_mb

    def reset_peak(self):
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()

    def summary(self) -> str:
        """Return a formatted memory summary."""
        lines = [f"{'='*50}", "GPU Memory Summary",
                 f"{'='*50}"]
        for s in self._snapshots[-7:]:  # last pipeline pass
            lines.append(
                f"  {s.stage:>12s}: {s.allocated_mb:6.0f} MB "
                f"(peak={s.max_allocated_mb:6.0f} MB)"
            )
        lines.append(f"  {'─'*40}")
        lines.append(f"  Peak total: {self._peak_total_mb:.0f} MB")
        return "\n".join(lines)

    def save_csv(self):
        """Save all snapshots to CSV."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "stage", "allocated_mb", "reserved_mb",
                "max_allocated_mb", "max_reserved_mb",
            ])
            if write_header:
                writer.writeheader()
            for s in self._snapshots[-20:]:  # last 20 entries
                writer.writerow(s.to_dict())
            self._snapshots = self._snapshots[-20:]  # keep only recent

    def clear(self):
        self._snapshots.clear()
