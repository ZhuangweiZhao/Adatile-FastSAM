"""Training logging utilities: logger setup, progress tracking, metric averaging."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


def setup_logger(
    name: str = "adatile",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    use_rich: bool = True,
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Args:
        name: Logger name.
        log_dir: Directory for log file output.
        level: Logging level.
        use_rich: Use RichHandler for colored console output (requires `rich`).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    if use_rich:
        try:
            from rich.logging import RichHandler
            console = RichHandler(rich_tracebacks=True, markup=True)
            console.setLevel(level)
            logger.addHandler(console)
        except ImportError:
            use_rich = False

    if not use_rich:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console.setFormatter(fmt)
        logger.addHandler(console)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "adatile") -> logging.Logger:
    """Get an existing logger or create a default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class AverageMeter:
    """Tracks running average of scalar values.

    Usage:
        meter = AverageMeter()
        meter.update(3.5)  # add a single value
        meter.update(4.2, n=2)  # add with weight
        print(meter.avg)  # current average
    """

    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Tracks and displays multiple AverageMeters.

    Usage:
        meters = [AverageMeter("loss"), AverageMeter("iou")]
        pm = ProgressMeter(total_steps, meters, prefix="Train")
        pm.display(step)
    """

    def __init__(
        self,
        num_batches: int,
        meters: list[AverageMeter],
        prefix: str = "",
    ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [f"{self.prefix} [{self.batch_fmtstr.format(batch)}]"]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = f"{{:{num_digits}d}}"
        return f"[{fmt}/{fmt.format(num_batches)}]"

    def get_avg(self, name: str) -> float:
        for m in self.meters:
            if m.name == name:
                return m.avg
        return 0.0

    def to_dict(self) -> Dict[str, float]:
        return {m.name: m.avg for m in self.meters}
