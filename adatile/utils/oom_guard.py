"""OOM Guard — predicts OOM before it happens and auto-saves crash reports.

Checks tensor sizes at key pipeline stages and warns if estimated
memory exceeds configurable thresholds. Saves crash reports on OOM.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger("adatile.oom_guard")


@dataclass
class CrashReport:
    step: int
    epoch: int
    gpu_memory_mb: float
    tiles: int
    tokens: int
    instances: int
    error: str
    stage: str = ""
    image_shape: str = ""
    timestamp: str = ""

    def save(self, path: str = "outputs/debug/crash_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.error(f"Crash report saved to {path}")


class OOMGuard:
    """Predictive OOM detection at key pipeline stages.

    Checks tensor allocations BEFORE they happen, not after.
    Logs warnings when estimated memory exceeds thresholds.
    """

    def __init__(
        self,
        warn_threshold_mb: float = 2000,
        critical_threshold_mb: float = 4000,
        total_gpu_mb: Optional[float] = None,
    ):
        self.warn_threshold_mb = warn_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        if total_gpu_mb is None and torch.cuda.is_available():
            total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        self.total_gpu_mb = total_gpu_mb or 6000

        self._last_state = {
            "step": 0, "epoch": 0, "tiles": 0, "tokens": 0, "instances": 0,
        }

    def update_state(self, **kwargs):
        self._last_state.update(kwargs)

    def check_tensor(self, shape: Tuple[int, ...], name: str = "",
                     dtype_bytes: int = 4) -> bool:
        """Check if a tensor allocation is safe.

        Args:
            shape: Tensor shape e.g. (N, H, W)
            name: Human-readable name for logging
            dtype_bytes: Bytes per element (4=float32, 2=float16)

        Returns:
            True if safe, False if risky.
        """
        elements = 1
        for dim in shape:
            elements *= dim
        estimated_mb = elements * dtype_bytes / (1024 * 1024)

        current_mb = (torch.cuda.memory_allocated() / (1024 ** 2)
                      if torch.cuda.is_available() else 0)
        projected_mb = current_mb + estimated_mb

        if projected_mb > self.total_gpu_mb * 0.9:
            logger.critical(
                "[OOM] ⛔ CRITICAL: %s shape=%s estimated=%.0fMB "
                "projected=%.0fMB / %.0fMB GPU",
                name, shape, estimated_mb, projected_mb, self.total_gpu_mb,
            )
            return False

        if estimated_mb > self.critical_threshold_mb:
            logger.critical(
                "[OOM] ⛔ CRITICAL: %s shape=%s estimated=%.0fMB "
                "(single tensor exceeds critical threshold %.0fMB)",
                name, shape, estimated_mb, self.critical_threshold_mb,
            )
            return False

        if estimated_mb > self.warn_threshold_mb:
            logger.warning(
                "[OOM] ⚠ WARNING: %s shape=%s estimated=%.0fMB",
                name, shape, estimated_mb,
            )

        return True

    def check_mask_allocation(
        self, n_inst: int, h: int, w: int, stage: str = "decoder",
    ) -> bool:
        """Specifically check mask allocation risk.

        Note: masks are CROP-level (h,w ≤ 256), not full-image.
        Full-image allocation would be N_inst × H_img × W_img, which is
        what causes OOM. Our crop-level masks are O(Σ h_i·w_i) instead.
        """
        elements = n_inst * h * w
        estimated_mb = elements * 4 / (1024 * 1024)  # float32

        # Crop-level masks: h,w ≤ 256. Even 1000 instances × 256×256 = 250 MB.
        # Only warn if single-tensor > 500 MB (which means full-image mode).
        if estimated_mb > 500:  # 500 MB — only full-image masks reach this
            logger.critical(
                "[OOM] ⛔ LARGE MASK ALLOCATION! %d instances × %d×%d mask = %.0f MB. "
                "Should be crop-level (≤256×256). If this is crop-level, "
                "max_h=%d is too large — check decoder padding logic.",
                n_inst, h, w, estimated_mb, max(h, w),
            )
            if estimated_mb > self.critical_threshold_mb:
                return False
        elif estimated_mb > 100:  # >100 MB is unusual for crop-level
            logger.info(
                "[OOM] ✓ mask allocation: %d instances × %d×%d = %.0f MB (crop-level, OK)",
                n_inst, h, w, estimated_mb,
            )
        return True

    def save_crash_report(self, error: str, stage: str = "",
                          image_shape: str = ""):
        """Save crash report on training failure."""
        report = CrashReport(
            step=self._last_state["step"],
            epoch=self._last_state["epoch"],
            gpu_memory_mb=(torch.cuda.max_memory_allocated() / (1024 ** 2)
                           if torch.cuda.is_available() else 0),
            tiles=self._last_state.get("tiles", 0),
            tokens=self._last_state.get("tokens", 0),
            instances=self._last_state.get("instances", 0),
            error=error,
            stage=stage,
            image_shape=image_shape,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        report.save()

    def wrap_forward(self, fn, stage_name: str, *args, **kwargs):
        """Wrap a forward call with OOM guarding."""
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.save_crash_report(
                    error=f"CUDA OOM in {stage_name}: {e}",
                    stage=stage_name,
                )
            raise
