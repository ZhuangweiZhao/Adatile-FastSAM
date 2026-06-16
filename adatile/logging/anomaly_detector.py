"""AnomalyDetector — automatic training anomaly detection and logging.

Detects:
  - NaN in loss / importance / gradients
  - Gradient explosion (>100× running average)
  - Gradient vanishing (<0.001× running average after warmup)
  - Importance collapse (std → 0)
  - Budget collapse (imp_mean → 0 or imp_mean → 1)
  - Early stop reason recording

Usage:
    detector = AnomalyDetector()
    detector.check(step, loss, metrics, importance, grad_norms)
    detector.write_report(run_dir)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AnomalyEvent:
    step: int
    event_type: str  # "nan", "grad_spike", "grad_vanish", "imp_collapse", "budget_collapse"
    detail: str
    values: Dict[str, float] = field(default_factory=dict)


class AnomalyDetector:
    """Detects and records training anomalies."""

    def __init__(
        self,
        warmup_steps: int = 50,
        grad_spike_factor: float = 100.0,
        grad_vanish_factor: float = 0.001,
        collapse_std_threshold: float = 0.01,
        collapse_mean_low: float = 0.005,
        collapse_mean_high: float = 0.995,
    ):
        self.warmup_steps = warmup_steps
        self.grad_spike_factor = grad_spike_factor
        self.grad_vanish_factor = grad_vanish_factor
        self.collapse_std_threshold = collapse_std_threshold
        self.collapse_mean_low = collapse_mean_low
        self.collapse_mean_high = collapse_mean_high

        # Events
        self.nan_events: List[AnomalyEvent] = []
        self.grad_spike_events: List[AnomalyEvent] = []
        self.grad_vanish_events: List[AnomalyEvent] = []
        self.collapse_events: List[AnomalyEvent] = []
        self.stop_reason: str = ""

        # Running statistics for gradient baseline
        self._grad_history: Dict[str, deque] = {}
        self._step_count: int = 0

    def check(
        self,
        step: int,
        loss: float,
        metrics: Dict[str, float],
        importance: Optional[torch.Tensor] = None,
        grad_norms: Optional[Dict[str, float]] = None,
    ):
        """Run all anomaly checks for one training step.

        Args:
            step: Global step number.
            loss: Total loss value.
            metrics: Dict from UnifiedLoss.
            importance: [B,1,H,W] SPM importance map.
            grad_norms: Dict with *_grad_norm keys from PaperLogger.
        """
        self._step_count = step

        # ── NaN detection ──────────────────────────────────
        if math.isnan(loss) or math.isinf(loss):
            self.nan_events.append(AnomalyEvent(
                step=step,
                event_type="nan_loss",
                detail=f"Loss is {'NaN' if math.isnan(loss) else 'Inf'} at step {step}",
                values={"loss": loss, **metrics},
            ))

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    self.nan_events.append(AnomalyEvent(
                        step=step,
                        event_type="nan_metric",
                        detail=f"Metric '{k}' is {'NaN' if math.isnan(v) else 'Inf'}",
                        values={"metric": k, "value": float(v)},
                    ))

        # Importance NaN
        if importance is not None:
            if torch.isnan(importance).any():
                self.nan_events.append(AnomalyEvent(
                    step=step,
                    event_type="nan_importance",
                    detail=f"Importance contains NaN at step {step}",
                    values={},
                ))

        # ── Gradient checks ────────────────────────────────
        if grad_norms and step > self.warmup_steps:
            for key, norm in grad_norms.items():
                if not key.endswith("_grad_norm"):
                    continue
                # Update running history
                if key not in self._grad_history:
                    self._grad_history[key] = deque(maxlen=20)
                self._grad_history[key].append(norm)

                hist = self._grad_history[key]
                if len(hist) < 10:
                    continue
                avg = sum(hist) / len(hist)

                if avg > 1e-8:
                    # Spike
                    if norm > avg * self.grad_spike_factor:
                        self.grad_spike_events.append(AnomalyEvent(
                            step=step,
                            event_type="grad_spike",
                            detail=f"{key}: {norm:.2f} (> {self.grad_spike_factor}x avg {avg:.2f})",
                            values={key: norm, f"{key}_avg": avg},
                        ))
                    # Vanish
                    if norm < avg * self.grad_vanish_factor:
                        self.grad_vanish_events.append(AnomalyEvent(
                            step=step,
                            event_type="grad_vanish",
                            detail=f"{key}: {norm:.6f} (< {self.grad_vanish_factor}x avg {avg:.6f})",
                            values={key: norm, f"{key}_avg": avg},
                        ))

        # ── Importance collapse ─────────────────────────────
        if importance is not None and step > self.warmup_steps:
            imp_std = float(importance.std().item())
            imp_mean = float(importance.mean().item())

            if imp_std < self.collapse_std_threshold:
                self.collapse_events.append(AnomalyEvent(
                    step=step,
                    event_type="imp_collapse",
                    detail=f"Importance std={imp_std:.6f} < {self.collapse_std_threshold} "
                            f"(mean={imp_mean:.4f})",
                    values={"imp_std": imp_std, "imp_mean": imp_mean},
                ))

            # Budget collapse: importance all 0 or all 1
            if imp_mean < self.collapse_mean_low:
                self.collapse_events.append(AnomalyEvent(
                    step=step,
                    event_type="budget_collapse_zero",
                    detail=f"Importance mean={imp_mean:.6f} — all tiles dropped",
                    values={"imp_mean": imp_mean},
                ))
            if imp_mean > self.collapse_mean_high:
                self.collapse_events.append(AnomalyEvent(
                    step=step,
                    event_type="budget_collapse_one",
                    detail=f"Importance mean={imp_mean:.6f} — all tiles kept",
                    values={"imp_mean": imp_mean},
                ))

        # ── NaN in gradients ────────────────────────────────
        if grad_norms:
            for key, norm in grad_norms.items():
                if key.endswith("_grad_norm") and (math.isnan(norm) or math.isinf(norm)):
                    self.nan_events.append(AnomalyEvent(
                        step=step,
                        event_type="nan_gradient",
                        detail=f"{key} is {'NaN' if math.isnan(norm) else 'Inf'}",
                        values={key: norm},
                    ))

    def record_stop_reason(self, reason: str):
        """Record why training stopped."""
        self.stop_reason = reason

    def write_report(self, run_dir: Path):
        """Write anomaly report to anomaly_report.txt."""
        path = run_dir / "anomaly_report.txt"
        lines = [
            "=" * 60,
            "Training Anomaly Report",
            "=" * 60,
            "",
        ]

        # NaN events
        lines.append(f"[NaN Events]  count={len(self.nan_events)}")
        for e in self.nan_events[-10:]:  # last 10
            lines.append(f"  step={e.step:>6d}  {e.event_type}: {e.detail}")
        if len(self.nan_events) > 10:
            lines.append(f"  ... ({len(self.nan_events) - 10} more)")

        # Gradient events
        lines.append(f"\n[Gradient Spikes]  count={len(self.grad_spike_events)}")
        for e in self.grad_spike_events[-10:]:
            lines.append(f"  step={e.step:>6d}  {e.detail}")

        lines.append(f"\n[Gradient Vanish]  count={len(self.grad_vanish_events)}")
        for e in self.grad_vanish_events[-10:]:
            lines.append(f"  step={e.step:>6d}  {e.detail}")

        # Collapse events
        lines.append(f"\n[Importance Collapse]  count={len(self.collapse_events)}")
        for e in self.collapse_events[-10:]:
            lines.append(f"  step={e.step:>6d}  {e.event_type}: {e.detail}")

        # Stop reason
        lines.append(f"\n[Stop Reason]  {self.stop_reason or 'N/A'}")

        # Summary
        lines.append(f"\n[Summary]")
        lines.append(f"  Total steps: {self._step_count}")
        has_issues = any([
            self.nan_events, self.grad_spike_events,
            self.collapse_events,
        ])
        lines.append(f"  Health: {'⚠️ ISSUES DETECTED' if has_issues else '✅ Clean'}")

        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
