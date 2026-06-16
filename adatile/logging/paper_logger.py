"""PaperLogger — paper-ready experiment tracking system.

Wraps RunLogger with enhanced logging for paper-grade experiments.
Adds loss decomposition, SPM distribution statistics, gradient norms,
anomaly detection, and automatic visualization.

Usage:
    from adatile.logging import PaperLogger
    plog = PaperLogger("outputs", "stageB", vars(args))
    plog.start(backbone, decoder, spm, loss_fn)
    plog.log_step(step, loss, metrics, imp, feats)
    plog.finish(best_dice=0.85, best_step=250)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from adatile.logging.run_logger import RunLogger
from adatile.logging.anomaly_detector import AnomalyDetector


class PaperLogger:
    """Paper-grade experiment logger wrapping RunLogger.

    Adds:
      - Loss decomposition (seg/spm/budget)
      - SPM distribution statistics
      - Gradient norm tracking
      - Anomaly detection (NaN, collapse, gradient issues)
      - Automatic visualization at finish()
      - experiment_summary.md generation
    """

    def __init__(
        self,
        root: str = "outputs",
        name: str = "experiment",
        args_dict: Optional[Dict] = None,
        log_interval: int = 50,
    ):
        self.runner = RunLogger(root, name, args_dict)
        self.log_interval = log_interval

        # Per-step buffers (accumulate then flush)
        self._step_losses: Dict[str, List[float]] = {}   # step → {loss_seg, loss_spm, ...}
        self._step_spm: Dict[str, List[float]] = {}       # step → {imp_mean, imp_std, ...}
        self._step_grads: Dict[str, List[float]] = {}     # step → {backbone_grad, ...}
        self._step_sparse: Dict[str, List[float]] = {}    # step → {keep_ratio, coverage, ...}
        self._step_timing: List[float] = []               # per-iteration time
        self._step: int = 0

        # Models (set by start())
        self.backbone: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.spm: Optional[nn.Module] = None

        # Anomaly detector
        self.anomaly = AnomalyDetector()

        # Timing
        self._iter_start: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(
        self,
        backbone: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        spm: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Path:
        """Start experiment, initialize all log files."""
        self.backbone = backbone
        self.decoder = decoder
        self.spm = spm

        run_dir = self.runner.start()
        self._setup_sparsity_csv()
        self._setup_grad_csv()
        return run_dir

    def begin_step(self):
        """Mark the start of an iteration (for timing)."""
        self._iter_start = time.perf_counter()

    def log_step(
        self,
        step: int,
        loss: torch.Tensor,
        metrics: Dict[str, float],
        importance: Optional[torch.Tensor] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
    ):
        """Log one training step with full metrics.

        Args:
            step: Global step number.
            loss: Total loss tensor.
            metrics: Dict from UnifiedLoss (seg_iou, seg_dice, imp_mean, coverage,
                     loss_seg, loss_spm, loss_budget, loss_seg_raw).
            importance: [B,1,H,W] SPM importance map (or None).
            val_metrics: Optional validation metrics dict.
            lr: Current learning rate.
        """
        elapsed = (time.perf_counter() - self._iter_start) * 1000  # ms per iter
        self._step = step
        self._step_timing.append(elapsed)

        # ── Build base log row ──────────────────────────────
        row: Dict[str, float] = {
            "loss": round(float(loss.item()), 6),
            "lr": round(lr, 8),
        }

        # Loss decomposition
        for key in ("loss_seg", "loss_seg_raw", "loss_spm", "loss_spm_raw",
                     "loss_budget", "loss_budget_raw", "lambda_used"):
            if key in metrics:
                row[key] = round(float(metrics[key]), 6)
                self._step_losses.setdefault(key, []).append(
                    (step, float(metrics[key]))
                )

        # Validation
        if val_metrics:
            for key in ("val_iou", "val_dice", "val_coverage", "val_imp_mean"):
                if key in val_metrics:
                    row[key] = round(float(val_metrics[key]), 6)

        # SPM statistics
        if importance is not None:
            imp = importance.detach()
            # .item() safe because importance is always small [B,1,H,W]
            imp_mean = float(imp.mean().item())
            imp_std  = float(imp.std().item())
            imp_max  = float(imp.max().item())
            imp_min  = float(imp.min().item())

            # Importance entropy: -Σ p*log(p) normalized
            imp_flat = imp.reshape(-1)
            imp_clamped = imp_flat.clamp(1e-7, 1 - 1e-7)
            imp_entropy = float(
                -(imp_clamped * imp_clamped.log()
                  + (1 - imp_clamped) * (1 - imp_clamped).log()
                ).mean().item()
            )
            # Normalize to [0,1] — max entropy at p=0.5 is ln(2) ≈ 0.693
            imp_entropy_norm = imp_entropy / math.log(2)

            # Positive tile ratio (importance > 0.5)
            positive_ratio = float((imp > 0.5).float().mean().item())

            row.update({
                "imp_mean": round(imp_mean, 6),
                "imp_std":  round(imp_std, 6),
                "imp_max":  round(imp_max, 6),
                "imp_min":  round(imp_min, 6),
                "imp_entropy": round(imp_entropy_norm, 6),
                "positive_tile_ratio": round(positive_ratio, 6),
            })

            # Budget gap
            if "coverage" in metrics:
                row["coverage"] = round(float(metrics["coverage"]), 6)
            if "imp_mean" in metrics:
                row["imp_mean_from_loss"] = round(float(metrics["imp_mean"]), 6)

            # SPM buffer
            for k in ("imp_mean", "imp_std", "imp_entropy", "positive_tile_ratio",
                       "coverage"):
                if k in row:
                    self._step_spm.setdefault(k, []).append((step, row[k]))

        # Sparse metrics (if available)
        sparse_keys = ("keep_ratio", "actual_keep_ratio", "kept_tiles",
                        "dropped_tiles", "sparse_iou", "sparse_dice")
        for key in sparse_keys:
            if metrics.get(key) is not None or (val_metrics and val_metrics.get(key) is not None):
                val = metrics.get(key, val_metrics.get(key, 0) if val_metrics else 0)
                row[key] = round(float(val), 6)
                self._step_sparse.setdefault(key, []).append((step, float(val)))

        # Timing
        row["iteration_time_ms"] = round(elapsed, 2)

        # ── Write to RunLogger CSV ───────────────────────────
        self.runner.log(step, **row)

        # ── Write sparse CSV ─────────────────────────────────
        if importance is not None or any(
            metrics.get(k) is not None for k in sparse_keys
        ):
            self._write_sparsity_csv(step, row)

        # ── Gradient norms ───────────────────────────────────
        grad_row = self._compute_grad_norms()
        if grad_row:
            self._write_grad_csv(step, grad_row)

        # ── Anomaly detection ────────────────────────────────
        self.anomaly.check(
            step=step,
            loss=float(loss.item()),
            metrics=metrics,
            importance=importance,
            grad_norms=grad_row,
        )

    def log_validation(self, step: int, val_metrics: Dict[str, float]):
        """Log validation results (called separately from log_step if needed)."""
        if not val_metrics:
            return
        row = {f"val_{k}": round(float(v), 6) for k, v in val_metrics.items()}
        self.runner.log(step, **row)

    def finish(
        self,
        best_dice: float = 0.0,
        best_step: int = 0,
        flops_saved: Optional[float] = None,
        n_params: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Finish experiment: visualize, generate summary, close logs."""
        # Save best
        self.runner.log_best(
            best_dice=best_dice,
            best_step=best_step,
        )

        # Auto-visualization
        try:
            from adatile.logging.visualizer import LogVisualizer
            viz = LogVisualizer(self.runner.run_dir)
            viz.generate_all(
                metrics_csv=self.runner.run_dir / "metrics.csv",
                sparsity_csv=self.runner.run_dir / "sparsity.csv",
            )
        except Exception as e:
            print(f"[PaperLogger] Visualization skipped: {e}")

        # Experiment summary
        try:
            self._write_summary_md(
                best_dice=best_dice,
                best_step=best_step,
                flops_saved=flops_saved,
                n_params=n_params,
                checkpoint_path=checkpoint_path,
            )
        except Exception as e:
            print(f"[PaperLogger] Summary generation skipped: {e}")

        # Anomaly report
        try:
            self.anomaly.write_report(self.runner.run_dir)
        except Exception:
            pass

        self.runner.finish()

    # ── Private: gradient norms ───────────────────────────────

    def _compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for all modules."""
        norms = {}
        for name, mod in [("backbone", self.backbone),
                          ("decoder", self.decoder),
                          ("spm", self.spm)]:
            if mod is None:
                continue
            total_norm = 0.0
            total_params = 0
            updated_params = 0
            for p in mod.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    total_params += p.numel()
                    if p.grad.abs().sum() > 1e-8:
                        updated_params += p.numel()
            if total_params > 0:
                norms[f"{name}_grad_norm"] = round(math.sqrt(total_norm), 6)
                norms[f"{name}_param_update_ratio"] = round(
                    updated_params / max(total_params, 1), 6
                )
        return norms

    # ── Private: CSV setup ────────────────────────────────────

    def _setup_sparsity_csv(self):
        path = self.runner.run_dir / "sparsity.csv"
        with open(path, "w", newline="") as f:
            f.write(
                "step,imp_mean,imp_std,imp_entropy,positive_tile_ratio,"
                "coverage,keep_ratio,kept_tiles,dropped_tiles,"
                "loss_spm,loss_budget,loss_seg\n"
            )

    def _write_sparsity_csv(self, step: int, row: Dict[str, float]):
        path = self.runner.run_dir / "sparsity.csv"
        fields = [
            str(step),
            str(row.get("imp_mean", "")),
            str(row.get("imp_std", "")),
            str(row.get("imp_entropy", "")),
            str(row.get("positive_tile_ratio", "")),
            str(row.get("coverage", "")),
            str(row.get("keep_ratio", "")),
            str(row.get("kept_tiles", "")),
            str(row.get("dropped_tiles", "")),
            str(row.get("loss_spm", "")),
            str(row.get("loss_budget", "")),
            str(row.get("loss_seg", "")),
        ]
        with open(path, "a", newline="") as f:
            f.write(",".join(fields) + "\n")

    def _setup_grad_csv(self):
        path = self.runner.run_dir / "gradients.csv"
        with open(path, "w", newline="") as f:
            f.write("step,backbone_grad_norm,decoder_grad_norm,spm_grad_norm,"
                     "backbone_param_update_ratio,decoder_param_update_ratio,"
                     "spm_param_update_ratio\n")

    def _write_grad_csv(self, step: int, norms: Dict[str, float]):
        path = self.runner.run_dir / "gradients.csv"
        fields = [str(step)] + [
            str(norms.get(k, ""))
            for k in ("backbone_grad_norm", "decoder_grad_norm", "spm_grad_norm",
                       "backbone_param_update_ratio", "decoder_param_update_ratio",
                       "spm_param_update_ratio")
        ]
        with open(path, "a", newline="") as f:
            f.write(",".join(fields) + "\n")

    # ── Private: summary markdown ─────────────────────────────

    def _write_summary_md(
        self,
        best_dice: float = 0.0,
        best_step: int = 0,
        flops_saved: Optional[float] = None,
        n_params: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Generate experiment_summary.md — copy-paste ready for paper."""
        path = self.runner.run_dir / "experiment_summary.md"
        best = self.runner._best

        lines = [
            f"# Experiment Summary: {self.runner.run_dir.name}",
            "",
            f"**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Best Results",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Best Dice | {best_dice:.4f} |",
            f"| Best Step | {best_step} |",
        ]

        for k, v in best.items():
            lines.append(f"| {k} | {v} |")

        if n_params:
            lines.append(f"| Params | {n_params:,} |")
        if flops_saved is not None:
            lines.append(f"| FLOPs Saved | {flops_saved:.1%} |")
        if checkpoint_path:
            lines.append(f"| Checkpoint | `{checkpoint_path}` |")

        lines += [
            "",
            "## Loss Components (Final)",
            "",
            "| Loss | Value |",
            "|------|-------|",
        ]
        for loss_name in ("loss_seg", "loss_spm", "loss_budget", "loss"):
            if loss_name in best:
                lines.append(f"| {loss_name} | {best[loss_name]} |")

        # Anomaly summary
        lines += [
            "",
            "## Training Health",
            "",
            f"- NaN events: {len(self.anomaly.nan_events)}",
            f"- Gradient spikes: {len(self.anomaly.grad_spike_events)}",
            f"- Importance collapse warnings: {len(self.anomaly.collapse_events)}",
            f"- Early stop reason: {self.anomaly.stop_reason or 'N/A'}",
            "",
            "## Generated Figures",
            "",
            "- `loss_curve.png` — Loss decomposition over steps",
            "- `dice_curve.png` — Dice/IoU over steps",
            "- `sparsity_curve.png` — SPM statistics over steps",
            "- `importance_distribution.png` — Importance map distribution",
            "- `budget_curve.png` — Budget gap over steps",
            "",
            "## Paper-Ready Copy",
            "",
            "```",
            f"Dice = {best_dice:.4f}",
        ]
        if flops_saved is not None:
            lines.append(f"FLOPs saved = {flops_saved:.1%}")
        if n_params:
            lines.append(f"Params = {n_params:,}")

        lines += ["```"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
