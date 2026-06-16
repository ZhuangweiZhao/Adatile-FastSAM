"""LogVisualizer — auto-generate paper-ready plots from experiment CSVs.

Generates at finish():
  - loss_curve.png          Loss decomposition over steps
  - dice_curve.png          Train/Val Dice + IoU over steps
  - sparsity_curve.png      SPM statistics (imp_mean, imp_std, coverage, positive_ratio)
  - importance_distribution.png  Histogram of importance values
  - budget_curve.png        Budget gap over steps

All figures use matplotlib with a clean, publication-ready style.
Dependency: pip install matplotlib (optional — gracefully skipped if missing).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# Publication-ready style
_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
}


class LogVisualizer:
    """Auto-generate publication-ready plots from experiment CSVs.

    Usage:
        viz = LogVisualizer(run_dir)
        viz.generate_all(metrics_csv, sparsity_csv)
    """

    def __init__(self, run_dir: Path):
        if not _HAS_MPL:
            raise RuntimeError("matplotlib not installed — pip install matplotlib")
        self.run_dir = Path(run_dir)
        self.viz_dir = self.run_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        with plt.style.context(_STYLE):
            pass  # ensure style loaded

    def generate_all(
        self,
        metrics_csv: Path,
        sparsity_csv: Optional[Path] = None,
        gradients_csv: Optional[Path] = None,
    ):
        """Generate all standard plots."""
        if metrics_csv.exists():
            self.loss_curve(metrics_csv)
            self.dice_curve(metrics_csv)
        if sparsity_csv and sparsity_csv.exists():
            self.sparsity_curve(sparsity_csv)
            self.importance_distribution(sparsity_csv)
            self.budget_curve(sparsity_csv)
        if gradients_csv and gradients_csv.exists():
            self.gradient_curve(gradients_csv)
        plt.close("all")

    # ── Individual plots ───────────────────────────────────────

    def loss_curve(self, csv_path: Path):
        """Plot loss decomposition: total, seg, spm, budget."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        loss_cols = [c for c in data[0].keys()
                     if c.startswith("loss") and c != "loss_spm_raw" and c != "loss_budget_raw"]
        if not loss_cols:
            # Fallback: just total loss
            loss_cols = ["loss"]

        steps = [int(r["step"]) for r in data]
        for col in loss_cols:
            vals = [float(r.get(col, np.nan)) for r in data]
            # Skip all-NaN columns
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(steps, vals, label=col.replace("loss_", "L_").replace("loss", "L_total"),
                    alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Decomposition")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.savefig(self.viz_dir / "loss_curve.png")
        plt.close(fig)

    def dice_curve(self, csv_path: Path):
        """Plot train/val Dice and IoU."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [int(r["step"]) for r in data]

        for col, label, color in [
            ("dice", "Train Dice", "#2196F3"),
            ("iou", "Train IoU", "#4CAF50"),
            ("val_dice", "Val Dice", "#FF5722"),
            ("val_iou", "Val IoU", "#FF9800"),
        ]:
            vals = [float(r[col]) for r in data if col in r and r[col] != ""]
            if not vals:
                continue
            # Sub-sample if length mismatch
            s = steps[:len(vals)]
            ax.plot(s, vals, label=label, color=color, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("Dice / IoU Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.savefig(self.viz_dir / "dice_curve.png")
        plt.close(fig)

    def sparsity_curve(self, csv_path: Path):
        """Plot SPM statistics: imp_mean, imp_std, coverage, positive_tile_ratio."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        steps = [int(r["step"]) for r in data]

        plots = [
            (0, 0, "imp_mean", "Importance Mean", "blue"),
            (0, 1, "imp_std", "Importance Std", "red"),
            (1, 0, "coverage", "Coverage", "green"),
            (1, 1, "positive_tile_ratio", "Positive Tile Ratio", "purple"),
        ]

        for row, col, key, title, color in plots:
            ax = axes[row][col]
            vals = [float(r[key]) for r in data if key in r and r[key] != ""]
            if vals:
                s = steps[:len(vals)]
                ax.plot(s, vals, color=color, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

        fig.suptitle("SPM Sparsity Statistics", fontsize=14)
        fig.tight_layout()

        fig.savefig(self.viz_dir / "sparsity_curve.png")
        plt.close(fig)

    def importance_distribution(self, csv_path: Path):
        """Plot histogram of importance values."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        # Collect importance statistics across all steps
        imp_means = [float(r["imp_mean"]) for r in data
                     if "imp_mean" in r and r["imp_mean"] != ""]
        imp_stds  = [float(r["imp_std"]) for r in data
                     if "imp_std" in r and r["imp_std"] != ""]

        if imp_means:
            ax.hist(imp_means, bins=30, alpha=0.6, label=f"Mean importance (μ={np.mean(imp_means):.3f})",
                    color="blue")
        if imp_stds:
            ax2 = ax.twinx()
            ax2.hist(imp_stds, bins=30, alpha=0.4, label=f"Std importance (μ={np.mean(imp_stds):.3f})",
                     color="red")
            ax2.set_ylabel("Frequency (std)", color="red")
            ax2.legend(loc="upper right")

        ax.set_xlabel("Importance Value")
        ax.set_ylabel("Frequency (mean)")
        ax.set_title("Importance Distribution Over Training")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.savefig(self.viz_dir / "importance_distribution.png")
        plt.close(fig)

    def budget_curve(self, csv_path: Path):
        """Plot budget gap = |imp_mean - target|."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [int(r["step"]) for r in data]

        # Budget gap (proxy: imp_mean vs loss_budget)
        imp_vals = [float(r["imp_mean"]) for r in data
                    if "imp_mean" in r and r["imp_mean"] != ""]
        if imp_vals:
            s = steps[:len(imp_vals)]
            ax.plot(s, imp_vals, label="imp_mean", color="blue")

        budget_vals = [float(r["loss_budget"]) for r in data
                       if "loss_budget" in r and r["loss_budget"] != ""]
        if budget_vals:
            s = steps[:len(budget_vals)]
            ax2 = ax.twinx()
            ax2.plot(s, budget_vals, label="Budget Loss", color="red", alpha=0.6)
            ax2.set_ylabel("Budget Loss", color="red")

        ax.set_xlabel("Step")
        ax.set_ylabel("Importance Mean")
        ax.set_title("Budget Control")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.savefig(self.viz_dir / "budget_curve.png")
        plt.close(fig)

    def gradient_curve(self, csv_path: Path):
        """Plot gradient norms over time."""
        data = _read_csv(csv_path)
        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [int(r["step"]) for r in data]

        for col, color in [
            ("backbone_grad_norm", "blue"),
            ("decoder_grad_norm", "green"),
            ("spm_grad_norm", "red"),
        ]:
            vals = [float(r[col]) for r in data if col in r and r[col] != ""]
            if vals and max(vals) > 0:
                s = steps[:len(vals)]
                ax.plot(s, vals, label=col.replace("_grad_norm", ""), color=color, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Norms")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        fig.savefig(self.viz_dir / "gradient_curve.png")
        plt.close(fig)


# ── Helpers ─────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[dict]:
    """Read CSV into list of dicts, handling empty/missing files."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
