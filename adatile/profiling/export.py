"""Export benchmark results as CSV, JSON, Chrome traces, and plots."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


def export_csv(results: List, path: str) -> str:
    """Export list of BenchmarkResult as CSV.

    Args:
        results: List of BenchmarkResult objects.
        path: Output CSV file path.

    Returns:
        Absolute path to saved file.
    """
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "stage", "time_ms", "time_std_ms",
            "memory_delta_mb", "memory_peak_mb", "num_tokens",
            "image_size", "sparsity",
        ])
        for r in results:
            for name, s in r.stages.items():
                writer.writerow([
                    r.config_name, name,
                    f"{s.time_ms:.3f}", f"{s.time_std_ms:.3f}",
                    f"{s.memory_delta_mb:.1f}", f"{s.memory_peak_mb:.1f}",
                    s.num_tokens,
                    f"{r.image_size[0]}x{r.image_size[1]}",
                    f"{r.effective_sparsity:.4f}",
                ])
            # Summary row
            writer.writerow([
                r.config_name, "TOTAL",
                f"{r.total_time_ms:.3f}", f"{r.total_time_std_ms:.3f}",
                "", f"{r.peak_memory_mb:.1f}",
                r.routed_tokens,
                f"{r.image_size[0]}x{r.image_size[1]}",
                f"{r.effective_sparsity:.4f}",
            ])

    return os.path.abspath(path)


def export_json(results: List, path: str) -> str:
    """Export list of BenchmarkResult as structured JSON.

    Args:
        results: List of BenchmarkResult objects.
        path: Output JSON file path.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    data = {
        "benchmark_meta": {
            "num_configs": len(results),
            "total_iterations": sum(r.num_iterations for r in results),
        },
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return os.path.abspath(path)


def export_chrome_trace(prof, path: str) -> str:
    """Export a torch.profiler trace as Chrome trace JSON.

    Open the resulting file in chrome://tracing to visualize the
    per-operator CUDA kernel timeline.

    Args:
        prof: torch.profiler.profile result.
        path: Output .json file path.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prof.export_chrome_trace(path)
    return os.path.abspath(path)


def plot_results(
    results: List,
    save_dir: str = "results/plots",
    prefix: str = "benchmark",
) -> Dict[str, str]:
    """Generate comparison plots from benchmark results.

    Produces:
        - {prefix}_latency_breakdown.png: stacked bar per config per stage
        - {prefix}_memory.png: peak memory per config
        - {prefix}_sparsity.png: effective sparsity bar chart
        - {prefix}_throughput.png: tokens/sec throughput

    Args:
        results: List of BenchmarkResult objects.
        save_dir: Directory to save plots.
        prefix: Filename prefix.

    Returns:
        Dict of plot_name → saved file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    if not results:
        return saved

    configs = [r.config_name for r in results]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    stage_order = ["backbone", "adaspm", "tokenizer", "router", "decoder"]

    # ── Latency breakdown ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w = 0.15
    x = np.arange(len(configs))
    for i, stage in enumerate(stage_order):
        vals = [
            r.stages[stage].time_ms if stage in r.stages else 0.0
            for r in results
        ]
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=stage, color=colors[i])
        for bar in bars:
            if bar.get_height() > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=6,
                )

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Breakdown")
    ax.set_xticks(x + bar_w * 2)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, f"{prefix}_latency_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["latency_breakdown"] = path

    # ── Peak memory ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    mem_vals = [r.peak_memory_mb for r in results]
    bars = ax.bar(configs, mem_vals, color="#e74c3c", alpha=0.8)
    for bar, val in zip(bars, mem_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{val:.0f} MB", ha="center", fontsize=10,
        )
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak GPU Memory Usage")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, f"{prefix}_memory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["memory"] = path

    # ── Effective sparsity ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sparsity_vals = [r.effective_sparsity * 100 for r in results]
    bars = ax.bar(configs, sparsity_vals, color="#2ecc71", alpha=0.8)
    for bar, val in zip(bars, sparsity_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=10,
        )
    ax.set_ylabel("Effective Sparsity (%)")
    ax.set_title("Real Token Sparsity (Empirically Measured)")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, f"{prefix}_sparsity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["sparsity"] = path

    # ── Throughput ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    throughput_vals = [r.throughput_tokens_per_sec for r in results]
    bars = ax.bar(configs, throughput_vals, color="#9b59b6", alpha=0.8)
    for bar, val in zip(bars, throughput_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:.0f}", ha="center", fontsize=9,
        )
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Processing Throughput")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, f"{prefix}_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["throughput"] = path

    return saved
