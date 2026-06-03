"""Benchmark statistics: per-stage measurements and aggregated results.

StageStats captures a single stage measurement.
BenchmarkResult aggregates all stages for one configuration run.
CompareResult holds side-by-side configuration comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageStats:
    """Timing and memory statistics for one pipeline stage."""

    name: str
    time_ms: float = 0.0
    time_std_ms: float = 0.0
    memory_delta_mb: float = 0.0
    memory_peak_mb: float = 0.0
    num_tokens: int = 0
    token_count: int = 0

    @property
    def time_pct(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "time_ms": round(self.time_ms, 3),
            "time_std_ms": round(self.time_std_ms, 3),
            "memory_delta_mb": round(self.memory_delta_mb, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "num_tokens": self.num_tokens,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one configuration.

    Aggregates per-stage timing, total latency, peak memory,
    and real sparsity metrics derived from actual token counts.
    """

    config_name: str
    image_size: tuple = (1024, 1024)
    stages: Dict[str, StageStats] = field(default_factory=dict)
    total_time_ms: float = 0.0
    total_time_std_ms: float = 0.0
    peak_memory_mb: float = 0.0
    total_tokens: int = 0
    skipped_tokens: int = 0
    routed_tokens: int = 0
    num_iterations: int = 1

    @property
    def effective_sparsity(self) -> float:
        """Real sparsity: fraction of tokens genuinely skipped before compute."""
        if self.total_tokens == 0:
            return 0.0
        return self.skipped_tokens / self.total_tokens

    @property
    def throughput_tokens_per_sec(self) -> float:
        """Tokens processed per second (accounting for sparsity)."""
        if self.total_time_ms == 0:
            return 0.0
        return self.routed_tokens / (self.total_time_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config_name,
            "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
            "total_time_ms": round(self.total_time_ms, 3),
            "total_time_std_ms": round(self.total_time_std_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "total_tokens": self.total_tokens,
            "skipped_tokens": self.skipped_tokens,
            "routed_tokens": self.routed_tokens,
            "effective_sparsity": round(self.effective_sparsity, 4),
            "throughput_tok_s": round(self.throughput_tokens_per_sec, 1),
            "iterations": self.num_iterations,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.config_name}",
            f"  Image: {self.image_size[0]}x{self.image_size[1]}  "
            f"Iterations: {self.num_iterations}",
            f"  Total: {self.total_time_ms:.2f} ± {self.total_time_std_ms:.2f} ms",
            f"  Peak memory: {self.peak_memory_mb:.1f} MB",
            f"  Tokens: {self.routed_tokens}/{self.total_tokens} routed "
            f"(sparsity={self.effective_sparsity:.1%})",
            f"  Throughput: {self.throughput_tokens_per_sec:.0f} tok/s",
            "",
        ]
        if self.stages:
            max_name = max(len(name) for name in self.stages)
            lines.append(f"  {'Stage':<{max_name}}  {'Time (ms)':>10}  {'Mem Δ (MB)':>10}  {'Tokens':>8}")
            lines.append(f"  {'-' * max_name}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
            for name, s in self.stages.items():
                lines.append(
                    f"  {name:<{max_name}}  {s.time_ms:>10.2f}  "
                    f"{s.memory_delta_mb:>10.1f}  {s.num_tokens:>8}"
                )
        return "\n".join(lines)


@dataclass
class CompareResult:
    """Side-by-side comparison of multiple benchmark configurations."""

    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dataframe(self):
        """Return pandas DataFrame for tabular display (optional dependency)."""
        try:
            import pandas as pd
        except ImportError:
            return None  # pandas is optional; use to_table() instead

        rows = []
        for r in self.results:
            row = {
                "config": r.config_name,
                "total_ms": round(r.total_time_ms, 2),
                "peak_mem_mb": round(r.peak_memory_mb, 1),
                "sparsity": round(r.effective_sparsity, 3),
                "tokens": r.routed_tokens,
                "tok/s": round(r.throughput_tokens_per_sec, 0),
            }
            for name, s in r.stages.items():
                row[f"{name}_ms"] = round(s.time_ms, 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_table(self) -> str:
        """Return a plain-text comparison table (no dependencies)."""
        lines = []
        header = f"{'Config':<20} {'Total ms':>10} {'Peak MB':>9} {'Sparsity':>9} {'Tok/s':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.results:
            lines.append(
                f"{r.config_name:<20} {r.total_time_ms:>10.2f} "
                f"{r.peak_memory_mb:>9.1f} {r.effective_sparsity:>9.1%} "
                f"{r.throughput_tokens_per_sec:>8.0f}"
            )
        # Stage breakdown
        lines.append("")
        lines.append("Per-stage latency (ms):")
        stage_names = list(self.results[0].stages.keys()) if self.results else []
        stage_header = f"{'Config':<20}"
        for s in stage_names:
            stage_header += f" {s:>10}"
        lines.append(stage_header)
        lines.append("-" * len(stage_header))
        for r in self.results:
            row = f"{r.config_name:<20}"
            for s in stage_names:
                if s in r.stages:
                    row += f" {r.stages[s].time_ms:>10.2f}"
                else:
                    row += f" {'-':>10}"
            lines.append(row)
        return "\n".join(lines)

    def speedup_vs(self, baseline_name: str) -> Dict[str, float]:
        """Compute speedup of each config relative to a baseline."""
        baseline = None
        for r in self.results:
            if r.config_name == baseline_name:
                baseline = r
                break
        if baseline is None:
            return {}
        out = {}
        for r in self.results:
            if r.config_name != baseline_name:
                out[r.config_name] = baseline.total_time_ms / max(r.total_time_ms, 0.001)
        return out
