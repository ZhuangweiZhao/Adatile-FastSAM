"""Real profiling subsystem for AdaTile-FastSAM.

Replaces analytical FLOPs/latency estimates with empirically measured
CUDA timings, peak memory tracking, and torch.profiler traces.

Components:
    - CUDATimer / StageTimer: GPU wall-clock timing
    - StageStats / BenchmarkResult / CompareResult: structured results
    - PipelineProfiler: end-to-end per-stage profiling
    - export_csv / export_json / export_chrome_trace / plot_results
"""

from adatile.profiling.timer import CUDATimer, StageTimer
from adatile.profiling.stats import StageStats, BenchmarkResult, CompareResult
from adatile.profiling.pipeline_profiler import PipelineProfiler
from adatile.profiling.export import (
    export_csv,
    export_json,
    export_chrome_trace,
    plot_results,
)

__all__ = [
    "CUDATimer",
    "StageTimer",
    "StageStats",
    "BenchmarkResult",
    "CompareResult",
    "PipelineProfiler",
    "export_csv",
    "export_json",
    "export_chrome_trace",
    "plot_results",
]
