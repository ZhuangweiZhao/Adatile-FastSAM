"""
FPS 测量 | Frames Per Second measurement.
===========================================

使用 torch.cuda.Event 进行高精度 GPU 计时（如果有 CUDA）。
Uses torch.cuda.Event for high-precision GPU timing (if CUDA available).

支持预热运行和多次平均。
Supports warm-up runs and multi-run averaging.
"""

from __future__ import annotations

import time

import torch


class FPSMeter:
    """
    FPS 测量器 | FPS (Frames Per Second) meter.

    使用 with 语句计时每次推理，排除预热运行。
    Use with-statement to time each inference, excluding warmup runs.

    支持 GPU 和 CPU 两种计时模式。
    Supports both GPU and CPU timing modes.

    Usage:
        >>> meter = FPSMeter(warmup=10, num_runs=100)
        >>> for batch in dataloader:
        ...     with meter:
        ...         output = model(batch)
        >>> fps = meter.compute()
        >>> logger.log_metric("fps/inference", fps)
    """

    def __init__(self, warmup: int = 10, num_runs: int = 100) -> None:
        """
        初始化 FPS 测量器 | Initialize FPS meter.

        Args:
            warmup:   预热运行次数（不计入统计）| Number of warmup runs (excluded from stats).
            num_runs: 最多保留的计时次数 | Maximum number of timings to retain.
        """
        self._warmup = warmup
        self._num_runs = num_runs
        self._times: list[float] = []      # 已记录的耗时（秒）| Recorded durations (seconds)
        self._call_count: int = 0           # 总调用次数 | Total call count
        self._use_cuda: bool = torch.cuda.is_available()

        # CUDA 事件（高精度 GPU 计时）| CUDA events (high-precision GPU timing)
        self._start_event = torch.cuda.Event(enable_timing=True) if self._use_cuda else None
        self._end_event = torch.cuda.Event(enable_timing=True) if self._use_cuda else None

        # CPU 计时起点 | CPU timing start
        self._start_time: float | None = None

    def __enter__(self) -> "FPSMeter":
        """开始计时 | Start timing."""
        if self._use_cuda and self._start_event is not None:
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """结束计时并记录 | End timing and record."""
        duration: float

        if self._use_cuda and self._end_event is not None and self._start_event is not None:
            self._end_event.record()
            # 等待 GPU 操作完成 | Wait for GPU ops to complete
            torch.cuda.synchronize()
            duration = self._start_event.elapsed_time(self._end_event) / 1000.0  # ms → s
        else:
            # CPU 计时 | CPU timing
            assert self._start_time is not None
            duration = time.perf_counter() - self._start_time
            self._start_time = None

        self._call_count += 1

        # 跳过预热运行 | Skip warmup runs
        if self._call_count <= self._warmup:
            return

        # 记录有效计时 | Record valid timing
        self._times.append(duration)

        # 保持窗口大小 | Maintain window size
        if len(self._times) > self._num_runs:
            self._times.pop(0)

    def compute(self) -> float:
        """
        计算平均 FPS | Compute average FPS.

        FPS = 1 / 平均耗时(秒) | average duration (seconds).

        Returns:
            float: FPS 值。如果没有有效计时，返回 -1.0。
                   FPS value. -1.0 if no valid timings.
        """
        if not self._times:
            return -1.0

        avg_time = sum(self._times) / len(self._times)
        if avg_time <= 0:
            return -1.0

        return 1.0 / avg_time

    def reset(self) -> None:
        """重置所有计时 | Reset all timings."""
        self._times.clear()
        self._call_count = 0

    @property
    def num_timings(self) -> int:
        """有效计时次数 | Number of valid timings recorded."""
        return len(self._times)
