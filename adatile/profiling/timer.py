"""CUDA event-based wall-clock timing for GPU pipeline stages.

Provides CUDATimer (raw events) and StageTimer (context manager)
for measuring real GPU kernel execution time, not CPU wall time.
"""

from __future__ import annotations

import torch


class CUDATimer:
    """CUDA event-based GPU timer with sub-millisecond precision.

    Uses torch.cuda.Event for accurate GPU-side timing.
    Falls back to CPU wall-clock if CUDA is unavailable.

    Usage:
        timer = CUDATimer()
        timer.start()
        # ... GPU work ...
        elapsed_ms = timer.stop()
    """

    def __init__(self, use_cuda: bool = True):
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._cpu_start: float = 0.0

    def start(self) -> None:
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = torch.cuda.Event().query if False else __import__("time").time()
            self._cpu_start = __import__("time").time()

    def stop(self) -> float:
        """Stop timer and return elapsed time in milliseconds."""
        if self._use_cuda:
            assert self._end_event is not None
            self._end_event.record()
            torch.cuda.synchronize()
            assert self._start_event is not None
            return self._start_event.elapsed_time(self._end_event)
        else:
            return (__import__("time").time() - self._cpu_start) * 1000.0

    @property
    def available(self) -> bool:
        return self._use_cuda


class StageTimer:
    """Named context-manager timer for a pipeline stage.

    Records CUDA elapsed time and optionally takes memory snapshots
    before and after the stage.

    Usage:
        timer = StageTimer("backbone")
        with timer:
            features = model.backbone(image)
        print(f"{timer.name}: {timer.elapsed_ms:.2f} ms")
    """

    def __init__(
        self,
        name: str,
        track_memory: bool = True,
        use_cuda: bool = True,
    ):
        self.name = name
        self._track_memory = track_memory
        self._timer = CUDATimer(use_cuda=use_cuda)
        self.elapsed_ms: float = 0.0
        self.memory_before_mb: float = 0.0
        self.memory_after_mb: float = 0.0
        self.memory_peak_mb: float = 0.0
        self.token_count: int = 0

    def __enter__(self):
        if self._track_memory and self._timer.available:
            torch.cuda.reset_peak_memory_stats()
            self.memory_before_mb = (
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
        self._timer.start()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = self._timer.stop()
        if self._track_memory and self._timer.available:
            self.memory_after_mb = (
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            self.memory_peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

    def set_token_count(self, n: int) -> None:
        self.token_count = n

    def memory_delta_mb(self) -> float:
        return self.memory_after_mb - self.memory_before_mb

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "time_ms": round(self.elapsed_ms, 3),
            "memory_before_mb": round(self.memory_before_mb, 1),
            "memory_after_mb": round(self.memory_after_mb, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "memory_delta_mb": round(self.memory_delta_mb(), 1),
            "token_count": self.token_count,
        }
