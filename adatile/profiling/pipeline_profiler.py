"""End-to-end pipeline profiler using torch.profiler and fvcore.

Profiles the full AdaTile-FastSAM pipeline with:
    - torch.profiler.profile() for kernel-level GPU timing and memory
    - fvcore.nn.FlopCountAnalysis for verified FLOPs counting
    - Per-stage breakdown via prof.step() markers
    - Export to CSV, JSON, and Chrome trace

Replaces the previous manual CUDA event timing with PyTorch's
official profiling API for CVPR-grade reproducibility.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from adatile.profiling.stats import StageStats, BenchmarkResult, CompareResult


class PipelineProfiler:
    """Profiles AdaTile-FastSAM using torch.profiler + fvcore.

    Usage:
        profiler = PipelineProfiler(model)
        result = profiler.profile(image, warmup=3, iterations=10)
        print(result.summary())

        # With FLOPs analysis
        flops_result = profiler.count_flops(image)

        # Chrome trace
        profiler.profile_with_trace(image)
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = True,
    ):
        self.model = model
        self._use_cuda = use_cuda and torch.cuda.is_available()

        if hasattr(model, "pipeline"):
            self._pipe = model.pipeline
        else:
            self._pipe = model

        self._stage_names = [
            "backbone", "adaspm", "tokenizer", "router", "decoder",
        ]
        self._last_trace = None

    # ── Primary profiling with torch.profiler ──────────────────────────

    def profile(
        self,
        image: torch.Tensor,
        warmup: int = 3,
        iterations: int = 10,
        use_torch_profiler: bool = True,
    ) -> BenchmarkResult:
        """Profile the full pipeline.

        Uses torch.profiler for GPU kernel timing. Falls back to
        CPU wall-clock timing if CUDA is unavailable.

        Args:
            image: [1, 3, H, W] input tensor.
            warmup: Number of warmup forward passes.
            iterations: Number of profiling iterations.
            use_torch_profiler: Whether to use torch.profiler (recommended).

        Returns:
            BenchmarkResult with per-stage and aggregate statistics.
        """
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                self._pipe(image)

        if use_torch_profiler and self._use_cuda:
            return self._profile_with_profiler(image, iterations)
        else:
            return self._profile_with_cpu_timing(image, iterations)

    def _profile_with_profiler(
        self, image: torch.Tensor, iterations: int,
    ) -> BenchmarkResult:
        """Profile using torch.profiler — primary path."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for i in range(iterations):
                with torch.no_grad():
                    self._pipe(image)
                prof.step()

        self._last_trace = prof

        # Extract per-stage timing from profiler key averages
        key_avgs = prof.key_averages()
        total_cuda_time_us = sum(
            e.cuda_time for e in key_avgs if hasattr(e, "cuda_time")
        )
        total_time_ms = total_cuda_time_us / 1000.0 / iterations

        # Map kernel names to stages
        stage_times = self._map_kernels_to_stages(key_avgs, iterations)

        # Memory
        peak_mb = 0.0
        if self._use_cuda:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return BenchmarkResult(
            config_name="profiled",
            image_size=(image.shape[-2], image.shape[-1]),
            stages=stage_times,
            total_time_ms=total_time_ms,
            total_time_std_ms=0.0,
            peak_memory_mb=peak_mb,
            total_tokens=0,  # filled after routing
            skipped_tokens=0,
            routed_tokens=0,
            num_iterations=iterations,
        )

    def _profile_with_cpu_timing(
        self, image: torch.Tensor, iterations: int,
    ) -> BenchmarkResult:
        """Fallback: CPU wall-clock timing when CUDA unavailable."""
        total_times = []
        per_stage_accum: Dict[str, List[float]] = {
            name: [] for name in self._stage_names
        }

        for _ in range(iterations):
            t_start = time.perf_counter()

            # Stage 1: Backbone
            t0 = time.perf_counter()
            features = self._pipe.backbone(image)
            per_stage_accum["backbone"].append((time.perf_counter() - t0) * 1000)

            # Stage 2: Ada-SPM
            t0 = time.perf_counter()
            spm_output = self._pipe.sparse_predictor(features)
            per_stage_accum["adaspm"].append((time.perf_counter() - t0) * 1000)

            # Stage 3: Tokenizer
            t0 = time.perf_counter()
            importance = spm_output.importance if hasattr(spm_output, 'importance') else None
            tile_infos, tile_tokens = self._pipe.tokenizer(image, features, importance)
            per_stage_accum["tokenizer"].append((time.perf_counter() - t0) * 1000)

            # Stage 4: Router
            t0 = time.perf_counter()
            route_decision = self._pipe.router(tile_tokens)
            per_stage_accum["router"].append((time.perf_counter() - t0) * 1000)

            # Stage 5: Decoder
            t0 = time.perf_counter()
            _ = self._pipe.decoder(
                route_decision.routed_tokens,
                tile_infos,
                image_size=image.shape[-2:],
            )
            per_stage_accum["decoder"].append((time.perf_counter() - t0) * 1000)

            total_times.append((time.perf_counter() - t_start) * 1000)

        stages = {}
        for name in self._stage_names:
            times = per_stage_accum[name]
            if times:
                stages[name] = StageStats(
                    name=name,
                    time_ms=sum(times) / len(times),
                    time_std_ms=float(
                        (sum((t - sum(times)/len(times))**2 for t in times) / max(len(times)-1, 1)) ** 0.5
                    ) if len(times) > 1 else 0.0,
                )

        mean_total = sum(total_times) / len(total_times)
        return BenchmarkResult(
            config_name="profiled",
            image_size=(image.shape[-2], image.shape[-1]),
            stages=stages,
            total_time_ms=mean_total,
            total_time_std_ms=float(
                (sum((t - mean_total)**2 for t in total_times) / max(len(total_times)-1, 1)) ** 0.5
            ) if len(total_times) > 1 else 0.0,
            peak_memory_mb=0.0,
            total_tokens=0,
            skipped_tokens=0,
            routed_tokens=0,
            num_iterations=iterations,
        )

    def _map_kernels_to_stages(
        self, key_avgs, iterations: int,
    ) -> Dict[str, StageStats]:
        """Heuristic: map profiler kernel names to pipeline stages.

        This is inherently approximate because kernels can span
        stages. For precise per-stage breakdown, use profile_with_trace()
        and inspect the Chrome trace manually.
        """
        stages: Dict[str, StageStats] = {}
        for name in self._stage_names:
            stages[name] = StageStats(name=name)

        for evt in key_avgs:
            if not hasattr(evt, "key"):
                continue
            key = evt.key.lower()
            cuda_ms = (evt.cuda_time if hasattr(evt, "cuda_time") else 0) / 1000

            if "backbone" in key or "timm" in key or "resnet" in key:
                stages["backbone"].time_ms += cuda_ms
            elif "spm" in key or "density" in key or "granularity" in key or "sparse" in key:
                stages["adaspm"].time_ms += cuda_ms
            elif "token" in key or "tile" in key or "planner" in key:
                stages["tokenizer"].time_ms += cuda_ms
            elif "rout" in key or "dtr" in key:
                stages["router"].time_ms += cuda_ms
            elif "decod" in key or "mask" in key or "proto" in key:
                stages["decoder"].time_ms += cuda_ms

        # Average over iterations
        for name in self._stage_names:
            stages[name].time_ms /= max(iterations, 1)

        return stages

    # ── Chrome trace export ────────────────────────────────────────────

    def profile_with_trace(
        self,
        image: torch.Tensor,
        output_path: str = "trace.json",
        warmup: int = 3,
    ) -> str:
        """Profile and export a Chrome trace.

        Returns:
            Path to the saved trace file.
        """
        for _ in range(warmup):
            with torch.no_grad():
                self._pipe(image)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                self._pipe(image)

        self._last_trace = prof
        prof.export_chrome_trace(output_path)
        return output_path

    # ── FLOPs counting via fvcore ──────────────────────────────────────

    def count_flops(self, image: torch.Tensor) -> Dict[str, Any]:
        """Count FLOPs using fvcore.nn.FlopCountAnalysis.

        Returns:
            Dict with total_flops, per_module_flops, parameter_count.
        """
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
        except ImportError:
            return {
                "error": "fvcore not installed. Install with: pip install fvcore",
                "total_flops": 0,
            }

        flops = FlopCountAnalysis(self._pipe, image)
        total = flops.total()

        return {
            "total_flops": total,
            "total_gflops": round(total / 1e9, 2),
            "by_operator": dict(flops.by_operator()),
            "by_module": {
                name: count
                for name, count in flops.by_module().items()
                if count > 0
            },
        }

    # ── Comparison mode ────────────────────────────────────────────────

    def compare_configs(
        self,
        configs: Dict[str, Dict[str, Any]],
        image: torch.Tensor,
        warmup: int = 3,
        iterations: int = 10,
    ) -> CompareResult:
        """Benchmark multiple configurations on the same input.

        Each config is a dict of parameter overrides applied to the
        model's tokenizer/router attributes.

        Args:
            configs: Dict of {name: {param_path: value}} overrides.
            image: [1, 3, H, W] shared input image.
            warmup: Warmup iterations per config.
            iterations: Profiling iterations per config.

        Returns:
            CompareResult with all BenchmarkResults.
        """
        results = []
        for config_name, overrides in configs.items():
            originals = self._apply_overrides(overrides)
            result = self.profile(image, warmup=warmup, iterations=iterations)
            result.config_name = config_name
            results.append(result)
            self._restore_overrides(overrides, originals)

        return CompareResult(results=results)

    def _apply_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        originals = {}
        for path, value in overrides.items():
            obj = self._pipe
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = parts[-1]
            originals[path] = getattr(obj, attr)
            setattr(obj, attr, value)
        return originals

    def _restore_overrides(
        self, overrides: Dict[str, Any], originals: Dict[str, Any],
    ) -> None:
        for path, _ in overrides.items():
            obj = self._pipe
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = parts[-1]
            if path in originals:
                setattr(obj, attr, originals[path])

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def last_trace(self):
        return self._last_trace
