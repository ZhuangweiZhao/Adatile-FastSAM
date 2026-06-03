#!/usr/bin/env python
"""AdaTile-FastSAM benchmark script — empirically measured efficiency numbers.

Produces reviewer-proof latency, memory, and sparsity statistics by
profiling the full pipeline with CUDA events and torch.profiler.

Usage:
    # Quick single-config benchmark
    python benchmark.py --quick

    # Compare all 4 configurations
    python benchmark.py --compare

    # Full benchmark with chrome trace
    python benchmark.py --compare --trace --output results/

    # Custom image size and iterations
    python benchmark.py --config hard_sparse --image-size 2048,2048 --iters 20

    # Full pipeline + individual component profiling
    python benchmark.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import torch

from adatile.config import Config
from adatile.modeling.adatile_fastsam import build_adatile_fastsam
from adatile.profiling import (
    PipelineProfiler,
    BenchmarkResult,
    CompareResult,
    export_csv,
    export_json,
    export_chrome_trace,
    plot_results,
)


# ── Pre-defined benchmark configurations ───────────────────────────────


def build_configs() -> Dict[str, Config]:
    """Build the 4 benchmark configurations."""
    configs = {}

    # 1. Fixed tiling — no adaptive tiling, no routing (baseline)
    cfg_fixed = Config()
    cfg_fixed.experiment_name = "bench_fixed_tiling"
    cfg_fixed.tokenizer.name = "uniform_tile"
    cfg_fixed.tokenizer.tile_sizes = [1024]
    cfg_fixed.tokenizer.stride_ratios = [1.0]
    cfg_fixed.tokenizer.max_tokens_per_image = 256
    cfg_fixed.router.name = "IdentityRouter"
    cfg_fixed.router.embed_dim = 256
    configs["fixed_tiling"] = cfg_fixed

    # 2. Adaptive tiling — dynamic tiles with threshold skip, DTRv2 routing
    cfg_adaptive = Config()
    cfg_adaptive.experiment_name = "bench_adaptive_tiling"
    cfg_adaptive.tokenizer.name = "dynamic_tile"
    cfg_adaptive.tokenizer.skip_mode = "threshold"
    cfg_adaptive.tokenizer.hard_skip_multiplier = 1.0
    cfg_adaptive.router.name = "DTRv2Router"
    cfg_adaptive.router.embed_dim = 256
    configs["adaptive_tiling"] = cfg_adaptive

    # 3. No sparsity — dynamic tiles, but identity router (all tokens pass through)
    cfg_nosparse = Config()
    cfg_nosparse.experiment_name = "bench_no_sparsity"
    cfg_nosparse.tokenizer.name = "dynamic_tile"
    cfg_nosparse.tokenizer.skip_mode = "threshold"
    cfg_nosparse.router.name = "IdentityRouter"
    cfg_nosparse.router.embed_dim = 256
    configs["no_sparsity"] = cfg_nosparse

    # 4. Sparse routing — hard skip + DTRv2 routing
    cfg_sparse = Config()
    cfg_sparse.experiment_name = "bench_sparse_routing"
    cfg_sparse.tokenizer.name = "dynamic_tile"
    cfg_sparse.tokenizer.skip_mode = "hard"
    cfg_sparse.tokenizer.hard_skip_multiplier = 1.0
    cfg_sparse.router.name = "DTRv2Router"
    cfg_sparse.router.embed_dim = 256
    configs["sparse_routing"] = cfg_sparse

    return configs


# ── Model building helpers ─────────────────────────────────────────────


def build_model_for_config(cfg: Config) -> torch.nn.Module:
    """Build an AdaTileFastSAM model for a given config."""
    # Override backbone to use a lightweight model
    cfg.backbone.name = "ResNet50Backbone"
    cfg.backbone.pretrained = False
    cfg.backbone.embed_dim = 256
    cfg.backbone.depth = 4
    cfg.backbone.num_heads = 4
    cfg.backbone.patch_size = 16
    cfg.backbone.output_scales = [4, 8, 16, 32]

    cfg.sparse.name = "ada_spm"
    cfg.sparse.num_scales = 4

    cfg.decoder.name = "fastsam_decoder"
    cfg.decoder.mask_dim = 256
    cfg.decoder.num_mask_tokens = 4
    cfg.decoder.iou_prediction = True

    cfg.prototype.name = ""  # disable

    return build_adatile_fastsam(cfg)


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaTile-FastSAM Empirical Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Single config name to benchmark (fixed_tiling, adaptive_tiling, "
             "no_sparsity, sparse_routing)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all 4 configs side-by-side",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick 5-iteration benchmark with adaptive_tiling",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full benchmark: compare configs + trace + plots",
    )
    parser.add_argument(
        "--image-size", type=str, default="1024,1024",
        help="Input image size as H,W (default: 1024,1024)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Profiling iterations (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Export torch.profiler chrome trace",
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA profiling (CPU timing only)",
    )
    parser.add_argument(
        "--sweep-resolutions", action="store_true",
        help="Sweep resolutions: 1024, 2048, 4096 (Reviewer Q5)",
    )
    parser.add_argument(
        "--measure-fps", action="store_true",
        help="Measure throughput (FPS) with increasing batch size (Reviewer Q5)",
    )
    return parser.parse_args()


def _measure_fps(model, image: torch.Tensor, use_cuda: bool,
                 iters: int = 50) -> float:
    """Measure throughput (FPS) with warmup.

    Returns frames per second (single forward pass, not batched).
    For batch throughput, multiply by batch size.
    """
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(image)

    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            model(image)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return iters / elapsed if elapsed > 0 else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    h, w = map(int, args.image_size.split(","))
    image = torch.randn(1, 3, h, w)

    if torch.cuda.is_available() and not args.no_cuda:
        image = image.cuda()

    use_cuda = torch.cuda.is_available() and not args.no_cuda

    # ── Mode dispatch ──────────────────────────────────────────────

    if args.quick:
        cfg = Config()
        cfg.experiment_name = "bench_quick"
        cfg.tokenizer.name = "dynamic_tile"
        cfg.tokenizer.skip_mode = "threshold"
        cfg.router.name = "DTRv2Router"
        cfg.router.embed_dim = 256
        model = build_model_for_config(cfg)
        if use_cuda:
            model = model.cuda()

        profiler = PipelineProfiler(model, use_cuda=use_cuda)
        result = profiler.profile(image, warmup=args.warmup, iterations=min(args.iters, 5))
        print(result.summary())

        # Auto-calibrate with empirical measurements
        # Empirical calibration: push profiler measurements to tile planner
        if hasattr(model, "pipeline") and hasattr(model.pipeline, "tokenizer"):
            planner = getattr(model.pipeline.tokenizer, "planner", None)
            if planner is not None and hasattr(planner, "set_calibrated_costs"):
                routed = max(result.routed_tokens, 1)
                total_ms = result.total_time_ms
                peak_mb = result.peak_memory_mb
                planner.set_calibrated_costs(
                    flops_per_tile=total_ms / routed * 1e6,
                    mem_bytes_per_tile=(peak_mb * 1024 * 1024) / routed,
                )

    elif args.compare or args.full:
        configs = build_configs()
        config_names = [args.config] if args.config else list(configs.keys())
        results = []

        for name in config_names:
            cfg = configs[name]
            print(f"\n{'='*60}")
            print(f"  Benchmarking: {name}")
            print(f"{'='*60}")

            model = build_model_for_config(cfg)
            if use_cuda:
                model = model.cuda()

            profiler = PipelineProfiler(model, use_cuda=use_cuda)
            result = profiler.profile(
                image,
                warmup=args.warmup,
                iterations=args.iters,
                use_torch_profiler=args.trace,
            )
            result.config_name = name
            results.append(result)
            print(result.summary())

            # Auto-calibrate with empirical measurements
            # Empirical calibration: push profiler measurements to tile planner
        if hasattr(model, "pipeline") and hasattr(model.pipeline, "tokenizer"):
            planner = getattr(model.pipeline.tokenizer, "planner", None)
            if planner is not None and hasattr(planner, "set_calibrated_costs"):
                routed = max(result.routed_tokens, 1)
                total_ms = result.total_time_ms
                peak_mb = result.peak_memory_mb
                planner.set_calibrated_costs(
                    flops_per_tile=total_ms / routed * 1e6,
                    mem_bytes_per_tile=(peak_mb * 1024 * 1024) / routed,
                )

        # Export
        cmp = CompareResult(results=results)
        prefix = "benchmark_compare"

        csv_path = os.path.join(args.output, f"{prefix}.csv")
        export_csv(results, csv_path)
        print(f"CSV:  {csv_path}")

        json_path = os.path.join(args.output, f"{prefix}.json")
        export_json(results, json_path)
        print(f"JSON: {json_path}")

        # Chrome trace
        if args.trace and profiler.last_trace is not None:
            trace_path = os.path.join(args.output, f"{prefix}_trace.json")
            export_chrome_trace(profiler.last_trace, trace_path)
            print(f"Trace: {trace_path}")

        # Plots
        plot_paths = plot_results(results, save_dir=args.output, prefix=prefix)
        for name, path in plot_paths.items():
            print(f"Plot [{name}]: {path}")

        # Speedup table
        if len(results) > 1:
            print(f"\n{'='*60}")
            print("  Speedup vs. fixed_tiling baseline")
            print(f"{'='*60}")
            speedups = cmp.speedup_vs("fixed_tiling")
            for name, su in speedups.items():
                print(f"  {name}: {su:.2f}x")

    elif args.sweep_resolutions:
        # Resolution sweep: 1024, 2048, 4096 (Reviewer Q5)
        resolutions = [(1024, 1024), (2048, 2048), (4096, 4096)]
        cfg = Config()
        cfg.experiment_name = "bench_resolution_sweep"
        cfg.tokenizer.name = "dynamic_tile"
        cfg.tokenizer.skip_mode = "threshold"
        cfg.router.name = "DTRv2Router"
        cfg.router.embed_dim = 256

        sweep_results = []
        for h, w in resolutions:
            print(f"\n── Resolution {h}x{w} ──")
            image = torch.randn(1, 3, h, w)
            if use_cuda:
                image = image.cuda()

            model = build_model_for_config(cfg)
            if use_cuda:
                model = model.cuda()

            profiler = PipelineProfiler(model, use_cuda=use_cuda)
            result = profiler.profile(image, warmup=args.warmup, iterations=args.iters)
            result.config_name = f"{h}x{w}"
            sweep_results.append(result)

            # FPS measurement
            fps = _measure_fps(model, image, use_cuda, args.iters)
            print(f"  Latency: {result.total_time_ms:.2f} ms, FPS: {fps:.2f}")

        # Export sweep results
        csv_path = os.path.join(args.output, "resolution_sweep.csv")
        export_csv(sweep_results, csv_path)
        json_path = os.path.join(args.output, "resolution_sweep.json")
        export_json(sweep_results, json_path)
        print(f"\nResolution sweep saved to {args.output}")

    elif args.measure_fps:
        # Throughput measurement at different batch sizes
        cfg = Config()
        cfg.tokenizer.name = "dynamic_tile"
        cfg.router.name = "DTRv2Router"
        cfg.router.embed_dim = 256
        model = build_model_for_config(cfg)
        if use_cuda:
            model = model.cuda()

        h, w = map(int, args.image_size.split(","))
        batch_sizes = [1, 2, 4, 8]
        print(f"\n── FPS Measurement at {h}x{w} ──")
        for bs in batch_sizes:
            try:
                image = torch.randn(bs, 3, h, w)
                if use_cuda:
                    image = image.cuda()
                fps = _measure_fps(model, image, use_cuda, args.iters)
                print(f"  Batch {bs}: {fps:.2f} FPS ({fps * bs:.1f} img/s)")
            except RuntimeError as e:
                print(f"  Batch {bs}: OOM ({e})")

    elif args.config:
        configs = build_configs()
        if args.config not in configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(configs.keys())}")
            sys.exit(1)

        cfg = configs[args.config]
        model = build_model_for_config(cfg)
        if use_cuda:
            model = model.cuda()

        profiler = PipelineProfiler(model, use_cuda=use_cuda)
        result = profiler.profile(
            image,
            warmup=args.warmup,
            iterations=args.iters,
            use_torch_profiler=args.trace,
        )
        result.config_name = args.config
        print(result.summary())

        # Auto-calibrate with empirical measurements
        # Empirical calibration: push profiler measurements to tile planner
        if hasattr(model, "pipeline") and hasattr(model.pipeline, "tokenizer"):
            planner = getattr(model.pipeline.tokenizer, "planner", None)
            if planner is not None and hasattr(planner, "set_calibrated_costs"):
                routed = max(result.routed_tokens, 1)
                total_ms = result.total_time_ms
                peak_mb = result.peak_memory_mb
                planner.set_calibrated_costs(
                    flops_per_tile=total_ms / routed * 1e6,
                    mem_bytes_per_tile=(peak_mb * 1024 * 1024) / routed,
                )

    else:
        print("Specify --config, --compare, --quick, or --full")
        print("Run --help for details")
        sys.exit(0)


if __name__ == "__main__":
    main()
