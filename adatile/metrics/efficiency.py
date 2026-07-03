"""
计算效率指标 | Computational Efficiency Metrics.
====================================================

提供 FLOPs、FPS、显存等计算效率指标的统一测量接口。
Unified interface for measuring FLOPs, FPS, memory, and parameter efficiency.

支持的指标 | Supported Metrics:
    - FLOPs: 浮点运算数 | Floating-point operations (via fvcore)
    - FPS: 每秒帧数 | Frames per second
    - Memory: GPU 显存占用 | GPU memory usage
    - Throughput: 吞吐量 (images/sec at various batch sizes)

用法 | Usage::

    from adatile.metrics.efficiency import EfficiencyBenchmark

    bench = EfficiencyBenchmark(model, input_shape=(3, 640, 640))
    results = bench.run(warmup=10, measure=100)
    print(f"FPS: {results['fps']:.1f}, FLOPs: {results['gflops']:.2f}G, "
          f"Memory: {results['memory_mb']:.1f}MB")
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# FLOPs 计算 | FLOPs Computation
# ═══════════════════════════════════════════════════════════════════

def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
) -> dict:
    """
    计算模型 FLOPs | Compute model FLOPs.

    优先使用 fvcore（更准确），回退到 thop。
    Prefers fvcore (more accurate), falls back to thop.

    :param model: PyTorch 模型 | PyTorch model.
    :param input_shape: 输入形状 (C, H, W) | Input shape (C, H, W).
    :param device: 设备 | Device.
    :return: {"gflops": float, "mparams": float, "method": str}
    """
    model = model.eval()

    # 尝试 fvcore | Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        dummy = torch.randn(1, *input_shape).to(device)
        flops = FlopCountAnalysis(model, dummy)
        total_flops = flops.total()
        total_params = sum(p.numel() for p in model.parameters())

        return {
            "gflops": total_flops / 1e9,
            "mparams": total_params / 1e6,
            "method": "fvcore",
        }
    except ImportError:
        pass

    # 回退到 thop | Fallback to thop
    try:
        from thop import profile, clever_format

        dummy = torch.randn(1, *input_shape).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)

        return {
            "gflops": flops / 1e9,
            "mparams": params / 1e6,
            "method": "thop",
        }
    except ImportError:
        pass

    # 手动估算（仅统计 Conv2d + Linear）| Manual estimate (Conv2d + Linear only)
    return {
        "gflops": _estimate_flops_manual(model, input_shape) / 1e9,
        "mparams": sum(p.numel() for p in model.parameters()) / 1e6,
        "method": "manual",
    }


def _estimate_flops_manual(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    手动估算 FLOPs（仅 Conv2d + Linear）| Manual FLOPs estimate (Conv2d + Linear only).

    公式 | Formula:
        Conv2d: 2 * Cin * Cout * Kh * Kw * Hout * Wout (MACs)
        Linear: 2 * in_features * out_features (MACs)
    """
    total_ops = 0
    C_in, H, W = input_shape

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # 估算输出空间尺寸 | Estimate output spatial size
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation

            H_out = (H + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            W_out = (W + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

            ops = 2 * module.in_channels * module.out_channels * kernel * kernel * H_out * W_out
            if module.bias is not None:
                ops += module.out_channels * H_out * W_out
            total_ops += ops

            # 更新尺寸用于下游层 | Update size for downstream layers
            C_in, H, W = module.out_channels, H_out, W_out

        elif isinstance(module, nn.Linear):
            ops = 2 * module.in_features * module.out_features
            if module.bias is not None:
                ops += module.out_features
            total_ops += ops

    return total_ops


# ═══════════════════════════════════════════════════════════════════
# FPS 测量 | FPS Measurement
# ═══════════════════════════════════════════════════════════════════

def measure_fps(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    device: str = "cuda",
    n_warmup: int = 10,
    n_measure: int = 100,
    use_amp: bool = False,
) -> dict:
    """
    测量推理 FPS | Measure inference FPS.

    使用 CUDA events 进行精确计时（GPU）或 time.perf_counter（CPU）。
    Uses CUDA events for precise timing (GPU) or time.perf_counter (CPU).

    :param model: PyTorch 模型 | PyTorch model.
    :param input_shape: 单张输入形状 (C, H, W) | Single input shape.
    :param batch_size: 批次大小 | Batch size.
    :param device: 设备 | Device.
    :param n_warmup: 预热迭代数 | Warmup iterations.
    :param n_measure: 测量迭代数 | Measurement iterations.
    :param use_amp: 是否使用 AMP 混合精度 | Whether to use AMP mixed precision.
    :return: {"fps": float, "latency_ms": float, "latency_std_ms": float, "batch_size": int}
    """
    model = model.eval()
    is_cuda = (device == "cuda" and torch.cuda.is_available())

    dummy = torch.randn(batch_size, *input_shape).to(device)

    # ── 预热 | Warmup ──
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if is_cuda:
        torch.cuda.synchronize()

    # ── 测量 | Measure ──
    latencies = []
    with torch.no_grad():
        for _ in range(n_measure):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(dummy)
                else:
                    _ = model(dummy)

                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # ms
            else:
                start = time.perf_counter()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(dummy)
                else:
                    _ = model(dummy)

                latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / avg_latency * batch_size

    return {
        "fps": round(fps, 2),
        "latency_ms": round(avg_latency, 2),
        "latency_std_ms": round(std_latency, 2),
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }


# ═══════════════════════════════════════════════════════════════════
# 显存测量 | Memory Measurement
# ═══════════════════════════════════════════════════════════════════

def measure_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """
    测量 GPU 显存占用 | Measure GPU memory usage.

    分别测量模型参数显存和推理峰值显存。
    Separately measures model parameter memory and inference peak memory.

    :param model: PyTorch 模型 | PyTorch model.
    :param input_shape: 单张输入形状 (C, H, W) | Single input shape.
    :param batch_size: 批次大小 | Batch size.
    :param device: 设备 | Device.
    :return: {"model_mb": float, "peak_mb": float, "input_mb": float, "total_mb": float}
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "model_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
            "peak_mb": 0.0,
            "input_mb": 0.0,
            "total_mb": 0.0,
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = model.eval().to(device)

    # 模型参数显存 | Model parameter memory
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    # 输入/激活峰值显存 | Input / activation peak memory
    dummy = torch.randn(batch_size, *input_shape).to(device)
    input_mem = dummy.numel() * dummy.element_size() / 1e6

    with torch.no_grad():
        _ = model(dummy)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()

    return {
        "model_mb": round(model_mem, 1),
        "peak_mb": round(peak_mem, 1),
        "input_mb": round(input_mem, 1),
        "total_mb": round(max(model_mem + input_mem, peak_mem), 1),
    }


# ═══════════════════════════════════════════════════════════════════
# 统一基准 | Unified Benchmark
# ═══════════════════════════════════════════════════════════════════

class EfficiencyBenchmark:
    """
    统一效率基准测试 | Unified Efficiency Benchmark.

    一键测量 FLOPs、FPS、显存、参数量。
    One-shot measurement of FLOPs, FPS, memory, and parameter count.

    用法 | Usage::

        bench = EfficiencyBenchmark(model, input_shape=(3, 640, 640))
        results = bench.run()
        print(bench.report())
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (3, 640, 640),
        device: str = "cuda",
    ):
        """
        :param model: PyTorch 模型 | PyTorch model.
        :param input_shape: 单张输入形状 (C, H, W) | Single input shape.
        :param device: 设备 | Device.
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self._results: Optional[dict] = None

    def run(
        self,
        batch_size: int = 1,
        n_warmup: int = 10,
        n_measure: int = 100,
    ) -> dict:
        """
        运行完整基准测试 | Run complete benchmark.

        :param batch_size: FPS/显存测量的批次大小 | Batch size for FPS/memory measurement.
        :param n_warmup: 预热迭代数 | Warmup iterations.
        :param n_measure: 测量迭代数 | Measurement iterations.
        :return: dict with all metrics.
        """
        self._results = {}

        # ── FLOPs | FLOPs ──
        flops = count_flops(self.model, self.input_shape, self.device)
        self._results.update(flops)

        # ── FPS | FPS ──
        fps = measure_fps(
            self.model, self.input_shape, batch_size,
            self.device, n_warmup, n_measure,
        )
        self._results.update(fps)

        # ── 显存 | Memory ──
        mem = measure_memory(self.model, self.input_shape, batch_size, self.device)
        self._results.update(mem)

        return self._results

    def report(self) -> str:
        """生成可读的基准报告 | Generate readable benchmark report."""
        if self._results is None:
            return "EfficiencyBenchmark: not yet run. Call .run() first."

        r = self._results
        lines = [
            "=" * 56,
            "Efficiency Benchmark | 效率基准测试",
            "=" * 56,
            f"  Input Shape  : {self.input_shape}",
            f"  Batch Size   : {r.get('batch_size', 1)}",
            f"  Device       : {self.device}",
            f"  FLOPs Method : {r.get('method', 'N/A')}",
            "-" * 56,
            f"  FLOPs        : {r.get('gflops', 0):.2f} GFLOPs",
            f"  Parameters   : {r.get('mparams', 0):.2f} M",
            f"  FPS          : {r.get('fps', 0):.1f} im/s",
            f"  Latency      : {r.get('latency_ms', 0):.1f} ± {r.get('latency_std_ms', 0):.1f} ms",
            f"  Model Memory : {r.get('model_mb', 0):.1f} MB",
            f"  Peak Memory  : {r.get('peak_mb', 0):.1f} MB",
            "=" * 56,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"EfficiencyBenchmark(shape={self.input_shape}, "
                f"device={self.device}, mparams={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M)")


# ═══════════════════════════════════════════════════════════════════
# 便捷函数 | Convenience Functions
# ═══════════════════════════════════════════════════════════════════

def throughput_sweep(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_sizes: list[int] = None,
    device: str = "cuda",
) -> dict:
    """
    吞吐量扫描：不同 batch size 下的 FPS | Throughput sweep: FPS at various batch sizes.

    :param model: PyTorch 模型 | PyTorch model.
    :param input_shape: 输入形状 (C, H, W) | Input shape.
    :param batch_sizes: 要测试的 batch size 列表 | List of batch sizes to test.
    :param device: 设备 | Device.
    :return: {batch_size: fps} 映射.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    results = {}
    for bs in batch_sizes:
        try:
            fps = measure_fps(model, input_shape, batch_size=bs, device=device,
                            n_warmup=5, n_measure=50)
            results[bs] = fps["fps"]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch_size={bs}")
                results[bs] = 0.0
                torch.cuda.empty_cache()
            else:
                raise

    return results
