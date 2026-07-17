#!/usr/bin/env python3
"""
效率测量脚本 | Efficiency Measurement Script.
=============================================

为论文 Benchmark 表统一测量四个模型的效率指标:
Measure efficiency metrics of all benchmark models under one protocol:

    - Params      : total / trainable (torch ``numel`` 统计 | counted via numel)
    - GFLOPs      : thop hook 统计, 数值为 MACs (与 mmcv/mmseg get_flops 同口径)
                    thop hook-based MACs (same convention as mmcv/mmseg counters)
    - FPS / Latency: bs=1, fp32, 预热后计时 (CUDA 同步) | warmup + timed loop w/ cuda sync
    - GPU Memory  : 前向峰值显存 | peak forward memory

测量分辨率 = 各方法的实际推理分辨率 (与评估协议一致):
Resolution = each method's ACTUAL inference resolution (matches eval protocol):
    SegNeXt 200x200 原生 | native; UNet/DeepLabV3+/Ours pad 200->224 (32 的倍数).

用法 | Usage::

    # 全部四个模型 | All four models
    python tools/analyze/measure_efficiency.py --all --device cuda

    # 单个模型 | Single model
    python tools/analyze/measure_efficiency.py --model ours --device cuda
    python tools/analyze/measure_efficiency.py --model segnext --model-size tiny
    python tools/analyze/measure_efficiency.py --model deeplabv3plus
    python tools/analyze/measure_efficiency.py --model unet

结果写入 analysis/efficiency_results.json (按模型合并), 并输出 CSV 行。
Results merged into analysis/efficiency_results.json; CSV rows printed at the end.
"""

from __future__ import annotations

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed

NUM_CLASSES = 4  # NEU-Seg: BG + Inclusion + Patch + Scratch

# ── 各方法实际推理分辨率 | Actual inference resolution per method ──
# segnext: 200 原生 | native. 其余 pad 200->224 (32 倍数) | others pad to /32.
MODEL_INPUT_SIZE = {"ours": 224, "segnext": 200, "unet": 224, "deeplabv3plus": 224}

ALL_MODELS = ["unet", "deeplabv3plus", "segnext", "ours"]


# ═══════════════════════════════════════════════════════════════════
# 统一前向包装 | Unified forward wrappers
# ═══════════════════════════════════════════════════════════════════

class SegNeXtWrapper(nn.Module):
    """SegNeXt = MSCAN backbone + LightHamHead (与 eval_segnext 前向一致
    | same forward as eval_segnext)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img)
        logits = self.head(feats[1:])  # C2, C3, C4
        # 上采样回输入分辨率 (推理协议一部分) | upsample back (part of inference)
        return F.interpolate(logits, size=img.shape[2:], mode="bilinear",
                             align_corners=False)


class OursWrapper(nn.Module):
    """Ours = frozen FastSAM + MultiScaleSpectralAttention + PureDecoderP2P3P4
    (与 eval_neuseg 前向一致 | same forward as eval_neuseg)."""

    def __init__(self, backbone: nn.Module, spectral_attn: nn.Module | None,
                 decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.spectral_attn = spectral_attn
        self.decoder = decoder

    def train(self, mode: bool = True):
        """FastSAM backbone 永久 eval (v2 教训 #1: .train() 会 crash YOLOv8 Detect 头),
        只切换可训练模块 | Backbone stays in permanent eval (v2 lesson #1);
        toggle trainable modules only."""
        self.training = mode
        if self.spectral_attn is not None:
            self.spectral_attn.train(mode)
        self.decoder.train(mode)
        return self

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img, extract_proto=False)
        if self.spectral_attn is not None:
            spec = self.spectral_attn(p2=feats.get("p2"), p3=feats.get("p3"),
                                      p4=feats.get("p4"))
            feats.update(spec)
        pred = self.decoder(feats["p2"], feats["p3"], feats["p4"])
        return F.interpolate(pred.unsqueeze(0) if pred.dim() == 3 else pred,
                             size=img.shape[2:], mode="bilinear",
                             align_corners=False)


# ═══════════════════════════════════════════════════════════════════
# 模型构建 | Model builders
# ═══════════════════════════════════════════════════════════════════

def build_model(name: str, args, device: torch.device):
    """构建模型 (权重随机即可: FLOPs/FPS 与权重值无关)
    Build model (random weights are fine: FLOPs/FPS independent of values).

    :return: (wrapper 或 None, 附加信息 dict | extra info dict)
    """
    if name == "unet":
        from adatile.baselines import UNet
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=64).to(device)
        return model, {"desc": "UNet-BN (base=64)"}

    if name == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise SystemExit(
                "[ERROR] 需要 segmentation_models_pytorch | required:\n"
                "    pip install segmentation-models-pytorch")
        # encoder_weights=None: 避免下载, 不影响效率测量 | skip download, no effect
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights=None,
            in_channels=3, classes=NUM_CLASSES,
        ).to(device)
        return model, {"desc": "DeepLabV3+ (ResNet-50, SMP)"}

    if name == "segnext":
        from adatile.backbone.mscan import MSCAN, SEGNEXT_CONFIGS
        from adatile.decoder.ham_head import LightHamHead
        cfg = SEGNEXT_CONFIGS[args.model_size]
        backbone = MSCAN(model_size=args.model_size, pretrained=None,
                         drop_path_rate=0.1).to(device)
        embed_dims = cfg["embed_dims"]
        head = LightHamHead(
            in_channels=[embed_dims[1], embed_dims[2], embed_dims[3]],
            num_classes=NUM_CLASSES, ham_channels=256, channels=256,
            ham_kwargs=dict(MD_R=16), dropout_ratio=0.1,
        ).to(device)
        return SegNeXtWrapper(backbone, head), {"desc": f"SegNeXt-{args.model_size} (MSCAN)"}

    if name == "ours":
        from adatile.backbone import FastSAMBackbone
        from adatile.frequency import MultiScaleSpectralAttention
        from adatile.decoder.pure_cnn_decoder import PureDecoderP2P3P4

        # 显式传 device: FastSAMBackbone 内部默认优先 CUDA | pass device explicitly
        backbone = FastSAMBackbone(
            checkpoint=args.fastsam_weights, freeze_backbone=True,
            device=str(device),
        ).to(device)
        backbone.eval()

        # 通道自动探测 (与 eval_neuseg 一致) | auto-detect channels (same as eval)
        with torch.no_grad():
            backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
        ch = backbone.channels if all(v > 0 for v in backbone.channels.values()) else \
            {"p2": 160, "p3": 960, "p4": 1280, "p8": 1280}

        spectral_attn = None
        if args.spectral:
            spectral_attn = MultiScaleSpectralAttention(
                p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
                reduction=4, n_freq=16,
            ).to(device)
        decoder = PureDecoderP2P3P4(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=NUM_CLASSES, mid_channels=128,
        ).to(device)
        return OursWrapper(backbone, spectral_attn, decoder), \
            {"desc": "Ours (FastSAM + MSSA)", "channels": ch}

    raise ValueError(f"Unknown model: {name}")


# ═══════════════════════════════════════════════════════════════════
# 测量 | Measurements
# ═══════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> tuple[int, int]:
    """参数量统计 (numel) | Parameter counts via numel.

    :return: (total, trainable) — trainable = requires_grad=True 的参数
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model: nn.Module, img_size: int, device: torch.device) -> float:
    """GFLOPs (thop hook 统计, 返回 MACs, 与 mmcv/mmseg 同口径)
    GFLOPs via thop hooks — value is MACs, same convention as mmcv/mmseg.

    注意: hook 只统计 nn.Conv2d/Linear 等标准层, functional 矩阵乘 (如 NMF)
    不计入 — 与官方 SegNeXt 报告口径一致。
    Note: hooks count standard layers only; functional matmuls (e.g. NMF)
    are excluded — consistent with the official SegNeXt reporting.
    """
    try:
        from thop import profile
    except ImportError:
        raise SystemExit("[ERROR] 需要 thop | required:  pip install thop")

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return macs / 1e9


def measure_fps(model: nn.Module, img_size: int, device: torch.device,
                warmup: int = 20, iters: int = 200) -> dict:
    """FPS / 延迟 / 峰值显存 | FPS, latency, peak GPU memory.

    bs=1, fp32, torch.no_grad, 每次迭代 CUDA 同步计时。
    bs=1, fp32, no_grad, CUDA-synchronized timing per iteration.
    """
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        # ── 预热 | Warmup ──
        for _ in range(warmup):
            model(dummy)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        # ── 计时 | Timed loop ──
        t0 = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    latency_ms = elapsed / iters * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                   if is_cuda else float("nan"))
    return {
        "latency_ms": round(latency_ms, 3),
        "fps": round(1000 / latency_ms, 2),
        "peak_mem_MB": round(peak_mem_mb, 1),
        "warmup": warmup,
        "iters": iters,
    }


def measure_one(name: str, args, device: torch.device, logger) -> dict:
    """测量单个模型的全部效率指标 | Measure all efficiency metrics for one model."""
    logger.log_info("build", f"Building {name}...")
    model, info = build_model(name, args, device)
    img_size = args.img_size or MODEL_INPUT_SIZE[name]

    total, trainable = count_params(model)

    # Ours 特殊处理: FastSAM 对象不是 nn.Module (真实网络在 backbone.model.model),
    # wrapper.parameters() 和 thop hook 都看不到它 → 单独统计后合并。
    # Ours special case: the FastSAM object is NOT an nn.Module (real net at
    # backbone.model.model); invisible to wrapper.parameters() and thop hooks
    # → measure separately and merge.
    backbone_gmacs = None
    if name == "ours":
        inner = model.backbone.model.model  # YOLO nn.Module
        total += sum(p.numel() for p in inner.parameters())
        trainable += sum(p.numel() for p in inner.parameters()
                         if p.requires_grad)  # 冻结应为 0 | expected 0 (frozen)
        head_gmacs = measure_flops(model, img_size, device)      # spectral+decoder
        backbone_gmacs = measure_flops(inner, img_size, device)  # frozen backbone
        gmacs = head_gmacs + backbone_gmacs
    else:
        gmacs = measure_flops(model, img_size, device)

    # thop 的 profile 会在模块上挂 buffer, 需在计时前重建以免干扰
    # thop attaches buffers during profiling; rebuild before timing to avoid noise
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _ = build_model(name, args, device)
    speed = measure_fps(model, img_size, device,
                        warmup=args.warmup, iters=args.iters)

    result = {
        "model": name,
        "desc": info["desc"],
        "input_size": img_size,
        "params_total_M": round(total / 1e6, 3),
        "params_trainable_M": round(trainable / 1e6, 3),
        "gflops": round(gmacs, 2),
        "gflops_backbone_frozen": (round(backbone_gmacs, 2)
                                   if backbone_gmacs is not None else None),
        "gflops_convention": "thop MACs (hook-based, mmcv/mmseg convention)",
        **speed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": "fp32",
        "batch_size": 1,
        "timestamp": datetime.now().isoformat(),
    }

    logger.log_metric(f"{name}/params_total_M", result["params_total_M"])
    logger.log_metric(f"{name}/params_trainable_M", result["params_trainable_M"])
    logger.log_metric(f"{name}/gflops", result["gflops"])
    logger.log_metric(f"{name}/fps", result["fps"])
    logger.log_metric(f"{name}/latency_ms", result["latency_ms"])
    logger.log_metric(f"{name}/peak_mem_MB", result["peak_mem_MB"])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════
# 入口 | Entry point
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NEU-Seg Benchmark Efficiency Measurement")
    p.add_argument("--model", type=str, default=None, choices=ALL_MODELS,
                   help="单个模型 | Single model to measure")
    p.add_argument("--all", action="store_true",
                   help="测量全部四个模型 | Measure all four models")
    p.add_argument("--model-size", type=str, default="tiny",
                   help="SegNeXt 型号 | SegNeXt size (tiny/small/base)")
    p.add_argument("--fastsam-weights", type=str, default="FastSAM-x.pt",
                   help="FastSAM 权重路径 (仅 ours) | FastSAM weights path (ours only)")
    p.add_argument("--spectral", action=argparse.BooleanOptionalAction, default=True,
                   help="ours 是否含频域注意力 | Include spectral attention in ours")
    p.add_argument("--img-size", type=int, default=0,
                   help="覆盖默认推理分辨率, 0=按方法默认 | Override resolution, 0=per-method default")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/efficiency_results.json",
                   help="结果 JSON (按模型合并) | Result JSON (merged per model)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.all and args.model is None:
        raise SystemExit("[ERROR] 需要 --model <name> 或 --all | need --model or --all")

    set_seed(args.seed)
    device = torch.device(args.device)

    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger("efficiency")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_path.parent / "efficiency_log.jsonl")))

    names = ALL_MODELS if args.all else [args.model]

    # ── 合并已有结果 (增量测量) | Merge existing results (incremental) ──
    existing: dict = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    for name in names:
        result = measure_one(name, args, device, logger)
        existing[name] = result
        # 每个模型测完立即落盘 (crash-safe) | flush after each model
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        logger.log_info("save", f"{name} -> {out_path}")

    # ── CSV 行输出 (直接填 benchmark 表) | CSV rows for the benchmark table ──
    logger.log_info("csv", "model,params_total_M,params_trainable_M,gflops,"
                           "fps,latency_ms,peak_mem_MB,input_size")
    for name in names:
        r = existing[name]
        logger.log_info("csv", f"{name},{r['params_total_M']},"
                               f"{r['params_trainable_M']},{r['gflops']},"
                               f"{r['fps']},{r['latency_ms']},"
                               f"{r['peak_mem_MB']},{r['input_size']}")


if __name__ == "__main__":
    main()
