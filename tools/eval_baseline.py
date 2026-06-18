#!/usr/bin/env python3
"""
Baseline 评测脚本 | Baseline Evaluation Script
=================================================

在 Massachusetts Buildings 数据集上评测 FastSAM Baseline：
- mIoU  (平均交并比 | Mean Intersection over Union)
- Dice  (Dice 系数 | Dice coefficient)
- FPS   (推理速度 | Inference speed)
- Params (模型参数量 | Model parameter count)

数据流 | Data Flow:
    Dataset → Backbone → Features → (简单的阈值预测) → Metrics → Recorder

用法 | Usage:
    python tools/eval_baseline.py
    python tools/eval_baseline.py --split val
    python tools/eval_baseline.py --tile-size 512
    python tools/eval_baseline.py --image-size 1024 1024 --batch-size 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 将项目根目录加入路径 | Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import FPSMeter, compute_dice, compute_miou, count_params, format_param_count
from adatile.backbone import FastSAMBackbone


def parse_args() -> argparse.Namespace:
    """解析命令行参数 | Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FastSAM Baseline 评测 | FastSAM Baseline Evaluation (Massachusetts Buildings)"
    )
    # 数据集参数 | Dataset args
    parser.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings",
                        help="数据集根目录 | Dataset root directory")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="评测的分割 | Split to evaluate (default: val)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="瓦片尺寸（None=全图1500x1500）| Tile size (None=full image)")
    # 模型参数 | Model args
    parser.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024],
                        help="输入图像尺寸 (H W) | Input image size (default: 1024 1024)")
    # 输出参数 | Output args
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="输出根目录 | Output root directory")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称（可选）| Experiment name (optional)")
    # 速度评测 | Speed benchmark
    parser.add_argument("--fps-samples", type=int, default=20,
                        help="FPS 采样次数 | FPS sample count (default: 20)")
    parser.add_argument("--fps-warmup", type=int, default=5,
                        help="FPS 预热次数 | FPS warmup runs (default: 5)")
    return parser.parse_args()


def evaluate_baseline(args: argparse.Namespace) -> None:
    """
    执行 Baseline 评测 | Run Baseline evaluation.

    评测流程 | Evaluation pipeline:
    1. 生成实验 ID + 配置 + 记录器
    2. 加载数据集
    3. 加载 FastSAM 骨干网络
    4. 统计参数量
    5. 逐样本：加载 → 前向 → 简单预测 → 计算指标
    6. 测 FPS
    7. 输出汇总
    """
    print("=" * 70)
    print("  AdaTile-FastSAM v2 — Baseline 评测 | Baseline Evaluation")
    print("  数据集 | Dataset: Massachusetts Buildings")
    print("=" * 70)

    # ── 1. 实验配置 | Experiment Config ────────────────────────
    exp_id = generate_exp_id(name=args.name or f"baseline_{args.split}")
    config = ExperimentConfig(
        exp_id=exp_id,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        dataset_name="Massachusetts_Buildings",
        dataset_root=args.data_root,
    )
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Experiment: {exp_id}")
    print(f"  Device:     {device}")
    print(f"  Output:     {config.output_dir}/{exp_id}")

    # ── 2. 数据集 | Dataset ────────────────────────────────────
    print(f"\n--- 加载数据集 | Loading Dataset ({args.split}) ---")
    ds = MassachusettsBuildingsDataset(
        root_dir=args.data_root,
        split=args.split,
        tile_size=args.tile_size,
    )
    print(f"  Samples: {len(ds)}  |  Classes: {ds.num_classes}  |  "
          f"Image size: {ds[0]['image'].shape[1:]}")

    # ── 3. Backbone | FastSAM ──────────────────────────────────
    print("\n--- 加载 FastSAM Backbone | Loading FastSAM Backbone ---")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    # V1 校验 | V1 check
    assert not backbone.model.model.training, "YOLO model must be in eval mode!"
    print(f"  Backbone: FastSAM-x  |  Device: {device}")

    # ── 4. 参数量 | Parameter Count ────────────────────────────
    print("\n--- 模型参数 | Model Parameters ---")
    params = count_params(backbone)
    print(f"  Total:      {format_param_count(params['total'])} ({params['total']:,})")
    print(f"  Trainable:  {format_param_count(params['trainable'])} ({params['trainable']:,})")
    print(f"  Frozen:     {format_param_count(params['frozen'])} ({params['frozen']:,})")
    recorder.logger.log_info(
        "model/params",
        f"Total={params['total']}, Trainable={params['trainable']}, "
        f"Frozen={params['frozen']}",
    )

    # ── 5. 评测循环 | Evaluation Loop ──────────────────────────
    print(f"\n--- 评测 | Evaluation ({len(ds)} samples) ---")
    all_mious: list[float] = []
    all_dices: list[float] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        gt_mask = sample["masks"].to(device)              # [1, H, W]

        # 前向传播 → 提取特征 | Forward → extract features
        with torch.no_grad():
            features = backbone(image)

        # 简单预测：用 P4 特征均值做二值化 | Simple prediction: threshold P4 feature mean
        # 实际使用需要 Decoder，这里做最简单的近似
        # Real use needs Decoder; here we approximate
        p4_feat = features["p4"]                          # [1, C, H_p4, W_p4]
        # 通道平均 → 缩放到 [0,1] → 双线性上采样到原图尺寸
        # Channel mean → scale to [0,1] → bilinear upsample to original size
        pred_prob = p4_feat.mean(dim=1, keepdim=True)     # [1, 1, H_p4, W_p4]
        pred_prob = (pred_prob - pred_prob.min()) / (pred_prob.max() - pred_prob.min() + 1e-8)
        # 上采样到 GT 尺寸 | Upsample to GT size
        pred_prob = F.interpolate(pred_prob, size=gt_mask.shape[1:], mode="bilinear",
                                  align_corners=False)
        pred_binary = (pred_prob > 0.5).float()            # [1, 1, H, W]

        # 计算 Dice | Compute Dice
        dice_val = compute_dice(pred_binary, gt_mask.unsqueeze(0))
        all_dices.append(dice_val.item())

        # 计算 mIoU（二分类）| Compute mIoU (binary)
        pred_labels = pred_binary.squeeze(0).squeeze(0).long()  # [H, W]
        gt_labels = gt_mask.squeeze(0).long()                    # [H, W]
        miou_result = compute_miou(pred_labels, gt_labels, num_classes=2)
        all_mious.append(miou_result["miou"])

        # 记录 | Record
        recorder.record_metric("dice/val", dice_val.item(), step=idx, phase="val")
        recorder.record_metric("miou/val", miou_result["miou"], step=idx, phase="val")

        if (idx + 1) % max(1, len(ds) // 10) == 0:
            print(f"  [{idx + 1:4d}/{len(ds)}]  "
                  f"mIoU={miou_result['miou']:.4f}  Dice={dice_val.item():.4f}")

    # ── 6. FPS 测速 | FPS Measurement ──────────────────────────
    print("\n--- FPS 测速 | FPS Benchmark ---")
    fps_meter = FPSMeter(warmup=args.fps_warmup, num_runs=args.fps_samples)
    sample_input = ds[0]["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        for i in range(args.fps_warmup + args.fps_samples):
            with fps_meter:
                _ = backbone(sample_input)
    fps = fps_meter.compute()
    print(f"  FPS: {fps:.2f}  (warmup={args.fps_warmup}, samples={args.fps_samples})")
    recorder.record_metric("fps/inference", fps, phase="val")

    # ── 7. 汇总 | Summary ──────────────────────────────────────
    mean_miou = np.mean(all_mious) if all_mious else 0.0
    mean_dice = np.mean(all_dices) if all_dices else 0.0

    print(f"\n{'=' * 70}")
    print(f"  Baseline 评测结果 | Baseline Results")
    print(f"  {'─' * 50}")
    print(f"  Split:       {args.split}")
    print(f"  Samples:     {len(ds)}")
    print(f"  mIoU (mean): {mean_miou:.4f}  (min={min(all_mious):.4f}, max={max(all_mious):.4f})")
    print(f"  Dice (mean): {mean_dice:.4f}  (min={min(all_dices):.4f}, max={max(all_dices):.4f})")
    print(f"  FPS:         {fps:.2f}  (GPU={device})")
    print(f"  Params:      {format_param_count(params['total'])}  "
          f"(trainable={format_param_count(params['trainable'])})")
    print(f"{'=' * 70}")

    # 记录最终汇总 | Record final summary
    recorder.record_metric("miou/mean", mean_miou, phase="val", tags=["baseline", "summary"])
    recorder.record_metric("dice/mean", mean_dice, phase="val", tags=["baseline", "summary"])
    recorder.finalize()
    recorder.close()

    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    args = parse_args()
    evaluate_baseline(args)
