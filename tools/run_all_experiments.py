#!/usr/bin/env python3
"""
批量实验运行脚本 — 最大化 RTX 5090 GPU 利用率.
Batch Experiment Runner — Maximize RTX 5090 GPU utilization.
=============================================================

功能 | Features:
    1. 按依赖顺序串行运行全部实验 (避免 GPU 竞争)
    2. 断点续跑 — 已完成实验自动跳过
    3. 每个训练完成后自动评估
    4. 实时汇总到 experiment_results.csv
    5. GPU 内存/温度监控 (nvidia-smi)
    6. 失败自动重试 (最多 3 次)

用法 | Usage::

    # 完整运行 (全部实验)
    python tools/run_all_experiments.py --data-root /root/autodl-tmp/iSAID_instance_fewshot

    # 仅运行指定 Phase
    python tools/run_all_experiments.py --data-root /path/to/data --phases A,B

    # 从指定实验开始续跑
    python tools/run_all_experiments.py --data-root /path/to/data --resume

    # 干跑 (仅打印实验列表, 不执行)
    python tools/run_all_experiments.py --data-root /path/to/data --dry-run

    # 跳过已完成实验, 强制重新运行
    python tools/run_all_experiments.py --data-root /path/to/data --force

5090 优化策略 | 5090 Optimization:
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (减少显存碎片)
    - CUDA_LAUNCH_BLOCKING=0 (异步 kernel launch)
    - tf32=on (Ampere+ tensor core 加速)
    - 连续运行无间隙, 避免 GPU 空闲
    - 评估脚本轻量 — 训练间隙快速完成

实验总数 | Total Experiments:
    训练: ~28 个 | 评估: ~28+ 个 | 预估总时长: 12-18 小时
"""

from __future__ import annotations

import sys
import os
import re
import json
import time
import signal
import subprocess
import argparse
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import OrderedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════
# 配置 | Configuration
# ═══════════════════════════════════════════════════════════════════

# 5090 环境变量 | RTX 5090 Environment Variables
ENV_5090 = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_LAUNCH_BLOCKING": "0",
    "NVIDIA_TF32_OVERRIDE": "1",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}

# 训练脚本路径 | Training script paths
TRAIN_FEWSHOT = "tools/train/train_fewshot_allclass.py"
TRAIN_SUPERVISED = "tools/train/train_supervised_full.py"
EVAL_FEWSHOT = "tools/eval/eval_fewshot_allclass.py"

# ═══════════════════════════════════════════════════════════════════
# 实验定义 | Experiment Definitions
# ═══════════════════════════════════════════════════════════════════

def build_experiments(data_root: str) -> list[dict]:
    """
    构建全部实验列表 | Build complete experiment list.

    每个实验是一个 dict:
        - id: 实验唯一标识
        - phase: 阶段名 (A/B/C/D/E/F/FS)
        - script: 脚本路径
        - args: 命令行参数
        - train: 是否为训练实验 (False = 纯评估)
        - depends_on: 依赖的实验 ID (训练完成后才评估)
        - description: 简短描述

    :param data_root: 数据根目录 | Data root directory
    :return: 实验列表 | Experiment list
    """
    BASE_ARGS = f"--data-root {data_root} --data-format isaid_instance --device cuda"

    experiments = []

    # ═══════════════════════════════════════════════════════════════
    # Phase A: 解冻曲线 | Unfreeze Ablation (8 experiments)
    #   K=1, seed=42, epochs=200, decoder=adaptive
    #   统一 200 epochs — 之前实验已证明 50 epochs 严重欠训练
    # ═══════════════════════════════════════════════════════════════
    phase_a_configs = [
        ("A-0",  0,  1e-3, None,  "全冻结 baseline | Fully frozen baseline"),
        ("A-1",  1,  1e-3, None,  "仅解冻 Segment/Proto head | Only Segment+Proto head"),
        ("A-5",  5,  1e-3, None,  "解冻 P4+P8+Segment | Unfreeze P4+P8+Segment"),
        ("A-8",  8,  1e-3, None,  "解冻完整 P3+P4+P8+FPN-top (BASELINE)"),
        ("A-10", 10, 1e-3, None,  "部分解冻 FPN | Partial FPN unfreeze"),
        ("A-12", 12, 1e-3, None,  "完整 FPN top-down pathway | Full FPN top-down"),
        ("A-14", 14, 1e-3, None,  "含 SPPF 解冻 | Including SPPF unfreeze"),
        ("A-23", 23, 1e-5, 1e-5,  "全解冻 (lr=1e-5) | Full backbone unfreeze, low LR"),
    ]

    for exp_id, uf, lr, lr_bb, desc in phase_a_configs:
        args = (f"{BASE_ARGS} --k-shot 1 --epochs 200 --seed 42 "
                f"--decoder adaptive --unfreeze-layers {uf} --lr {lr}")
        if lr_bb is not None:
            args += f" --lr-backbone {lr_bb}"
        experiments.append({
            "id": exp_id, "phase": "A", "script": TRAIN_FEWSHOT,
            "args": args, "train": True, "depends_on": None,
            "description": desc, "tags": "unfreeze,adaptive,k1,200ep",
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase B: K-shot 缩放 | K-shot Scaling (9 experiments)
    #   uf=8, decoder=adaptive, epochs=50, 3 seeds × 3 K
    # ═══════════════════════════════════════════════════════════════
    phase_b_configs = []
    for k in [1, 3, 5]:
        for seed in [42, 123, 456]:
            phase_b_configs.append((f"B-{k}_s{seed}", k, seed,
                                    f"K={k}, seed={seed}"))

    for exp_id, k, seed, desc in phase_b_configs:
        args = (f"{BASE_ARGS} --k-shot {k} --epochs 50 --seed {seed} "
                f"--decoder adaptive --unfreeze-layers 8 --lr 1e-3")
        experiments.append({
            "id": exp_id, "phase": "B", "script": TRAIN_FEWSHOT,
            "args": args, "train": True, "depends_on": None,
            "description": desc, "tags": "kshot,adaptive,uf8",
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase C: 消融矩阵 | Ablation Matrix (4 experiments)
    #   K=1, seed=42, epochs=50
    # ═══════════════════════════════════════════════════════════════
    phase_c_configs = [
        ("C-E1", "adaptive",       "p4", 8,  "BASELINE: P4 decoder + P4 proto"),
        ("C-E2", "adaptive",       "p8", 8,  "P4 decoder + P8 proto (验证 P8 语义)"),
        ("C-E3", "adaptive-p3p4",  "p4", 8,  "P3P4 decoder + P4 proto (验证 P3 贡献)"),
        ("C-E4", "adaptive-p3p4",  "p8", 8,  "P3P4 decoder + P8 proto (三层协同最优)"),
    ]

    for exp_id, decoder, proto_src, uf, desc in phase_c_configs:
        args = (f"{BASE_ARGS} --k-shot 1 --epochs 50 --seed 42 "
                f"--decoder {decoder} --unfreeze-layers {uf} --lr 1e-3 "
                f"--prototype-source {proto_src}")
        experiments.append({
            "id": exp_id, "phase": "C", "script": TRAIN_FEWSHOT,
            "args": args, "train": True, "depends_on": None,
            "description": desc, "tags": f"ablation,{decoder},proto{proto_src}",
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase D: Proto 消融 | Proto Ablation (纯评估, 依赖 C-E1/C-E4)
    # ═══════════════════════════════════════════════════════════════
    phase_d_configs = [
        ("D-zero",   "C-E1", "adaptive",     "zero",   "ProtoCoeffPredictor 零输入消融"),
        ("D-random", "C-E1", "adaptive",     "random", "ProtoCoeffPredictor 随机输入消融"),
        ("D-E4-zero","C-E4", "adaptive-p3p4","zero",   "P3P4 decoder + 零原型消融"),
    ]

    for exp_id, depends, decoder, proto_abl, desc in phase_d_configs:
        experiments.append({
            "id": exp_id, "phase": "D", "script": EVAL_FEWSHOT,
            "args": (f"--checkpoint __CHECKPOINT_OF_{depends}__ "
                     f"--k-shot 1 {BASE_ARGS} --decoder {decoder} "
                     f"--proto-ablation {proto_abl} "
                     f"--per-class 9999"),  # 使用全部 val 源图
            "train": False, "depends_on": depends,
            "description": desc, "tags": "proto-ablation,eval-only",
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase E: Pure Decoder | 纯 CNN 解码器 (1 experiment)
    # ═══════════════════════════════════════════════════════════════
    experiments.append({
        "id": "E-pure", "phase": "E", "script": TRAIN_FEWSHOT,
        "args": (f"{BASE_ARGS} --k-shot 1 --epochs 50 --seed 42 "
                 f"--decoder pure --unfreeze-layers 8 --lr 1e-3"),
        "train": True, "depends_on": None,
        "description": "Pure CNN decoder, 无原型通路 | No prototype pathway",
        "tags": "pure,cnn-only,uf8",
    })

    # ═══════════════════════════════════════════════════════════════
    # Phase F: P3P4 + uf12 | 强 Backbone + 多尺度 Decoder (1 experiment)
    # ═══════════════════════════════════════════════════════════════
    experiments.append({
        "id": "F-p3p4-uf12", "phase": "F", "script": TRAIN_FEWSHOT,
        "args": (f"{BASE_ARGS} --k-shot 1 --epochs 50 --seed 42 "
                 f"--decoder adaptive-p3p4 --unfreeze-layers 12 --lr 1e-3 "
                 f"--prototype-source p4"),
        "train": True, "depends_on": None,
        "description": "P3P4 decoder + uf=12 strong backbone",
        "tags": "p3p4,uf12,strong-bb",
    })

    # ═══════════════════════════════════════════════════════════════
    # Phase FS: Full Supervision | 全监督上限 (3 experiments)
    #   decoder=*, uf=12, epochs=100
    # ═══════════════════════════════════════════════════════════════
    phase_fs_configs = [
        ("FS-pure",          "pure",          "Pure decoder: 全监督架构上限"),
        ("FS-adaptive",      "adaptive",      "Adaptive decoder: 全监督上限 (主要基线)"),
        ("FS-adaptive-p3p4", "adaptive-p3p4", "P3P4 decoder: 全监督上限 (最优架构)"),
    ]

    for exp_id, decoder, desc in phase_fs_configs:
        args = (f"--data-root {data_root} --epochs 100 --seed 42 "
                f"--decoder {decoder} --unfreeze-layers 12 --lr 1e-3 "
                f"--device cuda --batch-size 1")
        experiments.append({
            "id": exp_id, "phase": "FS", "script": TRAIN_SUPERVISED,
            "args": args, "train": True, "depends_on": None,
            "description": desc, "tags": "full-supervision,ceiling",
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase NS: NEU_Seg 工业缺陷分割 | Industrial Defect Segmentation
    #   多类别 (4-class): BG + Inclusion + Patch + Scratch
    # ═══════════════════════════════════════════════════════════════
    TRAIN_NEUSEG = "tools/train/train_neuseg.py"
    EVAL_NEUSEG = "tools/eval/eval_neuseg.py"

    # ── NS-1: 基础 Pure CNN Decoder | Baseline Pure CNN ──
    experiments.append({
        "id": "NS-pure", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 baseline (CE+Dice)", "tags": "neuseg,pure,baseline",
    })

    # ── NS-2: Adaptive Decoder | 自适应原型解码器 ──
    experiments.append({
        "id": "NS-adaptive", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type adaptive "
                 "--epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: AdaptiveSparseDecoder (prototype-conditioned)", "tags": "neuseg,adaptive,prototype",
    })

    # ── NS-3: LoRA r=2 | 轻量微调 ──
    experiments.append({
        "id": "NS-lora2", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--lora-rank 2 --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 + ConvLoRA r=2", "tags": "neuseg,lora,r2",
    })

    # ── NS-4: LoRA r=4 | 中等微调 ──
    experiments.append({
        "id": "NS-lora4", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--lora-rank 4 --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 + ConvLoRA r=4", "tags": "neuseg,lora,r4",
    })

    # ── NS-5: MultiScaleAdapter | CAT-SAM 风格适配器 ──
    experiments.append({
        "id": "NS-adapter", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type adaptive "
                 "--use-adapter --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: Adaptive + MultiScaleAdapter (CAT-SAM style)", "tags": "neuseg,adapter,catsam",
    })

    # ── NS-6: Spectral Attention + Pure | 频域注意力 ──
    experiments.append({
        "id": "NS-spectral", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--use-spectral --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 + DCT SpectralAttention", "tags": "neuseg,spectral,dct",
    })

    # ── NS-7: Lovász loss | 边界优化损失 ──
    experiments.append({
        "id": "NS-lovasz", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--loss-type lovasz --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 + Lovász-Softmax loss", "tags": "neuseg,lovasz,boundary",
    })

    # ── NS-8: Augmentation v2 | 数据增强 ──
    experiments.append({
        "id": "NS-augv2", "phase": "NS", "script": TRAIN_NEUSEG,
        "args": ("--config configs/neu_seg.yaml --decoder-type pure_p3p4 "
                 "--augment --epochs 200 --device cuda"),
        "train": True, "depends_on": None,
        "description": "NEU_Seg: PureDecoderP3P4 + full augmentation (CLAHE+Gamma+Blur)", "tags": "neuseg,augmentation",
    })

    return experiments


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def get_gpu_info() -> str:
    """获取 GPU 状态 | Get GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "N/A"


def print_gpu_status(prefix: str = ""):
    """打印 GPU 状态 | Print GPU status."""
    info = get_gpu_info()
    if info != "N/A":
        print(f"  {prefix}[GPU] {info}")


def find_checkpoint(exp_id: str, runs_dir: Path, progress: ProgressTracker = None) -> Optional[Path]:
    """在 runs 目录中查找指定实验的 checkpoint.

    优先从 progress tracker 中获取 output_dir (最可靠),
    然后回退到文件系统搜索.
    """
    # 优先: 从 progress tracker 获取已知路径 | Priority: use stored output_dir
    if progress is not None and progress.is_done(exp_id):
        stored_dir = progress.data["completed"][exp_id].get("output_dir", "")
        if stored_dir:
            ckpt = Path(stored_dir) / "best_model.pt"
            if ckpt.exists():
                return ckpt

    # 回退: 文件系统搜索 | Fallback: filesystem search
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        dname = d.name.lower()
        eid = exp_id.lower().replace("_", "").replace("-", "")
        if eid in dname.replace("_", "").replace("-", ""):
            ckpt = d / "best_model.pt"
            if ckpt.exists():
                return ckpt
    return None


def resolve_checkpoint_arg(args: str, runs_dir: Path, progress: ProgressTracker = None) -> str:
    """将 __CHECKPOINT_OF_xxx__ 占位符替换为实际路径."""
    pattern = r"__CHECKPOINT_OF_([\w-]+)__"
    match = re.search(pattern, args)
    if match:
        dep_id = match.group(1)
        ckpt = find_checkpoint(dep_id, runs_dir, progress)
        if ckpt is None:
            raise FileNotFoundError(
                f"Cannot find checkpoint for dependency '{dep_id}'. "
                f"Make sure it has completed training."
            )
        args = re.sub(pattern, str(ckpt), args)
    return args


class ProgressTracker:
    """进度追踪器 | Progress tracker — JSON-based, crash-safe."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[WARN] Corrupted progress file, starting fresh")
                self.data = {}

    def is_done(self, exp_id: str) -> bool:
        return exp_id in self.data.get("completed", {})

    def mark_done(self, exp_id: str, result: dict):
        if "completed" not in self.data:
            self.data["completed"] = {}
        self.data["completed"][exp_id] = result
        self.data["last_updated"] = datetime.now().isoformat()
        self._save()

    def mark_failed(self, exp_id: str, error: str):
        if "failed" not in self.data:
            self.data["failed"] = OrderedDict()
        self.data["failed"][exp_id] = {
            "error": error,
            "time": datetime.now().isoformat(),
        }
        self._save()

    def get_done_ids(self) -> set:
        return set(self.data.get("completed", {}).keys())

    def _save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


class ResultsCSV:
    """结果 CSV 写入器 | Results CSV writer — 实时追加, crash-safe."""

    COLUMNS = [
        "exp_id", "phase", "description", "status",
        "mIoU", "per_class_ious", "best_epoch", "total_epochs",
        "duration_min", "gpu_peak_mem_mb",
        "train_args", "output_dir", "timestamp"
    ]

    def __init__(self, path: Path):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def add_result(self, result: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([result.get(col, "") for col in self.COLUMNS])


# ═══════════════════════════════════════════════════════════════════
# 实验执行器 | Experiment Runner
# ═══════════════════════════════════════════════════════════════════

def _snapshot_runs(runs_dir: Path) -> set:
    """获取 runs 目录快照 | Take a snapshot of the runs directory."""
    return {d.name for d in runs_dir.iterdir() if d.is_dir()}


def run_single_experiment(
    exp: dict,
    runs_dir: Path,
    progress: ProgressTracker,
    results_csv: ResultsCSV,
    force: bool = False,
    retry_count: int = 3,
) -> bool:
    """
    运行单个实验 | Run a single experiment.

    :param exp: 实验配置 | Experiment config dict
    :param runs_dir: runs 输出目录 | Runs output directory
    :param progress: 进度追踪器 | Progress tracker
    :param results_csv: 结果 CSV 写入器 | Results CSV writer
    :param force: 强制重新运行 | Force re-run even if completed
    :param retry_count: 失败重试次数 | Max retry attempts
    :return: True 如果成功 | True if successful
    """
    exp_id = exp["id"]
    phase = exp["phase"]

    # ── 跳过已完成实验 | Skip completed ──
    if not force and progress.is_done(exp_id):
        prev = progress.data["completed"][exp_id]
        print(f"  [{exp_id}] [SKIP] done at {prev.get('timestamp', '?')}, "
              f"mIoU={prev.get('mIoU', 'N/A')})")
        return True

    # ── 解析依赖 | Resolve dependencies ──
    args_str = exp["args"]
    if exp.get("depends_on"):
        args_str = resolve_checkpoint_arg(args_str, runs_dir, progress)

    script_path = _PROJECT_ROOT / exp["script"]
    cmd = [sys.executable, str(script_path)] + args_str.split()

    # ── 打印实验信息 | Print experiment info ──
    print(f"\n{'=' * 70}")
    print(f"  [{exp_id}] Phase={phase} | {'TRAIN' if exp['train'] else 'EVAL'}")
    print(f"  {exp['description']}")
    print(f"  CMD: {' '.join(cmd[2:4])} ...")
    print(f"{'=' * 70}")
    print_gpu_status("Before: ")

    # ── 快照 runs 目录 (用于之后查找输出目录) ──
    before_dirs = _snapshot_runs(runs_dir)

    # ── 执行 (带重试) | Execute (with retry) ──
    start_time = time.time()
    success = False
    last_error = ""
    stdout_lines = []  # 捕获输出以解析路径 | Capture output for path parsing

    for attempt in range(1, retry_count + 1):
        if attempt > 1:
            print(f"  [{exp_id}] [RETRY] Attempt {attempt}/{retry_count}...")
            time.sleep(5)  # 等待 GPU 冷却 | Wait for GPU cool-down

        try:
            # 合并环境变量 | Merge environment variables
            env = os.environ.copy()
            env.update(ENV_5090)

            # 使用 Popen 实现 tee: 同时输出到终端和捕获
            process = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Tee: 逐行读取, 同时输出到终端和缓存
            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                stdout_lines.append(line)

            process.wait()

            if process.returncode == 0:
                success = True
                break
            else:
                last_error = f"Exit code {process.returncode}"
                print(f"  [{exp_id}] [FAIL] Attempt {attempt} failed: {last_error}")

        except KeyboardInterrupt:
            print(f"\n  [{exp_id}] [STOP]  Interrupted by user")
            raise
        except Exception as e:
            last_error = str(e)
            print(f"  [{exp_id}] [FAIL] Attempt {attempt} crashed: {last_error}")

    duration = (time.time() - start_time) / 60.0  # minutes

    # ── 提取结果 | Extract results ──
    result = {
        "exp_id": exp_id,
        "phase": phase,
        "description": exp["description"],
        "status": "OK" if success else "FAILED",
        "mIoU": None,
        "per_class_ious": "",
        "best_epoch": None,
        "total_epochs": None,
        "duration_min": f"{duration:.1f}",
        "gpu_peak_mem_mb": "",
        "train_args": exp["args"],
        "output_dir": "",
        "timestamp": datetime.now().isoformat(),
    }

    # ── 从日志提取 mIoU | Extract mIoU from output logs ──
    if success:
        # 通过 runs 目录 diff 找到新创建的目录
        after_dirs = _snapshot_runs(runs_dir)
        new_dirs = after_dirs - before_dirs

        if new_dirs:
            # 取最新的目录 (按名称中的时间戳排序)
            result = _extract_result_from_dirs(
                new_dirs, runs_dir, exp, result
            )
        else:
            # Fallback: 从 stdout 解析输出路径
            result = _extract_result_from_stdout(
                stdout_lines, runs_dir, exp, result
            )

    # ── 记录结果 | Record result ──
    print_gpu_status("After:  ")
    print(f"  [{exp_id}] {'[OK]' if success else '[FAIL]'} Done in {duration:.1f} min"
          + (f", mIoU={result['mIoU']}" if result['mIoU'] else ""))

    if success:
        progress.mark_done(exp_id, result)
    else:
        progress.mark_failed(exp_id, last_error)

    results_csv.add_result(result)

    # ── 自动评估 | Auto-eval (if enabled and training succeeded) ──
    if success and exp["train"] and auto_eval_enabled:
        _run_auto_eval(exp, result, runs_dir, progress, results_csv, force, retry_count)

    return success


# 全局标记 | Global flag for auto-eval
auto_eval_enabled = False


def _run_auto_eval(
    train_exp: dict,
    train_result: dict,
    runs_dir: Path,
    progress: ProgressTracker,
    results_csv: ResultsCSV,
    force: bool,
    retry_count: int,
):
    """
    训练后自动评估 | Auto-evaluate after training.

    从训练参数中解析 k-shot, decoder, proto-source 等,
    在 best_model.pt 上运行 eval_fewshot_allclass.py.
    """
    train_id = train_exp["id"]
    eval_id = f"{train_id}-eval"

    # 跳过已完成的评估 | Skip completed eval
    if not force and progress.is_done(eval_id):
        return

    # 找到训练输出目录和 checkpoint | Find training output dir and checkpoint
    output_dir = train_result.get("output_dir", "")
    if not output_dir:
        print(f"  [{eval_id}] [SKIP] Cannot find output directory for auto-eval")
        return

    ckpt_path = Path(output_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [{eval_id}] [SKIP] No best_model.pt found in {output_dir}")
        return

    # 从训练参数中提取 eval 需要的参数 | Extract eval params from training args
    train_args = train_exp["args"]
    k_shot = _extract_arg(train_args, "--k-shot", "1")
    decoder = _extract_arg(train_args, "--decoder", "adaptive")
    proto_src = _extract_arg(train_args, "--prototype-source", "p4")
    seed = _extract_arg(train_args, "--seed", "42")
    data_root = _extract_arg(train_args, "--data-root", "")
    data_format = _extract_arg(train_args, "--data-format", "isaid_instance")

    eval_args = (
        f"--checkpoint {ckpt_path} "
        f"--k-shot {k_shot} "
        f"--decoder {decoder} "
        f"--prototype-source {proto_src} "
        f"--seed {seed} "
        f"--data-root {data_root} "
        f"--data-format {data_format} "
        f"--device cuda "
        f"--per-class 9999"  # 使用全部 val 源图 | Use all val source images
    )

    eval_exp = {
        "id": eval_id,
        "phase": f"{train_exp['phase']}-Eval",
        "script": EVAL_FEWSHOT,
        "args": eval_args,
        "train": False,
        "depends_on": None,
        "description": f"Auto-eval for {train_id}",
        "tags": "auto-eval",
    }

    print(f"\n  [{eval_id}] Running auto-evaluation...")
    run_single_experiment(
        eval_exp, runs_dir, progress, results_csv,
        force=force, retry_count=retry_count,
    )


def _extract_arg(args_str: str, flag: str, default: str = "") -> str:
    """从参数字符串中提取参数值 | Extract argument value from args string."""
    # 匹配 --flag value 或 --flag=value
    pattern = rf'{re.escape(flag)}\s+(\S+)'
    match = re.search(pattern, args_str)
    if match:
        return match.group(1)
    return default


def _extract_result_from_dirs(
    new_dirs: set, runs_dir: Path, exp: dict, result: dict
) -> dict:
    """从新创建的 runs 子目录中提取结果 | Extract results from new run directories."""
    for dirname in sorted(new_dirs):
        d = runs_dir / dirname
        if not d.is_dir():
            continue

        if exp["train"]:
            # ── 训练结果: 读取 train_log.json ──
            train_log = d / "train_log.json"
            if train_log.exists():
                try:
                    log_data = json.loads(train_log.read_text(encoding="utf-8"))
                    result["mIoU"] = f"{log_data.get('best_val_miou', 0):.4f}"
                    # 从 entries 中获取 epoch 信息
                    entries = log_data.get("entries", [])
                    if entries:
                        result["total_epochs"] = entries[-1].get("epoch", "?")
                        # 找最佳 epoch
                        best_entry = max(
                            (e for e in entries if "val_miou" in e),
                            key=lambda e: e.get("val_miou", 0),
                            default=None
                        )
                        if best_entry:
                            result["best_epoch"] = best_entry.get("epoch", "?")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
        else:
            # ── 评估结果: 读取 comparison.json ──
            comparison_json = d / "comparison.json"
            if comparison_json.exists():
                try:
                    data = json.loads(comparison_json.read_text(encoding="utf-8"))
                    result["mIoU"] = f"{data.get('ft_overall_mean_iou', 0):.4f}"
                    # 提取 per-class IoU
                    per_class = data.get("per_class", {})
                    if per_class:
                        ious = [v["ft_mean_iou"] for v in per_class.values()]
                        result["per_class_ious"] = ",".join(f"{x:.3f}" for x in ious)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass

        result["output_dir"] = str(d)
        break  # 只处理第一个新目录

    return result


def _extract_result_from_stdout(
    stdout_lines: list, runs_dir: Path, exp: dict, result: dict
) -> dict:
    """从 stdout 回退解析结果 | Fallback: parse results from stdout."""
    # 尝试从输出中查找 "Output: runs/..." 行
    for line in stdout_lines:
        if "Output:" in line and "runs/" in line:
            # 提取路径
            import re
            match = re.search(r'runs/\S+', line)
            if match:
                out_path = _PROJECT_ROOT / match.group(0)
                if out_path.is_dir():
                    # 递归提取
                    fake_new_dirs = {out_path.name}
                    return _extract_result_from_dirs(
                        fake_new_dirs, runs_dir, exp, result
                    )
            break
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量实验运行脚本 | Batch Experiment Runner for RTX 5090",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 | Examples:
  # 完整运行, 数据在 autodl-tmp
  python tools/run_all_experiments.py --data-root /root/autodl-tmp/iSAID_instance_fewshot

  # 仅运行 Phase A + B (快速验证)
  python tools/run_all_experiments.py --data-root /path/to/data --phases A,B

  # 续跑 (自动跳过已完成)
  python tools/run_all_experiments.py --data-root /path/to/data --resume

  # 干跑 — 仅列出实验, 不执行
  python tools/run_all_experiments.py --data-root /path/to/data --dry-run

  # 从指定实验 ID 开始
  python tools/run_all_experiments.py --data-root /path/to/data --start-from C-E1
        """
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="数据根目录 (iSAID_instance_fewshot 路径) | Data root directory")
    parser.add_argument("--phases", type=str, default=None,
                        help="仅运行指定 Phase, 逗号分隔 (如 A,B,C) | Only run specific phases")
    parser.add_argument("--start-from", type=str, default=None,
                        help="从指定实验 ID 开始运行 | Start from a specific experiment ID")
    parser.add_argument("--resume", action="store_true",
                        help="续跑模式: 自动跳过已完成实验 | Resume mode: skip completed")
    parser.add_argument("--force", action="store_true",
                        help="强制重新运行所有实验 (忽略进度) | Force re-run all experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑: 仅打印实验列表 | Dry run: only list experiments")
    parser.add_argument("--retry", type=int, default=3,
                        help="失败重试次数 (默认 3) | Max retry attempts (default 3)")
    parser.add_argument("--auto-eval", action="store_true",
                        help="每个训练完成后自动运行 eval_fewshot_allclass.py")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Runs 输出目录 (默认 runs/) | Runs output directory")
    args = parser.parse_args()

    # 设置全局 auto-eval 标记 | Set global auto-eval flag
    global auto_eval_enabled
    auto_eval_enabled = args.auto_eval

    # ── 验证数据目录 | Validate data root ──
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        print(f"  Tip: on AutoDL, try --data-root /root/autodl-tmp/iSAID_instance_fewshot")
        sys.exit(1)

    # ── 构建实验列表 | Build experiment list ──
    all_experiments = build_experiments(str(data_root))

    # 过滤 Phase | Filter by phase
    if args.phases:
        allowed = set(p.strip() for p in args.phases.split(","))
        experiments = [e for e in all_experiments if e["phase"] in allowed]
        if not experiments:
            print(f"[ERROR] No experiments match phases: {args.phases}")
            print(f"  Available phases: {sorted(set(e['phase'] for e in all_experiments))}")
            sys.exit(1)
    else:
        experiments = all_experiments

    # 从指定 ID 开始 | Start from specific ID
    if args.start_from:
        found = False
        for i, e in enumerate(experiments):
            if e["id"] == args.start_from:
                experiments = experiments[i:]
                found = True
                break
        if not found:
            print(f"[ERROR] Experiment ID not found: {args.start_from}")
            sys.exit(1)

    # ── 打印实验总览 | Print experiment overview ──
    n_train = sum(1 for e in experiments if e["train"])
    n_eval = sum(1 for e in experiments if not e["train"])
    print(f"\n{'=' * 70}")
    print(f"  >> AdaTile-FastSAM — 批量实验运行脚本")
    print(f"  Data: {data_root}")
    print(f"  Total experiments: {len(experiments)} ({n_train} train + {n_eval} eval)")
    print(f"  Phases: {sorted(set(e['phase'] for e in experiments))}")
    print(f"{'=' * 70}")
    print(f"\n  Experiment Plan:")
    print(f"  {'ID':<20s} {'Phase':<8s} {'Type':<6s} {'Description'}")
    print(f"  {'-' * 70}")
    for e in experiments:
        etype = "TRAIN" if e["train"] else "EVAL"
        deps = f" (depends: {e['depends_on']})" if e.get("depends_on") else ""
        print(f"  {e['id']:<20s} {e['phase']:<8s} {etype:<6s} {e['description'][:50]}{deps}")
    print()

    if args.dry_run:
        print("  [DRY RUN] 未执行任何实验. | No experiments executed.")
        return

    # ── 确认 | Confirmation ──
    print(f"  !!  即将运行 {len(experiments)} 个实验. 预估 GPU 时间: "
          f"{len([e for e in experiments if e['train']]) * 25 // 60:.0f}-"
          f"{len([e for e in experiments if e['train']]) * 40 // 60:.0f} 小时")
    print(f"  按 Ctrl+C 可随时中断, 已完成的实验将自动保存.")
    print(f"  继续? (y/n): ", end="", flush=True)
    if input().strip().lower() not in ("y", "yes"):
        print("  已取消 | Cancelled")
        return

    # ── 初始化 | Initialize ──
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    progress_file = runs_dir / ".experiment_progress.json"
    progress = ProgressTracker(progress_file)

    results_file = runs_dir / "experiment_results.csv"
    results_csv = ResultsCSV(results_file)

    # 如果指定 --resume, 打印已完成的实验 | Print completed experiments if resume
    done_ids = progress.get_done_ids()
    if done_ids and not args.force:
        print(f"\n  >> 已完成 {len(done_ids)} 个实验, 将自动跳过:")
        for eid in sorted(done_ids):
            prev = progress.data["completed"][eid]
            print(f"     [OK] {eid}: mIoU={prev.get('mIoU', 'N/A')} "
                  f"({prev.get('timestamp', '?')})")
        print()

    # ── 逐实验运行 | Run experiments one by one ──
    total_start = time.time()
    n_success = 0
    n_failed = 0
    n_skipped = 0

    for i, exp in enumerate(experiments):
        exp_id = exp["id"]

        # 续跑跳过 | Skip if already done in resume mode
        if args.resume and not args.force and progress.is_done(exp_id):
            n_skipped += 1
            continue

        print(f"\n{'━' * 70}")
        print(f"  [{i+1}/{len(experiments)}] {exp_id} ({exp['phase']})")
        print(f"{'━' * 70}")

        try:
            ok = run_single_experiment(
                exp, runs_dir, progress, results_csv,
                force=args.force, retry_count=args.retry,
            )
            if ok:
                n_success += 1
            else:
                n_failed += 1
        except KeyboardInterrupt:
            print(f"\n\n  [STOP]  User interrupted at experiment {exp_id}")
            print(f"  Progress saved to: {progress_file}")
            print(f"  Resume with: --resume --start-from {exp_id}")
            break

    # ── 最终汇总 | Final Summary ──
    total_duration = (time.time() - total_start) / 60.0
    print(f"\n{'=' * 70}")
    print(f"  == Experiment Run Complete!")
    print(f"  Total time: {total_duration:.1f} min ({total_duration/60:.1f} h)")
    print(f"  Success: {n_success} | Failed: {n_failed} | Skipped: {n_skipped}")
    print(f"  Progress: {progress_file}")
    print(f"  Results:  {results_file}")
    print(f"{'=' * 70}")

    if n_failed > 0:
        print(f"\n  [FAIL] Failed experiments:")
        for eid, info in progress.data.get("failed", {}).items():
            print(f"     {eid}: {info.get('error', '?')}")
        print(f"\n  Fix issues and resume with:")
        print(f"  python tools/run_all_experiments.py --data-root {data_root} --resume")

    # ── 打印结果汇总表 | Print results summary table ──
    if n_success > 0:
        print(f"\n  >> Results Summary:")
        print(f"  {'ID':<20s} {'mIoU':<10s} {'Duration':<10s} {'Status'}")
        print(f"  {'-' * 50}")
        for eid, info in sorted(progress.data.get("completed", {}).items()):
            miou = info.get('mIoU', 'N/A')
            dur = info.get('duration_min', '?')
            print(f"  {eid:<20s} {miou:<10s} {dur+' min':<10s} [OK]")


if __name__ == "__main__":
    main()
