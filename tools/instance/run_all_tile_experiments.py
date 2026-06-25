#!/usr/bin/env python3
"""
Tile 实验自动运行器 | Automated Tile Experiment Runner.
=========================================================

按顺序运行所有 tile 相关实验，出错跳过，已完成跳过。
Run all tile experiments sequentially, skip on error, skip completed.

用法 | Usage::
    # 全部实验 (Phase 1-3)
    python tools/instance/run_all_tile_experiments.py

    # 仅 Phase 1
    python tools/instance/run_all_tile_experiments.py --phase 1

    # 带自定义 GPU / 输出目录
    python tools/instance/run_all_tile_experiments.py --device cuda:0 --output-dir runs/tile_sweep

    # 干运行 (仅打印命令，不执行)
    python tools/instance/run_all_tile_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ═══════════════════════════════════════════════════════════════════
# 实验定义 | Experiment Definitions
# ═══════════════════════════════════════════════════════════════════

# 公共基础参数 | Common base args
BASE_ARGS = [
    "--src-root", "data/iSAID_processed",
    "--tile", "--tile-size", "896", "--tile-stride", "512",
    "--warmup-epochs", "3",
    "--amp",
]

# 3 类分组 | 3-class group
CLASSES_3 = ["--classes", "1,4,5"]

# 全 15 类 | All 15 classes
CLASSES_ALL = ["--classes", "all"]


def build_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    构建实验列表 | Build experiment list.

    每个实验 dict: {name, cmd: list[str], output_key: str}
    """
    exps: list[dict[str, Any]] = []

    device = args.device
    output_dir = args.output_dir
    cache = str(args.tile_cache_size)
    shots_3 = args.shots_3class
    shots_all = args.shots_all
    epochs_3 = args.epochs_3class
    epochs_all = args.epochs_all
    proto_list = [int(k) for k in args.proto_k_list.split(",")]

    # ══════════════════════════════════════════════════════════
    # Phase 1: 3 类快速扫网 | Phase 1: 3-class quick sweep
    # ══════════════════════════════════════════════════════════

    # P1.0: Non-Parametric Baseline (秒级)
    exps.append({
        "name": "P1.0_nonparam_3class",
        "phase": 1,
        "cmd": [
            sys.executable, str(_PROJECT_ROOT / "tools/instance/eval_c04_full_fewshot.py"),
            *BASE_ARGS,
            "--tile-cache-size", cache,
            "--device", device,
            "--output-dir", f"{output_dir}/p1_nonparam",
            "--shots", shots_3,
            *CLASSES_3,
            "--non-parametric",
        ],
    })

    # Decoder 列表: (suffix, decoder_arg)
    decoders_3class = [
        ("baseline", "--decoder", "baseline"),
    ]

    if not args.skip_heavy:
        decoders_3class += [
            ("film", "--decoder", "film"),
            ("crossattn_k1", "--decoder", "crossattn", "--num-prototypes", "1"),
            ("crossattn_k4", "--decoder", "crossattn", "--num-prototypes", "4"),
            ("contrastive", "--decoder", "contrastive"),
        ]

    for suffix, *dec_args in decoders_3class:
        exps.append({
            "name": f"P1_{suffix}_3class",
            "phase": 1,
            "cmd": [
                sys.executable, str(_PROJECT_ROOT / "tools/instance/eval_c04_full_fewshot.py"),
                *BASE_ARGS,
                "--tile-cache-size", cache,
                "--device", device,
                "--output-dir", f"{output_dir}/p1_{suffix}",
                "--shots", shots_3,
                "--epochs", str(epochs_3),
                "--episodes-per-epoch", str(args.episodes_3class),
                "--eval-episodes", str(args.eval_3class),
                *dec_args,
                *CLASSES_3,
            ],
        })

    if args.phase >= 2:
        # ══════════════════════════════════════════════════════
        # Phase 2: 全 15 类 | Phase 2: Full 15-class
        # ══════════════════════════════════════════════════════

        for suffix, *dec_args in [
            ("baseline", "--decoder", "baseline"),
            ("crossattn_k4", "--decoder", "crossattn", "--num-prototypes", "4"),
        ]:
            exps.append({
                "name": f"P2_{suffix}_15class",
                "phase": 2,
                "cmd": [
                    sys.executable, str(_PROJECT_ROOT / "tools/instance/eval_c04_full_fewshot.py"),
                    *BASE_ARGS,
                    "--tile-cache-size", cache,
                    "--device", device,
                    "--output-dir", f"{output_dir}/p2_{suffix}",
                    "--shots", shots_all,
                    "--epochs", str(epochs_all),
                    "--episodes-per-epoch", str(args.episodes_all),
                    "--eval-episodes", str(args.eval_all),
                    *dec_args,
                    *CLASSES_ALL,
                ],
            })

    if args.phase >= 3:
        # ══════════════════════════════════════════════════════
        # Phase 3: Multi-Prototype Ablation (1-shot, 3-class)
        # ══════════════════════════════════════════════════════

        for K in proto_list:
            exps.append({
                "name": f"P3_proto_k{K}_3class",
                "phase": 3,
                "cmd": [
                    sys.executable, str(_PROJECT_ROOT / "tools/instance/eval_c04_full_fewshot.py"),
                    *BASE_ARGS,
                    "--tile-cache-size", cache,
                    "--device", device,
                    "--output-dir", f"{output_dir}/p3_proto",
                    "--shots", "1",
                    "--epochs", str(epochs_3),
                    "--episodes-per-epoch", str(args.episodes_3class),
                    "--eval-episodes", str(args.eval_3class),
                    "--decoder", "crossattn",
                    "--num-prototypes", str(K),
                    *CLASSES_3,
                ],
            })

    return exps


# ═══════════════════════════════════════════════════════════════════
# 主运行器 | Main Runner
# ═══════════════════════════════════════════════════════════════════


def is_completed(exp: dict, output_dir: str) -> bool:
    """检查实验是否已完成 (results.json 存在且有内容) | Check if experiment is done."""
    cmd = exp["cmd"]
    # 从命令中提取 output-dir
    for i, arg in enumerate(cmd):
        if arg == "--output-dir" and i + 1 < len(cmd):
            results_path = Path(cmd[i + 1]) / "c04_results.json"
            if results_path.exists():
                try:
                    data = json.loads(results_path.read_text())
                    if data.get("results"):
                        return True
                except (json.JSONDecodeError, KeyError):
                    pass
            return False
    return False


def run_one(exp: dict, dry_run: bool = False) -> tuple[str, bool, str]:
    """
    运行单个实验 | Run a single experiment.

    :returns: (name, success, message)
    """
    name = exp["name"]
    cmd = exp["cmd"]

    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    if dry_run:
        print(f"\n{'='*70}")
        print(f"[DRY-RUN] {name}")
        print(f"{'='*70}")
        print(f"  {cmd_str}")
        return (name, True, "dry-run")

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] START: {name}")
    print(f"{'='*70}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=False,  # 实时输出到终端 | stream to terminal
            text=True,
        )
        dt = time.perf_counter() - t0

        if result.returncode == 0:
            msg = f"OK ({dt/60:.1f} min)"
            print(f"\n[DONE] {name}: {msg}")
            return (name, True, msg)
        else:
            msg = f"FAILED with exit code {result.returncode} ({dt/60:.1f} min)"
            print(f"\n[FAIL] {name}: {msg}")
            return (name, False, msg)

    except KeyboardInterrupt:
        print(f"\n[STOP] {name}: User interrupt")
        raise

    except Exception as e:
        dt = time.perf_counter() - t0
        msg = f"EXCEPTION: {type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"\n[ERROR] {name}: {msg}")
        return (name, False, msg)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Tile Experiment Runner | 自动实验运行器"
    )
    # ── 实验范围 | Experiment scope ──
    parser.add_argument("--phase", type=int, default=3,
                       help="运行的 Phase 编号 (1/2/3, 默认全部=3) | Phase number to run")
    parser.add_argument("--skip-heavy", action="store_true",
                       help="跳过 FiLM/Contrastive, 仅跑 baseline+crossattn | Skip heavy decoders")

    # ── 超参数 | Hyperparameters ──
    parser.add_argument("--device", type=str, default="cuda",
                       help="运行设备 | Device (default: cuda)")
    parser.add_argument("--output-dir", type=str, default="runs/tile_sweep",
                       help="输出根目录 | Output root directory")
    parser.add_argument("--tile-cache-size", type=int, default=32,
                       help="原图 LRU 缓存大小 (RTX 3060=32) | Tile cache size")

    # ── 3 类配置 | 3-class config ──
    parser.add_argument("--shots-3class", type=str, default="1,3,5",
                       help="3 类 shot 数 | Shots for 3-class experiments")
    parser.add_argument("--epochs-3class", type=int, default=30,
                       help="3 类 epoch 数 | Epochs for 3-class")
    parser.add_argument("--episodes-3class", type=int, default=100,
                       help="3 类每 epoch episode 数 | Episodes/epoch for 3-class")
    parser.add_argument("--eval-3class", type=int, default=200,
                       help="3 类评估 episode 数 | Eval episodes for 3-class")

    # ── 全 15 类配置 | Full 15-class config ──
    parser.add_argument("--shots-all", type=str, default="1,3,5",
                       help="全类 shot 数 | Shots for full experiments")
    parser.add_argument("--epochs-all", type=int, default=30,
                       help="全类 epoch 数 | Epochs for full-class")
    parser.add_argument("--episodes-all", type=int, default=200,
                       help="全类每 epoch episode 数 | Episodes/epoch for full-class")
    parser.add_argument("--eval-all", type=int, default=200,
                       help="全类评估 episode 数 | Eval episodes for full-class")

    # ── Phase 3 proto 消融 | Phase 3 proto ablation ──
    parser.add_argument("--proto-k-list", type=str, default="1,2,4,8",
                       help="多原型 K 值列表 | Comma-separated K values")

    # ── 运行控制 | Run control ──
    parser.add_argument("--dry-run", action="store_true",
                       help="仅打印命令不执行 | Print commands without running")
    parser.add_argument("--force", action="store_true",
                       help="强制重跑已完成实验 | Force re-run completed experiments")
    parser.add_argument("--no-skip", action="store_true",
                       help="重跑所有，包括已完成 | Re-run all, including completed")

    args = parser.parse_args()

    # ── 构建实验列表 | Build experiment list ──
    experiments = build_experiments(args)

    # ── 过滤已完成 | Filter completed ──
    if not args.no_skip and not args.force:
        skipped = 0
        filtered = []
        for exp in experiments:
            if is_completed(exp, args.output_dir):
                print(f"[SKIP] {exp['name']}: already completed (results.json exists)")
                skipped += 1
            else:
                filtered.append(exp)
        if skipped:
            print(f"\n  Skipped {skipped} already-completed experiment(s). "
                  f"Use --no-skip to re-run.\n")
        experiments = filtered

    if not experiments:
        print("No experiments to run. All done!")
        return

    # ── 打印计划 | Print plan ──
    print(f"\n{'='*70}")
    print(f"  Experiment Plan: {len(experiments)} to run")
    print(f"  Device: {args.device} | Output: {args.output_dir}")
    print(f"  Dry-run: {args.dry_run} | Force: {args.force}")
    print(f"{'='*70}")
    for exp in experiments:
        print(f"  [{exp['phase']}] {exp['name']}")
    print(f"{'='*70}\n")

    # ── 按顺序运行 | Run sequentially ──
    results: list[tuple[str, bool, str]] = []
    t_total_start = time.perf_counter()

    for i, exp in enumerate(experiments):
        print(f"\n{'#'*70}")
        print(f"# [{i+1}/{len(experiments)}] {exp['name']}")
        print(f"{'#'*70}")

        try:
            name, ok, msg = run_one(exp, dry_run=args.dry_run)
            results.append((name, ok, msg))
        except KeyboardInterrupt:
            print("\n\nUser interrupt. Saving partial results...")
            break

    dt_total = time.perf_counter() - t_total_start

    # ── 最终汇总 | Final Summary ──
    print(f"\n\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {len(results)} | Time: {dt_total/60:.1f} min")
    print(f"{'='*70}")

    success_list, fail_list = [], []
    for name, ok, msg in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name:<45s}  {msg}")
        if ok:
            success_list.append(name)
        else:
            fail_list.append(name)

    print(f"\n  Passed:  {len(success_list)}/{len(results)}")
    print(f"  Failed:  {len(fail_list)}/{len(results)}")

    if fail_list:
        print(f"\n  Failed experiments:")
        for f in fail_list:
            print(f"    - {f}")
        print(f"\n  Re-run failed only:")
        print(f"    python tools/instance/run_all_tile_experiments.py "
              f"--phase {args.phase} --no-skip")
        # 保存失败列表供重试 | Save failure list for retry
        fail_path = Path(args.output_dir) / "failed_experiments.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "device": args.device,
                "failed": fail_list,
            }, f, indent=2)
        print(f"  Failure list saved to: {fail_path}")

    if success_list:
        # 保存成功列表 | Save success list
        success_path = Path(args.output_dir) / "completed_experiments.json"
        success_path.parent.mkdir(parents=True, exist_ok=True)
        with open(success_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "device": args.device,
                "completed": success_list,
            }, f, indent=2)

    print(f"\nDone. Total time: {dt_total/60:.1f} min\n")


if __name__ == "__main__":
    main()
