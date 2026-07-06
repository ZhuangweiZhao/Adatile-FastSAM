#!/usr/bin/env python3
"""
统一消融实验 | Unified Ablation Runner.
=========================================

一键运行完整消融 + 输出论文级对比报告.
One-click full ablation → paper-grade comparison report.

实验矩阵 | Experiment Matrix:
  E1+E2: Zero-Shot + Bbox Prompt   (eval_fastsam_prompted.py)
  E3:    Three-Stage Diagnosis       (eval_fastsam_prompted.py --diagnose)
  E4:    FT vs ZS                    (eval_fewshot_allclass.py, 需checkpoint)

用法 | Usage::

    # 仅 ZS (无微调)
    python tools/eval/run_ablation.py --per-class 10 --device cuda

    # ZS + FT 对比
    python tools/eval/run_ablation.py --per-class 10 --device cuda \
        --ft-checkpoint runs/train_fewshot_allcls_K3_*/best_model.pt --k-shot 3

输出 | Output:
    runs/ablation_{timestamp}/
    ├── ablation_report.txt    # 文本报告
    └── zs_eval/               # ZS 详细数据
"""

import sys, argparse, json, random, subprocess
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from adatile.utils.seed import set_seed


def run_cmd(cmd: str, desc: str = "") -> int:
    """Run shell command, print output."""
    print(f"\n  [{desc}]")
    print(f"  $ {cmd}")
    return subprocess.call(cmd, shell=True, cwd=str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Unified Ablation Runner")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ft-checkpoint", type=str, default=None)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    ts = datetime.now().strftime("%m%d_%H%M")
    if args.output_dir is None:
        args.output_dir = f"runs/ablation_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  AdaTile-FastSAM Ablation")
    print(f"  Per-class: {args.per_class} | Seed: {args.seed}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    report = [
        f"AdaTile-FastSAM Ablation Report",
        f"==============================",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Per-class: {args.per_class} | Seed: {args.seed}",
    ]

    # ── E1+E2+E3: Zero-shot evaluation + diagnosis ──
    zs_out = out_dir / "zs_eval"
    cmd = (f'python tools/eval/eval_fastsam_prompted.py '
           f'--all-classes --per-class {args.per_class} '
           f'--mode bbox --diagnose '
           f'--device {args.device} --seed {args.seed} '
           f'--output-dir "{zs_out}"')
    ret = run_cmd(cmd, "E1+E2+E3: Zero-Shot + Bbox Prompt + Diagnosis")

    zs_stats = zs_out / "stats.json"
    if zs_stats.exists():
        with open(zs_stats) as f:
            zs = json.load(f)
        s = zs.get("summary", {}).get("bbox", {})
        diag = zs.get("summary", {}).get("_diagnosis_per_class", {})

        report.append(f"\n## E1+E2 — Zero-Shot Bbox Prompt")
        report.append(f"Overall Prompt IoU:  {s.get('mean_iou', 'N/A')}")
        report.append(f"AP@50:               {s.get('ap50', 'N/A')}")
        report.append(f"AP@75:               {s.get('ap75', 'N/A')}")
        report.append(f"Boundary IoU:        {s.get('mean_boundary_iou', 'N/A')}")

        if diag:
            ceilings = [d["mean_ceiling_iou"] for d in diag.values()]
            missings = [d["missing_rate"] for d in diag.values()]
            report.append(f"\n## E3 — Three-Stage Diagnosis")
            report.append(f"Oracle Ceiling IoU:  {np.mean(ceilings):.4f}")
            report.append(f"Mean Missing Rate:   {np.mean(missings) * 100:.1f}%")
            report.append(f"Selection Gap:       {np.mean(ceilings) - s.get('mean_iou', 0):+.4f}")

            report.append(f"\n### Per-Class")
            report.append(f"{'Class':<22s} {'N':>5s} {'Prompt':>8s} {'Ceiling':>8s} {'Miss%':>7s}")
            report.append(f"{'-' * 55}")
            for c in sorted(diag.keys(), key=lambda x: int(x)):
                d = diag[c]
                report.append(f"{c}:{d['name']:<19s} {d['n']:>5d} "
                              f"{d['mean_prompt_iou']:>8.4f} {d['mean_ceiling_iou']:>8.4f} "
                              f"{d['missing_rate']:>7.1%}")

    # ── E4: FT vs ZS ──
    if args.ft_checkpoint:
        ft_out = out_dir / "ft_eval"
        cmd = (f'python tools/eval/eval_fewshot_allclass.py '
               f'--checkpoint "{args.ft_checkpoint}" '
               f'--k-shot {args.k_shot} --per-class {args.per_class} '
               f'--device {args.device} --seed {args.seed} '
               f'--output-dir "{ft_out}"')
        ret2 = run_cmd(cmd, "E4: Fine-tuned vs Zero-Shot")

        ft_stats = ft_out / "comparison.json"
        if ft_stats.exists():
            with open(ft_stats) as f:
                ft = json.load(f)
            report.append(f"\n## E4 — Few-Shot (K={ft.get('k_shot', '?')})")
            report.append(f"FT Overall IoU:  {ft.get('ft_overall_mean_iou', 'N/A')}")
            report.append(f"ZS Overall IoU:  {ft.get('zs_overall_mean_iou', 'N/A')}")
            report.append(f"Delta (FT - ZS): {ft.get('delta_overall', 0):+.4f}")

            pc = ft.get("per_class", {})
            if pc:
                report.append(f"\n{'Class':<22s} {'FT':>8s} {'ZS':>8s} {'Delta':>8s}")
                report.append(f"{'-' * 50}")
                for c in sorted(pc.keys(), key=lambda x: int(x)):
                    d = pc[c]
                    report.append(f"{c}:{d['name']:<19s} {d['ft_mean_iou']:>8.4f} "
                                  f"{d['zs_mean_iou']:>8.4f} {d['delta']:>+8.4f}")
    else:
        report.append(f"\n## E4 — Skipped (no --ft-checkpoint)")

    # ── Save ──
    report_path = out_dir / "ablation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n{'=' * 60}")
    print(f"  Report: {report_path}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
