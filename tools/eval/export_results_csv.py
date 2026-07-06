#!/usr/bin/env python3
"""
将 K-shot 实验结果导出为 CSV | Export K-shot experiment results to CSV.
======================================================================

从 runs/ 目录收集所有 Baseline 结果，生成两个 CSV:
Collects all Baseline results from runs/ directory, generates two CSVs:
    1. per_class.csv   — Per-class FT-IoU / ZS-IoU / Delta for K=1,3,5
    2. training.csv    — Training curves (loss, iou per epoch)

用法 | Usage::
    python tools/eval/export_results_csv.py

输出 | Output:
    runs/baseline_results/per_class.csv
    runs/baseline_results/training_curves.csv
"""

from __future__ import annotations

import sys
import json
import csv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

CATEGORY_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

# ── 找到所有 eval 结果目录 | Find all eval result dirs ──
def find_eval_dirs(runs_root: Path) -> dict[int, Path]:
    """扫描 runs/ 找 eval_fewshot_baseline_K* 目录 | Scan runs/ for eval dirs."""
    result: dict[int, Path] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "eval_fewshot_baseline_K" in name:
            # 提取 K 值 | Extract K value
            for k in [1, 3, 5, 10]:
                if f"_K{k}_" in name or f"_K{k}I" in name or name.endswith(f"_K{k}"):
                    result[k] = d
                    break
    return result


def find_train_dirs(runs_root: Path) -> dict[int, Path]:
    """扫描 runs/ 找 baseline_K*_ImageShot 目录 | Scan runs/ for train dirs."""
    result: dict[int, Path] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "baseline_K" in name and "ImageShot" in name:
            for k in [1, 3, 5, 10]:
                if f"_K{k}_" in name or f"_K{k}I" in name:
                    result[k] = d
                    break
    return result


def main():
    runs_root = _PROJECT_ROOT / "runs"
    out_dir = runs_root / "baseline_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dirs = find_eval_dirs(runs_root)
    train_dirs = find_train_dirs(runs_root)
    k_values = sorted(eval_dirs.keys())

    print(f"Found eval dirs: {[(k, str(d.name)) for k, d in eval_dirs.items()]}")
    print(f"Found train dirs: {[(k, str(d.name)) for k, d in train_dirs.items()]}")

    # ═══════════════════════════════════════════════════════════════
    # CSV 1: Per-Class Results | 逐类结果
    # ═══════════════════════════════════════════════════════════════
    # 收集所有数据 | Collect all data
    all_data: dict[int, dict] = {}  # cls_id → {k_shot: {ft, zs, delta, n}}
    overall: dict[int, dict] = {}    # k_shot → {ft, zs, delta}

    for k in k_values:
        edir = eval_dirs[k]
        comp_path = edir / "comparison.json"
        if not comp_path.exists():
            print(f"  [WARN] No comparison.json in {edir.name}")
            continue
        with open(comp_path, encoding="utf-8") as f:
            data = json.load(f)
        overall[k] = {
            "ft": data.get("ft_overall_mean_iou", 0),
            "zs": data.get("zs_overall_mean_iou", 0),
            "delta": data.get("delta_overall", 0),
        }
        for cls_str, info in data.get("per_class", {}).items():
            cls_id = int(cls_str)
            if cls_id not in all_data:
                all_data[cls_id] = {}
            all_data[cls_id][k] = {
                "ft": info.get("ft_mean_iou", 0),
                "zs": info.get("zs_mean_iou", 0),
                "delta": info.get("delta", 0),
                "n": info.get("n", 0),
            }

    # 写入 CSV | Write CSV
    per_class_path = out_dir / "per_class.csv"
    with open(per_class_path, "w", newline="", encoding="utf-8-sig") as f:
        # 构建表头 | Build header
        header = ["Class ID", "Class Name"]
        for k in k_values:
            header += [f"K={k} FT-IoU", f"K={k} ZS-IoU", f"K={k} Delta", f"K={k} N"]
        header += ["Best K", "Best FT-IoU", "ZS beats FT?"]
        writer = csv.writer(f)
        writer.writerow(header)

        for cls_id in sorted(all_data.keys()):
            name = CATEGORY_NAMES.get(cls_id, f"cls_{cls_id}")
            row = [cls_id, name]
            best_k = None
            best_ft = 0
            for k in k_values:
                info = all_data[cls_id].get(k, {"ft": "", "zs": "", "delta": "", "n": ""})
                row += [info.get("ft", ""), info.get("zs", ""), info.get("delta", ""), info.get("n", "")]
                if isinstance(info.get("ft"), (int, float)) and info["ft"] > best_ft:
                    best_ft = info["ft"]
                    best_k = k
            row.append(best_k if best_k is not None else "")
            row.append(round(best_ft, 4) if best_ft else "")
            # ZS beats FT? 看 K=5 (最大 K)
            k5_info = all_data[cls_id].get(max(k_values), {})
            zs_beats = "YES" if k5_info.get("delta", 0) < 0 else ""
            row.append(zs_beats)
            writer.writerow(row)

        # 空行 | Blank row
        writer.writerow([])

        # Overall 行 | Overall row
        overall_row = ["", "OVERALL"]
        for k in k_values:
            ov = overall.get(k, {})
            overall_row += [ov.get("ft", ""), ov.get("zs", ""), ov.get("delta", ""), ""]
        # K=1→K=3 delta, K=1→K=5 delta
        overall_row += ["", "", ""]
        writer.writerow(overall_row)

        # Delta between K values
        if len(k_values) >= 2:
            delta_row = ["", "Δ(K=1→K=last)"]
            for k in k_values:
                delta_row += ["", "", "", ""]
            k1_ft = overall.get(k_values[0], {}).get("ft", 0)
            klast_ft = overall.get(k_values[-1], {}).get("ft", 0)
            delta_row += [f"{klast_ft - k1_ft:+.4f}", "", ""]
            writer.writerow(delta_row)

    print(f"  ✅ Per-class CSV → {per_class_path}")

    # ═══════════════════════════════════════════════════════════════
    # CSV 2: Training Curves | 训练曲线
    # ═══════════════════════════════════════════════════════════════
    training_path = out_dir / "training_curves.csv"
    with open(training_path, "w", newline="", encoding="utf-8-sig") as f:
        # 收集所有 K 的训练日志 | Collect training logs for all K
        all_logs: dict[int, list[dict]] = {}
        max_epochs = 0
        for k in sorted(train_dirs.keys()):
            log_path = train_dirs[k] / "train_log.json"
            if not log_path.exists():
                print(f"  [WARN] No train_log.json in {train_dirs[k].name}")
                continue
            with open(log_path, encoding="utf-8") as lf:
                log_data = json.load(lf)
            entries = log_data.get("entries", [])
            all_logs[k] = entries
            max_epochs = max(max_epochs, len(entries))

        # 构建表头 | Build header
        header = ["Epoch"]
        for k in sorted(all_logs.keys()):
            header += [f"K={k} Train Loss", f"K={k} Train IoU", f"K={k} Val IoU"]
        writer = csv.writer(f)
        writer.writerow(header)

        # 按 epoch 写入 | Write by epoch
        for ep_idx in range(max_epochs):
            row = [ep_idx + 1]
            for k in sorted(all_logs.keys()):
                entries = all_logs[k]
                if ep_idx < len(entries):
                    e = entries[ep_idx]
                    row += [
                        round(e.get("train_loss", 0), 6),
                        round(e.get("train_iou", 0), 6),
                        round(e.get("val_iou", 0), 6),
                    ]
                else:
                    row += ["", "", ""]
            writer.writerow(row)

        # 底部: Best Val IoU summary
        writer.writerow([])
        best_row = ["", "BEST"]
        for k in sorted(all_logs.keys()):
            entries = all_logs[k]
            best_val = max((e.get("val_iou", 0) for e in entries), default=0)
            best_row += ["", "", round(best_val, 6)]
        writer.writerow(best_row)

    print(f"  ✅ Training curves CSV → {training_path}")
    print(f"\n  All done! Output: {out_dir}/")


if __name__ == "__main__":
    main()
