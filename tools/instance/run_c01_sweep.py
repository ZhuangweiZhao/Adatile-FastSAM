#!/usr/bin/env python3
"""
C-01 Sweep: max_det × conf — 验证 Recall 瓶颈是 proposal 数量还是目标尺度.
================================================================================
"""
import subprocess, json
from pathlib import Path

BASE = "python tools/instance/eval_fastsam_zero_shot.py --n-images 20 --split val --device cuda"

CONFIGS = [
    ("m100_c10",  "--max-det 100 --conf 0.10"),
    ("m300_c10",  "--max-det 300 --conf 0.10"),
    ("m500_c10",  "--max-det 500 --conf 0.10"),
    ("m100_c05",  "--max-det 100 --conf 0.05"),
    ("m300_c05",  "--max-det 300 --conf 0.05"),
    ("m500_c05",  "--max-det 500 --conf 0.05"),
]

results = {}
for name, args in CONFIGS:
    out_dir = f"runs/c01_sweep/{name}"
    jf = Path(out_dir) / "fastsam_zero_shot.json"

    # 跳过已完成的 | skip if already done
    if jf.exists():
        d = json.loads(jf.read_text())
        results[name] = d
        print(f"\nSKIP {name}: already done, mR@50={d['mRecall50']*100:.1f}%")
        continue

    cmd = f"{BASE} {args} --output-dir {out_dir}"
    timeout = 7200 if "500" in name else 3600  # max_det=500 更慢
    print(f"\n{'='*60}")
    print(f"Running: {name} ({args})  timeout={timeout}s")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, shell=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s — partial results may exist")

    if jf.exists():
        d = json.loads(jf.read_text())
        results[name] = d
        print(f"  mR@50={d['mRecall50']*100:.1f}%  "
              f"smallV={d['per_class_Recall50'].get('1',0)*100:.1f}%  "
              f"largeV={d['per_class_Recall50'].get('2',0)*100:.1f}%")
    else:
        print(f"  FAILED for {name}")

# ── 汇总表 ──
print("\n" + "=" * 90)
print("SWEEP RESULTS: max_det × conf")
print("=" * 90)
h = f"{'Config':<18} {'mR@50':>7} {'mR@75':>7} {'AP@50':>7} {'smallV':>7} {'largeV':>7} {'bball':>7} {'pool':>7} {'ship':>7}"
print(h)
print("-" * 90)
for name in [c[0] for c in CONFIGS]:
    r = results.get(name)
    if not r: continue
    sv = r['per_class_Recall50'].get('1', 0) * 100
    lv = r['per_class_Recall50'].get('2', 0) * 100
    bb = r['per_class_Recall50'].get('12', 0) * 100
    sp = r['per_class_Recall50'].get('10', 0) * 100
    sh = r['per_class_Recall50'].get('5', 0) * 100
    print(f"{name:<18} {r['mRecall50']*100:>6.1f}% {r['mRecall75']*100:>6.1f}% "
          f"{r['approx_AP50']*100:>6.1f}% {sv:>6.1f}% {lv:>6.1f}% {bb:>6.1f}% {sp:>6.1f}% {sh:>6.1f}%")

# ── 关键对比 ──
print("\n── 关键问题 ──")
if results:
    r100 = results.get("m100_c10", {})
    r500 = results.get("m500_c10", {})
    if r100 and r500:
        delta = (r500.get('mRecall50', 0) - r100.get('mRecall50', 0)) * 100
        print(f"   conf=0.1: max_det 100→500, mR@50 变化 = {delta:+.1f}%")
        if delta > 5:
            print(f"   → 瓶颈主要在 proposal 数量")
        else:
            print(f"   → 瓶颈主要在目标尺度，增加 proposal 帮助有限")
