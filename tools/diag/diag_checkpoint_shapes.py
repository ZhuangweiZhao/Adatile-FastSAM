#!/usr/bin/env python3
"""检查 checkpoint 中的 key 和 shape，诊断与当前代码的差异."""
import torch, sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
    "runs/fewshot_f0_k5_0628_1935/decoder_p3p4film_5shot_best.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

print(f"Checkpoint: {ckpt_path}")
print(f"Keys: {len(ckpt)}\n")

# 逐层打印
p3_params, p4_params, other_params = [], [], []
for k, v in sorted(ckpt.items()):
    shape_str = str(list(v.shape))
    print(f"  {k:<50s} {shape_str}")
    if "proj_p3" in k or "p3" in k.lower():
        p3_params.append((k, shape_str))
    elif "proj_p4" in k or "p4" in k.lower():
        p4_params.append((k, shape_str))
    else:
        other_params.append((k, shape_str))

# 分析
print(f"\n=== P3-related parameters ===")
for k, s in p3_params:
    print(f"  {k}: {s}")

print(f"\n=== Dimension Analysis ===")
for k, v in ckpt.items():
    if "proj_p3.0.weight" == k:
        out_ch, in_ch = v.shape[:2]
        print(f"  proj_p3 input channels:  {in_ch}")
        print(f"  Expected (local code):   640")
        if in_ch == 960:
            print(f"  → Server used 960-dim P3 (possibly concat P3:640 + P4↑:320?)")
        elif in_ch == 640:
            print(f"  → Matches local code, something else is wrong")
    if "proj_p4.0.weight" == k:
        out_ch, in_ch = v.shape[:2]
        print(f"  proj_p4 input channels:  {in_ch}")
        print(f"  Expected (local code):   1280")
    if "gate_mlp.0.weight" == k:
        out_f, in_f = v.shape
        print(f"  gate_mlp proto_dim:      {in_f}")
        print(f"  Expected (local code):   1280")
