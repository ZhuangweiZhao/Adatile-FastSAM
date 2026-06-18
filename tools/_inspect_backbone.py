#!/usr/bin/env python3
"""检查 FastSAM backbone 层结构 | Inspect FastSAM backbone layer structure."""
import sys
sys.path.insert(0, "thirdLibrary/FastSAM")
from fastsam import FastSAM

model = FastSAM("thirdLibrary/FastSAM/weights/FastSAM-x.pt")
seq = model.model.model

print("Layer structure (idx, type, param count):")
print("=" * 70)
total = 0
for i, layer in enumerate(seq):
    n = sum(p.numel() for p in layer.parameters())
    total += n
    name = type(layer).__name__
    print(f"  [{i:2d}] {name:<25s} params={n:>10,}  ({n/1e6:.2f}M)")
print(f"  Total: {total:>10,} ({total/1e6:.2f}M)")

print()
print("C2f blocks detail:")
for i, layer in enumerate(seq):
    if "C2f" in type(layer).__name__:
        n = sum(p.numel() for p in layer.parameters())
        print(f"  [{i:2d}] {type(layer).__name__} params={n:>10,} ({n/1e6:.2f}M):")
        if hasattr(layer, "cv1"):
            print(f"        cv1: {type(layer.cv1).__name__}  "
                  f"ch_in={layer.cv1.conv.in_channels}, ch_out={layer.cv1.conv.out_channels}")
        if hasattr(layer, "cv2"):
            print(f"        cv2: {type(layer.cv2).__name__}  "
                  f"ch_in={layer.cv2.conv.in_channels}, ch_out={layer.cv2.conv.out_channels}")
        if hasattr(layer, "m"):
            m = layer.m
            if hasattr(m, "__len__"):
                print(f"        m: nn.Sequential with {len(m)} bottlenecks")
                for j, b in enumerate(m):
                    bn = sum(p.numel() for p in b.parameters())
                    print(f"          [{j}] {type(b).__name__} params={bn:>10,}")
            else:
                print(f"        m: {type(m).__name__}")
