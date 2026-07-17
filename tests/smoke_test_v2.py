"""
Quick smoke test for v2 modules: FrequencyFeatureEnhancer + ConvLoRA.
"""
import torch
import torch.nn as nn

# Test FrequencyFeatureEnhancer
from adatile.frequency import FrequencyFeatureEnhancer, MultiScaleFrequencyEnhancer

print("=== FrequencyFeatureEnhancer ===")
fe = FrequencyFeatureEnhancer(channels=64, n_bands=4, reduction=4)
for size in [32, 50, 64]:
    x = torch.randn(2, 64, size, size)
    y = fe(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f"  {size}x{size}: {x.shape} -> {y.shape} OK")

print("\n=== MultiScaleFrequencyEnhancer ===")
mfe = MultiScaleFrequencyEnhancer(p2_channels=160, p3_channels=960, p4_channels=1280, reduction=4)
p2 = torch.randn(2, 160, 50, 50)
p3 = torch.randn(2, 960, 25, 25)
p4 = torch.randn(2, 1280, 13, 13)
result = mfe(p2=p2, p3=p3, p4=p4)
for k, v in result.items():
    print(f"  {k}: {v.shape} OK")
assert set(result.keys()) == {"p2", "p3", "p4"}

# Test partial input (only p3)
result_partial = mfe(p3=p3)
print(f"  partial (p3 only): {list(result_partial.keys())} OK")

print("\n=== ConvLoRA ===")
from adatile.backbone.fastsam_backbone import ConvLoRA, _inject_conv_lora, _collect_lora_modules

conv = nn.Conv2d(64, 128, 3, padding=1)
lora_conv = ConvLoRA(conv, rank=4, alpha=1.0)
x = torch.randn(2, 64, 32, 32)
y = lora_conv(x)
assert y.shape == (2, 128, 32, 32)
print(f"  Forward: {x.shape} -> {y.shape} OK, LoRA params={lora_conv.lora_params}")
# Verify original conv is frozen
assert not any(p.requires_grad for p in conv.parameters())
# Verify LoRA params are trainable
assert lora_conv.lora_down.weight.requires_grad
assert lora_conv.lora_up.weight.requires_grad
print("  Freeze/requires_grad: correct")

print("\n=== _inject_conv_lora ===")
class SimpleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 1)  # 1x1 -> should be skipped
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

block = SimpleBlock()
n = _inject_conv_lora(block, rank=4)
print(f"  Added {n} LoRA params")
print(f"  conv1 -> {type(block.conv1).__name__} (expect ConvLoRA)")
print(f"  conv2 -> {type(block.conv2).__name__} (expect Conv2d)")
print(f"  conv3 -> {type(block.conv3).__name__} (expect ConvLoRA)")
assert isinstance(block.conv1, ConvLoRA), f"conv1 is {type(block.conv1)}"
assert isinstance(block.conv2, nn.Conv2d), f"conv2 is {type(block.conv2)}"
assert isinstance(block.conv3, ConvLoRA), f"conv3 is {type(block.conv3)}"

# Forward pass
y = block(torch.randn(2, 32, 16, 16))
print(f"  Forward: (2,32,16,16) -> {tuple(y.shape)} OK")
assert y.shape == (2, 32, 16, 16)

# Collect LoRA modules
lora_mods = _collect_lora_modules(block)
print(f"  _collect_lora_modules: {len(lora_mods)} modules")
assert len(lora_mods) == 2

print("\n✅ All smoke tests passed!")
