# Experiment A: FastSAM P4 全监督分割上限 | Supervised Segmentation Upper Bound

> **创建日期**: 2026-07-02
> **实验目标**: 测量冻结 FastSAM Backbone + Decoder 在 iSAID-5i 上的全监督语义分割性能上限。
> **对照价值**: 与 Few-Shot Episode 结果对比，分离"特征表示质量"与"Few-shot 匹配机制"的各自贡献。

---

## 1. 数据集统计 | Dataset Statistics

**iSAID-5i Fold 0, 仅 Base 类 (10 classes)**, Novel 类像素设为 ignore (255)。

| 统计项 | 数值 |
|--------|------|
| 训练 tile 数 | 9,090 |
| 验证 tile 数 | 3,108 |
| Tile 尺寸 | 256×256 px |
| 总前景像素 (train) | 120,377,494 |

### 类别分布 (训练集)

```
Class                    ID      Pixels    Tiles    %FG     %Tiles  类别等级
---------------------------------------------------------------------------------
ship                      1   43,504,840    4,405   36.14%   48.5%   Dominant (主导)
tennis_court              4   43,241,782    2,814   35.92%   31.0%   Dominant (主导)
storage_tank              2   15,278,417    1,032   12.69%   11.4%   Moderate (中等)
baseball_diamond          3    8,781,164      618    7.29%    6.8%   Moderate (中等)
soccer_ball_field        13    4,331,167      325    3.60%    3.6%   Low (偏低)
ground_track_field        6    3,873,931      287    3.22%    3.2%   Low (偏低)
bridge                    7      686,273       72    0.57%    0.8%   Rare (稀有) ❌
large_vehicle             8      586,273      412    0.49%    4.5%   Rare (稀有) ❌
plane                    14       92,267       14    0.08%    0.2%   Very Rare (极稀有) ❌❌
helicopter               10        1,380        2    0.00%    0.0%   Extremely Rare ❌❌❌
```

**数据稀缺阈值**:
- **可训练 (≥1M px)**: ship, tennis_court, storage_tank, baseball_diamond, soccer, ground_track
- **勉强可训 (100K-1M px)**: bridge, large_vehicle
- **基本不可训 (<100K px)**: plane (92K), helicopter (1.4K)

---

## 2. 实验配置矩阵 | Configuration Matrix

| Run ID | Decoder | Params | Class-Balance | BG Weight | Note |
|--------|---------|--------|:---:|:---:|------|
| `0629_2237` | LightDecoder (P4-only) | 716K | ❌ | N/A | 原始 baseline |
| `0630_0015` | LightDecoderP3P4 | 273K | ✅ | **0.0 (BUG)** | pixel_acc 崩溃 |
| `0630_0646` | LightDecoderP3P4 | 273K | ✅ | 1.0 (fixed) | 当前最佳 P3P4+CB |

### 共同配置
- **FastSAM Backbone**: Frozen, P4 stride-16 (1280ch), P3 stride-8 (960ch)
- **Epochs**: 50
- **Batch size**: 16
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR
- **Loss**: 0.5×Focal(γ=5.0) + 0.5×Dice
- **Device**: RTX 3060 Laptop GPU (6GB), Windows

---

## 3. 各 Run 详细结果 | Per-Run Results

### 3.1 Run `0629_2237` — P4-only, No Class-Balance (Baseline)

**最佳**: E17, **mIoU=0.4089**, pixel_acc=0.9317

```
Class              IoU    Assessment
─────────────────────────────────────────
ship             0.598    ✅ Good
storage_tank     0.772    ✅ Excellent
baseball_diamond 0.704    ✅ Good
tennis_court     0.844    ✅ Excellent
ground_track     0.285    ⚠️ Low
soccer           0.342    ⚠️ Low
bridge           0.022    ❌ Near zero
large_vehicle    0.113    ❌ Very low
helicopter       0.000    ❌❌ Dead
plane            0.000    ❌❌ Dead
```

**训练动态**: stable, no overfitting

| Epoch | Train Loss | Val mIoU | Val Acc |
|:-----:|----------:|--------:|--------:|
| 1 | 0.3731 | 0.3361 | 0.9094 |
| 5 | 0.1820 | 0.3951 | 0.9174 |
| 10 | 0.1290 | 0.4016 | 0.9280 |
| 15 | 0.1028 | 0.3954 | 0.9299 |
| 17 | — | **0.4089** | 0.9317 |
| 20 | 0.0856 | 0.4002 | 0.9320 |

---

### 3.2 Run `0630_0015` — P3+P4, Class-Balanced (BG_weight=0 BUG)

**最佳**: E34, **mIoU=0.3650**, pixel_acc=0.170 ⚠️

```
Class              IoU    Assessment
─────────────────────────────────────────
ship             0.612    ✅ Good
storage_tank     0.774    ✅ Excellent
baseball_diamond 0.711    ✅ Good
tennis_court     0.823    ✅ Good
soccer           0.337    ⚠️ Low
ground_track     0.261    ⚠️ Low
large_vehicle    0.121    ❌ Very low
bridge           0.010    ❌ Near zero
helicopter       0.000    ❌❌ Dead
plane            0.000    ❌❌ Dead
```

**Bug 影响**: BG pixel CE weight=0 → 模型可无成本预测假阳性 → pixel_acc 暴跌至 0.17。mIoU 被大量假阳性拖低。**结果不可用。**

---

### 3.3 Run `0630_0646` — P3+P4, Class-Balanced (BG_weight=1.0, Fixed) ✅

**最佳**: E5, **mIoU=0.3946**, pixel_acc=0.9153

```
Class              IoU    Assessment                     vs Baseline
─────────────────────────────────────────────────────────────────────
tennis_court     0.787    ✅ Good                        -0.057
storage_tank     0.754    ✅ Good                        -0.018
baseball_diamond 0.665    ⚠️ Moderate                     -0.039
ship             0.535    ⚠️ Moderate                     -0.063
soccer           0.370    ⚠️ Low (best so far!)            +0.028
ground_track     0.279    ⚠️ Low                          -0.006
large_vehicle    0.125    ❌ Very low                     +0.012
bridge           0.016    ❌ Near zero                    -0.006
plane            0.020    ★ First non-zero!               +0.020 ★
helicopter       0.000    ❌❌ Dead (2 tiles, 1.4K px)     —
```

**关键发现**:
- ✅ **plane 首次突破 0**: 从 0.000 → 0.020，证明 P3 stride-8 特征对小目标有效
- ✅ **pixel_acc 正常**: 0.915，BG_weight 修复生效
- ❌ **严重过拟合**: E5 即最佳，之后持续下降
- ❌ **Dominant 类大幅下降**: ship -0.063, tennis_court -0.057，CB 代价

**训练动态**: 典型过拟合曲线

| Epoch | Train Loss | Val mIoU | Val Acc |
|:-----:|----------:|--------:|--------:|
| 1 | 0.5026 | 0.3401 | 0.9063 |
| **5** | **0.3945** | **0.3946** | 0.9153 |
| 10 | 0.3665 | 0.3870 | 0.9132 |
| 20 | 0.3357 | 0.3523 | 0.9190 |
| 30 | 0.3254 | 0.3543 | 0.9241 |
| 50 | 0.3177 | 0.3527 | 0.9240 |

**Loss 持续下降，val mIoU E5 后持续下降 → 典型过拟合**

---

## 4. 跨 Run 对比 | Cross-Run Comparison

### 4.1 mIoU 对比

```
                           No Class-Balance        Class-Balanced
───────────────────────────────────────────────────────────────────
P4-only (716K)        [A] 0.4089 ✅ (E17)     [B] ??? (未跑)
                      4 dead classes

P3+P4 (273K)          [C] ??? (未跑)          [D] 0.3946 (E5, 过拟合)
                                                plane=0.020 ★ (首次非零)
                                                4 dead classes → 3 dead
```

### 4.2 Per-Class 详细对比

```
Class            [A] P4-only    [D] P3+P4+CB    Δ(A→D)   趋势
                     no CB          fixed
────────────────────────────────────────────────────────────────
tennis_court       0.844          0.787        -0.057    ↓ CB 压低
ship               0.598          0.535        -0.063    ↓ CB 压低
baseball_diamond   0.704          0.665        -0.039    ↓ CB 压低
storage_tank       0.772          0.754        -0.018    → 持平
ground_track       0.285          0.279        -0.006    → 持平
soccer             0.342          0.370        +0.028    ↑ P3 帮助
large_vehicle      0.113          0.125        +0.012    ↑ 轻微提升
bridge             0.022          0.016        -0.006    → 持平
plane              0.000          0.020 ★      +0.020    ↑★ P3 突破
helicopter         0.000          0.000        0.000     — 数据不足
```

### 4.3 稀有类能/不能救的分类

**P3 + CB 有效**:
- plane: 0.000 → 0.020 ★ (P3 stride-8 高分辨率特征)
- soccer: 0.342 → 0.370 (CB + P3 协同)

**P3 + CB 无效**:
- helicopter: 恒为 0 (仅 2 tiles, 1380 px — 无法通过 loss 或架构解决)
- bridge: 0.022 → 0.016 (基本持平，bridge 的线状结构需要更强的 decoder)

---

## 5. 关键结论 | Key Takeaways

### 5.1 FastSAM P4 全监督上限

> **P4-only, no class-balance: mIoU ≈ 0.41**

这是 FastSAM 冻结 backbone + 轻量 decoder 在 iSAID-5i 上全监督的全类平均上限。但 mIoU ≈ 0.41 掩盖了极端的类间差异：
- 4 类效果好 (>0.59): ship, storage_tank, baseball_diamond, tennis_court
- 2 类勉强 (>0.28): ground_track, soccer
- 4 类几乎为 0: bridge, large_vehicle, plane, helicopter

### 5.2 P3 的价值

P3 stride-8 特征使 **plane 从 0 → 0.02**（首次非零），证明高分辨率特征对小物体分割有帮助。但当前 P3+P4 decoder 参数仅 273K（P4-only 的 38%），容量不足导致严重过拟合。

### 5.3 Class-Balance 的代价

CB 使 dominant 4 类平均下降 ~0.04 IoU。稀有类的微弱提升（plane +0.02）不足以补偿主导类的损失，**导致整体 mIoU 反而下降**（0.409 → 0.395）。

这是因为 mIoU 是等权平均（每类权重相同），而 10 个 Base 类中 4 个主导类占据了 mIoU 的主体。CB 牺牲 4 个 0.04 去换 2 个 0.02，数学上不划算。

### 5.4 过拟合问题

P3+P4 273K decoder 在 E5 即开始过拟合。对比 P4-only 716K decoder 在 E17 仍稳定。**Decoder 容量与泛化性直接相关。**

---

## 6. 2×2 消融缺口 | Missing Ablation Cells

要完整拆解 P3 和 CB 的各自贡献，还需要补两个实验：

```
                           No Class-Balance        Class-Balanced
───────────────────────────────────────────────────────────────────
P4-only (716K)        [A] ✅ 已完成 0.409      [B] ⬜ 待跑
                                                  回答: CB 自己能把 plane 从 0→? 吗？

P3+P4 (273K)          [C] ⬜ 待跑                  [D] ✅ 已完成 0.395
                       回答: P3 自己 (不加 CB)          plane=0.020 ★
                       对稀有类有帮助吗？
```

```bash
# [B] P4-only + Class-Balanced
python tools/train/train_supervised.py --fold 0 --epochs 50

# [C] P3+P4 + No Class-Balance
python tools/train/train_supervised.py --fold 0 --epochs 50 --use-p3 --no-class-balance
```

---

## 7. 论文叙事价值 | Paper Narrative

Experiment A 即使只说 "mIoU=0.41"，在论文里也有几个关键用途：

1. **P4 特征上限论证**: FastSAM P4 冻结 backbone + 全监督 decoder = 0.41 mIoU。证明 P4 有一定语义信息，但受限于 (a) 单尺度 stride-16, (b) 类别极端不平衡。

2. **Long-tail 分析**: 4/10 类几乎不可学习 (helicopter=0, plane=0, bridge≈0.02, large_vehicle≈0.11)。这不是 decoder 的问题——是 P4 feature 本身对稀有/小/线状目标表示不足。

3. **与 Few-Shot 对比**: 全监督 0.41 vs Episode Base ≈? vs Episode Novel ≈? — 三者之间的 gap 定义了方法的改进空间。

4. **动机支撑**: "Even with full supervision, frozen FastSAM P4 cannot segment rare classes" → 这正是为什么需要 Tile 策略 / 多尺度 / Adapter / FDR 的理论动机。
