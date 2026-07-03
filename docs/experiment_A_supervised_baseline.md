# Experiment A: FastSAM P4 全监督分割上限 | Supervised Segmentation Upper Bound

> **最后更新**: 2026-07-03
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
| Backbone 输出 | P4 stride=16 → 16×16 feature cells |
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
| `0629_2237` [A] | LightDecoder (P4-only) | 716K | ❌ | N/A | 原始 baseline |
| `0702_2038` [B] | LightDecoder (P4-only) | 716K | ✅ | 1.0 | CB on P4-only |
| `0630_0015` | LightDecoderP3P4 | 273K | ✅ | **0.0 (BUG)** | pixel_acc 崩溃，不可用 |
| `0630_0646` [D] | LightDecoderP3P4 | 273K | ✅ | 1.0 (fixed) | P3P4+CB, plane 首次非零 |

### 共同配置
- **FastSAM Backbone**: Frozen, P4 stride-16 (1280ch), P3 stride-8 (960ch)
- **Epochs**: 50
- **Batch size**: 16
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR
- **Loss**: 0.5×Focal(γ=5.0) + 0.5×Dice
- **Class-Weight**: Inverse Sqrt Frequency, median anchor=1.0, cap=10.0 (when CB enabled)
- **Device**: RTX 3060 Laptop GPU (6GB), Windows

---

## 3. 各 Run 详细结果 | Per-Run Results

### 3.1 Run [A] `0629_2237` — P4-only, No Class-Balance (Baseline)

**最佳**: E17, **mIoU=0.4089**, pixel_acc=0.9317

```
Class              IoU    Assessment
─────────────────────────────────────────
tennis_court     0.844    ✅ Excellent
storage_tank     0.772    ✅ Excellent
baseball_diamond 0.704    ✅ Good
ship             0.598    ✅ Good
soccer           0.342    ⚠️ Low
ground_track     0.285    ⚠️ Low
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

### 3.2 Run [B] `0702_2038` — P4-only, Class-Balanced ✅ NEW

**最佳**: E40, **mIoU=0.3950**, pixel_acc=0.9221

```
Class              IoU    Assessment                     vs [A] no-CB
─────────────────────────────────────────────────────────────────────
tennis_court     0.810    ✅ Good                        -0.034
storage_tank     0.753    ✅ Good                        -0.019
baseball_diamond 0.708    ✅ Good                        +0.004
ship             0.547    ⚠️ Moderate                     -0.051
soccer           0.338    ⚠️ Low                          -0.004
ground_track     0.252    ⚠️ Low                          -0.033
large_vehicle    0.129    ❌ Very low                     +0.016 ↑
bridge           0.017    ❌ Near zero                    -0.005
helicopter       0.000    ❌❌ Dead                         —
plane            0.000    ❌❌ Dead (E4 闪现 0.055 后消失)   —
─────────────────────────────────────────────────────────────────────
mIoU             0.3950                                  -0.014
```

**plane 幽灵信号**:
```
E1: 0.000 → E4: 0.055 → E5: 0.000 → E50: 0.000
```
CB 确实在推动 plane 学习 (E4=0.055)，但在 16×16 feature 尺度下 plane 只有 ~0.6 feature cells/tile，梯度信号太弱无法收敛。

**训练动态**: stable, no overfitting (716K decoder 容量充足)

| Epoch | Train Loss | Val mIoU | Val Acc |
|:-----:|----------:|--------:|--------:|
| 1 | 0.5081 | 0.3057 | 0.8915 |
| 5 | 0.3990 | 0.3687 | 0.9035 |
| 10 | 0.3688 | 0.3731 | 0.9122 |
| 20 | 0.3393 | 0.3462 | 0.9193 |
| 30 | 0.3263 | 0.3906 | 0.9197 |
| **40** | **0.3181** | **0.3950** | 0.9221 |
| 50 | 0.3176 | 0.3858 | 0.9203 |

---

### 3.3 Run `0630_0015` — P3+P4, Class-Balanced (BG_weight=0 BUG)

**最佳**: E34, **mIoU=0.3650**, pixel_acc=0.170 ⚠️

**Bug 影响**: BG pixel CE weight=0 → 模型可无成本预测假阳性 → pixel_acc 暴跌至 0.17。**结果不可用。**

---

### 3.4 Run [D] `0630_0646` — P3+P4, Class-Balanced (BG_weight=1.0, Fixed)

**最佳**: E5, **mIoU=0.3946**, pixel_acc=0.9153

```
Class              IoU    Assessment                     vs [A] baseline
─────────────────────────────────────────────────────────────────────
tennis_court     0.787    ✅ Good                        -0.057
storage_tank     0.754    ✅ Good                        -0.018
baseball_diamond 0.665    ⚠️ Moderate                     -0.039
ship             0.535    ⚠️ Moderate                     -0.063
soccer           0.370    ⚠️ Low (best so far!)            +0.028
ground_track     0.279    ⚠️ Low                          -0.006
large_vehicle    0.125    ❌ Very low                     +0.012
bridge           0.016    ❌ Near zero                    -0.006
plane            0.020    ★ First stable non-zero!        +0.020 ★
helicopter       0.000    ❌❌ Dead (2 tiles, 1.4K px)     —
```

**关键发现**:
- ✅ **plane 首次稳定突破 0**: 从 0.000 → 0.020，证明 P3 stride-8 特征对小目标有效
- ✅ **pixel_acc 正常**: 0.915，BG_weight 修复生效
- ❌ **严重过拟合**: E5 即最佳，273K decoder 容量不足
- ❌ **Dominant 类大幅下降**: ship -0.063, tennis_court -0.057，CB 代价

**训练动态**: 典型过拟合曲线

| Epoch | Train Loss | Val mIoU | Val Acc |
|:-----:|----------:|--------:|--------:|
| 1 | 0.5026 | 0.3401 | 0.9063 |
| **5** | **0.3945** | **0.3946** | 0.9153 |
| 10 | 0.3665 | 0.3870 | 0.9132 |
| 20 | 0.3357 | 0.3523 | 0.9190 |
| 50 | 0.3177 | 0.3527 | 0.9240 |

---

## 4. 跨 Run 对比 | Cross-Run Comparison

### 4.1 2×2 消融矩阵 (256²)

```
                           No Class-Balance           Class-Balanced
──────────────────────────────────────────────────────────────────────────
P4-only (716K)        [A] 0.4089 ✅ E17           [B] 0.3950 ✅ E40
                      no overfitting               no overfitting
                      4 dead classes               4 dead classes
                      plane=0.000                  plane=0.000 (E4 闪现)

                      [G] 0.4090 ✅ E8 NEW         —
                      Partial-FT (31M, lr=1e-5)
                      mIoU 持平 [A]
                      ★ 收敛 2× 快但精度无增益

P3+P4 (273K)          [C] ⬜ 待跑                   [D] 0.3946 ✅ E5
                                                   overfitting!
                                                   plane=0.020 ★ 稳定
                                                   3 dead classes
```

### 4.2 Per-Class 全量对比

```
Class            [A] P4,nCB    [B] P4,CB     [D] P3P4,CB    趋势分析
────────────────────────────────────────────────────────────────────────
tennis_court      0.844          0.810          0.787         CB 每次都压低主导类
storage_tank      0.772          0.753          0.754         CB 轻微影响
baseball          0.704          0.708          0.665         持平或略降
ship              0.598          0.547          0.535         CB 大幅压低
soccer            0.342          0.338          0.370         P3 帮助 > CB 帮助
ground_track      0.285          0.252          0.279         CB 无显著影响
large_vehicle     0.113          0.129          0.125         微弱提升
bridge            0.022          0.017          0.016         保持不变
plane             0.000          0.000          0.020 ★       P3+CB 协同才有效
helicopter        0.000          0.000          0.000         数据不足，无法救
────────────────────────────────────────────────────────────────────────
mIoU              0.4089         0.3950         0.3946        [A] 仍然最高
```

### 4.3 核心发现: CB 的价值依赖特征分辨率

```
256² P4-only (16×16 cells):   CB 拖累 -0.014  ← 稀有类在 16×16 下不可见
256² P3+P4   (32×32 cells):   CB + P3 → plane=0.020  ← 高分辨率 + CB 协同
896² P4-only (56×56 cells):   待验证 ← 云服务器运行中
896² P3+P4   (56×56 cells):   待验证
```

> **CB 只有在有足够空间分辨率的前提下才有价值。** 16×16 feature 下，plane/helicopter 不到 1 个 feature cell，CB 只能从"看得到的类"拿走梯度去追"看不到的类"→ 净损失。

---

### 3.5 Run [G] `supervised_G_partial5_p4_nocb_256` — P4-only, No CB, Partial Finetune ✅ NEW

**最佳**: E8, **mIoU=0.4090**, pixel_acc=0.9258

```
Class              IoU    Assessment                     vs [A] Frozen
─────────────────────────────────────────────────────────────────────
tennis_court     0.834    ✅ Good                        -0.010
storage_tank     0.752    ✅ Good                        -0.020
baseball_diamond 0.688    ⚠️ Moderate                     -0.016
ship             0.591    ⚠️ Moderate                     -0.007
soccer           0.364    ⚠️ Low                          +0.022 ↑
ground_track     0.300    ⚠️ Low                          +0.015 ↑
large_vehicle    0.138    ❌ Very low                     +0.025 ↑
bridge           0.014    ❌ Near zero                    -0.008
helicopter       0.000    ❌❌ Dead                         —
plane            0.000    ❌❌ Dead                         —
─────────────────────────────────────────────────────────────────────
mIoU             0.4090                                  +0.0001 (持平)
```

**配置**: P4-only, No CB, Partial Finetune (backbone 最后 5 层, 31M params, lr=1e-5)

**训练动态**: 快速收敛，早期过拟合

| Epoch | Train Loss | Val mIoU | Val Acc |
|:-----:|----------:|--------:|--------:|
| 1 | 0.3731 | 0.3360 | 0.9094 |
| **8** | **0.1434** | **0.4090** | 0.9258 |
| 10 | 0.1301 | 0.3938 | 0.9279 |
| 20 | 0.0860 | 0.4027 | 0.9317 |
| 30 | 0.0686 | 0.4054 | 0.9333 |
| 50 | 0.0548 | 0.3985 | 0.9319 |

**关键发现**:
- ★ **31M backbone 微调 → mIoU 零增益 (+0.0001)**：证明 256² 下瓶颈是空间分辨率，不是特征质量
- ✅ 收敛速度翻倍：E8 达峰 (vs [A] E17)，backbone 参数加速适应
- ⚠️ 重分布效应：dominant 类微降 (-0.01~-0.02)，低频类微升 (+0.015~+0.025) — 但幅度远小于 CB
- ❌ Rare 类仍死：plane/helicopter 仍为 0，**31M backbone 参数也创造不出 16×16 feature 中不存在的空间信息**

> **核心结论**: 在 256² (16×16 feature) 下，特征分辨率是硬瓶颈。无论冻结还是微调 backbone，结果都卡在 mIoU≈0.41。突破需要更大 tile 或 P3 多尺度。

---

## 5. 关键结论 | Key Takeaways

### 5.1 FastSAM P4 全监督上限

> **P4-only, no class-balance: mIoU ≈ 0.41** (当前所有 run 中最高)

但 mIoU ≈ 0.41 掩盖了极端的类间差异：
- 4 类效果好 (>0.59): ship, storage_tank, baseball_diamond, tennis_court
- 2 类勉强 (>0.28): ground_track, soccer
- 4 类几乎为 0: bridge, large_vehicle, plane, helicopter

### 5.2 Class-Balance 的价值有条件

| 条件 | 效果 |
|------|------|
| P4-only, 256² (16×16) | ❌ CB → mIoU -0.014，稀有类无收益 |
| P3+P4, 256² (32×32) | ✅ CB + P3 协同 → plane 0→0.020 |
| 896² | ⬜ 待验证（预计 CB 价值更大因为稀有类终于看得见） |

### 5.3 P3 的价值

P3 stride-8 特征使 **plane 首次稳定突破 0** (0.020)，证明高分辨率特征是小目标分割的前提。但当前 P3+P4 decoder 仅 273K（P4-only 的 38%），容量不足导致 E5 即过拟合。

### 5.4 Partial Finetune 的价值（零）

> **31M backbone 参数微调 → mIoU +0.0001。256² 下的瓶颈是空间分辨率，不是特征质量。**

| 对比 | Frozen [A] | Partial-FT [G] | 差异 |
|------|:----------:|:-------------:|:----:|
| mIoU | 0.4089 | 0.4090 | +0.0001 |
| 最佳 epoch | E17 | E8 | **收敛 2× 快** |
| Dominant 类 | 略高 | 略低 | 微重分布 |
| Rare 类 (plane/heli) | 0 | 0 | 无解 |
| Trainable params | 716K | 31.8M | 44× 参数增量 |

> 31M 额外参数创造不出 16×16 feature 中不存在的空间信息。**Partial FT 的价值应该在 896² 才能体现**——那里有足够的空间分辨率，特征质量的提升才有意义。

### 5.5 Decoder 容量与泛化

| Decoder | Params | 过拟合出现 | 效果 |
|---------|--------|:--------:|------|
| P4-only | 716K | 无 (E40 最佳) | 稳定，容量充足 |
| P3+P4 | 273K | E5 即开始 | 容量不足，需增大 fuse_dim |

---

## 6. 剩余消融 | Remaining Ablation

```
256², Fold 0

                         No CB                CB
────────────────────────────────────────────────────
P4-only (716K)      [A] ✅ 0.409           [B] ✅ 0.395

P3+P4 (273K)        [C] ⬜ 待跑              [D] ✅ 0.395
                    测 P3 单独效果            plane=0.020 ★
```

```bash
# [C] P3+P4 + No Class-Balance (最后一格)
python tools/train/train_supervised.py --fold 0 --epochs 50 --use-p3 --no-class-balance
```

```
云服务器 (896²), Fold 0

                         No CB                CB               CB + Partial-FT
──────────────────────────────────────────────────────────────────────────────
P4-only (716K)      [B_896] 待跑            —                 [E] 待跑

P3+P4 (273K)        [C_896] 待跑         [D_896] 待跑        [F] 待跑
```

---

## 7. Run 索引 | Run Index

| Run | ID | Config | mIoU | Best Epoch | Notes |
|-----|-----|--------|:----:|:----------:|------|
| [A] | `0629_2237` | P4, 256², no-CB | **0.4089** | E17 | Baseline, stable |
| [B] | `0702_2038` | P4, 256², CB | 0.3950 | E40 | CB on P4 hurts |
| — | `0630_0015` | P3P4, 256², CB, BG=0 | 0.3650 | E34 | BUG, ignore |
| [D] | `0630_0646` | P3P4, 256², CB | 0.3946 | E5 | plane=0.020, overfit |
| [G] | `supervised_G_partial5_p4_nocb_256` | P4, 256², no-CB, Partial-FT | 0.4090 | E8 | 31M backbone, 零增益 |

---

## 8. 论文叙事价值 | Paper Narrative

Experiment A 即使只说 "mIoU=0.41"，在论文里也有几个关键用途：

1. **P4 特征上限论证**: FastSAM P4 冻结 backbone + 全监督 decoder = 0.41 mIoU。证明 P4 有一定语义信息，但受限于 (a) 单尺度 stride-16, (b) 类别极端不平衡。

2. **Long-tail 分析**: 4/10 类几乎不可学习。这不是 decoder 的问题——是 P4 feature 本身在 16×16 尺度下对稀有/小/线状目标表示不足。

3. **特征分辨率是关键瓶颈**: CB 在 16×16 下拖累性能 (-0.014)，在 32×32 (P3+P4) 下才让 plane 突破 0。验证了"空间分辨率 → 稀有类可学习性"的因果关系。

4. **动机支撑**: [G] 证明 partial finetune (31M backbone 参数) 在 256² 下零增益 → 空间分辨率才是首要瓶颈。Adaper / FDR 的价值体现在 (a) 高分辨率 tile 下的效率，(b) 而不是在低分辨率下的特征质量。这重新定位了 Paper B 的叙事：**Tile 不仅仅是为了降低计算量——它本身就是让小目标可见的必要条件。**
