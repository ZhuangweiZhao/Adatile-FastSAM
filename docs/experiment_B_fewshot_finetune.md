# Experiment B: Few-Shot Fine-Tuning — 实验文档

## 概述 | Overview

**目标 | Goal**: 验证 Frozen FastSAM + LightDecoder 能否通过少量样本快速适配新类别（Novel classes）。

**路线 | Approach**: Few-shot Fine-tuning（少样本微调），而非经典 Few-Shot Segmentation (FSS)。

**范式 | Paradigm**:
1. **Pre-train**: Base 类全量数据训练 → 获得通用分割能力
2. **Fine-tune**: Novel 类 K-shot 样本微调 → 快速适配新类别
3. **Evaluate**: 直接推理，无需 Support 图像

**与经典 FSS 的关键区别 | Key Difference from Classic FSS**:
- FSS: Support → Prototype → Query Matching → 每次推理需 Support
- Ours: K-shot Fine-tune → 直接推理 → 推理不需 Support

---

## 实验配置 | Experiment Configuration

### 数据集 | Dataset

| 属性 | 值 |
|------|-----|
| 数据集 | iSAID-5i (15 类遥感目标) |
| 图像尺寸 | 256×256 tiles |
| Fold 0 Base | ship(1), storage_tank(2), baseball_diamond(3), tennis_court(4), ground_track_field(6), bridge(7), large_vehicle(8), helicopter(10), soccer_ball_field(13), plane(14) |
| Fold 0 Novel | small_vehicle(9), harbor(15), swimming_pool(11), basketball_court(5), roundabout(12) |
| 预训练数据 | 所有 Base 类 train tiles（全量） |
| 微调数据 | Novel 类 K-shot train tiles（每类最多 K 个） |
| 测试数据 | 所有 val tiles（全类评估） |

### 预训练 Checkpoint | Pre-trained Checkpoint

| Checkpoint | Decoder | Base mIoU (Fold 0) | 参数量 |
|------------|---------|---------------------|--------|
| `supervised_F0_P4_Frz_nocb_256` | P4-only | 0.4089 | 716K |
| `supervised_F0_P3_Frz_nocb_256` | P3-only | 0.4106 | 253K |
| `supervised_F0_P3P4_Frz_nocb_256` | P3+P4 | 0.4203 | 274K |

### 微调配置 | Fine-Tuning Config

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 (Decoder) | 1e-4 (预训练的 1/10) |
| 学习率 (Backbone) | 1e-5 (如果微调) |
| 调度器 | CosineAnnealingLR (eta_min=1e-6) |
| Loss | 0.5×Focal(γ=5) + 0.5×Dice |
| 类别平衡 | Inverse sqrt frequency, cap=10.0 |
| Batch Size | 16 |
| 微调 Epochs | 20 |
| 随机种子 | 42 (训练), 42/123/456 (K-shot 采样) |

---

## 实验矩阵 | Experiment Matrix

### B-01: K-Shot Scaling（核心实验 | Core）

**问题**: Novel 类 mIoU 如何随 K-shot 数量变化？

```
Fold 0, 256², P4 Decoder, Decoder-only

K=0 (Zero-Shot)   K=1        K=3        K=5        K=10       Full (上限)
──────────────────────────────────────────────────────────────────────────
?                  ?          ?          ?          ?          0.4089 (Base)
```

**命令**:
```bash
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \
    --k-shot 1,3,5,10 --k-shot-seed 42,123,456 \
    --finetune-epochs 20 --batch-size 16
```

### B-02: Decoder Architecture × K-Shot

**问题**: 不同解码器架构在 few-shot 场景下的表现差异？

```
K=5, Fold 0, 256², 3 seeds

Decoder    Params    Zero-Shot    K=5 Fine-Tune    Δ
─────────────────────────────────────────────────────
P4-only    716K      ?            ?                ?
P3-only    253K      ?            ?                ?
P3+P4      274K      ?            ?                ?
```

**命令**:
```bash
# P3 Decoder
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P3_Frz_nocb_256/best_model.pt \
    --k-shot 5 --k-shot-seed 42,123,456 --finetune-epochs 20

# P3P4 Decoder
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt \
    --k-shot 5 --k-shot-seed 42,123,456 --finetune-epochs 20
```

### B-03: Fine-tuning Strategy

**问题**: 微调 Decoder 还是同时微调 Backbone？

```
K=5, Fold 0, P3P4 Decoder, 3 seeds

Strategy               Trainable    Novel mIoU    Base mIoU (遗忘?)
─────────────────────────────────────────────────────────────────
Decoder-only           274K         ?             ?
Decoder + Partial BB   ~31M         ?             ?
```

**命令**:
```bash
# Decoder-only (默认)
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt \
    --k-shot 5 --k-shot-seed 42,123,456 --finetune-epochs 20

# Decoder + Partial Backbone (last 5 layers)
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt \
    --k-shot 5 --k-shot-seed 42,123,456 --finetune-epochs 20 \
    --partial-finetune 5
```

### B-04: Resolution × Few-Shot（可选 | Optional）

**问题**: 更高分辨率 (896²) 是否能改善 few-shot 微调效果？

```
K=5, Fold 0, P3P4 Decoder

Resolution    Novel mIoU
────────────────────────
256²          ?
896²          ?
```

---

## 运行方式 | How to Run

### 本地 | Local

```bash
# Zero-shot 评估（仅评估预训练模型在 Novel 类上的表现）
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \
    --eval-zero-shot-only

# 单次 K=5 微调测试
python tools/train/train_fewshot_finetune.py \
    --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \
    --k-shot 5 --k-shot-seed 42 --finetune-epochs 5

# 一键消融矩阵（预览）
bash tools/train/run_fewshot_ablation.sh --dry

# 一键消融矩阵（运行，仅 B-01）
bash tools/train/run_fewshot_ablation.sh --only "B01"
```

### 云端 | Cloud

```bash
# 一键运行全部实验（并行 nohup）
bash tools/train/run_fewshot_ablation.sh --cloud

# 仅 B-01 K-shot scaling
bash tools/train/run_fewshot_ablation.sh --cloud --only "B01"

# 使用 Fold 1 预训练模型
bash tools/train/run_fewshot_ablation.sh --cloud --fold 1 \
    --pretrained runs/supervised_F1_P4_Frz_nocb_256/best_model.pt
```

---

## 结果记录 | Results

### B-01: K-Shot Scaling

| K | Seed | Tiles | Zero-Shot Novel | Fine-Tune Novel | Δ | Base | All |
|---|------|-------|-----------------|-----------------|---|------|-----|
| 1 | 42   | ?     | ?               | ?               | ? | ?    | ?   |
| 1 | 123  | ?     | ?               | ?               | ? | ?    | ?   |
| 1 | 456  | ?     | ?               | ?               | ? | ?    | ?   |
| 3 | 42   | ?     | ?               | ?               | ? | ?    | ?   |
| 3 | 123  | ?     | ?               | ?               | ? | ?    | ?   |
| 3 | 456  | ?     | ?               | ?               | ? | ?    | ?   |
| 5 | 42   | ?     | ?               | ?               | ? | ?    | ?   |
| 5 | 123  | ?     | ?               | ?               | ? | ?    | ?   |
| 5 | 456  | ?     | ?               | ?               | ? | ?    | ?   |
| 10| 42   | ?     | ?               | ?               | ? | ?    | ?   |
| 10| 123  | ?     | ?               | ?               | ? | ?    | ?   |
| 10| 456  | ?     | ?               | ?               | ? | ?    | ?   |

### B-02: Decoder Architecture

| Decoder | K | Seed | Zero-Shot Novel | Fine-Tune Novel | Δ | Base | All |
|---------|---|------|-----------------|-----------------|---|------|-----|
| P4      | 5 | 42   | ?               | ?               | ? | ?    | ?   |
| P3      | 5 | 42   | ?               | ?               | ? | ?    | ?   |
| P3P4    | 5 | 42   | ?               | ?               | ? | ?    | ?   |

### B-03: Fine-tuning Strategy

| Strategy | K | Seed | Novel | Base (遗忘?) | All |
|----------|---|------|-------|-------------|-----|
| Decoder-only | 5 | 42 | ? | ? | ? |
| Partial BB 5 | 5 | 42 | ? | ? | ? |

### B-04: Resolution × Few-Shot

| Resolution | K | Seed | Zero-Shot Novel | Fine-Tune Novel | Δ |
|------------|---|------|-----------------|-----------------|---|
| 256²       | 5 | 42   | ?               | ?               | ? |
| 896²       | 5 | 42   | ?               | ?               | ? |

---

## 分析维度 | Analysis Dimensions

1. **K-shot 饱和点**: K 多大时 Novel mIoU 不再显著提升？
2. **Decoder 架构效率**: 哪种 decoder 在少样本下最有效？（参数越少越好）
3. **类别遗忘**: 微调 Novel 类是否导致 Base 类性能下降？
4. **种子敏感性**: 不同 K-shot 采样种子对结果的影响有多大？
5. **稀有类分析**: 哪些 Novel 类最容易/最难通过 few-shot 适配？
6. **与全监督对比**: K-shot fine-tune 的 Novel mIoU 能达到全监督的多少百分比？

---

## 预期结果 | Expected Results

基于 Experiment A 的发现:
- **Zero-shot Novel mIoU 应接近 0**: 模型从未见过 Novel 类，无法预测
- **K=5 应显著提升**: 少量样本即可快速适配
- **P3P4 应优于 P4-only**: 多尺度 fusion 在少样本下可能更有优势
- **Decoder-only 应优于 Partial BB**: 少样本下 backbone 微调可能过拟合
- **K-shot 饱和效应**: 预计 K=5-10 附近达到瓶颈
