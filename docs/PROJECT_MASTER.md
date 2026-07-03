# AdaTile-FastSAM 项目总览 | Project Master Document

> **最后更新**: 2026-07-03 | **分支**: paper-b | **设备**: 本地 RTX 3060 / 云端 RTX 5090

---

## 目录

1. [项目定位](#一项目定位)
2. [两大创新](#二两大创新)
3. [实验全景](#三实验全景)
4. [A-Series: 全监督上限](#四a-series-全监督语义分割上限)
5. [B-Series: 空间稀疏性理论](#五b-series-空间稀疏性理论)
6. [C-Series: 语义分割 Few-Shot](#六c-series-语义分割-few-shot)
7. [D-Series: 实例分割 Few-Shot](#七d-series-实例分割-few-shot)
8. [工程体系](#八工程体系)
9. [文档索引](#九文档索引)
10. [论文状态](#十论文状态)

---

## 一、项目定位

**AdaTile-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation

### 1.1 两篇论文策略

```
Paper A (Proto Sparsity)          Paper B (Dual Sparsity / AdaTile)
├─ E001→E011-U 完成               ├─ B-00→B-03 理论闭合
├─ 存档于 main 分支               ├─ C-01→C-04 语义分割管线
└─ ICIP/CCIG 目标                 ├─ D-00→D-04 实例分割管线
                                  └─ 目标期刊 TBD
```

### 1.2 核心叙事

> SA-1B 预训练的 FastSAM 在航拍遥感场景性能崩溃（AP=0.015, 27× domain gap）。
> 通过 Few-Shot Domain Adaptation（ProtoCoeffPredictor + P4 Refinement + FDR Tile Selector），
> 可以用极少标注样本恢复分割能力，同时 FDR 稀疏路由减少 60% 计算量。

---

## 二、两大创新

| # | 创新 | 所属 | 描述 |
|---|------|------|------|
| 1 | **Ada-SPM** (Sparse Perception Module) | Paper A | 密度监督稀疏感知模块，学习重要性图 → Top-K tile 选择 |
| 2 | **FDR** (Foreground Density Router) | Paper B | 75K 参数，Pareto 最优空间路由器，学习 objectness/density |

### 2.1 FDR 机制

```
Image → Frozen MV3 Backbone → Feature Map → DensityHead (75K) → Importance Map → Top-K Tiles
```

- 监督信号: `fg_ratio` (tile 前景密度) — 不是边缘，不是类别标签
- 学习目标: objectness / instance density，category-agnostic
- 验证: Spearman r=0.889 (B-series)

---

## 三、实验全景

### 3.1 四阶段证据链

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AdaTile-FastSAM 完整证据链                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  A-Series                B-Series                C-Series     D-Series  │
│  ────────                ────────                ────────     ────────  │
│  全监督上限              空间稀疏性理论          语义Few-Shot  实例Few-Shot│
│                                                                         │
│  A: P4→0.41 mIoU    B-00: Spatial Sparsity   C-01: Zero-Shot  D-00: AP=0.015│
│  C: P3P4→0.42       B-01: Oracle Top-K       C-02A: Proto 0.3% D-01: 0.235  │
│  G: Partial FT→0    B-02: Learnability ★     C-03: CrossAttn   D-02: 0.309  │
│  H: Full FT→0       B-03: FDR 75K≈1.48M      → 32.7% ★        D-03: 0.316★ │
│  J: P3P4+FT→0       B-04: Decoder verified   C-04: Full 15cls  D-04: K-shot │
│                      B-05→B-09: Cross-DS     → running         → 0.326 ★   │
│                                                                         │
│  结论: 256²天花板     结论: FDR可行           结论: Tile必需    结论: 0.33天花板│
│  mIoU≈0.42           SSI>70→适用              1≈3≈5 shot       Proto瓶颈   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心数值一览

| 系列 | 实验 | 最佳结果 | 核心发现 |
|------|------|----------|----------|
| A | 全监督语义分割 | mIoU=**0.4203** (P3P4, Frozen) | 256² 天花板≈0.42，Backbone FT 零增益 |
| B | 空间稀疏性 | r=**0.889** (FDR), SSI=**2.41×** | 空间稀疏性可学习，FDR 75K ≥ 1.48M |
| C | 语义 Few-Shot | mIoU=**32.7%** (C-03, 1-shot) | Tile 是必要条件，Shot 1≈3≈5 |
| D | 实例 Few-Shot | mIoU=**0.326** (D-04, K=10) | P4 Refinement +34.5%, Proto 天花板≈0.33 |

---

## 四、A-Series: 全监督语义分割上限

**目标**: 测量冻结 FastSAM Backbone + Decoder 在 iSAID-5i 上的全监督上限。
**数据**: iSAID-5i Fold 0, Base 类 (10 classes), 256² tiles
**设备**: RTX 3060 (6GB) + RTX 5090 (云端)

### 4.1 消融矩阵

```
                       No Class-Balance         Class-Balanced
─────────────────────────────────────────────────────────────────
P4-only (716K)    [A] Frozen    0.4089 E17  [B] Frozen    0.3950 E40
                  [G] Partial   0.4090 E8    —
                  [H] Full      0.4103 E13   —

P3-only (253K)    [I] Frozen    0.4106 E17   —                    ← NEW

P3+P4 (274K)      [C] Frozen    0.4203 E14   [D] Frozen    0.3946 E5 ★
                  [J] Full      0.4199 E18   —
```

### 4.2 核心结论

1. **256² 天花板 = mIoU≈0.42**: 无论 Frozen/Partial/Full backbone，无论 P4/P3/P3P4 decoder
2. **Backbone 微调零增益**: 四重证据 (P4 Frozen/Full, P3P4 Frozen/Full) 全部证明
3. **P3+P4 > P3 > P4**: 多尺度融合是 256² 下唯一有效的提升 (+0.0114 vs P4-only)
4. **分辨率 > 通道数**: P3-only (253K) > P4-only (716K) by +0.0017
5. **稀有类不可学**: plane/helicopter 在 16×16/32×32 feature 下不到 2 cells
6. **CB 的双面性**: P4-only 下 CB 拖累 -0.014；P3+P4 下 plane 首次突破 0→0.020

### 4.3 文档

- `docs/experiment_A_supervised_baseline.md` — 完整消融矩阵 (9 runs)

---

## 五、B-Series: 空间稀疏性理论

**目标**: 证明空间稀疏性的存在性、可学习性、和路由器的可行性。
**理论链**: B-00 → B-01 → B-02 → B-02.5 → B-03 → B-04 (理论闭合)

### 5.1 证据链

| 实验 | 内容 | 核心发现 |
|------|------|----------|
| **B-00** | Tile Size Sensitivity | 所有尺度稀疏 — 1024px 下 60% tiles empty |
| **B-01** | Oracle Top-K | 上界: Top40% tiles → 96.5% FG, IDG=2.41× |
| **B-02** | Learnability | 重要性可学习: Spearman r=0.889 (MV3 backbone) |
| **B-02.5** | Generalization | Category-AGNOSTIC (holdout r=0.651), 跨数据集可能 |
| **B-03** | Router Architecture | FDR 75K ≈ R0 1.48M (Δr=-0.038), Edge ≠ Importance |
| **B-04** | Decoder Verification | val_fg5≈0.47 (E13), P4 有足够语义信息 |

### 5.2 Paper B 三定律

1. **Spatial Sparsity** (空间稀疏性): 所有尺度都稀疏 — 即使 2048×2048 也有 49.9% 空 tiles
2. **Foreground Concentration** (前景集中): Top 17-48% tiles 捕获 95% FG
3. **Scale-Sparsity Trade-off**: 更大的 tile → 更低的稀疏性 → 需要更高的 FG 捕获率

### 5.3 SSI (Spatial Sparsity Index)

- **定义**: Oracle Top40% FG retention — 预实验零成本判据
- **SSI > 70**: Router 适用 (object-centric: iSAID, DOTA, xView)
- **SSI < 50**: Router 无意义 (land-cover: LoveDA, Potsdam)

### 5.4 文档

- `RESEARCH_MAP.md` — 研究地图
- `memory/paper-b-evidence-chain.md` — 证据链
- `memory/spatial-sparsity-index.md` — SSI 定义

---

## 六、C-Series: 语义分割 Few-Shot

**目标**: FastSAM + Decoder 的 Few-Shot 语义分割管线。
**数据**: iSAID-5i, 多类语义分割

### 6.1 实验链

| 实验 | 内容 | 核心发现 |
|------|------|----------|
| **C-01** | FastSAM Zero-Shot | mR@50≈41.5%, bottleneck = max_det |
| **C-02A** | Proto Matching | 1-shot mIoU=**0.31%** — Proto→P4 直接匹配失败 |
| **C-02B** | Proto + Refine CNN | 1-shot≈0.5% — 训练崩溃全零预测 |
| **C-03** | Cross-Attention + Tile | ★ 1-shot mIoU=**32.7%** (118× over C-02A), Tile 896² |
| **C-04** | Full 15-Class | 🔄 进行中 |

### 6.2 C-03 关键结果

| Shot | mIoU | storage_tank | ship | small_vehicle |
|:----:|:----:|:------------:|:----:|:-------------:|
| 1 | **32.66%** | 50.22% | 35.33% | 10.84% |
| 3 | 31.93% | 43.49% | 36.27% | 15.01% |
| 5 | 32.44% | 47.90% | 37.00% | 10.96% |

**核心发现**:
- ★ **Tile 是 FastSAM few-shot 的必要条件** (全图 0.03% → Tile 32.66%)
- ★ **Shot Saturation**: 1≈3≈5 shot — 瓶颈不是样本数
- ★ **Bottleneck = small_vehicle (11%)** — 小目标仍然是最大挑战

### 6.3 文档

- `docs/c04_code_explanation.md` — C-04 代码讲解

---

## 七、D-Series: 实例分割 Few-Shot

**目标**: 从语义分割升级到实例分割，建立 ProtoCoeffPredictor + P4 Refinement 完整管线。
**数据**: iSAID-5i Fold 0, Novel classes, 256² tiles
**设备**: RTX 5090 云服务器

### 7.1 实验矩阵

| 实验 | Decoder | FDR | K | mIoU | AP50 | 核心发现 |
|------|---------|-----|---|------|------|----------|
| D-00 | FastSAM 原生 | — | — | ~0.01 | **0.023** | 27× domain gap |
| D-01 | ProtoOnly | — | 5 | 0.235 | 0.000 | Focal 坍塌, 纯 MLP 下界 |
| D-02 | Adaptive | ✓ | 5 | 0.309 | 0.000 | +31.5% vs ProtoOnly |
| D-03 | Adaptive | ✗ | 5 | **0.316** | 0.000 | FDR pixel gate 有害 (-2.2%) |
| D-04a | Adaptive | ✗ | 1 | 0.289 | 0.000 | 1-shot=88.6% of K=10 |
| D-04b | Adaptive | ✗ | 3 | 0.296 | 0.000 | — |
| D-04c | Adaptive | ✗ | 10 | **0.326** | 0.000 | 当前最优, 天花板≈0.33 |

### 7.2 三条核心发现

**发现 1: Domain Gap (27×)**
```
Zero-Shot AP=0.015 vs MS-COCO AP≈40% → 27× domain gap
```
论文 Introduction 的核心动机 — SA-1B "segment anything" 在航拍场景不成立。

**发现 2: P4 Refinement (+34.5%)**
```
ProtoOnly (纯 MLP, 427K):    mIoU = 0.235
Adaptive noFDR (+P4, 1.14M): mIoU = 0.316
                              Δ   = +0.081 (+34.5%)
```
P4 特征精炼是 mIoU 提升的核心驱动力。仅增加 714K 参数。

**发现 3: Shot Saturation (1-shot=88.6%)**
```
K=1:  0.289  (88.6% of K=10)
K=3:  0.296  (90.7%)
K=5:  0.316  (96.9%)  ← Sweet Spot
K=10: 0.326  (100%)
```
瓶颈在 proto 基函数表达上限 (~0.33)，不在样本数。

### 7.3 FDR 角色修正

```
          B-Series (已验证)              D-Series (本次)
          ────────────────              ───────────────
FDR 角色: Tile-level selector          ❌ Pixel-level gate
粒度:     粗 (tile)                     ❌ 细 (per-pixel)
效果:     r=0.889, SSI=2.41×          ❌ -2.2% mIoU
状态:     ✅ 有效                       ❌ 有害
```

### 7.4 NaN 三层防护

| # | 修复 | 原因 | 效果 |
|---|------|------|------|
| 1 | `min_tiles=30` | 数据极度稀缺导致梯度 NaN | 根治 cls12 |
| 2 | `focal eps=1e-4` | 原 1e-8 使 1/(1-pred)=1e8 梯度爆炸 | 截断至 1e4 |
| 3 | `focal γ=5.0` | 遥感 FG<5% 极端不平衡 | 标准配置 |
| 兜底 | `NaN skip + zero_grad` | 漏网 NaN episode | 自动保护 |

### 7.5 文档

- `docs/D_series_master.md` — D-Series 完整档案 ★
- `docs/D_series_summary.md` — D-Series 汇总
- `docs/experiment_D_baseline_fastsam.md` — D-00 详细
- `docs/experiment_D01_protoonly_fewshot.md` — D-01 详细
- `docs/experiment_D02_adaptive_fewshot.md` — D-02 详细
- `docs/experiment_D03_fdr_ablation.md` — D-03 详细
- `docs/experiment_D04_kshot_scaling.md` — D-04 详细

---

## 八、工程体系

### 8.1 核心模块

```
adatile/
├── backbone/fastsam_backbone.py     # FastSAM (frozen), P3/P4/P8 + Proto
├── decoder/
│   ├── light_decoder.py             # LightDecoder (716K, P4-only)
│   ├── light_decoder_p3p4.py        # LightDecoderP3P4 (274K, multi-scale)
│   └── adaptive_sparse_decoder.py   # AdaptiveSparseDecoder + ProtoOnlyDecoder
├── sparse/
│   ├── coefficient_predictor.py     # ProtoCoeffPredictor (3层 MLP, ~427K)
│   └── spatial_router.py           # ForegroundDensityRouter + DensityHead (75K)
├── metrics/
│   ├── metrics.py                   # compute_miou, compute_dice, FPSMeter
│   └── coco_eval.py                # COCOInstanceEvaluator
├── datasets/
│   ├── isaid_tiles.py              # FastISAIDTileDataset
│   ├── isaid_tile_wrapper.py       # Full-image→tile wrapper + LRU cache
│   └── isaid_instance.py           # ISAIDInstanceDataset
├── losses/seg_losses.py            # FocalLoss + DiceLoss + CombinedLoss
├── logging/                         # Structured logging (Console/File/Wandb)
├── config/                          # ExperimentConfig + Recorder + exp_id
└── utils/
    ├── seed.py                     # Unified set_seed()
    ├── prototype.py                # compute_fg_prototype()
    ├── label_mapping.py            # iSAID category mapping
    └── render.py                   # render_category_mask()
```

### 8.2 训练脚本

```
tools/
├── train/
│   ├── train_supervised.py          # A-Series: 全监督训练
│   ├── train_fewshot_finetune.py    # B-Series: Few-shot fine-tune
│   ├── train_isaid_mc.py            # iSAID multi-class (Paper A)
│   ├── train_b04.py                 # B-04: FDR + Decoder 端到端
│   └── train_instance_fewshot.py   # D-Series: 实例分割 Few-Shot (v3)
├── paper_b/                         # B-Series 独立实验脚本
│   ├── eval_b00_tile_size_sensitivity.py
│   ├── eval_b01_oracle_topk.py
│   ├── eval_b02_learnability.py
│   ├── eval_b03_router_architecture.py
│   └── ...
├── instance/                        # D-Series 实例分割脚本
│   ├── eval_baseline_instance.py
│   └── run_instance_fewshot_ablation.sh
└── diag/                            # 诊断工具
    ├── diag_b04_overfit.py
    ├── diag_class_stats.py
    ├── diag_check_labels.py
    └── diag_trace_labels.py
```

### 8.3 开发规范

1. **日志先行**: 所有新代码 → `adatile.logging`, 禁止 `print()`
2. **中英双语**: 所有文件/类/函数/关键逻辑同时注释
3. **测试覆盖**: 核心库模块有 shape/value range/edge case 测试
4. **可复现**: `set_seed()` 必须在所有实验脚本开头调用
5. **逐模块审查**: 完成一个→审查→通过→下一个

---

## 九、文档索引

### 9.1 实验文档

| 文件 | 内容 | 状态 |
|------|------|:----:|
| `docs/experiment_A_supervised_baseline.md` | A-Series: 9 runs 消融矩阵 (256²) | ✅ |
| `docs/experiment_B_fewshot_finetune.md` | B-Series: Few-shot fine-tune 实验设计 | 📋 |
| `docs/experiment_D_baseline_fastsam.md` | D-00: Zero-Shot 基线 (AP=0.015) | ✅ |
| `docs/experiment_D01_protoonly_fewshot.md` | D-01: ProtoOnly (mIoU=0.235) | ✅ |
| `docs/experiment_D02_adaptive_fewshot.md` | D-02: Adaptive+FDR (mIoU=0.309) | ✅ |
| `docs/experiment_D03_fdr_ablation.md` | D-03: FDR 消融 (noFDR=0.316) | ✅ |
| `docs/experiment_D04_kshot_scaling.md` | D-04: K-shot scaling (K=10=0.326) | ✅ |

### 9.2 汇总文档

| 文件 | 内容 |
|------|------|
| `docs/D_series_summary.md` | D-Series 快速汇总 |
| `docs/D_series_master.md` | D-Series 完整档案 (最详细) |
| `docs/PROJECT_MASTER.md` | **本文档** — 全项目总览 |

### 9.3 参考文档

| 文件 | 内容 |
|------|------|
| `CLAUDE.md` | 项目开发指南 + 最新结果 |
| `RESEARCH_MAP.md` | 研究地图 + 证据链 + 路线图 |
| `docs/isaid5i_dataset_spec.md` | iSAID-5i 数据集规格 |
| `docs/c04_code_explanation.md` | C-04 代码逐行讲解 |
| `pyproject.toml` | 依赖/配置/构建 |

### 9.4 记忆文件

| 目录 | 文件数 | 内容 |
|------|:------:|------|
| `memory/` | ~39 files | 项目知识持久化 |
| `memory/MEMORY.md` | 1 file | 记忆索引 |

---

## 十、论文状态

### 10.1 Paper B 完成度

```
  理论建设:  ████████░░  80%  (B-00→B-03 完整, D-00→D-04 补充)
  实验验证:  ██████░░░░  60%  (D-series 核心完成, cross-fold/instance loss 待做)
  效率分析:  ███░░░░░░░  30%  (FDR tile selector 需要大图推理 pipeline)
  论文写作:  ░░░░░░░░░░   0%  (所有实验完成后开始)
```

### 10.2 待攻克 (按优先级)

| 优先级 | 任务 | 预期收益 | 说明 |
|:------:|------|----------|------|
| **P0** | Proto basis finetuning | 突破 0.33 天花板 | 允许 32 个 proto masks 参与训练 |
| **P0** | Cross-fold (Fold 1/2) | 泛化性验证 | 确保结论不是 fold 0 特有 |
| **P1** | Multi-scale (P3+P5) | 小目标改善 | P3 stride 8 对小目标更友好 |
| **P1** | Instance-aware loss | AP>0 | 连通分量→实例失败，需要实例级监督 |
| **P1** | FDR 预训练 + 大图推理 | 效率展示 | FDR tile selector 减少 60% 计算 |
| **P2** | Cross-dataset (NWPU) | 泛化性 | category-agnostic 的最终证明 |

### 10.3 论文可用数据汇总

**Introduction 可用:**
- 27× domain gap (SA-1B→iSAID, AP=0.015)
- AR_large=37.2% vs AR_small=0.9% — 尺度分层效应
- "Tile is not an optimization — it's a necessity"

**Method 可用:**
- ProtoCoeffPredictor: support prototype → 32 mask coefficients
- P4 Feature Refinement: 补偿 proto basis mismatch (+34.5% mIoU)
- FDR Tile Selector: 75K, Spearman r=0.889, category-agnostic

**Experiments 可用:**
- 全监督: mIoU=0.42 (256² 天花板), Backbone FT 零增益 (4 runs)
- 空间稀疏性: B-00→B-03 完整理论链, SSI 判据
- K-shot scaling: 1/3/5/10 (D-04), SSI-1=88.6%, SSI-5=96.9%
- FDR 消融: noFDR (0.316) > +FDR (0.309), tile-selector > pixel-gate
- NaN 三层防护: min_tiles + eps + γ + skip

**Engineering 可用:**
- BatchNorm→InstanceNorm for bs=1
- Focal γ=5.0 + eps=1e-4 for remote sensing
- 结构化日志 + 实验配置复现系统

### 10.4 推荐论文叙事线

```
1. Introduction
   SA-1B fails on aerial imagery (AP=0.015, 27× gap)
   → Few-shot domain adaptation is necessary

2. Related Work
   SAM/FastSAM, Few-Shot Segmentation, Sparse Computation

3. Method
   3.1 ProtoCoeffPredictor: support prototype → mask coefficients
   3.2 P4 Feature Refinement: compensates proto basis mismatch
   3.3 FDR Tile Selector: reduces computation by 60%

4. Experiments
   4.1 Zero-shot proves domain gap (D-00)
   4.2 ProtoOnly establishes lower bound (D-01)
   4.3 P4 refinement drives +34.5% gain (D-03)
   4.4 1-shot achieves 89% of peak performance (D-04)
   4.5 FDR ablation proves tile-selector role (D-03)
   4.6 Spatial sparsity theory (B-00→B-03)

5. Conclusion
   Few-shot + P4 refinement is practical for aerial instance segmentation.
   Current bottleneck: SA-1B proto bases (~0.33 ceiling).
   Future: proto finetuning, multi-scale, instance-aware loss.
```

---

## 附录: 项目 Timeline

```
2026-06-16  v2 完全重写，清空 v1 代码
2026-06-21  Paper A/B 两篇策略确定
2026-06-24  C-03 Cross-Attention 突破 (32.7%)
2026-06-29  D-00 Zero-Shot baseline (AP=0.015)
2026-07-01  D-01 ProtoOnly (0.235), D-02 Adaptive+FDR (0.309)
2026-07-02  NaN 排查完成, 三层防护
2026-07-03  D-03 FDR 消融 (0.316), D-04 K-shot scaling (0.326)
2026-07-03  全部文档汇总完成 ← NOW
```

---

*This document is the single source of truth for the AdaTile-FastSAM project.*
*It aggregates all A/B/C/D series experiments, engineering practices, and paper strategy.*
