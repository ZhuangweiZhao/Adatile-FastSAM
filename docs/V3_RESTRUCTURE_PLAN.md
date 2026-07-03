# AdaTile-FastSAM v3 重构方案

> **日期**: 2026-07-03
> **触发**: 论文 Task 混杂、评价体系不一致、模块重复、叙事线混乱
> **决策**: 合并 Paper A+B → 一篇论文，统一 Task/Protocol/Eval/Module

---

## 零、四项关键决策 (已确认)

| # | 决策 | 旧 | 新 |
|---|------|-----|-----|
| 1 | Paper 策略 | Paper A + Paper B 两篇 | **合并为一篇** |
| 2 | Core Task | FSS / Instance / Sparse 混杂 | **High-Resolution Few-Shot Instance Segmentation** |
| 3 | 数据协议 | iSAID-5i (FSS protocol) | **自定义 Instance Few-Shot Split** |
| 4 | 模块命名 | FDR + Ada-SPM 两个名字 | **统一为 SPM (Sparse Perception Module)** |

---

## 一、论文定义 (v3)

### 1.1 One-Sentence Summary

> **FastSAM fails on aerial imagery (AP=0.015, 27× domain gap). We enable it for high-resolution aerial instance segmentation via Few-Shot Fine-tuning with Adaptive Sparse Computation — achieving competitive performance while processing only 40% of image tiles.**

### 1.2 Core Elements

```
Task:        High-Resolution Few-Shot Instance Segmentation
Backbone:    FastSAM (YOLOv8-S, SA-1B pretrained, frozen)
Training:    Few-shot Fine-tuning (pre-train on Base → fine-tune on Novel)
Innovation:  Adaptive Sparse Computation
             ├── SPM (Sparse Perception Module)      ← 统一 FDR + Ada-SPM
             │   ├── Importance Head (DensityHead)    ← 预测 tile 重要性
             │   └── Dynamic Tile Router              ← Top-K tile 选择
             └── Adaptive Decoder
                 ├── ProtoCoeffPredictor              ← support proto → 32 coeffs
                 ├── Feature Refinement (P4)           ← 空间细节补偿
                 └── Mask Head                        ← per-pixel instance logit
Data:        iSAID Instance Few-Shot Split (自定义)
             ├── Base: 10 classes (预训练)
             ├── Novel: 5 classes (K-shot fine-tune)
             ├── 3-Fold cross-validation
             └── 896×896 tiles, COCO JSON annotations
Evaluation:
  主指标:    COCO AP, AP50, AP75, AP_small, AP_medium, AP_large
  效率:      FLOPs, FPS, GPU Memory (peak)
  辅助:      mIoU, per-class IoU, AR (Supplement)
```

### 1.3 论文叙事线

```
1. Introduction
   SA-1B "segment anything" fails on aerial imagery
   → AP=0.015, AR_large=37.2% vs AR_small=0.9%
   → 27× domain gap quantified
   → We propose: Few-Shot Fine-tuning + Sparse Tile Routing

2. Related Work
   SAM/FastSAM, Few-Shot Segmentation, Sparse Computation, Aerial Imagery

3. Method (3 modules)
   3.1 Sparse Perception Module (SPM)
       Importance Head → tile importance map
       Dynamic Tile Router → Top-K selection
   3.2 Adaptive Decoder
       ProtoCoeffPredictor → 32-d mask coefficients
       P4 Feature Refinement → spatial detail compensation
       Mask Head → instance logit
   3.3 Few-Shot Fine-tuning Protocol
       Base pre-training → Novel K-shot fine-tune → direct inference

4. Experiments
   4.1 Zero-Shot Baseline: FastSAM on iSAID (AP=0.015)
   4.2 Spatial Sparsity Analysis: 60% tiles empty, Top-40% → 96.5% FG
   4.3 Few-Shot Scaling: K=1/3/5/10 → SSI-1=88.6%, K=5 sweet spot
   4.4 SPM Ablation: w/ vs w/o tile routing, FLOPs vs AP tradeoff
   4.5 Decoder Ablation: ProtoCoeff only vs +P4 Refinement (+34.5%)
   4.6 Cross-Dataset: NWPU-VHR-10 (category-agnostic verification)

5. Conclusion
   Few-shot + Sparse Routing enables practical aerial instance segmentation.
   Current bottleneck: SA-1B proto bases (~0.33 mIoU ceiling).
```

---

## 二、模块重命名映射

```
旧名称 (v2, 混乱)                新名称 (v3, 统一)
═══════════════════════════════════════════════════════
Ada-SPM (Paper A)            ┐
FDR / ForegroundDensityRouter├──→ SPM (Sparse Perception Module)
DensityHead                  │    ├── ImportanceHead
EdgeHead                     │    ├── DensityHead (保留内部类名)
TinyCNNRouter                ┘    └── TileRouter

ProtoCoeffPredictor          ──→ AdaptiveDecoder
P4 Feature Refinement             ├── ProtoCoeffPredictor (保留)
FDR Pixel Gate (废弃!)            ├── FeatureRefinement
Mask Head                         └── MaskHead
```

### 2.1 文件重命名

```
旧文件                                    新文件
─────────────────────────────────────────────────────────────────
adatile/sparse/spatial_router.py      →  adatile/sparse/spm.py
  class ForegroundDensityRouter       →  class SparsePerceptionModule
  class DensityHead                   →  class ImportanceHead
  class EdgeHead                      →  (保留, ablation only)
  class TinyCNNRouter                 →  (保留, lower-bound baseline)

adatile/decoder/adaptive_sparse_decoder.py
  class AdaptiveSparseDecoder         →  class AdaptiveDecoder
  class ProtoOnlyDecoder              →  (保留, ablation baseline)
  class ProtoCoeffPredictor           →  (保留, 子模块)
```

---

## 三、数据协议重构

### 3.1 弃用

```
❌ iSAID-5i (FSS protocol)
   - 10 Base / 5 Novel class split is fine
   - But: semantic masks (not instance), Episode-based evaluation
   - Name "5i" implies FSS benchmark → confusing for instance segmentation
```

### 3.2 新建

```
✅ iSAID Instance Few-Shot Split

目录结构:
data/iSAID_instance_fewshot/
├── images/                     # 896×896 tiles (PNG)
├── annotations/                # COCO JSON per split
│   ├── instances_base.json     # Base class instances (10 classes)
│   ├── instances_novel.json    # Novel class instances (5 classes)
│   └── instances_val.json      # Validation (all classes)
├── folds/
│   ├── fold_0.json             # Base/Novel split definition
│   ├── fold_1.json
│   └── fold_2.json
└── stats/
    └── class_distribution.json # Per-class instance count + area stats

Split 定义:
  Base (10):  ship, storage_tank, baseball_diamond, tennis_court,
              ground_track_field, bridge, large_vehicle, helicopter,
              soccer_ball_field, plane
  Novel (5):  small_vehicle, harbor, swimming_pool,
              basketball_court, roundabout

Tile 规格:
  Size:   896×896 px
  Stride: 512 px (overlap for boundary objects)
  Min FG: >5% foreground pixels
  格式:     COCO JSON (instance segmentation format)
```

### 3.3 需要新建的代码

```python
# tools/data/prep_isaid_instance.py
"""
Step 1: 从原始 iSAID COCO 生成 instance masks
Step 2: 按 896×896 切 tile
Step 3: 生成 COCO JSON annotations (per tile)
Step 4: 生成 Base/Novel split 定义
Step 5: 统计 per-class instance distribution
"""

# adatile/datasets/isaid_instance_fewshot.py
class ISAIDInstanceFewShotDataset:
    """
    iSAID Instance Few-Shot Dataset.
    替代 ISAIDInstanceDataset (旧, 256² tiles, 5i protocol).
    """
```

---

## 四、Evaluation 重构

### 4.1 主指标切换

```
旧 (v2):                         新 (v3):
─────────────────────────────────────────────────
mIoU (主)                        COCO AP (主)
Dice (辅助)                       AP50, AP75 (主)
COCO AP (尝试但非标准)             AP_small, AP_medium, AP_large (主)
                                  AR_max1, AR_max10, AR_max100 (辅助)
                                  FLOPs, FPS, GPU Memory (效率)
                                  mIoU (Supplement only)
```

### 4.2 评估脚本更新

```python
# tools/eval/eval_instance.py (新建, 替代 eval_baseline_instance.py)
"""
统一实例分割评估脚本:
  --mode zero-shot | few-shot
  --checkpoint path/to/model.pt
  --split val
  --tile-size 896
  --output results.json
"""
```

---

## 五、实验线重构

### 5.1 旧实验归档

```
归档 (不再出现在主论文中, 保留作为参考):

A-Series (全监督语义):
  → 保留 runs/ 和 docs/, 论文中不引用
  → 价值: 内部验证 P4 特征上限

C-Series (语义 FSS):
  → C-01, C-02A, C-02B, C-03 保留但降级
  → C-03 (32.7%) 可作补充材料引用
  → 论文中不展开

D-02 (+FDR pixel gate):
  → 已证明有害, 仅作为工程反面教材保留
```

### 5.2 新实验线

```
V3 实验线 (按论文叙事顺序):

Phase 1: Zero-Shot Baseline (1个实验)
  V3-01: FastSAM on iSAID Instance Split
  → AP, AP50, AP75, per-class, area-stratified
  → 证明 domain gap (AP≈0.015)
  → Script: tools/eval/eval_zero_shot.py

Phase 2: Spatial Sparsity Analysis (3个实验)
  V3-02: Tile FG distribution (all tile sizes)
  → 证明空间稀疏性存在 (60% empty @ 1024px)
  V3-03: Oracle Top-K FG retention → SSI
  → 上界: Top-40% → 96.5% FG
  V3-04: SPM learnability
  → Spearman r=0.889, category-agnostic
  → 重用 B-00/B-01/B-02 结果, 重新组织

Phase 3: Few-Shot Fine-tuning (4个实验)
  V3-05: Base pre-training (10 classes, full data)
  V3-06: K-shot scaling (K=1/3/5/10, Novel classes)
  → SSI-shot: 1-shot=88.6% of K=10
  V3-07: Decoder architecture ablation
  → ProtoOnly (427K) vs Adaptive noSPM vs Adaptive full
  V3-08: Per-class analysis (哪些类最难?)

Phase 4: Sparse Routing Efficiency (3个实验)
  V3-09: SPM tile routing @ full-image inference
  → FLOPs reduction vs AP tradeoff curve
  V3-10: Top-K% sweep (10%/20%/30%/40%/50%/100%)
  → Pareto frontier
  V3-11: FPS + GPU Memory measurement

Phase 5: Cross-Dataset (1个实验)
  V3-12: NWPU-VHR-10 few-shot transfer
  → category-agnostic verification
```

---

## 六、代码重构清单

### 6.1 新建文件

| 文件 | 内容 |
|------|------|
| `tools/data/prep_isaid_instance.py` | iSAID Instance Few-Shot Split 数据预处理 |
| `adatile/datasets/isaid_instance_fewshot.py` | 新数据集类 |
| `tools/eval/eval_zero_shot.py` | 统一 Zero-Shot 评估 |
| `tools/eval/eval_fewshot.py` | 统一 Few-Shot 评估 |
| `tools/train/train_base.py` | Base 预训练 |
| `tools/train/train_fewshot.py` | Novel Few-Shot Fine-tune |
| `docs/V3_RESTRUCTURE_PLAN.md` | 本文档 |

### 6.2 重命名文件

| 旧 | 新 |
|-----|-----|
| `adatile/sparse/spatial_router.py` | `adatile/sparse/spm.py` |
| `adatile/decoder/adaptive_sparse_decoder.py` | `adatile/decoder/adaptive_decoder.py` |

### 6.3 归档/废弃文件

| 文件 | 处理 |
|------|------|
| `tools/train/train_instance_fewshot.py` (v3) | 归档 → `tools/archive/` |
| `adatile/datasets/isaid_instance.py` | 归档 |
| `tools/instance/eval_baseline_instance.py` | 归档 |
| `tools/instance/eval_c03_catsam_fewshot.py` | 归档 (FSS) |
| `tools/instance/eval_c04_full_fewshot.py` | 归档 (FSS) |
| 所有 `tools/paper_a/` 脚本 | 保留 (Paper A 存档) |

---

## 七、文档更新清单

| 文件 | 操作 | 内容 |
|------|:----:|------|
| `CLAUDE.md` | 重写 | v3 策略 + 新实验线 + 弃用旧内容 |
| `RESEARCH_MAP.md` | 重写 | 单篇论文证据链 + V3 实验线 |
| `docs/PROJECT_MASTER.md` | 更新 | 反映 v3 重构 |
| `docs/V3_RESTRUCTURE_PLAN.md` | 新建 | 本文档 |
| `memory/two-paper-strategy.md` | 更新 | → single-paper-strategy |
| `memory/paper-b-evidence-chain.md` | 更新 | → paper-evidence-chain (单篇) |
| `pyproject.toml` | 不变 | — |

---

## 八、执行顺序

```
Day 1: 策略确认 + 文档更新
  ✅ 四项决策确认
  ✅ V3_RESTRUCTURE_PLAN.md (本文档)
  ⬜ 更新 CLAUDE.md
  ⬜ 更新 RESEARCH_MAP.md
  ⬜ 更新 PROJECT_MASTER.md
  ⬜ 更新 memory files

Day 2: 数据协议重构
  ⬜ prep_isaid_instance.py
  ⬜ ISAIDInstanceFewShotDataset
  ⬜ 数据验证 (instance count, class distribution)

Day 3: 模块重命名 + 代码清理
  ⬜ spatial_router.py → spm.py
  ⬜ AdaptiveSparseDecoder → AdaptiveDecoder
  ⬜ 归档旧文件 → tools/archive/

Day 4: Phase 1 实验
  ⬜ V3-01: Zero-Shot Baseline (896² tiles, COCO AP)

Day 5: Phase 2 实验
  ⬜ V3-02/03/04: Spatial Sparsity (重用 B-series 结果)

Day 6-7: Phase 3 实验
  ⬜ V3-05: Base pre-training
  ⬜ V3-06/07/08: K-shot + Decoder ablation

Day 8-9: Phase 4 实验
  ⬜ V3-09/10/11: SPM tile routing efficiency

Day 10: Phase 5 + 论文 outline
  ⬜ V3-12: Cross-dataset
  ⬜ 论文 outline
```

---

*This plan defines the v3 restructuring of the AdaTile-FastSAM project.*
*All changes are driven by the four confirmed decisions above.*
