# AdaTile-FastSAM 研究地图 | Research Map

> 最后更新：2026-07-03 | 分支：`paper-b` | 版本：v3

---

## 一、项目定义 (v3)

```
Task:  High-Resolution Few-Shot Instance Segmentation
Paper: 单篇 (Paper A + B 合并)
Innovation: Adaptive Sparse Computation
            ├── SPM (Sparse Perception Module)    ← 统一 FDR + Ada-SPM
            └── Adaptive Decoder                  ← ProtoCoeffPredictor + P4 Refinement
Data:   iSAID Instance Few-Shot Split (自定义, 896² tiles, COCO format)
Eval:   COCO AP + FLOPs/FPS (mIoU → Supplement)
```

### v3 vs v2 关键变化

| 维度 | v2 (旧) | v3 (新) |
|------|---------|---------|
| Paper 数 | 2 篇 (A + B) | **1 篇** |
| Task | FSS / Instance / Sparse 混杂 | **High-Resolution Few-Shot Instance Segmentation** |
| 数据 | iSAID-5i (FSS protocol) | **自定义 Instance Few-Shot Split** |
| 主指标 | mIoU | **COCO AP** |
| 模块名 | FDR + Ada-SPM (两个) | **SPM (统一)** |
| Decoder | AdaptiveSparseDecoder | **AdaptiveDecoder** |

---

## 二、完整证据链 (v3)

```
V3-01  Zero-Shot Baseline        → FastSAM on iSAID: AP=0.015, 27× domain gap
  │
V3-02  Spatial Sparsity           → 所有尺度稀疏 (60% empty @ 1024px)
  │
V3-03  Oracle Top-K               → 上界: Top-40% tiles → 96.5% FG, SSI 判据
  │
V3-04  SPM Learnability           → 重要性可学习: Spearman r=0.889, category-agnostic
  │
V3-05  Base Pre-training          → 10 classes, full data → COCO AP baseline
  │
V3-06  K-Shot Scaling             → K=1/3/5/10: SSI-1=88.6%, K=5 sweet spot
  │
V3-07  Decoder Ablation           → ProtoOnly (427K) vs Adaptive (1.14M): Δ=+34.5% mIoU
  │
V3-08  Per-Class Breakdown        → small_vehicle 最困难, harbor 最容易
  │
V3-09  SPM Tile Routing           → full-image inference, FLOPs reduction
  │
V3-10  Top-K% Sweep               → 10%-100% tiles: AP vs FLOPs Pareto frontier
  │
V3-11  Efficiency                 → FPS + GPU Memory measurement
  │
V3-12  Cross-Dataset              → NWPU-VHR-10 few-shot transfer (category-agnostic)
```

### SSI (Spatial Sparsity Index)

- **定义**: Oracle Top40% FG retention — 预实验零成本判据
- **SSI > 70**: SPM 适用 (object-centric: iSAID, DOTA, xView)
- **SSI < 50**: SPM 无意义 (land-cover: LoveDA, Potsdam)

### SSI-shot (Shot Saturation Index)

- **定义**: mIoU(K) / mIoU(K=10) — K shot 达到 full-shot 性能的百分比
- **SSI-1=88.6%, SSI-5=96.9%** — 1-shot 即接近饱和
- **意义**: 瓶颈在 proto 基函数表达上限 (~0.33 mIoU)，不在样本数

---

## 三、核心结果矩阵

### 3.1 历史实验摘要 (v2, 归档参考)

**B-Series (空间稀疏性理论, B-00→B-03):**

| 实验 | 核心发现 |
|------|----------|
| B-00 | 所有尺度稀疏 — 1024px 下 60% tiles empty |
| B-01 | Oracle Top-K: Top-40% tiles → 96.5% FG, IDG=2.41× |
| B-02 | 重要性可学习: Spearman r=0.889 (MV3 backbone) |
| B-02.5 | Category-AGNOSTIC (holdout r=0.651) |
| B-03 | FDR 75K ≈ R0 1.48M (Δr=-0.038), Edge≠Importance |
| B-04 | LightDecoder val_fg5≈0.47, P4 有语义信息 |

**C-Series (语义 FSS, C-01→C-03):**

| 实验 | 核心发现 |
|------|----------|
| C-01 | FastSAM Zero-Shot mR@50≈41.5% |
| C-02A | Proto Matching 1-shot mIoU=0.31% (Proto→P4 直接匹配失败) |
| C-03 | ★ Cross-Attention + Tile: 1-shot mIoU=**32.7%** (118× over C-02A) |

**D-Series (实例分割, D-00→D-04):**

| 实验 | Decoder | FDR | K | mIoU | AP50 |
|------|---------|-----|---|------|------|
| D-00 | FastSAM | — | — | ~0.01 | 0.023 |
| D-01 | ProtoOnly | — | 5 | 0.235 | 0.000 |
| D-02 | Adaptive | ✓ | 5 | 0.309 | 0.000 |
| D-03 | Adaptive | ✗ | 5 | **0.316** | 0.000 |
| D-04 | Adaptive noFDR | ✗ | 1/3/5/10 | 0.289→0.326 | 0.000 |

### 3.2 三条核心发现 (来自 D-Series)

1. **Domain Gap (27×)**: SA-1B→iSAID AP=0.015. "Segment anything" 在航拍不成立.
2. **P4 Refinement (+34.5%)**: ProtoOnly (0.235) → Adaptive noFDR (0.316). P4 精炼是核心驱动力.
3. **Shot Saturation**: 1-shot=88.6% of K=10. 瓶颈是 proto 基函数 (~0.33 天花板), 不是样本数.

### 3.3 FDR/SPM 角色

```
SPM 正确用法 (B-Series 验证):
  大图 → SPM → tile importance → Top-K tiles → Adaptive Decoder per tile

SPM 错误用法 (D-Series 验证):
  256² tile → per-pixel density gate → -2.2% mIoU
```

---

## 四、架构 (v3)

```
High-Res Image (H×W, up to 4000²)
      │
      ▼
┌─────────────────────────────────────────┐
│  FastSAM Backbone (frozen)              │
│  P3 [H/8, W/8, 960]                     │
│  P4 [H/16, W/16, 1280]   ← decoder      │
│  P8 [H/32, W/32, 1280]   ← SPM          │
│  Proto [H/4, W/4, 32]                   │
└─────────────────────────────────────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌───────────┐  ┌────────────────┐
│   SPM    │  │   Proto   │  │  Support Set    │
│ P8→Imp.  │  │  Masks    │  │  K-shot/class   │
│ →Tiles   │  │ [32,H,W]  │  │  → Prototype    │
└────┬─────┘  └─────┬─────┘  └───────┬────────┘
     │              │                 │
     │   K tiles    │   crop to tile  │
     └──────┬───────┘                 │
            ▼                         ▼
   ┌─────────────────────────────────────┐
   │  Adaptive Decoder                   │
   │  ├── ProtoCoeffPredictor (427K)     │
   │  │   support proto → 32 coeffs      │
   │  ├── Feature Refinement (P4, 714K)  │
   │  │   1280→256→128→64 spatial detail │
   │  └── Mask Head                      │
   │      64→32→1 per-pixel FG logit     │
   └─────────────────────────────────────┘
            │
            ▼
   Instance Masks + Scores
            │
            ▼
   COCO AP Evaluation
```

---

## 五、模块命名 (v3)

```
SPM (Sparse Perception Module)    ← was FDR + Ada-SPM
  └── ImportanceHead              ← was DensityHead
  └── TileRouter                  ← was select_tiles()

AdaptiveDecoder                   ← was AdaptiveSparseDecoder
  └── ProtoCoeffPredictor         (unchanged)
  └── FeatureRefinement           ← was feat_proj + feat_refine
  └── MaskHead                    ← was mask_head
```

---

## 六、下一步

| 优先级 | 任务 | 预期收益 |
|:------:|------|----------|
| **P0** | 数据协议: 生成 896² Instance Few-Shot Split | v3 基础 |
| **P0** | V3-01: Zero-Shot COCO AP baseline (896²) | 新基线 |
| **P1** | V3-05/06: Base pre-train + K-shot fine-tune | 核心实验 |
| **P1** | V3-09/10: SPM tile routing efficiency | SPM 价值证明 |
| **P1** | Proto basis finetuning | 突破 ~0.33 ceiling |
| **P2** | Cross-dataset (NWPU) | 泛化性 |

---

## 七、代码地图 (v3)

```
adatile/
├── backbone/          ✅ FastSAMBackbone (P3/P4/P8 + Proto)
├── decoder/
│   ├── light_decoder.py         ✅ LightDecoder (716K, P4-only)
│   ├── light_decoder_p3p4.py    ✅ LightDecoderP3P4 (274K)
│   └── adaptive_decoder.py      ✅ AdaptiveDecoder (1.14M) + ProtoOnlyDecoder (427K)
├── sparse/
│   ├── spm.py                   🔄 SparsePerceptionModule (→ rename)
│   └── coefficient_predictor.py ✅ ProtoCoeffPredictor
├── datasets/
│   ├── isaid_tiles.py              ✅ 1024² tiles
│   ├── isaid_instance_fewshot.py   🔄 NEW: 896², COCO format
│   ├── nwpu.py                     ✅ NWPU-VHR-10
│   └── loveda_tiles.py             ✅ LoveDA
├── metrics/           ✅ COCOInstanceEvaluator + compute_miou + FPSMeter
├── losses/            ✅ FocalLoss (eps=1e-4, γ=5.0) + DiceLoss
├── logging/           ✅ Console + File/JSONL + Wandb
└── utils/             ✅ seed, prototype, label_mapping, render

tools/
├── data/prep_isaid_instance.py       🔄 NEW: Instance Few-Shot Split
├── train/
│   ├── train_base.py                 🔄 NEW: Base pre-training
│   └── train_fewshot.py              🔄 NEW: K-shot fine-tune
├── eval/
│   ├── eval_zero_shot.py             🔄 NEW: Zero-shot COCO AP
│   └── eval_fewshot.py               🔄 NEW: Few-shot COCO AP
├── archive/                          # v2 归档
└── paper_b/                          # B-Series (可重用)
```

---

## 八、关键经验教训

1. **YOLOv8 eval mode**: `model.train()` crash → eval mode + requires_grad
2. **Decoder-SPM 解耦**: Decoder 始终接收全特征
3. **Budget loss 可导**: `(imp>0.5).float()` 无梯度 → `(imp.mean−target)²`
4. **Focal for remote sensing**: eps=1e-4 (防梯度爆炸), γ=5.0 (极端不平衡)
5. **BatchNorm→InstanceNorm**: bs=1 训练必须 InstanceNorm2d(affine=True)
6. **min_tiles=30**: 稀有类 NaN → 过滤 + eps + γ + skip 四层防护
7. **FDR=SPM 是 tile-selector**: per-pixel gate 在已选 tiles 上冗余且有害 (-2.2%)
8. **Tile 是必要条件**: 全图 resize→小目标消失→全预测背景崩塌
9. **Shot Saturation**: 瓶颈是 proto 基函数 (~0.33), 不是样本数
10. **iSAID 双映射 bug**: train/val 不同 category_id→per-split name matching

---

## 九、记忆索引

项目记忆存储于 `memory/` 目录 (39 files). 索引：`memory/MEMORY.md`.

关键文件:
- `PROJECT_MASTER.md` — 全项目总览
- `D_series_master.md` — D-Series 完整档案
- `V3_RESTRUCTURE_PLAN.md` — v3 重构方案
