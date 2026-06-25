# AdaTile-FastSAM 研究地图 | Research Map

> 最后更新：2026-06-24 | 分支：`paper-b`

---

## 一、项目全景

```
AdaTile-FastSAM: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper A (Proto Sparsity)          Paper B (Dual Sparsity / AdaTile)
├─ E001→E011-U 完成                ├─ B-00→B-03 理论闭合
├─ 存档于 main 分支                ├─ C-01→C-03 实验进行中
└─ ICIP/CCIG 目标                  └─ 目标期刊 TBD
```

### 两大创新

| # | 创新 | 描述 | 所属 |
|---|------|------|------|
| 1 | **Ada-SPM** | 密度监督稀疏感知模块，学习重要性图→Top-K tile 选择 | Paper A |
| 2 | **FDR** | Foreground Density Router，75K 参数，学习 objectness/density | Paper B |

---

## 二、Paper B 证据链

```
B-00  Spatial Sparsity     → 所有尺度都存在空间稀疏性（60% tiles empty @1024px）
  │
B-01  Oracle Top-K         → 上界：Top40% tiles → 96.5% FG，定义 SSI 判据
  │
B-02  Learnability         → 重要性可学习：Spearman r=0.889（MV3 backbone）
  │
B-02.5 Generalization      → 类别无关（holdout r=0.651），跨数据集泛化可能
  │
B-03  Router Architecture  → FDR 75K ≈ R0 1.48M（Δr=−0.038），Edge≠Importance
  │
B-04  Decoder Verification → LightDecoder val_fg5≈0.47（E13），P4 含语义信息
  │
B-05→B-09 Few-Shot +       → LoveDA/Vaihingen/NWPU 多数据集扩展
         Cross-Dataset
```

### SSI (Spatial Sparsity Index)

- **定义**：Oracle Top40% FG retention
- **SSI > 70**：Router 适用（object-centric: iSAID, DOTA, xView）
- **SSI < 50**：Router 无意义（land-cover: LoveDA, Potsdam）

---

## 三、C 系列实验管线（Few-Shot 实例分割）

```
C-01  FastSAM Zero-Shot           mR@50≈41.5%（召回上限）
  │                                bottleneck = max_det（不是 conf）
  │
C-02A Proto Baseline              1-shot mIoU=0.31%
  │                                Proto Matching→P4 不可用
  │                                但不证明 P4 没信息
  │
C-02B Proto + Refine CNN          1-shot mIoU≈0.5%
  │                                训练崩溃→全零预测
  │                                1维相似度图 refine 是架构死路
  │
C-03  Cross-Attention + Tile      ★ 1-shot mIoU=32.7%（118×提升！）
  │                                Tile 896×896, stride=512
  │                                Shot Saturation: 1=3=5 shot
  │                                bottleneck = small_vehicle (11%)
  │
C-04  Full 15-Class               🔄 进行中
       baseline / FiLM / CrossAttn
```

### C-03 详细结果（Tile 896×896）

| Shot | mIoU | storage_tank | ship | small_vehicle |
|:----:|:----:|:------------:|:----:|:-------------:|
| 1 | **32.66%** | 50.22% | 35.33% | 10.84% |
| 3 | 31.93% | 43.49% | 36.27% | 15.01% |
| 5 | 32.44% | 47.90% | 37.00% | 10.96% |

### C-03 全图 vs Tile 对照

| Shot | 全图 1024×1024 | Tile 896×896 |
|:----:|:-----------:|:----------:|
| 1 | 0.03% | 32.66% |
| 3 | 3.11% | 31.93% |
| 5 | 2.73% | 32.44% |

> **Tile 是 FastSAM few-shot 的必要条件。** 全图 resize→P4 stride=16→小目标消失→模型学习"全预测背景"。

---

## 四、关键发现

### 1. Proto Matching 失败 ≠ P4 没信息
- C-02A: cosine_sim(proto, P4) → 0.3% mIoU
- C-03: CrossAttn(proto, P4) → 32.7% mIoU
- **118× 差距证明：问题在匹配机制，不在特征质量**

### 2. Shot Saturation — 1-shot 已达上限
- 1-shot = 3-shot = 5-shot ≈ 32-33%
- Support 数量不是瓶颈，瓶颈在特征表示（P4 stride=16 + Mean Pooling）
- **正面叙事：SES≈1.0，方法在 1-shot 时已提取最大信息**

### 3. 瓶颈是小目标，不是 Few-Shot
- storage_tank（规则大目标）= 50% ← decoder/proto 都在工作
- small_vehicle（密集小目标）= 11% ← P4 stride=16 太粗
- **下一步优先：Multi-Prototype + P3（stride=8）**

### 4. 全图模式训练崩塌机制
- 对象 resize 到 1024→P4 中 <1.6px
- BCE loss 在 99.95% BG 下找到退化解"全预测背景"→IoU=0
- **Tile 不是优化，是必要条件**

---

## 五、架构演进

```
v1 (2026-06 前)
  Decoupled Sparse Training + SPM + Proto
  → Proto 删除（ΔDice=0），SPM 保留

v2 (2026-06-16)
  完全重写 → adatile 库
  Backbone / Decoder / SPM / FDR / Loss / Logging

C-03 当前架构：
  Support: P4 → masked_mean → proto [1280]
  Query:   P4 → Proj [256] → CrossAttn(Q=x, K=proto, V=proto) → residual(+)
           → Up1(×4) → Up2(×2) → Up3(×2) → Head → Binary Mask
  参数量: ~1.1M（ProtoMLP 394K + Proj 328K + Decoder 388K + CrossAttn 0）
  Backbone: Frozen FastSAM-x, ~0 可训练参数
```

---

## 六、下一步路线图（精简版）

```
C-03 3-class Baseline (done: 32.7% mIoU)
    ↓
C-04 15-class CrossAttn (running)
    ↓
Multi-Scale CAT-SAM (P4+P8, 解决 small_vehicle 11% 瓶颈)
    ↓
FDR-CATSAM (density-aware tile selection + CrossAttn, AdaTile 独有)
    ↓
Ablation Study (tile/scale/proto/loss)
    ↓
Paper Writing
```

| 阶段 | 任务 | 状态 | 核心问题 |
|:----:|------|:----:|------|
| **1** | C-03 3-class Baseline | ✅ 完成 | CrossAttn 有效吗？→ 32.7%, 118× |
| **2** | C-04 15-class Scaling | 🔄 进行中 | 3 类结论在 15 类上成立吗？ |
| **3** | Multi-Scale CAT-SAM | ⏳ | P4+P8 能否拉 small_vehicle 11%→20%+？ |
| **4** | FDR-CATSAM | ⏳ | 密度路由能否在 tile selection + few-shot 上联合优化？ |
| **5** | Ablation | ⏳ | 各组件贡献分解 |
| **6** | Paper | ⏳ | — |

### 不再分散投入

- Multi-Prototype → 合并到 Multi-Scale 一起做
- Attention Proto → 同上
- Contrastive Learning → 不做
- P3 获取 → 用 P8 替代（无需改 backbone）

---

## 七、代码地图

```
adatile/                          # 核心库
├── backbone/          ✅ FastSAMBackbone (P4/P8 hooks, eval-mode, freeze)
├── decoder/           ✅ InstanceDecoder, LightDecoder, LinearProbe, FusionProbe
├── sparse/            ✅ ForegroundDensityRouter, DensityHead, EdgeHead
├── datasets/          ✅ BaseSegDataset, iSAID, iSAID Tiles, LoveDA, NWPU, Vaihingen
│   ├── isaid_tile_wrapper.py  ✅ 全图→tile 包装器 (bbox 重叠, LRU 缓存)
│   └── p4_cache.py            ✅ P4 预计算缓存 (GPU/CPU/fp16/持久化/build_fast)
├── losses/            ✅ FocalLoss, DiceLoss, CombinedLoss
├── logging/           ✅ Console, File/JSONL, Wandb backends
├── metrics/           ✅ compute_miou, compute_dice, FPSMeter, count_params
└── utils/
    ├── seed.py             ✅ set_seed (Python/NumPy/PyTorch/cuDNN)
    ├── label_mapping.py    ✅ iSAID per-split category ID 映射
    └── prototype.py        ✅ compute_fg_prototype (canonical, 4 files→1)

tools/instance/                  # C 系列实验
├── eval_fastsam_zero_shot.py    ✅ C-01: FastSAM 零样本 AP/Recall
├── run_c01_sweep.py             ✅ C-01: max_det×conf 网格搜索
├── eval_c02a_fastsam_fewshot.py ✅ C-02A: Proto Baseline (0 参数)
├── eval_c02b_decoder_fewshot.py ✅ C-02B: Proto + Refine CNN (~10K)
├── eval_c03_catsam_fewshot.py   🔄 C-03: Cross-Attention + Tile (当前活跃)
└── eval_c04_full_fewshot.py     🔄 C-04: 全 15 类 (进行中)

tools/paper_b/                   # Paper B 理论实验
├── eval_b00 → b03   ✅ 理论链闭合
├── eval_b05 → b07   ✅ Oracle/FDR 分析
├── eval_b08 → b09   ✅ 多数据集 few-shot
└── eval_paper_b_pipeline.py ✅ 统一 pipeline
```

---

## 八、关键经验教训

1. **YOLOv8 eval mode**：`model.train()` 触发 YOLOv8 detect head crash，必须 eval mode + requires_grad
2. **Decoder-SPM 解耦**：Decoder 始终接收全特征，SPM 并行训练
3. **Budget loss 可导**：`(imp>0.5).float()` 无梯度 → `(imp.mean−target)²`
4. **SPM 三件套**：GT density focal + Top-K BCE + budget loss，缺一不可
5. **Episodic training**：Baseline 也必须 episodic，否则不公平
6. **Dice GT broadcast**：`unsqueeze(0)` + batch>1 → `[1,B,H,W]` 爆炸
7. **日志先行**：所有新代码接 `adatile.logging`，FileBackend crash-safe
8. **iSAID 双映射 bug**：train/val 用不同 category_id 编号→per-split name matching
9. **P4 缓存 189GB→OOM**：Tile 模式 23k tiles×8MB 不能全缓存→on-the-fly backbone
10. **PyTorch 2.x 兼容**：`torch.cat(x, self.d)` → `torch.cat(x, dim=self.d)`

---

## 九、记忆索引

项目记忆存储于 `memory/` 目录。关键文件：

| 文件 | 内容 |
|------|------|
| `two-paper-strategy.md` | Paper A/B 拆分策略与发表目标 |
| `paper-b-evidence-chain.md` | B-00→B-03 完整理论链 |
| `spatial-sparsity-index.md` | SSI 定义与数据集适用性 |
| `c03-execution-roadmap.md` | C-03 执行路线与优先级 |
| `adatile-evolution-roadmap.md` | 四阶段演化：模型→插件→框架→模块 |
| `publication-strategy.md` | 期刊选择、审稿攻击点、评分 |
| `paper-positioning.md` | 与 11 篇相关工作的关系分析 |

---

## 十、Citation & 对比基准

### 相关方法

| 方法 | 类型 | 与 AdaTile 关系 |
|------|------|----------------|
| CAT-SAM | Few-shot 条件化 | C-03 借鉴其 Cross-Attention 思想 |
| FastSAM | Backbone | 冻结作为特征提取器 |
| ProtoNet | Prototype matching | C-02A 验证不适用的基线 |
| PFENet | Few-shot seg | 对比基线 |
| HSNet | Few-shot seg | 对比基线 |

### AdaTile 独特优势

1. **Tile-level few-shot**：遥感特有，CAT-SAM/PFENet/HSNet 都是 image-level
2. **FDR**：75K 参数的 density router，可学习 tile 重要性
3. **Spatial Sparsity 理论**：SSI 判据统一 B→C 系列实验
4. **Decoder 解耦**：SPM/Router 与 Decoder 独立训练，各自收敛
