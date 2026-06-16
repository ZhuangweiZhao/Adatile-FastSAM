# AS-FastSAM：项目现状

> Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation  
> 面向小样本高分辨率实例分割的自适应稀疏感知 FastSAM  
> 最后更新：2026-06-12

---

## 一、论文主线

### 核心问题

SAM/FastSAM 在高分辨率分割中存在三个根本问题：

1. **高分辨率计算冗余**：4000×4000 图像中大量区域为背景，但 SAM 对所有区域等成本推理
2. **目标分布稀疏非均匀**：遥感场景中目标集中于局部区域，固定计算策略不合理
3. **小样本泛化能力差**：Few-shot 微调容易过拟合数据集纹理，而非学习真正的 object-aware representation

### 三大创新点

| # | 创新 | 核心机制 | 验证状态 |
|---|------|---------|---------|
| ① | **Ada-SPM** | GT-driven importance learning（density focal + planning BCE + budget），让模型学会"哪里值得算" | ✅ SPM-only: Dice=0.797, Coverage=86.5%@15% tiles |
| ② | **解耦稀疏训练** | Decoder 始终收全量特征，SPM 独立训练，无循环依赖 | ✅ 梯度双路径独立，无坍缩 |
| ③ | **Prototype-Guided Decoder** | Support prototype 通过 cosine similarity gate P8 特征 | ⚠️ Proto→Decoder 有效(+0.028)，Proto→Importance 中性 |

### 核心设计原则

> **Decoder 和 Importance 完全解耦**：Decoder 始终接收全量特征进行训练，Importance 通过 GT 驱动 loss 并行独立学习。避免"低 importance → 少 tile → 弱 decoder 梯度 → 更低 importance"的正反馈坍缩。

---

## 二、代码架构

### 目录结构

```
AdaTile-FastSAM/
├── adatile/
│   ├── backbone/
│   │   ├── fastsam_hook.py      ← ★ Hook 特征提取 (P4, P8)，避开 YOLOv8 head
│   │   ├── fastsam_backbone.py   ← 旧 YOLOv8 API（已废弃）
│   │   └── base.py               ← timm backbones
│   │
│   ├── decoder/
│   │   ├── light_decoder.py      ← ★ LightDecoder (U-Net-like, 支持多类别)
│   │   └── base.py               ← DifferentiableDecoder (复杂，保留备用)
│   │
│   ├── sparse/
│   │   ├── light_spm.py          ← ★ LightSPM (3 conv → sigmoid)
│   │   ├── ada_spm.py            ← 完整 Ada-SPM (FPN + transformer，保留备用)
│   │   └── cat_adapter.py        ← CAT adapter + prompt bridge
│   │
│   ├── datasets/
│   │   ├── universal.py          ← ★ UniversalDataset (auto-detect: BSDSeg/LoveDA/flat)
│   │   ├── bsseg.py              ← BSSegDataset (支持 binary + multi-class)
│   │   ├── coco.py               ← COCO dataset
│   │   ├── base.py               ← BaseDataset
│   │   ├── collate.py            ← Batch collation
│   │   └── stats.py              ← Dataset statistics
│   │
│   ├── logging/
│   │   └── run_logger.py         ← ★ RunLogger (CSV + JSON + summary + tee stdout)
│   │
│   ├── routing/                  ← DTR-v2 Router
│   ├── prototype/                ← Prototype Memory
│   ├── tokenizer/                ← Dynamic Tile Tokenizer
│   ├── segmentation/             ← Loss functions
│   ├── engine/trainer.py         ← Hook-based Trainer
│   ├── evaluation/               ← Metrics
│   └── config/                   ← Config system
│
├── tools/
│   ├── train_as_fastsam.py       ← ★ 统一训练入口 (importable + CLI, Stage A/B/C/D)
│   ├── train_stageA.py           ← Stage A: Backbone + Decoder
│   ├── train_stageB.py           ← Stage B: + Ada-SPM
│   ├── train_stageC.py           ← Stage C: + TilePlanner + Sparse Inference
│   ├── train_stageD.py           ← Stage D: + Prototype Memory + Few-shot
│   ├── ablation_proto_path.py    ← Proto 路径隔离消融 (A/B/C 三模型)
│   ├── ablation_domain_shift.py  ← 域偏移消融 (验证 SPM 学密度而非纹理)
│   ├── ablation_tile_ratio.py    ← Tile 保留率消融
│   ├── ablation_proto_importance.py ← Proto→Importance 路径验证
│   ├── exp_fewshot.py            ← Few-shot 对比实验 (1/5/10/full-shot × 3 seeds)
│   ├── diagnose_proto_importance.py ← Proto vs SPM 相关性诊断
│   ├── convert_isaid_binary.py   ← iSAID COCO → binary mask PNGs
│   ├── convert_loveda.py         ← LoveDA → BSSegDataset 布局
│   ├── train.py                  ← COCO/iSAID 训练 (旧管线)
│   └── eval.py                   ← 模型评测
│
├── .claude/
│   ├── CLAUDE.md                 ← 项目指引
│   └── skills/                   ← Claude Code skills
│
├── memory/                       ← ★ 项目知识持久化 (10+ 条记忆)
│   ├── MEMORY.md                 ← 记忆索引
│   ├── paper-innovations.md      ← 三创新框架
│   ├── decoder-gradient-bug.md   ← 历史教训：.item() 切断梯度
│   ├── importance-collapse.md    ← SPM 坍缩原因 + 修复
│   ├── spm-best-result.md        ← SPM 最佳结果记录
│   ├── proto-importance-correlation.md ← Proto vs SPM 相关性
│   ├── proto-path-isolation.md   ← Proto 路径消融设计
│   ├── episodic-baseline.md      ← Episodic training 基线决策
│   ├── unified-trainer.md        ← 统一训练入口设计
│   ├── exp-fewshot.md            ← Few-shot 实验设计
│   └── dataset-architecture.md   ← 数据集架构
│
└── outputs/                      ← 实验结果 (按时间戳命名)
```

### Active Pipeline (Stage A/B/C/D)

```
Stage A:  FastSAM Hook (P4/P8) → LightDecoder → mask [B,1,H/4,W/4]

Stage B:  FastSAM Hook (P4/P8) → LightDecoder → mask
                                → LightSPM → importance [B,1,H/32,W/32]

Stage C:  Stage B + TilePlanner → sparse tiles (top-K% by importance)
              └→ Sparse mask (skip background, keep objects)

Stage D:  Stage C + Prototype Memory (support → class prototypes)
              └→ ProtoGuidedDecoder (proto gate → P8 features)
              └→ Few-shot episodic training
```

**关键设计**：Decoder 训练时始终接收全量特征，SPM 通过 GT-driven loss 并行独立训练，无循环依赖。

---

## 三、关键实验结果

### 3.1 Stage A/B/C 基准 (BSDSeg binary, 640×640)

| 指标 | Stage A | Stage B | Stage C |
|------|---------|---------|---------|
| **Decoder Dice** | 0.846 | 0.856 | — |
| **Sparse IoU (15% tiles)** | — | — | 0.697 |
| **Full IoU** | — | — | 0.750 |
| **精度保持率** | — | — | **93%** |
| **物体覆盖率** | — | — | **86%** |
| **Token 缩减率** | — | — | **85%** |

### 3.2 Proto 路径隔离消融 (`ablation_proto_path.py`)

**结论：Proto 增益全部来自 Decoder 路径，Importance 路径为中性。**

| 实验 | 数据集 | Δ_train (B-A) | 结论 |
|------|--------|--------------|------|
| iSAID binary | 490×490 | **+0.028** | Proto→Decoder 有效 ✅ |
| LoveDA 8-class | 640×640 | -0.001~+0.007 | Proto gate 自关闭 (-0.01)，多类别 Proto 无效 ⚠️ |
| LoveDA 8-class (3000步) | 640×640 | -0.001 | gate 学习到忽略 Proto，(full_dice 可达 0.41) |

### 3.3 SPM 最佳结果 (iSAID binary)

| 指标 | 值 |
|------|-----|
| Sparse Dice @15% tiles | **0.797** |
| Coverage @15% tiles | **86.5%** |
| imp_mean | ~0.05 (健康稀疏) |
| Proto fusion | **中性** (不提升也不损害 SPM) |

### 3.4 Proto vs SPM 相关性诊断 (`diagnose_proto_importance.py`)

| 对比 | Pearson r |
|------|-----------|
| SPM importance vs GT density | **0.6–0.7** (强相关) |
| Proto similarity vs GT density | **0.1–0.3** (弱相关) |

→ SPM 学会的远多于 Proto 提供的，Proto 信号不驱动 importance。

### 3.5 LoveDA 多类别

| 指标 | 值 |
|------|-----|
| Decoder full Dice (3000步) | 0.41 (8 类，可学习) |
| SPM | ❌ 坍缩 (imp→0.93)，前景太密 (30-50%) |
| Proto gate | ❌ 自关闭到 -0.01 |

### 3.6 域偏移消融 (`ablation_domain_shift.py`)

验证 Ada-SPM 学的是目标密度而非数据集纹理。

```
实验设计: Urban↔Rural 交叉训练/测试 (LoveDA)
核心指标: ΔCoverage = Coverage_in - Coverage_cross
判断标准: <5% = SPM 学密度 ✅ | >15% = 过拟合纹理 ❌
```

| 修复 | 状态 |
|------|------|
| 旧方案：密度回归 → 多类别 SPM 坍缩 | ❌ |
| 新方案：per-image top-K 二值化排序 | 🔄 待验证 |

---

## 四、数据集支持

| 数据集 | 类型 | 分辨率 | 类别数 | 加载方式 |
|--------|------|--------|--------|---------|
| BSDSeg | Binary 语义分割 | 640×640 | 1 | UniversalDataset (auto) |
| iSAID | Instance seg (binary) | ~800-4000² | 1 | UniversalDataset + 预处理 |
| LoveDA | 语义分割 | 1024×1024 | 8 | UniversalDataset (auto) |
| COCO | Instance seg | 可变 | 80 | CocoDataset |

**UniversalDataset** 自动检测布局：
- `bsseg`: `{Train,Val}Dataset/{images,masks}/`
- `loveda`: `Train/Train/{Urban,Rural}/images_png/`
- `flat`: `{split}/{images,masks}/`

---

## 五、已完成 vs 待完成

### ✅ 已完成

| 条目 | 说明 |
|------|------|
| FastSAM Hook 特征提取 | P4/P8，避开 YOLOv8 head |
| LightDecoder | 支持 binary + multi-class，Dice > 0.84 |
| LightSPM (Ada-SPM) | GT-driven 三件套 loss，学会稀疏性 |
| TilePlanner 稀疏推理 | 15% tiles → 86% coverage |
| 解耦训练 | Decoder 全量 + SPM 独立，双路径梯度 |
| 三创新框架确定 | ① Ada-SPM ② 解耦训练 ③ Proto-Guided Decoder |
| Proto 路径隔离验证 | Proto→Decoder ✅ / Proto→Importance 中性 |
| SPM vs Proto 相关性诊断 | SPM≫Proto，SPM 独立有效 |
| UniversalDataset | 自动检测布局、类别数 |
| 多类别 decoder + loss | per-image per-class Dice + CE |
| 域偏移消融实验 | 脚本完成，SPM 排序损失修复 |
| RunLogger | CSV + JSON + 双路 stdout |
| 项目记忆系统 | 10+ 条结构化记忆 |
| 统一训练入口 | `train_as_fastsam.py` (importable) |

### ⚠️ 进行中 / 待验证

| 条目 | 优先级 | 说明 |
|------|--------|------|
| 域偏移消融运行 | **高** | 验证 SPM 泛化性，论文核心论据 |
| 多类别 SPM 修复 | **高** | per-image top-K 排序监督（已实现） |
| Few-shot 实验 | **高** | `exp_fewshot.py` 未运行 |
| Stage D (Proto Memory + Few-shot) | 中 | 代码已有，未完整训练 |
| iSAID 多类别 | 中 | DOTA 原图下载状态未知 |
| 更大分辨率 (2048+) | 中 | 当前 640，需验证高分辨率 |
| FLOPs/FPS 精确测量 | 低 | torch.profiler |

### 🔜 短期路线图

```
本周:
  1. 跑通域偏移消融 → 拿到 SPM 泛化证据
  2. 修复多类别 SPM 坍缩 → per-image top-K 排序
  3. 跑 exp_fewshot.py → 1/5/10/full-shot 对比数据
  4. Proto→Decoder 在多类别上的修复（gate 自关闭问题）

下周:
  5. Stage D 完整训练
  6. iSAID 多类别适配
  7. 开始论文写作（Introduction + Method）
```

---

## 六、关键教训 (Lessons Learned)

| # | 教训 | 影响 |
|---|------|------|
| 1 | **YOLOv8 segment head 不可微调** | 必须用 hook + 自定义 decoder |
| 2 | **Importance 和 Decoder 必须解耦** | 联合训练→正反馈坍缩 |
| 3 | **`.item()` 切断梯度链** | FastSAMDecoder 所有参数 grad=None |
| 4 | **SPM 缺少 GT 监督会坍缩** | 需 density focal + planning BCE + budget 三件套 |
| 5 | **多类别前景密集 → SPM 坍缩** | 需 per-image top-K 排序监督 |
| 6 | **Proto 不能做 importance，只能做 decoder** | Proto→Importance 全中性 |
| 7 | **Episodic training 对 few-shot 必须** | 每个 epoch 重采样 support/query |

---

## 七、常用命令

```bash
# ── 统一训练入口 ──────────────────────────
python tools/train_as_fastsam.py                          # Stage A (decoder only)
python tools/train_as_fastsam.py --use-spm                # Stage B (+ Ada-SPM)
python tools/train_as_fastsam.py --use-spm --use-planner  # Stage C (+ sparse)
python tools/train_as_fastsam.py --use-spm --use-planner --use-proto  # Stage D

# ── 消融实验 ──────────────────────────────
python tools/ablation_proto_path.py --quick               # Proto 路径隔离
python tools/ablation_domain_shift.py --quick             # 域偏移消融
python tools/ablation_tile_ratio.py --quick               # Tile 保留率消融
python tools/exp_fewshot.py --quick                       # Few-shot 对比

# ── 诊断工具 ──────────────────────────────
python tools/diagnose_proto_importance.py                 # Proto vs SPM 相关性

# ── 数据集转换 ────────────────────────────
python tools/convert_isaid_binary.py                      # iSAID → binary PNGs
python tools/convert_loveda.py                            # LoveDA → BSSegDataset

# ── 测试 & 代码质量 ──────────────────────
pytest tests/ -v
mypy adatile/
ruff check adatile/
```

---

## 八、记忆系统

项目使用 `.claude/memory/` 持久化关键发现：

| 记忆 | 内容 |
|------|------|
| `paper-innovations` | 三创新框架 + 理论支撑 |
| `decoder-gradient-bug` | `.item()` 切断梯度 → IoU/Dice 恒为 0 |
| `importance-collapse` | SPM 坍缩三件套修复 |
| `spm-best-result` | Dice=0.797 @15%, Coverage=86.5% |
| `proto-importance-correlation` | SPM≫Proto，Proto 不驱动 importance |
| `proto-path-isolation` | Proto→Decoder ✅ / Proto→Importance 中性 |
| `episodic-baseline` | Baseline 必须 episodic |
| `unified-trainer` | 统一训练入口设计 |
| `exp-fewshot` | Few-shot 实验设计 |
| `dataset-architecture` | 多分辨率、few-shot split、tile cache |

新对话开始时，Claude Code 自动加载 `MEMORY.md` 索引。
