---
name: record-memory
description: 记录 AdaTile-FastSAM 项目的实验数据、设计决策、Bug 修复、关键发现到持久记忆文件。每次有重要发现或用户说"记录一下""记住""保存"时触发。
---

# Record Memory — AdaTile-FastSAM

把对话中发现的关键信息、实验结论、Bug 根因、架构决策写入记忆文件，跨会话保留。

## 记忆存储位置

```
C:\Users\20871\.Codex\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\
```

## 工作流程

### 1. 识别 → 2. 确认 → 3. 查重 → 4. 写入 → 5. 更新索引

**Step 1: 自动识别**（无需用户指令，发现以下情况时主动记录）

- 实验跑出好坏结果 → 记录数据
- 发现 Bug 根因并修复 → 记录原因+方案
- 用户做出设计决策 → 记录决策+理由
- 用户表达偏好/纠正方向 → 记录偏好
- 架构/流程发生变更 → 记录新架构
- 某个 λ/batch_size/image_size 参数调到最佳 → 记录最佳配置

**Step 2: 列出候选，让用户确认**

简短列出 1-3 个候选主题，问"记录这些？"。用户可能只选部分，
或合并，或拒绝。不要跳过确认。

**Step 3: 查重** — 先看 MEMORY.md，有同类文件就更新而非新建。

**Step 4: 写文件** — 用下面模板，每个记忆一个文件。

**Step 5: 更新 MEMORY.md** — 在索引末尾加一行。

## 项目特定记忆类型

### 📊 实验结果
```
## 适合记录的内容
- SPM ablation: λ=5.0 时 imp_mean 从 0.37→0.22 最接近 0.15 target
- Few-shot: 5-shot Seed=2 下 Dice 提升 +0.03
- 新脚本首次运行结果
- Coverage/Dice 的 trade-off 关系

## 文件名: exp-<实验简称>.md
```

### 🐛 Bug 根因
```
## 适合记录的内容
- budget loss gradient=0：hard threshold > 0.5 不可导
- num_classes=256：mask={0,255} 未做二值化
- IoU>1：inter/(union-inter) 公式错误

## 文件名: <bug简述>.md
## 必须包含: 症状 | 根因 | 修复 | 验证方法
```

### 🏗️ 架构决策
```
## 适合记录的内容
- 为什么用 UnifiedLoss 替代各处定义 loss
- 为什么 Decoder 训练时用 full feature
- 为什么 episodic training 对 baseline 也要用

## 文件名: <决策简称>.md
## 必须包含: 决策 | Why | How to apply
```

### 🎯 关键参数
```
适合记录：最优 lr、λ_spm、λ_budget、batch_size、keep_ratio 等
```

### 📝 用户偏好
```
适合记录：代码风格偏好、命名习惯、实验流程偏好等
```

## 文件模板

```markdown
---
name: <kebab-case-slug>
description: <一行概要，检索用>
metadata:
  type: project | feedback | user | reference
---

# <标题>

<核心内容>

**Why:** <原因>

**How to apply:** <如何应用>

参见 [[related-memory]]
```

## 什么不用记录

- AGENTS.md 或 PROJECT_STATUS.md 已有的内容
- 临时调试 print（除非揭示了关键 insight）
- 注解读完代码就能知道的信息
- 微小语法修复

## 索引格式 (MEMORY.md)

每行一个记忆：
```
- [<短标题>](<文件名>.md) — <一句话hook>
```

## 写记忆前必做的检查清单

- [ ] 已有同类文件？→ 更新，不新建
- [ ] 内容不重复 AGENTS.md？
- [ ] 包含 Why + How to apply？
- [ ] 链接了相关记忆 [[like-this]]？
- [ ] 文件名是 kebab-case？
- [ ] MEMORY.md 索引已更新？

## 项目上下文速查

| 项目 | AdaTile-FastSAM |
|------|----------------|
| 统一入口 | `tools/train_as_fastsam.py` (UnifiedLoss, ProtoGuidedDecoder, ...) |
| 数据集 | `dataset/` — bsseg, loveda, flat 三种布局 → UniversalDataset 自动检测 |
| 核心组件 | FastSAMHookBackbone → LightDecoder + LightSPM |
| 双创新 | ① Ada-SPM ② 解耦稀疏训练 |
| 关键指标 | Dice, IoU, Coverage, imp_mean |
| L_total | L_seg + λ_spm × L_spm + λ_budget × (imp.mean() − keep_ratio)² |
| 当前最优 | Top-K λ=5.0: Dice=0.8326, Coverage=86.7%, imp_mean=0.2241 |

## 活跃实验脚本

| 脚本 | 用途 |
|------|------|
| `tools/ablation_spm_supervision.py` | Density vs Top-K 监督方式对比 |
| `tools/ablation_spm_fewshot.py` | SPM 小样本优势：Baseline vs SPM-only × shot |
| `tools/ablation_domain_shift.py` | Urban/Rural 域迁移实验 |
| `tools/exp_fewshot.py` | 1/5/10/full-shot baseline vs AS-FastSAM |
| `tools/train_bsseg.py` | BSDSeg 单数据集训练 |
| `tools/train_cat_adatile.py` | 多类数据集训练 |
| `tools/verify_innovations.py` | 创新点验证脚本 |
