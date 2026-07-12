# 评估协议对账表 | Evaluation Protocol Reconciliation

> **创建日期**: 2026-07-12
> **目的**: 澄清 `experiment_master.csv`(旧协议)与 `run_all_experiments.py` 自动评估(新协议)
> 之间 mIoU 差近一倍的根因,避免论文写作时把两套数字混用。
>
> **一句话结论**: 模型没有退化。两批数字用的是**两套不同的评估协议**,绝对量级不可比。
> 新协议每一行大约是旧协议的一半;而 **zero-shot 下限差了 15 倍(0.238 vs 0.015)**,
> 这直接证明差异来自协议本身,与训练无关。
>
> **⚠️ 状态更新(2026-07-12)**: 本文诊断的是两套**历史评估口径**(ImageShot-oracle 与 auto-eval 语义并集)。
> 此后项目已采用 **Evaluation Protocol V3**(实例级 COCO AP + 非-oracle,见
> [`../EVALUATION_PROTOCOL_V3.md`](../EVALUATION_PROTOCOL_V3.md))作为**唯一官方协议**,V3 同时取代上述**两套**
> 历史口径。**本文仅作历史对账;其所称"新/旧协议"均非 V3。**
>
> *The historical evaluation protocol was developed during the exploratory stage of the project. It was
> suitable for rapid ablation and debugging, but does not fully conform to standard instance segmentation
> evaluation. Therefore, all results reported in this paper are produced exclusively under Evaluation
> Protocol V3.*

---

## 1. 铁证:zero-shot 下限差 15 倍

zero-shot 是**完全不训练**的下限,只反映评估协议:

| 协议 | zero-shot mIoU | 逐类示例 |
|---|---|---|
| 旧(ImageShot, per-class=20) | **0.2383** | roundabout 0.511, ground_track 0.507, soccer 0.512 |
| 新(allcls, per-class=9999) | **0.0153** | 逐类几乎全 0.000 |

同一个 "zero-shot mIoU" 差 15 倍 → 只可能是评估方式变了,不可能是模型问题。

> **对论文立论的意义**: 立论是 "FastSAM zero-shot 失败(AP≈0.015, 27× 领域差距)"。
> 新协议 zero-shot=0.015 与立论**一致** ✅;旧协议 zero-shot=0.238 会**打脸立论** ❌
> (审稿人会问:zero-shot 都 0.238 了,27× gap 在哪?)。

---

## 2. 两个协议的差异(已读 `tools/eval/eval_fewshot_allclass.py` 确认)

| 维度 | 旧协议(master.csv) | 新协议(auto eval) |
|---|---|---|
| `--per-class` | **20**(脚本默认) | **9999**(`run_all_experiments.py:593` 写死) |
| 每类样本量 | 20 个裁好的单实例图 | 全部实例(small_vehicle n=326 等) |
| query 构造 | 单个裁剪好的实例 tile | 整幅图切 tile → 合并预测 |
| GT mask | 单实例 mask,bbox 紧 | **该类所有实例的语义并集 mask**,bbox 覆盖满图 |
| zero-shot 定义 | GT bbox 提示 FastSAM + **Oracle 选最优 mask**(上界) | 同函数,但目标是语义并集 → 崩到 0 |
| 数据脚本来源 | `ImageShot` 系列(0706–0707) | `train_fewshot_allclass.py`(0710+) |

**为什么新协议 zero-shot≈0**: query GT 是"该类所有实例合并"的二值 mask,
其 bbox 变成覆盖整个 tile 的大框;FastSAM 只能吐单目标 mask,无法覆盖散布实例的并集 → IoU≈0。
本质上新协议把任务评成了**语义并集分割**,比旧的**单实例分割**更难。

---

## 3. 三套数值对照(核心)

> 新协议列格式:`train_val_miou / held-out_eval_miou`。
> train_val_miou 来自 `train_log.json` 的 `best_val_miou`(噪声大,std≈0.06,仅供参考);
> held-out eval 才是可上报数。

### 3.1 解冻曲线 A(K=1, seed=42, adaptive, P4)

| 实验 | 解冻档 | epochs | 旧协议 (master) | 新协议 train/eval |
|---|---|---|---|---|
| A-0  | frozen | 旧50/新200 | 0.2852 | 0.2481 / 0.1526 |
| A-1  | uf1    | 旧50/新200 | 0.2833 | 0.2428 / (采集失败*) |
| A-5  | uf5    | 旧50/新200 | 0.3559 | 0.3016 / 0.1970 |
| A-8  | uf8 (BASELINE) | 旧50/新200 | 0.4185 | 0.3601 / 0.2107 |
| A-10 | uf10   | 旧50/新200 | 0.4001 | 0.3540 / 0.2258 |
| A-12 | uf12   | 旧50/新200 | 0.4414 (v1) | 0.3669 / 0.2139 |
| A-14 | uf14   | 旧50/新200 | 0.4188 | 0.3596 / 0.2128 |
| A-23 | uf23 lr1e-3 | 50 | 0.0000 (崩溃) | — |
| A-23 | uf23 lr1e-5 | 旧50/新200 | 0.4468 | 0.3788 / **0.2322** |

\* A-1 新协议 eval 采集失败:`.ipynb_checkpoints` 污染了 `_snapshot_runs()` 的新目录 diff,
`output_dir` 被误记成 `runs/.ipynb_checkpoints`,eval 未触发。真实 train_val=0.2428 仍在。

### 3.2 A 长训练(旧协议,epochs 消融)

| 实验 | epochs | 旧协议 mIoU |
|---|---|---|
| A-12 uf12 | 100 | 0.4834 |
| A-12 uf12 | 200 | 0.5149 |
| A-23 uf23 lr1e-5 | 200 | **0.5205** ← master 当前最优 |

### 3.3 K-shot 缩放 B(uf8, 50 epochs)

| K | 旧协议均值 (3 seeds) | 新协议 eval |
|---|---|---|
| K=1 | 0.4134 (±0.014) | 0.1602/0.1760/0.1796 → 均值 0.172 |
| K=3 | 0.3936 (±0.010) | (未跑) |
| K=5 | 0.4092 (±0.005) | (未跑) |

> B 系列两协议都用 **uf8 + 50 epochs**,是最干净的对照(见 §4)。
> 注意:Phase A 已证明 50 epochs 严重欠训练(A-12: 50ep 0.4414 → 200ep 0.5149),
> 但 B/C/E/F 仍用 50 epochs → B 的绝对值本就偏低,与协议问题叠加。

### 3.4 其它(旧协议)

| 实验 | 配置 | mIoU | 备注 |
|---|---|---|---|
| C-E1 | P4 dec + P4 proto | 0.4185 | baseline |
| C-E2 | P4 dec + P8 proto | 0.3914 | P8 proto 有害 −0.027 |
| C-E4-v2 | P3P4 dec + P8 proto | 0.4351 | |
| D-zero / D-random | proto 零/随机输入 | 0.4185 | **proto 通路功能死亡**(=Normal) |
| E-pure | 纯 CNN 无 proto | 0.3835 | |
| F-p3p4-uf12 | P3P4 + uf12 | 0.4282 | P3 在强 backbone 下仍无正贡献 |

---

## 4. 最干净的苹果对苹果对比

完全同一训练配置(**uf8 / K=1 / seed=42 / 50 epochs**),只有评估协议不同:

| | 旧协议 | 新协议 eval | 说明 |
|---|---|---|---|
| fine-tuned mIoU | **0.4185** | **0.1602** | 训练一模一样,纯协议差异 |
| zero-shot mIoU | **0.2383** | **0.0153** | 无任何训练,纯协议差异 |

→ 训练完全相同,mIoU 从 0.42 掉到 0.16;协议是唯一变量。

---

## 5. 定性结论一致,只有量级不同

两个协议下这些结论都成立(不受协议影响):
- 解冻越多越好,uf23 + 低 LR 最优;
- K-shot 基本饱和(K=1≈K=5);
- proto 通路功能死亡(zero/random = normal),收益主要来自 backbone 解冻 + P4 refinement;
- 训练时长仍是瓶颈(50ep 欠训练,200ep 更好)。

变的只是**绝对 mIoU 的量级**。

---

## 6. 待决策 + 行动项

### 必须先定协议(二选一,然后整表按同一协议重跑)
- **方案 A(推荐)**: 统一新协议(per-class=9999 全量 + 语义并集)。数字低但诚实,
  zero-shot=0.015 对得上立论。代价:master 表 A/B/C/H 全部重测。
- **方案 B**: 统一旧协议(per-class=20)。数字好看,但 zero-shot=0.238 与 Intro 冲突,
  且 20 样本噪声大、Oracle 基线易被攻击。

### 独立于协议、都要修的问题
1. **实例 vs 语义**: 新协议用"语义并集 mask"评实例分割,可能本身就是错的度量。
   正确做法应按实例匹配算 COCO AP / per-instance mask IoU。需确认。
2. **zero-shot 用 Oracle 选 mask** 是上界基线,是否公允需评估(审稿点)。
3. **`.ipynb_checkpoints` bug**: `_snapshot_runs()` 未过滤,导致 A-1 结果丢失。
4. **模型选择用裸 `val_miou`**: 噪声峰(std≈0.06),`best_model.pt` 可能存在运气峰上,
   建议改 EMA 或固定全类别评估。
5. **B/C/E/F epochs=50 欠训练**: 与 Phase A(200ep)不一致,建议对齐。

---

## 附:数据来源

- 旧协议官方表: `runs/experiment_master.csv`
- 旧协议原始 eval: `runs/云服务器/runs/runs/eval_fewshot_*_ImageShot_*/comparison.json`(0706–0707)
- 新协议自动 CSV: `runs/云服务器/experiment_results.csv`(0710–0712)
- 新协议原始 eval: `runs/云服务器/runs (1)/runs/eval_fewshot_train_fewshot_allcls_*/comparison.json`
- eval 脚本: `tools/eval/eval_fewshot_allclass.py`(`zero_shot_bbox_iou` at line 87, `--per-class` default 20 at line 141)
- 批量脚本: `tools/run_all_experiments.py`(auto-eval 写死 `--per-class 9999` at line 593)
