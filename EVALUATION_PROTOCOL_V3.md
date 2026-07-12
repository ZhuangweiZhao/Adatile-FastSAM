# EVALUATION PROTOCOL V3 — Official & Frozen
# AdaTile-FastSAM · Few-shot Remote Sensing Instance Segmentation

> **状态 | Status: FROZEN (冻结) · 官方唯一 (single official).**
> 本文件是本项目**唯一官方评估协议**。所有论文实验的**最终上报指标必须**由 V3 官方路径产生。
> 关键定义由自动化守卫锁定;任何静默修改或历史协议渗回都会**报错并阻断 commit / CI**。
> V3 是**本论文全部上报实验**的官方协议。后续修订并非被禁止,但**必须版本化、书面记录、并对受影响
> 结果全量重评**后方可发布(§12)。

---

## 1. Protocol Scope | 适用范围

**适用**:一切**上报到论文/报告**的评估数字(AP 家族、Instance mIoU、per-class、overall、zero-shot 基线)。
**唯一官方实现面 (OFFICIAL V3 surface)**:
- `tools/eval/evaluate_instance.py`(评估入口)
- `adatile/metrics/instance_match.py`(IoU / 一对一匹配 / Instance mIoU)
- `adatile/metrics/coco_eval.py`(COCO AP 官方封装 —— **唯一**允许直接调用 `pycocotools.COCOeval` 的文件)

**不适用**:训练脚本用于**模型选择**的训练内 val 指标(train-internal),但这些数字**不得**作为论文结果。

**本协议只定义 Evaluation,不限制模型 / 训练。** 以下均在协议之外,可自由改进而**不构成协议变更**:
- 训练方式、损失、优化器、epochs、few-shot 微调策略;
- decoder 设计与**实例生成方式**(连通域 / proposal / 其它);
- **score 计算与聚合**(mean / max / mask-score)、`score_thr`、`min_area` 等推理超参;
- backbone、prototype 方法、特征来源(P3/P4/P8)。

> **一句话边界 | One-line boundary**:
> **Anything that changes model behavior without changing metric semantics is outside the protocol.**
> 凡改变模型行为、但**不改变指标语义**者,均在协议之外。

**弃用且禁止用于论文的历史评估器(隔离区)** 见 §12.3。

---

## 2. Task Definition | 任务定义

**Few-shot Remote Sensing Instance Segmentation**:在 iSAID Instance Few-shot Split 上,给定每类 K-shot
support,对 query 图像输出**实例掩码**并按 COCO 实例分割标准评估。**实例分割,非语义分割。**

---

## 3. Ground Truth Definition | GT 定义

- GT = COCO **Instance Annotation**(`data/iSAID_instance_fewshot/annotations/instances_{split}.json`,polygon)。
- 逐实例掩码由 `pycocotools.COCO.annToMask(ann)` 生成(`load_gt_instances`),每个 annotation = 一个 GT 实例。
- 类别 = iSAID 15 类(id 1–15)。评估单元 = tile(896²,`image_id = tile`)。
- **禁止 Union Mask**:严禁把一类的多个实例合并成一张前景 GT 再评估。

---

## 4. Prediction Definition | 预测定义(模型侧 · 非冻结)

> **边界声明 | Boundary**:预测的**生成方式属模型侧**,不属评估协议冻结面(§12.1)。协议只要求预测以
> `(mask, category_id, score)` 三元组交给评估器,并**禁止 GT / oracle / FastSAM proposal**(微调路径);
> 至于**如何实例化、如何打分**,可随模型演进更换,**无需协议修订**。

- **当前 decoder 的限制**:decoder 仅预测"某类语义前景概率图"(该类所有实例合并),**本身不产实例**。
  故当前实现用**连通域**从前景图导出实例假设 —— 这是对 **decoder 能力的权宜转换,不是协议选择**。
  > The current decoder predicts category-wise foreground probability maps; connected-component
  > analysis is therefore adopted to derive instance hypotheses. This is a **model-side** step,
  > driven by the decoder's limitation, not a protocol choice.
- **当前默认实现**(`prob_map_to_instances`,**可更换**):`binary = prob > score_thr(0.5)` →
  `connected_components_to_instances(min_area=16)`;每个连通块 = 一个预测实例;
  `score` = 连通块内前景概率均值。
- **协议对预测的唯一硬约束**:`score` 必须是**模型自产**置信度;**无 oracle、不看 GT、不借 FastSAM
  proposal**。评估器**消费**模型给出的 `score`,**不规定**其聚合方式(mean / max / mask-score 均合法)。

---

## 5. Metric Definition | 指标定义(官方 pycocotools)

| 指标 | 定义 |
|---|---|
| **AP** | segm,IoU=0.50:0.95(step 0.05)均值,maxDets=100,`COCOeval.stats[0]` |
| **AP50 / AP75** | IoU=0.50 / 0.75 时的 AP(`stats[1]/stats[2]`) |
| **APS / APM / APL** | 面积 <32² / 32²–96² / >96² 的 AP(`stats[3..5]`) |
| **Per-class AP** | `get_per_category_ap`:逐类 catIds 单独 COCOeval,取 AP50 |
| **Overall AP** | pycocotools 对"评估图上有 GT 的类别"做 macro-mean(= 上面的 `AP`) |
| **Instance mIoU** | 对**每个 GT 实例**取最大 IoU 预测(类感知),再平均。`overall`=所有 GT 池化(实例频次加权);`class_mean`=先每类均值再对类平均(类均衡) |

- AP 分母限定为 **manifest 评估集**(`image_ids`)。**禁止**第二套 AP 定义或自造 AP。
- 完整数学 + schema 见 `docs/metrics_instance_v3.md`(随本协议一同冻结)。

---

## 6. Matching Definition | 匹配定义

- **COCO AP 侧**:官方 `COCOeval` 内部一对一匹配(按 score 降序、逐 IoU 阈值)。
- **调试计数侧**(`greedy_match`):预测按 score 降序,每个预测认领 IoU 最大且 ≥ 阈值的**未被认领** GT。
- **强制**:一个 prediction ↔ 至多一个 GT;一个 GT ↔ 至多一个 prediction。不变式 `TP+FN==n_gt`, `TP+FP==n_pred`。
- **禁止**:一个 prediction 匹配多个 GT。

---

## 7. Zero-shot Definition | Zero-shot 定义(非 oracle)

- 用**原始权重 FastSAM**(`load_clean_fastsam`,与微调 backbone 隔离)。
- `zero_shot_tile_instances`:`model(img, retina_masks=True)` 默认输出,score = `boxes.conf`(类无关)。
- **禁止**:GT bbox 提示、按 GT-IoU 选最优 mask、任何形式的 oracle。仅报 class-agnostic AP。

---

## 8. Support / Query / Episode Definition | 支持/查询/回合定义

- **Support**:每类 K 个 source 的全部 tile,**仅用于**计算 prototype(`compute_support_prototype`),永不进入评估。
- **Query**:被评估的 tile;评估**只用 query**。
- **隔离**:support 与 query **源图不相交(0% 场景重叠)**(`build_class_prototypes` 中 support_pool 排除 query 源图)。
- **Episode**:K-shot fine-tune 后直接推理(非 episodic FSS)。

---

## 9. Evaluation Dataset Definition | 固定评估集(Manifest)

- **Evaluation Manifest**:`--manifest <path>`(默认 `<data_root>/evaluation_manifest_<split>.json`),存图像文件名列表。
- **首次生成即冻结**;之后所有实验**只读复用同一固定评估集**,`Random Sampling: Disabled`。
- **禁止**每次重新随机采样评估集。**建议**首次用大 `--per-class`(如 9999)生成权威 manifest。

---

## 10. Reproducibility | 可复现性(所有随机来源)

| 来源 | 处理 |
|---|---|
| Query tile 选择 | `_det_hash`(md5),**禁用**内置 `hash()`(PYTHONHASHSEED 随机盐) → 跨 OS/Python 恒定 |
| Support 采样 | `random.Random(seed)`(stdlib Mersenne Twister,跨平台一致) |
| 评估集 | Evaluation Manifest(冻结) |
| 全局种子 | `set_seed(args.seed)`(Python/NumPy/torch/cuDNN deterministic) |

**保证**:给定 **模型 + 权重 + Manifest + Seed** → 任何研究者得到一致的 AP/AP50/AP75/APS/APM/APL/Instance mIoU/per-class/overall。
(通用注记:GPU 前向的 cuDNN 浮点非确定性由 `cudnn.deterministic` 在同硬件抑制,是所有 GPU DL 论文的共性,非本 Evaluator 缺陷;位级跨机复现请随附 checkpoint。)

---

## 11. Known Limitations | 已披露限制(须写入论文)

1. **连通域欠分割**:同类粘连实例会并成一个预测,压低 recall 上界(Evaluator 处理正确:1 TP + 1 FN)。
2. **Instance mIoU 为类感知**(GT 只与同类预测比)。
3. **overall vs class_mean**:`overall` 实例频次加权(被高频类主导),`class_mean` 类均衡;二者本就不等,均上报。
4. **冻结集代表性 = 生成时的 `--per-class`**:务必用大 `--per-class` 生成权威 manifest。
5. **低 AP 反映模型能力,不是评估协议**:当前微调 AP 偏低源于 decoder 只能输出语义前景(连通域欠分割
   + prototype 通路失效,见 [[prototype-functional-death]]),**非**评估口径所致。协议只如实度量;
   换更强的实例化 decoder 即可提升 AP,**无需改协议**。
   > Low AP reflects the current **model** capability (semantic-only decoder), **not** the evaluation
   > protocol. The protocol measures faithfully; a stronger decoder raises AP with no protocol change.

---

## 12. Change Control | 变更管制

### 12.1 何为 Protocol Change
凡涉及以下之一即为协议变更:**Metric Definition / Matching Strategy / IoU Definition / AP Definition /
Zero-shot Definition / Support-Query Definition / Evaluation Manifest / Ground Truth Definition**。
**不得直接修改。**

> **不属协议变更(模型侧,可自由更改,不冻结)**:预测的**实例化方式**(连通域或其它)、**置信度聚合**
> (mean / max / mask-score)、`score_thr`、`min_area` 等推理超参。这些随实验记录即可。冻结面刻意
> **不含** `decoder_prob_map` / `prob_map_to_instances`——协议固定"如何度量",不固定"模型如何产出预测"。

### 12.2 合法修订流程(唯一路径)
1. 明确说明**修改原因**;
2. 评估**对已有实验结果的影响**;
3. **重新验证协议一致性**(下方守卫全绿);
4. 更新本文件 **§Changelog 版本记录**,并重新生成锁:`python tests/test_protocol_frozen.py --update`;
5. 代码 + `tests/protocol_freeze.lock.json` + 本文件 **同一 commit** 提交。
未经 1–5 的修改 = **协议违规**。在正式修订发布前,**不存在 V4/V5**。

### 12.3 自动化守卫 | Enforcement
| 守卫 | 作用 |
|---|---|
| `tests/test_protocol_frozen.py` | 对 8 类 V3 协议定义(含 Ground Truth,**不含**模型侧预测生成)源码取 SHA-256,任何改动 → FAIL(锁:`tests/protocol_freeze.lock.json`) |
| `tests/test_protocol_audit.py` | 项目级审计:V3 面须绝对干净;历史协议模式只允许在隔离区;新文件引入 → FAIL |
| `.githooks/pre-commit` | 提交前跑上述测试,失败即阻断 commit(`git config core.hooksPath .githooks`) |

### 12.4 隔离区(弃用,禁止用于论文)| Quarantine — deprecated, NOT for paper results
以下脚本使用**历史协议(Historical Protocol)**,已列入 `tests/test_protocol_audit.py::QUARANTINE`,仅供历史参考。
(历史协议非"故意错误",而是研究早期的探索;其口径与本任务的实例级评估不可比,故弃用。)

| 文件 | 旧协议问题 |
|---|---|
| `tools/eval/eval_fewshot_allclass.py` | oracle zero-shot(`zero_shot_bbox_iou`)+ union-mask 语义 IoU(`ft_overall_mean_iou`)+ 内置 `hash()` 随机评估集 |
| `tools/eval/eval_zero_shot.py` | 直接 `COCOeval`(未走官方封装) |
| `tools/eval/eval_fastsam_prompted.py` | 随机采样评估集,无 manifest |
| `tools/eval/eval_novel_fewshot.py` | 随机采样评估集,无 manifest |
| `tools/instance/eval_fastsam_zero_shot.py` | 自造 `compute_ap`(非 pycocotools) |
| `tools/instance/eval_c03_catsam_fewshot.py` | 自造 `compute_instance_ap` |

旧 `runs/**/comparison.json`(`ft_overall_mean_iou`)一律不得混入 V3 结果表。对比说明见 `docs/protocol_reconciliation.md`。

---

## Changelog

- **V3.1 clarification (2026-07-12)** — 非破坏性修订(按 §12.2 走完流程)。将**预测生成**
  (`decoder_prob_map` / `prob_map_to_instances`——连通域实例化 + 置信度聚合)移出冻结面,归为**模型侧**;
  冻结面改列 **GroundTruth**(`load_gt_instances`)。**理由**:§12.1 从未把"预测/打分"列为协议变更项,
  原冻结面过度覆盖模型侧逻辑;本次使守卫范围与 §12.1 声明一致。**对已有结果影响**:**无**——未改动任何
  Metric / Matching / IoU / AP / Zero-shot / GT / Support-Query / Manifest 定义,数值不变;已重新生成
  `protocol_freeze.lock.json` 且守卫全绿。
- **V3 (frozen 2026-07-12)** — 首次冻结。实例级 COCO AP + Instance mIoU + 一对一匹配 + 非-oracle zero-shot +
  support/query 0% 重叠 + 确定性 md5 哈希 + Evaluation Manifest。取代历史语义 union-IoU + oracle 协议(隔离区,见 §12.4)。

---

## Appendix A · Protocol Validation | 协议合规性验证

> **已拆分为独立的论文向文档**(职责分离:本文件面向开发者/定义唯一官方协议与变更流程;附录面向论文/
> 回答"为什么这样定义")→ [`docs/appendix_A_protocol_validation.md`](docs/appendix_A_protocol_validation.md)。
> 内含 V3 ↔ COCO 七维逐项对照、Model vs Protocol 边界、Historical Protocol 排除说明,可近乎原样放入
> Supplementary Material。
