# Appendix A · Evaluation Protocol Validation | 评估协议合规性验证

> **面向 | Audience**: 论文 / Supplementary Material(paper-facing). 本文回答"**为什么 V3 是合理的评估
> 协议**",可近乎原样放入投稿的补充材料。工程侧的唯一官方定义与变更流程见
> [`EVALUATION_PROTOCOL_V3.md`](../EVALUATION_PROTOCOL_V3.md)(developer-facing)。
>
> This appendix explains *why* the evaluation protocol is sound and how it maps onto the standard
> COCO instance-segmentation benchmark. The normative, developer-facing definition (with change
> control and CI enforcement) lives in `EVALUATION_PROTOCOL_V3.md`.

---

## A.1 Motivation | 为什么需要一个受控的评估协议

在早期探索阶段,项目曾使用便于快速消融/调试的评估口径(见 §A.4 Historical Protocol)。这些口径不完全符合
标准实例分割评估,导致绝对量级不可比。为使**本论文的全部上报结果可比、可复现、且与社区标准一致**,我们把
评估固定为一套受控协议 **Evaluation Protocol V3**,并用自动化守卫(源码 SHA-256 冻结 + 项目级审计 +
提交钩子)防止其被静默修改或历史口径渗回。

The project initially used evaluation shortcuts convenient for rapid ablation and debugging. Those did not
fully conform to standard instance-segmentation evaluation, making absolute numbers incomparable. To make
**all results reported in this paper comparable, reproducible, and consistent with community standards**, we
fix evaluation to a controlled protocol (V3), guarded by automated freeze/audit checks.

---

## A.2 V3 ↔ COCO Correspondence | 与 COCO 标准的逐项对应

V3 在七个关键维度上与标准 COCO 实例分割评估一致:

| 标准要件 | COCO 标准做法 | V3 实现 | 一致 |
|---|---|---|---|
| **实例级 GT** | 每 annotation 一个实例掩码 | `load_gt_instances` = `pycocotools.COCO.annToMask` 逐实例;**禁 union mask** | ✓ |
| **一对一匹配** | COCOeval 内部按 score 降序、每 GT 至多配一个预测 | AP 侧走官方 `COCOeval`;调试计数侧 `greedy_match` 同语义,不变式 `TP+FN=n_gt`、`TP+FP=n_pred` | ✓ |
| **实例级 IoU** | 逐实例掩码 IoU | `pairwise_iou` 逐实例计算,**从不 union** | ✓ |
| **AP 计算** | 官方 `pycocotools`,`iouType="segm"` | `COCOInstanceEvaluator` 直接调用官方实现,**不自造 AP** | ✓ |
| **非 oracle** | 检测器自产 `(mask, score)`,评估不看 GT | zero-shot 用 clean-weight FastSAM 默认输出,`score=boxes.conf`,**不看 GT** | ✓ |
| **确定性评估** | 同预测 → 同指标 | `set_seed` + `_det_hash`(md5,禁内置 `hash()` 随机盐) | ✓ |
| **固定评估集** | 官方 val 集固定 | Evaluation Manifest 首次冻结、之后只读复用 | ✓ |

> **Conclusion.** V3 aligns with the standard COCO instance-segmentation benchmark on ground-truth
> definition, matching, IoU, AP computation, non-oracle prediction, deterministic evaluation, and a fixed
> evaluation set. **Therefore the numbers reported under V3 are on the same footing as COCO-family
> benchmarks: comparable, reproducible, and free of the inflation introduced by union-IoU or oracle
> mask selection.**

---

## A.3 Model vs Protocol | 模型与协议的边界(务必区分)

评估协议只定义**如何度量**,不定义**模型如何产生预测**。二者严格分离:

| 层 | 负责 | 是否受协议冻结 |
|---|---|---|
| **Protocol(协议层)** | GT 定义 · Matching · IoU · AP · Zero-shot 定义 · Support/Query 定义 · Evaluation Manifest | **是**(冻结) |
| **Model(模型层)** | 实例生成方式(连通域 / proposal)· score 聚合(mean / max / mask-score)· `score_thr` · `min_area` · backbone · prototype 方法 | **否**(可自由改进) |

一句话概括边界:

> **Anything that changes model behavior without changing metric semantics is outside the protocol.**
> 凡是改变模型行为、但不改变指标语义的,都在协议之外。

**推论(对读者/审稿人的重要澄清)**:本文报告的绝对 AP 偏低,反映的是**当前模型能力**——decoder 仅能输出
类别语义前景图(需连通域后处理才能得到实例假设),且 prototype 通路已功能性失效(收益主要来自 backbone
解冻 + P4 refinement)。这是**模型侧**的上界,**不是**评估协议造成的,也**不因**协议而可被"调高"。更强的实例化
decoder 可在**同一协议**下直接提升 AP。

The low absolute AP reflects the **current model** capability (a semantic-only decoder requiring
connected-component post-processing, with a functionally dead prototype pathway), **not** the evaluation
protocol. A stronger instance-level decoder would raise AP **under the very same protocol**.

---

## A.4 Historical Protocol | 历史协议为何不作为论文结果

> The historical evaluation protocol was developed during the exploratory stage of the project. It was
> suitable for rapid ablation and debugging, but does not fully conform to standard instance segmentation
> evaluation. Therefore, all results reported in this paper are produced exclusively under Evaluation
> Protocol V3.

历史协议(探索期)存在两类与实例分割标准不符的口径:(1)**语义 union-mask IoU**——把一类的多个实例合并成
一张前景图再算 IoU,无法区分"检出 1/3"与"检出 3/3",且与 COCO 不可比;(2)**oracle zero-shot**——用 GT
IoU 挑选最优 mask 作为 zero-shot 基线,人为抬高下限。两者叠加使历史 zero-shot 下限达 0.238,而实例级、
非-oracle 的 V3 下限为 0.015——后者才与本文立论(SA-1B→航拍存在约 27× 领域差距)一致。

历史协议仍具工程价值(快速消融),故未删除,而是**隔离**(见 `EVALUATION_PROTOCOL_V3.md` §12.4,运行时守卫
阻断其产出论文数字)。完整逐项对账见 [`protocol_reconciliation.md`](protocol_reconciliation.md)。

---

## A.5 Reproducibility | 可复现性(论文声明用)

给定 **模型权重 + Evaluation Manifest + Seed**,任何研究者可复现完全一致的
AP / AP50 / AP75 / APS / APM / APL / Instance mIoU / per-class 计数。全部随机来源均被固定:query tile 选择用
确定性 md5 哈希(禁用带进程随机盐的内置 `hash()`),support 采样用 stdlib `random.Random(seed)`,评估集用冻结
的 Manifest,全局 `set_seed` 覆盖 Python/NumPy/torch/cuDNN。(跨机位级复现建议随附 checkpoint;GPU 前向的
cuDNN 浮点非确定性由 `cudnn.deterministic` 在同硬件抑制,为所有 GPU 深度学习论文的共性。)

---

## A.6 See also | 相关

- 唯一官方协议(工程侧)| Normative protocol: [`EVALUATION_PROTOCOL_V3.md`](../EVALUATION_PROTOCOL_V3.md)
- 度量数学定义 | Metric math: [`metrics_instance_v3.md`](metrics_instance_v3.md)
- 历史协议对账 | Reconciliation: [`protocol_reconciliation.md`](protocol_reconciliation.md)
- 代码 | Code: `tools/eval/evaluate_instance.py`, `adatile/metrics/{instance_match,coco_eval}.py`
