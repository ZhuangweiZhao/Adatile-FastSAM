"""
评估协议 V3 冻结守卫 | Evaluation Protocol V3 Freeze Guard.
============================================================

本测试是 Evaluation Protocol V3 的**强制冻结机制**。它对协议关键定义 (metric / matching /
IoU / AP / zero-shot / support-query / ground-truth / manifest) 的源码逐一取 SHA-256, 与锁文件比对。
任何对这些定义的改动都会使测试 **FAIL** → 阻断 commit (见 .githooks/pre-commit) 与 CI。

注意 | Note: **预测生成 (连通域实例化 + 置信度聚合) 属模型侧, 刻意不在冻结面**。协议只固定
"如何度量" (how we measure), 不固定"模型如何产出预测" (what the model outputs) —— 更换打分方式
(mean / max / mask-score) 或实例化方式不是协议变更。see EVALUATION_PROTOCOL_V3.md §4 / §12.1.

This test is the ENFORCEMENT mechanism for Evaluation Protocol V3. It SHA-256-hashes each
protocol-critical definition and compares against a lock file. ANY change FAILS the test →
blocks the commit (.githooks/pre-commit) and CI.

唯一合法的修改路径 | The only legitimate way to change a frozen definition:
    1. 发布协议修订 (EVALUATION_PROTOCOL_V3.md §Amendment), 经批准;
    2. 重新生成锁: `python tests/test_protocol_frozen.py --update`。
未经上述流程擅自修改 = 协议违规。See EVALUATION_PROTOCOL_V3.md.

禁止产生 V4/V5 定义除非正式发布协议变更。No V4/V5 without a published amendment.
"""

from __future__ import annotations

import ast
import sys
import json
import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = Path(__file__).resolve().parent / "protocol_freeze.lock.json"

# ═══════════════════════════════════════════════════════════════════
# 冻结清单 | Frozen registry: component → [ "file::qualified_name", ... ]
#   qualified_name 支持 "func" 或 "Class.method"
# ═══════════════════════════════════════════════════════════════════

FROZEN: dict[str, list[str]] = {
    # ── IoU (逐实例, 绝不 union) | per-instance IoU, never union ──
    "IoU": [
        "adatile/metrics/instance_match.py::_stack_masks",
        "adatile/metrics/instance_match.py::pairwise_iou",
    ],
    # ── Instance Matching (一对一贪心) | one-to-one greedy matching ──
    "Matching": [
        "adatile/metrics/instance_match.py::greedy_match",
    ],
    # ── Instance mIoU (每个 GT 取最大 IoU 预测) | per-GT max-IoU prediction ──
    "InstanceMIoU": [
        "adatile/metrics/instance_match.py::instance_miou",
    ],
    # ── COCO AP (官方 pycocotools 封装) | official pycocotools wrapper ──
    "COCO_AP": [
        "adatile/metrics/coco_eval.py::mask_to_bbox",
        "adatile/metrics/coco_eval.py::_json_default_bytes",
        "adatile/metrics/coco_eval.py::_summarize",
        "adatile/metrics/coco_eval.py::COCOInstanceEvaluator.add_prediction",
        "adatile/metrics/coco_eval.py::COCOInstanceEvaluator.evaluate",
        "adatile/metrics/coco_eval.py::COCOInstanceEvaluator.evaluate_class_agnostic",
        "adatile/metrics/coco_eval.py::COCOInstanceEvaluator.get_per_category_ap",
        "adatile/metrics/coco_eval.py::connected_components_to_instances",
        "adatile/metrics/coco_eval.py::instances_to_coco_predictions",
    ],
    # ── Zero-shot Definition (非 oracle) | non-oracle zero-shot ──
    "ZeroShot": [
        "tools/eval/evaluate_instance.py::zero_shot_tile_instances",
    ],
    # ── Support / Query Definition (0% 场景重叠, prototype) | support/query protocol ──
    "SupportQuery": [
        "tools/eval/evaluate_instance.py::build_class_prototypes",
    ],
    # ── Ground Truth (COCO annToMask, 逐实例) | per-instance GT construction ──
    #    只冻结 GT 构造 (协议侧, §12.1 明列 Ground Truth Definition)。
    #    预测生成 decoder_prob_map / prob_map_to_instances (连通域实例化 + 置信度聚合) 属**模型侧**,
    #    刻意不冻结: 允许更换实例化与打分 (mean / max / mask-score) 而无需协议修订。
    #    Only GT construction is frozen; prediction generation is model-side (see §4 / §12.1).
    "GroundTruth": [
        "tools/eval/evaluate_instance.py::load_gt_instances",
    ],
    # ── Evaluation Manifest (确定性哈希 + 冻结评估集) | deterministic hash + frozen set ──
    "Manifest": [
        "tools/eval/evaluate_instance.py::_det_hash",
        "tools/eval/evaluate_instance.py::load_manifest",
        "tools/eval/evaluate_instance.py::save_manifest",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# 源码提取 + 哈希 | Source extraction + hashing
# ═══════════════════════════════════════════════════════════════════

def _extract_source(entry: str) -> str:
    """提取 "file::qualname" 的原始源码段 | Extract the raw source segment of "file::qualname".

    支持顶层函数与 "Class.method"。找不到则抛错 (协议符号被删除/改名 = 违规).
    Supports top-level funcs and "Class.method". Missing symbol raises (deletion/rename = violation).
    """
    file_rel, qual = entry.split("::", 1)
    path = _REPO_ROOT / file_rel
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _find(nodes, name):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
                return n
        return None

    if "." in qual:  # Class.method
        cls_name, meth_name = qual.split(".", 1)
        cls = _find(tree.body, cls_name)
        if cls is None:
            raise AssertionError(f"[FROZEN] class '{cls_name}' not found in {file_rel}")
        node = _find(cls.body, meth_name)
        if node is None:
            raise AssertionError(f"[FROZEN] method '{qual}' not found in {file_rel}")
    else:
        node = _find(tree.body, qual)
        if node is None:
            raise AssertionError(f"[FROZEN] function '{qual}' not found in {file_rel}")

    segment = ast.get_source_segment(src, node)
    if segment is None:
        raise AssertionError(f"[FROZEN] could not read source of '{entry}'")
    return segment


def compute_hashes() -> dict[str, str]:
    """对每个 component 计算 SHA-256 (其所有成员源码拼接).
    Compute SHA-256 per component (concatenation of its members' raw source)."""
    out = {}
    for component, entries in FROZEN.items():
        h = hashlib.sha256()
        for entry in entries:
            h.update(entry.encode("utf-8"))
            h.update(b"\x00")
            h.update(_extract_source(entry).encode("utf-8"))
            h.update(b"\x00")
        out[component] = h.hexdigest()
    return out


def _load_lock() -> dict:
    if not _LOCK_PATH.exists():
        raise AssertionError(
            f"[FROZEN] lock file missing: {_LOCK_PATH}\n"
            f"  Generate it once with:  python tests/test_protocol_frozen.py --update")
    return json.loads(_LOCK_PATH.read_text(encoding="utf-8"))


def _write_lock() -> None:
    lock = {
        "protocol": "EVALUATION_PROTOCOL_V3",
        "note": "FROZEN. Do not edit by hand. Regenerate ONLY after a published protocol amendment.",
        "hashes": compute_hashes(),
    }
    _LOCK_PATH.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[FROZEN] lock written → {_LOCK_PATH}")
    for k, v in lock["hashes"].items():
        print(f"    {k:<14s} {v[:16]}...")


# ═══════════════════════════════════════════════════════════════════
# 测试 | Test
# ═══════════════════════════════════════════════════════════════════

def test_evaluation_protocol_v3_frozen():
    """Evaluation Protocol V3 冻结校验 | Freeze verification.

    任一 component 哈希与锁不符 → FAIL, 指出被改动的协议部分.
    Any component-hash mismatch → FAIL, naming the changed protocol part."""
    expected = _load_lock()["hashes"]
    actual = compute_hashes()

    drifted = []
    for component in FROZEN:
        exp = expected.get(component)
        act = actual[component]
        if exp is None:
            drifted.append(f"{component}: NOT in lock (new frozen component — regenerate lock)")
        elif exp != act:
            drifted.append(f"{component}: CHANGED (lock {exp[:12]}… != now {act[:12]}…)")

    removed = [c for c in expected if c not in FROZEN]
    for c in removed:
        drifted.append(f"{c}: removed from FROZEN registry (protocol shrink — not allowed)")

    assert not drifted, (
        "\n\n  ================ EVALUATION PROTOCOL V3 VIOLATION ================\n"
        "  A FROZEN evaluation-protocol definition was modified:\n"
        + "".join(f"    - {d}\n" for d in drifted)
        + "  Frozen: Metric / Matching / IoU / AP / Zero-shot / Support-Query / Ground-Truth / Manifest.\n"
        "  These are LOCKED by EVALUATION_PROTOCOL_V3.md and MUST NOT change silently.\n"
        "  If this change is an APPROVED protocol amendment:\n"
        "      python tests/test_protocol_frozen.py --update\n"
        "  Otherwise REVERT the change. No V4/V5 without a published amendment.\n"
        "  =================================================================\n")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _write_lock()
    else:
        test_evaluation_protocol_v3_frozen()
        print("[FROZEN] OK — all protocol definitions match the lock.")
