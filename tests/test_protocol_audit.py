"""
评估协议 V3 项目级审计守卫 | Project-wide Evaluation Protocol V3 Audit Guard.
=============================================================================

扫描全项目, 确保**旧协议定义**(oracle zero-shot / union-mask IoU / 自造 AP / 未固定评估集)
不出现在**官方 V3 评估面**, 且不被**新文件**引入。已知的历史脚本列入 QUARANTINE(隔离区,
明确弃用, 禁止用于论文)。任一禁用模式出现在隔离区之外 → **FAIL**。

Scans the whole project so that pre-V3 definitions never appear on the official V3 surface
and are not introduced by new files. Known legacy scripts are QUARANTINEd (deprecated, forbidden
for paper results). Any forbidden pattern outside the quarantine → FAIL.

与 test_protocol_frozen.py 互补: 后者锁定 V3 定义不被改, 本测试防止旧协议渗回。
Complements test_protocol_frozen.py (which locks the V3 defs); this prevents old protocol creep.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 官方 V3 评估面 (必须绝对干净) | Official V3 surface — must be spotless ──
OFFICIAL_V3 = {
    "tools/eval/evaluate_instance.py",
    "adatile/metrics/instance_match.py",
    "adatile/metrics/coco_eval.py",   # 唯一允许直接调用 pycocotools COCOeval 的文件
}

# ── 隔离区: 已知历史脚本, 明确弃用, 禁止用于论文 | Quarantine: legacy, deprecated, not for paper ──
#    这些文件承认使用旧协议; 列在此处 = 明示技术债, 不得再增加。
QUARANTINE = {
    "tools/eval/eval_fewshot_allclass.py",       # oracle zero-shot + union-mask IoU + built-in hash()
    "tools/eval/eval_zero_shot.py",              # 旧 V3-01 基线: 直接 COCOeval (非官方封装)
    "tools/eval/eval_fastsam_prompted.py",       # 随机采样评估集, 无 manifest
    "tools/eval/eval_novel_fewshot.py",          # 随机采样评估集, 无 manifest
    "tools/instance/eval_fastsam_zero_shot.py",  # 自造 compute_ap (非 pycocotools)
    "tools/instance/eval_c03_catsam_fewshot.py", # 自造 compute_instance_ap
}

# ── 禁用模式 | Forbidden patterns (name → regex, per-pattern file exceptions) ──
FORBIDDEN = [
    # oracle zero-shot: 用 GT 选最优 mask / GT bbox 提示
    ("oracle_zero_shot", re.compile(r"def\s+zero_shot_bbox_iou\b|Select best mask by GT IoU"), set()),
    # 自造 AP (禁止重复实现 COCO AP)
    ("self_rolled_ap", re.compile(r"def\s+compute_ap\b|def\s+compute_instance_ap\b|def\s+voc_ap\b"), set()),
    # 原始 COCOeval (必须走官方封装 COCOInstanceEvaluator); coco_eval.py 为封装本体, 豁免
    ("raw_cocoeval", re.compile(r"COCOeval\("), {"adatile/metrics/coco_eval.py"}),
    # 内置 hash() 参与采样 (非确定性); V3 用 _det_hash 不会命中
    ("builtin_hash_sampling", re.compile(r"\+\s*hash\("), set()),
]


def _iter_py():
    for base in ("tools", "adatile"):
        for p in (_REPO_ROOT / base).rglob("*.py"):
            rel = p.relative_to(_REPO_ROOT).as_posix()
            yield rel, p


def test_official_v3_surface_is_clean():
    """官方 V3 评估面不得含任何旧协议模式 | Official V3 surface must contain no legacy pattern."""
    problems = []
    for rel in sorted(OFFICIAL_V3):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for name, rx, exceptions in FORBIDDEN:
            if rel in exceptions:
                continue
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{rel}:{line}  [{name}]  '{m.group(0)}'")
    assert not problems, (
        "\n\n  ===== OFFICIAL V3 SURFACE CONTAMINATED =====\n"
        + "".join(f"    - {p}\n" for p in problems)
        + "  The V3 evaluator/metrics must never contain oracle / self-AP / raw-COCOeval / hash().\n")


def test_no_legacy_protocol_outside_quarantine():
    """旧协议模式只允许出现在隔离区 | Legacy patterns may appear ONLY in quarantined files."""
    allowed = OFFICIAL_V3 | QUARANTINE
    leaks = []
    for rel, path in _iter_py():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, rx, exceptions in FORBIDDEN:
            if rel in exceptions:
                continue
            m = rx.search(text)
            if m and rel not in allowed:
                line = text[:m.start()].count("\n") + 1
                leaks.append(f"{rel}:{line}  [{name}]  '{m.group(0)}'")
    assert not leaks, (
        "\n\n  ===== NEW LEGACY-PROTOCOL LEAK (outside quarantine) =====\n"
        + "".join(f"    - {p}\n" for p in leaks)
        + "  Route metrics through tools/eval/evaluate_instance.py + adatile/metrics.\n"
        "  If a file is legacy & deprecated, add it to QUARANTINE with a deprecation banner.\n")


def test_quarantine_files_exist():
    """隔离区文件必须真实存在 (防止清单腐烂) | Quarantine entries must exist (no rot)."""
    missing = [q for q in QUARANTINE if not (_REPO_ROOT / q).exists()]
    assert not missing, f"Quarantine list references missing files: {missing}"


def test_quarantine_files_have_deprecation_guard():
    """每个隔离脚本必须在入口调用弃用守卫 | Each quarantined script must call the deprecation guard.

    防止有人悄悄移除守卫、让旧协议评估器又能直接运行。
    Prevents silently removing the guard so a legacy evaluator becomes runnable again.
    """
    missing = []
    for q in sorted(QUARANTINE):
        text = (_REPO_ROOT / q).read_text(encoding="utf-8")
        if "require_legacy_optin(" not in text:
            missing.append(q)
    assert not missing, (
        "\n\n  ===== QUARANTINED SCRIPT MISSING DEPRECATION GUARD =====\n"
        + "".join(f"    - {m}\n" for m in missing)
        + "  Each must call require_legacy_optin(__file__) at __main__ "
        "(tools/_deprecated_guard.py).\n")
