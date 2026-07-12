"""
旧评估器弃用守卫 | Deprecated evaluator opt-in guard.
====================================================

被 EVALUATION_PROTOCOL_V3.md §12.4 隔离的旧评估脚本 (oracle / union-IoU / 自造 AP / 随机评估集)
在 __main__ 处调用本守卫。默认**拒绝运行**并指向官方 V3 评估器; 仅当显式传入 `--force-legacy`
(明知其非 V3、结果不得用于论文) 时才放行。

Legacy evaluators quarantined by EVALUATION_PROTOCOL_V3.md §12.4 call this guard at __main__.
It REFUSES to run by default and points to the official V3 evaluator; it proceeds only when the
user explicitly passes `--force-legacy` (acknowledging the result is NOT valid for paper numbers).
"""

from __future__ import annotations

import sys
from pathlib import Path

_FLAG = "--force-legacy"
_V3 = "tools/eval/evaluate_instance.py"
_PROTOCOL = "EVALUATION_PROTOCOL_V3.md"


def require_legacy_optin(script_path: str) -> None:
    """拒绝运行隔离脚本, 除非显式 --force-legacy | Refuse to run unless --force-legacy is given.

    :param script_path: 通常传 __file__ | usually pass __file__.
    """
    name = Path(script_path).name

    if _FLAG in sys.argv:
        # 用户显式覆盖: 移除该 flag 以免干扰下游 argparse, 并高声警告
        # Explicit override: strip the flag so downstream argparse is unaffected, then warn loudly
        sys.argv = [a for a in sys.argv if a != _FLAG]
        sys.stderr.write(
            f"\n  [DEPRECATED-OVERRIDE] Running quarantined evaluator '{name}' with {_FLAG}.\n"
            f"  Its output uses a PRE-V3 protocol and MUST NOT be reported as a paper result.\n"
            f"  Official protocol: {_PROTOCOL} · Official evaluator: {_V3}\n\n")
        return

    sys.stderr.write(
        "\n"
        "  ============================================================\n"
        f"  DEPRECATED EVALUATOR — BLOCKED: {name}\n"
        "  ------------------------------------------------------------\n"
        "  This script uses a pre-V3 evaluation protocol (one or more of:\n"
        "  oracle zero-shot / union-mask IoU / self-rolled AP / random\n"
        "  evaluation set). It is QUARANTINED and its numbers are NOT\n"
        "  valid for any paper/report.\n\n"
        f"  Use the official evaluator instead:\n"
        f"      python {_V3} --help\n"
        f"  Protocol: {_PROTOCOL} (§12.4 Quarantine)\n\n"
        f"  To run anyway (NON-paper, at your own risk): re-run with {_FLAG}\n"
        "  ============================================================\n\n")
    raise SystemExit(2)
