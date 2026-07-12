#!/usr/bin/env bash
# ============================================================================
# 确认实验: uf8 + normalize-proto=l2, 匹配 none 基线 200ep, 云端执行 (与基线同环境)
# Confirmatory run: uf8 + normalize-proto=l2, matched to the `none` baseline
# (200 epochs), executed on the CLOUD server (same env the baseline was trained on).
#
# 目的 | Purpose: 把"预判 AP 平"变成"实测 AP 平" —— 三级判据 Mechanism/Function/Task。
#   与 none 基线唯一差别 = --normalize-proto l2 --log-forward-stats。
#   evaluate_instance.py 会从各 checkpoint 自动读取 normalize_proto 并复原 (none→none, l2→l2),
#   两次评估协议完全一致 (frozen V3), 故 AP 直接可比。
#
# 用法 | Usage:
#   bash tools/run_confirmatory_uf8_l2.sh <path/to/none_baseline/best_model.pt>
# ============================================================================
set -euo pipefail

NONE_CKPT="${1:?用法 | usage: bash tools/run_confirmatory_uf8_l2.sh <none_baseline/best_model.pt>}"
[ -f "$NONE_CKPT" ] || { echo "找不到 none 基线 checkpoint | none baseline not found: $NONE_CKPT"; exit 1; }

echo "==== [1/4] 训练 l2 (预算与 none 完全一致, 仅归一化不同) | Train l2 (matched budget) ===="
python tools/train/train_fewshot_allclass.py \
  --decoder adaptive --unfreeze-layers 8 --k-shot 1 \
  --epochs 200 --episodes-per-epoch 200 --lr 1e-3 --seed 42 \
  --data-format isaid_instance --device cuda \
  --normalize-proto l2 --log-forward-stats

L2_DIR=$(ls -dt runs/train_fewshot_allcls_K1_adaptive_uf8_norml2_* | head -1)
L2_CKPT="$L2_DIR/best_model.pt"
echo "  l2 checkpoint -> $L2_CKPT"

echo "==== [2/4] 冻结 V3 评估 — l2 | Frozen V3 eval — l2 ===="
python tools/eval/evaluate_instance.py --decoder adaptive --k-shot 1 --per-class 20 \
  --checkpoint "$L2_CKPT" --output-dir runs/eval/uf8_l2

echo "==== [3/4] 冻结 V3 评估 — none 基线 (同协议) | Frozen V3 eval — none baseline ===="
python tools/eval/evaluate_instance.py --decoder adaptive --k-shot 1 --per-class 20 \
  --checkpoint "$NONE_CKPT" --output-dir runs/eval/uf8_none

echo "==== [4/4] 功能探针 — l2 (coeff cos + Normal-vs-Zero) | Function probe — l2 ===="
python tools/diag/diag_gradient_starvation.py --function-check \
  --checkpoint "$L2_CKPT" --per-class 5 --device cuda \
  --output-dir runs/diag/uf8_l2_function

echo ""
echo "==== 完成. 三级判据对照 | DONE. Read the three-level gate ===="
echo "  Mechanism: $L2_DIR/train_log.json"
echo "     期望 | expect: proto_basis_l2 有界(~1), proto_mask_sat_frac↓, coeff_grad_norm>0"
echo "     (none 基线: basis~3.4e7, sat=1.0, grad=0)"
echo "  Task:      runs/eval/uf8_l2  vs  runs/eval/uf8_none   (AP / Instance mIoU)"
echo "     预判 | predicted: l2 ≈ none (归一化不为 support_proto 凭空造类别信息)"
echo "  Function:  runs/diag/uf8_l2_function/gradient_starvation... (—见 stdout FUNCTION 段)"
echo "     判据 | gate: coeff_offdiag_cosine<1 且 normal_vs_zero_divergence>0 → 功能恢复"
echo "                  仍 cos≈1 且 div≈0 → 原型仍死 (符合审计预判)"
