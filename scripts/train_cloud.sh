#!/bin/bash
# ============================================================
# AdaTile-FastSAM Cloud Training Script
#
# Usage:
#   bash scripts/train_cloud.sh                          # 1-shot, default config
#   bash scripts/train_cloud.sh 5shot                    # 5-shot
#   bash scripts/train_cloud.sh full                     # full COCO training
#   bash scripts/train_cloud.sh 1shot --steps 50000      # custom steps
# ============================================================
set -e

cd "$(dirname "$0")/.."

# ── Default settings ──────────────────────────────────────
MODE="${1:-1shot}"
STEPS="${2:-200000}"
BATCH_SIZE="${3:-4}"

# ── Per-mode config ───────────────────────────────────────
case $MODE in
    1shot)
        CONFIG="configs.fewshot.one_shot.get_1shot_config"
        EXTRA="-o train.max_steps=$STEPS data.batch_size=$BATCH_SIZE"
        ;;
    5shot)
        CONFIG="configs.fewshot.five_shot.get_5shot_config"
        EXTRA="-o train.max_steps=$STEPS data.batch_size=$BATCH_SIZE"
        ;;
    10shot)
        CONFIG="configs.fewshot.ten_shot.get_10shot_config"
        EXTRA="-o train.max_steps=$STEPS data.batch_size=$BATCH_SIZE"
        ;;
    full)
        CONFIG="configs.isaid.get_isaid_config"
        EXTRA="-o train.epochs=100 data.batch_size=$BATCH_SIZE train.mixed_precision=bf16"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash scripts/train_cloud.sh [1shot|5shot|10shot|full] [steps] [batch_size]"
        exit 1
        ;;
esac

# ── GPU info ──────────────────────────────────────────────
echo "============================================================"
echo "  AdaTile-FastSAM Cloud Training"
echo "============================================================"
echo "  Mode:       $MODE"
echo "  Config:     $CONFIG"
echo "  Max steps:  $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "============================================================"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
echo "  GPU:        $GPU_NAME"
echo "  VRAM:       $GPU_MEM"
echo "============================================================"

# ── Auto-detect BF16 support (Ampere+) ────────────────────
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "0.0")
SM_MAJOR=$(echo $SM | cut -d. -f1)
if [ "$SM_MAJOR" -ge 8 ]; then
    echo "  Precision:  BF16 (Ampere+ GPU detected)"
    EXTRA="$EXTRA train.mixed_precision=bf16"
else
    echo "  Precision:  FP16"
fi
echo "============================================================"
echo ""

# ── Launch training ───────────────────────────────────────
python tools/train.py \
    --config "$CONFIG" \
    $EXTRA

echo ""
echo "Training finished. Check output/ for results."
