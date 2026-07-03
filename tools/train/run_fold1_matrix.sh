#!/bin/bash
# ===========================================================================
# Fold 1 256² 实验矩阵一键脚本 | Fold 1 256² Experiment Matrix Runner.
# ===========================================================================
#
# 用法 | Usage:
#   本地 | Local:
#     bash tools/train/run_fold1_matrix.sh
#
#   云端 | Cloud:
#     bash tools/train/run_fold1_matrix.sh --cloud
#
#   仅预览命令 (不执行) | Dry-run only:
#     bash tools/train/run_fold1_matrix.sh --dry
#
#   仅跑指定实验 | Run specific experiments:
#     bash tools/train/run_fold1_matrix.sh --only "P4,P3P4"
#
# ===========================================================================

set -euo pipefail

# ── 默认配置 | Default Config ──
DATA_ROOT="data/iSAID-5i/iSAID"
FOLD=1
EPOCHS=50
BATCH_SIZE=16
OUTPUT_BASE="runs"
CLOUD_MODE=false
DRY_RUN=false
ONLY_EXPS=""

# ── 解析参数 | Parse Args ──
for arg in "$@"; do
    case $arg in
        --cloud)
            CLOUD_MODE=true
            DATA_ROOT="/root/autodl-tmp/iSAID-5i"
            ;;
        --dry)
            DRY_RUN=true
            ;;
        --only=*)
            ONLY_EXPS="${arg#*=}"
            ;;
        --data-root=*)
            DATA_ROOT="${arg#*=}"
            ;;
        --epochs=*)
            EPOCHS="${arg#*=}"
            ;;
        *)
            echo "Unknown arg: $arg"
            echo "Usage: bash tools/train/run_fold1_matrix.sh [--cloud] [--dry] [--only=P4,P3,P3P4] [--data-root=PATH] [--epochs=N]"
            exit 1
            ;;
    esac
done

# ── 实验定义 | Experiment Definitions ──
# 格式: "EXP_ID | Decoder | 描述 | 额外参数"
declare -a EXPS=(
    "F1_P4_Frz    | P4-only | P4 Frozen Baseline          | "
    "F1_P3_Frz    | P3-only | P3 Frozen (253K)             | --use-p3-only"
    "F1_P3P4_Frz  | P3+P4   | P3+P4 Frozen (273K)          | --use-p3"
)

# ── 过滤实验 | Filter Experiments ──
if [ -n "$ONLY_EXPS" ]; then
    IFS=',' read -ra FILTER <<< "$ONLY_EXPS"
    FILTERED=()
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        for f in "${FILTER[@]}"; do
            if [[ "$exp_id" == *"$f"* ]]; then
                FILTERED+=("$exp")
                break
            fi
        done
    done
    EXPS=("${FILTERED[@]}")
fi

# ── 打印头部 | Print Header ──
echo "============================================================================"
echo "Fold 1 256² Experiment Matrix | Fold 1 256² 实验矩阵"
echo "============================================================================"
echo ""
echo "  Data Root : $DATA_ROOT"
echo "  Fold      : $FOLD"
echo "  Epochs    : $EPOCHS"
echo "  Batch     : $BATCH_SIZE"
echo "  Mode      : $(if $DRY_RUN; then echo 'DRY-RUN (preview only)'; elif $CLOUD_MODE; then echo 'CLOUD (nohup parallel)'; else echo 'LOCAL (sequential)'; fi)"
echo "  Exps      : ${#EXPS[@]} experiments"
echo ""

# ── Fold 1 类别信息 | Fold 1 Class Info ──
echo "Fold 1 Class Split | 类别划分:"
echo "  Base  (10): ship(1), storage_tank(2), baseball_diamond(3), basketball_court(5),"
echo "              small_vehicle(9), helicopter(10), swimming_pool(11), roundabout(12),"
echo "              soccer_ball_field(13), harbor(15)"
echo "  Novel ( 5): plane(14), large_vehicle(8), bridge(7), ground_track_field(6), tennis_court(4)"
echo ""
echo "  ⚠ Fold 1 比 Fold 0 更难: helicopter=78 tiles (Fold 0=2), swimming_pool=82,"
echo "    basketball_court=204, roundabout=147. 极稀有类更多。"
echo ""

# ── 预览模式 | Preview Mode ──
if $DRY_RUN; then
    echo "--- DRY RUN: 以下命令将不会执行 | Commands below will NOT be executed ---"
    echo ""
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        exp_name="${exp#* | }"; exp_name="${exp_name%% |*}"
        exp_desc="${exp#* | }"; exp_desc="${exp_desc#* | }"; exp_desc="${exp_desc%% |*}"
        extra_args="${exp##* | }"

        OUT_DIR="${OUTPUT_BASE}/supervised_${exp_id}_nocb_256"
        echo "[$exp_id] $exp_desc ($exp_name)"
        echo "  python tools/train/train_supervised.py \\"
        echo "    --data-root $DATA_ROOT --fold $FOLD --epochs $EPOCHS \\"
        echo "    --no-class-balance $extra_args \\"
        echo "    --output-dir $OUT_DIR_DIR"
        echo ""
    done
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════
# 执行 | Execute
# ═══════════════════════════════════════════════════════════════════════════

COMMON_ARGS="--data-root $DATA_ROOT --fold $FOLD --epochs $EPOCHS --no-class-balance"

if $CLOUD_MODE; then
    # ── 云端模式: 并行启动 (nohup) | Cloud mode: parallel launch ──
    echo "Launching all experiments in parallel (nohup)..."
    echo ""

    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        exp_name="${exp#* | }"; exp_name="${exp_name%% | *}"
        exp_desc="${exp#* | }"; exp_desc="${exp_desc#* | }"; exp_desc="${exp_desc%% | *}"
        extra_args="${exp##* | }"
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"  # trim leading space

        OUT_DIR="${OUTPUT_BASE}/supervised_${exp_id}_nocb_256"
        LOG_FILE="/root/autodl-tmp/${exp_id}.log"

        echo "[$exp_id] Starting: $exp_desc ($exp_name)"
        echo "  Output: $OUT_DIR"
        echo "  Log   : $LOG_FILE"

        nohup python tools/train/train_supervised.py \
            $COMMON_ARGS $extra_args \
            --output-dir "$OUT_DIR" \
            > "$LOG_FILE" 2>&1 &

        echo "  PID   : $!"
        echo ""
    done

    echo "============================================================================"
    echo "All experiments launched. Monitor with:"
    echo "  tail -f /root/autodl-tmp/F1_*.log"
    echo "  nvidia-smi"
    echo "============================================================================"

else
    # ── 本地模式: 顺序执行 | Local mode: sequential execution ──
    echo "Running experiments sequentially..."
    echo ""

    START_TIME=$(date +%s)
    TOTAL=${#EXPS[@]}
    CURRENT=0

    for exp in "${EXPS[@]}"; do
        CURRENT=$((CURRENT + 1))
        exp_id="${exp%% |*}"
        exp_name="${exp#* | }"; exp_name="${exp_name%% | *}"
        exp_desc="${exp#* | }"; exp_desc="${exp_desc#* | }"; exp_desc="${exp_desc%% | *}"
        extra_args="${exp##* | }"
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        OUT_DIR="${OUTPUT_BASE}/supervised_${exp_id}_nocb_256"

        echo "============================================================================"
        echo "[$CURRENT/$TOTAL] $exp_id: $exp_desc ($exp_name)"
        echo "============================================================================"
        echo ""

        EXP_START=$(date +%s)

        python tools/train/train_supervised.py \
            $COMMON_ARGS $extra_args \
            --output-dir "$OUT_DIR"

        EXP_END=$(date +%s)
        EXP_MIN=$(( (EXP_END - EXP_START) / 60 ))
        echo ""
        echo "  Done in ${EXP_MIN}min. Output: $OUT_DIR"
        echo ""
    done

    END_TIME=$(date +%s)
    TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
    echo "============================================================================"
    echo "All $TOTAL experiments complete in ${TOTAL_MIN}min."
    echo "============================================================================"
fi
