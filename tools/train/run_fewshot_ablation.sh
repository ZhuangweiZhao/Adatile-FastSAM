#!/bin/bash
# ===========================================================================
# Few-Shot Fine-Tuning 消融实验矩阵 | Few-Shot Fine-Tuning Ablation Matrix.
# ===========================================================================
#
# 实验矩阵 | Experiment Matrix:
#   B-01: K-Shot Scaling        — K=1,3,5,10 × P4 Decoder (核心实验)
#   B-02: Decoder Architecture   — P4 vs P3 vs P3P4 @ K=5
#   B-03: Fine-tuning Strategy   — Decoder-only vs Partial Backbone @ K=5, P3P4
#   B-04: Resolution × Few-Shot  — 256² vs 896² @ K=5, P3P4 (可选)
#
# 用法 | Usage:
#   本地预览 | Local preview:
#     bash tools/train/run_fewshot_ablation.sh --dry
#
#   本地运行 | Local run:
#     bash tools/train/run_fewshot_ablation.sh
#
#   云端运行 | Cloud run:
#     bash tools/train/run_fewshot_ablation.sh --cloud
#
#   仅跑指定实验 | Run specific only:
#     bash tools/train/run_fewshot_ablation.sh --only "B01,B02"
#
#   仅 Zero-shot 评估 | Zero-shot evaluation only:
#     bash tools/train/run_fewshot_ablation.sh --zero-shot-only
# ===========================================================================

set -euo pipefail

# ── 默认配置 | Default Config ──
DATA_ROOT="data/iSAID-5i/iSAID"
FOLD=0
FINETUNE_EPOCHS=20
BATCH_SIZE=16
OUTPUT_BASE="runs"
CLOUD_MODE=false
DRY_RUN=false
ZERO_SHOT_ONLY=false
ONLY_EXPS=""

# ── 预训练 Checkpoint | Pre-trained Checkpoint ──
# 默认使用 Fold 0 P4 Frozen 的预训练模型
# Default: use Fold 0 P4 Frozen pre-trained model
PRETRAINED_CKPT="runs/supervised_F0_P4_Frz_nocb_256/best_model.pt"

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
        --zero-shot-only)
            ZERO_SHOT_ONLY=true
            ;;
        --only=*)
            ONLY_EXPS="${arg#*=}"
            ;;
        --data-root=*)
            DATA_ROOT="${arg#*=}"
            ;;
        --pretrained=*)
            PRETRAINED_CKPT="${arg#*=}"
            ;;
        --fold=*)
            FOLD="${arg#*=}"
            ;;
        --epochs=*)
            FINETUNE_EPOCHS="${arg#*=}"
            ;;
        --batch-size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        *)
            echo "Unknown arg: $arg"
            echo "Usage: bash tools/train/run_fewshot_ablation.sh [--cloud] [--dry] [--zero-shot-only]"
            echo "       [--only=B01,B02] [--pretrained=PATH] [--fold=N] [--epochs=N] [--batch-size=N]"
            exit 1
            ;;
    esac
done

# ── 实验定义 | Experiment Definitions ──
# 格式: "EXP_ID | K-Shot | 种子 | 描述 | 额外参数"
# Format: "EXP_ID | K-Shot | Seeds | Description | Extra Args"

declare -a EXPS=()

# ═══════════════════════════════════════════════════════════════════
# B-01: K-Shot Scaling (核心实验 | Core Experiment)
# ═══════════════════════════════════════════════════════════════════
# P4 Decoder, Decoder-only fine-tuning, K=1/3/5/10, 3 seeds
for k in 1 3 5 10; do
    EXPS+=("B01_K${k} | $k | 42,123,456 | P4 K=${k}-shot scaling | ")
done

# ═══════════════════════════════════════════════════════════════════
# B-02: Decoder Architecture × K-Shot
# ═══════════════════════════════════════════════════════════════════
# K=5, 3 seeds, P3 / P3P4 decoders
# 注意: P3/P3P4 需要对应的预训练 checkpoint
# Note: P3/P3P4 need corresponding pre-trained checkpoints
EXPS+=("B02_P3_K5    | 5 | 42,123,456 | P3 K=5-shot             | --pretrained runs/supervised_F0_P3_Frz_nocb_256/best_model.pt")
EXPS+=("B02_P3P4_K5  | 5 | 42,123,456 | P3P4 K=5-shot           | --pretrained runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt")

# ═══════════════════════════════════════════════════════════════════
# B-03: Fine-tuning Strategy
# ═══════════════════════════════════════════════════════════════════
# K=5, P3P4 Decoder, Decoder-only vs Partial Backbone (last 5 layers)
EXPS+=("B03_DecOnly  | 5 | 42,123,456 | P3P4 Decoder-only K=5    | --pretrained runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt")
EXPS+=("B03_Partial5 | 5 | 42,123,456 | P3P4 Partial-BB K=5     | --pretrained runs/supervised_F0_P3P4_Frz_nocb_256/best_model.pt --partial 5")

# ═══════════════════════════════════════════════════════════════════
# B-04: Resolution × Few-Shot (可选，需 896² pre-cut tiles)
# ═══════════════════════════════════════════════════════════════════
# EXPS+=("B04_896_K5   | 5 | 42 | P3P4 896² K=5-shot | --pretrained runs/supervised_F0_P3P4_896/best_model.pt --tile-root data/iSAID5i_tiles/tile_896")

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
echo "Few-Shot Fine-Tuning Ablation Matrix | 少样本微调消融矩阵"
echo "============================================================================"
echo ""
echo "  Data Root     : $DATA_ROOT"
echo "  Fold          : $FOLD"
echo "  Epochs        : $FINETUNE_EPOCHS"
echo "  Batch         : $BATCH_SIZE"
echo "  Mode          : $(if $DRY_RUN; then echo 'DRY-RUN (preview only)'; elif $CLOUD_MODE; then echo 'CLOUD (nohup parallel)'; else echo 'LOCAL (sequential)'; fi)"
echo "  Zero-Shot Only: $ZERO_SHOT_ONLY"
echo "  Exps          : ${#EXPS[@]} experiments"
echo ""

# ── Fold 信息 | Fold Info ──
echo "Fold $FOLD Class Split | 类别划分:"
if [ "$FOLD" = "0" ]; then
    echo "  Base  (10): ship(1), storage_tank(2), baseball_diamond(3), tennis_court(4),"
    echo "              ground_track_field(6), bridge(7), large_vehicle(8), helicopter(10),"
    echo "              soccer_ball_field(13), plane(14)"
    echo "  Novel ( 5): small_vehicle(9), harbor(15), swimming_pool(11), basketball_court(5), roundabout(12)"
elif [ "$FOLD" = "1" ]; then
    echo "  Base  (10): ship(1), storage_tank(2), baseball_diamond(3), basketball_court(5),"
    echo "              small_vehicle(9), helicopter(10), swimming_pool(11), roundabout(12),"
    echo "              soccer_ball_field(13), harbor(15)"
    echo "  Novel ( 5): plane(14), large_vehicle(8), bridge(7), ground_track_field(6), tennis_court(4)"
elif [ "$FOLD" = "2" ]; then
    echo "  Base  (10): tennis_court(4), basketball_court(5), ground_track_field(6), bridge(7),"
    echo "              large_vehicle(8), small_vehicle(9), helicopter(10), roundabout(12),"
    echo "              soccer_ball_field(13), plane(14)"
    echo "  Novel ( 5): ship(1), storage_tank(2), baseball_diamond(3), swimming_pool(11), harbor(15)"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Zero-Shot Only 模式 | Zero-Shot Only Mode
# ═══════════════════════════════════════════════════════════════════
if $ZERO_SHOT_ONLY; then
    echo "============================================================================"
    echo "Zero-Shot Evaluation Mode | 零样本评估模式"
    echo "============================================================================"
    echo ""

    python tools/train/train_fewshot_finetune.py \
        --data-root "$DATA_ROOT" \
        --load-ckpt "$PRETRAINED_CKPT" \
        --eval-zero-shot-only \
        --output-dir "${OUTPUT_BASE}/fewshot_zero_shot_F${FOLD}"

    echo ""
    echo "Zero-shot evaluation complete."
    echo "Results: ${OUTPUT_BASE}/fewshot_zero_shot_F${FOLD}/zero_shot_results.json"
    exit 0
fi

# ── 预览模式 | Preview Mode ──
if $DRY_RUN; then
    echo "--- DRY RUN: 以下命令将不会执行 | Commands below will NOT be executed ---"
    echo ""
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        rest="${exp#* | }"
        k_val="${rest%% |*}"
        rest="${rest#* | }"
        seeds="${rest%% |*}"
        rest="${rest#* | }"
        desc="${rest%% |*}"
        extra_args="${rest#* | }"
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        # 解析 extra_args 中的 --pretrained 覆盖 | Parse --pretrained override in extra_args
        _ckpt="$PRETRAINED_CKPT"
        if echo "$extra_args" | grep -q "\-\-pretrained"; then
            _ckpt=$(echo "$extra_args" | sed 's/.*--pretrained \([^ ]*\).*/\1/')
            extra_args=$(echo "$extra_args" | sed 's/--pretrained [^ ]*//')
        fi

        echo "[$exp_id] $desc"
        echo "  python tools/train/train_fewshot_finetune.py \\"
        echo "    --data-root $DATA_ROOT --load-ckpt $_ckpt \\"
        echo "    --k-shot $k_val --k-shot-seed $seeds \\"
        echo "    --finetune-epochs $FINETUNE_EPOCHS --batch-size $BATCH_SIZE \\"
        if [ -n "$extra_args" ]; then
            echo "    $extra_args \\"
        fi
        echo "    --output-dir ${OUTPUT_BASE}/fewshot_${exp_id}_F${FOLD}"
        echo ""
    done
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# 执行 | Execute
# ═══════════════════════════════════════════════════════════════════

COMMON_ARGS="--data-root $DATA_ROOT --finetune-epochs $FINETUNE_EPOCHS --batch-size $BATCH_SIZE"

if $CLOUD_MODE; then
    # ── 云端模式: 并行启动 | Cloud mode: parallel launch ──
    echo "Launching all experiments in parallel (nohup)..."
    echo ""

    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        rest="${exp#* | }"
        k_val="${rest%% |*}"
        rest="${rest#* | }"
        seeds="${rest%% |*}"
        rest="${rest#* | }"
        desc="${rest%% |*}"
        extra_args="${rest#* | }"
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        # 解析 extra_args 中的 --pretrained 覆盖 | Parse --pretrained override
        _ckpt="$PRETRAINED_CKPT"
        if echo "$extra_args" | grep -q "\-\-pretrained"; then
            _ckpt=$(echo "$extra_args" | sed 's/.*--pretrained \([^ ]*\).*/\1/')
            extra_args=$(echo "$extra_args" | sed 's/--pretrained [^ ]*//')
        fi
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        OUT_DIR="${OUTPUT_BASE}/fewshot_${exp_id}_F${FOLD}"
        LOG_FILE="/root/autodl-tmp/fewshot_${exp_id}.log"

        echo "[$exp_id] Starting: $desc"
        echo "  K=$k_val, seeds=$seeds"
        echo "  Output: $OUT_DIR"
        echo "  Log   : $LOG_FILE"

        nohup python tools/train/train_fewshot_finetune.py \
            $COMMON_ARGS \
            --load-ckpt "$_ckpt" \
            --k-shot "$k_val" --k-shot-seed "$seeds" \
            $extra_args \
            --output-dir "$OUT_DIR" \
            > "$LOG_FILE" 2>&1 &

        echo "  PID   : $!"
        echo ""
    done

    echo "============================================================================"
    echo "All experiments launched. Monitor with:"
    echo "  tail -f /root/autodl-tmp/fewshot_B0*.log"
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
        rest="${exp#* | }"
        k_val="${rest%% |*}"
        rest="${rest#* | }"
        seeds="${rest%% |*}"
        rest="${rest#* | }"
        desc="${rest%% |*}"
        extra_args="${rest#* | }"
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        # 解析 extra_args 中的 --pretrained 覆盖 | Parse --pretrained override
        _ckpt="$PRETRAINED_CKPT"
        if echo "$extra_args" | grep -q "\-\-pretrained"; then
            _ckpt=$(echo "$extra_args" | sed 's/.*--pretrained \([^ ]*\).*/\1/')
            extra_args=$(echo "$extra_args" | sed 's/--pretrained [^ ]*//')
        fi
        extra_args="${extra_args#"${extra_args%%[![:space:]]*}"}"

        OUT_DIR="${OUTPUT_BASE}/fewshot_${exp_id}_F${FOLD}"

        echo "============================================================================"
        echo "[$CURRENT/$TOTAL] $exp_id: $desc"
        echo "  K=$k_val, seeds=$seeds"
        echo "============================================================================"
        echo ""

        EXP_START=$(date +%s)

        python tools/train/train_fewshot_finetune.py \
            $COMMON_ARGS \
            --load-ckpt "$_ckpt" \
            --k-shot "$k_val" --k-shot-seed "$seeds" \
            $extra_args \
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
