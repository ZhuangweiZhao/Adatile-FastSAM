#!/bin/bash
# ===========================================================================
# Instance Few-Shot 消融实验矩阵 | Instance Few-Shot Ablation Matrix.
# ===========================================================================
#
# 实验矩阵 | Experiment Matrix:
#   D-01: K-Shot Scaling        — K=1/3/5/10, ProtoOnly decoder
#   D-02: Decoder Architecture   — ProtoOnly vs AdaptiveSparse @ K=5
#   D-03: FDR Contribution       — K=5, AdaptiveSparse, w/ vs w/o FDR
#
# 用法 | Usage:
#   预览 | Preview:
#     bash tools/instance/run_instance_fewshot_ablation.sh --dry
#   云端 | Cloud:
#     bash tools/instance/run_instance_fewshot_ablation.sh --cloud
#   仅指定实验 | Specific experiments:
#     bash tools/instance/run_instance_fewshot_ablation.sh --only "D01,D02"
# ===========================================================================

set -euo pipefail

DATA_ROOT="data/iSAID-5i/iSAID"
FOLD=0
EPISODES=500
OUTPUT_BASE="runs"
CLOUD_MODE=false
DRY_RUN=false
ONLY_EXPS=""

for arg in "$@"; do
    case $arg in
        --cloud)
            CLOUD_MODE=true
            DATA_ROOT="/root/autodl-tmp/iSAID-5i"
            ;;
        --dry) DRY_RUN=true ;;
        --only=*) ONLY_EXPS="${arg#*=}" ;;
        --data-root=*) DATA_ROOT="${arg#*=}" ;;
        --fold=*) FOLD="${arg#*=}" ;;
        --episodes=*) EPISODES="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── 实验定义 | Experiment Definitions ──
# 格式: "EXP_ID | K-Shot | Classes | Decoder | FDR | Description"
declare -a EXPS=()

# D-01: K-Shot Scaling — ProtoOnly, Novel classes
for k in 1 3 5 10; do
    EXPS+=("D01_K${k} | $k | novel | proto_only | no | ProtoOnly K=${k}-shot Novel")
done

# D-02: Decoder Architecture — K=5, Novel classes
EXPS+=("D02_Adaptive_K5  | 5 | novel | adaptive | yes | AdaptiveSparse K=5 +FDR")
EXPS+=("D02_ProtoOnly_K5 | 5 | novel | proto_only | no  | ProtoOnly K=5 (baseline)")

# D-03: FDR Ablation — K=5, AdaptiveSparse
EXPS+=("D03_Adaptive_FDR  | 5 | novel | adaptive | yes | AdaptiveSparse +FDR")
EXPS+=("D03_Adaptive_noFDR| 5 | novel | adaptive | no  | AdaptiveSparse -FDR")

# ── 过滤 | Filter ──
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

# ── 头部 | Header ──
echo "============================================================================"
echo "Instance Few-Shot Ablation | 实例分割少样本消融"
echo "============================================================================"
echo "  Data: $DATA_ROOT  Fold: $FOLD  Episodes: $EPISODES"
echo "  Exps: ${#EXPS[@]}  Mode: $(if $DRY_RUN; then echo 'DRY-RUN'; elif $CLOUD_MODE; then echo 'CLOUD'; else echo 'LOCAL'; fi)"
echo ""

if [ "$FOLD" = "0" ]; then
    echo "Fold 0: Base=[ship,storage_tank,baseball,tennis,ground_track,"
    echo "              bridge,large_veh,helicopter,soccer_ball,plane]"
    echo "       Novel=[small_veh(9),harbor(15),swimming_pool(11),"
    echo "              basketball(5),roundabout(12)]"
fi
echo ""

# ── 预览 | Preview ──
if $DRY_RUN; then
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        rest="${exp#* | }"; k_val="${rest%% |*}"
        rest="${rest#* | }"; classes="${rest%% |*}"
        rest="${rest#* | }"; decoder="${rest%% |*}"
        rest="${rest#* | }"; fdr="${rest%% |*}"

        fdr_flag=""; [ "$fdr" = "no" ] && fdr_flag="--no-fdr"
        echo "[$exp_id] K=$k_val decoder=$decoder FDR=$fdr"
        echo "  python tools/train/train_instance_fewshot.py --fold $FOLD \\"
        echo "    --k-shot $k_val --classes $classes \\"
        echo "    --decoder-type $decoder $fdr_flag \\"
        echo "    --episodes $EPISODES --output-dir ${OUTPUT_BASE}/ifewshot_${exp_id}_F${FOLD}"
        echo ""
    done
    exit 0
fi

# ── 执行 | Execute ──
COMMON="--data-root $DATA_ROOT --fold $FOLD --episodes $EPISODES"

run_exp() {
    local exp_id="$1" k_val="$2" classes="$3" decoder="$4" fdr="$5"
    local out_dir="${OUTPUT_BASE}/ifewshot_${exp_id}_F${FOLD}"
    local fdr_flag=""; [ "$fdr" = "no" ] && fdr_flag="--no-fdr"

    python tools/train/train_instance_fewshot.py \
        $COMMON --k-shot "$k_val" --classes "$classes" \
        --decoder-type "$decoder" $fdr_flag \
        --output-dir "$out_dir"
}

if $CLOUD_MODE; then
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        rest="${exp#* | }"; k_val="${rest%% |*}"
        rest="${rest#* | }"; classes="${rest%% |*}"
        rest="${rest#* | }"; decoder="${rest%% |*}"
        rest="${rest#* | }"; fdr="${rest%% |*}"

        fdr_flag=""; [ "$fdr" = "no" ] && fdr_flag="--no-fdr"
        LOG="/root/autodl-tmp/ifewshot_${exp_id}.log"

        echo "[$exp_id] Launching... log=$LOG"
        nohup python tools/train/train_instance_fewshot.py \
            $COMMON --k-shot "$k_val" --classes "$classes" \
            --decoder-type "$decoder" $fdr_flag \
            --output-dir "${OUTPUT_BASE}/ifewshot_${exp_id}_F${FOLD}" \
            > "$LOG" 2>&1 &
        echo "  PID=$!"
    done
    echo "All launched."
else
    for exp in "${EXPS[@]}"; do
        exp_id="${exp%% |*}"
        rest="${exp#* | }"; k_val="${rest%% |*}"
        rest="${rest#* | }"; classes="${rest%% |*}"
        rest="${rest#* | }"; decoder="${rest%% |*}"
        rest="${rest#* | }"; fdr="${rest%% |*}"

        echo "[$exp_id] K=$k_val decoder=$decoder FDR=$fdr"
        run_exp "$exp_id" "$k_val" "$classes" "$decoder" "$fdr"
        echo "  Done: ${OUTPUT_BASE}/ifewshot_${exp_id}_F${FOLD}"
    done
    echo "All complete."
fi
