#!/usr/bin/env bash
# ============================================================================
# Full LCR pipeline: generate -> audit -> subset -> train -> eval
# Asks every question upfront, prints a single summary, confirms once, runs.
# Each stage is independently toggleable; existing outputs can be reused.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Expect the user to activate their conda env BEFORE running this script.
# We don't auto-activate. If nothing is active, warn but allow continuing
# (the user may be using a system Python on purpose).
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: no conda env or venv is active."
    echo "         You probably want to run: conda activate <name>"
    read -r -p "Continue anyway? (y/N): " _CONT
    if [[ ! "${_CONT:-n}" =~ ^[Yy]$ ]]; then
        echo "Aborted. Activate your env and re-run."
        exit 1
    fi
fi

# ============================================================================
# Small helpers
# ============================================================================
ask() {
    # ask "prompt" default_value -> value
    local prompt="$1" default="$2" answer
    read -r -p "$prompt [$default]: " answer
    echo "${answer:-$default}"
}

ask_yn() {
    # ask_yn "prompt" default_yn -> 0 (yes) or 1 (no)
    local prompt="$1" default="$2" answer
    if [ "$default" = "y" ]; then
        read -r -p "$prompt (Y/n): " answer
        answer="${answer:-y}"
    else
        read -r -p "$prompt (y/N): " answer
        answer="${answer:-n}"
    fi
    [[ "$answer" =~ ^[Yy] ]]
}

model_short() {
    # 'Qwen/Qwen2.5-0.5B' -> 'qwen2.5-0.5b'
    local m="$1"
    echo "${m##*/}" | tr '[:upper:]' '[:lower:]'
}

temp_str() {
    # 1.25 -> temp1_25
    local t="$1"
    echo "temp${t//./_}"
}

# Decide what to do with an existing output dir.
# Returns: REUSE | RERUN | ABORT
decide_existing() {
    local label="$1" path="$2"
    if [ ! -e "$path" ]; then
        echo "RUN"
        return
    fi
    echo ""
    echo "[$label] output already exists: $path"
    echo "  r) reuse (skip this stage)"
    echo "  R) re-run (overwrite)"
    echo "  a) abort pipeline"
    local choice
    read -r -p "  choice [r]: " choice
    case "${choice:-r}" in
        r|reuse) echo "REUSE" ;;
        R|RERUN) echo "RERUN" ;;
        a|abort) echo "ABORT" ;;
        *)       echo "REUSE" ;;
    esac
}

# ============================================================================
# Banner
# ============================================================================
echo "=========================================="
echo "  LCR Pipeline"
echo "=========================================="
echo ""
echo "Stages (each can be skipped):"
echo "  [1] Generate     synthetic corpus (vLLM, BOS-only)"
echo "  [2] Audit        n-gram decontamination vs benchmarks"
echo "  [3] Subset       carve out N training subsets"
echo "  [4] Train        llamafactory continued pretraining"
echo ""
echo "Evaluation is a separate script: bash evaluate.sh"
echo ""
echo "All questions are asked up front."
echo ""

# ============================================================================
# Global config
# ============================================================================
echo "─── Global ──────────────────────────────────────"

# Curated model menu. Paper-relevant models marked with *.
echo "Model selection:"
echo "  Qwen 2.5 base:"
echo "    1.  Qwen/Qwen2.5-0.5B           *  (paper main student)"
echo "    2.  Qwen/Qwen2.5-1.5B"
echo "    3.  Qwen/Qwen2.5-3B"
echo "    4.  Qwen/Qwen2.5-7B             *  (paper same-lineage source)"
echo "  Qwen 3 base:"
echo "    5.  Qwen/Qwen3-1.7B-Base"
echo "    6.  Qwen/Qwen3-4B-Base"
echo "    7.  Qwen/Qwen3-8B-Base          *  (paper different-recipe source)"
echo "  LLaMA 3.2:"
echo "    8.  meta-llama/Llama-3.2-1B     *  (paper cross-family / second student)"
echo "    9.  meta-llama/Llama-3.2-3B"
echo "    c.  custom HuggingFace id"
echo ""
MODEL_CHOICE=$(ask "Choose (1-9 or c)" "1")
case "$MODEL_CHOICE" in
    1) MODEL="Qwen/Qwen2.5-0.5B" ;;
    2) MODEL="Qwen/Qwen2.5-1.5B" ;;
    3) MODEL="Qwen/Qwen2.5-3B" ;;
    4) MODEL="Qwen/Qwen2.5-7B" ;;
    5) MODEL="Qwen/Qwen3-1.7B-Base" ;;
    6) MODEL="Qwen/Qwen3-4B-Base" ;;
    7) MODEL="Qwen/Qwen3-8B-Base" ;;
    8) MODEL="meta-llama/Llama-3.2-1B" ;;
    9) MODEL="meta-llama/Llama-3.2-3B" ;;
    c|C|custom)
        MODEL=$(ask "  Custom HF id" "Qwen/Qwen2.5-0.5B")
        ;;
    *)
        echo "  Unrecognized choice, treating as a custom HF id: $MODEL_CHOICE"
        MODEL="$MODEL_CHOICE"
        ;;
esac
echo "  Selected: $MODEL"
echo ""

TEMPERATURE=$(ask "Sampling temperature" "1.25")
SEED=$(ask "Master seed" "42")

if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS_DETECTED=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS_DETECTED=1
fi
NUM_GPUS=$(ask "Number of GPUs" "$NUM_GPUS_DETECTED")

MODEL_SHORT=$(model_short "$MODEL")
TEMP_STR=$(temp_str "$TEMPERATURE")

SYNTH_DIR="synthetic_data/${MODEL_SHORT}/${TEMP_STR}"
CLEAN_DIR="synthetic_data_clean/${MODEL_SHORT}/${TEMP_STR}_clean"

echo ""

# ============================================================================
# Stage 1: Generate
# ============================================================================
echo "─── Stage 1: Generate ───────────────────────────"
if ask_yn "Run generation?" "y"; then
    DO_GEN=1
    TOTAL_SAMPLES=$(ask "  Total samples" "20000")
    MAX_NEW_TOKENS=$(ask "  Max new tokens per sample" "1024")
    MIN_TOKENS=$(ask "  Min tokens per sample (vLLM min_tokens)" "50")
    PROMPT_KIND=$(ask "  Prompt (bos|random|diverse|creative|document|continue)" "bos")
else
    DO_GEN=0
fi
echo ""

# ============================================================================
# Stage 2: Audit
# ============================================================================
echo "─── Stage 2: Audit ──────────────────────────────"
if ask_yn "Run contamination audit?" "y"; then
    DO_AUDIT=1
    DEFAULT_BENCH="arc_challenge mmlu truthfulqa_mc2 hellaswag gsm8k minerva_math humaneval"
    BENCHMARKS=$(ask "  Benchmarks (space-separated)" "$DEFAULT_BENCH")
    NGRAM_N=$(ask "  N-gram size (8=paper, 13=GPT-3 standard)" "8")
    MIN_CUTOFF=$(ask "  Min n-gram overlap to flag" "1")
    if ask_yn "  Save NLL histograms?" "y"; then
        AUDIT_HIST=1
        AUDIT_BINS=$(ask "    Histogram bins" "50")
    else
        AUDIT_HIST=0
        AUDIT_BINS=50
    fi
    # Adjust clean dir to include n-gram size, matching original convention
    CLEAN_DIR="synthetic_data_clean/${MODEL_SHORT}/${TEMP_STR}_clean_n${NGRAM_N}"
else
    DO_AUDIT=0
fi
echo ""

# ============================================================================
# Stage 3: Subset
# ============================================================================
echo "─── Stage 3: Subset ─────────────────────────────"
if ask_yn "Build training subsets?" "y"; then
    DO_SUBSET=1
    NUM_SUBSETS=$(ask "  Number of subsets" "3")
    TOKENS_PER_SUBSET_INPUT=$(ask "  Tokens per subset (supports 5M, 500K, or raw int)" "5M")
    case "$TOKENS_PER_SUBSET_INPUT" in
        *M|*m) TOKENS_PER_SUBSET=$(python - <<PY
v="$TOKENS_PER_SUBSET_INPUT"
print(int(float(v[:-1]) * 1_000_000))
PY
) ;;
        *K|*k) TOKENS_PER_SUBSET=$(python - <<PY
v="$TOKENS_PER_SUBSET_INPUT"
print(int(float(v[:-1]) * 1_000))
PY
) ;;
        *) TOKENS_PER_SUBSET="$TOKENS_PER_SUBSET_INPUT" ;;
    esac
    # Subsets source: if audit ran, use clean dir; else raw.
    if [ "$DO_AUDIT" = "1" ]; then
        SUBSET_SRC="$CLEAN_DIR"
    else
        SUBSET_SRC="$SYNTH_DIR"
    fi
    SUBSET_SRC=$(ask "  Source corpus dir" "$SUBSET_SRC")
else
    DO_SUBSET=0
fi
echo ""

# ============================================================================
# Stage 4: Train
# ============================================================================
echo "─── Stage 4: Train ──────────────────────────────"
if ask_yn "Train on a subset?" "y"; then
    DO_TRAIN=1
    TRAIN_SUBSET_IDX=$(ask "  Which subset index to train on" "1")
    LR=$(ask "  Learning rate" "1e-6")
    EPOCHS=$(ask "  Epochs" "40")
    BATCH_SIZE_PER_GPU=$(ask "  Per-device train batch size" "8")
    # Default grad_accum targets effective batch = 64 (paper Appendix C),
    # matching the original run_training.sh logic: 64 / (per_device * num_gpus).
    _DEFAULT_GRAD_ACCUM=$(( 64 / (BATCH_SIZE_PER_GPU * NUM_GPUS) ))
    [ "$_DEFAULT_GRAD_ACCUM" -lt 1 ] && _DEFAULT_GRAD_ACCUM=1
    GRAD_ACCUM=$(ask "  Gradient accumulation (effective batch = per_device * grad_accum * num_gpus; paper uses 64)" "$_DEFAULT_GRAD_ACCUM")
    CUTOFF_LEN=$(ask "  Cutoff length (sequence packing)" "2048")
    SAVE_STRATEGY=$(ask "  Save strategy (steps|epoch|no)" "epoch")
    if [ "$SAVE_STRATEGY" = "steps" ]; then
        SAVE_STEPS=$(ask "    Save every N steps" "500")
    fi
    TRAIN_RUN_NAME=$(ask "  Run name" "runs__${MODEL_SHORT}_T${TEMPERATURE//./_}__pt__subset_v${TRAIN_SUBSET_IDX}_lr${LR}_ep${EPOCHS}")
    TRAIN_OUTPUT_DIR="output/${TRAIN_RUN_NAME}"
    # If subset stage is off, ask where subsets already live.
    if [ "$DO_SUBSET" != "1" ]; then
        TRAIN_SUBSET_DIR=$(ask "  Subset dir (contains subset_v*.json)" "$SYNTH_DIR/subsets")
    fi
else
    DO_TRAIN=0
fi
echo ""

# Note: evaluation is in a separate script. After training completes:
#   bash evaluate.sh
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
printf "  Model:         %s  (short=%s)\n" "$MODEL" "$MODEL_SHORT"
printf "  Temperature:   %s  (folder=%s)\n" "$TEMPERATURE" "$TEMP_STR"
printf "  Seed:          %s\n" "$SEED"
printf "  GPUs:          %s\n" "$NUM_GPUS"
echo ""
echo "  Stage 1 Generate:  $([ "$DO_GEN" = 1 ] && echo on || echo off)"
if [ "$DO_GEN" = 1 ]; then
    printf "    samples=%s  max_new_tokens=%s  min_tokens=%s  prompt=%s\n" \
        "$TOTAL_SAMPLES" "$MAX_NEW_TOKENS" "$MIN_TOKENS" "$PROMPT_KIND"
    printf "    output: %s\n" "$SYNTH_DIR"
fi
echo ""
echo "  Stage 2 Audit:     $([ "$DO_AUDIT" = 1 ] && echo on || echo off)"
if [ "$DO_AUDIT" = 1 ]; then
    printf "    n=%s  min_cutoff=%s  histogram=%s\n" \
        "$NGRAM_N" "$MIN_CUTOFF" "$([ "$AUDIT_HIST" = 1 ] && echo yes || echo no)"
    printf "    benchmarks: %s\n" "$BENCHMARKS"
    printf "    output: %s\n" "$CLEAN_DIR"
fi
echo ""
echo "  Stage 3 Subset:    $([ "$DO_SUBSET" = 1 ] && echo on || echo off)"
if [ "$DO_SUBSET" = 1 ]; then
    printf "    num=%s  tokens/subset=%s\n" \
        "$NUM_SUBSETS" "$TOKENS_PER_SUBSET"
    printf "    source: %s\n" "$SUBSET_SRC"
fi
echo ""
echo "  Stage 4 Train:     $([ "$DO_TRAIN" = 1 ] && echo on || echo off)"
if [ "$DO_TRAIN" = 1 ]; then
    _EFF_BATCH=$(( BATCH_SIZE_PER_GPU * GRAD_ACCUM * NUM_GPUS ))
    printf "    subset_v=%s  lr=%s  epochs=%s  cutoff=%s\n" \
        "$TRAIN_SUBSET_IDX" "$LR" "$EPOCHS" "$CUTOFF_LEN"
    printf "    batch: per_device=%s * grad_accum=%s * gpus=%s = effective %s\n" \
        "$BATCH_SIZE_PER_GPU" "$GRAD_ACCUM" "$NUM_GPUS" "$_EFF_BATCH"
    printf "    run name: %s\n" "$TRAIN_RUN_NAME"
fi
echo ""
echo "=========================================="
echo ""

if ! ask_yn "Run the pipeline now?" "y"; then
    echo "Aborted before run."
    exit 0
fi
echo ""

# ============================================================================
# RUN
# ============================================================================

# ---- Stage 1 ---------------------------------------------------------------
if [ "$DO_GEN" = "1" ]; then
    echo "================ Stage 1: Generate ================"
    DECISION=$(decide_existing "Generate" "$SYNTH_DIR/texts.json")
    case "$DECISION" in
        ABORT) echo "Aborted."; exit 1 ;;
        REUSE) echo "  reusing existing corpus at $SYNTH_DIR" ;;
        RUN|RERUN)
            if [ "$DECISION" = "RERUN" ]; then
                echo "  removing existing $SYNTH_DIR"
                rm -rf "$SYNTH_DIR"
            fi
            mkdir -p "$SYNTH_DIR"
            echo "  launching $NUM_GPUS GPU workers..."
            PIDS=()
            for gid in $(seq 0 $((NUM_GPUS - 1))); do
                CUDA_VISIBLE_DEVICES="$gid" python src/generate_synthetic.py \
                    --model "$MODEL" \
                    --temperature "$TEMPERATURE" \
                    --total_samples "$TOTAL_SAMPLES" \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --min_tokens "$MIN_TOKENS" \
                    --prompt "$PROMPT_KIND" \
                    --gpu_id "$gid" \
                    --num_gpus "$NUM_GPUS" \
                    > "$SYNTH_DIR/gen_gpu_${gid}.log" 2>&1 &
                PIDS+=($!)
                echo "    GPU $gid -> PID ${PIDS[-1]}  (log: $SYNTH_DIR/gen_gpu_${gid}.log)"
            done
            FAIL=0
            for pid in "${PIDS[@]}"; do
                if ! wait "$pid"; then
                    FAIL=1
                fi
            done
            if [ "$FAIL" = "1" ]; then
                echo "ERROR: one or more generation workers failed. See logs in $SYNTH_DIR/" >&2
                exit 1
            fi
            echo "  merging shards..."
            python src/generate_synthetic.py \
                --model "$MODEL" \
                --temperature "$TEMPERATURE" \
                --num_gpus "$NUM_GPUS" \
                --merge_only
            ;;
    esac
    echo ""
fi

# ---- Stage 2 ---------------------------------------------------------------
if [ "$DO_AUDIT" = "1" ]; then
    echo "================ Stage 2: Audit ==================="
    DECISION=$(decide_existing "Audit" "$CLEAN_DIR/train.json")
    case "$DECISION" in
        ABORT) echo "Aborted."; exit 1 ;;
        REUSE) echo "  reusing existing clean corpus at $CLEAN_DIR" ;;
        RUN|RERUN)
            if [ "$DECISION" = "RERUN" ]; then
                rm -rf "$CLEAN_DIR"
            fi
            HIST_FLAG=""
            [ "$AUDIT_HIST" = "1" ] && HIST_FLAG="--save_histogram --num_bins $AUDIT_BINS"
            # shellcheck disable=SC2086
            python src/audit.py \
                --corpus_dir "$SYNTH_DIR" \
                --benchmarks $BENCHMARKS \
                --n "$NGRAM_N" \
                --min_cutoff "$MIN_CUTOFF" \
                --output_dir "$CLEAN_DIR" \
                $HIST_FLAG
            ;;
    esac
    echo ""
fi

# ---- Stage 3 ---------------------------------------------------------------
SUBSET_OUT_DIR=""
if [ "$DO_SUBSET" = "1" ]; then
    echo "================ Stage 3: Subset =================="
    SUBSET_OUT_DIR="$SUBSET_SRC/subsets"
    DECISION=$(decide_existing "Subset" "$SUBSET_OUT_DIR/summary.json")
    case "$DECISION" in
        ABORT) echo "Aborted."; exit 1 ;;
        REUSE) echo "  reusing existing subsets at $SUBSET_OUT_DIR" ;;
        RUN|RERUN)
            if [ "$DECISION" = "RERUN" ]; then
                rm -rf "$SUBSET_OUT_DIR"
            fi
            python src/create_subsets.py \
                --data_dir "$SUBSET_SRC" \
                --num_subsets "$NUM_SUBSETS" \
                --tokens_per_subset "$TOKENS_PER_SUBSET" \
                --seed "$SEED"
            ;;
    esac
    echo ""
fi

# ---- Stage 4 ---------------------------------------------------------------
if [ "$DO_TRAIN" = "1" ]; then
    echo "================ Stage 4: Train ==================="
    if ! command -v llamafactory-cli >/dev/null 2>&1; then
        echo "ERROR: llamafactory-cli not found. Run setup.sh first." >&2
        exit 1
    fi
    DECISION=$(decide_existing "Train" "$TRAIN_OUTPUT_DIR")
    case "$DECISION" in
        ABORT) echo "Aborted."; exit 1 ;;
        REUSE) echo "  reusing existing run at $TRAIN_OUTPUT_DIR" ;;
        RUN|RERUN)
            if [ "$DECISION" = "RERUN" ]; then
                rm -rf "$TRAIN_OUTPUT_DIR"
            fi
            mkdir -p "$TRAIN_OUTPUT_DIR"
            # Resolve where the training subsets live.
            if [ "$DO_SUBSET" = "1" ]; then
                _SUBSET_DIR="$SUBSET_OUT_DIR"
            else
                _SUBSET_DIR="$TRAIN_SUBSET_DIR"
            fi
            TRAIN_FILE="$_SUBSET_DIR/subset_v${TRAIN_SUBSET_IDX}.json"
            if [ ! -f "$TRAIN_FILE" ]; then
                echo "ERROR: training file not found: $TRAIN_FILE" >&2
                echo "  (looked in: $_SUBSET_DIR)" >&2
                exit 1
            fi

            # llamafactory expects datasets to be registered in dataset_info.json.
            # We register the subset file dynamically.
            DATASET_NAME="lcr_${MODEL_SHORT}_${TEMP_STR}_subset_v${TRAIN_SUBSET_IDX}"
            DATASET_INFO_DIR="$REPO_ROOT/data"
            mkdir -p "$DATASET_INFO_DIR"
            cp "$TRAIN_FILE" "$DATASET_INFO_DIR/${DATASET_NAME}.json"
            python - <<PY
import json
from pathlib import Path
info_path = Path("$DATASET_INFO_DIR/dataset_info.json")
info = json.loads(info_path.read_text()) if info_path.exists() else {}
info["$DATASET_NAME"] = {
    "file_name": "${DATASET_NAME}.json",
    "columns": {"prompt": "text"}
}
info_path.write_text(json.dumps(info, indent=2))
print(f"registered dataset: $DATASET_NAME")
PY

            SAVE_ARGS="save_strategy: $SAVE_STRATEGY"
            [ "$SAVE_STRATEGY" = "steps" ] && SAVE_ARGS+=$'\n'"save_steps: $SAVE_STEPS"

            CONFIG_FILE="$TRAIN_OUTPUT_DIR/config.yaml"
            cat > "$CONFIG_FILE" <<YAML
# Generated by pipeline.sh
model_name_or_path: $MODEL
trust_remote_code: true

stage: pt
do_train: true
finetuning_type: full

dataset_dir: $DATASET_INFO_DIR
dataset: $DATASET_NAME
template: empty
cutoff_len: $CUTOFF_LEN
overwrite_cache: true
preprocessing_num_workers: 16
packing: true

output_dir: $TRAIN_OUTPUT_DIR
overwrite_output_dir: true
logging_steps: 10
$SAVE_ARGS
save_only_model: true
plot_loss: true
report_to: none

per_device_train_batch_size: $BATCH_SIZE_PER_GPU
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: $LR
num_train_epochs: $EPOCHS
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

deepspeed: $REPO_ROOT/configs/ds_z3.json
YAML
            echo "  config written: $CONFIG_FILE"
            GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
            echo "  launching: CUDA_VISIBLE_DEVICES=$GPU_IDS llamafactory-cli train $CONFIG_FILE"
            CUDA_VISIBLE_DEVICES="$GPU_IDS" llamafactory-cli train "$CONFIG_FILE"
            ;;
    esac
    echo ""
fi

echo "=========================================="
echo "Pipeline complete."
[ "$DO_GEN"    = "1" ] && echo "  Generation:  $SYNTH_DIR"
[ "$DO_AUDIT"  = "1" ] && echo "  Audit:       $CLEAN_DIR"
[ "$DO_SUBSET" = "1" ] && echo "  Subsets:     $SUBSET_OUT_DIR"
[ "$DO_TRAIN"  = "1" ] && echo "  Training:    $TRAIN_OUTPUT_DIR"
echo ""
echo "Next: evaluate the checkpoints with:"
echo "    bash evaluate.sh"
echo "=========================================="
