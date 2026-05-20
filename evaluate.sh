#!/usr/bin/env bash
# ============================================================================
# Standalone evaluation script.
#
# Scans output/ for training runs and their checkpoints, lets you pick what
# to evaluate, runs lm-evaluation-harness via vLLM, writes results to evals/.
#
# Reads the same conda env as pipeline.sh — activate it first:
#   conda activate <name>
#   bash evaluate.sh
# ============================================================================
set -uo pipefail   # NOT -e: we tolerate per-job failures and keep going

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Env sanity check
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: no conda env or venv is active." >&2
    read -r -p "Continue anyway? (y/N): " _CONT
    [[ ! "${_CONT:-n}" =~ ^[Yy]$ ]] && { echo "Activate your env and re-run."; exit 1; }
fi

# ---- Determinism ------------------------------------------------------------
# These mirror your original script. Some flags reduce speed but make the
# numbers reproducible across runs.
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC_ALGORITHMS=1
export VLLM_DETERMINISTIC_OPS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
SEED=42

# ---- Small helpers ----------------------------------------------------------
ask() {
    local prompt="$1" default="$2" answer
    read -r -p "$prompt [$default]: " answer
    echo "${answer:-$default}"
}
ask_yn() {
    local prompt="$1" default="$2" answer
    if [ "$default" = "y" ]; then
        read -r -p "$prompt (Y/n): " answer; answer="${answer:-y}"
    else
        read -r -p "$prompt (y/N): " answer; answer="${answer:-n}"
    fi
    [[ "$answer" =~ ^[Yy] ]]
}

echo "=========================================="
echo "  LCR Evaluation"
echo "=========================================="
echo ""

# ============================================================================
# Step 1: Benchmark selection
# ============================================================================
echo "Step 1: Benchmarks"
echo "------------------"
echo ""
echo "  REASONING / KNOWLEDGE:"
echo "    1.  arc_challenge"
echo "    2.  arc_easy"
echo "    3.  mmlu"
echo "    4.  truthfulqa_mc2"
echo "    5.  hellaswag"
echo "    6.  winogrande"
echo ""
echo "  MATH:"
echo "    7.  gsm8k"
echo "    8.  minerva_math"
echo ""
echo "  CODE:"
echo "    9.  humaneval"
echo ""
echo "  SUITES:"
echo "    A.  Suite A (zero-shot)     - mmlu, arc_challenge, gsm8k, hellaswag,"
echo "                                  truthfulqa, minerva_math, humaneval"
echo "    B.  Suite B (standard shot) - arc_challenge(25), gsm8k(8),"
echo "                                  hellaswag(5), minerva_math(4)"
echo "    AB. Suite A + Suite B (both shot counts where applicable)"
echo "    P.  Paper core 7  (= Suite A)"
echo ""
read -r -p "Select (1-9, A, B, AB, P, or comma-separated, default=P): " BENCH_INPUT
BENCH_INPUT="${BENCH_INPUT:-P}"

# Benchmark display names
declare -A BENCH_DISPLAY
BENCH_DISPLAY["arc_challenge"]="ARC-Challenge"
BENCH_DISPLAY["arc_easy"]="ARC-Easy"
BENCH_DISPLAY["mmlu"]="MMLU"
BENCH_DISPLAY["truthfulqa_mc2"]="TruthfulQA"
BENCH_DISPLAY["hellaswag"]="HellaSwag"
BENCH_DISPLAY["winogrande"]="WinoGrande"
BENCH_DISPLAY["gsm8k"]="GSM8K"
BENCH_DISPLAY["minerva_math"]="MATH"
BENCH_DISPLAY["humaneval"]="HumanEval"

# (benchmark, fewshot) pairs to actually run. Stored as "bench||fs" strings.
EVAL_PAIRS=()

add_pair() {
    local bench="$1" fs="$2"
    local key="${bench}||${fs}"
    for existing in "${EVAL_PAIRS[@]}"; do
        [ "$existing" = "$key" ] && return
    done
    EVAL_PAIRS+=("$key")
}

# Default few-shot when user picks an individual benchmark
# (Suite A and Suite B override this with fixed counts.)
DEFAULT_FS=0

parse_benchmark_input() {
    local input="$1"
    case "$input" in
        A|a)
            # Suite A: zero-shot, all 7 paper benchmarks
            for b in mmlu arc_challenge gsm8k hellaswag truthfulqa_mc2 minerva_math humaneval; do
                add_pair "$b" 0
            done
            ;;
        B|b)
            # Suite B: standard few-shot
            add_pair arc_challenge 25
            add_pair gsm8k 8
            add_pair hellaswag 5
            add_pair minerva_math 4
            ;;
        AB|ab|Ab|aB)
            # Suite A + Suite B
            for b in mmlu arc_challenge gsm8k hellaswag truthfulqa_mc2 minerva_math humaneval; do
                add_pair "$b" 0
            done
            add_pair arc_challenge 25
            add_pair gsm8k 8
            add_pair hellaswag 5
            add_pair minerva_math 4
            ;;
        P|p|paper)
            # Paper core 7 == Suite A
            for b in mmlu arc_challenge gsm8k hellaswag truthfulqa_mc2 minerva_math humaneval; do
                add_pair "$b" 0
            done
            ;;
        *)
            # Comma-separated numbers; ask for few-shot
            IFS=',' read -ra PARTS <<< "$input"
            local picked=()
            for p in "${PARTS[@]}"; do
                p=$(echo "$p" | tr -d ' ')
                case "$p" in
                    1) picked+=(arc_challenge) ;;
                    2) picked+=(arc_easy) ;;
                    3) picked+=(mmlu) ;;
                    4) picked+=(truthfulqa_mc2) ;;
                    5) picked+=(hellaswag) ;;
                    6) picked+=(winogrande) ;;
                    7) picked+=(gsm8k) ;;
                    8) picked+=(minerva_math) ;;
                    9) picked+=(humaneval) ;;
                esac
            done
            if [ ${#picked[@]} -eq 0 ]; then
                echo "ERROR: no benchmarks selected." >&2
                exit 1
            fi
            FS_INPUT=$(ask "Few-shot (single number, or comma list like 0,5)" "0")
            local fs_vals=()
            IFS=',' read -ra FSP <<< "$FS_INPUT"
            for fs in "${FSP[@]}"; do
                fs=$(echo "$fs" | tr -d ' ')
                [[ "$fs" =~ ^[0-9]+$ ]] && fs_vals+=("$fs")
            done
            [ ${#fs_vals[@]} -eq 0 ] && fs_vals=(0)
            for b in "${picked[@]}"; do
                for fs in "${fs_vals[@]}"; do
                    add_pair "$b" "$fs"
                done
            done
            ;;
    esac
}

parse_benchmark_input "$BENCH_INPUT"
[ ${#EVAL_PAIRS[@]} -eq 0 ] && { echo "ERROR: nothing to evaluate."; exit 1; }

# Check if humaneval is in the selection -> ask for code-specific gen params
HAS_HUMANEVAL=0
for pair in "${EVAL_PAIRS[@]}"; do
    [ "${pair%%||*}" = "humaneval" ] && HAS_HUMANEVAL=1
done
HE_TEMP="0.2"; HE_N_SAMPLES="10"; HE_MAX_TOKENS="512"
if [ "$HAS_HUMANEVAL" = "1" ]; then
    echo ""
    echo "HumanEval generation parameters:"
    HE_TEMP=$(ask "  Temperature (0 = greedy, single sample)" "0.2")
    if python3 -c "import sys; sys.exit(0 if float('$HE_TEMP') > 0 else 1)"; then
        HE_N_SAMPLES=$(ask "  Samples per problem (pass@1 averaged over N)" "10")
    else
        HE_N_SAMPLES="1"
    fi
    HE_MAX_TOKENS=$(ask "  Max generation tokens" "512")
    echo ""
fi

echo ""
echo "Selected (benchmark, few-shot):"
for pair in "${EVAL_PAIRS[@]}"; do
    b="${pair%%||*}"; fs="${pair##*||}"
    printf "  %-20s  %s-shot\n" "${BENCH_DISPLAY[$b]}" "$fs"
done
echo ""

# ============================================================================
# Step 2: Scan output/ for training runs
# ============================================================================
echo "Step 2: Training runs"
echo "----------------------"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: $OUTPUT_DIR/ not found. (Set OUTPUT_DIR env var if your runs live elsewhere.)" >&2
    exit 1
fi

RUN_PATHS=()
RUN_NAMES=()
RUN_BASE_MODELS=()       # HF id inferred from the run's config.json or training_args.bin
RUN_CHECKPOINT_COUNTS=()

# Try to recover the base model HF id from the run dir. llamafactory writes
# config.json with `_name_or_path`, which is the base model.
recover_base_model() {
    local rdir="$1" hf
    # First look at the run dir itself
    if [ -f "$rdir/config.json" ]; then
        hf=$(python3 -c "import json; print(json.load(open('$rdir/config.json')).get('_name_or_path',''))" 2>/dev/null || true)
        [ -n "$hf" ] && [ "$hf" != "null" ] && echo "$hf" && return
    fi
    # Then the first checkpoint
    local first
    first=$(ls -d "$rdir"/checkpoint-* 2>/dev/null | sort -V | head -n1)
    if [ -n "$first" ] && [ -f "$first/config.json" ]; then
        hf=$(python3 -c "import json; print(json.load(open('$first/config.json')).get('_name_or_path',''))" 2>/dev/null || true)
        [ -n "$hf" ] && [ "$hf" != "null" ] && echo "$hf" && return
    fi
    echo ""
}

idx=0
echo ""
for run in "$OUTPUT_DIR"/*/; do
    [ -d "$run" ] || continue
    rname=$(basename "$run")
    cp_count=$(ls -d "$run"checkpoint-* 2>/dev/null | wc -l)
    [ "$cp_count" -eq 0 ] && continue
    base=$(recover_base_model "$run")
    RUN_PATHS+=("$run")
    RUN_NAMES+=("$rname")
    RUN_BASE_MODELS+=("$base")
    RUN_CHECKPOINT_COUNTS+=("$cp_count")
    idx=$((idx + 1))
    base_disp="${base:-?}"
    printf "  %d.  %-60s  cp=%-3d  base=%s\n" "$idx" "$rname" "$cp_count" "$base_disp"
done

if [ ${#RUN_PATHS[@]} -eq 0 ]; then
    echo "  no training runs with checkpoints under $OUTPUT_DIR/"
    echo "  (only base-model eval is possible)"
fi
echo ""

# ============================================================================
# Step 3: Run selection
# ============================================================================
echo "Step 3: Select runs"
echo "--------------------"
if [ ${#RUN_PATHS[@]} -gt 0 ]; then
    echo "  comma-separated (e.g. 1,3) or range (1-3) or 'all' or 'none' (base-only)"
    SEL=$(ask "Select" "all")
else
    SEL="none"
fi

SELECTED_INDICES=()
case "$SEL" in
    none|NONE) ;;
    all|ALL)
        for ((i=0; i<${#RUN_PATHS[@]}; i++)); do SELECTED_INDICES+=("$i"); done
        ;;
    *)
        IFS=',' read -ra PARTS <<< "$SEL"
        for part in "${PARTS[@]}"; do
            part=$(echo "$part" | tr -d ' ')
            if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for ((j=${BASH_REMATCH[1]}; j<=${BASH_REMATCH[2]}; j++)); do
                    [ "$j" -ge 1 ] && [ "$j" -le "${#RUN_PATHS[@]}" ] && SELECTED_INDICES+=("$((j-1))")
                done
            elif [[ "$part" =~ ^[0-9]+$ ]]; then
                [ "$part" -ge 1 ] && [ "$part" -le "${#RUN_PATHS[@]}" ] && SELECTED_INDICES+=("$((part-1))")
            fi
        done
        ;;
esac
echo ""

# ============================================================================
# Step 4: Checkpoint selection
# ============================================================================
CKPT_MODE="all"; CKPT_STEP=""; CKPT_SPECIFIC=""
if [ ${#SELECTED_INDICES[@]} -gt 0 ]; then
    echo "Step 4: Checkpoints per run"
    echo "----------------------------"
    echo "  1.  all          - every checkpoint-*"
    echo "  2.  latest       - last checkpoint only"
    echo "  3.  every N      - keep one every N (sorted by step)"
    echo "  4.  specific     - by step number (e.g. 50,100,200)"
    CKPT_OPT=$(ask "Mode" "1")
    case "$CKPT_OPT" in
        2) CKPT_MODE="latest" ;;
        3) CKPT_STEP=$(ask "  Every N" "5"); CKPT_MODE="every_n" ;;
        4) CKPT_SPECIFIC=$(ask "  Steps (comma-separated)" "100,200"); CKPT_MODE="specific" ;;
        *) CKPT_MODE="all" ;;
    esac
    echo ""
fi

resolve_checkpoints() {
    local rpath="$1"
    case "$CKPT_MODE" in
        latest)   ls -d "$rpath"checkpoint-* 2>/dev/null | sort -V | tail -1 ;;
        every_n)
            local c=0
            for cp in $(ls -d "$rpath"checkpoint-* 2>/dev/null | sort -V); do
                [ $((c % CKPT_STEP)) -eq 0 ] && echo "$cp"
                c=$((c + 1))
            done
            ;;
        specific)
            IFS=',' read -ra NUMS <<< "$CKPT_SPECIFIC"
            for n in "${NUMS[@]}"; do
                [ -d "${rpath}checkpoint-${n}" ] && echo "${rpath}checkpoint-${n}"
            done
            ;;
        *) ls -d "$rpath"checkpoint-* 2>/dev/null | sort -V ;;
    esac
}

# ============================================================================
# Step 5: Base model eval
# ============================================================================
echo "Step 5: Base model"
echo "-------------------"
if ask_yn "Also evaluate base model(s)?" "y"; then
    EVAL_BASE=1
else
    EVAL_BASE=0
fi
echo ""

# Collect unique base models we need to evaluate
declare -A UNIQUE_BASE
if [ "$EVAL_BASE" = "1" ]; then
    for idx in "${SELECTED_INDICES[@]}"; do
        b="${RUN_BASE_MODELS[$idx]}"
        [ -n "$b" ] && UNIQUE_BASE["$b"]=1
    done
    # If user picked no runs but still wants base eval, ask for one
    if [ ${#UNIQUE_BASE[@]} -eq 0 ]; then
        EXTRA_BASE=$(ask "Base model HF id" "Qwen/Qwen2.5-0.5B")
        UNIQUE_BASE["$EXTRA_BASE"]=1
    fi
fi

# ============================================================================
# Step 6: GPUs
# ============================================================================
echo "Step 6: GPUs"
echo "-------------"
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED=$(nvidia-smi -L | wc -l)
else
    DETECTED=1
fi
NUM_GPUS=$(ask "How many GPUs to use" "$DETECTED")
GPU_IDS_STR=$(seq -s, 0 $((NUM_GPUS - 1)))
GPU_IDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
echo "  using GPUs: $GPU_IDS_STR"
echo ""

# ============================================================================
# Build the job list
# ============================================================================
EVAL_ROOT="evals"
mkdir -p "$EVAL_ROOT"

# Each job: model_path | output_dir | label | benchmark | fewshot
JOB_MODEL=(); JOB_OUT=(); JOB_LABEL=(); JOB_BENCH=(); JOB_FS=()

short_name() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]'; }

add_job() {
    JOB_MODEL+=("$1"); JOB_OUT+=("$2"); JOB_LABEL+=("$3")
    JOB_BENCH+=("$4"); JOB_FS+=("$5")
}

# Base model jobs
for hf in "${!UNIQUE_BASE[@]}"; do
    short=$(short_name "$hf")
    for pair in "${EVAL_PAIRS[@]}"; do
        b="${pair%%||*}"; fs="${pair##*||}"
        out="$EVAL_ROOT/${short}/base/${b}_fs${fs}"
        add_job "$hf" "$out" "BASE/${short}/${b}-fs${fs}" "$b" "$fs"
    done
done

# Checkpoint jobs
for idx in "${SELECTED_INDICES[@]}"; do
    rpath="${RUN_PATHS[$idx]}"
    rname="${RUN_NAMES[$idx]}"
    base="${RUN_BASE_MODELS[$idx]}"
    short=$(short_name "${base:-unknown}")
    while IFS= read -r ckpt; do
        [ -z "$ckpt" ] && continue
        cp_name=$(basename "$ckpt")
        for pair in "${EVAL_PAIRS[@]}"; do
            b="${pair%%||*}"; fs="${pair##*||}"
            out="$EVAL_ROOT/${short}/${rname}/${cp_name}/${b}_fs${fs}"
            add_job "$ckpt" "$out" "${rname}/${cp_name}/${b}-fs${fs}" "$b" "$fs"
        done
    done < <(resolve_checkpoints "$rpath")
done

TOTAL_INIT=${#JOB_MODEL[@]}
if [ "$TOTAL_INIT" -eq 0 ]; then
    echo "Nothing to evaluate."
    exit 0
fi

# ============================================================================
# Filter jobs whose results already exist
# ============================================================================
results_exist() {
    local out="$1"
    [ -e "$out/results.json" ] && return 0
    # lm_eval typically writes results_<timestamp>.json nested inside a model-named dir
    find "$out" -maxdepth 4 -name "results_*.json" 2>/dev/null | head -n1 | grep -q . && return 0
    return 1
}

F_MODEL=(); F_OUT=(); F_LABEL=(); F_BENCH=(); F_FS=()
SKIPPED=0
for ((i=0; i<TOTAL_INIT; i++)); do
    if results_exist "${JOB_OUT[$i]}"; then
        SKIPPED=$((SKIPPED + 1))
    else
        F_MODEL+=("${JOB_MODEL[$i]}"); F_OUT+=("${JOB_OUT[$i]}")
        F_LABEL+=("${JOB_LABEL[$i]}"); F_BENCH+=("${JOB_BENCH[$i]}"); F_FS+=("${JOB_FS[$i]}")
    fi
done
JOB_MODEL=("${F_MODEL[@]}"); JOB_OUT=("${F_OUT[@]}"); JOB_LABEL=("${F_LABEL[@]}")
JOB_BENCH=("${F_BENCH[@]}"); JOB_FS=("${F_FS[@]}")
TOTAL=${#JOB_MODEL[@]}

# ============================================================================
# Summary + confirm
# ============================================================================
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo "  Total requested:  $TOTAL_INIT"
echo "  Already complete: $SKIPPED"
echo "  Remaining:        $TOTAL"
echo "  GPUs:             $NUM_GPUS ($GPU_IDS_STR)"
if [ "$HAS_HUMANEVAL" = "1" ]; then
    echo "  HumanEval:        T=$HE_TEMP  n_samples=$HE_N_SAMPLES  max_gen_toks=$HE_MAX_TOKENS"
fi
echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "Everything already evaluated."
    exit 0
fi
echo "  First few:"
for ((i=0; i<TOTAL && i<5; i++)); do echo "    ${JOB_LABEL[$i]}"; done
[ "$TOTAL" -gt 5 ] && echo "    ... and $((TOTAL - 5)) more"
echo "=========================================="
echo ""
if ! ask_yn "Run evaluation now?" "y"; then
    echo "Aborted."; exit 0
fi
echo ""

# ============================================================================
# Run jobs in batches across GPUs.
# Group by (benchmark, fewshot) so all GPUs in a wave run the same task —
# avoids slow-task stragglers blocking the others.
# ============================================================================
declare -A GROUP_BY_KEY
GROUP_KEYS=()
for ((i=0; i<TOTAL; i++)); do
    key="${JOB_BENCH[$i]}||${JOB_FS[$i]}"
    if [ -z "${GROUP_BY_KEY[$key]+x}" ]; then
        GROUP_BY_KEY[$key]="$i"
        GROUP_KEYS+=("$key")
    else
        GROUP_BY_KEY[$key]="${GROUP_BY_KEY[$key]} $i"
    fi
done

run_one_job() {
    local gpu="$1" model="$2" outdir="$3" bench="$4" fs="$5" label="$6"
    mkdir -p "$outdir"
    local log="$outdir/eval.log"

    # vLLM backend, deterministic seed. Use auto batch and bf16.
    local model_args="pretrained=$model,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.75,max_model_len=4096,seed=${SEED}"

    # Per-benchmark generation args:
    #   humaneval is generation-based -> use sampling at T (default 0.2) with
    #   N samples per problem (pass@1 averaged), and request enough max_gen_toks.
    #   Also requires --confirm_run_unsafe_code (executes generated code in sandbox)
    #   and HF_ALLOW_CODE_EVAL=1 environment variable.
    #   Everything else is loglikelihood/greedy.
    local gen_kwargs unsafe_flag extra_env
    if [ "$bench" = "humaneval" ]; then
        if python3 -c "import sys; sys.exit(0 if float('$HE_TEMP') > 0 else 1)"; then
            gen_kwargs="do_sample=True,temperature=${HE_TEMP},top_p=0.95,num_samples=${HE_N_SAMPLES},max_gen_toks=${HE_MAX_TOKENS}"
        else
            gen_kwargs="do_sample=False,temperature=0,num_samples=1,max_gen_toks=${HE_MAX_TOKENS}"
        fi
        unsafe_flag="--confirm_run_unsafe_code"
        extra_env="HF_ALLOW_CODE_EVAL=1"
    else
        gen_kwargs="temperature=0,top_p=1.0"
        unsafe_flag=""
        extra_env=""
    fi

    echo "  GPU $gpu -> $label"
    env CUDA_VISIBLE_DEVICES="$gpu" $extra_env lm_eval \
        --model vllm \
        --model_args "$model_args" \
        --tasks "$bench" \
        --num_fewshot "$fs" \
        --batch_size auto \
        --gen_kwargs "$gen_kwargs" \
        --seed "$SEED" \
        --output_path "$outdir" \
        $unsafe_flag \
        > "$log" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        # lm_eval writes results_<timestamp>.json nested; symlink to a stable name
        local found
        found=$(find "$outdir" -maxdepth 4 -name "results_*.json" 2>/dev/null | head -n1)
        [ -n "$found" ] && [ ! -e "$outdir/results.json" ] && ln -sf "$found" "$outdir/results.json"
        echo "  GPU $gpu DONE  $label"
    else
        echo "  GPU $gpu FAIL  $label  (rc=$rc, log: $log)"
    fi
}

group_num=0
total_groups=${#GROUP_KEYS[@]}
for key in "${GROUP_KEYS[@]}"; do
    group_num=$((group_num + 1))
    bench="${key%%||*}"; fs="${key##*||}"
    INDICES=(${GROUP_BY_KEY[$key]})
    g_size=${#INDICES[@]}
    echo ""
    echo "=== Group $group_num/$total_groups: ${BENCH_DISPLAY[$bench]} (${fs}-shot)  —  $g_size jobs ==="

    waves=$(( (g_size + NUM_GPUS - 1) / NUM_GPUS ))
    for ((w=0; w<waves; w++)); do
        start=$((w * NUM_GPUS))
        end=$((start + NUM_GPUS - 1))
        [ "$end" -ge "$g_size" ] && end=$((g_size - 1))
        echo "  Wave $((w+1))/$waves:"
        PIDS=()
        slot=0
        for ((k=start; k<=end; k++)); do
            i="${INDICES[$k]}"
            gpu="${GPU_IDS[$slot]}"
            run_one_job "$gpu" "${JOB_MODEL[$i]}" "${JOB_OUT[$i]}" \
                        "${JOB_BENCH[$i]}" "${JOB_FS[$i]}" "${JOB_LABEL[$i]}" &
            PIDS+=($!)
            slot=$((slot + 1))
        done
        for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    done

    # ---- Group summary: parse results.json from each job's output dir ----
    echo ""
    echo "  --- Results for ${BENCH_DISPLAY[$bench]} (${fs}-shot) ---"
    {
        for idx in "${INDICES[@]}"; do
            printf '%s\t%s\t%s\n' "${JOB_LABEL[$idx]}" "${JOB_OUT[$idx]}" "$bench"
        done
    } | python3 -c '
import json, sys
from pathlib import Path

# Metrics to look for per task, in priority order. lm-eval names them
# slightly differently across tasks (some have _norm variants, etc).
PRIORITY = {
    "arc_challenge":   ["acc_norm,none", "acc,none"],
    "arc_easy":        ["acc_norm,none", "acc,none"],
    "hellaswag":       ["acc_norm,none", "acc,none"],
    "winogrande":      ["acc,none"],
    "mmlu":            ["acc,none"],
    "truthfulqa_mc2":  ["acc,none"],
    "gsm8k":           ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"],
    "minerva_math":    ["exact_match,none", "exact_match,strict-match"],
    "humaneval":       ["pass@1,create_test", "pass@1,none"],
}

def find_results_file(outdir: Path):
    if (outdir / "results.json").exists():
        return outdir / "results.json"
    cands = list(outdir.rglob("results_*.json"))
    return cands[0] if cands else None

print("  %-70s %10s  %8s  %8s" % ("job", "metric", "value", "stderr"))
print("  " + "-" * 102)
for line in sys.stdin:
    label, outdir, bench = line.rstrip("\n").split("\t")
    rf = find_results_file(Path(outdir))
    if rf is None:
        print("  %-70s %10s  %8s  %8s" % (label[:70], "-", "no file", "-"))
        continue
    try:
        data = json.loads(rf.read_text())
    except Exception as e:
        print("  %-70s %10s  %8s  %8s" % (label[:70], "-", "parse?", "-"))
        continue
    results = data.get("results", {})
    # lm-eval keys tasks by name; for things like mmlu it may have sub-tasks
    # but the aggregate is under the bench name.
    task_data = results.get(bench, {})
    if not task_data:
        # try first key
        if results:
            task_data = next(iter(results.values()))
    val = "-"; err = "-"; chosen = "-"
    for key in PRIORITY.get(bench, []):
        if key in task_data:
            v = task_data[key]
            chosen = key.split(",")[0]
            val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            stderr_key = key.replace(",", "_stderr,", 1)
            if stderr_key in task_data and isinstance(task_data[stderr_key], (int, float)):
                err = f"{task_data[stderr_key]:.4f}"
            break
    print("  %-70s %10s  %8s  %8s" % (label[:70], chosen, val, err))
'
    echo ""
done

echo ""
echo "=========================================="
echo "All evaluations finished."
echo "Results under: $EVAL_ROOT/"
echo "=========================================="
