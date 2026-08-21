#!/bin/bash
#SBATCH --job-name=skill-native
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/%x-%j.err
#
# run_native.sbatch — single-model benchmark on a compute node WITHOUT
# Singularity. Replaces the singularity-exec wrapping in
# anvil_run_model.sbatch + run_inside_container.sh.
#
# What it does:
#   1. (idempotent) create/activate conda env at $SCRATCH/skill/.conda/vllm
#      with vllm + chromadb + sentence-transformers + openclaude.
#   2. start `vllm serve` on the assigned GPUs (TP=2 default).
#   3. wait for vllm /v1/models to be ready.
#   4. generate self_gen skill if missing for this model.
#   5. run_benchmark.py with three conditions (no_skill, with_skill, self_gen)
#      and k=1 trial per task.
#
# Required env (set via sbatch --export when calling, or shell):
#   MODEL_HF        HuggingFace repo id (e.g., Qwen/Qwen3.6-27B)
#   MODEL_PARSER    tool-call parser hint (hermes / qwen3_coder / llama3_json / gemma)
#   ANVIL_USER      defaults to $USER
#
# Optional:
#   TP_SIZE              default 2
#   BENCH_WORKERS        default 14
#   VAST_BENCH_LABEL     auto-generated if absent
set -uo pipefail
trap 'cleanup; exit 130' INT TERM

USER_PATH="/anvil/scratch/$USER"
PROJECT_ROOT="$USER_PATH/skill"
CONDA_ENV_DIR="$PROJECT_ROOT/.conda/vllm"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

: "${MODEL_HF:?MODEL_HF must be set, e.g., Qwen/Qwen3.6-27B}"
: "${MODEL_PARSER:?MODEL_PARSER must be set (hermes/qwen3_coder/llama3_json/gemma)}"

LABEL="${VAST_BENCH_LABEL:-anvil_native_$(echo "$MODEL_HF" | tr '/.-' '___')_$(date -u +%Y%m%d_%H%M)}"
TP_SIZE="${TP_SIZE:-2}"
BENCH_WORKERS="${BENCH_WORKERS:-14}"

VLLM_PORT=8001
PROXY_PORT=8080

VLLM_LOG="$LOG_DIR/vllm-$SLURM_JOB_ID.log"
PROXY_LOG="$LOG_DIR/proxy-$SLURM_JOB_ID.log"
VLLM_PID=""
PROXY_PID=""

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

cleanup() {
    log "cleanup: killing vllm/proxy"
    [[ -n "${VLLM_PID:-}" ]] && kill -9 $VLLM_PID 2>/dev/null
    [[ -n "${PROXY_PID:-}" ]] && kill -9 $PROXY_PID 2>/dev/null
    pkill -9 -u "$USER" -f 'vllm serve' 2>/dev/null
}

log "=== run_native.sbatch ==="
log "  job_id=$SLURM_JOB_ID node=$SLURMD_NODENAME"
log "  MODEL_HF=$MODEL_HF parser=$MODEL_PARSER tp=$TP_SIZE"
log "  LABEL=$LABEL"
log "  GPUs=${CUDA_VISIBLE_DEVICES:-?}"

# ---- 1. Conda env (idempotent) -----------------------------------------------
log "loading anaconda module"
module load anaconda/2024.02-py311
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_ENV_DIR/bin/python" ]]; then
    log "creating conda env at $CONDA_ENV_DIR (one-time, ~10 min)"
    conda create -y -p "$CONDA_ENV_DIR" python=3.12 nodejs=22 >/dev/null
    conda activate "$CONDA_ENV_DIR"
    # Force binary wheels only, and pin vllm to a known wheel-having version.
    # Without --only-binary, pip will fall back to a source build for any
    # vllm version that doesn't ship a Python-3.12 wheel — the same 1-2 hour
    # CUDA compile we just escaped.
    pip install --no-cache-dir --only-binary=:all: \
        "vllm==0.20.2" \
        chromadb sentence-transformers \
        fastapi uvicorn httpx \
        pydantic anthropic openai
    # openclaude — vllm-talking Claude Code CLI; same install as
    # Singularity.dflash, just into the conda prefix.
    npm install -g --prefix "$CONDA_ENV_DIR" \
        @gitlawb/openclaude@0.8.0 --no-audit --no-fund
    OC_CLI="$CONDA_ENV_DIR/lib/node_modules/@gitlawb/openclaude/dist/cli.mjs"
    sed -i \
        -e "s/MAX_OUTPUT_TOKENS_DEFAULT = 32000/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" \
        -e "s/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64000/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 8000/g" \
        -e "s/ESCALATED_MAX_TOKENS = 64000/ESCALATED_MAX_TOKENS = 8000/g" \
        "$OC_CLI"
    cat > "$CONDA_ENV_DIR/bin/openclaude-vllm" <<'WRAP'
#!/bin/bash
exec "$(dirname "$0")/openclaude" --provider openai "$@"
WRAP
    chmod +x "$CONDA_ENV_DIR/bin/openclaude-vllm"
    log "conda env ready"
else
    log "conda env already exists at $CONDA_ENV_DIR — reusing"
    conda activate "$CONDA_ENV_DIR"
fi

# ---- 2. Start vllm serve ------------------------------------------------------
export HF_HOME="$USER_PATH/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_DISABLE_USAGE_STATS=1
# Force V0 engine — V1 init hung at shm_broadcast on Anvil A100 TP=2 even
# with --disable-custom-all-reduce --distributed-executor-backend mp.
export VLLM_USE_V1=0

# Skip vllm pre-flight (each `vllm` invocation pays a ~9 min cold-cache
# import on Anvil's shared scratch FS). Just log the env state cheaply.
log "vllm pre-flight (env only)"
{
    echo "--- node: $SLURMD_NODENAME ---"
    echo "--- which python ---"; which python
    echo "--- nvidia-smi ---"; nvidia-smi 2>&1 | head -10 || true
    echo "--- CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ---"
    echo "--- env vllm 0.10 + torch 2.7 + cu126 ---"
} >>"$VLLM_LOG" 2>&1

log "starting vLLM model=$MODEL_HF tp=$TP_SIZE"
PYTHONUNBUFFERED=1 vllm serve "$MODEL_HF" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --disable-custom-all-reduce \
    --distributed-executor-backend mp \
    --tool-call-parser "$MODEL_PARSER" \
    --enable-auto-tool-choice \
    --served-model-name "$MODEL_HF" \
    >>"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Wait up to 45 min for vLLM to be ready. vllm 0.10.0 V1 engine init on
# this shared-scratch conda env takes ~25-30 min cold (10 min import + 5
# min torch.compile + 5 min model load + 5 min CUDA graph capture). The
# previous 25-min cap was too tight; smoke hit it right at engine init.
log "waiting for vLLM /v1/models (PID $VLLM_PID)"
READY=0
for i in $(seq 1 270); do
    if curl -fsS -m 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        log "vLLM ready after ~$((i * 10))s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM died early; tail of $VLLM_LOG:"
        tail -40 "$VLLM_LOG" | sed 's/^/    /'
        cleanup; exit 1
    fi
    sleep 10
done
if [[ "$READY" != "1" ]]; then
    log "ERROR: vLLM not ready after 45 min; tail of $VLLM_LOG:"
    tail -60 "$VLLM_LOG" | sed 's/^/    /'
    cleanup; exit 1
fi

# ---- 3. Tool-call proxy --------------------------------------------------------
log "starting proxy"
python3 "$PROJECT_ROOT/benchmark/tool_call_proxy.py" \
    --port "$PROXY_PORT" --upstream "http://127.0.0.1:${VLLM_PORT}" \
    >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 20); do
    curl -fsS -m 3 "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null 2>&1 \
        && { log "proxy ready"; break; }
    sleep 1
done

# ---- 4. Self-gen skill (if absent) --------------------------------------------
cd "$PROJECT_ROOT"
export CLAUDE_BIN="$CONDA_ENV_DIR/bin/openclaude-vllm"
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export RF_SKILL_HINT_LEVEL=full
export OPENAI_API_STREAM=0
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export CLAUDE_CODE_TOOLS_OVERRIDE="Bash,Edit,Read,Write,Glob,Grep"
export CHROMA_HOST=""
export SIONNA_SKILL_CHROMA_PATH="$USER_PATH/chroma_db"
mkdir -p "$SIONNA_SKILL_CHROMA_PATH"

MODEL_BASENAME=$(echo "$MODEL_HF" | awk -F/ '{print tolower($NF)}')
SELF_GEN_PATH="benchmark/self_gen_skill/${MODEL_BASENAME}/SKILL.md"
if [[ ! -s "$SELF_GEN_PATH" ]]; then
    log "generating self_gen skill: $SELF_GEN_PATH"
    mkdir -p "$(dirname "$SELF_GEN_PATH")"
    python3 benchmark/generate_baseline_skill.py \
        --backend openai-compat --model "$MODEL_HF" \
        --out-file "$SELF_GEN_PATH" \
        || log "WARN: self_gen skill generation failed; self_gen trials will skip"
else
    log "self_gen skill already present: $SELF_GEN_PATH"
fi

# ---- 5. Run benchmark ----------------------------------------------------------
log "running benchmark label=$LABEL"
# Random task ordering each run so when walltime cuts the benchmark short
# every task has an equal chance to be sampled (uniform over the 95 train
# tasks × 3 conditions). Pass BENCH_SHUFFLE_SEED to reproduce a specific
# sample; default uses $RANDOM so iterations after skill edits hit a
# different sample.
SHUFFLE_SEED="${BENCH_SHUFFLE_SEED:-$RANDOM}"
echo "→ shuffle seed: $SHUFFLE_SEED" >> "$LOG_DIR/seed-$SLURM_JOB_ID.txt" 2>/dev/null || true

# Optional task subset (e.g., BENCH_TIERS=T1 for a smaller debug run).
BENCH_EXTRA=()
if [[ -n "${BENCH_TIERS:-}" ]]; then
    BENCH_EXTRA+=(--tiers $BENCH_TIERS)
fi
if [[ -n "${BENCH_TASK_IDS:-}" ]]; then
    BENCH_EXTRA+=(--task-ids $BENCH_TASK_IDS)
fi

log "running benchmark label=$LABEL shuffle_seed=$SHUFFLE_SEED"
python3 benchmark/run_benchmark.py \
    --label "$LABEL" --shuffle-seed "$SHUFFLE_SEED" --timeout 1200 \
    --split train --conditions with_skill no_skill self_gen --k 1 \
    --workers "$BENCH_WORKERS" \
    --max-turns 25 --model "$MODEL_HF" \
    "${BENCH_EXTRA[@]}"
RC=$?
log "benchmark finished rc=$RC"

cleanup
exit $RC
