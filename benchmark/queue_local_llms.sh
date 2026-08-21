#!/bin/bash
# queue_local_llms.sh
# ----------------------------------------------------------------------
# Sequentially run multiple local-LLM benchmarks on sunlab, hot-swapping
# the model in GPU between runs. Designed for unattended overnight use.
#
# For each config (label / backend / model):
#   1. Stop any backend daemon this script started in a previous iteration.
#   2. Wait for the GPU to release VRAM (poll nvidia-smi, ~1 GB threshold).
#   3. Start the requested backend (ollama or vLLM) and wait for /v1/models.
#   4. Invoke benchmark/run_benchmark.py with the matching env vars + label.
#   5. Resume-friendly: skip if benchmark/results/<label>/progress.json
#      already has finished_at; otherwise pass --resume.
#   6. Append start/end times + exit codes to benchmark/results/_queue.log.
#
# Only daemons spawned by this script are killed (PIDs tracked in $BACKEND_PID).
# Other users' processes on the box are left alone.
#
# Usage:  bash benchmark/queue_local_llms.sh
# ----------------------------------------------------------------------

set -uo pipefail
# NOTE: no `set -e` — we want a single failed run to log + continue, not
# abort the whole overnight queue.

# ─────────────────────────────────────────────────────────────────────
# Paths & static config — every variable below is overridable via env.
# Sensible defaults for both sunlab (RTX 5090 32GB) and vast.ai (H200
# 141GB). Edit RUN_CONFIGS at the bottom to add/remove model runs.
# ─────────────────────────────────────────────────────────────────────

# Anchor PROJECT_ROOT to the repo regardless of where the script is invoked
# from. Resolves $PROJECT_ROOT to the parent of benchmark/.
_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_THIS_DIR}/.." && pwd)}"
RESULTS_DIR="$PROJECT_ROOT/benchmark/results"
LOG_FILE="$RESULTS_DIR/_queue.log"

# Binary discovery: env override → `which` → fallback to known sunlab path.
_locate_bin() {  # $1=env_value $2=command_name $3=fallback_path
  if [[ -n "$1" && -x "$1" ]]; then echo "$1"; return; fi
  local found; found="$(command -v "$2" 2>/dev/null || true)"
  if [[ -n "$found" ]]; then echo "$found"; return; fi
  echo "$3"
}
CLAUDE_BIN="$(_locate_bin "${CLAUDE_BIN:-}" openclaude \
    "/home/myid/js66916/.local/share/fnm/node-versions/v22.22.2/installation/bin/openclaude")"
VLLM_BIN="$(_locate_bin "${VLLM_BIN:-}" vllm \
    "/home/myid/js66916/miniconda3/envs/vllm/bin/vllm")"
PYTHON_BIN="$(_locate_bin "${PYTHON_BIN:-}" python3 "/usr/bin/python3")"
# Conda env's python (for the FastAPI proxy) — falls back to system python3
# if not present. The proxy needs fastapi/uvicorn/httpx; on vast.ai the base
# image typically has these or `pip install` is one-shot.
PROXY_PYTHON="${PROXY_PYTHON:-$(dirname "$VLLM_BIN")/python3}"
[[ -x "$PROXY_PYTHON" ]] || PROXY_PYTHON="$PYTHON_BIN"

OLLAMA_BIN="$(_locate_bin "${OLLAMA_BIN:-}" ollama "/home/myid/js66916/ollama/bin/ollama")"
OLLAMA_PORT=11500
OLLAMA_HOST_URL="http://127.0.0.1:${OLLAMA_PORT}"

VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_HOST_URL="http://127.0.0.1:${VLLM_PORT}"
HF_HOME_DEFAULT="${HF_HOME:-${HOME}/hf_cache}"

# Tool-call repair proxy. Sits between OpenClaude and vLLM, catches small-LLM
# malformed tool calls (Write without `content`, Bash with `command=undefined`).
PROXY_PORT="${PROXY_PORT:-8002}"
PROXY_HOST_URL="http://127.0.0.1:${PROXY_PORT}"
PROXY_PID=""

# ─────────────────────────────────────────────────────────────────────
# VRAM tier — auto-detected from nvidia-smi total memory.
# Affects vLLM concurrency (max-num-seqs), CUDA graphs, KV cache, and
# the parallel-trial worker count.
# Override via VRAM_TIER=low|high or set individual VLLM_* / BENCH_WORKERS.
# ─────────────────────────────────────────────────────────────────────
detect_vram_tier() {
  if [[ -n "${VRAM_TIER:-}" && "$VRAM_TIER" != "auto" ]]; then
    echo "$VRAM_TIER"; return
  fi
  local total_mib
  total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
               | head -n1 | tr -d ' ')"
  if [[ -z "$total_mib" ]]; then echo "low"; return; fi
  # >64 GB → high tier (H100, H200, A100-80, MI300X, etc.)
  if (( total_mib > 65536 )); then echo "high"; else echo "low"; fi
}
VRAM_TIER="$(detect_vram_tier)"

case "$VRAM_TIER" in
  high)
    # H200 / H100 / A100-80GB. Drop --enforce-eager (use CUDA graphs),
    # raise concurrency, give the OS some headroom.
    VLLM_NUM_SEQS="${VLLM_NUM_SEQS:-16}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
    VLLM_EXTRA_FLAGS_DEFAULT=()  # no --enforce-eager; CUDA graphs are fine
    BENCH_WORKERS="${BENCH_WORKERS:-6}"
    ;;
  low|*)
    # RTX 5090 32GB and similar. CUDA graphs OOM at high context, so
    # eager mode + max-num-seqs=1 is the only stable config.
    VLLM_NUM_SEQS="${VLLM_NUM_SEQS:-1}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.93}"
    VLLM_EXTRA_FLAGS_DEFAULT=( --enforce-eager )
    BENCH_WORKERS="${BENCH_WORKERS:-1}"
    ;;
esac

# Default benchmark args (overridable per-config; see RUN_CONFIGS).
# To run the RAG comparison (P6.5), set:
#   DEFAULT_CONDITIONS="with_skill no_skill"
#   CLAUDE_CODE_USE_RAG=1
# That makes with_skill = skill+RAG; no_skill stays bare. To get all
# three conditions (skill-only / RAG-only / skill+RAG), run three queue
# launches with different env settings.
DEFAULT_CONDITIONS="${DEFAULT_CONDITIONS:-with_skill no_skill}"
DEFAULT_MAX_TURNS="${DEFAULT_MAX_TURNS:-25}"
DEFAULT_TIMEOUT="${DEFAULT_TIMEOUT:-400}"
DEFAULT_RETRY_TIMEOUT="${DEFAULT_RETRY_TIMEOUT:-1200}"
DEFAULT_SHUFFLE_SEED="${DEFAULT_SHUFFLE_SEED:-42}"
DEFAULT_SPLIT="${DEFAULT_SPLIT:-train}"

# GPU readiness threshold (MiB). Wait until used VRAM drops below this
# before launching the next backend.
GPU_FREE_MIB_THRESHOLD="${GPU_FREE_MIB_THRESHOLD:-1024}"
GPU_WAIT_TIMEOUT_S="${GPU_WAIT_TIMEOUT_S:-180}"
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-300}"

# ─────────────────────────────────────────────────────────────────────
# Run queue. Each line:
#   "label|backend|model|conditions|max_turns|timeout"
# Empty fields after `model` fall back to defaults. Edit freely.
# ─────────────────────────────────────────────────────────────────────
RUN_CONFIGS=(
  # Sunlab (RTX 5090 32 GB) — works in low VRAM tier:
  # "paired_qwen3_6_v4|vllm|groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit|||"
  # "paired_llama31_8b_v4|vllm|unsloth/Meta-Llama-3.1-8B-Instruct|||"
  # "paired_gemma4_31b_v4|vllm|QuantTrio/gemma-4-31B-it-AWQ|||"

  # vast.ai (H200 141 GB) — high VRAM tier, BF16 70B-class fits:
  "paired_llama31_70b_v5|vllm|meta-llama/Meta-Llama-3.1-70B-Instruct|||"
  "paired_qwen25_72b_v5|vllm|Qwen/Qwen2.5-72B-Instruct|||"
  "paired_llama33_70b_v5|vllm|meta-llama/Llama-3.3-70B-Instruct|||"
  # Mid-size for sanity (also works in low tier):
  "paired_llama31_8b_v5|vllm|unsloth/Meta-Llama-3.1-8B-Instruct|||"
)

# ─────────────────────────────────────────────────────────────────────
# Bookkeeping
# ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
touch "$LOG_FILE"

BACKEND_PID=""        # PID of currently-running backend daemon (this script's child).
BACKEND_KIND=""       # "ollama" | "vllm" | ""
BACKEND_LOG=""        # Path to its stdout/stderr log.

log() {
  local ts
  ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE"
}

start_proxy() {
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    return 0
  fi
  local proxy_log="$RESULTS_DIR/_proxy_${PROXY_PORT}.log"
  log "starting tool-call repair proxy on :${PROXY_PORT} (python=$PROXY_PYTHON, log=$proxy_log)"
  # Use the vllm env python (or the system python3 fallback). vLLM pulls
  # fastapi/uvicorn/httpx as deps so the proxy works without an extra install.
  "$PROXY_PYTHON" \
      "$PROJECT_ROOT/benchmark/tool_call_proxy.py" \
      --port "$PROXY_PORT" \
      --upstream "$VLLM_HOST_URL" \
      >>"$proxy_log" 2>&1 &
  PROXY_PID=$!
  # Wait for /healthz
  for _ in $(seq 1 20); do
    if curl -fsS "${PROXY_HOST_URL}/healthz" >/dev/null 2>&1; then
      log "  proxy ready"
      return 0
    fi
    sleep 1
  done
  log "  ERROR: proxy did not become ready within 20s"
  return 1
}

stop_proxy() {
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    log "stopping tool-call proxy (pid=$PROXY_PID)"
    kill "$PROXY_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PROXY_PID" 2>/dev/null || true
  fi
  PROXY_PID=""
}

cleanup_backend() {
  # Stop only the daemon WE started — don't pkill broadly.
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    log "stopping $BACKEND_KIND (pid=$BACKEND_PID)"
    kill "$BACKEND_PID" 2>/dev/null || true
    # Give it 15s to drain, then SIGKILL its process group.
    for _ in $(seq 1 15); do
      kill -0 "$BACKEND_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
      log "  forcing SIGKILL"
      kill -9 "$BACKEND_PID" 2>/dev/null || true
      # vLLM spawns engine workers under the parent — kill the whole pgrp.
      pkill -9 -P "$BACKEND_PID" 2>/dev/null || true
    fi
  fi
  BACKEND_PID=""
  BACKEND_KIND=""
}

trap 'log "received signal — cleaning up"; stop_proxy; cleanup_backend; exit 130' INT TERM

wait_for_gpu_free() {
  local deadline=$((SECONDS + GPU_WAIT_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    if [[ -z "$used" ]]; then
      log "  nvidia-smi unavailable — skipping GPU wait"
      return 0
    fi
    if (( used <= GPU_FREE_MIB_THRESHOLD )); then
      log "  GPU free (${used} MiB used)"
      return 0
    fi
    sleep 3
  done
  log "  WARNING: GPU still busy after ${GPU_WAIT_TIMEOUT_S}s — continuing anyway"
  return 0
}

start_ollama() {
  local model="$1"
  BACKEND_LOG="$RESULTS_DIR/_ollama_${OLLAMA_PORT}.log"
  log "starting ollama daemon on :${OLLAMA_PORT} (log=$BACKEND_LOG)"
  OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" "$OLLAMA_BIN" serve \
    >>"$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!
  BACKEND_KIND="ollama"
  # Poll /api/tags until ready.
  local deadline=$((SECONDS + BACKEND_READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1; then
      log "  ollama ready"
      # Warm the model so first benchmark trial doesn't pay the cold-load cost.
      OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" \
        "$OLLAMA_BIN" run "$model" "" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 2
  done
  log "  ERROR: ollama did not become ready within ${BACKEND_READY_TIMEOUT_S}s"
  return 1
}

# v2.8.5: synthetic chat/completion warmup probe. /v1/models 200 != engine
# is ready for /v1/chat/completions — Gemma cycle3+4 lost all 4 trials
# (turns=0, full 1200s timeout) to a 503 on first real call. This probe
# closes the race; on any 200 we proceed; otherwise we proceed anyway
# (warning logged) so we don't infinite-loop on stuck engines.
warmup_chat() {
  local model="$1"
  for i in $(seq 1 6); do
    local rc
    rc=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${VLLM_HOST_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      --data "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
      --max-time 30)
    if [[ "$rc" == "200" ]]; then
      log "  warmup ok after attempt $i"
      return 0
    fi
    log "  warmup attempt $i: HTTP $rc, retry in 5s"
    sleep 5
  done
  log "  WARNING: warmup never returned 200 after 6 attempts; proceeding anyway"
  return 0
}

start_vllm() {
  local model="$1"
  BACKEND_LOG="$RESULTS_DIR/_vllm_${VLLM_PORT}.log"
  log "starting vLLM (model=$model, port=$VLLM_PORT, log=$BACKEND_LOG)"

  # Per-model flag selection. Each family's vLLM parsers + quant differ.
  # max-model-len is bumped on high-VRAM tier where the KV cache fits.
  # tools: per-model CLAUDE_CODE_TOOLS_OVERRIDE (8B Write parser regresses).
  # `tools` is intentionally NOT local — run_benchmark's subshell reads it
  # via the parent shell's variable scope.
  local extra_flags=()
  local low_ctx high_ctx mml
  # H.3: Gemma disables --enable-auto-tool-choice (grammar layer deadlocks vLLM
  # with openclaude's tool schemas). Proxy parses Gemma's native markup instead.
  local use_auto_tool_choice=1
  case "$model" in
    *Qwen3.6-27B-GPTQ-Pro-4bit*)
      low_ctx=25000; high_ctx=65536
      extra_flags+=( --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
                     --quantization gptq_marlin )
      tools="Bash,Read,Write,Edit"
      ;;
    *Meta-Llama-3.1-8B-Instruct*)
      low_ctx=32768; high_ctx=131072
      extra_flags+=( --tool-call-parser llama3_json )
      tools="Bash,Read"
      ;;
    *Meta-Llama-3.1-70B-Instruct*|*Llama-3.3-70B-Instruct*)
      low_ctx=16384; high_ctx=131072
      extra_flags+=( --tool-call-parser llama3_json )
      tools="Bash,Read,Write,Edit"
      ;;
    *Qwen2.5-72B-Instruct*)
      low_ctx=16384; high_ctx=32768
      extra_flags+=( --tool-call-parser hermes )
      tools="Bash,Read,Write,Edit"
      ;;
    *DeepSeek-R1-Distill-Qwen-32B-AWQ*)
      low_ctx=16384; high_ctx=32768
      extra_flags+=( --tool-call-parser deepseek_v3 --reasoning-parser deepseek_r1 \
                     --quantization awq_marlin )
      tools="Bash,Read,Write,Edit"
      ;;
    *gemma-4-31B-it-AWQ*)
      low_ctx=32768; high_ctx=32768
      # H.3: drop --tool-call-parser (grammar layer deadlocks); proxy parses markup
      extra_flags+=( --quantization awq_marlin )
      use_auto_tool_choice=0
      tools="Bash,Read,Write,Edit"
      ;;
    *)
      log "  WARN: no per-model vLLM flags for $model — using defaults"
      low_ctx=16384; high_ctx=32768
      tools="Bash,Read"
      ;;
  esac
  if [[ "$VRAM_TIER" == "high" ]]; then mml="$high_ctx"; else mml="$low_ctx"; fi
  extra_flags+=( --max-model-len "$mml" )
  # Conditionally include --enable-auto-tool-choice (disabled for Gemma, H.3)
  local auto_tool_flag=()
  if [[ "$use_auto_tool_choice" == "1" ]]; then
    auto_tool_flag=( --enable-auto-tool-choice )
  fi

  log "  vram_tier=$VRAM_TIER max_num_seqs=$VLLM_NUM_SEQS gpu_util=$VLLM_GPU_UTIL max_model_len=$mml auto_tool_choice=$use_auto_tool_choice"

  HF_HOME="$HF_HOME_DEFAULT" "$VLLM_BIN" serve "$model" \
      --port "$VLLM_PORT" \
      "${auto_tool_flag[@]}" \
      --enable-prefix-caching \
      --disable-log-stats \
      --gpu-memory-utilization "$VLLM_GPU_UTIL" \
      --max-num-seqs "$VLLM_NUM_SEQS" \
      "${VLLM_EXTRA_FLAGS_DEFAULT[@]}" \
      "${extra_flags[@]}" \
      >>"$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!
  BACKEND_KIND="vllm"
  # vLLM startup can take minutes (weight load + CUDA graphs). Poll /v1/models.
  local deadline=$((SECONDS + BACKEND_READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -fsS "${VLLM_HOST_URL}/v1/models" >/dev/null 2>&1; then
      log "  vLLM /v1/models ready — running warmup probe"
      warmup_chat "$model"
      return 0
    fi
    # If the daemon died early, bail rather than wait forever.
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      log "  ERROR: vLLM exited early — see $BACKEND_LOG"
      BACKEND_PID=""
      return 1
    fi
    sleep 5
  done
  log "  ERROR: vLLM did not become ready within ${BACKEND_READY_TIMEOUT_S}s"
  return 1
}

is_run_finished() {
  # Returns 0 if benchmark/results/<label>/progress.json already has finished_at.
  local label="$1"
  local pjson="$RESULTS_DIR/$label/progress.json"
  [[ -f "$pjson" ]] || return 1
  python3 -c "
import json, sys
try:
    p = json.load(open('$pjson'))
except Exception:
    sys.exit(1)
sys.exit(0 if p.get('finished_at') else 1)
" 2>/dev/null
}

run_benchmark() {
  local label="$1" backend="$2" model="$3" conditions="$4" max_turns="$5" timeout="$6"
  local base_url
  case "$backend" in
    ollama) base_url="${OLLAMA_HOST_URL}/v1" ;;
    vllm)   base_url="${PROXY_HOST_URL}/v1" ;;  # via tool-call repair proxy
    *) log "  ERROR: unknown backend=$backend"; return 2 ;;
  esac

  local resume_flag=""
  if [[ -d "$RESULTS_DIR/$label" ]]; then
    log "  $label/ exists — passing --resume"
    resume_flag="--resume"
  fi

  local conds="${conditions:-$DEFAULT_CONDITIONS}"
  local mt="${max_turns:-$DEFAULT_MAX_TURNS}"
  local to="${timeout:-$DEFAULT_TIMEOUT}"

  log "running benchmark label=$label model=$model backend=$backend"
  log "  conditions=[$conds] max_turns=$mt timeout=${to}s retry=${DEFAULT_RETRY_TIMEOUT}s"

  # Env vars consumed by run_benchmark.py / trial.py / openclaude.
  local rc
  (
    export PATH="$(dirname "$CLAUDE_BIN"):$PATH"
    export CLAUDE_BIN="$CLAUDE_BIN"
    export CLAUDE_CODE_USE_OPENAI=1
    export OPENAI_BASE_URL="$base_url"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"
    # v1.4 / fix #1: drop Write/Edit/Glob/Grep entirely. Small-LLM parsers
    # drop `content` on Write ~40% of the time, sending agents into a
    # malformation retry loop. Bash + Read can do everything those four
    # tools can (heredoc to write, sed to edit, find/grep via shell).
    export CLAUDE_CODE_TOOLS_OVERRIDE="${CLAUDE_CODE_TOOLS_OVERRIDE:-${tools:-Bash,Read}}"
    cd "$PROJECT_ROOT" || exit 90
    # shellcheck disable=SC2086  # intentional word splitting on $conds, $resume_flag
    "$PYTHON_BIN" benchmark/run_benchmark.py \
        --label "$label" \
        --shuffle-seed "$DEFAULT_SHUFFLE_SEED" \
        --timeout "$to" \
        --retry-timeout "$DEFAULT_RETRY_TIMEOUT" \
        --split "$DEFAULT_SPLIT" \
        --conditions $conds \
        --k "${BENCH_K:-1}" \
        --workers "$BENCH_WORKERS" \
        --max-turns "$mt" \
        --model "$model" \
        $resume_flag
  )
  rc=$?
  log "  benchmark exited rc=$rc"
  return $rc
}

# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────
log "=== queue start: ${#RUN_CONFIGS[@]} configs ==="
log "config:"
log "  PROJECT_ROOT  = $PROJECT_ROOT"
log "  CLAUDE_BIN    = $CLAUDE_BIN"
log "  VLLM_BIN      = $VLLM_BIN"
log "  PYTHON_BIN    = $PYTHON_BIN"
log "  PROXY_PYTHON  = $PROXY_PYTHON"
log "  HF_HOME       = $HF_HOME_DEFAULT"
log "  VRAM_TIER     = $VRAM_TIER  (override with VRAM_TIER=low|high)"
log "  VLLM_NUM_SEQS = $VLLM_NUM_SEQS"
log "  VLLM_GPU_UTIL = $VLLM_GPU_UTIL"
log "  BENCH_WORKERS = $BENCH_WORKERS"
log "  TOOLS_SUBSET  = ${CLAUDE_CODE_TOOLS_OVERRIDE:-Bash,Read}"
log "  BENCH_K       = ${BENCH_K:-1}  (override: BENCH_K=5 for paper-quality variance)"

# Sanity: bail early if any required binary is missing.
for b in CLAUDE_BIN VLLM_BIN PYTHON_BIN PROXY_PYTHON; do
  if [[ ! -x "${!b}" ]]; then
    log "  ERROR: $b='${!b}' is not executable. Set the env var or install."
    exit 2
  fi
done

for entry in "${RUN_CONFIGS[@]}"; do
  IFS='|' read -r label backend model conditions max_turns timeout <<<"$entry"

  if [[ -z "$label" || -z "$backend" || -z "$model" ]]; then
    log "SKIP malformed entry: $entry"
    continue
  fi

  log "----- next config: label=$label backend=$backend model=$model -----"

  if is_run_finished "$label"; then
    log "  already finished (progress.json has finished_at) — skipping"
    continue
  fi

  # Tear down whatever's currently in GPU before swapping models.
  cleanup_backend
  wait_for_gpu_free

  # Bring up the right backend.
  case "$backend" in
    ollama)
      if ! start_ollama "$model"; then
        log "  FAIL: could not start ollama for $label — skipping run"
        cleanup_backend
        continue
      fi
      ;;
    vllm)
      if ! start_vllm "$model"; then
        log "  FAIL: could not start vLLM for $label — skipping run"
        cleanup_backend
        continue
      fi
      if ! start_proxy; then
        log "  FAIL: could not start tool-call proxy for $label — skipping"
        stop_proxy; cleanup_backend
        continue
      fi
      ;;
    *)
      log "  FAIL: unknown backend $backend — skipping"
      continue
      ;;
  esac

  # Per-model skill_hint verbosity: 8B-class gets minimal (no LOOKUP block),
  # larger models get full. Matched on label suffix (e.g. _8b_ in the label).
  label_suffix="${label##*_}"   # last segment after final underscore (e.g. "v5")
  # Match model size component anywhere in the label.
  case "$label" in
    *_8b_*) export RF_SKILL_HINT_LEVEL=minimal ;;
    *)      export RF_SKILL_HINT_LEVEL=full    ;;
  esac
  log "  RF_SKILL_HINT_LEVEL=$RF_SKILL_HINT_LEVEL (label=$label)"

  start_ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  log "RUN start  $label  ($start_ts)"

  run_benchmark "$label" "$backend" "$model" "$conditions" "$max_turns" "$timeout"
  rc=$?

  end_ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  log "RUN end    $label  ($end_ts)  rc=$rc"
done

log "=== queue end: tearing down final backend ==="
stop_proxy
cleanup_backend
log "=== done ==="
