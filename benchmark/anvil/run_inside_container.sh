#!/bin/bash
# run_inside_container.sh — entrypoint for the Singularity container started
# by anvil_run_model.sbatch. Equivalent to bootstrap_vastai_tp.sh's "step 11+"
# (start vLLM → start proxy → run_benchmark.py → cleanup) but stripped of
# vast.ai-specific bits (rsync, catalog probing, multi-instance loops).
#
# Inputs (env vars set by the sbatch wrapper):
#   MODEL_HF, MODEL_PARSER, USE_DFLASH, VAST_TP_SIZE, VAST_BENCH_WORKERS,
#   VAST_DFLASH_SPEC_TOKENS, VAST_BENCH_LABEL, HF_HOME, HF_TOKEN
#
# Working directory: /work/skill (the project repo, bind-mounted from
# $SCRATCH/skill).

set -uo pipefail

# Triton (via torch.compile in vllm) requires libcuda.so but Anvil's --nv
# mounts only libcuda.so.1. Search every plausible directory and create the
# symlink. Verbose so failures are diagnosable.
echo "[startup] searching for libcuda.so.1 to symlink:"
for d in /usr/local/cuda/compat/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu \
         /.singularity.d/libs /usr/lib64 /usr/lib; do
    if [[ -d "$d" ]]; then
        if [[ -f "$d/libcuda.so.1" || -L "$d/libcuda.so.1" ]]; then
            if [[ ! -e "$d/libcuda.so" ]]; then
                if ln -sf libcuda.so.1 "$d/libcuda.so" 2>/dev/null; then
                    echo "[startup]   $d/libcuda.so -> libcuda.so.1 (CREATED)"
                else
                    echo "[startup]   $d/libcuda.so could not write (readonly?)"
                fi
            else
                echo "[startup]   $d/libcuda.so already exists"
            fi
        fi
    fi
done
# Fallback: tell triton explicitly where to look. Some triton versions honor
# this env var.
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib
export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
echo "[startup] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Derive ports from SLURM_JOB_ID to avoid loopback collisions when SLURM
# packs two of our jobs onto the same node (Singularity shares the host
# network namespace). Fallback to fixed 8001/8002 outside SLURM.
_JOB_OFFSET=$(( ${SLURM_JOB_ID:-0} % 1000 ))
VLLM_PORT=$(( 8000 + _JOB_OFFSET ))
PROXY_PORT=$(( 9000 + _JOB_OFFSET ))
# torch.distributed master port — must also be unique when two jobs co-locate
# on the same node (default 29500 collides → DistStoreError 1/2 clients joined).
export MASTER_PORT=$(( 25000 + _JOB_OFFSET ))
export MASTER_ADDR=127.0.0.1
VLLM_PID=""
PROXY_PID=""

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cleanup() {
    log "cleanup: vllm($VLLM_PID) proxy($PROXY_PID)"
    [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null
    [[ -n "$VLLM_PID" ]]  && kill "$VLLM_PID"  2>/dev/null
    sleep 5
    [[ -n "$PROXY_PID" ]] && kill -9 "$PROXY_PID" 2>/dev/null
    [[ -n "$VLLM_PID" ]]  && kill -9 "$VLLM_PID"  2>/dev/null
}
trap 'log "signal — cleanup"; cleanup; exit 130' INT TERM

# Per-family vLLM flag selection (mirror of bootstrap_vastai_tp.sh:707-740).
EXTRA_VLLM_FLAGS=()
AUTO_TOOL_FLAGS=( --enable-auto-tool-choice )
case "$MODEL_HF" in
    *Qwen3-Coder-30B-A3B-Instruct*)
        EXTRA_VLLM_FLAGS+=( --tool-call-parser qwen3_coder )
        ;;
    *Qwen3.6-27B*|*Qwen3.6-35B-A3B*)
        # vLLM 0.10.0 has no qwen3_xml parser; reuse qwen3_coder (closest fmt).
        EXTRA_VLLM_FLAGS+=( --tool-call-parser qwen3_coder )
        ;;
    *[Ll]lama-3.3-70[Bb]*-[Aa][Ww][Qq]*)
        EXTRA_VLLM_FLAGS+=( --tool-call-parser llama3_json --quantization awq_marlin )
        ;;
    *Gemma-4-31B-it*|*Gemma-4-26B-A4B-it*|*gemma-4-31[Bb]*|*gemma-4-26[Bb]*)
        AUTO_TOOL_FLAGS=()  # vLLM grammar layer hangs Gemma; proxy parses native markup.
        ;;
    *Mixtral-8x7B*)
        # 2026-05-15: Mixtral-8x7B-v0.1 lacks [TOOL_CALLS] / [TOOL_RESULTS]
        # tokens that the `mistral` parser expects. Drop --enable-auto-tool-choice
        # AND strip `tools`/`tool_choice` from client requests via the proxy.
        # PROXY_AUTO_TOOLS (further down) synthesizes Write+Bash from raw text.
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        EXTRA_VLLM_FLAGS+=( --disable-sliding-window )
        if [[ "$MODEL_HF" == *AWQ* || "$MODEL_HF" == *awq* ]]; then
            EXTRA_VLLM_FLAGS+=( --quantization awq_marlin )
        fi
        EXTRA_VLLM_FLAGS+=( --chat-template /work/skill/benchmark/anvil/templates/mistral_tool.jinja )
        export PROXY_COALESCE_ROLES=1
        ;;
    *[Pp]hi-4-mini*|*microsoft/Phi-4-mini*)
        # 2026-05-15: Phi-4-mini empirically emits prose w/ ```python blocks,
        # not native tool tokens. Drop parser, strip tool_choice via proxy,
        # use PROXY_AUTO_TOOLS. Also Phi-4-mini uses sliding-window attention
        # which conflicts with --enable-prefix-caching.
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        EXTRA_VLLM_FLAGS+=( --disable-sliding-window )
        ;;
    *DeepSeek-R1-Distill*)
        # 2026-05-15: R1-Distill is a reasoning model that emits <think>...</think>
        # then prose+code. The hermes parser conflicts with <think> tokens.
        # Drop parser, strip tool_choice, use PROXY_AUTO_TOOLS.
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        ;;
    *QwQ*)
        # 2026-05-15: QwQ-32B is also a reasoning model with <think> tokens.
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        ;;
    *[Pp]hi-4*|*microsoft/Phi-4*)
        # 2026-05-15: Phi-4 (base, not mini) has no <|tool|> tokens in its
        # tokenizer. Drop parser, strip tool_choice via proxy, rely on
        # PROXY_AUTO_TOOLS for text→tool_call synthesis. Also Phi-4 caps at
        # 16k context (max_position_embeddings=16384), shorter than our default.
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        export VAST_MAX_MODEL_LEN=16384
        ;;
    *granite-3.3-8b*|*granite-3.3-8B*|*Granite-3.3-8[Bb]*)
        # 2026-05-15: granite-3.3-8b's tokenizer lacks Hermes-style tool tokens.
        # Drop parser + strip tool_choice + rely on PROXY_AUTO_TOOLS (which now
        # also synthesizes from ```bash``` blocks, the Granite failure mode).
        AUTO_TOOL_FLAGS=()
        export PROXY_STRIP_TOOLS=1
        ;;
    *Mistral-Small-3*|*mistralai/Mistral-Small-3*)
        # 2026-05-15: Mistral-Small-3.1 has native tool-call training. transformers
        # v4.44+ requires an explicit chat_template, so we pass our pre-built
        # mistral_tool.jinja (same one Mixtral uses).
        EXTRA_VLLM_FLAGS+=( --tool-call-parser mistral )
        EXTRA_VLLM_FLAGS+=( --chat-template /work/skill/benchmark/anvil/templates/mistral_tool.jinja )
        ;;
    *granite-4.0*|*Granite-4.0*)
        # 2026-05-15: Granite-4.0 has native tool-call training. GraniteMoeHybrid
        # does not support prefix caching, so we strip --enable-prefix-caching
        # from the serve invocation below via DISABLE_PREFIX_CACHE=1.
        EXTRA_VLLM_FLAGS+=( --tool-call-parser granite )
        DISABLE_PREFIX_CACHE=1
        ;;
    *)
        EXTRA_VLLM_FLAGS+=( --tool-call-parser "${MODEL_PARSER:-hermes}" )
        ;;
esac

# Plain-text models: DeepSeek-Coder-V2-Lite, Phi-4 (base), Granite-3.3-8B,
# Mixtral-8x7B-v0.1 emit markdown ```python or ```bash blocks instead of
# structured tool_calls. Activate the proxy's PROXY_AUTO_TOOLS shim so it
# synthesizes Write+Bash from the first python block (or concatenates bash
# blocks). Default OFF for other families (must not change behavior for
# Q3-Coder, Llama, or the modern tool-call-trained models below).
case "$MODEL_HF" in
    *DeepSeek-Coder-V2-Lite*|*deepseek-coder-v2-lite*|\
    *granite-3.3-8b*|*Granite-3.3-8B*|*granite-3.3-8B*|\
    *Mixtral-8x7B*|\
    *DeepSeek-R1-Distill*|\
    *Phi-4-mini*|*phi-4-mini*|\
    *QwQ*)
        # 2026-05-15: empirically, all sub-frontier models — even the
        # "tool-call-trained" ones (Phi-4-mini, DSR1-Distill) and the reasoning
        # variants (QwQ, R1-Distill) — emit prose with embedded ```python/```bash
        # blocks rather than structured tool_calls. Force the proxy shim to
        # synthesize Write+Bash from those blocks.
        export PROXY_AUTO_TOOLS=1
        log "PROXY_AUTO_TOOLS=1 (plain-text shim active for $MODEL_HF)"
        ;;
    *[Pp]hi-4*|*microsoft/Phi-4*)
        # Phi-4 base (mini handled above)
        if [[ "$MODEL_HF" != *Phi-4-mini* && "$MODEL_HF" != *phi-4-mini* ]]; then
            export PROXY_AUTO_TOOLS=1
            log "PROXY_AUTO_TOOLS=1 (plain-text shim active for $MODEL_HF)"
        fi
        ;;
esac

# DFlash speculative decoding flags (mirror of the new block in bootstrap_vastai_tp.sh).
DFLASH_DRAFT=""
if [[ "${USE_DFLASH:-0}" == "1" ]]; then
    case "$MODEL_HF" in
        *Qwen3.6-27B*)              DFLASH_DRAFT="z-lab/Qwen3.6-27B-DFlash" ;;
        *Qwen3.6-35B-A3B*)          DFLASH_DRAFT="z-lab/Qwen3.6-35B-A3B-DFlash" ;;
        *Qwen3-Coder-30B-A3B*)      DFLASH_DRAFT="z-lab/Qwen3-Coder-30B-A3B-DFlash" ;;
        *Gemma-4-31B-it*|*gemma-4-31[Bb]*)
                                    DFLASH_DRAFT="z-lab/gemma-4-31B-it-DFlash" ;;
        *Gemma-4-26B-A4B-it*|*gemma-4-26[Bb]*-A4B*)
                                    DFLASH_DRAFT="z-lab/gemma-4-26B-A4B-it-DFlash" ;;
        *[Ll]lama-3.1-8[Bb]*)       DFLASH_DRAFT="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat" ;;
        *)
            log "WARN: USE_DFLASH=1 but no draft for $MODEL_HF — running baseline"
            ;;
    esac
fi
if [[ -n "$DFLASH_DRAFT" ]]; then
    SPEC_TOKENS="${VAST_DFLASH_SPEC_TOKENS:-15}"
    EXTRA_VLLM_FLAGS+=( --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT\",\"num_speculative_tokens\":$SPEC_TOKENS}" )
    EXTRA_VLLM_FLAGS+=( --attention-backend flash_attn )
    log "DFlash: draft=$DFLASH_DRAFT spec_tokens=$SPEC_TOKENS"
fi

VLLM_LOG="/work/scratch/skill/logs/vllm_${SLURM_JOB_ID:-local}.log"
PROXY_LOG="/work/scratch/skill/logs/proxy_${SLURM_JOB_ID:-local}.log"
mkdir -p "$(dirname "$VLLM_LOG")"

log "starting vLLM model=$MODEL_HF tp=${VAST_TP_SIZE:-2}"
log "  flags: ${AUTO_TOOL_FLAGS[*]:-} ${EXTRA_VLLM_FLAGS[*]}"
# Force V0 engine — V1 init hangs on shm_broadcast with TP=2 on Anvil A100
# nodes, even with --disable-custom-all-reduce. V0 init is simpler.
export VLLM_USE_V1=0
export VLLM_DISABLE_USAGE_STATS=1
PREFIX_CACHE_FLAG="--enable-prefix-caching"
if [[ "${DISABLE_PREFIX_CACHE:-0}" == "1" ]]; then
    PREFIX_CACHE_FLAG=""
    log "prefix caching disabled per family override"
fi
vllm serve "$MODEL_HF" \
    --tensor-parallel-size "${VAST_TP_SIZE:-2}" \
    --max-model-len "${VAST_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    $PREFIX_CACHE_FLAG \
    --disable-custom-all-reduce \
    --port "$VLLM_PORT" --host 127.0.0.1 \
    --served-model-name "$MODEL_HF" \
    "${AUTO_TOOL_FLAGS[@]}" \
    "${EXTRA_VLLM_FLAGS[@]}" \
    >>"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Wait up to 15 min for vLLM to be ready (TP init + DFlash draft load).
READY=0
for i in $(seq 1 90); do
    if curl -fsS -m 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
        log "vLLM ready after ~$((i * 10))s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM died early; tail:"
        tail -40 "$VLLM_LOG" | sed 's/^/    /'
        cleanup; exit 1
    fi
    sleep 10
done
if [[ "$READY" != "1" ]]; then
    log "ERROR: vLLM not ready after 15 min; bailing"
    cleanup; exit 1
fi

log "starting proxy"
python3 /work/skill/benchmark/tool_call_proxy.py \
    --port "$PROXY_PORT" --upstream "http://127.0.0.1:${VLLM_PORT}" \
    >>"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 20); do
    curl -fsS -m 3 "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null && { log "proxy ready"; break; }
    sleep 1
done

log "running benchmark label=${VAST_BENCH_LABEL}"
cd /work/skill
# openclaude-vllm is the wrapper installed by the Singularity recipe; it
# invokes the openclaude binary with --provider openai so OPENAI_BASE_URL
# below points it at our local proxy. trial/invoke.py honors $CLAUDE_BIN.
export CLAUDE_BIN="${CLAUDE_BIN:-/usr/local/bin/openclaude-vllm}"
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export RF_SKILL_HINT_LEVEL=minimal
export OPENAI_API_STREAM=0
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export CLAUDE_CODE_TOOLS_OVERRIDE="Bash,Edit,Read,Write,Glob,Grep"
# Chroma persistent client in scratch — no reverse SSH tunnel on Anvil.
# store.py honors SIONNA_SKILL_CHROMA_PATH (see store.py:_resolve_db_path).
# Empty CHROMA_HOST disables the HttpClient code path.
export CHROMA_HOST=""
export SIONNA_SKILL_CHROMA_PATH="/work/scratch/chroma_db"
mkdir -p "$SIONNA_SKILL_CHROMA_PATH"

# Pre-generate the self_gen skill for this model if absent. trial/invoke.py
# expects benchmark/self_gen_skill/<model_basename>/SKILL.md (where basename
# = lowercased final segment of MODEL_HF). If the file is missing all
# self_gen trials abort with "self_gen skill not found".
MODEL_BASENAME=$(echo "$MODEL_HF" | awk -F/ '{print tolower($NF)}')
SELF_GEN_PATH="benchmark/self_gen_skill/${MODEL_BASENAME}/SKILL.md"
if [[ ! -s "$SELF_GEN_PATH" ]]; then
    log "generating self_gen skill: $SELF_GEN_PATH"
    mkdir -p "$(dirname "$SELF_GEN_PATH")"
    python3 benchmark/generate_baseline_skill.py \
        --backend openai-compat --model "$MODEL_HF" \
        --out-file "$SELF_GEN_PATH" \
        || log "WARN: self_gen skill generation failed; self_gen trials will be skipped"
else
    log "self_gen skill already present: $SELF_GEN_PATH"
fi

BENCH_SPLIT="${BENCH_SPLIT:-train}"
BENCH_CONDITIONS="${BENCH_CONDITIONS:-with_skill no_skill self_gen}"
HELD_OUT_FLAG=()
[[ "$BENCH_SPLIT" == "test" || "$BENCH_SPLIT" == "all" ]] && HELD_OUT_FLAG=( --i-understand-held-out )
log "benchmark split=$BENCH_SPLIT conditions=$BENCH_CONDITIONS"
python3 benchmark/run_benchmark.py \
    --label "$VAST_BENCH_LABEL" --shuffle-seed 42 --timeout 1200 \
    --split "$BENCH_SPLIT" --conditions $BENCH_CONDITIONS --k 1 \
    "${HELD_OUT_FLAG[@]}" \
    --workers "${VAST_BENCH_WORKERS:-14}" \
    --max-turns 25 --model "$MODEL_HF"
RC=$?

cleanup
exit $RC
