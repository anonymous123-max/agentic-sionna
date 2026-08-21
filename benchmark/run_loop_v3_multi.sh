#!/bin/bash
# run_loop_v3_multi.sh — orchestrator for 2× A100 SXM4 with VAST_NUM_INSTANCES=2.
# Falls back to multi-instance pattern after TP=2 hung on shm_broadcast (memory
# iter 19 + 2026-05-06 confirmation).
#
# Each GPU runs ONE TP=1 vLLM instance. 2 chunks parallel × 2 workers = 4
# concurrent trials per model. Lower throughput than TP=2 but reliable.
#
# 35B-A3B (~70 GB BF16) is SKIPPED — won't fit single 80GB A100 with KV space.
# Run separately if needed via tp.sh once NCCL hang is resolved.
#
# Usage: bash /root/run_loop_v3_multi.sh
#
# Env overrides:
#   USE_DFLASH=1 enable DFlash speculative decoding for supported models
#                (Qwen3.6-27B, Qwen3-Coder-30B, Gemma-4-31B/26B). Llama 3.3-70B
#                has no published draft; runs baseline. See run_loop_v3_tp.sh
#                header for vLLM PR build requirements.
set -uo pipefail
exec > >(tee -a /workspace/loop.log) 2>&1

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
USE_DFLASH="${USE_DFLASH:-0}"

MODELS=(
    "Qwen/Qwen3.6-27B                          hermes"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct         qwen3_coder"
    "casperhansen/llama-3.3-70b-instruct-awq   llama3_json"
    "google/gemma-4-31B-it                     gemma"
    "google/gemma-4-26B-A4B-it                 gemma"
)

for MODEL_CONFIG in "${MODELS[@]}"; do
    set -- $MODEL_CONFIG
    HF=$1; PARSER=$2
    LABEL="train_$(echo $HF | tr '/.-' '___')"

    echo ""; echo "════════════════════════════════════════════════"
    echo "▶ [$(date +%H:%M:%S)] Starting model: $HF (parser hint=$PARSER) label=$LABEL"
    echo "  multi-instance × 2 (one TP=1 vLLM per GPU)"
    echo "════════════════════════════════════════════════"

    # Kill any leftover sessions from prior model
    for i in 0 1; do
        tmux kill-session -t "vllm$i"  2>/dev/null
        tmux kill-session -t "proxy$i" 2>/dev/null
        tmux kill-session -t "bench$i" 2>/dev/null
    done
    sleep 5

    # Per-model parser logic lives in bootstrap_vastai_multi.sh's case block
    # (Qwen3.6 → qwen3_xml + reasoning + thinking-disabled, Qwen3-Coder → qwen3_coder,
    #  Llama-AWQ → llama3_json + awq_marlin, Gemma → drop both flags).
    # PARSER hint here is fallback for models not in the case block.
    VAST_NUM_INSTANCES=2 \
    VAST_MODEL_HF="$HF" \
    VAST_MODEL_PARSER="$PARSER" \
    VAST_BENCH_WORKERS=2 \
    VAST_BENCH_LABEL="$LABEL" \
    VAST_AUTOLAUNCH_BENCHMARK=1 \
    VAST_SKIP_REPO_RSYNC=1 \
    VAST_CATALOG_SRC=skip \
    VAST_REQUIRE_CATALOG=0 \
    USE_DFLASH="$USE_DFLASH" \
    bash /root/bootstrap_vastai_multi.sh
    BOOT_RC=$?
    if [[ "$BOOT_RC" != "0" ]]; then
        echo "✗ bootstrap returned $BOOT_RC for $HF; skipping"
        continue
    fi

    echo "[$(date +%H:%M:%S)] bootstrap returned 0; benchmark running. Waiting for chunks..."
    # P0.4: also poll aggregate progress across chunk0+chunk1 dirs. If no
    # new result.json across either chunk for 10 min (5 ticks × 120 s), kill
    # remaining bench sessions and move on. Covers the case from 2026-05-06
    # where a chunk's worker stalled but its tmux session lingered.
    LAST_DONE=0
    STALL_TICKS=0
    while tmux ls 2>/dev/null | grep -qE "^bench[0-1]:"; do
        ACTIVE=$(tmux ls 2>&1 | grep -cE "^bench[0-1]:")
        DONE0=$(find /workspace/skill/benchmark/results/${LABEL}_chunk0 -name result.json 2>/dev/null | wc -l)
        DONE1=$(find /workspace/skill/benchmark/results/${LABEL}_chunk1 -name result.json 2>/dev/null | wc -l)
        DONE=$((DONE0 + DONE1))
        if [[ "$DONE" -gt "$LAST_DONE" ]]; then
            STALL_TICKS=0
            LAST_DONE=$DONE
        else
            STALL_TICKS=$((STALL_TICKS + 1))
        fi
        echo "  [$(date +%H:%M:%S)] $ACTIVE bench session(s) running: $DONE trials done (chunk0=$DONE0, chunk1=$DONE1, stall=${STALL_TICKS}/5)"
        if [[ "$STALL_TICKS" -ge 5 ]]; then
            echo "  ! benchmark stalled (no progress for 10 min); killing chunks and moving on"
            for i in 0 1; do
                tmux kill-session -t "bench$i" 2>/dev/null
            done
            break
        fi
        sleep 120
    done

    echo "[$(date +%H:%M:%S)] all chunks done. merging..."
    cd /workspace/skill && python3 benchmark/merge_chunks.py \
        --label-prefix "$LABEL" --num-chunks 2 2>&1 | tail -10

    echo "[$(date +%H:%M:%S)] aggregating proxy token usage into chunk dirs..."
    # See run_loop_v3_tp.sh comment: openclaude doesn't translate vLLM
    # token counts; proxy logs them sidecar. Run on each chunk dir.
    for c in 0 1; do
        python3 /workspace/skill/benchmark/analysis/aggregate_token_usage.py \
            --results-dir /workspace/skill/benchmark/results/${LABEL}_chunk${c} \
            --proxy-log-dir /workspace/logs \
            2>&1 | tail -3
    done

    echo "[$(date +%H:%M:%S)] $HF complete"
done

echo ""; echo "ALL 5 MODELS COMPLETE at $(date +%H:%M:%S)"
echo "(35B-A3B skipped — needs TP=2 which hangs on shm_broadcast)"
ls /workspace/skill/benchmark/results/ 2>&1 | head -30
