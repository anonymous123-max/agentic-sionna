#!/bin/bash
# run_loop_v3_tp.sh — orchestrator for 2× A100 SXM4 NVLink with TP=2.
# Iterates 6 models (Tier 1 + 2), bootstrapping each via bootstrap_vastai_tp.sh.
# Difference vs v2 (which targeted 8× A100 PCIe via multi-instance): single
# vLLM with --tensor-parallel-size 2 sharded across both NVLink GPUs.
#
# Usage: bash /root/run_loop_v3_tp.sh
#
# Env overrides:
#   HF_TOKEN     required for gated models (Llama 3.3, Gemma 4)
#   USE_DFLASH=1 enable DFlash speculative decoding (z-lab/dflash). Drafts
#                published for 5 of 6 models below; Llama 3.3-70B-AWQ has
#                no draft and runs baseline unchanged.
#                Requires DFlash-capable vLLM build (mainline or PR #40898
#                for Qwens; PR #41703 for Gemmas — separate env recommended).
set -uo pipefail
exec > >(tee -a /workspace/loop.log) 2>&1

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
USE_DFLASH="${USE_DFLASH:-0}"

MODELS=(
    "Qwen/Qwen3.6-27B                          hermes"
    "Qwen/Qwen3.6-35B-A3B                      hermes"
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
    echo "  TP=2 across 2× A100 SXM4 NVLink"
    echo "════════════════════════════════════════════════"

    # Kill any leftover sessions from prior model. tp.sh creates session names
    # `vllm`, `proxy0`, `benchmark` (NOT `vllm_tp`/`bench` — bug from earlier
    # version). The grep below for the wait-loop must use ^benchmark:.
    for s in vllm proxy0 benchmark; do
        tmux kill-session -t "$s" 2>/dev/null
    done
    sleep 5

    # Note: PARSER above is just a default; the bootstrap's per-model case
    # block in step 11 OVERRIDES it based on $VAST_MODEL_HF (Qwen3.6 →
    # qwen3_xml, Qwen3-Coder → qwen3_coder, Llama-AWQ → llama3_json+
    # awq_marlin, Gemma → drops both flags). So `hermes` as a default is
    # harmless — never actually used after the case block runs.
    # Max-throughput TP=2 settings on 2× A100 SXM4 NVLink:
    # - max-num-seqs=32: KV pool fits ~27 seqs at 32K context for Qwen3.6-27B BF16,
    #   slight over-provision lets short-prefix sharing absorb the gap
    # - workers=28: aggressive concurrency. Risk: brief queueing if 28 trials all
    #   want full 32K context; at typical bench prompt sizes (~5K) we can run ~50.
    VAST_TP_SIZE=2 \
    VAST_MODEL_HF="$HF" \
    VAST_MODEL_PARSER="$PARSER" \
    VAST_MODEL_NUM_SEQS_TP=16 \
    VAST_BENCH_WORKERS=14 \
    VAST_BENCH_LABEL="$LABEL" \
    VAST_AUTOLAUNCH_BENCHMARK=1 \
    VAST_SKIP_REPO_RSYNC=1 \
    VAST_CATALOG_SRC=skip \
    VAST_REQUIRE_CATALOG=0 \
    USE_DFLASH="$USE_DFLASH" \
    bash /root/bootstrap_vastai_tp.sh
    BOOT_RC=$?
    if [[ "$BOOT_RC" != "0" ]]; then
        echo "✗ bootstrap returned $BOOT_RC for $HF; skipping"
        continue
    fi

    echo "[$(date +%H:%M:%S)] bootstrap returned 0; benchmark running. Waiting for bench..."
    # tp.sh launches ONE benchmark process in tmux session named `benchmark`
    # (NOT `bench` — that was a bug). Trial count + pass rate per loop tick.
    # P0.4: also poll for progress delta. If no new result.json for 10 min
    # (5 ticks × 120 s), kill the bench session and move on. This catches
    # the failure mode from 2026-05-06 where tmux session presence flipped
    # false-positive when an iteration takeover stalled the worker.
    LAST_DONE=0
    STALL_TICKS=0
    while tmux ls 2>/dev/null | grep -qE "^benchmark:"; do
        DONE=$(find /workspace/skill/benchmark/results/$LABEL -name result.json 2>/dev/null | wc -l)
        if [[ "$DONE" -gt "$LAST_DONE" ]]; then
            STALL_TICKS=0
            LAST_DONE=$DONE
        else
            STALL_TICKS=$((STALL_TICKS + 1))
        fi
        echo "  [$(date +%H:%M:%S)] benchmark running: $DONE trials done (stall=${STALL_TICKS}/5)"
        if [[ "$STALL_TICKS" -ge 5 ]]; then
            echo "  ! benchmark stalled (no progress for 10 min); killing and moving on"
            tmux kill-session -t benchmark 2>/dev/null
            break
        fi
        sleep 120
    done

    echo "[$(date +%H:%M:%S)] benchmark session exited. aggregating proxy token usage..."
    # Token counts in result.json.usage are zero because openclaude doesn't
    # translate vLLM's prompt_tokens/completion_tokens to Claude format.
    # The proxy captures it to /workspace/logs/proxy_usage_*.jsonl; this
    # join attributes per-trial usage by time-window. Idempotent (only
    # adds usage_from_proxy field; existing fields untouched).
    python3 /workspace/skill/benchmark/analysis/aggregate_token_usage.py \
        --results-dir /workspace/skill/benchmark/results/$LABEL \
        --proxy-log-dir /workspace/logs \
        2>&1 | tail -5

    echo "[$(date +%H:%M:%S)] $HF complete"
done

echo ""; echo "ALL 6 MODELS COMPLETE at $(date +%H:%M:%S)"
ls /workspace/skill/benchmark/results/ 2>&1 | head -30
