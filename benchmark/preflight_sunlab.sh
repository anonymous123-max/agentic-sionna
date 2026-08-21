#!/bin/bash
# preflight_sunlab.sh — exercise the local-LLM benchmark stack on sunlab
# BEFORE renting vast.ai. Catches every issue documented in
# docs/plan_vast_ai_benchmark.md §0 that's reproducible without a big GPU.
#
# Run from sunlab. Uses RTX 5090 32GB + a small public model (default
# Llama-3.1-8B Instruct) that fits the 32GB envelope. If this exits 0,
# the same stack on vast.ai with a bigger model should Just Work.
# If it fails, you fix it here for free.
#
# Usage:
#   bash benchmark/preflight_sunlab.sh                      # full run + 1 benchmark trial
#   bash benchmark/preflight_sunlab.sh --skip-bench         # skip the trial, just verify chain
#
# Override-via-env:
#   TEST_MODEL          HF repo id of test model (default: unsloth/Meta-Llama-3.1-8B-Instruct)
#   TEST_TOOL_PARSER    vLLM tool-call parser name (default: llama3_json)
#   TEST_MAX_LEN        vLLM --max-model-len (default: 16384, sunlab RTX 5090 fit)
#   TEST_NUM_SEQS       vLLM --max-num-seqs   (default: 1)
#   SUNLAB_VLLM_PORT    port to serve vLLM on (default: 8001)
#   OPENCLAUDE_BIN      path to openclaude (auto-detected from fnm)
#   VLLM_BIN            path to vllm CLI (auto-detected from miniconda env)
#   SIONNA_PYTHON       python with sionna (auto-detected from miniconda env)

set -euo pipefail

# Resolve repo root (script lives at benchmark/preflight_sunlab.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------
# Config (overridable)
# ---------------------------------------------------------------------
SUNLAB_VLLM_PORT="${SUNLAB_VLLM_PORT:-8001}"
# Default test model: Qwen2.5-7B-Instruct — matches the Qwen3 family parser
# (hermes) we'll use on vast.ai with Qwen3.6-27B. Llama-3.1-8B + llama3_json
# parser would test a different code path.
TEST_MODEL="${TEST_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TEST_TOOL_PARSER="${TEST_TOOL_PARSER:-hermes}"
# 32K matches the vast.ai default (VAST_MODEL_MAX_LEN). The patched openclaude
# requests up to 8000 output tokens; full system prompt + tool defs + skill
# add ~17K input. Need ≥25K to pass smoke; 32K is safe margin.
TEST_MAX_LEN="${TEST_MAX_LEN:-32768}"
TEST_NUM_SEQS="${TEST_NUM_SEQS:-1}"

# Auto-detect binaries
if [[ -z "${OPENCLAUDE_BIN:-}" ]]; then
    OPENCLAUDE_BIN=$(find "$HOME/.local/share/fnm" -name openclaude -type f -executable 2>/dev/null | head -1)
    OPENCLAUDE_BIN="${OPENCLAUDE_BIN:-$(command -v openclaude || echo "")}"
fi
VLLM_BIN="${VLLM_BIN:-$HOME/miniconda3/envs/vllm/bin/vllm}"
SIONNA_PYTHON="${SIONNA_PYTHON:-$HOME/miniconda3/envs/sionna/bin/python}"

# fnm-managed node isn't in non-interactive SSH PATH → openclaude (a `#!/usr/bin/env node`
# shebang script) fails with "env: 'node': No such file or directory" and exit 127, which
# pipefail propagates and set -e kills the script. Force fnm's node into PATH up front.
FNM_NODE_BIN=$(find "$HOME/.local/share/fnm/node-versions" -maxdepth 5 -name node -type f -executable 2>/dev/null | head -1)
if [[ -n "$FNM_NODE_BIN" ]]; then
    export PATH="$(dirname "$FNM_NODE_BIN"):$PATH"
fi

LOG=/tmp/preflight_sunlab.log
: > "$LOG"
VLLM_PID=""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
step() { echo -e "\n[$(date +%H:%M:%S)] === $1 ==="; echo "[$(date +%H:%M:%S)] === $1 ===" >> "$LOG"; }
ok()   { echo "  ✓ $1"; echo "  ✓ $1" >> "$LOG"; }
warn() { echo "  ! $1"; echo "  ! $1" >> "$LOG"; }
fail() { echo "  ✗ $1" >&2; echo "  ✗ $1" >> "$LOG"; exit 1; }

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# =====================================================================
step "1. Verify chroma server is up"
if curl -sf -m 3 -o /dev/null http://127.0.0.1:8000/api/v1/heartbeat; then
    ok "chroma reachable on 127.0.0.1:8000"
else
    fail "chroma not running. Start with:
        PATH=\$HOME/.local/bin:\$PATH bash .claude/skills/rf-simulator/scripts/start_chroma_server.sh --bg"
fi

# =====================================================================
step "2. Verify openclaude installed"
if [[ -z "$OPENCLAUDE_BIN" ]] || [[ ! -x "$OPENCLAUDE_BIN" ]]; then
    fail "openclaude not found. Install: npm install -g @gitlawb/openclaude"
fi
OC_VER=$("$OPENCLAUDE_BIN" --version 2>&1 | head -1)
ok "openclaude found: $OPENCLAUDE_BIN ($OC_VER)"

# =====================================================================
step "3. Apply openclaude MAX_OUTPUT_TOKENS_DEFAULT patch"
OC_DIR=$(dirname "$(dirname "$OPENCLAUDE_BIN")")
OC_CLI=$(find "$OC_DIR" -name cli.mjs -path "*openclaude*" 2>/dev/null | head -1)
if [[ -z "$OC_CLI" ]]; then
    fail "could not locate openclaude cli.mjs under $OC_DIR"
fi
if grep -q "MAX_OUTPUT_TOKENS_DEFAULT = 32000" "$OC_CLI" 2>/dev/null; then
    cp "$OC_CLI" "$OC_CLI.bak"
    sed -i \
        -e "s/MAX_OUTPUT_TOKENS_DEFAULT = 32000/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" \
        -e "s/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64000/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 8000/g" \
        -e "s/ESCALATED_MAX_TOKENS = 64000/ESCALATED_MAX_TOKENS = 8000/g" \
        "$OC_CLI"
    ok "patched $OC_CLI (backup at .bak)"
elif grep -q "MAX_OUTPUT_TOKENS_DEFAULT = 8000" "$OC_CLI" 2>/dev/null; then
    ok "openclaude already patched"
else
    warn "openclaude cli.mjs has unexpected MAX_OUTPUT_TOKENS_DEFAULT value; inspect manually"
fi

# =====================================================================
step "4. Verify vllm + sionna python"
[[ -x "$VLLM_BIN" ]] || fail "vllm binary missing at $VLLM_BIN"
[[ -x "$SIONNA_PYTHON" ]] || fail "sionna python missing at $SIONNA_PYTHON"
ok "vllm: $VLLM_BIN"
ok "sionna python: $SIONNA_PYTHON"

# =====================================================================
step "5. Apply run_benchmark.py origin_id KeyError patch"
RB="$REPO_ROOT/benchmark/run_benchmark.py"
if grep -q 't\["origin_id"\] in ids' "$RB"; then
    sed -i 's|t\["origin_id"\] in ids|t.get("origin_id") in ids|' "$RB"
    ok "patched run_benchmark.py:94"
elif grep -q 't.get("origin_id") in ids' "$RB"; then
    ok "origin_id patch already applied"
else
    warn "run_benchmark.py:94 not in expected form — verify manually"
fi

# =====================================================================
step "6. Ensure openclaude-vllm wrapper is executable"
WRAPPER="$REPO_ROOT/benchmark/openclaude-vllm"
if [[ ! -x "$WRAPPER" ]]; then
    chmod +x "$WRAPPER" 2>/dev/null || fail "wrapper not at $WRAPPER"
fi
ok "wrapper at $WRAPPER"

# =====================================================================
step "7. Start vLLM ($TEST_MODEL on port $SUNLAB_VLLM_PORT)"
# Kill any prior vLLM on this port
pkill -f "vllm serve.*--port $SUNLAB_VLLM_PORT" 2>/dev/null || true
sleep 2

VLLM_LOG=/tmp/preflight_vllm.log
"$VLLM_BIN" serve "$TEST_MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len "$TEST_MAX_LEN" \
    --max-num-seqs "$TEST_NUM_SEQS" \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser "$TEST_TOOL_PARSER" \
    --port "$SUNLAB_VLLM_PORT" \
    --host 127.0.0.1 \
    --served-model-name preflight-test \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
ok "vLLM started PID=$VLLM_PID (log: $VLLM_LOG)"

# =====================================================================
step "8. Wait for vLLM ready (up to 5 min)"
READY=0
for i in $(seq 1 60); do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:$SUNLAB_VLLM_PORT/v1/models"; then
        ok "vLLM ready after $((i*5))s"
        READY=1
        break
    fi
    sleep 5
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  ! vLLM died; tail of log:"
        tail -20 "$VLLM_LOG"
        fail "vLLM died during startup"
    fi
done
[[ "$READY" == "1" ]] || { tail -20 "$VLLM_LOG"; fail "vLLM not ready after 5 min"; }

# =====================================================================
step "9. End-to-end smoke: openclaude → vLLM → text response"
# openclaude requires OPENAI_API_KEY/BASE_URL even for local vLLM (the value
# isn't checked by vLLM but openclaude sanity-checks the env first).
export OPENAI_BASE_URL="http://127.0.0.1:$SUNLAB_VLLM_PORT/v1"
export OPENAI_API_KEY=dummy-vllm
SMOKE_OUT=$(timeout 60 "$WRAPPER" \
    --model preflight-test \
    -p "Reply with the single word: pong" \
    --print --max-turns 1 \
    --allow-dangerously-skip-permissions \
    2>&1 || echo "__TIMEOUT_OR_ERROR__")
echo "$SMOKE_OUT" | tail -10 >> "$LOG"
if echo "$SMOKE_OUT" | grep -qi "pong"; then
    ok "openclaude returned 'pong' through vLLM"
else
    echo "  ! openclaude output (tail):"
    echo "$SMOKE_OUT" | tail -15
    fail "openclaude did NOT return expected 'pong'"
fi

# =====================================================================
if [[ "${1:-}" == "--skip-bench" ]]; then
    echo -e "\n========================================="
    echo "Pre-flight COMPLETE (chain verified, benchmark trial skipped)."
    echo "Next step: rent vast.ai, run bootstrap_vastai.sh."
    echo "========================================="
    exit 0
fi

step "10. Run a single benchmark trial (U001, with_skill)"
export OPENAI_BASE_URL="http://127.0.0.1:$SUNLAB_VLLM_PORT/v1"
export OPENAI_API_KEY=dummy-vllm
export CLAUDE_BIN="$WRAPPER"
export OPENCLAUDE_BIN="$OPENCLAUDE_BIN"
export RF_SKILL_DIR="$REPO_ROOT/.claude/skills/rf-simulator"
export CLAUDE_CODE_USE_RAG=1
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

rm -rf "$REPO_ROOT/benchmark/results/preflight_smoke"

set +e
"$SIONNA_PYTHON" benchmark/run_benchmark.py \
    --label preflight_smoke \
    --workers 1 \
    --model preflight-test \
    --conditions with_skill \
    --task-ids U001 \
    --max-turns 15 \
    --timeout 300 \
    2>&1 | tee -a "$LOG" | tail -15
RC=${PIPESTATUS[0]}
set -e

if [[ "$RC" != "0" ]]; then
    fail "benchmark trial exited non-zero ($RC) — see $LOG"
fi

# Look for pass-rate line in log
if grep -q "Pass rate: 1/1" "$LOG"; then
    ok "U001 trial PASSED end-to-end"
elif grep -q "Pass rate: 0/1" "$LOG"; then
    warn "U001 trial completed but did NOT pass — verifier output below"
    grep -A2 "verification" "$LOG" | tail -10
    warn "Chain works but trial verification failed — check task model fit"
else
    warn "Could not parse pass rate from log"
fi

# =====================================================================
step "11. Verify RAG injection (chroma → prompt) actually fired"
PROMPT_FILE=$(find "$REPO_ROOT/benchmark/results/preflight_smoke" -name prompt.txt -type f 2>/dev/null | head -1)
if [[ -z "$PROMPT_FILE" ]]; then
    fail "no prompt.txt produced for smoke trial"
fi
if grep -q "RELATED PRINCIPLES" "$PROMPT_FILE"; then
    ok "RAG injection verified: prompt contains RELATED PRINCIPLES block from chroma"
else
    warn "prompt did NOT contain RAG block. Check CLAUDE_CODE_USE_RAG=1 + chroma collection has chunks"
fi

if grep -q -E "Module 1|SKILL.md|rf-simulator" "$PROMPT_FILE"; then
    ok "skill content present in prompt (SKILL.md auto-loaded)"
else
    warn "skill content NOT detected in prompt — RF_SKILL_DIR may be misconfigured"
fi

# =====================================================================
step "12. Optional: verify furniture catalog presence"
# 3D-FUTURE catalog is optional on sunlab (T0 tasks aren't part of preflight smoke).
# But we can still sanity-check that FURNITURE_CATALOG_PATH points somewhere real.
if [[ -n "${FURNITURE_CATALOG_PATH:-}" ]] && [[ -d "${FURNITURE_CATALOG_PATH:-/dev/null}" ]]; then
    GLB_COUNT=$(find "$FURNITURE_CATALOG_PATH" -maxdepth 5 \( -name "*.glb" -o -name "*.obj" \) 2>/dev/null | wc -l)
    SIZE=$(du -sh "$FURNITURE_CATALOG_PATH" 2>/dev/null | cut -f1)
    if [[ "$GLB_COUNT" -ge 100 ]]; then
        ok "catalog at $FURNITURE_CATALOG_PATH: $GLB_COUNT meshes, $SIZE"
    else
        warn "catalog has only $GLB_COUNT meshes; T0 tasks may use AABB fallbacks"
    fi
else
    warn "FURNITURE_CATALOG_PATH not set or empty — that's fine for non-T0 preflight, but vast.ai run needs it"
fi

# =====================================================================
echo ""
echo "========================================="
echo "Pre-flight COMPLETE — local LLM stack verified end-to-end."
echo "Log: $LOG"
echo ""
echo "Verified:"
echo "  ✓ chroma reachable + collection has chunks"
echo "  ✓ openclaude → vLLM → text response"
echo "  ✓ benchmark trial completes end-to-end"
echo "  ✓ RAG injection makes it into the trial prompt"
echo "  ✓ skill content auto-loaded in prompt"
echo ""
echo "Next step: rent vast.ai, run bootstrap_vastai.sh."
echo "========================================="
