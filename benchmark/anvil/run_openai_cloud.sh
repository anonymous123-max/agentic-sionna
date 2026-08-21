#!/bin/bash
# Run the rf-simulator benchmark against an OpenAI cloud model via `claude -p`
# in CLAUDE_CODE_USE_OPENAI mode. One-shot k=1 over the full task set.
#
# Args:
#   $1  model id (e.g. gpt-4o, gpt-5.4-mini)
#   $2  label suffix (e.g. gpt4o or gpt54mini)
set -uo pipefail
MODEL="${1:?model id required, e.g. gpt-4o}"
SUFFIX="${2:?label suffix required}"

module load python/3.9.5 2>/dev/null || true
PYTHON=$(which python3.9 2>/dev/null || which python3 2>/dev/null)

cd /anvil/scratch/x-jsong16/skill

# Route claude CLI to OpenAI instead of Anthropic.
export CLAUDE_BIN=claude
export CLAUDE_CODE_USE_OPENAI=1
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://148.113.224.153:3000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?must be passed via env}"

export RF_SKILL_DIR=/anvil/scratch/x-jsong16/skill/.claude/skills/rf-simulator
export RF_NO_PROMPT=1
export PYTHONNOUSERSITE=1

LABEL="openai_${SUFFIX}_$(date -u +%Y%m%d_%H%M)"
LOG=/anvil/scratch/x-jsong16/skill/logs/${LABEL}.log
echo "[start $(date -u)] label=$LABEL model=$MODEL" | tee "$LOG"

"$PYTHON" benchmark/run_benchmark.py \
    --label "$LABEL" \
    --shuffle-seed 42 \
    --timeout 600 \
    --split all \
    --i-understand-held-out \
    --conditions with_skill no_skill \
    --k 1 \
    --workers 6 \
    --max-turns 20 \
    --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "[end $(date -u)] rc=$?" | tee -a "$LOG"
