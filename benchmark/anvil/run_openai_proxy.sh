#!/bin/bash
# Run rf-simulator benchmark via openclaude (inside Singularity baseline.sif)
# pointed at a custom OpenAI-compatible proxy. One-shot k=1, full task set.
#
# Args:
#   $1  model id (e.g. gpt-4o, gpt-5.4-mini)
#   $2  label suffix (e.g. gpt4o or gpt54mini)
set -uo pipefail
MODEL="${1:?model id required}"
SUFFIX="${2:?label suffix required}"

SCRATCH=/anvil/scratch/$USER
PROJECT_ROOT=$SCRATCH/skill
HF_HOME=$SCRATCH/hf_cache
SIF=$SCRATCH/skill_baseline.sif

LABEL="openai_${SUFFIX}_$(date -u +%Y%m%d_%H%M)"
LOG=$PROJECT_ROOT/logs/${LABEL}.log
echo "[start $(date -u)] label=$LABEL model=$MODEL" | tee "$LOG"

# Run the benchmark INSIDE the container so openclaude is on PATH.
module load singularity 2>/dev/null || true
singularity exec \
    --bind "$PROJECT_ROOT:/work/skill" \
    --bind "$HF_HOME:/work/hf_cache" \
    --bind "$SCRATCH:/work/scratch" \
    --pwd /work/skill \
    --env "CLAUDE_BIN=openclaude" \
    --env "CLAUDE_CODE_USE_OPENAI=1" \
    --env "OPENAI_API_KEY=${OPENAI_API_KEY:?must be set}" \
    --env "OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://148.113.224.153:3000/v1}" \
    --env "OPENCLAUDE_EXTRA_FLAGS=--bare --dangerously-skip-permissions" \
    --env "RF_SKILL_DIR=/work/skill/.claude/skills/rf-simulator" \
    --env "RF_NO_PROMPT=1" \
    --env "PYTHONNOUSERSITE=1" \
    --env "HF_HOME=/work/hf_cache" \
    "$SIF" \
    python3 benchmark/run_benchmark.py \
        --label "$LABEL" \
        --shuffle-seed 42 \
        --timeout 600 \
        --split all \
        --i-understand-held-out \
        --conditions with_skill no_skill \
        --k 1 \
        --workers 4 \
        --max-turns 15 \
        --model "$MODEL" 2>&1 | tee -a "$LOG"

echo "[end $(date -u)] rc=$?" | tee -a "$LOG"
