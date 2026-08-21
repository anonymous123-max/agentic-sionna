#!/bin/bash
#SBATCH --job-name=skill-bench
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/%x-%j.err
#
# anvil_run_model.sbatch — single-model benchmark job for Purdue Anvil.
#
# Spawned by submit_loop.sh, one job per model in the sweep. Equivalent
# to one iteration of run_loop_v3_tp.sh on vast.ai, but runs to completion
# inside this single SLURM job (no tmux, no persistent vLLM service).
#
# Required env vars (set via sbatch --export when calling, or manually):
#   MODEL_HF        HuggingFace repo id (e.g., Qwen/Qwen3.6-27B)
#   MODEL_PARSER    tool-call parser hint (e.g., hermes / qwen3_coder / llama3_json / gemma)
#   USE_DFLASH      "1" to enable DFlash speculative decoding, "0" for baseline
#   ANVIL_ALLOC     ACCESS allocation name (passed via -A on the command line, but
#                   we re-export here for visibility)
#
# Optional:
#   VAST_TP_SIZE              default 2 (2× A100 or 2× H100)
#   VAST_BENCH_WORKERS        default 14
#   VAST_DFLASH_SPEC_TOKENS   default 15
#
# Where files live (inside the job):
#   /anvil/scratch/$USER/skill/        the project working tree (rsync'd once)
#   /anvil/scratch/$USER/skill_dflash.sif     Singularity image (built once)
#   /anvil/scratch/$USER/hf_cache/     pre-staged HuggingFace cache
#   /anvil/scratch/$USER/skill/benchmark/results/<label>/    results
#
# Why partition=gpu (not gpu-debug): debug is 30 min walltime; full bench
# needs ~1-3 hours. Switch to gpu-debug for sinteractive smoke testing only.
#
# Why --gpus-per-node=2: matches our TP=2 tooling from vast.ai. Bump to 4
# if model needs more memory (e.g., Llama-3.3-70B-AWQ at FP16). 70B AWQ
# fits on 2× A100; FP16 needs 4×.

set -uo pipefail

# Required vars.
: "${MODEL_HF:?MODEL_HF must be set, e.g., Qwen/Qwen3.6-27B}"
: "${MODEL_PARSER:?MODEL_PARSER must be set, e.g., hermes}"
USE_DFLASH="${USE_DFLASH:-0}"

# Pick the right .sif based on USE_DFLASH and family.
SCRATCH="/anvil/scratch/$USER"
SIF_QWEN_LLAMA="$SCRATCH/skill_dflash.sif"
SIF_GEMMA="$SCRATCH/skill_dflash_gemma.sif"
SIF_BASELINE="$SCRATCH/skill_baseline.sif"

case "$MODEL_HF" in
    *Gemma*|*gemma*) SIF_DFLASH="$SIF_GEMMA" ;;
    *)               SIF_DFLASH="$SIF_QWEN_LLAMA" ;;
esac
if [[ "$USE_DFLASH" == "1" ]]; then
    SIF="$SIF_DFLASH"
else
    SIF="$SIF_BASELINE"
fi
# 2026-05-15: per-job override for models needing flash-attn 2.7+ (Granite-4.0,
# possibly Gemma-3 for reasoning). Set SIF_OVERRIDE=v2 in --export to pick
# skill_baseline_v2.sif.
if [[ "${SIF_OVERRIDE:-}" == "v2" ]]; then
    SIF="$SCRATCH/skill_baseline_v2.sif"
fi

if [[ ! -f "$SIF" ]]; then
    echo "✗ Singularity image missing: $SIF" >&2
    echo "  Build it from the login node before submitting jobs:" >&2
    echo "    singularity build $SIF benchmark/anvil/Singularity.<variant>" >&2
    exit 2
fi

PROJECT_ROOT="$SCRATCH/skill"
LABEL="anvil_$(echo "$MODEL_HF" | tr '/.-' '___')_$(date -u +%Y%m%d_%H%M)_dflash${USE_DFLASH}"
HF_HOME="$SCRATCH/hf_cache"

mkdir -p "$SCRATCH/skill/logs" "$HF_HOME"

echo "========================================================"
echo "anvil_run_model.sbatch"
echo "  MODEL_HF=$MODEL_HF"
echo "  MODEL_PARSER=$MODEL_PARSER"
echo "  USE_DFLASH=$USE_DFLASH"
echo "  SIF=$SIF"
echo "  LABEL=$LABEL"
echo "  GPUs allocated: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Wall budget: ${SLURM_TIMELIMIT:-unknown}"
echo "  Started:    $(date -u)"
echo "========================================================"

module load singularity 2>/dev/null || true

# All real work happens inside the container so paths/libs are deterministic.
# --nv exposes GPUs; --bind exposes scratch + project + hf cache; --pwd
# sets working directory to the benchmark root.
singularity exec --nv --writable-tmpfs \
    --bind "$PROJECT_ROOT:/work/skill" \
    --bind "$HF_HOME:/work/hf_cache" \
    --bind "$SCRATCH:/work/scratch" \
    --pwd /work/skill \
    --env "MODEL_HF=$MODEL_HF" \
    --env "MODEL_PARSER=$MODEL_PARSER" \
    --env "USE_DFLASH=$USE_DFLASH" \
    --env "VAST_TP_SIZE=${VAST_TP_SIZE:-2}" \
    --env "VAST_BENCH_WORKERS=${VAST_BENCH_WORKERS:-14}" \
    --env "VAST_DFLASH_SPEC_TOKENS=${VAST_DFLASH_SPEC_TOKENS:-15}" \
    --env "VAST_BENCH_LABEL=$LABEL" \
    --env "VAST_AUTOLAUNCH_BENCHMARK=1" \
    --env "VAST_SKIP_REPO_RSYNC=1" \
    --env "VAST_CATALOG_SRC=skip" \
    --env "VAST_REQUIRE_CATALOG=0" \
    --env "HF_HOME=/work/hf_cache" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "PROJECT_ROOT=/work/skill" \
    --env "RESULTS_DIR=/work/skill/benchmark/results" \
    --env "PYTHONNOUSERSITE=1" \
    --env "BENCH_SPLIT=${BENCH_SPLIT:-train}" \
    --env "BENCH_CONDITIONS=${BENCH_CONDITIONS:-with_skill no_skill self_gen}" \
    --env "VAST_MAX_MODEL_LEN=${VAST_MAX_MODEL_LEN:-32768}" \
    "$SIF" \
    bash benchmark/anvil/run_inside_container.sh

RC=$?
echo "========================================================"
echo "anvil_run_model.sbatch finished rc=$RC at $(date -u)"
echo "  results in: $PROJECT_ROOT/benchmark/results/$LABEL"
echo "========================================================"
exit $RC
