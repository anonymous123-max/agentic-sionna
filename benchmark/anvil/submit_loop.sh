#!/bin/bash
# submit_loop.sh — driver that submits one anvil_run_model.sbatch job per
# model. Equivalent to run_loop_v3_tp.sh on vast.ai, but instead of looping
# in tmux it queues independent SLURM jobs that can run in parallel
# (subject to the 12-GPUs-per-user / 32-per-allocation limit on Anvil).
#
# Usage:
#   bash benchmark/anvil/submit_loop.sh -A <ACCESS_alloc>
#   USE_DFLASH=1 bash benchmark/anvil/submit_loop.sh -A <ACCESS_alloc>
#   PARTITION=gpu-debug bash benchmark/anvil/submit_loop.sh -A <ACCESS_alloc>  # for smoke
#
# Required args:
#   -A | --account    ACCESS allocation name (e.g., MED230001)
# Optional env:
#   USE_DFLASH        0|1 (default 0)
#   PARTITION         gpu | gpu-debug (default gpu)
#   WALLTIME          HH:MM:SS (default 04:00:00 — well below the 48h cap;
#                     bump to 24:00:00 for k=3 sampling sweeps)
#   MODELS_FILTER     regex to subset the MODELS list (e.g., 'Qwen3.6-27B')
#   GPUS_PER_NODE     2 (default; 4 for FP16 70B)

set -euo pipefail

ALLOC=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -A|--account) ALLOC="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$ALLOC" ]]; then
    echo "✗ -A/--account is required (your ACCESS allocation name)" >&2
    exit 1
fi

USE_DFLASH="${USE_DFLASH:-0}"
PARTITION="${PARTITION:-gpu}"
WALLTIME="${WALLTIME:-04:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MODELS_FILTER="${MODELS_FILTER:-.}"

# Same 6 models as run_loop_v3_tp.sh.
MODELS=(
    "Qwen/Qwen3.6-27B                          hermes"
    "Qwen/Qwen3.6-35B-A3B                      hermes"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct         qwen3_coder"
    "casperhansen/llama-3.3-70b-instruct-awq   llama3_json"
    "google/gemma-4-31B-it                     gemma"
    "google/gemma-4-26B-A4B-it                 gemma"
)

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$THIS_DIR/anvil_run_model.sbatch"
[[ -f "$SBATCH_FILE" ]] || { echo "✗ missing $SBATCH_FILE" >&2; exit 1; }

echo "========================================================"
echo "submit_loop.sh"
echo "  alloc=$ALLOC partition=$PARTITION walltime=$WALLTIME"
echo "  USE_DFLASH=$USE_DFLASH GPUS_PER_NODE=$GPUS_PER_NODE"
echo "  filter=$MODELS_FILTER"
echo "========================================================"

JOB_IDS=()
for MODEL_CONFIG in "${MODELS[@]}"; do
    set -- $MODEL_CONFIG
    HF=$1; PARSER=$2

    if [[ ! "$HF" =~ $MODELS_FILTER ]]; then
        echo "  - skip $HF (filtered out)"
        continue
    fi

    JOB_NAME="skill-$(echo "$HF" | tr '/.' '__' | cut -c1-30)"
    echo "▶ submitting $HF (parser=$PARSER) → job-name=$JOB_NAME"

    JID=$(sbatch \
        --parsable \
        --account="$ALLOC" \
        --partition="$PARTITION" \
        --time="$WALLTIME" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --job-name="$JOB_NAME" \
        --export=ALL,MODEL_HF="$HF",MODEL_PARSER="$PARSER",USE_DFLASH="$USE_DFLASH" \
        "$SBATCH_FILE")
    echo "  → job_id=$JID"
    JOB_IDS+=("$JID")
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs:"
printf '  %s\n' "${JOB_IDS[@]}"
echo ""
echo "Watch them: squeue -u \$USER"
echo "After completion: ls /anvil/scratch/\$USER/skill/benchmark/results/"
