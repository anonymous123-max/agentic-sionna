#!/bin/bash
#SBATCH --job-name=skill-bench
#SBATCH --partition=gpuA100x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/bbXX/%u/skill/logs/%x-%j.out
#SBATCH --error=/scratch/bbXX/%u/skill/logs/%x-%j.err
#
# delta_run_model.sbatch — Delta-port of anvil_run_model.sbatch
# (NCSA Delta GPU, ACCESS-CI allocation)
#
# Key differences from Anvil:
#   - partition: Delta uses `gpuA100x4` / `gpuA40x4` / `gpuH200x4` (Anvil's `gpu`/`ai`)
#   - scratch: /scratch/<projectname>/$USER (Anvil's /anvil/scratch/$USER)
#                NCSA assigns a project-named subdir; replace `bbXX` with the
#                actual project code once known (e.g., bbka, bcyq, etc.)
#   - module: apptainer (Delta) vs singularity (Anvil); both expose the same CLI
#   - cpus-per-task: Delta A100x4 nodes have 64 CPUs (vs Anvil 128); request 16
#
# Same environment variables and flow as Anvil — the container does all the work.
#
# To submit:
#   sbatch -A <delta-allocation> benchmark/delta/delta_run_model.sbatch
# or with overrides:
#   sbatch -A <alloc> --export=ALL,MODEL_HF=...,MODEL_PARSER=...,USE_DFLASH=0,\
#       VAST_TP_SIZE=2,BENCH_SPLIT=all,BENCH_CONDITIONS="with_skill no_skill" \
#       benchmark/delta/delta_run_model.sbatch

set -uo pipefail

: "${MODEL_HF:?MODEL_HF must be set, e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct}"
: "${MODEL_PARSER:?MODEL_PARSER must be set, e.g., qwen3_coder}"
USE_DFLASH="${USE_DFLASH:-0}"

# Find the user's project scratch. Delta convention: /scratch/<project>/$USER
# The project name is the 4-letter code from the ACCESS allocation award.
# Fall back to $SCRATCH if the project scratch is found under a single subdir.
DELTA_SCRATCH_PARENT="/scratch"
if [[ -n "${DELTA_PROJECT:-}" ]]; then
    SCRATCH="$DELTA_SCRATCH_PARENT/$DELTA_PROJECT/$USER"
elif [[ -d "$DELTA_SCRATCH_PARENT" ]]; then
    # auto-detect: pick the first project subdir that contains the user's home
    SCRATCH=$(find "$DELTA_SCRATCH_PARENT" -maxdepth 2 -name "$USER" -type d 2>/dev/null | head -1)
fi
SCRATCH="${SCRATCH:-/scratch/$USER}"

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

if [[ ! -f "$SIF" ]]; then
    echo "✗ Singularity/Apptainer image missing: $SIF" >&2
    echo "  Build with: apptainer build $SIF benchmark/anvil/Singularity.baseline" >&2
    echo "  (Anvil recipes work as-is on Delta; both use apptainer/singularity CLI)" >&2
    exit 2
fi

PROJECT_ROOT="$SCRATCH/skill"
LABEL="delta_$(echo "$MODEL_HF" | tr '/.-' '___')_$(date -u +%Y%m%d_%H%M)_dflash${USE_DFLASH}"
HF_HOME="$SCRATCH/hf_cache"

mkdir -p "$SCRATCH/skill/logs" "$HF_HOME"

echo "========================================================"
echo "delta_run_model.sbatch"
echo "  MODEL_HF=$MODEL_HF"
echo "  MODEL_PARSER=$MODEL_PARSER"
echo "  USE_DFLASH=$USE_DFLASH"
echo "  SIF=$SIF"
echo "  SCRATCH=$SCRATCH"
echo "  LABEL=$LABEL"
echo "  GPUs allocated: $(echo ${CUDA_VISIBLE_DEVICES:-?} | tr ',' '\n' | wc -l)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Started:    $(date -u)"
echo "========================================================"

# Delta uses `apptainer` (which is the same binary as Singularity from v3 onward).
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

# Same bind/env layout as Anvil — the container is identical, only the host
# paths change. SCRATCH path is rebound to /work/scratch inside, same as Anvil.
apptainer exec --nv --writable-tmpfs \
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
echo "delta_run_model.sbatch finished rc=$RC at $(date -u)"
echo "  results in: $PROJECT_ROOT/benchmark/results/$LABEL"
echo "========================================================"
exit $RC
