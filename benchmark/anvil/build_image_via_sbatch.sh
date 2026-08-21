#!/bin/bash
#SBATCH --job-name=build_skill_dflash
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/build_skill_dflash-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/build_skill_dflash-%j.err
#
# build_image_via_sbatch.sh — submit the Singularity build as a SLURM job
# on a dedicated compute node, instead of running it on the login node.
#
# Why: login-node builds repeatedly OOM-killed at peak compile pressure
# because RAM is shared with ~22 other interactive users. A compute node
# gets dedicated 16 cores / 128 GB; build finishes in ~25-40 min reliably.
#
# Usage:  sbatch benchmark/anvil/build_image_via_sbatch.sh
#         (submit from $SCRATCH/skill/, log/sif paths assume that)
#
# The sbatch directives request 1 GPU because cis260614-gpu is a GPU-only
# account and can't submit to CPU partitions; the GPU is unused during the
# build (apptainer build is CPU+RAM only). Cost: ~4 SU for one job-hour.
set -euo pipefail

cd /anvil/scratch/$USER/skill

export SINGULARITY_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
export APPTAINER_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
# TMPDIR on compute-node-local /tmp (NVMe ~1-2 GB/sec) instead of scratch
# (~100 KB/sec) — squash phase was untenable on scratch.
export SINGULARITY_TMPDIR=/tmp/$USER-singularity
export APPTAINER_TMPDIR=/tmp/$USER-singularity
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" /anvil/scratch/$USER/skill/logs

LOG=/anvil/scratch/$USER/skill/logs/build_dflash.log
{
    echo ""
    echo "================================================================"
    echo "→ sbatch build start: $(date -u) (compute-node, MAX_JOBS=4)"
    echo "  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
    echo "  cpus=$SLURM_CPUS_PER_TASK  mem=128G  walltime=${SLURM_TIMELIMIT:-${SBATCH_TIMELIMIT:-unknown}}"
    echo "================================================================"
} >> "$LOG"

module load singularity 2>/dev/null || true

singularity build /anvil/scratch/$USER/skill_dflash.sif benchmark/anvil/Singularity.dflash >> "$LOG" 2>&1
RC=$?
echo "→ sbatch build end: $(date -u) rc=$RC" >> "$LOG"
ls -la /anvil/scratch/$USER/skill_dflash.sif >> "$LOG" 2>&1 || true
exit $RC
