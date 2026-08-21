#!/bin/bash
#SBATCH --job-name=build_skill_baseline_v2
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/build_skill_baseline_v2-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/build_skill_baseline_v2-%j.err
#
# Sibling of build_image_via_sbatch.sh — builds the v2 baseline image
# (Singularity.baseline_v2: vllm 0.10.0 + flash-attn 2.7.4.post1) on a
# compute node.
#
# Walltime: 3h reserved, expected ~25-45 min (no source compile if flash-attn
# wheel is found; ~60-90 min if it falls back to source build).
#
# Output: /anvil/scratch/$USER/skill_baseline_v2.sif (does NOT overwrite
# the existing skill_baseline.sif used by current bench jobs).
set -euo pipefail

cd /anvil/scratch/$USER/skill

export SINGULARITY_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
export APPTAINER_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
# TMPDIR on compute-node-local /tmp (fast NVMe) — see build_baseline_via_sbatch.sh.
export SINGULARITY_TMPDIR=/tmp/$USER-singularity
export APPTAINER_TMPDIR=/tmp/$USER-singularity
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" /anvil/scratch/$USER/skill/logs

# Clean any stale tmp from prior failed/running builds (compute-local /tmp
# is per-node-per-job so usually empty on a fresh allocation, but be safe).
rm -rf "$SINGULARITY_TMPDIR"/build-temp-* "$SINGULARITY_TMPDIR"/bundle-temp-* 2>/dev/null || true

LOG=/anvil/scratch/$USER/skill/logs/build_baseline_v2.log
{
    echo ""
    echo "================================================================"
    echo "→ baseline_v2 build start: $(date -u)"
    echo "  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
    echo "  cpus=$SLURM_CPUS_PER_TASK  mem=128G  walltime=3h"
    echo "  recipe: benchmark/anvil/Singularity.baseline_v2"
    echo "  target: /anvil/scratch/$USER/skill_baseline_v2.sif"
    echo "================================================================"
} >> "$LOG"

module load singularity 2>/dev/null || true

singularity build /anvil/scratch/$USER/skill_baseline_v2.sif \
    benchmark/anvil/Singularity.baseline_v2 >> "$LOG" 2>&1
RC=$?
echo "→ baseline_v2 build end: $(date -u) rc=$RC" >> "$LOG"
ls -la /anvil/scratch/$USER/skill_baseline_v2.sif >> "$LOG" 2>&1 || true
exit $RC
