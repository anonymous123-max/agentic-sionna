#!/bin/bash
#SBATCH --job-name=build_skill_baseline
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/build_skill_baseline-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/build_skill_baseline-%j.err
#
# build_baseline_via_sbatch.sh — submit baseline (no DFlash) image build.
# Mainline vLLM from PyPI, no source compile. Build typically ~10-15 min.
#
# Usage:  sbatch benchmark/anvil/build_baseline_via_sbatch.sh
set -euo pipefail

cd /anvil/scratch/$USER/skill

# OCI cache stays on scratch (shared, persistent, large layers reused across
# builds). But TMPDIR (where extract+squash write GB of intermediate files)
# goes to compute-node-local /tmp — Anvil GPU nodes have NVMe SSD there
# that hits 1-2 GB/sec vs ~100 KB/sec on shared scratch. The previous
# baseline build squashfs growth was ~120 KB/sec on scratch, would have
# taken 40+ hours.
export SINGULARITY_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
export APPTAINER_CACHEDIR=/anvil/scratch/$USER/.singularity_cache
export SINGULARITY_TMPDIR=/tmp/$USER-singularity
export APPTAINER_TMPDIR=/tmp/$USER-singularity
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" /anvil/scratch/$USER/skill/logs

LOG=/anvil/scratch/$USER/skill/logs/build_baseline.log
{
    echo ""
    echo "================================================================"
    echo "→ baseline build start: $(date -u) (mainline vLLM, no compile)"
    echo "  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
    echo "  cpus=$SLURM_CPUS_PER_TASK  mem=64G"
    echo "================================================================"
} >> "$LOG"

module load singularity 2>/dev/null || true

singularity build /anvil/scratch/$USER/skill_baseline.sif benchmark/anvil/Singularity.baseline >> "$LOG" 2>&1
RC=$?
echo "→ baseline build end: $(date -u) rc=$RC" >> "$LOG"
ls -la /anvil/scratch/$USER/skill_baseline.sif >> "$LOG" 2>&1 || true
exit $RC
