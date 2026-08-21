#!/bin/bash
#SBATCH --job-name=build_skill_dflash_gemma
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/build_skill_dflash_gemma-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/build_skill_dflash_gemma-%j.err
#
# Sibling of build_image_via_sbatch.sh — builds the Gemma DFlash image
# (Singularity.dflash_gemma, vLLM PR #41703) on a compute node.
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

LOG=/anvil/scratch/$USER/skill/logs/build_dflash_gemma.log
{
    echo ""
    echo "================================================================"
    echo "→ dflash_gemma build start: $(date -u)"
    echo "  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
    echo "  cpus=$SLURM_CPUS_PER_TASK  mem=128G"
    echo "================================================================"
} >> "$LOG"

module load singularity 2>/dev/null || true

singularity build /anvil/scratch/$USER/skill_dflash_gemma.sif benchmark/anvil/Singularity.dflash_gemma >> "$LOG" 2>&1
RC=$?
echo "→ dflash_gemma build end: $(date -u) rc=$RC" >> "$LOG"
ls -la /anvil/scratch/$USER/skill_dflash_gemma.sif >> "$LOG" 2>&1 || true
exit $RC
