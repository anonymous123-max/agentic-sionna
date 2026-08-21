#!/bin/bash
#SBATCH --job-name=hf_dl_top6v2
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
set -uo pipefail
SCRATCH=/anvil/scratch/$USER
HF_HOME="$SCRATCH/hf_cache"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — start $(date -u)"
module load singularity 2>/dev/null || true
for MODEL in mistralai/Mixtral-8x7B-Instruct-v0.1 deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct microsoft/Phi-4; do
    echo ">>> downloading $MODEL"
    singularity exec \
        --bind "$HF_HOME:/work/hf_cache" \
        --env "HF_HOME=/work/hf_cache" \
        --env "HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment}" \
        --env "HF_HUB_ENABLE_HF_TRANSFER=0" \
        --env "PYTHONNOUSERSITE=1" \
        "$SCRATCH/skill_baseline.sif" \
        huggingface-cli download "$MODEL" \
        --include "*.safetensors" "*.json" "*.txt" "*.model" "tokenizer*" \
        --cache-dir /work/hf_cache 2>&1 | tail -5
    RC=$?
    # Verify
    DIR=$(echo "$MODEL" | sed 's|/|--|')
    SIZE=$(du -sb "$HF_HOME/hub/models--$DIR/" 2>/dev/null | awk '{print $1}')
    SIZE=${SIZE:-0}
    SIZE_GB=$((SIZE/1024/1024/1024))
    NUM_SF=$(find "$HF_HOME/hub/models--$DIR/blobs/" -type f -size +100M 2>/dev/null | wc -l)
    echo ">>> $MODEL: cli_rc=$RC  ${SIZE_GB}GB  blob_count=$NUM_SF"
    if (( SIZE_GB < 5 )); then
        echo ">>> $MODEL FAILED: too small ($SIZE_GB GB)"
        exit 1
    fi
done
echo "all downloads verified at $(date -u)"
