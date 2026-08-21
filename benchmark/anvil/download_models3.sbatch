#!/bin/bash
#SBATCH --job-name=hf_dl_top6v3
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
        hf download "$MODEL" \
        --include "*.safetensors" "*.json" "*.txt" "*.model" "tokenizer*" 2>&1 | tail -3
    DIR=$(echo "$MODEL" | sed 's|/|--|')
    SIZE_GB=$(du -sb "$HF_HOME/hub/models--$DIR/" 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    SIZE_GB=${SIZE_GB:-0}
    echo ">>> $MODEL : ${SIZE_GB} GB at $HF_HOME/hub/models--$DIR/"
    if (( SIZE_GB < 5 )); then
        echo ">>> $MODEL FAILED: too small"
        exit 1
    fi
done
echo "all downloads verified at $(date -u)"
