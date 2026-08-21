#!/bin/bash
#SBATCH --job-name=hf_dl_top6
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
singularity exec \
    --bind "$HF_HOME:/work/hf_cache" \
    --env "HF_HOME=/work/hf_cache" \
    --env "HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment}" \
    --env "PYTHONNOUSERSITE=1" \
    "$SCRATCH/skill_baseline.sif" \
    python3 - <<'PY'
import os, sys
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download
models = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "microsoft/Phi-4",
]
for m in models:
    print(f"[{m}] starting", flush=True)
    p = snapshot_download(repo_id=m, cache_dir="/work/hf_cache",
                          max_workers=8,
                          allow_patterns=["*.safetensors","*.json","*.txt","*.model","tokenizer*"],
                          force_download=False)  # only fetch missing
    # Verify safetensors actually present
    import glob
    sf = glob.glob(f"{p}/*.safetensors")
    print(f"[{m}] done -> {p}  ({len(sf)} safetensors files)", flush=True)
    if len(sf) == 0:
        print(f"[{m}] ERROR: no safetensors found", flush=True)
        sys.exit(1)
PY
echo "exit rc=$? at $(date -u)"
