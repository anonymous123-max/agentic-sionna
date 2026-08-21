#!/bin/bash
#SBATCH --job-name=hf_dl_gemma_llama
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
set -uo pipefail
SCRATCH=/anvil/scratch/$USER
HF_HOME="$SCRATCH/hf_cache"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — start $(date -u)"
module load singularity 2>/dev/null || true
singularity exec \
    --bind "$HF_HOME:/work/hf_cache" \
    --env "HF_HOME=/work/hf_cache" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "PYTHONNOUSERSITE=1" \
    "$SCRATCH/skill_baseline.sif" \
    python3 - <<'PY'
import os, sys, glob, traceback
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download

# User confirmed access to both — retry with same token after gate acceptance.
models = [
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]
for m in models:
    print(f"\n========== [{m}] STARTING ==========", flush=True)
    try:
        p = snapshot_download(
            repo_id=m, cache_dir="/work/hf_cache/hub",
            max_workers=4,
            allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.md",
                "*.model", "tokenizer*", "*.py", "consolidated*",
            ],
            force_download=False, resume_download=True,
        )
        sf = sorted(glob.glob(f"{p}/*.safetensors"))
        total_bytes = sum(os.path.getsize(f) for f in sf if os.path.exists(f))
        cfg = os.path.exists(f"{p}/config.json")
        print(f"[{m}] OK -> {p}  ({len(sf)} safetensors, {total_bytes/1e9:.1f}GB, cfg={cfg})", flush=True)
    except Exception as e:
        print(f"[{m}] FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

# Normalize nested hub/hub/ if present
import shutil
nested = "/work/hf_cache/hub/hub"
if os.path.isdir(nested):
    for d in os.listdir(nested):
        src = os.path.join(nested, d); dst = os.path.join("/work/hf_cache/hub", d)
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.move(src, dst)
    os.rmdir(nested)
PY
echo "exit rc=$? at $(date -u)"
