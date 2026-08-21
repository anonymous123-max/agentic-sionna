#!/bin/bash
#SBATCH --job-name=hf_dl_v2
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
# baseline.sif sets HF_HUB_ENABLE_HF_TRANSFER=1 in its env but doesn't install
# the hf_transfer package — pop it before any HF call to avoid ValueError.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download

models = [
    "microsoft/Phi-4-mini-instruct",
    "ibm-granite/granite-4.0-tiny-preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
]
results = {}
for m in models:
    print(f"\n========== [{m}] STARTING ==========", flush=True)
    try:
        p = snapshot_download(
            repo_id=m, cache_dir="/work/hf_cache",
            max_workers=4,
            allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.md",
                "*.model", "tokenizer*", "*.py", "consolidated*",
            ],
            force_download=False,
            resume_download=True,
        )
        sf = sorted(glob.glob(f"{p}/*.safetensors"))
        total_bytes = sum(os.path.getsize(f) for f in sf if os.path.exists(f))
        cfg = os.path.exists(f"{p}/config.json")
        print(f"[{m}] OK -> {p}", flush=True)
        print(f"  safetensors: {len(sf)} files, total {total_bytes/1e9:.1f}GB", flush=True)
        print(f"  config.json exists: {cfg}", flush=True)
        if cfg and total_bytes > 100e6:  # > 100MB total weights
            results[m] = "OK"
        else:
            results[m] = f"INCOMPLETE (cfg={cfg}, bytes={total_bytes/1e9:.1f}GB)"
    except Exception as e:
        print(f"[{m}] FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[m] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

print("\n========== SUMMARY ==========")
for m, status in results.items():
    print(f"  {m}: {status}")
ok = sum(1 for s in results.values() if s == "OK")
print(f"\n{ok}/{len(results)} models successfully downloaded")
PY
echo "exit rc=$? at $(date -u)"
