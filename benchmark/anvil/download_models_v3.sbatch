#!/bin/bash
#SBATCH --job-name=hf_dl_v3
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
# baseline.sif sets HF_HUB_ENABLE_HF_TRANSFER=1 but doesn't install hf_transfer
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download

# Family-diverse additions for paper-grade coverage:
#   Qwen2.5-32B-Instruct       — fills 24-32B Qwen-family gap (tool-trained dense)
#   Qwen2.5-72B-Instruct       — 70B-class replicate for Qwen family (vs Llama-70B)
#   Qwen/QwQ-32B-Preview        — reasoning model (vs DeepSeek-R1-Distill comparison)
#   google/gemma-3-27b-it       — Gemma family currently missing entirely
models = [
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B-Preview",
    "google/gemma-3-27b-it",
]
results = {}
for m in models:
    print(f"\n========== [{m}] STARTING ==========", flush=True)
    try:
        # CRITICAL: cache_dir must include /hub for newer HF Hub layout
        p = snapshot_download(
            repo_id=m, cache_dir="/work/hf_cache/hub",
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
        if cfg and total_bytes > 1e9:
            results[m] = "OK"
        else:
            results[m] = f"INCOMPLETE (cfg={cfg}, bytes={total_bytes/1e9:.1f}GB)"
    except Exception as e:
        print(f"[{m}] FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[m] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

# Some downloads put files at cache_dir/hub/hub/models--*/, which breaks vLLM resolution.
# Normalize: if cache_dir/hub/hub exists, move contents up to cache_dir/hub.
import shutil
nested = "/work/hf_cache/hub/hub"
if os.path.isdir(nested):
    for d in os.listdir(nested):
        src = os.path.join(nested, d)
        dst = os.path.join("/work/hf_cache/hub", d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    os.rmdir(nested)
    print(f"\nMoved nested hub/ contents up to /work/hf_cache/hub/")

print("\n========== SUMMARY ==========")
for m, status in results.items():
    print(f"  {m}: {status}")
ok = sum(1 for s in results.values() if s == "OK")
print(f"\n{ok}/{len(results)} models successfully downloaded")
PY
echo "exit rc=$? at $(date -u)"
