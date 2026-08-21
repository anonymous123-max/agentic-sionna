#!/bin/bash
#SBATCH --job-name=hf_dl_2026_05
#SBATCH --partition=shared
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
import os, sys, glob
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
from huggingface_hub import snapshot_download

# Newest tool-call-trained replacements for the 4 zero-pass models.
# - Phi-4-mini-instruct (3.8B, March 2025): Microsoft's tool-trained Phi.
# - Mistral-Small-3.1-24B-Instruct-2503 (March 2025): native function-calling.
# - DeepSeek-R1-Distill-Qwen-32B (Jan 2025): reasoning + tool-use distillation.
# - Granite-4.0-Tiny-Preview-7B (Oct 2025): IBM's latest tool-trained, if available.
models = [
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "ibm-granite/granite-4.0-tiny-preview",
]
failures = []
for m in models:
    print(f"[{m}] starting", flush=True)
    try:
        p = snapshot_download(
            repo_id=m, cache_dir="/work/hf_cache",
            max_workers=8,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"],
            force_download=False,
        )
        sf = glob.glob(f"{p}/*.safetensors")
        print(f"[{m}] done -> {p}  ({len(sf)} safetensors files)", flush=True)
        if len(sf) == 0:
            failures.append((m, "no safetensors"))
    except Exception as e:
        print(f"[{m}] FAIL: {type(e).__name__}: {e}", flush=True)
        failures.append((m, str(e)[:200]))

if failures:
    print("=== FAILURES ===")
    for m, why in failures:
        print(f"  {m}: {why}")
    sys.exit(1)
PY
echo "exit rc=$? at $(date -u)"
