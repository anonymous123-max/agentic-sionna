#!/bin/bash
#SBATCH --job-name=skill-diag2
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/%x-%j.err

set -uo pipefail
ENV=/anvil/scratch/$USER/skill/.conda/vllm
source /apps/anvil/external/apps/conda/2024.02/etc/profile.d/conda.sh
conda activate $ENV

echo "=== try vllm import with 10 min timeout, line-buffered ==="
timeout 600 stdbuf -oL python -c "
import sys; sys.stderr = sys.stdout
print('1. importing vllm', flush=True)
import vllm
print('2. vllm version:', vllm.__version__, flush=True)
print('3. importing entry point', flush=True)
from vllm.entrypoints.openai import api_server
print('4. ok', flush=True)
" 2>&1
echo "import-rc=$?"
echo ""
echo "=== try plain vllm serve --help (lightest invocation) ==="
timeout 60 stdbuf -oL vllm serve --help 2>&1 | head -25
echo "serve-help-rc=$?"
