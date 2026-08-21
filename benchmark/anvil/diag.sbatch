#!/bin/bash
#SBATCH --job-name=skill-diag
#SBATCH --account=cis260614-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/anvil/scratch/%u/skill/logs/%x-%j.out
#SBATCH --error=/anvil/scratch/%u/skill/logs/%x-%j.err

set -uo pipefail
ENV=/anvil/scratch/$USER/skill/.conda/vllm

echo "=== node info ==="
hostname
nvidia-smi 2>&1 | head -15
echo ""
echo "=== driver / CUDA ==="
nvidia-smi --query-gpu=driver_version,name --format=csv 2>&1
which nvcc 2>&1; nvcc --version 2>&1 | head -3
ls /usr/local/cuda* 2>&1 | head -3
echo ""
echo "=== conda env activate ==="
source /apps/anvil/external/apps/conda/2024.02/etc/profile.d/conda.sh
conda activate $ENV
which python
python --version
echo ""
echo "=== try torch first (simpler than vllm) ==="
timeout 60 python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
" 2>&1
echo "torch-import-rc=$?"
echo ""
echo "=== then try vllm ==="
timeout 120 python -c "
import vllm
print('vllm', vllm.__version__)
" 2>&1
echo "vllm-import-rc=$?"
echo ""
echo "=== done ==="
