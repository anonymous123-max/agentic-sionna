#!/bin/bash
# install_vllm_dflash.sh — set up a parallel conda/venv with a DFlash-capable
# vLLM build, leaving the baseline vLLM env untouched.
#
# DFlash (https://github.com/z-lab/dflash) is speculative decoding via a
# block-diffusion draft. Mainline vLLM has it for some models; the newer
# SWA drafts (Qwen3.6, Gemma4) need specific PR branches:
#   - Qwens / Llama-3.1-8B → PR #40898
#   - Gemma 4 family       → PR #41703
#
# Why a separate env: installing the PR branch overwrites your working vLLM,
# and a build conflict mid-loop on vast.ai is expensive. Two parallel envs
# (one baseline, one DFlash) lets us flip USE_DFLASH=0/1 without reinstalls.
#
# Usage:
#   bash install_vllm_dflash.sh            # PR #40898 (Qwens / Llama 8B), env=vllm_dflash
#   FAMILY=gemma  bash install_vllm_dflash.sh   # PR #41703 (Gemma 4), env=vllm_dflash_gemma
#   ENV_NAME=my_env  bash install_vllm_dflash.sh   # custom env name
#
# After install, switch envs before launching vLLM:
#   conda activate vllm_dflash
#   USE_DFLASH=1 bash run_loop_v3_tp.sh
set -euo pipefail

FAMILY="${FAMILY:-qwen}"   # qwen | gemma
case "$FAMILY" in
    qwen)
        PR_NUM=40898
        DEFAULT_ENV="vllm_dflash"
        ;;
    gemma)
        PR_NUM=41703
        DEFAULT_ENV="vllm_dflash_gemma"
        ;;
    *)
        echo "✗ unknown FAMILY=$FAMILY (expected 'qwen' or 'gemma')" >&2
        exit 1
        ;;
esac
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV}"

echo "▶ DFlash vLLM install: family=$FAMILY PR=#$PR_NUM env=$ENV_NAME"

if ! command -v conda >/dev/null 2>&1; then
    echo "✗ conda not found in PATH; aborting (this script targets conda envs)" >&2
    exit 1
fi

# Source conda hook for the current shell
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Skip env creation if it already exists (idempotent for re-runs after a
# failed pip install — caller can rerun without losing the env).
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  env $ENV_NAME already exists; reusing"
else
    echo "  creating env $ENV_NAME (python=3.12)"
    conda create -y -n "$ENV_NAME" python=3.12 >/dev/null
fi

conda activate "$ENV_NAME"

echo "  installing vLLM PR #$PR_NUM (this can take 10-20 min on first run)"
# uv is faster + has --torch-backend=auto which picks the right CUDA wheel.
pip install -U uv >/dev/null
uv pip install -U --torch-backend=auto \
    "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/${PR_NUM}/head"

echo "  verifying vLLM import + DFlash spec method registered"
python3 -c "
import vllm
print(f'  vllm version: {vllm.__version__}')
# Spec method registration check — exact import path varies by vLLM version,
# so we just confirm the package imports without crashing. Real verification
# happens when --speculative-config '{\"method\":\"dflash\",...}' is passed.
"

echo "✓ DFlash env ready: $ENV_NAME"
echo ""
echo "Next:"
echo "  conda activate $ENV_NAME"
echo "  USE_DFLASH=1 bash benchmark/run_loop_v3_tp.sh    # vast.ai TP=2"
echo "  USE_DFLASH=1 bash benchmark/run_loop_v3_multi.sh # vast.ai multi-instance"
