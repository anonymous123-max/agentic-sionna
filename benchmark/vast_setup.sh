#!/bin/bash
# vast_setup.sh — one-shot bootstrap for a fresh vast.ai instance.
#
# Run this on a vast.ai container with NVIDIA drivers + CUDA already
# installed (most "PyTorch" or "CUDA-devel" base images qualify).
# Designed to be idempotent: re-running is safe.
#
# Usage on a freshly rented box:
#   git clone https://github.com/Pervasive-Intelligence-Lab/sionna-skill.git
#   cd sionna-skill && git checkout rf-sim-agent-v2
#   bash benchmark/vast_setup.sh
#   bash benchmark/queue_local_llms.sh           # auto-detects high-VRAM tier
#
# After completion the queue script can be launched directly via nohup:
#   nohup bash benchmark/queue_local_llms.sh > /tmp/queue.log 2>&1 & disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[setup] PROJECT_ROOT=$PROJECT_ROOT"
echo "[setup] using python: $(python3 --version)"
echo

# ─────────────────────────────────────────────────────────────────────
# 1. Python deps
# ─────────────────────────────────────────────────────────────────────
echo "[setup] installing pip deps (vllm, fastapi, uvicorn, httpx, ...)"
# vllm pulls fastapi/uvicorn/httpx as transitive deps — no need to list.
# Pin vllm if you've validated a specific version; otherwise latest.
pip install --quiet --upgrade pip
pip install --quiet \
  "vllm>=0.7.0" \
  "torch" \
  "scipy" \
  "numpy<2.0" \
  "matplotlib"

# ─────────────────────────────────────────────────────────────────────
# 2. OpenClaude (the OpenAI-compatible Claude Code fork)
# ─────────────────────────────────────────────────────────────────────
if ! command -v openclaude >/dev/null 2>&1; then
  echo "[setup] installing openclaude (Node 22 + npm i -g openclaude)"
  if ! command -v node >/dev/null 2>&1 || \
     [[ "$(node --version 2>/dev/null | cut -dv -f2 | cut -d. -f1)" -lt 20 ]]; then
    echo "[setup]   bootstrapping fnm + Node 22"
    curl -fsSL https://fnm.vercel.app/install | bash -s -- --skip-shell
    export PATH="$HOME/.local/share/fnm:$PATH"
    eval "$(fnm env --use-on-cd)"
    fnm install 22 && fnm use 22
  fi
  npm install -g openclaude
fi
echo "[setup] openclaude: $(command -v openclaude)"

# ─────────────────────────────────────────────────────────────────────
# 3. Hugging Face cache directory
# ─────────────────────────────────────────────────────────────────────
HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"
echo "[setup] (set HF_TOKEN=... in env if any planned model is gated)"

# ─────────────────────────────────────────────────────────────────────
# 4. Quick environment summary
# ─────────────────────────────────────────────────────────────────────
echo
echo "[setup] ── ENVIRONMENT ──"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "  python:    $(command -v python3)"
echo "  vllm:      $(command -v vllm || echo NOT-FOUND)"
echo "  openclaude:$(command -v openclaude || echo NOT-FOUND)"
echo
echo "[setup] done. Launch the queue with:"
echo "  cd $PROJECT_ROOT"
echo "  nohup bash benchmark/queue_local_llms.sh > benchmark/results/_queue_nohup.log 2>&1 & disown"
echo
echo "[setup] the queue auto-detects VRAM tier from nvidia-smi and picks"
echo "        max-num-seqs/CUDA-graphs/worker-count accordingly."
