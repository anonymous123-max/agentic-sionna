#!/usr/bin/env bash
# AutoNetSim installer
# Creates a conda env `sionna`, installs Python deps, prompts for API key setup.
set -euo pipefail

ENV_NAME="sionna"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "──────────────────────────────────────────────────────"
echo "  AutoNetSim installer"
echo "──────────────────────────────────────────────────────"

# 1. Check conda
if ! command -v conda >/dev/null 2>&1; then
    echo "✗ conda not found. Install Miniconda first: https://docs.conda.io/miniconda/"
    exit 1
fi

# 2. Create env if missing
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "✓ conda env '$ENV_NAME' already exists"
else
    echo "→ Creating conda env '$ENV_NAME' (python 3.11)..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

# 3. Activate + install deps
echo "→ Installing Python dependencies into '$ENV_NAME'..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Core deps (verifier + web + models)
pip install --upgrade pip
pip install \
    flask \
    numpy \
    scipy \
    shapely \
    trimesh \
    mapbox-earcut \
    matplotlib \
    pydantic \
    "sionna>=2.0" \
    "chromadb>=0.5,<1.0" \
    sentence-transformers

# 4. .env setup
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo ""
    echo "✓ Created .env from template. Edit it to add your API key:"
    echo "    $EDITOR $REPO_ROOT/.env"
    echo "  (see docs/DASHBOARD.md for provider-specific setup)"
else
    echo "✓ .env already exists — not overwriting"
fi

# 5. 3D-FUTURE dataset check
if [ -n "${FUTURE_DATASET_PATH:-}" ] && [ -f "$FUTURE_DATASET_PATH/model_info.json" ]; then
    echo "✓ 3D-FUTURE dataset found at $FUTURE_DATASET_PATH"
else
    echo ""
    echo "ℹ 3D-FUTURE dataset not configured (optional)."
    echo "  Without it, furniture renders as boxes — RF simulation still works."
    echo "  To install: register at https://tianchi.aliyun.com/dataset/98063"
    echo "  then set FUTURE_DATASET_PATH in .env"
fi

echo ""
echo "──────────────────────────────────────────────────────"
echo "  ✓ Install complete"
echo "──────────────────────────────────────────────────────"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API key"
echo "  2. conda activate $ENV_NAME"
echo "  3. PYTHONPATH=. python web/dashboard_app.py --port 8080"
echo "  4. Open http://localhost:8080"
