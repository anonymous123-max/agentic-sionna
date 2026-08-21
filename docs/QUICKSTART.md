# Quick start

Full setup, ~10 minutes.

## Prerequisites

- **conda** (Miniconda or Anaconda) — https://docs.conda.io/miniconda/
- **Python 3.11+** (managed via conda)
- **git**
- **Optional**: NVIDIA GPU + CUDA 12.3+ for GPU ray tracing (CPU falls
  back automatically)
- **Optional**: LLM API key (Anthropic, OpenAI-compat relay, or local
  model) — only needed for the chat panel

## 1. Clone + install

```bash
git clone https://github.com/Pervasive-Intelligence-Lab/sionna-skill.git
cd sionna-skill
bash install.sh
```

`install.sh`:
- Creates conda env `sionna` (Python 3.11)
- Installs Python deps (flask, numpy, scipy, shapely, trimesh,
  sionna>=2.0, chromadb, sentence-transformers, ...)
- Copies `.env.example` → `.env`
- Prompts about the optional 3D-FUTURE dataset

## 2. Configure your API key

```bash
$EDITOR .env
```

Fill in the three `DASHBOARD_CHAT_*` variables. See
[DASHBOARD.md](DASHBOARD.md) for provider-specific setup.

## 3. Launch the dashboard

```bash
conda activate sionna
PYTHONPATH=. python web/dashboard_app.py --port 8080
```

Open http://localhost:8080. On a remote server, use SSH port
forwarding or your VPN.

## 4. Verify it works

In the chat panel, try:

```
Build a 6x5 meter room with a desk and two chairs. Put the AP at
(3, 2.5, 2.8) with 20 dBm at 5 GHz, use a 2x2 iso antenna, then
compute coverage.
```

Expected: room appears in the 3D viewport, furniture drops in,
diamond marker (AP) moves, coverage heatmap renders. Total ~10-30 s.

## 5. (Optional) Real furniture meshes

```bash
# Download from https://tianchi.aliyun.com/dataset/98063 (~19 GB)
# Extract to /path/to/3D-FUTURE-model/
# Then in .env:
FUTURE_DATASET_PATH=/absolute/path/to/3D-FUTURE-model
```

Restart the dashboard. The catalog should now show ~14K furniture
models across ~50 categories (was 12 box placeholders).

## 6. (Optional) Run the benchmark

See [BENCHMARK.md](BENCHMARK.md).
