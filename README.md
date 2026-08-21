# AutoNetSim

**Turn "build a 6×5 office and compute coverage at 5 GHz" into a runnable
NVIDIA Sionna simulation — with a live 3D dashboard.**

AutoNetSim is a Claude-Code-native agent that generates, executes, and
verifies wireless network simulations from natural-language requests.
It ships as:

- **`.claude/skills/rf-simulator/`** — the procedural skill (routing,
  templates, references, memory) that turns intent into Sionna 2.0 code
- **`web/`** — a Flask + Three.js dashboard for interactive scene
  building, coverage visualisation, and chat-driven control
- **`benchmark/`** — 100+ tasks with a 3-layer verifier (artifact,
  executable, oracle) that scored the paper's results

Paper: **[AutoNetSim: Intent-Driven Wireless Network Experimentation
with Self-Evolving Agents](main.pdf)**  
Runnan Si\*, John Song\*, Haijian Sun, Zhenlin An — University of Georgia  
**IEEE ICNP 2026** · [Project page](https://pervasive-intelligence-lab.github.io/agentic-sionna/)

## Abstract

Wireless network simulation and optimization are difficult because they
require solving a complex cross-layer configuration problem while
interacting with complex 3D radio environments. For decades, engineers
have relied on channel models, channel simulators, and extensive
standards to guide this process, but using them correctly still requires
substantial domain expertise and engineering effort. Recent
language-model agents can automate parts of wireless reasoning and code
generation, yet they are not trained for the cross-layer setting in
which an agent must construct a 3D radio environment, bind it to radio
and network assumptions, and run simulations inside it. We present
**AutoNetSim**, a self-evolving language-agent system for intent-driven
wireless network simulation. Given a natural-language request, it
constructs an executable *radio environment* that combines a 3D scene
with a cross-layer wireless-system configuration and optimization. The
system uses a multi-agent architecture with specialized scene-building,
simulation, reflection, planning, and skill-learning agents. It learns
from previous tutorials, worked examples, prior simulation results, and
its own failed trajectories, then updates a procedural skill and
knowledge base through verifier-driven optimization. We evaluate the
system on benchmark suites covering 3D radio environment generation,
wireless simulation, and token efficiency across multiple indoor and
outdoor scenarios. On 3D radio environment generation, AutoNetSim
reaches 72.5% held-out pass rate, whereas both baselines fail to solve
any test task. Across ten ray-tracing, physical-layer, and
network/system-level simulation families, it averages 95.5% pass rate,
while both baselines stay below 70.0%.

---

## Quick start (5 min)

```bash
# 1. Clone
git clone https://github.com/Pervasive-Intelligence-Lab/agentic-sionna.git
cd agentic-sionna

# 2. Install (creates conda env `sionna`, installs deps)
bash install.sh

# 3. Configure your LLM key
cp .env.example .env
$EDITOR .env               # fill in DASHBOARD_CHAT_API_KEY

# 4. Launch the dashboard
conda activate sionna
PYTHONPATH=. python web/dashboard_app.py --port 8080

# 5. Open http://localhost:8080
```

Ask the chat panel: *"Build a 6×5 room with a desk and two chairs, put
the AP at (3, 2.5, 2.8), then compute coverage at 5 GHz."*

You should see the room render, furniture drop in, AP marker move,
and a coverage heatmap appear — all driven by one prompt.

---

## Bring your own LLM

No API key is bundled. The dashboard reads its provider config from
environment variables (or from `.env`, which is gitignored):

| Provider | `DASHBOARD_CHAT_BASE_URL` | `DASHBOARD_CHAT_MODEL` | Get a key |
|---|---|---|---|
| **Anthropic direct** | `https://api.anthropic.com/v1` | `claude-sonnet-4-6` | https://console.anthropic.com |
| OpenAI-compat relay (Kimi, PackyCode, exchangetoken, ZetaAPI, ...) | `https://<gateway>/v1` | any model the relay offers | relay dashboard |
| Local model (Ollama, LM Studio, vLLM) | `http://localhost:11434/v1` | e.g. `llama3.1` | no key needed |

If either the URL or the key is missing at startup, the chat panel
returns *"Chat is disabled — please set …"* rather than silently
falling back to any shared account.

See [docs/DASHBOARD.md](docs/DASHBOARD.md) for provider-specific setup
and the full list of chat actions.

---

## What the chat can do (17 actions)

| Category | Examples |
|---|---|
| **Scene** | `Build a 6x5 room`, `Change ceiling to 3.5 m`, `Change walls to drywall` |
| **Furniture** | `Add a sofa and two chairs`, `Move the desk to (3, 2)`, `Rotate the chair 90°`, `Remove the bookcase` |
| **AP / antenna** | `Move the AP to (5, 4, 2.8)`, `Set TX power to 15 dBm`, `Change frequency to 28 GHz`, `Use a 4x4 tr38901 antenna`, `Point the AP north with 15° downtilt` |
| **Simulation** | `Compute the coverage map` |
| **Scenes** | `Load Room_5x4_abc`, `Delete <scene>`, `Switch to outdoor`, `Fetch downtown Athens from OSM` |

Chat and direct 3D interaction (drag / rotate / delete) share the same
scene state — you can seed a room by chat and then hand-tune it in the
viewport.

---

## Optional: real furniture meshes (3D-FUTURE)

Without the dataset, furniture renders as box primitives — the RF
simulation is still correct (ray tracing uses AABBs). For photorealistic
meshes, install the ~19 GB **3D-FUTURE** dataset:

1. Register at [Tianchi](https://tianchi.aliyun.com/dataset/98063)
2. Download `3D-FUTURE-model.zip`, extract somewhere
3. In `.env`: `FUTURE_DATASET_PATH=/absolute/path/to/3D-FUTURE-model`
4. Restart the dashboard

The dataset directory must contain `model_info.json` and per-model
`<uuid>/raw_model.obj` files (this is the layout the Tianchi ZIP uses).

---

## Repo layout

```
agentic-sionna/
├── .claude/skills/rf-simulator/    # The skill (SKILL.md + references + templates + agents)
├── web/                            # Flask dashboard (backend + frontend)
│   ├── dashboard_app.py            #   Flask app entry
│   ├── routes/                     #   API endpoints (chat, scenes, coverage, catalog, ...)
│   ├── static/                     #   Three.js viewport + CSS
│   └── templates/dashboard.html
├── src/                            # Runtime libraries (models, optimizer, exporters, wireless)
├── benchmark/                      # Benchmark suite (verifier + tasks + oracles + metrics)
│   ├── verifier.py                 #   3-layer verifier
│   ├── tasks/                      #   Task specs (100+ scene-gen, RT, PHY, opt, system tasks)
│   ├── oracles/                    #   Reference answers per task
│   ├── compute_metrics.py          #   Extract continuous quality metrics from trial output
│   ├── paper_appendix_table.md     #   Failure taxonomy + wall-clock tables (paper appendix)
│   └── metrics_per_trial.csv       #   All 2094 trials' quality numbers
├── docs/
│   ├── QUICKSTART.md
│   ├── DASHBOARD.md                #   Chat actions + provider setup
│   └── BENCHMARK.md                #   How to reproduce paper numbers
├── main.pdf, architecture.pdf      # The paper + system diagram
├── install.sh, pyproject.toml
└── .env.example                    # Copy to .env with your keys
```

---

## Reproducing the paper

The `benchmark/` folder is self-contained.

```bash
# Run one trial
PYTHONPATH=. python benchmark/run_benchmark.py \
    --skill rf-simulator --task N1_cov_box_one_screen --k 1

# Compute continuous quality metrics on existing trial output
python benchmark/compute_metrics.py
python benchmark/aggregate_metrics.py
python benchmark/paper_appendix_table.py

# Read the results
less benchmark/paper_appendix_table.md
```

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for the full pipeline.

---

## Citation

If you use AutoNetSim in your work, please cite:

```bibtex
@inproceedings{si2026autonetsim,
  title     = {AutoNetSim: Intent-Driven Wireless Network Experimentation with Self-Evolving Agents},
  author    = {Si, Runnan and Song, John and Sun, Haijian and An, Zhenlin},
  booktitle = {Proceedings of the IEEE International Conference on Network Protocols (ICNP)},
  year      = {2026}
}
```

---

## Contributing

We welcome bug reports, tasks, and skill improvements. Please open an
issue before large PRs. Do **not** commit API keys or the 3D-FUTURE
dataset — both are excluded via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).

3D-FUTURE dataset is separately licensed by Alibaba; obtain it from
[Tianchi](https://tianchi.aliyun.com/dataset/98063) under their terms.
Sionna is licensed by NVIDIA under Apache-2.0.
