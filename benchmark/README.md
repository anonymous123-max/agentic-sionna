# Sionna Skill Benchmark

Infrastructure for evaluating the `rf-simulator` skill. Not runtime skill
content — the agent running a task only reads `.claude/skills/rf-simulator/`.

## Layout

```
benchmark/
├── README.md                       # This file
├── docs/
│   ├── BENCHMARK_METHODOLOGY.md
│   └── SKILL_UPDATE_PROCESS.md
│
├── run_benchmark.py                # Orchestrator (entry point)
├── trial.py                        # Per-trial worker (called by orchestrator)
├── verifier.py                     # Verification dispatcher (9 check types)
│
├── build_tasks.py                  # Build tasks.json from _sources/*
├── generate_scenes.py              # Build scenes/ fixtures (one-time setup)
├── generate_baseline_skill.py      # Stage-1 of self_gen condition
├── audit_oracles.py                # Methodology check: would oracle pass verifier?
├── audit_leakage.py                # Methodology check: skill leakage
│
├── tasks/
│   ├── tasks.json                  # ← active benchmark (134 tasks)
│   ├── _sources/
│   │   ├── capability_grid.json    # 60 hand-authored capability tasks
│   │   └── tutorial_variants.json  # 99-task corpus (Sionna tutorials + variants)
│   └── _audits/
│       ├── oracle_audit.json
│       └── leakage_audit.json
│
├── scenes/                         # 27 deterministic scene fixtures (tracked)
├── results/                        # Per-run outputs (gitignored)
└── plots/                          # Generated figures (gitignored)
```

## Run it

### Full 134-task benchmark (crash-protected, resumable)
```bash
python benchmark/run_benchmark.py --label run_v1 --workers 4
```

### Paired eval (with_skill / no_skill / self_gen)
```bash
python benchmark/run_benchmark.py --label paired \
    --conditions with_skill no_skill self_gen --workers 6
```

### Targeted rerun (e.g., T1 PHY tasks on opus, 3 trials each)
```bash
python benchmark/run_benchmark.py --label opus_phy \
    --tiers T1 --model opus --k 3 --workers 4
```

### Resume after a crash / SIGTERM / reboot
```bash
python benchmark/run_benchmark.py --label run_v1 --resume --workers 4
```
Already-completed trials are skipped; partial workdirs (killed mid-trial) are
cleaned and re-run.

### Single task via the lower-level worker
```bash
python benchmark/trial.py \
    --task-ids U001 --output-root benchmark/results/smoke \
    --model sonnet --max-turns 15 --timeout 300 --k 1 \
    --condition with_skill
```

### Re-verify existing results with an updated verifier
```bash
python benchmark/verifier.py \
    --task-id U001 \
    --output-dir benchmark/results/run_v1/with_skill/U001/t1
```

### Rebuild tasks.json from sources
```bash
python benchmark/build_tasks.py
# -> benchmark/tasks/tasks.json
```

### Generate the 27 scene fixtures (one-time)
```bash
python benchmark/generate_scenes.py
# -> benchmark/scenes/{easy,medium,hard}/scene_*/scene_state.json
```

### Stage-1 self_gen skill (run once before paired eval)
```bash
python benchmark/generate_baseline_skill.py --model sonnet
# -> benchmark/self_gen_skill/SKILL.md
```

## Multi-model loop on Anvil (Purdue, ACCESS allocation)

For free-via-NSF runs on Purdue's Anvil supercomputer (16× A100 nodes +
21× H100 nodes), the SLURM-based equivalent of the vast.ai loop lives in
`benchmark/anvil/`:

```
benchmark/anvil/
├── README.md                  # operational runbook (start here)
├── Singularity.dflash         # build recipe, vLLM PR #40898 (Qwens + Llama-8B)
├── Singularity.dflash_gemma   # build recipe, vLLM PR #41703 (Gemma 4)
├── anvil_run_model.sbatch     # sbatch template for one-model job
├── run_inside_container.sh    # entrypoint executed inside the .sif
└── submit_loop.sh             # driver: queue all 6 model jobs in parallel
```

Quick start once setup is done (`benchmark/anvil/README.md` walks through
the one-time setup):
```bash
USE_DFLASH=1 bash benchmark/anvil/submit_loop.sh -A <ACCESS_alloc>
```

Differences from the vast.ai path: jobs are bounded at 48h walltime,
container runtime is Singularity (not Docker), and parallelism comes from
queueing independent sbatch jobs rather than a long-lived tmux loop.

## Multi-model loop on vast.ai

For sweeping the 6-model local-LLM matrix on a rented vast.ai instance, the
orchestration sits one layer above `run_benchmark.py`:

```
benchmark/
├── run_loop_v3_tp.sh          # 2× A100 NVLink, TP=2 (one vLLM, two GPUs)
├── run_loop_v3_multi.sh       # 2× A100 multi-instance (one vLLM per GPU)
├── bootstrap_vastai_tp.sh     # invoked by run_loop_v3_tp.sh per model
├── bootstrap_vastai_multi.sh  # invoked by run_loop_v3_multi.sh per model
└── install_vllm_dflash.sh     # one-time setup for DFlash speculative decode
```

Each `run_loop_v3_*.sh` iterates the 6 models and shells out to the matching
`bootstrap_vastai_*.sh` per iteration; the bootstrap downloads weights, starts
vLLM with per-family parser flags, launches the proxy, and runs
`run_benchmark.py`.

### DFlash speculative decoding (optional)

Set `USE_DFLASH=1` to enable [DFlash](https://github.com/z-lab/dflash) drafts.
Five of the six loop models have published drafts (Qwen3.6-27B, Qwen3.6-35B-A3B,
Qwen3-Coder-30B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it); Llama-3.3-70B-AWQ has
no draft and silently falls back to baseline. Default behavior (`USE_DFLASH`
unset) is bit-identical to the pre-DFlash path.

One-time install of a DFlash-capable vLLM into a parallel conda env:

```bash
bash benchmark/install_vllm_dflash.sh                # PR #40898 → env=vllm_dflash       (Qwens/Llama 8B)
FAMILY=gemma bash benchmark/install_vllm_dflash.sh   # PR #41703 → env=vllm_dflash_gemma (Gemma 4)
```

Then activate the env and run the loop with the gate:
```bash
conda activate vllm_dflash
USE_DFLASH=1 bash benchmark/run_loop_v3_tp.sh
```

Tunables: `VAST_DFLASH_SPEC_TOKENS=15` (default 15) controls the number of
speculative tokens per verify step.

## Result schema

Each trial writes `benchmark/results/<run>/<condition>/<task_id>/t<k>/`:
- `result.json`     — task_id, condition, exec_success, wall_sec, verification
- `stdout.txt`      — agent CLI stdout (stream-json)
- `stderr.txt`      — agent CLI stderr
- `prompt.txt`      — the prompt the agent received
- `simulation.py`   — code the agent wrote (if any)
- `simulation_result.json` — agent's output (if any)
- plus any artifacts: `*.npy`, `*.png`, `scene_state.json`, etc.
