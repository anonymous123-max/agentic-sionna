# Anvil benchmark runbook

Operational guide for running the local-LLM benchmark on Purdue's Anvil
supercomputer. This is the Anvil-side equivalent of the vast.ai workflow
(`run_loop_v3_tp.sh` + `bootstrap_vastai_tp.sh`), adapted for SLURM jobs
and Singularity containers.

## For an agent reading this on the Anvil server

If you are a Claude (or other agent) running on the Anvil login node and
the human user just told you "follow `benchmark/anvil/README.md` to run
the benchmark," start here:

1. Read the **REQUIRED-FROM-USER inputs** section directly below. If any
   value is unknown, **stop and ask the human user**. Do not invent.
2. Walk through **One-time setup** (Steps 1-5) in order. Each step has a
   concrete bash block; copy it verbatim, substituting `$ANVIL_ALLOC`,
   `$HF_TOKEN`, etc. with the values the user gave you.
3. Run the **Smoke** flow before any full sweep. If the smoke job
   doesn't produce a passing trial, jump to **Troubleshooting**.
4. After smoke passes, run **Full sweep** or **Paired** depending on
   what the user asked for.
5. Hard-coded assumptions in the .sbatch / `run_inside_container.sh`
   (TP=2, max-num-seqs=16, walltime=4h, etc.) are listed in **OOM /
   tuning knobs** with rules for when to change them.

The runbook is designed to be self-contained — every command shown is
intended to work as written, with placeholders only in the
REQUIRED-FROM-USER table. If a step says "ask the user" or "stop here,"
take that literally.

## Why Anvil over vast.ai

| | vast.ai (today) | Anvil (target) |
|---|---|---|
| Cost | ~$1.50–3/hr per node | free under ACCESS allocation |
| Best GPU | A100 SXM4 NVLink | **H100** (Anvil AI partition) |
| Parallelism | 1 instance at a time | up to 12 GPUs/user simultaneously |
| Persistence | always-on SSH+tmux | SLURM jobs, ≤48h walltime |
| Container | Docker | Singularity |
| Internet from compute | unrestricted | login: yes; compute: assumed limited |

The 48-hour walltime cap is the only structural constraint. Each model in
the 6-model sweep takes ≤3 hours on H100 with DFlash, so a single sbatch
covers one model with margin. The driver `submit_loop.sh` queues all 6
jobs in parallel — total wall-clock for the sweep drops from days
(vast.ai sequential loop) to one job's worth.

## Files in this directory

```
benchmark/anvil/
├── README.md                       (this file — operational runbook)
├── Singularity.dflash              build recipe: vLLM PR #40898 (Qwens + Llama-8B)
├── Singularity.dflash_gemma        build recipe: vLLM PR #41703 (Gemma 4)
├── anvil_run_model.sbatch          sbatch template for one-model job
├── run_inside_container.sh         entrypoint executed inside the .sif
└── submit_loop.sh                  driver: queue all 6 model jobs
```

All four shell scripts are reachable from `benchmark/README.md` via the
"Multi-model loop on Anvil" section.

## REQUIRED-FROM-USER inputs

A fresh Claude session running this runbook needs these values from the
human operator before it can proceed. Fill them in here once approved:

| Variable | Example | Where to get it |
|---|---|---|
| `ANVIL_ALLOC` | `MED230001` | ACCESS allocation page after approval |
| `ANVIL_USER` | `x-jsong42` | `whoami` on the Anvil login node |
| `HF_TOKEN` | `hf_VyVODcv...` | huggingface.co → Settings → Access Tokens |
| `LAPTOP_REPO` | `/Users/johnsong/PycharmProjects/new-sionna-skill` | local laptop path used for rsync |

**Stop and ask the user for any value that's still a placeholder. Do not
guess.** All five `z-lab/*-DFlash` drafts are gated repos — the user must
have clicked "Request access" on each model page (huggingface.co/z-lab/...)
before any DFlash run can pull weights, even with a valid token.

## One-time setup (~1 day, mostly waiting on builds)

### Step 1 — Get an ACCESS allocation
1. Apply at <https://allocations.access-ci.org/> using your institutional
   email.
2. Request the **Discover** tier first (instant, ~400k credits free) for
   testing, then **Maximize** for the full benchmark (~1M+ credits).
3. NAIRR-Pilot track is fastest for AI workloads — check that box.
4. Once approved, note your allocation name (looks like `MED230001`).
   Set `ANVIL_ALLOC=<your-alloc>` in your shell rc.

### Step 2 — Login to Anvil and stage the repo
```bash
ssh login.anvil.rcac.purdue.edu      # use ACCESS credentials
echo "ANVIL_USER=$(whoami)"          # capture for later

# One-time directory layout on scratch (100 TB, 30-day purge by access time)
mkdir -p $SCRATCH/skill/logs $SCRATCH/hf_cache
```

**Stage the repo via rsync from the user's laptop.** This is the only
documented path right now — branch `rf-sim-agent-v2` lives only on the
laptop and on sunlab; the Anvil-specific files (everything in this
directory) are not yet on the public origin. **Run from the laptop**:
```bash
# REQUIRES: ANVIL_USER set on the laptop (e.g., export ANVIL_USER=x-jsong42)
rsync -avz \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='benchmark/results' --exclude='benchmark/_studies_archive' \
    --exclude='.venv' --exclude='node_modules' --exclude='docs/' \
    --exclude='scene_state.json' --exclude='simulation_result.json' \
    --exclude='.claude/skills/rf-simulator/memory/chroma_db' \
    "$LAPTOP_REPO/" \
    "${ANVIL_USER}@login.anvil.rcac.purdue.edu:/anvil/scratch/${ANVIL_USER}/skill/"
```
Once the branch is pushed to origin, the alternative `git clone … &&
git checkout rf-sim-agent-v2` from the Anvil login node will work too.

### Step 3 — Build the Singularity images
Done **once** on the login node (which has internet); then compute jobs
mount the read-only `.sif` files. ~30-40 minutes per image.
```bash
module load singularity
cd $SCRATCH/skill

# Image A: Qwens + Llama-3.1-8B (PR #40898)
singularity build $SCRATCH/skill_dflash.sif \
    benchmark/anvil/Singularity.dflash

# Image B: Gemmas (PR #41703) — separate because the two PRs conflict
singularity build $SCRATCH/skill_dflash_gemma.sif \
    benchmark/anvil/Singularity.dflash_gemma

# Optional baseline image (no DFlash) for paired comparison —
# build a Singularity.baseline that uses mainline vllm if needed.
# For now the same skill_dflash.sif works since USE_DFLASH=0 just
# omits the --speculative-config flag.
ln -s $SCRATCH/skill_dflash.sif $SCRATCH/skill_baseline.sif
```

### Step 4 — Pre-stage HuggingFace caches
Compute nodes likely cannot download from huggingface.co directly.
Pull all 6 model weights (and the 5 DFlash drafts if `USE_DFLASH=1`) on
the login node, where they land in `$SCRATCH/hf_cache`.

**Before running: confirm the user has been granted access to every
`z-lab/*-DFlash` repo.** The drafts are gated — a valid `HF_TOKEN` alone
is not enough; the human user must visit each model page below in a
browser and click "Request access". Approval is usually instant for
z-lab but is per-account, not per-token.

```bash
# REQUIRED FROM USER: HF_TOKEN with z-lab access approved (see above)
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before running step 4}"
export HF_HOME=$SCRATCH/hf_cache
pip install --user -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

FAILED=()
for repo in \
  Qwen/Qwen3.6-27B \
  Qwen/Qwen3.6-35B-A3B \
  Qwen/Qwen3-Coder-30B-A3B-Instruct \
  casperhansen/llama-3.3-70b-instruct-awq \
  google/gemma-4-31B-it \
  google/gemma-4-26B-A4B-it \
  z-lab/Qwen3.6-27B-DFlash \
  z-lab/Qwen3.6-35B-A3B-DFlash \
  z-lab/Qwen3-Coder-30B-A3B-DFlash \
  z-lab/gemma-4-31B-it-DFlash \
  z-lab/gemma-4-26B-A4B-it-DFlash; do
    echo "▶ $repo"
    huggingface-cli download "$repo" --quiet || FAILED+=("$repo")
done
du -sh $SCRATCH/hf_cache
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "✗ Could not download these repos:"
    printf '   - https://huggingface.co/%s\n' "${FAILED[@]}"
    echo "  → ASK THE HUMAN USER to visit each URL in a browser and click"
    echo "    'Request access', then re-run this step."
    exit 1
fi
```
Expect ~600 GB across all 6 base models + 5 DFlash drafts. Within the
100 TB scratch quota.

### Step 5 — Confirm internet policy on compute nodes
Ask `rcac-help@purdue.edu` *or* test directly via `gpu-debug`:
```bash
sinteractive -p gpu-debug -A $ANVIL_ALLOC --time=00:10:00 --gpus-per-node=1
# inside the interactive session:
curl -fsS -m 10 https://huggingface.co || echo "compute nodes are airgapped"
exit
```
If airgapped: Step 4 is mandatory; otherwise the .sbatch can lazy-download.

## Running the benchmark

### Smoke (5–10 min on `gpu-debug`)
Validates the .sif image, model load, vLLM startup, single trial completion.
```bash
cd $SCRATCH/skill

# DFlash-enabled smoke on Qwen3.6-27B (one task, 30-min walltime)
PARTITION=gpu-debug WALLTIME=00:25:00 USE_DFLASH=1 \
MODELS_FILTER='Qwen3.6-27B' \
bash benchmark/anvil/submit_loop.sh -A $ANVIL_ALLOC

# Watch it
squeue -u $USER
tail -f $SCRATCH/skill/logs/skill-Qwen__Qwen3.6-27B-*.out
```

### Full sweep (parallel, all 6 models)
```bash
# Baseline (no DFlash)
bash benchmark/anvil/submit_loop.sh -A $ANVIL_ALLOC

# DFlash sweep
USE_DFLASH=1 \
bash benchmark/anvil/submit_loop.sh -A $ANVIL_ALLOC
```
Each call submits 5–6 independent jobs. SLURM scheduler decides when each
runs; expect 0–12 hour queue wait depending on `gpu` partition load.

### Paired DFlash vs baseline
The cleanest paper-table data comes from running both as a paired sweep:
```bash
# Submit baseline run, capture job ids
JIDS_BASELINE=$(USE_DFLASH=0 bash benchmark/anvil/submit_loop.sh -A $ANVIL_ALLOC | grep job_id)
# Submit DFlash run
JIDS_DFLASH=$(USE_DFLASH=1   bash benchmark/anvil/submit_loop.sh -A $ANVIL_ALLOC | grep job_id)
```
Same task seeding, same workers — only the `--speculative-config` flag differs.
Compare `wall_sec` and `usage.totals.output_tokens / wall_sec` across the
two label sets when both finish.

### k=3 sampling for confidence bands
Bump `--k 3` in `run_inside_container.sh` and use `WALLTIME=24:00:00`.
Each model takes ~3× longer; still well under the 48h cap.

## Monitoring & retrieving results

```bash
# Job status (all your jobs)
squeue -u $USER

# Detailed status of one job
sacct -j <JID> --format=JobID,State,Elapsed,MaxRSS,ReqGRES

# Live tail of a running job
tail -f $SCRATCH/skill/logs/<job-name>-<JID>.out

# Aggregate finished result files (mirrors vast.ai pattern)
ls $SCRATCH/skill/benchmark/results/anvil_*

# Pull results back to laptop (if not running analysis on Anvil directly)
rsync -avz \
    <anvil_user>@login.anvil.rcac.purdue.edu:/anvil/scratch/<anvil_user>/skill/benchmark/results/anvil_* \
    ~/PycharmProjects/new-sionna-skill/benchmark/results/
```

## Chroma vector DB on Anvil

Vast.ai used a reverse SSH tunnel from sunlab to expose a network chroma
to the rented instance. Anvil compute nodes can't accept that tunnel, so
we use chroma as a local persistent client in scratch:

- `run_inside_container.sh` exports `CHROMA_HOST=""` (disables the
  HttpClient code path in `store.py:_ensure_initialized`) and
  `SIONNA_SKILL_CHROMA_PATH=/work/scratch/chroma_db` (overrides
  `_resolve_db_path`'s default location).
- The first time you run, the chroma_db dir is empty; populate it via:
  ```bash
  # On the Anvil login node, after Step 2 (repo staged) and Step 3 (.sif built):
  singularity exec --nv \
      --bind $SCRATCH/skill:/work/skill \
      --bind $SCRATCH:/work/scratch \
      --pwd /work/skill \
      --env SIONNA_SKILL_CHROMA_PATH=/work/scratch/chroma_db \
      $SCRATCH/skill_dflash.sif \
      python3 .claude/skills/rf-simulator/scripts/index_skill_artifacts.py
  ```
  This indexes the skill's scripts/templates/SKILL.md into the local
  chroma. Re-run after edits to the skill content.
- Different sbatch jobs sharing the same `$SCRATCH/chroma_db` is fine —
  chroma's PersistentClient is process-local but the underlying SQLite
  + HNSW files survive readers; conflicting concurrent writes are rare
  in practice (the indexer is the only writer, and it runs once).

## Differences from vast.ai workflow

| Vast.ai pattern | Anvil equivalent |
|---|---|
| `tmux new-session -d -s loop "bash run_loop_v3_tp.sh"` | `bash submit_loop.sh -A $ALLOC` (returns immediately, jobs queue) |
| `tmux ls`, `tmux attach -t loop` | `squeue -u $USER`, `tail -f logs/*.out` |
| Reverse SSH chroma tunnel from sunlab | local persistent client (see "Chroma vector DB on Anvil" above) |
| `bootstrap_vastai_tp.sh` (1100+ lines: rsync, env probe, vLLM, proxy, bench) | `anvil_run_model.sbatch` (sbatch directives) → `run_inside_container.sh` (~150 lines: vLLM, proxy, bench) |
| `ssh root@ssh6.vast.ai -L 8080:localhost:8080` for openclaude OAuth | n/a — openclaude is baked into the .sif and uses `--provider openai` against the local vLLM proxy, no OAuth flow |

## Cost / credit budget

Rough estimates assuming Anvil A100 = 4 credits/GPU-hr (verify in your
allocation page; the rate is partition-specific).

| Sweep | GPUs × hours | Credits |
|---|---|---|
| One-model smoke (gpu-debug) | 2 × 0.25 | 2 |
| Single-model bench (DFlash) | 2 × 1.5 | 12 |
| Full 6-model sweep, baseline | 6 × 2 × 2 | 48 |
| Full 6-model sweep, DFlash | 6 × 2 × 1 | 24 |
| Paired sweep (both above) | — | 72 |
| k=3 sampling on the paired set | — | ~220 |

Budget the **Discover** tier (~400k credits) handles >50 paired sweeps —
plenty of room for paper iterations. H100 (Anvil AI) is more expensive
per credit but ~2× faster; once benchmarked on A100, port the long
production runs to H100 partition by changing `--partition=gpu-h100` (or
the equivalent name — confirm via `sinfo -s`).

## Known unknowns to confirm before going live

These need a quick email to `rcac-help@purdue.edu` or a test on `gpu-debug`:

1. Does the H100 partition have a separate name (e.g., `gpu-h100`) or is
   it a `--gpus h100:2` constraint within `gpu`?
2. Outbound internet from compute nodes — yes/no?
3. Do NGC pytorch:25.01 images run unmodified under the `--nv` flag, or
   does Anvil require a patched base?
4. Is there a faster scratch tier (NVMe burst buffer) for HF caches?
5. Real per-credit GPU-hour rate — the table above is an estimate.

## Troubleshooting

Diagnostic order: first check `sacct` for the SLURM-level outcome, then
the per-job log, then the vLLM log, then the proxy log.

### Job ended at the walltime limit (`State=TIMEOUT` from `sacct`)
The benchmark didn't finish within the requested `--time`. Either:
- The model is slower than expected (Llama-3.3-70B at FP16 with no
  DFlash can take 4-5h for the full task set). **Fix**: bump
  `WALLTIME=08:00:00` in the env passed to `submit_loop.sh`.
- vLLM took >15 min to load (TP init or DFlash draft download). **Fix**:
  pre-warm by running a 1-task `gpu-debug` job first; second run hits
  the HF cache.

### Job died early (`State=FAILED`, `Elapsed` <5 min)
Almost always one of:
- Singularity image missing: `ls $SCRATCH/skill_dflash.sif` should exist.
- `MODEL_HF` not set: check the sbatch `--export=` line includes it.
- HF gated repo (z-lab/*): see Step 4. Look for `403 Client Error` in
  the job log. Resolution requires the human user to request access.
- vLLM `_C` import failure (libstdc++ CXXABI mismatch): only a problem
  if you replaced the NGC base; the `pytorch:25.01` base ships GCC 13+.

### vLLM never reaches `/v1/models`
Tail `$SCRATCH/skill/logs/vllm_<JOBID>.log`:
- `shm_broadcast: No available shared memory broadcast block found in
  60 seconds` → TP=2 NCCL hang. The `--disable-custom-all-reduce
  --distributed-executor-backend mp` flags in `run_inside_container.sh`
  should mitigate; if not, fall back to `VAST_TP_SIZE=1` (smaller
  models) or escalate to RCAC.
- `CUDA out of memory` during `--gpu-memory-utilization 0.85` profile
  run → drop to `0.80` or lower `--max-num-seqs` from 16 to 8.
- `KeyError: 'dflash'` → mainline vLLM was used instead of the DFlash
  PR build. Verify which `.sif` got bound (job log echoes `SIF=...`).

### Trial completes but `verification.passed` always False
Verifier rejects the agent's output. Common causes:
- `simulation_result.json` still has `status="placeholder_pre_shipped_by_harness"`
  → openclaude is reaching the model but the model isn't actually
  running the workload. Tail the trial's `stdout.txt` for the agent's
  reasoning. Often a sign the prompt isn't reaching the model
  (proxy / `OPENAI_BASE_URL` mismatch).
- Verifier complains about missing artifacts → the trial's tool-use
  failed. Check `stderr.txt`.

## OOM / tuning knobs

The defaults in `run_inside_container.sh` target H100 80GB / 4× A100 80GB
nodes with TP=2:

| Flag | Default | Lower if … | Raise if … |
|---|---|---|---|
| `--max-model-len` | 32768 | OOM during profile run | very long contexts cut off |
| `--max-num-seqs` | 16 | OOM at high concurrency | bench workers idle |
| `--gpu-memory-utilization` | 0.85 | OOM during model load | KV cache full → eviction |
| `--tensor-parallel-size` | 2 (`VAST_TP_SIZE`) | hardware mismatch | model too big for 2 GPUs (set 4 for FP16 70B) |
| `VAST_BENCH_WORKERS` | 14 | CPU saturated | GPU underutilized |
| `VAST_DFLASH_SPEC_TOKENS` | 15 | acceptance rate <50% | acceptance rate >85% (more spec tokens = more skip) |

## Open work items (future)

- [ ] Adapt `bootstrap_vastai_*.sh` per-family flag block into a shared
      Bash function library so `run_inside_container.sh` doesn't duplicate it.
- [ ] Sentinel-job pattern (`afterok` dependency) to chain
      paired with-skill / no-skill runs deterministically.
- [ ] Switch chroma to a persistent client; drop the autossh tunnel.
- [ ] Singularity image diet — current ~12 GB; could shrink by removing
      torch test data and unused vendored kernels.
