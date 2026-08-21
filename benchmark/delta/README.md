# Delta-port of the rf-simulator benchmark

This directory holds the NCSA Delta GPU port of `benchmark/anvil/`. The
container image, skill files, templates, and `run_inside_container.sh`
remain identical — only the SLURM submission script and host-path
conventions change.

## When to use

When Anvil's `cis260614-gpu` allocation depletes and you've been
approved for a Delta-GPU ACCESS allocation. As of 2026-05-13 we're at
95.3 / 100 SU on Anvil; one more parallel pair exhausts it.

## Pre-flight checklist

1. ACCESS-CI Explore (or Discover) allocation **on Delta GPU**,
   approved by ACCESS Allocations team. Submit at
   <https://allocations.access-ci.org/> — Explore needs no proposal,
   processed continuously.
2. NCSA Duo MFA enrolled at <https://identity.ncsa.illinois.edu/>.
   This is **separate** from Purdue's MFA — re-enroll there.
3. SSH access tested: `ssh x-<username>@login.delta.ncsa.illinois.edu`
   should land you on the login node.

## Staging the workload to Delta

```bash
# On Delta login node:
DELTA_PROJECT=<project_code>  # e.g. bbka — from allocation award
SCRATCH=/scratch/$DELTA_PROJECT/$USER
mkdir -p $SCRATCH

# Pull the repo (use the same branch that was last validated on Anvil)
git clone https://github.com/Pervasive-Intelligence-Lab/sionna-skill.git $SCRATCH/skill
cd $SCRATCH/skill
git checkout rf-sim-agent-v2

# Pre-stage HuggingFace cache (largest cost). Either:
# (a) rsync from Anvil (~330 GB):
rsync -aP x-<username>@anvil.rcac.purdue.edu:/anvil/scratch/x-<username>/hf_cache/ $SCRATCH/hf_cache/
# OR
# (b) Re-download in a single Delta job using the same dl_models sbatch
#     approach we used on Anvil (see benchmark/anvil/download_models.sbatch).

# Build the baseline Singularity/Apptainer image on a Delta gpuA40x4 node:
#   sbatch -A <delta-alloc> -p gpuA40x4 ... build_image.sh
# Or rsync the prebuilt baseline image:
rsync -P x-<username>@anvil.rcac.purdue.edu:/anvil/scratch/x-<username>/skill_baseline.sif $SCRATCH/
```

## Submission

```bash
sbatch -A <delta-alloc> --export=ALL,\
DELTA_PROJECT=<project_code>,\
MODEL_HF=casperhansen/llama-3.3-70b-instruct-awq,\
MODEL_PARSER=llama3_json,\
USE_DFLASH=0,\
VAST_TP_SIZE=2,\
BENCH_SPLIT=all,\
BENCH_CONDITIONS="with_skill no_skill" \
benchmark/delta/delta_run_model.sbatch
```

## What's different from Anvil

| Aspect | Anvil | Delta |
|---|---|---|
| Partition | `gpu` / `ai` | `gpuA100x4` / `gpuA40x4` / `gpuH200x4` |
| Scratch path | `/anvil/scratch/$USER` | `/scratch/<project>/$USER` |
| Container CLI | `singularity` | `apptainer` (drop-in compatible) |
| MFA | Purdue Duo (BoilerKey) | NCSA Duo |
| Account format | `cis260614-gpu` | depends on award |
| TP=2 hardware | 2× A100-40GB or H100 | 2× A100-40GB (`gpuA100x4`) or 2× A40-48GB (`gpuA40x4`) or 2× H200 (`gpuH200x4`) |
| GPUs per node | 4 | 4 |

## What's identical

- `benchmark/anvil/run_inside_container.sh` — runs unchanged inside the container
- `.claude/skills/rf-simulator/*` — skill substrate, all references, templates
- `benchmark/run_benchmark.py` — task driver, conditions, splits
- `benchmark/tool_call_proxy.py` — proxy with all shims (`PROXY_AUTO_TOOLS`, `PROXY_COALESCE_ROLES`)
- All paper-compliance reference docs under `references/`
- The four Singularity recipes (build commands are the same; image is identical)

The benchmark harness is intentionally portable — only the outer SLURM
header and host-side paths needed adaptation.

## Cost expectations

On Delta-GPU, each Llama-3.3-70B-AWQ FULL bench (322 trials, 2× A100,
50-min walltime) consumes roughly **the A100-equivalent of ~3-4 SU**
(Delta SU exchange rate vs Anvil may differ; check `accounting` on the
login node after a test job).

A full sweep matching the paper's "top 6 LLMs" goal (Llama-70B,
Q3-Coder-30B, Mixtral-AWQ, Phi-4, Granite, plus one variant) is
~20-30 Delta SU — well within an Explore allocation's typical 25k SU
ceiling.
