#!/bin/bash
# bootstrap_vastai_tp.sh — sibling of bootstrap_vastai.sh that launches ONE
# vLLM instance with --tensor-parallel-size $VAST_TP_SIZE (default 8) so the
# model weights are SHARDED across all GPUs. The KV-cache pool is therefore
# the sum of (per-GPU mem * util - per-GPU weight slice), which on 8× A100 80GB
# yields ~600 GB and unlocks much higher --max-num-seqs (~24-32) than the
# TP=1 single-instance bootstrap can.
#
# WARNING — read before launching:
#
#   1. PCIe all-reduce penalty. If the rented box is 8× A100 PCIe (NOT NVLink/
#      SXM4 NVSwitch), every TP forward pass goes through PCIe Gen4 (~32 GB/s
#      bidirectional) for the all-reduce after each MLP/attention shard. On
#      8-way TP this typically costs **50-70% of per-token throughput** vs
#      NVLink. The win is concurrency (50× more in-flight requests), not
#      per-token latency. If you only need 1-2 concurrent requests, prefer
#      bootstrap_vastai.sh (TP=1 + Sionna on GPU 1).
#
#   2. Multi-rank NCCL hangs. We have empirical evidence (§3, §8.3 of
#      docs/plan_vast_ai_benchmark.md) that on vLLM 0.20.0 / CUDA 13 / NCCL
#      2.28.9, even TP=2 hung at
#         "shm_broadcast: No available shared memory broadcast block found in
#          60 seconds"
#      on 2× A100 SXM4. TP=8 over PCIe carries the SAME risk plus PCIe latency.
#      This script DOES NOT promise TP=8 will work; it ensures we fail fast
#      with a useful error if NCCL hangs at init. If you see shm_broadcast,
#      fall back to bootstrap_vastai.sh (TP=1) or try VAST_TP_SIZE=2 first.
#
#   3. Workers > sunlab-validated 6. The TP default works out to ~23 workers,
#      which exceeds the empirically-validated 6 from sunlab. The harness has
#      a known multiprocessing-pool teardown bug (§3.5) that gets MORE likely
#      at higher concurrency. Have `--resume` ready as the escape valve when
#      the run truncates. Results from completed trials are saved.
#
# Pasteable into vast.ai's "On-start command" field, or run via SSH:
#   scp bootstrap_vastai_tp.sh root@<host>:/root/
#   ssh -i ~/.ssh/vastai -p <port> root@<host> 'bash /root/bootstrap_vastai_tp.sh'
#
# Required env vars:
#   VAST_REPO_RSYNC_SRC   e.g. "you@sunlab:~/PycharmProjects/new-sionna-skill"
#                         (same as bootstrap_vastai.sh)
#
# Optional env vars (TP-specific defaults; all others identical to
# bootstrap_vastai.sh — see that file's header for full reference):
#
#   VAST_TP_SIZE                  number of GPUs for the single vLLM instance
#                                 to shard across. Default: 8. Must be ≤ the
#                                 actual GPU count on the box.
#   VAST_MODEL_NUM_SEQS_TP        vLLM --max-num-seqs for the TP run.
#                                 Default: 24. (TP=8 on 8× A100 80GB with
#                                 Qwen3.6-27B BF16: 51 GB total weights → 6.4
#                                 GB per GPU; KV pool per GPU is
#                                 80 × 0.85 - 6.4 ≈ 57.6 GB, summed → ~460 GB.
#                                 At 8.3 GB/seq @ 32K context × 8 shards, that
#                                 supports comfortably more than 24 in flight,
#                                 so 24 is conservative.)
#                                 NOTE: this script consumes VAST_MODEL_NUM_SEQS_TP,
#                                 NOT VAST_MODEL_NUM_SEQS, so the two bootstraps
#                                 can be invoked side-by-side without env collision.
#   VAST_BENCH_WORKERS            harness workers. If unset, auto-set to
#                                 max-num-seqs - 1.
#   VAST_VLLM_WAIT_S              max seconds to wait for vLLM ready.
#                                 Default: 900 (15 min). TP init is slower than
#                                 TP=1 due to NCCL handshake + per-shard load.
#   VAST_GPU_MEM_UTIL_TP          --gpu-memory-utilization for the TP run.
#                                 Default: 0.85 (lower than TP=1's 0.92 because
#                                 each rank also holds working buffers).
#
# Exit behavior: same as bootstrap_vastai.sh — fail fast on any step. The TP
# init step (12) explicitly watches for shm_broadcast and exits with a
# diagnostic message if it sees it, instead of letting the 15-min timer
# silently expire.

set -euo pipefail

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG=/workspace/boot.log
mkdir -p /workspace /workspace/logs
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------
# Config (overridable via env)
# ---------------------------------------------------------------------
VAST_REPO_RSYNC_SRC="${VAST_REPO_RSYNC_SRC:-}"
VAST_REPO_RSYNC_KEY="${VAST_REPO_RSYNC_KEY:-/root/.ssh/sunlab_pull}"
VAST_TUNNEL_PORT="${VAST_TUNNEL_PORT:-8765}"
VAST_TUNNEL_WAIT_S="${VAST_TUNNEL_WAIT_S:-1800}"

VAST_MODEL_HF="${VAST_MODEL_HF:-Qwen/Qwen3.6-27B}"
VAST_MODEL_PARSER="${VAST_MODEL_PARSER:-hermes}"
VAST_MODEL_MAX_LEN="${VAST_MODEL_MAX_LEN:-32768}"

# --- TP-specific knobs ---
VAST_TP_SIZE="${VAST_TP_SIZE:-8}"
VAST_MODEL_NUM_SEQS_TP="${VAST_MODEL_NUM_SEQS_TP:-24}"
VAST_VLLM_WAIT_S="${VAST_VLLM_WAIT_S:-900}"
VAST_GPU_MEM_UTIL_TP="${VAST_GPU_MEM_UTIL_TP:-0.85}"

# Workers default = max-num-seqs - 1 (leave one slot for any in-flight retry
# without immediately queueing). User can still override.
VAST_BENCH_WORKERS="${VAST_BENCH_WORKERS:-$((VAST_MODEL_NUM_SEQS_TP - 1))}"

VAST_BENCH_LABEL="${VAST_BENCH_LABEL:-train_full_v1_tp${VAST_TP_SIZE}}"
VAST_AUTOLAUNCH_BENCHMARK="${VAST_AUTOLAUNCH_BENCHMARK:-1}"

VAST_CATALOG_SRC="${VAST_CATALOG_SRC:-prestaged}"
VAST_CATALOG_RSYNC_SRC="${VAST_CATALOG_RSYNC_SRC:-}"
VAST_CATALOG_PATH="${VAST_CATALOG_PATH:-/data/3D-FUTURE-model}"
VAST_REQUIRE_CATALOG="${VAST_REQUIRE_CATALOG:-1}"

BENCH_MODEL=benchmark-model

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
T0=$(date +%s)
ts() { printf "[%s | t+%ds] " "$(date +%H:%M:%S)" "$(($(date +%s) - T0))"; }
step() { echo -e "\n$(ts)=== $1 ==="; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ! $1"; }
err()  { echo "  ✗ $1" >&2; exit 1; }

DOWNLOAD_PID=""
PIP_PID=""

# =====================================================================
step "0. Sanity check (TP=$VAST_TP_SIZE)"
[[ "$(id -u)" == "0" ]] || err "must run as root on vast.ai"
[[ "${VAST_SKIP_REPO_RSYNC:-0}" == "1" ]] || [[ -n "$VAST_REPO_RSYNC_SRC" ]] || err "VAST_REPO_RSYNC_SRC env var is required (or set VAST_SKIP_REPO_RSYNC=1 if /workspace/skill is pre-staged)"
[[ "${VAST_SKIP_REPO_RSYNC:-0}" == "1" ]] || [[ -f "$VAST_REPO_RSYNC_KEY" ]] || err "rsync SSH key missing at $VAST_REPO_RSYNC_KEY"

# TP-specific GPU count check
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
GPU_COUNT=${GPU_COUNT:-0}
if [[ "$GPU_COUNT" -lt "$VAST_TP_SIZE" ]]; then
    err "VAST_TP_SIZE=$VAST_TP_SIZE but only $GPU_COUNT GPUs visible.
    Either rent a box with ≥$VAST_TP_SIZE GPUs, or lower VAST_TP_SIZE (e.g. =$GPU_COUNT)."
fi
ok "GPUs visible: $GPU_COUNT (need ≥$VAST_TP_SIZE for TP=$VAST_TP_SIZE)"

# Heads-up if interconnect is PCIe (NVLink/NVSwitch usually report 'NV#' in
# `nvidia-smi topo -m`; PCIe-only shows 'PHB'/'PIX'/'SYS'). Best-effort; do
# not fail because we can't reliably tell from inside an unprivileged
# container on every host.
if command -v nvidia-smi > /dev/null; then
    TOPO=$(nvidia-smi topo -m 2>/dev/null || true)
    if [[ -n "$TOPO" ]] && ! echo "$TOPO" | grep -qE "NV[0-9]"; then
        warn "no NVLink detected in 'nvidia-smi topo -m'. TP=$VAST_TP_SIZE all-reduce
       will run over PCIe. Expect 50-70% per-token throughput penalty vs NVLink.
       Win comes from concurrency (max-num-seqs=$VAST_MODEL_NUM_SEQS_TP), not latency."
    else
        ok "interconnect appears NVLink-capable"
    fi
fi

# Catalog config sanity (identical to bootstrap_vastai.sh)
case "$VAST_CATALOG_SRC" in
    prestaged)
        if [[ -z "$VAST_CATALOG_RSYNC_SRC" ]]; then
            err "VAST_CATALOG_SRC=prestaged requires VAST_CATALOG_RSYNC_SRC.
    Set it to your bucket holding registered 3D-FUTURE (native layout):
       VAST_CATALOG_RSYNC_SRC='you@sunlab:/data/3D-FUTURE-model'
    Or to skip catalog (T0 tasks degrade to fallback boxes):
       VAST_CATALOG_SRC=skip VAST_REQUIRE_CATALOG=0"
        fi
        ;;
    hf_3dfront|abo)
        warn "VAST_CATALOG_SRC=$VAST_CATALOG_SRC: skill's get_mesh_path() resolver
       expects native 3D-FUTURE layout. The catalog will download but the skill
       MAY fall back to AABB cubes at trial time until the resolver patch lands.
       For verified real-mesh resolution, use VAST_CATALOG_SRC=prestaged."
        ;;
    skip)
        if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
            err "VAST_CATALOG_SRC=skip but VAST_REQUIRE_CATALOG=1 — these conflict.
    To proceed without catalog (T0 tasks degraded): VAST_REQUIRE_CATALOG=0"
        fi
        ;;
    *)
        err "VAST_CATALOG_SRC must be one of: prestaged, hf_3dfront, abo, skip"
        ;;
esac

ok "running as root, repo=$VAST_REPO_RSYNC_SRC, catalog=$VAST_CATALOG_SRC"

# Disk-space sanity. Roughly: model weights + catalog + 50 GB overhead.
# (Identical to bootstrap_vastai.sh — TP doesn't change disk needs; weights
# are sharded at LOAD time, on disk they're still one set.)
DISK_AVAIL_GB=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
DISK_AVAIL_GB=${DISK_AVAIL_GB:-0}
case "$VAST_CATALOG_SRC" in
    skip)        CATALOG_GB=0 ;;
    abo)         CATALOG_GB=30 ;;
    hf_3dfront)  CATALOG_GB=100 ;;
    prestaged)   CATALOG_GB=50 ;;
esac
case "$VAST_MODEL_HF" in
    *27B*|*26[Bb]*) MODEL_GB=55 ;;
    *31[Bb]*)       MODEL_GB=65 ;;
    *35B*)          MODEL_GB=75 ;;
    *70B*-AWQ*|*70B*-INT4*|*-int4*) MODEL_GB=40 ;;
    *70B*)          MODEL_GB=140 ;;
    *)              MODEL_GB=80 ;;
esac
DISK_NEEDED=$((MODEL_GB + CATALOG_GB + 50))
echo "  disk: $DISK_AVAIL_GB GB free, ~$DISK_NEEDED GB estimated need (model=$MODEL_GB + catalog=$CATALOG_GB + overhead=50)"
if [[ "$DISK_AVAIL_GB" -lt "$DISK_NEEDED" ]]; then
    err "insufficient disk: $DISK_AVAIL_GB GB free, need ~$DISK_NEEDED GB. Re-rent with larger volume."
fi
ok "disk-space OK"

# =====================================================================
step "1. Stop vast.ai's pre-installed vLLM (eats 76 GB / GPU per §0 #1)"
supervisorctl stop vllm 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 4
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
ok "supervisor vllm stopped, GPU 0 mem now ${GPU_USED} MiB"

# =====================================================================
step "2. Write /workspace/skill.env"
cat > /workspace/skill.env <<EOF
export PATH=/root/.local/bin:/usr/local/bin:/opt/node-v22.11.0-linux-x64/bin:\$PATH
export IS_SANDBOX=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CHROMA_HOST=localhost
export CHROMA_PORT=$VAST_TUNNEL_PORT
export RF_SKILL_DIR=/workspace/skill/.claude/skills/rf-simulator
export CLAUDE_CODE_USE_RAG=1
export PYTHONPATH=/workspace/skill:\${PYTHONPATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=1
export CLAUDE_BIN=/usr/local/bin/openclaude-vllm
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=dummy-vllm
export OPENCLAUDE_BIN=/usr/local/bin/openclaude
export FURNITURE_CATALOG_PATH=$VAST_CATALOG_PATH
EOF
source /workspace/skill.env
ok "/workspace/skill.env written"

# =====================================================================
step "3. [PARALLEL START] Model download in background ($VAST_MODEL_HF)"
MODEL_DIR=/workspace/models/$(basename "$VAST_MODEL_HF")

# CONDITIONAL prune: only delete prior model weights if disk is too tight to
# fit the next model. Lets us cache multiple models when there's space (faster
# re-runs, less re-download on resume) and only sacrifice cache when needed.
# Threshold: keep at least 100 GB free after the new download lands.
MODEL_BASENAME=$(basename "$VAST_MODEL_HF")
mkdir -p /workspace/models
PRUNED=$(find /workspace/models -mindepth 1 -maxdepth 1 -type d ! -name "$MODEL_BASENAME" 2>/dev/null)
if [[ -n "$PRUNED" ]]; then
    DISK_FREE_GB=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    DISK_FREE_GB=${DISK_FREE_GB:-0}
    # Estimate new model size by HF id pattern. Conservative bands; better-safe
    # over-estimates trigger prune, under-estimates risk OOM mid-download.
    case "$VAST_MODEL_HF" in
        *70[Bb]*-[Aa][Ww][Qq]*|*70[Bb]*-instruct-[Aa][Ww][Qq]*) NEW_GB=45 ;;  # AWQ-quantized
        *35B-A3B*|*35[Bb]-A3B*) NEW_GB=75 ;;
        *27[Bb]*) NEW_GB=55 ;;
        *31[Bb]*|*30[Bb]*) NEW_GB=65 ;;
        *26[Bb]*-A4[Bb]*) NEW_GB=55 ;;
        *) NEW_GB=70 ;;  # conservative default
    esac
    HEADROOM_GB=100
    NEED_GB=$((NEW_GB + HEADROOM_GB))
    if [[ "$DISK_FREE_GB" -lt "$NEED_GB" ]]; then
        echo "  disk free ${DISK_FREE_GB} GB < needed ${NEED_GB} GB ($NEW_GB model + $HEADROOM_GB headroom)"
        echo "  pruning prior weights to make room: $PRUNED"
        echo "$PRUNED" | xargs rm -rf
        ok "pruned previous models from /workspace/models (disk was tight)"
    else
        ok "keeping prior weights cached (disk free=${DISK_FREE_GB} GB ≥ needed=${NEED_GB} GB): $PRUNED"
    fi
fi

if [[ -f "$MODEL_DIR/.complete" ]]; then
    ok "model already on disk at $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
    DOWNLOAD_PID=""
elif command -v hf > /dev/null; then
    mkdir -p "$MODEL_DIR"
    ( hf download "$VAST_MODEL_HF" --local-dir "$MODEL_DIR" \
        > /workspace/logs/download.log 2>&1 \
        && touch "$MODEL_DIR/.complete" ) &
    DOWNLOAD_PID=$!
    ok "model download PID=$DOWNLOAD_PID kicked off (log: /workspace/logs/download.log)"
else
    warn "hf CLI not in PATH yet — will defer download to step 9 (slower)"
    DOWNLOAD_PID=""
fi

# =====================================================================
step "4. [PARALLEL START] pip install in background"
( pip install --root-user-action=ignore -q -U \
    huggingface_hub hf_transfer chromadb-client sionna \
    trimesh shapely \
    'sentence-transformers>=3.0,<6.0' \
    > /workspace/logs/pip.log 2>&1 ) &
PIP_PID=$!
ok "pip install PID=$PIP_PID kicked off (log: /workspace/logs/pip.log)"

# =====================================================================
step "4b. [PARALLEL START] Furniture catalog ($VAST_CATALOG_SRC) in background"
CATALOG_PID=""
mkdir -p "$VAST_CATALOG_PATH"
case "$VAST_CATALOG_SRC" in
    hf_3dfront)
        if [[ -f "$VAST_CATALOG_PATH/.complete" ]]; then
            ok "catalog already on disk at $VAST_CATALOG_PATH ($(du -sh "$VAST_CATALOG_PATH" | cut -f1))"
        elif command -v hf > /dev/null; then
            (
                cd "$VAST_CATALOG_PATH" || exit 1
                hf download huanngzh/3D-Front --repo-type dataset \
                    --include "3D-FRONT-SCENE.part*" "valid_furniture_ids.json" \
                    --local-dir . > /workspace/logs/catalog.log 2>&1 \
                && cat 3D-FRONT-SCENE.part* > 3D-FRONT-SCENE.tar.gz \
                && tar -xzf 3D-FRONT-SCENE.tar.gz \
                && rm 3D-FRONT-SCENE.part* 3D-FRONT-SCENE.tar.gz \
                && touch "$VAST_CATALOG_PATH/.complete"
            ) &
            CATALOG_PID=$!
            ok "catalog download PID=$CATALOG_PID (HF huanngzh/3D-Front, ~50 GB after extract)"
        else
            warn "hf CLI not in PATH yet — defer catalog to after pip install"
        fi
        ;;
    abo)
        if [[ -f "$VAST_CATALOG_PATH/.complete" ]]; then
            ok "ABO catalog already on disk"
        elif command -v aws > /dev/null; then
            (
                aws s3 sync s3://amazon-berkeley-objects/3dmodels/original/ \
                    "$VAST_CATALOG_PATH/3dmodels/" --no-sign-request --quiet \
                    > /workspace/logs/catalog.log 2>&1 \
                && touch "$VAST_CATALOG_PATH/.complete"
            ) &
            CATALOG_PID=$!
            ok "ABO catalog download PID=$CATALOG_PID (~30 GB)"
        else
            warn "aws CLI missing — installing..."
            pip install --root-user-action=ignore -q awscli &> /dev/null
            (
                aws s3 sync s3://amazon-berkeley-objects/3dmodels/original/ \
                    "$VAST_CATALOG_PATH/3dmodels/" --no-sign-request --quiet \
                    > /workspace/logs/catalog.log 2>&1 \
                && touch "$VAST_CATALOG_PATH/.complete"
            ) &
            CATALOG_PID=$!
            ok "ABO catalog download PID=$CATALOG_PID"
        fi
        ;;
    prestaged)
        [[ -n "$VAST_CATALOG_RSYNC_SRC" ]] || err "VAST_CATALOG_SRC=prestaged requires VAST_CATALOG_RSYNC_SRC"
        (
            rsync -az -e "ssh -i $VAST_REPO_RSYNC_KEY -o StrictHostKeyChecking=accept-new" \
                "$VAST_CATALOG_RSYNC_SRC/" "$VAST_CATALOG_PATH/" \
                > /workspace/logs/catalog.log 2>&1 \
            && touch "$VAST_CATALOG_PATH/.complete"
        ) &
        CATALOG_PID=$!
        ok "prestaged catalog rsync PID=$CATALOG_PID from $VAST_CATALOG_RSYNC_SRC"
        ;;
    skip)
        warn "VAST_CATALOG_SRC=skip — T0_scene_gen tasks will use AABB box fallbacks (logged as caveat)"
        ;;
    *)
        err "VAST_CATALOG_SRC must be one of: hf_3dfront, abo, prestaged, skip (got: $VAST_CATALOG_SRC)"
        ;;
esac

# =====================================================================
step "5. Rsync repo from $VAST_REPO_RSYNC_SRC"
mkdir -p /workspace/skill
if [[ "${VAST_SKIP_REPO_RSYNC:-0}" == "1" ]]; then
    [[ -f /workspace/skill/benchmark/run_benchmark.py ]] || err "VAST_SKIP_REPO_RSYNC=1 but /workspace/skill/benchmark/run_benchmark.py missing — pre-stage the repo first"
    ok "repo at /workspace/skill (rsync skipped via VAST_SKIP_REPO_RSYNC=1, $(du -sh /workspace/skill 2>/dev/null | cut -f1))"
else
rsync -az -e "ssh -i $VAST_REPO_RSYNC_KEY -o StrictHostKeyChecking=accept-new" \
    --exclude='.git/' --exclude='__pycache__/' --exclude='benchmark/results/' \
    --exclude='benchmark/_studies_archive/' --exclude='*.pyc' --exclude='.venv/' \
    --exclude='node_modules/' --exclude='docs/superpowers/' \
    "$VAST_REPO_RSYNC_SRC/" /workspace/skill/
[[ -f /workspace/skill/benchmark/run_benchmark.py ]] || err "rsync didn't produce run_benchmark.py"
ok "repo at /workspace/skill ($(du -sh /workspace/skill | cut -f1))"
fi

# =====================================================================
step "6. Patch run_benchmark.py origin_id KeyError (§0 #13)"
RB=/workspace/skill/benchmark/run_benchmark.py
if grep -q 't\["origin_id"\] in ids' "$RB"; then
    sed -i 's|t\["origin_id"\] in ids|t.get("origin_id") in ids|' "$RB"
    ok "patched run_benchmark.py:94"
else
    ok "patch already applied or upstream-fixed"
fi

# =====================================================================
step "6b. Patch trial/invoke.py to pin Sionna RT to GPU 1 (harmless under TP=N)"
# In the TP variant, vLLM holds ALL $VAST_TP_SIZE GPUs. The CUDA_VISIBLE_DEVICES=1
# pin in the simulation subprocess is still applied for parity with bootstrap_vastai.sh
# and is harmless: the sim subprocess sees only "GPU 1" via this env, and Sionna
# will share that GPU with vLLM rank 1. On large radio-map tasks under heavy
# vLLM concurrency this CAN push GPU 1 toward OOM; if you see CUDA-OOM on
# Phase-2 RT-heavy tasks, lower workers or set BENCH_SIM_GPU to a less busy
# index at benchmark launch time (does not require a re-bootstrap).
INVOKE=/workspace/skill/benchmark/trial/invoke.py
if grep -q "BENCH_SIM_GPU" "$INVOKE"; then
    ok "trial/invoke.py already pinned (CUDA_VISIBLE_DEVICES patch present)"
else
    INVOKE_PATH="$INVOKE" python3 - <<'PY'
import os, re, sys
p = os.environ["INVOKE_PATH"]
src = open(p).read()
new = re.sub(
    r"(def _build_invoke_env\([^)]*\)[^:]*:.*?)\n(\s+)return env",
    r'\1\n\2# Pin Sionna RT in simulation subprocess to GPU 1 (vLLM holds GPU 0)\n\2env["CUDA_VISIBLE_DEVICES"] = os.environ.get("BENCH_SIM_GPU", "1")\n\2return env',
    src,
    count=1,
    flags=re.DOTALL,
)
if new == src:
    print("PATCH_FAIL: regex didn't match _build_invoke_env return env")
    sys.exit(1)
open(p, "w").write(new)
print("patched")
PY
    PATCH_RC=$?
    if [[ "$PATCH_RC" != "0" ]] || ! grep -q "BENCH_SIM_GPU" "$INVOKE"; then
        warn "couldn't auto-patch trial/invoke.py — Sionna will contend with vLLM on GPU 0
        Manual fix: in benchmark/trial/invoke.py's _build_invoke_env, before 'return env':
            env['CUDA_VISIBLE_DEVICES'] = '1'"
    else
        if ! PYTHONPATH=/workspace/skill python3 -c "import benchmark.trial.invoke" 2>/dev/null; then
            warn "trial/invoke.py broken after patch! Reverting via rsync from source..."
            rsync -az -e "ssh -i $VAST_REPO_RSYNC_KEY -o StrictHostKeyChecking=accept-new" \
                "$VAST_REPO_RSYNC_SRC/benchmark/trial/invoke.py" "$INVOKE"
            err "trial/invoke.py syntax broken after step 6b patch. File restored from source.
        Apply manually: add 'env[\"CUDA_VISIBLE_DEVICES\"] = \"1\"' before 'return env' in _build_invoke_env."
        fi
        ok "patched trial/invoke.py: simulation.py runs with CUDA_VISIBLE_DEVICES=1 (syntax verified)"
    fi
fi

# =====================================================================
step "7. Install Node + OpenClaude (apt PPAs blocked → use tarball, §0 #7)"

if [[ ! -x /opt/node-v22.11.0-linux-x64/bin/node ]]; then
    echo "  installing Node v22.11.0 from nodejs.org tarball..."
    curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz \
        -o /tmp/node.tar.xz
    tar -xJf /tmp/node.tar.xz -C /opt/
    rm /tmp/node.tar.xz
    for b in node npm npx; do
        ln -sf /opt/node-v22.11.0-linux-x64/bin/$b /usr/local/bin/$b
    done
fi
ok "node $(node --version)"

if [[ ! -d /opt/node-v22.11.0-linux-x64/lib/node_modules/@gitlawb/openclaude ]]; then
    echo "  npm install -g @gitlawb/openclaude@0.8.0 (this takes ~1 min; pinning 0.8.0 — 0.9.1 broke -p prompt mode)..."
    npm install -g --prefix /opt/node-v22.11.0-linux-x64 \
        @gitlawb/openclaude@0.8.0 --no-audit --no-fund 2>&1 | tail -3
fi
# Patch openclaude default max_tokens to fit in 32K context (vLLM rejects 32K+prompt)
OC_DIR=/opt/node-v22.11.0-linux-x64/lib/node_modules/@gitlawb/openclaude
sed -i -E "s/MAX_OUTPUT_TOKENS_DEFAULT = [0-9]+/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" "$OC_DIR/dist/cli.mjs"
grep -q "MAX_OUTPUT_TOKENS_DEFAULT = 8000" "$OC_DIR/dist/cli.mjs" || { echo "ERROR: max_tokens patch failed to apply"; exit 1; }
ln -sf /opt/node-v22.11.0-linux-x64/bin/openclaude /usr/local/bin/openclaude
ok "openclaude $(openclaude --version 2>&1 | head -1)"

OC_CLI=/opt/node-v22.11.0-linux-x64/lib/node_modules/@gitlawb/openclaude/dist/cli.mjs
if grep -q "MAX_OUTPUT_TOKENS_DEFAULT = 32000" "$OC_CLI"; then
    sed -i \
        -e "s/MAX_OUTPUT_TOKENS_DEFAULT = 32000/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" \
        -e "s/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64000/MAX_OUTPUT_TOKENS_UPPER_LIMIT = 8000/g" \
        -e "s/ESCALATED_MAX_TOKENS = 64000/ESCALATED_MAX_TOKENS = 8000/g" \
        "$OC_CLI"
    ok "openclaude max_tokens patched (32000 → 8000)"
else
    ok "openclaude already patched"
fi

if [[ -x /workspace/skill/benchmark/openclaude-vllm ]]; then
    cp /workspace/skill/benchmark/openclaude-vllm /usr/local/bin/openclaude-vllm
else
    cat > /usr/local/bin/openclaude-vllm <<'WRAP'
#!/bin/bash
exec /usr/local/bin/openclaude --provider openai "$@"
WRAP
fi
chmod +x /usr/local/bin/openclaude-vllm
ok "openclaude-vllm wrapper at /usr/local/bin/openclaude-vllm"

# =====================================================================
step "8. Wait for reverse SSH tunnel from sunlab (chroma at port $VAST_TUNNEL_PORT)"
TUNNEL_READY=0
WAIT_INTERVAL=5
ITERS=$((VAST_TUNNEL_WAIT_S / WAIT_INTERVAL))
for i in $(seq 1 $ITERS); do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:$VAST_TUNNEL_PORT/api/v1/heartbeat"; then
        ok "chroma reachable via tunnel (took $((i * WAIT_INTERVAL))s)"
        TUNNEL_READY=1
        break
    fi
    sleep $WAIT_INTERVAL
done
if [[ "$TUNNEL_READY" != "1" ]]; then
    err "chroma tunnel not up after ${VAST_TUNNEL_WAIT_S}s. Start from sunlab:
    nohup ssh -i ~/.ssh/vast_tunnel -N -R $VAST_TUNNEL_PORT:127.0.0.1:8000 \\
        -p <vast-port> root@<vast-host> > /tmp/forward.log 2>&1 &
    disown"
fi

# =====================================================================
step "9. Wait for pip install to finish (background since step 4)"
if wait $PIP_PID 2>/dev/null; then
    ok "pip install completed (PID was $PIP_PID)"
else
    warn "pip background returned non-zero; checking imports anyway"
    tail -10 /workspace/logs/pip.log
fi
python3 -c "import sionna, chromadb; print('sionna', sionna.__version__, '| chroma', chromadb.__version__)" \
    || err "sionna/chromadb not importable — see /workspace/logs/pip.log"
ok "deps OK"

# =====================================================================
step "10. Wait for model download (background since step 3)"
if [[ -n "$DOWNLOAD_PID" ]]; then
    if wait "$DOWNLOAD_PID" 2>/dev/null; then
        ok "model download completed ($(du -sh "$MODEL_DIR" | cut -f1))"
    else
        tail -20 /workspace/logs/download.log
        err "model download failed"
    fi
elif [[ ! -f "$MODEL_DIR/.complete" ]]; then
    echo "  falling back to inline download (pip install just finished)..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    mkdir -p "$MODEL_DIR"
    hf download "$VAST_MODEL_HF" --local-dir "$MODEL_DIR" 2>&1 | tail -3
    touch "$MODEL_DIR/.complete"
    ok "model downloaded ($(du -sh "$MODEL_DIR" | cut -f1))"
else
    ok "model already complete on disk"
fi

# =====================================================================
step "10b. Wait for catalog, fall back if step 4b couldn't kick it off"
if [[ -n "$CATALOG_PID" ]]; then
    if wait "$CATALOG_PID" 2>/dev/null; then
        ok "catalog download completed"
    else
        tail -20 /workspace/logs/catalog.log
        if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
            err "catalog download failed (set VAST_REQUIRE_CATALOG=0 to proceed with fallback boxes)"
        else
            warn "catalog download failed; continuing with fallback boxes"
            VAST_CATALOG_SRC=skip
        fi
    fi
elif [[ "$VAST_CATALOG_SRC" != "skip" ]] && [[ ! -f "$VAST_CATALOG_PATH/.complete" ]]; then
    echo "  catalog not started in parallel; running inline now (pip install is done)"
    case "$VAST_CATALOG_SRC" in
        hf_3dfront)
            (
                cd "$VAST_CATALOG_PATH" || exit 1
                hf download huanngzh/3D-Front --repo-type dataset \
                    --include "3D-FRONT-SCENE.part*" "valid_furniture_ids.json" \
                    --local-dir . > /workspace/logs/catalog.log 2>&1 \
                && cat 3D-FRONT-SCENE.part* > 3D-FRONT-SCENE.tar.gz \
                && tar -xzf 3D-FRONT-SCENE.tar.gz \
                && rm 3D-FRONT-SCENE.part* 3D-FRONT-SCENE.tar.gz \
                && touch "$VAST_CATALOG_PATH/.complete"
            ) || { tail -20 /workspace/logs/catalog.log; err "inline catalog download failed"; }
            ;;
        abo)
            pip install --root-user-action=ignore -q awscli &>/dev/null
            aws s3 sync s3://amazon-berkeley-objects/3dmodels/original/ \
                "$VAST_CATALOG_PATH/3dmodels/" --no-sign-request --quiet \
                > /workspace/logs/catalog.log 2>&1 \
                && touch "$VAST_CATALOG_PATH/.complete" \
                || { tail -20 /workspace/logs/catalog.log; err "inline ABO download failed"; }
            ;;
        prestaged)
            rsync -az -e "ssh -i $VAST_REPO_RSYNC_KEY -o StrictHostKeyChecking=accept-new" \
                "$VAST_CATALOG_RSYNC_SRC/" "$VAST_CATALOG_PATH/" \
                > /workspace/logs/catalog.log 2>&1 \
                && touch "$VAST_CATALOG_PATH/.complete" \
                || { tail -20 /workspace/logs/catalog.log; err "inline prestaged rsync failed"; }
            ;;
    esac
    ok "inline catalog download done"
fi

# =====================================================================
step "10c. Verify catalog usability (file count + size + skill-resolver test)"
if [[ "$VAST_CATALOG_SRC" == "skip" ]]; then
    warn "running without catalog — T0 scene_gen tasks will use AABB fallback boxes"
else
    GLB_COUNT=$(find "$VAST_CATALOG_PATH" -maxdepth 5 \( -name "*.glb" -o -name "*.obj" \) 2>/dev/null | wc -l)
    echo "  catalog disk:    $(du -sh "$VAST_CATALOG_PATH" 2>/dev/null | cut -f1)"
    echo "  mesh files:      $GLB_COUNT (.glb + .obj)"
    if [[ "$GLB_COUNT" -lt 100 ]]; then
        if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
            err "catalog has only $GLB_COUNT mesh files at $VAST_CATALOG_PATH (expected ≥100)"
        else
            warn "catalog mesh count low ($GLB_COUNT); benchmark T0 tasks may use fallback boxes"
        fi
    else
        ok "catalog has $GLB_COUNT mesh files"
    fi

    SAMPLE=$(find "$VAST_CATALOG_PATH" -maxdepth 5 \( -name "*.glb" -o -name "*.obj" \) 2>/dev/null | head -1)
    if [[ -n "$SAMPLE" ]]; then
        SAMPLE_SIZE=$(stat -c%s "$SAMPLE" 2>/dev/null || stat -f%z "$SAMPLE" 2>/dev/null || echo 0)
        if [[ "$SAMPLE_SIZE" -lt 10000 ]]; then
            warn "sample mesh is suspiciously small ($SAMPLE_SIZE bytes): $SAMPLE"
        else
            ok "sample mesh: $(basename "$SAMPLE") = ${SAMPLE_SIZE} bytes (looks real)"
        fi
    fi

    if [[ -f "$VAST_CATALOG_PATH/model_info.json" ]]; then
        N_ENTRIES=$(python3 -c "
import json, sys
try:
    d = json.load(open('$VAST_CATALOG_PATH/model_info.json'))
    if isinstance(d, list): print(len(d))
    elif isinstance(d, dict): print(len(d))
    else: print(0)
except Exception as e:
    print(f'parse-error: {e}')
" 2>&1)
        if [[ "$N_ENTRIES" =~ ^[0-9]+$ ]] && [[ "$N_ENTRIES" -gt 100 ]]; then
            ok "model_info.json has $N_ENTRIES entries — invoke.py will recognize this catalog"
        else
            warn "model_info.json parsed but only $N_ENTRIES entries (or invalid JSON)"
        fi
    else
        if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
            err "no model_info.json at $VAST_CATALOG_PATH — invoke.py can't recognize this as a 3D-FUTURE catalog.
    invoke.py probes for: <catalog>/model_info.json (3D-FUTURE native marker)
    Likely cause:
      - VAST_CATALOG_SRC=hf_3dfront or abo: layout doesn't match (fix: synthesize model_info.json)
      - VAST_CATALOG_SRC=prestaged: rsync source is missing model_info.json
      - corrupted/partial download
    Or set VAST_REQUIRE_CATALOG=0 to proceed with AABB fallback boxes."
        else
            warn "no model_info.json — trial subprocess will use AABB box fallbacks"
        fi
    fi
fi

# =====================================================================
step "10d. Verify chroma vector DB is reachable AND has content"
HEARTBEAT_RC=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$VAST_TUNNEL_PORT/api/v1/heartbeat")
[[ "$HEARTBEAT_RC" == "200" ]] || err "chroma tunnel heartbeat returned $HEARTBEAT_RC"

COLLECTION_CHECK=$(python3 -c "
import chromadb
try:
    c = chromadb.HttpClient(host='localhost', port=$VAST_TUNNEL_PORT)
    cols = c.list_collections()
    target = next((col for col in cols if col.name == 'sionna_skill_memory'), None)
    if target is None:
        print('ERR: no sionna_skill_memory collection')
    else:
        n = target.count()
        print(f'OK: sionna_skill_memory has {n} chunks')
except Exception as e:
    print(f'ERR: {e}')
" 2>&1)
if [[ "$COLLECTION_CHECK" == OK:* ]]; then
    ok "chroma RAG ready ($COLLECTION_CHECK)"
else
    err "chroma collection check failed: $COLLECTION_CHECK
    Re-seed via: ssh sunlab 'PATH=\$HOME/.local/bin:\$PATH bash benchmark/...scripts/seed_memory.py'"
fi

# =====================================================================
step "11. Start vLLM with TP=$VAST_TP_SIZE (max_len=$VAST_MODEL_MAX_LEN, num_seqs=$VAST_MODEL_NUM_SEQS_TP)"
# TP-mode rationale (vs bootstrap_vastai.sh's TP=1):
#   - --tensor-parallel-size $VAST_TP_SIZE: shard the LLM weights across all
#     $VAST_TP_SIZE GPUs. On 8× A100 80GB with Qwen3.6-27B BF16 (51 GB total),
#     each rank holds ~6.4 GB of weights, leaving ~57.6 GB per GPU for KV
#     cache (at gpu-memory-utilization=$VAST_GPU_MEM_UTIL_TP). Aggregate KV
#     pool across all ranks: ~460 GB → max-num-seqs=$VAST_MODEL_NUM_SEQS_TP
#     comfortably supported at 32K context.
#   - --enforce-eager: REQUIRED, not optional, on multi-rank vLLM. Per §0 #20,
#     CUDA-graph compilation is the most common multi-rank deadlock source.
#     Trades 10-20% per-token throughput for reliable startup.
#   - --gpu-memory-utilization $VAST_GPU_MEM_UTIL_TP (=0.85 default): lower
#     than TP=1's 0.92 because each rank also holds NCCL all-reduce buffers,
#     activation tensors at TP-shard sizes, and assorted bookkeeping. Going
#     higher risks per-rank OOM at peak concurrency.
#   - NO CUDA_VISIBLE_DEVICES export: vLLM TP needs to see all $VAST_TP_SIZE
#     GPUs. trial/invoke.py's CUDA_VISIBLE_DEVICES=1 (set in step 6b) only
#     applies to the simulation subprocess and is harmless here — the sim
#     subprocess just sees whichever GPU is at logical index 1 (one of the
#     ranks) and shares it with that rank. Heavy radio-map tasks under high
#     concurrency may trigger OOM there; if so, lower workers.

# Per-model vLLM flag selection — ported from bootstrap_vastai_multi.sh.
# Different families need different parsers / quantization / chat-template kwargs.
# Gemma family disables --enable-auto-tool-choice (vLLM grammar layer hangs).
# Thinking-mode Qwens need reasoning-parser + enable_thinking=false template arg.
EXTRA_VLLM_FLAGS=()
AUTO_TOOL_FLAGS=( --enable-auto-tool-choice )
case "$VAST_MODEL_HF" in
    *Qwen3-Coder-30B-A3B-Instruct*)
        # Non-thinking Qwen3-Coder: qwen3_coder parser, no reasoning-parser.
        EXTRA_VLLM_FLAGS+=( --tool-call-parser qwen3_coder )
        ;;
    *Qwen3.6-27B*|*Qwen3.6-35B-A3B*)
        # Qwen3.6 chat template emits XML format:
        #   <tool_call><function=NAME><parameter=name>VALUE</parameter></function></tool_call>
        # so the right parser is qwen3_xml (NOT hermes which expects JSON).
        # Thinking is disabled via chat-template kwargs (vLLM 0.20.0 syntax).
        EXTRA_VLLM_FLAGS+=( --tool-call-parser qwen3_xml \
                            --reasoning-parser qwen3 \
                            --default-chat-template-kwargs '{"enable_thinking":false}' )
        ;;
    *[Ll]lama-3.3-70[Bb]*-[Aa][Ww][Qq]*|*[Ll]lama-3.3-70[Bb]*-instruct-[Aa][Ww][Qq]*)
        EXTRA_VLLM_FLAGS+=( --tool-call-parser llama3_json --quantization awq_marlin )
        ;;
    *Gemma-4-31B-it*|*Gemma-4-26B-A4B-it*|*gemma-4-31[Bb]*|*gemma-4-26[Bb]*)
        # Gemma: drop --enable-auto-tool-choice and --tool-call-parser entirely
        # (vLLM grammar layer hangs Gemma per memory iteration 35).
        # Proxy parses Gemma's native markup instead.
        AUTO_TOOL_FLAGS=()
        ;;
    *)
        warn "no per-model vLLM flags for $VAST_MODEL_HF — falling back to VAST_MODEL_PARSER=$VAST_MODEL_PARSER"
        EXTRA_VLLM_FLAGS+=( --tool-call-parser "$VAST_MODEL_PARSER" )
        ;;
esac

# DFlash speculative decoding (https://github.com/z-lab/dflash). Opt-in via
# USE_DFLASH=1. Adds --speculative-config with the matching official draft
# from z-lab/. 5 of the 6 vast.ai loop models have published drafts; Llama
# 3.3-70B-AWQ has no published draft (only 8B-instruct), so DFlash is a
# silent no-op for it (warn + run baseline).
#
# vLLM build requirements (NOT installed by this script — the user must have
# the right vLLM in the active env):
#   - Qwens, Llama-3.1-8B: mainline vLLM or PR #40898
#   - Gemmas: PR #41703 (separate branch, separate env recommended)
# Helper: `bash benchmark/install_vllm_dflash.sh` (FAMILY=qwen|gemma) creates
# a parallel conda env without touching the baseline vLLM install.
DFLASH_DRAFT=""
if [[ "${USE_DFLASH:-0}" == "1" ]]; then
    case "$VAST_MODEL_HF" in
        *Qwen3.6-27B*)              DFLASH_DRAFT="z-lab/Qwen3.6-27B-DFlash" ;;
        *Qwen3.6-35B-A3B*)          DFLASH_DRAFT="z-lab/Qwen3.6-35B-A3B-DFlash" ;;
        *Qwen3-Coder-30B-A3B*)      DFLASH_DRAFT="z-lab/Qwen3-Coder-30B-A3B-DFlash" ;;
        *Gemma-4-31B-it*|*gemma-4-31[Bb]*)
                                    DFLASH_DRAFT="z-lab/gemma-4-31B-it-DFlash" ;;
        *Gemma-4-26B-A4B-it*|*gemma-4-26[Bb]*-A4B*)
                                    DFLASH_DRAFT="z-lab/gemma-4-26B-A4B-it-DFlash" ;;
        *[Ll]lama-3.1-8[Bb]*)       DFLASH_DRAFT="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat" ;;
        *)
            warn "USE_DFLASH=1 but no DFlash draft published for $VAST_MODEL_HF — running baseline"
            ;;
    esac
fi
if [[ -n "$DFLASH_DRAFT" ]]; then
    SPEC_TOKENS="${VAST_DFLASH_SPEC_TOKENS:-15}"
    EXTRA_VLLM_FLAGS+=( --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT\",\"num_speculative_tokens\":$SPEC_TOKENS}" )
    # Mainline DFlash recommends flash_attn for the speculative path.
    EXTRA_VLLM_FLAGS+=( --attention-backend flash_attn )
    echo "  DFlash: draft=$DFLASH_DRAFT spec_tokens=$SPEC_TOKENS"
fi

# Render flags into a string the heredoc can interpolate verbatim. Each flag
# already shell-safe (single-quoted JSON kept intact). printf %q would
# double-escape the JSON, so we trust the case-block's literal contents.
RENDERED_AUTO_TOOL=""
for f in "${AUTO_TOOL_FLAGS[@]}"; do
    RENDERED_AUTO_TOOL+=" $f"
done
RENDERED_EXTRA=""
for f in "${EXTRA_VLLM_FLAGS[@]}"; do
    # Re-quote args containing JSON / spaces so the generated script preserves them.
    if [[ "$f" == *"{"* || "$f" == *" "* ]]; then
        RENDERED_EXTRA+=" '$f'"
    else
        RENDERED_EXTRA+=" $f"
    fi
done

echo "  vllm flags: auto_tool=[${AUTO_TOOL_FLAGS[*]:-<empty>}] extra=[${EXTRA_VLLM_FLAGS[*]}]"

cat > /workspace/start_vllm.sh <<EOS
#!/bin/bash
# =====================================================================
# NCCL / distributed-execution env to fix the TP=2 shm_broadcast hang
# observed on vLLM 0.20.0 / CUDA 13.0 / NCCL 2.28.9 / 2× A100 SXM4 NVLink
# (memory iter 19 + repro 2026-05-06 on 108.231.141.46).
#
# Symptom we are mitigating:
#   [shm_broadcast.py:681] No available shared memory broadcast block
#   found in 60 seconds.
#
# Mechanism: vLLM's custom_all_reduce kernel + NCCL's cuMem allocator make
# assumptions about P2P / IOMMU / ACS that hold on bare metal but routinely
# break inside vast.ai containers (no BIOS access to disable ACS). The
# workers init NCCL fine, then deadlock at the first all-reduce shm
# rendezvous. Forcing NCCL for collectives (--disable-custom-all-reduce)
# and disabling cuMem (NCCL_CUMEM_ENABLE=0) is the documented recovery.
#
# Why we do NOT set NCCL_P2P_DISABLE=1: that would force traffic to PCIe
# and waste the NV12 NVLink (≈600 GB/s → ≈64 GB/s), killing TP=2's whole
# value proposition. NCCL_P2P_LEVEL=NVL keeps NVLink fast-path on while
# preventing fallback to non-existent same-host PCIe routes.
#
# Sources:
#   - vLLM forum: discuss.vllm.ai/t/.../2540 (Blackwell/PCIe hang fix)
#   - vLLM #30682, #27313, #8058 (recurring shm_broadcast hangs)
#   - NCCL troubleshooting: docs.nvidia.com/.../nccl/.../troubleshooting
# =====================================================================
export NCCL_DEBUG=WARN                         # log NCCL hangs without flooding
export NCCL_ASYNC_ERROR_HANDLING=1             # surface NCCL errors as exceptions
export NCCL_CUMEM_ENABLE=0                     # bypass cuMem allocator (root of shm_broadcast hang)
export NCCL_P2P_LEVEL=NVL                      # restrict P2P to NVLink (skip flaky PCIe paths)
export NCCL_SHM_DISABLE=0                      # keep shm transport (we have /dev/shm); explicit so vast.ai can't override
export VLLM_USE_TRITON_FLASH_ATTN=1            # standard fast-attn path
export VLLM_HOST_IP=127.0.0.1                  # pin to loopback (single-node TP); avoids NIC autodetect
exec vllm serve $MODEL_DIR \\
    --tensor-parallel-size $VAST_TP_SIZE \\
    --max-model-len $VAST_MODEL_MAX_LEN \\
    --max-num-seqs $VAST_MODEL_NUM_SEQS_TP \\
    --gpu-memory-utilization $VAST_GPU_MEM_UTIL_TP \\
    --enforce-eager \\
    --enable-prefix-caching \\
    --disable-custom-all-reduce \\
    --distributed-executor-backend mp \\
   $RENDERED_AUTO_TOOL \\
   $RENDERED_EXTRA \\
    --port 8001 \\
    --host 127.0.0.1 \\
    --served-model-name $BENCH_MODEL
EOS
chmod +x /workspace/start_vllm.sh
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "/workspace/start_vllm.sh 2>&1 | tee /workspace/logs/vllm.log"
ok "vLLM tmux session started (TP=$VAST_TP_SIZE, all $GPU_COUNT GPUs visible)"

# =====================================================================
step "12. Wait for vLLM ready (up to $((VAST_VLLM_WAIT_S / 60)) min — TP=$VAST_TP_SIZE init is slow + watch for NCCL hang)"
# TP init phases we expect to see in /workspace/logs/vllm.log:
#   1. Each rank loads its weight shard (~10-30s)
#   2. NCCL handshake / shm_broadcast / all-reduce warmup (the danger zone)
#   3. KV-cache profile run (single fwd pass with synthetic input)
#   4. Server starts listening on :8001
#
# The NCCL handshake is where TP=2 hung on the previous attempt. On vLLM
# 0.20.0 / CUDA 13 / NCCL 2.28.9 the failure mode is a stuck thread that
# eventually emits:
#   "shm_broadcast: No available shared memory broadcast block found in 60 seconds"
# We poll for that string every iteration and bail immediately if it appears
# instead of letting the 15-min timer run out.
VLLM_READY=0
NCCL_HANG_DETECTED=0
ITERS=$((VAST_VLLM_WAIT_S / 10))
for i in $(seq 1 $ITERS); do
    if curl -sf -m 3 -o /dev/null http://127.0.0.1:8001/v1/models; then
        ok "vLLM ready after $((i * 10))s (TP=$VAST_TP_SIZE)"
        VLLM_READY=1
        break
    fi

    # Watch for the known NCCL hang signature and bail fast. Real log signature
    # is "[shm_broadcast.py:681] No available shared memory broadcast block
    # found in 60 seconds" — the unique trailing string is what we match on.
    if [[ -f /workspace/logs/vllm.log ]] && \
       grep -q "No available shared memory broadcast block found in 60 seconds" \
            /workspace/logs/vllm.log 2>/dev/null; then
        NCCL_HANG_DETECTED=1
        break
    fi

    # Also bail on hard process death (vllm crashed instead of hanging).
    if ! tmux has-session -t vllm 2>/dev/null; then
        echo "  ! vLLM tmux session died; log tail:"
        tail -40 /workspace/logs/vllm.log 2>/dev/null || true
        err "vLLM process exited during startup. See /workspace/logs/vllm.log for the cause."
    fi

    sleep 10
done

if [[ "$NCCL_HANG_DETECTED" == "1" ]]; then
    echo "  ! vLLM log tail:"
    tail -30 /workspace/logs/vllm.log
    err "TP=$VAST_TP_SIZE hung at NCCL all-reduce init (saw 'shm_broadcast: No available shared memory broadcast block found in 60 seconds').
    Known issue on vLLM 0.20.0 / CUDA 13 / NCCL 2.28.9 (see docs/plan_vast_ai_benchmark.md §0 #8, §3, §8.3).
    NOTE: as of 2026-05-06 we ship NCCL_CUMEM_ENABLE=0 + NCCL_P2P_LEVEL=NVL +
    --disable-custom-all-reduce + --distributed-executor-backend mp by default.
    If we STILL hit shm_broadcast on top of that, escalate:
      (a) Try NCCL_P2P_DISABLE=1 (forces TCP, kills NVLink perf but unblocks).
      (b) Pin vLLM to 0.10.x (V0 engine) on a fresh rent — V1 shm rendezvous
          is the known fragile path.
      (c) Set VLLM_USE_V1=0 to fall back to V0 engine on this image.
      (d) Last resort: bash bootstrap_vastai_multi.sh (TP=1 × 2 instances)."
fi

if [[ "$VLLM_READY" != "1" ]]; then
    echo "  ! vLLM log tail:"
    tail -30 /workspace/logs/vllm.log
    err "vLLM not ready after $((VAST_VLLM_WAIT_S / 60)) min. TP=$VAST_TP_SIZE init exceeded the wait budget without the known shm_broadcast signature.
    Inspect /workspace/logs/vllm.log directly. If it shows progress (rank N loaded shard, etc.) but no listener,
    bump VAST_VLLM_WAIT_S and retry. If it shows NCCL/CUDA errors, fall back to bootstrap_vastai.sh."
fi

# =====================================================================
step "12.5. Launch tool-call repair proxy (port 8101 → vLLM 8001)"
# benchmark/tool_call_proxy.py sits between openclaude and the single TP vLLM,
# repairing small/Gemma model malformations:
#   1. Bash run_in_background "True"/"False" string → bool (memory iteration 2)
#   2. Inserts default content="" if Write omits content (memory iteration 1)
#   3. Parses Gemma's call:name{k:"v"} markup → OpenAI tool_calls (H.3)
#   4. Forces stream=False to avoid grammar-deadlock (sunlab cycles 11-13)
# The proxy forwards 1:1 to the TP vLLM:
#   openclaude → proxy port 8101 → vLLM port 8001
# tool_call_proxy.py is configured exclusively via CLI flags (--port, --host,
# --upstream). Optional env vars: PROXY_DEBUG_LOG, PROXY_MAX_TOKENS_CAP,
# PROXY_STRIP_TOOLS (only when --enable-auto-tool-choice is disabled).
# Deps: fastapi, httpx, uvicorn — vLLM already installs these as transitive
# deps, but pip-install defensively (idempotent: no-op if present).
pip install --quiet fastapi httpx uvicorn 2>/dev/null || \
    pip install fastapi httpx uvicorn || \
    warn "pip install fastapi/httpx/uvicorn failed — proxy may not start"

PROXY_LOG=/workspace/logs/proxy_0.log
tmux kill-session -t "proxy0" 2>/dev/null || true
tmux new-session -d -s "proxy0" \
    "python3 /workspace/skill/benchmark/tool_call_proxy.py \
        --host 127.0.0.1 \
        --port 8101 \
        --upstream http://127.0.0.1:8001 \
        2>&1 | tee $PROXY_LOG"
PROXY_READY=0
for attempt in $(seq 1 30); do  # 30 * 1s = 30s
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:8101/healthz"; then
        ok "proxy0 ready on port 8101 → vLLM 8001 (took ${attempt}s)"
        PROXY_READY=1
        break
    fi
    sleep 1
done
if [[ "$PROXY_READY" != "1" ]]; then
    echo "  ! proxy0 log tail:"
    tail -20 "$PROXY_LOG" 2>/dev/null || echo "  (no log yet)"
    err "tool-call proxy failed to start within 30s — see $PROXY_LOG"
fi

# =====================================================================
step "13. End-to-end smoke test (openclaude → vLLM, single instance at :8001)"
source /workspace/skill.env
SMOKE=$(timeout 60 /usr/local/bin/openclaude-vllm \
    --model "$BENCH_MODEL" \
    -p "Reply with the word OK and nothing else." \
    --print --max-turns 1 --allow-dangerously-skip-permissions 2>&1 || true)

# Check that openclaude got SOMETHING back (model output, not just stderr noise).
# Strip openclaude's own warnings (lines starting with "[context]" or "Warning:")
# then look for any non-empty line from the model. Thinking models often
# interpret the smoke prompt creatively, so a strict grep for a magic word
# rejects working chains.
SMOKE_BODY=$(echo "$SMOKE" | grep -vE "^\[context\]|^Warning:" | grep -vE "^\s*$" | head -20)
if [[ -n "$SMOKE_BODY" ]]; then
    ok "smoke test passed (model returned text: $(echo "$SMOKE_BODY" | head -1 | cut -c1-80))"
else
    echo "  ! smoke output (last 20 lines):"
    echo "$SMOKE" | tail -20
    err "smoke test failed — chain not working end-to-end. Check:
    - vLLM serving correctly (curl localhost:8001/v1/models)
    - tool-call parser '$VAST_MODEL_PARSER' correct for $VAST_MODEL_HF
    - openclaude max_tokens patch applied"
fi

# =====================================================================
step "13b. Real-trial verification: U001 + U136 (catches RAG + real-mesh issues)"
rm -rf /workspace/skill/benchmark/results/preflight_real
cd /workspace/skill

if [[ "$VAST_CATALOG_SRC" != "skip" ]]; then
    PREFLIGHT_TASKS="U001 U136"
else
    PREFLIGHT_TASKS="U001"
fi

set +e
( source /workspace/skill.env && \
  timeout 1200 python3 benchmark/run_benchmark.py \
    --label preflight_real \
    --workers 1 \
    --model "$BENCH_MODEL" \
    --conditions with_skill \
    --task-ids $PREFLIGHT_TASKS \
    --max-turns 25 \
    --timeout 400 \
) > /workspace/logs/preflight_real.log 2>&1
RC=$?
set -e

PROMPT_FILE=$(find /workspace/skill/benchmark/results/preflight_real -name prompt.txt -type f 2>/dev/null | head -1)
if [[ -z "$PROMPT_FILE" ]]; then
    tail -30 /workspace/logs/preflight_real.log
    err "real-trial verification: no prompt.txt produced"
fi

if grep -qE "RELATED (PRINCIPLES|MEMORY)" "$PROMPT_FILE"; then
    ok "RAG injection verified: prompt contains RELATED MEMORY block"
else
    if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
        echo "  ! prompt.txt did NOT contain RAG block. Checked: $PROMPT_FILE"
        err "RAG not being injected. Verify CLAUDE_CODE_USE_RAG=1 and trial/rag.py works."
    else
        warn "RAG not injected — benchmark will run without retrieval boost"
    fi
fi

if grep -q -E "Module 1|SKILL.md|rf-simulator" "$PROMPT_FILE"; then
    ok "skill content present in prompt (SKILL.md auto-loaded)"
else
    warn "skill content NOT detected — RF_SKILL_DIR may be misconfigured"
fi

if [[ "$VAST_CATALOG_SRC" != "skip" ]]; then
    U136_DIR=$(find /workspace/skill/benchmark/results/preflight_real -path "*U136*" -type d 2>/dev/null | head -1)
    if [[ -n "$U136_DIR" ]]; then
        GLB_FILES=$(find "$U136_DIR" -name "*.glb" -type f 2>/dev/null)
        if [[ -z "$GLB_FILES" ]]; then
            warn "U136 produced no .glb file — trial may not have completed scene export"
        else
            BIGGEST_GLB=$(echo "$GLB_FILES" | xargs -I{} stat -c "%s {}" 2>/dev/null | sort -rn | head -1)
            BIGGEST_SIZE=$(echo "$BIGGEST_GLB" | awk '{print $1}')
            BIGGEST_PATH=$(echo "$BIGGEST_GLB" | cut -d' ' -f2-)
            BIGGEST_SIZE=${BIGGEST_SIZE:-0}
            echo "  largest exported GLB: $BIGGEST_SIZE bytes ($BIGGEST_PATH)"
            if [[ "$BIGGEST_SIZE" -gt 500000 ]]; then
                ok "real meshes confirmed in U136 export (>500 KB GLB → 3D-FUTURE meshes used)"
            elif [[ "$BIGGEST_SIZE" -gt 100000 ]]; then
                warn "U136 GLB is $BIGGEST_SIZE B — borderline (could be small real or large fallback)"
            else
                if [[ "$VAST_REQUIRE_CATALOG" == "1" ]]; then
                    err "U136 GLB is only $BIGGEST_SIZE B — skill is using AABB box fallbacks, NOT real 3D-FUTURE meshes.
    The catalog file count check passed but the resolver isn't picking up real meshes at trial time.
    Likely cause: catalog layout doesn't match skill's get_mesh_path() expectations."
                else
                    warn "U136 GLB is small ($BIGGEST_SIZE B); benchmark using fallback boxes"
                fi
            fi
        fi
    else
        warn "U136 trial workdir not found — benchmark may have errored on this task"
    fi
fi

if grep -qE "Pass rate: [1-9][0-9]*/[1-9]" /workspace/logs/preflight_real.log; then
    ok "preflight trials: $(grep "Pass rate:" /workspace/logs/preflight_real.log | tail -1)"
else
    warn "preflight trials did not pass — see /workspace/logs/preflight_real.log
        (acceptable if model capability is the cause; chain is still verified by checks 1-3)"
fi

rm -rf /workspace/skill/benchmark/results/preflight_real

# =====================================================================
step "13c. Generate per-model self_gen SKILL.md (idempotent)"
# Each model self_gen condition uses a SKILL.md generated by THIS model,
# not the project's Sonnet baseline. Cached at
# /workspace/skill/benchmark/self_gen_skill/<model_basename>/SKILL.md;
# regenerate only if missing. ~3-8 min on 27B-class via vLLM (one-shot
# generation, no trial loop).
SELF_GEN_DIR=/workspace/skill/benchmark/self_gen_skill/$(basename "$VAST_MODEL_HF")
SELF_GEN_FILE=$SELF_GEN_DIR/SKILL.md
if [[ -f "$SELF_GEN_FILE" ]] && [[ "$(wc -c < "$SELF_GEN_FILE")" -gt 1000 ]]; then
    ok "self_gen skill already cached at $SELF_GEN_FILE ($(wc -c < "$SELF_GEN_FILE") bytes)"
else
    mkdir -p "$SELF_GEN_DIR"
    OPENAI_BASE_URL=http://127.0.0.1:8101/v1 \
    OPENAI_API_KEY=dummy-vllm \
    BENCHMARK_MODEL_ID=benchmark-model \
    timeout 600 python3 /workspace/skill/benchmark/generate_baseline_skill.py \
        --backend openai-compat \
        --out-file "$SELF_GEN_FILE" 2>&1 | tail -10
    if [[ ! -s "$SELF_GEN_FILE" ]]; then
        warn "self_gen generation produced empty file; using legacy single-file fallback"
        rm -f "$SELF_GEN_FILE"
    else
        ok "self_gen skill generated ($(wc -c < "$SELF_GEN_FILE") bytes)"
    fi
fi

# =====================================================================
if [[ "$VAST_AUTOLAUNCH_BENCHMARK" != "1" ]]; then
    echo ""
    echo "========================================="
    echo "Bootstrap COMPLETE — chain verified after $(($(date +%s) - T0))s."
    echo "Mode:               TP=$VAST_TP_SIZE single-instance"
    echo "max-num-seqs:       $VAST_MODEL_NUM_SEQS_TP"
    echo "Auto-launch disabled (VAST_AUTOLAUNCH_BENCHMARK=$VAST_AUTOLAUNCH_BENCHMARK)."
    echo ""
    echo "Manual launch command (point at proxy 8101 NOT vLLM 8001 — proxy repairs"
    echo "tool-call malformations before openclaude sees them):"
    echo "  source /workspace/skill.env && cd /workspace/skill && \\"
    echo "  OPENAI_BASE_URL=http://127.0.0.1:8101/v1 \\"
    echo "  CLAUDE_CODE_TOOLS_OVERRIDE=Bash,Read,Write,Edit \\"
    echo "  python3 benchmark/run_benchmark.py --label $VAST_BENCH_LABEL \\"
    echo "    --workers $VAST_BENCH_WORKERS --model $BENCH_MODEL \\"
    echo "    --conditions with_skill no_skill self_gen --max-turns 40 \\"
    echo "    --timeout 900 --retry-timeout 1800 --resume"
    echo ""
    echo "NOTE: --workers $VAST_BENCH_WORKERS exceeds sunlab-validated 6."
    echo "      Have --resume ready (already in the command above) — multiprocessing-pool"
    echo "      teardown bug is more likely at this concurrency. Completed trials are saved."
    echo "========================================="
    exit 0
fi

step "14. Launch benchmark in tmux session 'benchmark'"
# Per-model tool-set restriction. Without CLAUDE_CODE_TOOLS_OVERRIDE the full
# Claude Code default tool defs eat ~10K tokens per request. trial/invoke.py
# (lines 142-148) reads this env var and restricts the tool set accordingly.
# Gemma's tool-format parser is fragile, so it gets a smaller list.
case "$VAST_MODEL_HF" in
    *Qwen3.6-27B*|*Qwen3.6-35B-A3B*|*Qwen3-Coder-30B-A3B-Instruct*|*[Ll]lama-3.3-70[Bb]*-[Aa][Ww][Qq]*)
        BENCH_TOOLS_OVERRIDE="Bash,Read,Write,Edit"
        ;;
    *Gemma-4-31B-it*|*Gemma-4-26B-A4B-it*|*gemma-4-31[Bb]*|*gemma-4-26[Bb]*)
        BENCH_TOOLS_OVERRIDE="Bash,Read"
        ;;
    *)
        warn "no tools-override mapping for $VAST_MODEL_HF — defaulting to Bash,Read"
        BENCH_TOOLS_OVERRIDE="Bash,Read"
        ;;
esac
echo "  CLAUDE_CODE_TOOLS_OVERRIDE=$BENCH_TOOLS_OVERRIDE"

tmux kill-session -t benchmark 2>/dev/null || true
# Point bench at the tool-call proxy (port 8101) NOT the raw vLLM (port 8001).
# The proxy repairs known small-LLM tool-call malformations (Gemma markup,
# run_in_background string→bool, missing Write content, stream-deadlock) before
# openclaude sees them.
tmux new-session -d -s benchmark "source /workspace/skill.env && \
    export OPENAI_BASE_URL=http://127.0.0.1:8101/v1 && \
    export CLAUDE_CODE_TOOLS_OVERRIDE=$BENCH_TOOLS_OVERRIDE && \
    export CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS='{\"benchmark-model\": 32768}' && \
    export CLAUDE_CODE_OPENAI_MAX_OUTPUT_TOKENS='{\"benchmark-model\": 8000}' && \
    cd /workspace/skill && \
    python3 benchmark/run_benchmark.py \
        --label $VAST_BENCH_LABEL \
        --workers $VAST_BENCH_WORKERS \
        --model $BENCH_MODEL \
        --conditions with_skill no_skill self_gen \
        --max-turns 40 \
        --timeout 900 \
        --retry-timeout 1800 \
        --resume \
        2>&1 | tee /workspace/logs/benchmark.log"
ok "benchmark launched (TP=$VAST_TP_SIZE single-instance, OPENAI_BASE_URL=http://127.0.0.1:8101/v1 → vLLM 8001)"

# =====================================================================
TOTAL=$(($(date +%s) - T0))
echo ""
echo "========================================="
echo "Bootstrap COMPLETE in ${TOTAL}s — first trial firing now."
echo "Mode:        TP=$VAST_TP_SIZE single vLLM instance (port 8001)"
echo "Model:       $BENCH_MODEL ($VAST_MODEL_HF)"
echo "Parser:      $VAST_MODEL_PARSER"
echo "max-num-seqs:$VAST_MODEL_NUM_SEQS_TP"
echo "Workers:     $VAST_BENCH_WORKERS  (exceeds sunlab-validated 6 — see warning below)"
echo ""
echo "Monitor:     tail -f /workspace/logs/benchmark.log"
echo "vLLM log:    tail -f /workspace/logs/vllm.log"
echo "Tear down:   tmux kill-session -t benchmark"
echo "Boot log:    $LOG"
echo ""
echo "If the run truncates with a multiprocessing-pool teardown hang (known harness"
echo "bug at high concurrency), re-launch with --resume — completed trials are saved."
echo "Point at the proxy (8101) NOT the raw vLLM (8001) so tool-call repair stays active:"
echo "  source /workspace/skill.env && cd /workspace/skill && \\"
echo "  OPENAI_BASE_URL=http://127.0.0.1:8101/v1 \\"
echo "  CLAUDE_CODE_TOOLS_OVERRIDE=$BENCH_TOOLS_OVERRIDE \\"
echo "  python3 benchmark/run_benchmark.py --label $VAST_BENCH_LABEL \\"
echo "      --workers $VAST_BENCH_WORKERS --model $BENCH_MODEL \\"
echo "      --conditions with_skill no_skill self_gen --max-turns 40 \\"
echo "      --timeout 900 --retry-timeout 1800 --resume"
echo "========================================="
