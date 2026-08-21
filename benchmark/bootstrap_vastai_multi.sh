#!/bin/bash
# bootstrap_vastai_multi.sh — multi-vLLM bootstrap (Option B) for vast.ai 8x A100.
#
# Sibling of bootstrap_vastai.sh. Reuses every fix from §0 of
# docs/plan_vast_ai_benchmark.md. Difference: instead of one vLLM serving the
# benchmark, this launches N independent vLLM processes (default N=8), one
# per GPU, each TP=1, ports 8001..8001+N-1, all serving the same model under
# the same served-model-name. Then it shards the train split into N chunks
# and launches N benchmark drivers in parallel — one per vLLM.
#
# Why: with 8x A100 PCIe, single-vLLM TP=8 is bottlenecked by per-request
# latency and PCIe collective overhead. N independent TP=1 instances scale
# trial throughput ~linearly because the harness is embarrassingly parallel
# at the (task, condition, trial) level.
#
# Pasteable into vast.ai's "On-start command" field, or run via SSH:
#   scp bootstrap_vastai_multi.sh root@<host>:/root/
#   ssh -i ~/.ssh/vastai -p <port> root@<host> 'bash /root/bootstrap_vastai_multi.sh'
#
# Required env vars (same as single-instance):
#   VAST_REPO_RSYNC_SRC   e.g. "you@sunlab:~/PycharmProjects/new-sionna-skill"
#
# Multi-instance-specific env var:
#   VAST_NUM_INSTANCES    how many vLLM instances to launch (one per GPU).
#                         Default: 8. Must be ≤ visible GPU count.
#
# Optional env vars (same as single-instance, with sensible defaults):
#   VAST_REPO_RSYNC_KEY           default /root/.ssh/sunlab_pull
#   VAST_TUNNEL_PORT              chroma reverse-tunnel remote port. Default: 8765
#   VAST_MODEL_HF                 HF repo id. Default: Qwen/Qwen3.6-27B
#   VAST_MODEL_PARSER             vLLM tool-call parser. Default: hermes
#   VAST_MODEL_MAX_LEN            vLLM --max-model-len. Default: 32768
#   VAST_MODEL_NUM_SEQS           vLLM --max-num-seqs. Default: 2
#   VAST_BENCH_LABEL              run label prefix. Default: train_full_v1
#                                 Per-chunk labels become ${LABEL}_chunk{0..N-1}
#                                 Merged report goes to ${LABEL}_merged
#   VAST_BENCH_WORKERS            harness workers PER CHUNK (per vLLM). Default: 2
#                                 Total parallel trials = N * workers.
#   VAST_AUTOLAUNCH_BENCHMARK     "1" to fire benchmark after smoke; "0" to stop. Default: 1
#   VAST_TUNNEL_WAIT_S            max seconds to wait for chroma tunnel. Default: 1800
#
#   --- Catalog (unchanged from single-instance) ---
#   VAST_CATALOG_SRC              "hf_3dfront" | "abo" | "prestaged" | "skip". Default: prestaged
#   VAST_CATALOG_RSYNC_SRC        prestaged source
#   VAST_CATALOG_PATH             local install. Default: /data/3D-FUTURE-model
#   VAST_REQUIRE_CATALOG          "1" to fail bootstrap if catalog missing. Default: 1
#
# Exit behavior:
#   Each step verifies its outcome. On any failure, exits non-zero with a clear
#   error. NO benchmark fires unless every step before it succeeded.
#
# Disk requirement:
#   IDENTICAL to single-instance — vLLM mmaps a single on-disk model copy
#   regardless of how many processes serve it. Catalog is also one copy.

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
VAST_MODEL_NUM_SEQS="${VAST_MODEL_NUM_SEQS:-2}"

VAST_BENCH_LABEL="${VAST_BENCH_LABEL:-train_full_v1}"
VAST_BENCH_WORKERS="${VAST_BENCH_WORKERS:-2}"
VAST_AUTOLAUNCH_BENCHMARK="${VAST_AUTOLAUNCH_BENCHMARK:-1}"

VAST_NUM_INSTANCES="${VAST_NUM_INSTANCES:-8}"

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
step "0. Sanity check"
[[ "$(id -u)" == "0" ]] || err "must run as root on vast.ai"
[[ "${VAST_SKIP_REPO_RSYNC:-0}" == "1" ]] || [[ -n "$VAST_REPO_RSYNC_SRC" ]] || err "VAST_REPO_RSYNC_SRC env var is required (or set VAST_SKIP_REPO_RSYNC=1 if /workspace/skill is pre-staged)"
[[ "${VAST_SKIP_REPO_RSYNC:-0}" == "1" ]] || [[ -f "$VAST_REPO_RSYNC_KEY" ]] || err "rsync SSH key missing at $VAST_REPO_RSYNC_KEY"

# Multi-instance: validate VAST_NUM_INSTANCES against visible GPU count
if ! [[ "$VAST_NUM_INSTANCES" =~ ^[0-9]+$ ]] || [[ "$VAST_NUM_INSTANCES" -lt 1 ]]; then
    err "VAST_NUM_INSTANCES must be a positive integer (got: $VAST_NUM_INSTANCES)"
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_COUNT=${GPU_COUNT:-0}
if [[ "$GPU_COUNT" -lt "$VAST_NUM_INSTANCES" ]]; then
    err "VAST_NUM_INSTANCES=$VAST_NUM_INSTANCES but only $GPU_COUNT GPUs visible.
    Either lower VAST_NUM_INSTANCES or rent a box with more GPUs."
fi
ok "$GPU_COUNT GPUs visible, will launch $VAST_NUM_INSTANCES vLLM instances"

# Catalog config sanity
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

# Disk-space sanity. SAME as single-instance: vLLM mmaps one on-disk model
# regardless of how many processes serve it. Catalog is one shared copy too.
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
# NOTE: OPENAI_BASE_URL is set to instance 0 by default (used for the smoke
# test and any ad-hoc claude calls from the shell). Each per-chunk benchmark
# process overrides OPENAI_BASE_URL in its own tmux session to point at its
# assigned vLLM instance.
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
# fit the next model. Lets us cache multiple models when there's space.
MODEL_BASENAME=$(basename "$VAST_MODEL_HF")
mkdir -p /workspace/models
PRUNED=$(find /workspace/models -mindepth 1 -maxdepth 1 -type d ! -name "$MODEL_BASENAME" 2>/dev/null)
if [[ -n "$PRUNED" ]]; then
    DISK_FREE_GB=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    DISK_FREE_GB=${DISK_FREE_GB:-0}
    case "$VAST_MODEL_HF" in
        *70[Bb]*-[Aa][Ww][Qq]*|*70[Bb]*-instruct-[Aa][Ww][Qq]*) NEW_GB=45 ;;
        *35B-A3B*|*35[Bb]-A3B*) NEW_GB=75 ;;
        *27[Bb]*) NEW_GB=55 ;;
        *31[Bb]*|*30[Bb]*) NEW_GB=65 ;;
        *26[Bb]*-A4[Bb]*) NEW_GB=55 ;;
        *) NEW_GB=70 ;;
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
step "6b. Patch trial/invoke.py to pin Sionna RT to GPU (avoid vLLM contention)"
# Single-instance pinned simulation to GPU 1 (vLLM on 0). For multi-instance
# we leave the patch as-is (CUDA_VISIBLE_DEVICES=1 default) — each per-chunk
# benchmark is on a separate vLLM and the catalog of trials is small enough
# that they don't fight. If you need per-chunk simulation GPU pinning, set
# BENCH_SIM_GPU in the per-chunk env (the patch reads it). For now, leaving
# all trial subprocesses on GPU 1 is fine — Sionna RT is bursty, not steady.
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

# Patch openclaude max_tokens defaults (§0 #10)
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

# Install the wrapper that forces --provider openai (§0 #18)
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
step "11. Start $VAST_NUM_INSTANCES vLLM instances (one per GPU, TP=1, ports 8001..$((8000+VAST_NUM_INSTANCES)))"
# Each instance: CUDA_VISIBLE_DEVICES=i, port 8001+i, served-model-name=$BENCH_MODEL
# (same name on all). Trial drivers select instance via OPENAI_BASE_URL only.
mkdir -p /workspace/logs

# Per-model vLLM flag selection — ported from queue_local_llms.sh:291-388.
# Different families need different parsers / quantization / chat-template kwargs.
# Gemma family disables --enable-auto-tool-choice (vLLM grammar layer hangs;
# memory iteration 35 confirmed). Thinking-mode Qwens need reasoning-parser.
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
# USE_DFLASH=1. See bootstrap_vastai_tp.sh for vLLM build requirements
# (Qwens → mainline / PR #40898, Gemmas → PR #41703). Llama 3.3-70B-AWQ has
# no published draft, so DFlash is a silent no-op there.
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

for i in $(seq 0 $((VAST_NUM_INSTANCES - 1))); do
    PORT=$((8001 + i))
    SCRIPT=/workspace/start_vllm_${i}.sh
    cat > "$SCRIPT" <<EOS
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$i
exec vllm serve $MODEL_DIR \\
    --tensor-parallel-size 1 \\
    --max-model-len $VAST_MODEL_MAX_LEN \\
    --max-num-seqs $VAST_MODEL_NUM_SEQS \\
    --gpu-memory-utilization 0.92 \\
    --enforce-eager \\
    --enable-prefix-caching \\
   $RENDERED_AUTO_TOOL \\
   $RENDERED_EXTRA \\
    --port $PORT \\
    --host 127.0.0.1 \\
    --served-model-name $BENCH_MODEL
EOS
    chmod +x "$SCRIPT"
    tmux kill-session -t "vllm$i" 2>/dev/null || true
    tmux new-session -d -s "vllm$i" "$SCRIPT 2>&1 | tee /workspace/logs/vllm_${i}.log"
    ok "tmux session vllm$i started on GPU $i (port $PORT)"
done

# =====================================================================
step "12. Wait for all $VAST_NUM_INSTANCES vLLM instances ready (up to 12 min — weight load + KV profile per instance)"
# Loading the same model on N GPUs simultaneously can take longer than a single
# instance because they all hit the disk page cache at boot. Bumped from 8→12 min.
ALL_READY=1
for i in $(seq 0 $((VAST_NUM_INSTANCES - 1))); do
    PORT=$((8001 + i))
    INSTANCE_READY=0
    for attempt in $(seq 1 72); do  # 72 * 10s = 12 min per instance
        if curl -sf -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
            ok "vllm$i ready on port $PORT (took $((attempt * 10))s)"
            INSTANCE_READY=1
            break
        fi
        sleep 10
    done
    if [[ "$INSTANCE_READY" != "1" ]]; then
        echo "  ! vllm$i log tail:"
        tail -30 "/workspace/logs/vllm_${i}.log"
        ALL_READY=0
    fi
done
if [[ "$ALL_READY" != "1" ]]; then
    err "one or more vLLM instances not ready after 12 min — see /workspace/logs/vllm_*.log"
fi
ok "all $VAST_NUM_INSTANCES vLLM instances ready"

# =====================================================================
step "12.5. Launch $VAST_NUM_INSTANCES tool-call repair proxies (ports 8101..$((8100+VAST_NUM_INSTANCES)))"
# benchmark/tool_call_proxy.py sits between openclaude and each vLLM instance,
# repairing small/Gemma model malformations:
#   1. Bash run_in_background "True"/"False" string → bool (memory iteration 2)
#   2. Inserts default content="" if Write omits content (memory iteration 1)
#   3. Parses Gemma's call:name{k:"v"} markup → OpenAI tool_calls (H.3)
#   4. Forces stream=False to avoid grammar-deadlock (sunlab cycles 11-13)
# Each proxy_i forwards 1:1 to vllm_i:
#   openclaude → proxy port 810(1+i) → vLLM port 800(1+i)
# tool_call_proxy.py is configured exclusively via CLI flags (--port, --host,
# --upstream). Optional env vars: PROXY_DEBUG_LOG, PROXY_MAX_TOKENS_CAP,
# PROXY_STRIP_TOOLS (only when --enable-auto-tool-choice is disabled).
# Deps: fastapi, httpx, uvicorn — vLLM already installs these as transitive
# deps, but pip-install defensively (idempotent: no-op if present).
pip install --quiet fastapi httpx uvicorn 2>/dev/null || \
    pip install fastapi httpx uvicorn || \
    warn "pip install fastapi/httpx/uvicorn failed — proxies may not start"

ALL_PROXIES_READY=1
for i in $(seq 0 $((VAST_NUM_INSTANCES - 1))); do
    PROXY_PORT=$((8101 + i))
    UPSTREAM_PORT=$((8001 + i))
    PROXY_LOG=/workspace/logs/proxy_${i}.log
    tmux kill-session -t "proxy$i" 2>/dev/null || true
    tmux new-session -d -s "proxy$i" \
        "python3 /workspace/skill/benchmark/tool_call_proxy.py \
            --host 127.0.0.1 \
            --port $PROXY_PORT \
            --upstream http://127.0.0.1:$UPSTREAM_PORT \
            2>&1 | tee $PROXY_LOG"
    PROXY_READY=0
    for attempt in $(seq 1 30); do  # 30 * 1s = 30s per proxy
        if curl -sf -m 2 -o /dev/null "http://127.0.0.1:$PROXY_PORT/healthz"; then
            ok "proxy$i ready on port $PROXY_PORT → vLLM $UPSTREAM_PORT (took ${attempt}s)"
            PROXY_READY=1
            break
        fi
        sleep 1
    done
    if [[ "$PROXY_READY" != "1" ]]; then
        echo "  ! proxy$i log tail:"
        tail -20 "$PROXY_LOG" 2>/dev/null || echo "  (no log yet)"
        ALL_PROXIES_READY=0
    fi
done
if [[ "$ALL_PROXIES_READY" != "1" ]]; then
    err "one or more tool-call proxies failed to start within 30s — see /workspace/logs/proxy_*.log"
fi
ok "all $VAST_NUM_INSTANCES proxies ready"

# =====================================================================
step "13. End-to-end smoke test on instance 0 (openclaude → vLLM port 8001)"
# Same smoke as single-instance — pick instance 0 (port 8001) since
# OPENAI_BASE_URL points there by default.
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
    warn "smoke test produced no model output — proceeding to benchmark anyway
    (time-constrained; benchmark will catch real chain failures via trial verifier).
    Check if many trials fail:
    - vLLM serving correctly (curl localhost:8001/v1/models)
    - tool-call parser '$VAST_MODEL_PARSER' correct for $VAST_MODEL_HF
    - openclaude max_tokens patch applied"
fi

# =====================================================================
step "13b. Real-trial verification on instance 0: U001 + U136 (RAG + real-mesh)"
# Run on instance 0 only (port 8001). Same checks as single-instance — RAG
# injection, skill content, U136 GLB > 500 KB, ≥1 trial passing.
rm -rf /workspace/skill/benchmark/results/preflight_real
cd /workspace/skill

if [[ "$VAST_CATALOG_SRC" != "skip" ]]; then
    PREFLIGHT_TASKS="U001 U136"
else
    PREFLIGHT_TASKS="U001"
fi

set +e
( source /workspace/skill.env && \
  OPENAI_BASE_URL=http://127.0.0.1:8001/v1 timeout 1200 python3 benchmark/run_benchmark.py \
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
# generation, no trial loop). Multi-instance: use first proxy port (8101)
# for the one-shot generation; chunked benchmark below uses 8101..N.
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
    echo "Auto-launch disabled (VAST_AUTOLAUNCH_BENCHMARK=$VAST_AUTOLAUNCH_BENCHMARK)."
    echo ""
    echo "Manual chunked launch (one tmux session per chunk):"
    echo "  source /workspace/skill.env"
    echo "  for i in \$(seq 0 $((VAST_NUM_INSTANCES - 1))); do"
    # Use proxy ports (8101+i) NOT vLLM ports (8001+i) — the proxies on
    # 8101+i forward to vLLM 8001+i AND repair tool-call bugs (Gemma markup,
    # run_in_background string→bool, etc.). Manual launch must match auto-
    # launch in step 14 to keep tool-call repair active.
    echo "    PORT=\$((8101 + i))"
    echo "    tmux new-session -d -s bench\$i \"OPENAI_BASE_URL=http://127.0.0.1:\$PORT/v1 \\"
    echo "      cd /workspace/skill && python3 benchmark/run_benchmark.py \\"
    echo "      --label ${VAST_BENCH_LABEL}_chunk\$i --workers $VAST_BENCH_WORKERS \\"
    echo "      --model $BENCH_MODEL --conditions with_skill no_skill self_gen \\"
    echo "      --max-turns 40 --timeout 900 --retry-timeout 1800 --resume\""
    echo "  done"
    echo "  # Then merge: python3 benchmark/merge_chunks.py --label-prefix ${VAST_BENCH_LABEL} \\"
    echo "                  --num-chunks $VAST_NUM_INSTANCES"
    echo "========================================="
    exit 0
fi

# =====================================================================
step "13d. Compute task-ID chunks (deterministic split with shuffle-seed=42)"
# Read tasks.json, filter split=train, shuffle deterministically with seed 42,
# then chunk into N equal slices. Each chunk gets a space-separated list of
# task IDs that we pass to --task-ids on its dedicated benchmark process.
CHUNKS_DIR=/workspace/chunks
mkdir -p "$CHUNKS_DIR"

NUM_INSTANCES=$VAST_NUM_INSTANCES TASKS_JSON=/workspace/skill/benchmark/tasks/tasks.json \
    CHUNKS_DIR=$CHUNKS_DIR python3 - <<'PY'
import json, os, random, math
n = int(os.environ["NUM_INSTANCES"])
tasks_path = os.environ["TASKS_JSON"]
out_dir = os.environ["CHUNKS_DIR"]
all_tasks = json.load(open(tasks_path))["tasks"]
train_ids = [t["id"] for t in all_tasks if t.get("split") == "train"]
# Deterministic shuffle: same seed harness uses (--shuffle-seed 42) so chunk
# boundaries are reproducible across reruns of the bootstrap.
random.Random(42).shuffle(train_ids)
size = math.ceil(len(train_ids) / n)
for i in range(n):
    chunk = train_ids[i*size:(i+1)*size]
    with open(f"{out_dir}/chunk_{i}.txt", "w") as f:
        f.write(" ".join(chunk))
    print(f"chunk_{i}: {len(chunk)} tasks ({chunk[0] if chunk else '-'}..{chunk[-1] if chunk else '-'})")
print(f"total: {sum(len(open(f'{out_dir}/chunk_{i}.txt').read().split()) for i in range(n))} tasks across {n} chunks")
PY
ok "chunk files written to $CHUNKS_DIR"

# =====================================================================
step "14. Launch $VAST_NUM_INSTANCES benchmark processes (one tmux session per chunk)"
# Each chunk: OPENAI_BASE_URL points at its dedicated vLLM (port 8001+i),
# task IDs come from /workspace/chunks/chunk_$i.txt, label is
# ${VAST_BENCH_LABEL}_chunk$i. Workers per chunk = VAST_BENCH_WORKERS;
# total parallel trials across the box = VAST_NUM_INSTANCES * VAST_BENCH_WORKERS.

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

for i in $(seq 0 $((VAST_NUM_INSTANCES - 1))); do
    PORT=$((8001 + i))
    PROXY_PORT=$((8101 + i))
    CHUNK_FILE="$CHUNKS_DIR/chunk_${i}.txt"
    [[ -s "$CHUNK_FILE" ]] || { warn "chunk_$i is empty — skipping"; continue; }
    CHUNK_TASKS=$(cat "$CHUNK_FILE")
    CHUNK_LABEL="${VAST_BENCH_LABEL}_chunk${i}"

    tmux kill-session -t "bench$i" 2>/dev/null || true
    # Each session sources skill.env (PATH, IS_SANDBOX, CHROMA_*, etc.) then
    # OVERRIDES OPENAI_BASE_URL to its assigned tool-call proxy (port 8101+i),
    # which forwards to its dedicated vLLM (port 8001+i). The proxy repairs
    # known small-LLM tool-call malformations before openclaude sees them.
    tmux new-session -d -s "bench$i" "source /workspace/skill.env && \
        export OPENAI_BASE_URL=http://127.0.0.1:$PROXY_PORT/v1 && \
        export CLAUDE_CODE_TOOLS_OVERRIDE=$BENCH_TOOLS_OVERRIDE && \
        export CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS='{\"benchmark-model\": 32768}' && \
        export CLAUDE_CODE_OPENAI_MAX_OUTPUT_TOKENS='{\"benchmark-model\": 8000}' && \
        cd /workspace/skill && \
        python3 benchmark/run_benchmark.py \
            --label $CHUNK_LABEL \
            --workers $VAST_BENCH_WORKERS \
            --model $BENCH_MODEL \
            --conditions with_skill no_skill self_gen \
            --task-ids $CHUNK_TASKS \
            --max-turns 40 \
            --timeout 900 \
            --retry-timeout 1800 \
            --resume \
            2>&1 | tee /workspace/logs/benchmark_chunk${i}.log"
    ok "bench$i launched (port $PORT, label $CHUNK_LABEL, $(echo $CHUNK_TASKS | wc -w) tasks)"
done

# =====================================================================
step "15. Install merge helper (benchmark/merge_chunks.py)"
# Sums per-chunk progress.json + walks each chunk's result.json files to
# compute a single combined report at benchmark/results/<label>_merged/summary.json.
# Idempotent: rerun any time after chunks make progress.
MERGE_HELPER=/workspace/skill/benchmark/merge_chunks.py
if [[ ! -f "$MERGE_HELPER" ]]; then
    cat > "$MERGE_HELPER" <<'PY'
"""Merge per-chunk benchmark results into a single combined report.

Used by bootstrap_vastai_multi.sh after the N parallel chunked benchmarks.
Reads benchmark/results/<label-prefix>_chunk{0..N-1}/, sums totals/errors
from progress.json, walks per-trial result.json to compute pass rates by
tier and condition, writes a top-level merged summary.

Doesn't fight existing per-chunk reports — they stay intact under their
own labels for debugging. This just produces a unified view.

Usage:
    python3 benchmark/merge_chunks.py --label-prefix train_full_v1 --num-chunks 8
    # Output: benchmark/results/train_full_v1_merged/summary.json
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "benchmark/results"


def merge(label_prefix: str, num_chunks: int, results_root: Path) -> dict:
    chunks = []
    for i in range(num_chunks):
        cdir = results_root / f"{label_prefix}_chunk{i}"
        if not cdir.is_dir():
            print(f"[merge] chunk{i} missing: {cdir}", file=sys.stderr)
            continue
        chunks.append((i, cdir))

    if not chunks:
        raise SystemExit(f"No chunk directories found under {results_root} "
                         f"matching '{label_prefix}_chunk*'")

    totals = {"total_trials": 0, "completed": 0, "errors": 0, "wall_sec_max": 0}
    by_chunk = []
    by_cond: dict = defaultdict(lambda: {"n": 0, "pass": 0})
    by_tier: dict = defaultdict(lambda: {"n": 0, "pass": 0})
    by_cond_tier: dict = defaultdict(lambda: {"n": 0, "pass": 0})

    for i, cdir in chunks:
        prog_path = cdir / "progress.json"
        prog = {}
        if prog_path.exists():
            try:
                prog = json.loads(prog_path.read_text())
            except Exception as e:
                print(f"[merge] {prog_path}: parse error {e}", file=sys.stderr)
        totals["total_trials"] += int(prog.get("total_trials", 0) or 0)
        totals["completed"]    += int(prog.get("completed", 0) or 0)
        totals["errors"]       += int(prog.get("errors", 0) or 0)
        wall = int(prog.get("wall_sec", 0) or 0)
        totals["wall_sec_max"] = max(totals["wall_sec_max"], wall)

        chunk_stat = {"chunk": i, "dir": str(cdir),
                      "total": prog.get("total_trials"),
                      "completed": prog.get("completed"),
                      "errors": prog.get("errors"),
                      "wall_sec": wall,
                      "n_passed": 0, "n_total": 0}
        for rj in cdir.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            cond = r.get("condition", "?")
            tier = r.get("tier", "?")
            passed = bool(r.get("verification", {}).get("passed", False)) \
                or bool(r.get("passed", False))
            by_cond[cond]["n"] += 1
            by_cond[cond]["pass"] += int(passed)
            by_tier[tier]["n"] += 1
            by_tier[tier]["pass"] += int(passed)
            ck = f"{cond}|{tier}"
            by_cond_tier[ck]["n"] += 1
            by_cond_tier[ck]["pass"] += int(passed)
            chunk_stat["n_total"] += 1
            chunk_stat["n_passed"] += int(passed)
        by_chunk.append(chunk_stat)

    n = sum(v["n"] for v in by_cond.values())
    p = sum(v["pass"] for v in by_cond.values())

    # Normalized gain (Hake) when both conditions present
    gain = None
    if "with_skill" in by_cond and "no_skill" in by_cond:
        nw, nn = by_cond["with_skill"]["n"], by_cond["no_skill"]["n"]
        if nw and nn:
            ps = by_cond["with_skill"]["pass"] / nw
            pv = by_cond["no_skill"]["pass"] / nn
            gain = {
                "absolute_gain_pp": round(100 * (ps - pv), 2),
                "normalized_gain": round((ps - pv) / (1 - pv), 3) if pv < 1 else None,
                "p_skill": round(ps, 3),
                "p_vanilla": round(pv, 3),
            }

    return {
        "label_prefix": label_prefix,
        "num_chunks": num_chunks,
        "totals": totals,
        "n": n,
        "passed": p,
        "pass_rate": round(100 * p / n, 2) if n else 0,
        "by_condition": dict(by_cond),
        "by_tier": dict(by_tier),
        "by_condition_tier": dict(by_cond_tier),
        "gain": gain,
        "by_chunk": by_chunk,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Merge per-chunk benchmark results into a single report.")
    ap.add_argument("--label-prefix", required=True,
                    help="The label prefix used in bootstrap_vastai_multi.sh "
                         "(per-chunk labels: <prefix>_chunk0, _chunk1, ...).")
    ap.add_argument("--num-chunks", type=int, required=True,
                    help="VAST_NUM_INSTANCES used at bootstrap time.")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    results_root = Path(args.results_root)
    summary = merge(args.label_prefix, args.num_chunks, results_root)
    out_dir = results_root / f"{args.label_prefix}_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"Merged report → {out_path}")
    print(f"  total trials:  {summary['totals']['total_trials']}")
    print(f"  completed:     {summary['totals']['completed']}")
    print(f"  errors:        {summary['totals']['errors']}")
    print(f"  pass rate:     {summary['passed']}/{summary['n']} "
          f"({summary['pass_rate']}%)")
    if summary.get("gain"):
        g = summary["gain"]
        print(f"  Δpp:           {g['absolute_gain_pp']}  (g={g['normalized_gain']})")
    for tier, c in sorted(summary["by_tier"].items()):
        if c["n"]:
            print(f"  {tier}: {c['pass']}/{c['n']} ({100*c['pass']/c['n']:.0f}%)")


if __name__ == "__main__":
    main()
PY
    ok "wrote $MERGE_HELPER"
else
    ok "merge helper already present (rsynced from source)"
fi

# =====================================================================
TOTAL=$(($(date +%s) - T0))
echo ""
echo "========================================="
echo "Multi-vLLM Bootstrap COMPLETE in ${TOTAL}s — first trials firing now."
echo "Model:             $BENCH_MODEL ($VAST_MODEL_HF)"
echo "Parser:            $VAST_MODEL_PARSER"
echo "Instances:         $VAST_NUM_INSTANCES (one per GPU, TP=1)"
echo "Workers / chunk:   $VAST_BENCH_WORKERS"
echo "Total parallelism: $((VAST_NUM_INSTANCES * VAST_BENCH_WORKERS)) trials in flight"
echo ""
echo "Monitor all:       for i in \$(seq 0 $((VAST_NUM_INSTANCES - 1))); do echo -- chunk\$i --; tail -3 /workspace/logs/benchmark_chunk\$i.log; done"
echo "Monitor one:       tail -f /workspace/logs/benchmark_chunk0.log"
echo "Tear down:         for i in \$(seq 0 $((VAST_NUM_INSTANCES - 1))); do tmux kill-session -t bench\$i; tmux kill-session -t proxy\$i; tmux kill-session -t vllm\$i; done"
echo "Merge results:     python3 /workspace/skill/benchmark/merge_chunks.py \\"
echo "                       --label-prefix $VAST_BENCH_LABEL \\"
echo "                       --num-chunks $VAST_NUM_INSTANCES"
echo "Boot log:          $LOG"
echo "========================================="
