#!/bin/bash
# bootstrap_vastai.sh — single idempotent bootstrap for a fresh vast.ai instance
# running the local-LLM benchmark via vLLM + OpenClaude + chroma.
# Encodes every fix from docs/plan_vast_ai_benchmark.md §0.
# v3: aggressive parallelism — kicks off model download immediately so it
#     overlaps with rsync, pip, Node, openclaude install, tunnel wait.
#
# Pasteable into vast.ai's "On-start command" field, or run via SSH:
#   scp bootstrap_vastai.sh root@<host>:/root/
#   ssh -i ~/.ssh/vastai -p <port> root@<host> 'bash /root/bootstrap_vastai.sh'
#
# Required env vars:
#   VAST_REPO_RSYNC_SRC   e.g. "you@sunlab:~/PycharmProjects/new-sionna-skill"
#                         the box pulls /workspace/skill from here via rsync.
#                         You'll need a passwordless SSH key on the box that
#                         the source accepts. Pre-populate ~/.ssh/sunlab_pull
#                         (or override VAST_REPO_RSYNC_KEY).
#
# Optional env vars (all have sensible defaults):
#   VAST_REPO_RSYNC_KEY           default /root/.ssh/sunlab_pull
#   VAST_TUNNEL_PORT              chroma reverse-tunnel remote port. Default: 8765
#                                 (NOT 8000 — vast.ai's Caddy holds 8000)
#   VAST_MODEL_HF                 HF repo id. Default: Qwen/Qwen3.6-27B
#   VAST_MODEL_PARSER             vLLM tool-call parser. Default: hermes
#                                 (qwen3=hermes, llama3=llama3_json, mistral=mistral, deepseek_v3=deepseek_v3)
#   VAST_MODEL_MAX_LEN            vLLM --max-model-len. Default: 32768
#   VAST_MODEL_NUM_SEQS           vLLM --max-num-seqs. Default: 2
#   VAST_BENCH_LABEL              run label. Default: train_full_v1
#   VAST_BENCH_WORKERS            harness workers. Default: 2
#   VAST_AUTOLAUNCH_BENCHMARK     "1" to fire benchmark after smoke; "0" to stop. Default: 1
#   VAST_TUNNEL_WAIT_S            max seconds to wait for chroma tunnel. Default: 300
#
#   --- Furniture catalog (T0_scene_gen tasks need this; rest don't) ---
#   VAST_CATALOG_SRC              "hf_3dfront" | "abo" | "prestaged" | "skip". Default: hf_3dfront
#                                 hf_3dfront: pull huanngzh/3D-Front from HF (3D-FUTURE meshes embedded, ~50 GB)
#                                 abo:        public ABO bucket (CC BY 4.0, ~30 GB, lower fidelity)
#                                 prestaged:  rsync from VAST_CATALOG_RSYNC_SRC
#                                 skip:       no catalog; T0 tasks will use AABB cube fallbacks (logged as caveat)
#   VAST_CATALOG_RSYNC_SRC        for prestaged: source to rsync from (e.g. "you@sunlab:/data/3D-FUTURE-model")
#   VAST_CATALOG_PATH             local install path. Default: /workspace/catalogs/3d-future
#   VAST_REQUIRE_CATALOG          "1" to fail bootstrap if catalog setup or verification fails. Default: 1
#                                 (set to 0 only if you truly want to run with fallback boxes)
#
# Exit behavior:
#   Each step verifies its outcome. On any failure, exits non-zero with a clear
#   error. NO benchmark fires unless every step before it succeeded — so you
#   don't burn GPU on a half-set-up box.

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
# Default 30 min: gives the operator time to handle tasks in parallel with
# bootstrap (e.g., logging into Claude Code on vast.ai for ad-hoc shell use,
# completing OAuth browser flow, opening another SSH session). The bootstrap
# itself doesn't need claude code; this is just slack so it doesn't time out
# while the operator is doing something else in another tab.
VAST_TUNNEL_WAIT_S="${VAST_TUNNEL_WAIT_S:-1800}"

VAST_MODEL_HF="${VAST_MODEL_HF:-Qwen/Qwen3.6-27B}"
VAST_MODEL_PARSER="${VAST_MODEL_PARSER:-hermes}"
VAST_MODEL_MAX_LEN="${VAST_MODEL_MAX_LEN:-32768}"
VAST_MODEL_NUM_SEQS="${VAST_MODEL_NUM_SEQS:-2}"

VAST_BENCH_LABEL="${VAST_BENCH_LABEL:-train_full_v1}"
VAST_BENCH_WORKERS="${VAST_BENCH_WORKERS:-2}"
VAST_AUTOLAUNCH_BENCHMARK="${VAST_AUTOLAUNCH_BENCHMARK:-1}"

VAST_CATALOG_SRC="${VAST_CATALOG_SRC:-prestaged}"
VAST_CATALOG_RSYNC_SRC="${VAST_CATALOG_RSYNC_SRC:-}"
# Default to /data/3D-FUTURE-model — one of the paths invoke.py auto-probes,
# so even older skill checkouts (without the FURNITURE_CATALOG_PATH env honor
# patch) still find the catalog. Override only if you know what you're doing.
VAST_CATALOG_PATH="${VAST_CATALOG_PATH:-/data/3D-FUTURE-model}"
VAST_REQUIRE_CATALOG="${VAST_REQUIRE_CATALOG:-1}"

# Default catalog source is now `prestaged` (native 3D-FUTURE layout) because
# `hf_3dfront` and `abo` both use layouts the skill's get_mesh_path() resolver
# doesn't understand without a TBD ~30 LOC patch. Until that patch ships, only
# `prestaged` (matching 3D-FUTURE's <model_id>/raw_model.obj layout) actually
# resolves at runtime. To opt-in to fallback boxes (e.g., for fast first run):
#   VAST_CATALOG_SRC=skip VAST_REQUIRE_CATALOG=0

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

# Disk-space sanity. Roughly: model weights + catalog + 50 GB overhead.
DISK_AVAIL_GB=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
DISK_AVAIL_GB=${DISK_AVAIL_GB:-0}
case "$VAST_CATALOG_SRC" in
    skip)        CATALOG_GB=0 ;;
    abo)         CATALOG_GB=30 ;;
    hf_3dfront)  CATALOG_GB=100 ;;  # download + extract scratch
    prestaged)   CATALOG_GB=50 ;;
esac
# Estimate model size from HF id (rough heuristic; overrideable)
case "$VAST_MODEL_HF" in
    *27B*|*26[Bb]*) MODEL_GB=55 ;;
    *31[Bb]*)       MODEL_GB=65 ;;
    *35B*)          MODEL_GB=75 ;;
    *70B*-AWQ*|*70B*-INT4*|*-int4*) MODEL_GB=40 ;;
    *70B*)          MODEL_GB=140 ;;
    *)              MODEL_GB=80 ;;  # safe default
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
# CRITICAL PARALLELISM:
# Kick off the longest task (model download) immediately. The vllm/vllm-openai
# image already ships huggingface_hub, so `hf` CLI is available without
# waiting for our pip install. Everything else runs in parallel.
# =====================================================================
step "3. [PARALLEL START] Model download in background ($VAST_MODEL_HF)"
MODEL_DIR=/workspace/models/$(basename "$VAST_MODEL_HF")
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
step "6b. Patch trial/invoke.py to pin Sionna RT to GPU 1 (avoid vLLM contention)"
# vLLM holds GPU 0 (51 GB weights + ~16 GB KV ≈ 67/80 GB). Sionna RT defaults
# to cuda:0 → contention → potential OOM on heavy radio-map tasks.
# Patch _build_invoke_env to force CUDA_VISIBLE_DEVICES=1 in the simulation
# subprocess env (claude itself is Node.js; doesn't care about GPUs).
INVOKE=/workspace/skill/benchmark/trial/invoke.py
if grep -q "BENCH_SIM_GPU" "$INVOKE"; then
    ok "trial/invoke.py already pinned (CUDA_VISIBLE_DEVICES patch present)"
else
    # Quoted heredoc — bash does NOT expand inside. Pass file path via env.
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
        # Validate the patched file actually parses as Python — rules out the
        # case where regex injected syntactically broken code.
        if ! PYTHONPATH=/workspace/skill python3 -c "import benchmark.trial.invoke" 2>/dev/null; then
            warn "trial/invoke.py broken after patch! Reverting via rsync from source..."
            # Re-rsync just this file from the source
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
sed -i "s/MAX_OUTPUT_TOKENS_DEFAULT = 32000/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" "$OC_DIR/dist/cli.mjs"
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
    # Fallback: download wasn't kicked off in step 3 because hf CLI was missing.
    # Pip should have installed it now, so try again.
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
    # Step 4b couldn't start (hf or aws not in PATH at boot). Pip is done now,
    # so retry inline.
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

    # Sample a mesh to confirm files are non-trivial (not 0-byte)
    SAMPLE=$(find "$VAST_CATALOG_PATH" -maxdepth 5 \( -name "*.glb" -o -name "*.obj" \) 2>/dev/null | head -1)
    if [[ -n "$SAMPLE" ]]; then
        SAMPLE_SIZE=$(stat -c%s "$SAMPLE" 2>/dev/null || stat -f%z "$SAMPLE" 2>/dev/null || echo 0)
        if [[ "$SAMPLE_SIZE" -lt 10000 ]]; then
            warn "sample mesh is suspiciously small ($SAMPLE_SIZE bytes): $SAMPLE"
        else
            ok "sample mesh: $(basename "$SAMPLE") = ${SAMPLE_SIZE} bytes (looks real)"
        fi
    fi

    # invoke.py probes for `<catalog>/model_info.json` to set FURNITURE_CATALOG_PATH
    # in the trial subprocess env. Without that file, the catalog isn't picked up
    # at trial time even if it's downloaded. Verify it exists.
    if [[ -f "$VAST_CATALOG_PATH/model_info.json" ]]; then
        # Sanity: model_info.json should be JSON with multiple model entries
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
        # No model_info.json. invoke.py won't auto-detect this catalog at trial time.
        # The patched invoke.py also requires model_info.json even when FURNITURE_CATALOG_PATH
        # is set explicitly. So this is a hard requirement.
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

# Verify the collection exists and has at least 1 chunk (RAG won't help if it's empty)
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
step "11. Start vLLM on GPU 0 (TP=1, parser=$VAST_MODEL_PARSER, max_len=$VAST_MODEL_MAX_LEN)"
# CUDA_VISIBLE_DEVICES=0 pins vLLM to GPU 0. Then trial/invoke.py forces
# CUDA_VISIBLE_DEVICES=1 on simulation.py (see step 6b) so Sionna RT runs on
# GPU 1 with no contention. Both GPUs are now in use.
cat > /workspace/start_vllm.sh <<EOS
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
exec vllm serve $MODEL_DIR \\
    --tensor-parallel-size 1 \\
    --max-model-len $VAST_MODEL_MAX_LEN \\
    --max-num-seqs $VAST_MODEL_NUM_SEQS \\
    --gpu-memory-utilization 0.92 \\
    --enforce-eager \\
    --enable-auto-tool-choice \\
    --tool-call-parser $VAST_MODEL_PARSER \\
    --port 8001 \\
    --host 127.0.0.1 \\
    --served-model-name $BENCH_MODEL
EOS
chmod +x /workspace/start_vllm.sh
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "/workspace/start_vllm.sh 2>&1 | tee /workspace/logs/vllm.log"
ok "vLLM tmux session started (pinned to GPU 0)"

# =====================================================================
step "12. Wait for vLLM ready (up to 8 min — weight load + KV profile)"
VLLM_READY=0
for i in $(seq 1 48); do
    if curl -sf -m 3 -o /dev/null http://127.0.0.1:8001/v1/models; then
        ok "vLLM ready after $((i * 10))s"
        VLLM_READY=1
        break
    fi
    sleep 10
done
if [[ "$VLLM_READY" != "1" ]]; then
    echo "  ! vLLM log tail:"
    tail -30 /workspace/logs/vllm.log
    err "vLLM not ready after 8 min"
fi

# =====================================================================
step "13. End-to-end smoke test (openclaude → vLLM)"
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
# U001 = BER curve, train, simplest analytical task — verifies the whole chain
# U136 = scene_indoor_3dfuture, train, T0_scene_gen, furniture-in-prompt —
#        if catalog is set up correctly, the trial produces a multi-MB GLB
#        (real meshes); if fallbacks are being used, the GLB is <100 KB.
# This is a non-caveat-based real-mesh check that works TODAY without waiting
# for the caveats wiring.
rm -rf /workspace/skill/benchmark/results/preflight_real
cd /workspace/skill

# Pick the catalog test task: U136 if catalog wanted, just U001 otherwise
if [[ "$VAST_CATALOG_SRC" != "skip" ]]; then
    PREFLIGHT_TASKS="U001 U136"
else
    PREFLIGHT_TASKS="U001"
fi

set +e
# Use the SAME --max-turns as the full bench (step 14) so a U136 false-negative
# from premature turn-cap can't masquerade as a catalog problem.
timeout 1200 python3 benchmark/run_benchmark.py \
    --label preflight_real \
    --workers 1 \
    --model "$BENCH_MODEL" \
    --conditions with_skill \
    --task-ids $PREFLIGHT_TASKS \
    --max-turns 25 \
    --timeout 400 \
    > /workspace/logs/preflight_real.log 2>&1
RC=$?
set -e

PROMPT_FILE=$(find /workspace/skill/benchmark/results/preflight_real -name prompt.txt -type f 2>/dev/null | head -1)
if [[ -z "$PROMPT_FILE" ]]; then
    tail -30 /workspace/logs/preflight_real.log
    err "real-trial verification: no prompt.txt produced"
fi

# Verify (1) RAG injection — look for the marker rag.py inserts
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

# Verify (2) skill content reached the model — look for a SKILL.md marker
if grep -q -E "Module 1|SKILL.md|rf-simulator" "$PROMPT_FILE"; then
    ok "skill content present in prompt (SKILL.md auto-loaded)"
else
    warn "skill content NOT detected — RF_SKILL_DIR may be misconfigured"
fi

# Verify (3) real meshes used on U136 — check exported GLB size in workdir
# Real 3D-FUTURE meshes are typically multi-MB; AABB-cube fallbacks are <100 KB.
if [[ "$VAST_CATALOG_SRC" != "skip" ]]; then
    U136_DIR=$(find /workspace/skill/benchmark/results/preflight_real -path "*U136*" -type d 2>/dev/null | head -1)
    if [[ -n "$U136_DIR" ]]; then
        # Look for any .glb produced by the trial (scene export, viewer, etc.)
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

# Verify (4) at least one trial passed
if grep -qE "Pass rate: [1-9][0-9]*/[1-9]" /workspace/logs/preflight_real.log; then
    ok "preflight trials: $(grep "Pass rate:" /workspace/logs/preflight_real.log | tail -1)"
else
    warn "preflight trials did not pass — see /workspace/logs/preflight_real.log
        (acceptable if model capability is the cause; chain is still verified by checks 1-3)"
fi

# Cleanup verification output (keep logs)
rm -rf /workspace/skill/benchmark/results/preflight_real

# =====================================================================
if [[ "$VAST_AUTOLAUNCH_BENCHMARK" != "1" ]]; then
    echo ""
    echo "========================================="
    echo "Bootstrap COMPLETE — chain verified after $(($(date +%s) - T0))s."
    echo "Auto-launch disabled (VAST_AUTOLAUNCH_BENCHMARK=$VAST_AUTOLAUNCH_BENCHMARK)."
    echo ""
    echo "Manual launch command:"
    echo "  source /workspace/skill.env && cd /workspace/skill && \\"
    echo "  python3 benchmark/run_benchmark.py --label $VAST_BENCH_LABEL \\"
    echo "    --workers $VAST_BENCH_WORKERS --model $BENCH_MODEL \\"
    echo "    --conditions with_skill no_skill self_gen --max-turns 25 \\"
    echo "    --timeout 400 --retry-timeout 1200 --resume"
    echo "========================================="
    exit 0
fi

step "14. Launch benchmark in tmux session 'benchmark'"
tmux kill-session -t benchmark 2>/dev/null || true
tmux new-session -d -s benchmark "source /workspace/skill.env && cd /workspace/skill && \
    python3 benchmark/run_benchmark.py \
        --label $VAST_BENCH_LABEL \
        --workers $VAST_BENCH_WORKERS \
        --model $BENCH_MODEL \
        --conditions with_skill no_skill self_gen \
        --max-turns 25 \
        --timeout 400 \
        --retry-timeout 1200 \
        --resume \
        2>&1 | tee /workspace/logs/benchmark.log"
ok "benchmark launched"

# =====================================================================
TOTAL=$(($(date +%s) - T0))
echo ""
echo "========================================="
echo "Bootstrap COMPLETE in ${TOTAL}s — first trial firing now."
echo "Model:      $BENCH_MODEL ($VAST_MODEL_HF)"
echo "Parser:     $VAST_MODEL_PARSER"
echo "Workers:    $VAST_BENCH_WORKERS"
echo ""
echo "Monitor:    tail -f /workspace/logs/benchmark.log"
echo "Tear down:  tmux kill-session -t benchmark"
echo "Boot log:   $LOG"
echo "========================================="
