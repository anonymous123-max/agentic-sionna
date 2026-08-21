#!/bin/bash
# bootstrap_vastai_frontier.sh — Tier-3 sibling of bootstrap_vastai_tp.sh that
# launches ONE vLLM instance with --tensor-parallel-size $VAST_TP_SIZE for
# FRONTIER-SCALE INT4 models that don't fit on smaller boxes:
#
#   - DeepSeek V3.1 INT4   (~350 GB sharded weights, parser=deepseek_v3)
#   - Kimi K2.6 INT4       (~500 GB sharded weights, custom parser per recipe)
#
# Target hardware (NVLink preferred, PCIe tolerated with warnings):
#
#   - 4× B200 (192 GB × 4 = 768 GB)         → fits both, TP=4 over NVLink. ~$24/hr spot.
#                                              Best choice for Kimi K2.6 (KV pool comfortable).
#   - 8× H100 SXM5 (80 GB × 8 = 640 GB)     → fits DeepSeek easily, tight for Kimi.
#                                              ~$15-20/hr spot. Best for DeepSeek V3.1.
#   - 8× A100 80GB PCIe (80 GB × 8 = 640 GB) → fits DeepSeek; Kimi is borderline.
#                                              ~$6.82/hr spot — ~2-3× cheaper than
#                                              SXM5/B200 but ~2-3× longer wall time
#                                              per trial due to PCIe all-reduce penalty.
#                                              Net: usually still cheaper end-to-end,
#                                              but only for non-latency-sensitive sweeps.
#
# WARNING — read before launching:
#
#   1. NVLink strongly preferred but no longer hard-required. By default, this
#      script now follows bootstrap_vastai_tp.sh's pattern: detect interconnect
#      via `nvidia-smi topo -m`, warn loudly if PCIe-only, and continue. To
#      restore the old hard-fail behavior (e.g. on shared CI where PCIe boxes
#      should never be used), set VAST_REQUIRE_NVLINK=1.
#
#      Frontier MoE models (DeepSeek V3.1, Kimi K2.6) push hundreds of GB
#      through TP all-reduce per second. PCIe Gen4 (~32 GB/s bidirectional)
#      means 2-3× longer per-token latency than NVLink/NVSwitch — but on
#      8× A100 PCIe at ~$6.82/hr (vs ~$18/hr SXM5, ~$24/hr B200), the slower
#      run is still ~half the total cost in many cases.
#
#      Per-model max-num-seqs defaults are auto-lowered when PCIe is detected
#      (DeepSeek 16→12, Kimi 8→6) because PCIe all-reduce halves throughput
#      and request queueing kicks in at a much lower concurrency than NVLink.
#
#   2. Disk volume sizing. Kimi K2.6 INT4 weights alone are ~500 GB on disk;
#      add 50 GB catalog + ~100 GB working-set headroom → rent ≥700 GB volume.
#      DeepSeek V3.1 INT4 needs ≥500 GB. Step 0 disk-space check enforces this.
#
#   3. Cost. Frontier runs are EXPENSIVE per benchmark sweep, but the cheap
#      PCIe path narrows the gap considerably:
#         4× B200          @ ~$24/hr   × 2 hr ≈ $48 per sweep (fast)
#         8× H100 SXM5     @ ~$18/hr   × 2 hr ≈ $36 per sweep (fast)
#         8× A100 80G PCIe @ ~$6.82/hr × 4-5 hr ≈ $27-34 per sweep (slow but cheap)
#      Only commit to Tier-3 sweeps if Tier-1/Tier-2 results justify it.
#      (Suggested rule: if Tier-2 27B/30B INT4 is already >70% pass rate,
#      Tier-3 may not move the needle enough to justify the spend.)
#      For 1-2 concurrent requests only, prefer single-instance bootstrap_vastai.sh
#      on a smaller GPU — frontier TP is overkill at low concurrency.
#
#   4. Multi-rank NCCL hangs (inherited from bootstrap_vastai_tp.sh §0 #8 / §3
#      / §8.3). The known shm_broadcast hang on vLLM 0.20.0 / CUDA 13 / NCCL
#      2.28.9 still applies. We poll for the signature and bail fast at step 12.
#
#   5. Kimi K2.6 vLLM recipe is fluid. The vLLM team publishes the canonical
#      flags at:
#         https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html
#      Verify --quantization fp8 / --tool-call-parser are still current there
#      BEFORE invoking (see VAST_MODEL_VLLM_EXTRA_ARGS env var below).
#
# Pasteable into vast.ai's "On-start command" field, or run via SSH:
#   scp bootstrap_vastai_frontier.sh root@<host>:/root/
#   ssh -i ~/.ssh/vastai -p <port> root@<host> 'bash /root/bootstrap_vastai_frontier.sh'
#
# Required env vars:
#   VAST_REPO_RSYNC_SRC   e.g. "you@sunlab:~/PycharmProjects/new-sionna-skill"
#                         (same as bootstrap_vastai.sh)
#   VAST_MODEL_HF         frontier model HF id (e.g. "deepseek-ai/DeepSeek-V3.1"
#                         or "moonshotai/Kimi-K2.6"). NO default — pick explicitly.
#   VAST_MODEL_PARSER     vLLM tool-call parser. Frontier-specific values:
#                            DeepSeek V3.1: "deepseek_v3"
#                            Kimi K2.6:     verify against vLLM recipe page
#                         NO default — pick explicitly.
#
# Optional env vars (frontier-specific defaults; all others identical to
# bootstrap_vastai_tp.sh — see that file's header for full reference):
#
#   VAST_TP_SIZE                    GPUs to shard across. Auto-detected from
#                                   nvidia-smi -L if unset (4 for 4× B200,
#                                   8 for 8× H100 SXM5). Must be ≤ actual count.
#   VAST_MODEL_MAX_LEN              vLLM --max-model-len. Default: 32768.
#                                   Bump to 131072 for DeepSeek (128K window)
#                                   or 262144 for Kimi K2.6 (256K window) if
#                                   the workload needs it.
#   VAST_MODEL_NUM_SEQS_FRONTIER    vLLM --max-num-seqs. Default depends on
#                                   model AND interconnect:
#                                     DeepSeek V3.1, NVLink → 16
#                                     DeepSeek V3.1, PCIe   → 12
#                                     Kimi K2.6,     NVLink → 8
#                                     Kimi K2.6,     PCIe   → 6  (borderline)
#                                   Frontier MoE models burn KV faster per
#                                   token than dense models because each token
#                                   activates O(8) experts; max-num-seqs is
#                                   conservatively low. PCIe defaults are
#                                   further halved-toward-2/3 because PCIe
#                                   all-reduce halves per-token throughput,
#                                   so request queueing happens at much lower
#                                   concurrency than NVLink.
#   VAST_REQUIRE_NVLINK             If 1, restore the old hard-fail behavior
#                                   when no NV-links are reported in
#                                   `nvidia-smi topo -m`. Default: 0 (warn only).
#                                   Set to 1 on shared CI if PCIe boxes should
#                                   never be allowed.
#   VAST_SKIP_REPO_RSYNC            If 1, skip the repo rsync at step 5 (assumes
#                                   /workspace/skill is already pre-staged).
#                                   Default: 0.
#   VAST_GPU_MEM_UTIL_FRONTIER      --gpu-memory-utilization. Default: 0.85
#                                   (matches TP script — leaves headroom for
#                                   NCCL all-reduce + activation buffers).
#   VAST_VLLM_WAIT_S                max seconds to wait for vLLM ready.
#                                   Default: 1500 (25 min) on NVLink, 1800
#                                   (30 min) on PCIe (slower interconnect can
#                                   stretch shard-load + warmup). DeepSeek
#                                   350 GB / Kimi 500 GB takes longer to load
#                                   than 27 B regardless of interconnect.
#   VAST_BENCH_WORKERS              harness workers. If unset, auto-set to
#                                   max-num-seqs - 1.
#   VAST_MODEL_VLLM_EXTRA_ARGS      extra args passed verbatim to `vllm serve`.
#                                   Use this for K2.6 recipe-specific flags
#                                   (e.g. "--quantization fp8") without
#                                   editing the script. Default: "" (empty).
#
# Exit behavior: same as bootstrap_vastai_tp.sh — fail fast on any step. The TP
# init step (12) explicitly watches for shm_broadcast and exits with a
# diagnostic message if it sees it, instead of letting the 25-min timer
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

# No defaults — frontier model + parser must be explicit.
VAST_MODEL_HF="${VAST_MODEL_HF:-}"
VAST_MODEL_PARSER="${VAST_MODEL_PARSER:-}"
VAST_MODEL_MAX_LEN="${VAST_MODEL_MAX_LEN:-32768}"

# Hard-fail if interconnect is PCIe (off by default; warn-only otherwise).
VAST_REQUIRE_NVLINK="${VAST_REQUIRE_NVLINK:-0}"
# Skip repo rsync (assume /workspace/skill is pre-staged). Mirrors TP script.
VAST_SKIP_REPO_RSYNC="${VAST_SKIP_REPO_RSYNC:-0}"

# --- Frontier-specific knobs ---
# Auto-detect TP size from visible GPUs if unset. We do this BEFORE the sanity
# step because step 0 uses VAST_TP_SIZE in error messages and disk-need math.
GPU_COUNT_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
GPU_COUNT_DETECTED=${GPU_COUNT_DETECTED:-0}
VAST_TP_SIZE="${VAST_TP_SIZE:-$GPU_COUNT_DETECTED}"

# Best-effort interconnect detection (used to scale per-model num-seqs defaults
# and the vLLM-ready timeout). The authoritative check + user-visible warning
# happens in step 0; this early probe is just for tuning defaults.
NVLINK_PRESENT=0
if command -v nvidia-smi > /dev/null 2>&1; then
    if nvidia-smi topo -m 2>/dev/null | grep -qE "NV[0-9]"; then
        NVLINK_PRESENT=1
    fi
fi

# Pick a per-model max-num-seqs default based on which frontier model is loaded
# AND the interconnect. PCIe all-reduce halves per-token throughput, so request
# queueing kicks in at much lower concurrency than NVLink — defaults are
# correspondingly lower in the PCIe column.
# (Kimi K2.6 weights are ~40% larger than DeepSeek V3.1, so per-shard KV pool
# is correspondingly smaller; defaults track that gap.)
case "$VAST_MODEL_HF" in
    *Kimi*|*kimi*|*K2*)
        if [[ "$NVLINK_PRESENT" == "1" ]]; then
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=8
        else
            # Kimi at TP=8 PCIe is genuinely borderline — see step 0 warning.
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=6
        fi
        ;;
    *DeepSeek*|*deepseek*|*V3*)
        if [[ "$NVLINK_PRESENT" == "1" ]]; then
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=16
        else
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=12
        fi
        ;;
    *)
        if [[ "$NVLINK_PRESENT" == "1" ]]; then
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=12
        else
            VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT=8
        fi
        ;;
esac
VAST_MODEL_NUM_SEQS_FRONTIER="${VAST_MODEL_NUM_SEQS_FRONTIER:-$VAST_MODEL_NUM_SEQS_FRONTIER_DEFAULT}"

# vLLM-ready timeout: bump from 25 min (NVLink) to 30 min (PCIe). Frontier
# weight-load time is dominated by disk I/O, not interconnect, but PCIe NCCL
# warmup adds a few extra minutes in our measurements.
if [[ "$NVLINK_PRESENT" == "1" ]]; then
    VAST_VLLM_WAIT_S="${VAST_VLLM_WAIT_S:-1500}"
else
    VAST_VLLM_WAIT_S="${VAST_VLLM_WAIT_S:-1800}"
fi
VAST_GPU_MEM_UTIL_FRONTIER="${VAST_GPU_MEM_UTIL_FRONTIER:-0.85}"
VAST_MODEL_VLLM_EXTRA_ARGS="${VAST_MODEL_VLLM_EXTRA_ARGS:-}"

# Workers default = max-num-seqs - 1
VAST_BENCH_WORKERS="${VAST_BENCH_WORKERS:-$((VAST_MODEL_NUM_SEQS_FRONTIER - 1))}"

VAST_BENCH_LABEL="${VAST_BENCH_LABEL:-train_full_v1_frontier_tp${VAST_TP_SIZE}}"
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
step "0. Sanity check (frontier, TP=$VAST_TP_SIZE, model=$VAST_MODEL_HF)"
[[ "$(id -u)" == "0" ]] || err "must run as root on vast.ai"
[[ "$VAST_SKIP_REPO_RSYNC" == "1" ]] || [[ -n "$VAST_REPO_RSYNC_SRC" ]] || err "VAST_REPO_RSYNC_SRC env var is required (or set VAST_SKIP_REPO_RSYNC=1 if /workspace/skill is pre-staged)"
[[ "$VAST_SKIP_REPO_RSYNC" == "1" ]] || [[ -f "$VAST_REPO_RSYNC_KEY" ]] || err "rsync SSH key missing at $VAST_REPO_RSYNC_KEY"
[[ -n "$VAST_MODEL_HF" ]] || err "VAST_MODEL_HF is required (no default for Tier-3 — pick explicitly:
    deepseek-ai/DeepSeek-V3.1   (or community AWQ/INT4 build)
    moonshotai/Kimi-K2.6)"
[[ -n "$VAST_MODEL_PARSER" ]] || err "VAST_MODEL_PARSER is required (no default for Tier-3:
    DeepSeek V3.1 → 'deepseek_v3'
    Kimi K2.6     → check vLLM recipe at docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html)"

# Frontier-specific GPU count check
GPU_COUNT="$GPU_COUNT_DETECTED"
if [[ "$GPU_COUNT" -lt "$VAST_TP_SIZE" ]]; then
    err "VAST_TP_SIZE=$VAST_TP_SIZE but only $GPU_COUNT GPUs visible.
    Either rent a box with ≥$VAST_TP_SIZE GPUs, or lower VAST_TP_SIZE (e.g. =$GPU_COUNT)."
fi
if [[ "$VAST_TP_SIZE" -lt 4 ]]; then
    err "VAST_TP_SIZE=$VAST_TP_SIZE is too small for Tier-3 (frontier) models.
    DeepSeek V3.1 (~350 GB INT4) and Kimi K2.6 (~500 GB INT4) need TP≥4 even on
    192 GB B200. If you have only 1-2 GPUs, use bootstrap_vastai.sh / smaller models."
fi
ok "GPUs visible: $GPU_COUNT (TP=$VAST_TP_SIZE)"

# NVLink check. Default behavior (VAST_REQUIRE_NVLINK=0): mirror
# bootstrap_vastai_tp.sh — detect via `nvidia-smi topo -m` and warn loudly if
# only PCIe is reported, then continue. Set VAST_REQUIRE_NVLINK=1 to restore
# the old hard-fail behavior (e.g. on shared CI where PCIe should never run).
#
# Detection: nvidia-smi topo -m output should contain NV<digit> (e.g., NV12,
# NV18) in the GPU↔GPU cells. PCIe-only shows PHB/PIX/SYS/NODE.
if ! command -v nvidia-smi > /dev/null; then
    if [[ "$VAST_REQUIRE_NVLINK" == "1" ]]; then
        err "nvidia-smi not on PATH — cannot verify NVLink. VAST_REQUIRE_NVLINK=1; aborting."
    else
        warn "nvidia-smi not on PATH — cannot verify interconnect. Continuing anyway."
    fi
fi
TOPO=$(nvidia-smi topo -m 2>/dev/null || true)
if [[ -z "$TOPO" ]]; then
    if [[ "$VAST_REQUIRE_NVLINK" == "1" ]]; then
        err "nvidia-smi topo -m produced no output. Cannot verify NVLink. Aborting (VAST_REQUIRE_NVLINK=1)."
    else
        warn "nvidia-smi topo -m produced no output. Cannot verify interconnect; assuming PCIe-tier penalty applies."
    fi
elif echo "$TOPO" | grep -qE "NV[0-9]"; then
    NVLINK_LINES=$(echo "$TOPO" | grep -cE "NV[0-9]" || true)
    ok "NVLink detected ($NVLINK_LINES rows mention NV# links in topo matrix)"
else
    echo "  ! nvidia-smi topo -m output (PCIe-only — no NV# links seen):"
    echo "$TOPO" | head -20
    if [[ "$VAST_REQUIRE_NVLINK" == "1" ]]; then
        err "NO NVLINK DETECTED and VAST_REQUIRE_NVLINK=1.
    DeepSeek V3.1 / Kimi K2.6 push hundreds of GB through TP all-reduce per second;
    PCIe Gen4 (~32 GB/s) makes per-token latency multi-second.
    Either re-rent NVSwitch-bearing hardware (4× B200 HGX, 8× H100 SXM5 HGX),
    or unset VAST_REQUIRE_NVLINK to accept the PCIe penalty in exchange for
    ~half the hourly rate."
    else
        # Warn-by-default. The user is opting into the cheaper PCIe path.
        # Make sure the trade-off is unmistakably clear, including the
        # Kimi-specific borderline call-out.
        warn "NO NVLINK DETECTED. TP=$VAST_TP_SIZE all-reduce will run over PCIe.
       - TP=$VAST_TP_SIZE over PCIe will lose 50-70% per-token throughput vs NVLink.
       - Frontier models (Kimi K2.6 ~500GB, DeepSeek V3.1 ~350GB) amplify this
         penalty even more (MoE all-reduce volume is hundreds of GB/sec).
       - Wall time per benchmark trial may be 2-3× longer than NVLink, but \$/hr
         is 2-3× cheaper (e.g. 8× A100 PCIe ~\$6.82/hr vs ~\$18/hr H100 SXM5).
       - If you only need 1-2 concurrent requests, single-instance
         bootstrap_vastai.sh on a smaller GPU is much cheaper.
       - Set VAST_REQUIRE_NVLINK=1 to force the old hard-fail behavior."
        case "$VAST_MODEL_HF" in
            *Kimi*|*kimi*|*K2*)
                warn "Kimi K2.6 INT4 specifically: TP=$VAST_TP_SIZE on PCIe is genuinely borderline.
       ~500GB sharded weights + MoE all-reduce + PCIe = expect long per-token latency
       AND tight KV-pool headroom (max-num-seqs auto-lowered to $VAST_MODEL_NUM_SEQS_FRONTIER).
       If the run exceeds ~30 min/trial or OOMs, fall back to DeepSeek V3.1 INT4 on this box,
       or rent NVLink hardware for Kimi specifically."
                ;;
        esac
    fi
fi

# Catalog config sanity (identical to bootstrap_vastai_tp.sh)
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

# Disk-space sanity — frontier models are MUCH bigger than Tier 1/2.
DISK_AVAIL_GB=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
DISK_AVAIL_GB=${DISK_AVAIL_GB:-0}
case "$VAST_CATALOG_SRC" in
    skip)        CATALOG_GB=0 ;;
    abo)         CATALOG_GB=30 ;;
    hf_3dfront)  CATALOG_GB=100 ;;
    prestaged)   CATALOG_GB=50 ;;
esac
case "$VAST_MODEL_HF" in
    *Kimi*|*kimi*|*K2*)
        # Kimi K2.6 INT4 ships ~500 GB sharded weights.
        MODEL_GB=520
        ;;
    *DeepSeek*|*deepseek*|*V3*)
        # DeepSeek V3.1 INT4 (community AWQ or native) ~350 GB.
        MODEL_GB=370
        ;;
    *27B*|*26[Bb]*) MODEL_GB=55 ;;
    *31[Bb]*)       MODEL_GB=65 ;;
    *35B*)          MODEL_GB=75 ;;
    *70B*-AWQ*|*70B*-INT4*|*-int4*) MODEL_GB=40 ;;
    *70B*)          MODEL_GB=140 ;;
    *)
        warn "unknown frontier model size for $VAST_MODEL_HF — assuming 400 GB"
        MODEL_GB=400
        ;;
esac
DISK_NEEDED=$((MODEL_GB + CATALOG_GB + 100))   # extra 100 GB headroom for frontier (vs 50 for Tier 1/2)
echo "  disk: $DISK_AVAIL_GB GB free, ~$DISK_NEEDED GB estimated need (model=$MODEL_GB + catalog=$CATALOG_GB + overhead=100)"
if [[ "$DISK_AVAIL_GB" -lt "$DISK_NEEDED" ]]; then
    err "insufficient disk: $DISK_AVAIL_GB GB free, need ~$DISK_NEEDED GB.
    Frontier-tier sweeps need a LARGE rented volume:
      DeepSeek V3.1 INT4: ≥500 GB
      Kimi K2.6 INT4:     ≥700 GB
    Re-rent with a larger volume."
fi
ok "disk-space OK"

# =====================================================================
step "1. Stop vast.ai's pre-installed vLLM (eats 76 GB / GPU per §0 #1)"
supervisorctl stop vllm 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 4
GPU_USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
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
step "3. [PARALLEL START] Model download in background ($VAST_MODEL_HF, ~$MODEL_GB GB)"
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
    ok "model download PID=$DOWNLOAD_PID kicked off (log: /workspace/logs/download.log, ~$MODEL_GB GB)"
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
if [[ "$VAST_SKIP_REPO_RSYNC" == "1" ]]; then
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
# In the frontier variant, vLLM holds ALL $VAST_TP_SIZE GPUs. The CUDA_VISIBLE_DEVICES=1
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
sed -i "s/MAX_OUTPUT_TOKENS_DEFAULT = 32000/MAX_OUTPUT_TOKENS_DEFAULT = 8000/g" "$OC_DIR/dist/cli.mjs"
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
step "10. Wait for model download (background since step 3, frontier ~$MODEL_GB GB)"
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
step "11. Start vLLM with TP=$VAST_TP_SIZE (frontier: parser=$VAST_MODEL_PARSER, max_len=$VAST_MODEL_MAX_LEN, num_seqs=$VAST_MODEL_NUM_SEQS_FRONTIER)"
# Frontier-mode rationale (vs bootstrap_vastai_tp.sh's TP=8 on Tier 2 27B):
#   - --tensor-parallel-size $VAST_TP_SIZE: shard the LLM weights across all
#     $VAST_TP_SIZE GPUs over NVLink. With $MODEL_GB GB of weights, each rank
#     holds ~$((MODEL_GB / VAST_TP_SIZE)) GB; KV pool per rank is therefore
#     (GPU_MEM × $VAST_GPU_MEM_UTIL_FRONTIER - $((MODEL_GB / VAST_TP_SIZE)) GB).
#     Each frontier MoE token activates O(8) experts so KV growth is faster
#     per token than dense models — max-num-seqs=$VAST_MODEL_NUM_SEQS_FRONTIER
#     is conservative.
#   - --enforce-eager: REQUIRED on multi-rank vLLM. Per §0 #20, CUDA-graph
#     compilation is the most common multi-rank deadlock source. Trades 10-20%
#     per-token throughput for reliable startup. Mandatory at TP≥4.
#   - --gpu-memory-utilization $VAST_GPU_MEM_UTIL_FRONTIER (=0.85 default):
#     same as bootstrap_vastai_tp.sh — leaves headroom for NCCL all-reduce
#     buffers and per-shard activation tensors. Frontier MoE has slightly
#     larger working set than dense at the same TP, but 0.85 still works.
#   - VAST_MODEL_VLLM_EXTRA_ARGS: passthrough for recipe-specific flags. For
#     Kimi K2.6, the vLLM team's recipe at
#         docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html
#     may require --quantization fp8 or other model-specific flags. VERIFY
#     AGAINST THAT PAGE before invoking. Pass via:
#         VAST_MODEL_VLLM_EXTRA_ARGS="--quantization fp8" bash bootstrap_vastai_frontier.sh
#   - NO CUDA_VISIBLE_DEVICES export: vLLM TP needs to see all $VAST_TP_SIZE
#     GPUs. trial/invoke.py's CUDA_VISIBLE_DEVICES=1 (set in step 6b) only
#     applies to the simulation subprocess and is harmless here.
cat > /workspace/start_vllm.sh <<EOS
#!/bin/bash
# NCCL-friendly env (defensive — these defaults are usually fine but set
# explicitly so a quirky vast.ai container can't override us):
export NCCL_DEBUG=WARN                         # log NCCL hangs without flooding
export NCCL_ASYNC_ERROR_HANDLING=1             # surface NCCL errors as exceptions
export VLLM_USE_TRITON_FLASH_ATTN=1            # standard fast-attn path
exec vllm serve $MODEL_DIR \\
    --tensor-parallel-size $VAST_TP_SIZE \\
    --max-model-len $VAST_MODEL_MAX_LEN \\
    --max-num-seqs $VAST_MODEL_NUM_SEQS_FRONTIER \\
    --gpu-memory-utilization $VAST_GPU_MEM_UTIL_FRONTIER \\
    --enforce-eager \\
    --enable-auto-tool-choice \\
    --tool-call-parser $VAST_MODEL_PARSER \\
    --port 8001 \\
    --host 127.0.0.1 \\
    --served-model-name $BENCH_MODEL \\
    $VAST_MODEL_VLLM_EXTRA_ARGS
EOS
chmod +x /workspace/start_vllm.sh
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "/workspace/start_vllm.sh 2>&1 | tee /workspace/logs/vllm.log"
ok "vLLM tmux session started (TP=$VAST_TP_SIZE, all $GPU_COUNT GPUs visible)"

# =====================================================================
step "12. Wait for vLLM ready (up to $((VAST_VLLM_WAIT_S / 60)) min — frontier load is slow + watch for NCCL hang)"
# Frontier vLLM init phases we expect to see in /workspace/logs/vllm.log:
#   1. Each rank loads its weight shard. At $MODEL_GB GB total / $VAST_TP_SIZE
#      ranks, that's ~$((MODEL_GB / VAST_TP_SIZE)) GB per rank — disk I/O
#      bound, takes 60-300s on fast NVMe.
#   2. NCCL handshake / shm_broadcast / all-reduce warmup (the danger zone).
#   3. KV-cache profile run (single fwd pass with synthetic input).
#   4. Server starts listening on :8001.
#
# The NCCL handshake is where TP=2 hung on the previous attempt (§0 #8, §3,
# §8.3). On vLLM 0.20.0 / CUDA 13 / NCCL 2.28.9 the failure mode is a stuck
# thread that eventually emits:
#   "shm_broadcast: No available shared memory broadcast block found in 60 seconds"
# We poll for that string every iteration and bail immediately if it appears
# instead of letting the 25-min timer run out.
VLLM_READY=0
NCCL_HANG_DETECTED=0
ITERS=$((VAST_VLLM_WAIT_S / 10))
for i in $(seq 1 $ITERS); do
    if curl -sf -m 3 -o /dev/null http://127.0.0.1:8001/v1/models; then
        ok "vLLM ready after $((i * 10))s (TP=$VAST_TP_SIZE, frontier model)"
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
    Frontier-tier workarounds:
      (a) Confirm NVLink is really live: nvidia-smi nvlink --status
      (b) Try a smaller TP (only Kimi K2.6 needs TP=8 on H100; DeepSeek V3.1 may run TP=4):
            VAST_TP_SIZE=4 bash bootstrap_vastai_frontier.sh
      (c) Try a newer vLLM release on a fresh rent (the bug is image-version-specific).
    Frontier models have NO TP=1 fallback — they don't fit on a single GPU."
fi

if [[ "$VLLM_READY" != "1" ]]; then
    echo "  ! vLLM log tail:"
    tail -30 /workspace/logs/vllm.log
    err "vLLM not ready after $((VAST_VLLM_WAIT_S / 60)) min. Frontier TP init exceeded the wait budget without the known shm_broadcast signature.
    Inspect /workspace/logs/vllm.log directly. If it shows progress (rank N loaded shard, etc.) but no listener,
    bump VAST_VLLM_WAIT_S and retry. If it shows NCCL/CUDA errors, see (b)/(c) workarounds above.
    Frontier models can take 5-10 minutes JUST to load weights from disk; the default 25 min wait is generous but not infinite."
fi

# =====================================================================
step "13. End-to-end smoke test (openclaude → vLLM, single instance at :8001)"
source /workspace/skill.env
SMOKE=$(timeout 120 /usr/local/bin/openclaude-vllm \
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
      (DeepSeek V3.1 → 'deepseek_v3'; Kimi K2.6 → see vLLM recipe page)
    - openclaude max_tokens patch applied
    - VAST_MODEL_VLLM_EXTRA_ARGS contains required model-specific flags
      (e.g. Kimi K2.6 may need --quantization fp8)"
fi

# Frontier-specific: probe a long-context prompt for Kimi (256K window) /
# DeepSeek (128K window) sanity. Skip for unrecognized models.
case "$VAST_MODEL_HF" in
    *Kimi*|*kimi*|*K2*)
        if [[ "$VAST_MODEL_MAX_LEN" -gt 65536 ]]; then
            echo "  ... probing long-context (max_len=$VAST_MODEL_MAX_LEN, Kimi 256K window)"
            # Quick token count check — don't do a real long prompt (expensive),
            # just confirm /v1/models reports the served max-len we configured.
            REPORTED_LEN=$(curl -s http://127.0.0.1:8001/v1/models | python3 -c \
                "import sys, json; d=json.load(sys.stdin); print(d['data'][0].get('max_model_len','unknown'))" 2>/dev/null || echo "unknown")
            ok "vLLM reports max_model_len=$REPORTED_LEN (configured: $VAST_MODEL_MAX_LEN)"
        fi
        ;;
esac

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
if [[ "$VAST_AUTOLAUNCH_BENCHMARK" != "1" ]]; then
    echo ""
    echo "========================================="
    echo "Bootstrap COMPLETE — chain verified after $(($(date +%s) - T0))s."
    echo "Mode:               TP=$VAST_TP_SIZE single-instance (frontier)"
    echo "Model:              $VAST_MODEL_HF (~$MODEL_GB GB on disk)"
    echo "Parser:             $VAST_MODEL_PARSER"
    echo "max-num-seqs:       $VAST_MODEL_NUM_SEQS_FRONTIER"
    echo "Auto-launch disabled (VAST_AUTOLAUNCH_BENCHMARK=$VAST_AUTOLAUNCH_BENCHMARK)."
    echo ""
    echo "Manual launch command:"
    echo "  source /workspace/skill.env && cd /workspace/skill && \\"
    echo "  python3 benchmark/run_benchmark.py --label $VAST_BENCH_LABEL \\"
    echo "    --workers $VAST_BENCH_WORKERS --model $BENCH_MODEL \\"
    echo "    --conditions with_skill no_skill self_gen --max-turns 25 \\"
    echo "    --timeout 400 --retry-timeout 1200 --resume"
    echo ""
    if [[ "$NVLINK_PRESENT" == "1" ]]; then
        echo "FRONTIER COST REMINDER (NVLink path): every hour you keep the box rented costs"
        echo "  ~\$24/hr on 4× B200, ~\$18/hr on 8× H100 SXM5."
    else
        echo "FRONTIER COST REMINDER (PCIe path):"
        echo "  8× A100 PCIe at ~\$6.82/hr × ~3-5 hr per model = \$20-35;"
        echo "  ~half the SXM5/B200 cost but ~2× the wall time."
        echo "  Net: usually still cheaper end-to-end. Tear down promptly."
    fi
    echo "Tear down promptly when done: tmux kill-session -t vllm; <vast.ai stop>"
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
ok "benchmark launched (TP=$VAST_TP_SIZE frontier, OPENAI_BASE_URL=http://127.0.0.1:8001/v1)"

# =====================================================================
TOTAL=$(($(date +%s) - T0))
echo ""
echo "========================================="
echo "Bootstrap COMPLETE in ${TOTAL}s — first frontier trial firing now."
echo "Mode:        TP=$VAST_TP_SIZE single vLLM instance (port 8001)"
echo "Model:       $BENCH_MODEL ($VAST_MODEL_HF, ~$MODEL_GB GB on disk)"
echo "Parser:      $VAST_MODEL_PARSER"
echo "max-num-seqs:$VAST_MODEL_NUM_SEQS_FRONTIER"
echo "Workers:     $VAST_BENCH_WORKERS"
echo ""
echo "Monitor:     tail -f /workspace/logs/benchmark.log"
echo "vLLM log:    tail -f /workspace/logs/vllm.log"
echo "Tear down:   tmux kill-session -t benchmark"
echo "Boot log:    $LOG"
echo ""
echo "*** FRONTIER COST REMINDER ***"
if [[ "$NVLINK_PRESENT" == "1" ]]; then
    echo "Per-hour rates (typical spot, NVLink path):"
    echo "  4× B200       ~\$24/hr  → ~\$48 per 2-hour sweep"
    echo "  8× H100 SXM5  ~\$18/hr  → ~\$36 per 2-hour sweep"
    echo "Two-condition sweeps (with_skill + no_skill) double the GPU time."
else
    echo "Per-hour rates (typical spot, PCIe path):"
    echo "  8× A100 80G PCIe  ~\$6.82/hr  →  ~\$20-35 per model (3-5 hr)"
    echo "PCIe trades 2-3× longer wall time for ~half the hourly rate vs SXM5/B200;"
    echo "net cost end-to-end is usually still lower. Two-condition sweeps double"
    echo "the GPU time, so still tear down promptly when done."
fi
echo "Tear down PROMPTLY when results are harvested:"
echo "  ssh sunlab 'rsync -az ...:/workspace/skill/benchmark/results/  ./results/'"
echo "  tmux kill-session -t vllm; <vast.ai destroy instance>"
echo ""
echo "If the run truncates with a multiprocessing-pool teardown hang (known harness"
echo "bug at high concurrency), re-launch with --resume — completed trials are saved:"
echo "  source /workspace/skill.env && cd /workspace/skill && \\"
echo "  python3 benchmark/run_benchmark.py --label $VAST_BENCH_LABEL \\"
echo "      --workers $VAST_BENCH_WORKERS --model $BENCH_MODEL \\"
echo "      --conditions with_skill no_skill self_gen --max-turns 25 \\"
echo "      --timeout 400 --retry-timeout 1200 --resume"
echo "========================================="
