#!/bin/bash
# run_heldout.sh — RUN ONCE, after iteration is complete.
#
# The 134-task benchmark is split 60/40 train (81) / test (53) in tasks.json.
# All v1.0–v1.5 iteration was done on the TRAIN split only. The TEST split is
# held out and evaluated EXACTLY ONCE per published skill version, after the
# auto-improvement loop has plateaued.
#
# This script intentionally has no arguments, no flags, and no resume mode:
# every safeguard against repeated test-set evaluation is by construction.
#
# Usage:
#   bash benchmark/run_heldout.sh <skill_version> <model>
# Example:
#   bash benchmark/run_heldout.sh v1.5 meta-llama/Llama-3.1-70B-Instruct
#
# After running, archive the results JSON to _studies_archive/ with the
# version tag and DO NOT use it to inform further skill edits.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash $0 <skill_version> <model>" >&2
  exit 1
fi
SKILL_VER="$1"
MODEL="$2"
LABEL="heldout_${SKILL_VER}_$(date -u +%Y-%m-%d)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmark/results/$LABEL"

if [[ -d "$RESULTS_DIR" ]]; then
  echo "ERROR: $RESULTS_DIR already exists." >&2
  echo "       Held-out evaluation MUST run exactly once per skill version." >&2
  echo "       If you intended to re-run, manually rename the existing dir first" >&2
  echo "       and document why in benchmark/_studies_archive/." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
python3 benchmark/run_benchmark.py \
    --label "$LABEL" \
    --shuffle-seed 42 \
    --timeout 400 \
    --retry-timeout 1200 \
    --split test \
    --conditions with_skill no_skill \
    --k 1 \
    --workers "${BENCH_WORKERS:-6}" \
    --max-turns 25 \
    --model "$MODEL"

echo
echo "=== HELD-OUT EVALUATION COMPLETE ==="
echo "Results: $RESULTS_DIR"
echo "Archive these results to _studies_archive/ and do NOT iterate the skill"
echo "based on them — that would invalidate the held-out evaluation."
