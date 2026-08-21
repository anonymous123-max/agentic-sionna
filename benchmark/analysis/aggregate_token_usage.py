"""Aggregate per-trial token usage from proxy_usage_*.jsonl.

Reads result.json files under a results dir. For each trial, computes the
trial's time window (start = workdir mtime of the EARLIEST file in t1/,
end = mtime of result.json). Reads all proxy_usage_*.jsonl in the
proxy-log-dir (default /workspace/logs). Sums prompt/completion tokens for
records whose `ts` falls in the trial window. Writes back to result.json
under "usage_from_proxy":
    {"prompt_tokens": N, "completion_tokens": N,
     "total_tokens": N, "n_calls": N}

The existing `usage` field is left untouched — `usage_from_proxy` is
additive so we can compare openclaude's (broken) view against the
ground-truth proxy view.

WHY THIS EXISTS
---------------
vLLM responses include OpenAI-format `prompt_tokens`/`completion_tokens`,
but openclaude doesn't translate them to Claude-format
`input_tokens`/`output_tokens`. As a result, every result.json on the
current vast.ai run has usage.input_tokens=0/output_tokens=0, which
breaks paper-grade token metrics. The proxy now logs every upstream
response's usage block to a sidecar JSONL; this script joins those
records back to trials.

CAVEAT: time-window matching is APPROXIMATE per-trial. When trials run
concurrently on the same proxy port, their windows OVERLAP and a given
proxy record may be attributed to multiple trials. Per-trial counts are
therefore best-effort. The SUM across all trials is still exact (modulo
records outside any window). Suitable for paper aggregates (model-level
total tokens, mean tokens/trial), good-enough for per-trial inspection.

Usage:
    python3 benchmark/analysis/aggregate_token_usage.py \
        --results-dir benchmark/results/train_Qwen_Qwen3_6_27B_chunk0 \
        --proxy-log-dir /workspace/logs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_proxy_records(proxy_log_dir: Path) -> list[dict]:
    """Read every proxy_usage_*.jsonl in `proxy_log_dir`. Returns a list
    of records sorted by `ts`. Skips malformed lines silently."""
    records: list[dict] = []
    if not proxy_log_dir.is_dir():
        print(f"[aggregate] proxy log dir not found: {proxy_log_dir}",
              file=sys.stderr)
        return records
    for path in sorted(proxy_log_dir.glob("proxy_usage_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "ts" not in rec:
                        continue
                    records.append(rec)
        except OSError as e:
            print(f"[aggregate] cannot read {path}: {e}", file=sys.stderr)
    records.sort(key=lambda r: r.get("ts", 0))
    return records


def _trial_window(result_json_path: Path) -> tuple[float, float] | None:
    """Compute (start_ts, end_ts) for a trial.

    start = min mtime of files inside the trial's t1/ directory, EXCLUDING
            result.json itself (which is written last).
    end   = mtime of result.json.

    Returns None if the trial dir layout is unrecognized.
    """
    trial_dir = result_json_path.parent  # …/<trial>/t1/
    if not trial_dir.is_dir():
        return None
    end_ts = result_json_path.stat().st_mtime
    earliest: float | None = None
    for child in trial_dir.rglob("*"):
        if child == result_json_path:
            continue
        if not child.is_file():
            continue
        try:
            mt = child.stat().st_mtime
        except OSError:
            continue
        if earliest is None or mt < earliest:
            earliest = mt
    if earliest is None:
        # No other files — fall back to result.json mtime as both bounds.
        earliest = end_ts
    return (earliest, end_ts)


def _sum_in_window(records: list[dict],
                   start_ts: float, end_ts: float) -> dict:
    """Sum prompt/completion/total tokens across records whose `ts` is
    in [start_ts, end_ts]. Records are pre-sorted by ts so we could
    bisect, but linear scan is fine for the volumes we expect."""
    p = c = t = 0
    n = 0
    for rec in records:
        ts = rec.get("ts", 0)
        if ts < start_ts:
            continue
        if ts > end_ts:
            # Records are sorted; everything after is also out of window.
            break
        p += int(rec.get("prompt_tokens", 0) or 0)
        c += int(rec.get("completion_tokens", 0) or 0)
        t += int(rec.get("total_tokens", 0) or 0)
        n += 1
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": t,
        "n_calls": n,
    }


def _walk_results(results_dir: Path) -> list[Path]:
    """Find every result.json under `results_dir`."""
    return sorted(results_dir.rglob("result.json"))


def aggregate(results_dir: Path, proxy_log_dir: Path,
              dry_run: bool = False, in_place: bool = True) -> int:
    """Walk `results_dir`, attribute proxy usage to each trial, write back.
    Returns count of result.json files updated."""
    records = _load_proxy_records(proxy_log_dir)
    print(f"[aggregate] loaded {len(records)} proxy usage records "
          f"from {proxy_log_dir}", file=sys.stderr)

    result_paths = _walk_results(results_dir)
    print(f"[aggregate] found {len(result_paths)} result.json files "
          f"under {results_dir}", file=sys.stderr)

    updated = 0
    for rp in result_paths:
        win = _trial_window(rp)
        if win is None:
            continue
        start_ts, end_ts = win
        usage = _sum_in_window(records, start_ts, end_ts)
        try:
            with open(rp) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[aggregate] skip {rp}: {e}", file=sys.stderr)
            continue
        data["usage_from_proxy"] = usage
        if dry_run or not in_place:
            print(f"[dry-run] {rp}: {usage}")
            continue
        try:
            tmp = rp.with_suffix(rp.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(rp)
            updated += 1
        except OSError as e:
            print(f"[aggregate] cannot write {rp}: {e}", file=sys.stderr)
    print(f"[aggregate] updated {updated} result.json files",
          file=sys.stderr)
    return updated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Root or chunk directory containing result.json files")
    ap.add_argument("--proxy-log-dir", type=Path,
                    default=Path("/workspace/logs"),
                    help="Directory holding proxy_usage_*.jsonl files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print computed usage but don't modify result.json")
    ap.add_argument("--in-place", dest="in_place",
                    action="store_true", default=True,
                    help="Write usage_from_proxy back to result.json (default)")
    ap.add_argument("--no-in-place", dest="in_place", action="store_false",
                    help="Disable writeback; only print")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"[aggregate] results-dir not found: {args.results_dir}",
              file=sys.stderr)
        return 2

    aggregate(args.results_dir, args.proxy_log_dir,
              dry_run=args.dry_run, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    sys.exit(main())
