"""Three-condition comparison report for the T0 redesign experiment.

Walks benchmark/results/<label>/{with_skill,no_skill,self_gen}/*/t*/result.json
and computes:

  - overall pass_strict per condition + mean score
  - per-difficulty (easy/hard) breakdown
  - per-capability breakdown
  - per-task summary (mean across k trials, useful for failure inspection)
  - token usage + wall time per condition (cost view)
  - paired deltas: with_skill - no_skill (skill effect) and
    with_skill - self_gen (vs free-form context)

Usage:
  python3 benchmark/analysis/t0v2_report.py --label t0v2_b
  python3 benchmark/analysis/t0v2_report.py --label t0v2_b --out report.md
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark" / "results"


def load_trials(label: str) -> list[dict]:
    base = RESULTS / label
    rows = []
    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        rows.append(d)
    return rows


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate(rows: list[dict]) -> dict:
    """Returns a structured report dict."""
    # Per-condition aggregates
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r.get("condition", "?")].append(r)

    overall = {}
    for cond, rs in by_cond.items():
        passes = [1 if r.get("pass_strict") else 0 for r in rs]
        scores = [r.get("verification", {}).get("score", 0.0) for r in rs]
        in_tok = [r.get("usage", {}).get("input_tokens", 0) for r in rs]
        out_tok = [r.get("usage", {}).get("output_tokens", 0) for r in rs]
        wall = [r.get("wall_sec", 0.0) for r in rs]
        overall[cond] = {
            "n": len(rs),
            "pass_rate_pct": 100.0 * sum(passes) / len(rs) if rs else 0.0,
            "score_mean": statistics.mean(scores) if scores else 0.0,
            "input_tokens_mean": statistics.mean(in_tok) if in_tok else 0.0,
            "output_tokens_mean": statistics.mean(out_tok) if out_tok else 0.0,
            "wall_sec_mean": statistics.mean(wall) if wall else 0.0,
        }

    # Per-difficulty
    by_diff: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        key = (r.get("condition", "?"), r.get("difficulty", "?"))
        by_diff[key].append(1 if r.get("pass_strict") else 0)
    diff_table: dict[str, dict[str, float]] = defaultdict(dict)
    for (cond, diff), ps in by_diff.items():
        diff_table[diff][cond] = 100.0 * sum(ps) / len(ps) if ps else 0.0

    # Per-capability
    by_cap: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        key = (r.get("condition", "?"), r.get("capability", "?"))
        by_cap[key].append(1 if r.get("pass_strict") else 0)
    cap_table: dict[str, dict[str, float]] = defaultdict(dict)
    for (cond, cap), ps in by_cap.items():
        cap_table[cap][cond] = 100.0 * sum(ps) / len(ps) if ps else 0.0

    # Per-task: pass rate over k trials, per condition
    by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    task_meta: dict[str, dict] = {}
    for r in rows:
        tid = r.get("task_id", "?")
        cond = r.get("condition", "?")
        by_task[(tid, cond)].append(1 if r.get("pass_strict") else 0)
        task_meta[tid] = {
            "capability": r.get("capability", "?"),
            "difficulty": r.get("difficulty", "?"),
        }

    return {
        "overall": overall,
        "by_difficulty": dict(diff_table),
        "by_capability": dict(cap_table),
        "by_task": {f"{t}|{c}": (sum(v) / len(v) if v else 0.0, len(v))
                    for (t, c), v in by_task.items()},
        "task_meta": task_meta,
    }


CONDITIONS = ["no_skill", "with_skill", "self_gen"]


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table."""
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
              for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(r[i]).ljust(widths[i])
                              for i in range(len(headers))) + " |"
            for r in rows]
    return "\n".join([line, sep] + body)


def render_report(label: str, agg: dict) -> str:
    out = [f"# T0 Redesign Benchmark Report — `{label}`\n"]

    # === Overall ===
    out.append("## Overall (all 60 train tasks × k trials)\n")
    headers = ["Condition", "n", "pass_strict%", "mean_score", "in_tok",
               "out_tok", "wall_s"]
    rows = []
    for cond in CONDITIONS:
        if cond not in agg["overall"]:
            continue
        o = agg["overall"][cond]
        rows.append([cond, o["n"], f"{o['pass_rate_pct']:.1f}",
                     f"{o['score_mean']:.3f}",
                     f"{o['input_tokens_mean']:.0f}",
                     f"{o['output_tokens_mean']:.0f}",
                     f"{o['wall_sec_mean']:.0f}"])
    out.append(fmt_table(headers, rows) + "\n")

    # === Deltas ===
    if "with_skill" in agg["overall"] and "no_skill" in agg["overall"]:
        ws = agg["overall"]["with_skill"]["pass_rate_pct"]
        ns = agg["overall"]["no_skill"]["pass_rate_pct"]
        out.append(f"**Δ skill effect (with_skill − no_skill): {ws - ns:+.1f} pp**\n")
    if "with_skill" in agg["overall"] and "self_gen" in agg["overall"]:
        ws = agg["overall"]["with_skill"]["pass_rate_pct"]
        sg = agg["overall"]["self_gen"]["pass_rate_pct"]
        out.append(f"**Δ curated vs self_gen: {ws - sg:+.1f} pp**\n")

    # === Per difficulty ===
    out.append("\n## By difficulty\n")
    headers = ["Difficulty"] + CONDITIONS + ["Δ_skill", "Δ_vs_self_gen"]
    rows = []
    for diff in sorted(agg["by_difficulty"].keys()):
        row_data = agg["by_difficulty"][diff]
        ns = row_data.get("no_skill", 0.0)
        ws = row_data.get("with_skill", 0.0)
        sg = row_data.get("self_gen", 0.0)
        rows.append([
            diff,
            f"{ns:.1f}", f"{ws:.1f}", f"{sg:.1f}",
            f"{ws - ns:+.1f}", f"{ws - sg:+.1f}",
        ])
    out.append(fmt_table(headers, rows) + "\n")

    # === Per capability ===
    out.append("\n## By capability\n")
    headers = ["Capability"] + CONDITIONS + ["Δ_skill", "n_tasks"]
    rows = []
    for cap in sorted(agg["by_capability"].keys()):
        row_data = agg["by_capability"][cap]
        ns = row_data.get("no_skill", 0.0)
        ws = row_data.get("with_skill", 0.0)
        sg = row_data.get("self_gen", 0.0)
        # task count = unique task_ids with this capability
        ntasks = sum(1 for k, v in agg["task_meta"].items() if v["capability"] == cap)
        rows.append([
            cap,
            f"{ns:.1f}", f"{ws:.1f}", f"{sg:.1f}",
            f"{ws - ns:+.1f}", str(ntasks),
        ])
    out.append(fmt_table(headers, rows) + "\n")

    # === Hardest tasks (where with_skill underperforms) ===
    out.append("\n## Per-task drilldown — with_skill regressions vs no_skill\n")
    headers = ["task_id", "capability", "difficulty", "no_skill", "with_skill",
               "self_gen", "Δ_skill"]
    rows = []
    task_ids = {k.split("|")[0] for k in agg["by_task"].keys()}
    for tid in sorted(task_ids):
        meta = agg["task_meta"].get(tid, {})
        rates = {}
        for cond in CONDITIONS:
            key = f"{tid}|{cond}"
            if key in agg["by_task"]:
                rate, _ = agg["by_task"][key]
                rates[cond] = rate
            else:
                rates[cond] = math.nan
        ns = rates.get("no_skill", math.nan)
        ws = rates.get("with_skill", math.nan)
        sg = rates.get("self_gen", math.nan)
        delta = ws - ns if not math.isnan(ws) and not math.isnan(ns) else math.nan
        # only print rows where with_skill regressed (delta < 0) OR pass < 1
        if (not math.isnan(delta) and delta < 0) or ws < 1.0:
            rows.append([
                tid, meta.get("capability", "?"), meta.get("difficulty", "?"),
                f"{ns:.2f}" if not math.isnan(ns) else "—",
                f"{ws:.2f}" if not math.isnan(ws) else "—",
                f"{sg:.2f}" if not math.isnan(sg) else "—",
                f"{delta:+.2f}" if not math.isnan(delta) else "—",
            ])
    if rows:
        out.append(fmt_table(headers, rows) + "\n")
    else:
        out.append("(no regressions — with_skill perfect on every task)\n")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None,
                    help="Output markdown path. Default: print to stdout.")
    args = ap.parse_args()

    rows = load_trials(args.label)
    print(f"# Loaded {len(rows)} trials from {RESULTS / args.label}", flush=True)
    agg = aggregate(rows)
    report = render_report(args.label, agg)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote report → {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
