"""aggregate_metrics.py — turn per-trial CSV into paper-ready summary tables.

Consumes benchmark/metrics_per_trial.csv (from compute_metrics.py) and
emits:

  benchmark/metrics_summary_by_condition.csv   — pass-rate + metric means
  benchmark/metrics_summary_by_tier.csv         — per-tier breakdown
  benchmark/metrics_paper_table.md              — markdown for the paper
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "benchmark" / "metrics_per_trial.csv"

METRIC_KEYS = [
    "path_gain_mae_db",
    "rss_grid_mae_db",
    "sinr_err_db",
    "ber_log_err",
    "bler_log_err",
    "throughput_re_pct",
]

METRIC_LABELS = {
    "path_gain_mae_db":  "Path-gain MAE (dB)",
    "rss_grid_mae_db":   "RSS grid MAE (dB)",
    "sinr_err_db":       "SINR error (dB)",
    "ber_log_err":       "BER log-err",
    "bler_log_err":      "BLER log-err",
    "throughput_re_pct": "Throughput RE (%)",
}


def _num(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _agg(vals: list[float]) -> dict[str, float | int]:
    """Return {n, mean, std, median, iqr_lo, iqr_hi} for a list of floats."""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = st.mean(vals)
    std = st.stdev(vals) if n > 1 else 0.0
    med = st.median(vals)
    vals_sorted = sorted(vals)
    q1 = vals_sorted[max(0, int(round(0.25 * (n - 1))))]
    q3 = vals_sorted[min(n - 1, int(round(0.75 * (n - 1))))]
    return {"n": n, "mean": mean, "std": std, "median": med,
            "iqr_lo": q1, "iqr_hi": q3}


def main():
    if not IN_CSV.exists():
        print(f"input {IN_CSV} not found — run compute_metrics.py first",
              file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(open(IN_CSV)))
    print(f"loaded {len(rows)} per-trial rows", file=sys.stderr)

    # Bucketize per (study, condition) and per (condition, tier)
    by_cond: dict[tuple, list[dict]] = defaultdict(list)
    by_cond_tier: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[(r["study"], r["condition"])].append(r)
        by_cond_tier[(r["study"], r["condition"], r["tier"])].append(r)

    # ----- summary by condition (per study) -----
    out_cond = ROOT / "benchmark" / "metrics_summary_by_condition.csv"
    with out_cond.open("w", newline="") as fh:
        keys = (["study", "condition", "n_trials", "n_pass", "pass_rate"] +
                [f"{m}_n" for m in METRIC_KEYS] +
                [f"{m}_mean" for m in METRIC_KEYS] +
                [f"{m}_std" for m in METRIC_KEYS] +
                [f"{m}_median" for m in METRIC_KEYS])
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for (study, cond), trials in sorted(by_cond.items()):
            row = {"study": study, "condition": cond,
                   "n_trials": len(trials)}
            passed = sum(1 for t in trials
                          if str(t.get("passed", "")).lower() == "true")
            row["n_pass"] = passed
            row["pass_rate"] = round(passed / len(trials), 4) if trials else 0
            for m in METRIC_KEYS:
                vals = [_num(t.get(m)) for t in trials]
                agg = _agg(vals)
                row[f"{m}_n"] = agg["n"]
                row[f"{m}_mean"] = round(agg["mean"], 4) if agg["n"] else None
                row[f"{m}_std"] = round(agg["std"], 4) if agg["n"] else None
                row[f"{m}_median"] = round(agg["median"], 4) if agg["n"] else None
            w.writerow(row)
    print(f"wrote {out_cond}", file=sys.stderr)

    # ----- summary by condition × tier (per study) -----
    out_ct = ROOT / "benchmark" / "metrics_summary_by_tier.csv"
    with out_ct.open("w", newline="") as fh:
        keys = (["study", "condition", "tier", "n_trials", "pass_rate"] +
                [f"{m}_mean" for m in METRIC_KEYS] +
                [f"{m}_n" for m in METRIC_KEYS])
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for (study, cond, tier), trials in sorted(by_cond_tier.items()):
            row = {"study": study, "condition": cond, "tier": tier,
                   "n_trials": len(trials)}
            passed = sum(1 for t in trials
                          if str(t.get("passed", "")).lower() == "true")
            row["pass_rate"] = round(passed / len(trials), 4) if trials else 0
            for m in METRIC_KEYS:
                vals = [_num(t.get(m)) for t in trials]
                agg = _agg(vals)
                row[f"{m}_n"] = agg["n"]
                row[f"{m}_mean"] = round(agg["mean"], 4) if agg["n"] else None
            w.writerow(row)
    print(f"wrote {out_ct}", file=sys.stderr)

    # ----- markdown paper table (aggregate ACROSS studies, per condition) -----
    # This is the main table for §Evaluation.
    by_cond_all: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_all[r["condition"]].append(r)

    # Metrics that are heavy-tailed → report median [IQR] instead of mean±std.
    HEAVY_TAILED = {"throughput_re_pct"}

    def fmt_cell(m, agg):
        if agg["n"] == 0:
            return "–"
        if m in HEAVY_TAILED:
            return (f"{agg['median']:.2f} [{agg['iqr_lo']:.2f}, "
                    f"{agg['iqr_hi']:.2f}] (n={agg['n']})")
        return f"{agg['mean']:.2f} ± {agg['std']:.2f} (n={agg['n']})"

    lines = []
    lines.append("# Continuous quantitative metrics — pooled across studies\n")
    lines.append("Per-trial values are compared against per-family oracles. "
                 "Only trials that produce the corresponding numeric field "
                 "are counted in `n`. Pass-rate is the binary verifier result. "
                 "Heavy-tailed metrics (marked ⋆) are reported as "
                 "**median [Q1, Q3]** to guard against unit-mismatch outliers.\n")
    hdr = "| Condition | Trials | Pass rate |"
    sep = "|-----------|-------:|----------:|"
    for m in METRIC_KEYS:
        tag = " ⋆" if m in HEAVY_TAILED else ""
        hdr += f" {METRIC_LABELS[m]}{tag} |"
        sep += "------:|"
    lines.append(hdr)
    lines.append(sep)
    for cond in sorted(by_cond_all.keys()):
        trials = by_cond_all[cond]
        passed = sum(1 for t in trials
                      if str(t.get("passed", "")).lower() == "true")
        pass_rate = passed / len(trials) if trials else 0
        cells = [cond, str(len(trials)), f"{pass_rate:.1%}"]
        for m in METRIC_KEYS:
            vals = [_num(t.get(m)) for t in trials]
            cells.append(fmt_cell(m, _agg(vals)))
        lines.append("| " + " | ".join(cells) + " |")

    # Unit-error footer: what fraction of throughput RE > 100%?
    lines.append("")
    lines.append("### Unit-error rate (throughput_re_pct > 100%)")
    lines.append("")
    lines.append("| Condition | Extreme outliers / Trials with metric | Rate |")
    lines.append("|---|---|---|")
    for cond in sorted(by_cond_all.keys()):
        vals = [_num(t.get("throughput_re_pct")) for t in by_cond_all[cond]]
        vals = [v for v in vals if v is not None]
        n_ext = sum(1 for v in vals if v > 100)
        rate = f"{100*n_ext/len(vals):.1f}%" if vals else "–"
        lines.append(f"| {cond} | {n_ext} / {len(vals)} | {rate} |")

    # Per-tier breakdown — only tiers that actually have metric coverage
    lines.append("\n## Per-tier breakdown (pooled across conditions)\n")
    hdr2 = "| Tier |"
    sep2 = "|------|"
    for m in METRIC_KEYS:
        tag = " ⋆" if m in HEAVY_TAILED else ""
        hdr2 += f" {METRIC_LABELS[m]}{tag} |"
        sep2 += "------:|"
    lines.append(hdr2)
    lines.append(sep2)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    for tier in sorted(by_tier.keys()):
        trials = by_tier[tier]
        # Skip tiers with zero numeric coverage across every metric
        any_metric = any(_num(t.get(m)) is not None for t in trials
                          for m in METRIC_KEYS)
        if not any_metric:
            continue
        cells = [tier]
        for m in METRIC_KEYS:
            vals = [_num(t.get(m)) for t in trials]
            cells.append(fmt_cell(m, _agg(vals)))
        lines.append("| " + " | ".join(cells) + " |")

    out_md = ROOT / "benchmark" / "metrics_paper_table.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
