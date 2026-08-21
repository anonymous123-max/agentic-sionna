"""paper_appendix_table.py — build the paper-aligned supplementary table.

Reads:
  - benchmark/metrics_per_trial.csv       (from compute_metrics.py)
  - benchmark/paper_cell_map.json         (study → paper Table IV/VI cell)

Emits:
  - benchmark/paper_appendix_table.md     (markdown, drop into Appendix)

Structure mirrors paper Table IV / VI: each row = task_family × prompt_tier.
Per row × per condition we report:
  - N (trials on disk)
  - L1 pass %  (artifact / schema)
  - L2 pass %  (executable + sionna-used)
  - L3 pass %  (oracle tolerance — this equals the paper's number)
  - 4 continuous Layer-3 distances (only for tasks with a numeric oracle)

Only trials that produce structured output contribute to the numeric-
distance columns; crashed trials count in the pass-rate columns but not
in the continuous ones (n reported per cell).
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "benchmark" / "metrics_per_trial.csv"
MAP_JSON = ROOT / "benchmark" / "paper_cell_map.json"
OUT_MD = ROOT / "benchmark" / "paper_appendix_table.md"

CONDITIONS = ["no_skill", "self_gen", "with_skill"]
COND_LABEL = {"no_skill": "Naive", "self_gen": "Self-Written", "with_skill": "AutoNetSim"}

CONT_KEYS = ["path_gain_mae_db", "rss_grid_mae_db", "sinr_err_db",
             "ber_log_err", "throughput_re_pct"]
CONT_LABEL = {
    "path_gain_mae_db":  "Path-gain MAE (dB)",
    "rss_grid_mae_db":   "RSS MAE (dB)",
    "sinr_err_db":       "SINR err (dB)",
    "ber_log_err":       "BER log-err",
    "throughput_re_pct": "Thr RE (%)",
}
# Only these families produce each continuous metric — used to blank
# irrelevant cells so the table stays readable.
FAMILY_CONT_KEYS = {
    "N1": ["rss_grid_mae_db"],
    "N2": ["rss_grid_mae_db"],
    "N3": ["rss_grid_mae_db"],
    "N4": ["ber_log_err", "throughput_re_pct"],
    "P1": ["path_gain_mae_db", "throughput_re_pct"],
    "P2": ["path_gain_mae_db", "throughput_re_pct"],
    "S1": ["sinr_err_db", "throughput_re_pct"],
    "S2": ["path_gain_mae_db", "sinr_err_db", "throughput_re_pct"],
    "S3": ["sinr_err_db", "throughput_re_pct"],
    "S4": ["path_gain_mae_db", "throughput_re_pct"],
}


def _num(x):
    if x is None or x == "" or (isinstance(x, str) and x.lower() == "none"):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _bool(x):
    if isinstance(x, bool):
        return x
    if x is None or x == "" or (isinstance(x, str) and x.lower() == "none"):
        return None
    return str(x).lower() == "true"


def _rate(vals):
    """Return pass-rate % or None if all None."""
    v2 = [b for b in vals if b is not None]
    if not v2:
        return None
    return 100.0 * sum(1 for b in v2 if b) / len(v2)


def _median(vals):
    v2 = [v for v in vals if v is not None]
    return (st.median(v2), len(v2)) if v2 else (None, 0)


def fmt_pct(x):
    return f"{x:.1f}%" if x is not None else "–"


def fmt_cont(m, cell, tier):
    if m not in FAMILY_CONT_KEYS.get(tier, []):
        return "n/a"
    if cell["n"] == 0:
        return "–"
    return f"{cell['med']:.2f}"


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    paper_map = json.loads(MAP_JSON.read_text())
    paper_map = {k: v for k, v in paper_map.items() if not k.startswith("_")}

    # Index rows by (study, condition, tier)
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        idx[(r["study"], r["condition"], r["tier"])].append(r)

    md = []
    md.append("# Appendix X. Layered verification — aggregated view\n")
    md.append(
        "**Table X: Aggregated per-layer results for the same experiments "
        "as Tables IV and VI in the main paper.** For each task family and "
        "prompt tier, we report per-layer pass rates and, where applicable, "
        "the continuous Layer-3 distances (median across trials, computed "
        "on trials that produced structured output). The **L3 pass** column "
        "matches the AutoNetSim column of Tables IV / VI up to a small "
        "denominator difference due to crashed trials that leave no "
        "artifact on disk.\n")
    md.append(
        "Notation: `L1` = artifact & schema; `L2` = executable & Sionna-namespace "
        "call detected; `L3` = numerical output within the audited tolerance "
        "window. Continuous columns report medians across trials that produced "
        "the corresponding numeric field. `n/a` = the task family has no such "
        "oracle; `–` = no trial produced the field.\n")

    hdr = "| Section | Family / prompt | Condition | L1 | L2 | L3 (=paper) |"
    sep = "|---------|-----------------|-----------|---:|---:|------------:|"
    for m in CONT_KEYS:
        hdr += f" {CONT_LABEL[m]} |"
        sep += "-----:|"
    md.append(hdr)
    md.append(sep)

    # Order paper cells: Sim simple → Sim enhanced → Opt simple → Opt enhanced
    #                     → Sys simple → Sys enhanced
    order = [k for k in paper_map.keys()
             if "Sim_simple" in k or "Sim_" in k.replace("Sim_enhanced","Sim_ZZZ")]
    def order_key(k):
        weight = 0
        if k.startswith("Sim_simple"): weight = 0
        elif k.startswith("Sim_enhanced"): weight = 1
        elif k.startswith("Opt_simple"): weight = 2
        elif k.startswith("Opt_enhanced"): weight = 3
        elif k.startswith("Sys_simple"): weight = 4
        elif k.startswith("Sys_enhanced"): weight = 5
        return (weight, k)
    for cell_name in sorted(paper_map.keys(), key=order_key):
        cell = paper_map[cell_name]
        study, tier, prompt = cell["study"], cell["tier"], cell["prompt"]
        section = cell_name.split("_")[0]
        prompt_lab = f"{tier} / {prompt}"
        for cond in CONDITIONS:
            trials = idx.get((study, cond, tier), [])
            n = len(trials)
            l1 = _rate([_bool(t.get("L1_pass")) for t in trials])
            l2 = _rate([_bool(t.get("L2_pass")) for t in trials])
            l3 = _rate([_bool(t.get("L3_pass")) for t in trials])

            cont_cells = []
            for m in CONT_KEYS:
                vals = [_num(t.get(m)) for t in trials]
                med, n_med = _median(vals)
                if m not in FAMILY_CONT_KEYS.get(tier, []):
                    cont_cells.append("n/a")
                elif n_med == 0:
                    cont_cells.append("–")
                else:
                    cont_cells.append(f"{med:.2f}")

            md.append(
                "| " +
                " | ".join([
                    section,
                    prompt_lab,
                    COND_LABEL[cond],
                    fmt_pct(l1),
                    fmt_pct(l2),
                    fmt_pct(l3),
                    *cont_cells,
                ]) +
                " |"
            )

    # Add per-family summary of continuous metrics pooled across paper cells
    md.append("\n## Summary — continuous metrics pooled per condition\n")
    md.append("(Pooled across every paper cell above; heavy-tail metrics — "
              "path-gain MAE and throughput RE — reported as median.)\n")
    md.append("| Condition | Path-gain MAE (dB) | RSS MAE (dB) | "
              "SINR err (dB) | BER log-err | Thr RE (%) |")
    md.append("|---|---:|---:|---:|---:|---:|")

    paper_cell_studies = {(c["study"], c["tier"]) for c in paper_map.values()}
    for cond in CONDITIONS:
        pool = []
        for r in rows:
            if (r["study"], r["tier"]) in paper_cell_studies and r["condition"] == cond:
                pool.append(r)
        row = [COND_LABEL[cond]]
        for m in CONT_KEYS:
            vals = [_num(t.get(m)) for t in pool]
            med, n_med = _median(vals)
            if n_med == 0:
                row.append("–")
            else:
                row.append(f"{med:.2f}")
        md.append("| " + " | ".join(row) + " |")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
