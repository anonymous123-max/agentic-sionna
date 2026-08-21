"""Aggregate N1 v2 + N2 v2 results into a paper-ready table."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "N1 v2": {
        "results_dir": ROOT / "benchmark" / "results" / "n1_v2_full",
        "tasks_file":  ROOT / "benchmark" / "tasks" / "_sources" / "n1_v2.json",
        "title": "Single-AP simulation (coverage + link probe)",
        "row_label": lambda t: f"{t.get('sub_type','?'):8s} {t['scene_name']}",
    },
    "N2 v2": {
        "results_dir": ROOT / "benchmark" / "results" / "n2_v2_full",
        "tasks_file":  ROOT / "benchmark" / "tasks" / "_sources" / "n2_v2.json",
        "title": "AP configuration edit + before/after rerun",
        "row_label": lambda t: f"{t.get('edit_type','?'):10s} {t['scene_name']}",
    },
}
CONDS = ["with_skill", "no_skill", "self_gen"]


def analyze(name: str, cfg: dict) -> dict:
    run_dir = cfg["results_dir"]
    tasks_by_id = {t["id"]: t
                   for t in json.loads(cfg["tasks_file"].read_text())["tasks"]}

    # Re-verify with current verifier
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from benchmark.verifier import verify
    for res in sorted(run_dir.rglob("result.json")):
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        task = tasks_by_id.get(r.get("task_id"))
        if not task:
            continue
        v = verify(task, res.parent, exec_success=r.get("exec_success", True))
        r["verification"] = {
            "passed": v.passed, "score": v.score,
            "checks": [{"name": c.name, "passed": c.passed,
                        "detail": c.detail} for c in v.checks],
        }
        res.write_text(json.dumps(r, indent=2))

    # Aggregate
    cell = defaultdict(lambda: defaultdict(list))
    for res in sorted(run_dir.rglob("result.json")):
        r = json.loads(res.read_text())
        cond = r.get("condition")
        tid = r.get("task_id")
        if cond not in CONDS or tid not in tasks_by_id:
            continue
        passed = bool(r.get("verification", {}).get("passed", False))
        failed_names = [c["name"] for c in
                        r.get("verification", {}).get("checks", [])
                        if not c.get("passed", True)]
        cell[cond][tid].append((passed, failed_names))

    print(f"\n=== {name}: {cfg['title']} ===")
    label_w = max(len(cfg["row_label"](t)) for t in tasks_by_id.values()) + 2
    header = f"{'Task':{label_w}s} | " + " | ".join(f"{c:>13s}" for c in CONDS)
    print(header)
    print("-" * len(header))
    totals = {c: [0, 0] for c in CONDS}
    for tid, task in tasks_by_id.items():
        label = cfg["row_label"](task)
        row = [label.ljust(label_w)]
        for cond in CONDS:
            entries = cell[cond][tid]
            n_pass = sum(1 for p, _ in entries if p)
            n_tot = len(entries)
            totals[cond][0] += n_pass
            totals[cond][1] += n_tot
            row.append(f"{n_pass}/{n_tot} = {100*n_pass/max(1,n_tot):5.1f}%".rjust(13))
        print(" | ".join(row))

    agg_row = [f"{'Aggregate':{label_w}s}"]
    for cond in CONDS:
        np_, nt_ = totals[cond]
        agg_row.append(f"{np_}/{nt_} = {100*np_/max(1,nt_):5.1f}%".rjust(13))
    print(" | ".join(agg_row))

    if all(c in totals for c in CONDS):
        ws_p = 100 * totals["with_skill"][0] / max(1, totals["with_skill"][1])
        ns_p = 100 * totals["no_skill"][0] / max(1, totals["no_skill"][1])
        sg_p = 100 * totals["self_gen"][0] / max(1, totals["self_gen"][1])
        print(f"\n  Δ_skill (with_skill − no_skill) = {ws_p - ns_p:+.1f} pp")
        print(f"  Δ_skill (with_skill − self_gen) = {ws_p - sg_p:+.1f} pp")

    print(f"\n  Failure modes per condition:")
    for cond in CONDS:
        modes = defaultdict(int)
        for tid in tasks_by_id:
            for passed, failed in cell[cond][tid]:
                if passed: continue
                seen = set()
                for nm in failed:
                    short = nm.split(":")[-1] if ":" in nm else nm
                    if short in seen: continue
                    seen.add(short)
                    modes[short] += 1
        if modes:
            print(f"    {cond}:")
            for m, c in sorted(modes.items(), key=lambda kv: -kv[1]):
                print(f"      {m:35s} {c}")
    return totals


def main():
    headline = {}
    for name, cfg in RUNS.items():
        headline[name] = analyze(name, cfg)

    print("\n" + "=" * 80)
    print("HEADLINE")
    print("=" * 80)
    print(f"{'Task':12s} | {'with_skill':>13s} | {'no_skill':>13s} | "
          f"{'self_gen':>13s} | {'Δ vs ns':>9s}")
    print("-" * 78)
    for name, totals in headline.items():
        ws, ws_t = totals["with_skill"]
        ns, ns_t = totals["no_skill"]
        sg, sg_t = totals["self_gen"]
        ws_p = 100 * ws / max(1, ws_t)
        ns_p = 100 * ns / max(1, ns_t)
        sg_p = 100 * sg / max(1, sg_t)
        print(f"{name:12s} | {ws}/{ws_t} = {ws_p:5.1f}% | "
              f"{ns}/{ns_t} = {ns_p:5.1f}% | "
              f"{sg}/{sg_t} = {sg_p:5.1f}% | "
              f"{ws_p - ns_p:+6.1f}pp")


if __name__ == "__main__":
    sys.exit(main())
