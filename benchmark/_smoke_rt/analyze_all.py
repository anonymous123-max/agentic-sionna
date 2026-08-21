"""Aggregate N1 v2 + N2 v2 + N3 v1 + N4 v1 + P1 v1 + P2 v1 into the full headline."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.verifier import verify

RUNS = [
    ("N1 v2", "n1_v2_full",  "n1_v2.json",        "sub_type"),
    ("N2 v2", "n2_v2_full",  "n2_v2.json",        "edit_type"),
    ("N3 v1", "n3_v1_full",  "n3_v1.json",        None),
    ("N4 v1", "n4_v1_full",  "n4_v1.json",        "metric_type"),
    ("P1 v2", "p1_v1_v2_full",  "p1_v1.json",     "scene_name"),
    ("P2 v2", "p2_v1_v2_full",  "p2_v1.json",     "scene_name"),
    ("S  v2", "s_v2_full",      "s_v2.json",      "capability"),
]
CONDS = ["with_skill", "no_skill", "self_gen"]


def analyze(name: str, results_subdir: str, tasks_filename: str,
            row_field: str | None) -> dict:
    results_dir = ROOT / "benchmark" / "results" / results_subdir
    tasks_file = ROOT / "benchmark" / "tasks" / "_sources" / tasks_filename
    tasks_by_id = {t["id"]: t for t in json.loads(tasks_file.read_text())["tasks"]}

    # Re-verify everything
    for res in sorted(results_dir.rglob("result.json")):
        try: r = json.loads(res.read_text())
        except: continue
        task = tasks_by_id.get(r.get("task_id"))
        if not task: continue
        v = verify(task, res.parent, exec_success=r.get("exec_success", True))
        r["verification"] = {
            "passed": v.passed, "score": v.score,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in v.checks]}
        res.write_text(json.dumps(r, indent=2))

    cell = defaultdict(lambda: defaultdict(list))
    for res in sorted(results_dir.rglob("result.json")):
        r = json.loads(res.read_text())
        cond, tid = r.get("condition"), r.get("task_id")
        if cond not in CONDS or tid not in tasks_by_id: continue
        passed = bool(r.get("verification", {}).get("passed", False))
        failed = [c["name"] for c in r.get("verification", {}).get("checks", [])
                  if not c.get("passed", True)]
        cell[cond][tid].append((passed, failed))

    print(f"\n=== {name} ===")
    label_field = row_field if row_field else "scene_name"
    rows = []
    totals = {c: [0, 0] for c in CONDS}
    for tid, task in tasks_by_id.items():
        label = task.get(label_field, "") + " " + task.get("scene_name", "")
        label = label.strip()
        row = [label.ljust(36)]
        for cond in CONDS:
            entries = cell[cond].get(tid, [])
            n_pass = sum(1 for p, _ in entries if p); n_tot = len(entries)
            totals[cond][0] += n_pass; totals[cond][1] += n_tot
            row.append(f"{n_pass}/{n_tot}={100*n_pass/max(1,n_tot):5.1f}%".rjust(13))
        print(" | ".join(row))

    print("  Aggregate:")
    for cond in CONDS:
        n, t = totals[cond]
        print(f"    {cond:12s} {n}/{t} = {100*n/max(1,t):.1f}%")
    # Failure modes
    print("  Failure modes:")
    for cond in CONDS:
        modes = defaultdict(int)
        for tid in tasks_by_id:
            for passed, failed in cell[cond].get(tid, []):
                if passed: continue
                seen = set()
                for nm in failed:
                    short = nm.split(":")[-1] if ":" in nm else nm
                    if short in seen: continue
                    seen.add(short)
                    modes[short] += 1
        if modes:
            top = sorted(modes.items(), key=lambda kv: -kv[1])
            top_str = ", ".join(f"{m}={c}" for m, c in top[:5])
            print(f"    {cond}: {top_str}")

    return totals


def main():
    headline = {}
    for name, sub, tf, rf in RUNS:
        headline[name] = analyze(name, sub, tf, rf)

    print("\n" + "=" * 84)
    print("HEADLINE — all tasks")
    print("=" * 84)
    print(f"{'Task':12s} | {'with_skill':>13s} | {'no_skill':>13s} | "
          f"{'self_gen':>13s} | {'Δ vs ns':>9s}")
    print("-" * 80)
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
    main()
