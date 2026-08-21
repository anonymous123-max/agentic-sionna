"""Aggregate N1 + N2 results into paper-ready tables."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "N1 (single-AP coverage)":  {
        "dir": ROOT / "benchmark" / "results" / "n1_full_v10",
        "task_ids": ["N1_box_two_screens", "N1_box_one_screen",
                     "N1_simple_street_canyon", "N1_etoile"],
    },
    "N2 (5 → 2.4 GHz edit)":  {
        "dir": ROOT / "benchmark" / "results" / "n2_full_v10",
        "task_ids": ["N2_box_two_screens", "N2_box_one_screen",
                     "N2_simple_street_canyon", "N2_etoile"],
    },
}
CONDS = ["with_skill", "no_skill", "self_gen"]


def analyze(name: str, run_dir: Path, task_ids: list[str]) -> dict:
    cell: dict = defaultdict(lambda: defaultdict(list))
    if not run_dir.exists():
        print(f"  [skip] {name}: no run at {run_dir}")
        return {}
    for res in sorted(run_dir.rglob("result.json")):
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        cond, tid = r.get("condition"), r.get("task_id")
        if cond not in CONDS or tid not in task_ids:
            continue
        passed = bool(r.get("verification", {}).get("passed", False))
        failed_names = [c["name"]
                        for c in r.get("verification", {}).get("checks", [])
                        if not c.get("passed", True)]
        cell[cond][tid].append((passed, failed_names))

    print(f"\n=== {name}   ({run_dir.name}) ===")
    header = f"{'Scene':28s} | " + " | ".join(f"{c:>13s}" for c in CONDS)
    print(header)
    print("-" * len(header))
    for tid in task_ids:
        short = tid.split("_", 1)[-1]
        row = [short.ljust(28)]
        for cond in CONDS:
            entries = cell[cond][tid]
            n_pass = sum(1 for p, _ in entries if p)
            n_tot = len(entries)
            row.append(f"{n_pass}/{n_tot} = {100*n_pass/max(1,n_tot):5.1f}%".rjust(13))
        print(" | ".join(row))

    # Aggregate
    print(f"\n  Aggregate (all {len(task_ids)} scenes pooled):")
    aggregate = {}
    for cond in CONDS:
        all_pass = sum(1 for tid in task_ids for p, _ in cell[cond][tid] if p)
        all_tot = sum(len(cell[cond][tid]) for tid in task_ids)
        aggregate[cond] = (all_pass, all_tot)
        print(f"    {cond:15s}: {all_pass}/{all_tot} = {100*all_pass/max(1,all_tot):.1f}%")

    # Δ_skill
    if all(cond in aggregate for cond in CONDS):
        ws_pct = 100*aggregate["with_skill"][0]/max(1, aggregate["with_skill"][1])
        ns_pct = 100*aggregate["no_skill"][0]/max(1, aggregate["no_skill"][1])
        sg_pct = 100*aggregate["self_gen"][0]/max(1, aggregate["self_gen"][1])
        print(f"\n  Δ_skill (with_skill − no_skill) = {ws_pct-ns_pct:+.1f} pp")
        print(f"  Δ_skill (with_skill − self_gen) = {ws_pct-sg_pct:+.1f} pp")

    # Failure modes
    print("\n  Failure modes:")
    for cond in CONDS:
        modes: dict[str, int] = defaultdict(int)
        for tid in task_ids:
            for passed, failed in cell[cond][tid]:
                if passed:
                    continue
                seen = set()
                for nm in failed:
                    short = nm.split(":")[-1] if ":" in nm else nm
                    if short in seen:
                        continue
                    seen.add(short)
                    modes[short] += 1
        if modes:
            print(f"    {cond}:")
            for m, c in sorted(modes.items(), key=lambda kv: -kv[1]):
                print(f"      {m:35s} {c}")

    return aggregate


def main() -> int:
    print("Re-verifying all trials before aggregation...")
    import sys
    sys.path.insert(0, str(ROOT))
    from benchmark.verifier import verify

    # Re-verify N1 and N2 with current verifier
    for name, cfg in RUNS.items():
        run_dir = cfg["dir"]
        if not run_dir.exists():
            continue
        # Find the tasks file
        if "n1" in run_dir.name:
            tfile = ROOT / "benchmark" / "tasks" / "_sources" / "n1_coverage.json"
        elif "n2" in run_dir.name:
            tfile = ROOT / "benchmark" / "tasks" / "_sources" / "n2_freq_edit.json"
        else:
            continue
        tasks_by_id = {t["id"]: t for t in json.loads(tfile.read_text())["tasks"]}
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

    all_aggregates = {}
    for name, cfg in RUNS.items():
        agg = analyze(name, cfg["dir"], cfg["task_ids"])
        if agg:
            all_aggregates[name] = agg

    print("\n" + "=" * 80)
    print("HEADLINE TABLE")
    print("=" * 80)
    print(f"{'Task':30s} | {'with_skill':>13s} | {'no_skill':>13s} | "
          f"{'self_gen':>13s} | {'Δ vs ns':>9s}")
    print("-" * 95)
    for name, agg in all_aggregates.items():
        ws, ws_t = agg["with_skill"]
        ns, ns_t = agg["no_skill"]
        sg, sg_t = agg["self_gen"]
        ws_p = 100*ws/max(1, ws_t)
        ns_p = 100*ns/max(1, ns_t)
        sg_p = 100*sg/max(1, sg_t)
        print(f"{name:30s} | {ws}/{ws_t} = {ws_p:5.1f}% | "
              f"{ns}/{ns_t} = {ns_p:5.1f}% | "
              f"{sg}/{sg_t} = {sg_p:5.1f}% | "
              f"{ws_p-ns_p:+6.1f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
