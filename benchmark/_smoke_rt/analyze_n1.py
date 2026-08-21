"""Aggregate N1 full-run results into a paper-ready table.

Reads benchmark/results/n1_full_v10/*/N1_*/t*/result.json and emits:
  - Per (task × condition) pass rate
  - Per-condition aggregate
  - Failure mode breakdown (which check failed)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "benchmark" / "results" / "n1_full_v10"

SCENES = ["N1_box_two_screens", "N1_box_one_screen",
          "N1_simple_street_canyon", "N1_etoile"]
CONDS = ["with_skill", "no_skill", "self_gen"]


def main() -> int:
    if not RUN.exists():
        print(f"Run dir missing: {RUN}")
        return 1

    # cell[cond][task] = list of (passed, failed_check_names)
    cell: dict = defaultdict(lambda: defaultdict(list))

    for res in sorted(RUN.rglob("result.json")):
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        cond = r.get("condition")
        tid = r.get("task_id")
        if cond not in CONDS or tid not in SCENES:
            continue
        passed = bool(r.get("verification", {}).get("passed", False))
        failed_names = [
            c["name"] for c in r.get("verification", {}).get("checks", [])
            if not c.get("passed", True)
        ]
        cell[cond][tid].append((passed, failed_names))

    n_trials = max(
        (len(cell[c][t]) for c in CONDS for t in SCENES if cell[c][t]),
        default=0)

    print(f"=== N1 results ({RUN.name}) — {n_trials} trials per cell ===\n")
    # Per-task table
    header = f"{'Scene':25s} | " + " | ".join(f"{c:>13s}" for c in CONDS)
    print(header)
    print("-" * len(header))
    for tid in SCENES:
        row = [tid.replace("N1_", "").ljust(25)]
        for cond in CONDS:
            entries = cell[cond][tid]
            n_pass = sum(1 for p, _ in entries if p)
            n_tot = len(entries)
            row.append(f"{n_pass}/{n_tot} = {100*n_pass/max(1,n_tot):5.1f}%".rjust(13))
        print(" | ".join(row))

    # Aggregate per condition
    print()
    print("=== Aggregate (all 4 scenes pooled) ===")
    for cond in CONDS:
        all_pass = sum(1 for tid in SCENES for p, _ in cell[cond][tid] if p)
        all_tot = sum(len(cell[cond][tid]) for tid in SCENES)
        print(f"  {cond:15s}: {all_pass}/{all_tot} = "
              f"{100*all_pass/max(1,all_tot):.1f}%")

    # Failure-mode breakdown
    print()
    print("=== Failure modes per condition ===")
    for cond in CONDS:
        modes: dict[str, int] = defaultdict(int)
        n_trials_cond = 0
        for tid in SCENES:
            for passed, failed in cell[cond][tid]:
                n_trials_cond += 1
                if passed:
                    continue
                # Deduplicate within a trial (artifact:* appears 3x)
                seen = set()
                for name in failed:
                    short = name.split(":")[-1] if ":" in name else name
                    if short in seen:
                        continue
                    seen.add(short)
                    modes[short] += 1
        print(f"\n  {cond} ({n_trials_cond} trials):")
        for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
            print(f"    {mode:30s} {count:>4d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
