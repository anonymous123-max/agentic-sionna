#!/usr/bin/env python3
"""verify_output.py — agent-callable self-verification gate.

Thin wrapper around _verifier_core's plausibility checks. Run AFTER
your simulation has produced simulation_result.json (and any required .npy
/ .pt artifacts) and BEFORE you end your turn. Prints PASS/FAIL per check
and exits non-zero on any failure.

This wrapper deliberately delegates to the canonical verifier — no
duplicate logic. If the real benchmark verifier rejects your output,
this script will reject it too.

Usage:
    python3 verify_output.py                                    # check cwd
    python3 verify_output.py --workdir path/to/results          # check elsewhere
    python3 verify_output.py --required simulation_result.json coverage_map.npy

The required-artifacts check is local (the canonical verifier needs a
task spec; we don't have one here). All physical-plausibility checks
are delegated to `_verifier_core.check_plausibility()`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Locate the repo root (parents: scripts/ → rf-simulator/ → skills/ → .claude/ → repo)
# and add it to sys.path so `benchmark._verifier_core` is importable regardless
# of whether PYTHONPATH is set by the caller.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
from benchmark._verifier_core import check_plausibility, load_sim_result, check_tier5_domain  # noqa: E402


GREEN = "\033[32m"; RED = "\033[31m"; RESET = "\033[0m"


def status(ok: bool, name: str, detail: str = "") -> bool:
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def check_artifacts_exist(workdir: Path, required: list[str]) -> bool:
    """Local-only check: requested artifacts present + non-trivial size."""
    all_ok = True
    for name in required:
        p = workdir / name
        if not p.exists():
            all_ok &= status(False, f"artifact:{name}", "missing")
            continue
        size = p.stat().st_size
        if size == 0:
            all_ok &= status(False, f"artifact:{name}", "0 bytes")
        elif name.endswith((".pt", ".npy")) and size < 1024:
            all_ok &= status(True, f"artifact:{name}",
                             f"{size}B — placeholder?")
        else:
            all_ok &= status(True, f"artifact:{name}", f"{size}B")
    return all_ok


def check_skeleton_overwritten(workdir: Path) -> bool:
    """Local-only check: did the agent overwrite the harness pre-shipped JSON?

    The verifier doesn't surface this distinction (it just checks values),
    but it's the most common silent-fail mode on the agent side.
    """
    sim = load_sim_result(workdir)
    if sim is None:
        return status(False, "skeleton:exists", "simulation_result.json missing")
    if sim.get("status") == "placeholder_pre_shipped_by_harness":
        return status(False, "skeleton:overwritten",
                      "harness placeholder still in place — your simulation didn't run")
    return status(True, "skeleton:overwritten", "agent wrote real output")


def print_caveats(workdir: Path) -> None:
    """Print [Caveats] block — informational only, never fails the trial.

    Also fires a cheap heuristic: if the result looks analytical (ber_gap_db
    exactly 0.0, or a path-loss-range value looks like exact FSPL) but no
    'degraded' caveat was logged, emit a WARN line to prompt the agent to
    document the degradation.
    """
    sim = load_sim_result(workdir)
    if sim is None:
        return

    warnings_raw = sim.get("warnings", [])
    # Tolerate legacy plain-string warnings (pre-A1 schema)
    caveats = [w for w in warnings_raw if isinstance(w, dict)]

    if caveats:
        print(f"  {len(caveats)} caveat(s) reported:")
        for c in caveats:
            kind = c.get("kind", "?")
            src = c.get("source", "?")
            msg = c.get("message", "?")
            print(f"    {kind:<12} {src:<30} {msg}")
    else:
        print("  0 caveats — no fallbacks/defaults/degradations logged")

    # Analytical-shape heuristic (cheap, conservative)
    if sim.get("status") == "completed":
        has_degraded = any(c.get("kind") == "degraded" for c in caveats)
        if not has_degraded:
            nm = sim.get("numerical_metrics", {})
            # Exact 0.0 ber_gap_db is a strong signal of analytical FSPL path
            ber_gap = nm.get("ber_gap_db")
            # path_loss_range both elements identical → exact FSPL (no multipath)
            plr = nm.get("path_loss_range_db")
            plr_exact = (
                isinstance(plr, (list, tuple))
                and len(plr) == 2
                and plr[0] is not None
                and plr[0] == plr[1]
            )
            if ber_gap == 0.0 or plr_exact:
                print("  WARN: result looks analytical but no degraded caveat logged")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--workdir", default=".",
                    help="Directory containing simulation outputs (default: cwd)")
    ap.add_argument("--required", nargs="*", default=["simulation_result.json"],
                    help="Required artifact filenames")
    ap.add_argument("--capability", default=None,
                    help="Optional task capability for tier-5 domain checks "
                         "(e.g., channel_charting, isac_tradeoff_curve)")
    args = ap.parse_args()

    wd = Path(args.workdir).resolve()
    print(f"Verifying outputs in {wd}\n")

    overall = True

    # 1. Local: artifact presence + sizes
    print("[Artifact presence]")
    overall &= check_artifacts_exist(wd, args.required)
    print()

    # 2. Local: skeleton overwrite (agent-side concern)
    print("[Skeleton overwritten]")
    overall &= check_skeleton_overwritten(wd)
    print()

    # 3. Caveats — informational only; never fails the trial
    print("[Caveats]")
    print_caveats(wd)
    print()

    # 4. Delegated: every plausibility check the real verifier runs.
    print("[Physical plausibility — delegated to _verifier_core]")
    plaus = check_plausibility(wd)
    if not plaus:
        print(f"  (no plausibility flags — output looks physically reasonable)")
    else:
        for c in plaus:
            overall &= status(c.passed, c.name, c.detail)
    print()

    if args.capability:
        print(f"[Tier-5 domain checks: {args.capability}]")
        domain = check_tier5_domain(args.capability, wd)
        if not domain:
            print(f"  (no domain checks registered for "
                  f"capability={args.capability})")
        else:
            for c in domain:
                overall &= status(c.passed, c.name, c.detail)
        print()

    print(f"=== {'OVERALL: PASS' if overall else 'OVERALL: FAIL'} ===")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
