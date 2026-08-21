"""improvement_loop.py — orchestrator for the auto-improvement loop.

Implements the master guide Part 7 flow:
    SKILL v_n
        ↓ run train suite (3 conds × N tasks)
        ↓ classify failures
        ↓ pick dominant failure class → propose ONE targeted edit
        ↓ apply edit, re-run
        ↓ if regressed → revert; if improved < 2pp → investigate; else → continue
    until plateau or 5 iterations
        ↓ run held-out (once)
        ↓ archive

CURRENT STATE: scaffold. The pipeline shell + classification + proposal
formatter are implemented. The actual edit is left as a manual diff
proposed to the user — fully autonomous edit-and-commit is opt-in via
`--auto-apply` once we trust the classifier.

Usage:
    python3 benchmark/improvement_loop.py \\
        --skill-version v1.5 \\
        --model meta-llama/Llama-3.1-70B-Instruct \\
        --max-iterations 5 \\
        [--auto-apply]    # default: dry-run; only print proposed edits
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────
# Failure classification taxonomy (master guide Part 9)
# ────────────────────────────────────────────────────────────────────

FAILURE_CLASSES = {
    "skill_not_consulted": "Agent proceeded without referencing skill",
    "skill_consulted_ignored": "Reads skill, then violates it",
    "skill_content_wrong": "Follows skill but skill is incorrect",
    "skill_gap": "Struggles with topic the skill doesn't cover",
    "model_capability_ceiling": "Understands what to do, cannot execute",
    "environment_error": "Docker/GPU/library failure",
    "reward_hacking": "Verifier passes, output implausible",
}
DISTILLABLE_CLASSES = {
    "skill_consulted_ignored", "skill_content_wrong", "skill_gap"
}


def classify_failure(result: dict, stdout: str, stderr: str) -> str:
    """Return a failure class for one trial. Heuristic — see Part 9."""
    fails = [c for c in result.get("verification", {}).get("checks", [])
             if not c.get("passed")]
    err = stderr.lower() + " " + stdout.lower()

    if "out of memory" in err or "cuda" in err and "error" in err:
        return "environment_error"
    if "modulenotfounderror" in err or "importerror" in err:
        if any(s in err for s in ("sionna.channel", "sionna.mimo", "sionna.ofdm")):
            return "skill_content_wrong"  # v0.x imports — version-mismatched
        return "skill_gap"
    if "runtimeerror: cdl" in err and "transmitter" in err:
        return "skill_consulted_ignored"  # SKILL #5 explicitly forbids this
    if any("artifact:" in c.get("name", "") for c in fails):
        if "placeholder_pre_shipped" in stdout:
            return "skill_not_consulted"
    if any(c.get("name", "").startswith("plausibility:") for c in fails):
        return "reward_hacking"
    if fails and not stderr.strip():
        return "skill_consulted_ignored"
    return "model_capability_ceiling"


# ────────────────────────────────────────────────────────────────────
# Train run + tally
# ────────────────────────────────────────────────────────────────────

def run_train(label: str, model: str, workers: int = 6) -> Path:
    """Invoke run_benchmark.py on the train split. Returns the results dir."""
    print(f"[loop] running train suite: label={label} model={model}")
    cmd = [
        sys.executable, "benchmark/run_benchmark.py",
        "--label", label,
        "--shuffle-seed", "42",
        "--timeout", "400",
        "--retry-timeout", "1200",
        "--split", "train",
        "--conditions", "with_skill", "no_skill",
        "--k", "1",
        "--workers", str(workers),
        "--max-turns", "25",
        "--model", model,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / "benchmark" / "results" / label


def tally(results_dir: Path) -> dict:
    """Aggregate metrics + failure-class distribution from a results dir."""
    rows = []
    for cond in ("with_skill", "no_skill"):
        d = results_dir / cond
        if not d.exists():
            continue
        for tid in d.iterdir():
            for trial in tid.iterdir():
                rj = trial / "result.json"
                if not rj.exists():
                    continue
                try:
                    r = json.loads(rj.read_text())
                except Exception:
                    continue
                stdout = (trial / "stdout.txt").read_text(errors="replace") \
                    if (trial / "stdout.txt").exists() else ""
                stderr = (trial / "stderr.txt").read_text(errors="replace") \
                    if (trial / "stderr.txt").exists() else ""
                cls = classify_failure(r, stdout, stderr) if not r["verification"]["passed"] else "pass"
                rows.append({
                    "task_id": r.get("task_id"),
                    "cond": r.get("condition"),
                    "tier": r.get("tier"),
                    "passed": r["verification"]["passed"],
                    "failure_class": cls,
                })
    by_cond: dict = {}
    for cond in ("with_skill", "no_skill"):
        sub = [r for r in rows if r["cond"] == cond]
        passed = sum(1 for r in sub if r["passed"])
        classes = Counter(r["failure_class"] for r in sub if not r["passed"])
        by_cond[cond] = {
            "n": len(sub),
            "pass": passed,
            "pass_rate": round(100 * passed / max(len(sub), 1), 1),
            "failure_classes": dict(classes),
        }
    delta = by_cond["with_skill"]["pass_rate"] - by_cond["no_skill"]["pass_rate"]
    dom = Counter()
    for r in rows:
        if r["cond"] == "with_skill" and not r["passed"]:
            dom[r["failure_class"]] += 1
    return {
        "n_total": len(rows),
        "by_cond": by_cond,
        "delta_pp": delta,
        "dominant_failure_class": dom.most_common(1)[0] if dom else (None, 0),
    }


# ────────────────────────────────────────────────────────────────────
# Edit proposal
# ────────────────────────────────────────────────────────────────────

# P3.4: integrate the canonical 7-class classifier from distill_failures.py
# instead of the ad-hoc one in this file. Falls back gracefully if import fails.
try:
    from distill_failures import classify_failure as _canonical_classify  # type: ignore
    _USE_CANONICAL = True
except ImportError:
    _USE_CANONICAL = False


def classify_failure_v2(result: dict, stdout: str, stderr: str) -> str:
    """Wrapper that uses distill_failures.classify_failure when available."""
    if not _USE_CANONICAL:
        return classify_failure(result, stdout, stderr)
    code_files = []
    # If we know the trial dir, glob for code; else accept stdout proxy.
    fc = [c for c in result.get("verification", {}).get("checks", []) if not c.get("passed")]
    return _canonical_classify(
        failed_checks=fc,
        stdout=stdout,
        stderr=stderr,
        code="\n".join(code_files),
        agent_final_text="",
        passed=result.get("verification", {}).get("passed", False),
    )


# Heuristic: which SKILL.md section a failure class implicates.
SECTION_FOR_CLASS = {
    "skill_not_consulted": "description (frontmatter)",
    "skill_consulted_ignored": "Module 2 numbered constraints (add consequence/Why:)",
    "skill_content_wrong": "Identify the wrong instruction; re-anchor to Sionna 2.0 docs",
    "skill_gap": "Add new constraint or new reference file",
    "model_capability_ceiling": "Not fixable by skill edits — model size / context",
    "environment_error": "Not in scope — investigate harness",
    "reward_hacking": "Strengthen verifier plausibility checks",
}


def propose_edit(metrics: dict, results_dir: Path | None = None,
                  skill_md_path: Path | None = None) -> dict:
    """Produce a structured edit proposal — one section per iteration.

    P3.4 enhancement: when `results_dir` and `skill_md_path` are provided,
    locates SKILL.md line range for the target section and outputs a
    concrete patch suggestion (line range + replacement text). Without
    those args, falls back to the abstract proposal from v1.5.
    """
    cls, count = metrics["dominant_failure_class"]
    proposal = {
        "dominant_failure_class": cls,
        "occurrences": count,
        "target_section": SECTION_FOR_CLASS.get(cls, "(unmapped class — investigate manually)"),
        "delta_pp": metrics["delta_pp"],
        "next_action_hint": (
            "Distill the failed trajectories of this class into a principle "
            "(see master guide Part 8 step-by-step). Add it to the target "
            "section as ONE constraint with a Why: rationale. Re-run train "
            "and check for ≥ 2pp lift OR ≥ 25% reduction in this class."
        ),
    }

    if skill_md_path is None:
        skill_md_path = ROOT / ".claude/skills/rf-simulator/SKILL.md"

    if skill_md_path.exists() and cls in DISTILLABLE_FOR_PATCH:
        section_anchor = SECTION_ANCHOR.get(cls)
        if section_anchor:
            patch = _locate_section_patch(skill_md_path, section_anchor)
            if patch:
                proposal["patch"] = {
                    "file": str(skill_md_path.relative_to(ROOT)),
                    "anchor": section_anchor,
                    "line_range": patch["line_range"],
                    "current_text_preview": patch["preview"],
                    "suggested_action": (
                        f"Insert a new bullet/constraint just before line {patch['line_range'][1]} "
                        f"following the format of surrounding entries. Include a 'Why:' "
                        f"rationale citing the {cls} failure class observed in {count} trials."
                    ),
                }
            else:
                proposal["patch"] = {
                    "file": str(skill_md_path.relative_to(ROOT)),
                    "error": f"could not locate anchor '{section_anchor}'",
                }
    return proposal


# Failure classes for which a structured patch can be suggested
DISTILLABLE_FOR_PATCH = {
    "skill_not_consulted",
    "skill_consulted_ignored",
    "skill_content_wrong",
    "skill_gap",
}

# Anchor strings (substrings expected to appear in SKILL.md) per class
SECTION_ANCHOR = {
    "skill_not_consulted":     "## Restate Before Coding",
    "skill_consulted_ignored": "### Twelve constraints",
    "skill_content_wrong":     "### Version check",
    "skill_gap":               "### Conditional reads",
}


def _locate_section_patch(skill_path: Path, anchor: str) -> dict | None:
    """Find the line range of a section in SKILL.md identified by anchor.

    Returns dict with line_range = (start, end_exclusive) and a small
    preview of the current section content. Section ends at the next
    same-level or higher-level heading."""
    lines = skill_path.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if anchor in line:
            start = i
            break
    if start is None:
        return None
    # Determine heading level: count leading '#'
    anchor_level = len(lines[start]) - len(lines[start].lstrip("#"))
    # Find next heading of same or higher level
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].lstrip().startswith("#"):
            continue
        level = len(lines[j]) - len(lines[j].lstrip("#"))
        if level <= anchor_level:
            end = j
            break
    return {
        "line_range": (start + 1, end + 1),  # 1-indexed for human readability
        "preview": "\n".join(lines[start:min(start + 6, end)]),
    }


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument("--skill-version", required=True, help="Tag for this iteration")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-iterations", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--auto-apply", action="store_true",
                    help="Apply proposed edits automatically (default: print only)")
    args = ap.parse_args()

    if args.auto_apply:
        print("[loop] --auto-apply NOT YET IMPLEMENTED. Falling back to print-only.")

    history = []
    for iteration in range(args.max_iterations):
        label = f"loop_{args.skill_version}_iter{iteration}"
        results_dir = run_train(label, args.model, workers=args.workers)
        metrics = tally(results_dir)
        print(f"\n[iter {iteration}] {json.dumps(metrics, indent=2)}")

        history.append(metrics)
        if iteration > 0:
            prev = history[-2]["by_cond"]["with_skill"]["pass_rate"]
            curr = metrics["by_cond"]["with_skill"]["pass_rate"]
            if curr - prev < 2.0:
                print(f"\n[loop] iteration {iteration}: lift {curr - prev:+.1f}pp < 2pp threshold.")
                print(f"[loop]   investigate description triggering or task difficulty.")
                break

        proposal = propose_edit(metrics)
        print(f"\n[loop] PROPOSED EDIT for next iteration:")
        print(json.dumps(proposal, indent=2))
        print(f"\n[loop] Apply manually, then re-run with --skill-version {args.skill_version}.iter{iteration+1}")
        # Placeholder: real auto-apply would call an LLM with the failure
        # corpus + the SKILL.md section to rewrite, validate the diff,
        # commit, and continue. Out of scope for this scaffold.
        break

    # Final summary
    log_path = ROOT / "benchmark/_studies_archive" / f"loop_{args.skill_version}.json"
    log_path.write_text(json.dumps(history, indent=2))
    print(f"\n[loop] history archived → {log_path}")


if __name__ == "__main__":
    main()
