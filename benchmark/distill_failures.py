"""distill_failures.py — automate trajectory → principle (master guide Part 8).

Walks per-trial result dirs (or aggregated archive JSONs), classifies each
failure into the 7-class taxonomy from master-guide Part 9, extracts
(observation, failure, correction, principle) tuples, filters for
generalizability, dedups via token Jaccard similarity, and writes to
references/failure_library.md.

Step-by-step (master guide Part 8):
  1. Identify failure point — last action before verifier FAIL
  2. Reconstruct reasoning chain — agent's stated reasoning + skill
     content available at decision time
  3. Classify the failure — uses the 7-class taxonomy
  4. Extract the (OBS, FAIL, FIX, PRINCIPLE) tuple
  5. Generalizability filter — would this prevent ≥3 distinct configs?
  6. Semantic dedup — token Jaccard similarity > 0.6 means merge, not add
     (Phase 6 layers cosine-embedding dedup on top)
  7. Store and embed — write to failure_library.md (flat) + future VDB
  8. Propagate to SKILL.md — only if [ACTIVE] block AND human-confirmed

Usage:
    # Distill from raw per-trial dirs:
    python3 benchmark/distill_failures.py --trials-root benchmark/results/<label>

    # Distill from an archive JSON (no per-trial fidelity, but fast):
    python3 benchmark/distill_failures.py --archive benchmark/_studies_archive/2026-04-28_paired_qwen3_6_v4.json

    # Dry-run (don't update failure_library.md):
    python3 benchmark/distill_failures.py --dry-run --archive ...

    # Update existing library (default: dry-run):
    python3 benchmark/distill_failures.py --archive ... --apply

The classifier and dedup are heuristic; outputs are PROPOSALS for review,
not auto-applied principles.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = ROOT / ".claude/skills/rf-simulator/references/failure_library.md"


# ────────────────────────────────────────────────────────────────────
# 7-class failure taxonomy (master guide Part 9)
# ────────────────────────────────────────────────────────────────────

CLASSES = {
    "skill_not_consulted":       "Agent proceeded without referencing skill",
    "skill_consulted_ignored":   "Reads skill, then violates it",
    "skill_content_wrong":       "Follows skill but skill is incorrect (e.g. v0.x guidance in v2.0 env)",
    "skill_gap":                 "Struggles with topic the skill doesn't cover",
    "model_capability_ceiling":  "Understands what to do, cannot execute",
    "environment_error":         "Docker/GPU/library failure",
    "reward_hacking":            "Verifier passes, output implausible",
}
DISTILLABLE = {"skill_consulted_ignored", "skill_content_wrong", "skill_gap"}


# ────────────────────────────────────────────────────────────────────
# Classifier
# ────────────────────────────────────────────────────────────────────

def classify_failure(*, failed_checks: list[dict], stdout: str, stderr: str,
                      code: str, agent_final_text: str = "",
                      passed: bool = False) -> str:
    """Best-guess failure classification. Rules are ordered by specificity:
    earlier rules win when multiple match.

    Returns one of CLASSES keys, or "pass" if `passed=True`.
    """
    if passed:
        return "pass"

    err = (stderr or "").lower()
    out = (stdout or "").lower()
    code_lc = (code or "").lower()
    fc_names = [c.get("name", "") for c in failed_checks]

    # 7. Reward hacking — verifier flagged plausibility, even though
    # task-specific checks may have passed.
    if any("plausibility:" in n for n in fc_names):
        return "reward_hacking"

    # 6. Environment error — GPU/CUDA/Docker/file-system failures.
    env_signatures = (
        "out of memory", "cuda runtime error", "cuda out of memory",
        "no cuda gpus are available", "permission denied",
        "no space left on device", "killed", "[timeout]",
    )
    if any(sig in err or sig in out for sig in env_signatures):
        return "environment_error"

    # 3. Skill content wrong — agent used v0.x imports the skill might
    # have suggested. We can't be sure without inspecting skill version,
    # but ImportError on legacy namespaces is a strong tell.
    legacy_imports = ("sionna.channel", "sionna.mimo", "sionna.ofdm",
                      "tf.gradienttape", "tensorflow")
    if "importerror" in err or "modulenotfounderror" in err:
        if any(sig in err or sig in code_lc for sig in legacy_imports):
            return "skill_content_wrong"
        return "skill_gap"  # unknown API → skill didn't tell agent the right one

    # 2. Skill consulted but ignored — agent's code violates a skill
    # constraint (e.g. CDL with multi-user, CP < delay spread).
    cdl_violation = (
        "cdl" in code_lc and
        any(t in code_lc for t in ("num_tx=", "num_users", "n_users", "multi_tx"))
        and "umi" not in code_lc and "uma" not in code_lc
    )
    if cdl_violation:
        return "skill_consulted_ignored"
    if "runtimeerror: cdl" in err and "transmitter" in err:
        return "skill_consulted_ignored"

    # 1. Skill not consulted — agent end_turned with empty result AND
    # the harness pre-shipped placeholder is still in place. Strong
    # signal the agent gave up before the skill could help.
    if "placeholder_pre_shipped_by_harness" in out and not (agent_final_text or "").strip():
        return "skill_not_consulted"

    # 4. Skill gap — agent tried, computed something, missed a metric
    # the verifier wanted. Often the skill simply doesn't cover this
    # task type (e.g. ISAC RMSE, channel charting Pearson r).
    threshold_or_exact = [n for n in fc_names if n.startswith(("threshold:", "exact:"))]
    if threshold_or_exact and stderr.strip() == "":
        # Code ran, threshold missed — likely a content gap.
        return "skill_gap"

    # 5. Model capability ceiling — fallback when nothing else matches.
    # Often: agent wrote SOMETHING, code crashed, no clear root cause.
    return "model_capability_ceiling"


# ────────────────────────────────────────────────────────────────────
# Principle extraction (Step 4)
# ────────────────────────────────────────────────────────────────────

@dataclass
class Principle:
    principle_id: str
    extraction_date: str
    source_task_id: str
    source_condition: str
    sionna_version: str
    observation: str
    failure: str
    correction: str
    principle: str
    failure_class: str
    update_target: str
    update_class: str
    generalizability: str  # "high", "medium", "low"


def extract_principle(trial: dict, classification: str) -> Principle | None:
    """Build a structured principle record from a classified failure.

    `trial` is a dict with at least: task_id, cond, tier, failed_checks,
    stderr_tail, agent_final_text. Heuristics here produce drafts; humans
    edit before they enter the library.
    """
    if classification not in DISTILLABLE:
        return None

    err_tail = (trial.get("stderr_tail") or "")[-300:]
    fc = trial.get("failed_checks") or [{}]
    first_fail = fc[0].get("name", "?") if fc else "?"

    # Templated principle text — humans will refine.
    if classification == "skill_content_wrong":
        principle = (
            "When Sionna v2.0 is in use, do not import from v0.x namespaces "
            "(`sionna.channel`, `sionna.mimo`, `sionna.ofdm`). Use "
            "`sionna.phy.channel`, `sionna.phy.mimo`, `sionna.phy.ofdm`. See "
            "references/sionna-version-guide.md."
        )
        target = "SKILL.md Module 1 'Version check' section"
    elif classification == "skill_consulted_ignored":
        principle = (
            f"Constraint violated on first failed check `{first_fail}`. "
            "Add a tighter rationale (Why:) to the relevant SKILL.md constraint "
            "explaining the consequence so the agent doesn't override it."
        )
        target = "SKILL.md Module 2 numbered constraints"
    else:  # skill_gap
        principle = (
            f"Task type produced `{first_fail}` failure not covered by current "
            "skill. Add a new constraint or new reference file documenting the "
            "expected pattern + rationale."
        )
        target = "SKILL.md Module 1 routing or new references/<topic>.md"

    return Principle(
        principle_id=f"P-{trial.get('source_task_id', 'X')}-auto-{classification[:3]}",
        extraction_date="2026-05-01",
        source_task_id=trial.get("task_id", "?"),
        source_condition=trial.get("cond", "?"),
        sionna_version="2.0.x",
        observation=f"Trial {trial.get('task_id', '?')} {trial.get('cond', '?')} on {trial.get('tier', '?')}",
        failure=err_tail.strip() or first_fail,
        correction="(see principle) — manual review required",
        principle=principle,
        failure_class=classification,
        update_target=target,
        update_class="[ACTIVE]",
        generalizability="medium",
    )


# ────────────────────────────────────────────────────────────────────
# Generalizability filter (Step 5) — needs ≥3 distinct task tiers
# ────────────────────────────────────────────────────────────────────

def filter_generalizable(principles: list[Principle], task_tiers: dict[str, str]) -> list[Principle]:
    """Keep principles that recur across ≥3 distinct task tiers in the
    same failure_class. `task_tiers` maps source_task_id → tier."""
    by_class: dict[str, list[Principle]] = {}
    for p in principles:
        by_class.setdefault(p.failure_class, []).append(p)

    keepers: list[Principle] = []
    for cls, ps in by_class.items():
        tiers = {task_tiers.get(p.source_task_id, "?") for p in ps}
        if len(tiers) >= 3:
            # Promote one representative principle for this class.
            rep = ps[0]
            rep.generalizability = "high"
            rep.observation = (
                f"Observed across {len(ps)} trials spanning tiers: "
                f"{', '.join(sorted(tiers))}"
            )
            keepers.append(rep)
        elif len(ps) >= 2:
            # Two distinct trials → medium confidence
            rep = ps[0]
            rep.generalizability = "medium"
            keepers.append(rep)
        # Single trial → discarded (not generalizable enough)
    return keepers


# ────────────────────────────────────────────────────────────────────
# Token Jaccard dedup (Step 6 — string-similarity placeholder for VDB)
# ────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate(new: Principle, existing: list[str], threshold: float = 0.6) -> bool:
    """existing is a list of principle-text strings (not Principle objects).
    Returns True if `new.principle` is too similar to any existing entry."""
    return any(jaccard(new.principle, e) >= threshold for e in existing)


# ────────────────────────────────────────────────────────────────────
# Library reader/writer
# ────────────────────────────────────────────────────────────────────

def existing_principles_text(library_md: Path) -> list[str]:
    """Pull the principle paragraphs out of failure_library.md so we can
    dedup against them. Heuristic: anything under '- **principle**:' OR
    the body of a `### Heading` block."""
    if not library_md.exists():
        return []
    text = library_md.read_text()
    # Split on '### ' headings; each chunk = one principle entry
    entries = re.split(r"^###\s+", text, flags=re.MULTILINE)[1:]
    return [e[:1500] for e in entries]


def append_to_library(new_principles: list[Principle], library_md: Path) -> None:
    """Append new principles to a 'Pipeline-distilled' section of
    failure_library.md. Doesn't touch existing hand-curated entries."""
    if not new_principles:
        return
    library_md.parent.mkdir(parents=True, exist_ok=True)
    section = ["\n\n---\n\n## Pipeline-distilled (auto-appended; review before promoting)\n"]
    for p in new_principles:
        section.append(f"\n### {p.principle_id}\n")
        section.append(f"- **Symptom**: {p.failure[:200]}\n")
        section.append(f"- **Class**: `{p.failure_class}`\n")
        section.append(f"- **Source**: {p.observation}\n")
        section.append(f"- **Principle**: {p.principle}\n")
        section.append(f"- **Update target**: {p.update_target}\n")
        section.append(f"- **Generalizability**: {p.generalizability}\n")
        section.append(f"- **Update class**: {p.update_class}\n")
    with library_md.open("a") as f:
        f.write("".join(section))


# ────────────────────────────────────────────────────────────────────
# Driver — accepts either an archive JSON or a per-trial dir
# ────────────────────────────────────────────────────────────────────

def iter_trials_from_archive(archive_path: Path) -> Iterable[dict]:
    """Archive JSONs are a list of per-trial summary dicts. We don't have
    full stdout/stderr in this format, only `failed_checks` and a brief
    `agent_final_text`. Classifier rules degrade gracefully on missing
    fields."""
    rows = json.loads(archive_path.read_text())
    if not isinstance(rows, list):
        return
    for r in rows:
        # Normalize archive schema → trial dict
        yield {
            "task_id": r.get("task_id"),
            "cond": r.get("cond") or r.get("condition"),
            "tier": r.get("tier", "?"),
            "passed": bool(r.get("passed", False)),
            "failed_checks": r.get("failed_checks", []),
            "stderr_tail": r.get("agent_final_text", "")[-1000:],
            "stdout_tail": "",
            "code": "",
            "agent_final_text": r.get("agent_final_text", ""),
        }


def iter_trials_from_dir(trials_root: Path) -> Iterable[dict]:
    """Walk benchmark/results/<label>/<cond>/<task_id>/<t>/ and pull each
    trial's full stdout/stderr/code."""
    for cond_dir in trials_root.iterdir():
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for tid_dir in cond_dir.iterdir():
            if not tid_dir.is_dir():
                continue
            for trial_dir in tid_dir.iterdir():
                rj = trial_dir / "result.json"
                if not rj.exists():
                    continue
                try:
                    r = json.loads(rj.read_text())
                except Exception:
                    continue
                stdout = (trial_dir / "stdout.txt").read_text(errors="replace") \
                    if (trial_dir / "stdout.txt").exists() else ""
                stderr = (trial_dir / "stderr.txt").read_text(errors="replace") \
                    if (trial_dir / "stderr.txt").exists() else ""
                code_files = list(trial_dir.glob("*.py"))
                code = "\n".join(f.read_text(errors="replace") for f in code_files)
                yield {
                    "task_id": r.get("task_id"),
                    "cond": cond,
                    "tier": r.get("tier", "?"),
                    "passed": r.get("verification", {}).get("passed", False),
                    "failed_checks": [c for c in r.get("verification", {}).get("checks", [])
                                       if not c.get("passed")],
                    "stderr_tail": stderr[-3000:],
                    "stdout_tail": stdout[-3000:],
                    "code": code,
                    "agent_final_text": "",
                }


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=Path,
                     help="Per-version aggregated JSON (lower fidelity)")
    src.add_argument("--trials-root", type=Path,
                     help="Per-trial result dir (full stdout/stderr/code)")
    ap.add_argument("--apply", action="store_true",
                    help="Append new principles to failure_library.md "
                         "(default: dry-run / print only)")
    ap.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = ap.parse_args()

    # Step 1 — gather trials
    iterator = (iter_trials_from_archive(args.archive)
                if args.archive
                else iter_trials_from_dir(args.trials_root))

    # Step 2-3 — classify
    trials = list(iterator)
    classified = []
    class_counts: Counter = Counter()
    task_tiers: dict[str, str] = {}
    for t in trials:
        cls = classify_failure(
            failed_checks=t["failed_checks"],
            stdout=t.get("stdout_tail", ""),
            stderr=t.get("stderr_tail", ""),
            code=t.get("code", ""),
            agent_final_text=t.get("agent_final_text", ""),
            passed=t["passed"],
        )
        classified.append((t, cls))
        class_counts[cls] += 1
        if t.get("task_id"):
            task_tiers[t["task_id"]] = t.get("tier", "?")

    print(f"=== Classified {len(trials)} trials ===")
    for cls, n in class_counts.most_common():
        print(f"  {cls:30s} {n}")

    # Step 4 — extract principle drafts (only for distillable classes)
    drafts: list[Principle] = []
    for t, cls in classified:
        p = extract_principle({**t, "source_task_id": t.get("task_id", "?")}, cls)
        if p:
            drafts.append(p)
    print(f"\nExtracted {len(drafts)} principle drafts (distillable classes only)")

    # Step 5 — generalizability filter
    keepers = filter_generalizable(drafts, task_tiers)
    print(f"After generalizability filter: {len(keepers)} principles")

    # Step 6 — Jaccard dedup against existing library
    existing = existing_principles_text(args.library)
    new_only = [p for p in keepers if not is_duplicate(p, existing)]
    print(f"After Jaccard dedup vs library ({len(existing)} existing entries): "
          f"{len(new_only)} truly new")

    if not new_only:
        print("\nNothing new to append. (All distilled principles already covered.)")
        return

    # Step 7 — write to library (or dry-run)
    if args.apply:
        append_to_library(new_only, args.library)
        print(f"\n✓ Appended to {args.library.name}")
    else:
        print("\n(DRY-RUN — pass --apply to append) Proposed principles:")
        for p in new_only:
            print(f"\n  [{p.failure_class}] {p.principle_id}")
            print(f"    target: {p.update_target}")
            print(f"    principle: {p.principle[:200]}")


if __name__ == "__main__":
    main()
