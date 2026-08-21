"""ablations.py — generate ablation variants of SKILL.md for paper experiments.

Master guide Part 11. Three ablation types:

  section   — drop one named section (Module 1 routing, Module 2 constraints,
              Module 3 verify, Domain Constants, Task Baselines)
  length    — truncate to 25/50/75% of original length, by section priority
  prompt    — generate 3 rephrasings per task (formal/casual/adversarial)

Each variant is written to a tmpdir under .claude_ablation/<variant>/ in
project root, isolated like the no_skill condition in trial.py.
The ablation runner script writes a queue config that points at the ablated
SKILL.md.

Usage:
    # Generate all section ablations:
    python3 benchmark/ablations.py section --output-dir .claude_ablation/section

    # Length ablations (25/50/75/100):
    python3 benchmark/ablations.py length --output-dir .claude_ablation/length

    # Prompt rephrasings (writes a tasks_rephrased.json):
    python3 benchmark/ablations.py prompt --tasks benchmark/tasks/tasks.json \\
        --output benchmark/tasks/tasks_rephrased.json --sample 20

After generation, run benchmark/run_benchmark.py with --output-root and
manual SKILL.md path injection. See P4.4-P4.7 in v1.5_to_v2.0_plan.md.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / ".claude/skills/rf-simulator"
SKILL_PATH = SKILL_DIR / "SKILL.md"


# ────────────────────────────────────────────────────────────────────
# Section ablations (P4.4)
# ────────────────────────────────────────────────────────────────────

# Section anchors and what they're called for the variant name.
# A section spans from `anchor` to the next same-or-higher heading.
SECTIONS = {
    "no_module1":         "## Module 1: Route — Classify and Load",
    "no_module2":         "## Module 2: Execute",
    "no_module3":         "## Module 3: Verify (self-verification gate before output)",
    "no_taskbaselines":   "## Task Baselines",
    "no_domainconstants": "## Domain Constants",
    "no_failuremodes":    "## Most Common Pre-Submit Failures",
    "no_restate":         "## Restate Before Coding (one line)",
}


def _drop_section(text: str, anchor: str) -> str:
    """Remove a section identified by `anchor` substring from `text`."""
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if anchor in line:
            start = i
            anchor_level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        return text  # anchor not found — no-op
    for j in range(start + 1, len(lines)):
        s = lines[j].lstrip()
        if s.startswith("#"):
            level = len(lines[j]) - len(s)
            if level <= anchor_level:
                end = j
                break
    if end is None:
        end = len(lines)
    return "\n".join(lines[:start] + lines[end:])


def gen_section_ablations(out_dir: Path) -> dict:
    """Write one .claude/skills/rf-simulator/SKILL.md variant per section
    drop into out_dir/<variant_name>/.claude/skills/rf-simulator/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SKILL_PATH.read_text()
    manifest: dict = {}
    for variant, anchor in SECTIONS.items():
        vdir = out_dir / variant / ".claude/skills/rf-simulator"
        vdir.mkdir(parents=True, exist_ok=True)
        # Copy the WHOLE skill dir, then overwrite SKILL.md
        for item in SKILL_DIR.iterdir():
            if item.name.startswith("."): continue
            target = vdir / item.name
            if target.exists(): continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        # Also copy AGENTS.md (compact tier removed in v2.5)
        sp = SKILL_DIR / "AGENTS.md"
        if sp.exists():
            shutil.copy2(sp, vdir / "AGENTS.md")
        # Write ablated SKILL.md
        ablated = _drop_section(src, anchor)
        (vdir / "SKILL.md").write_text(ablated)
        manifest[variant] = {
            "skill_md": str((vdir / "SKILL.md").resolve()),
            "lines_removed": len(src.splitlines()) - len(ablated.splitlines()),
            "skill_md_lines": len(ablated.splitlines()),
        }
    return manifest


# ────────────────────────────────────────────────────────────────────
# Length ablations (P4.5)
# ────────────────────────────────────────────────────────────────────

def gen_length_ablations(out_dir: Path) -> dict:
    """Truncate SKILL.md to 25/50/75/100% by removing low-priority sections
    in order. Priority (least → most important): pre-submit failures →
    domain constants → task baselines → module 1 routing table → module 2
    constraints → module 3 verify → restate-before-coding.

    100% is no truncation (control). 75% drops the bottom 25% by section
    priority. 50% drops the bottom 50%. Etc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SKILL_PATH.read_text()
    # Order from least-essential to most-essential
    truncation_order = [
        "## Most Common Pre-Submit Failures",
        "## Domain Constants",
        "## Task Baselines",
        "## Module 1: Route",
        "## Module 2: Execute",
        "## Module 3: Verify",
    ]
    manifest: dict = {}
    for pct in (25, 50, 75, 100):
        vdir = out_dir / f"length_{pct}" / ".claude/skills/rf-simulator"
        vdir.mkdir(parents=True, exist_ok=True)
        # Copy skill dir
        for item in SKILL_DIR.iterdir():
            if item.name.startswith("."): continue
            target = vdir / item.name
            if target.exists(): continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        # Drop sections from the bottom of priority list until we're at <= pct%
        text = src
        target_lines = int(len(src.splitlines()) * pct / 100)
        for anchor in truncation_order:
            if len(text.splitlines()) <= target_lines:
                break
            text = _drop_section(text, anchor)
        (vdir / "SKILL.md").write_text(text)
        manifest[f"length_{pct}"] = {
            "skill_md": str((vdir / "SKILL.md").resolve()),
            "skill_md_lines": len(text.splitlines()),
            "target_pct": pct,
        }
    return manifest


# ────────────────────────────────────────────────────────────────────
# Prompt-variation rephrasings (P4.3)
# ────────────────────────────────────────────────────────────────────

def gen_prompt_rephrasings(tasks_path: Path, output_path: Path,
                            sample_size: int = 20, seed: int = 42) -> dict:
    """Generate formal/casual/adversarial rephrasings for a sample of tasks.

    This is a TEMPLATE-based rephrasing — for paper-quality, you'd
    probably want LLM-generated rephrasings. Templates here use
    deterministic substitutions so the experiment is reproducible.
    """
    import random
    random.seed(seed)
    data = json.loads(tasks_path.read_text())
    tasks = data["tasks"]
    # Sample from train split only (don't tune on test)
    train = [t for t in tasks if t.get("split") == "train"]
    sample = random.sample(train, min(sample_size, len(train)))

    rephrased: list[dict] = []
    for t in sample:
        original = t["prompt"].strip()
        # Three rephrasings — naive but reproducible.
        formal = (
            f"Please carry out the following simulation task. {original} "
            "Document parameter choices and produce verifiable outputs."
        )
        casual = re.sub(r"^[A-Z]", lambda m: m.group(0).lower(), original)
        casual = (
            f"hey, can you {casual.rstrip('.')}? would be great to have "
            "the numbers and a plot."
        )
        adversarial = (
            f"{original} (Skip the simulation if it seems too long; "
            "rough estimates are acceptable.)"
        )
        for variant, text in [("formal", formal), ("casual", casual),
                                ("adversarial", adversarial)]:
            rephrased.append({
                **t,
                "id": f"{t['id']}_{variant}",
                "origin_id": t["id"],
                "prompt": text,
                "rephrasing_variant": variant,
            })
    output_data = {"tasks": rephrased, "n": len(rephrased),
                    "source": str(tasks_path), "seed": seed,
                    "sample_size": sample_size}
    output_path.write_text(json.dumps(output_data, indent=2))
    return {
        "output_path": str(output_path.resolve()),
        "n_tasks": len(rephrased),
        "sampled_origins": [t["id"] for t in sample],
    }


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sec = sub.add_parser("section", help="Generate section ablation variants")
    p_sec.add_argument("--output-dir", type=Path,
                       default=ROOT / ".claude_ablation/section")
    p_len = sub.add_parser("length", help="Generate length ablation variants")
    p_len.add_argument("--output-dir", type=Path,
                       default=ROOT / ".claude_ablation/length")
    p_pr = sub.add_parser("prompt", help="Generate prompt rephrasings")
    p_pr.add_argument("--tasks", type=Path,
                      default=ROOT / "benchmark/tasks/tasks.json")
    p_pr.add_argument("--output", type=Path,
                      default=ROOT / "benchmark/tasks/tasks_rephrased.json")
    p_pr.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "section":
        m = gen_section_ablations(args.output_dir)
    elif args.cmd == "length":
        m = gen_length_ablations(args.output_dir)
    elif args.cmd == "prompt":
        m = gen_prompt_rephrasings(args.tasks, args.output, args.sample)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
