"""Oracle leakage audit — flag any task-specific content that appears
verbatim in the skill body.

The skill should teach principles, not solutions. If a task prompt mentions
"NVE < 60 on CDL-C at 30 kHz SCS" and the skill body also says
"NVE < 60 on CDL-C at 30 kHz SCS", that's leakage — the skill contains
the answer key.

Rules:
  - Extract all specific numeric-with-unit tokens from each task prompt
    (e.g., "15 dB", "64-QAM", "8 antennas", "CDL-C", "30 kHz").
  - Grep every skill/reference/template file for those tokens.
  - Flag tokens that appear in BOTH a task prompt AND the skill body.

Output: benchmark/tasks/_audits/leakage_audit.json — per-task list of potential leaks.

Run: python benchmark/audit_leakage.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = ROOT / "benchmark/tasks/tasks.json"
SKILL_DIR = ROOT / ".claude/skills/rf-simulator"


# Patterns for task-specific content that could be leaked
NUMERIC_PATTERNS = [
    # Numeric + unit combinations that are task-specific parameters
    r"\b\d+\.?\d*\s*(?:dB|dBm|GHz|MHz|kHz|Hz|m/s|km/h|meters?|m|ns|µs|ms)\b",
    # Antenna / bit / code configurations
    r"\b\d+\s*(?:antennas?|streams?|bits?|users?|cells?|RX|TX|BS|UE)\b",
    # Explicit constellation / code names
    r"\b(?:BPSK|QPSK|16-QAM|64-QAM|256-QAM)\b",
    # Specific channel model variants
    r"\b(?:CDL-[A-E]|TDL-[A-E]|UMi|UMa|RMa)\b",
    # Code rates as fractions
    r"\b(?:rate|R)\s*=?\s*\d+/\d+\b",
    # Message / block lengths
    r"\b\d{2,4}\s*(?:bits?|blocks?|symbols?|subcarriers?)\b",
]


def extract_task_specific_tokens(text: str) -> set[str]:
    tokens = set()
    for pat in NUMERIC_PATTERNS:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            # Normalize whitespace + casing
            tokens.add(re.sub(r"\s+", " ", m).strip().lower())
    return tokens


def load_skill_text() -> str:
    """Concatenate all text the agent might see from the skill — SKILL.md,
    references/*.md, templates/*.py, templates/*.json, AGENTS.md."""
    parts = []
    for subdir in ["", "references", "templates", "agents"]:
        d = SKILL_DIR / subdir if subdir else SKILL_DIR
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.is_file() and p.suffix in (".md", ".py", ".json", ".txt"):
                try:
                    parts.append(p.read_text(errors="replace"))
                except Exception:
                    pass
    return "\n".join(parts)


def main():
    tasks = json.loads(TASKS_FILE.read_text())["tasks"]
    skill_text = load_skill_text().lower()

    findings = []
    total_leaks = 0
    for t in tasks:
        prompt = t.get("prompt", "")
        task_tokens = extract_task_specific_tokens(prompt)
        leaks = sorted(tok for tok in task_tokens if tok in skill_text)
        if leaks:
            total_leaks += len(leaks)
            findings.append({
                "id": t["id"], "origin_id": t["origin_id"],
                "tier": t["tier"],
                "prompt_excerpt": prompt[:140],
                "leaked_tokens": leaks,
            })

    print(f"Leakage audit: {len(findings)}/{len(tasks)} tasks with at least "
          "one leaked token (may be benign — see notes).")
    print(f"Total leaked tokens (cross-product): {total_leaks}")
    print()

    # Leaked tokens that appear in MANY tasks — most likely these are
    # generic terms, not real leaks (e.g., "CDL-A" appears because the
    # skill references channel models generally, not because the task
    # has the answer key). Surface these as "probable false positives".
    token_task_count: dict[str, int] = {}
    for f in findings:
        for tok in f["leaked_tokens"]:
            token_task_count[tok] = token_task_count.get(tok, 0) + 1

    # Tokens leaked in 5+ tasks are generic domain terminology —
    # the skill should contain them. Below that threshold, worth review.
    shared = sorted([(tok, cnt) for tok, cnt in token_task_count.items()
                     if cnt >= 5], key=lambda x: -x[1])
    unique = sorted([(tok, cnt) for tok, cnt in token_task_count.items()
                     if cnt < 5], key=lambda x: -x[1])

    print(f"Generic domain terms (in ≥5 tasks — probably benign, "
          f"the skill teaches the domain): {len(shared)}")
    for tok, cnt in shared[:10]:
        print(f"  {cnt}x: {tok!r}")

    print()
    print(f"Task-specific tokens leaked (< 5 tasks, HIGHER-RISK — review):")
    print(f"  {len(unique)} unique tokens")
    for tok, cnt in unique[:20]:
        tasks_with = [f["id"] for f in findings if tok in f["leaked_tokens"]]
        print(f"  {cnt}x {tok!r} — in {tasks_with}")

    Path(ROOT / "benchmark/tasks/_audits/leakage_audit.json").write_text(
        json.dumps({"findings": findings,
                     "token_counts": token_task_count}, indent=2))
    print(f"\nWrote {ROOT / 'benchmark/tasks/_audits/leakage_audit.json'}")


if __name__ == "__main__":
    main()
