#!/usr/bin/env python3
"""version_review.py — flag version-mismatched skill content on Sionna release.

Per master guide Part 7. When a new Sionna version ships:
  1. Detect installed version vs the version named in SKILL.md / failure_library
  2. Flag every block that mentions the old version with `[REVIEW_NEEDED]`
  3. Re-run the eval suite (separate concern; this script just tags)
  4. After re-eval, blocks that pass have their tag stripped via --clear

Usage:
    # Check for version mismatch and tag (dry-run by default):
    python3 version_review.py

    # Apply tags to SKILL.md / failure_library.md:
    python3 version_review.py --apply

    # After re-eval pass, strip [REVIEW_NEEDED] tags:
    python3 version_review.py --clear

  When `installed_sionna ≠ documented_sionna` and version-bump signal,
  ACTIVE blocks that name the old version get tagged `[REVIEW_NEEDED]`.

Detection rules (regex):
  - Mentions of "Sionna 2.0.x" / "Sionna v2.0.x" / specific dotted version
  - Mentions of TF / tf.GradientTape / sionna.channel (legacy v0.x)
  - "Last verified: Sionna v2.0.x" (failure_library only)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".claude/skills/rf-simulator"

# Files this script may tag/clear
TARGETS = [
    SKILL_DIR / "SKILL.md",
    SKILL_DIR / "AGENTS.md",
    SKILL_DIR / "references/failure_library.md",
    SKILL_DIR / "references/sionna-version-guide.md",
]

# Patterns that anchor version-sensitive blocks
VERSION_PATTERNS = [
    re.compile(r"Sionna\s*v?2\.0[\.\dx]*", re.IGNORECASE),
    re.compile(r"sionna\.(channel|mimo|ofdm)\b"),  # v0.x legacy namespaces
    re.compile(r"tf\.GradientTape"),
    re.compile(r"Last verified:?\s*Sionna", re.IGNORECASE),
]

REVIEW_TAG = "`[REVIEW_NEEDED]`"


def installed_sionna_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("sionna")
    except Exception:
        return None


def documented_versions(text: str) -> set[str]:
    """Pull dotted-version-looking strings out of skill text."""
    return set(re.findall(r"2\.0\.\d+|2\.0\.x|2\.\d+\.\d+", text))


def find_version_blocks(text: str) -> list[tuple[int, str]]:
    """Return list of (line_index, line) for lines that match any version pattern."""
    hits = []
    for i, line in enumerate(text.splitlines()):
        for pat in VERSION_PATTERNS:
            if pat.search(line):
                hits.append((i, line))
                break
    return hits


def add_review_tags(text: str) -> tuple[str, int]:
    """Tag version-sensitive lines with `[REVIEW_NEEDED]` (idempotent)."""
    out_lines = []
    n_added = 0
    for line in text.splitlines():
        if REVIEW_TAG in line:
            out_lines.append(line)
            continue
        for pat in VERSION_PATTERNS:
            if pat.search(line):
                # Append tag at end of line, before any markdown trailing chars
                line = line.rstrip() + " " + REVIEW_TAG
                n_added += 1
                break
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), n_added


def strip_review_tags(text: str) -> tuple[str, int]:
    """Remove `[REVIEW_NEEDED]` tags (idempotent)."""
    new_text, n = re.subn(r"\s*" + re.escape(REVIEW_TAG), "", text)
    return new_text, n


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help="Add [REVIEW_NEEDED] tags to version-sensitive blocks")
    g.add_argument("--clear", action="store_true",
                   help="Strip [REVIEW_NEEDED] tags (after re-eval pass)")
    ap.add_argument("--current-version", default=None,
                    help="Override detected Sionna version (for testing)")
    args = ap.parse_args()

    installed = args.current_version or installed_sionna_version()

    print(f"Installed Sionna version: {installed or '(not installed locally)'}")
    print(f"Targets: {len(TARGETS)} files")

    total_hits = 0
    total_tagged = 0
    total_stripped = 0
    for tgt in TARGETS:
        if not tgt.exists():
            print(f"  (missing) {tgt.relative_to(ROOT)}")
            continue
        text = tgt.read_text()
        hits = find_version_blocks(text)
        docs_versions = documented_versions(text)
        action = ""

        if args.clear:
            new_text, n = strip_review_tags(text)
            if n > 0:
                tgt.write_text(new_text)
                action = f" — STRIPPED {n} tag(s)"
                total_stripped += n
        elif args.apply:
            mismatch = bool(installed and docs_versions and
                            not any(installed.startswith(v.rstrip("x").rstrip(".")) for v in docs_versions))
            if mismatch:
                new_text, n = add_review_tags(text)
                tgt.write_text(new_text)
                action = f" — TAGGED {n} block(s) (version mismatch: installed={installed})"
                total_tagged += n
            else:
                action = " — no mismatch detected; skipped"

        total_hits += len(hits)
        print(f"  {tgt.relative_to(ROOT)}: "
              f"{len(hits)} version-sensitive line(s){action}")

    if not args.apply and not args.clear:
        print(f"\nDry-run total: {total_hits} version-sensitive lines across {len(TARGETS)} files.")
        print("Pass --apply to tag (only when version mismatch detected) "
              "or --clear to strip after re-eval.")
    else:
        if args.apply:
            print(f"\n✓ Tagged {total_tagged} blocks total")
        if args.clear:
            print(f"\n✓ Stripped {total_stripped} tags total")


if __name__ == "__main__":
    main()
