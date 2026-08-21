"""seed_memory.py — bulk-load failure_library.md → vector store.

Parses references/failure_library.md, extracts each `### <heading>` block
as a principle, embeds it, and stores in the ChromaDB collection. Idempotent:
re-running upserts the same `principle_id`.

Usage:
    pip install chromadb sentence-transformers
    python3 .claude/skills/rf-simulator/scripts/seed_memory.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".claude/skills/rf-simulator"
LIBRARY = SKILL_DIR / "references/failure_library.md"

sys.path.insert(0, str(SKILL_DIR / "memory"))
import store  # type: ignore


def parse_library(text: str) -> list[dict]:
    """Extract one record per `### Heading` block."""
    # Split on `### ` (3-hash heading), preserve the heading itself.
    chunks = re.split(r"^###\s+", text, flags=re.MULTILINE)[1:]
    records = []
    for i, chunk in enumerate(chunks):
        # First line is the heading
        lines = chunk.splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        # Pull symptom / fix / class out of the bullet structure
        symptom = _extract_field(body, "Symptom") or ""
        root = _extract_field(body, "Root cause") or ""
        fix = _extract_field(body, "Fix") or ""
        cls_field = _extract_field(body, "Class") or ""
        update_class = _extract_field(body, "Update class") or "[ACTIVE]"

        # Synthesize principle text: heading + first 1500 chars of body
        principle_text = (
            f"{heading}. {root or symptom}. Fix: {fix or 'see body'}".strip()
        )

        records.append({
            "principle_id": f"P-libseed-{i:03d}-" + _slugify(heading)[:20],
            "observation": symptom[:300],
            "failure": symptom[:200],
            "correction": fix[:300],
            "principle": principle_text[:1200],
            "failure_class": cls_field.strip().strip("`") or "unknown",
            "sionna_version": "2.0.x",
            "task_tier": "various",
            "generalizability": "medium",
            "_heading": heading,
            "_update_class": update_class,
        })
    return records


def _extract_field(body: str, field: str) -> str | None:
    """Pull '**Field**: value' from a bullet list."""
    m = re.search(
        rf"\*\*{re.escape(field)}\*\*:\s*(.+?)(?=\n\s*-\s*\*\*|\Z)",
        body, re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    if not LIBRARY.exists():
        print(f"ERROR: {LIBRARY} missing", file=sys.stderr)
        sys.exit(1)
    s = store.stats()
    if not s.get("available"):
        print(f"ERROR: vector store unavailable: {s.get('reason')}", file=sys.stderr)
        sys.exit(1)
    print(f"Vector store: {s}")

    records = parse_library(LIBRARY.read_text())
    print(f"Parsed {len(records)} principles from {LIBRARY.name}")

    n_stored = 0
    for r in records:
        pid = store.store_principle(r)
        if pid:
            n_stored += 1
    print(f"Stored {n_stored} / {len(records)} principles")

    # Final stats
    print(f"After seed: {store.stats()}")

    # Smoke test retrieval
    test_queries = [
        "CDL channel multi-user error",
        "v0.x import error TensorFlow",
        "agent gives up after one error",
        "BER curve flat",
    ]
    print("\nRetrieval smoke tests:")
    for q in test_queries:
        results = store.retrieve(q, top_k=1)
        if results:
            print(f"  '{q}' → {results[0]['document'][:80]}")
        else:
            print(f"  '{q}' → (no match)")


if __name__ == "__main__":
    main()
