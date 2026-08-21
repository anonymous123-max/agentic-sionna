"""ingest.py — Backfill the vector store from references/*.md.

Run after material changes to references — this is offline tooling, not
called by the agent at runtime. Each markdown file is split at H2 (## )
boundaries; chunks larger than max_chars are further split at H3 (### ).

The store's existing dedup_threshold (0.85) prevents near-duplicate adds,
so re-running ingest after edits is safe and idempotent.

Usage:
    python3 .claude/skills/rf-simulator/memory/ingest.py
    # or from repo root:
    PYTHONPATH=.claude/skills/rf-simulator/memory python3 .claude/skills/rf-simulator/memory/ingest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import store  # type: ignore

REFS_DIR = SKILL_ROOT / "references"
# Maximum characters per chunk before we try to split further
MAX_CHARS = 800


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def chunk_markdown(path: Path, max_chars: int = MAX_CHARS) -> list[tuple[str, str]]:
    """Split markdown into (title, body) chunks at heading boundaries.

    Returns list of (title, body). Title is the first heading-line of the
    chunk, capped at 80 chars. Body is the full chunk text including the
    title line.
    """
    text = path.read_text()
    if not text.strip():
        return []

    # Split at H2 boundaries (preserve the heading line at start of each chunk)
    chunks = re.split(r"\n(?=## [^#])", text)
    out: list[tuple[str, str]] = []

    for c in chunks:
        c = c.strip()
        if not c:
            continue
        first_line = c.split("\n", 1)[0].lstrip("# ").strip()[:80]

        if len(c) <= max_chars:
            out.append((first_line, c))
        else:
            # Split further at H3 boundaries
            subs = re.split(r"\n(?=### [^#])", c)
            for sub in subs:
                sub = sub.strip()
                if not sub:
                    continue
                sub_title = sub.split("\n", 1)[0].lstrip("# ").strip()[:80]
                title = sub_title or first_line

                if len(sub) <= max_chars * 2:
                    out.append((title, sub))
                else:
                    # Final fallback: split by paragraph
                    paragraphs = sub.split("\n\n")
                    buf = ""
                    for p in paragraphs:
                        if len(buf) + len(p) + 2 > max_chars * 2 and buf:
                            out.append((title, buf.strip()))
                            buf = p
                        else:
                            buf = (buf + "\n\n" + p) if buf else p
                    if buf.strip():
                        out.append((title, buf.strip()))
    return out


def main() -> None:
    s = store.stats()
    if not s.get("available"):
        print(f"ERROR: vector store unavailable: {s.get('reason')}", file=sys.stderr)
        sys.exit(1)
    print(f"Vector store before: n_items={s.get('n_items')}  path={s.get('db_path')}")

    md_files = sorted(REFS_DIR.glob("*.md"))
    print(f"Ingesting from {len(md_files)} reference files in {REFS_DIR}\n")

    n_added = 0
    n_skipped = 0

    for md in md_files:
        chunks = chunk_markdown(md)
        file_added = 0
        file_skipped = 0
        for idx, (title, body) in enumerate(chunks):
            # Build a stable chunk_id: <stem>-<chunk-index>-<slug-of-title>
            cid = f"ref-{md.stem}-{idx:03d}-{_slugify(title)}"
            chunk = {
                "chunk_id": cid,
                "class_name": md.stem,          # reuse class_name for file stem
                "method_name": title,            # section title in method_name slot
                "version": "2.0",
                "summary": body[:1500],          # full body as summary for embedding
                "signature": "",
                "example": "",
                "source_url": f"references/{md.name}",
            }
            result_id = store.store_api_chunk(chunk)
            if result_id == cid:
                file_added += 1
                n_added += 1
            else:
                # store returned a different id → matched a near-dup
                file_skipped += 1
                n_skipped += 1

        print(f"  {md.name}: {len(chunks)} chunks  (+{file_added} added, ~{file_skipped} dedup-skipped)")

    print(f"\nIngested: {n_added} new + {n_skipped} dedup-skipped")
    final = store.stats()
    print(f"Store stats: {final}")

    # Quick smoke-test retrieval
    print("\nRetrieval smoke tests (top-1 hit per query):")
    test_queries = [
        "LDPC encoder code rate",
        "scene from json sionna",
        "RIS phase shift gradient",
        "load_scene radio map",
        "neural receiver demapper",
        "ITU material concrete glass",
        "OFDM resource grid",
        "Eb/N0 simulation BER",
    ]
    for q in test_queries:
        hits = store.retrieve(q, top_k=2)
        if hits:
            top = hits[0]
            score = top.get("score", 0.0)
            snip = (top.get("document", "") or "")[:80].replace("\n", " ")
            print(f"  {q!r}: score={score:.3f} → {snip}…")
        else:
            print(f"  {q!r}: 0 hits")


if __name__ == "__main__":
    main()
