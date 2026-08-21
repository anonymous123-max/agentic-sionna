#!/usr/bin/env python3
"""lookup.py — agent-callable semantic search over the skill's vector store.

Usage:
    python3 $RF_SKILL_DIR/scripts/lookup.py "your query"
    python3 $RF_SKILL_DIR/scripts/lookup.py "CDL multi-tx error" --top-k 5
    python3 $RF_SKILL_DIR/scripts/lookup.py "LDPC encoder" --kind api

Outputs human-readable hits to stdout. Exits 0 silently if chromadb is
unavailable so agents can chain `lookup.py "..." || true` without breaking.

The store at $RF_SKILL_DIR/memory/store.py handles the lazy import +
graceful no-op when chromadb is missing. We just need a thin CLI here.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Resolve the skill memory dir relative to this script.
_THIS = Path(__file__).resolve()
_MEMORY_DIR = _THIS.parent.parent / "memory"
sys.path.insert(0, str(_MEMORY_DIR))

# Map CLI-friendly short names to store.py's internal kind values.
_KIND_MAP = {
    "principle": "failure_principle",
    "api": "api_chunk",
    "script": "script",
    "template": "template",
    "skill": "skill_module",
    "any": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Semantic search over skill memory (failure principles + Sionna API chunks)."
    )
    ap.add_argument("query", help="Free-text query to search for.")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of hits to return (default: 3)")
    ap.add_argument("--kind",
                    choices=["principle", "api", "script", "template",
                             "skill", "any"],
                    default="any",
                    help="Filter by record kind: principle | api | script "
                         "| template | skill | any (default: any)")
    args = ap.parse_args()

    try:
        import store
    except Exception as e:
        # store.py itself catches chromadb ImportError lazily, so a hard
        # failure here means the script can't find the memory module.
        print(f"# lookup unavailable: {e}", file=sys.stderr)
        return 0

    try:
        fr = store.ensure_fresh()
        if fr.get("added", 0) > 0:
            print(f"[lookup] auto-ingested {fr['added']} stale chunks", file=sys.stderr)
    except Exception:
        pass  # never block lookup

    kind_arg = _KIND_MAP.get(args.kind)  # None means "any"
    try:
        hits = store.retrieve(args.query, top_k=args.top_k, kind=kind_arg)
    except TypeError:
        # Older store API may not accept kind=
        hits = store.retrieve(args.query, top_k=args.top_k)

    if not hits:
        # Either chromadb missing OR the corpus has nothing for this query.
        print("# no hits (chromadb may be unavailable, or corpus is empty for this query)")
        return 0

    import json as _json
    for i, hit in enumerate(hits, 1):
        # `hit` is whatever store.retrieve returns — typically dict with
        # `document`, `metadata`, `distance`, `score`. Be defensive about shape.
        if isinstance(hit, dict):
            text = hit.get("document") or hit.get("text") or str(hit)
            meta = hit.get("metadata", {})
            kind = meta.get("kind", "?")
            score = hit.get("score")
            score_str = f"  score={score:.3f}" if score is not None else ""
            # script/template/skill_module entries store JSON-serialized
            # {title, file_path, summary, snippet, tags}. Format them with
            # the path + snippet so the agent can act on the result.
            if kind in ("script", "template", "skill_module"):
                try:
                    obj = _json.loads(text)
                    title = obj.get("title", meta.get("title", ""))
                    fp = obj.get("file_path", meta.get("file_path", ""))
                    summary = obj.get("summary", "")
                    snippet = obj.get("snippet", "")
                    print(f"--- hit {i}  kind={kind}{score_str} ---")
                    print(f"title: {title}")
                    print(f"file:  $RF_SKILL_DIR/{fp}")
                    if summary:
                        print(f"summary: {summary[:400]}")
                    print()
                    print("snippet:")
                    print("-" * 60)
                    print(snippet[:1500])
                    if len(snippet) > 1500:
                        print(f"... [{len(snippet) - 1500} more chars; "
                              f"read full file at $RF_SKILL_DIR/{fp}]")
                    print("-" * 60)
                    print()
                    continue
                except Exception:
                    pass  # fall through to default formatting
            # Default formatting for failure_principle / api_chunk.
            print(f"--- hit {i}  kind={kind}{score_str} ---")
            print(text[:600])
            if len(text) > 600:
                print("...")
        else:
            print(f"--- hit {i} ---")
            print(str(hit)[:600])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
