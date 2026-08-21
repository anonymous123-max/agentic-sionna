"""index_sionna_docs.py — bulk-index Sionna v2.0 API into vector store.

Crawls the Sionna source tree (via tools/online_apis._get_sionna_tree),
fetches each .py file, extracts public class names + docstrings, and
embeds one chunk per class. Idempotent — re-running upserts the same
chunk_id.

Usage:
    pip install chromadb sentence-transformers
    python3 .claude/skills/rf-simulator/scripts/index_sionna_docs.py

    # Limit to N classes for testing:
    python3 .claude/skills/rf-simulator/scripts/index_sionna_docs.py --limit 20
"""
from __future__ import annotations
import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = ROOT / ".claude/skills/rf-simulator"
sys.path.insert(0, str(SKILL_DIR / "memory"))
sys.path.insert(0, str(SKILL_DIR / "tools"))
import store  # type: ignore
import online_apis  # type: ignore

CLASS_PATTERN = re.compile(
    r'^class\s+([A-Z]\w+)\b[^:]*:\s*\n'
    r'(?:\s*#[^\n]*\n)*'
    r'\s*(?:r)?"""\s*(.+?)"""',
    re.MULTILINE | re.DOTALL,
)


def fetch_source(rel_path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/NVlabs/sionna/main/{rel_path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rf-skill/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [{rel_path}] {e}", file=sys.stderr)
        return None


def index_class(class_name: str, docstring: str, src_path: str) -> str | None:
    chunk = {
        "chunk_id": f"sionna-api-{class_name}",
        "class_name": class_name,
        "method_name": "",
        "version": "2.0",
        "summary": docstring[:1500],
        "signature": "",
        "example": "",
        "source_url": f"https://raw.githubusercontent.com/NVlabs/sionna/main/{src_path}",
    }
    return store.store_api_chunk(chunk)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Index at most N classes (for testing)")
    args = ap.parse_args()

    s = store.stats()
    if not s.get("available"):
        print(f"ERROR: vector store unavailable: {s.get('reason')}", file=sys.stderr)
        sys.exit(1)
    print(f"Vector store: {s}")

    paths = online_apis._get_sionna_tree()
    print(f"Sionna tree: {len(paths)} .py files")

    n_indexed = 0
    seen_classes: set[str] = set()
    for rel in paths:
        if args.limit and n_indexed >= args.limit:
            break
        src = fetch_source(rel)
        if not src:
            continue
        for m in CLASS_PATTERN.finditer(src):
            cls = m.group(1)
            doc = m.group(2).strip()
            if not doc or len(doc) < 30:
                continue  # skip empty or trivial classes
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            cid = index_class(cls, doc, rel)
            if cid:
                n_indexed += 1
            if args.limit and n_indexed >= args.limit:
                break

    print(f"\nIndexed {n_indexed} classes")
    print(f"After index: {store.stats()}")

    # Smoke test
    print("\nRetrieval smoke tests:")
    for q in ["LDPC encoder for 5G", "OFDM resource grid", "ray tracing",
              "channel estimation"]:
        results = store.retrieve(q, top_k=2, kind="api_chunk")
        if results:
            for r in results:
                print(f"  '{q}' → {r['metadata'].get('class_name')}: "
                      f"{r['document'][:60]}")
        else:
            print(f"  '{q}' → (no match)")


if __name__ == "__main__":
    main()
