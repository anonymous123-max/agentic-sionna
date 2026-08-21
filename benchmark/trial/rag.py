"""RAG context retrieval — lazy chromadb integration."""
from __future__ import annotations
import functools
import os
import sys
from pathlib import Path

# benchmark/trial/rag.py is 2 levels below repo root
ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _rag_store():
    """Lazy: only imports chromadb when CLAUDE_CODE_USE_RAG=1 AND a caller
    actually invokes _retrieve_rag_context. Returns the store module
    handle, or None if unavailable."""
    if os.environ.get("CLAUDE_CODE_USE_RAG", "0") != "1":
        return None
    try:
        sys.path.insert(0, str(ROOT / ".claude/skills/rf-simulator/memory"))
        import store as _store
        if not _store.stats().get("available"):
            return None
        return _store
    except Exception:
        return None


def _retrieve_rag_context(task_prompt: str, top_k: int = 5) -> str:
    """Pre-prompt block of relevant principles, API chunks, AND script /
    template / SKILL.md pointers. Empty string if RAG unavailable or
    returns no results.

    Each hit is formatted by `kind`:
      - failure_principle / api_chunk: bullet with truncated doc text
      - script / template / skill_module: title + file path + short snippet
        (so the agent can copy directly without re-running lookup.py)
    """
    import json as _json
    store = _rag_store()
    if store is None:
        return ""
    try:
        chunks = store.retrieve(task_prompt, top_k=top_k)
    except Exception:
        return ""
    if not chunks:
        return ""
    bullets = []
    for c in chunks:
        doc = c.get("document", "").strip()
        meta = c.get("metadata", {}) or {}
        kind = meta.get("kind", "")
        if not doc:
            continue
        if kind in ("script", "template", "skill_module"):
            try:
                obj = _json.loads(doc)
                title = obj.get("title", "")[:100]
                fp = obj.get("file_path", "")
                summary = obj.get("summary", "")[:200]
                snippet = obj.get("snippet", "")[:500]
                bullets.append(
                    f"- [{kind}] {title}\n"
                    f"    file: $RF_SKILL_DIR/{fp}\n"
                    f"    {summary}\n"
                    f"    snippet:\n      "
                    + snippet.replace("\n", "\n      ")
                )
                continue
            except Exception:
                pass
        bullets.append(f"- [{kind or 'mem'}] {doc[:300]}")
    if not bullets:
        return ""
    return (
        "\n\nRELATED MEMORY (retrieved from skill vector store; consult "
        "BEFORE writing code from scratch — copying a matching template "
        "is much faster than rediscovering the Sionna 2.x API):\n"
        + "\n".join(bullets) + "\n"
    )
