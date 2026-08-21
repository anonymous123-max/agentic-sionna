"""memory/store.py — vector store for skill memory (failures + Sionna API).

Implements the master-guide Part 6/8 vector store. Provides:
  - store_principle(record)        : add a failure principle
  - store_api_chunk(chunk)         : add a Sionna API doc chunk
  - retrieve(query, top_k, kind)   : semantic retrieval
  - stats()                        : health check

Persistence path resolution (in order):
  1. SIONNA_SKILL_CHROMA_PATH env var (if set)
  2. ~/.local/share/sionna-skill/chroma_db   (XDG default — survives
     working-tree resets, branch switches, multi-clone)
  3. .claude/skills/rf-simulator/memory/chroma_db   (legacy in-tree
     fallback; only used if a chroma_db dir already exists there)

Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-dim, ~80 MB cache)

Cosine-deduplication at write time (master guide §8 Step 6):
  When `store_principle()` or `store_api_chunk()` is called with text
  that has cosine similarity > DEDUP_THRESHOLD to an existing chunk
  (excluding the same id), the write is skipped and the existing id
  returned. This prevents corpus bloat as new principles are distilled
  from each eval round.

DEPENDENCIES (not installed by default):
    python3 -m pip install --user "chromadb>=0.5,<1.0" sentence-transformers
"""
from __future__ import annotations

import json
import os
import re
import sys
import time as _time
from pathlib import Path

# Lazy imports — the skill works without these installed.
_CLIENT = None
_COLLECTION = None
_ENCODER = None


def _resolve_db_path() -> Path:
    """Resolve the chroma_db directory per the order documented above."""
    env = os.environ.get("SIONNA_SKILL_CHROMA_PATH")
    if env:
        return Path(env).expanduser()
    legacy = Path(__file__).parent / "chroma_db"
    if legacy.exists() and any(legacy.iterdir()):
        return legacy
    return Path.home() / ".local/share/sionna-skill/chroma_db"


DB_PATH = _resolve_db_path()
COLLECTION_NAME = "sionna_skill_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cosine similarity threshold for dedup. Above this, a candidate is
# considered a near-duplicate of an existing chunk and the write is
# skipped. Master guide §8 Step 6 prescribes 0.85; lower = more
# aggressive merging, higher = more permissive.
DEDUP_THRESHOLD = 0.85


def _ensure_initialized():
    """Lazy-init ChromaDB + embedding model. Returns False if deps missing.

    Client selection (in order):
      1. CHROMA_HOST set → HttpClient(host, port=CHROMA_PORT default 8000).
         This is the path for vast.ai workers reaching sunlab's chroma via
         the reverse SSH tunnel (CHROMA_HOST=localhost, CHROMA_PORT=8765).
      2. Else PersistentClient(DB_PATH). Local dev / sunlab's own host.

    Falls back to PersistentClient if HttpClient fails to connect (so a
    dropped tunnel doesn't kill all lookups; we just lose the live updates
    flowing back from sunlab and use whatever was last rsync'd locally).
    """
    global _CLIENT, _COLLECTION, _ENCODER
    if _COLLECTION is not None:
        return True
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    chroma_host = os.environ.get("CHROMA_HOST", "").strip()
    if chroma_host:
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
        try:
            _CLIENT = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            # Force a heartbeat so we fail fast if the tunnel is down.
            _CLIENT.heartbeat()
        except Exception as e:
            print(f"[store] HttpClient {chroma_host}:{chroma_port} failed: {e}; "
                  f"falling back to PersistentClient at {DB_PATH}",
                  file=sys.stderr)
            _CLIENT = None
    if _CLIENT is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(DB_PATH))
    _COLLECTION = _CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _ENCODER = SentenceTransformer(EMBEDDING_MODEL)
    return True


def _is_near_duplicate(
    embedding: list[float], own_id: str
) -> tuple[bool, str | None, float]:
    """Check if this embedding has a >DEDUP_THRESHOLD cosine match in the
    collection (excluding own_id).

    Returns (is_dup, matched_id, similarity). If is_dup is True, the
    caller should skip the write and surface matched_id to the user.
    """
    assert _COLLECTION is not None  # guarded by _ensure_initialized
    if _COLLECTION.count() == 0:
        return False, None, 0.0
    res = _COLLECTION.query(query_embeddings=[embedding], n_results=2)
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for cand_id, dist in zip(ids, dists):
        if cand_id == own_id:
            continue
        # cosine distance = 1 - cosine similarity; threshold check inverted
        sim = 1.0 - dist
        if sim >= DEDUP_THRESHOLD:
            return True, cand_id, sim
        return False, None, sim  # nearest is below threshold, no dup
    return False, None, 0.0


def store_principle(principle: dict, *, dedup: bool = True) -> str | None:
    """Store a failure principle. Schema per master guide Part 8.

    Required fields:
        principle_id, observation, failure, correction, principle,
        failure_class, sionna_version, task_tier, generalizability

    Returns:
        The principle_id of the stored (or matched-duplicate) entry.
        None if the dependencies aren't installed.
    """
    if not _ensure_initialized():
        return None
    text = (
        f"{principle.get('observation','')} "
        f"{principle.get('failure','')} "
        f"{principle.get('principle','')}"
    ).strip()
    embedding = _ENCODER.encode(text).tolist()
    pid = principle["principle_id"]
    if dedup:
        is_dup, matched_id, sim = _is_near_duplicate(embedding, own_id=pid)
        if is_dup:
            print(
                f"[store_principle] skipped {pid}: cosine sim {sim:.3f} "
                f"to existing {matched_id} (>= {DEDUP_THRESHOLD})"
            )
            return matched_id
    metadata = {
        "kind": "failure_principle",
        "failure_class": principle.get("failure_class", "unknown"),
        "sionna_version": principle.get("sionna_version", "unknown"),
        "task_tier": principle.get("task_tier", "unknown"),
        "generalizability": principle.get("generalizability", "medium"),
    }
    _COLLECTION.upsert(
        documents=[principle["principle"]],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[pid],
    )
    return pid


def store_api_chunk(chunk: dict, *, dedup: bool = True) -> str | None:
    """Store a Sionna API documentation chunk.

    Schema:
        chunk_id, class_name, method_name (optional), version, summary,
        signature, example, source_url

    Cosine-dedup defaults ON. Pass dedup=False to force a write
    (e.g., re-indexing a class whose docs were rewritten).
    """
    if not _ensure_initialized():
        return None
    text = (
        f"{chunk.get('class_name','')} "
        f"{chunk.get('method_name','')} "
        f"{chunk.get('summary','')} "
        f"{chunk.get('signature','')}"
    ).strip()
    embedding = _ENCODER.encode(text).tolist()
    cid = chunk["chunk_id"]
    if dedup:
        is_dup, matched_id, sim = _is_near_duplicate(embedding, own_id=cid)
        if is_dup:
            print(
                f"[store_api_chunk] skipped {cid}: cosine sim {sim:.3f} "
                f"to existing {matched_id} (>= {DEDUP_THRESHOLD})"
            )
            return matched_id
    metadata = {
        "kind": "api_chunk",
        "class_name": chunk.get("class_name", ""),
        "method_name": chunk.get("method_name", ""),
        "version": chunk.get("version", "2.0"),
    }
    full_doc = json.dumps({
        "summary": chunk.get("summary"),
        "signature": chunk.get("signature"),
        "example": chunk.get("example"),
        "source_url": chunk.get("source_url"),
    })
    _COLLECTION.upsert(
        documents=[full_doc],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[cid],
    )
    return cid


def store_artifact(
    artifact_id: str,
    kind: str,
    title: str,
    file_path: str,
    snippet: str,
    summary: str = "",
    tags: list[str] | None = None,
    *,
    dedup: bool = True,
) -> str | None:
    """Store a skill artifact (script / template / skill_module).

    Args:
        artifact_id: stable id like "script_run_ber_analytical" or
                     "template_ber" or "skill_module_routing".
        kind: "script" | "template" | "skill_module"
        title: short human title (e.g. "Analytical BER QPSK runner")
        file_path: filesystem path relative to skill root
                   (e.g. "scripts/run_ber_analytical.py")
        snippet: code/text snippet for the agent (≤2K chars; first 30-50
                 lines of script, or one section of template, etc.)
        summary: short prose explanation (1-3 sentences)
        tags: optional list of keywords for embedding boost

    The embedding indexes title + summary + tags + first ~200 chars of
    snippet. Retrieved hits include the FULL snippet in `document`.
    """
    if kind not in ("script", "template", "skill_module"):
        raise ValueError(
            f"kind must be script/template/skill_module, got {kind!r}")
    if not _ensure_initialized():
        return None
    tags = tags or []
    embedding_text = (
        f"{title} {summary} {' '.join(tags)} {snippet[:200]}"
    ).strip()
    embedding = _ENCODER.encode(embedding_text).tolist()
    if dedup:
        is_dup, matched_id, sim = _is_near_duplicate(
            embedding, own_id=artifact_id)
        if is_dup:
            print(
                f"[store_artifact] skipped {artifact_id}: cosine sim "
                f"{sim:.3f} to {matched_id} (>= {DEDUP_THRESHOLD})"
            )
            return matched_id
    metadata = {
        "kind": kind,
        "title": title,
        "file_path": file_path,
        "tags": ",".join(tags),
    }
    full_doc = json.dumps({
        "title": title,
        "file_path": file_path,
        "summary": summary,
        "snippet": snippet,
        "tags": tags,
    })
    _COLLECTION.upsert(
        documents=[full_doc],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[artifact_id],
    )
    return artifact_id


def retrieve(
    query: str,
    top_k: int = 4,
    kind: str | None = None,
    sionna_version: str | None = None,
) -> list[dict]:
    """Semantic retrieval of stored principles or API chunks.

    Args:
        query: free-form text (task description + error message)
        top_k: max chunks to return (≤4 recommended per master guide)
        kind: filter to "failure_principle" or "api_chunk", None = both
        sionna_version: filter by version string match

    Returns:
        list of {document, metadata, distance, score} dicts, ordered by
        similarity. ``score`` is in [0, 1] where 1 = identical:
        ``score = 1 - distance / 2`` (ChromaDB cosine distance range 0-2).
    """
    if not _ensure_initialized():
        return []
    embedding = _ENCODER.encode(query).tolist()
    where: dict = {}
    if kind:
        where["kind"] = kind
    if sionna_version:
        where["sionna_version"] = sionna_version
    result = _COLLECTION.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where if where else None,
    )
    out = []
    for doc, meta, dist in zip(
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite.
        # Convert to similarity in [0, 1] where 1 = identical.
        score = round(1 - dist / 2, 3)
        out.append({"document": doc, "metadata": meta, "distance": dist, "score": score})
    return out


def stats() -> dict:
    """Return collection size + a small sample. For health checks."""
    if not _ensure_initialized():
        return {"available": False, "reason": "chromadb / sentence-transformers not installed"}
    return {
        "available": True,
        "collection": COLLECTION_NAME,
        "n_items": _COLLECTION.count(),
        "db_path": str(DB_PATH),
        "dedup_threshold": DEDUP_THRESHOLD,
    }


_LAST_INGEST_FILE = Path(__file__).parent / "chroma_db" / ".last_ingest"


def _slugify(s: str) -> str:
    """Slug helper for building stable chunk IDs (mirrors ingest.py)."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def ensure_fresh() -> dict:
    """Idempotent: re-ingest references/*.md that are newer than .last_ingest.

    Returns {"added": N, "scanned": N, "status": "fresh|ingested|unavailable|error"}.
    NEVER raises — silent on any failure (chromadb missing, FS errors, etc).

    Called by lookup.py before each retrieve() and by refresh_memory.py CLI.
    """
    try:
        s = stats()
        if not s.get("available"):
            return {"added": 0, "scanned": 0, "status": "unavailable"}

        skill_root = Path(__file__).resolve().parent.parent
        refs_dir = skill_root / "references"
        if not refs_dir.exists():
            return {"added": 0, "scanned": 0, "status": "no_refs"}

        last_ts = 0.0
        if _LAST_INGEST_FILE.exists():
            try:
                last_ts = float(_LAST_INGEST_FILE.read_text().strip())
            except Exception:
                last_ts = 0.0

        stale = []
        scanned = 0
        for md in sorted(refs_dir.glob("*.md")):
            scanned += 1
            try:
                if md.stat().st_mtime > last_ts:
                    stale.append(md)
            except Exception:
                continue

        if not stale:
            return {"added": 0, "scanned": scanned, "status": "fresh"}

        # Lazy-import ingest (heavy) only when stale files exist
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import ingest as _ingest
        except Exception:
            return {"added": 0, "scanned": scanned, "status": "ingest_unavailable"}

        added = 0
        for md in stale:
            try:
                chunks = _ingest.chunk_markdown(md)
                for idx, (title, body) in enumerate(chunks):
                    cid = f"ref-{md.stem}-{idx:03d}-{_slugify(title)}"
                    chunk = {
                        "chunk_id": cid,
                        "class_name": md.stem,
                        "method_name": title,
                        "version": "2.0",
                        "summary": body[:1500],
                        "signature": "",
                        "example": "",
                        "source_url": f"references/{md.name}",
                    }
                    result_id = store_api_chunk(chunk)
                    if result_id == cid:
                        added += 1
            except Exception:
                continue

        try:
            _LAST_INGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_INGEST_FILE.write_text(str(_time.time()))
        except Exception:
            pass

        return {"added": added, "scanned": scanned, "status": "ingested"}
    except Exception as e:
        return {"added": 0, "scanned": 0, "status": f"error: {type(e).__name__}"}


if __name__ == "__main__":
    # Self-test: print availability
    s = stats()
    print(json.dumps(s, indent=2))
