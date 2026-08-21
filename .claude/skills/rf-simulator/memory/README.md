# memory/ — Vector Store for Skill Memory

Per master guide Part 6 (Memory + RAG) and Part 8 (Failure Distillation).
Holds two kinds of records, retrieved by semantic similarity:

| Kind | Source | Update cadence |
|---|---|---|
| `failure_principle` | Distilled from failed trajectories (manual or auto) | After every eval round |
| `api_chunk` | Sionna v2.0 API docstrings (planned) | On Sionna release |

## Status

**Scaffolded; activate per-task.** SKILL.md routes load
`references/failure_library.md` directly when the file fits in context.
Activate the vector store (run the two bulk scripts below) when:

- The failure library outgrows ~10K tokens (~100 principles), OR
- We add the Sionna API doc corpus (planned — large enough to need RAG), OR
- The auto-improvement loop generates principles fast enough that the
  flat file becomes a bottleneck.

## Activate

```bash
pip install chromadb sentence-transformers

# Stats only:
python3 -c "
import sys; sys.path.insert(0, '.claude/skills/rf-simulator/memory')
from store import stats; print(stats())"

# Bulk-seed failure principles into the store:
python3 .claude/skills/rf-simulator/scripts/seed_memory.py

# Bulk-index Sionna v2.0 API into the store (limit 20 for testing):
python3 .claude/skills/rf-simulator/scripts/index_sionna_docs.py --limit 20
```

This creates `chroma_db/` next to this README on first call. The
embedding model (`all-MiniLM-L6-v2`, ~80 MB) downloads on first
encode.

## Wire into the agent flow

The retrieval point is at task start — query with the task description
plus any error message in scope:

```python
from rf_simulator.memory.store import retrieve

chunks = retrieve(
    query=f"{task['prompt']} {recent_error_msg}",
    top_k=4,
    sionna_version="2.0",
)
# Prepend `chunks` to the agent's context BEFORE SKILL.md routing.
```

## Storage layout

- `chroma_db/`   — ChromaDB persistent index (binary, gitignored)
- `store.py`     — load/store/retrieve helpers
- `README.md`    — this file
- `_seeds/`      — JSONL files used to bulk-seed the store; readable
  source-of-truth in case the index is rebuilt

## Index update cadence

| Index | Trigger | Tool |
|---|---|---|
| failure principles | Bulk-seed from `failure_library.md` | `scripts/seed_memory.py` |
| failure principles | Auto-distill after eval round | `benchmark/distill_failures.py` |
| Sionna API docs | On `sionna.__version__` change | `scripts/index_sionna_docs.py` |
| Research papers | Weekly arXiv pull | `tools/online_apis.py arxiv "<query>"` + cron |

The vector store complements (not replaces) the local human-readable
files. The flat file is the source-of-truth; the store is a retrieval
optimization layered on top.
