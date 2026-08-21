#!/usr/bin/env python3
"""Drain _pending_principles.jsonl into chromadb via store.store_principle.

Designed to run periodically (cron, manual). Dedups via store's existing
0.85 threshold. Truncates queue file on success.

Silent on chromadb unavailable."""
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

try:
    import store
except Exception as e:
    print(f"[promote_pending] store unavailable: {e}", file=sys.stderr)
    sys.exit(0)

if not store.stats().get("available"):
    print("[promote_pending] store unavailable")
    sys.exit(0)

queue = _THIS.parent / "_pending_principles.jsonl"
if not queue.exists():
    print("[promote_pending] no queue")
    sys.exit(0)

added = 0
seen_set = set()  # in-memory dedup within this batch
for line in queue.read_text().splitlines():
    try:
        rec = json.loads(line)
    except Exception:
        continue
    key = (rec.get("task_id"), rec.get("verifier_name"))
    if key in seen_set:
        continue
    seen_set.add(key)
    try:
        # store_principle requires a full principle dict with principle_id
        pid = f"trial_{rec.get('task_id', 'unknown')}_{rec.get('verifier_name', 'unknown')}"
        principle = {
            "principle_id": pid,
            "observation": rec.get("principle", ""),
            "failure": rec.get("verifier_detail", ""),
            "correction": "",
            "principle": rec.get("principle", ""),
            "failure_class": rec.get("verifier_name", "unknown"),
            "sionna_version": "2.0",
            "task_tier": rec.get("tier", "unknown"),
            "generalizability": "medium",
        }
        ok = store.store_principle(principle)
        if ok:
            added += 1
    except Exception as e:
        print(f"  failed to add {rec.get('task_id')}: {e}", file=sys.stderr)

# On success, truncate the queue
if added > 0:
    queue.write_text("")
print(f"[promote_pending] promoted {added} principles, queue truncated.")
