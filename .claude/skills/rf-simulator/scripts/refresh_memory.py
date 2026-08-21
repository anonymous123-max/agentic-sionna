#!/usr/bin/env python3
"""Refresh the vector store from references/*.md.

Idempotent + cheap (~10ms when nothing changed). Called by lookup.py
before each retrieve, and may be invoked manually. Silent on chromadb
unavailability so it never breaks scripts that chain it.

Usage:
    python3 $RF_SKILL_DIR/scripts/refresh_memory.py
"""
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_MEMORY_DIR = _THIS.parent.parent / "memory"
sys.path.insert(0, str(_MEMORY_DIR))

try:
    import store
except Exception as e:
    print(f"[refresh_memory] store unavailable: {type(e).__name__}: {e}",
          file=sys.stderr)
    sys.exit(0)  # silent success — no point blocking

result = store.ensure_fresh()
print(json.dumps(result, indent=2))
