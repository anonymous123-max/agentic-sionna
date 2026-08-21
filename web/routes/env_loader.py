"""Load environment variables from a .env file at repo root.

Zero-dependency (pure stdlib). Called at dashboard startup so users can
put their API keys in a local .env (gitignored) instead of exporting
env vars manually every session.

Precedence: existing shell env > .env file. So if you `export
DASHBOARD_CHAT_API_KEY=...` in your shell, that takes priority over
whatever is in .env.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path | str) -> int:
    """Read KEY=VALUE lines from `path` into os.environ.

    Skips comments (# ...) and blank lines. Strips matching surrounding
    quotes. Does NOT overwrite variables already set in the shell env.

    Returns the number of variables loaded.
    """
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip inline comments (only when unquoted)
        if val and val[0] not in ("'", '"'):
            hashpos = val.find(" #")
            if hashpos > 0:
                val = val[:hashpos].rstrip()
        # Strip matching surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            n += 1
    return n
