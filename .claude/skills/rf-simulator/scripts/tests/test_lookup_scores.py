"""lookup.py CLI must surface a similarity score for each hit so agents
can judge relevance."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
LOOKUP = REPO / ".claude/skills/rf-simulator/scripts/lookup.py"


def test_lookup_output_contains_score_marker():
    """Run lookup.py with a query that's known to hit (per iter-34 audit:
    'sionna v2 namespace' returns the v0.x→v2.0 import principle)."""
    r = subprocess.run([sys.executable, str(LOOKUP), "sionna v2 namespace",
                        "--top-k", "2"],
                       capture_output=True, text=True, timeout=60)
    # Don't require returncode 0 — chromadb may not be installed.
    if "available=False" in r.stdout or "chromadb" in r.stderr.lower():
        import pytest
        pytest.skip("chromadb not available")
    # If chromadb IS available, output must include score=
    assert "score=" in r.stdout, (
        f"lookup.py output missing score marker. stdout:\n{r.stdout[:500]}"
    )
