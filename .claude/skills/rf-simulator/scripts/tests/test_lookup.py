"""Structural tests for scripts/lookup.py — verify CLI shape without requiring
chromadb to be installed on the test machine."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
LOOKUP = SCRIPTS / "lookup.py"


def test_lookup_exists_and_executable():
    assert LOOKUP.is_file(), "lookup.py should be present"


def test_lookup_help_runs():
    """`lookup.py --help` should exit 0 even with no chromadb."""
    proc = subprocess.run(
        [sys.executable, str(LOOKUP), "--help"],
        capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "Semantic search" in proc.stdout
    assert "--top-k" in proc.stdout
    assert "--kind" in proc.stdout


def test_lookup_query_returns_zero_exit():
    """Even when the corpus is empty / chromadb missing, the script must
    exit 0 silently so agents can chain it with `||` etc."""
    proc = subprocess.run(
        [sys.executable, str(LOOKUP), "fictional query that probably has no hits"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0


def test_lookup_accepts_kind_flag():
    proc = subprocess.run(
        [sys.executable, str(LOOKUP), "test", "--kind", "principle", "--top-k", "1"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
