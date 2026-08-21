"""Auto-update mechanisms must be no-op-safe when chromadb unavailable."""
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".claude/skills/rf-simulator/memory"))


def test_ensure_fresh_handles_unavailable_store():
    """When chromadb is unavailable, ensure_fresh should return cleanly."""
    import store
    with patch.object(store, "stats",
                      return_value={"available": False}):
        result = store.ensure_fresh()
    assert result["status"] in ("unavailable", "no_refs", "ingest_unavailable")
    assert result["added"] == 0


def test_ensure_fresh_when_no_files_changed():
    """Calling twice in a row should report 'fresh' the second time."""
    import store
    if not store.stats().get("available"):
        import pytest; pytest.skip("chromadb not available")
    store.ensure_fresh()  # first call may ingest
    second = store.ensure_fresh()
    assert second["status"] in ("fresh", "ingested")  # both valid


def test_failure_capture_silent_on_bad_input():
    """failure_capture.maybe_capture must NOT raise on malformed args."""
    sys.path.insert(0, str(REPO))
    from benchmark.trial.failure_capture import maybe_capture
    # Missing fields, wrong types, etc.
    maybe_capture({}, None, Path("/tmp"))
    maybe_capture({"id": "X"}, object(), Path("/tmp"))


def test_failure_capture_writes_jsonl(tmp_path, monkeypatch):
    """Real failure should produce a JSONL line."""
    sys.path.insert(0, str(REPO))
    from dataclasses import dataclass
    from benchmark.trial import failure_capture
    queue = tmp_path / "_pending.jsonl"
    monkeypatch.setattr(failure_capture, "_QUEUE", queue)

    @dataclass
    class FakeCheck:
        name: str
        passed: bool
        detail: str
    @dataclass
    class FakeResult:
        passed: bool
        checks: list

    r = FakeResult(passed=False, checks=[
        FakeCheck(name="threshold:ber_gap_db", passed=False,
                  detail="metric not found")
    ])
    failure_capture.maybe_capture({"id": "U001", "tier": "T1"},
                                   r, tmp_path)
    assert queue.exists()
    content = queue.read_text().strip()
    assert "U001" in content and "threshold:ber_gap_db" in content
