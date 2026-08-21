"""trial.py retry-pass condition — verify that a successful trial whose
stderr contained [TIMEOUT] is NOT retried (and overwritten). Only retry
when the trial actually failed."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def _make_workdir(tmp_path, score, has_timeout_in_stderr):
    """Set up a fake completed trial dir."""
    (tmp_path / "stderr.txt").write_text(
        "some output\n[TIMEOUT]\nmore output\n" if has_timeout_in_stderr else "ok\n"
    )
    (tmp_path / "result.json").write_text(json.dumps({
        "verification": {"score": score, "passed": score >= 1.0, "checks": []}
    }))
    return tmp_path


def test_should_retry_when_score_zero_and_timeout(tmp_path):
    """Real failure that timed out — should retry."""
    from trial import should_retry_trial
    wd = _make_workdir(tmp_path, score=0.0, has_timeout_in_stderr=True)
    assert should_retry_trial(wd) is True


def test_should_not_retry_when_passed_despite_timeout_marker(tmp_path):
    """Trial passed (score=1.0) but stderr contains [TIMEOUT]. Should NOT retry."""
    from trial import should_retry_trial
    wd = _make_workdir(tmp_path, score=1.0, has_timeout_in_stderr=True)
    assert should_retry_trial(wd) is False


def test_should_not_retry_when_no_timeout_marker(tmp_path):
    """No timeout marker — never retry regardless of score."""
    from trial import should_retry_trial
    wd = _make_workdir(tmp_path, score=0.0, has_timeout_in_stderr=False)
    assert should_retry_trial(wd) is False


def test_should_retry_when_result_json_missing(tmp_path):
    """No result.json — real timeout, harness killed mid-write. Retry."""
    from trial import should_retry_trial
    (tmp_path / "stderr.txt").write_text("[TIMEOUT]\n")
    assert should_retry_trial(tmp_path) is True


def test_should_retry_when_partial_score(tmp_path):
    """Score below 0.5 with [TIMEOUT] — likely killed mid-execution. Retry."""
    from trial import should_retry_trial
    wd = _make_workdir(tmp_path, score=0.3, has_timeout_in_stderr=True)
    assert should_retry_trial(wd) is True
