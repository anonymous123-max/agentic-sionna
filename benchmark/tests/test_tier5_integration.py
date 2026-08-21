"""benchmark.verifier.verify() runs tier-5 domain checks for known capabilities."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def test_verify_calls_tier5_for_known_capability(tmp_path):
    from verifier import verify
    (tmp_path / "simulation_result.json").write_text(json.dumps({
        "numerical_metrics": {"pearson_r": 0.4, "n_points": 500},
    }))
    task = {"id": "X", "verifier": {"type": "execution_ok"},
            "capability": "channel_charting", "required_artifacts": []}
    res = verify(task, tmp_path, exec_success=True)
    names = [c.name for c in res.checks]
    assert any("channel_charting" in n for n in names), \
        f"no channel_charting in checks: {names}"
    assert not res.passed  # pearson r = 0.4 < 0.7


def test_verify_skips_tier5_for_unknown_capability(tmp_path):
    """Unknown capabilities pass through — generic checks only."""
    from verifier import verify
    (tmp_path / "simulation_result.json").write_text(json.dumps({}))
    task = {"id": "X", "verifier": {"type": "execution_ok"},
            "capability": "not_real", "required_artifacts": []}
    res = verify(task, tmp_path, exec_success=True)
    names = [c.name for c in res.checks]
    assert not any("channel_charting" in n for n in names)
