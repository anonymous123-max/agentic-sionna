# .claude/skills/rf-simulator/scripts/tests/test_verifier_core.py
"""Tests for the verifier core (now lives in benchmark/_verifier_core.py after D.5).

verify_output.py locates the repo root via __file__ and inserts it into
sys.path, so it never needs PYTHONPATH to be set by the caller.
"""
import json
from pathlib import Path


def test_module_importable_via_benchmark_package():
    """After D.5, _verifier_core lives in benchmark/ — confirm it imports."""
    import benchmark._verifier_core  # noqa: F401


def test_check_result_dataclass_shape():
    from benchmark._verifier_core import CheckResult
    r = CheckResult(name="x", passed=True, detail="ok")
    assert r.name == "x" and r.passed is True and r.detail == "ok"


def test_load_sim_result_missing_returns_none(tmp_path):
    from benchmark._verifier_core import load_sim_result
    assert load_sim_result(tmp_path) is None


def test_load_sim_result_parses_json(tmp_path):
    from benchmark._verifier_core import load_sim_result
    (tmp_path / "simulation_result.json").write_text('{"a": 1}')
    assert load_sim_result(tmp_path) == {"a": 1}


def test_extract_scalar_finds_nested():
    from benchmark._verifier_core import extract_scalar
    sim = {"numerical_metrics": {"coverage_pct": 87.5}}
    assert extract_scalar(sim, "coverage_pct", {"metric": "coverage_pct"}) == 87.5


def test_check_plausibility_flags_impossible_ber(tmp_path):
    """BER=1e-5 at -10 dB SNR is physically impossible — should fail."""
    from benchmark._verifier_core import check_plausibility
    sim = {
        "numerical_metrics": {
            "ebn0_db": [-10.0, 0.0, 10.0],
            "ber_simulated": [1e-5, 1e-5, 1e-6],
        }
    }
    (tmp_path / "simulation_result.json").write_text(json.dumps(sim))
    out = check_plausibility(tmp_path)
    names = [c.name for c in out if not c.passed]
    assert any("ber_at_lowest_snr" in n for n in names)


def test_check_plausibility_passes_realistic_ber(tmp_path):
    """BER 0.4 at low SNR, 1e-4 at high — physically reasonable."""
    from benchmark._verifier_core import check_plausibility
    sim = {
        "numerical_metrics": {
            "ebn0_db": [-2.0, 4.0, 10.0],
            "ber_simulated": [0.4, 1e-2, 1e-4],
        }
    }
    (tmp_path / "simulation_result.json").write_text(json.dumps(sim))
    assert all(c.passed for c in check_plausibility(tmp_path))


def test_verify_output_runs_without_benchmark_on_path(tmp_path):
    """verify_output.py main() must succeed when benchmark/ is not importable."""
    import subprocess, sys, json as _json
    skill_root = Path(__file__).resolve().parents[2]
    script = skill_root / "scripts" / "verify_output.py"
    # Realistic sim result so plausibility checks pass
    (tmp_path / "simulation_result.json").write_text(_json.dumps({
        "numerical_metrics": {
            "ebn0_db": [0, 5, 10],
            "ber_simulated": [0.08, 0.01, 1e-3],
        }
    }))
    # Run with PYTHONPATH stripped of the repo root so benchmark/ can't be found
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": ""}
    proc = subprocess.run(
        [sys.executable, str(script), "--workdir", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "OVERALL: PASS" in proc.stdout
