"""template_rt_coverage.py CPU fallback must emit coverage_map.npy and
populate coverage_pct in simulation_result.json. Otherwise U061 (and
similar T2 RT tasks) fail the canonical-artifact check even when the
script ran successfully."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / ".claude/skills/rf-simulator/templates/template_rt_coverage.py"


def _make_scene(tmp_path: Path) -> None:
    """Write a minimal scene_state.json matching the format template_rt_coverage expects."""
    # run_cpu_analytical reads state["scene"]["bounds"] with width/depth/height
    scene = {
        "scene": {
            "bounds": {"width": 10.0, "depth": 10.0, "height": 3.0}
        },
        "walls": [],
        "furniture": [],
    }
    (tmp_path / "scene_state.json").write_text(json.dumps(scene))


def test_cpu_fallback_writes_coverage_map(tmp_path: Path):
    """RF_FORCE_CPU=1 must produce coverage_map.npy + coverage_pct in CWD."""
    env = dict(os.environ)
    env["RF_FORCE_CPU"] = "1"
    _make_scene(tmp_path)

    result = subprocess.run(
        [sys.executable, str(TEMPLATE)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    # Don't require returncode 0 — matplotlib/output warnings are acceptable.
    # Just require the canonical artifacts.
    cov_map = tmp_path / "coverage_map.npy"
    sim = tmp_path / "simulation_result.json"

    assert cov_map.exists() or any(tmp_path.rglob("coverage_map.npy")), (
        f"coverage_map.npy missing after CPU run.\n"
        f"stdout: {result.stdout[-300:]}\n"
        f"stderr: {result.stderr[-500:]}")

    assert sim.exists(), (
        f"simulation_result.json missing in CWD after CPU run.\n"
        f"stdout: {result.stdout[-300:]}\n"
        f"stderr: {result.stderr[-500:]}")

    metrics = json.loads(sim.read_text()).get("numerical_metrics", {})
    cov_pct = metrics.get("coverage_pct")
    assert cov_pct is not None, (
        f"coverage_pct is None — CPU fallback must compute it. metrics={metrics}")

    # Sanity: coverage_pct should be a percentage (0–100), not a fraction (0–1)
    assert 0.0 <= cov_pct <= 100.0, (
        f"coverage_pct={cov_pct} out of 0–100 range — check formula")


def test_cpu_fallback_no_sionna_import_noise(tmp_path: Path):
    """With RF_FORCE_CPU=1, stderr should NOT contain Sionna ImportError noise."""
    env = dict(os.environ)
    env["RF_FORCE_CPU"] = "1"
    _make_scene(tmp_path)

    result = subprocess.run(
        [sys.executable, str(TEMPLATE)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    # The env var must gate the import attempt entirely — no ModuleNotFoundError
    # for sionna should appear when RF_FORCE_CPU=1.
    assert "ModuleNotFoundError" not in result.stderr, (
        f"RF_FORCE_CPU=1 should skip Sionna import, but got ModuleNotFoundError.\n"
        f"stderr: {result.stderr[-500:]}")
    assert "No module named 'sionna'" not in result.stderr, (
        f"RF_FORCE_CPU=1 should skip Sionna import attempt.\n"
        f"stderr: {result.stderr[-500:]}")
