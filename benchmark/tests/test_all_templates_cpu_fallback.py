"""Every Sionna-using template must run under RF_FORCE_CPU=1 without
crashing. Plan B.4 fixed template_rt_coverage; this test guards against
the same gap re-appearing in others."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO / ".claude/skills/rf-simulator/templates"


def _all_templates() -> list[Path]:
    return sorted(TEMPLATES_DIR.glob("template_*.py"))


# Templates known to require an existing scene_state.json or similar
# external input — skipping is OK; the F.1 sanity check already proved
# these terminate cleanly with the input-missing error.
_REQUIRES_INPUT_FILE = {"template_rt_coverage.py", "template_rt_to_phy.py"}


@pytest.mark.parametrize("template", _all_templates(),
                          ids=lambda p: p.name)
def test_template_cpu_fallback_no_crash(template: Path, tmp_path):
    """Run each template under RF_FORCE_CPU=1 in an empty workdir.
    Expected: returncode 0, OR a clean exit with stderr message about
    missing input (for templates that legitimately need a scene file)."""
    env = {**os.environ,
           "RF_FORCE_CPU": "1",
           "RF_SKIP_TEMPLATE_WARN": "1"}
    r = subprocess.run([sys.executable, str(template)],
                       cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    if template.name in _REQUIRES_INPUT_FILE:
        # OK to fail with a clean "scene_path not found" message
        if r.returncode != 0:
            stderr_tail = (r.stderr or "")[-300:]
            assert "scene" in stderr_tail.lower() or "not found" in stderr_tail.lower(), (
                f"{template.name} crashed unexpectedly:\n{stderr_tail}"
            )
        return
    # Other templates should exit 0 with default PARAMS
    assert r.returncode == 0, (
        f"{template.name} crashed under RF_FORCE_CPU=1:\n"
        f"stdout: {r.stdout[-300:]}\n"
        f"stderr: {r.stderr[-500:]}"
    )
