"""iteration_log_auto.md path must be ROOT-relative, not CWD-relative."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import run_benchmark


def test_log_path_is_root_relative():
    """Walk the source for the literal path and confirm it derives from ROOT."""
    src = Path(run_benchmark.__file__).read_text()
    assert 'Path("benchmark/_studies_archive/iteration_log_auto.md")' not in src, (
        "iteration log path is CWD-relative — should derive from ROOT")
    assert "ROOT /" in src and "iteration_log_auto.md" in src
