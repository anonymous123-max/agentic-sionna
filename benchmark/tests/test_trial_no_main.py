"""Trial.py is a worker module — not an entry point. Its only public
surface is run_one + should_retry_trial. A separate orchestrator
(run_benchmark.py) handles the loop. This test prevents accidentally
re-introducing a competing main()."""
import importlib
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def test_trial_has_no_main():
    trial = importlib.import_module("trial")
    assert not hasattr(trial, "main"), (
        "trial.py must not define main() — orchestration lives in "
        "run_benchmark.py")


def test_trial_has_no_filter_tasks():
    trial = importlib.import_module("trial")
    assert not hasattr(trial, "filter_tasks"), (
        "trial.filter_tasks was only used by the deleted main() — "
        "run_benchmark.build_work_queue replaces it")


def test_trial_does_not_import_argparse():
    """trial.py is a worker; orchestration (and argparse) lives in
    run_benchmark.py. Guards against re-introducing argparse via either
    the old `ap = argparse.ArgumentParser()` block or any other form."""
    import trial
    src = inspect.getsource(trial)
    assert "import argparse" not in src
    assert "argparse." not in src
