"""Backward-compat: benchmark/verifier.py exposes the same surface after D.5.

After Task D.5, the task-agnostic plausibility helpers live in
benchmark/_verifier_core.py. The benchmark verifier imports from there
directly (no sys.path hack) so existing callers keep working unchanged.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def test_benchmark_verifier_reexports_core_names():
    from benchmark.verifier import (
        CheckResult, load_sim_result, load_all_code,
        extract_scalar, extract_array, check_plausibility,
    )
    # Identity check — must be the same objects, not duplicates
    import benchmark._verifier_core as core
    assert CheckResult is core.CheckResult
    assert check_plausibility is core.check_plausibility
    assert load_sim_result is core.load_sim_result
    assert load_all_code is core.load_all_code
    assert extract_scalar is core.extract_scalar
    assert extract_array is core.extract_array


def test_benchmark_verifier_specific_dispatch_still_present():
    """The task-spec-driven side stays in benchmark/verifier.py."""
    from benchmark.verifier import (
        run_checks, check_metric_threshold, check_metric_range,
        check_metric_monotone, check_count, check_value_exact,
        check_composite, check_code_contains, verify, VerificationResult,
    )
    assert callable(run_checks)
    assert callable(verify)
    assert callable(check_metric_threshold)
