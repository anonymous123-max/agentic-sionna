"""Continuous-metrics extraction in run_one must populate every alias path."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import trial
from benchmark.verifier import extract_scalar, load_sim_result


def test_canonical_metric_extracted(tmp_path):
    sim = {"numerical_metrics": {"ber_gap_db": 1.5}}
    (tmp_path / "simulation_result.json").write_text(json.dumps(sim))
    s = load_sim_result(tmp_path)
    out = {}
    for canon, aliases in trial.CONTINUOUS_METRICS:
        for a in aliases:
            v = extract_scalar(s, a, None)
            if v is not None:
                out[canon] = v
                break
    assert out["ber_gap_db"] == 1.5


def test_alias_resolves_to_canonical(tmp_path):
    """received_power_gain_db should resolve to ris_gain_db (canonical)."""
    sim = {"numerical_metrics": {"received_power_gain_db": 6.2}}
    (tmp_path / "simulation_result.json").write_text(json.dumps(sim))
    s = load_sim_result(tmp_path)
    out = {}
    for canon, aliases in trial.CONTINUOUS_METRICS:
        for a in aliases:
            v = extract_scalar(s, a, None)
            if v is not None:
                out[canon] = v
                break
    assert out["ris_gain_db"] == 6.2  # canonical name, alias-resolved


def test_first_alias_wins(tmp_path):
    """If multiple aliases match, the first one in the list wins."""
    sim = {"numerical_metrics": {
        "map_mae_db": 1.0,
        "radio_map_mae": 2.0,
        "path_loss_mae_db": 3.0,
    }}
    (tmp_path / "simulation_result.json").write_text(json.dumps(sim))
    s = load_sim_result(tmp_path)
    out = {}
    for canon, aliases in trial.CONTINUOUS_METRICS:
        for a in aliases:
            v = extract_scalar(s, a, None)
            if v is not None:
                out[canon] = v
                break
    assert out["map_mae_db"] == 1.0  # first alias in [map_mae_db, radio_map_mae, ...]


def test_manifest_covers_all_old_metrics():
    """Confirm the manifest has at least every metric the old code extracted."""
    canonical_names = {c for c, _ in trial.CONTINUOUS_METRICS}
    expected = {"ber_gap_db", "ber_at_snr", "nmse_db", "doppler_hz",
                "coverage_pct", "path_loss_range", "ris_gain_db",
                "noise_power_dbm", "peak_se", "snr_at_ber_1e4_db",
                "nve", "map_mae_db", "coding_gain_db"}
    missing = expected - canonical_names
    assert not missing, f"missing canonical metrics: {missing}"
