"""Tests for tier-5 domain-specific verifier checks."""
import json


def _write_sim(tmp_path, payload):
    (tmp_path / "simulation_result.json").write_text(json.dumps(payload))


def test_channel_charting_passes_with_high_pearson(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"pearson_r": 0.82, "n_points": 500},
    })
    out = check_tier5_domain("channel_charting", tmp_path)
    assert all(c.passed for c in out), [c.detail for c in out if not c.passed]


def test_channel_charting_fails_with_low_pearson(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"pearson_r": 0.42, "n_points": 500},
    })
    out = check_tier5_domain("channel_charting", tmp_path)
    failed = [c for c in out if not c.passed]
    assert any("pearson" in c.name for c in failed)


def test_channel_charting_fails_with_too_few_points(tmp_path):
    """A chart on 30 CSI vectors is overfitting — require >=100."""
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"pearson_r": 0.95, "n_points": 30},
    })
    out = check_tier5_domain("channel_charting", tmp_path)
    failed = [c for c in out if not c.passed]
    assert any("n_points" in c.name for c in failed)


def test_unknown_capability_returns_empty(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    assert check_tier5_domain("not_a_real_thing", tmp_path) == []


# ── C.2 ISAC Pareto-front ──────────────────────────────────────────────

def test_isac_passes_with_pareto_curve(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "comm_rate_bps_hz": [10, 9.5, 9, 8.5, 8, 7.5],
            "sensing_rmse_m": [1.0, 0.6, 0.4, 0.3, 0.2, 0.15],
            "n_realizations": 25,
        },
    })
    out = check_tier5_domain("isac_tradeoff_curve", tmp_path)
    assert all(c.passed for c in out), [c.detail for c in out if not c.passed]


def test_isac_fails_too_few_realizations(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "comm_rate_bps_hz": [10, 9, 8],
            "sensing_rmse_m": [1.0, 0.5, 0.3],
            "n_realizations": 3,
        },
    })
    out = check_tier5_domain("isac_tradeoff_curve", tmp_path)
    assert any("realizations" in c.name and not c.passed for c in out)


def test_isac_fails_when_no_tradeoff(tmp_path):
    """If sensing improves and comm rate also improves, it's not a tradeoff."""
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "comm_rate_bps_hz": [8, 9, 10],
            "sensing_rmse_m": [1.0, 0.5, 0.3],
            "n_realizations": 25,
        },
    })
    out = check_tier5_domain("isac_tradeoff_curve", tmp_path)
    assert any("monotone" in c.name and not c.passed for c in out)


# ── C.3 OTFS 2D pilot pattern ─────────────────────────────────────────

def test_otfs_passes_with_2d_pilot_code(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "delay_doppler_grid = torch.zeros(N_delay, N_doppler)\n"
        "delay_doppler_grid[pilot_delay, pilot_doppler] = 1.0\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("otfs_waveform", tmp_path)
    assert all(c.passed for c in out), [c.detail for c in out if not c.passed]


def test_otfs_fails_with_1d_pilots_only(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "pilot_subcarriers = torch.arange(0, N, 4)\n"
        "tx_grid[pilot_subcarriers] = 1.0\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("otfs_waveform", tmp_path)
    assert any("delay_doppler" in c.name and not c.passed for c in out)


# ── C.4 Semantic communication accuracy ──────────────────────────────

def test_semantic_passes_with_accuracy(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"classification_accuracy": 0.87},
    })
    out = check_tier5_domain("semantic_communication", tmp_path)
    assert all(c.passed for c in out), [c.detail for c in out if not c.passed]


def test_semantic_fails_when_only_ber_reported(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"ber_simulated": [0.1, 0.01]},
    })
    out = check_tier5_domain("semantic_communication", tmp_path)
    assert any("accuracy" in c.name and not c.passed for c in out)


# ── C.5 THz molecular absorption ──────────────────────────────────────

def test_thz_passes_with_absorption_term(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "import numpy as np\n"
        "molecular_absorption_db = 10  # ITU-R P.676 at 300 GHz\n"
        "path_loss_total = fspl + molecular_absorption_db\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("thz_channel", tmp_path)
    assert all(c.passed for c in out)


def test_thz_fails_friis_only(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "fspl = 20*np.log10(d) + 20*np.log10(f) + 32.45\n"
        "received_power = tx_power - fspl\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("thz_channel", tmp_path)
    assert any("absorption" in c.name and not c.passed for c in out)


# ── C.6 Near-field Rayleigh distance ─────────────────────────────────

def test_nearfield_passes_with_rayleigh_term(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "rayleigh_distance = 2 * D**2 / wavelength\n"
        "if user_dist < rayleigh_distance:\n"
        "    use_spherical_wave_steering()\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("near_field_beamforming", tmp_path)
    assert all(c.passed for c in out)


def test_nearfield_fails_without_rayleigh(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    (tmp_path / "simulation.py").write_text(
        "steering_vec = exp(1j * 2 * pi * d * sin(theta) / lambda_)\n")
    _write_sim(tmp_path, {"numerical_metrics": {}})
    out = check_tier5_domain("near_field_beamforming", tmp_path)
    assert any("rayleigh" in c.name and not c.passed for c in out)


# ── C.7 Federated client variance ─────────────────────────────────────

def test_federated_passes_with_varying_sizes(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"client_dataset_sizes": [100, 250, 80, 410]},
    })
    out = check_tier5_domain("federated_learning", tmp_path)
    assert all(c.passed for c in out)


def test_federated_fails_with_equal_sizes(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {"client_dataset_sizes": [200, 200, 200, 200]},
    })
    out = check_tier5_domain("federated_learning", tmp_path)
    assert any("client_size_variance" in c.name and not c.passed for c in out)


# ── C.8 STAR-RIS energy conservation ──────────────────────────────────

def test_starris_passes_with_unit_energy(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "t_coeffs_abs2": [0.4, 0.5, 0.6],
            "r_coeffs_abs2": [0.6, 0.5, 0.4],
        },
    })
    out = check_tier5_domain("star_ris", tmp_path)
    assert all(c.passed for c in out)


def test_starris_fails_when_t_plus_r_not_unit(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "t_coeffs_abs2": [0.5, 0.5, 0.5],
            "r_coeffs_abs2": [0.5, 0.6, 0.7],
        },
    })
    out = check_tier5_domain("star_ris", tmp_path)
    assert any("energy_conservation" in c.name and not c.passed for c in out)


# ── C.9 Channel prediction NMSE-vs-horizon ────────────────────────────

def test_channel_pred_passes_with_increasing_nmse(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "horizons": [1, 5, 10, 20],
            "nmse_db_at_horizon": [-15.0, -12.0, -8.0, -4.0],
        },
    })
    out = check_tier5_domain("channel_prediction", tmp_path)
    assert all(c.passed for c in out)


def test_channel_pred_fails_with_flat_nmse(tmp_path):
    from benchmark._verifier_core import check_tier5_domain
    _write_sim(tmp_path, {
        "numerical_metrics": {
            "horizons": [1, 5, 10, 20],
            "nmse_db_at_horizon": [-12.0, -12.0, -12.0, -12.0],
        },
    })
    out = check_tier5_domain("channel_prediction", tmp_path)
    assert any("horizon" in c.name and not c.passed for c in out)


# ── C.10 Integration: verify_output.py --capability wiring ───────────────────

def test_verify_output_runs_tier5_check(tmp_path):
    """verify_output.py --capability channel_charting must run the domain check."""
    import subprocess, sys
    from pathlib import Path
    skill_root = Path(__file__).resolve().parents[2]
    script = skill_root / "scripts" / "verify_output.py"
    _write_sim(tmp_path, {
        "numerical_metrics": {"pearson_r": 0.42, "n_points": 500},
    })
    proc = subprocess.run(
        [sys.executable, str(script), "--workdir", str(tmp_path),
         "--capability", "channel_charting"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0  # 0.42 < 0.7 should fail
    assert "pearson_r_threshold" in proc.stdout, proc.stdout


def test_verify_output_skips_tier5_without_capability(tmp_path):
    """Without --capability, no tier-5 domain check runs."""
    import subprocess, sys
    from pathlib import Path
    skill_root = Path(__file__).resolve().parents[2]
    script = skill_root / "scripts" / "verify_output.py"
    _write_sim(tmp_path, {
        "numerical_metrics": {"pearson_r": 0.42, "n_points": 500},
    })
    proc = subprocess.run(
        [sys.executable, str(script), "--workdir", str(tmp_path)],
        capture_output=True, text=True, timeout=30)
    # No --capability means no tier-5 dispatch — overall passes (BER not present).
    assert "pearson_r_threshold" not in proc.stdout


def test_action_plan_schema_validates_minimal_plan():
    """The published action-plan schema accepts a minimal valid example."""
    import json as _json
    from pathlib import Path as _Path
    schema_path = (_Path(__file__).resolve().parents[2]
                   / "templates" / "result_schema_action_plan.json")
    schema = _json.loads(schema_path.read_text())
    minimal = {
        "coverage_current": 82.5, "coverage_target": 90.0,
        "blind_spots": [{"location": [3.0, 4.0], "area_m2": 1.2,
                          "cause": "wall_occlusion"}],
        "actions": [{"type": "reposition", "ap_id": "AP1",
                     "delta": [0.0, 1.5, 0.0], "expected_gain_pp": 4.0}],
        "confidence": 0.7, "stop_recommended": False,
    }
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(minimal, schema)
    except ImportError:
        for k in schema["required"]:
            assert k in minimal, f"required key {k} missing from minimal example"
