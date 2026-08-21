"""Coverage, BER, and OFDM grid routes."""

import logging
import threading
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from src.wireless.ray_tracing import compute_thz_coverage, THzConfig

from routes.shared import (
    _create_job,
    _update_job,
    _finish_job,
    _fail_job,
    _add_job_slice,
    _parse_furniture_items,
    _buildings_to_obstacles,
    is_job_cancelled,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)

coverage_bp = Blueprint("coverage", __name__)


@coverage_bp.route("/api/coverage/thz", methods=["POST"])
def thz_coverage():
    """Compute theoretical THz coverage (fast, no Sionna needed)."""
    data = request.json or {}
    room_width = float(data.get("room_width", 8.0))
    room_length = float(data.get("room_length", 6.0))
    frequency = float(data.get("frequency", 300e9))
    tx_x = float(data.get("tx_x", room_width / 2))
    tx_y = float(data.get("tx_y", room_length / 2))
    tx_z = float(data.get("tx_z", 2.5))
    resolution = float(data.get("resolution", 0.2))
    array_elements = int(data.get("array_elements", 256))

    furniture_items = _parse_furniture_items(data.get("furniture", []))

    config = THzConfig(
        frequency=frequency,
        tx_position=(tx_x, tx_y, tx_z),
        resolution=resolution,
        array_elements=array_elements,
    )

    coverage = compute_thz_coverage(room_width, room_length, furniture_items, config)

    return jsonify({
        "coverage": coverage.tolist(),
        "width": room_width,
        "length": room_length,
        "min_dbm": float(np.min(coverage)),
        "max_dbm": float(np.max(coverage)),
        "mean_dbm": float(np.mean(coverage)),
    })


@coverage_bp.route("/api/ber/compute", methods=["POST"])
def ber_compute():
    """Compute BER vs SNR curves for LDPC and Polar codes."""
    data = request.json or {}
    snr_min = float(data.get("snr_min", -2))
    snr_max = float(data.get("snr_max", 10))
    num_points = int(data.get("num_points", 25))

    snr_db = np.linspace(snr_min, snr_max, num_points)
    snr_linear = 10 ** (snr_db / 10)

    # Theoretical BPSK BER
    from scipy.special import erfc
    ber_uncoded = 0.5 * erfc(np.sqrt(snr_linear))

    # Approximate coded BER via effective SNR shift (coding gain).
    # Typical values: LDPC ~6 dB at rate 1/2, Polar ~5 dB at rate 1/2.
    ldpc_gain = 10 ** (6.0 / 10)
    polar_gain = 10 ** (5.0 / 10)

    ber_ldpc = 0.5 * erfc(np.sqrt(snr_linear * ldpc_gain))
    ber_polar = 0.5 * erfc(np.sqrt(snr_linear * polar_gain))

    # Clamp to reasonable floor
    ber_ldpc = np.maximum(ber_ldpc, 1e-7)
    ber_polar = np.maximum(ber_polar, 1e-7)
    ber_uncoded = np.maximum(ber_uncoded, 1e-7)

    return jsonify({
        "snr_db": snr_db.tolist(),
        "ber_uncoded": ber_uncoded.tolist(),
        "ber_ldpc": ber_ldpc.tolist(),
        "ber_polar": ber_polar.tolist(),
    })


@coverage_bp.route("/api/ofdm/grid", methods=["POST"])
def ofdm_grid():
    """Generate OFDM resource grid visualization data."""
    data = request.json or {}
    num_subcarriers = int(data.get("num_subcarriers", 72))
    num_symbols = int(data.get("num_symbols", 14))
    pilot_spacing = int(data.get("pilot_spacing", 6))

    rng = np.random.default_rng(42)
    grid = rng.uniform(0.3, 1.0, (num_subcarriers, num_symbols))

    # Mark pilots
    pilots = []
    for sc in range(0, num_subcarriers, pilot_spacing):
        for sym in [0, 7]:  # Pilot positions at symbol 0 and 7
            if sym < num_symbols:
                grid[sc, sym] = 1.0
                pilots.append([sc, sym])

    return jsonify({
        "grid": grid.tolist(),
        "pilots": pilots,
        "num_subcarriers": num_subcarriers,
        "num_symbols": num_symbols,
    })


def _adaptive_z_steps(z_min: float, z_max: float) -> int:
    """Choose number of Z slices based on the height range.

    Thin ranges (indoor rooms, ~2.7m) get fewer slices; tall ranges
    (outdoor, up to 30m) get more.  Clamped to [4, 12].
    """
    height = z_max - z_min
    if height <= 3:
        return 6
    if height <= 10:
        return 8
    return 12


@coverage_bp.route("/api/coverage/thz/multi-z", methods=["POST"])
def thz_coverage_multi_z():
    """Compute THz coverage at multiple Z heights with progressive streaming.

    Each Z slice is streamed to the frontend via SSE as soon as it's computed.
    The frontend starts rendering after a few slices arrive and updates live.

    Tries GPU-accelerated Sionna RT first, falls back to analytical THz model.
    Supports cancellation via is_job_cancelled().
    """
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            room_width = float(data.get("room_width", 8.0))
            room_length = float(data.get("room_length", 6.0))
            frequency = float(data.get("frequency", 300e9))
            tx_x = float(data.get("tx_x", room_width / 2))
            tx_y = float(data.get("tx_y", room_length / 2))
            tx_z = float(data.get("tx_z", 2.5))
            resolution = float(data.get("resolution", 0.2))
            # Support both legacy array_elements and new antenna_rows x antenna_cols
            ant_rows = int(data.get("antenna_rows", 0))
            ant_cols = int(data.get("antenna_cols", 0))
            if ant_rows > 0 and ant_cols > 0:
                array_elements = ant_rows * ant_cols
            else:
                array_elements = int(data.get("array_elements", 256))
            z_min = float(data.get("z_min", 0.5))
            z_max = float(data.get("z_max", 2.5))

            # Adaptive Z steps: backend decides based on scene height,
            # but client can still override via z_steps param
            z_steps = int(data.get("z_steps", 0)) or _adaptive_z_steps(z_min, z_max)

            furniture_items = _parse_furniture_items(data.get("furniture", []))
            building_obstacles = _buildings_to_obstacles(data.get("buildings", []))

            # Extract antenna config for analytical directivity
            ant_pattern = data.get("antenna_pattern", "tr38901")
            ant_azimuth = float(data.get("antenna_azimuth", 0))
            ant_elevation = float(data.get("antenna_elevation", 0))

            z_values = np.linspace(z_min, z_max, z_steps).tolist()
            slices = []

            # --- GPU-first: try Sionna RT if scene XML exists ---
            scene_id = data.get("scene_id")
            max_depth = int(data.get("max_depth", 5))
            ant_polarization = data.get("antenna_polarization", "cross")

            use_sionna = False
            sionna_scene = None
            if scene_id:
                xml_path = OUTPUTS_DIR / scene_id / "scene.xml"
                if xml_path.exists():
                    try:
                        from src.wireless.scene import load_scene, configure_transmitter, SceneConfig
                        from src.wireless.ray_tracing import CoverageSession, CoverageConfig

                        _update_job(job_id, 5, "Loading Sionna RT scene (GPU)...")
                        sc_config = SceneConfig(
                            frequency=frequency,
                            tx_antenna_pattern=ant_pattern,
                            tx_polarization=ant_polarization,
                            tx_array_rows=ant_rows if ant_rows > 0 else 8,
                            tx_array_cols=ant_cols if ant_cols > 0 else 8,
                        )
                        sionna_scene = load_scene(xml_path, sc_config)
                        configure_transmitter(sionna_scene, [tx_x, tx_y, tx_z])
                        use_sionna = True
                    except ImportError:
                        logger.info("Sionna not available, falling back to analytical model")
                    except Exception:
                        logger.warning("Sionna RT scene load failed, falling back to analytical model",
                                       exc_info=True)

            if use_sionna:
                # Create session once: reuses solver + cached scene bounds across Z slices
                session = CoverageSession(sionna_scene)
                for i, z_h in enumerate(z_values):
                    if is_job_cancelled(job_id):
                        return
                    _update_job(
                        job_id,
                        int(10 + 80 * i / z_steps),
                        f"GPU ray tracing Z={z_h:.1f}m ({i+1}/{z_steps})...",
                    )
                    cov_config = CoverageConfig(
                        max_depth=max_depth,
                        cell_size=(resolution, resolution),
                        samples_per_tx=100_000,
                        z_height=z_h,
                    )
                    radio_map = session.compute(cov_config)
                    coverage_data = radio_map.path_gain.numpy().squeeze()
                    coverage_db = 10 * np.log10(np.abs(coverage_data) + 1e-12)
                    slice_obj = {"z": z_h, "grid": coverage_db.tolist()}
                    slices.append(slice_obj)
                    _add_job_slice(job_id, slice_obj)
                engine = "Sionna RT (GPU)"
            else:
                # Analytical THz fallback (CPU)
                for i, z_h in enumerate(z_values):
                    if is_job_cancelled(job_id):
                        return
                    _update_job(
                        job_id,
                        int(10 + 80 * i / z_steps),
                        f"Computing Z={z_h:.1f}m ({i+1}/{z_steps})...",
                    )
                    config = THzConfig(
                        frequency=frequency,
                        tx_position=(tx_x, tx_y, tx_z),
                        resolution=resolution,
                        array_elements=array_elements,
                        rx_height=z_h,
                    )
                    coverage = compute_thz_coverage(
                        room_width, room_length, furniture_items, config,
                        obstacles=building_obstacles,
                    )

                    # Apply analytical directivity from antenna pattern + steering
                    if ant_pattern != "iso":
                        coverage = _apply_directivity(
                            coverage, room_width, room_length,
                            tx_x, tx_y, tx_z, z_h,
                            ant_pattern, ant_azimuth, ant_elevation,
                        )

                    slice_obj = {"z": z_h, "grid": coverage.tolist()}
                    slices.append(slice_obj)
                    _add_job_slice(job_id, slice_obj)
                engine = "THz Model + Directivity"

            # Aggregate stats from middle slice
            mid = slices[len(slices) // 2]["grid"]
            mid_flat = np.array(mid).flatten()

            result = {
                "slices": slices,
                "z_values": z_values,
                "z_steps": z_steps,
                "room": {"width": room_width, "length": room_length},
                "min_dbm": float(np.min(mid_flat)),
                "max_dbm": float(np.max(mid_flat)),
                "mean_dbm": float(np.mean(mid_flat)),
                "engine": engine,
            }
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Multi-Z coverage computation failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


def _apply_directivity(coverage, room_width, room_length,
                       tx_x, tx_y, tx_z, z_h,
                       ant_pattern, ant_azimuth, ant_elevation):
    """Apply antenna directivity pattern to coverage grid.

    Supports 'dipole' (half-wave, 2.15 dBi peak) and 'tr38901' (3GPP TR 38.901
    Table 7.3-1 with 65 deg half-power beamwidth).
    """
    nx, ny = coverage.shape[1], coverage.shape[0]
    gx = np.linspace(0, room_width, nx)
    gy = np.linspace(0, room_length, ny)
    GX, GY = np.meshgrid(gx, gy)
    dx = GX - tx_x
    dy = GY - tx_y
    dz = z_h - tx_z

    cell_az = np.degrees(np.arctan2(dy, dx))
    horiz_dist = np.sqrt(dx**2 + dy**2)
    cell_el = np.degrees(np.arctan2(dz, np.maximum(horiz_dist, 0.01)))

    az_off = cell_az - ant_azimuth
    az_off = (az_off + 180) % 360 - 180
    el_off = cell_el - ant_elevation

    if ant_pattern == "dipole":
        # Half-wave dipole: cos^2(theta) pattern, 2.15 dBi peak gain
        az_rad = np.radians(az_off)
        dir_factor = np.cos(az_rad) ** 2
        dir_db = 10 * np.log10(np.maximum(dir_factor, 1e-3)) + 2.15
    else:
        # 3GPP TR 38.901 Table 7.3-1: A(theta) = -min(12*(theta/theta_3dB)^2, Am)
        # theta_3dB = 65 deg (half-power beamwidth), Am = 30 dB (front-to-back ratio)
        theta = np.sqrt(az_off**2 + el_off**2)
        dir_db = np.maximum(-12 * (theta / 65) ** 2, -30)

    return coverage + dir_db


@coverage_bp.route("/api/coverage/compute", methods=["POST"])
def compute_coverage():
    """Compute Sionna RT coverage map (requires sionna)."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            xml_path = data.get("xml_path")
            if not xml_path:
                _fail_job(job_id, "xml_path required")
                return

            _update_job(job_id, 10, "Loading scene...")

            room_width = float(data.get("room_width", 8.0))
            room_length = float(data.get("room_length", 6.0))
            frequency = float(data.get("frequency", 60e9))
            tx_x = float(data.get("tx_x", room_width / 2))
            tx_y = float(data.get("tx_y", room_length / 2))
            tx_z = float(data.get("tx_z", 2.5))

            _update_job(job_id, 30, "Computing coverage...")

            try:
                from src.wireless.scene import load_scene, configure_transmitter, SceneConfig
                from src.wireless.ray_tracing import compute_coverage_map, CoverageConfig

                config = SceneConfig(frequency=frequency)
                scene = load_scene(Path(xml_path), config)
                configure_transmitter(scene, [tx_x, tx_y, tx_z])

                if is_job_cancelled(job_id):
                    return

                _update_job(job_id, 50, "Ray tracing...")
                cov_config = CoverageConfig(
                    max_depth=int(data.get("max_depth", 5)),
                    cell_size=(
                        float(data.get("cell_size", 0.5)),
                        float(data.get("cell_size", 0.5)),
                    ),
                    samples_per_tx=int(data.get("samples", 100000)),
                    z_height=float(data.get("z_height", 1.5)),
                )
                radio_map = compute_coverage_map(scene, cov_config)

                _update_job(job_id, 80, "Extracting data...")
                coverage_data = radio_map.path_gain.numpy().squeeze()
                coverage_db = 10 * np.log10(np.abs(coverage_data) + 1e-12)

                result = {
                    "coverage": coverage_db.tolist(),
                    "min_dbm": float(np.min(coverage_db)),
                    "max_dbm": float(np.max(coverage_db)),
                    "mean_dbm": float(np.mean(coverage_db)),
                    "engine": "sionna_rt",
                }
            except ImportError:
                _update_job(job_id, 50, "Using theoretical model (Sionna not available)...")
                config = THzConfig(
                    frequency=frequency,
                    tx_position=(tx_x, tx_y, tx_z),
                    resolution=float(data.get("cell_size", 0.5)),
                )
                coverage = compute_thz_coverage(room_width, room_length, [], config)
                result = {
                    "coverage": coverage.tolist(),
                    "min_dbm": float(np.min(coverage)),
                    "max_dbm": float(np.max(coverage)),
                    "mean_dbm": float(np.mean(coverage)),
                    "engine": "theoretical_thz",
                }

            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Coverage computation failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})
