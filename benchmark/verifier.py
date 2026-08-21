"""Unified verifier for the merged task set.

Task-agnostic plausibility lives in benchmark/_verifier_core.py — this module
imports from there and adds the task-spec-driven dispatch (metric_threshold,
metric_range, code_contains, scene-collision, composite, etc.).
"""
from __future__ import annotations
import json
import re
from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path

from benchmark._verifier_core import (
    CheckResult,
    load_sim_result,
    load_all_code,
    load_bash_commands,
    extract_scalar,
    extract_array,
    check_plausibility,
    check_tier5_domain,
    _METRIC_ALIASES,
    _get_ber_arrays,
    _find_snr_at_ber,
    _find_scalar_anywhere,
    _find_array_anywhere,
)


def _find_list_len_anywhere(d: dict, keyword_substrings: list[str]) -> int | None:
    """Recursively search for the first list/dict whose key contains any of the
    given lowercase substrings — returns its length. If nothing matches as a
    collection, also accept an integer scalar at a key like 'num_cells' as the
    count itself (agents often pre-compute the count rather than producing a
    list of items)."""
    if not isinstance(d, dict):
        return None
    # Priority 1: lists or dicts keyed on the keyword
    for k, v in d.items():
        lk = k.lower()
        if isinstance(v, (list, dict)) and any(s in lk for s in keyword_substrings):
            return len(v)
    # Priority 2: integer scalars at keys like num_cells / n_users / total_bs
    # OR user_count / cell_count (suffix variant).
    for k, v in d.items():
        lk = k.lower()
        if isinstance(v, int) and any(s in lk for s in keyword_substrings):
            has_prefix = any(p in lk for p in ("num_", "n_", "total_", "count_"))
            has_suffix = lk.endswith(("_count", "_num", "_total"))
            if has_prefix or has_suffix:
                return int(v)
    for v in d.values():
        if isinstance(v, dict):
            found = _find_list_len_anywhere(v, keyword_substrings)
            if found is not None:
                return found
    return None


@dataclass
class VerificationResult:
    passed: bool
    score: float               # fraction of assertions passed, [0,1]
    checks: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [c.__dict__ for c in self.checks],
            "notes": self.notes,
        }


def _scene_from_dir(output_dir: Path) -> dict | None:
    p = output_dir / "scene_state.json"
    if not p.exists():
        candidates = list(output_dir.rglob("scene_state.json"))
        if not candidates:
            return None
        p = candidates[0]
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _iter_furniture(scene: dict):
    """Yield (id, x, y, w, d, theta) tuples for furniture with usable AABBs.

    Entries lacking position OR dimensions are silently skipped — they
    contribute no collision/in-bounds geometry. Tolerant of:
      - rooms-as-singular `room`
      - bare-string furniture lists
      - schema variants (position_m, dims_m, etc.) handled in _furniture_tuple
    """
    if not isinstance(scene, dict):
        return
    rooms = scene.get("rooms") or []
    if not rooms and isinstance(scene.get("room"), dict):
        rooms = [scene["room"]]
    if isinstance(rooms, list):
        for r in rooms:
            if not isinstance(r, dict):
                continue
            for f in (r.get("furniture") or []):
                if isinstance(f, dict):
                    t = _furniture_tuple(f)
                    if t is not None:
                        yield t
    for f in (scene.get("furniture") or []):
        if isinstance(f, dict):
            t = _furniture_tuple(f)
            if t is not None:
                yield t


def _furniture_tuple(f):
    """Extract (id, x, y, w, d, theta) tuple from a furniture dict.

    Tolerant of multiple schema variants observed in agent output:
      position:   `position` | `position_m` | `pos`
      dimensions: `dimensions` | `dimensions_m` | `dims_m` | `dims` | `size`
                  | (`radius_m` → square 2r×2r)

    Returns None when neither position nor dimensions can be recovered —
    callers MUST skip None entries (they have no usable AABB and are not
    a collision candidate).
    """
    if not isinstance(f, dict):
        return None

    # Position: try multiple keys, expect list/tuple of length ≥ 2
    pos = None
    for k in ("position", "position_m", "pos", "location", "center"):
        v = f.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            pos = v
            break
    if pos is None:
        return None

    # Dimensions: try multiple keys, expect list/tuple of length ≥ 2.
    # Special: `radius_m` (circle/round) → square AABB of 2r × 2r.
    dim = None
    for k in ("dimensions", "dimensions_m", "dims_m", "dims", "size"):
        v = f.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            dim = v
            break
    if dim is None:
        r = f.get("radius_m") or f.get("radius")
        if isinstance(r, (int, float)):
            dim = [2 * float(r), 2 * float(r)]
    if dim is None:
        # No dimension info at all: skip from collision/in-bounds checks
        # (the agent provided position but no extent — we can't construct AABB).
        return None

    try:
        x = float(pos[0])
        y = float(pos[1])
        w = float(dim[0])
        d = float(dim[1])
    except (TypeError, ValueError):
        return None

    theta = float(f.get("theta") or f.get("rotation") or f.get("orientation_deg") or 0)
    return (f.get("id") or f.get("name") or f.get("type") or "?", x, y, w, d, theta)


def _check_scene_collision_free(output_dir: Path) -> CheckResult:
    scene = _scene_from_dir(output_dir)
    if scene is None:
        return CheckResult(name="collision_free", passed=False,
                           detail="no scene_state.json")
    boxes = []
    for fid, x, y, w, d, _theta in _iter_furniture(scene):
        # Treat each furniture as axis-aligned box centered at (x, y)
        boxes.append((fid, x - w/2, y - d/2, x + w/2, y + d/2))
    # Check pairwise overlap
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if (a[1] < b[3] and a[3] > b[1] and a[2] < b[4] and a[4] > b[2]):
                overlaps.append((a[0], b[0]))
    passed = len(overlaps) == 0
    return CheckResult(name="collision_free", passed=passed,
                       detail=f"boxes={len(boxes)} overlaps={len(overlaps)}")


def _check_scene_in_bounds(output_dir: Path) -> CheckResult:
    scene = _scene_from_dir(output_dir)
    if scene is None:
        return CheckResult(name="in_bounds", passed=False,
                           detail="no scene_state.json")
    # Scene bounds come from rooms' dimensions; furniture must fit inside its room
    rooms = scene.get("rooms") or []
    if not rooms:
        # No rooms → scene has a top-level `bounds` key?
        b = scene.get("bounds") or scene.get("scene", {}).get("bounds")
        if not b:
            return CheckResult(name="in_bounds", passed=True,
                               detail="no rooms/bounds to check")
    out_of_bounds = 0
    total = 0
    for r in rooms:
        if not isinstance(r, dict):
            continue
        # Support both `dimensions` array and `bounds` dict
        dims = r.get("dimensions")
        if isinstance(dims, list) and len(dims) >= 2:
            rw, rd = float(dims[0]), float(dims[1])
        else:
            b = r.get("bounds") or {}
            rw = float(b.get("width", 0)) if isinstance(b, dict) else 0
            rd = float(b.get("depth", b.get("length", 0))) if isinstance(b, dict) else 0
        for f in (r.get("furniture") or []):
            if not isinstance(f, dict):
                # agent used a reference-string schema (e.g., "desk_1");
                # skip — the actual dict will be found at scene-level furniture
                continue
            t = _furniture_tuple(f)
            if t is None:
                # Position or dimensions missing → no AABB to check; skip
                # (verifier_report still reflects this via the total count)
                continue
            total += 1
            _, x, y, w, d, _theta = t
            if x - w/2 < 0 or y - d/2 < 0 or x + w/2 > rw or y + d/2 > rd:
                out_of_bounds += 1
    # Also iterate top-level furniture (legacy/alternate schema)
    for f in (scene.get("furniture") or []):
        if not isinstance(f, dict):
            continue
        t = _furniture_tuple(f)
        if t is None:
            continue
        total += 1
        _, x, y, w, d, _theta = t
        # Use scene bounds if available
        bounds = scene.get("scene", {}).get("bounds") or scene.get("bounds") or {}
        if isinstance(bounds, dict):
            rw = float(bounds.get("width", 0))
            rd = float(bounds.get("depth", bounds.get("length", 0)))
            if rw > 0 and rd > 0:
                if x - w/2 < 0 or y - d/2 < 0 or x + w/2 > rw or y + d/2 > rd:
                    out_of_bounds += 1
    return CheckResult(name="in_bounds", passed=out_of_bounds == 0,
                       detail=f"total={total} out_of_bounds={out_of_bounds}")


HABITABLE_ROOM_TYPES = {"living", "bedroom", "kitchen", "dining", "office", "study"}
PERIMETER_WALLS = {"north", "south", "east", "west"}


def check_irc_aperture(task: dict, output_dir: Path) -> CheckResult:
    """Verify scene_state.json's habitable rooms meet IRC §R303 8% window
    aperture on perimeter walls. Non-habitable rooms (storage etc.) skip
    the check.
    """
    v = task["verifier"]
    min_ratio = v.get("min_ratio", 0.08)
    p = output_dir / "scene_state.json"
    if not p.exists():
        return CheckResult(name="irc_aperture", passed=False,
                           detail="scene_state.json missing")
    try:
        scene = json.loads(p.read_text())
    except Exception as e:
        return CheckResult(name="irc_aperture", passed=False,
                           detail=f"scene_state.json malformed: {e}")
    rooms = scene.get("rooms") or []
    if not rooms:
        return CheckResult(name="irc_aperture", passed=False,
                           detail="no rooms in scene_state.json")
    failures = []
    for room in rooms:
        rtype = room.get("room_type", "living")
        if rtype not in HABITABLE_ROOM_TYPES:
            continue  # exempt
        dims = room.get("dimensions", [0, 0])
        floor_area = dims[0] * dims[1]
        if floor_area <= 0:
            failures.append(f"room {rtype}: dimensions invalid")
            continue
        target = min_ratio * floor_area
        windows = room.get("windows") or []
        good = [w for w in windows if w.get("wall") in PERIMETER_WALLS]
        bad = [w for w in windows if w.get("wall") not in PERIMETER_WALLS]
        aperture = sum(w.get("width", 0) * w.get("height", 0) for w in good)
        if bad:
            failures.append(
                f"room {rtype}: {len(bad)} window(s) on non-perimeter walls")
        if aperture < target:
            failures.append(
                f"room {rtype}: aperture {aperture:.2f} m² below "
                f"{target:.2f} m² ({min_ratio*100:.0f}% of {floor_area} m²)")
    if failures:
        return CheckResult(name="irc_aperture", passed=False,
                           detail="; ".join(failures))
    return CheckResult(name="irc_aperture", passed=True,
                       detail=f"all habitable rooms meet ≥{min_ratio*100:.0f}% aperture")


_SIONNA_MATERIAL_ALIASES = {
    "itu_concrete": "itu_concrete", "concrete": "itu_concrete",
    "itu_drywall": "itu_plasterboard", "drywall": "itu_plasterboard",
    "itu_plasterboard": "itu_plasterboard", "plasterboard": "itu_plasterboard",
    "itu_glass": "itu_glass", "glass": "itu_glass",
    "itu_wood": "itu_wood", "wood": "itu_wood", "hardwood": "itu_wood", "oak": "itu_wood",
    "itu_metal": "itu_metal", "metal": "itu_metal", "stainless": "itu_metal", "steel": "itu_metal",
    "itu_brick": "itu_brick", "brick": "itu_brick",
    "itu_marble": "itu_marble", "marble": "itu_marble",
    "itu_ceramic": "itu_ceramic", "ceramic": "itu_ceramic", "tile": "itu_ceramic",
}


def _scene_to_minimal_mitsuba(scene: dict) -> str:
    """Build a minimal valid Mitsuba 3.0 XML from scene_state.json so Sionna RT
    can attempt to load it. Just rooms-as-floor-boxes + ITU material BSDFs."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    root = Element("scene", version="3.0.0")
    SubElement(root, "integrator", type="path")
    sensor = SubElement(root, "sensor", type="perspective")
    SubElement(sensor, "float", name="fov", value="45")
    t = SubElement(sensor, "transform", name="to_world")
    SubElement(t, "lookat", origin="0,0,5", target="0,0,0", up="0,1,0")
    samp = SubElement(sensor, "sampler", type="independent")
    SubElement(samp, "integer", name="sample_count", value="1")
    film = SubElement(sensor, "film", type="hdrfilm")
    SubElement(film, "integer", name="width", value="64")
    SubElement(film, "integer", name="height", value="64")

    def norm(m):
        return _SIONNA_MATERIAL_ALIASES.get((m or "").lower(), "itu_plasterboard")

    materials = set()
    for r in scene.get("rooms") or []:
        if isinstance(r, dict):
            for v in (r.get("materials") or {}).values():
                if v:
                    materials.add(norm(v))
    for w in scene.get("walls") or []:
        if isinstance(w, dict):
            m = w.get("material")
            if m:
                materials.add(norm(m))
    if not materials:
        materials.add("itu_plasterboard")
    for m in materials:
        b = SubElement(root, "bsdf", type="itu-radio-material", id=m)
        SubElement(b, "string", name="material", value=m)

    for i, r in enumerate(scene.get("rooms") or []):
        if not isinstance(r, dict):
            continue
        dims = r.get("dimensions")
        if not isinstance(dims, list) or len(dims) < 2:
            continue
        try:
            w, d = float(dims[0]), float(dims[1])
        except (TypeError, ValueError):
            continue
        mat = norm((r.get("materials") or {}).get("walls", "itu_concrete"))
        shape = SubElement(root, "shape", type="cube", id=f"room_{i}_floor")
        tx = SubElement(shape, "transform", name="to_world")
        SubElement(tx, "scale", value=f"{w/2} {d/2} 0.05")
        SubElement(tx, "translate", value=f"{w/2} {d/2} 0")
        SubElement(shape, "ref", id=mat)
    return tostring(root, encoding="unicode")


def _check_sionna_loadable(output_dir: Path) -> CheckResult:
    """Verify the agent's scene_state.json can be parsed into a Mitsuba 3.0 XML
    and loaded by Sionna RT's load_scene(). This is a stronger downstream check
    than the structural verifier — it tests the actual usability of the scene."""
    scene_path = output_dir / "scene_state.json"
    if not scene_path.exists():
        return CheckResult(name="sionna_loadable", passed=False,
                           detail="no scene_state.json")
    try:
        scene = json.loads(scene_path.read_text())
    except Exception as e:
        return CheckResult(name="sionna_loadable", passed=False,
                           detail=f"JSON parse: {e}")

    # Lazy import — keep verifier light when sionna is not needed
    try:
        import sionna.rt as rt
    except ImportError:
        return CheckResult(name="sionna_loadable", passed=True,
                           detail="sionna not installed; check skipped")

    try:
        xml = _scene_to_minimal_mitsuba(scene)
        xml_path = output_dir / "_sionna_loadable_check.xml"
        xml_path.write_text(xml)
        loaded = rt.load_scene(str(xml_path))
        n_obj = len(list(loaded.objects)) if hasattr(loaded, "objects") else "?"
        return CheckResult(name="sionna_loadable", passed=True,
                           detail=f"Sionna RT loaded ({n_obj} named objects)")
    except Exception as e:
        return CheckResult(name="sionna_loadable", passed=False,
                           detail=f"{type(e).__name__}: {str(e)[:200]}")


def _find_num(d, *keys):
    """Find a numeric value in a nested dict by trying multiple key names."""
    if not isinstance(d, dict):
        return None
    for key in keys:
        # Direct
        if key in d and isinstance(d[key], (int, float)):
            return float(d[key])
    # Search nested numerical_metrics
    metrics = d.get("numerical_metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                return float(metrics[key])
    return None


def _find_arr(d, *keys):
    """Find a list of numbers in a nested dict by trying multiple key names."""
    if not isinstance(d, dict):
        return None
    for key in keys:
        if key in d and isinstance(d[key], list):
            arr = d[key]
            if all(isinstance(v, (int, float)) for v in arr):
                return [float(v) for v in arr]
    metrics = d.get("numerical_metrics")
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics and isinstance(metrics[key], list):
                arr = metrics[key]
                if all(isinstance(v, (int, float)) for v in arr):
                    return [float(v) for v in arr]
    return None


def _check_rt_oracle(output_dir: Path) -> CheckResult:
    """RT-level oracle: energy conservation + material/frequency trends.
    Pass conditions (each applies only if relevant data present):
      - max_rss_dbm <= tx_power_dbm + 5  (no free energy)
      - coverage_low_freq >= coverage_high_freq - 5pp  (low freq travels further)
      - coverage_drywall >= coverage_concrete - 3pp     (drywall lower loss)
      - rss values are in physical range [-120, +30] dBm
    """
    sim_path = output_dir / "simulation_result.json"
    if not sim_path.exists():
        return CheckResult(name="rt_oracle", passed=False, detail="no simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="rt_oracle", passed=False, detail=f"JSON: {e}")

    failures = []
    # 1. Energy conservation
    max_rss = _find_num(sim, "max_rss_dbm", "rss_max_dbm", "received_power_max_dbm")
    tx_power = _find_num(sim, "tx_power_dbm", "ap_power_dbm")
    if max_rss is not None and tx_power is not None:
        if max_rss > tx_power + 5:
            failures.append(f"max_rss {max_rss:.1f} > tx_power+5 = {tx_power+5:.1f} (free energy)")

    # 2. RSS values in physical range
    for key in ("max_rss_dbm", "min_rss_dbm", "mean_rss_dbm", "p5_received_power_dbm"):
        v = _find_num(sim, key)
        if v is not None and not (-120 <= v <= 30):
            failures.append(f"{key}={v:.1f} out of physical band [-120, +30]")

    # 3. Frequency trend: low freq should have >= high freq coverage (FSPL)
    pairs = [
        ("coverage_pct_24_ghz", "coverage_pct_5_ghz"),
        ("coverage_pct_5_ghz", "coverage_pct_28_ghz"),
        ("coverage_pct_5_ghz", "coverage_pct_60_ghz"),
        ("coverage_pct_28_ghz", "coverage_pct_60_ghz"),
    ]
    for low, high in pairs:
        c_low = _find_num(sim, low)
        c_high = _find_num(sim, high)
        if c_low is not None and c_high is not None:
            if c_low < c_high - 5:  # 5pp tolerance
                failures.append(f"{low}={c_low:.1f} < {high}={c_high:.1f} - 5 (FSPL trend violated)")

    # 4. Material trend: drywall should give >= concrete (drywall lower loss)
    c_drywall = _find_num(sim, "coverage_pct_drywall")
    c_concrete = _find_num(sim, "coverage_pct_concrete")
    if c_drywall is not None and c_concrete is not None:
        if c_drywall < c_concrete - 3:
            failures.append(f"coverage_pct_drywall={c_drywall:.1f} < concrete={c_concrete:.1f} - 3 (material trend)")

    if failures:
        return CheckResult(name="rt_oracle", passed=False, detail="; ".join(failures)[:200])
    return CheckResult(name="rt_oracle", passed=True, detail="energy + freq + material trends OK")


def _check_phy_oracle(output_dir: Path) -> CheckResult:
    """PHY-level oracle: BER monotonicity + coding gain + physical range."""
    sim_path = output_dir / "simulation_result.json"
    if not sim_path.exists():
        return CheckResult(name="phy_oracle", passed=False, detail="no simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="phy_oracle", passed=False, detail=f"JSON: {e}")

    failures = []
    # BER in [0, 1]
    ber = _find_num(sim, "ber", "ber_simulated", "ber_at_snr_10db", "ber_theoretical_awgn")
    if ber is not None and not (0 <= ber <= 1):
        failures.append(f"ber={ber:.4f} not in [0,1]")
    # BER monotone vs SNR
    snr_arr = _find_arr(sim, "snr_db", "ebnodb", "ebn0_db")
    ber_arr = _find_arr(sim, "ber_simulated", "ber_array", "ber")
    if snr_arr and ber_arr and len(snr_arr) == len(ber_arr) and len(snr_arr) >= 3:
        # Sort by SNR ascending, BER should be non-increasing
        pairs_sorted = sorted(zip(snr_arr, ber_arr))
        bers = [b for _, b in pairs_sorted]
        violations = sum(1 for i in range(len(bers)-1) if bers[i+1] > bers[i] * 1.5)  # tolerate small bumps
        if violations > 1:
            failures.append(f"BER not monotone vs SNR ({violations} violations)")
    # Coding gain non-trivial
    cg = _find_num(sim, "coding_gain_db")
    if cg is not None and cg < 0:
        failures.append(f"coding_gain_db={cg:.2f} < 0 (coding should not hurt)")
    # NMSE in physical range
    nmse = _find_num(sim, "nmse_db")
    if nmse is not None and not (-30 <= nmse <= 10):
        failures.append(f"nmse_db={nmse:.1f} out of physical band [-30, +10]")
    if failures:
        return CheckResult(name="phy_oracle", passed=False, detail="; ".join(failures)[:200])
    return CheckResult(name="phy_oracle", passed=True, detail="BER + coding gain + ranges OK")


def _check_sys_oracle(output_dir: Path) -> CheckResult:
    """SYS-level oracle: scheduler/fairness sanity + multi-cell consistency."""
    sim_path = output_dir / "simulation_result.json"
    if not sim_path.exists():
        return CheckResult(name="sys_oracle", passed=False, detail="no simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="sys_oracle", passed=False, detail=f"JSON: {e}")

    failures = []
    # Jain's fairness in [0, 1]
    fi = _find_num(sim, "fairness_index", "jains_fairness", "fairness")
    if fi is not None and not (0 <= fi <= 1):
        failures.append(f"fairness_index={fi:.3f} not in [0,1]")
    # Throughput in physical range (bps/Hz)
    tp = _find_num(sim, "mean_throughput_bps_hz", "sum_rate_bps_hz", "spectral_efficiency_bps_hz")
    if tp is not None and not (0 <= tp <= 30):
        failures.append(f"throughput {tp:.2f} bps/Hz out of range [0, 30]")
    # SINR mean in physical range (dB)
    sinr = _find_num(sim, "sinr_dbm_mean", "sinr_mean_db", "sinr_db_mean")
    if sinr is not None and not (-20 <= sinr <= 60):
        failures.append(f"sinr_mean={sinr:.1f} out of range [-20, +60]")
    # per_user array length sanity if both present
    n_users = _find_num(sim, "num_users", "n_users")
    rates = _find_arr(sim, "per_user_avg_rate", "user_rates_bps_hz")
    if rates and n_users is not None:
        if len(rates) != int(n_users):
            failures.append(f"per_user_avg_rate length {len(rates)} != n_users {int(n_users)}")
    if failures:
        return CheckResult(name="sys_oracle", passed=False, detail="; ".join(failures)[:200])
    return CheckResult(name="sys_oracle", passed=True, detail="fairness + throughput + SINR ranges OK")


def _check_geometry_oracle(output_dir: Path) -> CheckResult:
    """Geometry oracle: scene_state.json ↔ simulation_result.json consistency.
    Pass conditions:
      - TX position in scene_state matches simulation_result.deployment.transmitters
      - if multi-room scene and per_room_coverage reported: TX's home room has highest coverage
    """
    scene_path = output_dir / "scene_state.json"
    sim_path = output_dir / "simulation_result.json"
    if not scene_path.exists() or not sim_path.exists():
        return CheckResult(name="geometry_oracle", passed=False, detail="missing artifact")
    try:
        scene = json.loads(scene_path.read_text())
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="geometry_oracle", passed=False, detail=f"JSON: {e}")

    failures = []
    # If scene has multi-room and per_room_coverage_pct array, the room containing the TX
    # should generally have the highest coverage
    rooms = scene.get("rooms") or []
    if isinstance(rooms, list) and len(rooms) >= 2:
        per_room = _find_arr(sim, "per_room_coverage_pct", "per_room_pct", "per_room_min_rss")
        if per_room and len(per_room) >= 2:
            # If there's wide variance, that's GOOD (proves agent modeled wall attenuation)
            spread = max(per_room) - min(per_room)
            if spread < 1:  # Less than 1 pp variation = agent didn't model walls
                failures.append(f"per_room values have tiny spread {spread:.2f} (no wall effect)")
    if failures:
        return CheckResult(name="geometry_oracle", passed=False, detail="; ".join(failures)[:200])
    return CheckResult(name="geometry_oracle", passed=True, detail="geometry/sim consistent")


# ─────────────────────────────────────────────────────────────
# Reference oracles — analytical ground truth re-derivation
# (shared helpers below, per-capability checks build on them)
# ─────────────────────────────────────────────────────────────

def _extract_wd(d):
    """Extract (width, depth) from a dict with various key names, or a 2/3-list."""
    if isinstance(d, dict):
        try:
            w = d.get("width")
            if w is None:
                w = d.get("w") or d.get("x") or d.get("size_x") or d.get("width_m")
            dp = d.get("depth")
            if dp is None:
                dp = (d.get("length") or d.get("d") or d.get("y")
                      or d.get("size_y") or d.get("depth_m"))
            # min/max bounding-box variant
            if w is None and "x_max" in d and "x_min" in d:
                w = float(d["x_max"]) - float(d["x_min"])
            if dp is None and "y_max" in d and "y_min" in d:
                dp = float(d["y_max"]) - float(d["y_min"])
            if w is not None and dp is not None:
                return float(w), float(dp)
        except Exception:
            pass
    if isinstance(d, list) and len(d) >= 2:
        try:
            return float(d[0]), float(d[1])
        except Exception:
            pass
    return 0.0, 0.0


def _parse_scene_geometry(scene: dict) -> dict | None:
    """Robust extractor for scene bounds + first-AP parameters.

    Returns dict with keys (W, D, ap_x, ap_y, ap_z, freq_hz, tx_power,
    threshold_dbm) or None if scene bounds/AP cannot be recovered.

    Handles 6+ scene_state.json schema variants:
      scene.bounds / scene.dimensions / top-level bounds / rooms[].dims_m /
      rooms[].dimensions / rooms[].bounds = {x_min,x_max,y_min,y_max}
    AP can be under access_points / transmitters; position can be list
    or dict; common alternate keys for power/freq covered.
    """
    W = D = 0.0
    scene_block = scene.get("scene") if isinstance(scene.get("scene"), dict) else None
    for source in (
        (scene_block or {}).get("bounds"),
        (scene_block or {}).get("dimensions"),
        scene.get("bounds"),
        scene.get("dimensions"),
    ):
        if source is not None:
            W, D = _extract_wd(source)
            if W > 0 and D > 0:
                break

    if W <= 0 or D <= 0:
        rooms = scene.get("rooms") or []
        max_x = max_y = 0.0
        for r in rooms:
            if not isinstance(r, dict):
                continue
            rw = rd = 0.0
            for src in (r.get("bounds"), r.get("dimensions"),
                        r.get("dims_m"), r.get("dims"), r.get("size"), r):
                if src is None:
                    continue
                rw, rd = _extract_wd(src)
                if rw > 0 and rd > 0:
                    break
            rx = ry = 0.0
            for src in (r.get("bounds"), r.get("position"), r.get("origin"),
                        r.get("origin_m"), r):
                if isinstance(src, dict):
                    try:
                        rx = float(src.get("x", src.get("x_min", 0)) or 0)
                        ry = float(src.get("y", src.get("y_min", 0)) or 0)
                    except Exception:
                        rx = ry = 0
                    if rx > 0 or ry > 0:
                        break
                elif isinstance(src, list) and len(src) >= 2:
                    try:
                        rx, ry = float(src[0]), float(src[1])
                    except Exception:
                        rx = ry = 0
                    if rx > 0 or ry > 0:
                        break
            max_x = max(max_x, rx + rw)
            max_y = max(max_y, ry + rd)
        W, D = max_x, max_y
    if W <= 0 or D <= 0:
        return None

    aps = scene.get("access_points") or scene.get("transmitters") or []
    if not aps or not isinstance(aps[0], dict):
        return None
    ap = aps[0]
    pos = ap.get("position") or [W/2, D/2, 2.5]
    try:
        if isinstance(pos, dict):
            ap_x = float(pos.get("x", pos.get("x_m", W/2)))
            ap_y = float(pos.get("y", pos.get("y_m", D/2)))
            ap_z = float(pos.get("z", pos.get("z_m", 2.5)))
        elif isinstance(pos, list) and len(pos) >= 2:
            ap_x = float(pos[0]); ap_y = float(pos[1])
            ap_z = float(pos[2]) if len(pos) >= 3 else 2.5
        else:
            # Flat schema variant: x_m, y_m, z_m on the AP itself
            ap_x = float(ap.get("x_m", ap.get("x", W/2)))
            ap_y = float(ap.get("y_m", ap.get("y", D/2)))
            ap_z = float(ap.get("z_m", ap.get("z", 2.5)))
    except Exception:
        return None

    freq_hz = (ap.get("frequency_hz") or ap.get("freq_hz")
               or scene.get("metadata", {}).get("frequency_hz"))
    try:
        freq_hz = float(freq_hz) if freq_hz else 5e9
    except Exception:
        freq_hz = 5e9
    try:
        tx_power = float(ap.get("power_dbm", ap.get("tx_power_dbm", 20.0)) or 20.0)
    except Exception:
        tx_power = 20.0
    try:
        threshold = float(scene.get("metadata", {}).get("coverage_threshold_dbm") or -75)
    except Exception:
        threshold = -75.0

    return {"W": W, "D": D, "ap_x": ap_x, "ap_y": ap_y, "ap_z": ap_z,
            "freq_hz": freq_hz, "tx_power": tx_power, "threshold_dbm": threshold}


def _analytical_fspl_coverage(W: float, D: float, ap_x: float, ap_y: float,
                              ap_z: float, freq_hz: float, tx_power_dbm: float,
                              threshold_dbm: float, rx_height: float = 1.5,
                              grid: float = 0.25) -> float:
    """Compute analytical FSPL coverage_pct on a regular grid.

    rss = tx_power − [20·log10(d) + 20·log10(f) − 147.55]
    coverage_pct = mean(rss > threshold) × 100
    """
    import math
    n_total = n_above = 0
    x = 0.0
    while x <= W + 1e-6:
        y = 0.0
        while y <= D + 1e-6:
            dx = x - ap_x; dy = y - ap_y; dz = ap_z - rx_height
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d < 0.1:
                d = 0.1
            fspl = 20*math.log10(d) + 20*math.log10(freq_hz) - 147.55
            if (tx_power_dbm - fspl) > threshold_dbm:
                n_above += 1
            n_total += 1
            y += grid
        x += grid
    return 100.0 * n_above / max(n_total, 1)


def _check_c1_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """C1 reference oracle: analytical FSPL coverage as ground truth.

    Reads scene_state.json + simulation_result.json, computes analytical
    coverage_pct on a 0.25 m grid, and compares to the agent's reported value.

    Tolerance:
      - upper:  +5 pp (energy conservation — agent can't exceed FSPL)
      - lower: -5 pp easy / -15 pp hard (walls/multipath drop RT below FSPL)
    """
    scene_path = output_dir / "scene_state.json"
    sim_path = output_dir / "simulation_result.json"
    if not scene_path.exists() or not sim_path.exists():
        return CheckResult(name="c1_ref_oracle", passed=False,
                           detail="missing scene_state.json or simulation_result.json")
    try:
        scene = json.loads(scene_path.read_text())
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="c1_ref_oracle", passed=False, detail=f"JSON parse: {e}")

    geom = _parse_scene_geometry(scene)
    if geom is None:
        return CheckResult(name="c1_ref_oracle", passed=False,
                           detail="could not parse scene bounds or AP from scene_state.json")

    ref_pct = _analytical_fspl_coverage(
        geom["W"], geom["D"], geom["ap_x"], geom["ap_y"], geom["ap_z"],
        geom["freq_hz"], geom["tx_power"], geom["threshold_dbm"])

    agent_pct = _find_num(sim, "coverage_pct")
    if agent_pct is None:
        return CheckResult(name="c1_ref_oracle", passed=False,
                           detail=f"coverage_pct missing; ref={ref_pct:.1f}")

    diff = task.get("difficulty", "easy")
    upper_tol = 5.0
    lower_tol = 5.0 if diff == "easy" else 15.0
    delta = agent_pct - ref_pct
    passed = (-lower_tol <= delta <= upper_tol)
    return CheckResult(
        name="c1_ref_oracle",
        passed=passed,
        detail=(f"agent={agent_pct:.1f} ref={ref_pct:.1f} delta={delta:+.1f}pp "
                f"tol=[-{lower_tol:.0f},+{upper_tol:.0f}] diff={diff}"),
    )


def _check_c3_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """C3 reference oracle: analytical FSPL at two frequencies + diff arithmetic.

    Requires (canonical, with fallbacks) the agent to emit:
      - simulation_config.frequencies_ghz : [low, high]   (or two coverage_pct_<f>_ghz keys)
      - numerical_metrics.coverage_pct_low_freq           (or coverage_pct_<f1>_ghz)
      - numerical_metrics.coverage_pct_high_freq          (or coverage_pct_<f2>_ghz)
      - numerical_metrics.coverage_diff_pp                (low − high)

    Verifier checks:
      (1) Two frequencies present and low < high (FSPL sanity).
      (2) Per-frequency coverage matches analytical FSPL within tolerance.
      (3) Arithmetic: |agent_diff_pp − (agent_low − agent_high)| ≤ 2 pp.
    """
    scene_path = output_dir / "scene_state.json"
    sim_path = output_dir / "simulation_result.json"
    if not scene_path.exists() or not sim_path.exists():
        return CheckResult(name="c3_ref_oracle", passed=False,
                           detail="missing scene_state.json or simulation_result.json")
    try:
        scene = json.loads(scene_path.read_text())
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="c3_ref_oracle", passed=False, detail=f"JSON parse: {e}")

    geom = _parse_scene_geometry(scene)
    if geom is None:
        return CheckResult(name="c3_ref_oracle", passed=False,
                           detail="could not parse scene geometry")

    # Recover the two frequencies. Canonical: simulation_config.frequencies_ghz.
    freqs_ghz = None
    cfg = sim.get("simulation_config") if isinstance(sim.get("simulation_config"), dict) else {}
    cand = cfg.get("frequencies_ghz") or sim.get("frequencies_ghz")
    if isinstance(cand, list) and len(cand) >= 2:
        try:
            freqs_ghz = (float(cand[0]), float(cand[1]))
        except Exception:
            freqs_ghz = None

    # Fallback: read from coverage_pct_<freq>_ghz keys in numerical_metrics.
    # Convention from tc_chained_gen.py: f1=2.4 → "coverage_pct_24_ghz" (×10).
    # So integer tags are decoded by /10. Tags with a decimal point are taken as-is.
    nm = sim.get("numerical_metrics", {})
    if freqs_ghz is None and isinstance(nm, dict):
        found = []
        for k in nm:
            ks = str(k).lower()
            if ks.startswith("coverage_pct_") and ks.endswith("_ghz"):
                tag = ks[len("coverage_pct_"):-len("_ghz")]
                try:
                    if "." in tag or "_" in tag:
                        # Decimal form (e.g., "2.4", "2_4") — interpret literally
                        val = float(tag.replace("_", "."))
                    else:
                        # Integer ×10 form (e.g., "24" → 2.4 GHz, "280" → 28 GHz)
                        val = float(tag) / 10.0
                    found.append(val)
                except Exception:
                    continue
        if len(found) >= 2:
            found = sorted(set(found))
            freqs_ghz = (found[0], found[-1])

    if freqs_ghz is None or freqs_ghz[0] >= freqs_ghz[1]:
        return CheckResult(name="c3_ref_oracle", passed=False,
                           detail=f"could not recover (low, high) frequency pair; got {freqs_ghz}")

    f_low_hz = freqs_ghz[0] * 1e9
    f_high_hz = freqs_ghz[1] * 1e9

    # Compute analytical coverage at both frequencies using the agent's AP position.
    ref_low = _analytical_fspl_coverage(
        geom["W"], geom["D"], geom["ap_x"], geom["ap_y"], geom["ap_z"],
        f_low_hz, geom["tx_power"], geom["threshold_dbm"])
    ref_high = _analytical_fspl_coverage(
        geom["W"], geom["D"], geom["ap_x"], geom["ap_y"], geom["ap_z"],
        f_high_hz, geom["tx_power"], geom["threshold_dbm"])
    ref_diff = ref_low - ref_high

    # Recover agent values. Canonical: numerical_metrics.coverage_pct_low_freq/high_freq.
    agent_low = _find_num(sim, "coverage_pct_low_freq")
    agent_high = _find_num(sim, "coverage_pct_high_freq")
    # Fallback: legacy keys can be in x10 form (24, 50, 280) or decimal (2.4, 5.0).
    def _legacy_tags(fghz: float) -> list[str]:
        out = [f"coverage_pct_{int(fghz*10)}_ghz"]  # 2.4 → 24
        # decimal forms
        out.append(f"coverage_pct_{fghz:.1f}_ghz".replace(".", "_"))  # 2_4
        out.append(f"coverage_pct_{fghz:.1f}_ghz")                     # 2.4
        if fghz == int(fghz):  # 5.0 → also try "5"
            out.append(f"coverage_pct_{int(fghz)}_ghz")
        return out
    if agent_low is None:
        for tag in _legacy_tags(freqs_ghz[0]):
            agent_low = _find_num(sim, tag)
            if agent_low is not None:
                break
    if agent_high is None:
        for tag in _legacy_tags(freqs_ghz[1]):
            agent_high = _find_num(sim, tag)
            if agent_high is not None:
                break
    if agent_low is None or agent_high is None:
        return CheckResult(name="c3_ref_oracle", passed=False,
                           detail=(f"missing per-freq coverage; "
                                   f"need coverage_pct_{{low,high}}_freq or "
                                   f"coverage_pct_{int(freqs_ghz[0]*10)}_ghz / "
                                   f"coverage_pct_{int(freqs_ghz[1]*10)}_ghz"))
    agent_diff = _find_num(sim, "coverage_diff_pp")

    diff = task.get("difficulty", "easy")
    cov_tol_lo = 10.0 if diff == "easy" else 20.0
    cov_tol_hi = 5.0
    failures = []

    d_low = agent_low - ref_low
    if not (-cov_tol_lo <= d_low <= cov_tol_hi):
        failures.append(f"low freq: agent={agent_low:.1f} ref={ref_low:.1f} Δ={d_low:+.1f}pp")
    d_high = agent_high - ref_high
    if not (-cov_tol_lo <= d_high <= cov_tol_hi):
        failures.append(f"high freq: agent={agent_high:.1f} ref={ref_high:.1f} Δ={d_high:+.1f}pp")
    # FSPL sanity: high freq should NOT exceed low freq by more than 5 pp.
    if agent_high > agent_low + 5:
        failures.append(f"FSPL violation: agent high {agent_high:.1f} > low {agent_low:.1f} + 5")
    if agent_diff is not None:
        arith = agent_low - agent_high
        if abs(agent_diff - arith) > 2.0:
            failures.append(f"diff arithmetic: reported={agent_diff:.1f} but low−high={arith:.1f}")

    passed = not failures
    summary = (f"ref(low={freqs_ghz[0]:.1f}GHz)={ref_low:.1f} "
               f"ref(high={freqs_ghz[1]:.1f}GHz)={ref_high:.1f} "
               f"ref_diff={ref_diff:.1f}pp")
    detail = summary if passed else f"{summary} | " + "; ".join(failures)
    return CheckResult(name="c3_ref_oracle", passed=passed, detail=detail[:280])


def _check_c4_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """C4 reference oracle: before/after coverage consistency + edit-occurred + sign.

    Requires (with fallbacks) the agent to emit:
      - numerical_metrics.coverage_pct_before
      - numerical_metrics.coverage_pct_after
      - numerical_metrics.coverage_delta_pp  (after − before, signed)

    Optional: edit_action passed via subcheck spec — drives the sign check.
      action='removed' (furniture/obstacle): expect delta_pp ≥ -5 pp
      action='added'   (partition / wall):  expect delta_pp ≤ +5 pp
      action='changed' material→concrete:   expect delta_pp ≤ 0
      action='changed' material→drywall:    expect delta_pp ≥ 0

    Verifier checks:
      (1) Both before/after present, both ∈ [0, 100].
      (2) Arithmetic: |agent_delta − (after − before)| ≤ 2 pp.
      (3) Sign matches edit type (if known).
      (4) Edit actually occurred: agent's simulation.py or scene_state mentions
          the edit verb (best-effort; the existing token grep already enforces this).
    """
    scene_path = output_dir / "scene_state.json"
    sim_path = output_dir / "simulation_result.json"
    if not scene_path.exists() or not sim_path.exists():
        return CheckResult(name="c4_ref_oracle", passed=False,
                           detail="missing scene_state.json or simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="c4_ref_oracle", passed=False, detail=f"JSON parse: {e}")

    before = _find_num(sim, "coverage_pct_before")
    after = _find_num(sim, "coverage_pct_after")
    delta = _find_num(sim, "coverage_delta_pp")

    if before is None or after is None:
        return CheckResult(
            name="c4_ref_oracle", passed=False,
            detail=(f"missing required fields: "
                    f"coverage_pct_before={before} coverage_pct_after={after}"))

    failures = []
    if not (0 <= before <= 100):
        failures.append(f"coverage_pct_before={before:.1f} not in [0,100]")
    if not (0 <= after <= 100):
        failures.append(f"coverage_pct_after={after:.1f} not in [0,100]")

    arith_delta = after - before
    if delta is not None and abs(delta - arith_delta) > 2.0:
        failures.append(f"delta arith: reported={delta:.1f} but after−before={arith_delta:.1f}")
    final_delta = delta if delta is not None else arith_delta

    # Sign check (only if edit_action is provided in the subcheck spec).
    subcheck = task.get("verifier", {}) if isinstance(task.get("verifier"), dict) else {}
    md = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    edit_action = (subcheck.get("edit_action")
                   or task.get("edit_action")
                   or md.get("edit_action"))
    if edit_action:
        ea = str(edit_action).lower()
        if "remov" in ea and final_delta < -5:
            failures.append(f"sign: removed→expected Δ≥-5 but got {final_delta:+.1f}")
        elif "add" in ea and final_delta > 5:
            failures.append(f"sign: added→expected Δ≤+5 but got {final_delta:+.1f}")
        elif "concrete" in ea and final_delta > 5:
            failures.append(f"sign: →concrete expected Δ≤+5 but got {final_delta:+.1f}")
        elif ("drywall" in ea or "wood" in ea) and final_delta < -5:
            failures.append(f"sign: →drywall expected Δ≥-5 but got {final_delta:+.1f}")

    # Fabrication detector: delta exactly 0 with no edit, or before == after to
    # 0.01 pp — agent likely didn't actually re-simulate.
    if abs(arith_delta) < 0.01 and edit_action:
        failures.append(f"suspicious: before==after to 0.01 pp under action={edit_action}")

    passed = not failures
    detail = (f"before={before:.1f} after={after:.1f} Δ={final_delta:+.1f}pp "
              f"action={edit_action or '?'}")
    if not passed:
        detail += " | " + "; ".join(failures)
    return CheckResult(name="c4_ref_oracle", passed=passed, detail=detail[:280])


def _check_c5_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """C5 / T3 reference oracle: BER must lie between AWGN floor and
    Rayleigh-fading ceiling for the declared modulation and SNR.

    For QPSK (or BPSK) at SNR_dB:
        ber_awgn      = Q(√(2·γ))                                     # ideal AWGN
        ber_rayleigh  = 0.5·(1 − √(γ/(γ+1)))                          # single-path Rayleigh

    Pass criterion:
        ber_awgn − ε ≤ agent_ber ≤ 1.5 · ber_rayleigh + ε
    (1.5× ceiling allows mild stochastic over-shoot at low symbol counts.)

    Also requires `ber_theoretical_awgn` field present and within 3× of the
    analytical AWGN BER (to catch agents that fabricate this baseline).
    """
    import math
    sim_path = output_dir / "simulation_result.json"
    if not sim_path.exists():
        return CheckResult(name="c5_ref_oracle", passed=False,
                           detail="no simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="c5_ref_oracle", passed=False, detail=f"JSON: {e}")

    # Recover SNR (default 10 dB if missing)
    snr_db = (_find_num(sim, "snr_db", "ebn0_db", "eb_n0_db")
              or _find_num(sim, "snr", "ebn0"))
    if snr_db is None:
        snr_db = 10.0  # task default for easy scenes
    snr_lin = 10 ** (float(snr_db) / 10.0)

    # Analytical AWGN BER for QPSK / BPSK: Q(√(2·SNR))
    arg = math.sqrt(2 * snr_lin)
    ber_awgn_ref = 0.5 * math.erfc(arg / math.sqrt(2))
    # Analytical Rayleigh single-path BER (BPSK/QPSK identical at Eb/N0)
    ber_rayleigh_ref = 0.5 * (1 - math.sqrt(snr_lin / (snr_lin + 1)))

    agent_ber = _find_num(sim, "ber", "ber_simulated", "ber_at_snr_10db")
    if agent_ber is None:
        return CheckResult(name="c5_ref_oracle", passed=False,
                           detail=f"ber missing; AWGN floor {ber_awgn_ref:.2e}, "
                                  f"Rayleigh ceil {ber_rayleigh_ref:.2e}")

    failures = []

    # 1. Agent BER within band.
    # Lower bound: AWGN floor (energy conservation).
    # Upper bound: max(1.5×Rayleigh, 0.5). The Rayleigh ceiling assumes
    # ideal equalization; real multipath without an equalizer suffers ISI
    # and can push BER much higher, but BER for any modulation is
    # physically bounded by 0.5 (random-guess limit). Anything > 0.5 is a
    # coding error (e.g., agent reported SER for QPSK and forgot 4-PSK
    # symbol→bit demapping).
    eps = max(ber_awgn_ref * 0.1, 1e-9)
    lo = ber_awgn_ref - eps
    hi = max(1.5 * ber_rayleigh_ref, 0.5)
    if agent_ber < lo:
        failures.append(
            f"ber={agent_ber:.2e} BELOW AWGN floor {lo:.2e} @ SNR={snr_db:.0f} dB "
            f"(physically impossible — multipath can't beat ideal AWGN)"
        )
    elif agent_ber > 0.5 + 1e-3:
        failures.append(
            f"ber={agent_ber:.4f} > 0.5 (physical max for any binary modulation; "
            f"likely SER reported as BER)"
        )

    # 2. Reported theoretical AWGN matches our analytical
    agent_awgn = _find_num(sim, "ber_theoretical_awgn", "ber_awgn_theoretical",
                           "ber_awgn")
    if agent_awgn is None:
        failures.append("ber_theoretical_awgn missing (required baseline)")
    elif ber_awgn_ref > 0:
        ratio = agent_awgn / ber_awgn_ref if ber_awgn_ref > 0 else 1.0
        # Allow 3× slack for differing Q-approximation choices
        if not (1/3 <= ratio <= 3 or abs(agent_awgn - ber_awgn_ref) < 1e-9):
            failures.append(
                f"agent_awgn={agent_awgn:.2e} vs analytical={ber_awgn_ref:.2e} "
                f"(ratio {ratio:.2f}, expected ∈ [0.33, 3])"
            )

    # 3. CIR path count sanity (RT should produce ≥ 1 path; multipath usually ≥ 2)
    n_paths = _find_num(sim, "cir_path_count", "num_paths", "n_paths")
    if n_paths is not None and n_paths < 1:
        failures.append(f"cir_path_count={n_paths} < 1 (no multipath)")

    passed = not failures
    detail = (f"snr={snr_db:.0f}dB  agent_ber={agent_ber:.2e}  "
              f"ref_band=[AWGN {ber_awgn_ref:.2e}, 1.5·Rayleigh {1.5*ber_rayleigh_ref:.2e}]  "
              f"n_paths={n_paths}")
    if not passed:
        detail += " | " + "; ".join(failures)
    return CheckResult(name="c5_ref_oracle", passed=passed, detail=detail[:280])


def _check_c7_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """C7 reference oracle: re-compute Jain's fairness + mean throughput
    from per-user rates.

    Requires the agent to emit:
      - numerical_metrics.per_user_avg_rate : list of non-negative floats
        (aliases: user_rates_bps_hz, per_user_rate_bps_hz)

    Verifier checks (all relative to the per-user array):
      (1) Array exists, len ≥ 2, all entries non-negative.
      (2) Jain's index: J = (Σr)² / (n·Σr²); |agent_fairness − J| ≤ 0.05.
      (3) Mean throughput: |agent_mean − mean(r)| ≤ 0.2 bps/Hz.
      (4) num_users consistency: if reported, == len(per_user_avg_rate).
      (5) Every user got non-zero rate (no starvation) — heuristic.
    """
    sim_path = output_dir / "simulation_result.json"
    if not sim_path.exists():
        return CheckResult(name="c7_ref_oracle", passed=False,
                           detail="missing simulation_result.json")
    try:
        sim = json.loads(sim_path.read_text())
    except Exception as e:
        return CheckResult(name="c7_ref_oracle", passed=False, detail=f"JSON parse: {e}")

    rates = _find_arr(sim, "per_user_avg_rate", "user_rates_bps_hz",
                      "per_user_rate_bps_hz", "per_user_rates")
    if rates is None or len(rates) < 2:
        return CheckResult(
            name="c7_ref_oracle", passed=False,
            detail=("per_user_avg_rate[] missing or too short; "
                    f"got {None if rates is None else len(rates)}"))

    failures = []
    if any(r < 0 for r in rates):
        failures.append(f"negative rates: min={min(rates):.3f}")

    n = len(rates)
    s = sum(rates)
    sq = sum(r*r for r in rates)
    ref_fairness = (s*s) / (n * sq) if sq > 0 else 0.0
    ref_mean_thr = s / n if n > 0 else 0.0

    agent_fairness = _find_num(sim, "fairness_index", "jains_fairness", "fairness")
    agent_mean = _find_num(sim, "mean_throughput_bps_hz", "sum_rate_bps_hz",
                           "spectral_efficiency_bps_hz")

    if agent_fairness is None:
        failures.append("fairness_index missing")
    elif abs(agent_fairness - ref_fairness) > 0.05:
        failures.append(
            f"fairness: agent={agent_fairness:.3f} ref={ref_fairness:.3f} Δ={agent_fairness-ref_fairness:+.3f}")

    if agent_mean is None:
        # tolerate if sum_rate equivalent is present
        pass
    elif abs(agent_mean - ref_mean_thr) > 0.2:
        failures.append(
            f"mean_thr: agent={agent_mean:.2f} ref={ref_mean_thr:.2f} Δ={agent_mean-ref_mean_thr:+.2f}")

    n_users = _find_num(sim, "num_users", "n_users")
    if n_users is not None and int(n_users) != n:
        failures.append(f"num_users={int(n_users)} != len(rates)={n}")

    # Starvation heuristic: under PF, expect every user to have a positive rate.
    if min(rates) <= 1e-6:
        failures.append(f"starvation: min rate {min(rates):.4f} ≤ 0")

    passed = not failures
    detail = (f"n={n} ref_J={ref_fairness:.3f} ref_mean={ref_mean_thr:.2f}bps/Hz "
              f"agent_J={agent_fairness} agent_mean={agent_mean}")
    if not passed:
        detail += " | " + "; ".join(failures)
    return CheckResult(name="c7_ref_oracle", passed=passed, detail=detail[:280])


def _check_n1_reference_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N1 reference oracle: compare agent's coverage_map.npy against the
    precomputed Sionna-RT ground truth in benchmark/oracles/n1/{scene}.npy.

    The pre-computed oracle was generated by `benchmark/_smoke_rt/run_n1_reference.py`
    using the same scene + AP + solver params as the task prompt, so a faithful
    agent run should agree to within RT Monte-Carlo noise (a few dB per cell).

    Pass conditions:
      (1) coverage_map.npy exists, is 2-D float
      (2) shape matches the oracle (same cell_size / extent)
      (3) MAE over jointly-valid cells <= 3 dB
      (4) >= 80% of jointly-valid cells agree within ±5 dB
    """
    import numpy as np

    cov_path = output_dir / "coverage_map.npy"
    if not cov_path.exists():
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail="coverage_map.npy missing")
    try:
        agent = np.load(cov_path)
    except Exception as e:
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail=f"could not load coverage_map.npy: {e}")
    if agent.ndim != 2:
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail=f"coverage_map.npy must be 2-D, got shape {agent.shape}")

    # Resolve oracle path. Task spec has a relative path; resolve from repo root.
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail="task spec missing oracle_path")
    repo_root = Path(__file__).resolve().parent.parent
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail=f"oracle not found: {oracle_abs}")
    try:
        ref = np.load(oracle_abs)
    except Exception as e:
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail=f"could not load oracle {oracle_abs}: {e}")

    if agent.shape == ref.shape:
        # Strict path: cell-wise comparison on jointly-valid cells
        both_valid = np.isfinite(agent) & np.isfinite(ref)
        n_joint = int(both_valid.sum())
        if n_joint == 0:
            return CheckResult(name="n1_ref_oracle", passed=False,
                               detail="no jointly-valid cells between agent and reference")
        diff = np.abs(agent[both_valid] - ref[both_valid])
        mae = float(diff.mean())
        cell_agree = float((diff <= 5.0).mean())
        pass_mae = mae <= 3.0
        pass_agree = cell_agree >= 0.80
        passed = pass_mae and pass_agree
        detail = (f"shape={agent.shape} joint_cells={n_joint} "
                  f"MAE={mae:.2f}dB ({'PASS' if pass_mae else 'FAIL'} <=3dB) "
                  f"cell_agree={cell_agree*100:.1f}% ({'PASS' if pass_agree else 'FAIL'} >=80%)")
        return CheckResult(name="n1_ref_oracle", passed=passed, detail=detail)

    # Fallback: shape mismatch (often because the agent passed an explicit
    # center/size to the solver instead of using the default bbox-derived grid).
    # The grids differ in extent/padding, not in physics, so we fall back to a
    # distribution-level check: the valid-cell RSS distribution must agree on
    # mean (≤3 dB), std (≤2 dB), and valid-cell fraction (≤25 percentage points).
    a_valid = agent[np.isfinite(agent)]
    r_valid = ref[np.isfinite(ref)]
    if a_valid.size == 0 or r_valid.size == 0:
        return CheckResult(name="n1_ref_oracle", passed=False,
                           detail=(f"shape mismatch agent={agent.shape} ref={ref.shape}; "
                                   f"no valid cells in {'agent' if a_valid.size==0 else 'ref'}"))
    a_mean, a_std = float(a_valid.mean()), float(a_valid.std())
    r_mean, r_std = float(r_valid.mean()), float(r_valid.std())
    a_frac = a_valid.size / agent.size
    r_frac = r_valid.size / ref.size
    d_mean = abs(a_mean - r_mean)
    d_std = abs(a_std - r_std)
    d_frac = abs(a_frac - r_frac) * 100  # percentage points
    pass_mean = d_mean <= 3.0
    pass_std = d_std <= 2.0
    pass_frac = d_frac <= 25.0
    passed = pass_mean and pass_std and pass_frac
    detail = (f"shape mismatch (agent={agent.shape} ref={ref.shape}) → "
              f"distribution check: "
              f"mean diff {d_mean:.2f}dB ({'PASS' if pass_mean else 'FAIL'} <=3), "
              f"std diff {d_std:.2f}dB ({'PASS' if pass_std else 'FAIL'} <=2), "
              f"valid-frac diff {d_frac:.1f}pp ({'PASS' if pass_frac else 'FAIL'} <=25)")
    return CheckResult(name="n1_ref_oracle", passed=passed, detail=detail)


def _grid_distribution_check(agent: "np.ndarray", ref: "np.ndarray",
                             mean_tol_db: float = 3.0,
                             std_tol_db: float = 2.0,
                             frac_tol_pp: float = 25.0) -> tuple[bool, str]:
    """Helper: compare two coverage grids when shapes may differ.
    Returns (passed, detail) using the same distribution-level logic as
    _check_n1_reference_oracle's fallback path.
    """
    import numpy as np
    if agent.shape == ref.shape:
        both_valid = np.isfinite(agent) & np.isfinite(ref)
        n_joint = int(both_valid.sum())
        if n_joint == 0:
            return False, "no jointly-valid cells"
        diff = np.abs(agent[both_valid] - ref[both_valid])
        mae = float(diff.mean())
        cell_agree = float((diff <= 5.0).mean())
        passed = mae <= 3.0 and cell_agree >= 0.80
        return passed, f"MAE={mae:.2f}dB cell_agree={cell_agree*100:.1f}% (joint={n_joint})"
    a_v = agent[np.isfinite(agent)]
    r_v = ref[np.isfinite(ref)]
    if a_v.size == 0 or r_v.size == 0:
        return False, f"shape mismatch {agent.shape} vs {ref.shape}; one side has no valid cells"
    d_mean = abs(float(a_v.mean()) - float(r_v.mean()))
    d_std = abs(float(a_v.std()) - float(r_v.std()))
    d_frac = abs(a_v.size/agent.size - r_v.size/ref.size) * 100
    passed = d_mean <= mean_tol_db and d_std <= std_tol_db and d_frac <= frac_tol_pp
    return passed, (f"shape mismatch {agent.shape} vs {ref.shape}; "
                    f"mean_diff={d_mean:.2f}dB std_diff={d_std:.2f}dB "
                    f"frac_diff={d_frac:.1f}pp")


def _check_n2_freq_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N2 frequency-edit reference oracle.

    The agent ran coverage at two frequencies on the SAME built-in scene
    (no geometric change). Pass conditions:
      1. coverage_5ghz.npy matches the 5 GHz reference (uses N1 oracle)
      2. coverage_2ghz.npy matches the 2.4 GHz reference (n2_freq oracle)
      3. The two grids are not identical (agent actually re-ran)
      4. mean(after − before) >= +3 dB on jointly-valid cells (FSPL theory says +6.4 dB
         going 5 GHz → 2.4 GHz; we allow +3 to +20 dB to catch obvious bugs)
      5. agent's reported delta_dbm_mean must agree with our computed delta (±2 dB)
    """
    import numpy as np

    sim_res = output_dir / "simulation_result.json"
    cov5 = output_dir / "coverage_5ghz.npy"
    cov2 = output_dir / "coverage_2ghz.npy"
    if not cov5.exists() or not cov2.exists():
        return CheckResult(name="n2_freq_oracle", passed=False,
                           detail=f"missing artifact: 5ghz={cov5.exists()} 2ghz={cov2.exists()}")

    try:
        agent_5 = np.load(cov5)
        agent_2 = np.load(cov2)
    except Exception as e:
        return CheckResult(name="n2_freq_oracle", passed=False,
                           detail=f"npy load: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    ref5_path = (repo_root / task["oracle_before_path"]).resolve()
    ref2_path = (repo_root / task["oracle_after_path"]).resolve()
    if not ref5_path.exists() or not ref2_path.exists():
        return CheckResult(name="n2_freq_oracle", passed=False,
                           detail=f"oracle missing 5GHz={ref5_path.exists()} 2GHz={ref2_path.exists()}")
    ref5 = np.load(ref5_path)
    ref2 = np.load(ref2_path)

    # Check 1: 5 GHz agent vs 5 GHz reference
    p5, d5 = _grid_distribution_check(agent_5, ref5)
    # Check 2: 2.4 GHz agent vs 2.4 GHz reference
    p2, d2 = _grid_distribution_check(agent_2, ref2)

    # Check 3: agent's two grids must differ (must have actually re-run at 2.4 GHz)
    if agent_5.shape == agent_2.shape:
        both = np.isfinite(agent_5) & np.isfinite(agent_2)
        if both.sum() == 0:
            differ_pp = 100.0
            differ_ok = False
        else:
            agent_delta = agent_2[both] - agent_5[both]
            differ_pp = float(np.abs(agent_delta).mean())
            differ_ok = differ_pp >= 1.0    # need at least 1 dB difference
    else:
        differ_pp = float("nan")
        differ_ok = True   # shape differ → trivially different

    # Check 4: mean delta sign and magnitude (lower freq → less FSPL → +3 to +20 dB)
    if agent_5.shape == agent_2.shape:
        both = np.isfinite(agent_5) & np.isfinite(agent_2)
        if both.sum() > 0:
            mean_delta = float((agent_2[both] - agent_5[both]).mean())
        else:
            mean_delta = float("nan")
    else:
        a5v = agent_5[np.isfinite(agent_5)]
        a2v = agent_2[np.isfinite(agent_2)]
        mean_delta = float(a2v.mean() - a5v.mean()) if (a5v.size and a2v.size) else float("nan")
    pdelta = np.isfinite(mean_delta) and (3.0 <= mean_delta <= 20.0)

    # Check 5: reported delta_dbm_mean vs computed
    sim_res_ok = True
    sim_res_detail = ""
    if sim_res.exists():
        try:
            sj = json.loads(sim_res.read_text())
            reported = sj.get("delta_dbm_mean")
            if reported is not None and np.isfinite(mean_delta):
                err = abs(float(reported) - mean_delta)
                sim_res_ok = err <= 2.0
                sim_res_detail = f" reported={float(reported):+.2f}dB (err={err:.2f}dB)"
            else:
                sim_res_detail = " reported=missing"
        except Exception as e:
            sim_res_detail = f" reported=parse-error({e})"
    else:
        sim_res_ok = False
        sim_res_detail = " simulation_result.json missing"

    passed = p5 and p2 and differ_ok and pdelta and sim_res_ok
    detail = (
        f"5GHz: {'PASS' if p5 else 'FAIL'}({d5}) | "
        f"2GHz: {'PASS' if p2 else 'FAIL'}({d2}) | "
        f"differ(MAE between freqs)={differ_pp:.2f}dB "
        f"{'PASS' if differ_ok else 'FAIL'} >=1dB | "
        f"mean_delta={mean_delta:+.2f}dB "
        f"{'PASS' if pdelta else 'FAIL'} in [+3,+20]"
        f"{sim_res_detail}"
    )
    return CheckResult(name="n2_freq_oracle", passed=passed, detail=detail[:400])


def _check_n1_probe_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N1.probe reference oracle.

    Compares agent's per-link path metrics to a precomputed Sionna RT
    PathSolver reference (in benchmark/oracles/n1_probe/{scene}.json).

    Pass conditions (ALL):
      1. path_gain_db within ±2 dB of reference
      2. num_paths within ±2 OR within 30% relative
      3. delay_spread_ns within ±20% relative
      4. mean_delay_ns   within ±15% relative
    """
    import numpy as np

    sim_res_path = output_dir / "simulation_result.json"
    if not sim_res_path.exists():
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail="simulation_result.json missing")
    try:
        agent = json.loads(sim_res_path.read_text())
    except Exception as e:
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail=f"JSON parse: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail="task spec missing oracle_path")
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail=f"oracle not found: {oracle_abs}")
    try:
        ref = json.loads(oracle_abs.read_text())
    except Exception as e:
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail=f"oracle parse: {e}")

    def _find(obj, *keys):
        if not isinstance(obj, dict):
            return None
        for k in keys:
            if k in obj and isinstance(obj[k], (int, float)):
                return float(obj[k])
        nm = obj.get("numerical_metrics")
        if isinstance(nm, dict):
            for k in keys:
                if k in nm and isinstance(nm[k], (int, float)):
                    return float(nm[k])
        return None

    a_pg = _find(agent, "path_gain_db")
    a_np = _find(agent, "num_paths")
    a_md = _find(agent, "mean_delay_ns")
    a_ds = _find(agent, "delay_spread_ns")

    r_pg = float(ref["path_gain_db"])
    r_np = int(ref["num_paths"])
    r_md = float(ref["mean_delay_ns"])
    r_ds = float(ref["delay_spread_ns"])

    missing = [name for name, v in [("path_gain_db", a_pg),
                                    ("num_paths", a_np),
                                    ("mean_delay_ns", a_md),
                                    ("delay_spread_ns", a_ds)] if v is None]
    if missing:
        return CheckResult(name="n1_probe_oracle", passed=False,
                           detail=f"missing fields in agent output: {missing}")

    # Per-field checks
    pg_err = abs(a_pg - r_pg)
    pg_ok = pg_err <= 2.0

    np_diff = abs(int(a_np) - r_np)
    np_ratio = min(int(a_np), r_np) / max(int(a_np), r_np) if max(int(a_np), r_np) > 0 else 0
    np_ok = np_diff <= 2 or np_ratio >= 0.70

    md_rel = abs(a_md - r_md) / abs(r_md) if abs(r_md) > 1e-6 else (
        0.0 if abs(a_md - r_md) < 0.5 else 999.0)
    md_ok = md_rel <= 0.15

    ds_rel = abs(a_ds - r_ds) / abs(r_ds) if abs(r_ds) > 1e-6 else (
        0.0 if abs(a_ds - r_ds) < 0.5 else 999.0)
    ds_ok = ds_rel <= 0.20

    passed = pg_ok and np_ok and md_ok and ds_ok
    detail = (
        f"path_gain {a_pg:+.2f} vs ref {r_pg:+.2f} (err {pg_err:.2f}dB, "
        f"{'PASS' if pg_ok else 'FAIL'} ≤2) | "
        f"num_paths {int(a_np)} vs {r_np} (diff {np_diff}, ratio {np_ratio:.2f}, "
        f"{'PASS' if np_ok else 'FAIL'}) | "
        f"mean_delay {a_md:.2f} vs {r_md:.2f}ns ({md_rel*100:.1f}%, "
        f"{'PASS' if md_ok else 'FAIL'} ≤15%) | "
        f"DS {a_ds:.2f} vs {r_ds:.2f}ns ({ds_rel*100:.1f}%, "
        f"{'PASS' if ds_ok else 'FAIL'} ≤20%)")
    return CheckResult(name="n1_probe_oracle", passed=passed, detail=detail[:400])


def _check_n2_edit_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N2 v2 AP-configuration edit reference oracle.

    Generic for the 4 N2 v2 edit types (frequency, power, position, antenna):
    each task spec provides oracle_before_path + oracle_after_path. We:
      1. Verify coverage_before.npy matches oracle_before (Layer C grid check
         with distribution fallback).
      2. Verify coverage_after.npy  matches oracle_after  (same).
      3. Verify the two grids actually differ (≥1 dB MAE between them) — so
         the agent really applied the edit and re-ran the solver.
      4. Verify simulation_result.json declares `edit_type` matching the
         task's `expected edit_type`, and `method == "sionna_rt"`.

    Optional edit-specific tightening (when the task spec provides
    `expected_delta_mean_db` / `expected_delta_std_db`):
      - mean(after − before) within ±2 dB of the expected mean
      - std(after − before)  ≤ expected std + 1 dB

    Used by frequency and power edits where the expected mean delta is
    physically determined (FSPL theory for freq, linearity for power);
    omitted for position/antenna where the redistribution is spatially
    complex.
    """
    import numpy as np

    before_npy = output_dir / "coverage_before.npy"
    after_npy = output_dir / "coverage_after.npy"
    sim_res_path = output_dir / "simulation_result.json"
    if not before_npy.exists() or not after_npy.exists():
        return CheckResult(name="n2_edit_oracle", passed=False,
                           detail=(f"missing artifact: before={before_npy.exists()} "
                                   f"after={after_npy.exists()}"))
    try:
        a_before = np.load(before_npy)
        a_after = np.load(after_npy)
    except Exception as e:
        return CheckResult(name="n2_edit_oracle", passed=False,
                           detail=f"npy load: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    ref_b_path = (repo_root / task["oracle_before_path"]).resolve()
    ref_a_path = (repo_root / task["oracle_after_path"]).resolve()
    if not ref_b_path.exists() or not ref_a_path.exists():
        return CheckResult(name="n2_edit_oracle", passed=False,
                           detail=(f"oracle missing  before={ref_b_path.exists()} "
                                   f"after={ref_a_path.exists()}"))
    ref_before = np.load(ref_b_path)
    ref_after = np.load(ref_a_path)

    # Grid checks (shape match → strict; otherwise distribution-level)
    pb, db = _grid_distribution_check(a_before, ref_before)
    pa, da = _grid_distribution_check(a_after, ref_after)

    # Grids must differ (ensures agent actually re-ran)
    if a_before.shape == a_after.shape:
        both = np.isfinite(a_before) & np.isfinite(a_after)
        if both.sum() == 0:
            differ_mae = float("nan"); differ_ok = False
        else:
            differ_mae = float(np.abs(a_after[both] - a_before[both]).mean())
            differ_ok = differ_mae >= 1.0
    else:
        differ_mae = float("nan"); differ_ok = True  # shape diff → already different

    # edit_type + method
    expected_edit_type = task.get("edit_type") or task.get("expected_edit_type")
    sim_ok = True; sim_detail = ""
    if sim_res_path.exists():
        try:
            sj = json.loads(sim_res_path.read_text())
            agent_edit_type = sj.get("edit_type")
            method = sj.get("method", "")
            if expected_edit_type and agent_edit_type != expected_edit_type:
                sim_ok = False
                sim_detail = (f" edit_type='{agent_edit_type}' "
                              f"!= expected '{expected_edit_type}'")
            elif str(method).lower() != "sionna_rt":
                sim_ok = False
                sim_detail = f" method='{method}' (not 'sionna_rt')"
        except Exception as e:
            sim_ok = False
            sim_detail = f" simulation_result.json parse: {e}"
    else:
        sim_ok = False
        sim_detail = " simulation_result.json missing"

    # Optional edit-specific delta check
    exp_dmean = task.get("expected_delta_mean_db")
    exp_dstd = task.get("expected_delta_std_db")
    delta_ok = True
    delta_detail = ""
    if exp_dmean is not None and a_before.shape == a_after.shape:
        both = np.isfinite(a_before) & np.isfinite(a_after)
        if both.sum() > 0:
            d = a_after[both] - a_before[both]
            obs_mean = float(d.mean())
            obs_std = float(d.std())
            mean_err = abs(obs_mean - float(exp_dmean))
            mean_ok = mean_err <= 2.0
            std_ok = True
            if exp_dstd is not None:
                std_ok = obs_std <= float(exp_dstd) + 1.0
            delta_ok = mean_ok and std_ok
            delta_detail = (f" delta_mean={obs_mean:+.2f} (exp {exp_dmean:+.2f}, "
                            f"err {mean_err:.2f}, {'PASS' if mean_ok else 'FAIL'})")
            if exp_dstd is not None:
                delta_detail += (f" delta_std={obs_std:.2f} (exp ≤{float(exp_dstd)+1.0:.2f}, "
                                 f"{'PASS' if std_ok else 'FAIL'})")

    passed = pb and pa and differ_ok and sim_ok and delta_ok
    detail = (f"before: {'PASS' if pb else 'FAIL'}({db}) | "
              f"after: {'PASS' if pa else 'FAIL'}({da}) | "
              f"differ={differ_mae:.2f}dB {'PASS' if differ_ok else 'FAIL'} ≥1 |"
              f"{delta_detail}{sim_detail}")
    return CheckResult(name="n2_edit_oracle", passed=passed, detail=detail[:400])


def _check_n3_multi_ap_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N3 multi-AP reference oracle.

    The agent ran a 2-AP coverage scenario. Required outputs:
      - coverage_ap_0.npy / coverage_ap_1.npy   per-AP RSS in dBm
      - coverage_best_server.npy                element-wise max
      - serving_ap_map.npy                       element-wise argmax (int)
      - simulation_result.json

    Pass conditions:
      1. Both per-AP .npy match the reference oracle (grid or distribution).
      2. coverage_best_server matches np.fmax.reduce of agent's per-AP grids
         within 0.5 dB MAE (consistency).
      3. serving_ap_map matches argmax of agent's per-AP grids on ≥95% of valid cells.
      4. simulation_result.json declares num_aps == 2, method == "sionna_rt",
         ap_positions has 2 entries.
    """
    import numpy as np

    files = {n: output_dir / f"{n}.npy" for n in
             ("coverage_ap_0", "coverage_ap_1",
              "coverage_best_server", "serving_ap_map")}
    missing = [n for n, p in files.items() if not p.exists()]
    if missing:
        return CheckResult(name="n3_multi_ap_oracle", passed=False,
                           detail=f"missing artifact(s): {missing}")
    try:
        agent_ap0 = np.load(files["coverage_ap_0"])
        agent_ap1 = np.load(files["coverage_ap_1"])
        agent_best = np.load(files["coverage_best_server"])
        agent_serv = np.load(files["serving_ap_map"])
    except Exception as e:
        return CheckResult(name="n3_multi_ap_oracle", passed=False,
                           detail=f"npy load: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    ref_dir_rel = task.get("oracle_dir")
    if not ref_dir_rel:
        return CheckResult(name="n3_multi_ap_oracle", passed=False,
                           detail="task spec missing oracle_dir")
    ref_dir = (repo_root / ref_dir_rel).resolve()
    if not ref_dir.exists():
        return CheckResult(name="n3_multi_ap_oracle", passed=False,
                           detail=f"oracle dir missing: {ref_dir}")
    try:
        ref_ap0 = np.load(ref_dir / "coverage_ap_0.npy")
        ref_ap1 = np.load(ref_dir / "coverage_ap_1.npy")
    except Exception as e:
        return CheckResult(name="n3_multi_ap_oracle", passed=False,
                           detail=f"oracle load: {e}")

    # Check 1: per-AP grids match reference
    p0, d0 = _grid_distribution_check(agent_ap0, ref_ap0)
    p1, d1 = _grid_distribution_check(agent_ap1, ref_ap1)

    # Check 2: best_server consistent with per-AP max
    stacked = np.stack([agent_ap0, agent_ap1], axis=0)
    expected_best = np.fmax.reduce(stacked, axis=0)
    if agent_best.shape != expected_best.shape:
        best_consistent = False
        best_detail = (f"shape mismatch best={agent_best.shape} "
                       f"derived={expected_best.shape}")
    else:
        both_valid = np.isfinite(agent_best) & np.isfinite(expected_best)
        if both_valid.sum() == 0:
            best_consistent = False
            best_detail = "no jointly-valid cells"
        else:
            best_mae = float(np.abs(agent_best[both_valid] -
                                    expected_best[both_valid]).mean())
            best_consistent = best_mae <= 0.5
            best_detail = f"MAE={best_mae:.3f}dB"

    # Check 3: serving_ap consistent with per-AP argmax
    finite = np.isfinite(stacked)
    vals = np.where(finite, stacked, -np.inf)
    expected_serv = np.argmax(vals, axis=0).astype(np.int8)
    expected_serv = np.where(np.any(finite, axis=0), expected_serv, -1)
    if agent_serv.shape != expected_serv.shape:
        serv_consistent = False
        serv_detail = f"shape {agent_serv.shape} vs {expected_serv.shape}"
    else:
        valid = expected_serv >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            serv_consistent = False
            serv_detail = "no valid cells"
        else:
            agreement = float((agent_serv[valid] == expected_serv[valid]).sum()
                              / n_valid)
            serv_consistent = agreement >= 0.95
            serv_detail = f"{agreement * 100:.1f}% agreement"

    # Check 4: simulation_result.json metadata
    sim_path = output_dir / "simulation_result.json"
    sim_ok = True; sim_detail = ""
    if sim_path.exists():
        try:
            sj = json.loads(sim_path.read_text())
            num_aps = sj.get("num_aps")
            method = str(sj.get("method", "")).lower()
            ap_positions = sj.get("ap_positions") or []
            if num_aps != 2:
                sim_ok = False; sim_detail = f" num_aps={num_aps} (≠2)"
            elif method != "sionna_rt":
                sim_ok = False; sim_detail = f" method='{method}'"
            elif not isinstance(ap_positions, list) or len(ap_positions) != 2:
                sim_ok = False; sim_detail = f" ap_positions len={len(ap_positions)}"
        except Exception as e:
            sim_ok = False; sim_detail = f" sim_res parse: {e}"
    else:
        sim_ok = False; sim_detail = " simulation_result.json missing"

    passed = p0 and p1 and best_consistent and serv_consistent and sim_ok
    detail = (f"AP0: {'PASS' if p0 else 'FAIL'}({d0}) | "
              f"AP1: {'PASS' if p1 else 'FAIL'}({d1}) | "
              f"best={'PASS' if best_consistent else 'FAIL'}({best_detail}) | "
              f"serv={'PASS' if serv_consistent else 'FAIL'}({serv_detail}) | "
              f"sim={'PASS' if sim_ok else 'FAIL'}({sim_detail})")
    return CheckResult(name="n3_multi_ap_oracle", passed=passed, detail=detail[:400])


def _check_sionna_phy_used(output_dir: Path) -> CheckResult:
    """Verify the agent ran an actual Sionna PHY Monte-Carlo chain (not just
    `scipy.special.erfc` analytical).

    Pass conditions (ALL):
      1. simulation.py has non-commented `import sionna` (or `from sionna...`).
      2. simulation_result.json `method` == "sionna_phy".
      3. simulation.py references at least one of: Mapper, Demapper, AWGN,
         BinarySource, ebnodb2no, sim_ber, compute_ber, LDPC5GEncoder.
      4. simulation.py does NOT call `scipy.special.erfc` directly to compute
         the curve (presence of erfc in a comment / docstring is fine; the
         heuristic is "if scipy.special.erfc is the SOLE numeric source and no
         Sionna PHY class is used, this is analytical fallback").
    """
    import re
    sim_py = output_dir / "simulation.py"
    if not sim_py.exists():
        return CheckResult(name="sionna_phy_used", passed=False,
                           detail="simulation.py missing")
    text = sim_py.read_text(errors="replace")
    code_lines = []
    for ln in text.splitlines():
        code_lines.append(ln.split("#", 1)[0])
    code = "\n".join(code_lines)

    has_sionna_import = bool(re.search(r"^\s*(import\s+sionna|from\s+sionna)",
                                       code, re.MULTILINE))
    if not has_sionna_import:
        return CheckResult(name="sionna_phy_used", passed=False,
                           detail="no `import sionna` in simulation.py")

    sionna_phy_apis = ("Mapper", "Demapper", "AWGN", "BinarySource",
                       "ebnodb2no", "sim_ber", "compute_ber",
                       "LDPC5GEncoder", "LDPC5GDecoder",
                       "Polar5GEncoder", "Polar5GDecoder",
                       "sionna.phy.")
    api_hit = next((api for api in sionna_phy_apis if api in code), None)
    if api_hit is None:
        return CheckResult(name="sionna_phy_used", passed=False,
                           detail="no Sionna PHY API used (Mapper/Demapper/"
                                  "AWGN/ebnodb2no/LDPC5G*/sionna.phy.*)")

    # Erfc-only fallback check: scipy.special.erfc is fine as a *reference*,
    # but if it's the only numeric source and there's no Monte Carlo loop, fail.
    uses_erfc = "erfc" in code
    has_mc_loop = bool(re.search(r"\bfor\b.*\bsnr\b|\bfor\b.*\beb_n0\b|\bfor\b.*\bsigma\b",
                                  code, re.IGNORECASE))
    erfc_only = uses_erfc and not has_mc_loop
    if erfc_only:
        return CheckResult(name="sionna_phy_used", passed=False,
                           detail="uses scipy.special.erfc as analytical fallback "
                                  "(no Sionna MC loop detected)")

    # Method field
    sim_res = output_dir / "simulation_result.json"
    method_ok = True; method_detail = ""
    if sim_res.exists():
        try:
            sj = json.loads(sim_res.read_text())
            method = str(sj.get("method", "")).lower()
            if "sionna_phy" not in method:
                method_ok = False
                method_detail = f" method='{method}' (not 'sionna_phy')"
        except Exception:
            method_ok = False
            method_detail = " simulation_result.json parse failed"

    if not method_ok:
        return CheckResult(name="sionna_phy_used", passed=False,
                           detail=f"`import sionna` + API {api_hit} found,"
                                  f" but{method_detail}")
    return CheckResult(name="sionna_phy_used", passed=True,
                       detail=f"`import sionna` + API {api_hit}; method=sionna_phy")


def _check_p1_oracle(task: dict, output_dir: Path) -> CheckResult:
    """P1 azimuth/downtilt optimization reference oracle.

    Reads agent's simulation_result.json and compares against precomputed
    Sionna RT + PHY oracle. Three sub-checks:
      (a) near-optimal:  agent_best_throughput >= 0.95 * ref_best_throughput
      (b) no-fabrication: agent_best_throughput <= 1.05 * ref_best_throughput
                          AND the (az, dt) it claims must really yield that
                          throughput in the agent's own sweep table.
      (c) sweep-consistency: per-config path_gain_db MAE <= 3 dB across the
                             15 matching grid points (vs ref).

    Path-gain (rather than throughput) is used for (c) because the LDPC
    waterfall is steep — small Eb/N0 noise flips throughput 0<->1 and
    swamps any meaningful MAE measurement.
    """
    import numpy as np

    result_path = output_dir / "simulation_result.json"
    if not result_path.exists():
        return CheckResult(name="p1_oracle", passed=False,
                           detail="simulation_result.json missing")
    try:
        agent = json.loads(result_path.read_text())
    except Exception as e:
        return CheckResult(name="p1_oracle", passed=False,
                           detail=f"simulation_result parse: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return CheckResult(name="p1_oracle", passed=False,
                           detail="task spec missing oracle_path")
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return CheckResult(name="p1_oracle", passed=False,
                           detail=f"oracle not found: {oracle_abs}")
    try:
        ref = json.loads(oracle_abs.read_text())
    except Exception as e:
        return CheckResult(name="p1_oracle", passed=False,
                           detail=f"oracle parse: {e}")

    agent_sweep = agent.get("sweep_table") or []
    if not isinstance(agent_sweep, list) or len(agent_sweep) == 0:
        return CheckResult(name="p1_oracle", passed=False,
                           detail=f"sweep_table missing/empty (got {type(agent_sweep).__name__})")

    agent_best = agent.get("best") or {}
    a_tput = agent_best.get("throughput")
    a_az   = agent_best.get("az_deg")
    a_dt   = agent_best.get("dt_deg")
    if a_tput is None or a_az is None or a_dt is None:
        return CheckResult(name="p1_oracle", passed=False,
                           detail=f"best.{{throughput,az_deg,dt_deg}} required, got {agent_best}")
    a_tput = float(a_tput)

    ref_best = ref["best"]
    r_tput = float(ref_best["throughput"])

    # (a) near-optimal
    a_check = a_tput >= 0.95 * r_tput - 1e-9
    # (b) no-fabrication: cap at 5% above ref AND best entry must be in agent's own sweep
    b_cap = a_tput <= 1.05 * r_tput + 1e-9
    b_in_sweep = False
    for row in agent_sweep:
        try:
            if (abs(float(row.get("az_deg", 1e9)) - float(a_az)) <= 0.51 and
                abs(float(row.get("dt_deg", 1e9)) - float(a_dt)) <= 0.51 and
                abs(float(row.get("throughput", -1)) - a_tput) <= 0.02):
                b_in_sweep = True
                break
        except Exception:
            continue
    b_check = b_cap and b_in_sweep

    # (c) sweep MAE on path_gain_db over matching (az, dt) grid cells
    ref_by_key = {(round(float(r["az_deg"]), 2), round(float(r["dt_deg"]), 2)): r
                  for r in ref["sweep_table"]}
    diffs = []
    matched = 0
    for row in agent_sweep:
        try:
            key = (round(float(row["az_deg"]), 2), round(float(row["dt_deg"]), 2))
        except Exception:
            continue
        ref_row = ref_by_key.get(key)
        if ref_row is None:
            continue
        try:
            apg = float(row.get("path_gain_db"))
            rpg = float(ref_row["path_gain_db"])
        except (TypeError, ValueError):
            continue
        if apg <= -180 or rpg <= -180:
            continue
        diffs.append(abs(apg - rpg))
        matched += 1
    if matched >= 12:
        mae = float(np.mean(diffs))
        c_check = mae <= 3.0
    else:
        mae = float("nan")
        c_check = False

    passed = a_check and b_check and c_check
    detail = (f"best=({a_az:+.0f},{a_dt:.0f})={a_tput:.3f} vs ref=({ref_best['az_deg']:+.0f},"
              f"{ref_best['dt_deg']:.0f})={r_tput:.3f}  "
              f"(a)near≥95%:{'✓' if a_check else '✗'}  "
              f"(b)≤105%+in_sweep:{'✓' if b_check else '✗'}"
              f"(cap={'✓' if b_cap else '✗'},in_sweep={'✓' if b_in_sweep else '✗'})  "
              f"(c)pg_MAE={mae:.2f}dB on {matched}/15:{'✓' if c_check else '✗'}")
    return CheckResult(name="p1_oracle", passed=passed, detail=detail[:400])


def _load_s_oracle(task: dict, output_dir: Path):
    """Common oracle-loading + agent-output parsing for S1-S4."""
    repo_root = Path(__file__).resolve().parent.parent
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return None, None, "task spec missing oracle_path"
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return None, None, f"oracle not found: {oracle_abs}"
    try:
        ref = json.loads(oracle_abs.read_text())
    except Exception as e:
        return None, None, f"oracle parse: {e}"
    p = Path(output_dir) / "simulation_result.json"
    if not p.exists():
        return ref, None, "simulation_result.json missing"
    try:
        agent = json.loads(p.read_text())
        return ref, agent, None
    except Exception as e:
        return ref, None, f"simulation_result parse: {e}"


def _check_s1_oracle(task: dict, output_dir: Path) -> CheckResult:
    """S1 fixed deployment: per-user rates, sum throughput, Jain's fairness."""
    import numpy as np
    ref, agent, err = _load_s_oracle(task, output_dir)
    if err:
        return CheckResult(name="s1_oracle", passed=False, detail=err)

    a_rates = agent.get("per_user_rate_bps_hz") or []
    a_sum   = agent.get("sum_throughput_bps_hz")
    a_fair  = agent.get("fairness_index")
    r_rates = ref["per_user_rate_bps_hz"]
    r_sum   = ref["sum_throughput_bps_hz"]
    r_fair  = ref["fairness_index"]

    if not isinstance(a_rates, list) or len(a_rates) != len(r_rates):
        return CheckResult(name="s1_oracle", passed=False,
                           detail=f"per_user_rate length {len(a_rates) if a_rates else 0} vs ref {len(r_rates)}")
    try:
        ar = np.array(a_rates, dtype=float)
    except Exception:
        return CheckResult(name="s1_oracle", passed=False,
                           detail=f"per_user_rate non-numeric")
    if (ar < -1e-6).any():
        return CheckResult(name="s1_oracle", passed=False,
                           detail=f"per_user_rate has negative values: {ar.tolist()}")

    sum_ok = a_sum is not None and abs(float(a_sum) - r_sum) / max(r_sum, 0.1) <= 0.30
    fair_ok = a_fair is not None and abs(float(a_fair) - r_fair) <= 0.15
    # Fairness recomputed from agent's own rates must agree (within 0.02)
    ar_sum = float(ar.sum())
    if ar_sum > 0:
        recomp = float((ar_sum ** 2) / (ar.size * (ar ** 2).sum()))
    else:
        recomp = 0.0
    fair_consistent = a_fair is not None and abs(float(a_fair) - recomp) <= 0.02

    passed = sum_ok and fair_ok and fair_consistent
    detail = (f"sum_rate={a_sum} vs ref={r_sum}:{'✓' if sum_ok else '✗'}  "
              f"fairness={a_fair} vs ref={r_fair}:{'✓' if fair_ok else '✗'}  "
              f"fairness_self_consistent (recomp={recomp:.3f}):"
              f"{'✓' if fair_consistent else '✗'}")
    return CheckResult(name="s1_oracle", passed=passed, detail=detail[:400])


def _check_s2_oracle(task: dict, output_dir: Path) -> CheckResult:
    """S2 fixed scheduler: per-user throughput, scheduled-UE trace, fairness."""
    import numpy as np
    ref, agent, err = _load_s_oracle(task, output_dir)
    if err:
        return CheckResult(name="s2_oracle", passed=False, detail=err)

    a_thr  = agent.get("per_user_throughput_bps_hz") or []
    a_sum  = agent.get("sum_throughput_bps_hz")
    a_fair = agent.get("fairness_index")
    a_sched = agent.get("scheduled_ue") or []
    T = ref["num_slots"]
    n_ue = len(ref["per_user_throughput_bps_hz"])

    if not isinstance(a_thr, list) or len(a_thr) != n_ue:
        return CheckResult(name="s2_oracle", passed=False,
                           detail=f"per_user_throughput len {len(a_thr) if a_thr else 0} vs ref {n_ue}")
    try:
        at = np.array(a_thr, dtype=float)
    except Exception:
        return CheckResult(name="s2_oracle", passed=False,
                           detail="per_user_throughput non-numeric")
    if (at < -1e-6).any():
        return CheckResult(name="s2_oracle", passed=False,
                           detail=f"per_user_throughput has negatives: {at.tolist()}")

    sched_ok = (isinstance(a_sched, list) and len(a_sched) == T and
                all(isinstance(u, int) and 0 <= u < n_ue for u in a_sched))
    r_sum = ref["sum_throughput_bps_hz"]
    r_fair = ref["fairness_index"]
    sum_ok = a_sum is not None and abs(float(a_sum) - r_sum) / max(r_sum, 0.1) <= 0.40
    fair_ok = a_fair is not None and abs(float(a_fair) - r_fair) <= 0.20
    at_sum = float(at.sum())
    recomp = float((at_sum ** 2) / (at.size * (at ** 2).sum())) if at_sum > 0 else 0.0
    fair_consistent = a_fair is not None and abs(float(a_fair) - recomp) <= 0.03

    passed = sched_ok and sum_ok and fair_ok and fair_consistent
    detail = (f"sched_len={len(a_sched)}/{T} valid:{'✓' if sched_ok else '✗'}  "
              f"sum={a_sum} vs {r_sum}:{'✓' if sum_ok else '✗'}  "
              f"fair={a_fair} vs {r_fair}:{'✓' if fair_ok else '✗'}  "
              f"fair_self ({recomp:.3f}):{'✓' if fair_consistent else '✗'}")
    return CheckResult(name="s2_oracle", passed=passed, detail=detail[:400])


def _check_s3_oracle(task: dict, output_dir: Path) -> CheckResult:
    """S3 joint beamforming: best (az1, az2) maximises sum_rate."""
    ref, agent, err = _load_s_oracle(task, output_dir)
    if err:
        return CheckResult(name="s3_oracle", passed=False, detail=err)

    a_sweep = agent.get("sweep_table") or []
    a_best = agent.get("best") or {}
    a_sum = a_best.get("sum_rate")
    r_sum = float(ref["best"]["sum_rate"])

    if not isinstance(a_sweep, list) or len(a_sweep) < 6:
        return CheckResult(name="s3_oracle", passed=False,
                           detail=f"sweep_table size {len(a_sweep) if a_sweep else 0} < 6 (expected 9)")
    if a_sum is None:
        return CheckResult(name="s3_oracle", passed=False,
                           detail=f"best.sum_rate missing")

    near = float(a_sum) >= 0.85 * r_sum - 1e-9
    cap  = float(a_sum) <= 1.15 * r_sum + 1e-9
    # best entry must exist in agent's own sweep table
    in_sweep = False
    a_az1 = a_best.get("az1_deg"); a_az2 = a_best.get("az2_deg")
    if a_az1 is not None and a_az2 is not None:
        for row in a_sweep:
            try:
                if (abs(float(row.get("az1_deg", 1e9)) - float(a_az1)) <= 0.51 and
                    abs(float(row.get("az2_deg", 1e9)) - float(a_az2)) <= 0.51 and
                    abs(float(row.get("sum_rate", -1)) - float(a_sum)) <= 0.05):
                    in_sweep = True
                    break
            except Exception:
                continue
    passed = near and cap and in_sweep
    detail = (f"best=({a_az1},{a_az2})={a_sum} vs ref={r_sum}  "
              f"near≥85%:{'✓' if near else '✗'}  ≤115%:{'✓' if cap else '✗'}  "
              f"in_sweep:{'✓' if in_sweep else '✗'}")
    return CheckResult(name="s3_oracle", passed=passed, detail=detail[:400])


def _check_s4_oracle(task: dict, output_dir: Path) -> CheckResult:
    """S4 RB Pareto: each pareto_set index must be truly non-dominated."""
    ref, agent, err = _load_s_oracle(task, output_dir)
    if err:
        return CheckResult(name="s4_oracle", passed=False, detail=err)

    a_sweep = agent.get("sweep_table") or []
    a_par = agent.get("pareto_set") or []
    if not isinstance(a_sweep, list) or len(a_sweep) < 3:
        return CheckResult(name="s4_oracle", passed=False,
                           detail=f"sweep_table size {len(a_sweep) if a_sweep else 0} < 3")
    if not isinstance(a_par, list) or len(a_par) == 0:
        return CheckResult(name="s4_oracle", passed=False,
                           detail="pareto_set missing/empty")
    try:
        a_par = [int(i) for i in a_par]
    except Exception:
        return CheckResult(name="s4_oracle", passed=False,
                           detail="pareto_set entries must be ints")
    n = len(a_sweep)
    if any(not (0 <= i < n) for i in a_par):
        return CheckResult(name="s4_oracle", passed=False,
                           detail="pareto_set out of range")

    pts = []
    for row in a_sweep:
        try:
            pts.append((float(row["sum_rate"]), float(row["fairness"])))
        except (KeyError, TypeError, ValueError):
            return CheckResult(name="s4_oracle", passed=False,
                               detail="sweep_table row missing sum_rate/fairness")
    truly = set()
    for i, (s_i, f_i) in enumerate(pts):
        dominated = False
        for j, (s_j, f_j) in enumerate(pts):
            if i == j: continue
            ge = (s_j >= s_i - 1e-9 and f_j >= f_i - 1e-9)
            sb = (s_j > s_i + 1e-9 or f_j > f_i + 1e-9)
            if ge and sb:
                dominated = True; break
        if not dominated:
            truly.add(i)
    fabricated = [i for i in a_par if i not in truly]
    fabric_ok = len(fabricated) == 0
    # Coverage: agent's Pareto must overlap >=50% of ref Pareto size in agent's table
    ref_par_size = len(ref["pareto_set"])
    coverage_ok = len(a_par) >= max(1, ref_par_size // 2)

    passed = fabric_ok and coverage_ok
    detail = (f"pareto={len(a_par)} (truly_nondom={len(truly)}, ref={ref_par_size})  "
              f"fabricated={len(fabricated)}:{'✓' if fabric_ok else '✗'}  "
              f"coverage:{'✓' if coverage_ok else '✗'}")
    return CheckResult(name="s4_oracle", passed=passed, detail=detail[:400])


def _check_p2_oracle(task: dict, output_dir: Path) -> CheckResult:
    """P2 Pareto-frontier (throughput vs tx_power_mw) reference oracle.

    Three sub-checks:
      (a) hypervolume: HV(agent) >= 0.9 * HV(ref)
          (agent's frontier captures >= 90% of the ref-frontier rectangle area
           wrt reference point (0_throughput, max_power_mW_in_grid))
      (b) no fabrication: every index agent claims as Pareto-optimal must
          actually be non-dominated within agent's own sweep table.
      (c) sweep consistency: per-config path_gain_db MAE <= 3 dB on
          >=12/15 matched grid cells (same convention as P1).
    """
    import numpy as np

    result_path = output_dir / "simulation_result.json"
    if not result_path.exists():
        return CheckResult(name="p2_oracle", passed=False,
                           detail="simulation_result.json missing")
    try:
        agent = json.loads(result_path.read_text())
    except Exception as e:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"simulation_result parse: {e}")

    repo_root = Path(__file__).resolve().parent.parent
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return CheckResult(name="p2_oracle", passed=False,
                           detail="task spec missing oracle_path")
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"oracle not found: {oracle_abs}")
    try:
        ref = json.loads(oracle_abs.read_text())
    except Exception as e:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"oracle parse: {e}")

    agent_sweep = agent.get("sweep_table") or []
    agent_par_idx = agent.get("pareto_set")
    if not isinstance(agent_sweep, list) or len(agent_sweep) == 0:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"sweep_table missing/empty")
    if not isinstance(agent_par_idx, list) or len(agent_par_idx) == 0:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"pareto_set missing/empty")

    # Coerce indices to ints
    try:
        agent_par_idx = [int(i) for i in agent_par_idx]
    except Exception:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"pareto_set must be list[int]")

    n = len(agent_sweep)
    bad_idx = [i for i in agent_par_idx if not (0 <= i < n)]
    if bad_idx:
        return CheckResult(name="p2_oracle", passed=False,
                           detail=f"pareto_set has out-of-range indices {bad_idx}")

    # ---- (b) no fabrication: agent's claimed pareto must be non-dominated in agent's own sweep ----
    def extract_tp_mw(row):
        try:
            return float(row["throughput"]), float(row["tx_power_mw"])
        except (KeyError, TypeError, ValueError):
            return None, None

    agent_points = []
    for row in agent_sweep:
        t, p = extract_tp_mw(row)
        if t is None or p is None:
            return CheckResult(name="p2_oracle", passed=False,
                               detail=f"sweep_table row missing throughput/tx_power_mw")
        agent_points.append((t, p))

    truly_pareto = set()
    for i in range(n):
        t_i, p_i = agent_points[i]
        dominated = False
        for j in range(n):
            if i == j: continue
            t_j, p_j = agent_points[j]
            ge = (t_j >= t_i - 1e-6) and (p_j <= p_i + 1e-6)
            sb = (t_j > t_i + 1e-6) or (p_j < p_i - 1e-6)
            if ge and sb:
                dominated = True; break
        if not dominated:
            truly_pareto.add(i)
    fabricated = [i for i in agent_par_idx if i not in truly_pareto]
    b_check = len(fabricated) == 0

    # ---- (a) hypervolume comparison ----
    def hv_2d(points, ref_max_power):
        # Pareto in (max throughput, min power): sort by ascending throughput
        # (which also means ascending power for a true Pareto set) and sum
        # the incremental rectangle (thr_i - thr_{i-1}) * (p_ref - pwr_i).
        if not points: return 0.0
        pts = sorted(points, key=lambda p: p[0])
        hv = 0.0
        prev_thr = 0.0
        for thr, pwr in pts:
            if thr <= prev_thr: continue
            hv += (thr - prev_thr) * max(0.0, ref_max_power - pwr)
            prev_thr = thr
        return hv

    ref_hv = float(ref.get("pareto_hypervolume", 0.0))
    ref_max_power = float(ref.get("hv_ref_point", {}).get("tx_power_mw",
                                  max(r["tx_power_mw"] for r in ref["sweep_table"])))
    agent_pareto_pts = [agent_points[i] for i in agent_par_idx]
    agent_hv = hv_2d(agent_pareto_pts, ref_max_power)
    if ref_hv <= 0:
        a_check = True   # degenerate reference; pass trivially
        hv_ratio = 1.0
    else:
        hv_ratio = agent_hv / ref_hv
        a_check = hv_ratio >= 0.9 - 1e-6

    # ---- (c) sweep consistency: path_gain_db MAE <= 3 dB across matched (az, P) keys ----
    def key_of(row):
        try:
            return (round(float(row["az_deg"]), 2),
                    round(float(row["tx_power_dbm"]), 2))
        except (KeyError, TypeError, ValueError):
            return None

    ref_by_key = {}
    for r in ref["sweep_table"]:
        k = key_of(r)
        if k is not None:
            ref_by_key[k] = r

    diffs = []
    matched = 0
    for row in agent_sweep:
        k = key_of(row)
        if k is None: continue
        rr = ref_by_key.get(k)
        if rr is None: continue
        try:
            apg = float(row.get("path_gain_db"))
            rpg = float(rr["path_gain_db"])
        except (TypeError, ValueError):
            continue
        if apg <= -180 or rpg <= -180: continue
        diffs.append(abs(apg - rpg))
        matched += 1
    if matched >= 12:
        mae = float(np.mean(diffs))
        c_check = mae <= 3.0
    else:
        mae = float("nan")
        c_check = False

    passed = a_check and b_check and c_check
    detail = (f"agent_pareto={len(agent_par_idx)} (truly nondom={len(truly_pareto)})  "
              f"HV ratio={hv_ratio:.3f}:{'✓' if a_check else '✗'} (a)  "
              f"fabricated={len(fabricated)}:{'✓' if b_check else '✗'} (b)  "
              f"pg_MAE={mae:.2f}dB on {matched}/15:{'✓' if c_check else '✗'} (c)")
    return CheckResult(name="p2_oracle", passed=passed, detail=detail[:400])


def _check_n4_phy_oracle(task: dict, output_dir: Path) -> CheckResult:
    """N4 single-user PHY link reference oracle.

    Compares agent's `{metric}_curve.npy` to the precomputed Sionna PHY
    reference. Pass conditions:
      1. Curve .npy exists with shape (N, 2): col 0 = Eb/N0 dB, col 1 = metric.
      2. Per-point check vs reference:
           BER/BLER: |log10(agent) - log10(ref)| ≤ 0.5  (≈ factor 3)
           throughput: |agent - ref| / max(ref, 0.01) ≤ 0.15  (≤15% relative)
         At least 4/5 points must pass.
      3. Monotonicity:
           BER/BLER must decrease with SNR.
           throughput must increase with SNR.
    """
    import numpy as np

    metric_type = task.get("metric_type", "ber")
    npy_name = {
        "ber": "ber_curve.npy",
        "bler": "bler_curve.npy",
        "throughput": "throughput_curve.npy",
    }.get(metric_type)
    if npy_name is None:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"unknown metric_type {metric_type!r}")

    curve_path = output_dir / npy_name
    if not curve_path.exists():
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"{npy_name} missing")
    try:
        curve = np.load(curve_path)
    except Exception as e:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"npy load: {e}")
    if curve.ndim != 2 or curve.shape[1] != 2:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"curve shape {curve.shape}, expected (N, 2)")

    repo_root = Path(__file__).resolve().parent.parent
    oracle_rel = task.get("oracle_path")
    if not oracle_rel:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail="task spec missing oracle_path")
    oracle_abs = (repo_root / oracle_rel).resolve()
    if not oracle_abs.exists():
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"oracle not found: {oracle_abs}")
    try:
        ref = json.loads(oracle_abs.read_text())
    except Exception as e:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"oracle parse: {e}")

    ref_ebn0 = np.array(ref["eb_n0_db"], dtype=float)
    ref_vals = np.array(ref["metric_values"], dtype=float)

    agent_ebn0 = curve[:, 0]
    agent_vals = curve[:, 1]

    # SNR points must match (tolerance 0.1 dB)
    if len(agent_ebn0) != len(ref_ebn0):
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"point count {len(agent_ebn0)} vs ref {len(ref_ebn0)}")
    snr_diff = float(np.abs(agent_ebn0 - ref_ebn0).max())
    if snr_diff > 0.1:
        return CheckResult(name="n4_phy_oracle", passed=False,
                           detail=f"Eb/N0 grid mismatch: max diff {snr_diff:.2f} dB")

    # Per-point comparison
    is_log = metric_type in ("ber", "bler")
    per_point = []
    pass_count = 0
    for i, (snr, a, r) in enumerate(zip(agent_ebn0, agent_vals, ref_vals)):
        if is_log:
            la = np.log10(max(a, 1e-10))
            lr = np.log10(max(r, 1e-10))
            err = abs(la - lr)
            ok = err <= 0.5
            per_point.append((float(snr), float(a), float(r), float(err), ok))
        else:
            err = abs(a - r) / max(r, 0.01)
            ok = err <= 0.15
            per_point.append((float(snr), float(a), float(r), float(err), ok))
        if ok:
            pass_count += 1

    points_ok = pass_count >= 4

    # Monotonicity
    if is_log:
        # decreasing
        diffs = np.diff(agent_vals)
        monotone_ok = bool(np.all(diffs <= 1e-6))  # allow tie, no increase
    else:
        # increasing (throughput)
        diffs = np.diff(agent_vals)
        monotone_ok = bool(np.all(diffs >= -1e-6))

    passed = points_ok and monotone_ok
    # Detail string
    point_str = " ".join(
        f"{snr:+.1f}:{a:.2e}vs{r:.2e}({'✓' if ok else '✗'}{err:.2f})"
        if is_log else
        f"{snr:+.1f}:{a:.3f}vs{r:.3f}({'✓' if ok else '✗'}{err:.0%})"
        for snr, a, r, err, ok in per_point)
    detail = (f"{metric_type} {pass_count}/{len(per_point)} points "
              f"{'PASS' if points_ok else 'FAIL'} (≥4); "
              f"monotone {'PASS' if monotone_ok else 'FAIL'}; {point_str}")
    return CheckResult(name="n4_phy_oracle", passed=passed, detail=detail[:400])


def _check_sionna_rt_used(output_dir: Path) -> CheckResult:
    """Verify the agent actually used Sionna RT (not FSPL fallback).

    Pass conditions (ALL must hold):
      1. simulation.py contains a non-commented `import sionna` or
         `from sionna` line.
      2. simulation_result.json's `method` field starts with "sionna" OR
         `status` is NOT in the analytical-fallback set.
      3. A Mitsuba 3 scene XML artifact exists in the trial dir (Sionna RT
         loads scenes from .xml). This is the strongest signal: an agent
         that uses Sionna RT MUST produce / consume a .xml file because
         sionna.rt.load_scene() takes an XML path. Agents using FSPL or
         FSPL-with-manual-wall-loss never produce a .xml.

    This catches agents that emit `method="completed_analytical"` or
    silently fall back to a hand-coded FSPL formula when the default
    `python3` lacks the sionna module.
    """
    import re
    sim_py = None
    for cand in ("simulation.py", "simulate.py"):
        p = output_dir / cand
        if p.exists():
            sim_py = p
            break
    if sim_py is None:
        # Try any .py file with "simulation" in name
        for p in output_dir.glob("*.py"):
            if "sim" in p.name.lower():
                sim_py = p
                break
    if sim_py is None:
        return CheckResult(name="sionna_rt_used", passed=False,
                           detail="no simulation.py found")

    text = sim_py.read_text(errors="replace")
    # Strip comments before matching, so `# import sionna` doesn't count
    lines_no_comment = []
    for ln in text.splitlines():
        # Trim inline comments and pure comment lines
        s = ln.split("#", 1)[0]
        lines_no_comment.append(s)
    code = "\n".join(lines_no_comment)
    has_import = bool(re.search(r"^\s*(import\s+sionna|from\s+sionna)",
                                code, re.MULTILINE))
    if not has_import:
        return CheckResult(name="sionna_rt_used", passed=False,
                           detail=f"{sim_py.name} has no `import sionna` "
                                  f"(non-commented) — agent did NOT use Sionna RT")

    # Check simulation_result.json's method / status
    sim_res = output_dir / "simulation_result.json"
    if sim_res.exists():
        try:
            sim = json.loads(sim_res.read_text())
        except Exception:
            sim = {}
        method = str(sim.get("method") or "").lower()
        status = str(sim.get("status") or "").lower()
        bad_fallbacks = ("analytical", "fspl_only", "completed_analytical",
                         "success_analytical_fallback")
        if any(b in status for b in bad_fallbacks):
            return CheckResult(name="sionna_rt_used", passed=False,
                               detail=f"status='{status}' indicates analytical "
                                      f"fallback (sionna not actually used)")
        if any(b in method for b in bad_fallbacks):
            return CheckResult(name="sionna_rt_used", passed=False,
                               detail=f"method='{method}' indicates analytical fallback")

    # Check for Mitsuba XML artifact (strongest signal — sionna.rt.load_scene
    # takes an .xml path, so any real RT run produces one. Two acceptance
    # paths: a workdir XML (custom scene), OR a `rt.scene.<builtin>` reference
    # in the agent's code (built-in scenes — XML lives in the sionna install dir).
    BUILTIN_SCENES = (
        "box", "box_knife", "box_one_screen", "box_two_screens",
        "double_reflector", "etoile", "floor_wall", "florence", "munich",
        "san_francisco", "simple_reflector", "simple_street_canyon",
        "simple_street_canyon_with_cars", "simple_wedge", "triple_reflector",
    )
    builtin_used = None
    for name in BUILTIN_SCENES:
        # Accept several common ways agents reference built-in scenes:
        #   rt.scene.munich              (direct attribute)
        #   sionna.rt.scene.munich       (qualified)
        #   getattr(rt.scene, "munich")  (programmatic, uses scene_name var)
        #   "munich"                     (in load_scene("munich") rare; keep tight via context)
        # Require word boundary so we don't match "boxes" or similar.
        patterns = [
            rf"\brt\.scene\.{name}\b",
            rf"\bgetattr\(\s*(?:sionna\.)?rt\.scene\s*,\s*[\"']{name}[\"']\s*\)",
        ]
        if any(re.search(p, code) for p in patterns):
            builtin_used = name
            break
    # Fallback: also accept if simulation_result.json declares a built-in
    # scene_name AND the agent used getattr(rt.scene, ...) generically.
    if builtin_used is None:
        sim_res = output_dir / "simulation_result.json"
        if sim_res.exists() and re.search(r"getattr\(\s*(?:sionna\.)?rt\.scene\s*,",
                                          code):
            try:
                sj = json.loads(sim_res.read_text())
                declared = sj.get("scene_name")
                if not declared:
                    scn = sj.get("scene")
                    if isinstance(scn, str):
                        declared = scn
                    elif isinstance(scn, dict):
                        declared = scn.get("name")
                if isinstance(declared, str) and declared in BUILTIN_SCENES:
                    builtin_used = declared
            except Exception:
                pass

    xml_artifacts = list(output_dir.rglob("*.xml"))
    # Filter to Mitsuba-style scene XMLs (heuristic: contains "<scene" tag)
    mitsuba_xmls = []
    for x in xml_artifacts:
        try:
            head = x.read_text(errors="replace")[:500].lower()
            if "<scene" in head or "version=\"2" in head or "version=\"3" in head:
                mitsuba_xmls.append(x)
        except Exception:
            continue

    if builtin_used is None and not mitsuba_xmls:
        return CheckResult(
            name="sionna_rt_used", passed=False,
            detail=(f"`import sionna` present in {sim_py.name} but neither "
                    f"a `rt.scene.<builtin>` reference nor a Mitsuba scene.xml "
                    f"artifact was found in {output_dir.name}/ — agent may "
                    f"have imported sionna without actually calling load_scene()."))

    detail = f"`import sionna` in {sim_py.name}; non-analytical status; "
    if builtin_used is not None:
        detail += f"built-in scene: rt.scene.{builtin_used}"
    if mitsuba_xmls:
        detail += f"; XML artifact: {mitsuba_xmls[0].relative_to(output_dir)}"
    return CheckResult(name="sionna_rt_used", passed=True, detail=detail)


def _check_execution_ok(exec_success: bool) -> CheckResult:
    return CheckResult(name="execution_ok", passed=exec_success,
                       detail="agent reported execution_success")


def _check_file_exists(task: dict, output_dir: Path) -> list[CheckResult]:
    out = []
    for art in task.get("required_artifacts", []):
        p = output_dir / art
        # Accept any path match (e.g., outputs/scene_01/scene_state.json)
        candidates: list[Path] = []
        if p.exists():
            candidates.append(p)
        else:
            candidates.extend(output_dir.rglob(art))
        if not candidates:
            out.append(CheckResult(name=f"artifact:{art}", passed=False,
                                   detail="missing"))
            continue
        # If the artifact is a JSON file the harness pre-ships
        # (simulation_result.json / scene_state.json), reject when its
        # `status` field is still the harness placeholder. Otherwise an
        # agent that never wrote anything would pass this check.
        actual = candidates[0]
        if art.endswith(".json"):
            try:
                content = json.loads(actual.read_text())
                # 2026-05-15: only reject as placeholder if the agent
                # ALSO left numerical_metrics empty/null. Agents following
                # SKILL.md often populate metrics but keep `status`
                # untouched, which we should accept.
                is_placeholder_status = (isinstance(content, dict) and
                    content.get("status") == "placeholder_pre_shipped_by_harness")
                metrics = (content or {}).get("numerical_metrics") or {}
                metrics_populated = isinstance(metrics, dict) and any(
                    v not in (None, [], {}, "") for v in metrics.values()
                )
                if is_placeholder_status and not metrics_populated:
                    out.append(CheckResult(
                        name=f"artifact:{art}", passed=False,
                        detail="file is harness-pre-shipped placeholder, "
                               "not real model output"))
                    continue
            except (json.JSONDecodeError, OSError):
                # Malformed/unreadable JSON — keep prior behavior of
                # treating presence as a pass; other checks (e.g.
                # plausibility, metric_threshold) will catch real issues.
                pass
        out.append(CheckResult(name=f"artifact:{art}", passed=True,
                               detail=str(actual)))
    return out


# ---------------------------------------------------------------------------
# Per-metric handler functions (one per known metric)
# ---------------------------------------------------------------------------

def _check_model_selection(code: str, _task: dict) -> CheckResult:
    # T06: multi-user must use UMi|UMa, single-user CDL|TDL
    # Heuristic: check both tokens appear
    expected = ["UMi", "UMa", "CDL", "TDL"]
    found = [e for e in expected if e in code]
    return CheckResult(name="model_selection",
                       passed=len(found) >= 2,
                       detail=f"found={found}")


def _check_v2_namespace(code: str, _task: dict) -> CheckResult:
    bad = any(m in code for m in ["sionna.channel.", "sionna.mapping.", "sionna.ofdm."])
    return CheckResult(name="uses_v2_namespace", passed=not bad,
                       detail="found legacy v0.x imports" if bad else "ok")


def _check_norm_constrained(code: str, _task: dict) -> CheckResult:
    ok = any(tok in code for tok in ["torch.linalg.norm", "torch.norm", "normalize", ".norm()"])
    return CheckResult(name="norm_constrained", passed=ok,
                       detail="no norm constraint found" if not ok else "ok")


def _check_power_normalized(code: str, _task: dict) -> CheckResult:
    ok = any(tok in code for tok in ["normalize", "/ torch.sqrt", "power_constraint", "P_max"])
    return CheckResult(name="power_normalized", passed=ok,
                       detail="no power normalization" if not ok else "ok")


def _check_learnable_params(code: str, _task: dict) -> CheckResult:
    ok = any(tok in code for tok in ["nn.Parameter", "requires_grad=True", "trainable_weights"])
    return CheckResult(name="learnable_params", passed=ok)


def _check_accuracy_metric(code: str, _task: dict) -> CheckResult:
    ok = "accuracy" in code.lower() and "classification" in code.lower()
    return CheckResult(name="accuracy_metric", passed=ok)


def _check_molecular_absorption(code: str, _task: dict) -> CheckResult:
    ok = "absorption" in code.lower() or "p.676" in code.lower()
    return CheckResult(name="molecular_absorption", passed=ok)


def _check_rayleigh(code: str, _task: dict) -> CheckResult:
    # Check for 2D²/λ or equivalent
    ok = re.search(r"2\s*\*\s*D\s*\*?\*?2?\s*/\s*lamb|rayleigh_distance", code) is not None
    return CheckResult(name="rayleigh_distance", passed=ok)


# Registry mapping metric name → handler(code, task) -> CheckResult
_CODE_CONTAINS_HANDLERS: dict[str, Callable] = {
    "correct_model_selection": _check_model_selection,
    "code_runs_v2":            _check_v2_namespace,
    "norm_constrained":        _check_norm_constrained,
    "power_normalized":        _check_power_normalized,
    "learnable_params_present": _check_learnable_params,
    "eval_metric_is_accuracy": _check_accuracy_metric,
    "absorption_applied":      _check_molecular_absorption,
    "rayleigh_distance_correct": _check_rayleigh,
}


def _check_generic_tokens(code: str, py_code: str, bash_code: str,
                           metric: str, task: dict) -> CheckResult:
    """Generic fallback: tokenize the metric name and require each token
    (ignoring connective words like "better", "than", "correct", "good")
    appears somewhere in the code. E.g. "coded_below_uncoded" requires
    both "coded" and "uncoded" in the code; "uses_correct_channel" + expected=UMi
    requires "UMi" in the code."""
    skip_tokens = {"is", "of", "to", "the", "a", "for", "by",
                   "better", "than", "correct", "good", "right", "bad",
                   "with", "without", "more", "less", "uses", "use",
                   "check", "get", "has", "do", "does", "does"}
    # Synonym groups: any one alternative satisfies the token. Catches
    # legitimate naming variants the agent may emit (e.g. mmWave/mm-wave,
    # interval/interval_ms, decouple/decoupling) without weakening the
    # check on otherwise-distinct identifiers.
    synonyms = {
        "mmwave": ["mmwave", "mm-wave", "mm_wave", "millimeter"],
        "thz":    ["thz", "terahertz", "tera-hertz"],
        "interval": ["interval", "interval_ms", "update_interval", "tti"],
        "decoupling": ["decouple", "decoupling", "decoupled"],
        "decouple":   ["decouple", "decoupling", "decoupled"],
        "monotone":   ["monotone", "monotonic", "monotonically"],
        "monotonic":  ["monotone", "monotonic", "monotonically"],
    }
    tokens = [t for t in metric.lower().split("_")
              if t and t not in skip_tokens and len(t) > 2]
    # Also require v["expected"] literal if it's a string (e.g., "UMi")
    expected_literal = task["verifier"].get("spec", {}).get("expected")
    if isinstance(expected_literal, str):
        tokens.append(expected_literal.lower())
    code_lower = code.lower()

    def _present(t: str) -> bool:
        if t in code_lower:
            return True
        for alt in synonyms.get(t, []):
            if alt in code_lower:
                return True
        return False

    missing = [t for t in tokens if not _present(t)]
    # Length guard: .py-only agents need >200 chars to prevent trivial stubs;
    # heredoc agents (bash_code present) relax this — the bash corpus itself
    # is the evidence of real work.
    has_substance = (len(py_code) > 200
                     or len(bash_code) > 10
                     or len(code) > 200)
    passed = len(tokens) > 0 and not missing and has_substance
    return CheckResult(name=f"code_contains:{metric}", passed=passed,
                       detail=(f"tokens={tokens} missing={missing}"
                               if tokens else "no tokens to check"))


def check_code_contains(task: dict, output_dir: Path) -> CheckResult:
    """Look for expected identifiers in the agent's code."""
    v = task["verifier"]
    metric = v.get("metric", "")
    py_code = load_all_code(output_dir)
    bash_code = load_bash_commands(output_dir)
    code = py_code + "\n" + bash_code

    # Special cases: scene checks need output_dir, not just code text
    if metric == "collision_free_check":
        # Inspect scene_state.json for actual bounding-box overlaps instead of
        # grepping the code — scene_gen tasks can be collision-free by
        # construction (e.g., placed along walls) without any of those tokens.
        return _check_scene_collision_free(output_dir)
    if metric == "in_bounds_check":
        return _check_scene_in_bounds(output_dir)
    if metric == "sionna_loadable_check":
        return _check_sionna_loadable(output_dir)
    if metric == "sionna_rt_used_check":
        return _check_sionna_rt_used(output_dir)
    if metric == "rt_oracle_check":
        return _check_rt_oracle(output_dir)
    if metric == "phy_oracle_check":
        return _check_phy_oracle(output_dir)
    if metric == "sys_oracle_check":
        return _check_sys_oracle(output_dir)
    if metric == "geometry_oracle_check":
        return _check_geometry_oracle(output_dir)
    if metric == "c1_ref_oracle_check":
        return _check_c1_reference_oracle(task, output_dir)
    if metric == "c3_ref_oracle_check":
        return _check_c3_reference_oracle(task, output_dir)
    if metric == "c4_ref_oracle_check":
        return _check_c4_reference_oracle(task, output_dir)
    if metric == "c5_ref_oracle_check":
        return _check_c5_reference_oracle(task, output_dir)
    if metric == "c7_ref_oracle_check":
        return _check_c7_reference_oracle(task, output_dir)
    if metric == "n1_ref_oracle_check":
        return _check_n1_reference_oracle(task, output_dir)
    if metric == "n2_freq_oracle_check":
        return _check_n2_freq_oracle(task, output_dir)
    if metric == "n1_probe_oracle_check":
        return _check_n1_probe_oracle(task, output_dir)
    if metric == "n2_edit_oracle_check":
        return _check_n2_edit_oracle(task, output_dir)
    if metric == "n3_multi_ap_oracle_check":
        return _check_n3_multi_ap_oracle(task, output_dir)
    if metric == "sionna_phy_used_check":
        return _check_sionna_phy_used(output_dir)
    if metric == "n4_phy_oracle_check":
        return _check_n4_phy_oracle(task, output_dir)
    if metric == "p1_oracle_check":
        return _check_p1_oracle(task, output_dir)
    if metric == "p2_oracle_check":
        return _check_p2_oracle(task, output_dir)
    if metric == "s1_oracle_check":
        return _check_s1_oracle(task, output_dir)
    if metric == "s2_oracle_check":
        return _check_s2_oracle(task, output_dir)
    if metric == "s3_oracle_check":
        return _check_s3_oracle(task, output_dir)
    if metric == "s4_oracle_check":
        return _check_s4_oracle(task, output_dir)

    handler = _CODE_CONTAINS_HANDLERS.get(metric)
    if handler is not None:
        return handler(code, task)

    return _check_generic_tokens(code, py_code, bash_code, metric, task)


def check_metric_threshold(task: dict, sim: dict | None) -> CheckResult:
    v = task["verifier"]
    metric = v.get("metric", "")
    threshold = v.get("threshold")
    direction = v.get("direction", "<=")
    if threshold is None:
        return CheckResult(name=f"threshold:{metric}", passed=False,
                           detail="no threshold specified")
    val = extract_scalar(sim, metric, v)
    if val is None:
        return CheckResult(name=f"threshold:{metric}", passed=False,
                           detail="metric not found in output")
    passed = (val <= threshold) if direction == "<=" else (val >= threshold)
    return CheckResult(name=f"threshold:{metric}", passed=passed,
                       detail=f"{val:.4g} {direction} {threshold}")


def check_metric_range(task: dict, sim: dict | None) -> CheckResult:
    v = task["verifier"]
    metric = v.get("metric", "")
    lo, hi = v.get("min"), v.get("max")
    val = extract_scalar(sim, metric, v)
    if val is None or lo is None or hi is None:
        return CheckResult(name=f"range:{metric}", passed=False,
                           detail=f"val={val} range=[{lo},{hi}]")
    passed = lo <= val <= hi
    return CheckResult(name=f"range:{metric}", passed=passed,
                       detail=f"{val:.4g} in [{lo},{hi}]")


def check_metric_monotone(task: dict, sim: dict | None) -> CheckResult:
    v = task["verifier"]
    metric = v.get("metric", "")
    direction = v.get("direction", "decreasing")
    min_points = v.get("min_points", 3)
    arr = extract_array(sim, metric)
    if arr is None or len(arr) < min_points:
        return CheckResult(name=f"monotone:{metric}", passed=False,
                           detail=f"n={len(arr) if arr else 0} < {min_points}")
    if direction == "decreasing":
        passed = all(arr[i] >= arr[i+1] for i in range(len(arr) - 1))
    else:
        passed = all(arr[i] <= arr[i+1] for i in range(len(arr) - 1))
    return CheckResult(name=f"monotone:{metric}", passed=passed,
                       detail=f"{direction} over {len(arr)} points")


def check_count(task: dict, sim: dict | None, output_dir: Path) -> CheckResult:
    v = task["verifier"]
    expected = v.get("expected")
    if expected is None:
        return CheckResult(name="count", passed=False, detail="no expected")
    # Try a few sources in order:
    #   1. sim.curves (list)            2. sim.ber_curves (list)
    #   3. sim.ber_curves (dict-of-X)   4. sim.numerical_metrics.modulations (list)
    #   5. sim.numerical_metrics.ber_curves (list or dict)
    #   6. count of .png/.pdf files in workdir (last resort)
    actual = None
    if isinstance(sim, dict):
        if isinstance(sim.get("curves"), list):
            actual = len(sim["curves"])
        elif isinstance(sim.get("ber_curves"), list):
            actual = len(sim["ber_curves"])
        elif isinstance(sim.get("ber_curves"), dict):
            actual = len(sim["ber_curves"])
        elif isinstance(sim.get("numerical_metrics"), dict):
            nm = sim["numerical_metrics"]
            if isinstance(nm.get("modulations"), list):
                actual = len(nm["modulations"])
            elif isinstance(nm.get("ber_curves"), (list, dict)):
                actual = len(nm["ber_curves"])
    # Last resort: recursively find any list whose key mentions
    # cells/users/bs/curves/sectors/modulations — the common benchmark-task
    # "things to count" keywords.
    if actual is None and isinstance(sim, dict):
        actual = _find_list_len_anywhere(
            sim, ["cells", "users", "bs", "curves", "sectors",
                  "modulations", "rounds"])
    if actual is None:
        actual = (len(list(output_dir.rglob("*.png")))
                  + len(list(output_dir.rglob("*.pdf"))))
    return CheckResult(name="count", passed=actual == expected,
                       detail=f"actual={actual} expected={expected}")


def check_value_exact(task: dict, sim: dict | None) -> CheckResult:
    v = task["verifier"]
    metric = v.get("metric", "")
    expected = v.get("expected")
    tolerance = v.get("tolerance", 0)
    val = extract_scalar(sim, metric, v)
    if val is None or expected is None:
        return CheckResult(name=f"exact:{metric}", passed=False,
                           detail=f"val={val}")
    passed = abs(val - expected) <= tolerance
    return CheckResult(name=f"exact:{metric}", passed=passed,
                       detail=f"{val:.4g} vs {expected}±{tolerance}")


_VALID_CAUSES = {"wall_occlusion", "furniture_blockage",
                  "distance_attenuation", "interference_shadow",
                  "material_penetration_loss"}
_VALID_ACTION_TYPES = {"reposition", "reorient", "add"}


def check_action_plan_schema(task: dict, output_dir: Path) -> CheckResult:
    """Validate action_plan.json shape against result_schema_action_plan.json.
    Used by Phase T diagnosis tasks. Implements the schema's required-keys
    and enum constraints inline (avoids bringing jsonschema as a hard dep).
    """
    p = output_dir / "action_plan.json"
    if not p.exists():
        return CheckResult(name="action_plan:exists",
                           passed=False, detail="action_plan.json missing")
    try:
        plan = json.loads(p.read_text())
    except Exception as e:
        return CheckResult(name="action_plan:parse",
                           passed=False, detail=f"malformed JSON: {e}")
    required = ["coverage_current", "coverage_target", "blind_spots",
                "actions", "confidence", "stop_recommended"]
    missing = [k for k in required if k not in plan]
    if missing:
        return CheckResult(name="action_plan:required",
                           passed=False, detail=f"missing keys: {missing}")
    for i, bs in enumerate(plan.get("blind_spots", [])):
        cause = bs.get("cause")
        if cause not in _VALID_CAUSES:
            return CheckResult(
                name="action_plan:cause", passed=False,
                detail=f"blind_spot[{i}].cause={cause!r} not in {sorted(_VALID_CAUSES)}")
    for i, act in enumerate(plan.get("actions", [])):
        atype = act.get("type")
        if atype not in _VALID_ACTION_TYPES:
            return CheckResult(
                name="action_plan:action_type", passed=False,
                detail=f"actions[{i}].type={atype!r} not in {sorted(_VALID_ACTION_TYPES)}")
    conf = plan.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return CheckResult(name="action_plan:confidence",
                           passed=False, detail=f"confidence={conf} not in [0,1]")
    return CheckResult(name="action_plan:schema", passed=True,
                       detail="all schema constraints satisfied")


def check_iterative_convergence(task: dict, output_dir: Path) -> CheckResult:
    """Verify planning_state.json's history array shows iterative
    coverage improvement and meets thresholds.
    """
    v = task.get("verifier", {})
    min_iter = v.get("min_iterations", 2)
    min_imp = v.get("min_improvement", 0.10)
    p = output_dir / "planning_state.json"
    if not p.exists():
        return CheckResult(name="iterative:exists", passed=False,
                           detail="planning_state.json missing")
    try:
        state = json.loads(p.read_text())
    except Exception as e:
        return CheckResult(name="iterative:parse", passed=False,
                           detail=f"malformed: {e}")
    hist = state.get("history") or []
    if len(hist) < min_iter:
        return CheckResult(name="iterative:n_iter", passed=False,
                           detail=f"history has {len(hist)} iter, need ≥{min_iter}")
    covs = [h.get("coverage", 0.0) for h in hist]
    improvement = covs[-1] - covs[0]
    if improvement < min_imp:
        return CheckResult(name="iterative:improvement", passed=False,
                           detail=f"end-to-end improvement {improvement:.2f} < {min_imp}")
    # Near-monotonicity: allow at most 1 dip > 0.02 (2 percentage points)
    dips = sum(1 for i in range(len(covs) - 1) if covs[i + 1] < covs[i] - 0.02)
    if dips > 1:
        return CheckResult(name="iterative:monotone", passed=False,
                           detail=f"{dips} dips >2pp in coverage history")
    return CheckResult(name="iterative:convergence", passed=True,
                       detail=f"{len(hist)} iter, {covs[0]:.2f}→{covs[-1]:.2f}")


def check_composite(task: dict, output_dir: Path, sim: dict | None,
                    exec_success: bool) -> list[CheckResult]:
    """Run every subcheck in a composite verifier spec."""
    out = []
    for sub in task["verifier"].get("subchecks", []):
        sub_task = dict(task, verifier=sub)
        out.extend(run_checks(sub_task, output_dir, sim, exec_success))
    return out


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────

def run_checks(task: dict, output_dir: Path, sim: dict | None,
               exec_success: bool) -> list[CheckResult]:
    vtype = task["verifier"].get("type", "execution_ok")
    if vtype == "execution_ok":
        return [_check_execution_ok(exec_success)]
    if vtype == "file_exists":
        return _check_file_exists(task, output_dir)
    if vtype == "code_contains":
        return [check_code_contains(task, output_dir)]
    if vtype == "metric_threshold":
        return [check_metric_threshold(task, sim)]
    if vtype == "metric_range":
        return [check_metric_range(task, sim)]
    if vtype == "metric_monotone":
        return [check_metric_monotone(task, sim)]
    if vtype == "count":
        return [check_count(task, sim, output_dir)]
    if vtype == "value_exact":
        return [check_value_exact(task, sim)]
    if vtype == "composite":
        return check_composite(task, output_dir, sim, exec_success)
    if vtype == "irc_aperture":
        return [check_irc_aperture(task, output_dir)]
    if vtype == "action_plan_schema":
        return [check_action_plan_schema(task, output_dir)]
    if vtype == "iterative_convergence":
        return [check_iterative_convergence(task, output_dir)]
    return [CheckResult(name=f"unknown:{vtype}", passed=False,
                        detail=f"no dispatcher for type={vtype}")]


def verify(task: dict, output_dir: Path, exec_success: bool = True) -> VerificationResult:
    sim = load_sim_result(output_dir)
    # Always run artifact presence check if required_artifacts is specified
    required_ok = _check_file_exists(task, output_dir)
    type_checks = run_checks(task, output_dir, sim, exec_success)
    # Plausibility checks — reward-hacking defense. Any fail here marks
    # the overall trial as failed regardless of task-specific checks.
    plausibility = check_plausibility(output_dir)
    # Tier-5 domain checks — capability-specific constraints.
    capability = task.get("capability", "")
    domain = check_tier5_domain(capability, output_dir) if capability else []
    all_checks = required_ok + type_checks + plausibility + domain
    passed = sum(1 for c in all_checks if c.passed)
    total = len(all_checks)
    score = passed / total if total else 0.0
    # Plausibility failures short-circuit: a physically impossible output
    # can't be a "pass" even if the threshold check happens to match.
    plausibility_failed = any(not c.passed for c in plausibility)
    domain_failed = any(not c.passed for c in domain)
    overall_passed = score == 1.0 and not plausibility_failed and not domain_failed
    notes: list[str] = []
    if sim is None:
        notes.append("simulation_result.json not found")
    if plausibility_failed:
        notes.append("plausibility check failed — possible reward hacking")
    if domain_failed:
        notes.append(f"tier-5 domain check failed for capability={capability}")
    return VerificationResult(passed=overall_passed, score=score,
                              checks=all_checks, notes=notes)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True, help="e.g. U001")
    ap.add_argument("--tasks-file", default="benchmark/tasks/tasks.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--exec-success", type=int, default=1)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks_file).read_text())["tasks"]
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if task is None:
        raise SystemExit(f"task {args.task_id} not found")
    out = Path(args.output_dir)
    res = verify(task, out, exec_success=bool(args.exec_success))
    print(json.dumps(res.as_dict(), indent=2))
    raise SystemExit(0 if res.passed else 1)


if __name__ == "__main__":
    main()
