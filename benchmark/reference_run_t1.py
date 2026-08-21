"""Validated Sionna RT reference runner for T1 (single-AP coverage).

Run with the sionna env:
    /home/myid/rs01778/miniconda3/envs/sionna/bin/python \
        benchmark/reference_run_t1.py --scene <path_to_scene_state.json> \
        --out <output_dir>

Produces:
    <output_dir>/reference_coverage_map.npy       # 2D RSS dBm grid
    <output_dir>/reference_meta.json              # config + summary stats
    <output_dir>/tmp_sionna/scene.xml + meshes/   # generated XML (kept for audit)

This is our human-validated reference. The advisor or a domain expert
must inspect this code once and confirm it is correct; subsequent runs
on different scenes treat its output as ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Silence TF logs that Sionna may emit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ─────────────────────────────────────────────────────────────────────────────
# Scene state → Mitsuba XML + PLY meshes
#   (extracted from .claude/skills/rf-simulator/templates/template_rt_coverage.py
#    so the reference runner stands alone; identical math)
# ─────────────────────────────────────────────────────────────────────────────

def _write_ply(path: Path, verts: list[list[float]]) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float u\nproperty float v\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
    with open(path, "wb") as f:
        f.write(header.encode())
        for i, v in enumerate(verts):
            f.write(struct.pack("<fffff", v[0], v[1], v[2], uvs[i][0], uvs[i][1]))
        f.write(struct.pack("<Biii", 3, 0, 1, 2))
        f.write(struct.pack("<Biii", 3, 0, 2, 3))


def export_scene_to_ply_xml(state: dict, out_dir: Path) -> Path:
    """Convert scene_state.json to Mitsuba 3 XML + PLY meshes.

    Walls / ceiling / floor get concrete or plasterboard BSDFs.
    Furniture is approximated as a top face at its height (z-extruded box).
    Materials mapped: itu_concrete→concrete, itu_plasterboard→plasterboard,
    itu_glass→glass, itu_wood→wood.

    Returns the path to scene.xml.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    bounds = state["scene"]["bounds"]
    w, l, h = bounds["width"], bounds["depth"], bounds["height"]

    shapes: list[tuple[str, str, str]] = []
    mats: set[str] = set()

    # 6 enclosing faces
    _write_ply(mesh_dir / "floor.ply",
               [[0, 0, 0], [w, 0, 0], [w, l, 0], [0, l, 0]])
    shapes.append(("floor", "meshes/floor.ply", "concrete"))
    _write_ply(mesh_dir / "ceiling.ply",
               [[0, l, h], [w, l, h], [w, 0, h], [0, 0, h]])
    shapes.append(("ceiling", "meshes/ceiling.ply", "plasterboard"))
    for name, verts in [
        ("wall_s", [[0, 0, 0], [w, 0, 0], [w, 0, h], [0, 0, h]]),
        ("wall_n", [[w, l, 0], [0, l, 0], [0, l, h], [w, l, h]]),
        ("wall_w", [[0, l, 0], [0, 0, 0], [0, 0, h], [0, l, h]]),
        ("wall_e", [[w, 0, 0], [w, l, 0], [w, l, h], [w, 0, h]]),
    ]:
        _write_ply(mesh_dir / f"{name}.ply", verts)
        shapes.append((name, f"meshes/{name}.ply", "concrete"))
    mats.update(["concrete", "plasterboard"])

    # Interior walls (multi-room scenes only)
    mat_map = {"itu_concrete": "concrete", "itu_plasterboard": "plasterboard",
               "itu_glass": "glass", "itu_wood": "wood"}
    for i, wall in enumerate(state.get("walls", []) or []):
        if not wall.get("is_interior"):
            continue
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        m = mat_map.get(wall.get("material", "itu_plasterboard"), "plasterboard")
        mats.add(m)
        _write_ply(mesh_dir / f"iw_{i}.ply",
                   [[sx, sy, 0], [ex, ey, 0], [ex, ey, h], [sx, sy, h]])
        shapes.append((f"iw_{i}", f"meshes/iw_{i}.ply", m))

    # Furniture as top-face plates (any furniture with height >= 0.3 m)
    for i, furn in enumerate(state.get("furniture", []) or []):
        if not isinstance(furn, dict):
            continue
        pos = furn.get("position") or [0, 0, 0]
        dims = furn.get("dimensions") or [1, 1, 1]
        fh = float(dims[2]) if len(dims) > 2 else 1.0
        if fh < 0.3:
            continue
        fx, fy = float(pos[0]), float(pos[1])
        fw, fd = float(dims[0]), float(dims[1])
        m = mat_map.get(furn.get("material", "itu_wood"), "wood")
        mats.add(m)
        x0, y0, x1, y1, z1 = fx - fw/2, fy - fd/2, fx + fw/2, fy + fd/2, fh
        _write_ply(mesh_dir / f"f_{i}.ply",
                   [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        shapes.append((f"f_{i}", f"meshes/f_{i}.ply", m))

    # Compose XML
    thick = {"concrete": 0.15, "plasterboard": 0.12, "glass": 0.01, "wood": 0.01}
    xml = ['<scene version="2.1.0">\n<!-- Materials -->']
    for m in sorted(mats):
        xml.append(
            f'  <bsdf type="itu-radio-material" id="{m}">'
            f'\n    <string name="type" value="{m}"/>'
            f'\n    <float name="thickness" value="{thick.get(m, 0.1)}"/>'
            f'\n  </bsdf>'
        )
    xml.append("<!-- Shapes -->")
    for sid, ply, mid in shapes:
        xml.append(
            f'  <shape type="ply" id="mesh-{sid}">'
            f'\n    <string name="filename" value="{ply}"/>'
            f'\n    <boolean name="face_normals" value="true"/>'
            f'\n    <ref id="{mid}" name="bsdf"/>'
            f'\n  </shape>'
        )
    xml.append("</scene>")
    xml_path = out_dir / "scene.xml"
    xml_path.write_text("\n".join(xml))
    return xml_path


# ─────────────────────────────────────────────────────────────────────────────
# T1 reference runner
# ─────────────────────────────────────────────────────────────────────────────

def run_t1_reference(scene_state_path: Path, ap_position: list[float],
                     frequency_hz: float, tx_power_dbm: float,
                     rx_height: float = 1.5, cell_size: float = 0.1,
                     num_samples: int = 1_000_000, max_depth: int = 5,
                     output_dir: Path | None = None) -> dict:
    """Run Sionna RT for one (scene, AP) config; return reference metrics."""
    state = json.loads(scene_state_path.read_text())
    bounds = state["scene"]["bounds"]
    W, D = float(bounds["width"]), float(bounds["depth"])

    if output_dir is None:
        output_dir = Path("benchmark/_review_demo/references")
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = output_dir / "tmp_sionna"
    xml_path = export_scene_to_ply_xml(state, xml_dir)

    # Defer the heavy imports until after XML is built (so XML build can be
    # tested in any env, only the RT call needs the sionna env).
    import sionna.rt as rt

    scene = rt.load_scene(str(xml_path))
    scene.frequency = frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    tx = rt.Transmitter("tx", position=ap_position, power_dbm=tx_power_dbm)
    scene.add(tx)

    rm = rt.RadioMapSolver()
    radio_map = rm(
        scene=scene,
        cell_size=[cell_size, cell_size],
        samples_per_tx=num_samples,
        max_depth=max_depth,
        center=[W / 2, D / 2, rx_height],
        orientation=[0.0, 0.0, 0.0],
        size=[W, D],
    )
    pg = np.array(radio_map.path_gain)
    if pg.ndim == 3:
        pg = pg[0]
    rss = tx_power_dbm + 10.0 * np.log10(np.where(pg > 0, pg, np.nan))

    threshold_dbm = float(state.get("metadata", {})
                          .get("coverage_threshold_dbm", -75))
    valid = np.isfinite(rss)
    above = valid & (rss >= threshold_dbm)
    coverage_pct = float(np.sum(above) / max(np.sum(valid), 1) * 100)

    np.save(output_dir / "reference_coverage_map.npy", rss)
    meta = {
        "task": "T1_single_ap_coverage",
        "scene_state_path": str(scene_state_path),
        "scene_bounds": {"width": W, "depth": D, "height": float(bounds["height"])},
        "ap_position": list(ap_position),
        "frequency_hz": frequency_hz,
        "tx_power_dbm": tx_power_dbm,
        "rx_height": rx_height,
        "cell_size": cell_size,
        "num_samples": num_samples,
        "max_depth": max_depth,
        "threshold_dbm": threshold_dbm,
        "grid_shape": list(rss.shape),
        "coverage_pct": coverage_pct,
        "rss_min_dbm": float(np.nanmin(rss)),
        "rss_max_dbm": float(np.nanmax(rss)),
        "rss_mean_dbm": float(np.nanmean(rss)),
        "rss_std_dbm": float(np.nanstd(rss)),
        "engine": "sionna_rt_radiomap_solver",
    }
    (output_dir / "reference_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[reference_run_t1] coverage_pct = {coverage_pct:.2f}%  "
          f"grid {rss.shape}  saved to {output_dir}/reference_coverage_map.npy")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True,
                    help="Path to scene_state.json")
    ap.add_argument("--ap-x", type=float, default=None,
                    help="AP x position (default: scene centroid)")
    ap.add_argument("--ap-y", type=float, default=None,
                    help="AP y position (default: scene centroid)")
    ap.add_argument("--ap-z", type=float, default=2.5)
    ap.add_argument("--frequency-ghz", type=float, default=5.0)
    ap.add_argument("--tx-power-dbm", type=float, default=20.0)
    ap.add_argument("--cell-size", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=int(1e6))
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--out", default="benchmark/_review_demo/references/S01")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    state = json.loads(scene_path.read_text())
    bounds = state["scene"]["bounds"]
    W, D = float(bounds["width"]), float(bounds["depth"])
    ap_x = args.ap_x if args.ap_x is not None else W / 2
    ap_y = args.ap_y if args.ap_y is not None else D / 2

    run_t1_reference(
        scene_state_path=scene_path,
        ap_position=[ap_x, ap_y, args.ap_z],
        frequency_hz=args.frequency_ghz * 1e9,
        tx_power_dbm=args.tx_power_dbm,
        cell_size=args.cell_size,
        num_samples=args.samples,
        max_depth=args.max_depth,
        output_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
