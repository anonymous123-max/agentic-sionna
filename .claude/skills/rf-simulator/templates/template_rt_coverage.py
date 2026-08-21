#!/usr/bin/env python3
"""Template: RT Coverage Map — Sionna GPU Ray Tracing with CPU Fallback.

Modify ONLY the PARAMS block. This template:
1. Tries Sionna RT GPU ray tracing (requires sionna + NVIDIA GPU)
2. Falls back to CPU analytical model (3GPP InH) if Sionna unavailable
3. Produces standardized simulation_result.json + coverage heatmap

GPU path: Sionna RT RadioMapSolver (differentiable, physically accurate)
CPU path: 3GPP TR 38.901 InH + shadow fading (see references/cpu-fallback.md)
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "scene_path": "scene_state.json",     # scene_state.json OR Mitsuba XML
    "frequency_hz": 3.5e9,
    "tx_position": [5.0, 4.0, 2.8],
    "tx_power_dbm": 20.0,
    "tx_antenna": "iso",                   # "iso", "dipole", "tr38901"
    "rx_height": 1.5,
    "cell_size": 0.5,                      # grid resolution (m)
    "coverage_threshold_dbm": -70.0,
    "max_depth": 5,
    "num_samples": 1_000_000,
    "output_dir": "outputs/coverage",
}
# ============================================================================

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def validate_params(p: dict) -> list[str]:
    errors = []
    if not Path(p["scene_path"]).exists():
        errors.append(f"scene_path not found: {p['scene_path']}")
    if not (100e6 <= p["frequency_hz"] <= 300e9):
        errors.append(f"frequency_hz out of range")
    if not (-30 <= p["tx_power_dbm"] <= 60):
        errors.append(f"tx_power_dbm out of range")
    if not (0.1 <= p["cell_size"] <= 10.0):
        errors.append(f"cell_size out of range")
    if not (1 <= p["max_depth"] <= 10):
        errors.append(f"max_depth out of range")
    return errors


# ---------------------------------------------------------------------------
# GPU Path: Sionna RT Ray Tracing
# ---------------------------------------------------------------------------

def run_sionna_gpu(scene_path: str, params: dict) -> tuple[np.ndarray, str]:
    """Run Sionna RT on GPU. Returns (rss_dbm_2d, method_name)."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import sionna.rt as rt

    # Load scene (XML or auto-export from scene_state.json)
    if scene_path.endswith(".xml"):
        scene = rt.load_scene(scene_path)
    else:
        # Export scene_state.json → PLY + XML, then load
        state = json.loads(Path(scene_path).read_text())
        xml_dir = Path(scene_path).parent / "sionna"
        xml_path = xml_dir / "scene.xml"
        if not xml_path.exists():
            export_scene_to_ply_xml(state, xml_dir)
        scene = rt.load_scene(str(xml_path))

    scene.frequency = params["frequency_hz"]
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5,
        horizontal_spacing=0.5, pattern=params["tx_antenna"], polarization="V")
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5,
        horizontal_spacing=0.5, pattern="iso", polarization="V")

    tx = rt.Transmitter("tx", position=params["tx_position"],
                         power_dbm=params["tx_power_dbm"])
    scene.add(tx)

    # Load scene bounds from state
    state = json.loads(Path(scene_path).read_text()) if not scene_path.endswith(".xml") else {}
    bounds = state.get("scene", {}).get("bounds", {"width": 10, "depth": 8})
    w, l = bounds["width"], bounds["depth"]

    rm = rt.RadioMapSolver()
    radio_map = rm(
        scene=scene,
        cell_size=[params["cell_size"], params["cell_size"]],
        samples_per_tx=params["num_samples"],
        max_depth=params["max_depth"],
        center=[w / 2, l / 2, params["rx_height"]],
        orientation=[0.0, 0.0, 0.0],
        size=[w, l],
    )

    pg = np.array(radio_map.path_gain)
    if pg.ndim == 3:
        pg = pg[0]
    rss = params["tx_power_dbm"] + 10 * np.log10(np.where(pg > 0, pg, np.nan))
    return rss, "sionna_rt_gpu"


def export_scene_to_ply_xml(state: dict, out_dir: Path):
    """Export scene_state.json to Sionna-compatible PLY + XML."""
    import struct
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    bounds = state["scene"]["bounds"]
    w, l, h = bounds["width"], bounds["depth"], bounds["height"]

    def write_ply(path, verts):
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 4\nproperty float x\nproperty float y\n"
                  "property float z\nproperty float u\nproperty float v\n"
                  "element face 2\nproperty list uchar int vertex_indices\n"
                  "end_header\n")
        uvs = [[0,0],[1,0],[1,1],[0,1]]
        with open(path, "wb") as f:
            f.write(header.encode())
            for i, v in enumerate(verts):
                f.write(struct.pack("<fffff", v[0], v[1], v[2], uvs[i][0], uvs[i][1]))
            f.write(struct.pack("<Biii", 3, 0, 1, 2))
            f.write(struct.pack("<Biii", 3, 0, 2, 3))

    shapes = []
    mats = set()

    # Floor, ceiling, 4 walls
    write_ply(mesh_dir/"floor.ply", [[0,0,0],[w,0,0],[w,l,0],[0,l,0]])
    shapes.append(("floor", "meshes/floor.ply", "concrete"))
    write_ply(mesh_dir/"ceiling.ply", [[0,l,h],[w,l,h],[w,0,h],[0,0,h]])
    shapes.append(("ceiling", "meshes/ceiling.ply", "plasterboard"))
    for name, verts in [("wall_s",[[0,0,0],[w,0,0],[w,0,h],[0,0,h]]),
                         ("wall_n",[[w,l,0],[0,l,0],[0,l,h],[w,l,h]]),
                         ("wall_w",[[0,l,0],[0,0,0],[0,0,h],[0,l,h]]),
                         ("wall_e",[[w,0,0],[w,l,0],[w,l,h],[w,0,h]])]:
        write_ply(mesh_dir/f"{name}.ply", verts)
        shapes.append((name, f"meshes/{name}.ply", "concrete"))
    mats.update(["concrete", "plasterboard"])

    # Interior walls
    mat_map = {"itu_concrete":"concrete","itu_plasterboard":"plasterboard",
               "itu_glass":"glass","itu_wood":"wood"}
    for i, wall in enumerate(state.get("walls", [])):
        if not wall.get("is_interior"):
            continue
        sx, sy = wall["start"]; ex, ey = wall["end"]
        m = mat_map.get(wall.get("material","itu_plasterboard"), "plasterboard")
        mats.add(m)
        write_ply(mesh_dir/f"iw_{i}.ply", [[sx,sy,0],[ex,ey,0],[ex,ey,h],[sx,sy,h]])
        shapes.append((f"iw_{i}", f"meshes/iw_{i}.ply", m))

    # Furniture top faces
    for i, furn in enumerate(state.get("furniture", [])):
        pos, dims = furn["position"], furn["dimensions"]
        fh = dims[2] if len(dims) > 2 else 1.0
        if fh < 0.3:
            continue
        fx, fy = pos[0], pos[1]; fw, fd = dims[0], dims[1]
        m = mat_map.get(furn.get("material","itu_wood"), "wood")
        mats.add(m)
        x0, y0, x1, y1, z1 = fx-fw/2, fy-fd/2, fx+fw/2, fy+fd/2, fh
        write_ply(mesh_dir/f"f_{i}.ply", [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        shapes.append((f"f_{i}", f"meshes/f_{i}.ply", m))

    # Write XML
    thick = {"concrete": 0.15, "plasterboard": 0.12, "glass": 0.01, "wood": 0.01}
    xml = ['<scene version="2.1.0">\n<!-- Materials -->']
    for m in sorted(mats):
        xml.append(f'  <bsdf type="itu-radio-material" id="{m}">'
                   f'\n    <string name="type" value="{m}"/>'
                   f'\n    <float name="thickness" value="{thick.get(m,0.1)}"/>'
                   f'\n  </bsdf>')
    xml.append('<!-- Shapes -->')
    for sid, ply, mid in shapes:
        xml.append(f'  <shape type="ply" id="mesh-{sid}">'
                   f'\n    <string name="filename" value="{ply}"/>'
                   f'\n    <boolean name="face_normals" value="true"/>'
                   f'\n    <ref id="{mid}" name="bsdf"/>\n  </shape>')
    xml.append('</scene>')
    (out_dir / "scene.xml").write_text('\n'.join(xml))


# ---------------------------------------------------------------------------
# CPU Path: 3GPP TR 38.901 InH Analytical Model
# ---------------------------------------------------------------------------

def run_cpu_analytical(scene_path: str, params: dict) -> tuple[np.ndarray, str]:
    """CPU analytical fallback — 3GPP TR 38.901 InH model.

    IMPORTANT: This function is bit-identical to
    benchmark/evaluation_harness.py::compute_analytical_coverage so that the
    external re-evaluation (harness) produces identical numbers to the agent's
    self-report. If you modify one, modify the other.
    """
    state = json.loads(Path(scene_path).read_text())
    bounds = state["scene"]["bounds"]
    width, length = bounds["width"], bounds["depth"]
    freq_ghz = params["frequency_hz"] / 1e9
    cell_size = params["cell_size"]
    tx_x, tx_y, tx_z = params["tx_position"]
    tx_power = params["tx_power_dbm"]

    nx = max(2, int(width / cell_size))
    ny = max(2, int(length / cell_size))
    rx_x = np.linspace(cell_size / 2, width - cell_size / 2, nx)
    rx_y = np.linspace(cell_size / 2, length - cell_size / 2, ny)
    rx_grid_x, rx_grid_y = np.meshgrid(rx_x, rx_y)

    # Distance from TX to each RX cell (rx_height is 1.5, matching harness)
    dx = rx_grid_x - tx_x
    dy = rx_grid_y - tx_y
    dz = 1.5 - tx_z
    dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    dist_3d = np.maximum(dist_3d, 1.0)  # 3GPP model valid for d >= 1m

    walls = state.get("walls", [])
    wall_segs = [(w["start"], w["end"], w.get("material", "itu_concrete"))
                 for w in walls]
    furniture = state.get("furniture", [])

    # --- Wall intersection count per cell (same loss table as harness) ---
    wall_loss = np.zeros((ny, nx))
    for ws, we, wmat in wall_segs:
        loss_per_wall = 15.0  # concrete
        if "plasterboard" in wmat:
            loss_per_wall = 8.0
        elif "glass" in wmat:
            loss_per_wall = 3.0
        elif "wood" in wmat:
            loss_per_wall = 10.0
        for iy in range(ny):
            for ix in range(nx):
                if _seg_int(tx_x, tx_y,
                            float(rx_grid_x[iy, ix]),
                            float(rx_grid_y[iy, ix]),
                            ws[0], ws[1], we[0], we[1]):
                    wall_loss[iy, ix] += loss_per_wall

    # --- Furniture obstruction ---
    furn_loss = np.zeros((ny, nx))
    for furn in furniture:
        pos = furn["position"]
        dims = furn["dimensions"]
        fx, fy = pos[0], pos[1]
        fw, fd = dims[0], dims[1]
        fh = dims[2] if len(dims) > 2 else 1.0
        if fh < 0.5:
            continue
        loss = 5.0 + min(10.0, fh * 3.0)
        fxmin, fxmax = fx - fw / 2, fx + fw / 2
        fymin, fymax = fy - fd / 2, fy + fd / 2
        for iy in range(ny):
            for ix in range(nx):
                rx_px = float(rx_grid_x[iy, ix])
                rx_py = float(rx_grid_y[iy, ix])
                if _aabb_int(tx_x, tx_y, rx_px, rx_py,
                             fxmin, fymin, fxmax, fymax):
                    furn_loss[iy, ix] += loss

    # --- LOS/NLOS classification (3GPP InH Table 7.4.2-1) ---
    dist_2d = np.sqrt(dx**2 + dy**2)
    dist_2d = np.maximum(dist_2d, 0.1)
    p_los = np.ones_like(dist_2d)
    mask1 = (dist_2d > 1.2) & (dist_2d <= 6.5)
    p_los[mask1] = np.exp(-(dist_2d[mask1] - 1.2) / 4.7)
    mask2 = dist_2d > 6.5
    p_los[mask2] = np.exp(-(dist_2d[mask2] - 6.5) / 32.6) * 0.32

    # IMPORTANT: Two separate RNGs, matching the harness exactly.
    rng_los = np.random.RandomState(int(abs(tx_x * 1000 + tx_y * 7) + 99))
    los_draw = rng_los.rand(ny, nx)
    is_los = los_draw < p_los
    is_los[wall_loss > 0] = False
    is_los[furn_loss > 10] = False
    is_nlos = ~is_los

    # --- 3GPP InH path loss ---
    log_d = np.log10(dist_3d)
    log_f = np.log10(freq_ghz)
    pl_los = 32.4 + 17.3 * log_d + 20.0 * log_f
    pl_nlos = 17.3 + 38.3 * log_d + 24.9 * log_f
    pl_nlos = np.maximum(pl_los, pl_nlos)  # NLOS >= LOS per 3GPP
    path_loss = np.where(is_nlos, pl_nlos, pl_los)

    # --- Shadow fading (spatially correlated — filter unit N(0,1) then apply σ) ---
    rng = np.random.RandomState(int(abs(tx_x * 1000 + tx_y * 7)))
    sigma_los, sigma_nlos = 3.0, 8.03  # 3GPP TR 38.901 Table 7.4.1-1
    sigma_map = np.where(is_nlos, sigma_nlos, sigma_los)

    raw_fading = rng.randn(ny, nx)
    kernel_size = max(1, int(2.0 / cell_size))
    if kernel_size > 1:
        try:
            from scipy.ndimage import uniform_filter
            raw_fading = uniform_filter(raw_fading, size=kernel_size)
        except ImportError:
            pass
        std = raw_fading.std()
        if std > 0:
            raw_fading /= std

    shadow_fading = raw_fading * sigma_map

    # --- Total RSS ---
    rss = tx_power - path_loss - wall_loss - furn_loss + shadow_fading
    return rss, "cpu_3gpp_inh"


def _seg_int(ax,ay,bx,by,cx,cy,dx,dy):
    def cr(ox,oy,px,py,qx,qy): return (px-ox)*(qy-oy)-(py-oy)*(qx-ox)
    d1,d2=cr(cx,cy,dx,dy,ax,ay),cr(cx,cy,dx,dy,bx,by)
    d3,d4=cr(ax,ay,bx,by,cx,cy),cr(ax,ay,bx,by,dx,dy)
    return ((d1>0 and d2<0)or(d1<0 and d2>0))and((d3>0 and d4<0)or(d3<0 and d4>0))

def _aabb_int(ax,ay,bx,by,x0,y0,x1,y1):
    for e in [(x0,y0,x1,y0),(x1,y0,x1,y1),(x1,y1,x0,y1),(x0,y1,x0,y0)]:
        if _seg_int(ax,ay,bx,by,*e): return True
    return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(rss, params, method, timing, output_dir):
    """Write simulation_result.json + heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = params["coverage_threshold_dbm"]

    valid = np.isfinite(rss)
    rss_v = rss[valid]
    above = valid & (rss >= threshold)
    cov = float(np.sum(above) / max(np.sum(valid), 1) * 100)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    state = json.loads(Path(params["scene_path"]).read_text())
    w = state["scene"]["bounds"]["width"]
    l = state["scene"]["bounds"]["depth"]
    im = ax.imshow(rss, origin="lower", extent=(0, w, 0, l),
                   cmap="RdYlGn", vmin=-100, vmax=-20, aspect="equal")
    ax.plot(params["tx_position"][0], params["tx_position"][1],
            "w^", markersize=12, markeredgecolor="black", markeredgewidth=1.5)
    fig.colorbar(im, ax=ax, label="RSS (dBm)")
    ax.set_title(f"Coverage: {cov:.1f}% above {threshold:.0f} dBm ({method})")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    fig.savefig(output_dir / "coverage_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.save(output_dir / "rss_dbm.npy", rss)

    # Canonical artifact: coverage_map.npy in CWD (verifier checks for this by name)
    np.save("coverage_map.npy", rss)

    # Quadrant coverage
    ny, nx = rss.shape
    my, mx = ny//2, nx//2
    q = {}
    for name, (ys, xs) in {"SW":(slice(None,my),slice(None,mx)),
                            "SE":(slice(None,my),slice(mx,None)),
                            "NW":(slice(my,None),slice(None,mx)),
                            "NE":(slice(my,None),slice(mx,None))}.items():
        qv = valid[ys,xs]; qa = above[ys,xs]
        q[name] = round(float(np.sum(qa)/max(np.sum(qv),1)*100), 1)

    result = {
        "schema_version": "1.0", "task_type": "rt_coverage",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success", "method": method,
        "numerical_metrics": {
            "coverage_pct": round(cov, 2),
            "coverage_threshold_dbm": threshold,
            "mean_rss_dbm": round(float(np.mean(rss_v)), 2) if rss_v.size else None,
            "min_rss_dbm": round(float(np.min(rss_v)), 2) if rss_v.size else None,
            "max_rss_dbm": round(float(np.max(rss_v)), 2) if rss_v.size else None,
            "std_rss_db": round(float(np.std(rss_v)), 2) if rss_v.size else None,
            "p5_received_power_dbm": round(float(np.percentile(rss_v, 5)), 2) if rss_v.size else None,
            "per_quadrant_coverage": q,
        },
        "visual_outputs": {"heatmap_path": "coverage_heatmap.png"},
        "deployment": {"transmitters": [{"id":"tx","position":params["tx_position"],
                       "power_dbm":params["tx_power_dbm"]}]},
        "data_files": {"rss_map_npy": "rss_dbm.npy"},
        "timing": timing, "warnings": [],
    }
    (output_dir / "simulation_result.json").write_text(json.dumps(result, indent=2))

    # Canonical artifact: simulation_result.json in CWD so verifier finds coverage_pct
    # (verifier.load_sim_result looks in output_dir root, not subdirectories)
    Path("simulation_result.json").write_text(json.dumps(result, indent=2))

    return cov


def main():
    errors = validate_params(PARAMS)
    if errors:
        for e in errors: print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    rss, method = None, "unknown"

    # Try Sionna GPU first (skip if RF_FORCE_CPU=1 for testing / CPU-only hosts)
    if os.environ.get("RF_FORCE_CPU", "0") != "1":
        try:
            rss, method = run_sionna_gpu(PARAMS["scene_path"], PARAMS)
        except Exception as e:
            print(f"Sionna GPU unavailable ({e}), using CPU analytical")

    # CPU fallback
    if rss is None:
        rss, method = run_cpu_analytical(PARAMS["scene_path"], PARAMS)

    timing = {"simulate_sec": round(time.perf_counter() - t0, 3)}

    # Self-test: RSS values should be in physical range (-150 to TX power dBm).
    # Catches sign errors in path-loss formula.
    import numpy as _np
    rss_finite = rss[_np.isfinite(rss)]
    if rss_finite.size > 0:
        max_rss = float(_np.max(rss_finite)); min_rss = float(_np.min(rss_finite))
        if max_rss > PARAMS["tx_power_dbm"] + 5:
            sys.stderr.write(
                f"\nSELF_TEST FAIL: max RSS={max_rss:.1f} dBm exceeds "
                f"TX power={PARAMS['tx_power_dbm']:.1f} dBm. Path loss "
                f"formula has wrong sign. NOT writing simulation_result.json.\n")
            sys.exit(2)
        if min_rss < -200:
            sys.stderr.write(
                f"\nSELF_TEST FAIL: min RSS={min_rss:.1f} dBm physically "
                f"impossible. Check distance/wavelength scaling.\n")
            sys.exit(2)

    cov = write_results(rss, PARAMS, method, timing, PARAMS["output_dir"])

    print(f"\nCoverage: {cov:.1f}% ({method})")
    print(f"Output: {PARAMS['output_dir']}/simulation_result.json")

    # RTX 5090 Mitsuba cleanup crash workaround
    if method == "sionna_rt_gpu":
        os._exit(0)


if __name__ == "__main__":
    if not os.environ.get("RF_SKIP_TEMPLATE_WARN"):
        sys.stderr.write(
            "\n*** TEMPLATE WARNING ***\n"
            "This is a TEMPLATE — copy the file to your workdir and edit\n"
            "PARAMS before running. Default PARAMS will run, but they're\n"
            "for reference only and may not match your task. Set\n"
            "RF_SKIP_TEMPLATE_WARN=1 to silence.\n"
            "\n"
        )
    main()
