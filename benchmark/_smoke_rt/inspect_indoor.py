"""Inspect Sionna 2.0 built-in indoor scenes.

For each of {box, box_one_screen, box_two_screens, floor_wall}:
  - load_scene
  - print bbox, #objects, materials, mesh sizes
  - place an AP at room center (z=2.5 m), run RadioMapSolver
  - save a coverage_map_*.png so we can compare them visually

This is the input to deciding which built-in scene to use for T1/T2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna
import sionna.rt as rt

SCENES = ["box", "box_one_screen", "box_two_screens", "floor_wall"]
OUT = Path(__file__).resolve().parent / "indoor_scenes"
OUT.mkdir(exist_ok=True)


def inspect(name: str) -> dict:
    xml = getattr(rt.scene, name)
    print(f"\n=== {name}  ({Path(xml).name}) ===")
    scene = rt.load_scene(xml)
    # Scene-level bbox via mi_scene
    bb = scene.mi_scene.bbox()
    mins = np.array([float(bb.min.x), float(bb.min.y), float(bb.min.z)])
    maxs = np.array([float(bb.max.x), float(bb.max.y), float(bb.max.z)])
    mat_counts: dict[str, int] = {}
    obj_info = []
    for obj_name, obj in scene.objects.items():
        try:
            mat = obj.radio_material.name if obj.radio_material else "(none)"
            mat_counts[mat] = mat_counts.get(mat, 0) + 1
            pos = np.array(obj.position).flatten()
            obj_info.append((obj_name, mat, pos))
        except Exception as e:
            print(f"  [warn] {obj_name}: {e}")
    bbox = {"min": mins.tolist(), "max": maxs.tolist(),
            "size": (maxs - mins).tolist()}
    print(f"  bbox min: ({mins[0]:.2f}, {mins[1]:.2f}, {mins[2]:.2f}) m")
    print(f"  bbox max: ({maxs[0]:.2f}, {maxs[1]:.2f}, {maxs[2]:.2f}) m")
    print(f"  size:     {maxs[0]-mins[0]:.2f} × {maxs[1]-mins[1]:.2f} × "
          f"{maxs[2]-mins[2]:.2f} m")
    print(f"  #objects: {len(scene.objects)}")
    print(f"  materials: {mat_counts}")
    print(f"  per-object (first 8):")
    for n, m, pos in obj_info[:8]:
        print(f"    {n:40s} mat={m:20s} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    if len(obj_info) > 8:
        print(f"    ... (+{len(obj_info)-8} more)")

    # Pick AP at room center (bbox centroid x/y, z=2.5 m or 0.7*size_z)
    cx = float((mins[0] + maxs[0]) / 2)
    cy = float((mins[1] + maxs[1]) / 2)
    cz = float(mins[2] + 0.7 * (maxs[2] - mins[2]))
    print(f"  AP @ ({cx:.2f}, {cy:.2f}, {cz:.2f})")

    scene.frequency = 5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="ap", position=[cx, cy, cz], power_dbm=20.0))

    solver = rt.RadioMapSolver()
    cm = solver(scene, max_depth=2, cell_size=(0.25, 0.25),
                samples_per_tx=50_000)
    rss = cm.rss[0].numpy()
    rss_dbm = 10 * np.log10(np.maximum(rss, 1e-15)) + 30
    print(f"  coverage shape: {rss.shape}  "
          f"RSS dBm range: {rss_dbm.min():.1f} .. {rss_dbm.max():.1f}")

    # Save coverage heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rss_dbm, origin="lower",
                   extent=[mins[0], maxs[0], mins[1], maxs[1]],
                   cmap="viridis", vmin=-90, vmax=-30)
    ax.plot(cx, cy, "r*", markersize=18, label=f"AP z={cz:.1f}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name}: {maxs[0]-mins[0]:.1f} × {maxs[1]-mins[1]:.1f} m, "
                 f"{len(scene.objects)} obj")
    plt.colorbar(im, ax=ax, label="RSS (dBm)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_coverage.png", dpi=110)
    plt.close()

    # Save scene preview PNG (top-down ortho)
    return {
        "name": name,
        "bbox": bbox,
        "n_objects": len(scene.objects),
        "materials": mat_counts,
        "ap": (cx, cy, cz),
        "rss_min_dbm": float(rss_dbm.min()),
        "rss_max_dbm": float(rss_dbm.max()),
    }


def main() -> int:
    print(f"sionna {sionna.__version__}")
    results = []
    for name in SCENES:
        try:
            results.append(inspect(name))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! {name} FAILED: {e}")

    print("\n=== Summary table ===")
    print(f"{'scene':22s} {'size (m)':>22s} {'#obj':>6s} {'materials':>30s}")
    for r in results:
        sz = r["bbox"]["size"]
        mats = ",".join(f"{k}:{v}" for k, v in r["materials"].items())[:30]
        print(f"{r['name']:22s} {sz[0]:6.2f}×{sz[1]:6.2f}×{sz[2]:5.2f}"
              f"  {r['n_objects']:>4d}  {mats:>30s}")
    print(f"\nCoverage PNGs in: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
