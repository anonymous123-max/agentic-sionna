"""Smoke test: build Sionna RT scene from S01/scene_state.json and run a coverage map.

Goal: prove the v10 fix chain works end-to-end before re-running the benchmark:
  1. conda env's python can `import sionna`
  2. scene_gen exporter writes a Mitsuba 3.0 XML
  3. sionna.rt.load_scene() consumes the XML
  4. PathSolver / RadioMapSolver produces a coverage map
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".claude" / "skills" / "rf-simulator"))

from lib.scene_gen.exporters import export_all
from lib.scene_gen.models import (
    BoundingBox, FurnitureItem, Position, Room, Scene, Transmitter,
)


def build_scene_from_state(state: dict) -> Scene:
    sc = state["scene"]
    bounds = sc["bounds"]
    width = float(bounds["width"])
    length = float(bounds["depth"])
    height = float(bounds.get("height", 3.0))

    furniture: list[FurnitureItem] = []
    for f in state.get("furniture", []):
        pos = f["position"]
        dim = f["dimensions"]
        cx = float(pos[0]) + float(dim[0]) / 2.0
        cy = float(pos[1]) + float(dim[1]) / 2.0
        furniture.append(FurnitureItem(
            id=f.get("id", f.get("label", "item")),
            category=f.get("type", "generic"),
            model_id="aabb_box",
            model_path="",
            position=Position(x=cx, y=cy, theta=0.0),
            dimensions=BoundingBox(width=float(dim[0]),
                                   depth=float(dim[1]),
                                   height=float(dim[2])),
        ))

    room = Room(width=width, length=length, height=height, furniture=furniture)

    transmitters: list[Transmitter] = []
    for ap in state.get("access_points", []):
        p = ap["position"]
        transmitters.append(Transmitter(
            id=ap.get("id", "tx0"),
            name=ap.get("label", ap.get("id", "tx0")),
            position=(float(p[0]), float(p[1]), float(p[2])),
            power_dbm=float(ap.get("power_dbm", 20.0)),
            frequency_hz=float(ap.get("frequency_hz", 3.5e9)),
        ))

    freq = float(state.get("metadata", {}).get(
        "frequency_hz",
        state.get("access_points", [{}])[0].get("frequency_hz", 3.5e9),
    ))
    return Scene(room=room, transmitters=transmitters, frequency_hz=freq)


def main() -> int:
    state_path = ROOT / "benchmark" / "_review_dataset" / "S01" / "scene_state.json"
    state = json.loads(state_path.read_text())
    print(f"[1/4] Loaded scene_state.json: {state['scene']['name']}")

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    scene_obj = build_scene_from_state(state)
    print(f"[2/4] Built Scene: room {scene_obj.room.width}x"
          f"{scene_obj.room.length}x{scene_obj.room.height} m, "
          f"{len(scene_obj.room.furniture)} furniture, "
          f"{len(scene_obj.transmitters)} TX")

    paths = export_all(scene_obj, out_dir)
    xml_path = paths["xml"]
    xml_size = xml_path.stat().st_size
    print(f"[3/4] Wrote Mitsuba XML: {xml_path} ({xml_size} bytes)")
    head = xml_path.read_text()[:200].replace("\n", " ")
    print(f"      head: {head}...")

    import sionna  # noqa: F401
    import sionna.rt as rt
    print(f"[4/4] sionna version: {sionna.__version__}")

    sionna_scene = rt.load_scene(str(xml_path))
    print(f"      load_scene OK: objects={len(sionna_scene.objects)}")

    # Place TX
    tx = scene_obj.transmitters[0]
    sionna_scene.frequency = float(scene_obj.frequency_hz)
    sionna_scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sionna_scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sionna_scene.add(rt.Transmitter(
        name="ap0", position=list(tx.position), power_dbm=tx.power_dbm))

    # Coverage map
    solver = rt.RadioMapSolver()
    cm = solver(sionna_scene,
                max_depth=2,
                cell_size=(0.5, 0.5),
                samples_per_tx=10_000)
    rss = cm.rss[0].numpy()  # tx 0
    print(f"      RadioMap OK: shape={rss.shape}, "
          f"min={rss.min():.3e} W, max={rss.max():.3e} W")

    # Save coverage map as PNG (the visual proof RT actually ran)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rss_dbm = 10 * np.log10(np.maximum(rss, 1e-15)) + 30  # W → dBm
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rss_dbm, origin="lower",
                   extent=[0, scene_obj.room.width, 0, scene_obj.room.length],
                   cmap="viridis", vmin=-90, vmax=-30)
    tx_pos = scene_obj.transmitters[0].position
    ax.plot(tx_pos[0], tx_pos[1], "r*", markersize=18, label=f"AP @ z={tx_pos[2]} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"S01 — Sionna RT coverage (5 GHz, max_depth=2, "
                 f"10k samples/tx)\n"
                 f"RSS range: {rss_dbm.min():.1f} to {rss_dbm.max():.1f} dBm")
    plt.colorbar(im, ax=ax, label="RSS (dBm)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    cov_png = out_dir / "coverage_map.png"
    plt.savefig(cov_png, dpi=120)
    plt.close()
    np.save(out_dir / "coverage_map.npy", rss_dbm)
    print(f"      Saved coverage_map.png ({cov_png.stat().st_size} bytes)")

    print("\nSMOKE TEST: PASS")
    print(f"  XML artifact:   {xml_path}")
    print(f"  Coverage shape: {rss.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
