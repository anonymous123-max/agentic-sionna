"""Pre-compute N2 v2 edit-type after-state reference oracles.

Four AP-configuration edit types, each on a different scene:
  1. frequency    box_two_screens: 5 GHz → 2.4 GHz
  2. power        box_one_screen:  20 → 10 dBm
  3. position     simple_street_canyon: AP (0,0,15) → (-30,0,15)
  4. antenna      etoile: iso → 4×4 panel directional (yaw -π/2, face +x)

Before-states reuse N1 coverage oracles in benchmark/oracles/n1/.

Output: benchmark/oracles/n2_edit/{scene}_{edit_type}_after.{npy,json,png}
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna
import sionna.rt as rt


CONFIGS = [
    {
        "scene":     "box_two_screens",
        "edit_type": "frequency",
        "before":    {"freq": 5e9,   "power_dbm": 20.0, "ap": [0, 0, 4.5], "antenna": "iso"},
        "after":     {"freq": 2.4e9, "power_dbm": 20.0, "ap": [0, 0, 4.5], "antenna": "iso"},
        "cell_size": 0.25, "max_depth": 3, "samples_per_tx": 100_000,
    },
    {
        "scene":     "box_one_screen",
        "edit_type": "power",
        "before":    {"freq": 5e9,   "power_dbm": 20.0, "ap": [0, 0, 4.5], "antenna": "iso"},
        "after":     {"freq": 5e9,   "power_dbm": 10.0, "ap": [0, 0, 4.5], "antenna": "iso"},
        "cell_size": 0.25, "max_depth": 3, "samples_per_tx": 100_000,
    },
    {
        "scene":     "simple_street_canyon",
        "edit_type": "position",
        "before":    {"freq": 5e9,   "power_dbm": 20.0, "ap": [0, 0, 15],   "antenna": "iso"},
        "after":     {"freq": 5e9,   "power_dbm": 20.0, "ap": [-30, 0, 15], "antenna": "iso"},
        "cell_size": 1.0,  "max_depth": 2, "samples_per_tx": 50_000,
    },
    {
        "scene":     "etoile",
        "edit_type": "antenna",
        "before":    {"freq": 5e9,   "power_dbm": 20.0, "ap": [0, 0, 30], "antenna": "iso"},
        "after":     {"freq": 5e9,   "power_dbm": 20.0, "ap": [0, 0, 30],
                      "antenna": "panel_4x4_face_east"},
        "cell_size": 2.0,  "max_depth": 2, "samples_per_tx": 50_000,
    },
]

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "n2_edit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_arrays(scene: "rt.Scene", antenna: str):
    """Configure scene.tx_array based on antenna spec."""
    if antenna == "iso":
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                        pattern="iso", polarization="V")
    elif antenna == "panel_4x4_face_east":
        # 4x4 panel, 3GPP tr38901 element pattern, V polarization
        scene.tx_array = rt.PlanarArray(num_rows=4, num_cols=4,
                                        pattern="tr38901", polarization="V",
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5)
    else:
        raise ValueError(f"unknown antenna spec: {antenna}")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")


def _orient_for_antenna(antenna: str) -> list[float]:
    if antenna == "panel_4x4_face_east":
        return [-math.pi / 2, 0.0, 0.0]    # yaw = -90° → face +x
    return [0.0, 0.0, 0.0]


def _coverage(scene_name: str, state: dict, cfg: dict) -> tuple[np.ndarray, dict]:
    scene = rt.load_scene(getattr(rt.scene, scene_name))
    scene.frequency = state["freq"]
    _build_arrays(scene, state["antenna"])
    tx = rt.Transmitter(name="tx", position=list(state["ap"]),
                        power_dbm=state["power_dbm"])
    tx.orientation = _orient_for_antenna(state["antenna"])
    scene.add(tx)

    t0 = time.time()
    solver = rt.RadioMapSolver()
    cm = solver(scene, max_depth=cfg["max_depth"],
                cell_size=(cfg["cell_size"], cfg["cell_size"]),
                samples_per_tx=cfg["samples_per_tx"])
    rss = cm.rss[0].numpy()
    elapsed = time.time() - t0

    rss_dbm = np.where(rss > 0, 10 * np.log10(np.maximum(rss, 1e-15)) + 30, np.nan)
    valid = ~np.isnan(rss_dbm)
    meta = {
        "scene_name":     scene_name,
        "state":          state,
        "cell_size":      cfg["cell_size"],
        "max_depth":      cfg["max_depth"],
        "samples_per_tx": cfg["samples_per_tx"],
        "grid_shape":     list(rss.shape),
        "elapsed_s":      round(elapsed, 2),
        "rss_dbm_min":    float(np.nanmin(rss_dbm)) if valid.any() else None,
        "rss_dbm_max":    float(np.nanmax(rss_dbm)) if valid.any() else None,
        "rss_dbm_mean":   float(np.nanmean(rss_dbm)) if valid.any() else None,
        "valid_cells":    int(valid.sum()),
        "total_cells":    int(valid.size),
    }
    return rss_dbm, meta


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle out: {OUT_DIR}")
    summary = []
    for cfg in CONFIGS:
        scene = cfg["scene"]
        edit = cfg["edit_type"]
        print(f"\n=== {scene}  edit={edit} ===")
        rss_after, m_after = _coverage(scene, cfg["after"], cfg)
        before_n1 = (Path(__file__).resolve().parents[1] / "oracles" / "n1"
                     / f"{scene}.npy")
        if before_n1.exists():
            ref_before = np.load(before_n1)
            if ref_before.shape == rss_after.shape:
                both = np.isfinite(ref_before) & np.isfinite(rss_after)
                if both.any():
                    delta_mean = float((rss_after[both] - ref_before[both]).mean())
                    delta_std = float((rss_after[both] - ref_before[both]).std())
                else:
                    delta_mean = delta_std = float("nan")
            else:
                delta_mean = delta_std = float("nan")
        else:
            delta_mean = delta_std = float("nan")

        np.save(OUT_DIR / f"{scene}_{edit}_after.npy", rss_after)
        meta = {
            **m_after,
            "edit_type":      edit,
            "before_state":   cfg["before"],
            "after_state":    cfg["after"],
            "before_ref":     f"benchmark/oracles/n1/{scene}.npy",
            "delta_mean_db":  None if math.isnan(delta_mean) else round(delta_mean, 3),
            "delta_std_db":   None if math.isnan(delta_std)  else round(delta_std, 3),
            "sionna_version": sionna.__version__,
        }
        (OUT_DIR / f"{scene}_{edit}_after.json").write_text(
            json.dumps(meta, indent=2))

        # PNG preview
        valid = ~np.isnan(rss_after)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        vmin = {"box_two_screens": -90, "box_one_screen": -90,
                "simple_street_canyon": -120, "etoile": -130}.get(scene, -120)
        vmax = {"box_two_screens": -20, "box_one_screen": -20,
                "simple_street_canyon": -30, "etoile": -30}.get(scene, -30)
        im = ax.imshow(rss_after, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ap = cfg["after"]["ap"]
        ax.set_title(f"{scene} after {edit} edit\n"
                     f"RSS {m_after['rss_dbm_min']:.1f}..{m_after['rss_dbm_max']:.1f} "
                     f"mean={m_after['rss_dbm_mean']:.1f} | "
                     f"delta vs before: mean={delta_mean:+.2f}dB std={delta_std:.2f}dB")
        plt.colorbar(im, ax=ax, label="RSS (dBm)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{scene}_{edit}_after.png", dpi=110)
        plt.close()

        print(f"  after: shape={m_after['grid_shape']}  "
              f"RSS {m_after['rss_dbm_min']:.1f}..{m_after['rss_dbm_max']:.1f} "
              f"mean={m_after['rss_dbm_mean']:.1f}")
        print(f"  delta vs N1 before: mean={delta_mean:+.2f}dB  std={delta_std:.2f}dB"
              f"  ({m_after['elapsed_s']:.1f}s)")
        summary.append(meta)

    print("\n=== N2 edit oracle summary ===")
    print(f"{'scene':25s} {'edit':10s} {'before mean':>12s} {'after mean':>12s} {'Δ mean':>9s} {'Δ std':>8s}")
    for m in summary:
        # Re-read before mean from N1 oracle
        before_path = Path(__file__).resolve().parents[1] / "oracles" / "n1" / f"{m['scene_name']}.npy"
        ref = np.load(before_path)
        ref_mean = float(np.nanmean(ref))
        print(f"{m['scene_name']:25s} {m['edit_type']:10s} {ref_mean:>10.2f}dBm "
              f"{m['rss_dbm_mean']:>10.2f}dBm "
              f"{m['delta_mean_db']:>+7.2f}dB {m['delta_std_db']:>+7.2f}dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
