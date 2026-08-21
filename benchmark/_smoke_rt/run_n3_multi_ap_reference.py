"""Pre-compute N3 multi-AP reference oracles.

For each of 4 scenes we deploy 2 APs and run Sionna RT once to get the
per-AP coverage maps (cm.rss[0] for AP 0, cm.rss[1] for AP 1). The
best-server map and serving-AP assignment are derived on the fly by the
verifier from these per-AP references.

Output: benchmark/oracles/n3_multi_ap/{scene}/
  coverage_ap_0.npy     RSS in dBm (NaN where no path), AP 0
  coverage_ap_1.npy     RSS in dBm (NaN where no path), AP 1
  metadata.json          AP positions, frequency, grid shape, per-AP RSS stats
  preview.png            4-panel: AP 0 + AP 1 + best_server + serving_ap
"""
from __future__ import annotations

import json
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
        "scene":           "box_two_screens",
        "kind":            "indoor",
        "ap_positions":    [[-3, 0, 4.5], [3, 0, 4.5]],
        "frequency_hz":    5e9,
        "tx_power_dbm":    20.0,
        "cell_size":       0.25,
        "max_depth":       3,
        "samples_per_tx":  100_000,
        "vmin_dbm":        -90, "vmax_dbm": -20,
    },
    {
        "scene":           "box_one_screen",
        "kind":            "indoor",
        "ap_positions":    [[-3, 0, 4.5], [3, 0, 4.5]],
        "frequency_hz":    5e9,
        "tx_power_dbm":    20.0,
        "cell_size":       0.25,
        "max_depth":       3,
        "samples_per_tx":  100_000,
        "vmin_dbm":        -90, "vmax_dbm": -20,
    },
    {
        "scene":           "simple_street_canyon",
        "kind":            "outdoor",
        "ap_positions":    [[-40, 0, 15], [40, 0, 15]],
        "frequency_hz":    5e9,
        "tx_power_dbm":    20.0,
        "cell_size":       1.0,
        "max_depth":       2,
        "samples_per_tx":  50_000,
        "vmin_dbm":        -120, "vmax_dbm": -30,
    },
    {
        "scene":           "etoile",
        "kind":            "outdoor",
        "ap_positions":    [[-150, 0, 30], [150, 0, 30]],
        "frequency_hz":    5e9,
        "tx_power_dbm":    20.0,
        "cell_size":       2.0,
        "max_depth":       2,
        "samples_per_tx":  50_000,
        "vmin_dbm":        -130, "vmax_dbm": -30,
    },
]

OUT_ROOT = Path(__file__).resolve().parents[1] / "oracles" / "n3_multi_ap"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def compute_multi_ap(cfg: dict) -> dict:
    name = cfg["scene"]
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = rt.load_scene(getattr(rt.scene, name))
    scene.frequency = cfg["frequency_hz"]
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    for i, pos in enumerate(cfg["ap_positions"]):
        scene.add(rt.Transmitter(name=f"ap{i}", position=list(pos),
                                 power_dbm=cfg["tx_power_dbm"]))

    t0 = time.time()
    solver = rt.RadioMapSolver()
    cm = solver(scene, max_depth=cfg["max_depth"],
                cell_size=(cfg["cell_size"], cfg["cell_size"]),
                samples_per_tx=cfg["samples_per_tx"])
    elapsed = time.time() - t0
    rss_arr = cm.rss.numpy()  # shape: (num_tx, n_y, n_x), Watts
    print(f"  cm.rss shape: {rss_arr.shape}")

    # Per-AP RSS in dBm
    per_ap_dbm = []
    per_ap_stats = []
    for i in range(rss_arr.shape[0]):
        rss = rss_arr[i]
        rss_dbm = np.where(rss > 0, 10 * np.log10(np.maximum(rss, 1e-15)) + 30, np.nan)
        per_ap_dbm.append(rss_dbm)
        valid = ~np.isnan(rss_dbm)
        per_ap_stats.append({
            "ap_id":         i,
            "position":      cfg["ap_positions"][i],
            "rss_dbm_min":   float(np.nanmin(rss_dbm)) if valid.any() else None,
            "rss_dbm_max":   float(np.nanmax(rss_dbm)) if valid.any() else None,
            "rss_dbm_mean":  float(np.nanmean(rss_dbm)) if valid.any() else None,
            "valid_cells":   int(valid.sum()),
            "total_cells":   int(valid.size),
        })
        np.save(out_dir / f"coverage_ap_{i}.npy", rss_dbm)

    # Derived: best-server + serving-AP
    stacked = np.stack(per_ap_dbm, axis=0)
    best_server = np.fmax.reduce(stacked, axis=0)
    # argmax over AP axis: when both NaN, argmax returns 0 (need mask)
    any_valid = np.any(np.isfinite(stacked), axis=0)
    serving_ap = np.full(best_server.shape, -1, dtype=np.int8)
    for i in range(serving_ap.shape[0]):
        for j in range(serving_ap.shape[1]):
            col = stacked[:, i, j]
            if np.any(np.isfinite(col)):
                # max ignoring NaN
                vals = np.where(np.isfinite(col), col, -np.inf)
                serving_ap[i, j] = int(np.argmax(vals))

    np.save(out_dir / "best_server.npy", best_server)
    np.save(out_dir / "serving_ap.npy", serving_ap)

    valid_bs = ~np.isnan(best_server)
    serving_fractions = []
    for i in range(rss_arr.shape[0]):
        frac = float((serving_ap[valid_bs] == i).sum() / max(1, valid_bs.sum()))
        serving_fractions.append(round(frac, 4))

    meta = {
        "scene_name":           name,
        "kind":                 cfg["kind"],
        "ap_positions":         cfg["ap_positions"],
        "frequency_hz":         cfg["frequency_hz"],
        "tx_power_dbm":         cfg["tx_power_dbm"],
        "cell_size_m":          cfg["cell_size"],
        "max_depth":            cfg["max_depth"],
        "samples_per_tx":       cfg["samples_per_tx"],
        "grid_shape":           list(per_ap_dbm[0].shape),
        "per_ap":               per_ap_stats,
        "best_server_metrics": {
            "rss_dbm_min":  float(np.nanmin(best_server)) if valid_bs.any() else None,
            "rss_dbm_max":  float(np.nanmax(best_server)) if valid_bs.any() else None,
            "rss_dbm_mean": float(np.nanmean(best_server)) if valid_bs.any() else None,
            "valid_cells":  int(valid_bs.sum()),
            "total_cells":  int(valid_bs.size),
        },
        "serving_ap_fractions": serving_fractions,
        "solver_elapsed_s":     round(elapsed, 2),
        "sionna_version":       sionna.__version__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 4-panel preview
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    vmin = cfg["vmin_dbm"]; vmax = cfg["vmax_dbm"]
    for i, ax in enumerate(axes.flat[:2]):
        im = ax.imshow(per_ap_dbm[i], origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ap = cfg["ap_positions"][i]
        ax.set_title(f"AP {i}: pos=({ap[0]}, {ap[1]}, {ap[2]})  "
                     f"mean={per_ap_stats[i]['rss_dbm_mean']:.1f} dBm")
        plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 0]
    im = ax.imshow(best_server, origin="lower", cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_title(f"Best-server  mean={meta['best_server_metrics']['rss_dbm_mean']:.1f} dBm")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 1]
    # Serving-AP map (-1 = no signal, 0 = AP 0, 1 = AP 1)
    serv_display = serving_ap.astype(float)
    serv_display[serving_ap == -1] = np.nan
    im = ax.imshow(serv_display, origin="lower", cmap="coolwarm")
    ax.set_title(f"Serving-AP  fractions={serving_fractions}")
    plt.colorbar(im, ax=ax, fraction=0.04, ticks=[0, 1])

    plt.suptitle(f"{name}  ({cfg['kind']})  2-AP RT reference",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "preview.png", dpi=110)
    plt.close()

    return meta


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle root: {OUT_ROOT}")
    summary = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['scene']}  ({cfg['kind']}) ===")
        m = compute_multi_ap(cfg)
        a, b = m["per_ap"]
        print(f"  AP 0: mean={a['rss_dbm_mean']:.1f}  AP 1: mean={b['rss_dbm_mean']:.1f}  "
              f"best-server mean={m['best_server_metrics']['rss_dbm_mean']:.1f}  "
              f"serv_frac={m['serving_ap_fractions']}  "
              f"({m['solver_elapsed_s']:.1f}s)")
        summary.append(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
