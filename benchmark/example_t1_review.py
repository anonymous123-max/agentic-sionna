"""One-scene example: T1 (single-AP coverage) review pipeline.

Loads agent's coverage_map.npy for S01, loads VALIDATED Sionna RT
reference coverage_map (pre-computed by benchmark/reference_run_t1.py),
resamples to a common grid, and emits a one-page review card with:
  - scene info
  - agent's heatmap
  - reference heatmap (Sionna RT validated)
  - pixel-wise diff
  - similarity metrics
  - auto verdict + expert sign-off slot
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRIAL_DIR = Path("benchmark/results/tc30_c1_train_v6/with_skill/TC1_S01/t1")
REF_DIR = Path("benchmark/_review_demo/references/S01")
OUT_DIR = Path("benchmark/_review_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resample_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resample a 2D dBm grid to `target_shape`. NaN-safe.

    Operates on the dB scale directly (acceptable for visual comparison and
    coarse similarity metrics; for strict reference matching, re-run sionna
    at the same cell_size as the agent's grid instead).
    """
    from scipy.ndimage import zoom
    src = np.where(np.isnan(src), -150.0, src)  # placeholder for NaN
    zy = target_shape[0] / src.shape[0]
    zx = target_shape[1] / src.shape[1]
    out = zoom(src, (zy, zx), order=1)
    return out


def similarity_metrics(agent: np.ndarray, ref: np.ndarray, threshold_dbm=-75):
    """Compute side-by-side comparison metrics."""
    diff = agent - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    within_3 = float((np.abs(diff) <= 3).mean() * 100)
    within_5 = float((np.abs(diff) <= 5).mean() * 100)
    agent_cov = float((agent > threshold_dbm).mean() * 100)
    ref_cov = float((ref > threshold_dbm).mean() * 100)
    return {
        "rmse_dbm": rmse, "mae_dbm": mae, "max_abs_dbm": max_abs,
        "pct_within_3dBm": within_3, "pct_within_5dBm": within_5,
        "agent_coverage_pct": agent_cov, "ref_coverage_pct": ref_cov,
        "coverage_delta_pp": agent_cov - ref_cov,
    }


def verdict(m: dict) -> tuple[bool, str]:
    """Pass criteria for T1."""
    rules = [
        (m["rmse_dbm"] <= 8, f"RMSE {m['rmse_dbm']:.1f} ≤ 8 dBm"),
        (m["pct_within_3dBm"] >= 70, f"≥ 70% within ±3 dBm (got {m['pct_within_3dBm']:.1f}%)"),
        (abs(m["coverage_delta_pp"]) <= 10, f"|coverage Δ| ≤ 10 pp (got {m['coverage_delta_pp']:+.1f})"),
        (m["agent_coverage_pct"] <= m["ref_coverage_pct"] + 5,
         f"agent ≤ ref + 5 (energy conservation: {m['agent_coverage_pct']:.1f} ≤ {m['ref_coverage_pct']+5:.1f})"),
    ]
    passed = all(r for r, _ in rules)
    detail = "\n".join(f"  {'✓' if r else '✗'} {d}" for r, d in rules)
    return passed, detail


def main():
    scene = json.loads((TRIAL_DIR / "scene_state.json").read_text())
    sim = json.loads((TRIAL_DIR / "simulation_result.json").read_text())
    agent_map = np.load(TRIAL_DIR / "coverage_map.npy")

    bounds = scene["scene"]["bounds"]
    W, D = bounds["width"], bounds["depth"]
    ap = scene["access_points"][0]
    ap_pos = ap["position"]
    freq = ap["frequency_hz"]
    tx_power = ap["power_dbm"]
    threshold = scene.get("metadata", {}).get("coverage_threshold_dbm", -75)

    # Validated Sionna RT reference (pre-computed by reference_run_t1.py)
    ref_meta_path = REF_DIR / "reference_meta.json"
    ref_map_path = REF_DIR / "reference_coverage_map.npy"
    if not ref_map_path.exists():
        raise SystemExit(
            f"Reference not found at {ref_map_path}. Run:\n"
            f"  /home/myid/rs01778/miniconda3/envs/sionna/bin/python "
            f"benchmark/reference_run_t1.py --scene {TRIAL_DIR}/scene_state.json "
            f"--ap-x {ap_pos[0]} --ap-y {ap_pos[1]} --ap-z {ap_pos[2]} "
            f"--out {REF_DIR}"
        )
    ref_raw = np.load(ref_map_path)
    ref_meta = json.loads(ref_meta_path.read_text())

    # Resample reference to match agent's grid for pixel-wise diff
    ref_map = resample_to_match(ref_raw, agent_map.shape)

    m = similarity_metrics(agent_map, ref_map, threshold_dbm=threshold)
    passed, detail = verdict(m)

    # ─── Plot review card (3-panel + scene preview + verdict text) ───────
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1.2, 0.8, 1.2],
                          hspace=0.45, wspace=0.25)

    # Header
    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.02, 0.85, "T1 — Single-AP Coverage Review",
              fontsize=15, fontweight="bold")
    ax_h.text(0.02, 0.55,
              f"Scene: TC1_S01 (home office, {W:.0f}×{D:.0f} m)  |  "
              f"AP: {ap_pos} @ {freq/1e9:.1f} GHz, {tx_power:.0f} dBm",
              fontsize=10)
    ax_h.text(0.02, 0.30,
              f"Prompt (verbatim): \"Generate a 5×4 m home office..., place 1 AP at the centroid at 2.5 m, "
              f"compute coverage at 5.0 GHz, threshold −75 dBm. Report coverage_pct.\"",
              fontsize=8, style="italic", wrap=True)
    ax_h.text(0.02, 0.05,
              f"Reference: validated Sionna RT (RadioMapSolver, cell={ref_meta['cell_size']}m, "
              f"samples={ref_meta['num_samples']:.0e}, max_depth={ref_meta['max_depth']}, "
              f"grid {ref_meta['grid_shape'][0]}×{ref_meta['grid_shape'][1]} → resampled to agent grid)",
              fontsize=7, color="gray")

    # 3-panel comparison: agent | reference | diff
    vmin, vmax = -100, -20
    extent = (0, W, 0, D)

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(agent_map, origin="lower", extent=extent,
                     cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")
    ax1.plot(ap_pos[0], ap_pos[1], "w^", markersize=10, markeredgecolor="black")
    ax1.set_title(f"Agent (Sionna RT)\ncov={m['agent_coverage_pct']:.1f}%",
                  fontsize=10)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    fig.colorbar(im1, ax=ax1, label="RSS (dBm)", fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(ref_map, origin="lower", extent=extent,
                     cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")
    ax2.plot(ap_pos[0], ap_pos[1], "w^", markersize=10, markeredgecolor="black")
    ax2.set_title(f"Reference (Sionna RT)\ncov={m['ref_coverage_pct']:.1f}%", fontsize=10)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 2])
    diff = agent_map - ref_map
    im3 = ax3.imshow(diff, origin="lower", extent=extent, cmap="RdBu_r",
                     vmin=-10, vmax=10, aspect="equal")
    ax3.set_title(f"Agent − Reference\nRMSE={m['rmse_dbm']:.1f} dBm", fontsize=10)
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
    fig.colorbar(im3, ax=ax3, label="ΔdBm", fraction=0.046, pad=0.04)

    # Metrics block
    ax_m = fig.add_subplot(gs[2, :]); ax_m.axis("off")
    ax_m.text(0.02, 0.85, "Similarity metrics", fontsize=11, fontweight="bold")
    metrics_text = (
        f"  RMSE: {m['rmse_dbm']:.2f} dBm     "
        f"MAE: {m['mae_dbm']:.2f} dBm     "
        f"Max |Δ|: {m['max_abs_dbm']:.2f} dBm\n"
        f"  Pixels within ±3 dBm: {m['pct_within_3dBm']:.1f}%   "
        f"within ±5 dBm: {m['pct_within_5dBm']:.1f}%\n"
        f"  Coverage Δ (agent − ref): {m['coverage_delta_pp']:+.2f} pp"
    )
    ax_m.text(0.02, 0.55, metrics_text, fontsize=9, family="monospace")

    # Verdict
    ax_v = fig.add_subplot(gs[3, :]); ax_v.axis("off")
    verdict_color = "green" if passed else "red"
    ax_v.text(0.02, 0.92, f"Auto verdict: {'PASS' if passed else 'FAIL'}",
              fontsize=13, fontweight="bold", color=verdict_color)
    ax_v.text(0.02, 0.70, detail, fontsize=9, family="monospace")
    ax_v.text(0.02, 0.18, "Expert review:    [ ] Pass    [ ] Fail",
              fontsize=11)
    ax_v.text(0.02, 0.05, "Notes: ____________________________________________________________________________",
              fontsize=10)

    out_path = OUT_DIR / "T1_S01_review_card.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Also dump JSON for later aggregation
    (OUT_DIR / "T1_S01_metrics.json").write_text(json.dumps(
        {"task_id": "T1_S01", "passed": passed, "metrics": m,
         "ap_pos": ap_pos, "freq_hz": freq, "tx_power_dbm": tx_power,
         "agent_reported_coverage_pct": sim["numerical_metrics"]["coverage_pct"]},
        indent=2,
    ))

    print(f"Wrote {out_path}")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
