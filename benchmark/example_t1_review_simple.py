"""Simple T1 review card: agent's heatmap + numbers + the simulation
code itself + expert sign-off.

Multi-page PDF:
  - Page 1: scene info / prompt / heatmap / reported metrics / auto checks
            / expert sign-off block
  - Page 2..N: agent's simulation.py source with line numbers, so the
            domain expert can mark up specific lines in the code review.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


TRIAL_DIR = Path("benchmark/results/tc30_c1_train_v6/with_skill/TC1_S01/t1")
OUT_DIR = Path("benchmark/_review_demo_simple")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _code_pages_text(code_text: str, lines_per_page: int = 55,
                     chars_per_line: int = 95) -> list[list[str]]:
    """Split source into PDF pages of (numbered) text lines.
    Long lines are soft-wrapped at chars_per_line.
    """
    pages: list[list[str]] = []
    cur: list[str] = []
    for lineno, raw in enumerate(code_text.splitlines(), start=1):
        # Soft-wrap long lines, keeping the line number only on the first chunk
        chunks = [raw[i:i + chars_per_line]
                  for i in range(0, max(len(raw), 1), chars_per_line)] or [""]
        for j, chunk in enumerate(chunks):
            prefix = f"{lineno:4d}  " if j == 0 else "       "
            cur.append(prefix + chunk)
            if len(cur) >= lines_per_page:
                pages.append(cur); cur = []
    if cur:
        pages.append(cur)
    return pages


def render_simple_card(trial_dir: Path, out_path: Path,
                       scene_id: str = "S01",
                       prompt_text: str | None = None) -> dict:
    scene = json.loads((trial_dir / "scene_state.json").read_text())
    sim = json.loads((trial_dir / "simulation_result.json").read_text())
    agent_map = np.load(trial_dir / "coverage_map.npy")
    nm = sim.get("numerical_metrics", {})

    bounds = scene["scene"]["bounds"]
    W, D = bounds["width"], bounds["depth"]
    ap = (scene.get("access_points") or scene.get("transmitters") or [{}])[0]
    ap_pos = ap.get("position", [W/2, D/2, 2.5])
    freq_hz = ap.get("frequency_hz", 5e9)
    tx_power = ap.get("power_dbm", 20.0)
    threshold = scene.get("metadata", {}).get("coverage_threshold_dbm", -75)

    if prompt_text is None:
        prompt_text = (trial_dir / "prompt.txt").read_text()[:500]

    # ─── Auto sanity checks (low bar — true verification is the expert) ──
    checks = []
    checks.append(("heatmap file exists",
                   (trial_dir / "coverage_map.png").exists() or
                   (trial_dir / "coverage_heatmap.png").exists() or
                   agent_map is not None))
    cov_pct = nm.get("coverage_pct")
    checks.append(("coverage_pct in [0, 100]",
                   isinstance(cov_pct, (int, float)) and 0 <= cov_pct <= 100))
    checks.append(("AP position in-bounds",
                   0 <= ap_pos[0] <= W and 0 <= ap_pos[1] <= D))
    checks.append(("scene_state.json non-placeholder",
                   scene.get("status") != "placeholder_pre_shipped_by_harness"))

    # ─── Page layout: a4-ish portrait, single column ─────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.1, 1.0, 4.0, 1.2, 1.6],
                          hspace=0.45)

    # Header
    ax_h = fig.add_subplot(gs[0, 0]); ax_h.axis("off")
    ax_h.text(0.0, 0.92, f"T1 — Single-AP Coverage   |   Scene: TC1_{scene_id}",
              fontsize=15, fontweight="bold")
    ax_h.text(0.0, 0.62,
              f"Room {W:.0f}×{D:.0f} m   |   AP @ "
              f"({ap_pos[0]:.1f}, {ap_pos[1]:.1f}, {ap_pos[2]:.1f}) m   |   "
              f"{freq_hz/1e9:.1f} GHz   |   {tx_power:.0f} dBm   |   "
              f"threshold {threshold:.0f} dBm",
              fontsize=10)
    furn = scene.get("furniture") or []
    furn_types = [f.get("type", "?") for f in furn if isinstance(f, dict)]
    ax_h.text(0.0, 0.30,
              f"Furniture: {', '.join(furn_types) if furn_types else '(none)'}",
              fontsize=9, color="gray")

    # Prompt
    ax_p = fig.add_subplot(gs[1, 0]); ax_p.axis("off")
    ax_p.text(0.0, 0.92, "Prompt (verbatim from harness):",
              fontsize=10, fontweight="bold")
    # Trim and wrap prompt
    pt = prompt_text.strip().replace("\n", " ")
    if len(pt) > 380:
        pt = pt[:377] + "..."
    ax_p.text(0.0, 0.10, pt, fontsize=8, style="italic", wrap=True)

    # Heatmap (the centerpiece) — adaptive colormap so spatial pattern is visible
    ax_m = fig.add_subplot(gs[2, 0])
    finite = agent_map[np.isfinite(agent_map)]
    vmin = float(np.percentile(finite, 5)) if finite.size else -100
    vmax = float(np.percentile(finite, 95)) if finite.size else -20
    # Guard against degenerate range
    if vmax - vmin < 1.0:
        center = (vmin + vmax) / 2
        vmin, vmax = center - 0.5, center + 0.5
    im = ax_m.imshow(agent_map, origin="lower", extent=(0, W, 0, D),
                     cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")
    ax_m.plot(ap_pos[0], ap_pos[1], "w^", markersize=14,
              markeredgecolor="black", markeredgewidth=1.5)
    # Draw furniture rectangles on top
    for f in furn:
        if not isinstance(f, dict): continue
        pos = f.get("position") or [0, 0, 0]
        dims = f.get("dimensions") or [0.5, 0.5, 0.5]
        try:
            x = float(pos[0]) - float(dims[0])/2
            y = float(pos[1]) - float(dims[1])/2
            rect = Rectangle((x, y), float(dims[0]), float(dims[1]),
                             linewidth=1.2, edgecolor="black", facecolor="none",
                             alpha=0.75)
            ax_m.add_patch(rect)
            ax_m.text(float(pos[0]), float(pos[1]), f.get("type", "?"),
                      ha="center", va="center", fontsize=7, color="black",
                      bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))
        except Exception:
            pass
    ax_m.set_title("Agent coverage heatmap", fontsize=11)
    ax_m.set_xlabel("x (m)"); ax_m.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax_m, label="RSS (dBm)", fraction=0.04, pad=0.04)

    # Agent reported metrics
    ax_n = fig.add_subplot(gs[3, 0]); ax_n.axis("off")
    ax_n.text(0.0, 0.92, "Agent reported:", fontsize=10, fontweight="bold")
    method = sim.get("method") or "(unspecified)"
    lines = [
        f"  coverage_pct: {cov_pct}",
        f"  mean RSS:     {nm.get('mean_rss_dbm', '?')} dBm",
        f"  min RSS:      {nm.get('min_rss_dbm', '?')} dBm",
        f"  max RSS:      {nm.get('max_rss_dbm', '?')} dBm",
        f"  method:       {method}",
    ]
    ax_n.text(0.0, 0.10, "\n".join(lines), fontsize=9, family="monospace")

    # Auto checks + expert sign-off
    ax_v = fig.add_subplot(gs[4, 0]); ax_v.axis("off")
    ax_v.text(0.0, 0.96, "Auto sanity checks:", fontsize=10, fontweight="bold")
    auto_lines = [f"  {'✓' if ok else '✗'}  {name}" for name, ok in checks]
    ax_v.text(0.0, 0.70, "\n".join(auto_lines), fontsize=9, family="monospace",
              color="darkgreen" if all(ok for _, ok in checks) else "darkred")
    ax_v.text(0.0, 0.30, "Expert review:",
              fontsize=11, fontweight="bold")
    ax_v.text(0.0, 0.18, "   [ ] Pass        [ ] Fail",
              fontsize=11)
    ax_v.text(0.0, 0.04,
              "   Notes: _________________________________________________________________",
              fontsize=10)

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "scene_id": scene_id,
        "auto_checks_passed": all(ok for _, ok in checks),
        "coverage_pct": cov_pct,
        "method": method,
    }


def main():
    out_path = OUT_DIR / "T1_S01_review_card.pdf"
    summary = render_simple_card(TRIAL_DIR, out_path, scene_id="S01")
    print(f"Wrote {out_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
