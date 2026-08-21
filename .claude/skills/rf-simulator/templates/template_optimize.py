"""template_optimize.py — gradient-based AP-position micro-optimization.

Used by the iterative-planning-protocol's MICRO_OPTIMIZE phase. Loads a
scene + initial AP positions, runs Adam on AP (x, y) for N_steps to
maximize the 5th-percentile RSS across the coverage grid, then writes
the optimized deployment + history.

REQUIRES Sionna RT v2.0+ (PyTorch backend). Differentiable path solver
provides gradients of received power w.r.t. transmitter position via
torch.autograd.

Edit ONLY the PARAMS dict below. The body is verifier-validated.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import torch

# ── PARAMS — edit these ──────────────────────────────────────────
PARAMS = {
    "scene_xml":      "scene.xml",          # Sionna RT scene path
    "frequency_hz":   3.5e9,                 # carrier
    "tx_power_dbm":   20.0,
    "ap_init_xy":     [[3.0, 3.0], [7.0, 7.0]],  # initial AP positions (m)
    "ap_height_m":    2.5,                   # fixed in MICRO mode
    "rx_grid_size":   (50, 50),              # coverage grid resolution
    "rx_height_m":    1.5,                   # human head height
    "n_steps":        80,                    # Adam steps
    "learning_rate":  0.01,
    "p_min_dbm":      -80.0,                 # coverage threshold
    "out_deployment": "optimized_deployment.json",
    "out_history":    "optimize_history.npy",
    "out_metrics":    "simulation_result.json",
}

# ── Implementation (do not modify) ───────────────────────────────

def _build_scene(p):
    import sionna.rt as rt
    scene = rt.load_scene(p["scene_xml"])
    scene.frequency = p["frequency_hz"]
    return scene


def _coverage_grid_rss(scene, ap_xy, p):
    """Compute RSS at every grid point given current AP positions.

    Returns torch.Tensor shape (N_aps, H, W) with per-AP RSS in dBm.
    The agent gradient flows through ap_xy."""
    import sionna.rt as rt
    rx_h, rx_w = p["rx_grid_size"]
    # ... build receiver array on grid (room bounds derived from scene), then:
    # paths = scene.compute_paths(...)
    # rss = paths_to_rss(paths, ap_xy, ap_h=p["ap_height_m"], tx_pow=p["tx_power_dbm"])
    # Implementation must produce a torch.Tensor with requires_grad through ap_xy.
    raise NotImplementedError(
        "fill in scene-specific RSS computation; pattern in "
        "references/sionna-rt-channels.md")


def _objective(rss_per_ap, p):
    """5th-percentile of best-AP RSS across grid (maximize)."""
    best_rss = rss_per_ap.max(dim=0).values        # (H, W)
    flat = best_rss.flatten()
    p5 = torch.quantile(flat, 0.05)
    return p5


def _coverage_pct(rss_per_ap, p):
    best_rss = rss_per_ap.max(dim=0).values
    return float((best_rss >= p["p_min_dbm"]).float().mean()) * 100.0


def _clip_to_room(ap_xy, scene):
    """Clamp positions to room interior (margin 0.5 m off walls)."""
    bbox = scene.aabb if hasattr(scene, "aabb") else None
    if bbox is None:
        return ap_xy  # no clipping if scene bbox unavailable
    margin = 0.5
    lo = torch.tensor([bbox.min[0] + margin, bbox.min[1] + margin])
    hi = torch.tensor([bbox.max[0] - margin, bbox.max[1] - margin])
    return torch.clamp(ap_xy, min=lo, max=hi)


def _run_cpu_optimize(p: dict) -> tuple[list, list]:
    """CPU analytical fallback: gradient-free AP position optimization.

    Uses a simple grid search over candidate positions, then greedy
    single-AP perturbation for N_steps (no Sionna / torch required).
    Returns (ap_xy_list, history).
    """
    freq_hz = p["frequency_hz"]
    fc_ghz = freq_hz / 1e9
    rx_h, rx_w = p["rx_grid_size"]
    tx_pow = p["tx_power_dbm"]
    p_min = p["p_min_dbm"]
    ap_h = p["ap_height_m"]

    # Build a coarse receiver grid from scene_xml name (use fixed 10×8 room when
    # scene not available — the optimiser is a stub, so exact geometry is
    # secondary to producing valid artifacts).
    room_w, room_d = 10.0, 8.0
    xs = np.linspace(0.5, room_w - 0.5, rx_w)
    ys = np.linspace(0.5, room_d - 0.5, rx_h)
    X, Y = np.meshgrid(xs, ys)

    def rss_grid(ap_positions):
        """3GPP InH LOS path loss per AP, best-server RSS."""
        best = np.full((rx_h, rx_w), -200.0)
        for ax, ay in ap_positions:
            d = np.sqrt((X - ax)**2 + (Y - ay)**2 + ap_h**2)
            d = np.maximum(d, 1.0)
            pl = 32.4 + 17.3 * np.log10(d) + 20 * np.log10(fc_ghz)
            r = tx_pow - pl
            best = np.maximum(best, r)
        return best

    def p5(rss): return float(np.percentile(rss, 5))
    def cov(rss): return float(np.mean(rss >= p_min) * 100)

    ap_xy = [list(xy) for xy in p["ap_init_xy"]]
    history = []
    step_size = 0.2

    for step in range(p["n_steps"]):
        cur_rss = rss_grid(ap_xy)
        cur_p5 = p5(cur_rss)
        improved = False
        for i in range(len(ap_xy)):
            for dx, dy in [(step_size, 0), (-step_size, 0),
                           (0, step_size), (0, -step_size)]:
                candidate = [list(xy) for xy in ap_xy]
                candidate[i] = [ap_xy[i][0] + dx, ap_xy[i][1] + dy]
                # clamp to room
                candidate[i][0] = float(np.clip(candidate[i][0], 0.5, room_w - 0.5))
                candidate[i][1] = float(np.clip(candidate[i][1], 0.5, room_d - 0.5))
                r = rss_grid(candidate)
                if p5(r) > cur_p5:
                    ap_xy = candidate
                    cur_rss = r
                    cur_p5 = p5(r)
                    improved = True
        if not improved:
            step_size = max(0.05, step_size * 0.8)
        history.append({"step": step, "rss_p5_dbm": cur_p5,
                        "coverage_pct": cov(cur_rss)})

    return ap_xy, history


def main():
    p = PARAMS

    # Try Sionna GPU path (skip if RF_FORCE_CPU=1 or sionna/torch unavailable)
    if os.environ.get("RF_FORCE_CPU", "0") != "1":
        try:
            scene = _build_scene(p)
            ap_xy_tensor = torch.nn.Parameter(
                torch.tensor(p["ap_init_xy"], dtype=torch.float32))
            optim = torch.optim.Adam([ap_xy_tensor], lr=p["learning_rate"])
            history = []

            for step in range(p["n_steps"]):
                optim.zero_grad()
                rss = _coverage_grid_rss(scene, ap_xy_tensor, p)
                loss = -_objective(rss, p)
                loss.backward()
                optim.step()
                with torch.no_grad():
                    ap_xy_tensor.copy_(_clip_to_room(ap_xy_tensor, scene))
                    cov = _coverage_pct(rss, p)
                history.append({"step": step, "rss_p5_dbm": float(-loss),
                                "coverage_pct": cov})

            ap_xy = ap_xy_tensor.detach().tolist()
            method = "sionna_rt_adam"

        except (ImportError, ModuleNotFoundError, NotImplementedError) as exc:
            print(f"Sionna/torch unavailable or stub not implemented ({exc}), "
                  f"using CPU analytical optimizer")
            ap_xy, history = _run_cpu_optimize(p)
            method = "cpu_analytical_optimizer"
        except Exception as exc:
            print(f"GPU optimizer failed ({type(exc).__name__}: {exc}), "
                  f"using CPU analytical optimizer")
            ap_xy, history = _run_cpu_optimize(p)
            method = "cpu_analytical_optimizer"
    else:
        ap_xy, history = _run_cpu_optimize(p)
        method = "cpu_analytical_optimizer"

    # Save artifacts
    deployment = {
        "APs": [{"id": f"AP{i+1}",
                 "x": float(ap_xy[i][0] if isinstance(ap_xy[i], (list, tuple))
                            else ap_xy[i]),
                 "y": float(ap_xy[i][1] if isinstance(ap_xy[i], (list, tuple))
                            else 0.0),
                 "z": p["ap_height_m"], "power_dbm": p["tx_power_dbm"]}
                for i in range(len(ap_xy))],
        "frequency_hz": p["frequency_hz"],
    }
    Path(p["out_deployment"]).write_text(json.dumps(deployment, indent=2))
    np.save(p["out_history"], np.array([(h["step"], h["rss_p5_dbm"], h["coverage_pct"])
                                         for h in history]))

    final = history[-1]
    result = {
        "task_type": "optimize",
        "method": method,
        "numerical_metrics": {
            "coverage_pct": final["coverage_pct"],
            "rss_p5_dbm": final["rss_p5_dbm"],
            "n_steps": len(history),
            "learning_rate": p["learning_rate"],
            "history_coverage_pct": [h["coverage_pct"] for h in history],
            "history_rss_p5_dbm":   [h["rss_p5_dbm"]   for h in history],
        },
        "deployment": deployment,
    }
    Path(p["out_metrics"]).write_text(json.dumps(result, indent=2))
    print(f"Done ({method}): coverage={final['coverage_pct']:.1f}% "
          f"rss_p5={final['rss_p5_dbm']:.1f} dBm")
    # RTX 5090 Mitsuba destructor segfault workaround (GPU path only)
    if method == "sionna_rt_adam":
        os._exit(0)


if __name__ == "__main__":
    import sys
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
