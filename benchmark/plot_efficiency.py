"""plot_efficiency.py — token-efficiency curve (master guide P5.5).

For each ablation condition (or full vs no_skill), plot:
  X-axis: extra tokens consumed vs no_skill baseline
  Y-axis: pass-rate gain (pp) vs no_skill baseline

A condition in the upper-right is "expensive but pays off"; lower-right is
"expensive but no payoff". Master guide cites SWE-Skills-Bench finding some
skills caused 451% token overhead with zero pass-rate improvement.

Usage:
    python3 benchmark/plot_efficiency.py \\
        --archives benchmark/_studies_archive/2026-04-28_paired_*.json \\
        --output paper/figures/token_efficiency.svg
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def aggregate(path: Path) -> dict:
    """Return {condition: {"pass_rate", "mean_tokens", "n"}}."""
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        return {}
    by_cond = defaultdict(lambda: {"n": 0, "pass": 0, "tokens": 0})
    for r in rows:
        c = r.get("cond") or r.get("condition")
        if not c:
            continue
        u = r.get("usage", {}) if isinstance(r.get("usage"), dict) else {}
        tokens = (u.get("input_tokens", 0) + u.get("output_tokens", 0)
                  + u.get("cache_read_input_tokens", 0)
                  + u.get("cache_creation_input_tokens", 0))
        by_cond[c]["n"] += 1
        by_cond[c]["pass"] += int(bool(r.get("passed", False)))
        by_cond[c]["tokens"] += tokens
    out = {}
    for c, v in by_cond.items():
        out[c] = {
            "n": v["n"],
            "pass_rate": 100 * v["pass"] / max(v["n"], 1),
            "mean_tokens": v["tokens"] / max(v["n"], 1),
        }
    return out


def render_svg(points: list[dict], width: int = 700, height: int = 500) -> str:
    """points: [{label, x_extra_tokens, y_gain_pp}]."""
    if not points:
        return f'<svg width="{width}" height="{height}"><text x="20" y="40">No data</text></svg>'

    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    xs = [p["x_extra_tokens"] for p in points]
    ys = [p["y_gain_pp"] for p in points]
    x_max = max(max(xs, default=0), 100)
    x_min = min(min(xs, default=0), 0)
    y_max = max(max(ys, default=0), 5)
    y_min = min(min(ys, default=0), -5)
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    def sx(x): return margin + plot_w * (x - x_min) / x_range
    def sy(y): return height - margin - plot_h * (y - y_min) / y_range

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="13">']
    svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="15" '
               f'font-weight="bold">Token efficiency: pass-rate gain vs token overhead</text>')
    # Axes
    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
               f'y2="{height-margin}" stroke="black"/>')
    svg.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>')
    # Zero gain reference
    if y_min < 0 < y_max:
        zy = sy(0)
        svg.append(f'<line x1="{margin}" y1="{zy:.1f}" x2="{width-margin}" '
                   f'y2="{zy:.1f}" stroke="#888" stroke-dasharray="4,4"/>')
    # Labels
    svg.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle">'
               f'Extra tokens vs no-skill baseline (mean per trial)</text>')
    svg.append(f'<text x="20" y="{height/2}" text-anchor="middle" '
               f'transform="rotate(-90 20 {height/2})">Pass-rate gain (pp)</text>')
    # Points
    for p in points:
        cx, cy = sx(p["x_extra_tokens"]), sy(p["y_gain_pp"])
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#2196f3" stroke="black"/>')
        svg.append(f'<text x="{cx + 10:.1f}" y="{cy - 4:.1f}" font-size="11">{p["label"]}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--archives", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    points = []
    for a in args.archives:
        if not a.exists():
            continue
        agg = aggregate(a)
        ws = agg.get("with_skill")
        ns = agg.get("no_skill")
        if not ws or not ns:
            continue
        label = a.stem.split("_", 1)[-1]
        x = ws["mean_tokens"] - ns["mean_tokens"]
        y = ws["pass_rate"] - ns["pass_rate"]
        points.append({"label": label, "x_extra_tokens": x, "y_gain_pp": y})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(points))
    print(f"Wrote {args.output}  ({len(points)} points)")
    for p in points:
        print(f"  {p['label']}: +{p['x_extra_tokens']:.0f} tokens, {p['y_gain_pp']:+.1f}pp")


if __name__ == "__main__":
    main()
