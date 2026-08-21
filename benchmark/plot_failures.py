"""plot_failures.py — failure-mode distribution figure (master guide P5.3).

Aggregates classified failures across one or more archive JSONs, then
emits a stacked-bar SVG showing per-condition × per-failure-class
proportions. SVG only — no matplotlib dependency.

Usage:
    python3 benchmark/plot_failures.py \\
        --archives benchmark/_studies_archive/2026-04-28_paired_qwen3_6_v4.json \\
        --output paper/figures/failure_modes_qwen3_6.svg

    # Multiple models on one figure:
    python3 benchmark/plot_failures.py \\
        --archives 2026-04-28_paired_*.json \\
        --output paper/figures/failure_modes_combined.svg
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from distill_failures import classify_failure  # noqa: E402

# Stable colors per failure class (master guide Part 9 ordering)
COLORS = {
    "pass":                       "#4caf50",
    "skill_not_consulted":        "#ff9800",
    "skill_consulted_ignored":    "#f44336",
    "skill_content_wrong":        "#9c27b0",
    "skill_gap":                  "#2196f3",
    "model_capability_ceiling":   "#607d8b",
    "environment_error":          "#795548",
    "reward_hacking":             "#000000",
}

CLASS_ORDER = list(COLORS.keys())


def classify_archive(path: Path) -> dict[str, dict[str, int]]:
    """Return {condition: {class: count}}."""
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        return {}
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        cond = r.get("cond") or r.get("condition", "?")
        cls = classify_failure(
            failed_checks=r.get("failed_checks", []),
            stdout=r.get("agent_final_text", ""),
            stderr=r.get("agent_final_text", ""),
            code="",
            agent_final_text=r.get("agent_final_text", ""),
            passed=bool(r.get("passed", False)),
        )
        out[cond][cls] += 1
    return {k: dict(v) for k, v in out.items()}


def render_svg(data: dict[str, dict[str, dict[str, int]]],
                width: int = 800, height: int = 480) -> str:
    """data: {bar_label: {class: count}} — flat one level. Each bar
    label could be 'qwen3.6/with_skill', 'qwen3.6/no_skill', etc.

    Renders a stacked horizontal bar per bar_label, normalized to 100%."""
    bar_labels = list(data.keys())
    margin_left = 200
    margin_right = 60
    margin_top = 40
    margin_bottom = 40
    plot_w = width - margin_left - margin_right
    n = max(len(bar_labels), 1)
    bar_h = (height - margin_top - margin_bottom) / n - 8
    bar_h = max(bar_h, 16)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="13">']
    # Title
    svg.append(f'<text x="{width/2}" y="20" text-anchor="middle" '
               f'font-size="15" font-weight="bold">Failure-mode distribution by condition</text>')
    # Bars
    for i, lbl in enumerate(bar_labels):
        y = margin_top + i * (bar_h + 8) + 8
        counts = data[lbl]
        total = sum(counts.values()) or 1
        x = margin_left
        for cls in CLASS_ORDER:
            c = counts.get(cls, 0)
            if c == 0:
                continue
            seg_w = plot_w * c / total
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{seg_w:.1f}" '
                       f'height="{bar_h:.1f}" fill="{COLORS[cls]}"/>')
            if seg_w > 28:
                svg.append(f'<text x="{x + seg_w/2:.1f}" y="{y + bar_h/2 + 4:.1f}" '
                           f'text-anchor="middle" fill="white" font-size="11">'
                           f'{c}</text>')
            x += seg_w
        svg.append(f'<text x="{margin_left - 8}" y="{y + bar_h/2 + 4:.1f}" '
                   f'text-anchor="end">{lbl}</text>')
    # Legend
    leg_y = height - margin_bottom + 14
    leg_x = margin_left
    for cls in CLASS_ORDER:
        sw = 12
        svg.append(f'<rect x="{leg_x}" y="{leg_y - 10}" width="{sw}" height="{sw}" fill="{COLORS[cls]}"/>')
        svg.append(f'<text x="{leg_x + sw + 4}" y="{leg_y}">{cls}</text>')
        leg_x += sw + 4 + 8 * len(cls) + 16
        if leg_x > width - 100:
            leg_y += 18
            leg_x = margin_left
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--archives", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    flat: dict[str, dict[str, int]] = {}
    for p in args.archives:
        if not p.exists():
            print(f"  MISSING {p}", file=sys.stderr)
            continue
        # Use filename stem (minus date prefix) as bar label
        stem = p.stem.split("_", 1)[-1] if "_" in p.stem else p.stem
        per_cond = classify_archive(p)
        for cond, counts in per_cond.items():
            flat[f"{stem}/{cond}"] = counts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(flat))
    print(f"Wrote {args.output}  ({len(flat)} bars)")


if __name__ == "__main__":
    main()
