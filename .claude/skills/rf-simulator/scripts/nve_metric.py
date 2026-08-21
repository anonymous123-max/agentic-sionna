#!/usr/bin/env python3
"""nve_metric.py — Normalized Validation Error for neural channel estimators.

NVE measures how close the agent's BLER is to the perfect-CSI BLER lower
bound. Defined per master guide Part 9 / NVlabs the-ai-telco-engineer:

    NVE = mean( BLER_agent / BLER_perfect_CSI )   (lower is better)

Baseline (LS estimator on a representative CDL channel) typically gives
NVE ≈ 94. Novel neural estimators target NVE < 60.

Usage:
    python3 nve_metric.py --agent-bler 0.13 0.07 0.03 0.01 \\
                          --perfect-bler 0.10 0.05 0.02 0.005

Or pass JSON paths:
    python3 nve_metric.py --agent-json out_agent.json \\
                          --perfect-json out_perfect.json \\
                          --agent-key numerical_metrics.bler_simulated \\
                          --perfect-key numerical_metrics.bler_simulated
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compute_nve(agent_bler: list[float], perfect_bler: list[float]) -> dict:
    if len(agent_bler) != len(perfect_bler):
        raise ValueError(
            f"agent ({len(agent_bler)} pts) and perfect ({len(perfect_bler)} pts) "
            "BLER curves must have the same length")
    if not agent_bler:
        raise ValueError("empty BLER curve")
    EPS = 1e-12  # guard against perfect = 0
    ratios = [a / max(p, EPS) for a, p in zip(agent_bler, perfect_bler)]
    nve = sum(ratios) / len(ratios)
    return {
        "nve": nve,
        "ratios": ratios,
        "n_points": len(ratios),
        "agent_bler": agent_bler,
        "perfect_bler": perfect_bler,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent-bler", nargs="+", type=float,
                    help="Agent BLER values (in order of SNR)")
    ap.add_argument("--perfect-bler", nargs="+", type=float,
                    help="Perfect-CSI BLER values (same order)")
    ap.add_argument("--agent-json", help="JSON file with agent results")
    ap.add_argument("--perfect-json", help="JSON file with perfect-CSI results")
    ap.add_argument("--agent-key", default="numerical_metrics.bler_simulated",
                    help="Dotted path to BLER list in agent JSON")
    ap.add_argument("--perfect-key", default="numerical_metrics.bler_simulated",
                    help="Dotted path to BLER list in perfect-CSI JSON")
    args = ap.parse_args()

    if args.agent_bler and args.perfect_bler:
        agent = args.agent_bler
        perfect = args.perfect_bler
    elif args.agent_json and args.perfect_json:
        agent = _get(json.loads(Path(args.agent_json).read_text()), args.agent_key)
        perfect = _get(json.loads(Path(args.perfect_json).read_text()), args.perfect_key)
        if agent is None or perfect is None:
            sys.exit(f"Could not extract BLER from JSON. "
                     f"Tried {args.agent_key} / {args.perfect_key}")
    else:
        ap.error("provide either --agent-bler+--perfect-bler OR "
                 "--agent-json+--perfect-json")

    out = compute_nve(agent, perfect)
    print(json.dumps(out, indent=2))
    print()
    print(f"NVE = {out['nve']:.2f}  ({'GOOD' if out['nve'] < 60 else 'BASELINE-LIKE' if out['nve'] < 100 else 'WORSE-THAN-LS'})")
    print("  Reference: LS baseline ≈ 94, novel-estimator target < 60.")


if __name__ == "__main__":
    main()
