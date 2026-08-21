"""gepa_description.py — Pareto-front optimization of the skill description.

Master guide Part 7. The description field is the triggering mechanism;
two competing objectives:
  1. Trigger rate    — should-trigger queries that DO trigger
  2. Specificity     — should-not-trigger queries that DO NOT trigger

We don't have a real Claude Code triggering decision API at hand, so this
script approximates trigger using cosine-similarity-style keyword overlap
against the description. For paper-quality, swap `_keyword_match()` for
an actual LLM trigger decision.

Usage:
    python3 benchmark/gepa_description.py score                    # score current desc
    python3 benchmark/gepa_description.py score --variants benchmark/_studies_archive/desc_variants.json
    python3 benchmark/gepa_description.py pareto --variants ...   # rank Pareto-optimal
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Trigger-eval queries (10 should-trigger + 10 should-not-trigger).
# Hand-curated for the rf-simulator skill.
TRIGGER_QUERIES = {
    "should_trigger": [
        "Compute coverage map for an indoor 8x6 m office at 60 GHz",
        "Plot BER vs Eb/N0 for 16-QAM with LDPC rate=0.5 over CDL-A",
        "Place 3 access points to maximize WiFi coverage in this room",
        "Train a neural channel estimator and compare to LS baseline",
        "What's the spectral efficiency for 2x2 MIMO OFDM at SNR=20 dB",
        "Simulate STAR-RIS with 100 elements; report sum-SNR",
        "Multi-cell interference: compute SINR distribution over 7 cells",
        "Channel charting using PCA on raw CSI from UMi",
        "Compute Doppler shift for a vehicle at 50 km/h, fc=3.5 GHz",
        "How much CP do I need for an OFDM system over 100ns delay spread",
    ],
    "should_not_trigger": [
        "Build me a chrome extension to summarize Twitter threads",
        "Write a SQL migration to add a unique index on user.email",
        "Help me debug a memory leak in my Rust async code",
        "What's the best React state management library in 2026?",
        "Write a poem about an autumn forest",
        "Translate this Spanish paragraph to English",
        "How do I configure pytest fixtures for parametrized tests",
        "Implement gradient descent from scratch in numpy",
        "Compare AWS Lambda vs Cloudflare Workers for cold start",
        "Generate a unit test for this Rust struct",
    ],
}


# ────────────────────────────────────────────────────────────────────
# Cheap trigger heuristic — keyword overlap
# ────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(s.lower()) if len(t) > 2}


def _keyword_match(description: str, query: str, threshold: int = 2) -> bool:
    """Heuristic: trigger if `threshold`+ keyword tokens from `description`
    appear in `query`. Approximates Claude Code's semantic-match decision.

    For paper-quality, replace this with an actual model call:
        claude(prompt=f"Should the rf-simulator skill (description: '{description}') "
                       f"trigger on this user query: '{query}'? Answer Yes/No.")
    """
    desc_tokens = _tokens(description)
    query_tokens = _tokens(query)
    return len(desc_tokens & query_tokens) >= threshold


def score(description: str) -> dict:
    """Return trigger rate, specificity, and per-query breakdown."""
    pos_hits = [_keyword_match(description, q) for q in TRIGGER_QUERIES["should_trigger"]]
    neg_hits = [_keyword_match(description, q) for q in TRIGGER_QUERIES["should_not_trigger"]]
    trigger_rate = sum(pos_hits) / len(pos_hits)
    specificity = 1 - sum(neg_hits) / len(neg_hits)
    return {
        "trigger_rate": round(trigger_rate, 3),
        "specificity": round(specificity, 3),
        "f1": round(2 * trigger_rate * specificity / max(trigger_rate + specificity, 1e-9), 3),
        "false_positives": [q for q, h in zip(TRIGGER_QUERIES["should_not_trigger"], neg_hits) if h],
        "false_negatives": [q for q, h in zip(TRIGGER_QUERIES["should_trigger"], pos_hits) if not h],
    }


def current_description() -> str:
    """Extract the description: field from SKILL.md frontmatter."""
    text = (ROOT / ".claude/skills/rf-simulator/SKILL.md").read_text()
    m = re.search(r"description:\s*>\s*\n((?:  .+\n)+)", text)
    if m:
        return " ".join(m.group(1).split())
    return ""


# ────────────────────────────────────────────────────────────────────
# Pareto-front ranking
# ────────────────────────────────────────────────────────────────────

def pareto_front(variants: list[dict]) -> list[dict]:
    """variants: list of {name, description, score} dicts. Returns
    non-dominated subset (no other variant has both higher trigger rate
    AND higher specificity)."""
    front = []
    for v in variants:
        dominated = False
        for w in variants:
            if w is v:
                continue
            if (w["score"]["trigger_rate"] >= v["score"]["trigger_rate"]
                and w["score"]["specificity"] >= v["score"]["specificity"]
                and (w["score"]["trigger_rate"] > v["score"]["trigger_rate"]
                     or w["score"]["specificity"] > v["score"]["specificity"])):
                dominated = True
                break
        if not dominated:
            front.append(v)
    return front


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sc = sub.add_parser("score", help="Score current or supplied descriptions")
    p_sc.add_argument("--variants", type=Path, default=None,
                     help="JSON file with {name: description, ...} variants")
    p_pa = sub.add_parser("pareto", help="Rank Pareto-optimal variants")
    p_pa.add_argument("--variants", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "score":
        if args.variants:
            variants = json.loads(args.variants.read_text())
            for name, desc in variants.items():
                s = score(desc)
                print(f"\n=== {name} ===")
                print(json.dumps(s, indent=2))
        else:
            current = current_description()
            print(f"Current description ({len(current.split())} words):")
            print(f"  {current[:200]}...")
            print(f"\nScore: {json.dumps(score(current), indent=2)}")
    elif args.cmd == "pareto":
        variants = json.loads(args.variants.read_text())
        scored = [{"name": n, "description": d, "score": score(d)}
                  for n, d in variants.items()]
        front = pareto_front(scored)
        print(f"=== Pareto front ({len(front)} of {len(scored)}) ===")
        for v in sorted(front, key=lambda x: x["score"]["f1"], reverse=True):
            print(f"  {v['name']}: trigger={v['score']['trigger_rate']} "
                  f"specificity={v['score']['specificity']} f1={v['score']['f1']}")


if __name__ == "__main__":
    main()
