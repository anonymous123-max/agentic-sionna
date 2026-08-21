"""Protocol-adherence analyzer for completed benchmark trials.

The skill's SKILL.md prescribes a 4-step protocol:
    1. RESTATE  — first assistant turn restates task type + key params
    2. ROUTE    — agent consults lookup.py OR references a template/script
    3. EXECUTE  — agent actually runs python (Bash python3 ...) on real code
    4. VERIFY   — agent runs verify_output.py OR writes simulation_result.json
                  with non-placeholder content

This script scans completed trial stdouts under benchmark/results/<label>/...
and reports per-trial which steps were performed, plus aggregate stats per
condition.

Usage:
    python3 benchmark/analysis/protocol_adherence.py \\
        --results-dir benchmark/results \\
        --label-prefix train_Qwen_Qwen3_6_27B_chunk
    # Or analyze a single label:
    python3 benchmark/analysis/protocol_adherence.py \\
        --label-dir benchmark/results/train_Qwen_Qwen3_6_27B_chunk0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict


# Patterns for each protocol step. Conservative — we'd rather miss than
# falsely-attribute, so the resulting numbers underestimate adherence.
_RESTATE_HINTS = re.compile(
    r"\b(BER|BLER|coverage|scene|MIMO|OFDM|LDPC|Polar|"
    r"ray tracing|RT|CDL|TDL|RIS|neural|"
    r"Eb/N0|SNR|QPSK|BPSK|QAM|"
    r"frequency|bandwidth|antenna|placement|simulation)\b",
    re.IGNORECASE,
)

_LOOKUP_RE = re.compile(r"lookup\.py\b")
_TEMPLATE_RE = re.compile(
    r"templates?/template_\w+\.py|"
    r"\$RF_SKILL_DIR/templates|"
    r"\bcp\b\s+\S*templates?/")
_SCRIPT_RE = re.compile(
    r"\$RF_SKILL_DIR/scripts/\w+\.py|"
    r"scripts/run_ber_analytical\.py|"
    r"scripts/verify_output\.py")

_BASH_PYTHON_RE = re.compile(r"python3?\s+\S+\.py|python3?\s+-c\b|python3?\s+<<")
_VERIFY_RE = re.compile(r"verify_output\.py")
_SIM_RESULT_WRITE_RE = re.compile(
    r"simulation_result\.json.*(?:json\.dump|json\.dumps|write_text|"
    r"open\(.*[\"\']w[\"\'])"
)


@dataclass
class Adherence:
    task_id: str
    condition: str
    chunk: str
    restate: bool = False
    route: bool = False
    execute: bool = False
    verify: bool = False
    restate_turn: int = -1
    route_turn: int = -1
    execute_turn: int = -1
    verify_turn: int = -1
    n_assistant_turns: int = 0
    n_tool_use: int = 0
    passed: bool = False
    score: float = 0.0
    sim_status: str = ""

    @property
    def step_count(self) -> int:
        return sum([self.restate, self.route, self.execute, self.verify])


def _iter_assistant_turns(stdout_path: Path):
    """Yield (turn_idx, text_blocks, tool_use_blocks) from a stream-json stdout.txt."""
    if not stdout_path.exists():
        return
    turn_idx = 0
    with stdout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "assistant":
                continue
            content = (ev.get("message") or {}).get("content") or []
            text_parts = [c.get("text", "") for c in content
                          if isinstance(c, dict) and c.get("type") == "text"]
            tool_calls = [c for c in content
                          if isinstance(c, dict) and c.get("type") == "tool_use"]
            yield turn_idx, text_parts, tool_calls
            turn_idx += 1


def _bash_command(tool_call: dict) -> str:
    """Extract the Bash command string from a tool_use block (any tool).
    Returns "" if the call isn't a Bash invocation or has no command."""
    if tool_call.get("name") != "Bash":
        return ""
    return str((tool_call.get("input") or {}).get("command", ""))


def analyze_trial(result_json: Path) -> Adherence | None:
    """Parse one trial's stdout + result and return an Adherence record."""
    trial_dir = result_json.parent
    stdout = trial_dir / "stdout.txt"
    sim_json = trial_dir / "simulation_result.json"
    task_id = trial_dir.parent.name
    condition = trial_dir.parent.parent.name
    chunk = trial_dir.parent.parent.parent.name
    a = Adherence(task_id=task_id, condition=condition, chunk=chunk)
    try:
        rj = json.loads(result_json.read_text())
        # P0.2: prefer pass_strict (added 2026-05-07) which requires BOTH
        # exec_success AND verification.passed. Fall back to plain
        # verification.passed for old result.json files written before the
        # field was introduced.
        a.passed = bool(rj.get("pass_strict",
                               rj.get("verification", {}).get("passed", False)))
        a.score = float(rj.get("verification", {}).get("score", 0.0) or 0.0)
        a.n_assistant_turns = int(rj.get("usage", {}).get("num_turns", 0) or 0)
    except Exception:
        return None
    if sim_json.exists():
        try:
            sj = json.loads(sim_json.read_text())
            a.sim_status = sj.get("status", "")
        except Exception:
            pass
    n_tool = 0
    for turn_idx, texts, tools in _iter_assistant_turns(stdout):
        n_tool += len(tools)
        joined_text = " ".join(texts)
        joined_cmds = " ".join(_bash_command(t) for t in tools)
        # RESTATE: first turn that mentions a domain term
        if not a.restate and turn_idx <= 1:
            if joined_text and _RESTATE_HINTS.search(joined_text):
                a.restate = True
                a.restate_turn = turn_idx
        # ROUTE: lookup.py reference, template ref, or script ref
        if not a.route:
            blob = joined_text + " " + joined_cmds
            if (_LOOKUP_RE.search(blob)
                    or _TEMPLATE_RE.search(blob)
                    or _SCRIPT_RE.search(blob)):
                a.route = True
                a.route_turn = turn_idx
        # EXECUTE: actually runs python on a script (not just listing files)
        if not a.execute and joined_cmds:
            if _BASH_PYTHON_RE.search(joined_cmds):
                a.execute = True
                a.execute_turn = turn_idx
        # VERIFY: invoked verify_output.py OR wrote simulation_result.json
        if not a.verify:
            blob = joined_text + " " + joined_cmds
            if _VERIFY_RE.search(blob) or _SIM_RESULT_WRITE_RE.search(blob):
                a.verify = True
                a.verify_turn = turn_idx
    a.n_tool_use = n_tool
    return a


def aggregate(records: list[Adherence]) -> dict:
    """Roll up adherence stats by condition + chunk."""
    by_cond: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "restate": 0, "route": 0, "execute": 0, "verify": 0,
        "all_4": 0, "n_pass": 0,
        "step_counts": Counter(),
    })
    for r in records:
        b = by_cond[r.condition]
        b["n"] += 1
        b["restate"] += int(r.restate)
        b["route"] += int(r.route)
        b["execute"] += int(r.execute)
        b["verify"] += int(r.verify)
        b["all_4"] += int(r.step_count == 4)
        b["n_pass"] += int(r.passed)
        b["step_counts"][r.step_count] += 1
    out = {}
    for cond, b in by_cond.items():
        n = b["n"]
        if n == 0:
            continue
        out[cond] = {
            "n": n,
            "n_pass": b["n_pass"],
            "pass_rate_pct": round(100 * b["n_pass"] / n, 1),
            "restate_pct": round(100 * b["restate"] / n, 1),
            "route_pct": round(100 * b["route"] / n, 1),
            "execute_pct": round(100 * b["execute"] / n, 1),
            "verify_pct": round(100 * b["verify"] / n, 1),
            "all_4_pct": round(100 * b["all_4"] / n, 1),
            "step_count_dist": dict(b["step_counts"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--results-dir", help="Root results dir (will glob "
                   "<root>/<label-prefix>*).")
    g.add_argument("--label-dir", help="Single label dir to analyze.")
    ap.add_argument("--label-prefix", default="",
                    help="Filter results-dir glob by label prefix.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of human-readable text.")
    ap.add_argument("--per-trial", action="store_true",
                    help="Print per-trial breakdown.")
    args = ap.parse_args()

    if args.label_dir:
        label_dirs = [Path(args.label_dir)]
    else:
        root = Path(args.results_dir)
        label_dirs = sorted(p for p in root.glob(f"{args.label_prefix}*")
                            if p.is_dir())

    records: list[Adherence] = []
    for ld in label_dirs:
        for result in sorted(ld.glob("*/*/t*/result.json")):
            r = analyze_trial(result)
            if r is not None:
                records.append(r)

    if not records:
        print("# No trials found", file=sys.stderr)
        return 1

    agg = aggregate(records)

    if args.json:
        out = {
            "n_trials": len(records),
            "aggregate": agg,
            "per_trial": [asdict(r) for r in records] if args.per_trial else [],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Analyzed {len(records)} trials\n")
        print(f"{'condition':<14} {'n':>4} {'pass%':>6} "
              f"{'restate':>8} {'route':>6} {'exec':>5} {'verify':>7} "
              f"{'all4':>5}")
        print("-" * 70)
        for cond in sorted(agg.keys()):
            a = agg[cond]
            print(f"{cond:<14} {a['n']:>4} {a['pass_rate_pct']:>5.1f}% "
                  f"{a['restate_pct']:>7.1f}% {a['route_pct']:>5.1f}% "
                  f"{a['execute_pct']:>4.1f}% {a['verify_pct']:>6.1f}% "
                  f"{a['all_4_pct']:>4.1f}%")
        if args.per_trial:
            print("\nPer-trial detail:")
            for r in records:
                steps = "".join("✓" if s else "·" for s in
                                [r.restate, r.route, r.execute, r.verify])
                print(f"  {r.chunk}/{r.condition}/{r.task_id} "
                      f"steps={steps} pass={r.passed} score={r.score:.2f} "
                      f"turns={r.n_assistant_turns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
