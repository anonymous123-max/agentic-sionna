"""run_anthropic_harness.py — alt benchmark runner using raw Anthropic API.

Master guide Part 9: "test on ≥ 2 harnesses". Our primary harness is
OpenClaude/Claude Code via openclaude CLI. This runner exercises the
SAME tasks via the raw Anthropic SDK with explicit tool-use loop, so
we can measure harness-effects on skill performance.

Limitations vs Claude Code:
  - No automatic skill auto-discovery from .claude/skills/. We inject
    SKILL.md into the system prompt manually.
  - No bundled Bash/Read tool runtime. We provide minimal Bash + Read
    tool implementations that execute against the trial workdir.
  - No skeleton pre-ship — we replicate trial.py's pre_ship_skeleton.
  - No retry hook — manual retries handled by the inner loop.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python3 benchmark/run_anthropic_harness.py \\
        --label paired_anthropic_v18 \\
        --model claude-sonnet-4-6 \\
        --conditions with_skill no_skill \\
        --workers 4

Results land alongside OpenClaude runs in benchmark/results/<label>/, so
the same aggregator + iteration_log work without modification.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.verifier import verify  # noqa: E402
from benchmark.trial import pre_ship_skeleton  # noqa: E402

ANTHROPIC_AVAILABLE = False
try:
    import anthropic  # type: ignore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]


# ────────────────────────────────────────────────────────────────────
# Tool implementations
# ────────────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "name": "Bash",
        "description": "Execute a bash command in the current working directory. Returns stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "default": 60},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file by absolute or relative path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    },
]


def execute_tool(name: str, params: dict, workdir: Path) -> str:
    """Run a tool call against the trial workdir; return string result."""
    try:
        if name == "Bash":
            cmd = params.get("command", "")
            timeout = params.get("timeout", 60)
            r = subprocess.run(cmd, shell=True, cwd=workdir,
                                capture_output=True, text=True, timeout=timeout)
            out = (r.stdout or "")[:4000]
            err = (r.stderr or "")[:2000]
            return f"exit={r.returncode}\n{out}\n--- stderr ---\n{err}".strip()
        elif name == "Read":
            p = params.get("file_path", "")
            target = workdir / p if not Path(p).is_absolute() else Path(p)
            if not target.exists():
                return f"FileNotFoundError: {p}"
            return target.read_text(errors="replace")[:6000]
    except subprocess.TimeoutExpired:
        return f"TimeoutExpired after {params.get('timeout', 60)}s"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return f"unknown tool: {name}"


# ────────────────────────────────────────────────────────────────────
# Trial loop
# ────────────────────────────────────────────────────────────────────

def build_system_prompt(condition: str) -> str:
    """For with_skill: inject SKILL.md into system prompt. For no_skill:
    minimal generic system prompt."""
    if condition == "with_skill":
        skill = (ROOT / ".claude/skills/rf-simulator/SKILL.md").read_text()
        return (
            "You are a wireless-simulation engineer with access to Bash and "
            "Read tools. Use them to write Python that produces the required "
            "artifacts.\n\n"
            "AVAILABLE SKILL DOCUMENTATION:\n\n"
            f"{skill}"
        )
    return (
        "You are a wireless-simulation engineer with access to Bash and "
        "Read tools. Use them to write Python that produces the required "
        "artifacts."
    )


def run_one_trial(client, task: dict, condition: str, model: str,
                  workdir: Path, max_turns: int = 20, timeout: int = 400) -> dict:
    """Single tool-using loop over the Anthropic API."""
    pre_ship_skeleton(task, workdir)
    artifacts = task.get("required_artifacts", ["simulation_result.json"])
    user_prompt = (
        f"{task['prompt'].strip()}\n\n"
        f"Required output artifacts: {artifacts}.\n"
        f"Working dir: {workdir}.\n"
        f"Use Bash + Read tools to produce them."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    system_prompt = build_system_prompt(condition)

    t0 = time.time()
    turns = 0
    while turns < max_turns:
        if time.time() - t0 > timeout:
            break
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS_SCHEMA,
                messages=messages,
            )
        except Exception as e:
            return {"error": f"API error: {e}", "turns": turns,
                    "wall_sec": time.time() - t0}
        turns += 1

        # Append assistant message
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            break
        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    out = execute_tool(block.name, dict(block.input), workdir)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue
        break  # other stop_reason: max_tokens, refusal, etc.

    return {"turns": turns, "wall_sec": time.time() - t0,
            "messages": len(messages)}


def run_label(label: str, model: str, conditions: list[str],
               tasks: list[dict], output_root: Path, workers: int = 4,
               max_turns: int = 20, timeout: int = 400) -> dict:
    """Sequential runner — keeps it simple for now. Add a Pool for parallel."""
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(2)
    assert anthropic is not None  # guaranteed by ANTHROPIC_AVAILABLE guard above
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var", file=sys.stderr)
        sys.exit(2)
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    for task in tasks:
        for cond in conditions:
            workdir = output_root / cond / task["id"] / "t1"
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "prompt.txt").write_text(task["prompt"])
            print(f"  [{cond}/{task['id']}] running...", flush=True)
            run_info = run_one_trial(client, task, cond, model, workdir,
                                       max_turns=max_turns, timeout=timeout)
            v = None
            try:
                v = verify(task, workdir, exec_success=True)
                passed = v.passed
            except Exception:
                passed = False
            result = {
                "task_id": task["id"],
                "tier": task.get("tier", "?"),
                "capability": task.get("capability", "?"),
                "condition": cond,
                "trial": 1,
                "model": model,
                "harness": "raw_anthropic_sdk",
                "wall_sec": round(run_info.get("wall_sec", 0), 2),
                "usage": {"num_turns": run_info.get("turns", 0)},
                "verification": v.as_dict() if v is not None else
                                  {"passed": False, "score": 0, "checks": []},
                "error": run_info.get("error"),
            }
            (workdir / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
    return {"label": label, "n": len(results),
            "passed": sum(1 for r in results if r.get("verification", {}).get("passed"))}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--conditions", nargs="+",
                    default=["with_skill", "no_skill"])
    ap.add_argument("--tasks-file", type=Path,
                    default=ROOT / "benchmark/tasks/tasks.json")
    ap.add_argument("--split", default="train",
                    help="train|test (test is held-out — see run_heldout.sh)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only first N tasks (smoke testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=400)
    args = ap.parse_args()

    data = json.loads(args.tasks_file.read_text())
    tasks = [t for t in data["tasks"] if t.get("split") == args.split]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Running {len(tasks)} {args.split} tasks × {len(args.conditions)} "
          f"conditions = {len(tasks) * len(args.conditions)} trials")

    output_root = ROOT / "benchmark/results" / args.label
    output_root.mkdir(parents=True, exist_ok=True)

    summary = run_label(args.label, args.model, args.conditions, tasks,
                          output_root, workers=args.workers,
                          max_turns=args.max_turns, timeout=args.timeout)
    print(f"\nDone. {summary['passed']}/{summary['n']} passed")


if __name__ == "__main__":
    main()
