"""Interaction-benchmark runner — drives a multi-turn dialog between the
agent under test and an LLM-powered user-simulator.

Workflow per task:
  1. Run the agent with `initial_prompt` (under-specified).
  2. After each agent response, parse for a clarifying question (a turn
     containing a "?" with no tool_use call).
  3. If found, dispatch to the user-simulator (Anthropic API or local LLM)
     with `user_persona` as system prompt + accumulated dialog history.
  4. Inject the user-simulator's response as the next user turn into the
     agent (via the harness's stream-json input format).
  5. Loop until agent calls verify_output OR writes simulation_result.json
     OR hits max_clarifying_qs cap.

This is a SCAFFOLD — production wiring (multi-turn input to openclaude /
claude CLI) needs:
  - openclaude/claude `--input-format stream-json` so we can feed user
    turns mid-conversation. Verify with `claude --help`.
  - User-simulator backend: defaults to ANTHROPIC_API_KEY (Haiku-3.5 cheap),
    falls back to OPENAI_BASE_URL local LLM if set.
  - Verifier extension: add `interaction` check kind (counts clarifying Qs
    from stdout.txt).

Usage (scaffold-only — single task dry-run):
    python3 benchmark/trial/interaction_runner.py \\
        --task-file benchmark/tasks/tasks_interaction.json \\
        --task-id I001 \\
        --workdir /tmp/i001_test \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# User simulator (driven by an LLM)
# ---------------------------------------------------------------------------

USER_SIM_SYSTEM_TEMPLATE = """You are roleplaying a USER who needs help with a
wireless simulation task. The AGENT will ask you clarifying questions; respond
naturally and concisely (≤2 sentences). Do not solve the task yourself — only
provide the parameters the agent asks for. Stay in persona.

GROUND TRUTH (only reveal when relevant to a question):
{ground_truth}

PERSONA:
{persona}
"""


def call_user_simulator(persona: str, ground_truth: dict,
                        dialog: list[dict]) -> str:
    """Call an LLM to generate the next user turn.

    `dialog` is a list of `{"role": "user"|"assistant", "content": "..."}`.
    Returns the simulator's text reply.

    Backend selection (in order):
      1. ANTHROPIC_API_KEY → claude-haiku via anthropic SDK
      2. OPENAI_BASE_URL set → local LLM via openai SDK
      3. Else → fallback canned response (echoes ground truth)
    """
    sys_msg = USER_SIM_SYSTEM_TEMPLATE.format(
        ground_truth=json.dumps(ground_truth, indent=2),
        persona=persona,
    )

    # Anthropic Haiku path. Use the alias `claude-haiku-4-5` (or
    # operator-overridden via INTERACTION_USER_SIM_MODEL) so the runner
    # doesn't break when a dated model id is retired.
    if os.environ.get("ANTHROPIC_API_KEY"):
        sim_model = os.environ.get("INTERACTION_USER_SIM_MODEL",
                                    "claude-haiku-4-5")
        # Errors here MUST raise (not fall through to canned reveal) —
        # the canned reveal leaks ground truth and silently invalidates
        # the benchmark. Operator should fix the API key / model id.
        import anthropic  # type: ignore
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=sim_model,
            max_tokens=200,
            system=sys_msg,
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in dialog],
        )
        # The first content block of a non-tool-use response is text; if it's
        # something else (rare), str() it so we still get a usable string.
        first = resp.content[0]
        return getattr(first, "text", str(first))

    # OpenAI-compatible (local LLM) path. Same hard-fail policy.
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        from openai import OpenAI  # type: ignore
        client = OpenAI(
            base_url=base,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("INTERACTION_USER_SIM_MODEL",
                                  "benchmark-model"),
            max_tokens=200,
            messages=[{"role": "system", "content": sys_msg}]
            + [{"role": m["role"], "content": m["content"]}
               for m in dialog],
        )
        return resp.choices[0].message.content or ""

    # No backend configured. REFUSE to fall through to a canned ground-truth
    # reveal — that would silently invalidate the trial. Operator must
    # provide ANTHROPIC_API_KEY OR OPENAI_BASE_URL.
    raise RuntimeError(
        "interaction_runner: no user-simulator backend configured. "
        "Set ANTHROPIC_API_KEY or OPENAI_BASE_URL. "
        "(The earlier canned-reveal fallback was removed because it "
        "leaks ground truth and invalidates the trial.)"
    )


# ---------------------------------------------------------------------------
# Clarifying-question detection
# ---------------------------------------------------------------------------

# A turn is a clarifying question if its assistant text contains a `?`
# and the turn produced NO tool_use blocks (i.e., it's pure dialog).
_QUESTION_RE = re.compile(r"\?")


def is_clarifying_question(assistant_text: str, tool_calls: list) -> bool:
    if tool_calls:
        return False
    return bool(_QUESTION_RE.search(assistant_text or ""))


# ---------------------------------------------------------------------------
# Main scaffold (dry-run only — production multi-turn wiring is TBD)
# ---------------------------------------------------------------------------

def run_interactive_trial(task: dict, workdir: Path, dry_run: bool) -> dict:
    """Run one interaction trial. Currently scaffold-only.

    Returns a dict suitable for serializing to interaction_result.json.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    persona = task["user_persona"]
    gt = task["ground_truth"]
    initial = task["initial_prompt"]
    max_qs = task.get("max_clarifying_qs", 5)

    dialog = [{"role": "user", "content": initial}]

    if dry_run:
        # Just exercise the user-simulator one turn, don't invoke the agent.
        agent_question = "What modulation should I use? AWGN or fading?"
        dialog.append({"role": "assistant", "content": agent_question})
        sim_reply = call_user_simulator(persona, gt, dialog)
        dialog.append({"role": "user", "content": sim_reply})
        return {
            "task_id": task["id"],
            "mode": "dry_run",
            "agent_q": agent_question,
            "sim_reply": sim_reply,
            "dialog": dialog,
            "n_clarifying_qs": 1,
        }

    # PRODUCTION (TODO): wire to claude/openclaude --input-format stream-json
    # and feed `dialog` updates as new user turns arrive. For now this raises
    # so calling code is forced to pass --dry-run.
    raise NotImplementedError(
        "Production interaction loop not yet implemented. Scaffold only "
        "supports --dry-run. See module docstring for what's needed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task-file", required=True,
                    help="Path to tasks_interaction.json")
    ap.add_argument("--task-id", required=True,
                    help="Task id within the file (e.g., I001)")
    ap.add_argument("--workdir", required=True,
                    help="Trial workdir")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't invoke the agent — just exercise the user "
                         "simulator + print one round of dialog.")
    args = ap.parse_args()

    tasks = json.loads(Path(args.task_file).read_text())["tasks"]
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if task is None:
        print(f"# task {args.task_id} not found in {args.task_file}",
              file=sys.stderr)
        return 1

    result = run_interactive_trial(
        task=task, workdir=Path(args.workdir), dry_run=args.dry_run,
    )
    out_file = Path(args.workdir) / "interaction_result.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
