"""Prompt template loading and construction."""
from __future__ import annotations
import functools
import os
from pathlib import Path

from benchmark.trial.rag import _retrieve_rag_context

# benchmark/trial/prompt.py is 2 levels below repo root
ROOT = Path(__file__).resolve().parents[2]

_PROMPTS_DIR = ROOT / "benchmark" / "prompts"


@functools.lru_cache(maxsize=8)
def _load_template(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def build_prompt(task: dict, condition: str) -> str:
    """Construct the user prompt for a given task + skill condition.

    All three conditions get the IDENTICAL task prompt. The only thing
    that differs is which skill (if any) is auto-discovered by Claude Code
    via the CWD chosen in invoke_claude().
    """
    base = task["prompt"].strip()
    artifacts = task.get("required_artifacts", ["simulation_result.json"])
    rag_block = _retrieve_rag_context(base) if condition == "with_skill" else ""
    if condition == "with_skill":
        level = os.environ.get("RF_SKILL_HINT_LEVEL", "minimal")
        if level not in ("minimal", "full"):
            level = "minimal"
        skill_hint = _load_template(f"skill_hint_{level}.txt")
    else:
        skill_hint = ""
    tail = _load_template("task_tail.txt").format(artifacts=artifacts)
    return base + rag_block + tail + skill_hint
