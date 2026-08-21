"""Append candidate failure principles to a queue for later promotion
into the vector store. Silent on any error — never blocks the trial
worker."""
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_QUEUE = (_REPO / ".claude/skills/rf-simulator/memory"
          / "_pending_principles.jsonl")


def maybe_capture(task: dict, result, workdir: Path) -> None:
    """If a trial failed with a recognizable verifier failure, append a
    candidate principle to the queue. Silent on error."""
    try:
        if getattr(result, "passed", False):
            return
        checks = getattr(result, "checks", []) or []
        failed = [c for c in checks if not getattr(c, "passed", True)]
        if not failed:
            return
        fc = failed[0]
        principle = (f"Task {task.get('id', '?')} ({task.get('tier', '?')}): "
                     f"{fc.name} - {fc.detail[:200]}")
        record = {
            "task_id": task.get("id"),
            "tier": task.get("tier"),
            "capability": task.get("capability"),
            "principle": principle,
            "verifier_name": fc.name,
            "verifier_detail": fc.detail,
        }
        _QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with _QUEUE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # NEVER raise from this hook
