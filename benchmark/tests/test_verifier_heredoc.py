"""verifier.check_code_contains — verify that agent code expressed via
`python -c "..."` Bash heredocs is also scanned, not just .py files."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def _setup_workdir(tmp_path, stdout_lines):
    """Write a stdout.txt with the given JSONL stream (assistant tool_use
    events containing Bash commands)."""
    lines = []
    for cmd in stdout_lines:
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}
            ]}
        }))
    (tmp_path / "stdout.txt").write_text("\n".join(lines) + "\n")


def test_code_contains_finds_token_in_heredoc(tmp_path):
    """No .py file, but `python -c` heredoc contains the tokens."""
    from verifier import check_code_contains
    _setup_workdir(tmp_path, [
        'python -c "import numpy; paths_computed = numpy.zeros((5,))"'
    ])
    task = {"verifier": {"metric": "paths_computed", "spec": {}}}
    result = check_code_contains(task, tmp_path)
    assert result.passed, f"should have found 'paths' and 'computed': {result.detail}"


def test_code_contains_still_fails_when_neither_py_nor_heredoc(tmp_path):
    """Truly empty workdir — should fail."""
    from verifier import check_code_contains
    _setup_workdir(tmp_path, ["ls -la", "echo hello"])
    task = {"verifier": {"metric": "paths_computed", "spec": {}}}
    result = check_code_contains(task, tmp_path)
    assert not result.passed


def test_code_contains_picks_either_py_or_heredoc(tmp_path):
    """When BOTH .py file AND heredoc exist, either source can satisfy."""
    from verifier import check_code_contains
    (tmp_path / "simulation.py").write_text("import numpy as np\n")
    _setup_workdir(tmp_path, ['python simulation.py'])
    task = {"verifier": {"metric": "uses_numpy", "spec": {}}}
    result = check_code_contains(task, tmp_path)
    assert result.passed
