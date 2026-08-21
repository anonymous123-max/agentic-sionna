"""Structural tests for template_optimize.py — does not require sionna.rt
to be installed (we test the file as a Python AST + import structure)."""
from __future__ import annotations
import ast
from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[3]
            / "templates" / "template_optimize.py")


def _parse():
    return ast.parse(TEMPLATE.read_text())


def test_template_parses():
    _parse()


def test_template_has_params_dict():
    """The agent should only edit PARAMS = {...}; it must exist as a top-level dict."""
    tree = _parse()
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PARAMS" in targets and isinstance(node.value, ast.Dict):
                found = True
    assert found, "PARAMS = {...} top-level dict not found"


def test_template_uses_adam():
    src = TEMPLATE.read_text()
    assert "torch.optim.Adam" in src, "must use Adam optimizer"


def test_template_uses_quantile_p5():
    src = TEMPLATE.read_text()
    assert "0.05" in src, "5th-percentile objective requires 0.05 quantile"


def test_template_clips_positions():
    """Position clipping is required per protocol."""
    src = TEMPLATE.read_text()
    assert "_clip_to_room" in src or "clamp" in src, \
        "AP positions must be clipped to room boundary"


def test_template_emits_canonical_artifacts():
    src = TEMPLATE.read_text()
    for art in ("simulation_result.json",
                "optimized_deployment.json",
                "optimize_history.npy"):
        assert art in src, f"template doesn't emit {art}"


def test_template_ends_with_os_exit():
    """RTX 5090 Mitsuba destructor segfault workaround per static-knowledge.md."""
    src = TEMPLATE.read_text()
    assert "os._exit(0)" in src
