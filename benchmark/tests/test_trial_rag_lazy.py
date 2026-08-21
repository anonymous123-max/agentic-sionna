"""RAG context retrieval must be lazy — module import should not touch
chromadb, sys.path, or os.environ['CLAUDE_CODE_USE_RAG']."""
import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def test_import_does_not_mutate_sys_path():
    sys_path_before = list(sys.path)
    if "trial" in sys.modules:
        del sys.modules["trial"]
    importlib.import_module("trial")
    extra = [p for p in sys.path if p not in sys_path_before]
    assert not any(".claude/skills/rf-simulator/memory" in p for p in extra), (
        f"trial.py inserted memory/ into sys.path on import: {extra}")


def test_import_works_without_rag_env_var():
    os.environ.pop("CLAUDE_CODE_USE_RAG", None)
    if "trial" in sys.modules:
        del sys.modules["trial"]
    importlib.import_module("trial")  # must not raise


def test_import_works_with_rag_off():
    os.environ["CLAUDE_CODE_USE_RAG"] = "0"
    if "trial" in sys.modules:
        del sys.modules["trial"]
    importlib.import_module("trial")  # must not raise
