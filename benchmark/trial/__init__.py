"""Trial worker package — split from monolithic trial.py for clarity.

Public surface (tests + benchmark callers rely on these names):
    run_one, should_retry_trial          — orchestration entrypoints
    TrialConfig                          — frozen dataclass replacing 7-param list
    pre_ship_skeleton                    — used by run_anthropic_harness
    build_prompt                         — used by tests
    _retrieve_rag_context, _rag_store    — referenced by tests
    _load_template                       — referenced by tests (.cache_clear() in test_skill_hint_levels)
    parse_stream_json_usage              — used internally + tests
    _build_invoke_env                    — used by tests (test_trial_env)
    _agent_gave_up_empty                 — used internally
    CONTINUOUS_METRICS                   — referenced by tests
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure the repo root is in sys.path so `from benchmark.verifier import ...`
# works when this package is loaded via `sys.path.insert(0, REPO/"benchmark")`
# (as the test suite does). Mirrors the original trial.py bootstrap.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark.trial.skeletons import (
    pre_ship_skeleton,
    _SKELETON_BASE,
    _SKELETON_BY_TIER,
    _SCENE_SKELETON,
)
from benchmark.trial.prompt import (
    build_prompt,
    _load_template,
    _PROMPTS_DIR,
)
from benchmark.trial.rag import (
    _retrieve_rag_context,
    _rag_store,
)
from benchmark.trial.invoke import (
    invoke_claude,
    parse_stream_json_usage,
    _build_invoke_env,
    _agent_gave_up_empty,
    ROOT,
    CLAUDE_BIN,
)
from benchmark.trial.run import (
    run_one,
    should_retry_trial,
    CONTINUOUS_METRICS,
    TrialConfig,
)
