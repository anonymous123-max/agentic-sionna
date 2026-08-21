# benchmark/__init__.py
"""Benchmark harness for Sionna skill evaluation.

Public modules:
    trial          — per-task worker (run_one, should_retry_trial)
    run_benchmark  — parallel orchestrator
    verifier       — verification dispatch + checks
    tool_call_proxy — OpenAI-compat repair proxy
"""
