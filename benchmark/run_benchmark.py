"""Single entry point for the full unified benchmark, with crash protection.

Features:
- Resumable: existing result.json files are detected and skipped on rerun.
- Worker-parallel: farms (task × condition × trial) tuples to N subprocesses.
- Per-trial isolation: exceptions don't kill the pool; each failure is logged.
- Heartbeat: writes progress JSON every 30 s for external monitoring.
- Graceful shutdown: SIGTERM finishes in-flight trials, then exits with state
  preserved, so `--resume` picks up exactly where it stopped.
- Crash recovery: if killed mid-trial (SIGKILL / server reboot), the partial
  workdir is cleaned on next launch and the trial re-runs.

Usage:
    # Full benchmark, single condition, 4 workers
    python benchmark/run_benchmark.py --label unified_v2 --workers 4

    # Targeted rerun of the 14 T1 PHY tasks on opus, k=3
    python benchmark/run_benchmark.py --label opus_phy \\
        --task-ids $(python -c "import json; \\
            print(' '.join(t['id'] for t in \\
            json.load(open('benchmark/tasks/tasks.json'))['tasks'] \\
            if t['tier']=='T1_phy_link_level'))") \\
        --model opus --k 3 --workers 4 --resume

    # Paired eval: with_skill AND no_skill baselines
    python benchmark/run_benchmark.py --label paired --workers 6 \\
        --conditions with_skill no_skill

    # Resume a killed run
    python benchmark/run_benchmark.py --label unified_v2 --resume
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.trial import run_one, should_retry_trial, TrialConfig  # noqa: E402
from benchmark.verifier import verify  # noqa: E402


TASKS_FILE = ROOT / "benchmark/tasks/tasks.json"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark/results"


# ─────────────────────────────────────────────────────────────
# State (resume-friendly)
# ─────────────────────────────────────────────────────────────

@dataclass
class TrialKey:
    task_id: str
    condition: str
    trial: int

    def dirname(self, root: Path) -> Path:
        return root / self.condition / self.task_id / f"t{self.trial}"

    def done(self, root: Path) -> bool:
        return (self.dirname(root) / "result.json").exists()

    def is_partial(self, root: Path) -> bool:
        """Has a workdir but no final result.json — likely killed mid-trial."""
        d = self.dirname(root)
        return d.exists() and not (d / "result.json").exists()


def cleanup_partials(root: Path, trials: list[TrialKey]) -> int:
    """Remove partial workdirs so the trial re-runs cleanly."""
    import shutil
    cleaned = 0
    for t in trials:
        if t.is_partial(root):
            shutil.rmtree(t.dirname(root), ignore_errors=True)
            cleaned += 1
    return cleaned


def build_work_queue(tasks: list[dict], conditions: list[str], k: int,
                     task_ids: list[str] | None, tiers: list[str] | None,
                     split: str | None,
                     output_root: Path, resume: bool) -> list[TrialKey]:
    if task_ids:
        ids = set(task_ids)
        tasks = [t for t in tasks
                 if t["id"] in ids or t.get("origin_id") in ids]
    if tiers:
        tset = set(tiers)
        tasks = [t for t in tasks
                 if t["tier"] in tset
                 or any(t["tier"].startswith(x) for x in tset)]
    if split and split != "all":
        # Test split is methodologically protected — require explicit
        # --split=test to run against it. Default train-only behavior in
        # run() passes split="train" unless the caller overrides.
        # `split="all"` means no filter.
        tasks = [t for t in tasks if t.get("split") == split]
    queue: list[TrialKey] = []
    for task in tasks:
        for cond in conditions:
            for trial in range(1, k + 1):
                key = TrialKey(task["id"], cond, trial)
                if resume and key.done(output_root):
                    continue
                queue.append(key)
    return queue


# ─────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────

# Process-wide config — set by pool initializer
_CONFIG: dict | None = None


def _worker_init(cfg: dict):
    global _CONFIG
    _CONFIG = cfg
    # Ignore SIGINT in workers; main handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_trial(key_dict: dict) -> dict:
    """Run one trial in a worker process. Catches all exceptions so a
    single failure doesn't poison the pool."""
    key = TrialKey(**key_dict)
    cfg = _CONFIG or {}
    tasks_by_id: dict = cfg["tasks_by_id"]
    task = tasks_by_id.get(key.task_id)
    if task is None:
        return {"task_id": key.task_id, "condition": key.condition,
                "trial": key.trial, "error": f"task {key.task_id} not in manifest"}
    try:
        result = run_one(
            task=task,
            trial=key.trial,
            results_root=Path(cfg["output_root"]),
            config=TrialConfig(
                model=cfg["model"],
                max_turns=cfg["max_turns"],
                timeout=cfg["timeout"],
                condition=key.condition,
            ),
        )
        return result
    except Exception as e:
        import traceback
        return {"task_id": key.task_id, "condition": key.condition,
                "trial": key.trial, "error": str(e),
                "traceback": traceback.format_exc()[-2000:]}


# ─────────────────────────────────────────────────────────────
# Progress + heartbeat
# ─────────────────────────────────────────────────────────────

def write_progress(output_root: Path, state: dict):
    path = output_root / "progress.json"
    tmp = output_root / ".progress.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, path)


def aggregate_results(output_root: Path, tasks_by_id: dict) -> dict:
    rows = []
    for p in sorted(output_root.rglob("result.json")):
        try:
            r = json.loads(p.read_text())
            task = tasks_by_id.get(r.get("task_id"))
            if task is None:
                continue
            # Re-verify with the CURRENT verifier so results reflect the
            # latest check logic, not whatever was running at trial time.
            v = verify(task, p.parent,
                       exec_success=bool(r.get("exec_success", True)))
            rows.append({
                "task_id": r["task_id"],
                "tier": r.get("tier"),
                "condition": r.get("condition"),
                "trial": r.get("trial"),
                "passed": v.passed, "score": v.score,
            })
        except Exception:
            continue
    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_cond: dict[str, dict[str, int]] = {}
    by_tier: dict[str, dict[str, int]] = {}
    for r in rows:
        c = by_cond.setdefault(r["condition"] or "?", {"n": 0, "pass": 0})
        c["n"] += 1
        c["pass"] += int(r["passed"])
        t = by_tier.setdefault(r["tier"] or "?", {"n": 0, "pass": 0})
        t["n"] += 1
        t["pass"] += int(r["passed"])
    # Normalized gain (Hake 1998 / SkillsBench):
    #   g = (p_skill - p_vanilla) / (1 - p_vanilla)
    # Captures what fraction of the remaining headroom the skill closed.
    gain = None
    if "with_skill" in by_cond and "no_skill" in by_cond:
        nw = by_cond["with_skill"]["n"]
        nn = by_cond["no_skill"]["n"]
        if nw > 0 and nn > 0:
            p_skill = by_cond["with_skill"]["pass"] / nw
            p_vanilla = by_cond["no_skill"]["pass"] / nn
            abs_gain_pp = 100 * (p_skill - p_vanilla)
            norm_gain = ((p_skill - p_vanilla) / (1 - p_vanilla)
                         if p_vanilla < 1 else None)
            gain = {"absolute_gain_pp": round(abs_gain_pp, 2),
                     "normalized_gain": (round(norm_gain, 3)
                                          if norm_gain is not None else None),
                     "p_skill": round(p_skill, 3),
                     "p_vanilla": round(p_vanilla, 3)}
    return {"n": n, "passed": passed,
            "pass_rate": (100 * passed / n) if n else 0,
            "by_condition": by_cond, "by_tier": by_tier,
            "gain": gain}


# ─────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────

def detect_timed_out(output_root: Path, queue: list[TrialKey]) -> list[TrialKey]:
    """Find trials that timed out AND actually failed (set by trial.invoke_claude
    when subprocess.TimeoutExpired fires). These are candidates for one
    retry pass with a longer timeout.

    Uses should_retry_trial() to avoid retrying trials that produced a
    passing result.json despite having [TIMEOUT] in their stderr."""
    out = []
    for k in queue:
        try:
            if should_retry_trial(k.dirname(output_root)):
                out.append(k)
        except Exception:
            pass
    return out


def archive_for_retry(output_root: Path, key: TrialKey) -> None:
    """Move a timed-out trial's workdir to <dir>_attempt1 so the retry
    runs in a fresh workdir. Idempotent — does nothing if already archived."""
    src = key.dirname(output_root)
    if not src.exists():
        return
    dest = src.with_name(src.name + "_attempt1")
    if dest.exists():
        return  # already retried once — don't recurse
    src.rename(dest)


def run(args) -> int:
    tasks_file = Path(args.tasks_file) if args.tasks_file else TASKS_FILE
    tasks = json.loads(tasks_file.read_text())["tasks"]
    tasks_by_id = {t["id"]: t for t in tasks}

    output_root = Path(args.output_root) / args.label
    output_root.mkdir(parents=True, exist_ok=True)

    # Build the work queue — filter, apply resume, clean partials
    # Split guard: default to train for iteration runs, require explicit
    # --split=test to touch the held-out set (and only once, at the end).
    if args.split == "test" and not args.i_understand_held_out:
        raise SystemExit(
            "ERROR: --split=test must be confirmed with "
            "--i-understand-held-out. The test split is evaluated exactly "
            "once — running it repeatedly invalidates the split guarantee.")

    queue = build_work_queue(
        tasks=tasks,
        conditions=args.conditions,
        k=args.k,
        task_ids=args.task_ids,
        tiers=args.tiers,
        split=args.split,
        output_root=output_root,
        resume=args.resume,
    )
    if args.shuffle_seed is not None:
        # Random task order avoids hot-loading one tier first (e.g., all
        # T1 PHY trials sharing GPU contention before moving on). With a
        # fixed seed it stays reproducible.
        random.Random(args.shuffle_seed).shuffle(queue)
        print(f"Shuffled queue with seed={args.shuffle_seed}")
    cleaned = cleanup_partials(output_root, queue)
    if cleaned:
        print(f"Cleaned {cleaned} partial workdirs (killed mid-trial)")

    if not queue:
        print("Nothing to do — all trials already complete. "
              "Drop --resume or change --label to rerun.")
        print()
        print(json.dumps(aggregate_results(output_root, tasks_by_id),
                         indent=2, default=str))
        return 0

    print(f"Launching: {len(queue)} trials across {args.workers} workers")
    print(f"  label={args.label}  model={args.model}  k={args.k}  "
          f"max_turns={args.max_turns}  timeout={args.timeout}s")
    print(f"  conditions={args.conditions}")
    print(f"  output={output_root}")

    cfg = {
        "tasks_by_id": tasks_by_id,
        "output_root": str(output_root),
        "model": args.model,
        "max_turns": args.max_turns,
        "timeout": args.timeout,
    }

    # Graceful shutdown
    stop_flag = mp.Value("i", 0)

    def handle_sigterm(signum, frame):
        print(f"\n[signal {signum}] Finishing in-flight trials then stopping. "
              "Run again with --resume to continue.", flush=True)
        stop_flag.value = 1

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    state = {
        "label": args.label,
        "total_trials": len(queue),
        "completed": 0,
        "errors": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "last_heartbeat": None,
    }

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(cfg,)) as pool:
        try:
            results_iter = pool.imap_unordered(
                _run_trial, [asdict(k) for k in queue], chunksize=1)
            for r in results_iter:
                state["completed"] += 1
                if "error" in r:
                    state["errors"] += 1
                state["last_heartbeat"] = time.strftime("%H:%M:%S")
                elapsed = time.time() - t0
                rate = state["completed"] / max(elapsed, 1)
                eta_s = (state["total_trials"] - state["completed"]) / max(rate, 1e-6)
                state["eta_sec"] = int(eta_s)
                if state["completed"] % 5 == 0 or state["completed"] == state["total_trials"]:
                    print(f"  [{state['completed']}/{state['total_trials']}]"
                          f" errors={state['errors']} "
                          f"rate={rate*60:.1f}/min eta={int(eta_s/60)}min",
                          flush=True)
                write_progress(output_root, state)
                if stop_flag.value:
                    print("Shutdown flag set — draining pool.", flush=True)
                    pool.terminate()
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt — draining pool.")
            pool.terminate()

    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["wall_sec"] = int(time.time() - t0)
    write_progress(output_root, state)

    # Retry pass: re-run timed-out trials once with longer timeout.
    # Prevents infinite-loop debug from killing whole trials, while still
    # giving genuinely-slow tasks a second chance.
    if args.retry_timeout and not stop_flag.value:
        timed_out = detect_timed_out(output_root, queue)
        if timed_out:
            print(f"\nRetry pass: {len(timed_out)} trial(s) timed out — "
                  f"re-running with timeout={args.retry_timeout}s")
            for k in timed_out:
                archive_for_retry(output_root, k)
            cfg["timeout"] = args.retry_timeout
            t1 = time.time()
            with ctx.Pool(args.workers, initializer=_worker_init,
                          initargs=(cfg,)) as pool:
                for r in pool.imap_unordered(
                        _run_trial, [asdict(k) for k in timed_out], chunksize=1):
                    state["completed"] += 1
                    if "error" in r:
                        state["errors"] += 1
                    state["last_heartbeat"] = time.strftime("%H:%M:%S")
                    write_progress(output_root, state)
            print(f"Retry pass done. Wall: {int(time.time() - t1)}s")
            state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            state["wall_sec"] = int(time.time() - t0)

    state["aggregate"] = aggregate_results(output_root, tasks_by_id)
    write_progress(output_root, state)

    print()
    print(f"Done. Wall: {state['wall_sec']}s  "
          f"Completed: {state['completed']}/{state['total_trials']}  "
          f"Errors: {state['errors']}")
    ag = state["aggregate"]
    print(f"Pass rate: {ag['passed']}/{ag['n']} ({ag['pass_rate']:.1f}%)")
    for tier, c in sorted(ag["by_tier"].items()):
        print(f"  {tier}: {c['pass']}/{c['n']} "
              f"({100*c['pass']/c['n']:.0f}%)")

    # P3.5: append a row to iteration_log.md for live tracking.
    try:
        _append_iteration_log(args, state, ag)
    except Exception as e:
        print(f"[iteration_log] append failed: {e}", file=sys.stderr)
    return 0


def _append_iteration_log(args, state: dict, aggregate: dict) -> None:
    """Append a one-line summary to benchmark/_studies_archive/iteration_log.md.
    Master guide Part 7 — live tracking instead of retrospective backfill.

    Computes per-condition pass rates by walking the output dir
    (run_benchmark.py is condition-agnostic at the runner level; the
    breakdown comes from the trial result.jsons)."""
    from collections import defaultdict
    log_path = ROOT / "benchmark" / "_studies_archive" / "iteration_log_auto.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    output_root = Path(args.output_root) / args.label
    cond_stats: dict = defaultdict(lambda: {"n": 0, "pass": 0})
    for cond_dir in output_root.iterdir() if output_root.exists() else []:
        if not cond_dir.is_dir() or cond_dir.name.startswith("_"):
            continue
        for tid_dir in cond_dir.iterdir():
            if not tid_dir.is_dir():
                continue
            for trial in tid_dir.iterdir():
                rj = trial / "result.json"
                if not rj.exists():
                    continue
                try:
                    r = json.loads(rj.read_text())
                except Exception:
                    continue
                cond_stats[cond_dir.name]["n"] += 1
                if r.get("verification", {}).get("passed"):
                    cond_stats[cond_dir.name]["pass"] += 1

    ws = cond_stats.get("with_skill", {"n": 0, "pass": 0})
    ns = cond_stats.get("no_skill", {"n": 0, "pass": 0})
    ws_rate = 100 * ws["pass"] / ws["n"] if ws["n"] else 0.0
    ns_rate = 100 * ns["pass"] / ns["n"] if ns["n"] else 0.0
    delta = ws_rate - ns_rate
    # Normalized gain: (ws - ns) / (1 - ns) — handles ns→100% edge case
    norm_gain = ((ws_rate - ns_rate) / (100 - ns_rate)
                 if ns_rate < 100 else 0.0)

    if not log_path.exists():
        log_path.write_text(
            "# Iteration Log (auto-appended by run_benchmark.py)\n\n"
            "Each row is a single benchmark run. ws/ns = with_skill / no_skill.\n"
            "Δpp = ws% − ns% (positive → skill helps).\n"
            "g = (ws − ns) / (100 − ns) — normalized gain.\n\n"
            "| timestamp | label | model | n | ws% | ns% | Δpp | g |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
    row = (
        f"| {state['finished_at']} "
        f"| {args.label} "
        f"| {args.model} "
        f"| {ws['n']}+{ns['n']} "
        f"| {ws_rate:.1f} | {ns_rate:.1f} "
        f"| {delta:+.1f} | {norm_gain:+.2f} |\n"
    )
    with log_path.open("a") as f:
        f.write(row)
    print(f"\n[iteration_log] appended {row.strip()}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--label", required=True,
                    help="Run label. Output goes to benchmark/results/<label>/")
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--tasks-file", default=None,
                    help="Override tasks JSON path. Default: "
                         "benchmark/tasks/tasks.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel worker processes")
    ap.add_argument("--model", default="sonnet",
                    help="Claude model name (sonnet, opus, haiku)")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=400,
                    help="Per-trial wall timeout in seconds. Pick the value "
                         "that fits ~90%% of trials for your model "
                         "(Sonnet ~150s, Gemma4 ~400s); slow tail gets one "
                         "retry via --retry-timeout.")
    ap.add_argument("--retry-timeout", type=int, default=None,
                    help="If set, after the main pass any trial whose stderr "
                         "contains [TIMEOUT] is rerun once with this longer "
                         "timeout. Original workdir is archived to "
                         "<dir>_attempt1. Recommended: 3x --timeout.")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="Seed to shuffle trial order. Avoids hot-loading one "
                         "tier first. Default: no shuffle (preserves task ID "
                         "order). Recommend a fixed integer for reproducible "
                         "task ordering across reruns.")
    ap.add_argument("--k", type=int, default=1,
                    help="Trials per (task, condition) pair")
    ap.add_argument("--conditions", nargs="+",
                    default=["with_skill"],
                    choices=["with_skill", "no_skill", "self_gen"])
    ap.add_argument("--task-ids", nargs="*", default=None,
                    help="Restrict to these IDs (unified U-ids or origin ids)")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="Restrict to these tier prefixes (e.g. T1 T3)")
    ap.add_argument("--split", default="train",
                    choices=["train", "test", "all"],
                    help="Restrict to train (default), test (requires "
                         "--i-understand-held-out), or all tasks")
    ap.add_argument("--i-understand-held-out", action="store_true",
                    help="Required with --split=test — acknowledges that "
                         "held-out tasks can only be evaluated once")
    ap.add_argument("--resume", action="store_true",
                    help="Skip trials whose result.json already exists")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
