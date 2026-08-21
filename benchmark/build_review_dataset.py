"""Assemble per-scene review dataset folders for human/expert review.

Layout produced under benchmark/_review_dataset/:
    INDEX.md
    S01/
      README.md
      scene_state.json
      T1_single_ap_coverage/
        prompt.txt
        simulation.py
        coverage_map.png
        coverage_map.npy
        simulation_result.json
        stdout.txt           (truncated for size)
        result.json
        verifier_report.txt  (human-readable verdict + per-check)
    S02/ ...
    S20/ ...

Currently fills T1 from existing tc30_c1_train_v6 with_skill results.
T2/T3/T4 stubs are created with a placeholder README explaining how
they'll be filled later.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Ensure `benchmark.verifier` is importable when running this script directly
# from the project root or from inside the benchmark/ dir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.verifier import verify as _verify  # noqa: E402


# T0 sources (100-task scene-gen benchmark, 60 train + 40 test):
SRC_T0_TRAIN_V5_WS = Path("benchmark/results/t0v2_b_ws_v5/with_skill")
SRC_T0_TEST_V5_WS  = Path("benchmark/results/t0v2_b_test/with_skill")
T0_TASK_SOURCES    = Path("benchmark/tasks/_sources/t0_redesign.json")

# T1 source order: v7 (placement-validator skill) preferred where available,
# otherwise fall back to v6. v7 covered only the 6 v6-fail scenes by design.
SRC_T1_PRIMARY = Path("benchmark/results/t1_v7_placement/with_skill")
SRC_T1_FALLBACK = Path("benchmark/results/tc30_c1_train_v6/with_skill")

# Cached task specs for re-verification (loaded lazily in main())
_TASK_SPECS: dict[str, dict] = {}
# T2 source: prefer v3_retry80 (max_turns=80 retry for the 4 placeholders)
# where available, then v3 (concrete partition + threshold -50 + v8 RT-only).
# v1/v2 are kept on disk for ablation but not used here.
SRC_T2_PRIMARY = Path("benchmark/results/t2_scene_edit_v3_retry80/with_skill")
SRC_T2_FALLBACK = Path("benchmark/results/t2_scene_edit_v3/with_skill")
TASK_SOURCES = Path("benchmark/tasks/_sources/tc_chained.json")
OUT = Path("benchmark/_review_dataset")

# Map TC1_S{NN} → S{NN}; 20 train scenes only (S01..S20)
SCENE_IDS = [f"S{i:02d}" for i in range(1, 21)]

# Generic copy mapping for any task type that produces these artifacts.
# Names are matched against trial dir; missing ones are silently skipped.
GENERIC_FILES = [
    "prompt.txt", "simulation.py",
    "coverage_map.png", "coverage_map.npy",
    "coverage_map_before.png", "coverage_map_before.npy",
    "coverage_map_after.png",  "coverage_map_after.npy",
    "coverage_heatmap.png",
    "scene_state_before.json", "scene_state_after.json",
    "simulation_result.json", "result.json",
]

T1_FILES = {
    "prompt.txt": "prompt.txt",
    "simulation.py": "simulation.py",
    "coverage_map.png": "coverage_map.png",
    "coverage_map.npy": "coverage_map.npy",
    "simulation_result.json": "simulation_result.json",
    "scene_state.json": "scene_state.json",
    "result.json": "result.json",
    "stdout.txt": "stdout.txt",
}

T2_T3_T4_README = """# Pending — needs agent run

This task type has not been run yet for {scene_id}.

To populate: extend `benchmark/tasks/_sources/tc_chained_gen.py` to
emit T2/T3/T4 tasks per scene, then run `benchmark/run_benchmark.py`
with the new task IDs.
"""


def _truncate_stdout(text: str, max_bytes: int = 200_000) -> str:
    """Trim long stdout to keep folder size sane while preserving start+end."""
    b = text.encode("utf-8", errors="replace")
    if len(b) <= max_bytes:
        return text
    half = max_bytes // 2 - 200
    head = b[:half].decode("utf-8", errors="replace")
    tail = b[-half:].decode("utf-8", errors="replace")
    return (head
            + f"\n\n--- [truncated {len(b) - max_bytes} bytes for review dataset] ---\n\n"
            + tail)


def _verifier_report_text(result: dict) -> str:
    """Render result.json's verification block into a readable text report."""
    lines = []
    v = result.get("verification") or {}
    lines.append(f"OVERALL: {'PASS' if v.get('passed') else 'FAIL'} "
                 f"(score {v.get('score', '?')})")
    lines.append("")
    for c in v.get("checks", []):
        mark = "✓" if c.get("passed") else "✗"
        lines.append(f"  {mark}  {c.get('name', '?'):<40s}  {c.get('detail', '')[:140]}")
    if v.get("notes"):
        lines.append("")
        lines.append("Notes:")
        for n in v["notes"]:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def populate_generic_task(src_dir: Path, dst_dir: Path,
                          task_prompt: str | None,
                          scene_state_destination: Path | None = None) -> dict:
    """Copy a trial's artifacts into the dataset folder for ANY task type.

    Honors GENERIC_FILES + stdout.txt (truncated) + verifier_report.txt
    (rendered from result.json). scene_state.json is special-cased: it's
    moved up to the scene-level destination if provided.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for name in GENERIC_FILES:
        s = src_dir / name
        if not s.exists():
            continue
        shutil.copy2(s, dst_dir / name)
        copied.append(name)

    # scene_state.json → scene level (one per scene, shared across tasks)
    s = src_dir / "scene_state.json"
    if s.exists() and scene_state_destination is not None and \
            not scene_state_destination.exists():
        shutil.copy2(s, scene_state_destination)
        copied.append(f"../{scene_state_destination.name}")

    # stdout (truncated)
    so = src_dir / "stdout.txt"
    if so.exists():
        (dst_dir / "stdout.txt").write_text(_truncate_stdout(so.read_text(errors="replace")))
        copied.append("stdout.txt")

    # Verifier report — re-run verify() with CURRENT verifier code so the
    # report reflects the latest robustness fixes (e.g., schema-tolerant
    # _furniture_tuple, dropped redundant token grep). The result.json on
    # disk reflects the verifier AT TIME OF RUN, which may be stale.
    res_path = src_dir / "result.json"
    if res_path.exists():
        stored = json.loads(res_path.read_text())
        # Re-verify with current verifier; fall back to stored if that fails.
        tcid = stored.get("task_id") or stored.get("origin_id")
        if tcid and tcid in _TASK_SPECS:
            sim_ok = (src_dir / "simulation_result.json").exists()
            v = _verify(_TASK_SPECS[tcid], src_dir, exec_success=sim_ok)
            fresh = {"verification": v.as_dict(), "task_id": tcid}
            (dst_dir / "verifier_report.txt").write_text(_verifier_report_text(fresh))
        else:
            (dst_dir / "verifier_report.txt").write_text(_verifier_report_text(stored))
        copied.append("verifier_report.txt")

    if task_prompt:
        (dst_dir / "prompt.txt").write_text(task_prompt)

    return {"copied": copied, "has_result": (src_dir / "result.json").exists()}


def populate_T1(scene_id: str, src_dir: Path, dst_dir: Path,
                task_prompt: str | None) -> dict:
    """Copy v6 train T1 outputs into the dataset folder."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts that exist
    copied: list[str] = []
    for src_name, dst_name in T1_FILES.items():
        s = src_dir / src_name
        if not s.exists():
            continue
        d = dst_dir / dst_name
        if src_name == "stdout.txt":
            d.write_text(_truncate_stdout(s.read_text(errors="replace")))
        elif src_name == "scene_state.json":
            # scene_state.json lives at the SCENE level, not per-task
            scene_dst = dst_dir.parent / "scene_state.json"
            shutil.copy2(s, scene_dst)
            copied.append("../scene_state.json")
            continue
        else:
            shutil.copy2(s, d)
        copied.append(dst_name)

    # If we have a task prompt from tc_chained.json, write a canonical prompt.txt
    if task_prompt:
        (dst_dir / "prompt.txt").write_text(task_prompt)

    # Verifier report
    res_path = src_dir / "result.json"
    if res_path.exists():
        try:
            result = json.loads(res_path.read_text())
            (dst_dir / "verifier_report.txt").write_text(_verifier_report_text(result))
        except Exception:
            pass

    return {"copied": copied}


def write_scene_readme(scene_id: str, scene_state_path: Path,
                       readme_path: Path) -> dict:
    """Per-scene human-readable summary."""
    if not scene_state_path.exists():
        readme_path.write_text(f"# Scene {scene_id}\n\nMissing scene_state.json.\n")
        return {"valid": False}

    state = json.loads(scene_state_path.read_text())
    bounds = state.get("scene", {}).get("bounds", {})
    W = bounds.get("width", "?")
    D = bounds.get("depth", "?")
    H = bounds.get("height", "?")
    rooms = state.get("rooms", [])
    furn = state.get("furniture", []) or [
        f for r in rooms if isinstance(r, dict) for f in r.get("furniture", []) or []]
    furn_types = []
    for f in furn:
        if isinstance(f, dict):
            furn_types.append(f.get("type", "?"))
    aps = state.get("access_points") or state.get("transmitters") or []

    txt = [
        f"# Scene {scene_id}",
        "",
        f"- Bounds: **{W} × {D} × {H} m**",
        f"- Rooms: {len(rooms)}",
        f"- Furniture: {len(furn_types)} items — "
        f"{', '.join(furn_types) if furn_types else '(none)'}",
        f"- AP(s): {len(aps)}",
        "",
        "## Files",
        "- `scene_state.json` — full scene structure (shared across tasks)",
        "- `T1_single_ap_coverage/` — single-AP coverage task (populated)",
        "- `T2_scene_edit/` — scene-edit + recompute task (pending)",
        "- `T3_ber/` — BER @ (TX, RX) task (pending)",
        "- `T4_optimization/` — MCS / multi-AP optimization (pending)",
        "",
        "## Expert review",
        "Open each task folder, inspect `simulation.py` + `coverage_map.png` "
        "(and supporting artifacts). Mark up code where needed; write a `REVIEW.md` "
        "in the task folder with verdict (pass/fail) + notes.",
    ]
    readme_path.write_text("\n".join(txt))
    return {"valid": True, "n_furniture": len(furn_types)}


def populate_T0(out_root: Path) -> dict:
    """Populate `T0/` subfolder with 60 train + 40 test scene-gen trials.

    Layout:
      T0/
        README.md          — task overview + per-split summary table
        train/T0xNNN/      — 60 train folders
        test/T0xNNN/       — 40 test folders
    Each per-task folder contains:
        prompt.txt, simulation.py, scene_state.json, simulation_result.json,
        stdout.txt (truncated), result.json, verifier_report.txt
    """
    if not T0_TASK_SOURCES.exists():
        return {"populated": 0, "missing_src": True}
    t0_doc = json.loads(T0_TASK_SOURCES.read_text())
    tasks = t0_doc.get("tasks", [])

    t0_root = out_root / "T0"
    if t0_root.exists():
        shutil.rmtree(t0_root)
    (t0_root / "train").mkdir(parents=True, exist_ok=True)
    (t0_root / "test").mkdir(parents=True, exist_ok=True)

    rows = []
    for t in tasks:
        tid = t["id"]
        split = t.get("split", "train")
        diff = t.get("difficulty", "?")
        src_root = SRC_T0_TRAIN_V5_WS if split == "train" else SRC_T0_TEST_V5_WS
        src = src_root / tid / "t1"
        dst = t0_root / split / tid
        if not src.exists() or not (src / "simulation_result.json").exists():
            rows.append({"id": tid, "split": split, "diff": diff,
                         "populated": False, "passed": None})
            continue
        dst.mkdir(parents=True, exist_ok=True)
        # Copy a small fixed set of artifacts
        for name in ("prompt.txt", "simulation.py", "scene_state.json",
                     "simulation_result.json", "result.json"):
            s = src / name
            if s.exists():
                shutil.copy2(s, dst / name)
        # Truncate stdout
        so = src / "stdout.txt"
        if so.exists():
            (dst / "stdout.txt").write_text(_truncate_stdout(so.read_text(errors="replace")))
        # Verifier report (re-verify with current verifier)
        passed = None
        try:
            sim_ok = (src / "simulation_result.json").exists()
            v = _verify(t, src, exec_success=sim_ok)
            passed = v.passed
            fresh = {"verification": v.as_dict(), "task_id": tid}
            (dst / "verifier_report.txt").write_text(_verifier_report_text(fresh))
        except Exception:
            stored = json.loads((src / "result.json").read_text())
            (dst / "verifier_report.txt").write_text(_verifier_report_text(stored))
            passed = stored.get("verification", {}).get("passed")
        # Pin which skill produced the data (v5 for both train + test)
        (dst / "_skill_version.txt").write_text("v5\n")

        rows.append({"id": tid, "split": split, "diff": diff,
                     "populated": True, "passed": bool(passed)})

    # Per-task README
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test  = sum(1 for r in rows if r["split"] == "test")
    n_train_pass = sum(1 for r in rows if r["split"] == "train" and r["passed"])
    n_test_pass  = sum(1 for r in rows if r["split"] == "test" and r["passed"])
    n_train_pop  = sum(1 for r in rows if r["split"] == "train" and r["populated"])
    n_test_pop   = sum(1 for r in rows if r["split"] == "test" and r["populated"])

    lines = [
        "# T0 — Scene Generation (100 tasks)",
        "",
        f"- Train: {n_train} tasks (populated: {n_train_pop}/{n_train})",
        f"- Test (held-out): {n_test} tasks (populated: {n_test_pop}/{n_test})",
        "",
        f"PASS (v5 with_skill, current verifier):",
        f"- Train: **{n_train_pass}/{n_train_pop} = "
        f"{(100*n_train_pass/n_train_pop if n_train_pop else 0):.1f}%**",
        f"- Test:  **{n_test_pass}/{n_test_pop} = "
        f"{(100*n_test_pass/n_test_pop if n_test_pop else 0):.1f}%**",
        "",
        "## Per-task index",
        "",
        "| ID | Split | Difficulty | Populated | PASS |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["split"], x["id"])):
        pop = "✓" if r["populated"] else "—"
        pas = "✓" if r["passed"] else ("✗" if r["populated"] else "—")
        lines.append(f"| {r['id']} | {r['split']} | {r['diff']} | {pop} | {pas} |")
    (t0_root / "README.md").write_text("\n".join(lines))

    return {
        "populated": n_train_pop + n_test_pop,
        "train_pop": n_train_pop, "train_pass": n_train_pass,
        "test_pop": n_test_pop, "test_pass": n_test_pass,
        "total": n_train + n_test,
    }


def write_index(out_root: Path, summary: list[dict]) -> None:
    lines = [
        "# Review Dataset Index",
        "",
        "Per-task agent outputs for human expert review.",
        "",
        "> **Read first**:",
        "> - [`RESULTS.md`](RESULTS.md) — paper-grade headline numbers (TL;DR table, "
        "skill-iteration trajectory v0→v8, failure-mode separation per layer).",
        "> - [`TASK_DESIGN.md`](TASK_DESIGN.md) — full per-task spec: inputs, required "
        "outputs, verifier subchecks, reference-oracle math, worked examples.",
        "",
        "## Top-level layout",
        "",
        "- [`T0/`](T0/README.md) — scene generation (100 prompts: 60 train + 40 test)",
        "- `S01/ ... S20/` — TC chained-task scenes (T1 single-AP coverage + T2 scene edit)",
        "",
        "## Structure",
        "",
        "Each `S{NN}/` folder contains:",
        "- `README.md` — scene summary",
        "- `scene_state.json` — the scene (shared across tasks)",
        "- `T1_single_ap_coverage/` — single-AP coverage (populated from "
        "`tc30_c1_train_v6` `with_skill`)",
        "- `T2_scene_edit/`, `T3_ber/`, `T4_optimization/` — pending (placeholders)",
        "",
        "## How to review",
        "",
        "1. Open `S{NN}/T{X}_*/simulation.py` — the agent-written code.",
        "2. Open the matching `coverage_map.png` (or whichever artifact the "
        "task produces) and check visually.",
        "3. Open `simulation_result.json` to see the reported metrics.",
        "4. Optional: read `verifier_report.txt` for the auto-verdict + per-check breakdown.",
        "5. Write `REVIEW.md` in the task folder with: verdict (pass/fail), "
        "notes pointing to specific lines or pixels.",
        "",
        "## Summary",
        "",
        "| Scene | Bounds | Furniture | T1 | T2 | T3 | T4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        sid = s["scene_id"]
        bounds = s.get("bounds", "?")
        nf = s.get("n_furniture", "?")
        t1 = "✓" if s.get("t1_ok") else "—"
        t2 = "✓" if s.get("t2_ok") else "—"
        lines.append(f"| {sid} | {bounds} | {nf} | {t1} | {t2} | — | — |")
    lines.append("")
    lines.append("Legend: ✓ populated  —  pending")
    (out_root / "INDEX.md").write_text("\n".join(lines))


def main():
    # Wipe only per-scene folders (S{NN}) and the T0 subtree + INDEX.md.
    # Preserve hand-written root docs (RESULTS.md, TASK_DESIGN.md) so they
    # survive re-populates.
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in OUT.iterdir():
        if sub.is_dir() and (
            (sub.name.startswith("S") and sub.name[1:].isdigit())
            or sub.name == "T0"
        ):
            shutil.rmtree(sub)
    idx = OUT / "INDEX.md"
    if idx.exists():
        idx.unlink()

    # Load task prompts for every task type by capability, and cache full task
    # specs for re-verification in populate_generic_task.
    prompts_by_id: dict[str, str] = {}
    if TASK_SOURCES.exists():
        tdoc = json.loads(TASK_SOURCES.read_text())
        for t in tdoc.get("tasks", []):
            prompts_by_id[t["id"]] = t.get("prompt", "")
            _TASK_SPECS[t["id"]] = t

    summary: list[dict] = []
    for sid in SCENE_IDS:
        scene_dir = OUT / sid
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_state_dst = scene_dir / "scene_state.json"

        # ── T1: single_ap_coverage (prefer v7 placement-validator skill; fall back to v6) ──
        t1_tcid = f"TC1_{sid}"
        t1_v7 = SRC_T1_PRIMARY / t1_tcid / "t1"
        t1_v6 = SRC_T1_FALLBACK / t1_tcid / "t1"
        if t1_v7.exists() and (t1_v7 / "simulation_result.json").exists():
            t1_src = t1_v7
            t1_skill_version = "v7"
        else:
            t1_src = t1_v6
            t1_skill_version = "v6"
        t1_dst = scene_dir / "T1_single_ap_coverage"
        t1_ok = t1_src.exists() and (t1_src / "simulation_result.json").exists()
        if t1_ok:
            populate_T1(sid, t1_src, t1_dst, prompts_by_id.get(t1_tcid))
            (t1_dst / "_skill_version.txt").write_text(t1_skill_version + "\n")
        else:
            t1_dst.mkdir(parents=True, exist_ok=True)
            (t1_dst / "README.md").write_text(
                f"# T1 missing for {sid}\n\nNo source data at {t1_src}.\n")

        # ── T2: prefer retry80 (max_turns=80 for the 4 placeholders) then v3 ──
        t2_tcid = f"TC4_{sid}"
        t2_retry = SRC_T2_PRIMARY / t2_tcid / "t1"
        t2_v3 = SRC_T2_FALLBACK / t2_tcid / "t1"
        if t2_retry.exists() and (t2_retry / "simulation_result.json").exists():
            t2_src = t2_retry
            t2_skill_run = "v3_retry80"
        else:
            t2_src = t2_v3
            t2_skill_run = "v3"
        t2_dst = scene_dir / "T2_scene_edit"
        t2_ok = t2_src.exists() and (t2_src / "simulation_result.json").exists()
        if t2_ok:
            populate_generic_task(t2_src, t2_dst, prompts_by_id.get(t2_tcid),
                                  scene_state_destination=scene_state_dst)
            (t2_dst / "_run_label.txt").write_text(t2_skill_run + "\n")
        else:
            t2_dst.mkdir(parents=True, exist_ok=True)
            (t2_dst / "README.md").write_text(T2_T3_T4_README.format(scene_id=sid))

        # ── T3 / T4: pending ──
        for slot in ("T3_ber", "T4_optimization"):
            d = scene_dir / slot
            d.mkdir(parents=True, exist_ok=True)
            (d / "README.md").write_text(T2_T3_T4_README.format(scene_id=sid))

        # Scene README
        scene_state_path = scene_dir / "scene_state.json"
        meta = write_scene_readme(sid, scene_state_path, scene_dir / "README.md")

        bounds_str = "?"
        if scene_state_path.exists():
            try:
                st = json.loads(scene_state_path.read_text())
                b = st.get("scene", {}).get("bounds", {})
                bounds_str = f"{b.get('width','?')} × {b.get('depth','?')} m"
            except Exception:
                pass

        summary.append({
            "scene_id": sid, "t1_ok": t1_ok, "t2_ok": t2_ok,
            "bounds": bounds_str, "n_furniture": meta.get("n_furniture", "?"),
        })

    # T0 (100 task scene-gen) goes into its own subfolder, parallel to S{NN}/
    t0_stats = populate_T0(OUT)

    write_index(OUT, summary)
    print(f"Wrote {OUT}/ with {len(summary)} scene folders.")
    print(f"  T1 populated: {sum(1 for s in summary if s['t1_ok'])}/{len(summary)}")
    print(f"  T2 populated: {sum(1 for s in summary if s['t2_ok'])}/{len(summary)}")
    if t0_stats.get("populated"):
        print(f"  T0 populated: {t0_stats['populated']}/{t0_stats['total']} "
              f"(train pass {t0_stats['train_pass']}/{t0_stats['train_pop']}, "
              f"test pass {t0_stats['test_pass']}/{t0_stats['test_pop']})")
    print(f"  Top-level index: {OUT}/INDEX.md")


if __name__ == "__main__":
    main()
