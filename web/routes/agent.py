"""Agent-step endpoint for the rf-simulator skill chatbox.

Replaces the old Haiku action-block parser. The chatbox now forwards prompts
into this endpoint, which spawns the `claude` CLI as a subprocess with the
rf-simulator skill available, captures any new files produced during the
turn, then packages them as a list of typed "panes" the frontend renders.

Pane types: ber_curve, scene, table, code, json, text.

Streaming: optional. POST /api/agent/step returns JSON in one shot.
GET /api/agent/stream?job_id=... is an SSE companion that mirrors the
SSE pattern from routes.pages.progress().
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, Response, jsonify, request

from routes.shared import (
    OUTPUTS_DIR,
    _create_job,
    _update_job,
    _finish_job,
    _fail_job,
    _get_job,
    _add_job_slice,
)


def _maybe_build_viewer(workdir: Path, new_files: List[Path]) -> None:
    """If the agent produced a scene_state.json (and no companion viewer),
    generate viewer.html via the rf-simulator skill's template_viewer.
    Self-contained HTML so the dashboard can iframe it directly.

    No-op when scene_state.json wasn't touched this turn, or when the
    template module fails to import (best-effort).
    """
    scene_files = [
        f for f in new_files
        if f.name == "scene_state.json"
    ]
    if not scene_files:
        return
    viewer = workdir / "viewer.html"
    # Always regenerate if scene_state was just written — picks up updates.
    try:
        import sys as _sys
        skill_templates = (
            REPO_ROOT / ".claude" / "skills" / "rf-simulator" / "templates"
        )
        if str(skill_templates) not in _sys.path:
            _sys.path.insert(0, str(skill_templates))
        from template_viewer import build_viewer  # type: ignore
        build_viewer(str(scene_files[0]), str(viewer))
    except Exception:
        # Don't crash the agent flow over a viewer generation failure.
        pass


def _summarize_tool_use(name: str, inp: dict) -> str:
    """One-line summary of a tool_use block for the chat UI."""
    if name == "Bash":
        cmd = (inp.get("command") or "")[:120]
        return f"$ {cmd}" + (" …" if len(inp.get("command") or "") > 120 else "")
    if name == "Read":
        return f"read {inp.get('file_path', '?')}"
    if name in ("Edit", "Write"):
        return f"{name.lower()} {inp.get('file_path', '?')}"
    if name == "Glob":
        return f"glob {inp.get('pattern', '?')}"
    if name == "Grep":
        return f"grep {inp.get('pattern', '?')!r}"
    if name == "Task":
        desc = (inp.get("description") or "")[:80]
        return f"subagent: {desc}"
    return name

agent_bp = Blueprint("agent", __name__)

# Working directory the agent CLI runs in.  Keep it under outputs/ so the
# scene browser already picks up anything the agent emits.
AGENT_WORKDIR = OUTPUTS_DIR / "_agent"
AGENT_WORKDIR.mkdir(parents=True, exist_ok=True)

# Resolve the repo root (skill/) from this file: web/routes/agent.py -> ../..
REPO_ROOT = Path(__file__).resolve().parents[2]

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
AGENT_MODEL = os.environ.get("RF_AGENT_MODEL", "claude-sonnet-4-5")
AGENT_MAX_TURNS = int(os.environ.get("RF_AGENT_MAX_TURNS", "12"))
AGENT_TIMEOUT_S = int(os.environ.get("RF_AGENT_TIMEOUT", "180"))


# ─────────────────────────────────────────────────────────
# Pane builders
# ─────────────────────────────────────────────────────────

def _pane(ptype: str, data: Any, title: Optional[str] = None) -> Dict[str, Any]:
    p: Dict[str, Any] = {"type": ptype, "data": data}
    if title:
        p["title"] = title
    return p


def _detect_pane_for_file(p: Path) -> Optional[Dict[str, Any]]:
    """Best-effort mapping of a produced file to a pane spec."""
    suffix = p.suffix.lower()
    name = p.name.lower()
    rel = str(p.relative_to(AGENT_WORKDIR)) if AGENT_WORKDIR in p.parents else p.name
    url = f"/api/agent/file/{rel}"

    if suffix in {".png", ".jpg", ".jpeg"}:
        # BER plots, coverage maps -- render as image.
        return _pane("image", {"url": url}, title=p.name)
    if suffix == ".glb" or suffix == ".gltf":
        return _pane("scene", {"url": url}, title=p.name)
    if suffix == ".html" and "viewer" in name:
        # Self-contained scene viewer (e.g. produced by template_viewer.py
        # auto-build) — iframe it directly so the 3D pane shows up next to
        # the chat instead of just listing the JSON file.
        return _pane("scene_iframe", {"url": url},
                     title=p.parent.name if p.parent.name != "_agent" else "3D viewer")
    if suffix == ".json":
        try:
            obj = json.loads(p.read_text())
        except Exception:
            return None
        # numerical_metrics-ish shape -> ber_curve
        if isinstance(obj, dict) and any(k in obj for k in ("ber", "snr_db", "ber_curve", "snr")):
            return _pane("ber_curve", obj, title=p.stem.replace("_", " "))
        # scene_state.json -> 3D scene viewer (handled by Three.js GLB loader
        # plus a JSON sidebar)
        if "scene_state" in name or "scene" in name:
            return _pane("json", obj, title=p.name)
        return _pane("json", obj, title=p.name)
    if suffix in {".py", ".js", ".ts", ".sh"}:
        try:
            src = p.read_text()
        except Exception:
            return None
        return _pane("code", {"language": suffix.lstrip("."), "src": src}, title=p.name)
    if suffix == ".md":
        try:
            return _pane("text", {"markdown": p.read_text()}, title=p.name)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────
# Claude CLI runner
# ─────────────────────────────────────────────────────────

def _snapshot_files(workdir: Path) -> Dict[str, float]:
    """Return {relpath: mtime} for every file in workdir."""
    out: Dict[str, float] = {}
    for f in workdir.rglob("*"):
        if f.is_file():
            try:
                out[str(f.relative_to(workdir))] = f.stat().st_mtime
            except OSError:
                continue
    return out


def _diff_files(before: Dict[str, float], workdir: Path) -> List[Path]:
    """Files created or modified since `before`."""
    new: List[Path] = []
    for f in workdir.rglob("*"):
        if not f.is_file():
            continue
        try:
            rel = str(f.relative_to(workdir))
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if rel not in before or mtime > before[rel] + 0.001:
            new.append(f)
    return new


def _parse_result_text(stdout: str) -> str:
    """Pull the final `result` event's text out of stream-json NDJSON."""
    for line in stdout.splitlines()[::-1]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "result":
            return (ev.get("result") or "").strip()
    # Fallback: last assistant message text.
    for line in stdout.splitlines()[::-1]:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return (blk.get("text") or "").strip()
    return ""


def _parse_layout_hint(prompt: str, n_panes: int) -> Dict[str, Any]:
    """Look for explicit layout requests in the prompt (e.g.
    "compare 3 scenes side-by-side"). Falls back to auto-grid."""
    p = prompt.lower()
    cols = None
    if "side-by-side" in p or "side by side" in p:
        cols = n_panes if n_panes > 0 else None
    if "two column" in p or "2 column" in p or "2-column" in p:
        cols = 2
    if "three column" in p or "3 column" in p or "3-column" in p:
        cols = 3
    return {"cols": cols} if cols else {"auto": True}


def _run_agent(prompt: str, history: List[Dict[str, str]], job_id: str) -> Dict[str, Any]:
    """Invoke the claude CLI, capture produced files, return result dict."""
    workdir = AGENT_WORKDIR / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    # Pre-pend conversation history into the prompt so the agent has context.
    transcript = ""
    for turn in history or []:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            transcript += f"[{role}] {content}\n"
    full_prompt = (transcript + f"[user] {prompt}").strip()

    env = dict(os.environ)
    env["RF_SKILL_DIR"] = str(REPO_ROOT / ".claude" / "skills" / "rf-simulator")
    env["RF_NO_PROMPT"] = "1"
    # If the user has Claude Code OAuth credentials, prefer those over any
    # ANTHROPIC_API_KEY that ~/.bashrc / dashboard env-loader may have set.
    # An expired/stale API key would override OAuth and produce
    # "Invalid API key" errors otherwise.
    oauth_cred = Path.home() / ".claude" / ".credentials.json"
    if oauth_cred.exists():
        env.pop("ANTHROPIC_API_KEY", None)

    cmd = [
        CLAUDE_BIN, "-p", full_prompt,
        "--model", AGENT_MODEL,
        "--max-turns", str(AGENT_MAX_TURNS),
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json",
        "--verbose",
    ]

    before = _snapshot_files(workdir)
    _update_job(job_id, 10, "invoking agent")

    t0 = time.time()
    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    rc: int = 0
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(workdir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,  # line-buffered
        )
    except FileNotFoundError:
        return {
            "text": (f"Claude CLI not found (`{CLAUDE_BIN}`). Set CLAUDE_BIN "
                     "or install claude-code."),
            "panes": [],
            "files_produced": [],
            "error": "cli_not_found",
        }

    # Stream stdout line-by-line, parse each as stream-json event, and push
    # user-visible "thinking" updates into the job's slice buffer so the
    # chat UI can render them as they arrive instead of waiting for the
    # whole agent to finish.
    deadline = t0 + AGENT_TIMEOUT_S
    assistant_text_so_far = ""
    tool_count = 0
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if time.time() > deadline:
            proc.kill()
            timed_out = True
            break
        stdout_buf.append(line)
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Stream-json shapes we care about:
        #   {"type":"system","subtype":"init",...}     — startup
        #   {"type":"assistant","message":{"content":[{"type":"text"|"tool_use",...}]}}
        #   {"type":"user","message":{"content":[{"type":"tool_result",...}]}}
        #   {"type":"result","subtype":"success",...}  — final
        t = ev.get("type")
        if t == "system":
            _update_job(job_id, 15, "session started")
        elif t == "assistant":
            msg = ev.get("message", {}) or {}
            for blk in (msg.get("content") or []):
                btype = blk.get("type")
                if btype == "text":
                    txt = (blk.get("text") or "").strip()
                    if txt:
                        assistant_text_so_far += txt + "\n"
                        _add_job_slice(job_id, {"kind": "thinking", "text": txt})
                        _update_job(job_id, 40, "thinking")
                elif btype == "tool_use":
                    tool_count += 1
                    name = blk.get("name") or "tool"
                    inp = blk.get("input") or {}
                    summary = _summarize_tool_use(name, inp)
                    _add_job_slice(job_id, {"kind": "tool_use", "tool": name,
                                            "summary": summary})
                    _update_job(job_id, min(40 + tool_count * 5, 85),
                                f"using {name}")
        elif t == "user":
            # tool_result blocks — short status only (avoid spamming full output)
            msg = ev.get("message", {}) or {}
            for blk in (msg.get("content") or []):
                if blk.get("type") == "tool_result":
                    is_err = blk.get("is_error", False)
                    _add_job_slice(job_id, {"kind": "tool_result",
                                            "ok": (not is_err)})
        elif t == "result":
            _update_job(job_id, 95, "wrapping up")

    proc.stdout.close()
    if proc.stderr is not None:
        stderr_buf.append(proc.stderr.read())
        proc.stderr.close()
    rc = proc.wait()
    if timed_out:
        rc = -1
        stderr_buf.append("\n[TIMEOUT]")

    stdout = "".join(stdout_buf)
    stderr = "".join(stderr_buf)
    wall = time.time() - t0

    text = _parse_result_text(stdout) or "(agent returned no text)"
    if rc != 0 and "[TIMEOUT]" in stderr:
        text = f"{text}\n\n_(agent timed out after {AGENT_TIMEOUT_S}s)_".strip()

    new_files = _diff_files(before, workdir)

    # If the agent emitted a scene_state.json but not a viewer, auto-generate
    # an interactive viewer.html so the chat can load the room into the 3D
    # pane (instead of dumping JSON to the user).
    _maybe_build_viewer(workdir, new_files)
    # Re-diff so the freshly-generated viewer.html is picked up below.
    new_files = _diff_files(before, workdir)

    panes: List[Dict[str, Any]] = []
    for f in sorted(new_files):
        pane = _detect_pane_for_file(f)
        if pane is not None:
            panes.append(pane)

    # Always prepend a text pane if there's a text reply with content.
    if text:
        panes.insert(0, _pane("text", {"markdown": text}, title="Reply"))

    layout = _parse_layout_hint(prompt, len(panes))

    return {
        "text": text,
        "panes": panes,
        "layout": layout,
        "files_produced": [str(f.relative_to(workdir)) for f in new_files],
        "wall_seconds": round(wall, 2),
        "exit_code": rc,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


# ─────────────────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────────────────

@agent_bp.route("/api/agent/step", methods=["POST"])
def agent_step():
    """Synchronous agent invocation. Blocks until the CLI exits."""
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history") or []
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    job_id = _create_job()
    try:
        result = _run_agent(prompt, history, job_id)
    except Exception as e:
        _fail_job(job_id, str(e))
        return jsonify({"error": str(e), "job_id": job_id}), 500

    _finish_job(job_id, result)
    result["job_id"] = job_id
    return jsonify(result)


@agent_bp.route("/api/agent/step-async", methods=["POST"])
def agent_step_async():
    """Kick off a background invocation; client polls /api/progress/<job_id>
    (the SSE endpoint in routes.pages) for incremental updates."""
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history") or []
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    job_id = _create_job()

    def _worker():
        try:
            result = _run_agent(prompt, history, job_id)
            _finish_job(job_id, result)
        except Exception as e:
            _fail_job(job_id, str(e))

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@agent_bp.route("/api/agent/file/<path:relpath>")
def agent_file(relpath: str):
    """Serve a file the agent produced. Paths are scoped to AGENT_WORKDIR."""
    # Resolve and ensure the requested path stays inside AGENT_WORKDIR.
    target = (AGENT_WORKDIR / relpath).resolve()
    try:
        target.relative_to(AGENT_WORKDIR.resolve())
    except ValueError:
        return jsonify({"error": "forbidden"}), 403
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    from flask import send_file
    return send_file(str(target))
