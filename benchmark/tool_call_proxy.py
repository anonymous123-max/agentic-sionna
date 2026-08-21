"""OpenAI-compatible chat-completions proxy that repairs malformed
tool calls before they reach OpenClaude / Claude Code.

WHY THIS EXISTS
---------------
On small local LLMs (Qwen3.6, Gemma4, Llama3.1) served via vLLM, the
tool-call parser regularly emits required arguments as `undefined` or
drops them entirely. Examples observed in v1.2/v1.3 trials:

  Write file_path="..." (no `content`)         ← Qwen3.6 ~40% miss rate
  Bash command=undefined                        ← Qwen3.6, intermittent
  Bash description="..." (no `command`)         ← Gemma4 occasionally

When OpenClaude receives these, it returns InputValidationError back
to the model, which then loops on the malformation. Single trial can
burn 25 turns × 3min = ~75 min on a Write retry loop.

This proxy sits between OpenClaude and vLLM:

  OpenClaude  →  this proxy (port 8002)  →  vLLM (port 8001)

For each tool_call in the model's response, it:
  1. Parses the tool's required-args schema (from the request's tools[]).
  2. Detects missing/`undefined`/empty required args.
  3. Repairs by either:
     a. extracting from the assistant message's text content
        (typical: model writes the script body in markdown then calls
         Write, so we pull the ```python block as `content`)
     b. defaulting to a safe no-op so the call doesn't crash the harness
        (e.g. Bash command="echo placeholder")
  4. Returns the repaired response upstream.

DESIGN NOTES
------------
- Stateless. Each chat completion is repaired independently.
- Forwards everything else verbatim (streaming, models endpoint, etc.).
- Logs every repair to stderr so post-run analysis can quantify lift.
- Falls back to no-op repair (returning original) on any internal error,
  so the proxy can't break a working pipeline.

USAGE
-----
  # Start vLLM on :8001 as usual.
  # Start the proxy:
  python3 benchmark/tool_call_proxy.py --port 8002 --upstream http://127.0.0.1:8001
  # Point OpenClaude at :8002 instead of :8001:
  export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
"""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


UPSTREAM = "http://127.0.0.1:8001"  # overridden by --upstream
REPAIR_LOG: deque = deque(maxlen=10000)  # bounded; oldest auto-evicted


# ─────────────────────────────────────────────────────────────
# Token-usage sidecar logging
# ─────────────────────────────────────────────────────────────
# vLLM responses include OpenAI-format usage (prompt_tokens/completion_tokens),
# but openclaude does not translate them to Claude-format
# (input_tokens/output_tokens). Result: every result.json today shows
# usage.input_tokens=0/output_tokens=0, breaking paper-grade token metrics.
# Capturing here at the proxy is the cleanest place — we already see every
# upstream response. Failures here MUST NEVER break a trial.

_USAGE_LOG_PATH = os.environ.get(
    "PROXY_USAGE_LOG",
    f"/workspace/logs/proxy_usage_{os.environ.get('PROXY_PORT', '8101')}.jsonl",
)


def _record_usage(resp_body: dict, port: str) -> None:
    """Append one line to the usage sidecar JSONL with token counts.

    Called with the ORIGINAL upstream body BEFORE any tool-call repair
    mutation, so the recorded counts reflect what vLLM actually billed.
    Best-effort: any exception is swallowed so logging can never break
    a trial.
    """
    try:
        u = (resp_body or {}).get("usage") or {}
        if not u:
            return
        rec = {
            "ts": time.time(),
            "port": port,
            "model": resp_body.get("model", ""),
            "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0),
            "finish_reason": (resp_body.get("choices") or [{}])[0].get(
                "finish_reason", ""
            ),
        }
        # Ensure parent dir exists; cheap idempotent op.
        try:
            Path(_USAGE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with open(_USAGE_LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # never break a trial because of logging


# ─────────────────────────────────────────────────────────────
# P0.3: max_tokens clamp + P0.5: truncation finish_reason remap
# ─────────────────────────────────────────────────────────────
# vLLM 0.20 returns 400 (VLLMValidationError) when prompt_tokens +
# max_tokens exceeds max_model_len. openclaude does not see the error,
# so the trial silently fails. Defensive ceiling: prefer truncated
# generation over a hard 400.

_HEADROOM = 64  # safety margin for chat-template tokens added upstream
_MAX_MODEL_LEN = int(os.environ.get("PROXY_MAX_MODEL_LEN", "32768"))


def _count_prompt_tokens(body: dict) -> int:
    """Best-effort prompt token count. This is a CHARACTER-BASED ESTIMATE,
    not a true tokenization — we trade exactness for the ability to clamp
    without paying tokenizer cost on every request. Conservative ratio:
    1 token per 3.5 chars (Qwen tokenizer is denser than GPT BPE).
    Returns int."""
    msgs = body.get("messages") or []
    total_chars = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total_chars += len(str(part.get("text") or part.get("content") or ""))
    # 100-token slop for system prompt + tool defs not captured above
    return int(total_chars / 3.5) + 100


def _clamp_max_tokens(body: dict) -> dict:
    """Clamp max_tokens so prompt + completion fits in max_model_len.
    Mutates body in place AND returns it."""
    requested = body.get("max_tokens", 8000)
    prompt_est = _count_prompt_tokens(body)
    cap = _MAX_MODEL_LEN - prompt_est - _HEADROOM
    if cap < 256:
        # Prompt nearly fills the window; leave at least 256 tokens for output.
        cap = 256
    if requested > cap:
        body["max_tokens"] = cap
    return body


def _remap_truncation(resp_body: dict, request_body: dict) -> dict:
    """vLLM bug: hits completion_tokens=max_tokens but reports
    finish_reason=tool_calls. Remap to 'length' so the harness retry
    logic kicks in. Does NOT touch usage counts."""
    requested_max = request_body.get("max_tokens", 8000)
    for choice in resp_body.get("choices") or []:
        usage = resp_body.get("usage") or {}
        comp = int(usage.get("completion_tokens", 0) or 0)
        # Off-by-one tolerance — vLLM may stop one token shy of cap
        if comp >= requested_max - 1 and choice.get("finish_reason") == "tool_calls":
            choice["finish_reason"] = "length"
    return resp_body


# ─────────────────────────────────────────────────────────────
# Repair logic
# ─────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|bash|sh|shell|js|json)?\s*\n([\s\S]+?)```",
    flags=re.IGNORECASE,
)


def extract_code_block(text: str) -> str | None:
    """Pull the first fenced code block out of `text`. Many small models
    write the script body in a markdown ``` block right before calling
    Write — so when `content` is missing, this is where it usually is.
    """
    if not text:
        return None
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).rstrip() if m else None


def extract_first_command(text: str) -> str | None:
    """For Bash command repair: prefer fenced block, else first non-empty
    line that looks like a shell command."""
    block = extract_code_block(text)
    if block:
        return block
    if not text:
        return None
    for line in text.strip().splitlines():
        s = line.strip().lstrip("$ ").strip()
        if s and not s.startswith("#"):
            return s
    return None


def _missing_or_undefined(args: dict, key: str) -> bool:
    """A required arg counts as missing if absent, None, the string
    literal 'undefined', or empty-string."""
    if key not in args:
        return True
    v = args[key]
    if v is None:
        return True
    if isinstance(v, str) and v.strip() in ("", "undefined"):
        return True
    return False


def repair_tool_call(tool_call: dict, message_text: str,
                      tool_defs: dict[str, dict]) -> dict | None:
    """Inspect one tool_call, repair its arguments in-place if needed.
    Returns the repair record (or None if no repair was made)."""
    fn = tool_call.get("function", {})
    name = fn.get("name")
    if not name:
        return None
    raw_args = fn.get("arguments")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        args = {}

    tdef = tool_defs.get(name)
    if not tdef:
        return None
    params = tdef.get("parameters") or {}
    required = params.get("required") or []
    if not required:
        return None

    missing = [k for k in required if _missing_or_undefined(args, k)]
    if not missing:
        return None

    repairs: list[str] = []

    # Tool-specific recovery heuristics
    if name == "Write":
        if _missing_or_undefined(args, "content"):
            recovered = extract_code_block(message_text) or ""
            if not recovered:
                # No code block to extract — write a placeholder so the
                # tool call at least succeeds. Better than the agent
                # looping on a validation error.
                recovered = (
                    "# proxy-injected placeholder — original tool call "
                    "had no `content` field\n"
                )
            args["content"] = recovered
            repairs.append("Write.content")
        if _missing_or_undefined(args, "file_path"):
            args["file_path"] = "simulation.py"
            repairs.append("Write.file_path")
    elif name == "Bash":
        # Coerce string booleans (e.g. "True"/"False") on bool fields.
        # Llama-3.1-8B emits "run_in_background": "True" which trips the
        # Bash tool validator (expects boolean). Repair before forwarding.
        for _bk in ('run_in_background', 'sandbox'):
            _bv = args.get(_bk)
            if isinstance(_bv, str):
                _norm = _bv.strip().lower()
                if _norm in ('true', '1', 'yes'):
                    args[_bk] = True; repairs.append(f'Bash.{_bk}=bool')
                elif _norm in ('false', '0', 'no', ''):
                    args[_bk] = False; repairs.append(f'Bash.{_bk}=bool')
                else:
                    args.pop(_bk, None); repairs.append(f'Bash.{_bk}=dropped')
        if _missing_or_undefined(args, "command"):
            # Be conservative: extracting a shell command from prose text
            # is unreliable, and running the wrong command is worse than
            # running nothing. If the assistant put a fenced code block,
            # detect language: python → wrap with `python3 -c`; shell →
            # use as-is; otherwise no-op.
            block = extract_code_block(message_text)
            lang_match = re.search(r"```(\w+)", message_text or "")
            lang = lang_match.group(1).lower() if lang_match else ""
            if block and lang in ("bash", "sh", "shell", ""):
                # Treat untagged or shell-tagged blocks as shell commands.
                # First non-empty line only — multi-line shell can break.
                cmd = block.strip().splitlines()[0] if block.strip() else ""
                args["command"] = cmd or "echo 'proxy: empty command'"
            elif block and lang == "python":
                # Write the block to a temp file and run it. This makes
                # the Bash call functionally equivalent to a Write+Bash.
                # PROXYEOF heredoc terminator avoids shell-quoting issues.
                args["command"] = (
                    f"cat > /tmp/proxy_inline.py <<'PROXYEOF'\n"
                    f"{block}\nPROXYEOF\n"
                    f"python3 /tmp/proxy_inline.py"
                )
            else:
                args["command"] = "echo 'proxy: empty command'"
            repairs.append("Bash.command")
    elif name == "Edit":
        # Edit needs file_path + old_string + new_string. Hard to repair
        # cleanly from text — fall back to a no-op edit that doesn't
        # crash the harness.
        for k in ("file_path", "old_string", "new_string"):
            if _missing_or_undefined(args, k):
                args[k] = args.get(k) or ""
                repairs.append(f"Edit.{k}")
    elif name == "Read":
        if _missing_or_undefined(args, "file_path"):
            args["file_path"] = "."
            repairs.append("Read.file_path")
    else:
        # Generic fallback: fill missing string-typed required args with empty.
        schema = params.get("properties") or {}
        for k in missing:
            ks = schema.get(k, {}) or {}
            if ks.get("type") == "string":
                args[k] = ""
                repairs.append(f"{name}.{k}")
            else:
                # Don't fabricate non-string values.
                return None

    if not repairs:
        return None
    fn["arguments"] = json.dumps(args)
    return {
        "ts": time.time(),
        "tool": name,
        "missing": missing,
        "repaired_fields": repairs,
        "had_text_block": bool(extract_code_block(message_text)),
    }


# Two markup formats observed from Gemma 3/4 without --enable-auto-tool-choice:
#   <|tool_call>name(args)<tool_call|>      — older format
#   call:name{k: "v", k2: 42}               — current format (cycle13 evidence)
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|?>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*<\/?tool_call\|?>",
    flags=re.DOTALL,
)
_GEMMA_CALL_BRACE_RE = re.compile(
    r"\bcall:([a-zA-Z_][a-zA-Z0-9_]*)\s*\{([^}]*)\}",
    flags=re.DOTALL,
)


def _parse_gemma_args(raw: str) -> dict:
    """Best-effort parse of `k=v, k=v` or `k:v, k:v`. Quoted strings get
    quotes stripped. Numeric-looking values become int/float. Anything
    else stays str."""
    out: dict = {}
    # Split on commas NOT inside brackets
    for part in re.split(r",(?![^\[\]{}]*[\]\}])", raw):
        if "=" in part:
            k, v = part.split("=", 1)
        elif ":" in part:
            k, v = part.split(":", 1)
        else:
            continue
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        try:
            out[k] = int(v) if "." not in v else float(v)
        except ValueError:
            out[k] = v
    return out


def _parse_gemma_markup_to_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Parse Gemma's native tool-call markup into an OpenAI tool_calls list.
    Handles both observed formats: `<|tool_call>name(args)<tool_call|>` and
    `call:name{k: "v"}`. Returns (cleaned_content, tool_calls) where
    cleaned_content has the markup stripped."""
    tool_calls: list[dict] = []
    cleaned = content
    idx = 0
    for m in _GEMMA_TOOL_CALL_RE.finditer(content):
        name, raw_args = m.group(1), m.group(2)
        tool_calls.append({
            "id": f"gemma_call_{idx}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(_parse_gemma_args(raw_args))},
        })
        cleaned = cleaned.replace(m.group(0), "")
        idx += 1
    for m in _GEMMA_CALL_BRACE_RE.finditer(content):
        name, raw_args = m.group(1), m.group(2)
        tool_calls.append({
            "id": f"gemma_call_{idx}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(_parse_gemma_args(raw_args))},
        })
        cleaned = cleaned.replace(m.group(0), "")
        idx += 1
    return cleaned.strip(), tool_calls


# ─────────────────────────────────────────────────────────────
# Plain-text → tool_call synthesis shim (PROXY_AUTO_TOOLS=1)
# ─────────────────────────────────────────────────────────────
# For models that don't emit OpenAI-format tool_calls at all (DeepSeek-
# Coder-V2-Lite, Phi-4, Granite-3.3-8B) we synthesize a one-shot
# Write(simulation.py) + Bash(python3 simulation.py) sequence from the
# first fenced ```python``` block in the assistant text. Default off; only
# activates when env PROXY_AUTO_TOOLS is set to a truthy value.

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", re.DOTALL)
_SYNTH_COUNTER = 0


def _auto_tools_enabled() -> bool:
    v = os.environ.get("PROXY_AUTO_TOOLS", "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def synthesize_tool_calls_from_text(body: dict) -> dict:
    """If the assistant emitted plain text with a ```python``` block AND no
    tool_calls, synthesize Write(simulation.py)+Bash(python3 simulation.py).

    Fallback: if no python block but one or more ```bash/sh/shell``` blocks
    exist (Granite-3.3-8B failure mode — emits `cp template_ber.py
    simulation.py` then `python3 simulation.py` in two separate bash blocks
    without ever running anything), concatenate all bash blocks into a single
    Bash tool call so the agent at least executes the commands. The verifier
    then has a chance to find the expected artifact.
    """
    global _SYNTH_COUNTER
    for choice in body.get("choices") or []:
        msg = choice.get("message") or {}
        if msg.get("tool_calls"):
            continue
        text = msg.get("content")
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text
                            if isinstance(p, dict))
        if not isinstance(text, str) or not text:
            continue

        m = _PYTHON_BLOCK_RE.search(text)
        if m:
            code = m.group(1).rstrip()
            if not code.strip():
                continue
            cid = _SYNTH_COUNTER
            _SYNTH_COUNTER += 2
            msg["tool_calls"] = [
                {"id": f"call_synth_{cid}", "type": "function",
                 "function": {"name": "Write",
                              "arguments": json.dumps({
                                  "file_path": "simulation.py",
                                  "content": code})}},
                {"id": f"call_synth_{cid + 1}", "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({
                                  "command": "python3 simulation.py",
                                  "description": "Run the synthesized simulation"})}},
            ]
            msg["content"] = None
            choice["finish_reason"] = "tool_calls"
            print(f"[proxy-autotools] synthesized Write+Bash (code_len={len(code)})",
                  file=sys.stderr, flush=True)
            continue

        bash_blocks = _BASH_BLOCK_RE.findall(text)
        if bash_blocks:
            cmd = " && ".join(b.strip() for b in bash_blocks if b.strip())
            if not cmd:
                continue
            cid = _SYNTH_COUNTER
            _SYNTH_COUNTER += 1
            msg["tool_calls"] = [
                {"id": f"call_synth_{cid}", "type": "function",
                 "function": {"name": "Bash",
                              "arguments": json.dumps({
                                  "command": cmd,
                                  "description": "Run concatenated bash blocks"})}},
            ]
            msg["content"] = None
            choice["finish_reason"] = "tool_calls"
            print(f"[proxy-autotools] synthesized Bash from {len(bash_blocks)} bash block(s)",
                  file=sys.stderr, flush=True)
    return body


# ─────────────────────────────────────────────────────────────
# EDIT-TAG: mixtral-template-fix (2026-05) — role coalescing
# ─────────────────────────────────────────────────────────────
# Mistral/Mixtral's stock chat template enforces strict user/assistant
# alternation and bans system/tool roles. openclaude routinely sends
# consecutive same-role turns (system+user, assistant text after assistant
# tool_call, two consecutive tool results). When PROXY_COALESCE_ROLES=1,
# we pre-process the request's `messages` list to:
#   - Concatenate consecutive same-role messages' text content with "\n\n"
#   - Merge consecutive assistant `tool_calls` arrays
#   - Convert `role=tool` messages into a `user` message that wraps the
#     payload (so the strict template can render it without raising)
#   - Lift `system` content into the first `user` message
# This runs ONLY when the env var is set, so other models are untouched.


def _flatten_content(c: Any) -> str:
    """OpenAI content can be a string or a list of parts ({type,text}).
    Flatten to a single string for concatenation."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "\n\n".join(parts)
    return str(c)


def coalesce_messages_for_mistral(messages: list[dict]) -> list[dict]:
    """Reshape messages so they satisfy Mistral's strict alternation rules.

    Steps:
      1. Lift all `system` content into a buffer prepended to the next user.
      2. Convert `tool` messages into a `user` message wrapping the payload
         as ``[TOOL_RESULTS] {...} [/TOOL_RESULTS]``.
      3. Coalesce consecutive same-role messages: text joined by "\n\n";
         tool_calls arrays concatenated.
    """
    if not messages:
        return messages
    sys_buf = ""
    intermediate: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = _flatten_content(m.get("content"))
        if role == "system":
            sys_buf = (sys_buf + "\n\n" + content).strip() if sys_buf else content
            continue
        if role == "tool":
            payload = json.dumps({
                "tool_call_id": m.get("tool_call_id", ""),
                "content": content,
            })
            intermediate.append({
                "role": "user",
                "content": f"[TOOL_RESULTS] {payload} [/TOOL_RESULTS]",
            })
            continue
        new_m: dict = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            new_m["tool_calls"] = list(m.get("tool_calls") or [])
        if role == "user" and sys_buf:
            new_m["content"] = sys_buf + "\n\n" + new_m["content"]
            sys_buf = ""
        intermediate.append(new_m)

    # Stray trailing system with no user after it: prepend to first user.
    if sys_buf:
        for m in intermediate:
            if m["role"] == "user":
                m["content"] = sys_buf + "\n\n" + m["content"]
                sys_buf = ""
                break
        if sys_buf:
            intermediate.insert(0, {"role": "user", "content": sys_buf})

    # Coalesce consecutive same-role.
    out: list[dict] = []
    for m in intermediate:
        if out and out[-1]["role"] == m["role"]:
            prev = out[-1]
            a = prev.get("content") or ""
            b = m.get("content") or ""
            if a and b:
                prev["content"] = a + "\n\n" + b
            else:
                prev["content"] = a or b
            if m["role"] == "assistant":
                merged = (prev.get("tool_calls") or []) + (m.get("tool_calls") or [])
                if merged:
                    prev["tool_calls"] = merged
        else:
            out.append(m)
    return out


def _maybe_coalesce_request(request_body: dict) -> None:
    if os.environ.get("PROXY_COALESCE_ROLES") != "1":
        return
    msgs = request_body.get("messages")
    if not isinstance(msgs, list):
        return
    try:
        request_body["messages"] = coalesce_messages_for_mistral(msgs)
    except Exception as e:
        print(f"[proxy-coalesce] error: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def repair_response_body(body: dict, request_body: dict) -> dict:
    """Walk all choices in the response, repair tool calls. Modifies
    `body` in place AND returns it."""
    tools_list = request_body.get("tools") or []
    tool_defs: dict[str, dict] = {}
    for t in tools_list:
        if isinstance(t, dict):
            fn = t.get("function") or t
            name = fn.get("name")
            if name:
                tool_defs[name] = fn

    for choice in body.get("choices") or []:
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []
        for tc in tcs:
            rec = repair_tool_call(tc, text, tool_defs)
            if rec is not None:
                REPAIR_LOG.append(rec)
                print(f"[proxy-repair] {rec}", file=sys.stderr, flush=True)
    return body


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "repairs_count": len(REPAIR_LOG),
            "last_5_repairs": list(REPAIR_LOG)[-5:]}


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{UPSTREAM}/v1/models")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type",
                                                  "application/json"))


_DEBUG_LOG = os.environ.get("PROXY_DEBUG_LOG")


def _debug_dump(label: str, payload: dict | str | bytes) -> None:
    if not _DEBUG_LOG:
        return
    try:
        if isinstance(payload, (dict, list)):
            text = json.dumps(payload)[:8000]
        elif isinstance(payload, bytes):
            text = payload[:8000].decode(errors="replace")
        else:
            text = str(payload)[:8000]
        with open(_DEBUG_LOG, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {label}: {text}\n")
    except Exception:
        pass


def _reconstruct_sse(repaired: dict):
    """Convert a single chat-completion JSON response into an SSE-format
    chunk stream that mimics OpenAI's streaming delta protocol.

    Generator yields bytes chunks compatible with openclaude's parser:
    delta with role, delta with content (if any), delta with tool_calls
    (if any), finish_reason chunk, and finally [DONE]."""
    choice = (repaired.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    base_meta = {
        "id": repaired.get("id", "proxy-reconstructed"),
        "object": "chat.completion.chunk",
        "model": repaired.get("model", ""),
    }

    def chunk(delta: dict, finish_reason=None):
        c = {**base_meta,
             "choices": [{"index": 0, "delta": delta,
                          "finish_reason": finish_reason}]}
        yield f"data: {json.dumps(c)}\n\n".encode()

    # 1. role chunk
    yield from chunk({"role": msg.get("role", "assistant")})
    # 2. content chunk (if any)
    if msg.get("content"):
        yield from chunk({"content": msg["content"]})
    # 3. tool_calls chunk (if any)
    if msg.get("tool_calls"):
        yield from chunk({"tool_calls": msg["tool_calls"]})
    # 4. final chunk with finish_reason
    yield from chunk({}, finish_reason=choice.get("finish_reason", "stop"))
    # 5. [DONE] marker per OpenAI SSE spec
    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body_bytes = await req.body()
    try:
        request_body = json.loads(body_bytes) if body_bytes else {}
    except Exception:
        request_body = {}

    is_stream = bool(request_body.get("stream"))
    _debug_dump(f"REQ stream={is_stream}", request_body)

    is_stream_requested = bool(request_body.get("stream"))
    # Force non-streaming to vLLM. vLLM's streaming path with
    # --enable-auto-tool-choice can hang on Gemma's tool-call grammar
    # (cycles 6-10 reproduced this empirically). Single-JSON responses
    # work reliably at any prompt size up to max_model_len.
    request_body["stream"] = False

    cap = os.environ.get("PROXY_MAX_TOKENS_CAP")
    if cap:
        try:
            cap_int = int(cap)
            current = request_body.get("max_tokens", cap_int + 1)
            if not isinstance(current, int) or current > cap_int:
                request_body["max_tokens"] = cap_int
        except ValueError:
            pass

    # H.3: When vLLM runs without --enable-auto-tool-choice (Gemma plain-text
    # mode), it rejects requests that include `tools` or `tool_choice`.
    # Strip those fields so vLLM accepts the request; the model emits its
    # native <|tool_call> markup in `content`, which the post-hoc parser
    # below converts to tool_calls.
    if os.environ.get("PROXY_STRIP_TOOLS"):
        request_body.pop("tools", None)
        request_body.pop("tool_choice", None)

    # EDIT-TAG: mixtral-template-fix (2026-05) — Mistral-family request
    # coalescing under env-var gate. Reshapes `messages` to satisfy the
    # strict-alternation chat template before forwarding to vLLM. No-op when
    # PROXY_COALESCE_ROLES != "1".
    _maybe_coalesce_request(request_body)

    # P0.3: clamp max_tokens vs. estimated prompt length so prompt +
    # completion fits in max_model_len. Best-effort char-based estimate;
    # see _count_prompt_tokens. Runs LAST so it sees the final request
    # shape (after any earlier PROXY_MAX_TOKENS_CAP / strip-tools mutation).
    _clamp_max_tokens(request_body)

    body_bytes = json.dumps(request_body).encode()

    headers = {k: v for k, v in req.headers.items()
               if k.lower() not in ("host", "content-length")}

    try:
        async with httpx.AsyncClient(timeout=1800) as client:
            r = await client.post(f"{UPSTREAM}/v1/chat/completions",
                                  content=body_bytes, headers=headers)
    except Exception as e:
        _debug_dump("UPSTREAM_EXCEPTION", f"{type(e).__name__}: {e}")
        print(f"[proxy] upstream error: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return JSONResponse(
            content={"error": {"type": "upstream_error",
                               "message": f"{type(e).__name__}: {e}"}},
            status_code=502)

    _debug_dump(f"UPSTREAM status={r.status_code} stream_requested={is_stream_requested}",
                r.content)

    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type",
                                                  "application/json"))

    try:
        upstream_body = r.json()
    except Exception:
        return Response(content=r.content, status_code=200,
                        media_type=r.headers.get("content-type",
                                                  "application/json"))

    # Capture token usage from the ORIGINAL upstream body BEFORE any
    # mutation (tool-call repair, Gemma markup conversion, etc.). vLLM's
    # `usage` block is the source of truth for what the GPU actually
    # processed; aggregator joins these records back to trials by ts.
    _record_usage(upstream_body, os.environ.get("PROXY_PORT", "8101"))

    # P0.5: vLLM bug — when completion_tokens hits max_tokens cap, it
    # reports finish_reason=tool_calls instead of "length". Remap so the
    # harness's truncation-retry logic fires. Runs AFTER _record_usage so
    # captured usage counts are untouched.
    _remap_truncation(upstream_body, request_body)

    repaired = repair_response_body(upstream_body, request_body)

    # Without --enable-auto-tool-choice, Gemma emits native markup in
    # `content`. Convert to tool_calls so openclaude sees the proper shape.
    for choice in repaired.get("choices") or []:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if "<|tool_call" in content:
            cleaned, calls = _parse_gemma_markup_to_tool_calls(content)
            if calls:
                msg["content"] = cleaned or None
                existing = msg.get("tool_calls") or []
                msg["tool_calls"] = existing + calls
                if not cleaned:
                    choice["finish_reason"] = "tool_calls"

    # PROXY_AUTO_TOOLS shim: for plain-text models (DeepSeek-Coder-V2-Lite,
    # Phi-4, Granite-3.3-8B) that never emit structured tool_calls, harvest
    # the first ```python block and synthesize Write+Bash. Default off.
    if _auto_tools_enabled():
        synthesize_tool_calls_from_text(repaired)

    if not is_stream_requested:
        return JSONResponse(content=repaired, status_code=200)

    # Reconstruct an SSE reply from the single JSON. openclaude expects
    # OpenAI's incremental delta format: chunks of {"choices":[{"delta":...}]}
    # then a final [DONE] marker.
    return StreamingResponse(_reconstruct_sse(repaired),
                             media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default="http://127.0.0.1:8001")
    args = ap.parse_args()

    global UPSTREAM
    UPSTREAM = args.upstream.rstrip("/")
    print(f"[proxy] starting on {args.host}:{args.port} → {UPSTREAM}",
          file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
