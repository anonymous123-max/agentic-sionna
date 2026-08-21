# benchmark/tests/test_proxy_force_nonstream.py
"""Proxy must force stream=false to vLLM regardless of openclaude's
request, then reconstruct an SSE-formatted reply when streaming was
requested."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import pytest
import tool_call_proxy
from fastapi.testclient import TestClient


@pytest.fixture
def fake_upstream():
    """Patch httpx.AsyncClient.post to return a synthetic vLLM response."""
    captured = {}

    async def fake_post(self, url, content=None, headers=None):
        captured["body"] = json.loads(content)
        resp = MagicMock()
        resp.status_code = 200
        resp.content = json.dumps({
            "id": "test-id", "object": "chat.completion",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi.",
                            "tool_calls": []},
                "finish_reason": "stop",
            }],
        }).encode()
        resp.json = MagicMock(return_value=json.loads(resp.content))
        resp.headers = {"content-type": "application/json"}
        return resp

    with patch("httpx.AsyncClient.post", new=fake_post):
        yield captured


def test_streaming_request_forwarded_with_stream_false(fake_upstream):
    """When openclaude sends stream=true, proxy must POST stream=false to vLLM."""
    client = TestClient(tool_call_proxy.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}],
                          "stream": True})
    assert r.status_code == 200
    assert fake_upstream["body"]["stream"] is False, (
        f"proxy did not force stream=false; sent: {fake_upstream['body']}")


def test_nonstreaming_request_returns_json(fake_upstream):
    client = TestClient(tool_call_proxy.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hi."


def test_streaming_request_returns_sse_format(fake_upstream):
    client = TestClient(tool_call_proxy.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}],
                          "stream": True})
    assert r.status_code == 200
    body = r.text
    # SSE format: each chunk is "data: {...}\n\n"; final line is "data: [DONE]\n\n"
    assert body.startswith("data: "), f"not SSE-formatted: {body[:200]!r}"
    assert "data: [DONE]" in body, f"missing [DONE] marker: {body[-200:]!r}"
    # Parse one mid-stream chunk
    chunks = [line for line in body.split("\n\n") if line.startswith("data: ") and "[DONE]" not in line]
    assert len(chunks) >= 2, "expected at least role+content chunks"
    first = json.loads(chunks[0].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert "delta" in first["choices"][0]


def test_streaming_with_tool_calls_emits_tool_calls_chunk(fake_upstream, monkeypatch):
    """If the upstream response has tool_calls, the SSE stream must emit a
    chunk with delta.tool_calls."""
    async def fake_post_with_tools(self, url, content=None, headers=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = json.dumps({
            "id": "t", "object": "chat.completion", "model": "x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "calc",
                                                         "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }],
        }).encode()
        resp.json = MagicMock(return_value=json.loads(resp.content))
        resp.headers = {"content-type": "application/json"}
        return resp

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post_with_tools)
    client = TestClient(tool_call_proxy.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}],
                          "stream": True})
    assert r.status_code == 200
    assert "tool_calls" in r.text, "SSE stream missing tool_calls chunk"
    assert "calc" in r.text, "tool_calls chunk missing function name"
