"""When PROXY_MAX_TOKENS_CAP is set, proxy clamps max_tokens before
forwarding to vLLM. Used as a debug knob to force termination on hung
generations."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import tool_call_proxy
from fastapi.testclient import TestClient


def _fake_post_factory(captured: dict):
    async def fake_post(self, url, content=None, headers=None):
        captured["body"] = json.loads(content)
        r = MagicMock()
        r.status_code = 200
        r.content = b'{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok","tool_calls":[]},"finish_reason":"stop"}],"model":"x","object":"chat.completion"}'
        r.json = MagicMock(return_value=json.loads(r.content))
        r.headers = {"content-type": "application/json"}
        return r
    return fake_post


def test_cap_applied_when_env_set(monkeypatch):
    monkeypatch.setenv("PROXY_MAX_TOKENS_CAP", "256")
    captured: dict = {}
    with patch("httpx.AsyncClient.post", new=_fake_post_factory(captured)):
        client = TestClient(tool_call_proxy.app)
        client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9999})
    assert captured["body"]["max_tokens"] == 256


def test_cap_not_applied_without_env(monkeypatch):
    monkeypatch.delenv("PROXY_MAX_TOKENS_CAP", raising=False)
    captured: dict = {}
    with patch("httpx.AsyncClient.post", new=_fake_post_factory(captured)):
        client = TestClient(tool_call_proxy.app)
        client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9999})
    assert captured["body"]["max_tokens"] == 9999


def test_cap_does_not_lower_already_smaller_value(monkeypatch):
    """If the request asks for fewer tokens than the cap, leave it alone."""
    monkeypatch.setenv("PROXY_MAX_TOKENS_CAP", "256")
    captured: dict = {}
    with patch("httpx.AsyncClient.post", new=_fake_post_factory(captured)):
        client = TestClient(tool_call_proxy.app)
        client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50})
    assert captured["body"]["max_tokens"] == 50  # cap doesn't raise it


def test_invalid_cap_value_silently_ignored(monkeypatch):
    """Bogus PROXY_MAX_TOKENS_CAP shouldn't crash the proxy."""
    monkeypatch.setenv("PROXY_MAX_TOKENS_CAP", "not-a-number")
    captured: dict = {}
    with patch("httpx.AsyncClient.post", new=_fake_post_factory(captured)):
        client = TestClient(tool_call_proxy.app)
        client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9999})
    assert captured["body"]["max_tokens"] == 9999  # unchanged
