"""Parse Gemma's native tool-call markup into OpenAI tool_calls."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

from tool_call_proxy import _parse_gemma_markup_to_tool_calls


def test_simple_call():
    cleaned, calls = _parse_gemma_markup_to_tool_calls(
        '<|tool_call>calc(a=2, b=2)<tool_call|>')
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "calc"
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 2}
    assert cleaned == ""


def test_string_args():
    _, calls = _parse_gemma_markup_to_tool_calls(
        '<|tool_call>bash(command="echo hi")<tool_call|>')
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": "echo hi"}


def test_text_around_call():
    cleaned, calls = _parse_gemma_markup_to_tool_calls(
        'I will use calc.\n<|tool_call>calc(a=1, b=2)<tool_call|>\nDone.')
    assert len(calls) == 1
    assert "I will use calc" in cleaned
    assert "Done." in cleaned


def test_multiple_calls():
    _, calls = _parse_gemma_markup_to_tool_calls(
        '<|tool_call>read(file_path="x")<tool_call|>'
        '<|tool_call>read(file_path="y")<tool_call|>')
    assert len(calls) == 2
    assert calls[0]["id"] == "gemma_call_0"
    assert calls[1]["id"] == "gemma_call_1"


def test_no_markup_returns_empty_calls_list():
    cleaned, calls = _parse_gemma_markup_to_tool_calls('Just plain text.')
    assert calls == []
    assert cleaned == 'Just plain text.'


def test_float_args():
    _, calls = _parse_gemma_markup_to_tool_calls(
        '<|tool_call>place(x=1.5, y=2.0)<tool_call|>')
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"x": 1.5, "y": 2.0}


def test_call_brace_format():
    """Gemma 4 31B (cycle13 evidence) emits `call:name{k: "v"}` format."""
    cleaned, calls = _parse_gemma_markup_to_tool_calls(
        'call:bash{command: "python3 hello.py"}')
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": "python3 hello.py"}
    assert cleaned == ""


def test_call_brace_text_around():
    cleaned, calls = _parse_gemma_markup_to_tool_calls(
        'I will run the script. call:bash{command: "ls -la"} Done.')
    assert len(calls) == 1
    assert "I will run" in cleaned
    assert "Done." in cleaned


def test_call_brace_multiple():
    _, calls = _parse_gemma_markup_to_tool_calls(
        'call:read{file_path: "a"} then call:read{file_path: "b"}')
    assert len(calls) == 2
    assert calls[0]["id"] == "gemma_call_0"
    assert calls[1]["id"] == "gemma_call_1"


def test_both_formats_in_same_response():
    cleaned, calls = _parse_gemma_markup_to_tool_calls(
        '<|tool_call>read(file_path="x")<tool_call|>'
        ' and call:bash{command: "ls"}')
    assert len(calls) == 2  # one of each format
    names = {c["function"]["name"] for c in calls}
    assert names == {"read", "bash"}
