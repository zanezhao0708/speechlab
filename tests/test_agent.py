"""Tests for the LLM agent: config, tool specs and the tool-execution loop.

No network access is required — the transport is exercised through a fake
``_chat_request`` that scripts a tool call followed by a final answer.
"""

import json
import urllib.error

import pytest

from speechlab.agent import (
    RESEARCH_SYSTEM_PROMPT,
    AgentConfig,
    SpeechResearchAgent,
    build_tool_specs,
)

from .helpers import vowel_like, write_wav


class _FakeResponse:
    """Minimal stand-in for the object returned by urllib.urlopen."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_tool_specs_valid():
    specs = build_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"analyze_audio", "compare_audio", "diarize_audio",
                     "transcribe_audio", "reference_ranges"}
    for s in specs:
        assert s["type"] == "function"
        json.dumps(s)  # serialisable


def test_config_env_defaults(monkeypatch):
    monkeypatch.delenv("SPEECHLAB_API_KEY", raising=False)
    cfg = AgentConfig()
    with pytest.raises(RuntimeError):
        cfg.validate()
    cfg.api_key = "sk-test"
    cfg.validate()


def test_agent_requires_key(monkeypatch):
    monkeypatch.delenv("SPEECHLAB_API_KEY", raising=False)
    agent = SpeechResearchAgent(AgentConfig(api_key="", model="test"))
    with pytest.raises(RuntimeError):
        agent.ask("hello")


def test_local_tool_analyze_audio(tmp_path):
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = agent.tools["analyze_audio"]({"path": str(path)})
    assert result["sample_rate_hz"] == 16000
    assert result["pitch"]["n_voiced_frames"] > 0


def test_tool_error_is_reported_not_raised(tmp_path):
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "analyze_audio", "arguments": json.dumps(
            {"path": str(tmp_path / "missing.wav")})},
    }))
    assert "error" in result


def test_unknown_tool():
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "does_not_exist", "arguments": "{}"},
    }))
    assert "error" in result


def test_agent_loop_with_fake_transport(tmp_path):
    """Script a tool call round-trip without touching the network."""
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)

    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="fake"))

    def fake_chat(messages, tools):
        # first round: model asks to analyse the file
        if not any(m.get("role") == "tool" for m in messages):
            return {"choices": [{
                "finish_reason": "tool_calls",
                "message": {"content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "analyze_audio",
                                 "arguments": json.dumps({"path": str(path)})},
                }]},
            }]}
        # second round: model answers using the tool result
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        payload = json.loads(tool_msg["content"])
        return {"choices": [{
            "finish_reason": "stop",
            "message": {"content": f"F0 median is {payload['pitch']['f0_median_hz']:.1f} Hz"},
        }]}

    agent._chat_request = fake_chat  # type: ignore[method-assign]
    answer = agent.ask("Analyse this file", context=None)
    assert "F0 median is" in answer
    # history contains system, user, assistant(tool call), tool, final assistant
    roles = [m["role"] for m in agent.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_system_prompt_content():
    assert "speech" in RESEARCH_SYSTEM_PROMPT.lower()
    assert "analyze_audio" in RESEARCH_SYSTEM_PROMPT


# ---------------------------------------------------------------- multi-turn
def test_agent_multi_turn_keeps_history():
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="fake"))

    def fake_chat(messages, tools):
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": "ok"}}]}

    agent._chat_request = fake_chat  # type: ignore[method-assign]
    agent.ask("first question")
    agent.ask("second question")
    roles = [m["role"] for m in agent.history]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert any(m.get("content") == "first question" for m in agent.history)


def test_agent_reset_clears_history_and_cache():
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="fake"))
    agent.history = [{"role": "system", "content": "x"}]
    agent._tool_cache["k"] = "v"
    agent.reset()
    assert agent.history == []
    assert agent._tool_cache == {}


# --------------------------------------------------------------------- retry
def test_chat_request_retries_on_429(monkeypatch):
    agent = SpeechResearchAgent(AgentConfig(api_key="sk", model="m",
                                            retry_backoff_s=0.0))
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                         None, None)
        return _FakeResponse({"choices": [{"finish_reason": "stop",
                                           "message": {"content": "hi"}}]})

    monkeypatch.setattr("speechlab.agent.urllib.request.urlopen", fake_urlopen)
    resp = agent._chat_request([{"role": "user", "content": "q"}], None)
    assert calls["n"] == 2
    assert resp["choices"][0]["message"]["content"] == "hi"


def test_chat_request_does_not_retry_auth_errors(monkeypatch):
    agent = SpeechResearchAgent(AgentConfig(api_key="sk", model="m",
                                            retry_backoff_s=0.0))
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized",
                                     None, None)

    monkeypatch.setattr("speechlab.agent.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        agent._chat_request([{"role": "user", "content": "q"}], None)
    assert calls["n"] == 1  # raised immediately, no retries


def test_chat_request_gives_up_after_max_retries(monkeypatch):
    agent = SpeechResearchAgent(AgentConfig(api_key="sk", model="m",
                                            max_retries=2, retry_backoff_s=0.0))
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("speechlab.agent.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        agent._chat_request([{"role": "user", "content": "q"}], None)
    assert calls["n"] == 3  # initial attempt + 2 retries


# -------------------------------------------------------------- tool caching
def test_tool_result_cached():
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    calls = {"n": 0}

    def counting_analyze(args):
        calls["n"] += 1
        return {"ok": True}

    agent.tools = {"analyze_audio": counting_analyze}
    call = {"id": "c1", "type": "function",
            "function": {"name": "analyze_audio",
                         "arguments": json.dumps({"path": "/tmp/x.wav"})}}
    first = agent._execute_tool(call)
    second = agent._execute_tool(call)
    assert calls["n"] == 1
    assert first == second


def test_tool_cache_ignores_errors():
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    calls = {"n": 0}

    def failing(args):
        calls["n"] += 1
        raise FileNotFoundError("missing.wav")

    agent.tools = {"analyze_audio": failing}
    call = {"id": "c1", "type": "function",
            "function": {"name": "analyze_audio",
                         "arguments": json.dumps({"path": "missing.wav"})}}
    r1 = json.loads(agent._execute_tool(call))
    r2 = json.loads(agent._execute_tool(call))
    assert calls["n"] == 2  # errors are not cached — file may appear later
    assert "error" in r1 and "error" in r2
