"""Tests for the LLM agent: config, tool specs and the tool-execution loop.

No network access is required — the transport is exercised through a fake
``_chat_request`` that scripts a tool call followed by a final answer.
"""

import json
import os
import urllib.error

import pytest

from speechlab.agent import (
    RESEARCH_SYSTEM_PROMPT,
    AgentConfig,
    SpeechResearchAgent,
    build_tool_specs,
    redact_path,
    resolve_tool_path,
)

from .helpers import vowel_like, write_wav


def _agent(tmp_path=None, **cfg) -> SpeechResearchAgent:
    """Agent whose file tools may read *tmp_path* (or nowhere)."""
    if tmp_path is not None and "allowed_dirs" not in cfg:
        cfg["allowed_dirs"] = [str(tmp_path)]
    return SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test", **cfg))


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
    agent = _agent(tmp_path)
    result = agent.tools["analyze_audio"]({"path": str(path)})
    assert result["sample_rate_hz"] == 16000
    assert result["pitch"]["n_voiced_frames"] > 0
    assert result["file"] == os.path.realpath(str(path))


def test_tool_error_is_reported_not_raised(tmp_path):
    agent = _agent(tmp_path)
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

    agent = _agent(tmp_path)

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


# ----------------------------------------------------------- path sandbox
def test_resolve_tool_path_allows_inside(tmp_path):
    real = resolve_tool_path("utt.wav", [str(tmp_path)])
    assert real == str(tmp_path / "utt.wav") or real.endswith("utt.wav")
    assert str(tmp_path) in real


def test_resolve_tool_path_denies_outside(tmp_path):
    with pytest.raises(PermissionError):
        resolve_tool_path("/etc/passwd", [str(tmp_path)])


def test_resolve_tool_path_denies_traversal(tmp_path):
    with pytest.raises(PermissionError):
        resolve_tool_path(str(tmp_path / ".." / "secret.wav"), [str(tmp_path)])


def test_resolve_tool_path_denies_empty():
    with pytest.raises(ValueError):
        resolve_tool_path("", ["."])


def test_resolve_tool_path_symlink_escape(tmp_path):
    """A symlink inside the workspace pointing outside must not pass."""
    target = tmp_path.parent / "speechlab_sandbox_escape_target.wav"
    target.write_bytes(b"x")
    link = tmp_path / "innocent.wav"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    with pytest.raises(PermissionError):
        resolve_tool_path(str(link), [str(tmp_path)])


def test_resolve_tool_path_relative_uses_first_root(tmp_path):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "utt.wav"
    a.write_bytes(b"a")
    b = tmp_path / "sub" / "utt.wav"
    b.write_bytes(b"b")
    # "utt.wav" must bind to the first root, not the second
    assert resolve_tool_path("utt.wav", [str(tmp_path), str(tmp_path / "sub")]) \
        == str(a)


def test_tool_denies_file_outside_workspace(tmp_path):
    """The sandbox is enforced at tool level: /etc/passwd is never read."""
    agent = _agent(tmp_path)
    result = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "analyze_audio",
                     "arguments": json.dumps({"path": "/etc/passwd"})},
    }))
    assert "error" in result
    assert "outside the allowed directories" in result["error"]


def test_tool_relative_path_resolves_in_workspace(tmp_path):
    write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)
    agent = _agent(tmp_path)
    result = agent.tools["analyze_audio"]({"path": "utt.wav"})
    assert result["sample_rate_hz"] == 16000


def test_allowed_dirs_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHLAB_ALLOWED_DIRS",
                       os.pathsep.join([str(tmp_path), "/data/audio"]))
    cfg = AgentConfig()
    assert cfg.allowed_dirs == [str(tmp_path), "/data/audio"]


def test_allowed_dirs_default_cwd(monkeypatch):
    monkeypatch.delenv("SPEECHLAB_ALLOWED_DIRS", raising=False)
    assert AgentConfig().allowed_dirs == [os.getcwd()]


# ---------------------------------------------------------- path redaction
def test_redact_path():
    assert redact_path("/home/alice/study/patient_042.wav") == "…/patient_042.wav"
    assert redact_path("utt.wav") == "…/utt.wav"
    assert redact_path(None) is None
    assert redact_path("") == ""


def test_tool_results_redacted(tmp_path):
    write_wav(str(tmp_path / "a.wav"), vowel_like(duration_s=0.5), 16000)
    write_wav(str(tmp_path / "b.wav"), vowel_like(duration_s=0.5, f0_hz=220), 16000)
    agent = _agent(tmp_path, redact_paths=True)

    report = agent.tools["analyze_audio"]({"path": "a.wav"})
    assert report["file"] == "…/a.wav"
    assert str(tmp_path) not in json.dumps(report)

    cmp = agent.tools["compare_audio"]({"path_a": "a.wav", "path_b": "b.wav"})
    assert cmp["files"] == ["…/a.wav", "…/b.wav"]


def test_tool_results_not_redacted_by_default(tmp_path):
    write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)
    agent = _agent(tmp_path)
    report = agent.tools["analyze_audio"]({"path": "utt.wav"})
    assert report["file"] == str(tmp_path / "utt.wav")


def test_redact_paths_from_env(monkeypatch):
    monkeypatch.setenv("SPEECHLAB_REDACT_PATHS", "1")
    assert AgentConfig().redact_paths is True
    monkeypatch.setenv("SPEECHLAB_REDACT_PATHS", "false")
    assert AgentConfig().redact_paths is False
    monkeypatch.delenv("SPEECHLAB_REDACT_PATHS", raising=False)
    assert AgentConfig().redact_paths is False


def test_missing_file_error_redacted(tmp_path):
    agent = _agent(tmp_path, redact_paths=True)
    result = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "analyze_audio",
                     "arguments": json.dumps(
                         {"path": str(tmp_path / "patient_042.wav")})},
    }))
    assert "error" in result
    assert str(tmp_path) not in result["error"]
    assert "patient_042.wav" in result["error"]


def test_load_error_message_redacted(tmp_path, monkeypatch):
    """Decoder errors that echo the path are redacted too, type preserved."""
    import speechlab.agent as agent_mod

    target = tmp_path / "utt.wav"
    target.write_bytes(b"not really audio")

    def boom(path, target_sr=None):
        raise RuntimeError(f"cannot decode {path}")

    monkeypatch.setattr(agent_mod, "load_audio", boom)
    agent = _agent(tmp_path, redact_paths=True)
    result = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "analyze_audio",
                     "arguments": json.dumps({"path": str(target)})},
    }))
    assert "error" in result
    assert result["error"].startswith("RuntimeError:")
    assert str(tmp_path) not in result["error"]
    assert "…/utt.wav" in result["error"]
