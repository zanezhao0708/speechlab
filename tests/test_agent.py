"""Tests for the LLM agent: config, tool specs and the tool-execution loop.

No network access is required — the transport is exercised through a fake
``_chat_request`` that scripts a tool call followed by a final answer.
"""

import json

import pytest

from speechlab.agent import (
    RESEARCH_SYSTEM_PROMPT,
    AgentConfig,
    SpeechResearchAgent,
    build_tool_specs,
)

from .helpers import vowel_like, write_wav


def test_tool_specs_valid():
    specs = build_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"analyze_audio", "audio_quality", "dataset_report"}
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


def test_local_tools_analyze_audio(tmp_path):
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = agent.tools["analyze_audio"]({"path": str(path)})
    assert result["sample_rate_hz"] == 16000
    assert result["pitch"]["n_voiced_frames"] > 0


def test_local_tools_audio_quality(tmp_path):
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=1.0), 16000)
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = agent.tools["audio_quality"]({"path": str(path)})
    assert "ok" in result and "issues" in result


def test_local_tools_dataset_report(tmp_path):
    write_wav(str(tmp_path / "spk01_a.wav"), vowel_like(duration_s=0.2), 16000)
    write_wav(str(tmp_path / "spk02_a.wav"), vowel_like(duration_s=0.2), 16000)
    agent = SpeechResearchAgent(AgentConfig(api_key="sk-test", model="test"))
    result = agent.tools["dataset_report"]({"path": str(tmp_path)})
    assert result["n_files"] == 2
    assert result["n_speakers"] == 2


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
    calls: list[str] = []

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
    calls.append(answer)
    assert "F0 median is" in answer
    # history contains system, user, assistant(tool call), tool, final assistant
    roles = [m["role"] for m in agent.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_system_prompt_content():
    assert "speech" in RESEARCH_SYSTEM_PROMPT.lower()
    assert "tools" in RESEARCH_SYSTEM_PROMPT.lower()
