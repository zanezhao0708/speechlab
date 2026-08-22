"""Tests for the agent's streaming loop (ask_stream) and new local tools.

The transport is faked: `_stream_chat_request` yields scripted SSE chunk
dicts, so no network access is required.
"""

import json
import os
import urllib.error

from speechlab.agent import AgentConfig, SpeechResearchAgent

from .helpers import vowel_like, write_wav


def _mk_agent(tmp_path=None):
    cfg: dict = {"api_key": "sk-test", "model": "fake"}
    if tmp_path is not None:
        cfg["allowed_dirs"] = [str(tmp_path)]
    return SpeechResearchAgent(AgentConfig(**cfg))


def test_ask_stream_simple_answer():
    agent = _mk_agent()

    def fake_stream(messages, tools):
        for tok in ["F0 is ", "188 Hz"]:
            yield {"choices": [{"delta": {"content": tok}}]}

    agent._stream_chat_request = fake_stream  # type: ignore[method-assign]
    events = list(agent.ask_stream("what is the F0?"))
    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert events[-1]["answer"] == "F0 is 188 Hz"
    roles = [m["role"] for m in agent.history]
    assert roles == ["system", "user", "assistant"]


def test_ask_stream_tool_call_round(tmp_path):
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=0.5), 16000)

    def fake_stream(messages, tools):
        if not any(m.get("role") == "tool" for m in messages):
            # model streams a tool call whose arguments arrive in fragments
            args = json.dumps({"path": path})
            frag1, frag2 = args[: len(args) // 2], args[len(args) // 2:]
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1",
                 "function": {"name": "analyze_audio", "arguments": frag1}}]}}]}
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": frag2}}]}}]}
            return
        for tok in ["jitter ", "looks normal"]:
            yield {"choices": [{"delta": {"content": tok}}]}

    agent = _mk_agent(tmp_path)
    agent._stream_chat_request = fake_stream  # type: ignore[method-assign]
    events = list(agent.ask_stream("analyse the file"))

    types = [e["type"] for e in events]
    assert types == ["tool", "tool", "delta", "delta", "done"]
    assert events[0] == {"type": "tool", "name": "analyze_audio", "status": "start"}
    assert events[1]["status"] == "done"
    assert events[-1]["answer"] == "jitter looks normal"

    roles = [m["role"] for m in agent.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    payload = json.loads(tool_msg["content"])
    assert payload["pitch"]["n_voiced_frames"] > 0  # real analysis ran


def test_ask_stream_reports_transport_error():
    agent = _mk_agent()

    def fake_stream(messages, tools):
        raise urllib.error.URLError("connection reset")
        yield  # pragma: no cover — make this a generator

    agent._stream_chat_request = fake_stream  # type: ignore[method-assign]
    events = list(agent.ask_stream("hi"))
    assert events[-1]["type"] == "error"
    assert "URLError" in events[-1]["error"]
    # failed exchange must not corrupt history
    assert [m["role"] for m in agent.history] == ["system"]


# ------------------------------------------------------------- new tools
def test_tool_reference_ranges():
    agent = _mk_agent()
    out = agent.tools["reference_ranges"]({"metric": "jitter"})
    assert "jitter_percent" in out
    out_all = agent.tools["reference_ranges"]({})
    assert {"f0_hz", "jitter_percent", "shimmer_db", "hnr_db"} <= set(out_all)


def test_tool_reference_ranges_unknown_metric():
    agent = _mk_agent()
    out = agent.tools["reference_ranges"]({"metric": "vibeness"})
    assert "error" in out and "available" in out


def test_tool_compare_audio(tmp_path):
    a = write_wav(str(tmp_path / "a.wav"), vowel_like(duration_s=0.6, f0_hz=150), 16000)
    b = write_wav(str(tmp_path / "b.wav"), vowel_like(duration_s=0.6, f0_hz=250), 16000)
    agent = _mk_agent(tmp_path)
    out = agent.tools["compare_audio"]({"path_a": a, "path_b": b})
    assert [os.path.basename(f) for f in out["files"]] == ["a.wav", "b.wav"]
    assert any(r["metric"] == "f0_median_hz" for r in out["metrics"])
    assert "f0_ttest" in out  # contours were compared inferentially


def test_tool_diarize_audio(tmp_path):
    a = write_wav(str(tmp_path / "a.wav"), vowel_like(duration_s=1.0), 16000)
    agent = _mk_agent(tmp_path)
    out = agent.tools["diarize_audio"]({"path": a, "n_speakers": 0})
    assert "n_speakers" in out and "turns" in out


def test_tool_transcribe_missing_file(tmp_path):
    agent = _mk_agent(tmp_path)
    out = json.loads(agent._execute_tool({
        "id": "c1", "type": "function",
        "function": {"name": "transcribe_audio",
                     "arguments": json.dumps({"path": str(tmp_path / "no.wav")})},
    }))
    assert "error" in out  # missing file surfaces to the model, no crash
