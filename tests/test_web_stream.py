"""Tests for the new web endpoints: SSE streaming, auth gate, history, trends.

The LLM agent is replaced by a fake so no network is required.
"""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest

pytest.importorskip("flask", reason="web extra not installed: pip install -e '.[web]'")

import web.app as webapp
from web.app import app
from web.store import Store


class FakeAgent:
    """Stands in for SpeechResearchAgent; scripts ask/ask_stream."""

    def __init__(self, script=None):
        self.script = script or [
            {"type": "delta", "text": "F0 is "},
            {"type": "delta", "text": "188 Hz"},
            {"type": "done", "answer": "F0 is 188 Hz"},
        ]

    def ask(self, message):
        return "F0 is 188 Hz"

    def ask_stream(self, message):
        yield from self.script


def _wav_bytes(sr=16000, seconds=1.0, freq=150.0):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    x = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # isolated DB per test
    store = Store(str(tmp_path / "web.db"))
    monkeypatch.setattr(webapp, "STORE", store)
    monkeypatch.setattr(webapp, "WEB_PASSWORD", "")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    store.close()


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


# ------------------------------------------------------------------ stream
def test_chat_stream_events_and_persistence(client, monkeypatch):
    monkeypatch.setattr(webapp, "_get_or_build_agent", lambda sess, body: FakeAgent())
    r = client.post("/api/chat/stream",
                    json={"message": "what is the F0?"},
                    headers={"X-User-Key": "u1"})
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    events = _parse_sse(r.get_data(as_text=True))
    types = [e["type"] for e in events]
    assert types == ["meta", "delta", "delta", "done"]
    conv_id = events[0]["conversation_id"]
    assert conv_id

    # exchange was persisted for this user
    msgs = webapp.STORE.get_messages(conv_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "F0 is 188 Hz"


def test_chat_stream_tool_events(client, monkeypatch):
    script = [
        {"type": "tool", "name": "analyze_audio", "status": "start"},
        {"type": "tool", "name": "analyze_audio", "status": "done"},
        {"type": "delta", "text": "done"},
        {"type": "done", "answer": "done"},
    ]
    monkeypatch.setattr(webapp, "_get_or_build_agent",
                        lambda sess, body: FakeAgent(script))
    r = client.post("/api/chat/stream", json={"message": "go"})
    events = _parse_sse(r.get_data(as_text=True))
    assert [e["type"] for e in events] == ["meta", "tool", "tool", "delta", "done"]


def test_chat_stream_error_event_not_persisted(client, monkeypatch):
    script = [{"type": "error", "error": "boom"}]
    monkeypatch.setattr(webapp, "_get_or_build_agent",
                        lambda sess, body: FakeAgent(script))
    r = client.post("/api/chat/stream", json={"message": "hi"},
                    headers={"X-User-Key": "u2"})
    events = _parse_sse(r.get_data(as_text=True))
    assert events[-1]["type"] == "error"
    assert webapp.STORE.list_conversations("u2") == []


def test_chat_stream_without_user_key_skips_persistence(client, monkeypatch):
    monkeypatch.setattr(webapp, "_get_or_build_agent", lambda sess, body: FakeAgent())
    r = client.post("/api/chat/stream", json={"message": "hi"})
    events = _parse_sse(r.get_data(as_text=True))
    assert events[0]["conversation_id"] == ""


# -------------------------------------------------------------------- auth
def test_auth_gate_blocks_and_allows(client, monkeypatch):
    monkeypatch.setattr(webapp, "WEB_PASSWORD", "sesame")
    # /api/config stays open so the UI can learn auth is required
    assert client.get("/api/config").status_code == 200
    assert client.get("/api/config").get_json()["auth_required"] is True

    r = client.post("/api/chat/stream", json={"message": "hi"})
    assert r.status_code == 401
    r = client.get("/api/trends")
    assert r.status_code == 401

    ok = {"X-Auth-Token": "sesame"}
    monkeypatch.setattr(webapp, "_get_or_build_agent", lambda sess, body: FakeAgent())
    assert client.post("/api/chat/stream", json={"message": "hi"},
                       headers=ok).status_code == 200
    assert client.get("/api/trends", headers=ok).status_code == 200


def test_wrong_password_rejected(client, monkeypatch):
    monkeypatch.setattr(webapp, "WEB_PASSWORD", "sesame")
    r = client.get("/api/trends", headers={"X-Auth-Token": "nope"})
    assert r.status_code == 401


# ----------------------------------------------------------------- history
def test_history_list_get_delete(client, monkeypatch):
    monkeypatch.setattr(webapp, "_get_or_build_agent", lambda sess, body: FakeAgent())
    r = client.post("/api/chat/stream", json={"message": "first question"},
                    headers={"X-User-Key": "u1"})
    conv_id = _parse_sse(r.get_data(as_text=True))[0]["conversation_id"]

    listing = client.get("/api/history", headers={"X-User-Key": "u1"}).get_json()
    assert [c["id"] for c in listing["conversations"]] == [conv_id]
    assert listing["conversations"][0]["title"] == "first question"

    # another user cannot see or fetch it
    assert client.get("/api/history", headers={"X-User-Key": "u2"}
                      ).get_json()["conversations"] == []
    assert client.get(f"/api/history?conversation_id={conv_id}",
                      headers={"X-User-Key": "u2"}).status_code == 404

    one = client.get(f"/api/history?conversation_id={conv_id}",
                     headers={"X-User-Key": "u1"}).get_json()
    assert [m["role"] for m in one["messages"]] == ["user", "assistant"]

    assert client.delete("/api/history",
                         json={"conversation_id": conv_id},
                         headers={"X-User-Key": "u1"}).status_code == 200
    assert client.get("/api/history", headers={"X-User-Key": "u1"}
                      ).get_json()["conversations"] == []


def test_history_without_user_key(client):
    assert client.get("/api/history").get_json() == {"conversations": [],
                                                     "messages": []}


# ------------------------------------------------------------------ trends
def test_trends_crud(client):
    h = {"X-User-Key": "u1"}
    assert client.get("/api/trends", headers=h).get_json() == {"measurements": []}

    r = client.post("/api/trends", json={"file": "a.wav", "f0": 220.0,
                                         "jitter": 0.4, "shimmer": 0.2,
                                         "hnr": 21.0}, headers=h)
    assert r.get_json()["ok"] is True
    rows = client.get("/api/trends", headers=h).get_json()["measurements"]
    assert len(rows) == 1 and rows[0]["f0"] == 220.0

    client.delete("/api/trends", headers=h)
    assert client.get("/api/trends", headers=h).get_json()["measurements"] == []


def test_analyze_saves_measurement(client):
    h = {"X-User-Key": "u1"}
    r = client.post("/api/analyze", data={
        "file": (io.BytesIO(_wav_bytes()), "a.wav"),
    }, content_type="multipart/form-data", headers=h)
    assert r.status_code == 200
    rows = client.get("/api/trends", headers=h).get_json()["measurements"]
    assert len(rows) == 1
    assert rows[0]["file"] == "a.wav"
    assert rows[0]["f0"] > 0  # F0 median of the 150 Hz tone came through
