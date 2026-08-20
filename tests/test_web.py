"""Tests for the Flask web app: endpoints, quota, rate limit, cache."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from web.app import _ANALYSIS_CACHE, app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


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


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert "has_server_key" in r.get_json()


def test_analyze_direct(client):
    r = client.post("/api/analyze", data={
        "file": (io.BytesIO(_wav_bytes()), "a.wav"),
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["file"] == "a.wav"
    assert d["pitch"]["n_voiced_frames"] > 0
    # strict JSON: NaN must have been replaced by null
    assert "NaN" not in r.get_data(as_text=True)


def test_analyze_rejects_bad_extension(client):
    r = client.post("/api/analyze", data={
        "file": (io.BytesIO(b"x"), "a.txt"),
    }, content_type="multipart/form-data")
    assert r.status_code == 400


def test_analyze_cache_hit(client, monkeypatch):
    _ANALYSIS_CACHE.clear()
    calls = {"n": 0}
    import web.app as webapp
    orig = webapp.analyze

    def counting(audio, contour=False):
        calls["n"] += 1
        return orig(audio, contour=contour)

    monkeypatch.setattr(webapp, "analyze", counting)
    raw = _wav_bytes()
    for _ in range(2):
        r = client.post("/api/analyze", data={
            "file": (io.BytesIO(raw), "b.wav"),  # fresh handle each time
        }, content_type="multipart/form-data")
        assert r.status_code == 200
    assert calls["n"] == 1  # second call served from cache


def test_upload_and_chat_without_key(client):
    r = client.post("/api/upload", data={
        "file": (io.BytesIO(_wav_bytes()), "a.wav"),
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    sid = r.get_json()["session_id"]

    r2 = client.post("/api/chat", json={
        "session_id": sid, "message": "hi",
    }, environ_base={})
    # no API key configured and none supplied → clear error, not a crash
    import os
    if os.environ.get("SPEECHLAB_API_KEY"):
        pytest.skip("server key configured in this environment")
    assert r2.status_code == 400
    assert "API Key" in r2.get_json()["error"]


def test_chat_rate_limit(client, monkeypatch):
    import os
    if os.environ.get("SPEECHLAB_API_KEY"):
        pytest.skip("server key configured in this environment")
    # establish a session first (uploads return the session id)
    r = client.post("/api/upload", data={
        "file": (io.BytesIO(_wav_bytes()), "a.wav"),
    }, content_type="multipart/form-data")
    sid = r.get_json()["session_id"]
    codes = [
        client.post("/api/chat", json={"message": "hi", "session_id": sid}).status_code
        for _ in range(10)
    ]
    # allowed failures (400 no-key) until the limiter kicks in with 429
    assert 429 in codes


def test_upload_quota(client, monkeypatch):
    from web import app as webapp_mod
    big = _wav_bytes()
    monkeypatch.setattr(webapp_mod, "SESSION_UPLOAD_QUOTA", len(big) + 100)
    r = client.post("/api/upload", data={
        "file": (io.BytesIO(big), "a.wav"),
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    r2 = client.post("/api/upload", data={
        "file": (io.BytesIO(big), "b.wav"),
        "session_id": r.get_json()["session_id"],
    }, content_type="multipart/form-data")
    assert r2.status_code == 413


def test_reset(client):
    r = client.post("/api/upload", data={
        "file": (io.BytesIO(_wav_bytes()), "a.wav"),
    }, content_type="multipart/form-data")
    sid = r.get_json()["session_id"]
    r2 = client.post("/api/reset", json={"session_id": sid})
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
