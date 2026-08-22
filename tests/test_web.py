"""Tests for the Flask web app: endpoints, quota, rate limit, cache."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

pytest.importorskip("flask", reason="web extra not installed: pip install -e '.[web]'")

from web.app import _ANALYSIS_CACHE, app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Test client with an isolated per-test database.

    Without this swap the routes would write into the module-level Store
    (``web/data/speechlab.db``), leaking rows between test runs and
    polluting any real deployment the tests happen to run against.
    """
    from web import app as webapp_mod
    from web.store import Store
    monkeypatch.setattr(webapp_mod, "STORE", Store(str(tmp_path / "test.db")))
    webapp_mod._ANALYZE_TIMES.clear()
    _ANALYSIS_CACHE.clear()
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


def test_analyze_rate_limit(client, monkeypatch):
    """/api/analyze is CPU-bound and keyless — it must be rate limited."""
    from web import app as webapp_mod
    monkeypatch.setattr(webapp_mod, "ANALYZE_RATE_LIMIT", (2, 60.0))
    webapp_mod._ANALYZE_TIMES.clear()
    try:
        codes = [
            client.post("/api/analyze", data={
                "file": (io.BytesIO(_wav_bytes()), "a.wav"),
            }, content_type="multipart/form-data").status_code
            for _ in range(3)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429
    finally:
        webapp_mod._ANALYZE_TIMES.clear()


def test_trends_rejects_bad_payloads(client):
    headers = {"X-User-Key": "tester"}
    # non-numeric metric
    r = client.post("/api/trends", json={"f0": "abc"}, headers=headers)
    assert r.status_code == 400
    # boolean masquerading as a number
    r = client.post("/api/trends", json={"f0": True}, headers=headers)
    assert r.status_code == 400
    # forged far-future timestamp
    r = client.post("/api/trends",
                    json={"f0": 120.0, "ts": 9999999999.0}, headers=headers)
    assert r.status_code == 400
    # valid row round-trips
    r = client.post("/api/trends",
                    json={"f0": 120.5, "jitter": 0.4, "file": "a.wav"},
                    headers=headers)
    assert r.status_code == 200
    rows = client.get("/api/trends", headers=headers).get_json()["measurements"]
    assert len(rows) == 1 and rows[0]["f0"] == 120.5
