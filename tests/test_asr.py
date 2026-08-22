"""Tests for the ASR module: privacy boundary, header injection, caching."""

from __future__ import annotations

import io
import sys
import types
import wave

import numpy as np
import pytest

from speechlab import asr


def _wav_bytes(sr=16000, seconds=0.3, freq=150.0):
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
def wav_file(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(_wav_bytes())
    return str(p)


# --------------------------------------------------------------- sanitisation
def test_sanitize_filename_blocks_header_injection():
    evil = "/tmp/evil\"\r\nX-Injected: yes\r\n.wav"
    safe = asr._sanitize_filename(evil)
    assert '"' not in safe and "\r" not in safe and "\n" not in safe
    assert "\\" not in safe
    assert safe.startswith("evil") and safe.endswith(".wav")


def test_sanitize_filename_falls_back_on_empty():
    # basename of a trailing-slash path is empty
    assert asr._sanitize_filename("/some/dir/") == "audio"
    # all-unsafe characters are still rendered harmless
    junk = asr._sanitize_filename('"\\\r\n')
    assert set(junk) == {"_"}


def test_multipart_body_has_clean_filename(wav_file):
    body, _ = asr._multipart({"model": "whisper-1"}, wav_file)
    text = body.decode("latin-1")
    assert 'filename="a.wav"' in text
    assert "\r\nX-" not in text.split("filename=")[1][:64]


# --------------------------------------------------------- privacy boundary
def test_local_empty_text_stays_local(wav_file, monkeypatch):
    """A local backend's empty result must not trigger a remote upload."""
    monkeypatch.setattr(
        asr, "_local_whisper",
        lambda p, lang: {"text": "", "language": None,
                         "engine": "local openai-whisper (base)"})

    def api_must_not_run(*a, **k):
        raise AssertionError("audio must not be uploaded when a local backend ran")

    monkeypatch.setattr(asr, "_api_whisper", api_must_not_run)
    monkeypatch.delenv("SPEECHLAB_API_KEY", raising=False)
    out = asr.transcribe(wav_file)
    assert out["text"] == ""
    assert out["note"] == "no speech detected by the local model"
    assert out["engine"].startswith("local")


def test_no_backend_and_no_key_raises(wav_file, monkeypatch):
    monkeypatch.setattr(asr, "_local_whisper", lambda p, lang: None)
    monkeypatch.delenv("SPEECHLAB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="transcription unavailable"):
        asr.transcribe(wav_file)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        asr.transcribe(str(tmp_path / "nope.wav"))


# ------------------------------------------------------------------ caching
def test_local_whisper_model_loaded_once(wav_file, monkeypatch):
    calls = {"load": 0}

    class FakeModel:
        def transcribe(self, path, language=None):
            return {"text": "hello world", "language": language or "en"}

    def fake_load(name):
        calls["load"] += 1
        return FakeModel()

    fake = types.ModuleType("whisper")
    fake.load_model = fake_load
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.delitem(asr._MODEL_CACHE, "openai-whisper", raising=False)

    r1 = asr.transcribe(wav_file)
    r2 = asr.transcribe(wav_file)
    assert calls["load"] == 1
    assert r1["text"] == r2["text"] == "hello world"
    assert r1["engine"] == "local openai-whisper (base)"
