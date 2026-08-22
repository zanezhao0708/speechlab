"""Tests for the WAV decoding paths (8-bit unsigned, 24-bit vectorised)."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from speechlab.audio import load_audio


def _write_wav(path, samples, sampwidth, sr=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        if sampwidth == 1:  # unsigned, midpoint 128
            raw = np.clip(np.round(samples * 127 + 128), 0, 255).astype(np.uint8)
        elif sampwidth == 2:
            raw = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
        elif sampwidth == 3:
            ints = np.clip(np.round(samples * 8388607), -8388608, 8388607).astype(np.int64)
            u = (ints & 0xFFFFFF).astype(np.uint32)
            raw = np.stack([u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
            w.writeframes(raw)
            return
        else:
            raise ValueError(sampwidth)
        w.writeframes(raw.tobytes())


@pytest.fixture()
def tone():
    sr = 16000
    t = np.arange(sr) / sr
    return sr, 0.5 * np.sin(2 * np.pi * 200 * t)


def test_wav8_unsigned_roundtrip(tmp_path, tone):
    sr, x = tone
    p = tmp_path / "u8.wav"
    _write_wav(str(p), x, sampwidth=1, sr=sr)
    got = load_audio(str(p))
    # 8-bit quantisation is coarse (~1/128); the sine must survive intact
    assert got.sample_rate == sr
    np.testing.assert_allclose(got.samples, x, atol=1.5 / 127)
    # sign matters: the old signed-int8 bug destroyed the waveform offset
    assert abs(np.mean(got.samples)) < 0.02


def test_wav24_vectorised_roundtrip(tmp_path, tone):
    sr, x = tone
    p = tmp_path / "s24.wav"
    _write_wav(str(p), x, sampwidth=3, sr=sr)
    got = load_audio(str(p))
    np.testing.assert_allclose(got.samples, x, atol=1.5 / 8388608)


def test_wav24_negative_values(tmp_path):
    # exercise the sign-bit branch of the unpacker
    sr = 8000
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0] * 100)
    p = tmp_path / "s24b.wav"
    _write_wav(str(p), x, sampwidth=3, sr=sr)
    got = load_audio(str(p))
    assert got.samples[0] < -0.99
    assert got.samples[-1] > 0.99
