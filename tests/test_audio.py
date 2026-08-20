"""Tests for audio I/O and utilities."""

import numpy as np
import pytest

from speechlab.audio import db, frame_signal, load_audio, resample

from .helpers import noise, tone, write_wav


def test_load_wav_roundtrip(tmp_path):
    x = tone(440.0, 0.5)
    path = write_wav(str(tmp_path / "a.wav"), x, 16000)
    audio = load_audio(path)
    assert audio.sample_rate == 16000
    assert audio.num_samples == len(x)
    assert audio.duration == pytest.approx(0.5, abs=1e-3)
    # 16-bit quantisation tolerance
    np.testing.assert_allclose(audio.samples, x, atol=1e-4)


def test_load_resamples(tmp_path):
    sr, target = 16000, 8000
    x = tone(440.0, 1.0, sr=sr)
    path = write_wav(str(tmp_path / "a.wav"), x, sr)
    audio = load_audio(path, target_sr=target)
    assert audio.sample_rate == target
    assert audio.num_samples == pytest.approx(len(x) // 2, abs=10)


def test_resample_identity():
    x = tone(200.0, 0.1)
    np.testing.assert_allclose(resample(x, 16000, 16000), x)


def test_resample_changes_length():
    x = tone(200.0, 1.0, sr=16000)
    y = resample(x, 16000, 24000)
    assert len(y) == pytest.approx(24000, abs=5)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/file.wav")


def test_db_floor():
    assert db(0.0) == -120.0
    assert db(1.0) == pytest.approx(0.0)
    assert db(10.0) == pytest.approx(10.0)


def test_frame_signal_shapes():
    sr = 16000
    x = noise(0.5, sr=sr, amp=0.1)
    fl, hl = 400, 160
    frames = frame_signal(x, fl, hl, center=False)
    expected = 1 + (len(x) - fl) // hl
    assert frames.shape == (expected, fl)

    # centered framing yields roughly duration/hop frames
    frames_c = frame_signal(x, fl, hl, center=True)
    assert frames_c.shape[0] == pytest.approx(len(x) // hl, abs=5)


def test_frame_signal_empty():
    frames = frame_signal(np.zeros(10), 400, 160, center=False)
    assert frames.shape == (0, 400)
