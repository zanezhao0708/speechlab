"""Tests for recording-quality checks."""

import numpy as np

from speechlab.audio import load_audio
from speechlab.quality import quality_report

from .helpers import noise, tone, vowel_like, write_wav


def _make(tmp_path, samples, sr=16000, name="utt.wav"):
    path = write_wav(str(tmp_path / name), samples, sr)
    return load_audio(path)


def test_clean_recording_has_no_issues(tmp_path):
    x = 0.3 * vowel_like(duration_s=2.0) + 0.001 * noise(2.0, seed=1)
    report = quality_report(_make(tmp_path, x))
    assert report["ok"]
    assert report["issues"] == []
    assert report["speech_ratio"] > 0.3


def test_clipping_detected(tmp_path):
    sr = 16000
    x = tone(200.0, 1.0, sr=sr, amp=2.0)  # beyond full scale -> clipped on write
    report = quality_report(_make(tmp_path, x))
    assert not report["ok"]
    assert any("clipping" in i for i in report["issues"])
    assert report["clipping_ratio"] > 0


def test_dc_offset_detected(tmp_path):
    sr = 16000
    x = 0.2 * tone(200.0, 1.0, sr=sr) + 0.2  # large DC offset
    report = quality_report(_make(tmp_path, x))
    assert any("DC offset" in i for i in report["issues"])


def test_low_snr_detected(tmp_path):
    sr = 16000
    # speech burst in heavy noise
    speech = 0.05 * vowel_like(duration_s=1.0, sr=sr)
    x = speech + 0.2 * noise(1.0, sr=sr, seed=5)
    report = quality_report(_make(tmp_path, x))
    assert any("SNR" in i for i in report["issues"])
    assert report["snr_db"] < 15.0


def test_mostly_silence_flagged(tmp_path):
    sr = 16000
    silence = 0.001 * noise(2.0, sr=sr, seed=2)
    burst = 0.5 * vowel_like(duration_s=0.2, sr=sr)
    x = np.concatenate([silence, burst, silence])
    report = quality_report(_make(tmp_path, x))
    assert any("speech energy" in i for i in report["issues"])


def test_low_sample_rate_flagged(tmp_path):
    sr = 8000
    x = 0.3 * vowel_like(duration_s=1.0, sr=sr)
    report = quality_report(_make(tmp_path, x, sr=sr))
    assert any("sample rate" in i for i in report["issues"])


def test_report_is_json_serialisable(tmp_path):
    import json

    x = 0.3 * tone(200.0, 0.5)
    report = quality_report(_make(tmp_path, x))
    json.dumps(report)  # must not raise
