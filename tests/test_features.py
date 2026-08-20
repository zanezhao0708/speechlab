"""Tests for the acoustic analysis core used by the agent."""

import numpy as np
import pytest

from speechlab.audio import load_audio
from speechlab.features import (
    analyze,
    f0_track,
    formants,
    hnr,
    jitter_shimmer,
)

from .helpers import noise, tone, vowel_like, write_wav


# --------------------------------------------------------------------- pitch
@pytest.mark.parametrize("f0", [85.0, 120.0, 220.0, 330.0])
def test_f0_on_pure_tone(f0):
    sr = 16000
    x = tone(f0, 1.0, sr=sr, amp=0.5)
    track = f0_track(x, sr)
    voiced_f0 = track.f0[track.voiced]
    assert len(voiced_f0) > 10, "tone should be mostly voiced"
    assert np.median(voiced_f0) == pytest.approx(f0, rel=0.03)


def test_f0_unvoiced_on_noise():
    sr = 16000
    x = noise(1.0, sr=sr, amp=0.3, seed=7)
    track = f0_track(x, sr)
    assert track.voiced_ratio < 0.15


def test_f0_summary_keys():
    track = f0_track(tone(150.0, 0.4), 16000)
    s = track.summary()
    assert {"f0_mean_hz", "f0_median_hz", "f0_std_hz", "voiced_ratio"} <= set(s)


# ------------------------------------------------------------------ formants
def test_formants_of_synthetic_vowel():
    sr = 16000
    x = vowel_like(f0_hz=120.0, duration_s=0.3, sr=sr,
                   formants=((500.0, 1.0), (1500.0, 0.6)))
    fl = int(0.025 * sr)
    center = x[len(x) // 2 : len(x) // 2 + fl]
    f = formants(center, sr)
    assert len(f) >= 2
    # F1 near 500 Hz, F2 near 1500 Hz (±20 %)
    assert f[0] == pytest.approx(500.0, rel=0.2)
    assert f[1] == pytest.approx(1500.0, rel=0.2)


def test_formants_order():
    sr = 16000
    x = vowel_like(duration_s=0.2, sr=sr)
    f = formants(x[: int(0.025 * sr)], sr)
    assert f == sorted(f)


# --------------------------------------------------------------- voice quality
def test_jitter_shimmer_periodic_signal():
    sr = 16000
    x = vowel_like(f0_hz=150.0, duration_s=1.0, sr=sr)
    js = jitter_shimmer(x, sr)
    assert js.n_periods >= 50
    assert js.jitter_local_percent < 2.0  # synthetic voice is very stable
    assert js.shimmer_local_db < 1.0


def test_jitter_shimmer_too_short():
    js = jitter_shimmer(np.zeros(100), 16000)
    assert js.n_periods == 0
    assert np.isnan(js.jitter_local_percent)


def test_hnr_tone_high_noise_low():
    sr = 16000
    tonal = hnr(tone(200.0, 1.0, sr=sr, amp=0.5), sr)
    noisy = hnr(0.5 * noise(1.0, sr=sr, amp=1.0, seed=3), sr)
    assert tonal > 15.0
    # noise has no periodicity: either no HNR at all or clearly lower
    assert np.isnan(noisy) or noisy < tonal - 5.0


# ---------------------------------------------------------------- full report
def test_analyze_end_to_end(tmp_path):
    sr = 16000
    x = vowel_like(f0_hz=140.0, duration_s=0.8, sr=sr)
    path = write_wav(str(tmp_path / "utt.wav"), x, sr)

    report = analyze(load_audio(path))
    assert report["duration_s"] == pytest.approx(0.8, abs=0.01)
    assert report["sample_rate_hz"] == 16000
    assert report["pitch"]["f0_median_hz"] == pytest.approx(140.0, rel=0.05)
    assert report["voice_quality"]["n_periods"] > 30
    assert set(report["formants"]) == {"F1_hz", "F2_hz", "F3_hz"}


def test_analyze_computes_f0_once(monkeypatch, tmp_path):
    """The F0 track is shared with jitter/shimmer and HNR, not recomputed."""
    import speechlab.features as feats

    calls = {"n": 0}
    real = feats.f0_track

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(feats, "f0_track", counting)
    sr = 16000
    x = vowel_like(f0_hz=140.0, duration_s=0.8, sr=sr)
    path = write_wav(str(tmp_path / "utt.wav"), x, sr)
    report = feats.analyze(load_audio(path))
    assert calls["n"] == 1
    # results are unchanged by the sharing
    assert report["pitch"]["f0_median_hz"] == pytest.approx(140.0, rel=0.05)
    assert report["voice_quality"]["n_periods"] > 30


def test_jitter_shimmer_accepts_precomputed_track():
    sr = 16000
    x = vowel_like(f0_hz=150.0, duration_s=0.6, sr=sr)
    track = f0_track(x, sr)
    with_track = jitter_shimmer(x, sr, track=track)
    without = jitter_shimmer(x, sr)
    assert with_track.n_periods == without.n_periods
    assert with_track.jitter_local_percent == pytest.approx(
        without.jitter_local_percent)


def test_hnr_accepts_precomputed_track():
    sr = 16000
    x = tone(200.0, 1.0, sr=sr, amp=0.5)
    track = f0_track(x, sr)
    assert hnr(x, sr, track=track) == pytest.approx(hnr(x, sr))
