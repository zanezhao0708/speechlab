"""Tests for acoustic feature extraction."""

import numpy as np
import pytest

from speechlab.audio import load_audio
from speechlab.features import (
    f0_track,
    formants,
    frame_energy,
    hnr,
    hz_to_mel,
    jitter_shimmer,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    mfcc,
)

from .helpers import noise, tone, vowel_like, write_wav


# ---------------------------------------------------------------- mel / MFCC
def test_mel_scale_roundtrip():
    for f in (100.0, 1000.0, 4000.0, 7999.0):
        m = hz_to_mel(f)
        assert float(mel_to_hz(m)) == pytest.approx(f, rel=1e-6)


def test_mel_filterbank_shape():
    fb = mel_filterbank(16000, 512, n_mels=26)
    assert fb.shape == (26, 257)
    assert np.all(fb >= 0)
    # filters should not overlap the whole spectrum
    assert np.all(fb[0] > -1e-9)


def test_mel_spectrogram_shape():
    sr = 16000
    x = noise(1.0, sr=sr)
    mel = mel_spectrogram(x, sr, n_mels=64)
    assert mel.shape[1] == 64
    # ~100 frames per second at 10 ms hop
    assert mel.shape[0] == pytest.approx(100, abs=15)


def test_mfcc_shape_and_finiteness():
    sr = 16000
    x = vowel_like(duration_s=0.5, sr=sr)
    c = mfcc(x, sr, n_mfcc=13)
    assert c.shape == (c.shape[0], 13)
    assert np.all(np.isfinite(c))


def test_mfcc_energy_rises_with_loudness():
    sr = 16000
    quiet = mfcc(0.1 * tone(500.0, 0.5, sr=sr), sr)
    loud = mfcc(0.9 * tone(500.0, 0.5, sr=sr), sr)
    # c0 tracks log energy
    assert np.mean(loud[:, 0]) > np.mean(quiet[:, 0]) + 3.0


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
    from speechlab.features import analyze

    report = analyze(load_audio(path))
    assert report["duration_s"] == pytest.approx(0.8, abs=0.01)
    assert report["sample_rate_hz"] == 16000
    assert report["pitch"]["f0_median_hz"] == pytest.approx(140.0, rel=0.05)
    assert report["voice_quality"]["n_periods"] > 30
    assert set(report["formants"]) == {"F1_hz", "F2_hz", "F3_hz"}


def test_frame_energy_range():
    sr = 16000
    e = frame_energy(tone(300.0, 0.3, sr=sr, amp=0.5), sr)
    assert len(e) > 10
    assert np.all(np.isfinite(e))
