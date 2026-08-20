"""Tests for the acoustic analysis core used by the agent."""

import json

import numpy as np
import pytest

from speechlab.audio import frame_signal, load_audio
from speechlab.features import (
    activity_summary,
    analyze,
    compare_reports,
    cpps,
    default_frame_lengths,
    f0_track,
    formants,
    hnr,
    jitter_shimmer,
    report_field,
    spectral_stats,
    vad_segments,
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


# ---------------------------------------------------------------------- CPPS
def test_cpps_discriminates_breathiness():
    """Harmonic-rich > breathy > noise: CPPS falls as noise mixes in."""
    sr = 16000
    clean = vowel_like(f0_hz=150.0, duration_s=1.0, sr=sr)
    breathy = clean + 0.6 * noise(1.0, sr=sr, amp=0.25, seed=5)
    noisy = noise(1.0, sr=sr, amp=0.25, seed=5)
    assert cpps(clean, sr) > cpps(breathy, sr) > cpps(noisy, sr)


def test_cpps_silence_is_nan():
    assert np.isnan(cpps(np.zeros(16000), 16000))


def test_cpps_short_signal_no_crash():
    """Smoothing window longer than the frame count must not raise."""
    sr = 16000
    for dur in (0.03, 0.05, 0.08):
        x = vowel_like(f0_hz=140.0, duration_s=dur, sr=sr)
        assert np.isfinite(cpps(x, sr))


# ------------------------------------------------------------------ spectral
def test_spectral_stats_vowel_vs_noise():
    sr = 16000
    v = spectral_stats(vowel_like(f0_hz=140.0, duration_s=1.0, sr=sr), sr)
    n = spectral_stats(noise(1.0, sr=sr, amp=0.3, seed=3), sr)
    assert 0 < v["spectral_centroid_hz"] < sr / 2
    # vowel energy is low-frequency, noise is spread to Nyquist
    assert v["spectral_centroid_hz"] < n["spectral_centroid_hz"]
    # vowel LTAS slopes down, white noise is flat
    assert v["spectral_tilt_db_per_khz"] < -3.0
    assert abs(n["spectral_tilt_db_per_khz"]) < 3.0
    # harmonic signal is far from flat, noise is nearly flat
    assert v["spectral_flatness"] < 0.1 < n["spectral_flatness"]


def test_spectral_stats_silence_is_empty():
    assert spectral_stats(np.zeros(16000), 16000) == {}


# ----------------------------------------------------------------------- VAD
def _speech_pause_speech(sr=16000):
    """1 s vowel, 0.5 s silence, 1 s vowel."""
    return np.concatenate([
        vowel_like(f0_hz=140.0, duration_s=1.0, sr=sr),
        np.zeros(round(0.5 * sr)),
        vowel_like(f0_hz=180.0, duration_s=1.0, sr=sr),
    ])


def test_vad_segments_split_by_silence():
    sr = 16000
    segs = vad_segments(_speech_pause_speech(sr), sr)
    assert len(segs) == 2
    (s0, e0), (s1, e1) = segs
    assert s0 == pytest.approx(0.0, abs=0.05)
    assert e0 == pytest.approx(1.0, abs=0.1)
    assert s1 == pytest.approx(1.5, abs=0.1)
    assert e1 == pytest.approx(2.5, abs=0.05)


def test_vad_bridges_short_pauses():
    """A 30 ms dip is shorter than min_pause_s (60 ms) and is bridged."""
    sr = 16000
    x = vowel_like(f0_hz=140.0, duration_s=0.6, sr=sr)
    i = round(0.3 * sr)
    x[i : i + round(0.03 * sr)] = 0.0
    assert len(vad_segments(x, sr)) == 1


def test_vad_drops_short_blips():
    """A 10 ms burst is not a speech segment (min_speech_s=50 ms here)."""
    sr = 16000
    x = np.zeros(round(1.0 * sr))
    i = round(0.5 * sr)
    x[i : i + round(0.01 * sr)] = 0.5
    assert vad_segments(x, sr, min_speech_s=0.05) == []


def test_vad_accepts_precomputed_frames():
    sr = 16000
    x = _speech_pause_speech(sr)
    fl, hl = default_frame_lengths(sr)
    frames = frame_signal(x, fl, hl, window="rect", center=True)
    assert vad_segments(x, sr, frames=frames) == vad_segments(x, sr)


def test_vad_all_silence():
    assert vad_segments(np.zeros(16000), 16000) == []


def test_activity_summary_structure():
    sr = 16000
    a = activity_summary(_speech_pause_speech(sr), sr)
    assert a["n_speech_segments"] == 2
    assert a["n_pauses"] == 1
    assert a["mean_pause_s"] == pytest.approx(0.5, abs=0.1)
    assert a["max_pause_s"] == pytest.approx(0.5, abs=0.1)
    assert a["speech_ratio"] == pytest.approx(0.8, abs=0.1)
    assert a["pause_time_s"] == pytest.approx(0.5, abs=0.15)


def test_activity_summary_all_silence():
    a = activity_summary(np.zeros(16000), 16000)
    assert a["n_speech_segments"] == 0
    assert a["n_pauses"] == 0
    assert a["speech_ratio"] == 0.0


# ---------------------------------------------------------------- comparison
def test_compare_reports_deltas():
    a = {
        "duration_s": 1.0,
        "pitch": {"f0_median_hz": 100.0},
        "voice_quality": {"jitter_local_percent": float("nan")},
        "hnr_db": 10.0,
    }
    b = {
        "duration_s": 2.0,
        "pitch": {"f0_median_hz": 110.0},
        "voice_quality": {"jitter_local_percent": 1.0},
        "hnr_db": 12.5,
    }
    d = compare_reports(a, b)
    assert d["duration_s"] == pytest.approx(1.0)
    assert d["pitch.f0_median_hz"] == pytest.approx(10.0)
    assert d["hnr_db"] == pytest.approx(2.5)
    # NaN jitter in A cannot be compared — skipped, not crashed
    assert "voice_quality.jitter_local_percent" not in d


def test_report_field_paths():
    r = {"pitch": {"f0_median_hz": 100.0, "note": "x"}, "file": "a.wav"}
    assert report_field(r, "pitch.f0_median_hz") == 100.0
    assert report_field(r, "pitch.f0_mean_hz") is None       # missing key
    assert report_field(r, "file") is None                   # non-numeric
    assert report_field(r, "missing.deeper.path") is None    # missing branch


# ---------------------------------------------------------------- full report
def test_analyze_includes_new_measures(tmp_path):
    sr = 16000
    x = _speech_pause_speech(sr)
    path = write_wav(str(tmp_path / "utt.wav"), x, sr)
    report = analyze(load_audio(path))
    assert np.isfinite(report["cpps_db"])
    assert 0 < report["spectral"]["spectral_centroid_hz"] < sr / 2
    assert report["activity"]["n_speech_segments"] == 2
    assert report["activity"]["n_pauses"] == 1
    # everything must survive a JSON round-trip (LLM tool payload)
    json.dumps(report)


def test_analyze_report_is_strict_json(tmp_path):
    """Undefined measures become null, never NaN/Infinity literals."""
    sr = 16000
    # unvoiced noise -> jitter/shimmer/HNR undefined
    path = write_wav(str(tmp_path / "noise.wav"), noise(0.5, sr=sr, amp=0.3, seed=1), sr)
    report = analyze(load_audio(path))
    # allow_nan=False raises ValueError if any NaN/inf remains
    json.dumps(report, allow_nan=False)
    assert report["voice_quality"]["jitter_local_percent"] is None
    assert report["hnr_db"] is None


def test_f0_track_edges_not_pulled_to_zero():
    """Median smoothing must not drag the first/last voiced F0 toward 0."""
    sr = 16000
    x = vowel_like(f0_hz=200.0, duration_s=0.5, sr=sr)
    track = f0_track(x, sr)
    v = track.f0[track.voiced]
    assert len(v) >= 3
    # edge values stay near the true F0, not collapsed by zero-padding
    assert v[0] == pytest.approx(200.0, rel=0.15)
    assert v[-1] == pytest.approx(200.0, rel=0.15)
