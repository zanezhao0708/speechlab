"""Tests for the acoustic analysis core used by the agent."""

import numpy as np
import pytest

from speechlab.audio import AudioData, load_audio
from speechlab.features import (
    _shimmer_local_db,
    analyze,
    f0_track,
    formants,
    hnr,
    jitter_shimmer,
)

from .helpers import noise, perturbed_vowel, tone, vowel_like, write_wav


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
    """Full pipeline recovers designed formants of a source-filter vowel.

    Tolerances match what Praat itself deviates on the same synthetic
    signals (measured during development; see test_formant_golden.py for
    the tight, ground-truth-level validation).
    """
    sr = 16000
    x = vowel_like(f0_hz=120.0, duration_s=0.3, sr=sr,
                   formants=((500.0, 1.0), (1500.0, 0.6), (2440.0, 0.3)))
    report = analyze(AudioData(samples=x, sample_rate=sr, path=None))
    fmt = report["formants"]
    assert fmt["F1_hz"] == pytest.approx(500.0, rel=0.12)
    assert fmt["F2_hz"] == pytest.approx(1500.0, rel=0.12)
    assert fmt["F3_hz"] == pytest.approx(2440.0, rel=0.2)


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


# ------------------------------------------- adversarial perturbation tests
# Stable-signal tests above prove the estimators do not explode; the tests
# here prove they actually *respond* to perturbation — the failure mode the
# old shimmer implementation hid for months: sign-cancelling dB steps made
# every perturbed voice look calm (see the telescoping comment in
# features.jitter_shimmer).
def test_shimmer_formula_alternating_amplitudes():
    """Reviewer's counter-example, at the formula level: 1 -> 2 -> 1 -> 2.

    Every consecutive period differs by 2x, so shimmer (local, dB) must be
    20·log10(2) = 6.02 dB.  The old sign-cancelling average telescoped the
    +6/−6 dB steps down to ~0 dB (log of last/first).  Tested directly on
    the amplitude sequence because the *signal* with exact alternation is
    genuinely ambiguous at the epoch level — see
    test_strict_amplitude_alternation_is_a_subharmonic below.
    """
    amps = np.tile([1.0, 2.0], 60)  # strict pulse alternans
    assert _shimmer_local_db(amps) == pytest.approx(6.0206, abs=0.01)
    # the same telescoping trap with unequal endpoints: a V-shaped envelope
    # returns to its start, so the buggy formula reported exactly 0 dB.
    v_shape = np.concatenate([np.linspace(1.0, 4.0, 50), np.linspace(4.0, 1.0, 50)])
    assert _shimmer_local_db(v_shape) > 0.05


def test_strict_amplitude_alternation_is_a_subharmonic():
    """Exact 1 -> 2 alternation resolves to F0/2, matching native Praat.

    Cross-checked against Praat's own autocorrelation tracker: on this
    signal it reports 60.0 Hz (59 pulses, shimmer 0.085 dB).  Period-based
    perturbation measures are only well-defined when the period is
    unambiguous; this locks our documented convention so a future change
    has to argue with Praat, not drift silently.
    """
    sr = 16000
    x = perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr,
                        amp_factors=(1.0, 2.0))
    track = f0_track(x, sr)
    voiced_f0 = track.f0[track.voiced]
    assert len(voiced_f0) > 10
    assert np.median(voiced_f0) < 80.0  # 60 Hz subharmonic, as Praat


def test_shimmer_ignores_slow_envelope_drift():
    """Anti-telescoping: first/last amplitude differing must NOT count.

    A slow 0.5 -> 1.0 ramp changes each period by ~0.6 %, so true shimmer
    is < 0.5 dB even though the envelope rises 6 dB overall.  The buggy
    telescoping sum measured the overall 6 dB drift instead.
    """
    sr = 16000
    n_periods = 120
    ramp = np.linspace(0.5, 1.0, n_periods)
    x = perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr,
                        amp_factors=ramp)
    js = jitter_shimmer(x, sr)
    assert js.shimmer_local_db < 0.5


def test_shimmer_rises_with_random_amplitude_perturbation():
    """±15 % cycle-to-cycle amplitude noise must raise shimmer clearly."""
    sr = 16000
    rng = np.random.default_rng(5)
    clean = jitter_shimmer(perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr), sr)
    rough = jitter_shimmer(
        perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr,
                        amp_factors=rng.normal(1.0, 0.15, 200)), sr)
    assert rough.shimmer_local_db > clean.shimmer_local_db + 0.5
    # E|20·log10(1+ε)| ≈ 8.686·0.15·√2·0.8 ≈ 1.5 dB
    assert rough.shimmer_local_db > 1.0


def test_jitter_rises_with_period_perturbation():
    """±5 % cycle-to-cycle period noise must raise jitter clearly.

    σ is capped at 0.05 deliberately — the cross-checked failure envelope:
    beyond ~5 % period jitter both this tracker and native Praat's
    autocorrelation pitch lock onto the first formant's ringing (Praat
    reports 492 Hz on the σ=0.08 version; we report 500 Hz), which
    fragments the epoch train and corrupts both jitter and shimmer.
    """
    sr = 16000
    rng = np.random.default_rng(6)
    clean = jitter_shimmer(perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr), sr)
    rough = jitter_shimmer(
        perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr,
                        period_factors=rng.normal(1.0, 0.05, 200)), sr)
    assert rough.n_periods > 100  # epoch train stayed intact
    assert rough.jitter_local_percent > clean.jitter_local_percent + 2.0
    # amplitude was untouched: shimmer must not move much
    assert rough.shimmer_local_db < clean.shimmer_local_db + 1.0


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
    assert {"F1_hz", "F2_hz", "F3_hz"} <= set(report["formants"])
    assert "formant_track" not in report  # full track only with contour=True


def test_formant_confidence_metrics():
    """The report states how much to trust each formant median."""
    sr = 16000
    x = vowel_like(f0_hz=120.0, duration_s=0.5, sr=sr)
    fmt = analyze(AudioData(samples=x, sample_rate=sr, path=None))["formants"]
    assert fmt["n_frames"] >= 5
    for k in ("F1", "F2", "F3"):
        assert 0.0 <= fmt["confidence"][k] <= 1.0
        assert fmt[f"{k}_iqr_hz"] >= 0.0
    # a clean synthetic vowel is exactly what LPC is built for: the
    # frame-wise F1 estimates should agree tightly
    assert fmt["confidence"]["F1"] >= 0.6


def test_formant_confidence_drops_when_estimate_scatters():
    """A moving formant must yield lower confidence than a steady one."""
    sr = 16000
    steady = vowel_like(f0_hz=120.0, duration_s=0.6, sr=sr,
                        formants=((600.0, 1.0), (1500.0, 0.6), (2440.0, 0.3)))
    # F1 steps 400 -> 800 Hz in short steady blocks: whichever frames the
    # energetic-half selection picks, they cannot agree on one F1
    blocks = [vowel_like(f0_hz=120.0, duration_s=0.06, sr=sr,
                         formants=((f1, 1.0), (1500.0, 0.6), (2440.0, 0.3)))
              for f1 in np.linspace(400.0, 800.0, 10)]
    moving = np.concatenate(blocks)

    s = analyze(AudioData(samples=steady, sample_rate=sr, path=None))["formants"]
    m = analyze(AudioData(samples=moving, sample_rate=sr, path=None))["formants"]
    assert m["F1_iqr_hz"] > s["F1_iqr_hz"] + 100.0  # estimates span the glide
    assert m["confidence"]["F1"] < s["confidence"]["F1"]


def test_formant_track_with_contour():
    """contour=True adds a per-frame F1-F3 track aligned in time."""
    sr = 16000
    x = vowel_like(f0_hz=120.0, duration_s=0.6, sr=sr)
    report = analyze(AudioData(samples=x, sample_rate=sr, path=None), contour=True)
    ft = report["formant_track"]
    assert set(ft) == {"times_s", "F1_hz", "F2_hz", "F3_hz"}
    assert len(ft["times_s"]) == len(ft["F1_hz"]) == len(ft["F2_hz"]) == len(ft["F3_hz"])
    assert len(ft["times_s"]) <= 200  # payload stays small
    voiced_f1 = [f for f in ft["F1_hz"] if f is not None]
    assert len(voiced_f1) > 5
    assert np.median(voiced_f1) == pytest.approx(730.0, rel=0.15)
    # unvoiced (or formant-less) frames are None, never stale values
    assert ft["times_s"] == sorted(ft["times_s"])


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
