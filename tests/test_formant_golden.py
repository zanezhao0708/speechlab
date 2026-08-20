"""Golden tests: formant/LPC estimation against known ground truth.

These are *validation* tests, not sanity tests: every signal here has a
mathematically known spectral structure (AR poles / designed resonances),
and the estimators must recover it.  Tolerances are informed by what Praat
itself deviates on the same synthetic signals (measured during development:
Praat's F1 estimates deviate by up to ~10 % and F3 by up to ~20 % from the
design on 3-formant synthetics, so we require no more of SpeechLab; where
all formant slots are occupied by designed resonances the agreement is
within a few percent).
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from speechlab.audio import AudioData
from speechlab.features import (
    FORMANT_CEILING_HZ,
    _auto_pre_emphasis,
    _resample_for_formants,
    analyze,
    formants,
    lpc,
)

# --------------------------------------------------------------------------
# helpers: signals with known spectral structure
# --------------------------------------------------------------------------

def ar_signal(pole_freqs, sr, duration_s=0.4, rho=0.97, noise=0.005, seed=0):
    """White-noise AR process whose poles sit at ``pole_freqs`` Hz."""
    rng = np.random.default_rng(seed)
    poles = []
    for f in pole_freqs:
        w = 2 * np.pi * f / sr
        poles += [rho * np.exp(1j * w), rho * np.exp(-1j * w)]
    c = np.real(np.poly(poles))
    c = c / c[0]  # denominator: 1 + c1 z^-1 + ... + cp z^-p
    n = int(sr * duration_s)
    x = np.zeros(n)
    e = rng.standard_normal(n) * noise
    p = len(c) - 1
    for i in range(p, n):
        x[i] = -sum(c[k] * x[i - k] for k in range(1, p + 1)) + e[i]
    return x[int(0.1 * sr):]  # drop start-up transient


def pulse_vowel(f0_hz, formant_freqs, sr, duration_s=0.5):
    """Source-filter vowel: impulse-train glottis through resonators.

    Unlike a free sum of damped sinusoids this is the model LPC formant
    analysis is designed for, and each designed resonance occupies one
    formant slot so no spurious pole can displace a real one.
    """
    n = int(sr * duration_s)
    src = np.zeros(n)
    pos = 0.0
    while pos < n - 1:
        src[round(pos)] = 1.0
        pos += sr / f0_hz
    x = src
    for i, f in enumerate(formant_freqs):
        bw = 70.0 + 40.0 * i
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * f / sr
        x = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], x)
    return x / (np.max(np.abs(x)) + 1e-12) * 0.8


def mid_frame(x, sr, frame_s=0.025):
    fl = int(frame_s * sr)
    return x[len(x) // 2 : len(x) // 2 + fl]


TRUTH_3 = (500.0, 1500.0, 2500.0)
TRUTH_4 = (500.0, 1500.0, 2500.0, 3500.0)   # fits an 8 kHz Nyquist
TRUTH_5 = (500.0, 1500.0, 2500.0, 3500.0, 4500.0)  # occupies all 5 slots


# --------------------------------------------------------------------------
# pre-emphasis: must be Praat's exact formula
# --------------------------------------------------------------------------

def test_pre_emphasis_matches_praat_formula():
    # Praat: alpha = exp(-2*pi*f*dt); at 10 kHz a 48.47 Hz corner == 0.97
    for sr in (8000, 10000, 16000, 44100):
        assert _auto_pre_emphasis(sr) == pytest.approx(
            np.exp(-2 * np.pi * 50.0 / sr), rel=1e-12)
    assert np.exp(-2 * np.pi * 48.47 / 10000) == pytest.approx(0.97, abs=0.001)


# --------------------------------------------------------------------------
# LPC: the [1, -a] denominator sign convention
# --------------------------------------------------------------------------

def test_lpc_returns_denominator_coefficients():
    """Roots of [1, *lpc(...)] must be the AR poles (regression test for the
    sign convention: with predictor coefficients used un-negated, a pure
    resonance yields only real roots and no formants at all)."""
    sr = 16000
    x = ar_signal([500.0], sr, seed=3)  # AR(2): one pole pair
    a = lpc(mid_frame(x, sr), 2)
    roots = np.roots(np.concatenate(([1.0], a)))
    freqs = sorted(abs(np.angle(r)) * sr / (2 * np.pi) for r in roots)
    assert freqs[0] == pytest.approx(500.0, rel=0.03)
    # pole radius ~ rho = 0.97 (damped resonance, inside the unit circle)
    assert abs(roots[0]) == pytest.approx(0.97, abs=0.03)


def test_formants_recover_known_ar_poles():
    """Two designed AR resonances recovered from a 100 ms frame (8 kHz).

    A white-noise AR process with narrow resonances becomes hard for
    autocorrelation LPC as the sample rate grows (the process is ever more
    narrowband relative to Nyquist — at 16 kHz the Toeplitz system loses
    identifiability on short frames).  That is exactly why Praat — and
    ``analyze()`` — resample to ~2× the formant ceiling before LPC; the
    end-to-end tests below cover that path at every sample rate.
    """
    for seed in (1, 2, 3):
        x = ar_signal([500.0, 1500.0], 8000, duration_s=0.4, seed=seed)
        fl = int(0.1 * 8000)
        mid = x[len(x) // 2 : len(x) // 2 + fl]
        f = formants(mid, 8000, order=4, max_formants=2, pre_emphasis=0.0)
        assert f[0] == pytest.approx(500.0, rel=0.05), (seed, f)
        assert f[1] == pytest.approx(1500.0, rel=0.05), (seed, f)


def test_formants_pulse_vowel_8k():
    """Full-default per-frame pipeline on a 4-formant source-filter vowel."""
    sr = 8000
    x = pulse_vowel(120.0, TRUTH_4, sr)
    f = formants(mid_frame(x, sr), sr)
    assert f[0] == pytest.approx(500.0, rel=0.13)
    assert f[1] == pytest.approx(1500.0, rel=0.03)
    assert f[2] == pytest.approx(2500.0, rel=0.03)


# --------------------------------------------------------------------------
# analyze(): ceiling resampling + voiced-frame gating
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sr,truth", [
    (8000, TRUTH_4),
    (16000, TRUTH_5),
    (22050, TRUTH_4),
    (44100, TRUTH_4),
    (48000, TRUTH_5),
])
def test_analyze_recovers_designed_formants_all_rates(sr, truth):
    """End-to-end regression: designed formants survive every sample rate.

    This fails without the Praat-style resampling to ~2×ceiling — at high
    sample rates naive full-rate LPC spends its poles on harmonic structure
    far above the formant region and the estimates collapse.
    """
    x = pulse_vowel(120.0, truth, sr)
    report = analyze(AudioData(samples=x, sample_rate=sr, path=None))
    fmt = report["formants"]
    assert fmt, f"no formants recovered at {sr}"
    assert fmt["F1_hz"] == pytest.approx(truth[0], rel=0.13), (sr, fmt)
    assert fmt["F2_hz"] == pytest.approx(truth[1], rel=0.04), (sr, fmt)
    assert fmt["F3_hz"] == pytest.approx(truth[2], rel=0.04), (sr, fmt)


def test_analyze_formants_require_voicing():
    """Unvoiced input must not yield formants (the loudest frames of an
    utterance are often plosive bursts — their LPC roots are not formants)."""
    silence = AudioData(samples=np.zeros(16000), sample_rate=16000, path=None)
    assert analyze(silence)["formants"] == {}


def test_analyze_formant_frames_survive_loud_burst():
    """A loud aperiodic burst mid-vowel must not leak into the medians."""
    sr = 16000
    x = pulse_vowel(120.0, TRUTH_5, sr, duration_s=0.6)
    rng = np.random.default_rng(11)
    i0 = sr // 2
    x[i0 : i0 + int(0.03 * sr)] += 0.9 * rng.standard_normal(int(0.03 * sr))
    report = analyze(AudioData(samples=x, sample_rate=sr, path=None))
    assert report["formants"]["F1_hz"] == pytest.approx(500.0, rel=0.15)


def test_resample_for_formants_targets_ceiling():
    x = np.ones(44100)
    assert _resample_for_formants(x, 8000, FORMANT_CEILING_HZ)[1] == 8000
    x16, sr16 = _resample_for_formants(x, 16000, FORMANT_CEILING_HZ)
    assert sr16 == 11000  # exact rational 11/16
    assert len(x16) == pytest.approx(44100 * 11 / 16, abs=2)
    # 44.1 kHz has no cheap exact factor: nearest decimation lands at 11025
    assert _resample_for_formants(x, 44100, FORMANT_CEILING_HZ)[1] == 11025
    assert _resample_for_formants(x, 48000, FORMANT_CEILING_HZ)[1] == 11000
    # a 5 kHz resonance must survive resampling (anti-aliasing is intact)
    sr = 16000
    t = np.arange(sr) / sr
    x5k = np.sin(2 * np.pi * 5000.0 * t)
    y, sr_y = _resample_for_formants(x5k, sr, FORMANT_CEILING_HZ)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak = np.fft.rfftfreq(len(y), 1 / sr_y)[np.argmax(sp)]
    assert peak == pytest.approx(5000.0, abs=100)
