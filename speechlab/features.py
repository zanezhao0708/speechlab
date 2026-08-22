"""Acoustic analysis core used by the research agent.

All analysers follow the same conventions:

* signals are mono float64 arrays (see :mod:`speechlab.audio`);
* frames are 25 ms long with a 10 ms hop by default;
* every function returns plain floats / numpy arrays so results can be
  JSON-serialised for LLM consumption.

Features
--------
- ``f0_track`` : autocorrelation pitch tracker with parabolic refinement,
- ``lpc`` / ``formants`` : linear prediction and formant estimation,
- ``jitter_shimmer`` : voice-quality perturbation measures,
- ``hnr`` : harmonics-to-noise ratio estimate,
- ``analyze`` : the single entry point the agent calls as a tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import AudioData, db, frame_signal

__all__ = [
    "F0Track",
    "JitterShimmer",
    "analyze",
    "compare_reports",
    "default_frame_lengths",
    "diarize",
    "f0_track",
    "formants",
    "hnr",
    "jitter_shimmer",
    "lpc",
    "pause_stats",
    "recording_quality",
    "reference_ranges",
    "spectrogram",
    "voiced_segments",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def default_frame_lengths(sr: int) -> tuple[int, int]:
    """Return (frame_length, hop_length) for 25 ms / 10 ms at ``sr``."""
    return max(1, round(sr * 0.025)), max(1, round(sr * 0.010))


def _defaults(sr: int, frame_length: int | None, hop_length: int | None):
    if frame_length is None or hop_length is None:
        fl, hl = default_frame_lengths(sr)
        frame_length = fl if frame_length is None else frame_length
        hop_length = hl if hop_length is None else hop_length
    return frame_length, hop_length


# --------------------------------------------------------------------------
# pitch
# --------------------------------------------------------------------------

@dataclass
class F0Track:
    """Result of pitch tracking.

    Attributes
    ----------
    times : np.ndarray  — frame centre times in seconds
    f0 : np.ndarray     — F0 in Hz (0 for unvoiced frames)
    voiced : np.ndarray — boolean voicing decisions
    """

    times: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray

    @property
    def voiced_ratio(self) -> float:
        """Fraction of frames marked as voiced."""
        return float(np.mean(self.voiced)) if len(self.voiced) else 0.0

    def summary(self) -> dict:
        """Descriptive statistics over voiced frames."""
        v = self.f0[self.voiced]
        if len(v) == 0:
            return {"voiced_ratio": 0.0, "n_voiced_frames": 0}
        return {
            "voiced_ratio": float(np.mean(self.voiced)),
            "n_voiced_frames": int(np.sum(self.voiced)),
            "f0_mean_hz": float(np.mean(v)),
            "f0_median_hz": float(np.median(v)),
            "f0_std_hz": float(np.std(v)),
            "f0_min_hz": float(np.min(v)),
            "f0_max_hz": float(np.max(v)),
        }


# Peak-selection constants for the pitch tracker (see _select_period).
_TOL_PEAK = 0.05    # candidate peaks must sit within this of the maximum
_DELTA_OCT = 0.008  # half-period rejection margin (see _select_period)
_TOL_MULT = 0.12    # multiple-chain validation floor, relative to maximum


def _select_period(a: np.ndarray, lags: np.ndarray) -> int | None:
    """Pick the fundamental-period lag index from one frame's NCCF ``a``.

    For a nearly periodic frame the NCCF peaks at T, 2T, 3T… are almost
    equally high — their differences are of the same order as per-frame
    jitter/noise (~0.01) — so "the global maximum" is a coin flip between
    subharmonics, and "first lag above 85 % of max" grabs peak *slopes*.
    Instead:

    1. candidates = local maxima within ``_TOL_PEAK`` of the maximum;
    2. walk from the shortest lag and *reject* a candidate P when the
       peak at 2P is clearly stronger (``_DELTA_OCT``): P is then just
       half of the true period, which happens when a formant lands on an
       even harmonic and the waveform nearly repeats at T/2 (the
       octave-up trap);
    3. accept P when every small multiple m·P also shows a peak
       (``_TOL_MULT``): a spurious 2T/3T candidate has no peak chain at
       the true-period multiples it would need;
    4. the first survivor wins — the shortest valid lag is the
       fundamental period.
    """
    n = len(a)
    lag0, lag_last = int(lags[0]), int(lags[-1])
    idx = [i for i in range(n)
           if a[i] >= (a[i - 1] if i > 0 else -np.inf)
           and a[i] >= (a[i + 1] if i < n - 1 else -np.inf) and a[i] > 0]
    if not idx:
        return None
    m_max = float(a.max())
    cands = sorted(i for i in idx if a[i] >= m_max - _TOL_PEAK)
    if not cands:
        cands = [max(idx, key=lambda i: a[i])]
    for i in cands:
        period = int(lags[i])
        if 2 * period <= lag_last:
            w2 = max(1, round(0.08 * period))
            lo = max(0, 2 * period - w2 - lag0)
            hi = min(n - 1, 2 * period + w2 - lag0)
            if lo <= hi and a[lo:hi + 1].max() > a[i] + _DELTA_OCT:
                continue  # half-period of a clearly stronger 2P peak
        k_max = min(lag_last // period, 6)
        if k_max < 2:
            return i  # too long to verify multiples — take it on trust
        ok = True
        for m in range(1, k_max + 1):
            center = m * period
            wm = max(1, round(0.04 * center))
            lo = max(0, center - wm - lag0)
            hi = min(n - 1, center + wm - lag0)
            if lo > hi or a[lo:hi + 1].max() < m_max - _TOL_MULT:
                ok = False
                break
        if ok:
            return i
    return max(idx, key=lambda i: a[i])


def f0_track(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
             frame_length: int | None = None, hop_length: int | None = None,
             voicing_threshold: float = 0.35,
             energy_floor_db: float = -55.0) -> F0Track:
    """Autocorrelation F0 tracker with octave-robust peak selection.

    The NCCF is Boersma's (1993) window-corrected autocorrelation —
    ``r(lag) = ACF(x·w)(lag) / ACF(w)(lag)``, normalised by ``r(0)`` —
    computed over a window of 3 periods of ``fmin`` (Praat's convention).
    Period selection runs on peaks of the temporally smoothed NCCF via
    :func:`_select_period`; sub-sample precision comes from parabolic
    interpolation on the raw NCCF.

    Parameters
    ----------
    fmin, fmax : pitch search range in Hz (60–500 covers adult speech).
    voicing_threshold : minimum normalised autocorrelation for a voiced frame.
    energy_floor_db : frames below this RMS dB are never voiced.
    frame_length : analysis window in samples.  ``None`` uses
        ``max(25 ms, 3/fmin)``; if given, it should be at least twice the
        longest lag (``2·sr/fmin``) or the search range is clipped.
    """
    # 3 periods of the lowest candidate F0, at least 25 ms: the NCCF at
    # lags near 1/fmin needs >= 2 periods of overlap to be trustworthy.
    if frame_length is None:
        frame_length = max(round(sr * 0.025), int(np.ceil(3.0 * sr / fmin)))
    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = min(int(np.ceil(sr / fmin)), frame_length // 2)
    if lag_min >= lag_max:
        raise ValueError("pitch search range does not fit the analysis frame")

    frames = frame_signal(samples, frame_length, hop_length, window="rect", center=True)
    n_frames = len(frames)
    times = (np.arange(n_frames) * hop_length + frame_length / 2 - frame_length // 2) / sr

    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)

    if n_frames == 0:
        return F0Track(times=times, f0=f0, voiced=voiced)

    # energy gate
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = db(rms ** 2)
    active = rms_db > energy_floor_db

    lags = np.arange(lag_min, lag_max + 1)
    act_idx = np.where(active)[0]
    if len(act_idx) == 0:
        return F0Track(times=times, f0=f0, voiced=voiced)

    # Window-corrected autocorrelation (Boersma 1993).  Dividing by the
    # window's own autocorrelation undoes the taper, so a perfectly
    # periodic frame scores ~1 at EVERY integer multiple of its period.
    # The overlap-length normalisation used previously
    # (num / sqrt(E_left·E_right)) inflates long lags — the shrinking
    # overlap divides away more energy than the numerator loses — which
    # systematically biased the global maximum towards 2T/3T.
    w = np.hanning(frame_length)
    fx = (frames[act_idx] - frames[act_idx].mean(axis=1, keepdims=True)) * w
    n_fft = 1 << (2 * frame_length - 1).bit_length()
    FX = np.fft.rfft(fx, n_fft, axis=1)
    acf = np.fft.irfft(FX * np.conj(FX), n_fft, axis=1)
    FW = np.fft.rfft(w, n_fft)
    acf_w = np.fft.irfft(FW * np.conj(FW), n_fft)[:frame_length]
    r = acf[:, lags] / np.maximum(acf_w[lags], 1e-12)
    r0 = acf[:, 0] / max(acf_w[0], 1e-12)
    nccf = r / np.maximum(r0, 1e-12)[:, None]

    # 3-frame temporal smoothing feeds only the octave decision: the
    # half-period vs full-period NCCF contrast (~0.01) is the same order
    # as per-frame noise, and averaging three frames recovers it.
    sm = nccf.copy()
    if len(sm) >= 3:
        sm[1:-1] = (nccf[:-2] + nccf[1:-1] + nccf[2:]) / 3.0

    for row, fi in enumerate(act_idx):
        if float(sm[row].max()) < voicing_threshold:
            continue
        j = _select_period(sm[row], lags)
        if j is None:
            continue
        # parabolic interpolation on the raw (unsmoothed) NCCF
        raw = nccf[row]
        lag_est = float(lags[j])
        if 0 < j < len(raw) - 1:
            y0, y1, y2 = raw[j - 1], raw[j], raw[j + 1]
            denom_p = y0 - 2 * y1 + y2
            if abs(denom_p) > 1e-9:
                lag_est = lags[j] + 0.5 * (y0 - y2) / denom_p
        if lag_est > 0:
            f0[fi] = sr / lag_est
            voiced[fi] = True

    # octave-jump suppression: median filter over voiced frames
    if np.any(voiced):
        from scipy.signal import medfilt

        v_idx = np.where(voiced)[0]
        if len(v_idx) >= 3:
            f0[v_idx] = medfilt(f0[v_idx], 3)

    return F0Track(times=times, f0=f0, voiced=voiced)


# --------------------------------------------------------------------------
# LPC / formants
# --------------------------------------------------------------------------

def lpc(samples: np.ndarray, order: int) -> np.ndarray:
    """Linear prediction denominator coefficients (a_1..a_order).

    Solves the Yule-Walker (Toeplitz) equations and returns the NEGATED
    predictor coefficients, so that the all-pole filter denominator is
    ``1 + a_1·z⁻¹ + … + a_order·z⁻ᵒʳᵈᵉʳ`` (leading 1 omitted) and the
    resonance frequencies are simply the roots of ``[1, *a]``.
    """
    from scipy.linalg import solve_toeplitz

    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    if len(x) <= order:
        raise ValueError("frame too short for the requested LPC order")
    r = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) - 1 + order + 1]
    if np.allclose(r[0], 0.0):
        return np.zeros(order)
    a = solve_toeplitz(r[:order], r[1 : order + 1])
    return -np.asarray(a, dtype=np.float64)


def _auto_pre_emphasis(sr: int) -> float:
    """Pre-emphasis coefficient for a 50 Hz corner, exactly as Praat does it.

    Praat's ``Pre-emphasize (from frequency f)`` uses ``α = exp(−2π·f·Δt)``
    with ``Δt = 1/sr`` (Praat manual: at 10 kHz, a corner of 48.47 Hz
    corresponds to the classic α = 0.97).  Applied as ``x[n] − α·x[n−1]``.
    """
    return float(np.exp(-2.0 * np.pi * 50.0 / sr))


def formants(samples: np.ndarray, sr: int, order: int | None = None,
             pre_emphasis: float | None = None,
             max_formants: int = 5) -> list[float]:
    """Estimate formant frequencies (Hz) for one analysis frame via LPC roots.

    Parameters
    ----------
    order : LPC order; defaults to ``2·max_formants`` (one complex pole
        pair per formant — Praat's Formant(Burg) uses the same rule and
        defaults to 5 formants / 10 poles).
    pre_emphasis : coefficient; ``None`` derives the 50 Hz-corner
        Praat coefficient from the sample rate.  Pass ``0.0`` to disable.

    Note
    ----
    Praat resamples to twice the formant ceiling before LPC (without it,
    formant estimates at high sample rates are markedly off — the LPC
    poles get spent on harmonic structure above the formant region).
    :func:`analyze` does this resampling; when calling ``formants`` on
    full-rate frames yourself, prefer frames already at ≤ 2×ceiling rate.
    """
    if order is None:
        order = 2 * max_formants
    if pre_emphasis is None:
        pre_emphasis = _auto_pre_emphasis(sr)
    x = np.asarray(samples, dtype=np.float64)
    if pre_emphasis:
        x = np.append(x[0], x[1:] - pre_emphasis * x[:-1])
    a = lpc(x, order)
    poly = np.concatenate(([1.0], a))
    roots = np.roots(poly)

    freqs = []
    for r in roots:
        if abs(r.imag) <= 1e-10:
            continue
        freq = np.angle(r) * sr / (2 * np.pi)
        if 90.0 < freq < sr / 2.0 - 50.0:  # skip DC/Nyquist-adjacent roots
            freqs.append((freq, abs(r)))
    freqs.sort()
    out = [float(f) for f, _ in freqs[:max_formants]]
    return out


# --------------------------------------------------------------------------
# voice quality
# --------------------------------------------------------------------------

@dataclass
class JitterShimmer:
    """Perturbation measures computed from glottal-pulse epochs."""

    jitter_local_percent: float
    shimmer_local_db: float
    n_periods: int

    def summary(self) -> dict:
        return {
            "jitter_local_percent": self.jitter_local_percent,
            "shimmer_local_db": self.shimmer_local_db,
            "n_periods": self.n_periods,
        }


def _find_epochs(samples: np.ndarray, sr: int, fmin: float = 60.0,
                 fmax: float = 500.0,
                 track: F0Track | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Locate glottal pulse epochs (waveform peaks) for perturbation measures.

    A rough F0 is estimated first (or reused from ``track``), the signal is
    low-pass smoothed over a quarter period to suppress formant ripple, and
    peaks are then picked with a minimum spacing of 0.6·period.

    Returns ``(epoch_indices, period_amplitudes)`` where
    ``period_amplitudes[k]`` is the peak-to-peak swing of the raw signal
    over period ``k``.  Epochs (smoothed-waveform peaks) lag the true
    excitation by a roughly constant offset, so a window from one epoch to
    the next straddles the period boundary: it mixes the current period's
    decay with the next period's onset and attenuates exactly the
    cycle-to-cycle amplitude contrast shimmer is supposed to measure.
    Amplitudes are therefore taken between the *attacks* — the sharpest
    waveform slopes preceding each epoch — which track the actual glottal
    periods (as Praat's PointProcess pulses do).
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < int(sr / fmin) * 3:
        return np.empty(0, dtype=int), np.empty(0)

    if track is None:
        track = f0_track(x, sr, fmin=fmin, fmax=fmax)
    voiced_f0 = track.f0[track.voiced]
    if len(voiced_f0) == 0:
        return np.empty(0, dtype=int), np.empty(0)
    period = sr / float(np.median(voiced_f0))

    # invert if the waveform is dominantly negative so peaks align with pulses
    if np.percentile(x, 75) < 0:
        x = -x

    # low-pass smoothing kills inter-pulse formant oscillation
    win_len = max(3, round(period * 0.25))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.hanning(win_len)
    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
    xs = np.convolve(x, kernel, mode="same")

    threshold = 0.1 * np.max(np.abs(xs))
    epochs: list[int] = []
    n = len(xs)

    # seed with the first local maximum above threshold, then walk: the
    # next epoch is the waveform maximum inside [0.75, 1.25]·period after
    # the last one.  Taking the window maximum (rather than the first
    # above-threshold peak) avoids locking onto formant ringing between
    # glottal pulses, which used to split periods and inflate jitter and
    # shimmer on impulse-excited vowels.
    i = 1
    while i < n - 1 and not epochs:
        if xs[i] > threshold and xs[i] >= xs[i - 1] and xs[i] > xs[i + 1]:
            epochs.append(i)
        i += 1
    if epochs:
        lo = max(1, round(0.75 * period))
        hi = max(lo + 1, round(1.25 * period))
        anchor = epochs[0]
        while anchor + hi <= n:  # only full windows: a clipped final
            w0 = anchor + lo     # window max is a ringing tail, not a pulse
            j = int(np.argmax(xs[w0:anchor + hi]))
            if xs[w0 + j] > threshold:
                epochs.append(w0 + j)
                anchor = w0 + j
            else:  # silent stretch: skip a window, resync afterwards
                anchor += hi

    epochs_arr = np.asarray(epochs, dtype=int)
    if len(epochs_arr) >= 2:
        # drop periods outside a *toleranced* plausible F0 range.  Strict
        # bounds flicker for signals sitting on the fmin/fmax edge (a
        # 60 Hz-tracked epoch train has periods straddling 1/fmin) and
        # mangle the epoch set instead of gating it.
        periods = np.diff(epochs_arr) / sr
        good = (periods >= 0.8 / fmax) & (periods <= 1.25 / fmin)
        if not np.all(good):
            keep = [epochs_arr[0]]
            for k in range(1, len(epochs_arr)):
                if good[k - 1]:
                    keep.append(epochs_arr[k])
            epochs_arr = np.asarray(keep, dtype=int)
    if len(epochs_arr) < 2:
        return epochs_arr, np.empty(0)
    # robust per-period amplitude: the full swing between consecutive
    # *attacks* (sharpest slopes), not between the smoothed-peak epochs —
    # see the docstring for why the epoch-to-epoch window leaks the next
    # period's onset into the current period's amplitude.
    half = round(0.5 * period)
    attacks = np.empty(len(epochs_arr), dtype=int)
    for k, e in enumerate(epochs_arr):
        lo = max(0, e - half)
        seg = np.abs(np.diff(x[lo:e + 1]))
        attacks[k] = lo + int(np.argmax(seg)) + 1 if len(seg) else e
    amps = np.array([float(np.ptp(x[attacks[k]:max(attacks[k] + 1, attacks[k + 1])]))
                     for k in range(len(attacks) - 1)])
    return epochs_arr, amps


def jitter_shimmer(samples: np.ndarray, sr: int, fmin: float = 60.0,
                   fmax: float = 500.0,
                   track: F0Track | None = None) -> JitterShimmer:
    """Local jitter (%) and shimmer (dB) from consecutive glottal periods.

    Jitter follows Praat's *jitter (local)*: the average absolute difference
    between consecutive periods, divided by the mean period.  Shimmer
    follows *shimmer (local, dB)*: the average of ``|20·log10(A_{k+1}/A_k)|``
    over consecutive periods, with ``A`` the peak-to-peak amplitude inside
    each period (validated against native Praat on perturbed synthetic
    vowels; see ``benchmarks/praat_benchmark.py``).

    Commonly cited sustained-vowel screening values are jitter < 1 %,
    shimmer < 0.4 dB — but these thresholds are algorithm-, recording- and
    population-dependent (Praat's own docs stress the sustained-vowel
    requirement); treat them as research triage, not diagnosis.  Pass a
    precomputed ``track`` from :func:`f0_track` to avoid recomputing pitch.

    Degenerate periodicity: amplitudes that repeat exactly every k periods
    (strict pulse alternans, k = 2) shift the best waveform repetition to
    F0/k — native Praat tracks such signals at the subharmonic and so does
    this tracker, which then averages the loud/quiet pulses together and
    under-reports shimmer.  Period-based perturbation measures are only
    well-defined when the period itself is unambiguous; treat strict
    alternation accordingly.
    """
    epochs, amps = _find_epochs(samples, sr, fmin, fmax, track=track)
    if len(epochs) < 3 or len(amps) < 2:
        return JitterShimmer(jitter_local_percent=float("nan"),
                             shimmer_local_db=float("nan"), n_periods=0)

    periods = np.diff(epochs) / sr
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100.0)
    return JitterShimmer(jitter_local_percent=jitter,
                         shimmer_local_db=_shimmer_local_db(amps),
                         n_periods=len(periods))


def _shimmer_local_db(amps: np.ndarray) -> float:
    """Praat's *shimmer (local, dB)* from per-period amplitudes.

    The average of ``|20·log10(A_{k+1}/A_k)|`` over consecutive periods.
    The absolute value must sit on the dB steps themselves: applied to the
    signed values (or omitted) the average telescopes down to
    ``20·log10(A_last/A_first)`` — a signal whose amplitudes swing
    1 → 2 → 1 → 2 … would report ~0 dB instead of 6.02 dB, measuring
    envelope drift rather than cycle-to-cycle perturbation.
    """
    amps = np.asarray(amps, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_ratios = amps[1:] / np.where(np.abs(amps[:-1]) < 1e-12, np.nan, amps[:-1])
        db_steps = 20.0 * np.log10(amp_ratios)
        return float(np.nanmean(np.abs(db_steps))) \
            if np.any(np.isfinite(db_steps)) else float("nan")


def hnr(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
        frame_length: int | None = None, hop_length: int | None = None,
        track: F0Track | None = None) -> float:
    """Harmonics-to-noise ratio in dB, estimated from autocorrelation.

    Uses the relation HNR ≈ 10·log10(r/(1−r)) at the best F0 lag, averaged
    over voiced frames.  Values above ~20 dB indicate a tonal, stable voice.
    Pass a precomputed ``track`` from :func:`f0_track` to avoid recomputing
    pitch.
    """
    if track is None:
        track = f0_track(samples, sr, fmin=fmin, fmax=fmax,
                         frame_length=frame_length, hop_length=hop_length)
    if not np.any(track.voiced):
        return float("nan")

    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    frames = frame_signal(samples, frame_length, hop_length, window="rect", center=True)
    vals = []
    for i in np.where(track.voiced)[0]:
        if i >= len(frames):
            continue
        fx = frames[i]
        fx = fx - fx.mean()
        d2 = np.sum(fx ** 2)
        if d2 < 1e-10:
            continue
        lag = round(sr / max(track.f0[i], 1e-6))
        if lag <= 0 or lag >= frame_length:
            continue
        num = float(np.sum(fx[:-lag] * fx[lag:]))
        # NCCF-style normalisation over the overlapping segments
        e1 = float(np.sum(fx[:-lag] ** 2))
        e2 = float(np.sum(fx[lag:] ** 2))
        r_val = num / max(np.sqrt(e1 * e2), 1e-20)
        # r <= 0 means no periodicity evidence at all: Praat reports
        # "undefined" for such frames.  Clamping them to r = 1e-6 instead
        # would inject a made-up -60 dB into the average and drag the
        # summary down arbitrarily — skip them like Praat does.
        if r_val <= 0.0:
            continue
        r_val = min(r_val, 0.999999)
        vals.append(10.0 * np.log10(r_val / (1.0 - r_val)))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------
# recording quality & segmentation
# --------------------------------------------------------------------------

def recording_quality(samples: np.ndarray, sr: int) -> dict:
    """Pre-analysis recording checks researchers need before trusting numbers.

    Returns peak level (dBFS), clipping ratio, broadband SNR estimate
    (signal vs. quietest-10 % frames), and a list of issues that make the
    recording unsuitable for perturbation measures.
    """
    x = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    peak_db = float(db(peak ** 2)) if peak > 0 else -120.0

    # clipping: only *runs* of consecutive samples pinned at full scale count —
    # a lone peak sample (e.g. from lossless normalisation) is not clipping
    at_peak = np.abs(x) >= 0.999 if len(x) else np.zeros(0, dtype=bool)
    clip_samples = 0
    if at_peak.any():
        edges = np.diff(np.concatenate(([False], at_peak, [False])).astype(np.int8))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        run_lens = ends - starts
        clip_samples = int(run_lens[run_lens >= 3].sum())
    clip_ratio = clip_samples / len(x) if len(x) else 0.0
    clipping = clip_samples > 0

    frame_length, hop_length = default_frame_lengths(sr)
    frames = frame_signal(x, frame_length, hop_length, window="rect", center=True)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12) if len(frames) else np.empty(0)
    if len(rms):
        rms_db = db(rms ** 2)
        noise_db = float(np.percentile(rms_db, 10))   # quietest frames ≈ noise floor
        signal_db = float(np.percentile(rms_db, 90))  # energetic frames ≈ signal
        # a sustained vowel / tone has no quiet gaps, so the percentile gap
        # is tiny and the "noise floor" is meaningless — report unknown
        snr_db = round(signal_db - noise_db, 1) if signal_db - noise_db >= 6.0 else float("nan")
    else:
        snr_db = float("nan")

    issues = []
    if len(x) == 0:
        issues.append("empty file")
    else:
        if clipping:
            issues.append("clipping detected — re-record at lower gain")
        if peak_db < -30.0:
            issues.append("recording too quiet — check microphone/gain")
        if np.isfinite(snr_db) and snr_db < 20.0:
            issues.append("low SNR — perturbation measures unreliable")
    return {
        "peak_dbfs": round(peak_db, 1),
        "clipping_ratio": round(clip_ratio, 5),
        "snr_db": snr_db,
        "issues": issues,
    }


def voiced_segments(track: F0Track, min_len_s: float = 0.3,
                    max_gap_s: float = 0.1) -> list[dict]:
    """Contiguous voiced stretches (e.g. sustained vowels) in a recording.

    Merges voiced frames separated by gaps shorter than ``max_gap_s`` and
    drops segments shorter than ``min_len_s``.  Useful for locating the
    analysis-worthy parts of a long recording.
    """
    if len(track.times) == 0:
        return []
    segs: list[dict] = []
    start: float | None = None
    last: float = 0.0
    for t, v in zip(track.times, track.voiced):
        if v:
            if start is None or t - last > max_gap_s:
                if start is not None and last - start >= min_len_s:
                    segs.append({"start_s": round(start, 3), "end_s": round(last, 3)})
                start = t
            last = t
        elif start is not None and t - last > max_gap_s:
            if last - start >= min_len_s:
                segs.append({"start_s": round(start, 3), "end_s": round(last, 3)})
            start = None
    if start is not None and last - start >= min_len_s:
        segs.append({"start_s": round(start, 3), "end_s": round(last, 3)})
    return segs


# --------------------------------------------------------------------------
# spectrogram & temporal structure (pauses, speech rate)
# --------------------------------------------------------------------------

def spectrogram(samples: np.ndarray, sr: int, max_time_bins: int = 120,
                max_freq_bins: int = 80, fmax_hz: float = 5000.0) -> dict:
    """Compact log-magnitude spectrogram for visualisation.

    Returns a downsampled dB matrix (``values[freq_bin][time_bin]``) plus the
    axis ranges, small enough to ship as JSON.
    """
    x = np.asarray(samples, dtype=np.float64)
    frame_length, hop_length = default_frame_lengths(sr)
    frames = frame_signal(x, frame_length, hop_length, window="hann", center=True)
    if len(frames) == 0:
        return {"times_s": [], "freqs_hz": [], "values_db": [], "t_max": 0.0}

    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)
    keep = freqs <= fmax_hz
    spec, freqs = spec[:, keep], freqs[keep]
    spec_db = db(spec.T)  # (freq_bins, time_bins)
    n_frames_total = spec_db.shape[1]

    # block-average to the target resolution
    def _pool(a: np.ndarray, axis: int, target: int) -> np.ndarray:
        n = a.shape[axis]
        if n <= target:
            return a
        size = int(np.ceil(n / target))
        trim = (n // size) * size
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(0, trim)
        a = a[tuple(sl)]
        return a.reshape(*a.shape[:axis], -1, size, *a.shape[axis + 1:]).mean(axis=axis + 1)

    spec_db = _pool(spec_db, 0, max_freq_bins)   # pool frequency rows …
    spec_db = _pool(spec_db, 1, max_time_bins)   # … and time columns
    freqs = _pool(freqs, 0, max_freq_bins)

    # pooled bins each cover n_frames_total/cols frames of hop_length/sr
    bin_span_s = n_frames_total * hop_length / sr / spec_db.shape[1]
    times = (np.arange(spec_db.shape[1]) + 0.5) * bin_span_s
    return {
        "t_max": round(float(len(x) / sr), 3),
        "times_s": [round(float(t), 3) for t in times],
        "freqs_hz": [round(float(f)) for f in freqs],
        "values_db": [[round(float(v), 1) for v in row] for row in spec_db],
    }


def pause_stats(samples: np.ndarray, sr: int, min_pause_s: float = 0.2,
                min_speech_db_rel: float = 32.0) -> dict:
    """Temporal structure: silence ratio, pauses, and a syllable-rate estimate.

    Pauses are energy-gated silence runs of at least ``min_pause_s``; syllable
    nuclei are counted as peaks of the smoothed energy envelope (de Jong &
    Wempe 2009 style) — a useful rough articulation-rate estimate for
    connected speech, not a substitute for forced alignment.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < sr // 4:
        return {}

    frame_length, hop_length = default_frame_lengths(sr)
    frames = frame_signal(x, frame_length, hop_length, window="rect", center=True)
    rms_db = db(np.mean(frames ** 2, axis=1) + 1e-12)
    t_frame = hop_length / sr

    peak_db = float(np.max(rms_db))
    speech_thr = peak_db - min_speech_db_rel
    is_speech = rms_db > speech_thr
    n_frames = len(rms_db)

    # silence runs >= min_pause_s sandwiched by speech
    pauses: list[float] = []
    run = 0
    seen_speech = False
    for v in is_speech:
        if not v:
            run += 1
            continue
        if seen_speech and run * t_frame >= min_pause_s:
            pauses.append(round(run * t_frame, 3))
        seen_speech = True
        run = 0
    speech_frames = int(np.sum(is_speech))
    speech_s = speech_frames * t_frame

    # syllable nuclei: peaks of the low-passed speech-region energy envelope
    env = 10 ** (rms_db / 20)
    win = max(3, round(0.05 / t_frame) | 1)  # ~50 ms smoothing
    kernel = np.hanning(win)
    kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
    env_s = np.convolve(env, kernel, mode="same")
    min_dist = max(1, round(0.12 / t_frame))  # ≥120 ms between nuclei
    thr = 0.25 * float(np.max(env_s))
    n_syll = 0
    i = 1
    last_peak = -10**9
    while i < n_frames - 1:
        if (is_speech[i] and env_s[i] > thr
                and env_s[i] >= env_s[i - 1] and env_s[i] > env_s[i + 1]
                and i - last_peak >= min_dist
                and _peak_prominence(env_s, i) >= 0.5):
            n_syll += 1
            last_peak = i
            i += min_dist
            continue
        i += 1

    return {
        "speech_s": round(speech_s, 2),
        "silence_ratio": round(1.0 - speech_frames / n_frames, 3),
        "n_pauses": len(pauses),
        "pause_total_s": round(float(np.sum(pauses)), 2),
        "pause_mean_s": round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "pause_max_s": round(float(np.max(pauses)), 3) if pauses else 0.0,
        "syllable_est": int(n_syll),
        "articulation_rate_syl_per_s": round(n_syll / speech_s, 2) if speech_s > 0.2 else None,
    }


# --------------------------------------------------------------------------
# speaker diarization (lightweight, offline)
# --------------------------------------------------------------------------

def _peak_prominence(env: np.ndarray, i: int, span: int = 0) -> float:
    """Prominence of env[i]: peak height over the higher adjacent valley.

    Valleys are searched within ~120 ms on each side (bounded by larger
    neighbours).  A real syllable nucleus rises clearly out of its valleys;
    ripple on a long vowel does not.
    """
    n = len(env)
    look = span or max(2, round(0.12 / 0.01))
    lo = max(0, i - look)
    hi = min(n, i + look + 1)
    seg = env[lo:hi]
    if len(seg) < 3:
        return 0.0
    peak = env[i]
    left = env[lo:i + 1]
    right = env[i:hi]
    lv = float(np.min(left))
    rv = float(np.min(right))
    valley = max(lv, rv)
    if peak <= 0:
        return 0.0
    return (peak - valley) / peak


def _logmel(x: np.ndarray, sr: int, n_filters: int = 20) -> np.ndarray:
    """Frame-wise log-Mel-band energies (simple triangular filterbank)."""
    frame_length, hop_length = default_frame_lengths(sr)
    frames = frame_signal(x, frame_length, hop_length, window="hann", center=True)
    if len(frames) == 0:
        return np.zeros((0, n_filters))
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)
    fmin, fmax = 80.0, min(6000.0, sr / 2 - 200.0)
    centers = np.geomspace(fmin, fmax, n_filters + 2)
    out = np.zeros((len(frames), n_filters))
    for k in range(n_filters):
        lo, mid, hi = centers[k], centers[k + 1], centers[k + 2]
        m = ((freqs >= lo) & (freqs <= hi)).astype(float)
        tri = np.zeros_like(freqs)
        idx = np.where(m > 0)[0]
        if len(idx) == 0:
            continue
        rising = (freqs[idx] - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs[idx]) / max(hi - mid, 1e-9)
        tri[idx] = np.minimum(rising, falling)
        out[:, k] = np.sqrt(np.maximum(spec @ tri, 1e-12))
    return np.log(out + 1e-10)


def diarize(audio: AudioData, n_speakers: int = 2, min_turn_s: float = 0.4,
            max_speakers: int = 4) -> dict:
    """Who speaks when — lightweight offline diarization.

    Frames of log-Mel spectral shape (energy-normalised) are clustered with
    agglomerative average linkage over cosine distance; energy gating keeps
    only speech frames.  ``n_speakers`` may be ``0`` for auto (1..max) via
    the largest silhouette.  Output includes per-speaker speaking time and
    merged turns.  This is a compact classical system — expect ~80-90 % frame
    accuracy on clean two-speaker audio, not production-grade diarization.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    sr, x = audio.sample_rate, np.asarray(audio.samples, dtype=np.float64)
    feats = _logmel(x, sr)
    if len(feats) < 10:
        return {"n_speakers": 0, "speaking_time_s": {}, "turns": [],
                "note": "recording too short for diarization"}

    frame_length, hop_length = default_frame_lengths(sr)
    t_frame = hop_length / sr
    rms = np.sqrt(np.mean(
        frame_signal(x, frame_length, hop_length, window="rect", center=True) ** 2,
        axis=1) + 1e-12)
    speech = rms > 0.15 * float(np.max(rms)) if len(rms) else np.zeros(len(feats), bool)
    speech = speech[:len(feats)]
    idx = np.where(speech)[0]
    if len(idx) < 10:
        return {"n_speakers": 0, "speaking_time_s": {}, "turns": [],
                "note": "no sustained speech detected"}

    # energy normalisation: speaker identity lives in spectral shape, not level
    sub = feats[idx].copy()
    sub -= sub.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sub = sub / np.maximum(norms, 1e-10)

    def _cluster(k: int) -> np.ndarray:
        link = linkage(sub, method="average", metric="cosine")
        return fcluster(link, t=k, criterion="maxclust") - 1

    def _silhouette(labels: np.ndarray) -> float:
        from scipy.spatial.distance import cdist
        ks = np.unique(labels)
        if len(ks) < 2:
            return -1.0
        cents = np.vstack([sub[labels == k].mean(axis=0) for k in ks])
        d = cdist(sub, cents, "cosine")
        own = d[np.arange(len(labels)), labels]
        other = np.min(d + np.eye(len(ks))[labels] * 10, axis=1)
        s = np.mean((other - own) / np.maximum(np.maximum(own, other), 1e-10))
        return float(s)

    if n_speakers and n_speakers > 1:
        best = _cluster(n_speakers)
    else:
        best, best_s = None, -2.0
        for k in range(2, max_speakers + 1):
            lab = _cluster(k)
            s = _silhouette(lab)
            if s > best_s:
                best, best_s = lab, s
        if best is None:  # single speaker
            best = np.zeros(len(idx), dtype=int)

    labels = np.zeros(len(feats), dtype=int)
    labels[idx] = best

    # majority smoothing over ±5 frames (50 ms) to kill spurious flickers
    from scipy.signal import medfilt
    labels_s = medfilt(labels, 5).astype(int)

    # merge into turns
    turns: list[dict] = []
    cur = labels_s[0]
    start = 0.0
    for i in range(1, len(labels_s)):
        if labels_s[i] != cur:
            turns.append({"speaker": int(cur),
                          "start_s": round(start, 2),
                          "end_s": round(i * t_frame, 2)})
            cur = labels_s[i]
            start = i * t_frame
    turns.append({"speaker": int(cur), "start_s": round(start, 2),
                  "end_s": round(len(labels_s) * t_frame, 2)})

    # drop too-short turns by folding them into the previous kept turn
    kept: list[dict] = []
    for t in turns:
        if kept and t["end_s"] - t["start_s"] < min_turn_s:
            kept[-1]["end_s"] = t["end_s"]  # absorb
        else:
            kept.append(dict(t))
    # re-merge consecutive same-speaker turns created by absorption
    merged: list[dict] = []
    for t in kept:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["end_s"] = t["end_s"]
        else:
            merged.append(t)

    speaking = {int(k): round(float(np.sum(labels_s == k) * t_frame), 1)
                for k in np.unique(labels_s)}
    return {"n_speakers": len(speaking), "speaking_time_s": speaking,
            "turns": merged}


# --------------------------------------------------------------------------
# statistical comparison of two analyses
# --------------------------------------------------------------------------

#: commonly cited screening ranges (adults, sustained vowel, unless noted).
#: Research triage only: jitter/shimmer/HNR thresholds are algorithm-,
#: recording- and population-dependent — they are not diagnostic criteria.
_NORMS: dict[str, dict] = {
    "f0_hz": {"men": (85, 180), "women": (165, 255), "children": (250, 350),
              "note": "modal speaking pitch"},
    "jitter_percent": {"typical": (0.0, 1.0), "borderline": (1.0, 1.5),
                       "note": "local jitter, sustained vowel; thresholds "
                               "are algorithm- and population-dependent"},
    "shimmer_db": {"typical": (0.0, 0.35), "borderline": (0.35, 0.7),
                   "note": "local shimmer, sustained vowel; thresholds "
                           "are algorithm- and population-dependent"},
    "hnr_db": {"typical": (20.0, 45.0), "borderline": (15.0, 20.0),
               "note": ">20 dB suggests stable phonation"},
    "snr_db": {"typical": (30.0, 60.0), "note": "recording quality target"},
}


def reference_ranges(metric: str = "") -> dict:
    """Lookup table of literature screening ranges (for the agent)."""
    if not metric:
        return _NORMS
    key = metric.lower().replace(" ", "_")
    for k, value in _NORMS.items():
        if k.split("_")[0] in key or key in k:
            return {k: value}
    return {"error": f"no reference data for '{metric}'",
            "available": sorted(_NORMS)}


def compare_reports(a: dict, b: dict) -> dict:
    """Compare two analyze() reports with inferential statistics.

    Welch t-test (F0 contour samples where available) and Cohen's d effect
    sizes for scalar summaries.
    """
    from scipy import stats

    def _get(rep, keys, default=None):
        cur = rep
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    pa, pb = a.get("pitch", {}), b.get("pitch", {})
    ja, jb = a.get("voice_quality", {}), b.get("voice_quality", {})
    metrics = [
        ("f0_median_hz", pa.get("f0_median_hz"), pb.get("f0_median_hz"), "Hz"),
        ("jitter_percent", ja.get("jitter_local_percent"), jb.get("jitter_local_percent"), "%"),
        ("shimmer_db", ja.get("shimmer_local_db"), jb.get("shimmer_local_db"), "dB"),
        ("hnr_db", a.get("hnr_db"), b.get("hnr_db"), "dB"),
    ]
    rows = []
    for name, va, vb, unit in metrics:
        if va is None or vb is None:
            continue
        row = {"metric": name, "a": round(float(va), 2), "b": round(float(vb), 2),
               "diff_b_minus_a": round(float(vb) - float(va), 2), "unit": unit}
        # pooled-std effect size when std available (F0 only)
        sa, sb = pa.get("f0_std_hz"), pb.get("f0_std_hz")
        if name == "f0_median_hz" and sa and sb:
            sp = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
            row["cohens_d"] = round((float(vb) - float(va)) / sp, 2) if sp > 1e-9 else None
        rows.append(row)

    out: dict = {"metrics": rows}

    # Welch t-test on the two F0 contour distributions (when present)
    ca, cb = a.get("pitch_contour"), b.get("pitch_contour")
    if ca and cb:
        fa = np.array([v for v in ca["f0_hz"] if v is not None], dtype=float)
        fb = np.array([v for v in cb["f0_hz"] if v is not None], dtype=float)
        if len(fa) >= 5 and len(fb) >= 5:
            res = stats.ttest_ind(fa, fb, equal_var=False)
            t, p = float(res.statistic), float(res.pvalue)
            # Welch–Satterthwaite df.  n1+n2-2 would be the *pooled*-variance
            # df — reporting it under a "Welch" label is the statistical
            # twin of the shimmer sign bug: a plausible number from the
            # wrong formula.  scipy computes it for us; fall back to the
            # textbook formula only for very old scipy.
            dof = getattr(res, "df", None)
            if dof is None:
                va, vb = np.var(fa, ddof=1), np.var(fb, ddof=1)
                wa, wb = va / len(fa), vb / len(fb)
                dof = (wa + wb) ** 2 / (wa ** 2 / (len(fa) - 1)
                                        + wb ** 2 / (len(fb) - 1))
            out["f0_ttest"] = {
                "t": round(t, 2), "p": float(p),
                "df": round(float(dof), 1), "n_a": len(fa), "n_b": len(fb),
                "significant_5pct": bool(p < 0.05),
                "note": ("Welch t-test on per-frame F0 samples; consecutive "
                         "frames are temporally autocorrelated, so treat p "
                         "as approximate (anti-conservative)"),
            }
    if not rows and "f0_ttest" not in out:
        out["error"] = "no comparable metrics between the two reports"
    return out


# --------------------------------------------------------------------------
# the agent's tool entry point
# --------------------------------------------------------------------------

FORMANT_CEILING_HZ = 5500.0
"""Default formant ceiling (Hz).  Praat resamples to twice this rate before
LPC so that the analysis bandwidth matches the range where formants live;
we follow the same practice."""


def _resample_for_formants(x: np.ndarray, sr: int,
                           ceiling: float) -> tuple[np.ndarray, int]:
    """Downsample to (at most) ``2·ceiling`` Hz for formant analysis.

    Praat's Formant(Burg) does exactly this — without it, high sample rates
    force very high LPC orders and the estimate degrades.  Uses an exact
    rational factor when it is cheap, otherwise plain decimation.
    """
    from math import gcd

    from scipy.signal import resample_poly

    target = 2.0 * ceiling
    if sr <= target:
        return x, sr
    t_int = round(target)
    g = gcd(int(sr), t_int)
    up, down = t_int // g, sr // g
    if up > 64:  # exact factor too costly — integer decimation instead
        # closest divisor-ish rate, never exceeding the target by >2 %
        up = 1
        down = max(1, round(sr / target))
        while sr / down > target * 1.02:
            down += 1
    return resample_poly(x, up, down), sr * up // down


def analyze(audio: AudioData, contour: bool = False,
            formant_ceiling: float = FORMANT_CEILING_HZ,
            backend: str = "native") -> dict:
    """Run a standard acoustic analysis and return a JSON-ready dict.

    ``backend`` selects the measurement engine: ``"native"`` (default) is
    the lightweight numpy/scipy implementation; ``"praat"`` computes the
    same report through praat-parselmouth for publication-grade numbers
    (install with ``pip install -e ".[bench]"``).  Every report carries a
    ``backend`` field so numbers stay traceable to the engine that
    produced them.

    ``formants`` carries not just the median F1–F3 but how trustworthy
    each median is: the number of frames behind it, the interquartile
    spread across frames (``F*_iqr_hz``) and a per-formant
    ``confidence`` in [0, 1] combining coverage (how often the formant
    was found at all) with stability (how little the frame-wise
    estimates scatter around the median).

    With ``contour=True`` a downsampled F0 track (``pitch_contour``)
    and per-frame formant track (``formant_track``) are included for
    plotting.  ``formant_ceiling`` follows Praat's Formant(Burg)
    convention: the signal is resampled to twice the ceiling before LPC.
    """
    if backend == "praat":
        from .praat import analyze_praat  # local import: no hard dep, no cycle
        return analyze_praat(audio, contour=contour,
                             formant_ceiling=formant_ceiling)
    if backend != "native":
        raise ValueError(f"unknown backend {backend!r}: use 'native' or 'praat'")
    sr, x = audio.sample_rate, audio.samples
    # pitch is the most expensive stage — compute once and share it with
    # the jitter/shimmer and HNR estimators
    track = f0_track(x, sr)
    segs = voiced_segments(track)
    if segs:
        # perturbation/HNR norms are sustained-vowel measures: prefer the
        # longest contiguous voiced stretch over the whole (possibly
        # connected-speech) recording — and use the *same* segment for
        # jitter/shimmer and HNR so the three numbers describe one signal
        seg = max(segs, key=lambda s: s["end_s"] - s["start_s"])
        i0, i1 = int(seg["start_s"] * sr), min(int(seg["end_s"] * sr) + 1, len(x))
        sub_x = x[i0:i1]
        m = (track.times >= seg["start_s"]) & (track.times <= seg["end_s"])
        sub_track = F0Track(times=track.times[m], f0=track.f0[m], voiced=track.voiced[m])
        js = jitter_shimmer(sub_x, sr, track=sub_track)
        hnr_db = hnr(sub_x, sr, track=sub_track)
        js_segment = seg
    else:
        js = jitter_shimmer(x, sr, track=track)
        hnr_db = hnr(x, sr, track=track)
        js_segment = None

    # ---- formants: voiced frames only, analysed at 2× the ceiling rate.
    # Voicing matters: the loudest frames of an utterance are often
    # plosive bursts or fricatives, whose LPC roots are not formants.
    formant_summary: dict = {}
    formant_rows_by_frame: dict[int, list[float]] = {}
    x_f, sr_f = _resample_for_formants(x, sr, formant_ceiling)
    fl_f, hl_f = default_frame_lengths(sr_f)
    frames_f = frame_signal(x_f, fl_f, hl_f, window="rect", center=True)
    if len(frames_f):
        voiced_f = track.voiced[: len(frames_f)]
        if np.any(voiced_f):
            energy_f = np.sqrt(np.mean(frames_f ** 2, axis=1) + 1e-12)
            v_idx = np.where(voiced_f)[0]
            # one LPC pass per voiced frame feeds both the median summary
            # and the per-frame track
            formant_rows_by_frame = {int(i): formants(frames_f[i], sr_f)
                                     for i in v_idx}
            # headline medians: most energetic half of the voiced frames
            # (≥5 when available) — keeps the vowel core, drops glide/
            # nasal tails
            sel = v_idx[np.argsort(energy_f[v_idx])]
            sel = sel[-max(5, len(v_idx) // 2):] if len(v_idx) > 5 else sel
            rows = [formant_rows_by_frame[int(i)] for i in sel]
            full_rows = [r for r in rows if len(r) >= 3]
            if full_rows:
                f_stack = np.vstack([r[:3] for r in full_rows])
                med = np.median(f_stack, axis=0)
                q25, q75 = np.percentile(f_stack, [25, 75], axis=0)
                iqr = q75 - q25
                # coverage: how many of the analysed frames yielded this
                # formant at all (missing LPC roots lower confidence)
                coverage = [float(np.mean([len(r) > k for r in rows])) for k in range(3)]
                # stability: relative scatter around the median; a 20 % IQR
                # (e.g. ±10 % around the median) already scores 0
                stability = [float(np.clip(1.0 - iqr[k] / (0.2 * max(med[k], 1.0)), 0.0, 1.0))
                             for k in range(3)]
                formant_summary = {
                    "F1_hz": float(med[0]),
                    "F2_hz": float(med[1]),
                    "F3_hz": float(med[2]),
                    "n_frames": len(f_stack),
                    "F1_iqr_hz": round(float(iqr[0]), 1),
                    "F2_iqr_hz": round(float(iqr[1]), 1),
                    "F3_iqr_hz": round(float(iqr[2]), 1),
                    "confidence": {f"F{k + 1}": round(coverage[k] * stability[k], 2)
                                   for k in range(3)},
                }

    report = {
        "file": audio.path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": sr,
        "n_samples": int(audio.num_samples),
        "backend": "native",
        "pitch": track.summary(),
        "voice_quality": js.summary(),
        "voice_quality_segment_s": (js_segment["start_s"], js_segment["end_s"])
                                   if js_segment else None,
        "hnr_db": round(hnr_db, 2) if len(x) else float("nan"),
        "formants": formant_summary,
        "recording_quality": recording_quality(x, sr),
        "voiced_segments": segs,
        "pause_stats": pause_stats(x, sr),
    }
    if contour:
        step = max(1, len(track.times) // 400)  # keep payloads small
        idx = np.arange(0, len(track.times), step)
        report["pitch_contour"] = {
            "times_s": [round(float(t), 3) for t in track.times[idx]],
            "f0_hz": [round(float(f), 1) if v else None
                      for f, v in zip(track.f0[idx], track.voiced[idx])],
        }
        # per-frame formant track (voiced frames only, None elsewhere),
        # downsampled to ≤ 200 points like the pitch contour
        step_f = max(1, len(frames_f) // 200)
        idx_f = np.arange(0, len(frames_f), step_f)
        times_f = (np.arange(len(frames_f)) * hl_f + fl_f / 2 - fl_f // 2) / sr_f
        cols: list[list] = [[], [], []]
        for i in idx_f:
            row = formant_rows_by_frame.get(int(i))
            for k in range(3):
                cols[k].append(round(row[k], 1)
                               if row is not None and len(row) > k else None)
        report["formant_track"] = {
            "times_s": [round(float(t), 3) for t in times_f[idx_f]],
            "F1_hz": cols[0],
            "F2_hz": cols[1],
            "F3_hz": cols[2],
        }
        report["spectrogram"] = spectrogram(x, sr)
    return report
