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
- ``cpps`` : smoothed cepstral peak prominence (dysphonia screening),
- ``spectral_stats`` : LTAS shape measures (centroid, tilt, flatness),
- ``vad_segments`` / ``activity_summary`` : energy-based voice activity
  detection and speech/pause structure,
- ``compare_reports`` : numeric deltas between two analysis reports,
- ``analyze`` : the single entry point the agent calls as a tool.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import AudioData, db, frame_signal

__all__ = [
    "F0Track",
    "JitterShimmer",
    "activity_summary",
    "analyze",
    "compare_reports",
    "cpps",
    "default_frame_lengths",
    "f0_track",
    "formants",
    "hnr",
    "jitter_shimmer",
    "lpc",
    "report_field",
    "spectral_stats",
    "vad_segments",
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


def _batch_autocorr(fx: np.ndarray, lags: np.ndarray | None = None,
                    chunk: int = 2048) -> np.ndarray:
    """Autocorrelation of every row of ``fx``, computed in batches via FFT.

    Returns ``out`` with ``out[i, k] = sum_t fx[i, t] * fx[i, t + k]`` for
    every requested lag ``k`` in ``lags`` (all lags ``0 .. n-1`` when
    ``lags`` is None).  This is O(n log n) per frame instead of the O(n *
    n_lags) direct product, and batches frames so long signals stay fast.
    """
    from scipy.fft import irfft, next_fast_len, rfft

    m, n = fx.shape
    nfft = next_fast_len(2 * n)  # zero-padded -> linear (non-circular) corr
    cols = np.arange(n) if lags is None else np.asarray(lags, dtype=np.intp)
    out = np.empty((m, len(cols)), dtype=np.float64)
    for s in range(0, m, chunk):
        F = rfft(fx[s : s + chunk], n=nfft, axis=1)
        power = F.real ** 2 + F.imag ** 2
        out[s : s + chunk] = irfft(power, n=nfft, axis=1)[:, cols]
    return out


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


def f0_track(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
             frame_length: int | None = None, hop_length: int | None = None,
             voicing_threshold: float = 0.35,
             energy_floor_db: float = -55.0,
             frames: np.ndarray | None = None) -> F0Track:
    """Autocorrelation-based F0 tracker.

    Parameters
    ----------
    fmin, fmax : pitch search range in Hz (60–500 covers adult speech).
    voicing_threshold : minimum normalised autocorrelation for a voiced frame.
    energy_floor_db : frames below this RMS dB are never voiced.
    frames : optional precomputed framing from
        ``frame_signal(samples, frame_length, hop_length, window="rect",
        center=True)``; lets callers that frame the signal anyway reuse it.
    """
    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = int(np.ceil(sr / fmin))
    if lag_max >= frame_length:
        lag_max = frame_length - 1
    if lag_min >= lag_max:
        raise ValueError("pitch search range does not fit the analysis frame")

    if frames is None:
        frames = frame_signal(samples, frame_length, hop_length,
                              window="rect", center=True)
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

    # normalised cross-correlation over the lag range, vectorised over frames.
    # NCCF: r(lag) = sum x(t)x(t+lag) / sqrt(sum x(t)^2 * sum x(t+lag)^2)
    # (over the overlapping region) so that perfectly periodic frames score 1.
    lags = np.arange(lag_min, lag_max + 1)
    n_active = int(np.sum(active))
    if n_active == 0:
        return F0Track(times=times, f0=f0, voiced=voiced)

    act_idx = np.where(active)[0]
    fx = frames[act_idx]
    fx = fx - fx.mean(axis=1, keepdims=True)
    # cumulative energies for O(1) segment-energy lookups
    cum = np.concatenate([np.zeros((n_active, 1)), np.cumsum(fx ** 2, axis=1)], axis=1)

    num = _batch_autocorr(fx, lags)
    e1 = cum[:, frame_length - lags]              # energy of x[:N-lag]
    e2 = cum[:, frame_length, None] - cum[:, lags]  # energy of x[lag:]
    ac = num / np.sqrt(np.maximum(e1 * e2, 1e-20))

    r_max = ac.max(axis=1)
    ok = r_max >= voicing_threshold
    if np.any(ok):
        # Autocorrelation of a periodic signal peaks at every integer
        # multiple of the period.  Choosing the *shortest* lag within a
        # tolerance of the maximum avoids subharmonic (octave-down) errors.
        a = ac[ok]
        thr = 0.85 * r_max[ok]
        j = np.argmax(a >= thr[:, None], axis=1)  # first lag within tolerance
        n_lags = len(lags)
        rows = np.arange(len(a))
        y1 = a[rows, j]
        y0 = a[rows, np.maximum(j - 1, 0)]
        y2 = a[rows, np.minimum(j + 1, n_lags - 1)]
        # parabolic interpolation around the peak
        denom = y0 - 2.0 * y1 + y2
        lag_est = lags[j].astype(np.float64)
        interp = (j > 0) & (j < n_lags - 1) & (np.abs(denom) > 1e-9)
        shift = np.zeros(len(a))
        np.divide(0.5 * (y0 - y2), denom, out=shift, where=interp)
        lag_est += shift
        good = lag_est > 0
        idx = act_idx[ok][good]
        f0[idx] = sr / lag_est[good]
        voiced[idx] = True

    # octave-jump suppression: median filter over voiced frames
    if np.any(voiced):
        from scipy.signal import medfilt

        v_idx = np.where(voiced)[0]
        if len(v_idx) >= 3:
            smoothed = medfilt(f0[v_idx], 3)
            f0[v_idx] = smoothed

    return F0Track(times=times, f0=f0, voiced=voiced)


# --------------------------------------------------------------------------
# LPC / formants
# --------------------------------------------------------------------------

def lpc(samples: np.ndarray, order: int) -> np.ndarray:
    """Linear prediction coefficients (a_1..a_order) via Levinson-Durbin.

    Returns the denominator polynomial without the leading 1.
    """
    from scipy.linalg import solve_toeplitz

    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    if len(x) <= order:
        raise ValueError("frame too short for the requested LPC order")
    # autocorrelation for lags 0..order only: O(N*order) instead of the
    # O(N^2) full np.correlate, which computes every lag and discards most
    r = np.empty(order + 1)
    r[0] = np.dot(x, x)
    for k in range(1, order + 1):
        r[k] = np.dot(x[:-k], x[k:])
    if np.allclose(r[0], 0.0):
        return np.zeros(order)
    a = solve_toeplitz(r[:order], r[1 : order + 1])
    return np.asarray(a, dtype=np.float64)


def _auto_pre_emphasis(sr: int) -> float:
    """Pre-emphasis coefficient giving a -3 dB corner at 50 Hz (Praat-style).

    A fixed 0.97 coefficient is far too aggressive at 16 kHz — it flattens
    the whole F1 region — so the coefficient is derived from the sample rate.
    """
    w = 2.0 * np.pi * 50.0 / sr
    c = float(np.cos(w))
    disc = c * c - 0.5
    if disc <= 0:  # only for absurdly low sample rates
        return 0.0
    return c - np.sqrt(disc)


def _batch_formants(frames: np.ndarray, sr: int, order: int | None = None,
                    pre_emphasis: float | None = None,
                    max_formants: int = 4) -> list[list[float]]:
    """Formant frequencies for a stack of frames; batched :func:`formants`.

    Identical pre-emphasis, LPC order and root filtering as
    :func:`formants`, but the Yule-Walker systems of all frames are solved
    together with a vectorised Levinson-Durbin recursion — the per-frame
    scipy/numpy call overhead otherwise dominates on long recordings.
    """
    x = np.asarray(frames, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a 2-D array of analysis frames")
    m, n = x.shape
    if order is None:
        order = min(16, int(2 + sr / 1000.0))
    if pre_emphasis is None:
        pre_emphasis = _auto_pre_emphasis(sr)
    if n <= order:
        raise ValueError("frame too short for the requested LPC order")

    if pre_emphasis:
        xp = np.empty_like(x)
        xp[:, 0] = x[:, 0]
        xp[:, 1:] = x[:, 1:] - pre_emphasis * x[:, :-1]
        x = xp
    x = x - x.mean(axis=1, keepdims=True)

    # per-frame autocorrelation for lags 0..order
    r = np.empty((m, order + 1))
    r[:, 0] = np.einsum("ij,ij->i", x, x)
    for k in range(1, order + 1):
        r[:, k] = np.einsum("ij,ij->i", x[:, :-k], x[:, k:])

    out: list[list[float]] = [[] for _ in range(m)]
    idx = np.where(np.abs(r[:, 0]) > 1e-8)[0]  # silent frames have no LPC
    if len(idx) == 0:
        return out
    rr = r[idx]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # vectorised Levinson-Durbin over all kept frames
        a = np.zeros((len(idx), order))
        E = rr[:, 0].copy()
        for k in range(1, order + 1):
            acc = np.zeros(len(idx))
            if k > 1:
                acc = np.einsum("ij,ij->i", a[:, : k - 1], rr[:, k - 1 : 0 : -1])
            rc = (rr[:, k] - acc) / E
            a[:, k - 1] = rc
            if k > 1:
                a[:, : k - 1] -= rc[:, None] * a[:, k - 2 :: -1]
            E = E * (1.0 - rc * rc)

    nyq_edge = sr / 2.0 - 50.0
    for row, i in enumerate(idx):
        roots = np.roots(np.concatenate(([1.0], a[row])))
        freqs = np.angle(roots) * sr / (2.0 * np.pi)
        band = ((np.abs(roots.imag) > 1e-10)
                & (freqs > 90.0) & (freqs < nyq_edge))  # skip DC/Nyquist roots
        sel = np.sort(freqs[band])
        out[i] = [float(f) for f in sel[:max_formants]]
    return out


def formants(samples: np.ndarray, sr: int, order: int | None = None,
             pre_emphasis: float | None = None,
             max_formants: int = 4) -> list[float]:
    """Estimate formant frequencies (Hz) for one analysis frame via LPC roots.

    Parameters
    ----------
    order : LPC order; defaults to ``min(16, 2 + sr/1000)``.
    pre_emphasis : coefficient; ``None`` derives one from the sample rate
        (50 Hz corner).  Pass ``0.0`` to disable.
    """
    x = np.asarray(samples, dtype=np.float64)
    return _batch_formants(x[None, :], sr, order=order,
                           pre_emphasis=pre_emphasis,
                           max_formants=max_formants)[0]


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

    Returns ``(epoch_indices, peak_amplitudes)``.
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
    from scipy.signal import oaconvolve

    xs = oaconvolve(x, kernel, mode="same")

    min_dist = max(2, round(0.6 * period))
    threshold = 0.1 * np.max(np.abs(xs))
    # vectorised peak picking: strict local maxima above the threshold...
    inner = xs[1:-1]
    cand = np.where(
        (inner > threshold) & (inner >= xs[:-2]) & (inner > xs[2:])
    )[0] + 1
    # ...then accepted greedily left-to-right with the minimum spacing
    # (candidates are ~one per period, so this loop is tiny)
    epochs_arr = np.empty(len(cand), dtype=np.int64)
    n_kept = 0
    last = -(1 << 62)
    for c in cand:
        if c - last >= min_dist:
            epochs_arr[n_kept] = c
            n_kept += 1
            last = c
    epochs_arr = epochs_arr[:n_kept]

    if len(epochs_arr) >= 2:
        # drop periods outside the plausible F0 range
        periods = np.diff(epochs_arr) / sr
        good = (periods >= 1.0 / fmax) & (periods <= 1.0 / fmin)
        if not np.all(good):
            keep = np.empty(len(epochs_arr), dtype=bool)
            keep[0] = True
            keep[1:] = good
            epochs_arr = epochs_arr[keep]
    return epochs_arr, x[epochs_arr] if len(epochs_arr) else np.empty(0)


def jitter_shimmer(samples: np.ndarray, sr: int, fmin: float = 60.0,
                   fmax: float = 500.0,
                   track: F0Track | None = None) -> JitterShimmer:
    """Local jitter (%) and shimmer (dB) from consecutive glottal periods.

    Typical sustained-vowel values: jitter < 1 %, shimmer < 0.4 dB indicate
    a healthy voice; both rise with vocal pathology.  Pass a precomputed
    ``track`` from :func:`f0_track` to avoid recomputing pitch.
    """
    epochs, peaks = _find_epochs(samples, sr, fmin, fmax, track=track)
    if len(epochs) < 3:
        return JitterShimmer(jitter_local_percent=float("nan"),
                             shimmer_local_db=float("nan"), n_periods=0)

    periods = np.diff(epochs) / sr
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_ratios = peaks[1:] / np.where(np.abs(peaks[:-1]) < 1e-12, np.nan, peaks[:-1])
        shimmer = 20.0 * np.log10(np.abs(amp_ratios))
        shimmer = float(np.nanmean(shimmer)) if np.any(np.isfinite(shimmer)) else float("nan")
    return JitterShimmer(jitter_local_percent=jitter,
                         shimmer_local_db=shimmer, n_periods=len(periods))


def hnr(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
        frame_length: int | None = None, hop_length: int | None = None,
        track: F0Track | None = None,
        frames: np.ndarray | None = None) -> float:
    """Harmonics-to-noise ratio in dB, estimated from autocorrelation.

    Uses the relation HNR ≈ 10·log10(r/(1−r)) at the best F0 lag, averaged
    over voiced frames.  Values above ~20 dB indicate a tonal, stable voice.
    Pass a precomputed ``track`` from :func:`f0_track` (and optionally the
    matching ``frames``) to avoid recomputing pitch and framing.
    """
    if track is None:
        track = f0_track(samples, sr, fmin=fmin, fmax=fmax,
                         frame_length=frame_length, hop_length=hop_length,
                         frames=frames)
    if not np.any(track.voiced):
        return float("nan")

    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    if frames is None:
        frames = frame_signal(samples, frame_length, hop_length,
                              window="rect", center=True)
    vi = np.where(track.voiced)[0]
    vi = vi[vi < len(frames)]
    if len(vi) == 0:
        return float("nan")

    fx = np.asarray(frames[vi], dtype=np.float64)
    fx = fx - fx.mean(axis=1, keepdims=True)
    m = len(fx)
    lag = np.rint(sr / np.maximum(track.f0[vi], 1e-6)).astype(np.intp)
    cum = np.concatenate([np.zeros((m, 1)), np.cumsum(fx ** 2, axis=1)], axis=1)
    ok = (cum[:, frame_length] >= 1e-10) & (lag > 0) & (lag < frame_length)
    if not np.any(ok):
        return float("nan")
    fx, lag = fx[ok], lag[ok]
    num = _batch_autocorr(fx)
    num = num[np.arange(len(fx)), lag]
    # NCCF-style normalisation over the overlapping segments
    e1 = cum[ok, frame_length - lag]
    e2 = cum[ok, frame_length] - cum[ok, lag]
    r_val = num / np.maximum(np.sqrt(e1 * e2), 1e-20)
    r_val = np.clip(r_val, 1e-6, 0.999999)
    vals = 10.0 * np.log10(r_val / (1.0 - r_val))
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# spectral & cepstral voice-quality measures
# --------------------------------------------------------------------------

def cpps(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
         frame_length: int | None = None, hop_length: int | None = None,
         smoothing_frames: int | None = None,
         energy_floor_db: float = -55.0) -> float:
    """Smoothed cepstral peak prominence (CPPS) in dB.

    The cepstrum of a periodic voice shows a peak at the quefrency of the
    pitch period; how far that peak rises above the cepstral trend line
    quantifies harmonic richness independent of F0.  CPPS — the prominence
    smoothed over time — falls with breathiness and is a standard
    dysphonia screening measure.

    Absolute values depend on implementation details and are not directly
    comparable to Praat/VoiceSauce calibrations: interpret them *relative*
    (within this library, e.g. pre- vs post-therapy) rather than against
    external clinical thresholds.

    Implementation: 40 ms / 5 ms Hann frames, real cepstrum of the dB
    magnitude spectrum, peak searched over the 60–500 Hz quefrency band,
    prominence measured against a per-frame least-squares trend line,
    smoothed over ~60 ms and averaged over energetic frames only.
    """
    from scipy.fft import irfft, rfft

    x = np.asarray(samples, dtype=np.float64)
    if frame_length is None:
        frame_length = max(1, round(sr * 0.040))
    if hop_length is None:
        hop_length = max(1, round(sr * 0.005))
    if smoothing_frames is None:
        smoothing_frames = max(1, round(0.060 * sr / hop_length))

    q_min = max(1, int(np.floor(sr / fmax)))          # quefrency of 1/fmax
    q_max = min(frame_length // 2, int(np.ceil(sr / fmin)))
    if q_min >= q_max:
        raise ValueError("pitch search range does not fit the analysis frame")

    frames = frame_signal(x, frame_length, hop_length, window="hann", center=True)
    if len(frames) == 0:
        return float("nan")
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    active = db(rms ** 2) > energy_floor_db
    if not np.any(active):
        return float("nan")

    # real cepstrum: IFFT of the log magnitude spectrum (in dB)
    spec = np.abs(rfft(frames, axis=1))
    cep = irfft(20.0 * np.log10(np.maximum(spec, 1e-10)), n=frame_length, axis=1)

    # per-frame linear trend over quefrency 0..q_max
    q = np.arange(q_max + 1, dtype=np.float64)
    qc = q - q.mean()
    c = cep[:, : q_max + 1]
    c_mean = c.mean(axis=1)
    slope = ((c - c_mean[:, None]) @ qc) / np.dot(qc, qc)
    intercept = c_mean - slope * q.mean()

    # prominence of the cepstral peak in the pitch quefrency band
    band = c[:, q_min : q_max + 1]
    j = np.argmax(band, axis=1)
    rows = np.arange(len(c))
    peak = band[rows, j]
    trend = intercept + slope * (q_min + j)
    prom = peak - trend

    # smooth over time, then average over energetic frames only
    w = np.ones(smoothing_frames) / smoothing_frames
    prom_s = np.convolve(prom, w, mode="same")
    return float(np.mean(prom_s[active]))


def spectral_stats(samples: np.ndarray, sr: int,
                   frame_length: int | None = None, hop_length: int | None = None,
                   energy_floor_db: float = -55.0) -> dict:
    """Long-term average spectrum (LTAS) shape measures.

    Returns ``{"spectral_centroid_hz", "spectral_tilt_db_per_khz",
    "spectral_flatness"}`` computed from the mean power spectrum of
    energetic Hann frames:

    * centroid — the energy-weighted mean frequency (rises with
      brightness/breathiness);
    * tilt — slope of the dB LTAS against frequency in dB/kHz (more
      negative = more low-frequency-dominated, e.g. a sonorant vowel);
    * flatness — geometric/arithmetic mean ratio of the power spectrum
      (→ 1 for white noise, ≪ 1 for a harmonic signal).
    """
    from scipy.fft import rfft, rfftfreq

    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    frames = frame_signal(samples, frame_length, hop_length, window="hann", center=True)
    if len(frames) == 0:
        return {}
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    active = db(rms ** 2) > energy_floor_db
    if not np.any(active):
        return {}

    spec = np.abs(rfft(frames[active], axis=1))
    power = spec.mean(axis=0) ** 2
    freqs = rfftfreq(frame_length, 1.0 / sr)

    total = float(np.sum(power))
    centroid = float(np.dot(freqs, power) / total) if total > 0 else 0.0

    ltas_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    f_khz = freqs / 1000.0
    fc = f_khz - f_khz.mean()
    tilt = float(np.dot(fc, ltas_db - ltas_db.mean()) / np.dot(fc, fc))

    logp = np.log(np.maximum(power, 1e-20))
    flatness = float(np.exp(logp.mean()) / max(float(power.mean()), 1e-20))

    return {
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_tilt_db_per_khz": round(tilt, 2),
        "spectral_flatness": round(flatness, 4),
    }


# --------------------------------------------------------------------------
# voice activity / pause structure
# --------------------------------------------------------------------------

def vad_segments(samples: np.ndarray, sr: int,
                 frame_length: int | None = None, hop_length: int | None = None,
                 energy_floor_db: float = -55.0,
                 min_speech_s: float = 0.02, min_pause_s: float = 0.06,
                 frames: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Energy-based voice activity detection.

    Returns speech intervals as ``[(start_s, end_s), ...]``.  Frames above
    ``energy_floor_db`` RMS are active; inactive gaps shorter than
    ``min_pause_s`` are bridged and segments shorter than ``min_speech_s``
    are dropped, so plosive closures and measurement jitter do not
    fragment the speech.

    ``frames`` may pass a precomputed rectangular framing from
    ``frame_signal(samples, frame_length, hop_length, window="rect",
    center=True)``.
    """
    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    if frames is None:
        frames = frame_signal(samples, frame_length, hop_length,
                              window="rect", center=True)
    duration = len(np.asarray(samples)) / float(sr)
    if len(frames) == 0 or duration <= 0:
        return []

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    active = db(rms ** 2) > energy_floor_db
    idx = np.where(active)[0]
    if len(idx) == 0:
        return []

    # runs of active frames, bridging gaps shorter than min_pause_s
    min_gap = max(1, round(min_pause_s * sr / hop_length))
    min_seg = max(1, round(min_speech_s * sr / hop_length))
    runs: list[tuple[int, int]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= min_gap:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))

    pad = frame_length // 2
    out: list[tuple[float, float]] = []
    for a, b in runs:
        if b - a + 1 < min_seg:
            continue
        t0 = max(0.0, (a * hop_length - pad) / sr)
        t1 = min(duration, (b * hop_length - pad + frame_length) / sr)
        if t1 > t0:
            out.append((round(float(t0), 4), round(float(t1), 4)))
    return out


def activity_summary(samples: np.ndarray, sr: int, **kwargs) -> dict:
    """Speech/pause structure derived from :func:`vad_segments`.

    Useful for fluency description and corpus quality checks (leading/
    trailing silences, pause count and length, speaking-time ratio).
    """
    segments = vad_segments(samples, sr, **kwargs)
    duration = len(np.asarray(samples)) / float(sr)
    speech = float(sum(e - s for s, e in segments))
    pauses = [s2 - e1 for (_, e1), (s2, _) in itertools.pairwise(segments)]
    return {
        "duration_s": round(duration, 3),
        "n_speech_segments": len(segments),
        "n_pauses": len(pauses),
        "speech_time_s": round(speech, 3),
        "pause_time_s": round(duration - speech, 3),
        "speech_ratio": round(speech / duration, 3) if duration > 0 else 0.0,
        "mean_pause_s": round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "max_pause_s": round(float(np.max(pauses)), 3) if pauses else 0.0,
    }


# --------------------------------------------------------------------------
# report comparison
# --------------------------------------------------------------------------

#: report fields (dotted paths) included in side-by-side comparisons
COMPARE_FIELDS = [
    "duration_s",
    "pitch.f0_mean_hz",
    "pitch.f0_median_hz",
    "pitch.f0_std_hz",
    "pitch.voiced_ratio",
    "voice_quality.jitter_local_percent",
    "voice_quality.shimmer_local_db",
    "hnr_db",
    "cpps_db",
    "formants.F1_hz",
    "formants.F2_hz",
    "formants.F3_hz",
]


def report_field(report: dict, dotted: str) -> float | None:
    """Fetch ``"a.b.c"`` from a nested report; ``None`` if absent/non-numeric."""
    cur: Any = report
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur)


def compare_reports(report_a: dict, report_b: dict) -> dict:
    """Numeric deltas (``b - a``) of the key measures of two reports.

    Fields missing from either report, or non-finite (e.g. NaN jitter on a
    too-short signal), are skipped.
    """
    deltas: dict[str, float] = {}
    for field in COMPARE_FIELDS:
        va = report_field(report_a, field)
        vb = report_field(report_b, field)
        if va is None or vb is None:
            continue
        if math.isfinite(va) and math.isfinite(vb):
            deltas[field] = round(vb - va, 4)
    return deltas


# --------------------------------------------------------------------------
# the agent's tool entry point
# --------------------------------------------------------------------------

def analyze(audio: AudioData) -> dict:
    """Run a standard acoustic analysis and return a JSON-ready dict."""
    sr, x = audio.sample_rate, audio.samples
    fl, hl = default_frame_lengths(sr)
    # one shared rectangular framing (a zero-copy view) for pitch, formants
    # and HNR; pitch is the most expensive stage, so compute the track once
    # and share it with the jitter/shimmer and HNR estimators too
    frames = frame_signal(x, fl, hl, window="rect", center=True)
    track = f0_track(x, sr, frames=frames)
    js = jitter_shimmer(x, sr, track=track)
    # median formants over the most energetic voiced frames
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12) if len(frames) else np.empty(0)
    top = np.argsort(energy)[-max(5, len(energy) // 10):] if len(energy) else []
    formant_rows = _batch_formants(frames[top], sr)
    formant_rows = [r for r in formant_rows if len(r) >= 3]
    if formant_rows:
        f_stack = np.vstack([r[:3] for r in formant_rows])
        formant_summary = {
            "F1_hz": float(np.median(f_stack[:, 0])),
            "F2_hz": float(np.median(f_stack[:, 1])),
            "F3_hz": float(np.median(f_stack[:, 2])),
        }
    else:
        formant_summary = {}

    return {
        "file": audio.path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": sr,
        "n_samples": int(audio.num_samples),
        "pitch": track.summary(),
        "voice_quality": js.summary(),
        "hnr_db": round(hnr(x, sr, track=track, frames=frames), 2) if len(x) else float("nan"),
        "cpps_db": round(cpps(x, sr), 2) if len(x) else float("nan"),
        "spectral": spectral_stats(x, sr),
        "activity": activity_summary(x, sr, frames=frames),
        "formants": formant_summary,
    }
