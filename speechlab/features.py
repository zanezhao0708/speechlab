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
    "default_frame_lengths",
    "f0_track",
    "formants",
    "hnr",
    "jitter_shimmer",
    "lpc",
    "recording_quality",
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


def f0_track(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
             frame_length: int | None = None, hop_length: int | None = None,
             voicing_threshold: float = 0.35,
             energy_floor_db: float = -55.0) -> F0Track:
    """Autocorrelation-based F0 tracker.

    Parameters
    ----------
    fmin, fmax : pitch search range in Hz (60–500 covers adult speech).
    voicing_threshold : minimum normalised autocorrelation for a voiced frame.
    energy_floor_db : frames below this RMS dB are never voiced.
    """
    frame_length, hop_length = _defaults(sr, frame_length, hop_length)
    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = int(np.ceil(sr / fmin))
    if lag_max >= frame_length:
        lag_max = frame_length - 1
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

    # normalised cross-correlation over the lag range, vectorised over frames.
    # NCCF: r(lag) = sum x(t)x(t+lag) / sqrt(sum x(t)^2 * sum x(t+lag)^2)
    # (over the overlapping region) so that perfectly periodic frames score 1.
    lags = np.arange(lag_min, lag_max + 1)
    n_active = int(np.sum(active))
    if n_active == 0:
        return F0Track(times=times, f0=f0, voiced=voiced)

    act_idx = np.where(active)[0]
    fx = frames[act_idx] - frames[act_idx].mean(axis=1, keepdims=True)
    # cumulative energies for O(1) segment-energy lookups
    cum = np.concatenate([np.zeros((n_active, 1)), np.cumsum(fx ** 2, axis=1)], axis=1)

    ac = np.empty((n_active, len(lags)), dtype=np.float64)
    for j, lag in enumerate(lags):
        num = np.einsum("ij,ij->i", fx[:, : frame_length - lag], fx[:, lag:])
        e1 = cum[:, frame_length - lag]                       # energy of x[:N-lag]
        e2 = cum[:, frame_length] - cum[:, lag]                # energy of x[lag:]
        ac[:, j] = num / np.sqrt(np.maximum(e1 * e2, 1e-20))

    for i, fi in enumerate(act_idx):
        a = ac[i]
        r_max = float(a.max())
        if r_max < voicing_threshold:
            continue
        # Autocorrelation of a periodic signal peaks at every integer
        # multiple of the period.  Choosing the *shortest* lag within a
        # tolerance of the maximum avoids subharmonic (octave-down) errors.
        thr = 0.85 * r_max
        j = int(np.argmax(a >= thr))
        # parabolic interpolation around the peak
        lag_est = float(lags[j])
        if 0 < j < len(a) - 1:
            y0, y1, y2 = a[j - 1], a[j], a[j + 1]
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
    r = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) - 1 + order + 1]
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
    if order is None:
        order = min(16, int(2 + sr / 1000.0))
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
    xs = np.convolve(x, kernel, mode="same")

    min_dist = max(2, round(0.6 * period))
    threshold = 0.1 * np.max(np.abs(xs))
    epochs: list[int] = []
    i = 1
    n = len(xs)
    while i < n - 1:
        is_peak = xs[i] > threshold and xs[i] >= xs[i - 1] and xs[i] > xs[i + 1]
        if is_peak and (not epochs or i - epochs[-1] >= min_dist):
            epochs.append(i)
            i += min_dist
            continue
        i += 1

    epochs_arr = np.asarray(epochs, dtype=int)
    if len(epochs_arr) >= 2:
        # drop periods outside the plausible F0 range
        periods = np.diff(epochs_arr) / sr
        good = (periods >= 1.0 / fmax) & (periods <= 1.0 / fmin)
        if not np.all(good):
            keep = [epochs_arr[0]]
            for k in range(1, len(epochs_arr)):
                if good[k - 1]:
                    keep.append(epochs_arr[k])
            epochs_arr = np.asarray(keep, dtype=int)
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
        r_val = min(max(r_val, 1e-6), 0.999999)
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

    # clipping: samples pinned at (or within 0.1 % of) full scale
    clip_ratio = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0

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
        if peak >= 0.999:
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
# the agent's tool entry point
# --------------------------------------------------------------------------

def analyze(audio: AudioData, contour: bool = False) -> dict:
    """Run a standard acoustic analysis and return a JSON-ready dict.

    With ``contour=True`` a downsampled F0 track (``pitch_contour``) is
    included for plotting.
    """
    sr, x = audio.sample_rate, audio.samples
    # pitch is the most expensive stage — compute once and share it with
    # the jitter/shimmer and HNR estimators
    track = f0_track(x, sr)
    js = jitter_shimmer(x, sr, track=track)
    fl, hl = default_frame_lengths(sr)
    # median formants over the most energetic voiced frames
    frames = frame_signal(x, fl, hl, window="rect", center=True)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12) if len(frames) else np.empty(0)
    top = np.argsort(energy)[-max(5, len(energy) // 10):] if len(energy) else []
    formant_rows = [formants(frames[i], sr) for i in top]
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

    report = {
        "file": audio.path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": sr,
        "n_samples": int(audio.num_samples),
        "pitch": track.summary(),
        "voice_quality": js.summary(),
        "hnr_db": round(hnr(x, sr, track=track), 2) if len(x) else float("nan"),
        "formants": formant_summary,
        "recording_quality": recording_quality(x, sr),
        "voiced_segments": voiced_segments(track),
    }
    if contour:
        step = max(1, len(track.times) // 400)  # keep payloads small
        idx = np.arange(0, len(track.times), step)
        report["pitch_contour"] = {
            "times_s": [round(float(t), 3) for t in track.times[idx]],
            "f0_hz": [round(float(f), 1) if v else None
                      for f, v in zip(track.f0[idx], track.voiced[idx])],
        }
    return report
