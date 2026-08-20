"""Recording-quality checks for speech corpora.

Detects the problems that most commonly invalidate speech recordings:
clipping, DC offset, excessive silence, and poor signal-to-noise ratio.
"""

from __future__ import annotations

import numpy as np

from .audio import AudioData, db, frame_signal
from .features import default_frame_lengths

__all__ = ["quality_report"]


def _silence_mask(samples: np.ndarray, sr: int,
                  energy_floor_db: float = -50.0) -> np.ndarray:
    """Energy-based voice-activity mask at the frame level."""
    fl, hl = default_frame_lengths(sr)
    frames = frame_signal(samples, fl, hl, window="rect", center=False)
    if len(frames) == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return db(rms ** 2) > energy_floor_db


def quality_report(audio: AudioData) -> dict:
    """Compute a battery of quality metrics for one recording."""
    sr, x = audio.sample_rate, audio.samples
    n = len(x)

    peak = float(np.max(np.abs(x))) if n else 0.0
    # clipping: samples at/near full scale
    clip_thr = 0.999
    n_clipped = int(np.sum(np.abs(x) >= clip_thr))
    clipping_ratio = n_clipped / n if n else 0.0

    dc_offset = float(np.mean(x)) if n else 0.0

    # silence / speech ratio from frame energies
    voiced = _silence_mask(x, sr)
    if len(voiced):
        speech_ratio = float(np.mean(voiced))
    else:
        speech_ratio = 0.0

    # SNR estimate from the spectral valley floor: for each windowed frame the
    # median power-bin sits near the noise level between harmonic peaks, while
    # the frame's mean power tracks signal+noise.  This works even for
    # continuously voiced signals that contain no silence to measure.
    fl, hl = default_frame_lengths(sr)
    frames = frame_signal(x, fl, hl, window="hann", center=False)
    if len(frames) >= 4:
        from scipy.fft import rfft

        n_fft = 1 << max(1, int(np.ceil(np.log2(fl))))
        spec = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2  # per-bin power
        frame_mean = spec.mean(axis=1)
        valley = np.median(spec, axis=1)
        signal_level = float(np.mean(frame_mean))
        noise_level = float(np.mean(valley))
        snr = 10.0 * np.log10(max(signal_level, 1e-20) / max(noise_level, 1e-20))
        # combine with the silence-based estimate when quiet frames exist
        p = frame_mean
        quiet = float(np.mean(np.sort(p)[: max(1, len(p) // 10)]))
        loud = float(np.mean(np.sort(p)[-max(1, len(p) // 10):]))
        if quiet < 0.1 * loud:  # genuine quiet frames -> silence-based estimate
            snr = max(snr, 10.0 * np.log10(max(loud, 1e-20) / max(quiet, 1e-20)))
    else:
        snr = float("nan")

    issues: list[str] = []
    if clipping_ratio > 1e-4:
        issues.append(f"clipping detected ({n_clipped} samples at full scale)")
    if abs(dc_offset) > 0.01:
        issues.append(f"DC offset {dc_offset:+.4f} exceeds 0.01")
    if not np.isnan(snr) and snr < 15.0:
        issues.append(f"low estimated SNR ({snr:.1f} dB < 15 dB)")
    if speech_ratio > 0 and speech_ratio < 0.25:
        issues.append(f"only {speech_ratio:.0%} of frames contain speech energy")
    if sr < 16000:
        issues.append(f"sample rate {sr} Hz < 16 kHz (limited for fricatives/formants)")
    if n and audio.duration < 0.5:
        issues.append("recording shorter than 0.5 s")

    return {
        "file": audio.path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": sr,
        "peak_amplitude": round(peak, 4),
        "clipping_ratio": clipping_ratio,
        "dc_offset": round(dc_offset, 5),
        "speech_ratio": round(speech_ratio, 3),
        "silence_ratio": round(1.0 - speech_ratio, 3),
        "snr_db": round(snr, 1) if not np.isnan(snr) else None,
        "issues": issues,
        "ok": not issues,
    }
