"""Shared helpers for synthesising test signals (no external fixtures needed)."""

import wave

import numpy as np


def tone(freq_hz: float, duration_s: float, sr: int = 16000,
         amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    """A pure sine tone."""
    t = np.arange(round(duration_s * sr)) / sr
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


def vowel_like(f0_hz: float = 120.0, duration_s: float = 1.0, sr: int = 16000,
               formants=((730, 1.0), (1090, 0.5), (2440, 0.25))) -> np.ndarray:
    """A buzzy 'vowel': sawtooth glottal source through resonant filters.

    Crude but produces a periodic, harmonically rich signal with stable F0
    and energy — enough to sanity-check pitch, jitter and HNR estimators.
    """
    from scipy.signal import lfilter

    n = round(duration_s * sr)
    t = np.arange(n) / sr
    # glottal-ish pulse train: softly clipped sawtooth
    source = 2.0 * ((t * f0_hz) % 1.0) - 1.0
    source = np.tanh(2.5 * source)

    out = np.zeros(n)
    for f_hz, bw in formants:
        # 2-pole resonator (simple biquad approximation via lfilter)
        r = np.exp(-np.pi * 80.0 / sr)  # ~80 Hz bandwidth
        theta = 2 * np.pi * f_hz / sr
        a = np.array([1.0, -2 * r * np.cos(theta), r * r])
        b = np.array([(1 - r) ** 2])
        out += bw * lfilter(b, a, source)
    out /= np.max(np.abs(out)) + 1e-12
    return 0.8 * out


def noise(duration_s: float, sr: int = 16000, amp: float = 0.1,
          seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(round(duration_s * sr))


def write_wav(path: str, samples: np.ndarray, sr: int) -> str:
    """Write float samples to a 16-bit PCM WAV file."""
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0 - 1e-9)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path
