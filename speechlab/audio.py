"""Audio loading and basic utilities.

Reads WAV files with the standard library / scipy, and any format supported
by ``soundfile`` (flac, mp3, ogg, ...) when that optional package is
installed.  All loaders return float64 samples in [-1, 1] and convert
multi-channel audio to mono by averaging channels.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "AudioData",
    "db",
    "frame_signal",
    "load_audio",
    "resample",
]


@dataclass
class AudioData:
    """Container for a mono, floating-point signal.

    Attributes
    ----------
    samples : np.ndarray
        1-D float64 array in [-1, 1].
    sample_rate : int
        Sampling rate in Hz.
    path : str or None
        Original file path when loaded from disk.
    """

    samples: np.ndarray
    sample_rate: int
    path: str | None = None

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / float(self.sample_rate)

    @property
    def num_samples(self) -> int:
        return len(self.samples)


def _wav_to_float(raw: bytes, sampwidth: int, dtype_code: str) -> np.ndarray:
    """Convert raw WAV bytes to float64 in [-1, 1]."""
    if sampwidth == 1:
        # WAV 8-bit PCM is UNSIGNED with midpoint 128
        vals = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (vals - 128.0) / 128.0
    if dtype_code == "f":  # 32/64-bit float PCM already in native range
        return np.frombuffer(raw, dtype="<f4" if sampwidth == 4 else "<f8").astype(np.float64)
    fmt = {2: "h", 3: "i", 4: "i"}[sampwidth]
    n_bytes = {"h": 2, "i": 4}[fmt]
    vals = np.frombuffer(raw, dtype=np.dtype(fmt).newbyteorder("<"))
    peak = float(2 ** (8 * n_bytes - 1))
    return vals.astype(np.float64) / peak


def _pcm24_to_float(raw: bytes) -> np.ndarray:
    """Vectorised little-endian signed 24-bit PCM → float64 in [-1, 1]."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    val = np.where(val >= 0x800000, val - 0x1000000, val)
    return val.astype(np.float64) / 8388608.0


def load_audio(path: str, target_sr: int | None = None) -> AudioData:
    """Load an audio file as mono float64.

    Parameters
    ----------
    path : str
        Path to an audio file.
    target_sr : int, optional
        If given, resample the signal to this rate.

    Returns
    -------
    AudioData
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")

    ext = os.path.splitext(path)[1].lower()
    samples: np.ndarray | None = None
    sample_rate: int | None = None

    # Try soundfile first for non-wav formats (and wav too, it handles edge cases).
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, dtype="float64", always_2d=True)
        samples = data.mean(axis=1)
        sample_rate = int(sr)
    except ImportError:
        if ext not in (".wav", ".wave"):
            raise ValueError(
                f"cannot read '{ext}' files: install the optional 'soundfile' package"
            )
    except Exception:
        # Fall through to the stdlib wave reader for .wav files.
        if ext not in (".wav", ".wave"):
            raise

    if samples is None:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            if wf.getcomptype() != "NONE":
                raise ValueError("compressed WAV is not supported")
        if sampwidth == 3:  # 24-bit PCM needs manual unpacking
            samples = _pcm24_to_float(raw)
        else:
            samples = _wav_to_float(raw, sampwidth, wf_dtype(sampwidth))
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

    samples = np.asarray(samples, dtype=np.float64).ravel()
    if target_sr is not None and target_sr != sample_rate:
        samples = resample(samples, sample_rate, target_sr)
        sample_rate = int(target_sr)

    return AudioData(samples=samples, sample_rate=int(sample_rate), path=path)


def wf_dtype(sampwidth: int) -> str:
    """Map WAV sample width to a float/pcm code used by :func:`_wav_to_float`."""
    if sampwidth == 4:
        # ambiguous: could be 32-bit int PCM or 32-bit float.  The stdlib
        # wave module does not distinguish; assume int PCM (most common for
        # research corpora) — 32-bit float WAVs are rare.
        return "i"
    return "i"


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling of a 1-D signal."""
    if orig_sr == target_sr:
        return np.asarray(samples, dtype=np.float64)
    from math import gcd

    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return resample_poly(np.asarray(samples, dtype=np.float64), up, down)


def db(x: float | np.ndarray, floor: float = -120.0) -> float | np.ndarray:
    """Convert a power/amplitude ratio to decibels with a floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(x, 1e-12))
    if np.isscalar(out):
        return float(max(out, floor))
    return np.maximum(out, floor)


def frame_signal(
    samples: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Slice a signal into overlapping frames.

    Returns an array of shape ``(n_frames, frame_length)``.
    """
    x = np.asarray(samples, dtype=np.float64)
    if center:  # pad so that frames are centred on sample indices
        pad = frame_length // 2
        mode = "reflect" if pad < len(x) else "constant"
        x = np.pad(x, (pad, pad), mode=mode)

    if len(x) < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)

    n_frames = 1 + (len(x) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = x[idx]

    if window == "hann":
        w = np.hanning(frame_length)
    elif window == "hamming":
        w = np.hamming(frame_length)
    elif window in (None, "rect", "rectangular"):
        w = np.ones(frame_length)
    else:
        raise ValueError(f"unknown window: {window}")
    return frames * w
