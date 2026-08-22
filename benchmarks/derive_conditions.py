#!/usr/bin/env python
"""Derive the benchmark condition sets from downloaded CMU Arctic files.

Builds the 100-file real-data benchmark directory from the raw 16 kHz
utterances:

* ``main/``   — exactly 100 files for the headline run:
    - 80 clean originals (4 speakers x a0001-a0020, 16 kHz)
    - 5 resampled to 44.1 kHz + 5 to 22.05 kHz (sample-rate robustness)
    - 10 noise-added at SNR {0, 5, 10, 15, 20} dB x 2 speakers (16 kHz)
* ``telephony/`` — 10 files resampled to 8 kHz for a separate run with
  ``--formant-max 3800`` (telephony analysis convention).

Gaussian noise is seeded, so the whole benchmark directory is
reproducible from ``bench_data/raw``.

Usage::

    python benchmarks/derive_conditions.py --raw bench_data/raw \
        --main bench_data/main --telephony bench_data/telephony
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SEED = 20260822


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float64) / 32768.0, sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    y = np.clip(x, -1.0, 1.0 - 1e-9)
    pcm = (y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_rms = float(np.sqrt(np.mean(x ** 2)))
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    n = rng.standard_normal(len(x)) * noise_rms
    return x + n


def resample_to(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    from math import gcd
    g = gcd(target, sr)
    return resample_poly(x, target // g, sr // g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="bench_data/raw")
    ap.add_argument("--main", default="bench_data/main")
    ap.add_argument("--telephony", default="bench_data/telephony")
    args = ap.parse_args()

    raw, main_d, tel_d = (Path(p) for p in (args.raw, args.main, args.telephony))
    main_d.mkdir(parents=True, exist_ok=True)
    tel_d.mkdir(parents=True, exist_ok=True)

    def raw_name(spk: str, n: int) -> Path:
        return raw / f"{spk}_arctic_a{n:04d}.wav"

    # ---- 80 clean originals ------------------------------------------------
    for spk in ("bdl", "rms", "slt", "clb"):
        for n in range(1, 21):
            src, dst = raw_name(spk, n), main_d / f"{spk}_a{n:04d}.wav"
            write_wav(dst, *read_wav(src))

    # ---- 10 sample-rate variants -------------------------------------------
    # (speaker, utterance, target rate); both genders at both rates
    resample_jobs = [
        ("bdl", 21, 44100), ("bdl", 22, 44100),
        ("slt", 21, 44100), ("slt", 22, 44100),
        ("rms", 21, 44100),
        ("clb", 21, 22050), ("clb", 22, 22050),
        ("slt", 23, 22050), ("bdl", 23, 22050),
        ("rms", 22, 22050),
    ]
    for spk, n, target in resample_jobs:
        x, sr = read_wav(raw_name(spk, n))
        tag = {44100: "sr44k", 22050: "sr22k"}[target]
        write_wav(main_d / f"{spk}_a{n:04d}_{tag}.wav",
                  resample_to(x, sr, target), target)

    # ---- 10 noise variants ---------------------------------------------------
    # SNR {20, 15, 10, 5, 0} dB, each level once male once female
    noise_jobs = [
        ("bdl", 24, 20.0), ("slt", 24, 20.0),
        ("rms", 24, 15.0), ("clb", 24, 15.0),
        ("bdl", 25, 10.0), ("slt", 25, 10.0),
        ("rms", 25, 5.0), ("clb", 25, 5.0),
        ("rms", 23, 0.0), ("clb", 23, 0.0),
    ]
    rng = np.random.default_rng(SEED)
    for spk, n, snr in noise_jobs:
        x, sr = read_wav(raw_name(spk, n))
        write_wav(main_d / f"{spk}_a{n:04d}_snr{int(snr)}db.wav",
                  add_noise(x, snr, rng), sr)

    # ---- 10 telephony files (8 kHz, separate run at ceiling 3800) ----------
    for spk, n in [("bdl", i) for i in range(1, 6)] + \
                  [("slt", i) for i in range(1, 6)]:
        x, sr = read_wav(raw_name(spk, n))
        write_wav(tel_d / f"{spk}_a{n:04d}_sr8k.wav",
                  resample_to(x, sr, 8000), 8000)

    n_main = len(list(main_d.glob("*.wav")))
    n_tel = len(list(tel_d.glob("*.wav")))
    print(f"main: {n_main} files (expect 100) -> {main_d}")
    print(f"telephony: {n_tel} files (expect 10) -> {tel_d}")
    assert n_main == 100, "main set must contain exactly 100 files"
    assert n_tel == 10, "telephony set must contain exactly 10 files"


if __name__ == "__main__":
    main()
