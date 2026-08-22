#!/usr/bin/env python
"""Per-condition breakdown of a praat_benchmark results.csv.

Splits the real-data run into its experimental conditions (clean /
resampled / noise level / speaker) and reports agreement metrics per
group, so a regression can be attributed to a condition instead of
drowning in the aggregate.

Usage::

    python benchmarks/analyze_conditions.py bench_results/real100/results.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict

import numpy as np


def pearson(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stats(a: list[float], b: list[float]) -> dict:
    """MAE / r / bias / LoA between two per-file series."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) == 0:
        return {"n": 0}
    d = a - b
    return {
        "n": len(a),
        "mae": float(np.mean(np.abs(d))),
        "r": pearson(list(a), list(b)),
        "bias": float(np.mean(d)),
        "loa": float(1.96 * np.std(d, ddof=1)) if len(a) > 1 else float("nan"),
    }


def condition(name: str) -> str:
    if "sr44k" in name:
        return "resampled 44.1 kHz"
    if "sr22k" in name:
        return "resampled 22.05 kHz"
    m = re.search(r"snr(\d+)db", name)
    if m:
        return f"noise {m.group(1)} dB SNR"
    return "clean 16 kHz"


METRICS = [
    ("F0 median (Hz)", "sl_f0_median", "praat_f0_median"),
    ("F1 (Hz)", "sl_f1", "praat_f1"),
    ("F2 (Hz)", "sl_f2", "praat_f2"),
    ("F3 (Hz)", "sl_f3", "praat_f3"),
    ("jitter (%)", "sl_jitter", "praat_jitter"),
    ("shimmer (dB)", "sl_shimmer", "praat_shimmer"),
    ("HNR (dB)", "sl_hnr", "praat_hnr"),
]


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[condition(r["file"])].append(r)
        groups[f"speaker {r['file'][:3]}"].append(r)

    order = ["clean 16 kHz", "resampled 44.1 kHz", "resampled 22.05 kHz",
             "noise 20 dB SNR", "noise 15 dB SNR", "noise 10 dB SNR",
             "noise 5 dB SNR", "noise 0 dB SNR",
             "speaker bdl", "speaker rms", "speaker slt", "speaker clb"]
    print(f"{'group':<22}" + "".join(
        f"{label:>34}" for label, _, _ in METRICS))
    for g in order:
        if g not in groups:
            continue
        rs = groups[g]
        cells = []
        for _, sl, pr in METRICS:
            s = stats([float(r[sl]) for r in rs], [float(r[pr]) for r in rs])
            if s["n"] == 0:
                cells.append("n=0".rjust(34))
            else:
                cells.append(
                    f"n={s['n']:3d} MAE={s['mae']:7.2f} "
                    f"r={s['r']:5.2f} b={s['bias']:+7.2f}")
        print(f"{g:<22}" + "".join(cells))

    # overall
    cells = []
    for _, sl, pr in METRICS:
        s = stats([float(r[sl]) for r in rows], [float(r[pr]) for r in rows])
        cells.append(f"n={s['n']:3d} MAE={s['mae']:7.2f} "
                     f"r={s['r']:5.2f} b={s['bias']:+7.2f}")
    print(f"{'ALL':<22}" + "".join(cells))


if __name__ == "__main__":
    main()
