#!/usr/bin/env python
"""SpeechLab vs Praat accuracy benchmark.

Compares SpeechLab's acoustic analysis against Praat (via the official
``praat-parselmouth`` binding) on the same files, and reports:

- **F0** — frame-wise MAE (Hz and cents) and Pearson r over frames both
  trackers judge voiced, plus median-F0 agreement;
- **Formants** — MAE and Pearson r for F1/F2/F3 (median per file);
- **Jitter / shimmer / HNR** — Pearson r, mean bias and Bland-Altman
  limits of agreement across files.

SpeechLab side
    Uses exactly the pipeline the agent/web UI ships: ``analyze()``
    (median formants over energetic frames, jitter/shimmer on the longest
    voiced segment, autocorrelation HNR) plus ``f0_track()`` for the
    frame-wise pitch comparison.

Praat side
    ``to_pitch_ac`` (autocorrelation, matching SpeechLab's NCCF family),
    ``to_formant_burg`` (Burg LPC, the Praat default), local jitter/shimmer
    via a PointProcess derived from the same pitch object, and
    ``To Harmonicity (cc)`` for HNR.

Usage
-----
Real data (recommended: CMU Arctic or the Saarbrücken Voice Database)::

    python benchmarks/praat_benchmark.py /path/to/wavs --out bench_results

Self-check without real data (synthetic vowels with jitter/shimmer/noise
and resampling conditions — validates the pipeline, not clinical accuracy)::

    python benchmarks/praat_benchmark.py --synthetic 24 --out bench_results

Outputs: ``results.csv`` (per file), ``summary.json`` + printed markdown
table, and matplotlib figures when available (scatter + Bland-Altman).

Install extras: ``pip install -e ".[bench]"``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechlab.audio import load_audio
from speechlab.features import analyze, f0_track

try:
    import parselmouth
except ImportError:  # pragma: no cover — CLI guard
    parselmouth = None

AUDIO_EXT = {".wav", ".wave", ".flac", ".mp3", ".ogg"}


# ---------------------------------------------------------------------------
# Praat reference measurements
# ---------------------------------------------------------------------------

def praat_reference(path: str, fmin: float, fmax: float,
                    formant_max: float) -> dict:
    """Praat's own numbers for one file (parselmouth bindings)."""
    snd = parselmouth.Sound(path)

    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)
    p_times = pitch.xs()
    p_f0 = pitch.selected_array["frequency"]  # 0 = unvoiced

    formant = snd.to_formant_burg(
        time_step=0.01, max_number_of_formants=5,
        maximum_formant=formant_max, window_length=0.025,
        pre_emphasis_from=50.0)
    f_vals = {1: [], 2: [], 3: []}
    for t in p_times[p_f0 > 0]:  # formants only on voiced frames
        for n in (1, 2, 3):
            v = formant.get_value_at_time(n, t)
            if not math.isnan(v) and v > 0:
                f_vals[n].append(v)

    # PointProcess built from the Sound (not from the Pitch): the Praat
    # manual warns that Pitch -> To PointProcess leaves pulses unaligned
    # with the waveform periods, which corrupts shimmer's amplitude
    # windows.  Shimmer/jitter use Praat's own commands.
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)",
                                fmin, fmax)
    jitter = parselmouth.praat.call(
        pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100.0
    shimmer = parselmouth.praat.call(
        [snd, pp], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
    hv = np.asarray(harm.values).ravel()
    hv = hv[hv > -200]  # Praat's "undefined" sentinel
    hnr = float(np.mean(hv)) if len(hv) else float("nan")

    voiced = p_f0[p_f0 > 0]
    return {
        "times": p_times, "f0": p_f0,
        "f0_median": float(np.median(voiced)) if len(voiced) else float("nan"),
        "f1": float(np.median(f_vals[1])) if f_vals[1] else float("nan"),
        "f2": float(np.median(f_vals[2])) if f_vals[2] else float("nan"),
        "f3": float(np.median(f_vals[3])) if f_vals[3] else float("nan"),
        "jitter": float(jitter), "shimmer": float(shimmer), "hnr": hnr,
    }


# ---------------------------------------------------------------------------
# SpeechLab measurements (the exact shipped pipeline)
# ---------------------------------------------------------------------------

def speechlab_measure(path: str) -> tuple[dict, dict]:
    """(per-file metrics, f0 track) using the same code the agent uses."""
    audio = load_audio(path)
    report = analyze(audio)
    track = f0_track(audio.samples, audio.sample_rate)
    return report, track


def compare_f0(sl_track, praat: dict, max_align_s: float = 0.005) -> dict:
    """Frame-wise F0 agreement on the shared voiced frames."""
    sl_voiced = sl_track.voiced & (sl_track.f0 > 0)
    if not np.any(sl_voiced):
        return {"n_frames": 0}
    times = sl_track.times[sl_voiced]
    f0_sl = sl_track.f0[sl_voiced]
    # nearest Praat frame within ±max_align_s (check both neighbours of the
    # insertion point — the closer side wins)
    pr_times = praat["times"]
    idx_r = np.clip(np.searchsorted(pr_times, times), 0, len(pr_times) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(pr_times) - 1)
    d_r = np.abs(pr_times[idx_r] - times)
    d_l = np.abs(pr_times[idx_l] - times)
    idx = np.where(d_l < d_r, idx_l, idx_r)
    ok = np.minimum(d_l, d_r) <= max_align_s + 1e-9
    f0_pr = praat["f0"][idx]
    ok &= f0_pr > 0  # both trackers voiced
    if np.sum(ok) < 3:
        return {"n_frames": int(np.sum(ok))}
    a, b = f0_sl[ok], f0_pr[ok]
    err = a - b
    cents = 1200.0 * np.log2(a / b)
    r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return {
        "n_frames": int(np.sum(ok)),
        "mae_hz": float(np.mean(np.abs(err))),
        "mae_cents": float(np.mean(np.abs(cents))),
        "median_diff_hz": float(np.median(err)),
        "r": r,
    }


# ---------------------------------------------------------------------------
# aggregation statistics
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _bland_altman(xs: list[float], ys: list[float]) -> dict:
    """Bias and limits of agreement for paired per-file values (x − y)."""
    d = np.asarray(xs, float) - np.asarray(ys, float)
    if len(d) < 3:
        return {"n": len(d)}
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return {"n": len(d), "bias": bias, "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-file rows into the headline benchmark table."""
    def pairs(sl, pr):
        out = [(r[sl], r[pr]) for r in rows
               if r.get(sl) == r.get(sl) and r.get(pr) == r.get(pr)]
        return [p[0] for p in out], [p[1] for p in out]

    f0_rows = [r for r in rows if r.get("f0_mae_hz") == r.get("f0_mae_hz")]
    summary = {
        "n_files": len(rows),
        "f0": {
            "n_files_with_frames": len(f0_rows),
            "total_frames": int(sum(r["n_frames"] for r in f0_rows)),
            "mae_hz": float(np.mean([r["f0_mae_hz"] for r in f0_rows])) if f0_rows else None,
            # robust to single-file octave errors
            "median_mae_hz": float(np.median([r["f0_mae_hz"] for r in f0_rows])) if f0_rows else None,
            "mae_cents": float(np.mean([r["f0_mae_cents"] for r in f0_rows])) if f0_rows else None,
        },
        "formants": {},
        "voice_quality": {},
    }
    sl_f0, pr_f0 = pairs("sl_f0_median", "praat_f0_median")
    summary["f0"]["median_r"] = _pearson(sl_f0, pr_f0)
    summary["f0"]["median_bland_altman"] = _bland_altman(sl_f0, pr_f0)

    for n in (1, 2, 3):
        sl, pr = pairs(f"sl_f{n}", f"praat_f{n}")
        errs = [abs(a - b) for a, b in zip(sl, pr)]
        summary["formants"][f"F{n}"] = {
            "n": len(errs),
            "mae_hz": float(np.mean(errs)) if errs else None,
            "r": _pearson(sl, pr),
            "bland_altman": _bland_altman(sl, pr),
        }
    for metric in ("jitter", "shimmer", "hnr"):
        sl, pr = pairs(f"sl_{metric}", f"praat_{metric}")
        summary["voice_quality"][metric] = {
            "n": len(sl),
            "r": _pearson(sl, pr),
            "bland_altman": _bland_altman(sl, pr),
        }
    return summary


def summary_markdown(s: dict) -> str:
    def f(v, p=2):
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else f"{v:.{p}f}"

    lines = [
        f"# SpeechLab vs Praat benchmark ({s['n_files']} files)",
        "",
        "| Metric | n | MAE | Pearson r | Bias (SL−Praat) | LoA 95% |",
        "|---|---|---|---|---|---|",
    ]
    fz = s["f0"]
    bc = fz["median_bland_altman"]
    lines.append(
        f"| F0 frame MAE | {fz['total_frames']} frames | "
        f"{f(fz['mae_hz'])} Hz (median {f(fz['median_mae_hz'])}) / "
        f"{f(fz['mae_cents'])} cents | — | — | — |")
    lines.append(
        f"| F0 median (per file) | {bc.get('n', 0)} | — | {f(fz['median_r'], 3)} | "
        f"{f(bc.get('bias'))} Hz | [{f(bc.get('loa_low'))}, {f(bc.get('loa_high'))}] |")
    for n in (1, 2, 3):
        m = s["formants"][f"F{n}"]
        bc = m["bland_altman"]
        lines.append(
            f"| F{n} | {m['n']} | {f(m['mae_hz'])} Hz | {f(m['r'], 3)} | "
            f"{f(bc.get('bias'))} | [{f(bc.get('loa_low'))}, {f(bc.get('loa_high'))}] |")
    for metric, unit in (("jitter", "%"), ("shimmer", "dB"), ("hnr", "dB")):
        m = s["voice_quality"][metric]
        bc = m["bland_altman"]
        lines.append(
            f"| {metric} | {m['n']} | — | {f(m['r'], 3)} | "
            f"{f(bc.get('bias'))} {unit} | [{f(bc.get('loa_low'))}, {f(bc.get('loa_high'))}] |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# plots (optional)
# ---------------------------------------------------------------------------

def make_plots(rows: list[dict], summary: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures "
              "(pip install matplotlib)")
        return

    def pairs(sl, pr):
        return zip(*[(r[sl], r[pr]) for r in rows
                     if r.get(sl) == r.get(sl) and r.get(pr) == r.get(pr)]
                   ) or ([], [])

    # scatter: SpeechLab vs Praat for F0 median and F1–F3
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [("sl_f0_median", "praat_f0_median", "F0 median (Hz)")]
    panels += [(f"sl_f{n}", f"praat_f{n}", f"F{n} (Hz)") for n in (1, 2, 3)]
    for ax, (sl, pr, label) in zip(axes, panels):
        xs, ys = pairs(sl, pr)
        ax.scatter(ys, xs, s=18, alpha=0.7)
        lo = min(list(xs) + list(ys)) if xs else 0
        hi = max(list(xs) + list(ys)) if xs else 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"Praat {label}")
        ax.set_ylabel(f"SpeechLab {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter.png"), dpi=130)
    plt.close(fig)

    # Bland-Altman for jitter / shimmer / HNR
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("jitter", "shimmer", "hnr")):
        xs, ys = pairs(f"sl_{metric}", f"praat_{metric}")
        d = [a - b for a, b in zip(xs, ys)]
        m = [(a + b) / 2 for a, b in zip(xs, ys)]
        ax.scatter(m, d, s=18, alpha=0.7)
        bc = summary["voice_quality"][metric]["bland_altman"]
        for y in (bc.get("bias"), bc.get("loa_low"), bc.get("loa_high")):
            if y is not None and not math.isnan(y):
                ax.axhline(y, color="k", ls="--", lw=0.8)
        ax.set_xlabel(f"mean of SpeechLab & Praat {metric}")
        ax.set_ylabel("difference (SL − Praat)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=130)
    plt.close(fig)
    print(f"figures written to {out_dir}/scatter.png, bland_altman.png")


# ---------------------------------------------------------------------------
# synthetic self-check data
# ---------------------------------------------------------------------------

def _resonator_coeffs(freq: float, bw: float, sr: int):
    """Second-order resonator (formant filter) as an IIR section."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    # y[n] = x[n] + 2r·cos(θ)·y[n-1] − r²·y[n-2]
    return np.array([1.0]), np.array([1.0, -2 * r * np.cos(theta), r * r])


def _synthetic_vowel(sr: int, dur: float, f0: float, formants_hz,
                     jitter: float = 0.0, shimmer: float = 0.0,
                     rng=None) -> np.ndarray:
    """Source-filter vowel: glottal pulse train through formant resonators.

    This is the classical synthesis that autocorrelation pitch trackers and
    LPC formant estimators are designed for, so both SpeechLab and Praat
    should track it cleanly — the point is to validate the benchmark
    pipeline itself, not to stress-test the algorithms.
    """
    from scipy.signal import lfilter
    rng = rng or np.random.default_rng()
    n = int(sr * dur)

    # glottal source: impulse train with optional period perturbation
    src = np.zeros(n)
    pos = 0.0
    while pos < n:
        idx = round(pos)
        if idx < n:
            amp = 1.0 + (shimmer * rng.standard_normal() if shimmer else 0.0)
            src[idx] = amp
        period_k = (sr / f0) * (1.0 + jitter * rng.standard_normal()
                                if jitter else 1.0)
        pos += period_k
    # Rosenberg-ish glottal shape: differentiate twice → integrate twice is
    # overkill; a short low-pass on the impulse train suffices for tracking
    src = lfilter([1.0], [1.0, -0.97], src)  # slight spectral tilt

    # cascade formant resonators (source-filter)
    out = src
    for i, ff in enumerate(formants_hz):
        bw = 60.0 + 50.0 * i
        b, a = _resonator_coeffs(ff, bw, sr)
        out = lfilter(b, a, out)
        out /= np.max(np.abs(out)) + 1e-12  # keep each stage in range

    if jitter > 0 or shimmer > 0:  # breathiness adds a noise floor
        out = out + 0.02 * (1 + 10 * jitter) * rng.standard_normal(n)
    return out / (np.max(np.abs(out)) + 1e-12) * 0.8


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def make_synthetic(out_dir: str, n: int, seed: int = 0) -> list[str]:
    """n files spanning conditions: clean / noisy / 8 kHz / perturbed."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        f0 = float(rng.uniform(90, 280))
        fmts = (float(rng.uniform(300, 800)),
                float(rng.uniform(900, 2200)),
                float(rng.uniform(2200, 3000)))
        sr = 16000
        x = _synthetic_vowel(sr, 1.0, f0, fmts,
                             jitter=float(rng.uniform(0, 0.01)),
                             shimmer=float(rng.uniform(0, 0.05)), rng=rng)
        cond = i % 4
        if cond == 1:  # additive noise, ~15 dB SNR
            x = x + 0.15 * rng.standard_normal(len(x))
        elif cond == 2:  # downsampled to 8 kHz (anti-alias then decimate)
            from scipy.signal import resample_poly
            x = resample_poly(x, 1, 2)
            sr = 8000
        elif cond == 3:  # slight low-pass + noise
            from scipy.signal import resample_poly
            x = resample_poly(resample_poly(x, 3, 4), 4, 3)
            x = x + 0.05 * rng.standard_normal(len(x))
        path = os.path.join(out_dir, f"syn_{i:03d}_cond{cond}.wav")
        _write_wav(path, x, sr)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def find_audio(data_dir: str, limit: int | None) -> list[str]:
    files = []
    for root, _, names in os.walk(data_dir):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in AUDIO_EXT:
                files.append(os.path.join(root, name))
    if limit:
        files = files[:limit]
    return files


def run(files: list[str], out_dir: str, fmin: float, fmax: float,
        formant_max: float) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, path in enumerate(files, 1):
        try:
            pr = praat_reference(path, fmin, fmax, formant_max)
            report, track = speechlab_measure(path)
            f0cmp = compare_f0(track, pr)
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the run
            print(f"[{i}/{len(files)}] SKIP {os.path.basename(path)}: {exc}")
            continue
        vq = report.get("voice_quality", {})
        fmt = report.get("formants", {})
        row = {
            "file": os.path.basename(path),
            "sl_f0_median": report["pitch"].get("f0_median_hz", float("nan")),
            "praat_f0_median": pr["f0_median"],
            "f0_mae_hz": f0cmp.get("mae_hz", float("nan")),
            "f0_mae_cents": f0cmp.get("mae_cents", float("nan")),
            "n_frames": f0cmp.get("n_frames", 0),
            "sl_f1": fmt.get("F1_hz", float("nan")),
            "sl_f2": fmt.get("F2_hz", float("nan")),
            "sl_f3": fmt.get("F3_hz", float("nan")),
            "praat_f1": pr["f1"], "praat_f2": pr["f2"], "praat_f3": pr["f3"],
            "sl_jitter": vq.get("jitter_local_percent", float("nan")),
            "praat_jitter": pr["jitter"],
            "sl_shimmer": vq.get("shimmer_local_db", float("nan")),
            "praat_shimmer": pr["shimmer"],
            "sl_hnr": report.get("hnr_db", float("nan")),
            "praat_hnr": pr["hnr"],
        }
        rows.append(row)
        print(f"[{i}/{len(files)}] {row['file']}: "
              f"F0 MAE {row['f0_mae_hz']:.1f} Hz ({row['n_frames']} frames)"
              if row["n_frames"] else
              f"[{i}/{len(files)}] {row['file']}: no shared voiced frames")

    if not rows:
        raise SystemExit("no files could be analysed")

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    md = summary_markdown(summary)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write(md + "\n")
    print()
    print(md)
    print(f"\nper-file results: {csv_path}")
    make_plots(rows, summary, out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", help="directory of audio files")
    ap.add_argument("--out", default="bench_results", help="output directory")
    ap.add_argument("--limit", type=int, help="analyse at most N files")
    ap.add_argument("--fmin", type=float, default=60.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--formant-max", type=float, default=5500.0,
                    help="Praat formant ceiling (5000 male / 5500 female-child)")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="generate N synthetic test files instead of using --data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if parselmouth is None:
        raise SystemExit("praat-parselmouth is required: pip install -e '.[bench]'")

    if args.synthetic:
        syn_dir = os.path.join(args.out, "synthetic")
        print(f"generating {args.synthetic} synthetic files in {syn_dir} ...")
        files = make_synthetic(syn_dir, args.synthetic, args.seed)
    elif args.data:
        files = find_audio(args.data, args.limit)
        if not files:
            raise SystemExit(f"no audio files found under {args.data}")
    else:
        ap.error("provide a data directory or --synthetic N")

    print(f"benchmarking {len(files)} files (SpeechLab vs Praat) ...")
    run(files, args.out, args.fmin, args.fmax, args.formant_max)


if __name__ == "__main__":
    main()
