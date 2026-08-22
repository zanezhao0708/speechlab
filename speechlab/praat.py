"""Publication-grade backend: Praat via ``praat-parselmouth``.

The native backend (:mod:`speechlab.features`) is lightweight numpy/scipy;
this module produces the *same report schema* using Praat's own algorithms —
``To Pitch (ac)``, ``To Formant (burg)``, ``To PointProcess (periodic, cc)``
and ``To Harmonicity (cc)`` — for users who need numbers they can publish
next to Praat-derived values from other labs.

Select it with ``analyze(audio, backend="praat")``; the optional dependency
is installed via ``pip install -e ".[bench]"``.

Methodological note
-------------------
Jitter/shimmer/HNR are computed on the longest voiced segment (Praat's own
docs define these as sustained-vowel measures), matching the native
backend's semantics exactly.  Signal-level utilities that Praat does not
define (recording quality, pause statistics, spectrogram) are reused from
the native implementation so both backends return an identical schema.
"""

from __future__ import annotations

import numpy as np

from .audio import AudioData
from .features import (
    FORMANT_CEILING_HZ,
    F0Track,
    pause_stats,
    recording_quality,
    spectrogram,
    voiced_segments,
)

__all__ = ["analyze_praat", "praat_available"]

#: Praat settings shared with the native backend's defaults
_FMIN, _FMAX = 60.0, 500.0


def praat_available() -> bool:
    """True when ``praat-parselmouth`` is importable."""
    try:
        import parselmouth  # noqa: F401
        return True
    except ImportError:
        return False


def _require():
    try:
        import parselmouth
        return parselmouth
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "backend='praat' needs praat-parselmouth: "
            "pip install -e '.[bench]'"
        ) from exc


def _praat_pitch_track(parselmouth, pitch) -> F0Track:
    """Convert a Praat Pitch object to an :class:`F0Track`."""
    times = np.asarray(pitch.xs(), dtype=np.float64)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    return F0Track(times=times, f0=f0, voiced=f0 > 0)


def analyze_praat(audio: AudioData, contour: bool = False,
                  formant_ceiling: float = FORMANT_CEILING_HZ) -> dict:
    """Praat-powered twin of :func:`speechlab.features.analyze`.

    Returns the same keys as the native report, with ``backend`` set to the
    parselmouth version string.  Formant medians/IQR/confidence follow the
    native computation over Praat's frame-wise Formant values on voiced
    frames, so downstream consumers (agent, web UI, CSV export) need no
    changes beyond reading ``report["backend"]``.
    """
    parselmouth = _require()
    snd = parselmouth.Sound(np.asarray(audio.samples, dtype=np.float64),
                            sampling_frequency=float(audio.sample_rate))

    # ---- pitch ----------------------------------------------------------
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=_FMIN,
                            pitch_ceiling=_FMAX)
    track = _praat_pitch_track(parselmouth, pitch)
    segs = voiced_segments(track)

    # ---- voice quality on the longest voiced segment (Praat's own
    # sustained-vowel convention, matching the native backend) ------------
    pp_src = snd
    js_segment = None
    if segs:
        seg = max(segs, key=lambda s: s["end_s"] - s["start_s"])
        js_segment = seg
        pp_src = snd.extract_part(from_time=seg["start_s"], to_time=seg["end_s"])
    pp = parselmouth.praat.call(pp_src, "To PointProcess (periodic, cc)",
                                _FMIN, _FMAX)
    n_points = parselmouth.praat.call(pp, "Get number of points")
    if n_points >= 3:
        jitter = parselmouth.praat.call(
            pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100.0
        shimmer = parselmouth.praat.call(
            [pp_src, pp], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02,
            1.3, 1.6)
        vq = {"jitter_local_percent": float(jitter),
              "shimmer_local_db": float(shimmer),
              "n_periods": int(n_points) - 1}
    else:
        vq = {"jitter_local_percent": float("nan"),
              "shimmer_local_db": float("nan"), "n_periods": 0}

    harm = parselmouth.praat.call(pp_src, "To Harmonicity (cc)",
                                  0.01, _FMIN, 0.1, 1.0)
    hv = np.asarray(harm.values).ravel()
    hv = hv[hv > -200]  # Praat's "undefined" sentinel
    hnr_db = float(np.mean(hv)) if len(hv) else float("nan")

    # ---- formants: median F1-F3 (+ IQR, confidence) on voiced frames ----
    formant = snd.to_formant_burg(
        time_step=0.01, max_number_of_formants=5,
        maximum_formant=formant_ceiling, window_length=0.025,
        pre_emphasis_from=50.0)
    f_stack_rows: list[list[float]] = []
    track_rows: dict[int, list[float]] = {}
    for i, t in enumerate(track.times):
        row = []
        for n in (1, 2, 3):
            v = formant.get_value_at_time(n, float(t))
            if not np.isnan(v) and v > 0:
                row.append(float(v))
            else:
                break  # Praat numbers formants consecutively
        track_rows[i] = row
        if track.voiced[i] and len(row) == 3:
            f_stack_rows.append(row)
    formant_summary: dict = {}
    if f_stack_rows:
        f_stack = np.vstack(f_stack_rows)
        med = np.median(f_stack, axis=0)
        q25, q75 = np.percentile(f_stack, [25, 75], axis=0)
        iqr = q75 - q25
        voiced_rows = [track_rows[i] for i in np.where(track.voiced)[0]]
        coverage = [float(np.mean([len(r) > k for r in voiced_rows]))
                    for k in range(3)]
        stability = [float(np.clip(1.0 - iqr[k] / (0.2 * max(med[k], 1.0)), 0.0, 1.0))
                     for k in range(3)]
        formant_summary = {
            "F1_hz": float(med[0]), "F2_hz": float(med[1]), "F3_hz": float(med[2]),
            "n_frames": len(f_stack),
            "F1_iqr_hz": round(float(iqr[0]), 1),
            "F2_iqr_hz": round(float(iqr[1]), 1),
            "F3_iqr_hz": round(float(iqr[2]), 1),
            "confidence": {f"F{k + 1}": round(coverage[k] * stability[k], 2)
                           for k in range(3)},
        }

    x, sr = audio.samples, audio.sample_rate
    report = {
        "file": audio.path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": sr,
        "n_samples": int(audio.num_samples),
        "backend": f"praat-parselmouth {parselmouth.VERSION}",
        "pitch": track.summary(),
        "voice_quality": vq,
        "voice_quality_segment_s": (js_segment["start_s"], js_segment["end_s"])
                                   if js_segment else None,
        "hnr_db": round(hnr_db, 2) if len(x) else float("nan"),
        "formants": formant_summary,
        "recording_quality": recording_quality(x, sr),
        "voiced_segments": segs,
        "pause_stats": pause_stats(x, sr),
    }
    if contour:
        step = max(1, len(track.times) // 400)
        idx = np.arange(0, len(track.times), step)
        report["pitch_contour"] = {
            "times_s": [round(float(t), 3) for t in track.times[idx]],
            "f0_hz": [round(float(f), 1) if v else None
                      for f, v in zip(track.f0[idx], track.voiced[idx])],
        }
        step_f = max(1, len(track.times) // 200)
        idx_f = np.arange(0, len(track.times), step_f)
        cols: list[list] = [[], [], []]
        for i in idx_f:
            row = track_rows.get(int(i), [])
            for k in range(3):
                cols[k].append(round(row[k], 1) if len(row) > k else None)
        report["formant_track"] = {
            "times_s": [round(float(t), 3) for t in track.times[idx_f]],
            "F1_hz": cols[0], "F2_hz": cols[1], "F3_hz": cols[2],
        }
        report["spectrogram"] = spectrogram(x, sr)
    return report
