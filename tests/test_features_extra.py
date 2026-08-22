"""Tests for recording-quality checks, segmentation and temporal structure."""

from __future__ import annotations

import numpy as np
import pytest

from speechlab.features import (
    F0Track,
    f0_track,
    pause_stats,
    recording_quality,
    spectrogram,
    voiced_segments,
)


def test_clipping_run_detected():
    sr = 16000
    x = np.zeros(sr)
    x[5000:5010] = 1.0  # 10 consecutive pinned samples
    q = recording_quality(x, sr)
    assert q["clipping_ratio"] > 0
    assert any("clipping" in i for i in q["issues"])


def test_lone_peak_not_clipping():
    sr = 16000
    x = np.zeros(sr)
    x[5000] = 1.0  # single normalised peak — not real clipping
    q = recording_quality(x, sr)
    assert q["clipping_ratio"] == 0.0
    assert not any("clipping" in i for i in q["issues"])


def test_voiced_segments_merges_short_gaps():
    times = np.arange(0, 2.0, 0.01)
    f0 = np.full(len(times), 150.0)
    voiced = np.ones(len(times), dtype=bool)
    voiced[100:105] = False  # 50 ms gap < max_gap_s=0.1 → merge
    segs = voiced_segments(F0Track(times=times, f0=f0, voiced=voiced))
    assert len(segs) == 1
    assert segs[0]["start_s"] == 0.0
    assert segs[0]["end_s"] == pytest.approx(1.99, abs=0.05)


def test_voiced_segments_drops_short():
    times = np.arange(0, 1.0, 0.01)
    f0 = np.full(len(times), 150.0)
    voiced = np.zeros(len(times), dtype=bool)
    voiced[10:20] = True  # 100 ms < min_len_s=0.3 → dropped
    assert voiced_segments(F0Track(times=times, f0=f0, voiced=voiced)) == []


def test_pause_stats_counts_bursts():
    sr = 16000
    x = np.zeros(sr * 3)
    for start in (0.0, 0.8, 1.7):  # three 0.3 s bursts, two ~0.5 s pauses
        i0, i1 = int(start * sr), int((start + 0.3) * sr)
        t = np.arange(i1 - i0) / sr
        x[i0:i1] = 0.6 * np.sin(2 * np.pi * 150 * t) * np.hanning(i1 - i0)
    ps = pause_stats(x, sr)
    assert ps["n_pauses"] == 2
    assert ps["silence_ratio"] > 0.4
    assert ps["syllable_est"] >= 3  # at least the three bursts


def test_spectrogram_shape_and_finite():
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 300 * t)
    spec = spectrogram(x, sr, max_time_bins=50, max_freq_bins=40)
    assert len(spec["freqs_hz"]) <= 40
    assert len(spec["times_s"]) <= 50
    vals = np.asarray(spec["values_db"], dtype=float)
    assert np.all(np.isfinite(vals))
    # rows must match the frequency labels (regression: labels were pooled
    # but the matrix was not, mislabelling energy by ~4x)
    assert len(spec["freqs_hz"]) == len(spec["values_db"])
    # 300 Hz tone: the hottest row must be a bin near 300 Hz
    row = int(np.unravel_index(np.argmax(vals), vals.shape)[0])
    assert abs(spec["freqs_hz"][row] - 300) <= 160  # one pooled bin width
    # time axis covers the whole 1 s signal
    assert spec["times_s"][-1] > 0.8


def test_f0_medfilt_suppresses_single_octave_error():
    # mostly-stable 150 Hz signal: the in-tracker median filter keeps the
    # summary honest even if individual frames wander
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 150 * t)
    track = f0_track(x, sr)
    med = np.median(track.f0[track.voiced])
    assert 140 < med < 160
