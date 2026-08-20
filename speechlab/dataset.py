"""Corpus statistics and speaker-independent dataset splitting.

Scans a directory tree for audio files, computes duration statistics,
optionally groups utterances by speaker using a filename regex, and can
produce train/dev/test splits that never place the same speaker in two
subsets (the standard evaluation protocol in speaker technology).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .audio import load_audio

__all__ = ["Utterance", "dataset_report", "scan_dataset", "speaker_independent_split"]

AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

# a few common naming conventions; the first match wins
DEFAULT_SPEAKER_PATTERNS = [
    r"^(?P<speaker>\d+)[_-]",           # 1234_utt.wav  (TIMIT-ish)
    r"^(?P<speaker>[A-Za-z]+\d+)[_-]",  # spk01_utt.wav
    r"^(?P<speaker>[^-_]+)[-]",         # speaker-14-0001.flac (LibriSpeech)
    r"^(?P<speaker>[^-_]+)[_]",         # speaker_utt1.wav
]

# directory names that are layout labels, never speaker IDs
GENERIC_DIR_NAMES = {
    "train", "dev", "test", "valid", "eval", "data", "audio",
    "wav", "wavs", "utterances", "utt",
}


@dataclass
class Utterance:
    """One audio file in a corpus."""

    path: str
    speaker: str | None
    duration_s: float
    sample_rate_hz: int

    def to_row(self) -> dict:
        return {
            "path": self.path,
            "speaker": self.speaker,
            "duration_s": round(self.duration_s, 3),
            "sample_rate_hz": self.sample_rate_hz,
        }


def _guess_speaker(filename: str, patterns: list[str]) -> str | None:
    stem = os.path.splitext(os.path.basename(filename))[0]
    for pat in patterns:
        m = re.match(pat, stem)
        if m:
            return m.group("speaker")
    return None


def scan_dataset(root: str, speaker_pattern: str | None = None,
                 with_durations: bool = True) -> list[Utterance]:
    """Recursively find audio files under ``root``.

    Parameters
    ----------
    speaker_pattern : regex with a named group ``(?P<speaker>...)`` used to
        extract speaker IDs from file names.  When ``None`` a small set of
        common conventions is tried automatically.
    with_durations : set False to skip opening files (fast metadata scan).
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(root)

    patterns = [speaker_pattern] if speaker_pattern else DEFAULT_SPEAKER_PATTERNS
    root_abs = os.path.abspath(root)
    utterances: list[Utterance] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in AUDIO_EXTENSIONS:
                continue
            path = os.path.join(dirpath, fn)
            speaker = _guess_speaker(fn, patterns)
            if speaker is None and os.path.abspath(dirpath) != root_abs:
                # TIMIT-style layout: speaker ID is the parent directory name
                parent = os.path.basename(dirpath)
                if parent and parent.lower() not in GENERIC_DIR_NAMES:
                    speaker = parent
            if with_durations:
                try:
                    audio = load_audio(path)
                    utterances.append(Utterance(path, speaker, audio.duration,
                                                audio.sample_rate))
                    continue
                except Exception:  # noqa: BLE001, S110 — unreadable files are kept with NaN
                    pass
            utterances.append(Utterance(path, speaker, float("nan"), 0))
    return utterances


def _percentiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def dataset_report(utterances: list[Utterance]) -> dict:
    """Aggregate statistics over a list of :class:`Utterance` objects."""
    if not utterances:
        return {"n_files": 0}

    durations = np.array([u.duration_s for u in utterances], dtype=float)
    durations = durations[np.isfinite(durations)]
    sr_counts: dict[int, int] = defaultdict(int)
    for u in utterances:
        if u.sample_rate_hz:
            sr_counts[u.sample_rate_hz] += 1

    report: dict = {
        "n_files": len(utterances),
        "total_duration_h": round(float(np.sum(durations) / 3600.0), 4)
        if len(durations) else 0.0,
        "n_speakers": len({u.speaker for u in utterances if u.speaker}) or None,
        "sample_rates": dict(sorted(sr_counts.items())),
        "duration_stats_s": {k: round(v, 3) for k, v in _percentiles(durations).items()},
    }

    # per-speaker duration breakdown (top 20 by duration)
    per_speaker: dict[str, float] = defaultdict(float)
    for u in utterances:
        if u.speaker:
            per_speaker[u.speaker] += u.duration_s if np.isfinite(u.duration_s) else 0.0
    if per_speaker:
        top = sorted(per_speaker.items(), key=lambda kv: -kv[1])[:20]
        report["per_speaker_duration_s"] = {k: round(v, 2) for k, v in top}
        spk_durs = np.array(list(per_speaker.values()))
        report["per_speaker_stats_s"] = {
            k: round(v, 3) for k, v in _percentiles(spk_durs).items()
        }
    return report


def speaker_independent_split(utterances: list[Utterance],
                              train_ratio: float = 0.8,
                              dev_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              seed: int = 0) -> dict[str, list[str]]:
    """Split a corpus so that no speaker appears in two subsets.

    Speakers are assigned greedily (largest first) to the subset whose
    duration budget is furthest from its target; a final pass moves speakers
    out of crowded subsets so that requested subsets are not left empty.
    Returns a dict with ``train`` / ``dev`` / ``test`` file-path lists.
    """
    ratios = (train_ratio, dev_ratio, test_ratio)
    names = ("train", "dev", "test")
    if abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0:
        raise ValueError("ratios must be non-negative and sum to 1")

    by_speaker: dict[str | None, list[Utterance]] = defaultdict(list)
    for u in utterances:
        by_speaker[u.speaker or os.path.basename(u.path)].append(u)

    speaker_items = []
    for spk, utts in by_speaker.items():
        dur = sum(u.duration_s for u in utts if np.isfinite(u.duration_s))
        paths = [u.path for u in utts]
        speaker_items.append((spk, dur, paths))
    total = sum(d for _, d, _ in speaker_items) or 1.0

    rng = np.random.default_rng(seed)
    rng.shuffle(speaker_items)  # randomise tie-breaking, then order by size
    speaker_items.sort(key=lambda t: -t[1])

    dur_of = {spk: d for spk, d, _ in speaker_items}
    assignment: dict[str, int] = {}
    bucket_dur = [0.0, 0.0, 0.0]
    for spk, dur, _paths in speaker_items:
        remaining = [ratios[k] * total - bucket_dur[k] for k in range(3)]
        target = int(np.argmax(remaining))
        assignment[spk] = target
        bucket_dur[target] += dur

    # keep every requested subset non-empty when there are enough speakers
    if all(r > 0 for r in ratios) and len(speaker_items) >= 3:
        for k in range(3):
            if any(b == k for b in assignment.values()):
                continue
            counts = [sum(1 for b in assignment.values() if b == i) for i in range(3)]
            donors = [i for i in range(3) if counts[i] >= 2]
            if not donors:
                continue
            donor = max(donors, key=lambda i: counts[i])
            movable = min((s for s, b in assignment.items() if b == donor),
                          key=lambda s: dur_of[s])
            assignment[movable] = k

    buckets: list[list[str]] = [[], [], []]
    for spk, _dur, paths in speaker_items:
        buckets[assignment[spk]].extend(paths)
    return {names[k]: sorted(buckets[k]) for k in range(3)}
