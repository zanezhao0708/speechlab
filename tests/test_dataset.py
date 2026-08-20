"""Tests for corpus scanning, reporting and speaker-independent splits."""

import json

import pytest

from speechlab.dataset import (
    dataset_report,
    scan_dataset,
    speaker_independent_split,
)

from .helpers import tone, vowel_like, write_wav


@pytest.fixture()
def corpus(tmp_path):
    """A tiny corpus: 3 speakers x 3 utterances of varying duration."""
    specs = [
        ("spk01", [0.5, 0.7, 0.6]),
        ("spk02", [1.0, 0.8]),
        ("spk03", [0.4, 0.5, 0.6, 0.5]),
    ]
    for spk, durs in specs:
        for i, d in enumerate(durs):
            x = 0.4 * vowel_like(f0_hz=110.0, duration_s=d)
            write_wav(str(tmp_path / f"{spk}_utt{i:02d}.wav"), x, 16000)
    # a non-audio file that must be ignored
    (tmp_path / "README.txt").write_text("not audio")
    return tmp_path


def test_scan_finds_audio_only(corpus):
    utterances = scan_dataset(str(corpus))
    assert len(utterances) == 9
    assert all(u.path.endswith(".wav") for u in utterances)


def test_scan_extracts_speakers(corpus):
    utterances = scan_dataset(str(corpus))
    speakers = {u.speaker for u in utterances}
    assert speakers == {"spk01", "spk02", "spk03"}


def test_scan_custom_pattern(corpus):
    utterances = scan_dataset(str(corpus), speaker_pattern=r"^spk(?P<speaker>\d+)_")
    speakers = {u.speaker for u in utterances}
    assert speakers == {"01", "02", "03"}


def test_scan_without_durations(corpus):
    utterances = scan_dataset(str(corpus), with_durations=False)
    assert len(utterances) == 9
    assert all(u.sample_rate_hz == 0 for u in utterances)


def test_scan_missing_dir():
    with pytest.raises(NotADirectoryError):
        scan_dataset("/nonexistent/dir")


def test_dataset_report(corpus):
    utterances = scan_dataset(str(corpus))
    report = dataset_report(utterances)
    assert report["n_files"] == 9
    assert report["n_speakers"] == 3
    assert report["sample_rates"] == {16000: 9}
    total = sum(sum(d) for _, d in [("a", [0.5, 0.7, 0.6]), ("b", [1.0, 0.8]),
                                    ("c", [0.4, 0.5, 0.6, 0.5])])
    assert report["total_duration_h"] == pytest.approx(total / 3600.0, abs=5e-4)
    stats = report["duration_stats_s"]
    assert stats["min"] == pytest.approx(0.4, abs=0.05)
    assert stats["max"] == pytest.approx(1.0, abs=0.05)
    json.dumps(report)  # JSON-serialisable


def test_speaker_independent_split_disjoint(corpus):
    utterances = scan_dataset(str(corpus))
    split = speaker_independent_split(utterances, seed=0)
    assert set(split) == {"train", "dev", "test"}
    # every file appears exactly once
    all_paths = split["train"] + split["dev"] + split["test"]
    assert sorted(all_paths) == sorted(u.path for u in utterances)
    # no speaker in two subsets
    import os
    import re

    def spk_of(path):
        return re.match(r"^(?P<s>spk\d+)_", os.path.basename(path)).group("s")

    sets = [{spk_of(p) for p in split[k]} for k in ("train", "dev", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_split_ratio_validation(corpus):
    utterances = scan_dataset(str(corpus))
    with pytest.raises(ValueError):
        speaker_independent_split(utterances, 0.5, 0.5, 0.5)


def test_scan_subdirectories(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "dev").mkdir()
    write_wav(str(tmp_path / "train" / "a_spk1_1.wav"), tone(200.0, 0.1), 16000)
    write_wav(str(tmp_path / "dev" / "a_spk2_1.wav"), tone(200.0, 0.1), 16000)
    utterances = scan_dataset(str(tmp_path))
    assert len(utterances) == 2


def test_speaker_from_parent_directory(tmp_path):
    """TIMIT-style layout: speaker ID is the directory, not the filename."""
    for spk in ("FCJF0", "MJFA0"):
        d = tmp_path / "DR1" / spk
        d.mkdir(parents=True)
        write_wav(str(d / "SX9.wav"), tone(200.0, 0.1), 16000)
    utterances = scan_dataset(str(tmp_path))
    speakers = {u.speaker for u in utterances}
    assert speakers == {"FCJF0", "MJFA0"}


def test_generic_dirs_not_speakers(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    write_wav(str(d / "SX9.wav"), tone(200.0, 0.1), 16000)
    utterances = scan_dataset(str(tmp_path))
    assert utterances[0].speaker is None


def test_split_keeps_subsets_nonempty(tmp_path):
    """Three equal speakers with 60/20/20 ratios -> every subset gets one."""
    for spk in ("spk01", "spk02", "spk03"):
        for i in range(2):
            write_wav(str(tmp_path / f"{spk}_{i}.wav"),
                      vowel_like(duration_s=0.3), 16000)
    utterances = scan_dataset(str(tmp_path))
    split = speaker_independent_split(utterances, 0.6, 0.2, 0.2, seed=0)
    assert all(len(split[k]) > 0 for k in ("train", "dev", "test"))
    assert sum(len(v) for v in split.values()) == 6
