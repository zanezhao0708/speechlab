"""Tests for the praat-parselmouth backend (features.analyze(backend="praat")).

These tests validate the contract that makes the backend usable, not Praat
itself: identical report schema, agreement with the native backend on a
clean synthetic vowel within the tolerances the Praat benchmark measured
(F0 median r = 1.000, bias -0.17 Hz), graceful degradation when
praat-parselmouth is missing, and the CLI/agent wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip(
    "parselmouth", reason="bench extra not installed: pip install -e '.[bench]'")

from speechlab.audio import load_audio
from speechlab.cli import main as cli_main
from speechlab.features import analyze

from .helpers import perturbed_vowel, vowel_like, write_wav


@pytest.fixture(scope="module")
def vowel_file(tmp_path_factory):
    """A clean 140 Hz synthetic vowel, once per module."""
    path = tmp_path_factory.mktemp("praat_backend") / "vowel.wav"
    return write_wav(str(path), vowel_like(f0_hz=140.0, duration_s=0.8, sr=16000),
                     16000)


def test_report_schema_matches_native(vowel_file):
    native = analyze(load_audio(vowel_file))
    praat = analyze(load_audio(vowel_file), backend="praat")
    assert native["backend"] == "native"
    assert praat["backend"].startswith("praat-parselmouth")
    # every key the agent / web UI / CSV export reads must exist in both
    assert set(native) == set(praat)
    assert set(native["formants"]) == set(praat["formants"])
    assert set(native["voice_quality"]) == set(praat["voice_quality"])


def test_f0_agrees_with_native(vowel_file):
    """Median F0 within 2 % of native on a clean vowel (benchmark bias: 0.17 Hz)."""
    native = analyze(load_audio(vowel_file))["pitch"]["f0_median_hz"]
    praat = analyze(load_audio(vowel_file),
                    backend="praat")["pitch"]["f0_median_hz"]
    assert praat == pytest.approx(native, rel=0.02)


def test_clean_vowel_measures_sane(vowel_file):
    rep = analyze(load_audio(vowel_file), backend="praat")
    vq = rep["voice_quality"]
    assert vq["jitter_local_percent"] < 2.0
    assert vq["shimmer_local_db"] < 1.0
    assert vq["n_periods"] > 50
    assert rep["hnr_db"] > 15.0
    # formants in plausible vowel territory with usable confidence
    assert 400 < rep["formants"]["F1_hz"] < 900
    assert all(c >= 0.5 for c in rep["formants"]["confidence"].values())


def test_voice_quality_uses_longest_voiced_segment(tmp_path):
    """Jitter/shimmer must analyse the sustained-vowel part, not the silence.

    A 1 s vowel followed by 0.5 s of near-silence: the perturbation segment
    must end before the silence starts (the native backend's convention).
    """
    sr = 16000
    voiced = perturbed_vowel(f0_hz=120.0, duration_s=1.0, sr=sr)
    silence = np.zeros(int(0.5 * sr))
    x = np.concatenate([voiced, silence])
    path = str(write_wav(str(tmp_path / "utt.wav"), x, sr))
    rep = analyze(load_audio(path), backend="praat")
    start, end = rep["voice_quality_segment_s"]
    assert end <= 1.05  # analysis confined to the voiced stretch
    assert end - start >= 0.5


def test_contour_payload_present(vowel_file):
    rep = analyze(load_audio(vowel_file), backend="praat", contour=True)
    assert len(rep["pitch_contour"]["f0_hz"]) > 10
    assert len(rep["formant_track"]["F1_hz"]) > 10
    assert len(rep["formant_track"]["times_s"]) == len(rep["formant_track"]["F1_hz"])


def test_unknown_backend_rejected(vowel_file):
    audio = load_audio(vowel_file)
    with pytest.raises(ValueError, match="unknown backend"):
        analyze(audio, backend="pyworld")


def _broken_require():
    raise RuntimeError("backend='praat' needs praat-parselmouth: "
                       "pip install -e '.[bench]'")


def test_missing_parselmouth_gives_install_hint(vowel_file, monkeypatch):
    """backend="praat" without the extra must fail with a actionable message."""
    import speechlab.praat as praat_mod

    monkeypatch.setattr(praat_mod, "_require", _broken_require)
    audio = load_audio(vowel_file)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[bench\]'"):
        analyze(audio, backend="praat")


def test_cli_backend_flag(vowel_file):
    out = CliRunner().invoke(cli_main, ["analyze", "--backend", "praat", vowel_file])
    assert out.exit_code == 0
    assert "F0" in out.output


def test_cli_batch_csv_carries_backend(vowel_file, tmp_path):
    csv_path = tmp_path / "batch.csv"
    out = CliRunner().invoke(cli_main, ["batch", "--backend", "praat",
                                        "--csv", str(csv_path), vowel_file])
    assert out.exit_code == 0
    text = csv_path.read_text()
    header = text.splitlines()[0]
    assert "backend" in header
    assert "praat-parselmouth" in text


def test_agent_tool_backend_param(vowel_file):
    from speechlab.agent import AgentConfig, SpeechResearchAgent

    agent = SpeechResearchAgent(
        AgentConfig(api_key="sk-test", model="test", allowed_dirs=["/"]))
    result = agent.tools["analyze_audio"]({"path": vowel_file,
                                           "backend": "praat"})
    assert result["backend"].startswith("praat-parselmouth")


def test_agent_tool_backend_error_is_reported(vowel_file, monkeypatch):
    """A missing parselmouth must surface as an error dict, not a crash."""
    import speechlab.praat as praat_mod

    monkeypatch.setattr(praat_mod, "_require", _broken_require)
    from speechlab.agent import AgentConfig, SpeechResearchAgent

    agent = SpeechResearchAgent(
        AgentConfig(api_key="sk-test", model="test", allowed_dirs=["/"]))
    result = agent.tools["analyze_audio"]({"path": vowel_file,
                                           "backend": "praat"})
    assert "error" in result
    assert "pip install" in result["error"]
