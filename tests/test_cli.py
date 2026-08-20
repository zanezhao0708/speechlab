"""Tests for the command line interface (via click's test runner)."""

import json

import pytest
from click.testing import CliRunner

from speechlab.cli import main

from .helpers import vowel_like, write_wav


@pytest.fixture()
def sample_wav(tmp_path):
    path = write_wav(str(tmp_path / "utt.wav"), vowel_like(duration_s=1.0, f0_hz=130.0),
                     16000)
    return str(path)


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "speechlab" in result.output


def test_cli_info(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["info", sample_wav])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["sample_rate_hz"] == 16000
    assert data["duration_s"] == pytest.approx(1.0, abs=0.01)


def test_cli_analyze(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", sample_wav])
    assert result.exit_code == 0
    assert "F0" in result.output
    assert "Jitter" in result.output


def test_cli_analyze_json(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", sample_wav, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["pitch"]["f0_median_hz"] == pytest.approx(130.0, rel=0.05)
    # new measures are part of every report
    assert "cpps_db" in data
    assert "spectral" in data and "activity" in data


def test_cli_analyze_multiple_files(sample_wav, tmp_path):
    other = write_wav(str(tmp_path / "utt2.wav"),
                      vowel_like(duration_s=1.0, f0_hz=160.0), 16000)
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", sample_wav, other, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list) and len(data) == 2
    assert data[0]["pitch"]["f0_median_hz"] == pytest.approx(130.0, rel=0.05)
    assert data[1]["pitch"]["f0_median_hz"] == pytest.approx(160.0, rel=0.05)


def test_cli_compare_table(sample_wav, tmp_path):
    other = write_wav(str(tmp_path / "utt2.wav"),
                      vowel_like(duration_s=1.0, f0_hz=160.0), 16000)
    runner = CliRunner()
    result = runner.invoke(main, ["compare", sample_wav, other])
    assert result.exit_code == 0
    assert "File A" in result.output and "File B" in result.output
    assert "F0 median (Hz)" in result.output
    assert "CPPS (dB)" in result.output


def test_cli_compare_json(sample_wav, tmp_path):
    other = write_wav(str(tmp_path / "utt2.wav"),
                      vowel_like(duration_s=1.0, f0_hz=160.0), 16000)
    runner = CliRunner()
    result = runner.invoke(main, ["compare", sample_wav, other, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data) == {"file_a", "file_b", "deltas"}
    assert data["file_a"]["file"] == sample_wav
    assert data["deltas"]["pitch.f0_median_hz"] == pytest.approx(30.0, abs=10.0)


def test_cli_agent_without_key(sample_wav, monkeypatch):
    monkeypatch.delenv("SPEECHLAB_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "hello"])
    assert result.exit_code != 0
    assert "SPEECHLAB_API_KEY" in result.output


def test_cli_agent_with_context(sample_wav, monkeypatch):
    """The agent path fails fast on a bad endpoint, but context loads first."""
    monkeypatch.setenv("SPEECHLAB_API_KEY", "sk-test")
    monkeypatch.setenv("SPEECHLAB_BASE_URL", "http://127.0.0.1:1/v1")
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "describe this", "--context", sample_wav])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()
