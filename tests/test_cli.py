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
