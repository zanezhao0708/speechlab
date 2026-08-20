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


@pytest.fixture()
def corpus(tmp_path):
    for spk in ("spk01", "spk02"):
        for i in range(2):
            write_wav(str(tmp_path / f"{spk}_{i}.wav"),
                      vowel_like(duration_s=0.3 + 0.1 * i), 16000)
    return str(tmp_path)


def _first_json(text: str) -> dict:
    """Parse the first JSON object in CLI output (stderr may be appended)."""
    return json.JSONDecoder().raw_decode(text)[0]


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


def test_cli_quality(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["quality", sample_wav])
    assert result.exit_code == 0
    data = _first_json(result.output)
    assert "snr_db" in data


def test_cli_features_mfcc(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["features", sample_wav, "--type", "mfcc"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["feature"] == "mfcc"
    assert len(data["matrix"]) > 10
    assert len(data["matrix"][0]) == 13


def test_cli_features_f0(sample_wav):
    runner = CliRunner()
    result = runner.invoke(main, ["features", sample_wav, "--type", "f0"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["f0_hz"]) == len(data["times_s"]) == len(data["voiced"])


def test_cli_report(corpus, tmp_path):
    out = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(main, ["report", corpus, "-o", str(out)])
    assert result.exit_code == 0
    data = _first_json(result.output)
    assert data["n_files"] == 4
    assert data["n_speakers"] == 2
    assert out.exists()


def test_cli_split(corpus, tmp_path):
    out = tmp_path / "split.json"
    runner = CliRunner()
    result = runner.invoke(main, ["split", corpus, "--train", "0.5", "--dev", "0.25",
                                  "--test", "0.25", "-o", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert set(data) == {"train", "dev", "test"}
    total = sum(len(v) for v in data.values())
    assert total == 4


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
