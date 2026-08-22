"""Command line interface: ``speechlab <command>``."""

from __future__ import annotations

import csv
import json
import os

import click

from . import __version__
from .agent import AgentConfig, SpeechResearchAgent
from .audio import load_audio
from .features import analyze


@click.group()
@click.version_option(version=__version__, prog_name="speechlab")
def main() -> None:
    """SpeechLab: an LLM research agent for speech science."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str) -> None:
    """Show basic metadata for an audio file."""
    audio = load_audio(path)
    click.echo(json.dumps({
        "file": path,
        "duration_s": round(audio.duration, 3),
        "sample_rate_hz": audio.sample_rate,
        "n_samples": audio.num_samples,
    }, indent=2, ensure_ascii=False))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="emit raw JSON instead of a table")
@click.option("--backend", type=click.Choice(["native", "praat"]), default="native",
              help="measurement engine: native (default) or praat via praat-parselmouth")
def analyze_cmd(path: str, as_json: bool, backend: str) -> None:
    """Acoustic analysis the agent uses: F0, voice quality, HNR, formants."""
    report = analyze(load_audio(path), backend=backend)
    if as_json:
        click.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return
    click.echo(f"File        : {report['file']}")
    click.echo(f"Duration    : {report['duration_s']} s @ {report['sample_rate_hz']} Hz")
    pitch = report["pitch"]
    if pitch.get("n_voiced_frames"):
        click.echo(
            f"F0          : median {pitch['f0_median_hz']:.1f} Hz "
            f"(mean {pitch['f0_mean_hz']:.1f}, sd {pitch['f0_std_hz']:.1f}, "
            f"range {pitch['f0_min_hz']:.0f}-{pitch['f0_max_hz']:.0f} Hz)"
        )
        click.echo(f"Voiced      : {pitch['voiced_ratio']:.0%} of frames")
    else:
        click.echo("F0          : no voiced frames detected")
    vq = report["voice_quality"]
    if vq.get("n_periods"):
        click.echo(
            f"Jitter      : {vq['jitter_local_percent']:.2f} %   "
            f"Shimmer: {vq['shimmer_local_db']:.2f} dB   "
            f"({vq['n_periods']} periods)"
        )
    click.echo(f"HNR         : {report['hnr_db']} dB")
    if report["formants"]:
        f = report["formants"]
        click.echo(
            f"Formants    : F1 {f['F1_hz']:.0f} Hz, F2 {f['F2_hz']:.0f} Hz, "
            f"F3 {f['F3_hz']:.0f} Hz (median of energetic frames)"
        )


main.add_command(analyze_cmd, name="analyze")

#: columns exported by ``speechlab batch`` for R / SPSS / pandas
_BATCH_COLUMNS = [
    "file", "duration_s", "sample_rate_hz", "backend",
    "f0_median_hz", "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz",
    "voiced_ratio", "jitter_local_percent", "shimmer_local_db", "n_periods",
    "hnr_db", "F1_hz", "F2_hz", "F3_hz",
    "peak_dbfs", "clipping_ratio", "snr_db", "quality_issues",
]


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--csv", "csv_path", type=click.Path(writable=True), default=None,
              help="also write a CSV table (one row per file) for R/SPSS/pandas")
@click.option("--backend", type=click.Choice(["native", "praat"]), default="native",
              help="measurement engine: native (default) or praat via praat-parselmouth")
def batch(paths: tuple[str, ...], csv_path: str | None, backend: str) -> None:
    """Analyse many recordings at once; optional CSV export."""
    rows = []
    for path in paths:
        try:
            report = analyze(load_audio(path), backend=backend)
        except Exception as exc:  # noqa: BLE001 — keep going through bad files
            click.echo(f"[skip] {path}: {exc}", err=True)
            continue
        p, v, q, f = (report.get(k, {}) for k in
                      ("pitch", "voice_quality", "recording_quality", "formants"))
        rows.append({
            "file": os.path.basename(path),
            "duration_s": report["duration_s"],
            "sample_rate_hz": report["sample_rate_hz"],
            "backend": report.get("backend", "native"),
            "f0_median_hz": round(p.get("f0_median_hz", float("nan")), 2),
            "f0_mean_hz": round(p.get("f0_mean_hz", float("nan")), 2),
            "f0_std_hz": round(p.get("f0_std_hz", float("nan")), 2),
            "f0_min_hz": round(p.get("f0_min_hz", float("nan")), 2),
            "f0_max_hz": round(p.get("f0_max_hz", float("nan")), 2),
            "voiced_ratio": round(p.get("voiced_ratio", 0.0), 3),
            "jitter_local_percent": round(v.get("jitter_local_percent", float("nan")), 3),
            "shimmer_local_db": round(v.get("shimmer_local_db", float("nan")), 3),
            "n_periods": v.get("n_periods", 0),
            "hnr_db": report["hnr_db"],
            "F1_hz": round(f.get("F1_hz", float("nan")), 1),
            "F2_hz": round(f.get("F2_hz", float("nan")), 1),
            "F3_hz": round(f.get("F3_hz", float("nan")), 1),
            "peak_dbfs": q.get("peak_dbfs"),
            "clipping_ratio": q.get("clipping_ratio"),
            "snr_db": q.get("snr_db"),
            "quality_issues": "; ".join(q.get("issues", [])),
        })
        click.echo(f"[ok] {rows[-1]['file']}  F0 {rows[-1]['f0_median_hz']} Hz  "
                   f"jitter {rows[-1]['jitter_local_percent']} %  "
                   f"HNR {rows[-1]['hnr_db']} dB")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_BATCH_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        click.echo(f"wrote {len(rows)} rows -> {csv_path}")


@main.command()
@click.argument("question")
@click.option("--context", "-c", "context_path", type=click.Path(exists=True),
              default=None, help="audio file whose analysis is attached as context")
def agent(question: str, context_path: str | None) -> None:
    """Ask the LLM research agent a question (needs SPEECHLAB_API_KEY)."""
    context = None
    if context_path:
        context = {"audio_analysis": analyze(load_audio(context_path))}
    try:
        cfg = AgentConfig()
        if context_path:  # let follow-up tool calls reach the context file
            cfg.allowed_dirs = [os.path.dirname(os.path.abspath(context_path)),
                                *cfg.allowed_dirs]
        ag = SpeechResearchAgent(cfg)
        answer = ag.ask(question, context=context)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:  # noqa: BLE001 — turn transport errors into CLI errors
        raise click.ClickException(f"agent request failed: {exc}")
    click.echo(answer)


if __name__ == "__main__":
    main()
