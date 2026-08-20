"""Command line interface: ``speechlab <command>``."""

from __future__ import annotations

import json

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
def analyze_cmd(path: str, as_json: bool) -> None:
    """Acoustic analysis the agent uses: F0, voice quality, HNR, formants."""
    report = analyze(load_audio(path))
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
        ag = SpeechResearchAgent(AgentConfig())
        answer = ag.ask(question, context=context)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:  # noqa: BLE001 — turn transport errors into CLI errors
        raise click.ClickException(f"agent request failed: {exc}")
    click.echo(answer)


if __name__ == "__main__":
    main()
