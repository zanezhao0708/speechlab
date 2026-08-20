"""Command line interface: ``speechlab <command>``."""

from __future__ import annotations

import json

import click

from . import __version__
from .agent import AgentConfig, SpeechResearchAgent
from .audio import load_audio
from .features import COMPARE_FIELDS, analyze, compare_reports, report_field


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
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="emit raw JSON instead of a table")
def analyze_cmd(paths: tuple[str, ...], as_json: bool) -> None:
    """Acoustic analysis the agent uses: F0, voice quality, HNR, CPPS, formants."""
    reports = [analyze(load_audio(p)) for p in paths]
    if as_json:
        out = reports[0] if len(reports) == 1 else reports
        click.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    for i, report in enumerate(reports):
        if i:
            click.echo("")
        _print_report_table(report)


def _print_report_table(report: dict) -> None:
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
    click.echo(f"CPPS        : {report['cpps_db']} dB")
    spec = report.get("spectral")
    if spec:
        click.echo(
            f"Spectral    : centroid {spec['spectral_centroid_hz']:.0f} Hz, "
            f"tilt {spec['spectral_tilt_db_per_khz']:+.1f} dB/kHz, "
            f"flatness {spec['spectral_flatness']:.3f}"
        )
    act = report.get("activity")
    if act:
        click.echo(
            f"Activity    : {act['n_speech_segments']} segment(s), "
            f"{act['speech_ratio']:.0%} speech, "
            f"{act['n_pauses']} pause(s) (mean {act['mean_pause_s']:.2f} s, "
            f"max {act['max_pause_s']:.2f} s)"
        )
    if report["formants"]:
        f = report["formants"]
        click.echo(
            f"Formants    : F1 {f['F1_hz']:.0f} Hz, F2 {f['F2_hz']:.0f} Hz, "
            f"F3 {f['F3_hz']:.0f} Hz (median of energetic frames)"
        )


main.add_command(analyze_cmd, name="analyze")


#: human-readable labels for the COMPARE_FIELDS shown by `speechlab compare`
_COMPARE_LABELS = {
    "duration_s": "Duration (s)",
    "pitch.f0_mean_hz": "F0 mean (Hz)",
    "pitch.f0_median_hz": "F0 median (Hz)",
    "pitch.f0_std_hz": "F0 SD (Hz)",
    "pitch.voiced_ratio": "Voiced ratio",
    "voice_quality.jitter_local_percent": "Jitter (%)",
    "voice_quality.shimmer_local_db": "Shimmer (dB)",
    "hnr_db": "HNR (dB)",
    "cpps_db": "CPPS (dB)",
    "formants.F1_hz": "F1 (Hz)",
    "formants.F2_hz": "F2 (Hz)",
    "formants.F3_hz": "F3 (Hz)",
}


@main.command()
@click.argument("path_a", type=click.Path(exists=True))
@click.argument("path_b", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="emit raw JSON instead of a table")
def compare_cmd(path_a: str, path_b: str, as_json: bool) -> None:
    """Compare two recordings: full analyses plus deltas (B minus A)."""
    report_a = analyze(load_audio(path_a))
    report_b = analyze(load_audio(path_b))
    deltas = compare_reports(report_a, report_b)
    if as_json:
        click.echo(json.dumps({
            "file_a": report_a,
            "file_b": report_b,
            "deltas": deltas,
        }, indent=2, ensure_ascii=False))
        return
    click.echo(f"File A      : {report_a['file']}")
    click.echo(f"File B      : {report_b['file']}")
    click.echo("")
    click.echo(f"{'Measure':<16}{'A':>12}{'B':>12}{'D(B-A)':>12}")
    for field in COMPARE_FIELDS:
        va = report_field(report_a, field)
        vb = report_field(report_b, field)
        if va is None or vb is None or field not in deltas:
            continue
        label = _COMPARE_LABELS.get(field, field)
        click.echo(f"{label:<16}{va:>12.2f}{vb:>12.2f}{deltas[field]:>+12.2f}")


main.add_command(compare_cmd, name="compare")


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
