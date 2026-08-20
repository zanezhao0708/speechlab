"""Command line interface: ``speechlab <command>``."""

from __future__ import annotations

import json
import sys

import click

from . import __version__
from .agent import AgentConfig, SpeechResearchAgent
from .audio import load_audio
from .dataset import dataset_report, scan_dataset, speaker_independent_split
from .features import analyze, f0_track, formants, frame_energy, mel_spectrogram, mfcc


@click.group()
@click.version_option(version=__version__, prog_name="speechlab")
def main() -> None:
    """SpeechLab: acoustic analysis, corpus tools and an LLM research agent."""


# ---------------------------------------------------------------------------
# info / analyze / quality
# ---------------------------------------------------------------------------

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
    """Full acoustic analysis: F0, voice quality, HNR, formants."""
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
@click.argument("path", type=click.Path(exists=True))
def quality(path: str) -> None:
    """Check a recording for clipping, DC offset, SNR and silence issues."""
    from .quality import quality_report

    report = quality_report(load_audio(path))
    click.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        click.echo("Issues found:", err=True)
        for issue in report["issues"]:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# feature matrices
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", "feat_type",
              type=click.Choice(["mfcc", "mel", "f0", "energy", "formants"]),
              default="mfcc", show_default=True, help="feature to dump")
@click.option("--frame-ms", default=25.0, show_default=True, help="frame length in ms")
@click.option("--hop-ms", default=10.0, show_default=True, help="hop length in ms")
def features(path: str, feat_type: str, frame_ms: float, hop_ms: float) -> None:
    """Dump a feature matrix as JSON (list of per-frame rows)."""
    audio = load_audio(path)
    sr, x = audio.sample_rate, audio.samples
    fl = max(1, round(sr * frame_ms / 1000.0))
    hl = max(1, round(sr * hop_ms / 1000.0))

    if feat_type == "mfcc":
        mat = mfcc(x, sr, frame_length=fl, hop_length=hl)
        out = {"feature": "mfcc", "shape": list(mat.shape),
               "matrix": [[round(v, 4) for v in row] for row in mat]}
    elif feat_type == "mel":
        mat = mel_spectrogram(x, sr, frame_length=fl, hop_length=hl)
        out = {"feature": "mel", "shape": list(mat.shape),
               "matrix": [[round(v, 2) for v in row] for row in mat]}
    elif feat_type == "f0":
        track = f0_track(x, sr, frame_length=fl, hop_length=hl)
        out = {"feature": "f0", "hop_ms": hop_ms,
               "times_s": [round(t, 4) for t in track.times],
               "f0_hz": [round(v, 1) for v in track.f0],
               "voiced": [bool(v) for v in track.voiced]}
    elif feat_type == "energy":
        e = frame_energy(x, sr, frame_length=fl, hop_length=hl)
        out = {"feature": "energy_db", "hop_ms": hop_ms,
               "values": [round(v, 2) for v in e]}
    else:  # formants
        from .audio import frame_signal
        frames = frame_signal(x, fl, hl, window="rect", center=True)
        rows = [formants(frames[i], sr) for i in range(0, len(frames),
                                                       max(1, len(frames) // 200))]
        out = {"feature": "formants", "rows": [[round(v, 1) for v in r] for r in rows]}

    click.echo(json.dumps(out, ensure_ascii=False))


# ---------------------------------------------------------------------------
# corpus tools
# ---------------------------------------------------------------------------

@main.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False))
@click.option("--speaker-pattern", default=None,
              help="regex with (?P<speaker>...) to extract speaker IDs from filenames")
@click.option("--output", "-o", default=None, help="also write the report to this JSON file")
def report(root: str, speaker_pattern: str | None, output: str | None) -> None:
    """Corpus statistics for a directory of audio files."""
    utterances = scan_dataset(root, speaker_pattern=speaker_pattern)
    result = dataset_report(utterances)
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        click.echo(f"report written to {output}", err=True)


@main.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False))
@click.option("--train", default=0.8, show_default=True, help="train duration ratio")
@click.option("--dev", default=0.1, show_default=True, help="dev duration ratio")
@click.option("--test", default=0.1, show_default=True, help="test duration ratio")
@click.option("--speaker-pattern", default=None,
              help="regex with (?P<speaker>...) for speaker IDs")
@click.option("--seed", default=0, show_default=True)
@click.option("--output", "-o", default=None, help="write split lists to this JSON file")
def split(root: str, train: float, dev: float, test: float,
          speaker_pattern: str | None, seed: int, output: str | None) -> None:
    """Speaker-independent train/dev/test split of a corpus directory."""
    utterances = scan_dataset(root, speaker_pattern=speaker_pattern)
    if not utterances:
        raise click.ClickException("no audio files found")
    result = speaker_independent_split(utterances, train, dev, test, seed=seed)
    counts = {k: len(v) for k, v in result.items()}
    click.echo(json.dumps(counts, indent=2))
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        click.echo(f"split written to {output}", err=True)


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

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
