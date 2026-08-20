# SpeechLab

A toolkit and LLM agent for speech-science research.

SpeechLab gives phoneticians, speech-language pathologists and speech-ML
researchers a single, dependency-light Python toolbox for the everyday jobs
of acoustic analysis — plus an LLM research assistant that can run those
analyses itself through tool calling.

```
$ speechlab analyze recording.wav
File        : recording.wav
Duration    : 0.55 s @ 16000 Hz
F0          : median 188.9 Hz (mean 188.9, sd 0.5, range 186-190 Hz)
Voiced      : 93% of frames
Jitter      : 0.52 %   Shimmer: 0.08 dB   (94 periods)
HNR         : 16.66 dB
Formants    : F1 442 Hz, F2 1213 Hz, F3 2460 Hz (median of energetic frames)
```

## Highlights

- **Acoustic analysis** — F0 tracking (NCCF autocorrelation with subharmonic
  suppression and parabolic interpolation), formants (LPC roots with
  sample-rate-aware pre-emphasis), MFCC / log-mel spectrograms, RMS energy,
  jitter / shimmer, HNR.
- **Recording quality checks** — clipping, DC offset, silence ratio, SNR
  (spectral-valley estimate that also works for continuously voiced
  signals), low sample rate warnings.
- **Corpus tools** — recursive scanning with duration/speaker statistics,
  automatic speaker-ID extraction from filenames or directory layout
  (TIMIT-style), speaker-independent train/dev/test splitting.
- **LLM research agent** — an OpenAI-compatible, tool-calling assistant that
  grounds its answers in measurements from your files instead of guessing.
- **Lean dependencies** — only `numpy`, `scipy` and `click`
  (`soundfile` optional, for mp3/flac/ogg).

## Installation

```bash
pip install -e .            # from a checkout
pip install -e ".[soundfile]"   # + mp3/flac/ogg support
```

Requires Python 3.10+.

## Command-line usage

```bash
# one-file analysis
speechlab info recording.wav            # duration, sample rate, size
speechlab analyze recording.wav         # F0, jitter/shimmer, HNR, formants
speechlab analyze recording.wav --json  # machine-readable report
speechlab quality recording.wav         # clipping / DC / SNR / silence issues

# feature matrices as JSON
speechlab features recording.wav --type mfcc
speechlab features recording.wav --type f0 --frame-ms 40 --hop-ms 5

# corpus statistics and splits
speechlab report /path/to/corpus -o report.json
speechlab split /path/to/corpus --train 0.8 --dev 0.1 --test 0.1 -o split.json

# LLM research agent
export SPEECHLAB_API_KEY=sk-...
speechlab agent "Design a speaker-independent evaluation protocol for my corpus"
speechlab agent "Interpret these voice measures" --context patient.wav
```

## Library usage

```python
from speechlab.audio import load_audio
from speechlab.features import analyze, f0_track, mfcc
from speechlab.quality import quality_report
from speechlab.dataset import scan_dataset, dataset_report, speaker_independent_split

audio = load_audio("recording.wav", target_sr=16000)

# full acoustic report (dict, JSON-ready)
report = analyze(audio)

# frame-level tracks
track = f0_track(audio.samples, audio.sample_rate)
print(track.summary())          # f0_mean_hz, f0_median_hz, voiced_ratio, ...

# recording quality
issues = quality_report(audio)

# corpus statistics + speaker-independent split
utterances = scan_dataset("/path/to/corpus")
print(dataset_report(utterances)["total_duration_h"])
split = speaker_independent_split(utterances, 0.8, 0.1, 0.1, seed=42)
```

## The research agent

`speechlab.agent.SpeechResearchAgent` talks to any OpenAI-compatible
`/chat/completions` endpoint and can call the local analysis tools
(`analyze_audio`, `audio_quality`, `dataset_report`) through standard
function calling:

```python
from speechlab.agent import AgentConfig, SpeechResearchAgent

agent = SpeechResearchAgent(AgentConfig(
    api_key="sk-...",            # or set SPEECHLAB_API_KEY
    base_url="https://api.openai.com/v1",   # any compatible endpoint
    model="gpt-4o-mini",
))
print(agent.ask("Summarise the corpus in ./data and flag problematic files."))
```

Configuration via environment variables:

| Variable              | Default                    |
|-----------------------|----------------------------|
| `SPEECHLAB_API_KEY`   | — (required for the agent) |
| `SPEECHLAB_BASE_URL`  | `https://api.openai.com/v1`|
| `SPEECHLAB_MODEL`     | `gpt-4o-mini`              |

The agent is instructed to always measure before interpreting, to report
units and typical clinical ranges, and to distinguish established results
from hypotheses.

## Method notes

- **F0**: 25 ms / 10 ms frames, normalised cross-correlation over 60–500 Hz,
  shortest-lag-within-tolerance peak picking (anti-subharmonic), parabolic
  refinement, 3-frame median smoothing.
- **Formants**: LPC (Levinson–Durbin) on 25 ms frames, pre-emphasis
  coefficient derived from the sample rate (50 Hz corner, Praat-style —
  a fixed 0.97 crushes the F1 region at 16 kHz), root finding with DC and
  Nyquist roots discarded.
- **Jitter/shimmer**: epoch picking on a low-pass smoothed waveform
  (quarter-period smoothing suppresses formant ripple), local
  period-to-period and amplitude perturbation.
- **HNR**: autocorrelation-based, `10·log10(r/(1−r))` at the tracked F0 lag.
- **SNR**: median power-spectrum bin (harmonic valley floor) vs frame mean
  power, combined with a silence-based estimate when quiet frames exist.

All estimators are classical, lightweight implementations intended for
research triage and corpus QC; for publication-grade clinical numbers,
cross-check against Praat or VoiceSauce.

## Examples

See [`examples/`](examples/) for runnable scripts:

- [`analyze_single.py`](examples/analyze_single.py) — full analysis of one file.
- [`dataset_report.py`](examples/dataset_report.py) — corpus statistics and split.
- [`research_agent.py`](examples/research_agent.py) — the LLM agent in action.

## Development

```bash
pip install -e ".[dev]"
pytest          # 60+ tests, all signals synthesised — no fixtures needed
ruff check .
```

## License

MIT
