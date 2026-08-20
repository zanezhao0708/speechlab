# SpeechLab Agent

An LLM research agent for speech science.

SpeechLab Agent talks to any OpenAI-compatible chat API and grounds its
answers in **locally measured acoustics**: when you point it at an audio
file, it calls its built-in `analyze_audio` tool (F0, formants,
jitter/shimmer, HNR, CPPS, spectral shape, speech/pause structure)
through standard function calling, or `compare_audio` to contrast two
recordings, and interprets the real numbers instead of guessing.

```
$ speechlab agent "这例嗓音的 F0 和 jitter 数据说明什么?" --context patient.wav

F0 median is 188.9 Hz ... jitter 0.52 % is within the <1 % norm ...
HNR 16.66 dB is slightly below the ~20 dB typical of a stable voice ...
(suggested next steps follow)
```

## How it works

```
        question                    tool call (JSON)         measurements
You ─────────────────▶ LLM (OpenAI-compatible) ──────────▶ analyze_audio()
                          ▲                                       │
                          └───────────── answer ◀────────────────┘
```

- **Agent loop** — `SpeechResearchAgent.ask()` runs a standard
  tool-calling loop (up to 6 rounds): the model requests
  `analyze_audio(path)` (or `compare_audio(path_a, path_b)` for
  pre/post comparisons), the tool returns a JSON report, and the model
  writes its grounded answer.
- **Acoustic core** — the tools the agent owns:
  F0 tracking (NCCF autocorrelation with subharmonic suppression and
  parabolic interpolation), formants F1–F3 (LPC roots with
  sample-rate-aware pre-emphasis), jitter / shimmer (epoch-based
  perturbation), HNR (autocorrelation estimate), CPPS (smoothed
  cepstral peak prominence), LTAS spectral shape (centroid, tilt,
  flatness), and energy-based voice activity / pause structure.
- **Lean** — only `numpy`, `scipy` and `click`; `soundfile` optional for
  mp3/flac/ogg.

## Installation

```bash
pip install -e .                # from a checkout
pip install -e ".[soundfile]"   # + mp3/flac/ogg support
```

Requires Python 3.10+.

## Usage

```bash
export SPEECHLAB_API_KEY=sk-...
# optional:
export SPEECHLAB_BASE_URL=https://api.openai.com/v1
export SPEECHLAB_MODEL=gpt-4o-mini

speechlab agent "Design a speaker-independent evaluation protocol"
speechlab agent "Interpret these voice measures" --context patient.wav

# the analysis the agent runs under the hood, standalone:
speechlab analyze patient.wav          # one file
speechlab analyze pre.wav post.wav     # several files
speechlab compare pre.wav post.wav     # side-by-side deltas (B minus A)
speechlab info patient.wav
```

Python API:

```python
from speechlab.agent import AgentConfig, SpeechResearchAgent

agent = SpeechResearchAgent(AgentConfig(
    api_key="sk-...",                      # or set SPEECHLAB_API_KEY
    base_url="https://api.openai.com/v1",  # any compatible endpoint
    model="gpt-4o-mini",
))
print(agent.ask("Compare the F0 statistics of ./a.wav and ./b.wav"))

# conversations are multi-turn — follow-ups keep the context:
agent.ask("Now which of the two has the more stable voice?")
agent.reset()  # start fresh

print(agent.ask("Is this voice measurement pathological?", context={
    "audio_analysis": analyze(load_audio("patient.wav")),
}))
```

Transient transport errors (rate limits, 5xx, connection resets) are
retried automatically with exponential backoff; repeated tool calls on the
same file reuse cached analysis results.

Configuration via environment variables:

| Variable              | Default                    |
|-----------------------|----------------------------|
| `SPEECHLAB_API_KEY`   | — (required)               |
| `SPEECHLAB_BASE_URL`  | `https://api.openai.com/v1`|
| `SPEECHLAB_MODEL`     | `gpt-4o-mini`              |

The system prompt instructs the agent to always measure before
interpreting, report units (Hz, dB, ms, %) with normal ranges, and
distinguish established results from hypotheses.

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
- **CPPS**: 40 ms / 5 ms Hann frames, real cepstrum of the dB magnitude
  spectrum, peak over the 60–500 Hz quefrency band measured against a
  per-frame least-squares trend line, smoothed ~60 ms and averaged over
  energetic frames.  Absolute values are implementation-specific —
  interpret changes within this library (e.g. pre vs post therapy)
  rather than against external thresholds.
- **Spectral shape**: LTAS of energetic Hann frames — energy-weighted
  centroid, dB/kHz tilt (least-squares slope), geometric/arithmetic
  flatness.
- **Voice activity**: RMS-energy VAD; inactive gaps under 60 ms are
  bridged, segments under 20 ms dropped, yielding speech segments,
  pause count/duration and speaking-time ratio.

These are classical, lightweight implementations intended for research
triage; for publication-grade clinical numbers, cross-check against Praat
or VoiceSauce.

## Example

See [`examples/research_agent.py`](examples/research_agent.py) for a
runnable script.

## Development

```bash
pip install -e ".[dev]"
pytest          # all signals synthesised — no fixtures needed
ruff check .
```

## License

MIT
