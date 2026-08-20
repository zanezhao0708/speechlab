# SpeechLab Agent

An LLM research agent for speech science.

SpeechLab Agent talks to any OpenAI-compatible chat API and grounds its
answers in **locally measured acoustics**: when you point it at an audio
file, it calls its built-in `analyze_audio` tool (F0, formants,
jitter/shimmer, HNR) through standard function calling and interprets the
real numbers instead of guessing.

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
                          ▲                     │                  │
                          │      compare_audio / diarize_audio /   │
                          │      transcribe_audio / reference_ranges
                          └───────────── answer ◀────────────────┘
```

- **Agent loop** — `SpeechResearchAgent.ask()` runs a standard
  tool-calling loop (up to 6 rounds): the model requests a tool, the
  tool returns a JSON report, and the model writes its grounded answer.
  `ask_stream()` is the SSE variant — token deltas plus tool-progress
  events — used by the web UI for live streaming replies.
- **Acoustic core** — five tools the agent owns:
  F0 tracking (NCCF autocorrelation with subharmonic suppression and
  parabolic interpolation), formants F1–F3 (LPC roots with
  sample-rate-aware pre-emphasis), jitter / shimmer (epoch-based
  perturbation), HNR (autocorrelation estimate);
  `compare_audio` (Welch t-test + Cohen's d between recordings),
  `diarize_audio` (offline speaker turns via log-Mel clustering),
  `transcribe_audio` (local whisper or a Whisper-compatible API) and
  `reference_ranges` (literature screening norms).
- **Lean** — only `numpy`, `scipy` and `click`; `soundfile` optional for
  mp3/flac/ogg; `openai-whisper` / `faster-whisper` optional for offline ASR.

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
speechlab analyze patient.wav
speechlab info patient.wav

# batch-analyse many files, one CSV row per file (20+ measures):
speechlab batch recordings/*.wav --csv results.csv
```

## Web chat interface

```bash
python web/app.py          # http://localhost:8000

# optional deployment knobs:
export SPEECHLAB_WEB_PASSWORD=...   # gate every /api route behind a shared password
export SPEECHLAB_WEB_DB=/path/x.db  # SQLite location for history/trends
```

A chatbot-style web UI for people who don't use the CLI:

- **Chat (streaming)** — multi-turn agent conversations with answers
  streamed token-by-token (SSE); tool invocations (`analyze_audio`,
  `compare_audio`, …) show live progress. Users bring their own API key
  (stored in their browser only) or the server provides one via
  `SPEECHLAB_API_KEY`.
- **Browser recording** — record from the mic, get an instant local
  analysis; the recording is also attached as agent context.
- **Direct analysis (no API key)** — upload or record audio and get the
  full local report: F0 statistics, jitter/shimmer (measured on the
  longest sustained voiced segment), HNR, formants, spectrogram,
  pitch contour, pause/speech-rate structure, recording-quality flags
  with literature-based ✓/△/✗ grading, and one-click Markdown export.
- **Multi-file comparison** — select several files at once for a side-by-side
  table (best values highlighted) with per-speaker F0 contours; ask the
  agent "对比这两个录音" and it runs inferential statistics itself via
  `compare_audio`.
- **Longitudinal trends, cross-device** — every analysis is saved to the
  server (SQLite) under an anonymous browser identity, so progress
  charts follow the user across laptops/phones, with local history
  merged in and de-duplicated.
- **History restore** — conversations persist server-side and can be
  reopened from any device via 🕘 历史.
- **Access password** — set `SPEECHLAB_WEB_PASSWORD` to require a shared
  password before the API is usable (for semi-public deployments).

Deployment notes: sessions are in-memory (dev server); add a reverse
proxy (with SSE buffering disabled) and TLS before exposing it publicly.
The server enforces per-session upload quotas (200 MB), chat rate limits
(8 requests/min) and caps concurrent LLM calls.

Python API:

```python
from speechlab.agent import AgentConfig, SpeechResearchAgent
from speechlab.audio import load_audio
from speechlab.features import analyze

agent = SpeechResearchAgent(AgentConfig(
    api_key="sk-...",                      # or set SPEECHLAB_API_KEY
    base_url="https://api.openai.com/v1",  # any compatible endpoint
    model="gpt-4o-mini",
))
print(agent.ask("Compare the F0 statistics of ./a.wav and ./b.wav"))

# streaming variant: yields {"type": "delta"|"tool"|"done"|"error", ...}
for ev in agent.ask_stream("What does the jitter of ./a.wav mean?"):
    if ev["type"] == "delta":
        print(ev["text"], end="", flush=True)

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
  period-to-period and amplitude perturbation. Computed on the longest
  sustained voiced segment when one exists — perturbation norms assume a
  sustained vowel, not connected speech.
- **HNR**: autocorrelation-based, `10·log10(r/(1−r))` at the tracked F0 lag.
- **Temporal structure**: energy-gated pauses (≥0.2 s) and a syllable-nuclei
  estimate from smoothed energy peaks requiring both a minimum prominence
  over the adjacent valleys and ≥120 ms spacing (de Jong & Wempe style) —
  a rough articulation-rate proxy, not a forced-alignment replacement.
- **Diarization**: energy-normalised log-Mel frame features, agglomerative
  average-linkage clustering over cosine distance, majority smoothing and
  turn merging; speaker count auto-selected by silhouette when unknown.
  Compact and offline — expect ~80–90 % frame accuracy on clean
  two-speaker audio, not production-grade diarization.

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
