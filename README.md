# SpeechLab Agent

[![CI](https://github.com/zanezhao0708/speechlab/actions/workflows/ci.yml/badge.svg)](https://github.com/zanezhao0708/speechlab/actions/workflows/ci.yml)

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
speechlab analyze patient.wav --backend praat   # publication-grade (see below)
speechlab info patient.wav

# batch-analyse many files, one CSV row per file (20+ measures):
speechlab batch recordings/*.wav --csv results.csv
speechlab batch recordings/*.wav --backend praat --csv results.csv
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

| Variable                   | Default                    |
|----------------------------|----------------------------|
| `SPEECHLAB_API_KEY`        | — (required)               |
| `SPEECHLAB_BASE_URL`       | `https://api.openai.com/v1`|
| `SPEECHLAB_MODEL`          | `gpt-4o-mini`              |
| `SPEECHLAB_ALLOWED_DIRS`   | current working directory  |
| `SPEECHLAB_REDACT_PATHS`   | off                        |

The system prompt instructs the agent to always measure before
interpreting, report units (Hz, dB, ms, %) with reference ranges, and
present jitter/shimmer/HNR thresholds as algorithm-, recording- and
population-dependent screening values (research triage, not diagnosis),
deferring clinical decisions to certified professionals.

## Privacy & security boundaries

What you should know before pointing the agent at recordings:

**What leaves the machine.** Audio itself never leaves your machine for
the acoustic tools — F0/formants/jitter/shimmer/HNR/diarization are all
computed locally. Two things do travel to the configured LLM endpoint:
your question (and any attached analysis context), and the JSON tool
results the model requests. If you use `transcribe_audio` without a
local whisper installation, the audio file is sent to the configured
Whisper-compatible API.

**Path sandbox.** The model chooses which files its tools read — and a
model can be confused or prompt-injected, so file access is confined to
an allowlist of directories (`AgentConfig.allowed_dirs`, default the
current working directory; override with `SPEECHLAB_ALLOWED_DIRS`, an
`os.pathsep`-separated list). Paths are resolved *including symlinks*
before the check, so `../` traversal and in-workspace symlinks pointing
outside both fail closed with a `PermissionError` the model sees as a
tool error. The web UI sandboxes further still: agents there can only
ever touch the current session's uploaded files.

**Path redaction.** Directory names can carry identifying information
(`/home/alice/patients/…`, study IDs, usernames). Set
`SPEECHLAB_REDACT_PATHS=1` (or `AgentConfig(redact_paths=True)`) and
tool results ship paths as `…/patient_042.wav` — basename only — before
anything reaches the API. The web UI always redacts server-side upload
paths this way.

**Clinical caveat.** Reference ranges are literature screening values;
the agent states them as such and defers clinical decisions to
certified professionals.

## Method notes

- **F0**: 25 ms / 10 ms frames (3 periods of the pitch floor, whichever is
  longer), Boersma's window-corrected normalised autocorrelation over
  60–500 Hz, octave-robust period selection (candidates within tolerance
  of the maximum; a half-period is rejected when the 2× lag peaks clearly
  higher; a candidate survives only if its small multiples all show NCCF
  peaks — the anti-subharmonic chain check), parabolic refinement on the
  raw NCCF, 3-frame median smoothing over voiced frames.
- **Formants**: LPC (Levinson–Durbin) on 25 ms frames following Praat's
  Formant(Burg) conventions — the signal is first resampled to twice the
  formant ceiling (5.5 kHz by default), pre-emphasis uses Praat's exact
  `α = exp(−2π·f·Δt)` (50 Hz corner), the analysis order is 2 poles per
  formant (5 formants / 10 poles), and only **voiced** frames (energetic
  half) enter the median, so plosive bursts and fricatives cannot pose as
  formants. DC and Nyquist-adjacent roots are discarded. The report also
  states how much to trust each median: frame count, interquartile spread
  across frames, and a per-formant `confidence` in [0, 1] (coverage ×
  stability — a gliding or noisy formant scores low); `contour=True` adds
  the per-frame F1–F3 track. Validated against known-pole synthetic
  signals in `tests/test_formant_golden.py`.
- **Jitter/shimmer**: epoch picking on a low-pass smoothed waveform
  (quarter-period smoothing suppresses formant ripple), local
  period-to-period and amplitude perturbation. Computed on the longest
  sustained voiced segment when one exists — perturbation norms assume a
  sustained vowel, not connected speech. Known limit: the epoch walk
  confines consecutive periods to [0.75, 1.25]× the median period, so
  extremely rough (pathological) voices are underestimated.
- **HNR**: autocorrelation-based, `10·log10(r/(1−r))` at the tracked F0
  lag, over the same sustained voiced segment as jitter/shimmer; frames
  with no positive periodicity are skipped (undefined, as in Praat).
- **Temporal structure**: energy-gated pauses (≥0.2 s) and a syllable-nuclei
  estimate from smoothed energy peaks requiring both a minimum prominence
  over the adjacent valleys and ≥120 ms spacing (de Jong & Wempe style) —
  a rough articulation-rate proxy, not a forced-alignment replacement.
- **Diarization**: energy-normalised log-Mel frame features, agglomerative
  average-linkage clustering over cosine distance, majority smoothing and
  turn merging; speaker count auto-selected by silhouette when unknown.
  Compact and offline — expect ~80–90 % frame accuracy on clean
  two-speaker audio, not production-grade diarization.

## Accuracy benchmark (SpeechLab vs Praat)

The unit tests only prove the implementation is self-consistent on clean
synthetic signals. For evidence against a reference implementation there is
a benchmark harness in `benchmarks/praat_benchmark.py` that runs
SpeechLab's shipped `analyze()` pipeline and Praat (via the official
`praat-parselmouth` binding) on the same files and reports:

- **F0** — frame-wise MAE (Hz and cents) on shared voiced frames, plus
  per-file median-F0 Pearson r and Bland-Altman agreement;
- **Formants** — MAE and Pearson r for F1/F2/F3;
- **Jitter / shimmer / HNR** — Pearson r, bias and Bland-Altman limits of
  agreement across files.

Both sides go through `analyze()` — `backend="native"` vs
`backend="praat"` — so they share one selection policy (voiced-frame
filtering, energetic-half formant medians, longest-voiced-segment rule for
jitter/shimmer/HNR). File-level comparisons therefore isolate the
measurement engine rather than mixing in selection differences; the only
exception is the frame-wise F0 metric, which compares the two raw pitch
trackers directly.

```bash
pip install -e ".[bench]"                     # parselmouth + matplotlib
# self-check of the harness itself (no real data needed):
python benchmarks/praat_benchmark.py --synthetic 24 --out bench_results
# real evaluation (recommended: CMU Arctic, Saarbrücken Voice Database):
python benchmarks/praat_benchmark.py /path/to/wavs --out bench_results
```

Outputs: `results.csv` (per file), `summary.json` / `summary.md`
(headline table), `scatter.png` and `bland_altman.png` when matplotlib is
installed.

### Real-speech results (CMU Arctic, 100 files)

Corpus: 100 utterances from four CMU Arctic speakers (bdl, rms = male;
slt, clb = female) — 80 clean 16 kHz originals, 10 resampled to
44.1/22.05 kHz, 10 noise-added at 0–20 dB SNR — plus a 10-file 8 kHz
telephony subset analysed at a 3800 Hz formant ceiling. Download and
condition derivation are scripted (`benchmarks/download_cmu_arctic.py`,
`benchmarks/derive_conditions.py`), and the full per-file CSV, scatter
and Bland-Altman figures are committed under `bench_results/`.

| Metric | n | MAE / bias (SL−Praat) | LoA 95% | Pearson r |
|---|---|---|---|---|
| F0 frame-wise | 20467 frames | 1.92 Hz (median 1.18) / 22.5 cents | — | — |
| F0 median per file | 100 | +1.06 Hz | ±3.3 Hz | 0.999 |
| F1 | 100 | 21.1 / +14.4 Hz | [−40, +69] Hz | 0.977 |
| F2 | 100 | 64.3 / +43.0 Hz | [−106, +192] Hz | 0.950 |
| F3 | 100 | 69.1 / +43.1 Hz | [−162, +248] Hz | 0.942 |
| jitter (local) | 100 | +0.39 pp | [−1.7, +2.4] pp | 0.37* |
| shimmer (local dB) | 100 | +0.11 dB | [−0.24, +0.47] dB | 0.69 |
| HNR | 100 | −2.66 dB | [−6.3, +1.0] dB | 0.93 |

\* Pearson r on jitter is depressed by range restriction: the corpus is
all-normal voices, so Praat's jitter spans only 0.6–4.4 % (IQR 0.9 pp)
while the disagreement sd is ~1.0 pp — the noise-to-signal ratio alone
caps r near 0.6. The agreement statistics (bias, LoA) are the meaningful
ones here. The 8 kHz telephony subset, whose jitter values spread wider,
gives r = 0.755 (bias +0.16 pp).

Agreement holds across conditions: per-file F0 r ≥ 0.999 and jitter
bias within ±1 pp in *every* group (clean / resampled / 0–20 dB SNR /
8 kHz telephony); F2/F3 show a systematic +40–100 Hz offset that grows
at non-native sample rates — LPC-pole vs Burg-pole disagreement, not a
selection artifact. HNR carries a stable −2.7 dB calibration offset
(autocorrelation-vs-cc definition).

This run also caught and fixed a real bug: without Praat's
stability-factor pair exclusion (maxPeriodFactor 1.3 / maxPeakFactor
1.6) the native jitter averaged epoch-walk glitches on connected speech
and reported a median 44 % where Praat reported 1.8 %.

> Synthetic-set agreement (24 source-filter vowels across
> clean/noisy/jittered/shimmered/8 kHz conditions): per-file median F0
> r = 1.000 (bias −0.17 Hz, LoA ±1.5 Hz, zero octave errors), F1 MAE
> 20 Hz (r = 0.92), F2 90 Hz (r = 0.91), F3 74 Hz (r = 0.97), HNR
> r = 0.98 (bias −1.8 dB). Jitter r = 0.88 (bias +0.6 %), shimmer
> r = 0.89 (bias +0.2 dB).
>
> Remaining gap to publication-grade evidence: sustained vowels and
> rough/pathological voices (the Saarbrücken Voice Database requires a
> licence agreement), and corpus sizes beyond 100 files. Until then
> cross-check publication-critical perturbation numbers with
> `backend="praat"`.

## Praat backend (publication-grade)

When numbers must sit next to Praat-derived values in a paper, switch the
measurement engine instead of hand-cross-checking: `backend="praat"`
computes the identical report schema through the official
[`praat-parselmouth`](https://github.com/praat/praat) binding — Praat's own
`To Pitch (ac)`, `To Formant (burg)`, `To PointProcess (periodic, cc)`
and `To Harmonicity (cc)`, with jitter/shimmer/HNR on the longest voiced
segment (Praat's sustained-vowel convention).

```bash
pip install -e ".[bench]"          # adds praat-parselmouth
speechlab analyze vowel.wav --backend praat
speechlab batch corpus/*.wav --backend praat --csv results.csv
```

```python
from speechlab.features import analyze
report = analyze(audio, backend="praat")     # same keys as native
report["backend"]                            # -> "praat-parselmouth 0.4.7"
```

Every report (both engines) carries a `backend` field, and the batch CSV
exports it as a column, so exported numbers stay traceable to the engine
that produced them. The agent's `analyze_audio` tool also accepts
`backend: "praat"` and will state which engine produced its numbers.

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
