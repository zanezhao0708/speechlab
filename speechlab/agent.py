"""An LLM-powered speech-research assistant with tool use.

The agent talks to any OpenAI-compatible ``/chat/completions`` endpoint and
can invoke SpeechLab's local acoustic analysis (``analyze_audio``) through
standard function calling, so the model grounds its answers in measured
acoustics instead of guessing.

Configuration (environment variables):

* ``SPEECHLAB_API_KEY``   — API key (required to use the agent).
* ``SPEECHLAB_BASE_URL``  — endpoint base URL, default OpenAI.
* ``SPEECHLAB_MODEL``     — model name, default ``gpt-4o-mini``.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .asr import transcribe as _transcribe
from .audio import load_audio
from .features import analyze as _analyze_audio
from .features import compare_reports as _compare_reports
from .features import diarize as _diarize
from .features import reference_ranges as _reference_ranges

__all__ = ["RESEARCH_SYSTEM_PROMPT", "AgentConfig", "SpeechResearchAgent", "build_tool_specs"]

RESEARCH_SYSTEM_PROMPT = """\
You are SpeechLab Agent, a research assistant for speech science and speech
technology.  Your users are phoneticians, speech-language pathologists,
audiologists and speech-ML researchers.

You are knowledgeable about:
- articulatory and acoustic phonetics (formants, F0, VOT, prosody),
- voice quality measures (jitter, shimmer, HNR) and clinical interpretation,
- speech corpus methodology (recording protocols, transcription, metadata),
- experimental design (speakers, stimuli, counterbalancing, statistics),
- speech technology (ASR, TTS, speaker verification) and standard benchmarks.

Ground rules:
1. When the user mentions a local audio file, ALWAYS use the analyze_audio
   tool to obtain measurements before interpreting them.  Never invent numbers.
2. Report units (Hz, dB, ms, %) alongside every measurement and note normal
   ranges when giving clinical interpretations, with the caveat that clinical
   decisions require a certified professional.  The reference_ranges tool
   gives literature screening values; use it rather than guessing norms.
3. Suggest concrete, feasible next steps: analyses to run, confounds to
   control, or papers/methods to consider.
4. Be honest about uncertainty; distinguish established results from
   hypotheses.

Available tools beyond analyze_audio:
- compare_audio: inferential comparison (Welch t-test, Cohen's d) of two
  recordings — use it for before/after or group-difference questions.
- diarize_audio: who speaks when (speaker turns, speaking time) — use it
  when a recording contains more than one speaker.
- transcribe_audio: speech-to-text (local whisper or a configured API) —
  use it when the wording or timing of the utterance matters.
- reference_ranges: normative screening ranges for F0/jitter/shimmer/HNR.
"""


@dataclass
class AgentConfig:
    """Connection settings for an OpenAI-compatible chat API."""

    api_key: str = field(default_factory=lambda: os.environ.get("SPEECHLAB_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "SPEECHLAB_BASE_URL", "https://api.openai.com/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SPEECHLAB_MODEL", "gpt-4o-mini")
    )
    temperature: float = 0.3
    max_tool_rounds: int = 6
    timeout_s: float = 120.0
    max_retries: int = 2
    """Retries (beyond the first attempt) for transient transport errors."""
    retry_backoff_s: float = 1.0
    """Base backoff delay; doubles after each failed attempt."""
    max_history_chars: int = 40_000
    """Rough history budget: older messages are dropped to stay under it."""

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "no API key: set SPEECHLAB_API_KEY (and optionally "
                "SPEECHLAB_BASE_URL / SPEECHLAB_MODEL) or pass AgentConfig(...)"
            )


def build_tool_specs() -> list[dict]:
    """OpenAI function-calling schemas for the local analysis tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_audio",
                "description": (
                    "Acoustic analysis of one audio file: duration, F0 statistics, "
                    "jitter/shimmer, HNR, formants, pauses and speech rate."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "path to the audio file"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_audio",
                "description": (
                    "Statistically compare two audio files (e.g. before/after "
                    "therapy): Welch t-test on F0 contours, Cohen's d effect "
                    "sizes and per-metric differences for jitter/shimmer/HNR."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path_a": {"type": "string", "description": "first audio file (baseline)"},
                        "path_b": {"type": "string", "description": "second audio file (condition)"},
                    },
                    "required": ["path_a", "path_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "diarize_audio",
                "description": (
                    "Who speaks when: lightweight offline speaker diarization. "
                    "Returns speaker turns, speaking time per speaker. Set "
                    "n_speakers=0 to auto-detect (1-4)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "path to the audio file"},
                        "n_speakers": {
                            "type": "integer",
                            "description": "number of speakers, 0 = auto-detect",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "transcribe_audio",
                "description": (
                    "Speech-to-text transcription of an audio file (local whisper "
                    "if installed, otherwise a configured Whisper-compatible API)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "path to the audio file"},
                        "language": {
                            "type": "string",
                            "description": "ISO-639-1 language hint, e.g. 'zh', 'en' (optional)",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reference_ranges",
                "description": (
                    "Literature-derived screening ranges for acoustic metrics "
                    "(F0 by sex/age, jitter, shimmer, HNR, SNR). Call without "
                    "arguments for the full table."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "description": "metric name, e.g. 'jitter', 'f0', 'hnr' (optional)",
                        },
                    },
                },
            },
        },
    ]


def _sanitize(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so results stay valid strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _safe_json(obj: Any) -> str:
    def default(o: Any) -> str:
        try:
            return str(o)
        except Exception:  # noqa: BLE001 — last-resort stringification
            return "<unserialisable>"

    return json.dumps(_sanitize(obj), ensure_ascii=False, default=default,
                      allow_nan=False)


#: HTTP status codes worth retrying: request timeout, rate limit, server errors
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class SpeechResearchAgent:
    """A tool-using LLM assistant for speech research."""

    def __init__(self, config: AgentConfig | None = None,
                 tools: dict[str, Callable[[dict], Any]] | None = None,
                 system_prompt: str = RESEARCH_SYSTEM_PROMPT):
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else self.default_tools()
        self.tool_specs = build_tool_specs()
        self.history: list[dict] = []
        self._tool_cache: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------
    # local tools the model can call
    # ------------------------------------------------------------------
    @staticmethod
    def default_tools() -> dict[str, Callable[[dict], Any]]:
        def analyze_audio(args: dict) -> dict:
            return _analyze_audio(load_audio(args["path"]))

        def compare_audio(args: dict) -> dict:
            ra = _analyze_audio(load_audio(args["path_a"]), contour=True)
            rb = _analyze_audio(load_audio(args["path_b"]), contour=True)
            out = _compare_reports(ra, rb)
            out["files"] = [args["path_a"], args["path_b"]]
            return out

        def diarize_audio(args: dict) -> dict:
            return _diarize(load_audio(args["path"]),
                            n_speakers=int(args.get("n_speakers") or 0))

        def transcribe_audio(args: dict) -> dict:
            return _transcribe(args["path"], language=args.get("language"))

        def reference_ranges(args: dict) -> dict:
            return _reference_ranges(args.get("metric", ""))

        return {
            "analyze_audio": analyze_audio,
            "compare_audio": compare_audio,
            "diarize_audio": diarize_audio,
            "transcribe_audio": transcribe_audio,
            "reference_ranges": reference_ranges,
        }

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _build_request(self, messages: list[dict], tools: list[dict] | None,
                       stream: bool) -> urllib.request.Request:
        cfg = self.config
        cfg.validate()
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )

    def _chat_request(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """POST one chat completion, retrying transient transport errors."""
        cfg = self.config
        req = self._build_request(messages, tools, stream=False)

        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS:
                    raise  # auth/argument errors will not fix themselves
                last_exc = exc
            except urllib.error.URLError as exc:
                last_exc = exc  # connection reset, DNS hiccup, timeouts ...
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_s * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def _stream_chat_request(self, messages: list[dict], tools: list[dict] | None):
        """POST one chat completion with ``stream=True``; yield SSE chunks.

        Retries only cover connection setup: once bytes start flowing the
        caller consumes the response directly.
        """
        cfg = self.config
        req = self._build_request(messages, tools, stream=True)

        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = urllib.request.urlopen(req, timeout=cfg.timeout_s)
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS:
                    raise
                last_exc = exc
            except urllib.error.URLError as exc:
                last_exc = exc
            else:
                with resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue  # keep-alive comments / partial frames
                return
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_s * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # agentic loop
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the conversation history and tool cache."""
        self.history = []
        self._tool_cache.clear()

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """Keep the system message + the newest messages within the budget.

        Trim points never split an assistant/tool call exchange: cutting at a
        ``role == "user"`` boundary keeps the remaining sequence valid.
        """
        budget = self.config.max_history_chars
        total = sum(len(str(m.get("content") or "")) for m in messages)
        if total <= budget or len(messages) <= 2:
            return messages
        system, rest = messages[0], messages[1:]
        acc = len(str(system.get("content") or ""))
        # walk from the newest message backwards, stop at a user turn once
        # we are over budget
        keep_from = len(rest)
        running = 0
        for i in range(len(rest) - 1, -1, -1):
            running += len(str(rest[i].get("content") or ""))
            if running + acc > budget and rest[i].get("role") == "user" and i + 1 < len(rest):
                keep_from = i
                break
            keep_from = i
        trimmed = [system] + rest[keep_from:]
        # last resort: even a single exchange is too big — keep just the tail
        if sum(len(str(m.get("content") or "")) for m in trimmed) > budget * 1.5:
            return [system, rest[-1]]
        return trimmed

    def ask(self, question: str, context: dict | None = None) -> str:
        """Ask a research question; returns the assistant's final answer.

        Conversations are multi-turn: previous exchanges are kept in
        :attr:`history` and sent along with each new question, so follow-ups
        like "now compare it with the other file" work.  Call
        :meth:`reset` to start fresh.

        Parameters
        ----------
        context : optional dict merged into the first user message, e.g.
            ``{"audio_analysis": {...}}`` produced by
            :func:`speechlab.features.analyze`.
        """
        cfg = self.config
        user_content = question
        if context:
            user_content += "\n\nAttached analysis context:\n" + _safe_json(context)

        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]
        messages = self.history + [{"role": "user", "content": user_content}]

        for _round in range(cfg.max_tool_rounds):
            response = self._chat_request(messages, self.tool_specs)
            choice = response["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                messages.append(self._assistant_message(msg))
                for call in msg["tool_calls"]:
                    result = self._execute_tool(call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    })
                continue

            messages.append({"role": "assistant", "content": msg.get("content", "")})
            self.history = self._trim_history(messages)
            return msg.get("content", "")

        # round limit reached without a final answer: close the exchange with
        # a synthetic assistant message so the history stays a valid
        # user/assistant alternation for the next turn
        notice = (
            "（已达到工具调用轮数上限，本次未能给出最终结论。"
            "请尝试缩小问题范围，或直接运行 speechlab analyze 查看原始数据。）"
        )
        messages.append({"role": "assistant", "content": notice})
        self.history = self._trim_history(messages)
        return notice

    def ask_stream(self, question: str, context: dict | None = None):
        """Streaming variant of :meth:`ask` — yields event dicts.

        Event types:

        - ``{"type": "delta", "text": str}`` — one token chunk of the final
          answer (streamed live from the model).
        - ``{"type": "tool", "name": str, "status": "start"|"done"}`` — a
          local tool is being executed (the UI can show progress).
        - ``{"type": "done", "answer": str}`` — final full answer; history is
          updated exactly like in :meth:`ask`.
        - ``{"type": "error", "error": str}`` — transport/tool failure; the
          exchange is not recorded in history.
        """
        cfg = self.config
        user_content = question
        if context:
            user_content += "\n\nAttached analysis context:\n" + _safe_json(context)

        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]
        messages = self.history + [{"role": "user", "content": user_content}]

        try:
            for _round in range(cfg.max_tool_rounds):
                content_parts: list[str] = []
                calls_acc: dict[int, dict] = {}

                for chunk in self._stream_chat_request(messages, self.tool_specs):
                    if not chunk.get("choices"):
                        continue  # usage-only frames
                    delta = chunk["choices"][0].get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        yield {"type": "delta", "text": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        acc = calls_acc.setdefault(
                            idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["args"] += fn["arguments"]

                if calls_acc:
                    calls = [{
                        "id": acc["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": acc["name"],
                                     "arguments": acc["args"] or "{}"},
                    } for i, acc in sorted(calls_acc.items())]
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant", "content": "".join(content_parts) or ""}
                    assistant_msg["tool_calls"] = calls
                    messages.append(assistant_msg)
                    for call in calls:
                        yield {"type": "tool", "name": call["function"]["name"],
                               "status": "start"}
                        result = self._execute_tool(call)
                        yield {"type": "tool", "name": call["function"]["name"],
                               "status": "done"}
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        })
                    continue

                answer = "".join(content_parts)
                messages.append({"role": "assistant", "content": answer})
                self.history = self._trim_history(messages)
                yield {"type": "done", "answer": answer}
                return

            notice = (
                "（已达到工具调用轮数上限，本次未能给出最终结论。"
                "请尝试缩小问题范围，或直接运行 speechlab analyze 查看原始数据。）"
            )
            messages.append({"role": "assistant", "content": notice})
            self.history = self._trim_history(messages)
            yield {"type": "done", "answer": notice}
        except Exception as exc:  # noqa: BLE001 — report, keep history usable
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _assistant_message(msg: dict) -> dict:
        """Rebuild an assistant message containing tool calls."""
        out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        calls = []
        for c in msg.get("tool_calls", []):
            fn = c["function"]
            calls.append({
                "id": c["id"],
                "type": "function",
                "function": {"name": fn["name"], "arguments": fn.get("arguments", "{}")},
            })
        if calls:
            out["tool_calls"] = calls
        return out

    def _execute_tool(self, call: dict) -> str:
        name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or "{}"

        # models sometimes re-request the same analysis within one session;
        # cached results keep that cheap
        cache_key = f"{name}:{raw_args}"
        cached = self._tool_cache.get(cache_key)
        if cached is not None:
            self._tool_cache.move_to_end(cache_key)
            return cached

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments were not valid JSON"})
        if name not in self.tools:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = self.tools[name](args)
            out = _safe_json(result)
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

        while len(self._tool_cache) >= 64:
            self._tool_cache.popitem(last=False)
        self._tool_cache[cache_key] = out
        return out
