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

from .audio import load_audio
from .features import analyze as _analyze_audio

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
   decisions require a certified professional.
3. Suggest concrete, feasible next steps: analyses to run, confounds to
   control, or papers/methods to consider.
4. Be honest about uncertainty; distinguish established results from
   hypotheses.
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
    """OpenAI function-calling schema for the local analysis tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_audio",
                "description": (
                    "Acoustic analysis of one audio file: duration, F0 statistics, "
                    "jitter/shimmer, HNR and formant estimates."
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

        return {"analyze_audio": analyze_audio}

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _chat_request(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """POST one chat completion, retrying transient transport errors."""
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
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )

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
