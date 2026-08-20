"""An LLM-powered speech-research assistant with tool use.

The agent talks to any OpenAI-compatible ``/chat/completions`` endpoint and
can invoke SpeechLab's local analysis tools (``analyze_audio``,
``audio_quality``, ``dataset_report``) through standard function calling,
so the model grounds its answers in measured acoustics instead of guessing.

Configuration (environment variables):

* ``SPEECHLAB_API_KEY``   — API key (required to use the agent).
* ``SPEECHLAB_BASE_URL``  — endpoint base URL, default OpenAI.
* ``SPEECHLAB_MODEL``     — model name, default ``gpt-4o-mini``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .audio import load_audio
from .dataset import dataset_report as _dataset_report
from .dataset import scan_dataset
from .features import analyze as _analyze_audio
from .quality import quality_report as _quality_report

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
1. When the user mentions a local audio file or corpus directory, ALWAYS use
   the provided tools (analyze_audio / audio_quality / dataset_report) to
   obtain measurements before interpreting them.  Never invent numbers.
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

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "no API key: set SPEECHLAB_API_KEY (and optionally "
                "SPEECHLAB_BASE_URL / SPEECHLAB_MODEL) or pass AgentConfig(...)"
            )


def build_tool_specs() -> list[dict]:
    """OpenAI function-calling schema for the local analysis tools."""
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
        {
            "type": "function",
            "function": {
                "name": "audio_quality",
                "description": (
                    "Recording-quality report: clipping, DC offset, SNR estimate, "
                    "silence ratio and a list of detected issues."
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
                "name": "dataset_report",
                "description": (
                    "Scan a corpus directory and report file counts, duration "
                    "statistics, speaker breakdown and sample rates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "corpus root directory"},
                        "speaker_pattern": {
                            "type": "string",
                            "description": (
                                "optional regex with a named group (?P<speaker>...) "
                                "for extracting speaker IDs from file names"
                            ),
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]


def _safe_json(obj: Any) -> str:
    def default(o: Any) -> str:
        try:
            return str(o)
        except Exception:  # noqa: BLE001 — last-resort stringification
            return "<unserialisable>"

    return json.dumps(obj, ensure_ascii=False, default=default)


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

    # ------------------------------------------------------------------
    # local tools the model can call
    # ------------------------------------------------------------------
    @staticmethod
    def default_tools() -> dict[str, Callable[[dict], Any]]:
        def analyze_audio(args: dict) -> dict:
            return _analyze_audio(load_audio(args["path"]))

        def audio_quality(args: dict) -> dict:
            return _quality_report(load_audio(args["path"]))

        def dataset_report(args: dict) -> dict:
            utterances = scan_dataset(args["path"],
                                       speaker_pattern=args.get("speaker_pattern"))
            return _dataset_report(utterances)

        return {
            "analyze_audio": analyze_audio,
            "audio_quality": audio_quality,
            "dataset_report": dataset_report,
        }

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _chat_request(self, messages: list[dict], tools: list[dict] | None) -> dict:
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
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # agentic loop
    # ------------------------------------------------------------------
    def ask(self, question: str, context: dict | None = None) -> str:
        """Ask a research question; returns the assistant's final answer.

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

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        self.history = list(messages)

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
            self.history = messages
            return msg.get("content", "")

        self.history = messages
        return "(reached the tool-call round limit without a final answer)"

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
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments were not valid JSON"})
        if name not in self.tools:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = self.tools[name](args)
            return _safe_json(result)
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
