"""Speech-to-text transcription for the agent's ``transcribe_audio`` tool.

Two tiers, tried in order:

1. **Local whisper** — if the ``openai-whisper`` or ``faster-whisper``
   package is installed, transcription runs fully offline on the machine.
2. **OpenAI-compatible API** — ``POST {base_url}/audio/transcriptions``
   (Whisper-style multipart endpoint), reusing the agent's API key.

No hard dependency on either: when neither is available the caller gets a
clear error it can surface to the user.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

__all__ = ["transcribe"]

#: module-level cache for loaded local models — loading whisper's "base"
#: takes seconds and would otherwise happen on every transcription call
_MODEL_CACHE: dict[str, Any] = {}


def _local_whisper(path: str, language: str | None) -> dict | None:
    """Transcribe with a locally installed whisper package, if any."""
    try:
        import whisper  # type: ignore — openai-whisper
    except ImportError:
        whisper = None

    if whisper is not None:
        if "openai-whisper" not in _MODEL_CACHE:
            _MODEL_CACHE["openai-whisper"] = whisper.load_model("base")
        result = _MODEL_CACHE["openai-whisper"].transcribe(
            path, language=language or None)
        return {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language"),
            "engine": "local openai-whisper (base)",
        }

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        return None

    if "faster-whisper" not in _MODEL_CACHE:
        _MODEL_CACHE["faster-whisper"] = WhisperModel("base", compute_type="int8")
    segments, info = _MODEL_CACHE["faster-whisper"].transcribe(
        path, language=language or None)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "engine": "local faster-whisper (base, int8)",
    }


def _sanitize_filename(name: str) -> str:
    r"""Make a filename safe to embed in a multipart Content-Disposition header.

    ``os.path.basename`` strips path separators but not quotes or CR/LF —
    a crafted name (reachable via the agent's ``transcribe_audio`` tool
    arguments) would inject arbitrary bytes into the HTTP headers.
    """
    name = os.path.basename(name)
    name = re.sub(r'[\r\n"\\]', "_", name)
    name = name.strip() or "audio"
    return name[:80]


def _multipart(fields: dict[str, str], file_path: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body with one file part (stdlib only)."""
    boundary = f"----speechlab{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n"
            f"\r\n{value}\r\n".encode()
        )
    fname = _sanitize_filename(file_path)
    with open(file_path, "rb") as fh:
        blob = fh.read()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n"
        f"\r\n".encode() + blob + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _api_whisper(path: str, language: str | None, api_key: str,
                 base_url: str, model: str, timeout_s: float = 300.0) -> dict:
    """POST the file to an OpenAI-compatible /audio/transcriptions endpoint."""
    fields = {"model": model}
    if language:
        fields["language"] = language
    fields["response_format"] = "json"
    body, boundary = _multipart(fields, path)
    url = base_url.rstrip("/") + "/audio/transcriptions"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return {
        "text": str(payload.get("text", "")).strip(),
        "language": language,
        "engine": f"API {model} via {base_url}",
    }


def transcribe(path: str, language: str | None = None,
               api_key: str = "", base_url: str = "",
               model: str = "whisper-1") -> dict[str, Any]:
    """Transcribe one audio file; returns ``{"text", "language", "engine"}``.

    Local whisper wins when installed; otherwise the API is used if a key
    was supplied (falls back to the ``SPEECHLAB_*`` environment variables).
    Raises ``RuntimeError`` when no transcription backend is available.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    local = _local_whisper(path, language)
    if local is not None:
        # A local backend ran: its answer is final.  Empty text is a
        # legitimate result (e.g. a silent recording) — falling through to
        # the remote API here would upload the audio to a third party the
        # user may have deliberately avoided by installing local whisper,
        # and would raise a confusing "no API key" error even though the
        # offline path succeeded.
        if not local["text"]:
            local = dict(local, note="no speech detected by the local model")
        return local

    api_key = api_key or os.environ.get("SPEECHLAB_API_KEY", "")
    base_url = base_url or os.environ.get(
        "SPEECHLAB_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError(
            "transcription unavailable: install `openai-whisper` or "
            "`faster-whisper` for offline ASR, or configure an API key "
            "(SPEECHLAB_API_KEY) to use a Whisper-compatible endpoint"
        )
    try:
        return _api_whisper(path, language, api_key, base_url, model)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"transcription endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"transcription endpoint unreachable: {exc}") from exc
