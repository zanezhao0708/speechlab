"""Chatbot-style web demo for SpeechLab Agent.

Run:

    export SPEECHLAB_API_KEY=sk-...   # or enter a key in the web UI
    python web/app.py                 # then open http://localhost:8000

Visitors can also configure their own API key / base URL / model in the
UI (stored only in their browser), and can run the local acoustic
analysis directly without any API key.

Optional environment variables:

* ``SPEECHLAB_WEB_PASSWORD`` — when set, all /api routes require the
  matching ``X-Auth-Token`` header (simple shared-password gate).
* ``SPEECHLAB_WEB_DB``       — SQLite path for chat history / measurement
  trends (default ``web/data/speechlab.db``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from flask import Flask, Response, jsonify, request, send_from_directory

if __package__ in (None, ""):  # executed directly: `python web/app.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechlab.agent import AgentConfig, SpeechResearchAgent
from speechlab.asr import transcribe as _asr_transcribe
from speechlab.audio import load_audio
from speechlab.features import analyze
from speechlab.features import compare_reports as _compare_reports
from speechlab.features import diarize as _diarize
from speechlab.features import reference_ranges as _reference_ranges
from web.store import Store

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ALLOWED_EXT = {".wav", ".wave", ".mp3", ".flac", ".ogg"}
MAX_SESSIONS = 64
SESSION_UPLOAD_QUOTA = 200 * 1024 * 1024   # total bytes per session
CHAT_RATE_LIMIT = (8, 60.0)                # max 8 chat calls per 60 s
CHAT_CONCURRENCY = 4                       # simultaneous LLM calls server-wide

#: shared-password gate: empty → auth disabled
WEB_PASSWORD = os.environ.get("SPEECHLAB_WEB_PASSWORD", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

#: conversations / messages / longitudinal measurements (SQLite)
STORE = Store(os.environ.get("SPEECHLAB_WEB_DB") or None)


@dataclass
class Session:
    """One browser conversation: an agent plus its uploaded audio files."""

    agent: SpeechResearchAgent | None = None
    config_key: tuple = ()
    upload_dir: str = field(
        default_factory=lambda: tempfile.mkdtemp(prefix="speechlab-web-")
    )
    files: dict[str, str] = field(default_factory=dict)  # stored name -> original name
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Serialises agent work; also lets eviction skip sessions in use."""
    uploaded_bytes: int = 0
    chat_times: deque = field(default_factory=deque)
    last_used: float = field(default_factory=time.time)


_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()
_CHAT_SEMAPHORE = threading.BoundedSemaphore(CHAT_CONCURRENCY)

# LRU cache of finished analyses keyed by (size, mtime_ns) so re-analysing
# the same file (chat tool call after a direct report, repeated questions)
# is instant.
_ANALYSIS_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 64


def _cached_analyze(path: str, contour: bool) -> dict:
    """Analyse with an LRU cache; always caches the full (contour) report."""
    with open(path, "rb") as fh:
        key = hashlib.md5(fh.read()).hexdigest()
    with _LOCK:
        hit = _ANALYSIS_CACHE.get(key)
        if hit is not None:
            _ANALYSIS_CACHE.move_to_end(key)
            full = hit
        else:
            full = None
    if full is None:
        full = analyze(load_audio(path), contour=True)
        with _LOCK:
            _ANALYSIS_CACHE[key] = full
            while len(_ANALYSIS_CACHE) > _CACHE_MAX:
                _ANALYSIS_CACHE.popitem(last=False)
    if not contour:
        full = {k: v for k, v in full.items()
                if k not in ("pitch_contour", "spectrogram")}
    return full


def _get_session(session_id: str | None) -> tuple[str, Session]:
    with _LOCK:
        if session_id and session_id in _SESSIONS:
            sess = _SESSIONS[session_id]
            sess.last_used = time.time()
            return session_id, sess
        sid = uuid.uuid4().hex
        _SESSIONS[sid] = Session()
        if len(_SESSIONS) > MAX_SESSIONS:  # evict least-recently-used idle sessions
            candidates = sorted(_SESSIONS.items(), key=lambda kv: kv[1].last_used)
            need = len(_SESSIONS) - MAX_SESSIONS
            for old_sid, old_sess in candidates:
                if need <= 0:
                    break
                if old_sess.lock.acquire(blocking=False):  # in use right now?
                    try:
                        shutil.rmtree(old_sess.upload_dir, ignore_errors=True)
                        del _SESSIONS[old_sid]
                        need -= 1
                    finally:
                        old_sess.lock.release()
        return sid, _SESSIONS[sid]


def _json_safe(obj):
    """Replace NaN/Inf with None so the payload is strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def _resolve_upload(upload_dir: str, raw: str) -> str:
    """Map a model-supplied path onto one file inside the session's upload dir."""
    path = os.path.join(upload_dir, os.path.basename(raw))
    if not os.path.isfile(path):
        available = sorted(os.listdir(upload_dir)) or ["(none)"]
        raise FileNotFoundError(
            f"'{raw}' is not an uploaded file; available files: {', '.join(available)}"
        )
    return path


def _make_tools(upload_dir: str, api_key: str = "", base_url: str = "") -> dict:
    """Agent tools restricted to this session's uploaded files."""

    def analyze_audio(args: dict) -> dict:
        return _cached_analyze(_resolve_upload(upload_dir, args.get("path", "")),
                               contour=False)

    def compare_audio(args: dict) -> dict:
        ra = _cached_analyze(_resolve_upload(upload_dir, args.get("path_a", "")),
                             contour=True)
        rb = _cached_analyze(_resolve_upload(upload_dir, args.get("path_b", "")),
                             contour=True)
        out = _compare_reports(ra, rb)
        out["files"] = [args.get("path_a"), args.get("path_b")]
        return out

    def diarize_audio(args: dict) -> dict:
        return _diarize(load_audio(_resolve_upload(upload_dir, args.get("path", ""))),
                        n_speakers=int(args.get("n_speakers") or 0))

    def transcribe_audio(args: dict) -> dict:
        return _asr_transcribe(
            _resolve_upload(upload_dir, args.get("path", "")),
            language=args.get("language"),
            api_key=api_key, base_url=base_url)

    def reference_ranges(args: dict) -> dict:
        return _reference_ranges(args.get("metric", ""))

    return {
        "analyze_audio": analyze_audio,
        "compare_audio": compare_audio,
        "diarize_audio": diarize_audio,
        "transcribe_audio": transcribe_audio,
        "reference_ranges": reference_ranges,
    }


@app.before_request
def _auth_gate():
    """When SPEECHLAB_WEB_PASSWORD is set, guard every /api route."""
    if not WEB_PASSWORD or not request.path.startswith("/api/"):
        return None
    if request.path == "/api/config":  # must announce auth_required to the UI
        return None
    token = request.headers.get("X-Auth-Token", "")
    if hmac.compare_digest(token, WEB_PASSWORD):
        return None
    return jsonify(error="需要访问密码", auth_required=True), 401


def _user_key() -> str:
    """Anonymous device/user identity supplied by the browser."""
    key = (request.headers.get("X-User-Key") or "").strip()
    return key[:64]


def _get_or_build_agent(sess: Session, body: dict) -> SpeechResearchAgent:
    api_key = (body.get("api_key") or os.environ.get("SPEECHLAB_API_KEY", "")).strip()
    base_url = (
        body.get("base_url")
        or os.environ.get("SPEECHLAB_BASE_URL", "https://api.openai.com/v1")
    ).strip()
    model = (body.get("model") or os.environ.get("SPEECHLAB_MODEL", "gpt-4o-mini")).strip()
    if not api_key:
        raise RuntimeError(
            "缺少 API Key：请点击页面右上角「设置」填写，或在服务器上设置 "
            "SPEECHLAB_API_KEY 环境变量"
        )
    key = (api_key, base_url, model)
    if sess.agent is None or sess.config_key != key:
        sess.agent = SpeechResearchAgent(
            AgentConfig(api_key=api_key, base_url=base_url, model=model),
            tools=_make_tools(sess.upload_dir, api_key=api_key, base_url=base_url),
        )
        sess.config_key = key
    return sess.agent


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/config")
def config():
    """Tell the UI whether the server already has an API key configured."""
    return jsonify(
        has_server_key=bool(os.environ.get("SPEECHLAB_API_KEY")),
        base_url=os.environ.get("SPEECHLAB_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("SPEECHLAB_MODEL", "gpt-4o-mini"),
        auth_required=bool(WEB_PASSWORD),
        tools=["analyze_audio", "compare_audio", "diarize_audio",
               "transcribe_audio", "reference_ranges"],
    )


@app.post("/api/upload")
def upload():
    sid, sess = _get_session(request.form.get("session_id"))
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="没有收到文件"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"不支持的格式 {ext or '(无扩展名)'}，请上传 wav/mp3/flac/ogg"), 400
    f.stream.seek(0, 2)
    size = f.stream.tell()
    f.stream.seek(0)
    with sess.lock:
        if sess.uploaded_bytes + size > SESSION_UPLOAD_QUOTA:
            return jsonify(error="本会话上传总量已达上限（200 MB），请开新对话"), 413
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(f.filename))
        stored = f"{uuid.uuid4().hex[:8]}_{safe}"
        f.save(os.path.join(sess.upload_dir, stored))
        sess.files[stored] = f.filename
        sess.uploaded_bytes += size
    return jsonify(session_id=sid, stored=stored, original=f.filename)


def _chat_prelude(body: dict):
    """Shared validation + rate limit + agent setup for chat endpoints.

    Returns ``(error, sid, sess, agent, message, names)`` where ``error`` is
    a ``(response, status)`` tuple ready to return, or ``None``.
    """
    message = (body.get("message") or "").strip()
    attached = body.get("attached") or []
    sid, sess = _get_session(body.get("session_id"))
    if not message and not attached:
        return (jsonify(error="消息为空"), 400), sid, sess, None, message, []

    max_calls, window_s = CHAT_RATE_LIMIT
    now = time.time()
    with sess.lock:
        while sess.chat_times and now - sess.chat_times[0] > window_s:
            sess.chat_times.popleft()
        if len(sess.chat_times) >= max_calls:
            return (jsonify(error="请求太频繁，请稍后再试"), 429), sid, sess, None, message, []
        sess.chat_times.append(now)

    try:
        with sess.lock:
            agent = _get_or_build_agent(sess, body)
    except RuntimeError as exc:
        return (jsonify(error=str(exc)), 400), sid, sess, None, message, []

    names = [n for n in attached if n in sess.files]
    if names:
        listing = "\n".join(
            f"- {n} (original name: {sess.files[n]})" for n in names
        )
        message += (
            "\n\n[The user attached these audio files, already uploaded to the "
            f"server — call analyze_audio on them as needed]:\n{listing}"
        )
    if not message:
        message = "请分析我上传的音频。"
    return None, sid, sess, agent, message, names


def _persist_exchange(user_key: str, conv_id: str | None,
                      question: str, answer: str, files: list[str]) -> str | None:
    """Save one user/assistant exchange to the store (best effort)."""
    if not user_key:
        return conv_id
    try:
        if not conv_id or STORE.conversation_owner(conv_id) != user_key:
            conv_id = STORE.create_conversation(user_key, title=question[:60])
        STORE.add_message(conv_id, "user", question, files)
        STORE.add_message(conv_id, "assistant", answer)
    except Exception:  # noqa: BLE001, S110 — persistence must never break chat
        pass
    return conv_id


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    error, sid, sess, agent, message, names = _chat_prelude(body)
    if error:
        return error

    question = (body.get("message") or "").strip() or "请分析我上传的音频。"
    original_files = [sess.files[n] for n in names if n in sess.files]
    try:
        with sess.lock, _CHAT_SEMAPHORE:
            answer = agent.ask(message)
    except Exception as exc:  # noqa: BLE001 — surface as a chat error
        return jsonify(error=f"请求失败：{exc}"), 502
    conv_id = _persist_exchange(_user_key(), body.get("conversation_id"),
                                question, answer, original_files)
    return jsonify(session_id=sid, answer=answer, conversation_id=conv_id)


@app.post("/api/chat/stream")
def chat_stream():
    """SSE variant of /api/chat: streams answer deltas as they arrive.

    Event payload (one JSON object per ``data:`` frame): ``meta`` first
    (session + conversation ids), then ``delta``/``tool`` events, ending
    with ``done`` (or ``error``).
    """
    body = request.get_json(force=True)
    error, sid, sess, agent, message, names = _chat_prelude(body)
    if error:
        return error

    user_key = _user_key()
    conv_id = body.get("conversation_id") or ""
    question = (body.get("message") or "").strip() or "请分析我上传的音频。"
    fresh_conv = False
    if user_key and (not conv_id or STORE.conversation_owner(conv_id) != user_key):
        conv_id = STORE.create_conversation(user_key, title=question[:60])
        fresh_conv = True
    original_files = [sess.files[n] for n in names if n in sess.files]

    def sse(obj: dict) -> str:
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    def _drop_fresh():
        """A failed first exchange must not leave an empty conversation row."""
        if fresh_conv:
            try:
                STORE.delete_conversation(conv_id)
            except Exception:  # noqa: BLE001, S110
                pass

    def generate():
        yield sse({"type": "meta", "session_id": sid, "conversation_id": conv_id})
        answer = ""
        ok = False
        try:
            with sess.lock, _CHAT_SEMAPHORE:
                for ev in agent.ask_stream(message):
                    if ev["type"] == "delta":
                        answer += ev.get("text", "")
                    elif ev["type"] == "done":
                        answer = ev.get("answer", answer)
                        ok = True
                    elif ev["type"] == "error":
                        ok = False
                    yield sse(ev)
        except Exception as exc:  # noqa: BLE001 — report as SSE error event
            yield sse({"type": "error", "error": f"请求失败：{exc}"})
            _drop_fresh()
            return
        if ok and user_key:
            _persist_exchange(user_key, conv_id, question, answer, original_files)
        elif not ok:
            _drop_fresh()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/analyze")
def analyze_direct():
    """Direct acoustic analysis — no LLM / API key needed."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="没有收到文件"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"不支持的格式 {ext or '(无扩展名)'}，请上传 wav/mp3/flac/ogg"), 400
    tmpdir = tempfile.mkdtemp(prefix="speechlab-direct-")
    try:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(f.filename))
        path = os.path.join(tmpdir, safe)
        f.save(path)
        try:
            report = _cached_analyze(path, contour=True)
        except Exception as exc:  # noqa: BLE001 — report file/decoding problems
            return jsonify(error=f"分析失败：{exc}"), 400
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    report = dict(report)  # copy: cached dict is shared
    report["file"] = f.filename
    # longitudinal tracking: store the headline metrics for this user
    user_key = _user_key()
    if user_key:
        try:
            STORE.add_measurement(
                user_key, f.filename,
                (report.get("pitch") or {}).get("f0_median_hz"),
                (report.get("voice_quality") or {}).get("jitter_local_percent"),
                (report.get("voice_quality") or {}).get("shimmer_local_db"),
                report.get("hnr_db"))
        except Exception:  # noqa: BLE001, S110 — persistence must never break analysis
            pass
    return jsonify(_json_safe(report))


# ------------------------------------------------------------- history sync
@app.get("/api/history")
def history():
    """List this user's conversations, or fetch one conversation's messages."""
    user_key = _user_key()
    if not user_key:
        return jsonify(conversations=[], messages=[])
    conv_id = request.args.get("conversation_id", "")
    if conv_id:
        if STORE.conversation_owner(conv_id) != user_key:
            return jsonify(error="conversation not found"), 404
        return jsonify(conversation_id=conv_id,
                       messages=STORE.get_messages(conv_id))
    return jsonify(conversations=STORE.list_conversations(user_key))


@app.delete("/api/history")
def history_delete():
    """Delete one stored conversation."""
    user_key = _user_key()
    conv_id = (request.get_json(force=True) or {}).get("conversation_id", "")
    if not conv_id or STORE.conversation_owner(conv_id) != user_key:
        return jsonify(error="conversation not found"), 404
    STORE.delete_conversation(conv_id)
    return jsonify(ok=True)


# -------------------------------------------------------------- trends sync
@app.route("/api/trends", methods=["GET", "POST", "DELETE"])
def trends():
    """Server-side measurement history — the cross-device part of 📈 趋势."""
    user_key = _user_key()
    if not user_key:
        return jsonify(measurements=[])
    if request.method == "GET":
        return jsonify(measurements=STORE.get_measurements(user_key))
    if request.method == "DELETE":
        STORE.clear_measurements(user_key)
        return jsonify(ok=True)
    body = request.get_json(force=True)
    row = body if isinstance(body, dict) else {}
    STORE.add_measurement(
        user_key, str(row.get("file", ""))[:200],
        row.get("f0"), row.get("jitter"), row.get("shimmer"), row.get("hnr"),
        ts=row.get("ts"))
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    body = request.get_json(force=True)
    sid, sess = _get_session(body.get("session_id"))
    with sess.lock:
        sess.agent = None
        sess.config_key = ()
        sess.files.clear()
        sess.uploaded_bytes = 0
        sess.chat_times.clear()
        for name in os.listdir(sess.upload_dir):
            os.remove(os.path.join(sess.upload_dir, name))
    return jsonify(session_id=sid, ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"SpeechLab web demo: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
