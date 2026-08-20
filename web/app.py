"""Chatbot-style web demo for SpeechLab Agent.

Run:

    export SPEECHLAB_API_KEY=sk-...   # or enter a key in the web UI
    python web/app.py                 # then open http://localhost:8000

Visitors can also configure their own API key / base URL / model in the
UI (stored only in their browser), and can run the local acoustic
analysis directly without any API key.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field

from flask import Flask, jsonify, request, send_from_directory

from speechlab.agent import AgentConfig, SpeechResearchAgent
from speechlab.audio import load_audio
from speechlab.features import analyze

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ALLOWED_EXT = {".wav", ".wave", ".mp3", ".flac", ".ogg"}
MAX_SESSIONS = 64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads


@dataclass
class Session:
    """One browser conversation: an agent plus its uploaded audio files."""

    agent: SpeechResearchAgent | None = None
    config_key: tuple = ()
    upload_dir: str = field(
        default_factory=lambda: tempfile.mkdtemp(prefix="speechlab-web-")
    )
    files: dict[str, str] = field(default_factory=dict)  # stored name -> original name


_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()


def _get_session(session_id: str | None) -> tuple[str, Session]:
    with _LOCK:
        if session_id and session_id in _SESSIONS:
            return session_id, _SESSIONS[session_id]
        sid = uuid.uuid4().hex
        _SESSIONS[sid] = Session()
        while len(_SESSIONS) > MAX_SESSIONS:  # evict oldest sessions
            oldest, sess = next(iter(_SESSIONS.items()))
            shutil.rmtree(sess.upload_dir, ignore_errors=True)
            del _SESSIONS[oldest]
        return sid, _SESSIONS[sid]


def _make_tools(upload_dir: str) -> dict:
    """analyze_audio restricted to this session's uploaded files."""

    def analyze_audio(args: dict) -> dict:
        raw = args.get("path", "")
        path = os.path.join(upload_dir, os.path.basename(raw))
        if not os.path.isfile(path):
            available = sorted(os.listdir(upload_dir)) or ["(none)"]
            raise FileNotFoundError(
                f"'{raw}' is not an uploaded file; available files: {', '.join(available)}"
            )
        return analyze(load_audio(path))

    return {"analyze_audio": analyze_audio}


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
            tools=_make_tools(sess.upload_dir),
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
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(f.filename))
    stored = f"{uuid.uuid4().hex[:8]}_{safe}"
    f.save(os.path.join(sess.upload_dir, stored))
    sess.files[stored] = f.filename
    return jsonify(session_id=sid, stored=stored, original=f.filename)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    message = (body.get("message") or "").strip()
    attached = body.get("attached") or []
    sid, sess = _get_session(body.get("session_id"))
    if not message and not attached:
        return jsonify(error="消息为空"), 400
    try:
        agent = _get_or_build_agent(sess, body)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400

    # List files attached to this turn so the model knows what it can analyse.
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

    try:
        answer = agent.ask(message)
    except Exception as exc:  # noqa: BLE001 — surface as a chat error
        return jsonify(error=f"请求失败：{exc}"), 502
    return jsonify(session_id=sid, answer=answer)


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
            report = analyze(load_audio(path))
        except Exception as exc:  # noqa: BLE001 — report file/decoding problems
            return jsonify(error=f"分析失败：{exc}"), 400
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    report["file"] = f.filename
    return jsonify(report)


@app.post("/api/reset")
def reset():
    body = request.get_json(force=True)
    sid, sess = _get_session(body.get("session_id"))
    sess.agent = None
    sess.config_key = ()
    sess.files.clear()
    for name in os.listdir(sess.upload_dir):
        os.remove(os.path.join(sess.upload_dir, name))
    return jsonify(session_id=sid, ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"SpeechLab web demo: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
