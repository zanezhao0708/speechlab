"""SQLite persistence for the SpeechLab web app.

Three tables back the cross-device features:

- ``conversations`` / ``messages`` — chat history, restorable in any browser
  that presents the same user key;
- ``measurements`` — acoustic measurements (F0 / jitter / shimmer / HNR)
  for longitudinal trend tracking across devices.

The store is intentionally tiny: one connection guarded by a lock (SQLite
serialises writers anyway) and plain SQL helpers, no ORM.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

__all__ = ["Store"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_key   TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    files           TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE TABLE IF NOT EXISTS measurements (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    ts       REAL NOT NULL,
    file     TEXT NOT NULL DEFAULT '',
    f0       REAL, jitter REAL, shimmer REAL, hnr REAL
);
CREATE INDEX IF NOT EXISTS idx_meas_user ON measurements(user_key, ts);
"""


def default_db_path() -> str:
    env = os.environ.get("SPEECHLAB_WEB_DB")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "speechlab.db")


class Store:
    """Thread-safe SQLite persistence (conversations / messages / trends)."""

    def __init__(self, path: str | None = None):
        self.path = path or default_db_path()
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------- conversations
    def create_conversation(self, user_key: str, title: str = "") -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO conversations (id, user_key, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)", (conv_id, user_key, title[:80], now, now))
        return conv_id

    def touch_conversation(self, conv_id: str, title: str | None = None) -> None:
        """Bump updated_at (and optionally rename) if the conversation exists."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if row is None:
                return
            if title:
                self._conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title[:80], time.time(), conv_id))
            else:
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (time.time(), conv_id))

    def list_conversations(self, user_key: str, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE user_key = ? ORDER BY updated_at DESC LIMIT ?",
                (user_key, limit)).fetchall()
        return [dict(r) for r in rows]

    def conversation_owner(self, conv_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_key FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return row["user_key"] if row else None

    def delete_conversation(self, conv_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM messages WHERE conversation_id = ?",
                               (conv_id,))
            self._conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    # ------------------------------------------------------------ messages
    def add_message(self, conv_id: str, role: str, content: str,
                    files: list[str] | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, files, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, json.dumps(files or []), time.time()))
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conv_id))

    def get_messages(self, conv_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, files, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id LIMIT ?",
                (conv_id, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["files"] = json.loads(d.get("files") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["files"] = []
            out.append(d)
        return out

    # -------------------------------------------------------- measurements
    def add_measurement(self, user_key: str, file: str, f0: float | None,
                        jitter: float | None, shimmer: float | None,
                        hnr: float | None, ts: float | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO measurements (user_key, ts, file, f0, jitter, shimmer, hnr) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_key, ts or time.time(), file, f0, jitter, shimmer, hnr))

    def get_measurements(self, user_key: str, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, file, f0, jitter, shimmer, hnr FROM measurements "
                "WHERE user_key = ? ORDER BY ts LIMIT ?",
                (user_key, limit)).fetchall()
        return [dict(r) for r in rows]

    def clear_measurements(self, user_key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM measurements WHERE user_key = ?",
                               (user_key,))
