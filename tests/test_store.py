"""Tests for the SQLite persistence layer (web/store.py)."""

import time

import pytest

from web.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()


def test_conversation_lifecycle(store):
    cid = store.create_conversation("user1", title="first chat")
    assert store.conversation_owner(cid) == "user1"
    assert store.conversation_owner("nope") is None

    store.add_message(cid, "user", "hello", ["a.wav"])
    store.add_message(cid, "assistant", "hi there")
    msgs = store.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["files"] == ["a.wav"]
    assert msgs[1]["content"] == "hi there"

    store.delete_conversation(cid)
    assert store.get_messages(cid) == []
    assert store.conversation_owner(cid) is None


def test_list_conversations_scoped_to_user_and_ordered(store):
    c1 = store.create_conversation("alice", "old")
    time.sleep(0.01)
    c2 = store.create_conversation("alice", "new")
    store.create_conversation("bob", "other")

    convs = store.list_conversations("alice")
    assert [c["id"] for c in convs] == [c2, c1]  # newest first
    assert all(c["id"] != "bob" for c in convs)


def test_touch_conversation_renames(store):
    cid = store.create_conversation("u", "before")
    store.touch_conversation(cid, title="after")
    convs = store.list_conversations("u")
    assert convs[0]["title"] == "after"
    # unknown id is a no-op, not an error
    store.touch_conversation("missing", title="x")


def test_measurements_roundtrip(store):
    store.add_measurement("u", "a.wav", 220.0, 0.5, 0.2, 21.0, ts=1000.0)
    store.add_measurement("u", "b.wav", 180.0, 0.8, 0.3, 18.0, ts=1001.0)
    store.add_measurement("other", "c.wav", 100.0, None, None, None, ts=1002.0)

    rows = store.get_measurements("u")
    assert len(rows) == 2
    assert rows[0]["f0"] == 220.0
    assert rows[0]["file"] == "a.wav"
    # NULL metrics survive as None
    store.clear_measurements("u")
    assert store.get_measurements("u") == []
    assert len(store.get_measurements("other")) == 1


def test_message_files_json_corruption_tolerated(store):
    cid = store.create_conversation("u")
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, files, created_at) "
            "VALUES (?, 'user', 'x', 'not-json', 0)", (cid,))
    msgs = store.get_messages(cid)
    assert msgs[0]["files"] == []
