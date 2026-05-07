#!/usr/bin/env python3
"""Unit tests for the session-management helpers in mcp_server.py.

Self-contained: stubs out the third-party imports (mcp, openai) and the
Keychain lookup so the test can import mcp_server without needing the
full runtime environment. Uses only stdlib (unittest, tempfile).

Run:
    python3 test_session_mgmt.py

NOTE: Secret-shape fixtures are assembled at runtime from broken-up
parts so secret scanners (semgrep, gitleaks, trufflehog) do not flag
this test file as containing real credentials. Every fixture below is
synthetic.

The test class points ``SESSIONS_DIR`` at a per-test
``tempfile.TemporaryDirectory`` so the user's real
``~/.claude/sessions/`` is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SERVER_PATH = HERE / "mcp_server.py"

# --- Synthetic secret-shape fixtures (assembled at runtime) ---------------
_GOOG_API_KEY_PREFIX = "AI" + "za"
FAKE_GOOG_API_KEY = _GOOG_API_KEY_PREFIX + "SyDsyntheticTestValue1234567890_-end"


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    """Install minimal stand-ins for mcp.* and openai so importing
    mcp_server does not require those packages or hit the Keychain."""

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    _stub_module("openai", OpenAI=_FakeOpenAI)

    class _FakeServer:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

        def create_initialization_options(self):
            return None

        async def run(self, *a, **kw):
            return None

    async def _fake_stdio_server():
        yield None, None

    _stub_module("mcp")
    _stub_module("mcp.server", Server=_FakeServer)
    _stub_module("mcp.server.stdio", stdio_server=_fake_stdio_server)

    class _Tool:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _TextContent:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _stub_module("mcp.types", Tool=_Tool, TextContent=_TextContent)


def _load_mcp_server():
    _install_stubs()
    fake_proc = types.SimpleNamespace(returncode=0, stdout="dummy-key\n")
    with mock.patch("subprocess.run", return_value=fake_proc):
        spec = importlib.util.spec_from_file_location(
            "mcp_server_under_test_session", SERVER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


mcp_server = _load_mcp_server()


class _SessionMgmtBase(unittest.TestCase):
    """Repoint SESSIONS_DIR at a temp directory for the duration of each
    test so we do not pollute ~/.claude/sessions/."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        # Patch the module-level SESSIONS_DIR. Our helpers reference
        # mcp_server.SESSIONS_DIR via the module global, so a single
        # rebind is sufficient.
        self._orig_sessions_dir = mcp_server.SESSIONS_DIR
        mcp_server.SESSIONS_DIR = self.tmp_path

    def tearDown(self) -> None:
        mcp_server.SESSIONS_DIR = self._orig_sessions_dir
        self._tmp.cleanup()


class TestSaveSession(_SessionMgmtBase):
    def test_save_session_returns_success_and_id(self):
        result = mcp_server.save_session(
            name="My Test", messages=[{"role": "user", "content": "hello"}]
        )
        self.assertTrue(result["success"])
        self.assertIn("session_id", result)
        self.assertEqual(result["name"], "My Test")
        self.assertEqual(result["message_count"], 1)

    def test_save_session_writes_file_at_expected_path(self):
        result = mcp_server.save_session(
            name="path-check", messages=[{"role": "user", "content": "hi"}]
        )
        expected = self.tmp_path / f"{result['session_id']}.json"
        self.assertTrue(expected.exists(), f"missing file at {expected}")

        with open(expected) as f:
            data = json.load(f)
        self.assertEqual(data["session_id"], result["session_id"])
        self.assertEqual(data["name"], "path-check")
        self.assertEqual(len(data["messages"]), 1)
        self.assertIn("created_at", data)
        self.assertIn("last_modified", data)
        # created_at and last_modified should match on creation
        self.assertEqual(data["created_at"], data["last_modified"])

    def test_save_session_default_name(self):
        result = mcp_server.save_session(messages=[])
        self.assertEqual(result["name"], "Untitled")
        self.assertEqual(result["message_count"], 0)

    def test_save_session_redacts_secrets_in_messages(self):
        """SECURITY: secret-shape strings inside `messages` MUST be
        redacted before the session file lands on disk. This is the
        single most important invariant of the save_session change."""
        messages = [
            {"role": "user", "content": f"my key is {FAKE_GOOG_API_KEY}"},
            {"role": "assistant", "content": "ok"},
        ]
        result = mcp_server.save_session(name="leak-test", messages=messages)
        session_file = self.tmp_path / f"{result['session_id']}.json"

        # Read the raw bytes off disk - we want to confirm the secret
        # string never reached the filesystem in plaintext.
        on_disk = session_file.read_text()
        self.assertNotIn(FAKE_GOOG_API_KEY, on_disk)
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", on_disk)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", on_disk)

        # And the parsed structure should also have the redaction in
        # place (defence in depth - if the read-back path ever bypasses
        # raw-text parsing this still catches the leak).
        data = json.loads(on_disk)
        self.assertEqual(
            data["messages"][0]["content"],
            "my key is [REDACTED_GOOGLE_API_KEY]",
        )

    def test_save_session_redacts_secrets_in_metadata(self):
        """Metadata fields are also walked - dict-key redaction included
        so a secret-as-key cannot smuggle plaintext past the boundary."""
        result = mcp_server.save_session(
            name="meta-leak",
            messages=[],
            metadata={"note": f"see {FAKE_GOOG_API_KEY}", FAKE_GOOG_API_KEY: "v"},
        )
        session_file = self.tmp_path / f"{result['session_id']}.json"
        on_disk = session_file.read_text()
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", on_disk)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", on_disk)


class TestLoadSession(_SessionMgmtBase):
    def test_load_session_round_trip_preserves_fields(self):
        original_messages = [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        original_metadata = {"project": "math", "tags": ["arithmetic"]}
        save_result = mcp_server.save_session(
            name="round-trip",
            messages=original_messages,
            metadata=original_metadata,
        )

        loaded = mcp_server.load_session(save_result["session_id"])
        self.assertEqual(loaded["session_id"], save_result["session_id"])
        self.assertEqual(loaded["name"], "round-trip")
        self.assertEqual(loaded["messages"], original_messages)
        self.assertEqual(loaded["metadata"], original_metadata)
        self.assertIn("created_at", loaded)
        self.assertIn("last_modified", loaded)

    def test_load_session_raises_for_missing_id(self):
        with self.assertRaises(ValueError):
            mcp_server.load_session("nonexistent-uuid-xxxxxxxx")


class TestUpdateSession(_SessionMgmtBase):
    def test_update_session_changes_name_and_bumps_last_modified(self):
        save_result = mcp_server.save_session(
            name="before", messages=[{"role": "user", "content": "x"}]
        )
        original = mcp_server.load_session(save_result["session_id"])
        # Sleep enough for ISO-8601 microsecond resolution to advance
        # without making the test slow. 10ms is plenty.
        time.sleep(0.01)

        update_result = mcp_server.update_session(
            save_result["session_id"], name="after"
        )
        self.assertTrue(update_result["success"])
        self.assertEqual(update_result["name"], "after")

        reloaded = mcp_server.load_session(save_result["session_id"])
        self.assertEqual(reloaded["name"], "after")
        self.assertEqual(reloaded["created_at"], original["created_at"])
        self.assertNotEqual(reloaded["last_modified"], original["last_modified"])
        self.assertGreater(reloaded["last_modified"], original["last_modified"])

    def test_update_session_without_name_only_bumps_modified(self):
        save_result = mcp_server.save_session(name="keepme", messages=[])
        original = mcp_server.load_session(save_result["session_id"])
        time.sleep(0.01)

        mcp_server.update_session(save_result["session_id"])
        reloaded = mcp_server.load_session(save_result["session_id"])
        self.assertEqual(reloaded["name"], "keepme")
        self.assertGreater(reloaded["last_modified"], original["last_modified"])

    def test_update_session_raises_for_missing_id(self):
        with self.assertRaises(ValueError):
            mcp_server.update_session("nonexistent-uuid-xxxxxxxx", name="x")


class TestDeleteSession(_SessionMgmtBase):
    def test_delete_session_removes_file(self):
        save_result = mcp_server.save_session(name="doomed", messages=[])
        session_file = self.tmp_path / f"{save_result['session_id']}.json"
        self.assertTrue(session_file.exists())

        delete_result = mcp_server.delete_session(save_result["session_id"])
        self.assertTrue(delete_result["success"])
        self.assertFalse(session_file.exists())

    def test_delete_session_raises_for_missing_id(self):
        with self.assertRaises(ValueError):
            mcp_server.delete_session("nonexistent-uuid-xxxxxxxx")


class TestListSessions(_SessionMgmtBase):
    def test_list_sessions_empty_when_dir_empty(self):
        self.assertEqual(mcp_server.list_sessions(), [])

    def test_list_sessions_returns_most_recent_first(self):
        first = mcp_server.save_session(name="first", messages=[])
        time.sleep(0.01)
        second = mcp_server.save_session(name="second", messages=[])
        time.sleep(0.01)
        third = mcp_server.save_session(name="third", messages=[])

        listing = mcp_server.list_sessions()
        self.assertEqual(len(listing), 3)
        # third is newest, first is oldest
        self.assertEqual(listing[0]["session_id"], third["session_id"])
        self.assertEqual(listing[1]["session_id"], second["session_id"])
        self.assertEqual(listing[2]["session_id"], first["session_id"])

    def test_list_sessions_reflects_update_ordering(self):
        a = mcp_server.save_session(name="a", messages=[])
        time.sleep(0.01)
        b = mcp_server.save_session(name="b", messages=[])
        time.sleep(0.01)
        # Touch `a` so it becomes most-recent.
        mcp_server.update_session(a["session_id"], name="a-renamed")

        listing = mcp_server.list_sessions()
        self.assertEqual(listing[0]["session_id"], a["session_id"])
        self.assertEqual(listing[0]["name"], "a-renamed")
        self.assertEqual(listing[1]["session_id"], b["session_id"])

    def test_list_sessions_skips_corrupt_files(self):
        good = mcp_server.save_session(name="ok", messages=[])
        # Drop a junk file in the sessions dir to simulate corruption.
        (self.tmp_path / "bad.json").write_text("{not json")

        listing = mcp_server.list_sessions()
        # Only the good session shows up; the bad file is silently
        # skipped by the JSONDecodeError guard.
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["session_id"], good["session_id"])

    def test_list_sessions_reports_message_count(self):
        result = mcp_server.save_session(
            name="counted",
            messages=[
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "2"},
                {"role": "user", "content": "3"},
            ],
        )
        listing = mcp_server.list_sessions()
        match = next(s for s in listing if s["session_id"] == result["session_id"])
        self.assertEqual(match["message_count"], 3)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    sys.exit(0 if runner.run(suite).wasSuccessful() else 1)
