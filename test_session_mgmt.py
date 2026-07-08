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

import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
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


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """Return the dict of fake mcp/openai/httpx/google.auth modules used
    during import. Caller is expected to scope these via
    mock.patch.dict(sys.modules) rather than mutating sys.modules
    directly (per PR #4 review, CodeRabbit Major: test stubs must not
    leak into other tests' sys.modules entries — would break unittest
    discover ordering and mask missing imports in sibling test files).

    google.auth.* fakes were added alongside the PR #8 test_redact.py
    scoping refactor: alphabetical pytest collection was previously
    running test_redact.py first, leaking its google.auth stubs into
    sys.modules and satisfying this file's `import google.auth` by
    accident. With the leak fixed, this file needs its own fakes.
    """

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    # httpx is imported at module level by mcp_server.py for the Gemini
    # Deep Research HTTP client. Tests don't exercise that code path
    # (Gemini helpers are mock.patch.object'd in test_redact.py), but
    # the bare `import httpx` still has to resolve.
    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

    class _FakeHTTPStatusError(Exception):
        pass

    class _FakeRequestError(Exception):
        pass

    class _FakeRequestsException(Exception):
        pass

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

    class _Tool:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _TextContent:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    # Fakes for google.auth ADC plumbing. mcp_server.py imports
    # `google.auth` and `google.auth.transport.requests` at module level
    # and uses dotted access like `google.auth.default(...)` and
    # `google.auth.exceptions.DefaultCredentialsError`. Tests never
    # invoke the real ADC path.
    class _FakeCredentials:
        valid = True
        token = "fake-bearer-token-for-tests"

        def refresh(self, request):
            self.token = "fake-bearer-token-for-tests"

    def _fake_default(scopes=None):
        return _FakeCredentials(), "fake-test-project"

    class _FakeDefaultCredentialsError(Exception):
        pass

    class _FakeRequest:
        def __init__(self, *a, **kw):
            pass

    def _make(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    # Build the google.auth chain with parent/child attribute wiring so
    # dotted access like `google.auth.exceptions.X` resolves after an
    # `import google.auth` statement.
    google_mod = _make("google")
    auth_exceptions_mod = _make(
        "google.auth.exceptions",
        DefaultCredentialsError=_FakeDefaultCredentialsError,
    )
    auth_mod = _make(
        "google.auth", default=_fake_default, exceptions=auth_exceptions_mod
    )
    transport_mod = _make("google.auth.transport")
    transport_requests_mod = _make(
        "google.auth.transport.requests", Request=_FakeRequest
    )
    google_mod.auth = auth_mod
    auth_mod.transport = transport_mod
    transport_mod.requests = transport_requests_mod

    return {
        "openai": _make("openai", OpenAI=_FakeOpenAI),
        "mcp": _make("mcp"),
        "mcp.server": _make("mcp.server", Server=_FakeServer),
        "mcp.server.stdio": _make("mcp.server.stdio", stdio_server=_fake_stdio_server),
        "mcp.types": _make("mcp.types", Tool=_Tool, TextContent=_TextContent),
        "httpx": _make(
            "httpx",
            AsyncClient=_FakeAsyncClient,
            HTTPStatusError=_FakeHTTPStatusError,
            RequestError=_FakeRequestError,
        ),
        "requests": _make(
            "requests",
            RequestException=_FakeRequestsException,
        ),
        "google": google_mod,
        "google.auth": auth_mod,
        "google.auth.exceptions": auth_exceptions_mod,
        "google.auth.transport": transport_mod,
        "google.auth.transport.requests": transport_requests_mod,
    }


def _load_mcp_server():
    """Import mcp_server.py with mcp.*/openai stubbed via a scoped
    sys.modules patch so the fakes don't leak into later test imports."""
    stubs = _build_stub_modules()
    fake_proc = types.SimpleNamespace(returncode=0, stdout="dummy-key\n")
    with mock.patch.dict(sys.modules, stubs):
        with mock.patch("subprocess.run", return_value=fake_proc):
            # DO NOT CONSOLIDATE this name with test_redact.py's. Today
            # the loaded module is bound to a local variable and the spec
            # name is not registered in sys.modules (manual exec_module
            # + scoped patch.dict), so collisions would be harmless. But
            # three plausible future changes would make them dangerous:
            # (a) adopting the docs' canonical
            # `sys.modules[spec.name] = module` pattern for circular-
            # import support, (b) weakening or removing the patch.dict
            # scope above, (c) switching to a loader that auto-registers.
            # Unique per-file suffixes cost nothing, surface in tracebacks
            # so you can tell which test loaded which copy, and pre-empt
            # all three regressions. Keep them distinct.
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
            mcp_server.load_session("00000000-0000-0000-0000-000000000000")


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
            mcp_server.update_session("00000000-0000-0000-0000-000000000000", name="x")


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
            mcp_server.delete_session("00000000-0000-0000-0000-000000000000")


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


class TestSessionIdValidation(_SessionMgmtBase):
    """PR #3 review (Codex P1): get_session_file must reject non-UUID
    inputs. Without this, MCP-callable load/update/delete tools could be
    pointed at arbitrary .json files on disk via a malicious session_id.
    """

    def test_rejects_path_traversal_attempts(self):
        bad_ids = [
            "/tmp/victim",
            "../../../etc/passwd",
            "..\\..\\windows\\config",
            "; rm -rf /",
            "../foo",
            "name with spaces",
            "",
            "not-a-uuid",
            "session-with-dot.in.the.middle",
        ]
        for bad_id in bad_ids:
            with self.assertRaises(
                ValueError, msg=f"should reject session_id {bad_id!r}"
            ):
                mcp_server.get_session_file(bad_id)

    def test_accepts_valid_uuid(self):
        """Sanity: a real uuid4 string should pass validation and yield
        a path INSIDE SESSIONS_DIR."""
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        path = mcp_server.get_session_file(sid)
        self.assertEqual(path.parent, mcp_server.SESSIONS_DIR)
        self.assertEqual(path.name, f"{sid}.json")

    def test_load_session_rejects_path_traversal(self):
        """End-to-end: an MCP caller should not be able to drive
        load_session into reading /tmp/victim.json or similar."""
        with self.assertRaises(ValueError):
            mcp_server.load_session("/tmp/victim")

    def test_delete_session_rejects_path_traversal(self):
        """End-to-end: delete_session must not unlink arbitrary files."""
        with self.assertRaises(ValueError):
            mcp_server.delete_session("../../../etc/passwd")

    def test_update_session_rejects_path_traversal(self):
        """End-to-end: update_session must not overwrite arbitrary files."""
        with self.assertRaises(ValueError):
            mcp_server.update_session("/etc/hosts", name="hijack")


class TestNameRedaction(_SessionMgmtBase):
    """PR #3 review (Codex P2): the ``name`` field must go through
    redact_secrets before persisting, matching messages/metadata."""

    def test_save_session_redacts_name(self):
        name_with_secret = f"My API key is {FAKE_GOOG_API_KEY}"
        result = mcp_server.save_session(
            name=name_with_secret,
            messages=[{"role": "user", "content": "x"}],
        )
        sid = result["session_id"]
        on_disk = (self.tmp_path / f"{sid}.json").read_text()
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", on_disk)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", on_disk)
        # Returned name reflects the redacted value too.
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", result["name"])
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", result["name"])

    def test_update_session_redacts_name(self):
        save = mcp_server.save_session(name="initial", messages=[])
        sid = save["session_id"]
        mcp_server.update_session(sid, name=f"new {FAKE_GOOG_API_KEY}")
        on_disk = (self.tmp_path / f"{sid}.json").read_text()
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", on_disk)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", on_disk)


class TestRobustness(_SessionMgmtBase):
    """PR #3 follow-up review (Gemini medium): error-handling + lazy mkdir."""

    def test_load_session_handles_corrupted_json(self):
        """Corrupted JSON surfaces as a clean ValueError, not raw JSONDecodeError."""
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        (self.tmp_path / f"{sid}.json").write_text("{not valid json")
        with self.assertRaises(ValueError) as ctx:
            mcp_server.load_session(sid)
        self.assertIn("invalid", str(ctx.exception).lower())

    def test_update_session_handles_corrupted_json(self):
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        (self.tmp_path / f"{sid}.json").write_text("not-json-either")
        with self.assertRaises(ValueError) as ctx:
            mcp_server.update_session(sid, name="x")
        self.assertIn("invalid", str(ctx.exception).lower())

    def test_save_session_creates_sessions_dir_lazily(self):
        """Module-level mkdir was removed for test isolation; save_session
        must create the directory on demand."""
        import tempfile

        with tempfile.TemporaryDirectory() as fresh_root:
            target = Path(fresh_root) / "fresh-sessions"
            self.assertFalse(target.exists())
            mcp_server.SESSIONS_DIR = target
            try:
                result = mcp_server.save_session(name="x", messages=[])
                self.assertTrue(target.exists())
                self.assertTrue((target / f"{result['session_id']}.json").exists())
            finally:
                mcp_server.SESSIONS_DIR = self.tmp_path

    def test_list_sessions_helper_returns_raw_name(self):
        """list_sessions() helper preserves the raw session name verbatim;
        the Markdown-table pipe escape happens in the call_tool rendering
        path. This test guards that the helper itself does not pre-escape."""
        original_name = "session | with | pipes"
        mcp_server.save_session(name=original_name, messages=[])
        listed = mcp_server.list_sessions()
        self.assertEqual(listed[0]["name"], original_name)

    def test_list_sessions_skips_non_object_json(self):
        """PR #3 follow-up review (Gemini medium + Codex P3): if a session
        file is valid JSON but not an object (e.g. "[]"), the .get(...)
        call would raise AttributeError. Verify list_sessions skips such
        files cleanly instead of failing the entire listing.

        NOTE: filenames must be valid UUID stems so they pass the
        round-7 UUID-filter check (mcp_server.py L271-L283); otherwise
        these files exit at the UUID gate and never reach the
        ``isinstance(data, dict)`` branch this test is meant to
        exercise (per PR #4 round-7 review, Gemini medium L475).
        """
        # Drop a list-shaped JSON file in the sessions dir; UUID stem
        # so the round-7 UUID filter doesn't short-circuit it.
        list_sid = "00000000-0000-0000-0000-000000000002"
        (self.tmp_path / f"{list_sid}.json").write_text("[]")
        # Drop a string-shaped JSON file.
        string_sid = "00000000-0000-0000-0000-000000000003"
        (self.tmp_path / f"{string_sid}.json").write_text('"hello"')
        # And a valid session for contrast
        good = mcp_server.save_session(name="ok", messages=[])

        listing = mcp_server.list_sessions()
        # Only the good session shows up
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["session_id"], good["session_id"])

    def test_list_sessions_skips_non_list_messages(self):
        """If messages is present but not a list (malformed file),
        len() raises TypeError. Skip the file, don't crash."""
        sid = "00000000-0000-0000-0000-000000000001"
        (self.tmp_path / f"{sid}.json").write_text(
            '{"name": "broken", "messages": "not-a-list"}'
        )
        good = mcp_server.save_session(name="ok", messages=[])

        listing = mcp_server.list_sessions()
        # Only good shows up
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["session_id"], good["session_id"])

    def test_list_sessions_handles_null_name(self):
        """PR #3 follow-up review (Gemini L196): JSON null round-trips
        to Python None. Use `or` fallback."""
        sid = "00000000-0000-0000-0000-000000000010"
        (self.tmp_path / f"{sid}.json").write_text(
            '{"name": null, "messages": [], "created_at": "x", "last_modified": "y"}'
        )
        listing = mcp_server.list_sessions()
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["name"], "Untitled")

    def test_load_session_rejects_non_dict_json(self):
        """PR #3 follow-up review (Gemini L275): list-shaped session file
        crashes .get() with AttributeError. Surface as ValueError."""
        sid = "00000000-0000-0000-0000-000000000020"
        (self.tmp_path / f"{sid}.json").write_text("[]")
        with self.assertRaises(ValueError) as ctx:
            mcp_server.load_session(sid)
        self.assertIn("not a JSON object", str(ctx.exception))

    def test_load_session_handles_null_fields(self):
        """PR #3 follow-up review (Gemini L284): JSON null fields round-trip
        to usable defaults, not None."""
        sid = "00000000-0000-0000-0000-000000000030"
        (self.tmp_path / f"{sid}.json").write_text(
            '{"session_id": "' + sid + '", "name": null, "messages": null, '
            '"metadata": null, "created_at": "x", "last_modified": "y"}'
        )
        loaded = mcp_server.load_session(sid)
        self.assertEqual(loaded["name"], "Untitled")
        self.assertEqual(loaded["messages"], [])
        self.assertEqual(loaded["metadata"], {})

    def test_update_session_rejects_non_dict_json(self):
        """PR #3 follow-up review (Gemini L301): same shape guard for update_session."""
        sid = "00000000-0000-0000-0000-000000000040"
        (self.tmp_path / f"{sid}.json").write_text('"a-string"')
        with self.assertRaises(ValueError) as ctx:
            mcp_server.update_session(sid, name="x")
        self.assertIn("not a JSON object", str(ctx.exception))

    def test_update_session_accepts_empty_string_name(self):
        """PR #4 follow-up review (CodeRabbit nitpick L360): callers
        can pass name='' to explicitly clear the name field; `if name:`
        was rejecting that as falsy."""
        save = mcp_server.save_session(name="initial", messages=[])
        sid = save["session_id"]
        result = mcp_server.update_session(sid, name="")
        self.assertEqual(result["name"], "")

    def test_delete_session_handles_concurrent_unlink(self):
        """PR #4 follow-up review (CodeRabbit nitpick L377): try/except
        FileNotFoundError on unlink() instead of exists()-then-unlink."""
        save = mcp_server.save_session(name="doomed", messages=[])
        sid = save["session_id"]
        # Simulate concurrent deletion
        (self.tmp_path / f"{sid}.json").unlink()
        with self.assertRaises(ValueError) as ctx:
            mcp_server.delete_session(sid)
        self.assertIn("Session not found", str(ctx.exception))


class TestToolRendering(_SessionMgmtBase):
    """Coverage for the call_tool handler's Markdown rendering paths.
    These exercise the public MCP surface (not just the helpers) so
    bugs in the wire format are caught."""

    def test_load_session_tool_includes_metadata(self):
        """PR #4 round-7 review (Codex P2 L760): metadata stored via
        save_session must surface in the load_session tool output.
        Previously the helper returned metadata in its dict but the
        tool handler dropped it before assembling the TextContent."""
        meta = {"project": "openclaw", "tags": ["security", "review"]}
        save = mcp_server.save_session(
            name="with-meta",
            messages=[{"role": "user", "content": "hi"}],
            metadata=meta,
        )
        sid = save["session_id"]

        result = asyncio.run(mcp_server.call_tool("load_session", {"session_id": sid}))
        text = result[0].text

        self.assertIn("### Metadata", text)
        # Both keys and the array element must round-trip through the
        # JSON-fenced render.
        self.assertIn("openclaw", text)
        self.assertIn("security", text)
        self.assertIn("review", text)
        # Conversation section still rendered.
        self.assertIn("### Conversation History", text)
        self.assertIn("USER:", text)

    def test_load_session_tool_omits_metadata_section_when_empty(self):
        """If metadata is empty/missing, no ``### Metadata`` heading
        should appear in the output (cosmetic — keeps the render
        clean for the common no-metadata case)."""
        save = mcp_server.save_session(
            name="no-meta",
            messages=[{"role": "user", "content": "hi"}],
        )
        sid = save["session_id"]

        result = asyncio.run(mcp_server.call_tool("load_session", {"session_id": sid}))
        text = result[0].text

        self.assertNotIn("### Metadata", text)
        self.assertIn("### Conversation History", text)

    def test_list_sessions_skips_non_uuid_filenames(self):
        """PR #4 round-7 review (Gemini medium L270): files in
        SESSIONS_DIR whose stems aren't valid UUIDs (e.g. a manual
        ``notes.json`` backup) must be skipped — not parsed and not
        listed under a misleading session id."""
        # Create a real session so we have a baseline.
        save = mcp_server.save_session(name="real", messages=[])
        real_sid = save["session_id"]

        # Drop a stray file with valid JSON shape but non-UUID stem.
        stray = self.tmp_path / "notes.json"
        stray.write_text(
            json.dumps(
                {
                    "name": "manual notes",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
        # And a stray file with a non-JSON stem to be doubly sure.
        odd = self.tmp_path / "config.json"
        odd.write_text(json.dumps({"foo": "bar"}))

        sessions = mcp_server.list_sessions()
        ids = {s["session_id"] for s in sessions}
        self.assertEqual(
            ids,
            {real_sid},
            f"non-UUID files leaked into listing: {ids}",
        )


class TestAtomicWrites(_SessionMgmtBase):
    """PR #4 follow-up review (Codex P2 L360 + Gemini med L270/L362):
    save/update must use temp-file + os.replace so a crash mid-write
    can't truncate or corrupt the session file."""

    def test_save_session_no_tmp_leak_on_success(self):
        """Successful save leaves no .tmp file behind."""
        result = mcp_server.save_session(
            name="clean", messages=[{"role": "user", "content": "hi"}]
        )
        sid = result["session_id"]
        self.assertTrue((self.tmp_path / f"{sid}.json").exists())
        leftover = list(self.tmp_path.glob("*.tmp"))
        self.assertEqual(leftover, [], f"Leftover temp files: {leftover}")

    def test_save_session_no_tmp_leak_on_write_failure(self):
        """If json.dump raises, the temp file must be cleaned up so
        SESSIONS_DIR doesn't accumulate .tmp litter on disk-full or
        permission errors."""
        with mock.patch.object(
            mcp_server.json, "dump", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                mcp_server.save_session(name="fail", messages=[])
        leftover = list(self.tmp_path.glob("*.tmp"))
        self.assertEqual(leftover, [], f"Leftover temp files: {leftover}")

    def test_update_session_preserves_file_on_write_failure(self):
        """If update_session's write fails, the original session file
        is unchanged (the whole point of the atomic temp+os.replace
        pattern). A non-atomic ``O_TRUNC`` write would have left a
        zero-length or partial JSON file here."""
        original = mcp_server.save_session(
            name="original",
            messages=[{"role": "user", "content": "important data"}],
        )
        sid = original["session_id"]
        sess_path = self.tmp_path / f"{sid}.json"
        with open(sess_path, "r", encoding="utf-8") as f:
            before = json.load(f)

        with mock.patch.object(
            mcp_server.json, "dump", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                mcp_server.update_session(sid, name="should-not-stick")

        with open(sess_path, "r", encoding="utf-8") as f:
            after = json.load(f)
        self.assertEqual(after, before)
        leftover = list(self.tmp_path.glob("*.tmp"))
        self.assertEqual(leftover, [])

    def test_atomic_temp_paths_are_unique_per_call(self):
        """PR #4 round-5 review (Codex P2 L392): _atomic_temp_for must
        return a different path on each call so two concurrent writers
        can't share an inode and corrupt each other's writes."""
        target = self.tmp_path / "abc.json"
        # mkstemp needs the dir to exist
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        paths = {mcp_server._atomic_temp_for(target) for _ in range(10)}
        # All 10 paths distinct.
        self.assertEqual(len(paths), 10, f"non-unique temp paths: {paths}")
        # All 10 inside the target's directory.
        for p in paths:
            self.assertEqual(p.parent, target.parent)
        # Cleanup so the test doesn't leave 10 empty temp files behind.
        for p in paths:
            p.unlink(missing_ok=True)

    def test_concurrent_updates_do_not_corrupt_session_file(self):
        """Codex P2 L392 in scenario form: two threads update the same
        session concurrently. The final file must (a) be valid JSON,
        (b) match one of the two writers' content end-to-end, and
        (c) leave no .tmp leak behind. With a fixed temp path this
        test would intermittently produce mixed/truncated JSON."""
        save = mcp_server.save_session(name="initial", messages=[])
        sid = save["session_id"]

        results: list[Exception | dict] = []
        barrier = threading.Barrier(2)

        def worker(new_name: str) -> None:
            try:
                # Sync both threads at the start so they actually race.
                barrier.wait(timeout=2)
                results.append(mcp_server.update_session(sid, name=new_name))
            except Exception as exc:  # pragma: no cover — defensive
                results.append(exc)

        # Run a handful of rounds to give the race opportunities to fire.
        # Single-round contention is unlikely to corrupt under the GIL,
        # so the loop is the actual coverage. With the fixed ``.tmp``
        # name this would have occasionally produced JSONDecodeError on
        # the post-write read.
        for round_idx in range(5):
            results.clear()
            barrier = threading.Barrier(2)
            threads = [
                threading.Thread(target=worker, args=(f"name-A-{round_idx}",)),
                threading.Thread(target=worker, args=(f"name-B-{round_idx}",)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
                self.assertFalse(t.is_alive(), "worker hung")

            # Both calls returned successfully (no exceptions).
            for r in results:
                self.assertIsInstance(
                    r, dict, f"update_session raised in concurrent run: {r!r}"
                )

            # The final on-disk file is valid JSON, and its name field
            # matches ONE of the two writers (whichever's os.replace
            # landed last). It is NEVER a mix of bytes from both.
            sess_path = self.tmp_path / f"{sid}.json"
            with open(sess_path, "r", encoding="utf-8") as f:
                final = json.load(f)
            self.assertIn(
                final["name"],
                {f"name-A-{round_idx}", f"name-B-{round_idx}"},
                f"final name corrupted: {final['name']!r}",
            )

            # No temp leak after either round.
            leftover = list(self.tmp_path.glob("*.tmp"))
            self.assertEqual(leftover, [], f"Leftover temp files: {leftover}")

    def test_cooperative_concurrent_update_and_delete_no_resurrection(self):
        """PR #4 round-6 review (Codex P2 L429): when two callers
        race ``update_session`` and ``delete_session`` on the same
        session via the public API, both honor ``_session_lock``
        and the result is consistent.

        Possible outcomes per round:
          - update wins lock first → completes → delete wins lock
            second, unlinks the updated file → final: no session.
          - delete wins lock first → unlinks → update wins lock
            second, hits FileNotFoundError on read → raises
            ``Session not found`` → final: no session.

        In BOTH outcomes the session ends up deleted; the file
        must NEVER exist at end of round (no resurrection)."""
        for round_idx in range(15):
            save = mcp_server.save_session(
                name=f"contested-{round_idx}",
                messages=[{"role": "user", "content": "hi"}],
            )
            sid = save["session_id"]
            sess_path = self.tmp_path / f"{sid}.json"
            self.assertTrue(
                sess_path.exists(),
                f"setup failed at round {round_idx}",
            )

            results: dict[str, Exception | dict] = {}
            barrier = threading.Barrier(2)

            def do_update(name: str = f"upd-{round_idx}") -> None:
                try:
                    barrier.wait(timeout=2)
                    results["update"] = mcp_server.update_session(sid, name=name)
                except Exception as exc:
                    results["update"] = exc

            def do_delete() -> None:
                try:
                    barrier.wait(timeout=2)
                    results["delete"] = mcp_server.delete_session(sid)
                except Exception as exc:
                    results["delete"] = exc

            threads = [
                threading.Thread(target=do_update),
                threading.Thread(target=do_delete),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
                self.assertFalse(t.is_alive(), "worker hung")

            # No resurrection: the session file must not exist at
            # end of round, regardless of which call won the lock
            # first.
            self.assertFalse(
                sess_path.exists(),
                f"session resurrected at round {round_idx}: {results}",
            )

            # If delete won the lock first, update sees FileNotFound
            # and raises ValueError("Session not found"). If update
            # won first, both calls succeed.
            update_res = results.get("update")
            if isinstance(update_res, Exception):
                self.assertIsInstance(update_res, ValueError)
                self.assertIn("not found", str(update_res).lower())
            else:
                self.assertIsInstance(update_res, dict)
            # Delete should never raise in this scenario — even if
            # update went first, delete's unlink targets the
            # post-update file, which still exists when delete
            # acquires the lock.
            self.assertIsInstance(
                results.get("delete"), dict, f"delete raised: {results}"
            )

            # Cleanup the leftover lockfile so glob counts stay
            # predictable across rounds.
            for lockfile in self.tmp_path.glob("*.lock"):
                lockfile.unlink()

    def test_session_lock_raises_clean_error_on_non_posix(self):
        """PR #4 round-9 review (Gemini medium L17): on Windows or
        any platform without ``fcntl``, ``_session_lock`` must raise
        a clear OSError instead of crashing with AttributeError. The
        module's top-level ``import fcntl`` is wrapped in try/except
        so the module loads fine on Windows; this test verifies the
        runtime guard inside the helper."""
        save = mcp_server.save_session(name="x", messages=[])
        sid = save["session_id"]
        session_file = mcp_server.get_session_file(sid)

        # Simulate a Windows environment by setting mcp_server.fcntl
        # to None — the same state the conditional import produces
        # when ImportError fires.
        with mock.patch.object(mcp_server, "fcntl", None):
            with self.assertRaises(OSError) as ctx:
                with mcp_server._session_lock(session_file):
                    pass  # pragma: no cover — should not reach
            self.assertIn("POSIX", str(ctx.exception))

    def test_session_lock_finally_survives_fcntl_set_to_none_mid_context(self):
        """PR #4 round-11 review (Gemini medium L258): if
        ``mcp_server.fcntl`` is monkey-patched to None *during* a
        held lock (test fixtures, runtime mutation), the finally
        block must not crash with AttributeError when it attempts
        to release. The original exception (if any) should propagate
        cleanly; the cleanup should be a quiet no-op for the unlock
        and still close the fd.

        PR #4 round-12 review (Codex P3 L912): the original version
        of this test used a nested ``mock.patch.object`` context
        manager inside ``_session_lock``. Because Python unwinds the
        inner context first, ``mcp_server.fcntl`` was restored to
        the real module BEFORE ``_session_lock.__exit__`` ran — so
        the defensive guard was never exercised and this test
        passed vacuously against the buggy implementation. Fix:
        mutate ``mcp_server.fcntl`` directly inside the lock and
        restore it from the OUTER ``try/finally`` so the patch
        stays active when the lock's finally fires.
        """
        save = mcp_server.save_session(name="y", messages=[])
        sid = save["session_id"]
        session_file = mcp_server.get_session_file(sid)

        class _Sentinel(Exception):
            pass

        # Manual save+restore (rather than nested with-statement) so
        # mcp_server.fcntl stays None when _session_lock.__exit__
        # runs the unlock-then-close cleanup. The outer try/finally
        # ensures the real fcntl is restored even if the inner
        # assertRaises itself raises something unexpected.
        real_fcntl = mcp_server.fcntl
        try:
            with self.assertRaises(_Sentinel):
                with mcp_server._session_lock(session_file):
                    # Lock is held; now flip fcntl to None so the
                    # finally block has to take the defensive
                    # ``if fcntl is not None:`` branch.
                    mcp_server.fcntl = None
                    raise _Sentinel("propagate me")
        finally:
            mcp_server.fcntl = real_fcntl

    def test_lockfile_released_on_update_exception(self):
        """If ``update_session`` raises mid-critical-section, the
        lockfile must be released so subsequent calls aren't
        deadlocked. This validates the ``finally`` arm of
        ``_session_lock``'s context manager."""
        save = mcp_server.save_session(name="x", messages=[])
        sid = save["session_id"]
        sess_path = self.tmp_path / f"{sid}.json"

        # Force update_session to raise during write.
        with mock.patch.object(
            mcp_server.json, "dump", side_effect=OSError("simulated")
        ):
            with self.assertRaises(OSError):
                mcp_server.update_session(sid, name="will-fail")

        # The lock file may exist (we don't unlink on release —
        # that's documented behavior to avoid a re-acquire race),
        # but it must NOT be flock'd by a stale fd. Re-acquiring
        # via another update_session call must succeed quickly,
        # not block.
        result = mcp_server.update_session(sid, name="recovered")
        self.assertEqual(result["name"], "recovered")
        self.assertTrue(sess_path.exists())

    def test_update_session_rejects_concurrent_deletion(self):
        """Codex P2 L360: if a session is deleted between update_session's
        read and its write, update_session must NOT silently re-create
        the file. The existence re-check before the atomic write should
        raise ValueError so the caller learns the operation lost a race."""
        save = mcp_server.save_session(name="vanishing", messages=[])
        sid = save["session_id"]
        sess_path = self.tmp_path / f"{sid}.json"

        # Wrap json.load so it deletes the file as a side effect after
        # successfully reading it. This simulates a concurrent
        # delete_session() landing between our read and our write.
        original_load = mcp_server.json.load

        def evil_load(fp):
            data = original_load(fp)
            sess_path.unlink()
            return data

        with mock.patch.object(mcp_server.json, "load", side_effect=evil_load):
            with self.assertRaises(ValueError) as ctx:
                mcp_server.update_session(sid, name="resurrected?")

        self.assertIn("deleted concurrently", str(ctx.exception))
        # And critically: the file stayed deleted (no silent recreation).
        self.assertFalse(sess_path.exists())
        # No temp leak either.
        leftover = list(self.tmp_path.glob("*.tmp"))
        self.assertEqual(leftover, [])


class TestLazyKeychainImport(unittest.TestCase):
    """PR #4 round-10 review (Codex P2 L38): the perplexity client must
    be constructed lazily so importing ``mcp_server`` succeeds even when
    the macOS ``security`` CLI is unavailable. This complements round-9's
    fcntl portability fix — the goal of "module loads cleanly on
    non-POSIX" requires *both* fcntl and the keychain lookup to be
    deferred. These tests do a fresh ``importlib`` load with
    ``subprocess.run`` mocked to fail, so we exercise the import-time
    path rather than poking the already-imported ``mcp_server`` global."""

    def _import_with_failing_keychain(self, *, drop_fcntl: bool):
        """Import mcp_server in a fresh module slot with the keychain
        CLI mocked to raise ``FileNotFoundError`` (simulating Windows
        or a stripped container where ``security`` is absent). When
        ``drop_fcntl`` is True we additionally make ``import fcntl``
        raise ImportError to mimic a true Windows environment.

        Returns the freshly imported module on success; raises the
        underlying error if the import itself blew up.
        """
        stubs = _build_stub_modules()

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("[Errno 2] No such file or directory: 'security'")

        # Use a unique slot name so the import isn't served from a
        # cached entry left by ``_load_mcp_server`` at module-load
        # time.
        slot_name = (
            "mcp_server_lazy_keychain_no_fcntl"
            if drop_fcntl
            else "mcp_server_lazy_keychain"
        )

        # Build a builtins.__import__ wrapper that raises ImportError
        # specifically for fcntl. This is what actually fires the
        # round-9 try/except path on Windows; setting ``sys.modules
        # ["fcntl"] = None`` is not sufficient when fcntl is a
        # statically-linked built-in (as on macOS), because the import
        # machinery resolves it directly without consulting
        # sys.modules.
        import builtins as _builtins

        original_import = _builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "fcntl":
                raise ImportError("No module named 'fcntl' (simulated)")
            return original_import(name, *args, **kwargs)

        ctx = mock.patch.dict(sys.modules, stubs)
        with ctx:
            # Force a fresh resolution of fcntl by removing any
            # already-cached entry; otherwise Python returns the
            # cached real fcntl without ever calling our import hook.
            sys.modules.pop("fcntl", None)
            import_patch = (
                mock.patch.object(_builtins, "__import__", side_effect=blocking_import)
                if drop_fcntl
                else mock.patch.object(_builtins, "__import__", wraps=original_import)
            )
            with import_patch:
                with mock.patch("subprocess.run", side_effect=fake_run):
                    spec = importlib.util.spec_from_file_location(
                        slot_name, SERVER_PATH
                    )
                    assert spec is not None and spec.loader is not None
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
        # Drop the freshly-loaded module from sys.modules so it doesn't
        # leak into other tests that import via _load_mcp_server.
        sys.modules.pop(slot_name, None)
        return module

    def test_module_imports_when_keychain_cli_missing(self):
        """The headline round-10 case: ``security`` is absent, but
        ``import mcp_server`` still succeeds. Without the lazy
        accessor this would raise FileNotFoundError at import time."""
        module = self._import_with_failing_keychain(drop_fcntl=False)
        # The cache must start uninitialised — proves we did not eagerly
        # call ``get_api_key_from_keychain`` at module load.
        self.assertIsNone(module._perplexity_client_cache)
        # And the helper functions we care about are still reachable
        # for tool discovery.
        self.assertTrue(callable(module._get_perplexity_client))
        self.assertTrue(callable(module.list_sessions))

    def test_module_imports_when_fcntl_and_keychain_missing(self):
        """Combined non-POSIX scenario: neither ``fcntl`` nor the
        ``security`` CLI is available. Round 9 fixed fcntl; round 10
        must additionally not regress when both fail. Together they
        deliver the round-9 stated goal of a clean Windows import."""
        module = self._import_with_failing_keychain(drop_fcntl=True)
        # fcntl flag was set to None by the stubbed import.
        self.assertIsNone(module.fcntl)
        # Keychain client still uninitialised.
        self.assertIsNone(module._perplexity_client_cache)

    def test_get_perplexity_client_propagates_keychain_error(self):
        """When ``deep_research`` is invoked on a system without the
        keychain CLI, the failure must surface — we are deferring the
        lookup, not silently swallowing it. v1.2 (issue #20): a missing
        security(1) no longer leaks a raw FileNotFoundError; it raises
        the actionable ValueError naming the env-var remedy, so the
        laziness still preserves a loud, actionable failure signal."""
        module = self._import_with_failing_keychain(drop_fcntl=False)
        module.os.environ.pop("PERPLEXITY_API_KEY", None)
        with mock.patch.object(
            module,
            "subprocess",
            mock.Mock(
                run=mock.Mock(
                    side_effect=FileNotFoundError(
                        "[Errno 2] No such file or directory: 'security'"
                    )
                ),
                CalledProcessError=Exception,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                module._get_perplexity_client()
        self.assertIn("PERPLEXITY_API_KEY", str(ctx.exception))

    def test_get_perplexity_client_caches_first_call(self):
        """The accessor must memoise — repeated invocations should
        not re-shell out to ``security`` once the first call has
        succeeded."""
        # Reuse the already-imported test module (it was loaded with a
        # stubbed succeeding ``subprocess.run`` returning ``dummy-key``)
        # and reset the cache so we can observe the first-call write.
        original = mcp_server._perplexity_client_cache
        try:
            mcp_server._perplexity_client_cache = None
            call_count = {"n": 0}

            def counting_run(*args, **kwargs):
                call_count["n"] += 1
                return types.SimpleNamespace(returncode=0, stdout="k\n")

            with mock.patch.object(
                mcp_server.subprocess, "run", side_effect=counting_run
            ):
                first = mcp_server._get_perplexity_client()
                second = mcp_server._get_perplexity_client()

            self.assertIs(first, second)
            self.assertEqual(call_count["n"], 1)
        finally:
            mcp_server._perplexity_client_cache = original


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    sys.exit(0 if runner.run(suite).wasSuccessful() else 1)
