#!/usr/bin/env python3
"""Unit tests for redact_secrets() in mcp_server.py.

Self-contained: stubs out the third-party imports (mcp, openai) and the
Keychain lookup so the test can import mcp_server without needing the
full runtime environment. Uses only stdlib (unittest).

Run:
    python3 test_redact.py

NOTE: Secret-shape fixtures are assembled at runtime from broken-up
parts so secret scanners (semgrep, gitleaks, trufflehog) do not flag
this test file as containing real credentials. Every fixture below is
synthetic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SERVER_PATH = HERE / "mcp_server.py"

# --- Synthetic secret-shape fixtures (assembled at runtime) ---------------
# Each prefix is split so the contiguous literal never appears in source.
_GOOG_OAUTH_ACCESS_PREFIX = "ya" + "29."
_GOOG_OAUTH_REFRESH_PREFIX = "1" + "//" + "0g"
_GOOG_API_KEY_PREFIX = "AI" + "za"
_JWT_HEADER = "ey" + "J" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9XX"

FAKE_GOOG_ACCESS = _GOOG_OAUTH_ACCESS_PREFIX + "synthetic_access_token_123-_xyz"
FAKE_GOOG_REFRESH = _GOOG_OAUTH_REFRESH_PREFIX + (
    "Abcdef0123456789ABCDEFGHIJKLmnop_qrs"
)
FAKE_GOOG_API_KEY = _GOOG_API_KEY_PREFIX + "SyDsyntheticTestValue1234567890_-end"
FAKE_JWT = (
    _JWT_HEADER
    + ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1"
    + ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """Return the dict of fake mcp/openai/httpx/google.auth modules used
    during import. Caller is expected to scope these via
    mock.patch.dict(sys.modules) rather than mutating sys.modules
    directly (per PR #8 review, Gemini comment #3285598397: test stubs
    must not leak into other tests' sys.modules entries — leakage masks
    missing imports in sibling test files and makes results depend on
    pytest's discovery order).
    """

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    # httpx is imported at module level by mcp_server.py for the Gemini
    # Deep Research HTTP client. Tests never exercise the network path —
    # the TestGemini* tool-boundary tests mock.patch.object the helper
    # functions, and the Test*GeminiInteractionHelper unit tests inject a
    # fake client — so the fakes here just need to be Exception
    # subclasses with the right names for the helpers' except clauses.
    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

    class _FakeHTTPStatusError(Exception):
        pass

    class _FakeRequestError(Exception):
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

    async def _fake_stdio_server():  # not actually awaited in tests
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
        "google": google_mod,
        "google.auth": auth_mod,
        "google.auth.exceptions": auth_exceptions_mod,
        "google.auth.transport": transport_mod,
        "google.auth.transport.requests": transport_requests_mod,
    }


def _load_mcp_server():
    """Import mcp_server.py with mcp.*/openai/httpx/google.auth stubbed
    via a scoped sys.modules patch so the fakes don't leak into later
    test imports."""
    stubs = _build_stub_modules()
    fake_proc = types.SimpleNamespace(returncode=0, stdout="dummy-key\n")
    with mock.patch.dict(sys.modules, stubs):
        with mock.patch("subprocess.run", return_value=fake_proc):
            # DO NOT CONSOLIDATE this name with test_session_mgmt.py's.
            # Today the loaded module is bound to a local variable and
            # the spec name is not registered in sys.modules (manual
            # exec_module + scoped patch.dict), so collisions would be
            # harmless. But three plausible future changes would make
            # them dangerous: (a) adopting the docs' canonical
            # `sys.modules[spec.name] = module` pattern for circular-
            # import support, (b) weakening or removing the patch.dict
            # scope above, (c) switching to a loader that auto-registers.
            # Unique per-file suffixes cost nothing, surface in tracebacks
            # so you can tell which test loaded which copy, and pre-empt
            # all three regressions. Keep them distinct.
            spec = importlib.util.spec_from_file_location(
                "mcp_server_under_test_redact", SERVER_PATH
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    return module


mcp_server = _load_mcp_server()
redact_secrets = mcp_server.redact_secrets


class TestRedactSecrets(unittest.TestCase):
    def test_google_oauth_access_token(self):
        s = f"auth={FAKE_GOOG_ACCESS} next"
        out = redact_secrets(s)
        self.assertIn("[REDACTED_GOOGLE_OAUTH_ACCESS]", out)
        self.assertNotIn(_GOOG_OAUTH_ACCESS_PREFIX, out)

    def test_google_oauth_refresh_token(self):
        s = f"refresh: {FAKE_GOOG_REFRESH} end"
        out = redact_secrets(s)
        self.assertIn("[REDACTED_GOOGLE_OAUTH_REFRESH]", out)
        self.assertNotIn(_GOOG_OAUTH_REFRESH_PREFIX, out)

    def test_google_api_key(self):
        s = f"key={FAKE_GOOG_API_KEY}"
        out = redact_secrets(s)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", out)

    def test_jwt(self):
        out = redact_secrets(f"Bearer {FAKE_JWT} more")
        self.assertIn("[REDACTED_JWT]", out)
        self.assertNotIn(_JWT_HEADER[:4], out)

    def test_no_apple_asp_redaction_after_codex_review(self):
        """PR #1 review (Codex P2): the Apple ASP regex was dropped because
        it false-positived on ordinary 4-word lowercase hyphenated phrases
        in research-prose output. None of these benign strings should be
        mutated, and the [REDACTED_APPLE_APP_PWD] marker should never
        appear.
        """
        for s in [
            "real-time-data-flow",  # Codex's first example
            "zero-shot-text-only",  # Codex's second example
            "abcd-efgh-ijkl-mnop",  # original ASP shape — now benign
            "self-hosted-build-tools",  # ML/devops slug
            "550e8400-e29b-41d4-a716-446655440000",  # UUID
            "AAAA-BBBB-CCCC-DDDD",  # uppercase 4x4 (always unmatched)
        ]:
            out = redact_secrets(s)
            self.assertEqual(out, s, f"unexpected mutation of {s!r}: got {out!r}")
            self.assertNotIn("REDACTED_APPLE", out)

    def test_private_key_block(self):
        # Inline a non-Google-shape body so the block test stays focused.
        block = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEAfooBarBaz\n"
            "synthetic-body-content-here-1234567890\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = redact_secrets(f"prefix\n{block}\nsuffix")
        self.assertIn("[REDACTED_PRIVATE_KEY_BLOCK]", out)
        self.assertNotIn("MIIEpAIBAAKCAQEA", out)

    def test_private_key_block_swallows_inner_secrets(self):
        # If a key-shape string is nested inside a PEM block, the outer
        # block pattern (matched first) should swallow it whole.
        block = (
            "-----BEGIN PRIVATE KEY-----\n"
            f"body-with-{FAKE_GOOG_API_KEY}-inside\n"
            "-----END PRIVATE KEY-----"
        )
        out = redact_secrets(block)
        self.assertEqual(out, "[REDACTED_PRIVATE_KEY_BLOCK]")

    def test_private_key_block_unlabeled(self):
        block = "-----BEGIN PRIVATE KEY-----\nABCDEFGH\n-----END PRIVATE KEY-----"
        out = redact_secrets(block)
        self.assertEqual(out, "[REDACTED_PRIVATE_KEY_BLOCK]")

    def test_multiple_patterns_in_one_string(self):
        s = f"Use {FAKE_GOOG_API_KEY} with token {FAKE_GOOG_ACCESS} then jwt {FAKE_JWT}"
        out = redact_secrets(s)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertIn("[REDACTED_GOOGLE_OAUTH_ACCESS]", out)
        self.assertIn("[REDACTED_JWT]", out)

    def test_nested_dict_and_list(self):
        nested = {
            "title": "Doc",
            "items": [
                f"key={FAKE_GOOG_API_KEY}",
                {"jwt": FAKE_JWT},
            ],
            "count": 5,
            "active": True,
            "skipped": None,
        }
        out = redact_secrets(nested)
        self.assertNotIn(_GOOG_API_KEY_PREFIX, str(out))
        self.assertNotIn(_JWT_HEADER[:4], str(out))
        self.assertEqual(out["count"], 5)
        self.assertIs(out["active"], True)
        self.assertIsNone(out["skipped"])
        self.assertEqual(out["title"], "Doc")

    def test_tuple_preserved_as_tuple(self):
        out = redact_secrets(("safe", FAKE_GOOG_API_KEY))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0], "safe")
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out[1])

    def test_passthrough_for_unsupported_types(self):
        for v in [None, 42, 3.14, True, b"bytes-not-walked"]:
            self.assertEqual(redact_secrets(v), v)

    def test_idempotent(self):
        s = f"{FAKE_GOOG_ACCESS} and {FAKE_GOOG_API_KEY}"
        once = redact_secrets(s)
        twice = redact_secrets(once)
        self.assertEqual(once, twice)

    def test_empty_string(self):
        self.assertEqual(redact_secrets(""), "")

    def test_clean_string_unchanged(self):
        s = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(redact_secrets(s), s)

    def test_short_jwt_is_caught_after_gemini_review(self):
        """PR #1 review (Gemini): JWT minimum lengths were relaxed from
        {30,30,20} to {10,10,10}. Catches minimal real JWTs — e.g. a header
        encoding `{"alg":"HS256"}` is 20 chars, which the original {30,}
        requirement missed entirely.
        """
        # Build a minimal-shape JWT at runtime to dodge secret scanners.
        header = "ey" + "J" + "hbGciOiJIUzI1NiJ9"  # 20 chars
        payload = "ey" + "J" + "zdWIiOiIifQ"  # 14 chars
        sig = "shortsigvalue123"  # 16 chars (>10)
        short_jwt = f"{header}.{payload}.{sig}"
        out = redact_secrets(f"Bearer {short_jwt} after")
        self.assertIn("[REDACTED_JWT]", out)
        self.assertNotIn(header, out)

    def test_secret_as_dict_key_is_redacted_after_gemini_review(self):
        """PR #1 review (Gemini): redact_secrets() now walks dict KEYS
        recursively, not just values. A secret appearing as a dict key
        (e.g. an API key passed as a parameter name in synthesized JSON)
        was previously left in plaintext.
        """
        d = {FAKE_GOOG_API_KEY: "value", "title": FAKE_GOOG_API_KEY}
        out = redact_secrets(d)
        # Raw key prefix should not appear anywhere in the output.
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", str(out))
        # The key got redacted to the marker; both occurrences collapse
        # into a single "[REDACTED_GOOGLE_API_KEY]" key.
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertEqual(out["title"], "[REDACTED_GOOGLE_API_KEY]")

    def test_dict_keys_collision_preserves_all_values_after_pr2_review(self):
        """PR #2 review (Codex P2 + Gemini medium): two DISTINCT secret-shape
        keys that both redact to the same marker would collapse into one
        dict entry under naive comprehension, silently losing values. Fix
        appends a numeric suffix (#2, #3, ...) on collision so every entry
        is preserved.
        """
        k1 = _GOOG_API_KEY_PREFIX + "SyDistinctKey1foobarbazquux1234567"
        k2 = _GOOG_API_KEY_PREFIX + "SyDistinctKey2foobarbazquux1234567"
        d = {k1: "value1", k2: "value2"}
        out = redact_secrets(d)
        # Both values must be preserved.
        self.assertEqual(set(out.values()), {"value1", "value2"})
        # Two distinct redacted keys: marker, then marker#2.
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]#2", out)
        # No leakage of original prefixes.
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", str(out))

    def test_dict_keys_collision_three_way(self):
        """Three distinct secret-shape keys -> marker, marker#2, marker#3."""
        keys = [
            _GOOG_API_KEY_PREFIX + "SyAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            _GOOG_API_KEY_PREFIX + "SyBbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            _GOOG_API_KEY_PREFIX + "SyCcccccccccccccccccccccccccccccc",
        ]
        d = {k: f"v{i}" for i, k in enumerate(keys)}
        out = redact_secrets(d)
        self.assertEqual(set(out.values()), {"v0", "v1", "v2"})
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]#2", out)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]#3", out)

    def test_dict_non_string_keys_preserve_type_after_pr2_review_gemini(self):
        """PR #2 follow-up review (Gemini medium): non-string dict keys
        (tuples, ints, etc.) must pass through with original type preserved.
        Collision-handling is gated on string keys only because non-string
        keys cannot collide with redacted-string markers.
        """
        # Tuple keys (no secret-shape) — pass through unchanged with type intact
        d_tuple = {(1, 2): "v1", (3, 4): "v2"}
        out = redact_secrets(d_tuple)
        self.assertEqual(out, d_tuple)
        self.assertTrue(all(isinstance(k, tuple) for k in out))

        # Integer keys
        d_int = {42: "v1", 99: "v2"}
        out = redact_secrets(d_int)
        self.assertEqual(out, d_int)
        self.assertTrue(all(isinstance(k, int) for k in out))

        # Mixed: secret-shape string + non-string keys all in one dict
        mixed = {FAKE_GOOG_API_KEY: "leaked", (1, 2): "tup", 42: "i"}
        out = redact_secrets(mixed)
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", out)
        self.assertIn((1, 2), out)
        self.assertIn(42, out)
        self.assertEqual(out[(1, 2)], "tup")
        self.assertEqual(out[42], "i")

    def test_tuple_keys_collision_preserves_all_values(self):
        """PR #2 follow-up review (Codex P2): tuple keys whose elements get
        recursively redacted into the same shape would collide and silently
        drop data. Fix: extend collision-handling to tuples by appending a
        '#N' string element.
        """
        k1 = (FAKE_GOOG_API_KEY + "_one", "x")
        k2 = (FAKE_GOOG_API_KEY + "_two", "x")
        d = {k1: "v1", k2: "v2"}
        out = redact_secrets(d)
        # Both values must be preserved.
        self.assertEqual(set(out.values()), {"v1", "v2"})
        # First collides as the bare redacted tuple; second gets "#2" suffix.
        bare = ("[REDACTED_GOOGLE_API_KEY]", "x")
        suffixed = ("[REDACTED_GOOGLE_API_KEY]", "x", "#2")
        self.assertIn(bare, out)
        self.assertIn(suffixed, out)


class TestValidateInteractionId(unittest.TestCase):
    """The Gemini interaction_id flows into the URL of an authenticated HTTP
    call, so the validator is the boundary that prevents credential leakage to
    an attacker-controlled host."""

    def setUp(self):
        self.validate = mcp_server._validate_interaction_id

    def test_accepts_alphanumeric(self):
        self.assertEqual(self.validate("abc123XYZ_-"), "abc123XYZ_-")

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.validate("../../etc/passwd")

    def test_rejects_slash(self):
        with self.assertRaises(ValueError):
            self.validate("abc/def")

    def test_rejects_at_sign_host_swap(self):
        # Classic URL trick: foo@evil.com would shift the host of a naive
        # f-string URL construction.
        with self.assertRaises(ValueError):
            self.validate("abc@evil.com")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.validate("")

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            self.validate("a" * 129)

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            self.validate(None)  # type: ignore[arg-type]

    def test_rejects_whitespace(self):
        with self.assertRaises(ValueError):
            self.validate("abc def")


class TestGeminiStartValidation(unittest.IsolatedAsyncioTestCase):
    """Strict input validation at the gemini_deep_research_start boundary.

    These checks defend against truthy-string traps (`bool("false") is True`),
    arbitrary thinking_summaries values, and silent acceptance of an empty/
    missing interaction id from the upstream response."""

    async def _call(self, args, post_return):
        async def _fake_post(payload):
            self.last_payload = payload
            return post_return

        with mock.patch.object(mcp_server, "_post_gemini_interaction", _fake_post):
            return await mcp_server.call_tool("gemini_deep_research_start", args)

    async def test_collaborative_planning_string_rejected(self):
        # `bool("false")` is True in Python — guard must catch this.
        out = await self._call(
            {"query": "x", "collaborative_planning": "false"},
            {"id": "abc123"},
        )
        text = out[0].text
        self.assertIn('"status": "failed"', text)
        self.assertIn("collaborative_planning", text)

    async def test_collaborative_planning_int_rejected(self):
        out = await self._call(
            {"query": "x", "collaborative_planning": 1},
            {"id": "abc123"},
        )
        self.assertIn('"status": "failed"', out[0].text)

    async def test_collaborative_planning_true_accepted(self):
        out = await self._call(
            {"query": "x", "collaborative_planning": True},
            {"id": "abc123", "status": "in_progress"},
        )
        self.assertIn('"interaction_id": "abc123"', out[0].text)
        self.assertTrue(self.last_payload["agent_config"]["collaborative_planning"])

    async def test_thinking_summaries_invalid_value_rejected(self):
        out = await self._call(
            {"query": "x", "thinking_summaries": "verbose"},
            {"id": "abc123"},
        )
        text = out[0].text
        self.assertIn('"status": "failed"', text)
        self.assertIn("thinking_summaries", text)

    async def test_thinking_summaries_none_accepted(self):
        out = await self._call(
            {"query": "x", "thinking_summaries": "none"},
            {"id": "abc123", "status": "in_progress"},
        )
        self.assertIn('"interaction_id": "abc123"', out[0].text)

    async def test_missing_interaction_id_fails_loudly(self):
        # Upstream returned no id → must raise RuntimeError so the MCP layer
        # surfaces a tool error rather than handing back a poll id of `null`.
        async def _fake_post(payload):
            return {"status": "in_progress"}  # no id

        with mock.patch.object(mcp_server, "_post_gemini_interaction", _fake_post):
            with self.assertRaises(RuntimeError):
                await mcp_server.call_tool("gemini_deep_research_start", {"query": "x"})

    async def test_malformed_interaction_id_fails_loudly(self):
        async def _fake_post(payload):
            return {"id": "has/slash"}

        with mock.patch.object(mcp_server, "_post_gemini_interaction", _fake_post):
            with self.assertRaises(RuntimeError):
                await mcp_server.call_tool("gemini_deep_research_start", {"query": "x"})

    async def test_previous_interaction_id_validated_and_forwarded(self):
        out = await self._call(
            {"query": "x", "previous_interaction_id": "prev_abc-123"},
            {"id": "abc123", "status": "in_progress"},
        )
        self.assertIn('"interaction_id": "abc123"', out[0].text)
        self.assertEqual(self.last_payload["previous_interaction_id"], "prev_abc-123")

    async def test_previous_interaction_id_invalid_rejected(self):
        out = await self._call(
            {"query": "x", "previous_interaction_id": "../etc/passwd"},
            {"id": "abc123"},
        )
        self.assertIn('"status": "failed"', out[0].text)


class TestGeminiResultTerminalStates(unittest.IsolatedAsyncioTestCase):
    """The result handler must treat cancelled/failed/completed as terminal,
    everything else as in-progress, and tolerate malformed step entries."""

    async def _call(self, response):
        async def _fake_get(interaction_id):
            return response

        with mock.patch.object(mcp_server, "_get_gemini_interaction", _fake_get):
            out = await mcp_server.call_tool(
                "gemini_deep_research_result", {"interaction_id": "abc123"}
            )
        return out[0].text

    async def test_cancelled_is_terminal(self):
        text = await self._call({"status": "cancelled"})
        self.assertIn('"status": "cancelled"', text)
        self.assertIn('"error"', text)
        self.assertNotIn("Still running", text)

    async def test_unknown_status_is_in_progress(self):
        text = await self._call({"status": "queued"})
        self.assertIn("Still running", text)

    async def test_steps_summary_skips_non_dict_entries(self):
        text = await self._call(
            {
                "status": "completed",
                "output_text": "done",
                "steps": [{"type": "search"}, "garbage", None, {"type": "synth"}],
            }
        )
        # Non-dict entries are skipped; the two dict types remain.
        self.assertIn('"search"', text)
        self.assertIn('"synth"', text)
        # steps_count includes ALL entries, not just dicts, so the upstream
        # count is preserved for diagnostic accuracy.
        self.assertIn('"steps_count": 4', text)


class TestPostGeminiInteractionHelper(unittest.TestCase):
    """Unit tests for the POST HTTP helper itself (client mocked, no network).

    Mirrors TestPostAgentResearchHelper in test_agent_research.py: network-
    layer failures (httpx.RequestError) and JSON decode failures (ValueError
    from response.json()) must return the structured
    ``{"status": "failed", "error": ...}`` envelope — the same treatment the
    Perplexity Agent API helpers got in PR #16 — instead of crashing the
    MCP tool call. Credential/ADC errors stay outside the try block and
    must keep propagating.
    """

    def _run_helper(self, fake_client):
        async def fake_get_client():
            return fake_client

        async def fake_headers():
            return {"Authorization": "Bearer test-token"}

        with mock.patch.object(mcp_server, "_gemini_headers", side_effect=fake_headers):
            with mock.patch.object(
                mcp_server, "_get_http_client", side_effect=fake_get_client
            ):
                return asyncio.run(
                    mcp_server._post_gemini_interaction({"input": "query"})
                )

    def test_http_error_becomes_failure_envelope(self):
        class _FakeErrorResponse:
            status_code = 500
            text = "internal error"

            def raise_for_status(self):
                exc = mcp_server.httpx.HTTPStatusError("500")
                exc.response = self
                raise exc

            def json(self):  # pragma: no cover - raise_for_status fires first
                return {}

        class _FakeClient:
            async def post(self, url, **kwargs):
                return _FakeErrorResponse()

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("500", data["error"])

    def test_request_error_becomes_failure_envelope(self):
        # Connect errors and read timeouts must return the structured
        # envelope, not crash the tool call.
        class _FakeClient:
            async def post(self, url, **kwargs):
                raise mcp_server.httpx.RequestError("connection refused")

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("connection refused", data["error"])

    def test_request_error_message_is_redacted(self):
        # Exception text is untrusted and must pass through redact_secrets
        # before reaching the envelope — same "never emit secret-shapes"
        # contract as _http_error_payload (per Qodo review on PR #17).
        class _FakeClient:
            async def post(self, url, **kwargs):
                raise mcp_server.httpx.RequestError(
                    f"proxy rejected key {FAKE_GOOG_API_KEY}"
                )

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("[REDACTED_GOOGLE_API_KEY]", data["error"])
        self.assertNotIn(_GOOG_API_KEY_PREFIX + "Sy", data["error"])

    def test_invalid_json_becomes_failure_envelope(self):
        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        class _FakeClient:
            async def post(self, url, **kwargs):
                return _FakeResponse()

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("JSON", data["error"])

    def test_credential_errors_propagate(self):
        # ADC/credential failures happen in _gemini_headers, outside the
        # try block — they must raise, not be swallowed into an envelope.
        async def failing_headers():
            raise RuntimeError("ADC credentials unavailable")

        with mock.patch.object(
            mcp_server, "_gemini_headers", side_effect=failing_headers
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(mcp_server._post_gemini_interaction({"input": "query"}))


class TestGetGeminiInteractionHelper(unittest.TestCase):
    """Unit tests for the GET (poll) HTTP helper — same envelope contract
    as TestPostGeminiInteractionHelper, mirroring TestGetAgentResponseHelper
    in test_agent_research.py."""

    def _run_helper(self, fake_client, interaction_id="abc123"):
        async def fake_get_client():
            return fake_client

        async def fake_headers():
            return {"Authorization": "Bearer test-token"}

        with mock.patch.object(mcp_server, "_gemini_headers", side_effect=fake_headers):
            with mock.patch.object(
                mcp_server, "_get_http_client", side_effect=fake_get_client
            ):
                return asyncio.run(mcp_server._get_gemini_interaction(interaction_id))

    def test_http_error_becomes_failure_envelope(self):
        class _FakeErrorResponse:
            status_code = 404
            text = "not found"

            def raise_for_status(self):
                exc = mcp_server.httpx.HTTPStatusError("404")
                exc.response = self
                raise exc

            def json(self):  # pragma: no cover - raise_for_status fires first
                return {}

        class _FakeClient:
            async def get(self, url, **kwargs):
                return _FakeErrorResponse()

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("404", data["error"])

    def test_helper_revalidates_id_as_defense_in_depth(self):
        # The validation ValueError is raised before the try block — it
        # must propagate, never be misread as a JSON decode failure.
        with self.assertRaises(ValueError):
            asyncio.run(mcp_server._get_gemini_interaction("abc/../evil"))

    def test_request_error_becomes_failure_envelope(self):
        class _FakeClient:
            async def get(self, url, **kwargs):
                raise mcp_server.httpx.RequestError("read timeout")

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("read timeout", data["error"])

    def test_request_error_message_is_redacted(self):
        # Same redaction contract as the POST helper — see
        # TestPostGeminiInteractionHelper.test_request_error_message_is_redacted.
        class _FakeClient:
            async def get(self, url, **kwargs):
                raise mcp_server.httpx.RequestError(
                    f"DNS failure for {FAKE_GOOG_ACCESS}"
                )

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("[REDACTED_GOOGLE_OAUTH_ACCESS]", data["error"])
        self.assertNotIn(_GOOG_OAUTH_ACCESS_PREFIX, data["error"])

    def test_invalid_json_becomes_failure_envelope(self):
        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        class _FakeClient:
            async def get(self, url, **kwargs):
                return _FakeResponse()

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("JSON", data["error"])


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromTestCase(TestRedactSecrets),
            loader.loadTestsFromTestCase(TestValidateInteractionId),
            loader.loadTestsFromTestCase(TestGeminiStartValidation),
            loader.loadTestsFromTestCase(TestGeminiResultTerminalStates),
            loader.loadTestsFromTestCase(TestPostGeminiInteractionHelper),
            loader.loadTestsFromTestCase(TestGetGeminiInteractionHelper),
        ]
    )
    sys.exit(0 if runner.run(suite).wasSuccessful() else 1)
