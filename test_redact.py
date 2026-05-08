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


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    """Install minimal stand-ins for mcp.* and openai so importing
    mcp_server does not require those packages or hit the Keychain."""

    class _FakeOpenAI:  # noqa: D401 - test stub
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

    async def _fake_stdio_server():  # not actually awaited in tests
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
            "mcp_server_under_test", SERVER_PATH
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


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRedactSecrets)
    sys.exit(0 if runner.run(suite).wasSuccessful() else 1)
