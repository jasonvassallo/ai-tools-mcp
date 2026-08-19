#!/usr/bin/env python3
"""Unit tests for the local_delegate tool family in mcp_server.py.

Self-contained: stubs out the third-party imports (mcp, openai, httpx,
google.auth) and the Keychain lookup so the test can import mcp_server
without needing the full runtime environment. Uses only stdlib
(unittest). Network paths are never exercised — endpoint resolution is
mock.patch.object'd, mirroring how test_redact.py and test_agent_research.py
treat network/keychain helpers.

Run:
    uv run --with pytest pytest test_local_delegate.py -q
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SERVER_PATH = HERE / "mcp_server.py"


async def _settle(predicate, timeout: float = 2.0) -> None:
    """Poll until predicate() is true or fail the test after `timeout`s.

    Replaces bare ``await asyncio.sleep(N)`` settle-waits for
    ``wait_for``-wrapped background tasks: fixed sleeps are either too
    short (flaky on a loaded CI box) or wastefully long. Polling on a
    tight interval settles as soon as the condition is met.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within settle timeout")


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """Return the dict of fake mcp/openai/httpx/google.auth modules used
    during import. Scoped via mock.patch.dict(sys.modules) so the fakes
    don't leak into other tests' imports (per PR #8 review)."""

    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

    class _FakeRequestError(Exception):
        pass

    class _FakeConnectError(_FakeRequestError):
        pass

    class _FakeTimeoutException(_FakeRequestError):
        pass

    class _FakeConnectTimeout(_FakeTimeoutException):
        pass

    class _FakeReadTimeout(_FakeTimeoutException):
        pass

    class _FakeRequestsException(Exception):
        pass

    class _FakeHTTPStatusError(Exception):
        def __init__(self, message="", *, request=None, response=None):
            super().__init__(message)
            self.request = request
            self.response = response

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
        "mcp": _make("mcp"),
        "mcp.server": _make("mcp.server", Server=_FakeServer),
        "mcp.server.stdio": _make("mcp.server.stdio", stdio_server=_fake_stdio_server),
        "mcp.types": _make("mcp.types", Tool=_Tool, TextContent=_TextContent),
        "httpx": _make(
            "httpx",
            AsyncClient=_FakeAsyncClient,
            HTTPStatusError=_FakeHTTPStatusError,
            RequestError=_FakeRequestError,
            ConnectError=_FakeConnectError,
            TimeoutException=_FakeTimeoutException,
            ConnectTimeout=_FakeConnectTimeout,
            ReadTimeout=_FakeReadTimeout,
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
    """Import mcp_server.py with third-party modules stubbed via a scoped
    sys.modules patch so the fakes don't leak into later test imports."""
    stubs = _build_stub_modules()
    fake_proc = types.SimpleNamespace(returncode=0, stdout="dummy-key\n")
    with mock.patch.dict(sys.modules, stubs):
        with mock.patch("subprocess.run", return_value=fake_proc):
            # Unique per-file module name — DO NOT CONSOLIDATE with the
            # other test files' spec names (see test_redact.py for the
            # full rationale: collision-proofing against future loader
            # changes that register the spec name in sys.modules).
            spec = importlib.util.spec_from_file_location(
                "mcp_server_under_test_local_delegate", SERVER_PATH
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            # OLLAMA_DELEGATE_MODELS is resolved from the environment at
            # import time; a developer/CI shell following the Windows
            # guidance may export AI_TOOLS_OLLAMA_MODELS, which would
            # replace the built-in allowlist this suite asserts against
            # (Codex P2, PR #22). Import under a scrubbed env; the
            # resolver tests exercise the override explicitly.
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AI_TOOLS_OLLAMA_MODELS", None)
                os.environ.pop("AI_TOOLS_OLLAMA_DEFAULT_MODEL", None)
                spec.loader.exec_module(module)
    return module


mcp_server = _load_mcp_server()


def _call(name: str, arguments: dict) -> list:
    return asyncio.run(mcp_server.call_tool(name, arguments))


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mcp_server.httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )

    def json(self):
        if self._json is None:
            raise ValueError("no JSON")
        return self._json


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls: list = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.response


def _with_client(client):
    return mock.patch.object(
        mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
    )


class TestOllamaAuthHeaders(unittest.TestCase):
    def _keychain(self, mapping):
        def fake(service, account):
            if service in mapping:
                return mapping[service]
            raise ValueError("not found")

        return mock.patch.object(
            mcp_server, "get_api_key_from_keychain", side_effect=fake
        )

    def test_localhost_needs_no_auth(self):
        for ep in ("http://localhost:11434", "http://127.0.0.1:11434"):
            self.assertEqual(mcp_server._ollama_auth_headers(ep), {})

    def test_remote_with_creds_gets_cf_access_headers(self):
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
            }
        ):
            headers = mcp_server._ollama_auth_headers("https://remote.example")
        self.assertEqual(
            headers,
            {"CF-Access-Client-Id": "id-123", "CF-Access-Client-Secret": "sec-456"},
        )

    def test_remote_missing_either_cred_returns_none(self):
        with self._keychain({"OLLAMA_CF_ACCESS_CLIENT_ID": "id-123"}):
            self.assertIsNone(mcp_server._ollama_auth_headers("https://remote.example"))
        with self._keychain({"OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456"}):
            self.assertIsNone(mcp_server._ollama_auth_headers("https://remote.example"))

    def test_remote_empty_client_id_returns_none(self):
        # A Keychain item can exist with an empty password: `security`
        # returns "" with returncode 0 (no ValueError). Must still fail
        # closed rather than send a malformed header.
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
            }
        ):
            self.assertIsNone(mcp_server._ollama_auth_headers("https://remote.example"))

    def test_remote_empty_client_secret_returns_none(self):
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "",
            }
        ):
            self.assertIsNone(mcp_server._ollama_auth_headers("https://remote.example"))

    def test_missing_security_binary_returns_none(self):
        # On non-macOS, `security` doesn't exist — subprocess.run raises
        # a miss (v1.2: the helper folds a missing security(1) into the
        # same ValueError as an ordinary miss). Cloudflare Access creds
        # are optional config here too, so the remote endpoint must be
        # skipped (None), not crash the whole delegate chain.
        with mock.patch.object(
            mcp_server,
            "get_api_key_from_keychain",
            side_effect=ValueError("Credential not found. Set the ..."),
        ):
            self.assertIsNone(mcp_server._ollama_auth_headers("https://remote.example"))

    def test_probe_sends_cf_headers_to_remote(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "AI_TOOLS_OLLAMA_URLS": "https://remote.example",
                    "AI_TOOLS_OLLAMA_URL": "",
                },
            ),
            self._keychain(
                {
                    "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                    "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
                }
            ),
        ):
            mcp_server._ollama_endpoint_cache.clear()
            client = _FakeTagsClient(tags_by_url={"https://remote.example": [_MODEL]})
            with mock.patch.object(
                mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
            ):
                endpoint = asyncio.run(mcp_server._select_ollama_endpoint(_MODEL))
        self.assertEqual(endpoint, "https://remote.example")
        _, kwargs = client.get_calls[0]
        self.assertEqual(kwargs["headers"]["CF-Access-Client-Id"], "id-123")

    def test_post_sends_cf_headers_and_never_leaks_secret_in_errors(self):
        with (
            self._keychain(
                {
                    "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                    "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
                }
            ),
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
            ),
        ):
            client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
            with mock.patch.object(
                mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
            ):
                out, _ = asyncio.run(
                    mcp_server._post_ollama_chat({"model": _MODEL}, 30.0)
                )
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("sec-456", out["error"])

    def test_http_error_body_echoing_secret_is_scrubbed(self):
        # An Access-gated host's 403 body can echo request headers verbatim.
        # redact_secrets has no CF-Access-token pattern, so the only backstop
        # is the value-aware scrub in _post_ollama_chat's HTTPStatusError
        # branch — assert it actually strips the live secret value.
        with (
            self._keychain(
                {
                    "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                    "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
                }
            ),
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
            ),
        ):
            client = _FakeClient(
                response=_FakeResponse(
                    status_code=403,
                    text="denied for CF-Access-Client-Secret: sec-456",
                )
            )
            with mock.patch.object(
                mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
            ):
                out, _ = asyncio.run(
                    mcp_server._post_ollama_chat({"model": _MODEL}, 30.0)
                )
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("sec-456", out["error"])
        self.assertIn("[REDACTED_CF_ACCESS]", out["error"])

    def test_http_error_scrub_survives_truncation_straddle(self):
        # The secret straddles the 500-char truncation cutoff: it starts at
        # index 495 and (being 7 chars) ends at index 502, i.e. the body[:500]
        # snippet contains only its first 5 chars ("sec-4"). If the scrub runs
        # AFTER truncation (substring match against the full secret), it never
        # finds the full value in the truncated snippet and the fragment
        # leaks. The scrub must run on the full body BEFORE truncation.
        with (
            self._keychain(
                {
                    "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                    "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
                }
            ),
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
            ),
        ):
            secret = "sec-456"
            filler_before = "a" * 495
            filler_after = "b" * (600 - len(filler_before) - len(secret))
            body = filler_before + secret + filler_after
            self.assertEqual(len(body), 600)
            client = _FakeClient(response=_FakeResponse(status_code=403, text=body))
            with mock.patch.object(
                mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
            ):
                out, _ = asyncio.run(
                    mcp_server._post_ollama_chat({"model": _MODEL}, 30.0)
                )
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("sec-456", out["error"])
        self.assertNotIn("sec-4", out["error"])


class TestPostOllamaChat(unittest.TestCase):
    def _with_selection(self, endpoint="http://localhost:11434"):
        return mock.patch.object(
            mcp_server,
            "_select_ollama_endpoint",
            mock.AsyncMock(return_value=endpoint),
        )

    def _post(self, client, payload=None, timeout_s=300.0):
        # The eviction warning is out of band; these tests are about the
        # body, and an unprotected call never has a warning to carry.
        with _with_client(client):
            data, warning = asyncio.run(
                mcp_server._post_ollama_chat(payload or {"model": "m"}, timeout_s)
            )
        self.assertEqual(warning, "")
        return data

    def test_happy_path_posts_to_api_chat_with_timeout(self):
        client = _FakeClient(
            response=_FakeResponse(json_data={"message": {"content": "hi"}})
        )
        with self._with_selection():
            out = self._post(
                client, payload={"model": "m", "stream": False}, timeout_s=42.0
            )
        self.assertEqual(out["message"]["content"], "hi")
        url, kwargs = client.calls[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(kwargs["timeout"], 42.0)
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertEqual(client.calls[0][1]["headers"], {})

    def test_connect_error_mentions_launchagent(self):
        client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
        with self._with_selection():
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("LaunchAgent", out["error"])
        self.assertIn("http://localhost:11434", out["error"])

    def test_connect_error_invalidates_implicit_resolution_cache(self):
        # Codex P2 (PR #32): the implicit cache pins (model, endpoint) pairs;
        # a dead endpoint must evict it too, or omitted-model calls keep
        # resolving to the corpse for up to a TTL.
        mcp_server._implicit_resolution_cache["gemma4:12b-nvfp4"] = (
            "qwen3.8:27b-nvfp4",
            "http://localhost:11434",
            "Note: substituted.\n\n",
            mcp_server.time.monotonic() + 60,
        )
        client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
        with self._with_selection():
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(mcp_server._implicit_resolution_cache, {})

    def test_404_adds_pull_hint(self):
        client = _FakeClient(
            response=_FakeResponse(status_code=404, text="model not found")
        )
        with self._with_selection():
            out = self._post(client, payload={"model": "qwen3.8:27b-nvfp4"})
        self.assertEqual(out["status"], "failed")
        self.assertIn("ollama pull qwen3.8:27b-nvfp4", out["error"])

    def test_non_404_http_error_no_pull_hint(self):
        client = _FakeClient(response=_FakeResponse(status_code=500, text="boom"))
        with self._with_selection():
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("ollama pull", out["error"])

    def test_non_json_200_is_failure_envelope(self):
        client = _FakeClient(response=_FakeResponse(json_data=None))
        with self._with_selection():
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("invalid JSON", out["error"])

    def test_selection_failure_is_failure_envelope(self):
        with mock.patch.object(
            mcp_server,
            "_select_ollama_endpoint",
            mock.AsyncMock(side_effect=ValueError("No Ollama endpoint serves 'm'")),
        ):
            client = _FakeClient(response=_FakeResponse(json_data={}))
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("No Ollama endpoint serves", out["error"])

    def test_connect_error_redacts_secret_in_url(self):
        # Assemble a JWT-shaped secret at runtime so scanners don't flag
        # this test. JWT pattern is eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}
        header = "ey" + "J" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQ6MTUxNjIzOTAyMn0"
        signature = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        jwt_token = f"{header}.{payload}.{signature}"
        # localhost host so the POST (and its ConnectError message) is
        # actually exercised — a remote host would be skipped for missing
        # credentials before the POST. The secret rides in the userinfo.
        url_with_secret = f"http://token:{jwt_token}@localhost:11434"
        with self._with_selection(endpoint=url_with_secret):
            client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        # The JWT token must NOT appear in the error message (redacted).
        self.assertNotIn(jwt_token, out["error"])
        # But "Ollama not running" and "LaunchAgent" must still be there.
        self.assertIn("Ollama not running", out["error"])
        self.assertIn("LaunchAgent", out["error"])
        # Verify redaction worked: should see [REDACTED_JWT] instead.
        self.assertIn("[REDACTED_JWT]", out["error"])

    def test_connect_error_drops_cache_entry(self):
        mcp_server._ollama_endpoint_cache["m"] = ("http://localhost:11434", 10**12)
        client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
        with self._with_selection():
            self._post(client, payload={"model": "m"})
        self.assertNotIn("m", mcp_server._ollama_endpoint_cache)


_MODEL = "qwen3.8:27b-nvfp4"


def _no_keychain(service, account):
    raise ValueError("not found")


def _no_security_binary(service, account):
    # v1.2: get_api_key_from_keychain folds a missing `security` CLI
    # (non-macOS) into the same actionable ValueError as an ordinary
    # miss — callers only ever see ValueError.
    raise ValueError(
        "Credential not found. Set the OLLAMA_URL environment variable ..."
    )


class TestResolveOllamaChain(unittest.TestCase):
    def _chain(self, env, keychain=_no_keychain):
        cleared = {k: "" for k in ("AI_TOOLS_OLLAMA_URLS", "AI_TOOLS_OLLAMA_URL")}
        with (
            mock.patch.dict(os.environ, {**cleared, **env}),
            mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
            ),
        ):
            return mcp_server._resolve_ollama_chain()

    def test_urls_env_is_ordered_chain(self):
        chain = self._chain(
            {"AI_TOOLS_OLLAMA_URLS": "http://localhost:11434/, https://mini.tail:443"}
        )
        self.assertEqual(chain, ["http://localhost:11434", "https://mini.tail:443"])

    def test_singular_env_compat_one_item(self):
        chain = self._chain({"AI_TOOLS_OLLAMA_URL": "http://localhost:11434"})
        self.assertEqual(chain, ["http://localhost:11434"])

    def test_default_chain_when_no_env(self):
        self.assertEqual(self._chain({}), list(mcp_server._OLLAMA_DEFAULT_CHAIN))

    def test_keychain_endpoint_appended(self):
        chain = self._chain(
            {"AI_TOOLS_OLLAMA_URLS": "http://localhost:11434"},
            keychain=lambda s, a: "https://kc.example",
        )
        self.assertEqual(chain, ["http://localhost:11434", "https://kc.example"])

    def test_duplicates_dropped_preserving_order(self):
        chain = self._chain(
            {"AI_TOOLS_OLLAMA_URLS": "http://localhost:11434,http://localhost:11434/"}
        )
        self.assertEqual(chain, ["http://localhost:11434"])

    def test_empty_entries_ignored(self):
        chain = self._chain({"AI_TOOLS_OLLAMA_URLS": "http://localhost:11434,,"})
        self.assertEqual(chain, ["http://localhost:11434"])

    def test_plain_http_remote_rejected(self):
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_OLLAMA_URLS": "http://remote.example:11434"})

    def test_embedded_credentials_rejected(self):
        # Credentials in the endpoint URL are never legitimate here — remote
        # auth is CF Access headers from the Keychain, not URL userinfo.
        # Embedded creds would otherwise flow into error messages / --check
        # stdout, and redact_secrets has no generic userinfo pattern.
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_OLLAMA_URLS": "http://user:pw@localhost:11434"})

    def test_embedded_credentials_password_not_leaked_in_error(self):
        # An arbitrary password has no secret "shape" redact_secrets can
        # match (unlike a JWT/API-key pattern), so the embedded-credentials
        # branch must never echo the raw url back — it must build a
        # display-safe form before formatting the error message.
        with self.assertRaises(ValueError) as ctx:
            self._chain(
                {"AI_TOOLS_OLLAMA_URLS": "http://user:hunter2-plain@localhost:11434"}
            )
        self.assertNotIn("hunter2-plain", str(ctx.exception))

    def test_invalid_scheme_with_userinfo_does_not_leak_password(self):
        # The scheme-rejection branch runs before the embedded-credentials
        # check, but it must still never echo a raw password back — it
        # shares the same display-safe url construction.
        with self.assertRaises(ValueError) as ctx:
            self._chain({"AI_TOOLS_OLLAMA_URLS": "ftp://user:hunter2-plain@host"})
        self.assertNotIn("hunter2-plain", str(ctx.exception))

    def test_garbage_url_rejected(self):
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_OLLAMA_URLS": "http://"})

    def test_missing_security_binary_degrades_gracefully(self):
        # On non-macOS, `security` doesn't exist at all; v1.2 folds that
        # into the same actionable ValueError as an ordinary miss.
        # The Keychain lookup here is optional config (an
        # extra chain entry), so the chain must still resolve from
        # env/default instead of the whole call crashing.
        chain = self._chain(
            {"AI_TOOLS_OLLAMA_URLS": "http://localhost:11434"},
            keychain=_no_security_binary,
        )
        self.assertEqual(chain, ["http://localhost:11434"])


class TestDelegateDefaultModel(unittest.TestCase):
    def test_base_tag_when_env_unset(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_DEFAULT_MODEL": ""}):
            self.assertEqual(
                mcp_server._delegate_default_model(),
                mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL,
            )

    def test_allowlisted_env_override_honored(self):
        tag = "qwen3.8:27b-nvfp4"
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_DEFAULT_MODEL": tag}):
            self.assertEqual(mcp_server._delegate_default_model(), tag)

    def test_non_allowlisted_env_falls_back_to_base(self):
        with mock.patch.dict(
            os.environ, {"AI_TOOLS_OLLAMA_DEFAULT_MODEL": "llama3:8b"}
        ):
            self.assertEqual(
                mcp_server._delegate_default_model(),
                mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL,
            )


class _FakeTagsClient:
    """Programmable fake for _select_ollama_endpoint probes."""

    def __init__(self, tags_by_url=None, exc_by_url=None, raw_json_by_url=None):
        self.tags_by_url = tags_by_url or {}
        self.exc_by_url = exc_by_url or {}
        # Overrides tags_by_url for a given base URL: returns this JSON body
        # verbatim (e.g. a bare list) instead of the {"models": [...]} shape,
        # to exercise probe responses that aren't a dict at the top level.
        self.raw_json_by_url = raw_json_by_url or {}
        self.get_calls: list = []

    async def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        base = url.removesuffix("/api/tags")
        if base in self.exc_by_url:
            raise self.exc_by_url[base]
        if base in self.raw_json_by_url:
            return _FakeResponse(json_data=self.raw_json_by_url[base])
        return _FakeResponse(
            json_data={"models": [{"name": t} for t in self.tags_by_url.get(base, [])]}
        )


class TestSelectOllamaEndpoint(unittest.TestCase):
    EP1 = "http://localhost:11434"
    EP2 = "http://127.0.0.1:11435"

    def setUp(self):
        mcp_server._ollama_endpoint_cache.clear()
        env = mock.patch.dict(
            os.environ,
            {
                "AI_TOOLS_OLLAMA_URLS": f"{self.EP1},{self.EP2}",
                "AI_TOOLS_OLLAMA_URL": "",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        kc = mock.patch.object(
            mcp_server, "get_api_key_from_keychain", side_effect=_no_keychain
        )
        kc.start()
        self.addCleanup(kc.stop)

    def _select(self, client, model=_MODEL):
        with mock.patch.object(
            mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
        ):
            return asyncio.run(mcp_server._select_ollama_endpoint(model))

    def test_picks_first_endpoint_with_tag(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [_MODEL], self.EP2: [_MODEL]})
        self.assertEqual(self._select(client), self.EP1)

    def test_skips_endpoint_missing_tag(self):
        client = _FakeTagsClient(
            tags_by_url={self.EP1: ["other:1b"], self.EP2: [_MODEL]}
        )
        self.assertEqual(self._select(client), self.EP2)

    def test_skips_unreachable_endpoint(self):
        client = _FakeTagsClient(
            tags_by_url={self.EP2: [_MODEL]},
            exc_by_url={self.EP1: mcp_server.httpx.ConnectError("refused")},
        )
        self.assertEqual(self._select(client), self.EP2)

    def test_all_miss_raises_naming_every_endpoint(self):
        client = _FakeTagsClient(
            tags_by_url={self.EP2: ["other:1b"]},
            exc_by_url={self.EP1: mcp_server.httpx.ConnectError("refused")},
        )
        with self.assertRaises(ValueError) as ctx:
            self._select(client)
        message = str(ctx.exception)
        self.assertIn(self.EP1, message)
        self.assertIn(self.EP2, message)
        self.assertIn("unreachable", message)
        self.assertIn("other:1b", message)

    def test_cache_prevents_reprobe_within_ttl(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [_MODEL]})
        self._select(client)
        calls_after_first = len(client.get_calls)
        self._select(client)
        self.assertEqual(len(client.get_calls), calls_after_first)

    def test_cache_expiry_reprobes(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [_MODEL]})
        with mock.patch.object(mcp_server, "_OLLAMA_PROBE_CACHE_TTL_S", 0.0):
            self._select(client)
            calls_after_first = len(client.get_calls)
            self._select(client)
        self.assertGreater(len(client.get_calls), calls_after_first)

    def test_remote_without_creds_is_skipped_with_reason(self):
        with mock.patch.dict(
            os.environ, {"AI_TOOLS_OLLAMA_URLS": "https://remote.example"}
        ):
            client = _FakeTagsClient(tags_by_url={"https://remote.example": [_MODEL]})
            with self.assertRaises(ValueError) as ctx:
                self._select(client)
        self.assertIn("skipped", str(ctx.exception))
        self.assertEqual(client.get_calls, [])  # never called bare

    def test_non_dict_tags_body_treated_as_no_models(self):
        # A probed endpoint can return valid JSON that isn't an object (e.g.
        # a bare list or string) — `.get("models", [])` on a non-dict blows
        # up with AttributeError, escaping the always-return-envelope
        # contract. Must be treated as "no models" and fall through to the
        # next endpoint instead of raising.
        client = _FakeTagsClient(
            raw_json_by_url={self.EP1: [1, 2]},
            tags_by_url={self.EP2: [_MODEL]},
        )
        self.assertEqual(self._select(client), self.EP2)


class _FakeShowClient(_FakeTagsClient):
    """_FakeTagsClient plus a programmable /api/show for capability reads."""

    def __init__(self, caps_by_model=None, show_exc=None, **kw):
        super().__init__(**kw)
        self.caps_by_model = caps_by_model or {}
        self.show_exc = show_exc
        self.post_calls: list = []

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.show_exc is not None:
            raise self.show_exc
        model = (kwargs.get("json") or {}).get("model", "")
        caps = self.caps_by_model.get(model)
        if caps is None:
            return _FakeResponse(json_data={})  # old Ollama: no capabilities key
        return _FakeResponse(json_data={"capabilities": list(caps)})


class TestResolveImplicitModel(unittest.TestCase):
    EP1 = "http://localhost:11434"
    EP2 = "http://127.0.0.1:11435"
    DEFAULT = "gemma4:12b-nvfp4"
    QWEN = "qwen3.8:27b-nvfp4"

    def setUp(self):
        mcp_server._ollama_endpoint_cache.clear()
        mcp_server._implicit_resolution_cache.clear()
        env = mock.patch.dict(
            os.environ,
            {
                "AI_TOOLS_OLLAMA_URLS": f"{self.EP1},{self.EP2}",
                "AI_TOOLS_OLLAMA_URL": "",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        kc = mock.patch.object(
            mcp_server, "get_api_key_from_keychain", side_effect=_no_keychain
        )
        kc.start()
        self.addCleanup(kc.stop)

    def _resolve(self, client):
        with mock.patch.object(
            mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
        ):
            return asyncio.run(mcp_server._resolve_implicit_model())

    def test_default_served_locally_no_note(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [self.DEFAULT, self.QWEN]})
        model, endpoint, note = self._resolve(client)
        self.assertEqual((model, endpoint, note), (self.DEFAULT, self.EP1, ""))

    def test_local_qwen_beats_remote_default(self):
        # The CWE-200 case from PR #32 review: gemma only remote, qwen local.
        # A local option must win so the prompt never leaves the machine.
        client = _FakeTagsClient(
            tags_by_url={self.EP1: [self.QWEN], self.EP2: [self.DEFAULT]}
        )
        model, endpoint, note = self._resolve(client)
        self.assertEqual(model, self.QWEN)
        self.assertEqual(endpoint, self.EP1)
        self.assertIn("default model gemma4:12b-nvfp4 is not served", note)

    def test_falls_through_to_later_endpoint(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [], self.EP2: [self.QWEN]})
        model, endpoint, note = self._resolve(client)
        self.assertEqual((model, endpoint), (self.QWEN, self.EP2))
        self.assertIn("using qwen3.8:27b-nvfp4", note)

    def test_nothing_served_anywhere_raises(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [], self.EP2: []})
        with self.assertRaises(ValueError) as ctx:
            self._resolve(client)
        self.assertIn("any allowlisted model", str(ctx.exception))

    def test_resolution_is_cached(self):
        client = _FakeTagsClient(tags_by_url={self.EP1: [self.DEFAULT]})
        self._resolve(client)
        first = len(client.get_calls)
        self._resolve(client)
        self.assertEqual(len(client.get_calls), first)  # served from cache


class TestModelCapabilities(unittest.TestCase):
    EP = "http://localhost:11434"

    def setUp(self):
        mcp_server._ollama_capability_cache.clear()
        kc = mock.patch.object(
            mcp_server, "get_api_key_from_keychain", side_effect=_no_keychain
        )
        kc.start()
        self.addCleanup(kc.stop)

    def _caps(self, client, model="gemma4:12b-nvfp4"):
        with mock.patch.object(
            mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
        ):
            return asyncio.run(mcp_server._model_capabilities(self.EP, model))

    def test_reports_capabilities(self):
        client = _FakeShowClient(
            caps_by_model={"gemma4:12b-nvfp4": ["completion", "tools", "thinking"]}
        )
        self.assertEqual(
            self._caps(client), frozenset({"completion", "tools", "thinking"})
        )

    def test_missing_capabilities_key_is_none(self):
        client = _FakeShowClient()  # /api/show returns {} (older Ollama)
        self.assertIsNone(self._caps(client))

    def test_transport_error_is_none(self):
        client = _FakeShowClient(show_exc=mcp_server.httpx.ConnectError("refused"))
        self.assertIsNone(self._caps(client))

    def test_result_is_cached(self):
        client = _FakeShowClient(caps_by_model={"gemma4:12b-nvfp4": ["thinking"]})
        self._caps(client)
        self._caps(client)
        self.assertEqual(len(client.post_calls), 1)


class TestThinkingModelAdvisory(unittest.TestCase):
    """Capability-driven `think` handling, measured — not assumed.

    2026-07-20: an earlier version hardcoded qwen-only thinking and shipped
    the claim "gemma4 is not a thinking model". /api/show says otherwise —
    BOTH built-in defaults report the 'thinking' capability. These tests pin
    the corrected behavior: capabilities come from the endpoint, a
    non-thinking tag gets `think` stripped plus an advisory (never Ollama's
    hard 400), and an indeterminate read changes nothing.
    """

    THINKING = frozenset({"completion", "tools", "thinking"})
    NO_THINKING = frozenset({"completion", "tools"})

    @staticmethod
    def _env(caps, resolved="gemma4:12b-nvfp4", note=""):
        """Patch endpoint resolution + capability lookup for one call."""
        ep = "http://127.0.0.1:11434"
        return (
            mock.patch.object(
                mcp_server,
                "_resolve_implicit_model",
                mock.AsyncMock(return_value=(resolved, ep, note)),
            ),
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value=ep),
            ),
            mock.patch.object(
                mcp_server,
                "_model_capabilities",
                mock.AsyncMock(return_value=caps),
            ),
        )

    def _delegate(
        self,
        args,
        caps,
        post=None,
        resolved="gemma4:12b-nvfp4",
        note="",
        queue_job_id=None,
    ):
        # _queue_submit is stubbed (default: no durable queue) so these
        # tests exercise the advisory plumbing, not queue reachability.
        fake = post or mock.AsyncMock(
            return_value=({"model": resolved, "message": {"content": "ok"}}, "")
        )
        r1, r2, r3 = self._env(caps, resolved=resolved, note=note)
        with (
            r1,
            r2,
            r3,
            mock.patch.object(mcp_server, "_post_ollama_chat", fake),
            mock.patch.object(
                mcp_server, "_queue_submit", mock.AsyncMock(return_value=queue_job_id)
            ),
        ):
            out = _call("local_delegate", args)
        return out, fake

    def test_thinking_capable_default_passes_think_through(self):
        # Regression pin for the shipped bug: gemma4 DOES think; think=true
        # on the default must produce NO advisory and stay in the payload.
        out, fake = self._delegate({"prompt": "hi", "think": True}, self.THINKING)
        self.assertTrue(out[0].text.startswith("## Local Delegate"))
        self.assertNotIn("Note:", out[0].text)
        payload, _ = fake.call_args.args
        self.assertIs(payload["think"], True)

    def test_non_thinking_model_strips_flag_and_advises(self):
        out, fake = self._delegate(
            {"prompt": "hi", "think": True, "model": "gemma4:12b-nvfp4"},
            self.NO_THINKING,
        )
        self.assertTrue(out[0].text.startswith("Note: think=true was disabled"))
        self.assertIn("## Local Delegate", out[0].text)
        payload, _ = fake.call_args.args
        self.assertIs(payload["think"], False)

    def test_indeterminate_capabilities_change_nothing(self):
        # /api/show unavailable → no advisory, payload untouched (fail
        # neutral; a wrong "flag ignored" note is worse than Ollama's error).
        out, fake = self._delegate({"prompt": "hi", "think": True}, None)
        self.assertNotIn("Note:", out[0].text)
        payload, _ = fake.call_args.args
        self.assertIs(payload["think"], True)

    def test_think_false_never_consults_capabilities(self):
        fake = mock.AsyncMock(
            return_value=(
                {"model": "gemma4:12b-nvfp4", "message": {"content": "ok"}},
                "",
            )
        )
        caps = mock.AsyncMock(return_value=self.THINKING)
        r1, _, _ = self._env(self.THINKING)
        with (
            r1,
            mock.patch.object(mcp_server, "_model_capabilities", caps),
            mock.patch.object(mcp_server, "_post_ollama_chat", fake),
        ):
            _call("local_delegate", {"prompt": "hi"})
        caps.assert_not_awaited()

    def test_implicit_substitution_note_reaches_answer(self):
        out, _ = self._delegate(
            {"prompt": "hi"},
            self.THINKING,
            resolved="qwen3.8:27b-nvfp4",
            note="Note: default model gemma4:12b-nvfp4 is not served by any "
            "reachable endpoint checked before localhost; using "
            "qwen3.8:27b-nvfp4 (localhost) instead.\n\n",
        )
        self.assertTrue(out[0].text.startswith("Note: default model"))
        self.assertIn("## Local Delegate", out[0].text)

    def test_explicit_model_bypasses_implicit_resolver(self):
        fake = mock.AsyncMock(
            return_value=(
                {"model": "gemma4:12b-nvfp4", "message": {"content": "ok"}},
                "",
            )
        )
        resolver = mock.AsyncMock()
        with (
            mock.patch.object(mcp_server, "_resolve_implicit_model", resolver),
            mock.patch.object(mcp_server, "_post_ollama_chat", fake),
        ):
            _call("local_delegate", {"prompt": "hi", "model": "gemma4:12b-nvfp4"})
        resolver.assert_not_awaited()

    def test_implicit_resolution_failure_is_an_error(self):
        resolver = mock.AsyncMock(side_effect=ValueError("no endpoint serves"))
        with mock.patch.object(mcp_server, "_resolve_implicit_model", resolver):
            out = _call("local_delegate", {"prompt": "hi"})
        self.assertTrue(out[0].text.startswith("Error:"))
        self.assertIn("no endpoint serves", out[0].text)

    def test_prefix_is_prepended_to_answer(self):
        out = mcp_server._render_delegate_answer(
            {"model": "gemma4:12b-nvfp4", "message": {"content": "answer"}},
            prefix="Note: think=true was disabled.\n\n",
        )
        self.assertTrue(out[0].text.startswith("Note: think=true was disabled."))
        self.assertIn("answer", out[0].text)

    def test_prefix_absent_by_default(self):
        out = mcp_server._render_delegate_answer(
            {"model": "gemma4:12b-nvfp4", "message": {"content": "answer"}}
        )
        self.assertTrue(out[0].text.startswith("## Local Delegate"))

    def test_both_advisories_compose(self):
        # Substituted default AND a non-thinking resolved model: both notes
        # must survive, implicit first, and the answer still renders.
        note = (
            "Note: default model gemma4:12b-nvfp4 is not served by any "
            "reachable endpoint checked before localhost; using "
            "qwen3.8:27b-nvfp4 (localhost) instead.\n\n"
        )
        out, fake = self._delegate(
            {"prompt": "hi", "think": True},
            self.NO_THINKING,
            resolved="qwen3.8:27b-nvfp4",
            note=note,
        )
        text = out[0].text
        self.assertTrue(text.startswith("Note: default model"))
        self.assertIn("think=true was disabled", text)
        self.assertLess(
            text.index("default model"), text.index("think=true was disabled")
        )
        self.assertIn("## Local Delegate", text)
        payload, _ = fake.call_args.args
        self.assertIs(payload["think"], False)

    def test_background_envelope_stays_parseable_json(self):
        """Advisories ride as a JSON field, never a prefix, on this path."""
        out, _ = self._delegate(
            {"prompt": "hi", "think": True, "background": True},
            self.NO_THINKING,
            resolved="gemma4:12b-nvfp4",
        )
        env = json.loads(out[0].text)  # must not raise
        self.assertIn("job_id", env)
        self.assertEqual(env["status"], "started")
        self.assertIn("think=true was disabled", env["warning"])
        # No queue reachable in this fixture: the fallback note composes
        # with the advisory in the same JSON field.
        self.assertIn("No durable queue endpoint reachable", env["warning"])

    def test_background_envelope_has_no_warning_key_when_not_needed(self):
        # A durable-queue submit with no advisory carries no warning at
        # all — the fallback note is only for the in-memory path. Model
        # passed explicitly: an implicit model on the queue path now
        # carries its own resolved-against-this-machine advisory.
        out, _ = self._delegate(
            {
                "prompt": "hi",
                "think": True,
                "background": True,
                "model": mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL,
            },
            self.THINKING,
            queue_job_id="q" + "c" * 32,
        )
        env = json.loads(out[0].text)
        self.assertNotIn("warning", env)
        self.assertEqual(env["queue"], "durable")

    def test_error_returns_are_not_prefixed(self):
        # Callers may pattern-match the "Error:" shape; keep it unpolluted.
        out = mcp_server._render_delegate_answer(
            {"status": "failed", "error": "boom"}, prefix="Note: ignored.\n\n"
        )
        self.assertTrue(out[0].text.startswith("Error"))
        self.assertNotIn("Note:", out[0].text)


class TestRenderDelegateAnswer(unittest.TestCase):
    def test_happy_path(self):
        out = mcp_server._render_delegate_answer(
            {"model": "qwen3.8:27b-nvfp4", "message": {"content": "answer"}}
        )
        self.assertIn("answer", out[0].text)
        self.assertIn("Local Delegate", out[0].text)

    def test_thinking_field_is_discarded(self):
        out = mcp_server._render_delegate_answer(
            {"model": "m", "message": {"content": "answer", "thinking": "scratchpad"}}
        )
        self.assertNotIn("scratchpad", out[0].text)

    def test_failure_envelope_surfaced(self):
        out = mcp_server._render_delegate_answer({"status": "failed", "error": "boom"})
        self.assertIn("Error", out[0].text)
        self.assertIn("boom", out[0].text)

    def test_failure_envelope_redacts_secrets(self):
        # Assemble a JWT-shaped secret at runtime so scanners don't flag
        # this test. JWT pattern is eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}
        header = "ey" + "J" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQ6MTUxNjIzOTAyMn0"
        signature = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        jwt_token = f"{header}.{payload}.{signature}"

        out = mcp_server._render_delegate_answer(
            {"status": "failed", "error": f"boom {jwt_token}"}
        )
        # Verify redaction worked: JWT must NOT appear, [REDACTED_JWT] must be present
        self.assertNotIn(jwt_token, out[0].text)
        self.assertIn("[REDACTED_JWT]", out[0].text)
        # Verify context survived: "Error" and "boom" must still be there
        self.assertIn("Error", out[0].text)
        self.assertIn("boom", out[0].text)

    def test_empty_content_is_error(self):
        out = mcp_server._render_delegate_answer({"message": {"content": ""}})
        self.assertIn("no content", out[0].text)

    def test_missing_message_is_error(self):
        out = mcp_server._render_delegate_answer({})
        self.assertIn("no content", out[0].text)


class TestDelegateJobs(unittest.TestCase):
    def setUp(self):
        mcp_server._delegate_jobs.clear()

    def test_lifecycle_start_running_collect_gone(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s, pre_unload=False):
                await gate.wait()
                return {"message": {"content": "done!"}}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                job_id = mcp_server._start_delegate_job({"model": "m"})
                running, _ = mcp_server._collect_delegate_job(job_id)
                self.assertEqual(running["status"], "running")
                self.assertIsInstance(running["elapsed_s"], int)
                gate.set()
                task = mcp_server._delegate_jobs[job_id]["task"]
                await _settle(task.done)  # let the wait_for-wrapped task finish
                done, _ = mcp_server._collect_delegate_job(job_id)
                self.assertEqual(done["message"]["content"], "done!")
                with self.assertRaises(ValueError):
                    mcp_server._collect_delegate_job(job_id)  # single-collect

        asyncio.run(scenario())

    def test_job_cap_rejects_fifth(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s, pre_unload=False):
                await gate.wait()
                return {}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                ids = [mcp_server._start_delegate_job({}) for _ in range(4)]
                with self.assertRaises(ValueError):
                    mcp_server._start_delegate_job({})
                tasks = [mcp_server._delegate_jobs[job_id]["task"] for job_id in ids]
                gate.set()
                await _settle(lambda: all(t.done() for t in tasks))
                for job_id in ids:  # drain so no pending tasks leak
                    mcp_server._collect_delegate_job(job_id)

        asyncio.run(scenario())

    def test_malformed_job_id_rejected(self):
        with self.assertRaises(ValueError):
            mcp_server._collect_delegate_job("not-a-job-id")

    def test_none_job_id_rejected(self):
        with self.assertRaises(ValueError):
            mcp_server._collect_delegate_job(None)

    def test_unknown_wellformed_job_id_rejected(self):
        with self.assertRaises(ValueError):
            mcp_server._collect_delegate_job("a" * 32)

    def test_timeout_result_is_failure_envelope(self):
        async def scenario():
            async def hang(payload, timeout_s, pre_unload=False):
                await asyncio.sleep(3600)

            with mock.patch.object(mcp_server, "_post_ollama_chat", hang):
                with mock.patch.object(mcp_server, "_DELEGATE_BG_CEILING_S", 0.01):
                    job_id = mcp_server._start_delegate_job({})
                    await asyncio.sleep(0.05)
                    out, warning = mcp_server._collect_delegate_job(job_id)
                    self.assertEqual(out["status"], "failed")
                    self.assertIn("ceiling", out["error"])
                    # No eviction result exists for a job cancelled before it
                    # ever returned one.
                    self.assertEqual(warning, "")

        asyncio.run(scenario())

    def test_completed_jobs_beyond_retention_are_evicted(self):
        # Completed-but-never-collected jobs must not accumulate forever.
        # Start well more than _DELEGATE_DONE_RETAINED jobs, let each
        # complete instantly, never call local_delegate_result on any of
        # them, and confirm the registry stays bounded — retaining only
        # the newest window — instead of growing unboundedly.
        async def scenario():
            async def fake_post(payload, timeout_s, pre_unload=False):
                return {"message": {"content": "done"}}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                with mock.patch.object(mcp_server, "_DELEGATE_DONE_RETAINED", 3):
                    retained = mcp_server._DELEGATE_DONE_RETAINED
                    total = retained + 5
                    ids = []
                    for _ in range(total):
                        job_id = mcp_server._start_delegate_job({"model": "m"})
                        task = mcp_server._delegate_jobs[job_id]["task"]
                        await _settle(task.done)
                        ids.append(job_id)

                    survivors = mcp_server._delegate_jobs
                    self.assertLessEqual(len(survivors), retained + 1)
                    # Newest jobs survive...
                    for jid in ids[-(retained + 1) :]:
                        self.assertIn(jid, survivors)
                    # ...oldest ones were evicted.
                    for jid in ids[: total - (retained + 1)]:
                        self.assertNotIn(jid, survivors)

        asyncio.run(scenario())

    def test_eviction_swallows_cancelled_task_exception_retrieval(self):
        # The eviction path calls task.exception() to mark any exception
        # as retrieved (avoiding asyncio's "exception was never
        # retrieved" warning). For a *cancelled* task, task.exception()
        # itself raises CancelledError (asyncio semantics) instead of
        # returning — eviction must swallow that, not let it escape
        # _start_delegate_job. A Mock stands in for the cancelled task so
        # this is deterministic (no real cancellation race).
        async def scenario():
            cancelled_task = mock.Mock()
            cancelled_task.done.return_value = True
            cancelled_task.exception.side_effect = asyncio.CancelledError()
            stale_job_id = "a" * 32
            mcp_server._delegate_jobs[stale_job_id] = {
                "task": cancelled_task,
                "started": time.monotonic() - 100,
            }

            async def fake_post(payload, timeout_s, pre_unload=False):
                return {"message": {"content": "done"}}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                with mock.patch.object(mcp_server, "_DELEGATE_DONE_RETAINED", 0):
                    # Retention of 0 forces immediate eviction of the
                    # pre-seeded cancelled entry; must not raise.
                    job_id = mcp_server._start_delegate_job({"model": "m"})
                    task = mcp_server._delegate_jobs[job_id]["task"]
                    await _settle(task.done)

            self.assertNotIn(stale_job_id, mcp_server._delegate_jobs)
            cancelled_task.exception.assert_called_once()

        asyncio.run(scenario())


class TestToolListing(unittest.TestCase):
    def _tools(self):
        tools = asyncio.run(mcp_server.list_tools())
        return {t.name: t for t in tools}

    def test_both_tools_listed(self):
        by_name = self._tools()
        self.assertIn("local_delegate", by_name)
        self.assertIn("local_delegate_result", by_name)

    def test_prompt_required_and_model_enum_matches_allowlist(self):
        schema = self._tools()["local_delegate"].inputSchema
        self.assertEqual(schema["required"], ["prompt"])
        self.assertEqual(
            schema["properties"]["model"]["enum"],
            list(mcp_server.OLLAMA_DELEGATE_MODELS),
        )

    def test_result_requires_job_id(self):
        schema = self._tools()["local_delegate_result"].inputSchema
        self.assertEqual(schema["required"], ["job_id"])


class TestLocalDelegateValidation(unittest.TestCase):
    def test_missing_prompt(self):
        out = _call("local_delegate", {})
        self.assertIn("prompt", out[0].text)

    def test_empty_prompt(self):
        out = _call("local_delegate", {"prompt": "   "})
        self.assertIn("prompt", out[0].text)

    def test_model_not_in_allowlist(self):
        out = _call("local_delegate", {"prompt": "x", "model": "llama3:8b"})
        self.assertIn("qwen3.8:27b-nvfp4", out[0].text)

    def test_think_must_be_bool(self):
        out = _call("local_delegate", {"prompt": "x", "think": "yes"})
        self.assertIn("think", out[0].text)

    def test_background_must_be_bool(self):
        out = _call("local_delegate", {"prompt": "x", "background": "yes"})
        self.assertIn("background", out[0].text)

    def test_keep_alive_pattern(self):
        for bad in ("5 m", "-1m", "10d", "", "99999s", "5m; rm -rf /"):
            out = _call("local_delegate", {"prompt": "x", "keep_alive": bad})
            self.assertIn("keep_alive", out[0].text, msg=bad)

    def test_timeout_bounds_and_bool_rejection(self):
        for bad in (0, -5, 601, True, "300"):
            out = _call("local_delegate", {"prompt": "x", "timeout_s": bad})
            self.assertIn("timeout_s", out[0].text, msg=repr(bad))

    def test_system_must_be_string(self):
        out = _call("local_delegate", {"prompt": "x", "system": 42})
        self.assertIn("system", out[0].text)


def _stub_resolution(case: unittest.TestCase, caps=frozenset({"thinking"})):
    """Stub endpoint/capability resolution for handler-level delegate tests.

    Without this, tests that omit `model` (or set think=true) would probe the
    real endpoint chain — green on a dev box running Ollama, red in CI. The
    stubs mirror the healthy-localhost case: default model resolves locally
    with no advisory, capabilities report 'thinking'.
    """
    ep = "http://127.0.0.1:11434"
    for target, repl in (
        (
            "_resolve_implicit_model",
            mock.AsyncMock(
                return_value=(mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL, ep, "")
            ),
        ),
        ("_select_ollama_endpoint", mock.AsyncMock(return_value=ep)),
        ("_model_capabilities", mock.AsyncMock(return_value=caps)),
    ):
        patcher = mock.patch.object(mcp_server, target, repl)
        patcher.start()
        case.addCleanup(patcher.stop)


class TestLocalDelegateSync(unittest.TestCase):
    def setUp(self):
        _stub_resolution(self)

    def test_payload_construction_defaults(self):
        fake = mock.AsyncMock(
            return_value=({"model": "m", "message": {"content": "ok"}}, "")
        )
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            out = _call("local_delegate", {"prompt": "do the thing"})
        payload, timeout_s = fake.call_args.args
        self.assertEqual(payload["model"], mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL)
        self.assertEqual(
            payload["messages"], [{"role": "user", "content": "do the thing"}]
        )
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], False)
        self.assertNotIn(
            "keep_alive", payload
        )  # gemma default: omitted → inherit server OLLAMA_KEEP_ALIVE
        self.assertEqual(timeout_s, 300.0)
        self.assertIn("ok", out[0].text)

    def test_keep_alive_zero_default_tag_matching(self):
        # PR #60 review findings: the env-overridable allowlist accepts
        # arbitrary tags, so the qwen match must survive namespacing and
        # casing rather than a bare whole-tag prefix check.
        applies = mcp_server._keep_alive_zero_default_applies
        for tag in (
            "qwen3.8:27b-nvfp4",
            "qwen3.6:27b-coding-nvfp4-64k",
            "Qwen3:latest",
            "QWEN2.5-coder:7b",
            "acme/qwen3:latest",
            "hf.co/acme/qwen-model",
        ):
            self.assertTrue(applies(tag), msg=tag)
        for tag in (
            "gemma4:12b-nvfp4",
            "acme/gemma:2b",
            "hf.co/qwenteam/gemma-x",  # qwen in namespace, not model name
        ):
            self.assertFalse(applies(tag), msg=tag)

    def test_qwen_defaults_keep_alive_zero(self):
        # Contamination mitigation: a resident qwen runner returns other
        # prompts' answers on repeat calls; omitted keep_alive → "0".
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {"prompt": "p", "model": "qwen3.8:27b-nvfp4"},
            )
        payload, _ = fake.call_args.args
        self.assertEqual(payload["keep_alive"], "0")

    def test_qwen_explicit_keep_alive_wins_over_default(self):
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {
                    "prompt": "p",
                    "model": "qwen3.8:27b-nvfp4",
                    "keep_alive": "5m",
                },
            )
        payload, _ = fake.call_args.args
        self.assertEqual(payload["keep_alive"], "5m")

    def test_implicitly_resolved_qwen_gets_keep_alive_zero(self):
        # The qwen default must key off the FINAL model: an omitted-model call
        # resolves via _resolve_implicit_model, which may pick a qwen tag.
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        resolver = mock.AsyncMock(
            return_value=(
                "qwen3.8:27b-nvfp4",
                "http://127.0.0.1:11434",
                "",
            )
        )
        with (
            mock.patch.object(mcp_server, "_resolve_implicit_model", resolver),
            mock.patch.object(mcp_server, "_post_ollama_chat", fake),
        ):
            _call("local_delegate", {"prompt": "p"})
        payload, _ = fake.call_args.args
        self.assertEqual(payload["model"], "qwen3.8:27b-nvfp4")
        self.assertEqual(payload["keep_alive"], "0")

    def test_qwen_default_also_requests_a_pre_unload(self):
        # keep_alive is a POST-response TTL, so the "0" default does not
        # protect the call carrying it: measured 2026-08-08 on JVMBPro, a
        # keep_alive:0 call landing on an already-resident dirty qwen runner
        # is contaminated at the same rate as an unprotected one. The
        # eviction is what puts the call on a fresh runner.
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {"prompt": "p", "model": "qwen3.8:27b-nvfp4"},
            )
        self.assertTrue(fake.call_args.kwargs["pre_unload"])

    def test_pre_unload_follows_the_effective_zero_ttl(self):
        # Protection keys off the EFFECTIVE keep_alive, not how it was chosen.
        # An explicit "0" MUST still be protected: commands/local-delegate.md
        # tells callers to pass keep_alive="0" for the long-context qwen
        # route, so gating on "we defaulted it" left that documented path
        # unprotected (Codex P1 + Gemini, PR #65). Explicit non-zero values
        # still opt out so deliberate warm-pinning works.
        for ka, expect_pre_unload in (("5m", False), ("0", True), ("1h", False)):
            fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
            with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
                _call(
                    "local_delegate",
                    {
                        "prompt": "p",
                        "model": "qwen3.8:27b-nvfp4",
                        "keep_alive": ka,
                    },
                )
            self.assertEqual(
                fake.call_args.kwargs["pre_unload"], expect_pre_unload, msg=ka
            )
            # The caller's explicit value is still what reaches Ollama.
            self.assertEqual(fake.call_args.args[0]["keep_alive"], ka, msg=ka)

    def test_explicit_zero_on_non_qwen_stays_unprotected(self):
        # The zero TTL alone must not trigger an eviction on an immune model.
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {"prompt": "p", "model": "gemma4:12b-nvfp4", "keep_alive": "0"},
            )
        self.assertFalse(fake.call_args.kwargs["pre_unload"])

    def test_non_qwen_model_gets_no_pre_unload(self):
        # gemma is immune to the contamination (0/141 lifetime) — it must not
        # pay a reload it does not need.
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call("local_delegate", {"prompt": "p", "model": "gemma4:12b-nvfp4"})
        self.assertFalse(fake.call_args.kwargs["pre_unload"])

    def test_payload_with_system_think_keepalive_timeout(self):
        fake = mock.AsyncMock(return_value=({"message": {"content": "ok"}}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {
                    "prompt": "p",
                    "system": "you are terse",
                    "think": True,
                    "keep_alive": "0",
                    "timeout_s": 600,
                    "model": "qwen3.8:27b-nvfp4",
                },
            )
        payload, timeout_s = fake.call_args.args
        self.assertEqual(
            payload["messages"][0], {"role": "system", "content": "you are terse"}
        )
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "p"})
        self.assertIs(payload["think"], True)
        self.assertEqual(payload["keep_alive"], "0")
        self.assertEqual(payload["model"], "qwen3.8:27b-nvfp4")
        self.assertEqual(timeout_s, 600.0)

    def test_failure_envelope_reaches_caller(self):
        fake = mock.AsyncMock(return_value=({"status": "failed", "error": "down"}, ""))
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            out = _call("local_delegate", {"prompt": "p"})
        self.assertIn("down", out[0].text)


class TestLocalDelegateBackground(unittest.TestCase):
    def setUp(self):
        mcp_server._delegate_jobs.clear()
        _stub_resolution(self)
        # No durable queue in this fixture — these tests pin the legacy
        # in-memory fallback path (TestLocalDelegateQueue covers the
        # queue-first path).
        queue = mock.patch.object(
            mcp_server, "_queue_submit", mock.AsyncMock(return_value=None)
        )
        queue.start()
        self.addCleanup(queue.stop)

    def test_background_returns_job_id_then_result_collects(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s, pre_unload=False):
                await gate.wait()
                return {"model": "m", "message": {"content": "bg answer"}}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                started = await mcp_server.call_tool(
                    "local_delegate", {"prompt": "p", "background": True}
                )
                envelope = json.loads(started[0].text)
                self.assertEqual(envelope["status"], "started")
                job_id = envelope["job_id"]

                running = await mcp_server.call_tool(
                    "local_delegate_result", {"job_id": job_id}
                )
                self.assertIn("running", running[0].text)

                gate.set()
                await asyncio.sleep(0.05)
                done = await mcp_server.call_tool(
                    "local_delegate_result", {"job_id": job_id}
                )
                self.assertIn("bg answer", done[0].text)

        asyncio.run(scenario())

    def test_background_qwen_payload_gets_keep_alive_zero(self):
        # The qwen keep_alive default is applied before the payload forks into
        # the background job path, so background calls are protected too.
        starter = mock.Mock(return_value="job-1")
        with mock.patch.object(mcp_server, "_start_delegate_job", starter):
            _call(
                "local_delegate",
                {
                    "prompt": "p",
                    "model": "qwen3.8:27b-nvfp4",
                    "background": True,
                },
            )
        (payload,) = starter.call_args.args
        self.assertEqual(payload["keep_alive"], "0")
        # ...and so is the eviction: a background job runs on the same shared
        # runner and is exposed to the same first-call gap.
        self.assertTrue(starter.call_args.kwargs["pre_unload"])

    def test_cap_error_is_clean_text(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s, pre_unload=False):
                await gate.wait()
                return {}, ""

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                ids = []
                for _ in range(4):
                    started = await mcp_server.call_tool(
                        "local_delegate", {"prompt": "p", "background": True}
                    )
                    ids.append(json.loads(started[0].text)["job_id"])
                fifth = await mcp_server.call_tool(
                    "local_delegate", {"prompt": "p", "background": True}
                )
                self.assertIn("cap", fifth[0].text)
                gate.set()
                await asyncio.sleep(0.05)
                for job_id in ids:
                    await mcp_server.call_tool(
                        "local_delegate_result", {"job_id": job_id}
                    )

        asyncio.run(scenario())

    def test_result_unknown_id_is_clean_error(self):
        out = _call("local_delegate_result", {"job_id": "b" * 32})
        self.assertIn("Error", out[0].text)

    def test_result_missing_id_is_clean_error(self):
        out = _call("local_delegate_result", {})
        self.assertIn("Error", out[0].text)


class TestEvictOllamaRunner(unittest.TestCase):
    """The pre-unload that makes keep_alive:"0" protect its OWN call.

    Measured 2026-08-08 on JVMBPro: a keep_alive:0 request landing on an
    already-resident, already-dirty qwen runner is contaminated at the same
    rate as an unprotected one, because keep_alive only sets the
    post-response TTL. Evicting first is the part that actually works.
    """

    _EP = "http://127.0.0.1:11434"
    _LOG = "ai_tools_mcp.delegate.evict"

    def setUp(self):
        # _evict_stats is module-global and the *_total assertions below are
        # absolute, so every test starts from a known count.
        mcp_server._evict_stats.update(ok=0, absent=0, failed=0)
        # Captured BEFORE the mute below, because one test needs the handler
        # that really writes in order to prove where the write happens.
        self._real_handlers = list(mcp_server._evict_log.handlers)
        # Mute the module's real stderr handler for the duration of each test:
        # it is the production channel, and letting it fire here interleaves
        # warning lines with unittest's own output. It must be a NullHandler
        # rather than an empty list — with no handlers AND propagate=False,
        # logging falls back to its lastResort handler, which writes to
        # stderr anyway. assertLogs/assertNoLogs swap logger.handlers
        # wholesale, so the assertions below are unaffected either way.
        muted = mock.patch.object(
            mcp_server._evict_log, "handlers", [logging.NullHandler()]
        )
        muted.start()
        self.addCleanup(muted.stop)

    @staticmethod
    def _stderr_sink(handler):
        """The handler that actually writes: with a queue in front, the
        listener's; without one, the handler itself."""
        listener = getattr(handler, "listener", None)
        return handler if listener is None else listener.handlers[0]

    @staticmethod
    def _unloaded():
        """What a REAL eviction answers: 200, done_reason "unload" — the same
        shape a PULLED-but-not-resident tag returns for its ~22 ms no-op. An
        UN-PULLED tag answers 404 instead; see the 404 test below."""
        return _FakeResponse(json_data={"done_reason": "unload"})

    def _client(self, evict=None, chat=None):
        """Two-leg fake: the eviction and the chat answer differently, which
        the single-response _FakeClient cannot express. Pass an Exception as
        `evict` to make that leg raise."""
        evict = self._unloaded() if evict is None else evict
        chat = _FakeResponse(json_data={"ok": True}) if chat is None else chat

        class _TwoLeg:
            def __init__(self):
                self.calls: list = []

            async def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                resp = evict if url.endswith("/api/generate") else chat
                if isinstance(resp, Exception):
                    raise resp
                return resp

        return _TwoLeg()

    def _run_chat(self, client, pre_unload):
        """Returns the (body, evict_warning) pair _post_ollama_chat hands
        back. The warning rides BESIDE the body, never inside it."""
        with (
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value=self._EP),
            ),
            _with_client(client),
        ):
            return asyncio.run(
                mcp_server._post_ollama_chat(
                    {"model": _MODEL}, 30.0, pre_unload=pre_unload
                )
            )

    def _remote_auth(self, headers):
        """Patch selection + credentials onto a remote Access-gated host."""
        return (
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
            ),
            mock.patch.object(
                mcp_server, "_ollama_auth_headers", mock.Mock(return_value=headers)
            ),
        )

    def test_evicts_before_the_chat_on_the_same_endpoint(self):
        client = self._client()
        out, warning = self._run_chat(client, pre_unload=True)
        # A PROVEN eviction annotates nothing: the body is byte-identical to
        # the pre-change one and there is no warning to carry.
        self.assertEqual(out, {"ok": True})
        self.assertEqual(warning, "")
        urls = [url for url, _ in client.calls]
        # Order is the whole point: evicting AFTER the chat would be the
        # no-op this change exists to fix.
        self.assertEqual(urls, [f"{self._EP}/api/generate", f"{self._EP}/api/chat"])
        body = client.calls[0][1]["json"]
        self.assertEqual(body["model"], _MODEL)
        self.assertEqual(body["keep_alive"], 0)
        # Empty prompt is Ollama's unload idiom — a non-empty one would
        # bill a whole generation just to evict.
        self.assertEqual(body["prompt"], "")

    def test_no_eviction_when_not_requested(self):
        client = _FakeClient(response=_FakeResponse(json_data={"ok": True}))
        self._run_chat(client, pre_unload=False)
        self.assertEqual([url for url, _ in client.calls], [f"{self._EP}/api/chat"])

    def test_eviction_failure_never_fails_the_callers_request(self):
        # A wedged/absent eviction must degrade to today's behaviour, not
        # turn a working delegate call into an error.
        client = self._client(evict=mcp_server.httpx.ConnectError("refused"))
        out, warning = self._run_chat(client, pre_unload=True)
        self.assertEqual(out["ok"], True)
        self.assertEqual(len(client.calls), 2)
        # ...but it is no longer SILENT.
        self.assertTrue(warning)

    def test_transport_failure_is_reported(self):
        # The connection never landed, so nothing was evicted — and the class
        # name is the whole reason string: no exception TEXT, which on a
        # request error can echo the URL and its userinfo.
        _, warning = self._run_chat(
            self._client(evict=mcp_server.httpx.ConnectError("refused")),
            pre_unload=True,
        )
        self.assertIn("request failed", warning)
        self.assertIn("ConnectError", warning)
        self.assertNotIn("refused", warning)

    def test_eviction_is_bounded_by_the_callers_timeout(self):
        # timeout_s is the documented ceiling for the whole delegate call, so
        # the eviction must be spent out of that budget, never added on top.
        # A slow eviction makes the deduction observable: if its cost were
        # added on top, the chat would still be handed the full 30s.
        class _SlowEvict:
            def __init__(self):
                self.calls: list = []

            async def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if url.endswith("/api/generate"):
                    await asyncio.sleep(0.2)
                return _FakeResponse(json_data={"ok": True})

        client = _SlowEvict()
        self._run_chat(client, pre_unload=True)
        evict_timeout = client.calls[0][1]["timeout"]
        chat_timeout = client.calls[1][1]["timeout"]
        self.assertLessEqual(evict_timeout, 30.0)
        self.assertLess(chat_timeout, 30.0 - 0.15)

    def test_short_caller_timeout_neither_starves_nor_overruns(self):
        # timeout_s may be as low as 1. The chat must still get a usable
        # slice, but never MORE than the caller asked for — the floor is
        # clamped to the ceiling.
        client = _FakeClient(response=_FakeResponse(json_data={"ok": True}))
        with (
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value=self._EP),
            ),
            _with_client(client),
        ):
            _, warning = asyncio.run(
                mcp_server._post_ollama_chat({"model": _MODEL}, 1.0, pre_unload=True)
            )
        # timeout_s=1 cannot afford an eviction on top of the chat's slice, so
        # the eviction is skipped entirely and the chat keeps the full budget.
        self.assertEqual([url for url, _ in client.calls], [f"{self._EP}/api/chat"])
        chat_timeout = client.calls[0][1]["timeout"]
        self.assertGreater(chat_timeout, 0.0)
        self.assertLessEqual(chat_timeout, 1.0)
        # A skipped eviction leaves the call just as unprotected as a failed
        # one, so it is reported the same way rather than passing for success.
        self.assertIn("skipped", warning)

    def test_total_budget_is_bounded_by_the_callers_timeout(self):
        # The regression CodeRabbit asked for: a SLOW eviction must not let
        # the two legs together exceed timeout_s.
        #
        # The bound is (time the eviction actually consumed) + (budget handed
        # to the chat) — NOT the sum of the two budgets, which double-counts
        # eviction budget that goes unused when the eviction returns early.
        # The pre-remediation code failed this at timeout_s=1: it spent up to
        # 1s evicting and then restored the chat to a full 1s floor.
        for caller_timeout in (1.0, 6.0, 30.0):

            class _SlowEvict:
                def __init__(self):
                    self.calls: list = []
                    self.evict_elapsed = 0.0

                async def post(self, url, **kwargs):
                    self.calls.append((url, kwargs))
                    if url.endswith("/api/generate"):
                        started = time.monotonic()
                        await asyncio.sleep(0.2)
                        self.evict_elapsed = time.monotonic() - started
                    return _FakeResponse(json_data={"ok": True})

            client = _SlowEvict()
            with (
                mock.patch.object(
                    mcp_server,
                    "_select_ollama_endpoint",
                    mock.AsyncMock(return_value=self._EP),
                ),
                _with_client(client),
            ):
                asyncio.run(
                    mcp_server._post_ollama_chat(
                        {"model": _MODEL}, caller_timeout, pre_unload=True
                    )
                )
            chat_budget = next(
                kwargs["timeout"]
                for url, kwargs in client.calls
                if url.endswith("/api/chat")
            )
            self.assertLessEqual(
                client.evict_elapsed + chat_budget,
                caller_timeout + 1e-9,
                msg=f"timeout_s={caller_timeout}",
            )

    def test_hung_eviction_is_cut_off_at_its_budget(self):
        # httpx's float timeout is per-phase, so it cannot bound total wall
        # clock on its own; asyncio.wait_for is what enforces evict_budget.
        # A fake client that ignores the httpx timeout entirely (as a wedged
        # connection effectively would) must still be cut off, and the chat
        # must still run.
        class _HungEvict:
            def __init__(self):
                self.calls: list = []

            async def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if url.endswith("/api/generate"):
                    await asyncio.sleep(60)  # never completes within budget
                return _FakeResponse(json_data={"ok": True})

        client = _HungEvict()
        started = time.monotonic()
        with (
            mock.patch.object(mcp_server, "_OLLAMA_EVICT_TIMEOUT_S", 0.3),
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value=self._EP),
            ),
            _with_client(client),
        ):
            out, warning = asyncio.run(
                mcp_server._post_ollama_chat({"model": _MODEL}, 30.0, pre_unload=True)
            )
        elapsed = time.monotonic() - started
        # Cut off near the 0.3s budget, nowhere near the 60s hang...
        self.assertLess(elapsed, 5.0)
        # ...and the caller's real request still went out and succeeded...
        self.assertEqual(out["ok"], True)
        self.assertEqual(client.calls[-1][0], f"{self._EP}/api/chat")
        # ...while the cut-off eviction is reported, not swallowed.
        self.assertIn("timed out", warning)

    def test_eviction_carries_the_same_auth_headers_as_the_chat(self):
        # A remote Access-gated endpoint would 403 the eviction — and a
        # silently-403ing eviction is an unprotected call that looks fine.
        client = _FakeClient(response=_FakeResponse(json_data={"ok": True}))
        select, auth = self._remote_auth({"CF-Access-Client-Id": "id-123"})
        with select, auth, _with_client(client):
            asyncio.run(
                mcp_server._post_ollama_chat({"model": _MODEL}, 30.0, pre_unload=True)
            )
        evict_headers = client.calls[0][1]["headers"]
        chat_headers = client.calls[1][1]["headers"]
        self.assertEqual(evict_headers, chat_headers)
        self.assertEqual(evict_headers["CF-Access-Client-Id"], "id-123")

    def test_rejected_eviction_is_reported_not_swallowed(self):
        # The live-demonstrated failure: a host that 403s /api/generate (stale
        # Access token) and 200s /api/chat used to hand back a clean answer
        # with no sign the protection never ran. A 5xx is the same silent
        # class. (404 has its own meaning; see the un-pulled-tag test.)
        for code in (403, 500):
            with self.subTest(code=code):
                client = self._client(
                    evict=_FakeResponse(json_data={}, status_code=code)
                )
                out, warning = self._run_chat(client, pre_unload=True)
                self.assertEqual(out["ok"], True)
                self.assertIn(str(code), warning)

    def test_an_unpulled_tag_is_named_not_dressed_up_as_a_broken_mitigation(self):
        # Measured twice on Ollama 0.32.7: an UN-PULLED tag answers HTTP 404
        # {"error":"model ... not found"} — NOT the 200/done_reason:"unload"
        # no-op a pulled-but-not-resident tag gives. Nothing was resident, so
        # nothing could be contaminated, and the chat that follows fails on
        # its own 404. Still reported (a proxy can 404 /api/generate while
        # serving /api/chat), but as an operating condition rather than a
        # broken mitigation an operator has to go chase.
        client = self._client(evict=_FakeResponse(json_data={}, status_code=404))
        with self.assertLogs(self._LOG, level="INFO") as logs:
            out, warning = self._run_chat(client, pre_unload=True)
        self.assertEqual(out["ok"], True)
        self.assertIn("not pulled", warning)
        line = logs.output[0]
        self.assertTrue(line.startswith("INFO:"), msg=line)
        self.assertIn("result=NO-RUNNER", line)
        self.assertEqual(mcp_server._evict_stats["absent"], 1)
        self.assertEqual(mcp_server._evict_stats["failed"], 0)

    def test_two_hundred_that_did_not_unload_is_reported(self):
        # The subtlest case: HTTP 200, well-formed body, but the runner is
        # still resident. Only the body distinguishes it from a real eviction.
        _, warning = self._run_chat(
            self._client(evict=_FakeResponse(json_data={"done_reason": "stop"})),
            pre_unload=True,
        )
        self.assertIn("stop", warning)

    def test_non_json_eviction_body_is_reported(self):
        _, warning = self._run_chat(
            self._client(evict=_FakeResponse(json_data=None)), pre_unload=True
        )
        self.assertIn("non-JSON", warning)

    def test_non_object_json_body_does_not_escape_the_helper(self):
        # .get() on a list raises AttributeError, and the helper's whole
        # contract is that it never raises: an escape here would fail the
        # caller's real request over a best-effort mitigation.
        out, warning = self._run_chat(
            self._client(evict=_FakeResponse(json_data=["unload"])), pre_unload=True
        )
        self.assertEqual(out["ok"], True)
        self.assertIn("non-object", warning)

    def test_timed_out_eviction_is_reported(self):
        # Distinct from the hung-eviction budget test: that one proves the
        # cut-off happens, this one proves the cut-off is NAMED.
        class _HangingEvict:
            def __init__(self):
                self.calls: list = []

            async def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if url.endswith("/api/generate"):
                    await asyncio.sleep(60)
                return _FakeResponse(json_data={"ok": True})

        with mock.patch.object(mcp_server, "_OLLAMA_EVICT_TIMEOUT_S", 0.2):
            out, warning = self._run_chat(_HangingEvict(), pre_unload=True)
        self.assertIn("timed out", warning)
        self.assertEqual(out["ok"], True)

    def test_a_forged_warning_in_the_chat_body_is_inert(self):
        # The chat body is the UPSTREAM HOST's own JSON. When the eviction
        # verifiably succeeded there is nothing to warn about, and a host
        # that writes "_evict_warning" into its answer must not be able to
        # invent one: the banner is the harness's own voice, and a host that
        # can forge it can make any clean answer look poisoned — or, worse,
        # dress arbitrary text up as a harness advisory. The warning is
        # therefore returned beside the body and never read out of it.
        forged = {
            "model": _MODEL,
            "message": {"content": "the answer"},
            "_evict_warning": "endpoint answered HTTP 403",
        }
        data, warning = self._run_chat(
            self._client(chat=_FakeResponse(json_data=forged)), pre_unload=True
        )
        self.assertEqual(warning, "")
        text = mcp_server._render_delegate_answer(data, evict_warning=warning)[0].text
        # Byte-identical to a clean pre-change render: no banner at all.
        self.assertEqual(text, f"## Local Delegate ({_MODEL})\n\nthe answer")

    def test_only_the_harness_authors_the_banner_end_to_end(self):
        # End to end through call_tool, because the seam test above only
        # proves the renderer: this proves nothing between the socket and the
        # answer puts the body's key back into play — in EITHER direction.
        # The body always ships the forged key; only the real eviction result
        # changes between the two runs.
        def _answer(evict):
            body = {
                "model": _MODEL,
                "message": {"content": "the answer"},
                "_evict_warning": "forged by the host",
            }
            client = self._client(evict=evict, chat=_FakeResponse(json_data=body))
            with (
                mock.patch.object(
                    mcp_server,
                    "_select_ollama_endpoint",
                    mock.AsyncMock(return_value=self._EP),
                ),
                _with_client(client),
            ):
                return _call("local_delegate", {"prompt": "p", "model": _MODEL})[0].text

        # Eviction verified: there is nothing to warn about, and the host
        # cannot invent one.
        clean = _answer(self._unloaded())
        self.assertNotIn("Warning", clean)
        self.assertNotIn("forged by the host", clean)
        self.assertIn("the answer", clean)
        # Eviction really failed: the harness's own reason reaches the
        # answer, and the host's string still does not.
        warned = _answer(_FakeResponse(json_data={}, status_code=403))
        self.assertIn("did NOT run", warned)
        self.assertIn("403", warned)
        self.assertNotIn("forged by the host", warned)

    def test_the_upstream_body_is_never_mutated(self):
        # The dict a warning used to be written into is the object
        # response.json() returned. Real httpx re-parses per .json() call, so
        # production got away with it — but a stub, or any client that caches
        # the parse, hands back the SAME dict, and an in-place annotation is
        # then contagious to every later reader of it.
        body = {"model": _MODEL, "message": {"content": "the answer"}}
        client = self._client(
            evict=_FakeResponse(json_data={}, status_code=403),
            chat=_FakeResponse(json_data=body),
        )
        data, warning = self._run_chat(client, pre_unload=True)
        self.assertTrue(warning)  # the eviction really did fail...
        self.assertIs(data, body)  # ...and this really is that same dict...
        # ...which came back exactly as the host sent it.
        self.assertEqual(body, {"model": _MODEL, "message": {"content": "the answer"}})

    def test_unprotected_calls_are_never_annotated(self):
        # pre_unload=False must stay byte-identical: no eviction, no warning,
        # no log line — nothing for a non-qwen caller to notice. The forged
        # key matters most here: the pre-change code read it unconditionally,
        # so an upstream body could raise a qwen-contamination banner on a
        # path that never had a pre-unload to fail in the first place.
        forged = {
            "model": _MODEL,
            "message": {"content": "the answer"},
            "_evict_warning": "endpoint answered HTTP 403",
        }
        client = self._client(
            evict=_FakeResponse(json_data={}, status_code=403),
            chat=_FakeResponse(json_data=forged),
        )
        with self.assertNoLogs(self._LOG):
            data, warning = self._run_chat(client, pre_unload=False)
        self.assertEqual([url for url, _ in client.calls], [f"{self._EP}/api/chat"])
        self.assertEqual(warning, "")
        self.assertEqual(mcp_server._evict_stats, {"ok": 0, "absent": 0, "failed": 0})
        text = mcp_server._render_delegate_answer(data, evict_warning=warning)[0].text
        self.assertEqual(text, f"## Local Delegate ({_MODEL})\n\nthe answer")

    def test_skipped_eviction_is_reported_too(self):
        # timeout_s=1 cannot afford an eviction, so the call runs
        # unprotected — the same state a failed eviction leaves, and it says so.
        client = self._client()
        with (
            mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value=self._EP),
            ),
            _with_client(client),
            self.assertLogs(self._LOG, level="WARNING") as logs,
        ):
            _, warning = asyncio.run(
                mcp_server._post_ollama_chat({"model": _MODEL}, 1.0, pre_unload=True)
            )
        # No eviction was even attempted...
        self.assertEqual([url for url, _ in client.calls], [f"{self._EP}/api/chat"])
        # ...and both channels say so.
        self.assertIn("skipped", warning)
        self.assertIn("ollama-preunload", logs.output[0])
        self.assertIn("result=FAILED", logs.output[0])

    def test_eviction_never_raises_out_of_the_helper(self):
        # Constraint 1: whatever the endpoint does, the helper returns an
        # outcome. Every arm it declares is exercised here, so a future edit
        # that drops one fails rather than escaping into the caller.
        cases = [
            ("verified unload", _FakeResponse(json_data={"done_reason": "unload"})),
            ("200 but resident", _FakeResponse(json_data={"done_reason": "stop"})),
            ("non-JSON body", _FakeResponse(json_data=None)),
            ("non-object JSON body", _FakeResponse(json_data=["unload"])),
            ("HTTP 500", _FakeResponse(json_data={}, status_code=500)),
            ("HTTP 404", _FakeResponse(json_data={}, status_code=404)),
            ("connect error", mcp_server.httpx.ConnectError("refused")),
            ("request error", mcp_server.httpx.RequestError("boom")),
            (
                "status error carrying a response",
                mcp_server.httpx.HTTPStatusError(
                    "boom",
                    request=None,
                    response=_FakeResponse(json_data={}, status_code=503),
                ),
            ),
            # httpx builds these WITH a response, but nothing here guarantees
            # the instance this arm catches has one — and reading
            # exc.response.status_code off a response-less error would raise
            # AttributeError straight out of the never-raises helper, making
            # "never raises" conditional on someone else's constructor.
            (
                "status error with no response",
                mcp_server.httpx.HTTPStatusError("boom", request=None, response=None),
            ),
        ]
        for label, case in cases:
            with self.subTest(case=label):
                client = self._client(evict=case)
                result = asyncio.run(
                    mcp_server._evict_ollama_runner(client, self._EP, _MODEL, {}, 5.0)
                )
                self.assertIsInstance(result, mcp_server._EvictOutcome)
                self.assertIsInstance(result.reason, str)

    def test_warning_never_carries_header_values(self):
        # The eviction sends Cloudflare Access service tokens; neither a 403
        # whose BODY echoes them nor a 200 whose done_reason IS one may turn
        # the call's own credentials into output — on the warning, the
        # rendered answer, or the log line.
        secret = "sec-456"
        evictions = [
            _FakeResponse(
                json_data={},
                status_code=403,
                text=f"CF-Access-Client-Secret: {secret}",
            ),
            # The remote-controlled fragment: done_reason is echoed into the
            # reason string, so a hostile/misconfigured body is the one way a
            # header value could round-trip back out.
            _FakeResponse(json_data={"done_reason": secret}),
        ]
        for evict in evictions:
            with self.subTest(status=evict.status_code):
                client = self._client(
                    evict=evict,
                    chat=_FakeResponse(
                        json_data={
                            "model": _MODEL,
                            "message": {"content": "the answer"},
                        }
                    ),
                )
                select, auth = self._remote_auth(
                    {
                        "CF-Access-Client-Id": "id-123",
                        "CF-Access-Client-Secret": secret,
                    }
                )
                with (
                    select,
                    auth,
                    _with_client(client),
                    self.assertLogs(self._LOG, level="WARNING") as logs,
                ):
                    data, warning = asyncio.run(
                        mcp_server._post_ollama_chat(
                            {"model": _MODEL}, 30.0, pre_unload=True
                        )
                    )
                rendered = mcp_server._render_delegate_answer(
                    data, evict_warning=warning
                )[0].text
                for surface in (warning, rendered, "\n".join(logs.output)):
                    self.assertNotIn(secret, surface)
                    self.assertNotIn("id-123", surface)

    def test_a_long_done_reason_is_capped(self):
        # done_reason is remote-controlled and lands on BOTH surfaces:
        # uncapped, a hostile host writes as much as it likes into the
        # operator's log line and onto the answer the caller reads.
        _, warning = self._run_chat(
            self._client(evict=_FakeResponse(json_data={"done_reason": "z" * 5000})),
            pre_unload=True,
        )
        self.assertLess(len(warning), 100)
        self.assertNotIn("z" * 60, warning)

    def test_done_reason_is_scrubbed_before_it_is_capped(self):
        # The straddle: 35 filler chars then the secret, so the 40-char cap
        # cuts through it. Scrub-then-cap redacts the whole value first and
        # the cap only ever sees "[REDA…". Cap-then-scrub has nothing left to
        # match the full value against and leaves "sec-4" in the clear —
        # exactly the ordering bug the /api/chat body scrub guards at 500.
        secret = "sec-456-and-more"
        client = self._client(
            evict=_FakeResponse(json_data={"done_reason": "a" * 35 + secret})
        )
        select, auth = self._remote_auth({"CF-Access-Client-Secret": secret})
        with select, auth, _with_client(client):
            _, warning = asyncio.run(
                mcp_server._post_ollama_chat({"model": _MODEL}, 30.0, pre_unload=True)
            )
        self.assertNotIn(secret, warning)
        self.assertNotIn("sec-4", warning)

    def test_failed_eviction_is_logged_even_when_the_chat_fails(self):
        # The case the banner CANNOT cover, and the whole reason the record is
        # written at eviction time: the chat fails, so the caller gets a bare
        # error. The stderr line is then the only surviving evidence that the
        # mitigation is broken.
        client = self._client(
            evict=_FakeResponse(json_data={}, status_code=403),
            chat=_FakeResponse(json_data={}, status_code=500),
        )
        with self.assertLogs(self._LOG, level="WARNING") as logs:
            out, _ = self._run_chat(client, pre_unload=True)
        self.assertEqual(out["status"], "failed")
        self.assertIn("ollama-preunload", logs.output[0])
        self.assertIn("result=FAILED", logs.output[0])
        self.assertIn("403", logs.output[0])

    def test_verified_unload_logs_ok(self):
        with self.assertLogs(self._LOG, level="INFO") as logs:
            self._run_chat(self._client(), pre_unload=True)
        line = logs.output[0]
        self.assertIn("ollama-preunload", line)
        self.assertIn("result=OK", line)
        self.assertIn("ok_total=1", line)
        self.assertIn("fail_total=0", line)
        # A localhost endpoint is named as such rather than echoed, so the
        # log never becomes a place a remote URL's userinfo could land.
        self.assertIn("endpoint=localhost", line)

    def test_running_totals_count_every_outcome_class(self):
        # One line in a log nobody greps is easy to miss; the running totals
        # are what turn a mitigation that has been broken for weeks into
        # something visible on any single later call.
        self._run_chat(self._client(), pre_unload=True)
        self._run_chat(
            self._client(evict=_FakeResponse(json_data={}, status_code=404)),
            pre_unload=True,
        )
        with self.assertLogs(self._LOG, level="WARNING") as logs:
            self._run_chat(
                self._client(evict=_FakeResponse(json_data={}, status_code=403)),
                pre_unload=True,
            )
        self.assertEqual(mcp_server._evict_stats, {"ok": 1, "absent": 1, "failed": 1})
        # ...and the line an operator actually reads carries all three.
        line = logs.output[0]
        self.assertIn("ok_total=1", line)
        self.assertIn("absent_total=1", line)
        self.assertIn("fail_total=1", line)

    def test_log_line_is_single_line(self):
        # The host writes one JSON record per stderr LINE, so a newline in
        # the message silently fragments the record into unparseable halves.
        with self.assertLogs(self._LOG, level="INFO") as logs:
            self._run_chat(
                self._client(evict=_FakeResponse(json_data={"done_reason": "a\nb"})),
                pre_unload=True,
            )
        for record in logs.records:
            self.assertNotIn("\n", record.getMessage())

    def test_the_stderr_write_never_happens_on_the_calling_thread(self):
        # This is called from the asyncio event loop. A StreamHandler
        # write+flush into a stderr pipe the host has stopped draining blocks
        # the WHOLE loop — every other MCP request with it — for as long as
        # the pipe stays full, outside every timeout this module sets. With a
        # queue in front, emit() is an enqueue and the block lands on the
        # listener's thread instead. Driven through the module's REAL handler
        # chain (restored for this test only), not a stand-in for it.
        release = threading.Event()
        self.addCleanup(release.set)
        seen = threading.Event()

        def _blocking_emit(record):
            release.wait(1.5)
            seen.set()

        self.assertTrue(self._real_handlers, "module handler chain missing")
        handler = self._real_handlers[0]
        sink = self._stderr_sink(handler)
        with (
            mock.patch.object(mcp_server._evict_log, "handlers", [handler]),
            mock.patch.object(sink, "emit", _blocking_emit),
        ):
            started = time.monotonic()
            self._run_chat(
                self._client(evict=_FakeResponse(json_data={}, status_code=403)),
                pre_unload=True,
            )
            elapsed = time.monotonic() - started
            # Let the blocked sink finish inside the patch, so the real
            # handler never gets the record and never writes to stderr.
            release.set()
            seen.wait(2.0)
        # The record really did reach the sink — otherwise "fast" would only
        # mean nothing was logged at all.
        self.assertTrue(seen.is_set())
        self.assertLess(elapsed, 0.5)


class TestEvictLoggerSetup(unittest.TestCase):
    """stdout is the MCP protocol stream, so this logger has to stay on
    stderr — including when it already exists. logging.getLogger() is
    process-global, so "we got here first" is not something this module can
    assume."""

    _LOG = "ai_tools_mcp.delegate.evict"

    def test_a_preconfigured_logger_is_still_isolated_from_the_root(self):
        log = logging.getLogger(self._LOG)
        saved_handlers = list(log.handlers)
        saved_propagate = log.propagate
        saved_level = log.level

        def _restore():
            log.handlers[:] = saved_handlers
            log.propagate = saved_propagate
            log.setLevel(saved_level)

        self.addCleanup(_restore)
        # A host, the SDK, or an earlier import of this module got here
        # first, so the "no handlers yet" guard will skip. Anything that
        # sheltered inside that guard skips with it — and propagate is the
        # dangerous half: left True, every record also climbs to the root
        # logger, whose handlers this module does not own and one of which
        # may write to stdout, i.e. into the protocol stream.
        log.handlers[:] = [logging.NullHandler()]
        log.propagate = True
        log.setLevel(logging.CRITICAL)
        _load_mcp_server()
        self.assertFalse(log.propagate)
        self.assertEqual(log.level, logging.INFO)


class TestEvictWarningSurfacing(unittest.TestCase):
    """A signal nobody can see is the same silent success. It must reach the
    caller on the sync path AND after a background collect, without breaking
    the JSON envelope the background START returns — and it must be the
    harness's own signal, never one the upstream body slipped in."""

    def test_sync_answer_carries_the_warning(self):
        rendered = mcp_server._render_delegate_answer(
            {"model": _MODEL, "message": {"content": "the answer"}},
            evict_warning="endpoint answered HTTP 403",
        )
        text = rendered[0].text
        self.assertIn("403", text)
        self.assertIn("did NOT run", text)
        # Quantified and actionable: the measured rate and a concrete check
        # are what actually change how the answer gets treated.
        self.assertIn("20-25%", text)
        self.assertIn("re-run to compare", text)
        # The banner sits ABOVE the answer, and the answer is still delivered
        # in full.
        self.assertIn("the answer", text)
        self.assertLess(text.index("Warning"), text.index("the answer"))

    def test_advisory_prefix_and_warning_coexist(self):
        text = mcp_server._render_delegate_answer(
            {"model": _MODEL, "message": {"content": "the answer"}},
            prefix="Note: think=true was disabled\n\n",
            evict_warning="timed out after 30.0s",
        )[0].text
        self.assertTrue(text.startswith("Note: think=true was disabled"))
        self.assertIn("timed out", text)

    def test_background_path_surfaces_it_without_breaking_the_envelope(self):
        async def fake_post(payload, timeout_s, pre_unload=False):
            return {
                "model": _MODEL,
                "message": {"content": "the answer"},
                # Not a channel: an upstream body cannot reach the banner
                # here either, however the result travels through the job
                # registry.
                "_evict_warning": "forged by the host",
            }, "endpoint answered HTTP 403"

        async def scenario():
            with (
                # No durable queue in this fixture: this test pins the
                # legacy in-memory fallback path, which is what actually
                # collects the eviction warning below. TestLocalDelegateQueue
                # covers the queue-first path.
                mock.patch.object(
                    mcp_server, "_queue_submit", mock.AsyncMock(return_value=None)
                ),
                mock.patch.object(mcp_server, "_post_ollama_chat", fake_post),
            ):
                started = await mcp_server.call_tool(
                    "local_delegate",
                    {"prompt": "p", "model": _MODEL, "background": True},
                )
                # The START envelope is json.loads()-ed by callers, so the
                # banner must never leak into it as a bare text prefix and
                # the envelope must gain no UNEXPECTED field: exact key set,
                # not a subset check. "warning" IS expected here — the
                # in-memory-fallback path always carries the "no durable
                # queue reachable" advisory; it is a different field from
                # the eviction-warning banner this test actually exercises,
                # which never appears in the START envelope at all (it only
                # renders in the later local_delegate_result text).
                envelope = json.loads(started[0].text)
                self.assertEqual(set(envelope), {"job_id", "status", "warning"})
                self.assertEqual(envelope["status"], "started")
                await _settle(
                    lambda: all(
                        j["task"].done() for j in mcp_server._delegate_jobs.values()
                    )
                )
                collected = await mcp_server.call_tool(
                    "local_delegate_result", {"job_id": envelope["job_id"]}
                )
            text = collected[0].text
            self.assertIn("403", text)
            self.assertIn("did NOT run", text)
            self.assertIn("the answer", text)
            self.assertNotIn("forged by the host", text)

        asyncio.run(scenario())


class TestRunCheckOllamaLine(unittest.TestCase):
    def _run_check_output(
        self, get_side_effect=None, json_version="0.9.0", vault=None, resolver=None
    ):
        fake_resp = mock.Mock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.json.return_value = {"version": json_version}
        fake_requests = types.SimpleNamespace(
            get=mock.Mock(return_value=fake_resp, side_effect=get_side_effect),
            RequestException=Exception,
        )
        self._last_fake_get = fake_requests.get

        def fake_resolve(service, account):
            # Perplexity key resolves; OLLAMA_URL absent so the chain comes
            # from the env var alone (a Keychain URL of "k" would fail
            # endpoint validation and mask what this test targets).
            # Patching _resolve_credential rather than
            # get_api_key_from_keychain covers both call paths —
            # get_api_key_from_keychain is now a thin wrapper over it.
            if service == "OLLAMA_URL":
                raise ValueError("not found")
            return "k", "env"

        # The vault reader is stubbed by default so run_check's output is
        # identical on every platform — otherwise a developer running the
        # suite on Windows with real CF credentials vaulted gets extra
        # divergence-warning lines that CI never sees.
        buf = io.StringIO()
        with mock.patch.object(mcp_server, "requests", fake_requests, create=True):
            with mock.patch.object(
                mcp_server,
                "_read_windows_credential",
                side_effect=vault or (lambda target: None),
            ):
                with mock.patch.object(
                    mcp_server,
                    "_resolve_credential",
                    side_effect=resolver or fake_resolve,
                ):
                    with mock.patch.object(
                        mcp_server, "_load_adc", side_effect=ValueError("no adc")
                    ):
                        with mock.patch.dict(
                            os.environ,
                            {
                                "AI_TOOLS_OLLAMA_URL": "http://localhost:11434",
                                "AI_TOOLS_OLLAMA_URLS": "",
                            },
                        ):
                            with contextlib.redirect_stdout(buf):
                                with self.assertRaises(SystemExit) as ctx:
                                    mcp_server.run_check()
        return buf.getvalue(), ctx.exception.code

    def test_reachable_prints_ok(self):
        out, code = self._run_check_output()
        self.assertIn("ok: ollama reachable at http://localhost:11434", out)
        self.assertEqual(code, 1)  # only the forced ADC failure counts

    def test_unreachable_prints_warn_not_fail(self):
        out, code = self._run_check_output(get_side_effect=Exception("refused"))
        self.assertIn("warn: ollama not reachable at http://localhost:11434", out)
        self.assertEqual(code, 1)  # ollama down did NOT add to errors

    def test_bad_env_default_model_warns(self):
        with mock.patch.dict(
            os.environ, {"AI_TOOLS_OLLAMA_DEFAULT_MODEL": "llama3:8b"}
        ):
            out, code = self._run_check_output()
        self.assertIn("not in", out)  # allowlist warn line
        self.assertIn("llama3:8b", out)
        self.assertEqual(code, 1)

    def test_probe_disables_redirects(self):
        # A CF Access service-token header must never follow a redirect
        # off-host — same rationale as the shared httpx client's
        # follow_redirects=False. requests.get needs the equivalent
        # allow_redirects=False on the --check probe.
        self._run_check_output()
        self.assertEqual(self._last_fake_get.call_count, 1)
        _, kwargs = self._last_fake_get.call_args
        self.assertEqual(kwargs.get("allow_redirects"), False)


class TestLoadADC(unittest.TestCase):
    def test_uses_credential_quota_project_when_default_project_is_missing(self):
        credentials = types.SimpleNamespace(quota_project_id="quota-project")

        with mock.patch.object(
            mcp_server.google.auth,
            "default",
            return_value=(credentials, None),
        ):
            loaded_credentials, project = mcp_server._load_adc()

        self.assertIs(loaded_credentials, credentials)
        self.assertEqual(project, "quota-project")

    def test_rejects_adc_without_default_or_quota_project(self):
        credentials = types.SimpleNamespace(quota_project_id=None)

        with (
            mock.patch.object(
                mcp_server.google.auth,
                "default",
                return_value=(credentials, None),
            ),
            self.assertRaisesRegex(ValueError, "billing project"),
        ):
            mcp_server._load_adc()


class TestCredentialResolution(unittest.TestCase):
    """v1.2 (issue #20): env-first credential lookup, Keychain fallback."""

    def test_env_override_wins_without_touching_keychain(self):
        with mock.patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pk-env"}):
            with mock.patch.object(mcp_server.subprocess, "run") as run:
                self.assertEqual(
                    mcp_server.get_api_key_from_keychain("api_tokens", "perplexity"),
                    "pk-env",
                )
                run.assert_not_called()

    def test_generic_env_name_is_the_service_name(self):
        with mock.patch.dict(os.environ, {"OLLAMA_CF_ACCESS_CLIENT_ID": "cid-env"}):
            with mock.patch.object(mcp_server.subprocess, "run") as run:
                self.assertEqual(
                    mcp_server.get_api_key_from_keychain(
                        "OLLAMA_CF_ACCESS_CLIENT_ID", "jasonvassallo"
                    ),
                    "cid-env",
                )
                run.assert_not_called()

    def test_blank_env_is_ignored_and_falls_through(self):
        ok = mock.Mock(returncode=0, stdout="from-keychain\n")
        with mock.patch.dict(os.environ, {"PERPLEXITY_API_KEY": "   "}):
            with mock.patch.object(mcp_server.subprocess, "run", return_value=ok):
                self.assertEqual(
                    mcp_server.get_api_key_from_keychain("api_tokens", "perplexity"),
                    "from-keychain",
                )

    def test_missing_everywhere_error_names_the_env_var(self):
        miss = mock.Mock(returncode=1, stdout="")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PERPLEXITY_API_KEY", None)
            with mock.patch.object(mcp_server.subprocess, "run", return_value=miss):
                with self.assertRaises(ValueError) as ctx:
                    mcp_server.get_api_key_from_keychain("api_tokens", "perplexity")
        self.assertIn("PERPLEXITY_API_KEY", str(ctx.exception))

    def test_non_macos_no_security_binary_degrades_to_same_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_URL", None)
            with (
                mock.patch.object(
                    mcp_server.subprocess, "run", side_effect=FileNotFoundError
                ),
                self.assertRaises(ValueError) as ctx,
            ):
                mcp_server.get_api_key_from_keychain("OLLAMA_URL", "u")
        self.assertIn("OLLAMA_URL", str(ctx.exception))


# ─── Windows Credential Manager tier ──────────────────────────────────

_CF_ID = "OLLAMA_CF_ACCESS_CLIENT_ID"
_CF_SECRET = "OLLAMA_CF_ACCESS_CLIENT_SECRET"
_CF_ID_TARGET = "ai-tools-mcp-cf-access/client-id"
_CF_SECRET_TARGET = "ai-tools-mcp-cf-access/client-secret"

# Fakes only — never a real credential shape, never a real value.
_FAKE_ENV_ID = "fake-env-client-id"
_FAKE_VAULT_ID = "fake-vault-client-id"
_FAKE_VAULT_SECRET = "fake-vault-client-secret"


@contextlib.contextmanager
def _no_cf_env():
    """Clear both CF env vars so the vault tier is what's under test."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_CF_ID, None)
        os.environ.pop(_CF_SECRET, None)
        yield


def _vault(mapping):
    """Stub _read_windows_credential from a {target: value} dict."""
    return lambda target: mapping.get(target)


class TestWindowsCredentialVaultTier(unittest.TestCase):
    """Windows Credential Manager as the middle credential tier.

    The vault tier exists because Claude Desktop launches the packaged
    .mcpb outside any shell — it inherits no profile, so a plaintext env
    var was the only thing that reached it. Env stays FIRST so existing
    installs are untouched and the cutover stays reversible.
    """

    def test_env_var_wins_over_vault(self):
        vault = mock.Mock(side_effect=_vault({_CF_ID_TARGET: _FAKE_VAULT_ID}))
        with mock.patch.dict(os.environ, {_CF_ID: _FAKE_ENV_ID}):
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                with mock.patch.object(mcp_server.subprocess, "run") as run:
                    value, source = mcp_server._resolve_credential(_CF_ID, "u")
        self.assertEqual(value, _FAKE_ENV_ID)
        self.assertEqual(source, "env")
        # Precedence must SHORT-CIRCUIT, not just win a comparison: the
        # vault is never consulted and the Keychain never shelled out to.
        vault.assert_not_called()
        run.assert_not_called()

    def test_vault_used_when_env_absent(self):
        vault = _vault({_CF_ID_TARGET: _FAKE_VAULT_ID})
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                with mock.patch.object(mcp_server.subprocess, "run") as run:
                    value, source = mcp_server._resolve_credential(_CF_ID, "u")
        self.assertEqual(value, _FAKE_VAULT_ID)
        self.assertEqual(source, "windows-credential-manager")
        run.assert_not_called()  # answered before the Keychain tier

    def test_both_cf_credentials_resolve_from_vault(self):
        vault = _vault(
            {_CF_ID_TARGET: _FAKE_VAULT_ID, _CF_SECRET_TARGET: _FAKE_VAULT_SECRET}
        )
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                headers = mcp_server._ollama_auth_headers("https://ollama.example.com")
        self.assertEqual(
            headers,
            {
                "CF-Access-Client-Id": _FAKE_VAULT_ID,
                "CF-Access-Client-Secret": _FAKE_VAULT_SECRET,
            },
        )

    def test_blank_env_falls_through_to_vault(self):
        # Whitespace is not a credential — same fail-closed rule the env
        # tier already applied before the Keychain.
        vault = _vault({_CF_ID_TARGET: _FAKE_VAULT_ID})
        with mock.patch.dict(os.environ, {_CF_ID: "   "}):
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                value, source = mcp_server._resolve_credential(_CF_ID, "u")
        self.assertEqual(value, _FAKE_VAULT_ID)
        self.assertEqual(source, "windows-credential-manager")

    def test_empty_vault_entry_is_treated_as_absent(self):
        # CredReadW succeeds with CredentialBlobSize 0 for an entry stored
        # with an empty secret — must not be mistaken for a credential.
        ok = mock.Mock(returncode=0, stdout="from-keychain\n")
        with (
            _no_cf_env(),
            mock.patch.object(
                mcp_server, "_read_windows_credential", _vault({_CF_ID_TARGET: ""})
            ),
            mock.patch.object(mcp_server.subprocess, "run", return_value=ok),
        ):
            value, source = mcp_server._resolve_credential(_CF_ID, "u")
        self.assertEqual(value, "from-keychain")
        self.assertEqual(source, "macos-keychain")

    def test_missing_everywhere_names_both_remedies(self):
        miss = mock.Mock(returncode=1, stdout="")
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", _vault({})):
                with mock.patch.object(mcp_server.subprocess, "run", return_value=miss):
                    with self.assertRaises(ValueError) as ctx:
                        mcp_server._resolve_credential(_CF_ID, "u")
        message = str(ctx.exception)
        self.assertIn(_CF_ID, message)  # the env var
        self.assertIn(_CF_ID_TARGET, message)  # the vault target
        self.assertIn("security add-generic-password", message)  # the Keychain

    def test_missing_credentials_skip_endpoint_rather_than_raise(self):
        # Fail closed: an Access-gated host is never called bare.
        miss = mock.Mock(returncode=1, stdout="")
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", _vault({})):
                with mock.patch.object(mcp_server.subprocess, "run", return_value=miss):
                    self.assertIsNone(
                        mcp_server._ollama_auth_headers("https://ollama.example.com")
                    )

    def test_non_cf_service_never_consults_the_vault(self):
        # Only services in _CRED_VAULT_TARGETS get a vault lookup; the
        # Perplexity key keeps its exact two-tier behaviour.
        vault = mock.Mock(return_value="should-not-be-read")
        ok = mock.Mock(returncode=0, stdout="from-keychain\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PERPLEXITY_API_KEY", None)
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                with mock.patch.object(mcp_server.subprocess, "run", return_value=ok):
                    value, source = mcp_server._resolve_credential(
                        "api_tokens", "perplexity"
                    )
        self.assertEqual(value, "from-keychain")
        self.assertEqual(source, "macos-keychain")
        vault.assert_not_called()

    def test_localhost_endpoint_needs_no_credentials_at_all(self):
        vault = mock.Mock(return_value=None)
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", vault):
                self.assertEqual(
                    mcp_server._ollama_auth_headers("http://localhost:11434"), {}
                )
        vault.assert_not_called()


class TestWindowsCredentialReaderIsPlatformGuarded(unittest.TestCase):
    """The ctypes path must be inert everywhere except Windows.

    CI runs this suite on Linux, where importing ctypes.wintypes raises —
    so the guard is what keeps `import mcp_server` working at all.
    """

    def setUp(self):
        mcp_server._windows_credential_api.cache_clear()
        self.addCleanup(mcp_server._windows_credential_api.cache_clear)

    def test_api_binding_is_none_off_windows(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                mcp_server._windows_credential_api.cache_clear()
                with mock.patch.object(mcp_server.sys, "platform", platform):
                    self.assertIsNone(mcp_server._windows_credential_api())

    def test_reader_returns_none_off_windows(self):
        with mock.patch.object(mcp_server.sys, "platform", "linux"):
            self.assertIsNone(mcp_server._read_windows_credential(_CF_ID_TARGET))

    def test_resolution_off_windows_is_unchanged_two_tier(self):
        # The whole non-Windows contract in one assertion: env, then
        # Keychain, with the vault tier structurally absent.
        ok = mock.Mock(returncode=0, stdout="from-keychain\n")
        with mock.patch.object(mcp_server.sys, "platform", "darwin"):
            with _no_cf_env():
                with mock.patch.object(mcp_server.subprocess, "run", return_value=ok):
                    value, source = mcp_server._resolve_credential(_CF_ID, "u")
            self.assertEqual((value, source), ("from-keychain", "macos-keychain"))

            with mock.patch.dict(os.environ, {_CF_ID: _FAKE_ENV_ID}):
                with mock.patch.object(mcp_server.subprocess, "run") as run:
                    self.assertEqual(
                        mcp_server._resolve_credential(_CF_ID, "u"),
                        (_FAKE_ENV_ID, "env"),
                    )
                    run.assert_not_called()


class TestCredentialBlobDecoding(unittest.TestCase):
    """CredentialBlob decoding — UTF-16LE in practice, UTF-8 tolerated."""

    def test_utf16le_blob_is_the_common_case(self):
        # What CredVault.psm1's [Text.Encoding]::Unicode, cmdkey, and
        # keyring all write.
        blob = _FAKE_VAULT_SECRET.encode("utf-16-le")
        self.assertEqual(mcp_server._decode_credential_blob(blob), _FAKE_VAULT_SECRET)

    def test_utf8_blob_is_tolerated(self):
        blob = _FAKE_VAULT_SECRET.encode("utf-8")
        self.assertEqual(mcp_server._decode_credential_blob(blob), _FAKE_VAULT_SECRET)

    def test_ascii_utf8_is_not_misread_as_utf16(self):
        # An even-length ASCII UTF-8 blob also decodes "successfully" as
        # UTF-16LE, into garbage. Guards against a decoder that just tries
        # UTF-16 first.
        self.assertEqual(mcp_server._decode_credential_blob(b"abcd"), "abcd")

    def test_nul_terminated_utf8_is_not_misread_as_utf16(self):
        # Regression (PR #48 review): "contains a NUL anywhere" is NOT a
        # safe UTF-16 discriminator. A C-style NUL-terminated UTF-8 blob of
        # even length contains one, decodes as UTF-16LE WITHOUT raising, and
        # yielded CJK garbage ("扡..."). The high byte of an ASCII
        # UTF-16LE character is what actually distinguishes them, so the
        # test is on blob[1].
        self.assertEqual(mcp_server._decode_credential_blob(b"abc\x00"), "abc")
        for text in ("a", "ab", "abc", "abcd", _FAKE_VAULT_SECRET):
            with self.subTest(text=text):
                blob = text.encode("utf-8") + b"\x00"
                self.assertEqual(mcp_server._decode_credential_blob(blob), text)

    def test_trailing_nul_terminator_is_stripped(self):
        blob = (_FAKE_VAULT_ID + "\x00").encode("utf-16-le")
        self.assertEqual(mcp_server._decode_credential_blob(blob), _FAKE_VAULT_ID)

    def test_empty_blob_is_empty_string(self):
        self.assertEqual(mcp_server._decode_credential_blob(b""), "")

    def test_undecodable_blob_fails_closed(self):
        # Lone UTF-16 surrogate half: invalid as both UTF-16LE and UTF-8.
        # "" reads as absent upstream rather than a garbage credential.
        self.assertEqual(mcp_server._decode_credential_blob(b"\x00\xd8\xff"), "")

    def test_decoder_never_raises_on_arbitrary_bytes(self):
        for blob in (b"\xff", b"\xff\xfe", b"\x00", b"\x80\x81\x82", bytes(range(8))):
            with self.subTest(blob=blob):
                self.assertIsInstance(mcp_server._decode_credential_blob(blob), str)


class TestCredentialSourceReporting(unittest.TestCase):
    """--check reports the tier, never the value (the cutover check)."""

    def _report(self, env, vault):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in (_CF_ID, _CF_SECRET):
                os.environ.pop(key, None)
            os.environ.update(env)
            with (
                mock.patch.object(
                    mcp_server, "_read_windows_credential", _vault(vault)
                ),
                mock.patch.object(
                    mcp_server.subprocess, "run", return_value=mock.Mock(returncode=1)
                ),
                contextlib.redirect_stdout(buf),
            ):
                mcp_server._report_cf_access_credentials()
        return buf.getvalue()

    def test_reports_vault_as_the_source(self):
        out = self._report(
            {}, {_CF_ID_TARGET: _FAKE_VAULT_ID, _CF_SECRET_TARGET: _FAKE_VAULT_SECRET}
        )
        self.assertIn("ok: cloudflare access client id found", out)
        self.assertIn("windows-credential-manager", out)
        self.assertNotIn(_FAKE_VAULT_ID, out)  # never the value
        self.assertNotIn(_FAKE_VAULT_SECRET, out)

    def test_reports_env_as_the_source(self):
        out = self._report({_CF_ID: _FAKE_ENV_ID, _CF_SECRET: "fake-env-secret"}, {})
        self.assertIn("(env)", out)
        self.assertNotIn(_FAKE_ENV_ID, out)

    def test_warns_when_env_and_vault_disagree(self):
        # The failure this catches: a rotated service token leaves the
        # vault current and the env var stale. Env wins by precedence, so
        # the only other symptom is a 403 indistinguishable from
        # "no credentials configured".
        out = self._report({_CF_ID: _FAKE_ENV_ID}, {_CF_ID_TARGET: _FAKE_VAULT_ID})
        self.assertIn("differs", out)
        self.assertIn(_CF_ID_TARGET, out)
        self.assertNotIn(_FAKE_ENV_ID, out)
        self.assertNotIn(_FAKE_VAULT_ID, out)

    def test_no_divergence_warning_when_copies_agree(self):
        out = self._report({_CF_ID: _FAKE_ENV_ID}, {_CF_ID_TARGET: _FAKE_ENV_ID})
        self.assertNotIn("differs", out)

    def test_warns_when_credential_is_absent_everywhere(self):
        out = self._report({}, {})
        self.assertIn("warn: cloudflare access client id not found", out)
        self.assertIn("warn: cloudflare access client secret not found", out)

    def test_empty_keychain_value_is_not_reported_as_found(self):
        # Regression (PR #48 review): a Keychain item that exists with an
        # empty password resolves to ("", "macos-keychain") instead of
        # raising, which printed `ok: ... found (macos-keychain)` — a FALSE
        # SUCCESS on the exact surface used to verify a vault cutover, while
        # _ollama_auth_headers rejects the blank and skips every endpoint.
        buf = io.StringIO()
        empty = mock.Mock(returncode=0, stdout="   \n")
        with _no_cf_env():
            with mock.patch.object(mcp_server, "_read_windows_credential", _vault({})):
                with mock.patch.object(
                    mcp_server.subprocess, "run", return_value=empty
                ):
                    with contextlib.redirect_stdout(buf):
                        mcp_server._report_cf_access_credentials()
        out = buf.getvalue()
        self.assertNotIn("ok:", out)
        self.assertIn("warn: cloudflare access client id not found", out)
        self.assertIn("warn: cloudflare access client secret not found", out)


class TestModelAllowlistOverride(unittest.TestCase):
    """v1.2 (issue #20): AI_TOOLS_OLLAMA_MODELS overrides the allowlist."""

    def test_unset_env_returns_builtin(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TOOLS_OLLAMA_MODELS", None)
            self.assertEqual(
                mcp_server._resolve_delegate_models(),
                mcp_server._OLLAMA_BUILTIN_DELEGATE_MODELS,
            )

    def test_override_parses_orders_and_dedupes(self):
        raw = " qwen2.5-coder:14b , qwen3.8:27b-nvfp4 ,qwen2.5-coder:14b "
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_MODELS": raw}):
            self.assertEqual(
                mcp_server._resolve_delegate_models(),
                ("qwen2.5-coder:14b", "qwen3.8:27b-nvfp4"),
            )

    def test_effectively_empty_override_fails_closed_to_builtin(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_MODELS": " ,,  , "}):
            self.assertEqual(
                mcp_server._resolve_delegate_models(),
                mcp_server._OLLAMA_BUILTIN_DELEGATE_MODELS,
            )


# ─── Durable queue client (v1.6) ──────────────────────────────────────

_QID = "q" + "d" * 32


class _FakeQueueClient:
    """Programmable GET/POST fake for the queue client helpers.

    Routes by exact URL. ``get_map``/``post_map`` values are either a
    _FakeResponse or an Exception instance to raise.
    """

    def __init__(self, get_map=None, post_map=None):
        self.get_map = get_map or {}
        self.post_map = post_map or {}
        self.get_calls: list = []
        self.post_calls: list = []

    @staticmethod
    def _resolve(mapping, url):
        outcome = mapping.get(url)
        if outcome is None:
            raise mcp_server.httpx.ConnectError(f"no route for {url}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._resolve(self.get_map, url)

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._resolve(self.post_map, url)


_LOCAL_QUEUE = "http://localhost:11438"
_REMOTE_QUEUE = "https://queue-mbp.example"
_HEALTH_OK = {"status": "ok", "queued": 0, "running": 0}


class TestResolveQueueChain(unittest.TestCase):
    def _chain(self, env):
        cleared = {"AI_TOOLS_QUEUE_URLS": ""}
        with mock.patch.dict(os.environ, {**cleared, **env}):
            return mcp_server._resolve_queue_chain()

    def test_default_chain(self):
        self.assertEqual(self._chain({}), list(mcp_server._QUEUE_DEFAULT_CHAIN))

    def test_env_csv_override_dedupes_and_strips(self):
        chain = self._chain(
            {
                "AI_TOOLS_QUEUE_URLS": (
                    " http://localhost:11438/ ,https://q.example,"
                    "http://localhost:11438,,"
                )
            }
        )
        self.assertEqual(chain, ["http://localhost:11438", "https://q.example"])

    def test_plain_http_remote_rejected(self):
        # Same fail-closed rule as the Ollama chain: a non-localhost
        # endpoint must be https (the Access-gated tunnel).
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_QUEUE_URLS": "http://remote.example:11438"})

    def test_embedded_credentials_rejected(self):
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_QUEUE_URLS": "https://u:pw@q.example"})


class _QueueClientCase(unittest.TestCase):
    """Shared env/keychain scaffolding for the queue helper tests."""

    URLS = _LOCAL_QUEUE

    def setUp(self):
        env = mock.patch.dict(os.environ, {"AI_TOOLS_QUEUE_URLS": self.URLS})
        env.start()
        self.addCleanup(env.stop)
        kc = mock.patch.object(
            mcp_server, "get_api_key_from_keychain", side_effect=_no_keychain
        )
        kc.start()
        self.addCleanup(kc.stop)

    def _with(self, client):
        return mock.patch.object(
            mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
        )


class TestSelectQueueEndpoint(_QueueClientCase):
    def test_healthy_localhost_selected_with_empty_headers(self):
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)}
        )
        with self._with(client):
            selected = asyncio.run(mcp_server._select_queue_endpoint())
        self.assertEqual(selected, (_LOCAL_QUEUE, {}))

    def test_unreachable_endpoint_yields_none(self):
        client = _FakeQueueClient(
            get_map={
                f"{_LOCAL_QUEUE}/healthz": mcp_server.httpx.ConnectError("refused")
            }
        )
        with self._with(client):
            self.assertIsNone(asyncio.run(mcp_server._select_queue_endpoint()))

    def test_unhealthy_endpoint_yields_none(self):
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(status_code=500)}
        )
        with self._with(client):
            self.assertIsNone(asyncio.run(mcp_server._select_queue_endpoint()))

    def test_remote_without_creds_is_skipped_and_never_called(self):
        # Fail closed: an Access-gated queue host is never probed bare.
        with mock.patch.dict(os.environ, {"AI_TOOLS_QUEUE_URLS": _REMOTE_QUEUE}):
            client = _FakeQueueClient(
                get_map={
                    f"{_REMOTE_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)
                }
            )
            with self._with(client):
                self.assertIsNone(asyncio.run(mcp_server._select_queue_endpoint()))
        self.assertEqual(client.get_calls, [])

    def test_remote_with_creds_gets_cf_access_headers(self):
        creds = {
            "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
            "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
        }

        def keychain(service, account):
            if service in creds:
                return creds[service]
            raise ValueError("not found")

        with (
            mock.patch.dict(os.environ, {"AI_TOOLS_QUEUE_URLS": _REMOTE_QUEUE}),
            mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
            ),
        ):
            client = _FakeQueueClient(
                get_map={
                    f"{_REMOTE_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)
                }
            )
            with self._with(client):
                selected = asyncio.run(mcp_server._select_queue_endpoint())
        endpoint, headers = selected
        self.assertEqual(endpoint, _REMOTE_QUEUE)
        self.assertEqual(headers["CF-Access-Client-Id"], "id-123")
        _, kwargs = client.get_calls[0]
        self.assertEqual(kwargs["headers"]["CF-Access-Client-Secret"], "sec-456")

    def test_falls_through_dead_localhost_to_healthy_remote(self):
        creds = {
            "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
            "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
        }

        def keychain(service, account):
            if service in creds:
                return creds[service]
            raise ValueError("not found")

        with (
            mock.patch.dict(
                os.environ,
                {"AI_TOOLS_QUEUE_URLS": f"{_LOCAL_QUEUE},{_REMOTE_QUEUE}"},
            ),
            mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
            ),
        ):
            client = _FakeQueueClient(
                get_map={
                    f"{_LOCAL_QUEUE}/healthz": mcp_server.httpx.ConnectError("dead"),
                    f"{_REMOTE_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK),
                }
            )
            with self._with(client):
                endpoint, _headers = asyncio.run(mcp_server._select_queue_endpoint())
        self.assertEqual(endpoint, _REMOTE_QUEUE)


class TestQueueSubmit(_QueueClientCase):
    def _submit(self, client, payload=None):
        with self._with(client):
            return asyncio.run(mcp_server._queue_submit(payload or {"model": "m"}))

    def test_happy_path_returns_job_id(self):
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)},
            post_map={
                f"{_LOCAL_QUEUE}/v1/jobs": _FakeResponse(
                    json_data={"job_id": _QID, "status": "queued"}
                )
            },
        )
        self.assertEqual(self._submit(client), _QID)
        url, kwargs = client.post_calls[0]
        self.assertEqual(url, f"{_LOCAL_QUEUE}/v1/jobs")
        self.assertEqual(kwargs["json"]["model"], "m")

    def test_no_reachable_endpoint_returns_none(self):
        client = _FakeQueueClient(
            get_map={
                f"{_LOCAL_QUEUE}/healthz": mcp_server.httpx.ConnectError("refused")
            }
        )
        self.assertIsNone(self._submit(client))
        self.assertEqual(client.post_calls, [])

    def test_submit_rejection_returns_none(self):
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)},
            post_map={
                f"{_LOCAL_QUEUE}/v1/jobs": _FakeResponse(
                    status_code=413, json_data={"error": "too big"}
                )
            },
        )
        self.assertIsNone(self._submit(client))

    def test_malformed_job_id_is_ambiguous_not_a_fallback(self):
        # HTTP 200 proves the queue ACCEPTED the job; a missing/garbage
        # job_id must not quietly fall back to the in-memory store, or
        # the same prompt executes twice (queue worker + in-memory).
        for bad in ("no-prefix", "q" + "z" * 32, "", None, 42):
            client = _FakeQueueClient(
                get_map={
                    f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)
                },
                post_map={
                    f"{_LOCAL_QUEUE}/v1/jobs": _FakeResponse(
                        json_data={"job_id": bad, "status": "queued"}
                    )
                },
            )
            with self.assertRaises(ValueError, msg=repr(bad)) as ctx:
                self._submit(client)
            self.assertIn("twice", str(ctx.exception))

    def test_connect_error_on_submit_falls_back(self):
        # The POST never reached the service — safe to fall back (None).
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)},
            post_map={
                f"{_LOCAL_QUEUE}/v1/jobs": mcp_server.httpx.ConnectError("refused")
            },
        )
        self.assertIsNone(self._submit(client))

    def test_read_timeout_on_submit_is_ambiguous(self):
        # The request was sent and the response never came back: the job
        # may be enqueued server-side, so falling back would risk double
        # execution. Must raise, not return None.
        client = _FakeQueueClient(
            get_map={f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK)},
            post_map={
                f"{_LOCAL_QUEUE}/v1/jobs": mcp_server.httpx.ReadTimeout("mid-flight")
            },
        )
        with self.assertRaises(ValueError) as ctx:
            self._submit(client)
        self.assertIn("unknown", str(ctx.exception))
        self.assertIn("twice", str(ctx.exception))


class TestQueuePoll(_QueueClientCase):
    RESULT_URL = f"{_LOCAL_QUEUE}/v1/jobs/{_QID}/result"

    def _poll(self, client):
        with self._with(client):
            return asyncio.run(mcp_server._queue_poll(_QID))

    def _client(self, result_response):
        return _FakeQueueClient(
            get_map={
                f"{_LOCAL_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK),
                self.RESULT_URL: result_response,
            }
        )

    def test_running_maps_to_delegate_running_envelope(self):
        out = self._poll(
            self._client(
                _FakeResponse(
                    json_data={"job_id": _QID, "status": "running", "elapsed_s": 7}
                )
            )
        )
        self.assertEqual(out, {"status": "running", "elapsed_s": 7, "queue": "durable"})

    def test_done_returns_stored_chat_response(self):
        stored = {"model": "m", "message": {"content": "queued answer"}}
        out = self._poll(
            self._client(_FakeResponse(json_data={"job_id": _QID, "result": stored}))
        )
        self.assertEqual(out, stored)

    def test_failed_envelope_passes_through(self):
        stored = {"status": "failed", "error": "upstream HTTP 500"}
        out = self._poll(
            self._client(_FakeResponse(json_data={"job_id": _QID, "result": stored}))
        )
        self.assertEqual(out, stored)

    def test_unknown_id_404_raises_purge_hint(self):
        with self.assertRaises(ValueError) as ctx:
            self._poll(
                self._client(_FakeResponse(status_code=404, json_data={"error": "x"}))
            )
        # The message must not flatly assert the job was purged: each
        # endpoint has its own store, so "unknown here" also covers a job
        # whose accepting endpoint is currently unreachable.
        self.assertIn("purged", str(ctx.exception))
        self.assertIn("unreachable", str(ctx.exception))
        self.assertIn(_LOCAL_QUEUE, str(ctx.exception))

    def test_404_walks_chain_to_endpoint_that_knows_the_job(self):
        # Two endpoints, each with its own store: a 404 from the first
        # must not end the poll — the second endpoint has the job.
        creds = {
            "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
            "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
        }

        def keychain(service, account):
            if service in creds:
                return creds[service]
            raise ValueError("not found")

        stored = {"model": "m", "message": {"content": "found remotely"}}
        client = _FakeQueueClient(
            get_map={
                self.RESULT_URL: _FakeResponse(
                    status_code=404, json_data={"error": "x"}
                ),
                f"{_REMOTE_QUEUE}/v1/jobs/{_QID}/result": _FakeResponse(
                    json_data={"job_id": _QID, "result": stored}
                ),
            }
        )
        with (
            mock.patch.dict(
                os.environ,
                {"AI_TOOLS_QUEUE_URLS": f"{_LOCAL_QUEUE},{_REMOTE_QUEUE}"},
            ),
            mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
            ),
            self._with(client),
        ):
            out = asyncio.run(mcp_server._queue_poll(_QID))
        self.assertEqual(out, stored)

    def test_no_endpoint_raises_retryable_error(self):
        client = _FakeQueueClient(
            get_map={
                f"{_LOCAL_QUEUE}/healthz": mcp_server.httpx.ConnectError("refused")
            }
        )
        with self.assertRaises(ValueError) as ctx:
            self._poll(client)
        self.assertIn("retry local_delegate_result later", str(ctx.exception))

    def test_request_error_raises_retryable_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._poll(self._client(mcp_server.httpx.ConnectError("mid-flight")))
        self.assertIn("durable", str(ctx.exception))

    def test_http_error_scrubs_cf_headers_from_body(self):
        creds = {
            "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
            "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
        }

        def keychain(service, account):
            if service in creds:
                return creds[service]
            raise ValueError("not found")

        remote_result = f"{_REMOTE_QUEUE}/v1/jobs/{_QID}/result"
        client = _FakeQueueClient(
            get_map={
                f"{_REMOTE_QUEUE}/healthz": _FakeResponse(json_data=_HEALTH_OK),
                remote_result: _FakeResponse(
                    status_code=500, text="echo CF-Access-Client-Secret: sec-456"
                ),
            }
        )
        with (
            mock.patch.dict(os.environ, {"AI_TOOLS_QUEUE_URLS": _REMOTE_QUEUE}),
            mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
            ),
            self._with(client),
        ):
            out = asyncio.run(mcp_server._queue_poll(_QID))
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("sec-456", out["error"])


class TestLocalDelegateQueue(unittest.TestCase):
    """Handler-level queue-first submit and q-id result routing."""

    def setUp(self):
        mcp_server._delegate_jobs.clear()
        _stub_resolution(self)

    def test_queue_submit_wins_and_envelope_is_durable(self):
        submit = mock.AsyncMock(return_value=_QID)
        start = mock.Mock()
        with (
            mock.patch.object(mcp_server, "_queue_submit", submit),
            mock.patch.object(mcp_server, "_start_delegate_job", start),
        ):
            out = _call("local_delegate", {"prompt": "p", "background": True})
        env = json.loads(out[0].text)
        self.assertEqual(env["job_id"], _QID)
        self.assertEqual(env["status"], "started")
        self.assertEqual(env["queue"], "durable")
        # The model was implicit: the queue executes against ITS upstream,
        # not the endpoint the model was resolved on, and the envelope
        # says so.
        self.assertIn("resolved against", env["warning"])
        start.assert_not_called()
        (payload,), _ = submit.call_args
        self.assertEqual(payload["messages"], [{"role": "user", "content": "p"}])

    def test_queue_submit_explicit_model_has_no_warning(self):
        submit = mock.AsyncMock(return_value=_QID)
        with mock.patch.object(mcp_server, "_queue_submit", submit):
            out = _call(
                "local_delegate",
                {
                    "prompt": "p",
                    "background": True,
                    "model": mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL,
                },
            )
        env = json.loads(out[0].text)
        self.assertEqual(env["queue"], "durable")
        self.assertNotIn("warning", env)

    def test_ambiguous_queue_submit_errors_without_memory_fallback(self):
        # An ambiguous submit outcome (job may already be enqueued) must
        # surface as an error — starting an in-memory copy could execute
        # the prompt twice.
        submit = mock.AsyncMock(
            side_effect=ValueError("submit outcome is unknown — twice")
        )
        start = mock.Mock()
        with (
            mock.patch.object(mcp_server, "_queue_submit", submit),
            mock.patch.object(mcp_server, "_start_delegate_job", start),
        ):
            out = _call("local_delegate", {"prompt": "p", "background": True})
        self.assertTrue(out[0].text.startswith("Error:"))
        self.assertIn("unknown", out[0].text)
        start.assert_not_called()

    def test_fallback_to_memory_carries_warning(self):
        async def scenario():
            fake = mock.AsyncMock(
                return_value={"model": "m", "message": {"content": "ok"}}
            )
            with (
                mock.patch.object(
                    mcp_server, "_queue_submit", mock.AsyncMock(return_value=None)
                ),
                mock.patch.object(mcp_server, "_post_ollama_chat", fake),
            ):
                out = await mcp_server.call_tool(
                    "local_delegate", {"prompt": "p", "background": True}
                )
                env = json.loads(out[0].text)
                self.assertNotIn("queue", env)
                self.assertIn("No durable queue endpoint reachable", env["warning"])
                self.assertIn(env["job_id"], mcp_server._delegate_jobs)
                # Drain the in-memory job so no pending task leaks.
                task = mcp_server._delegate_jobs[env["job_id"]]["task"]
                await _settle(task.done)
                mcp_server._collect_delegate_job(env["job_id"])

        asyncio.run(scenario())

    def test_sync_call_never_touches_the_queue(self):
        submit = mock.AsyncMock()
        # _post_ollama_chat returns (data, evict_warning) since #69; a bare
        # dict here unpacks by iterating its keys instead of raising, which
        # silently turned `data` into the string "model" — caught by running
        # this test in isolation after a merge, not by the type checker.
        fake = mock.AsyncMock(
            return_value=({"model": "m", "message": {"content": "x"}}, "")
        )
        with (
            mock.patch.object(mcp_server, "_queue_submit", submit),
            mock.patch.object(mcp_server, "_post_ollama_chat", fake),
        ):
            _call("local_delegate", {"prompt": "p"})
        submit.assert_not_awaited()

    def test_result_routes_q_id_to_queue_poll(self):
        poll = mock.AsyncMock(
            return_value={"model": "m", "message": {"content": "durable answer"}}
        )
        collect = mock.Mock()
        with (
            mock.patch.object(mcp_server, "_queue_poll", poll),
            mock.patch.object(mcp_server, "_collect_delegate_job", collect),
        ):
            out = _call("local_delegate_result", {"job_id": _QID})
        self.assertIn("durable answer", out[0].text)
        poll.assert_awaited_once_with(_QID)
        collect.assert_not_called()

    def test_result_routes_legacy_id_to_memory_store(self):
        poll = mock.AsyncMock()
        with mock.patch.object(mcp_server, "_queue_poll", poll):
            out = _call("local_delegate_result", {"job_id": "b" * 32})
        self.assertIn("Error", out[0].text)  # unknown in-memory id
        poll.assert_not_awaited()

    def test_result_running_envelope_stays_json(self):
        poll = mock.AsyncMock(
            return_value={"status": "running", "elapsed_s": 3, "queue": "durable"}
        )
        with mock.patch.object(mcp_server, "_queue_poll", poll):
            out = _call("local_delegate_result", {"job_id": _QID})
        env = json.loads(out[0].text)
        self.assertEqual(env["status"], "running")
        self.assertEqual(env["queue"], "durable")

    def test_result_queue_unreachable_is_clean_retryable_error(self):
        poll = mock.AsyncMock(
            side_effect=ValueError("No queue endpoint reachable — retry later.")
        )
        with mock.patch.object(mcp_server, "_queue_poll", poll):
            out = _call("local_delegate_result", {"job_id": _QID})
        self.assertTrue(out[0].text.startswith("Error:"))
        self.assertIn("retry", out[0].text)

    def test_result_failed_envelope_renders_as_error(self):
        poll = mock.AsyncMock(
            return_value={"status": "failed", "error": "upstream HTTP 500"}
        )
        with mock.patch.object(mcp_server, "_queue_poll", poll):
            out = _call("local_delegate_result", {"job_id": _QID})
        self.assertIn("Error", out[0].text)
        self.assertIn("upstream HTTP 500", out[0].text)


if __name__ == "__main__":
    unittest.main()
