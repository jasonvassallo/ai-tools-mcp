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
import os
import sys
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

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

    class _FakeRequestError(Exception):
        pass

    class _FakeConnectError(_FakeRequestError):
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
            ConnectError=_FakeConnectError,
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
        with mock.patch.dict(
            os.environ,
            {
                "AI_TOOLS_OLLAMA_URLS": "https://remote.example",
                "AI_TOOLS_OLLAMA_URL": "",
            },
        ):
            with self._keychain(
                {
                    "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                    "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
                }
            ):
                mcp_server._ollama_endpoint_cache.clear()
                client = _FakeTagsClient(
                    tags_by_url={"https://remote.example": [_MODEL]}
                )
                with mock.patch.object(
                    mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
                ):
                    endpoint = asyncio.run(mcp_server._select_ollama_endpoint(_MODEL))
        self.assertEqual(endpoint, "https://remote.example")
        _, kwargs = client.get_calls[0]
        self.assertEqual(kwargs["headers"]["CF-Access-Client-Id"], "id-123")

    def test_post_sends_cf_headers_and_never_leaks_secret_in_errors(self):
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
            }
        ):
            with mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
            ):
                client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
                with mock.patch.object(
                    mcp_server, "_get_http_client", mock.AsyncMock(return_value=client)
                ):
                    out = asyncio.run(
                        mcp_server._post_ollama_chat({"model": _MODEL}, 30.0)
                    )
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("sec-456", out["error"])

    def test_http_error_body_echoing_secret_is_scrubbed(self):
        # An Access-gated host's 403 body can echo request headers verbatim.
        # redact_secrets has no CF-Access-token pattern, so the only backstop
        # is the value-aware scrub in _post_ollama_chat's HTTPStatusError
        # branch — assert it actually strips the live secret value.
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
            }
        ):
            with mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
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
                    out = asyncio.run(
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
        with self._keychain(
            {
                "OLLAMA_CF_ACCESS_CLIENT_ID": "id-123",
                "OLLAMA_CF_ACCESS_CLIENT_SECRET": "sec-456",
            }
        ):
            with mock.patch.object(
                mcp_server,
                "_select_ollama_endpoint",
                mock.AsyncMock(return_value="https://remote.example"),
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
                    out = asyncio.run(
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
        with _with_client(client):
            return asyncio.run(
                mcp_server._post_ollama_chat(payload or {"model": "m"}, timeout_s)
            )

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

    def test_404_adds_pull_hint(self):
        client = _FakeClient(
            response=_FakeResponse(status_code=404, text="model not found")
        )
        with self._with_selection():
            out = self._post(client, payload={"model": "qwen3.6:35b-a3b-coding-nvfp4"})
        self.assertEqual(out["status"], "failed")
        self.assertIn("ollama pull qwen3.6:35b-a3b-coding-nvfp4", out["error"])

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


_MODEL = "qwen3.6:35b-a3b-coding-nvfp4"


def _no_keychain(service, account):
    raise ValueError("not found")


def _no_security_binary(service, account):
    # v1.2: get_api_key_from_keychain folds a missing `security` CLI
    # (non-macOS) into the same actionable ValueError as an ordinary
    # miss — callers only ever see ValueError.
    raise ValueError("Credential not found. Set the OLLAMA_URL environment variable ...")


class TestResolveOllamaChain(unittest.TestCase):
    def _chain(self, env, keychain=_no_keychain):
        cleared = {k: "" for k in ("AI_TOOLS_OLLAMA_URLS", "AI_TOOLS_OLLAMA_URL")}
        with mock.patch.dict(os.environ, {**cleared, **env}):
            with mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=keychain
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
        tag = "qwen3.6:35b-a3b-coding-nvfp4-32k"
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


class TestRenderDelegateAnswer(unittest.TestCase):
    def test_happy_path(self):
        out = mcp_server._render_delegate_answer(
            {"model": "qwen3.6:35b-a3b-coding-nvfp4", "message": {"content": "answer"}}
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

            async def fake_post(payload, timeout_s):
                await gate.wait()
                return {"message": {"content": "done!"}}

            with mock.patch.object(mcp_server, "_post_ollama_chat", fake_post):
                job_id = mcp_server._start_delegate_job({"model": "m"})
                running = mcp_server._collect_delegate_job(job_id)
                self.assertEqual(running["status"], "running")
                self.assertIsInstance(running["elapsed_s"], int)
                gate.set()
                task = mcp_server._delegate_jobs[job_id]["task"]
                await _settle(task.done)  # let the wait_for-wrapped task finish
                done = mcp_server._collect_delegate_job(job_id)
                self.assertEqual(done["message"]["content"], "done!")
                with self.assertRaises(ValueError):
                    mcp_server._collect_delegate_job(job_id)  # single-collect

        asyncio.run(scenario())

    def test_job_cap_rejects_fifth(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s):
                await gate.wait()
                return {}

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
            async def hang(payload, timeout_s):
                await asyncio.sleep(3600)

            with mock.patch.object(mcp_server, "_post_ollama_chat", hang):
                with mock.patch.object(mcp_server, "_DELEGATE_BG_CEILING_S", 0.01):
                    job_id = mcp_server._start_delegate_job({})
                    await asyncio.sleep(0.05)
                    out = mcp_server._collect_delegate_job(job_id)
                    self.assertEqual(out["status"], "failed")
                    self.assertIn("ceiling", out["error"])

        asyncio.run(scenario())

    def test_completed_jobs_beyond_retention_are_evicted(self):
        # Completed-but-never-collected jobs must not accumulate forever.
        # Start well more than _DELEGATE_DONE_RETAINED jobs, let each
        # complete instantly, never call local_delegate_result on any of
        # them, and confirm the registry stays bounded — retaining only
        # the newest window — instead of growing unboundedly.
        async def scenario():
            async def fake_post(payload, timeout_s):
                return {"message": {"content": "done"}}

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

            async def fake_post(payload, timeout_s):
                return {"message": {"content": "done"}}

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
        self.assertIn("qwen3.6:35b-a3b-coding-nvfp4", out[0].text)

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


class TestLocalDelegateSync(unittest.TestCase):
    def test_payload_construction_defaults(self):
        fake = mock.AsyncMock(return_value={"model": "m", "message": {"content": "ok"}})
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
        )  # v1.1: omitted → inherit server OLLAMA_KEEP_ALIVE
        self.assertEqual(timeout_s, 300.0)
        self.assertIn("ok", out[0].text)

    def test_payload_with_system_think_keepalive_timeout(self):
        fake = mock.AsyncMock(return_value={"message": {"content": "ok"}})
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            _call(
                "local_delegate",
                {
                    "prompt": "p",
                    "system": "you are terse",
                    "think": True,
                    "keep_alive": "0",
                    "timeout_s": 600,
                    "model": "qwen3.6:35b-a3b-coding-nvfp4-256k",
                },
            )
        payload, timeout_s = fake.call_args.args
        self.assertEqual(
            payload["messages"][0], {"role": "system", "content": "you are terse"}
        )
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "p"})
        self.assertIs(payload["think"], True)
        self.assertEqual(payload["keep_alive"], "0")
        self.assertEqual(payload["model"], "qwen3.6:35b-a3b-coding-nvfp4-256k")
        self.assertEqual(timeout_s, 600.0)

    def test_failure_envelope_reaches_caller(self):
        fake = mock.AsyncMock(return_value={"status": "failed", "error": "down"})
        with mock.patch.object(mcp_server, "_post_ollama_chat", fake):
            out = _call("local_delegate", {"prompt": "p"})
        self.assertIn("down", out[0].text)


class TestLocalDelegateBackground(unittest.TestCase):
    def setUp(self):
        mcp_server._delegate_jobs.clear()

    def test_background_returns_job_id_then_result_collects(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s):
                await gate.wait()
                return {"model": "m", "message": {"content": "bg answer"}}

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

    def test_cap_error_is_clean_text(self):
        async def scenario():
            gate = asyncio.Event()

            async def fake_post(payload, timeout_s):
                await gate.wait()
                return {}

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


class TestRunCheckOllamaLine(unittest.TestCase):
    def _run_check_output(self, get_side_effect=None, json_version="0.9.0"):
        fake_resp = mock.Mock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.json.return_value = {"version": json_version}
        fake_requests = types.SimpleNamespace(
            get=mock.Mock(return_value=fake_resp, side_effect=get_side_effect),
            RequestException=Exception,
        )
        self._last_fake_get = fake_requests.get

        def fake_keychain(service, account):
            # Perplexity key resolves; OLLAMA_URL absent so the chain comes
            # from the env var alone (a Keychain URL of "k" would fail
            # endpoint validation and mask what this test targets).
            if service == "OLLAMA_URL":
                raise ValueError("not found")
            return "k"

        buf = io.StringIO()
        with mock.patch.object(mcp_server, "requests", fake_requests, create=True):
            with mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=fake_keychain
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
        with mock.patch.dict(
            os.environ, {"OLLAMA_CF_ACCESS_CLIENT_ID": "cid-env"}
        ):
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
            with mock.patch.object(
                mcp_server.subprocess, "run", side_effect=FileNotFoundError
            ):
                with self.assertRaises(ValueError) as ctx:
                    mcp_server.get_api_key_from_keychain("OLLAMA_URL", "u")
        self.assertIn("OLLAMA_URL", str(ctx.exception))


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
        raw = " qwen2.5-coder:14b , qwen3.6:35b-a3b-coding-nvfp4 ,qwen2.5-coder:14b "
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_MODELS": raw}):
            self.assertEqual(
                mcp_server._resolve_delegate_models(),
                ("qwen2.5-coder:14b", "qwen3.6:35b-a3b-coding-nvfp4"),
            )

    def test_effectively_empty_override_fails_closed_to_builtin(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_MODELS": " ,,  , "}):
            self.assertEqual(
                mcp_server._resolve_delegate_models(),
                mcp_server._OLLAMA_BUILTIN_DELEGATE_MODELS,
            )


if __name__ == "__main__":
    unittest.main()
