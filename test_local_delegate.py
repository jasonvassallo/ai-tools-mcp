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
import getpass
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SERVER_PATH = HERE / "mcp_server.py"


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


class TestPostOllamaChat(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(
            os.environ, {"AI_TOOLS_OLLAMA_URL": "http://localhost:11434"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, client, payload=None, timeout_s=300.0):
        with _with_client(client):
            return asyncio.run(
                mcp_server._post_ollama_chat(payload or {"model": "m"}, timeout_s)
            )

    def test_happy_path_posts_to_api_chat_with_timeout(self):
        client = _FakeClient(
            response=_FakeResponse(json_data={"message": {"content": "hi"}})
        )
        out = self._post(
            client, payload={"model": "m", "stream": False}, timeout_s=42.0
        )
        self.assertEqual(out["message"]["content"], "hi")
        url, kwargs = client.calls[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(kwargs["timeout"], 42.0)
        self.assertEqual(kwargs["json"]["model"], "m")

    def test_connect_error_mentions_launchagent(self):
        client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
        out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("LaunchAgent", out["error"])
        self.assertIn("http://localhost:11434", out["error"])

    def test_404_adds_pull_hint(self):
        client = _FakeClient(
            response=_FakeResponse(status_code=404, text="model not found")
        )
        out = self._post(client, payload={"model": "qwen3.6:35b-a3b-coding-nvfp4"})
        self.assertEqual(out["status"], "failed")
        self.assertIn("ollama pull qwen3.6:35b-a3b-coding-nvfp4", out["error"])

    def test_non_404_http_error_no_pull_hint(self):
        client = _FakeClient(response=_FakeResponse(status_code=500, text="boom"))
        out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertNotIn("ollama pull", out["error"])

    def test_non_json_200_is_failure_envelope(self):
        client = _FakeClient(response=_FakeResponse(json_data=None))
        out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("invalid JSON", out["error"])

    def test_bad_configured_url_is_failure_envelope(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_URL": "ftp://nope"}):
            client = _FakeClient(response=_FakeResponse(json_data={}))
            out = self._post(client)
        self.assertEqual(out["status"], "failed")
        self.assertIn("Invalid Ollama URL", out["error"])


class TestResolveOllamaUrl(unittest.TestCase):
    def test_env_var_wins(self):
        with mock.patch.dict(
            os.environ, {"AI_TOOLS_OLLAMA_URL": "http://jvmacmini:11434/"}
        ):
            self.assertEqual(mcp_server._resolve_ollama_url(), "http://jvmacmini:11434")

    def test_keychain_second(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TOOLS_OLLAMA_URL", None)
            with mock.patch.object(
                mcp_server,
                "get_api_key_from_keychain",
                return_value="https://mini.tail:11434",
            ) as kc:
                self.assertEqual(
                    mcp_server._resolve_ollama_url(), "https://mini.tail:11434"
                )
        kc.assert_called_once_with("OLLAMA_URL", getpass.getuser())

    def test_default_localhost_when_neither(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TOOLS_OLLAMA_URL", None)
            with mock.patch.object(
                mcp_server,
                "get_api_key_from_keychain",
                side_effect=ValueError("not found"),
            ):
                self.assertEqual(
                    mcp_server._resolve_ollama_url(), "http://localhost:11434"
                )

    def test_rejects_non_http_scheme(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_URL": "file:///etc/passwd"}):
            with self.assertRaises(ValueError):
                mcp_server._resolve_ollama_url()

    def test_rejects_garbage_url(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_URL": "http://"}):
            with self.assertRaises(ValueError):
                mcp_server._resolve_ollama_url()


if __name__ == "__main__":
    unittest.main()
