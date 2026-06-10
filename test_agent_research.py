#!/usr/bin/env python3
"""Unit tests for the agent_research tool in mcp_server.py.

Self-contained: stubs out the third-party imports (mcp, openai, httpx,
google.auth) and the Keychain lookup so the test can import mcp_server
without needing the full runtime environment. Uses only stdlib
(unittest). Network paths are never exercised — the Agent API POST
helper is mock.patch.object'd, mirroring how test_redact.py treats the
Gemini helpers.

Run:
    python3 test_agent_research.py

NOTE: Secret-shape fixtures are assembled at runtime from broken-up
parts so secret scanners (semgrep, gitleaks, trufflehog) do not flag
this test file as containing real credentials. Every fixture below is
synthetic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SERVER_PATH = HERE / "mcp_server.py"

# Synthetic JWT assembled at runtime (see module docstring).
_JWT_HEADER = "ey" + "J" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9XX"
FAKE_JWT = (
    _JWT_HEADER
    + ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1"
    + ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


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
                "mcp_server_under_test_agent_research", SERVER_PATH
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    return module


mcp_server = _load_mcp_server()


def _call(name: str, arguments: dict) -> list:
    return asyncio.run(mcp_server.call_tool(name, arguments))


def _sample_response(
    *,
    answer: str = "28",
    sandbox_results: list | None = None,
    status: str = "completed",
    output: list | None = None,
) -> dict:
    """Agent API response shaped like the live smoke test of 2026-06-09
    (POST /v1/responses, model perplexity/sonar, tools=[{type: sandbox}])."""
    if sandbox_results is None:
        sandbox_results = [
            {
                "exit_code": 0,
                "status": "completed",
                "stdout": "28\n",
                "stderr": "",
                "duration_ms": 135,
            }
        ]
    if output is None:
        output = [
            {
                "type": "sandbox_results",
                "status": "completed",
                "container_id": "irzdnf5kdho6bn91ejom7",
                "language": "bash",
                "code": 'python3 -c "print(2+3+5+7+11)"',
                "results": sandbox_results,
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            },
        ]
    return {
        "id": "resp_79b0f91b-e4c6-44e9-86cf-8ab09e9c88d0",
        "object": "response",
        "status": status,
        "model": "perplexity/sonar",
        "output": output,
        "usage": {
            "input_tokens": 4701,
            "output_tokens": 123,
            "total_tokens": 4824,
            "cost": {
                "currency": "USD",
                "input_cost": 0.00118,
                "output_cost": 0.00031,
                "tool_calls_cost": 0,
                "total_cost": 0.00149,
            },
        },
    }


class TestAgentResearchToolListing(unittest.TestCase):
    def _get_tool(self):
        tools = asyncio.run(mcp_server.list_tools())
        by_name = {t.name: t for t in tools}
        self.assertIn("agent_research", by_name)
        return by_name["agent_research"]

    def test_tool_is_listed_with_query_required(self):
        tool = self._get_tool()
        self.assertEqual(tool.inputSchema["required"], ["query"])

    def test_model_enum_matches_server_allowlist(self):
        tool = self._get_tool()
        enum = tool.inputSchema["properties"]["model"]["enum"]
        self.assertEqual(sorted(enum), sorted(mcp_server.AGENT_RESEARCH_MODELS))

    def test_max_output_tokens_bounds_in_schema(self):
        tool = self._get_tool()
        prop = tool.inputSchema["properties"]["max_output_tokens"]
        self.assertEqual(prop["minimum"], 256)
        self.assertEqual(prop["maximum"], 8192)


class TestAgentResearchValidation(unittest.TestCase):
    """Invalid arguments must fail closed with a structured error and
    must never reach the network helper."""

    def _assert_fails_without_network(self, arguments: dict, expect_substr: str):
        with mock.patch.object(mcp_server, "_post_agent_research") as post:
            result = _call("agent_research", arguments)
        post.assert_not_called()
        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn(expect_substr, payload["error"])

    def test_missing_query(self):
        self._assert_fails_without_network({}, "query")

    def test_empty_query(self):
        self._assert_fails_without_network({"query": "   "}, "query")

    def test_non_string_query(self):
        self._assert_fails_without_network({"query": 123}, "query")

    def test_model_not_in_allowlist(self):
        self._assert_fails_without_network(
            {"query": "q", "model": "openai/gpt-5.5"}, "model"
        )

    def test_max_output_tokens_string(self):
        self._assert_fails_without_network(
            {"query": "q", "max_output_tokens": "2048"}, "max_output_tokens"
        )

    def test_max_output_tokens_bool(self):
        # bool is an int subclass in Python; True must not pass as 1.
        self._assert_fails_without_network(
            {"query": "q", "max_output_tokens": True}, "max_output_tokens"
        )

    def test_max_output_tokens_below_minimum(self):
        self._assert_fails_without_network(
            {"query": "q", "max_output_tokens": 100}, "max_output_tokens"
        )

    def test_max_output_tokens_above_maximum(self):
        self._assert_fails_without_network(
            {"query": "q", "max_output_tokens": 100_000}, "max_output_tokens"
        )


class TestAgentResearchSuccess(unittest.TestCase):
    def test_answer_cost_and_run_count_in_output(self):
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=_sample_response()
        ) as post:
            result = _call("agent_research", {"query": "sum the first 5 primes"})
        self.assertEqual(len(result), 1)
        text = result[0].text
        self.assertIn("28", text)
        self.assertIn("0.00149", text)
        self.assertIn("USD", text)
        self.assertIn("sandbox executions: 1", text)
        post.assert_called_once()

    def test_default_payload_pins_model_and_sandbox_tool(self):
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=_sample_response()
        ) as post:
            _call("agent_research", {"query": "sum the first 5 primes"})
        payload = post.call_args.args[0]
        self.assertEqual(payload["model"], "anthropic/claude-sonnet-4-6")
        self.assertEqual(payload["input"], "sum the first 5 primes")
        self.assertEqual(payload["tools"], [{"type": "sandbox"}])
        self.assertEqual(payload["max_output_tokens"], 4096)
        # Server-fixed instructions steer the agent toward sandbox use.
        self.assertIsInstance(payload.get("instructions"), str)
        self.assertTrue(payload["instructions"].strip())
        # Synchronous phase-1 contract: background mode must not be set.
        self.assertNotIn("background", payload)

    def test_allowlisted_model_override_passes_through(self):
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=_sample_response()
        ) as post:
            _call(
                "agent_research",
                {"query": "q", "model": "perplexity/sonar", "max_output_tokens": 512},
            )
        payload = post.call_args.args[0]
        self.assertEqual(payload["model"], "perplexity/sonar")
        self.assertEqual(payload["max_output_tokens"], 512)

    def test_answer_is_redacted(self):
        response = _sample_response(answer=f"The token is {FAKE_JWT} apparently.")
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        text = result[0].text
        self.assertNotIn(FAKE_JWT, text)
        self.assertIn("[REDACTED_JWT]", text)

    def test_non_string_output_text_is_skipped_not_crashed(self):
        # The Agent API response is untrusted input — a truthy non-string
        # `text` (e.g. a dict) must be skipped, not crash the join (per
        # PR #16 review, Qodo bug #3).
        output = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": {"malformed": "dict"}},
                    {"type": "output_text", "text": "real answer"},
                ],
            },
        ]
        with mock.patch.object(
            mcp_server,
            "_post_agent_research",
            return_value=_sample_response(output=output),
        ):
            result = _call("agent_research", {"query": "q"})
        self.assertIn("real answer", result[0].text)
        self.assertNotIn("malformed", result[0].text)

    def test_model_and_status_from_response_are_redacted(self):
        # `model` and `status` are API-emitted strings rendered into the
        # output — route them through the redactor like everything else
        # (per PR #16 review, Claude bot).
        response = _sample_response(status=f"weird-{FAKE_JWT}")
        response["model"] = f"perplexity/{FAKE_JWT}"
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        text = result[0].text
        self.assertNotIn(FAKE_JWT, text)
        self.assertIn("[REDACTED_JWT]", text)

    def test_missing_cost_block_omits_cost_line(self):
        response = _sample_response()
        response["usage"] = {}
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        text = result[0].text
        self.assertNotIn("cost:", text)
        self.assertIn("sandbox executions: 1", text)

    def test_multiple_message_items_are_joined(self):
        output = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "part one"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "part two"}],
            },
        ]
        with mock.patch.object(
            mcp_server,
            "_post_agent_research",
            return_value=_sample_response(output=output),
        ):
            result = _call("agent_research", {"query": "q"})
        text = result[0].text
        self.assertIn("part one", text)
        self.assertIn("part two", text)


class TestAgentResearchFailures(unittest.TestCase):
    def test_http_failure_envelope_is_surfaced(self):
        failure = {"status": "failed", "error": "403: forbidden"}
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=failure
        ):
            result = _call("agent_research", {"query": "q"})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("403", payload["error"])

    def test_non_completed_status_is_flagged(self):
        response = _sample_response(status="incomplete")
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        self.assertIn("incomplete", result[0].text)

    def test_failed_sandbox_execution_includes_truncated_stderr(self):
        long_stderr = "boom! " * 200  # 1200 chars, must be truncated
        response = _sample_response(
            sandbox_results=[
                {
                    "exit_code": 1,
                    "status": "completed",
                    "stdout": "",
                    "stderr": long_stderr,
                    "duration_ms": 90,
                }
            ]
        )
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        text = result[0].text
        self.assertIn("exit_code=1", text)
        self.assertIn("boom!", text)
        self.assertNotIn(long_stderr, text)  # truncated, not dumped verbatim

    def test_clean_sandbox_run_has_no_warning_section(self):
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=_sample_response()
        ):
            result = _call("agent_research", {"query": "q"})
        self.assertNotIn("exit_code=", result[0].text)

    def test_empty_output_returns_error(self):
        response = _sample_response(output=[])
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=response
        ):
            result = _call("agent_research", {"query": "q"})
        self.assertIn("no assistant message", result[0].text)


class TestAgentResearchBackgroundStart(unittest.TestCase):
    def test_background_true_returns_response_id_envelope(self):
        queued = {
            "id": "resp_abc-123",
            "status": "queued",
            "model": "anthropic/claude-sonnet-4-6",
        }
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=queued
        ) as post:
            result = _call("agent_research", {"query": "q", "background": True})
        payload = post.call_args.args[0]
        self.assertIs(payload["background"], True)
        env = json.loads(result[0].text)
        self.assertEqual(env["response_id"], "resp_abc-123")
        self.assertEqual(env["status"], "queued")
        self.assertIn("agent_research_result", env["hint"])

    def test_background_must_be_json_boolean(self):
        # `bool("false")` is True in Python — a JSON-stringified flag must
        # be rejected, not silently coerced (same contract as the
        # collaborative_planning flag on gemini_deep_research_start).
        with mock.patch.object(mcp_server, "_post_agent_research") as post:
            result = _call("agent_research", {"query": "q", "background": "true"})
        post.assert_not_called()
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("background", payload["error"])

    def test_background_false_behaves_synchronously(self):
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=_sample_response()
        ) as post:
            result = _call("agent_research", {"query": "q", "background": False})
        self.assertNotIn("background", post.call_args.args[0])
        self.assertIn("28", result[0].text)

    def test_http_failure_on_start_is_surfaced(self):
        failure = {"status": "failed", "error": "429: rate limited"}
        with mock.patch.object(
            mcp_server, "_post_agent_research", return_value=failure
        ):
            result = _call("agent_research", {"query": "q", "background": True})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("429", payload["error"])

    def test_malformed_response_id_fails_loudly(self):
        # A null/malformed id breaks the poll contract — fail loudly rather
        # than hand the caller an envelope they can never poll (mirrors the
        # gemini_deep_research_start contract).
        queued = {"id": "resp/../evil", "status": "queued"}
        with mock.patch.object(mcp_server, "_post_agent_research", return_value=queued):
            with self.assertRaises(RuntimeError):
                _call("agent_research", {"query": "q", "background": True})


class TestAgentResearchResult(unittest.TestCase):
    def test_tool_is_listed_with_response_id_required(self):
        tools = asyncio.run(mcp_server.list_tools())
        by_name = {t.name: t for t in tools}
        self.assertIn("agent_research_result", by_name)
        self.assertEqual(
            by_name["agent_research_result"].inputSchema["required"],
            ["response_id"],
        )

    def test_invalid_response_id_fails_without_network(self):
        with mock.patch.object(mcp_server, "_get_agent_response") as get:
            result = _call(
                "agent_research_result", {"response_id": "resp/../../etc/passwd"}
            )
        get.assert_not_called()
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")

    def test_missing_response_id_fails_without_network(self):
        with mock.patch.object(mcp_server, "_get_agent_response") as get:
            result = _call("agent_research_result", {})
        get.assert_not_called()
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        # A missing id should say "required", not dump the regex contract
        # (per PR #16 review, Claude bot).
        self.assertIn("required", payload["error"])

    def test_in_progress_returns_poll_hint(self):
        for pending_status in ("queued", "in_progress"):
            with self.subTest(status=pending_status):
                data = {"id": "resp_abc", "status": pending_status}
                with mock.patch.object(
                    mcp_server, "_get_agent_response", return_value=data
                ):
                    result = _call("agent_research_result", {"response_id": "resp_abc"})
                payload = json.loads(result[0].text)
                self.assertEqual(payload["status"], pending_status)
                self.assertIn("Poll", payload["hint"])

    def test_completed_returns_formatted_answer(self):
        with mock.patch.object(
            mcp_server, "_get_agent_response", return_value=_sample_response()
        ):
            result = _call("agent_research_result", {"response_id": "resp_abc"})
        text = result[0].text
        self.assertIn("28", text)
        self.assertIn("0.00149", text)
        self.assertIn("sandbox executions: 1", text)

    def test_terminal_failure_status_is_surfaced(self):
        data = {"id": "resp_abc", "status": "cancelled", "error": "user cancelled"}
        with mock.patch.object(mcp_server, "_get_agent_response", return_value=data):
            result = _call("agent_research_result", {"response_id": "resp_abc"})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("cancelled", payload["error"])

    def test_http_failure_envelope_is_surfaced(self):
        failure = {"status": "failed", "error": "404: not found"}
        with mock.patch.object(mcp_server, "_get_agent_response", return_value=failure):
            result = _call("agent_research_result", {"response_id": "resp_abc"})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("404", payload["error"])


class TestGetAgentResponseHelper(unittest.TestCase):
    def test_gets_response_by_id_with_bearer(self):
        captured: dict = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": 1}

        class _FakeClient:
            async def get(self, url, **kwargs):
                captured["url"] = url
                captured["kwargs"] = kwargs
                return _FakeResponse()

        async def fake_get_client():
            return _FakeClient()

        with mock.patch.object(
            mcp_server, "get_api_key_from_keychain", return_value="test-key"
        ):
            with mock.patch.object(
                mcp_server, "_get_http_client", side_effect=fake_get_client
            ):
                data = asyncio.run(mcp_server._get_agent_response("resp_abc-123"))
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(
            captured["url"], "https://api.perplexity.ai/v1/responses/resp_abc-123"
        )
        self.assertEqual(
            captured["kwargs"]["headers"]["Authorization"], "Bearer test-key"
        )

    def test_helper_revalidates_id_as_defense_in_depth(self):
        with self.assertRaises(ValueError):
            asyncio.run(mcp_server._get_agent_response("resp/../evil"))

    def test_request_error_becomes_failure_envelope(self):
        class _FakeClient:
            async def get(self, url, **kwargs):
                raise mcp_server.httpx.RequestError("connection timed out")

        async def fake_get_client():
            return _FakeClient()

        with mock.patch.object(
            mcp_server, "get_api_key_from_keychain", return_value="test-key"
        ):
            with mock.patch.object(
                mcp_server, "_get_http_client", side_effect=fake_get_client
            ):
                data = asyncio.run(mcp_server._get_agent_response("resp_abc"))
        self.assertEqual(data["status"], "failed")
        self.assertIn("connection timed out", data["error"])


class TestPostAgentResearchHelper(unittest.TestCase):
    """Unit tests for the HTTP helper itself (client mocked, no network)."""

    def _run_helper(self, fake_client):
        async def fake_get_client():
            return fake_client

        with mock.patch.object(
            mcp_server, "get_api_key_from_keychain", return_value="test-key"
        ):
            with mock.patch.object(
                mcp_server, "_get_http_client", side_effect=fake_get_client
            ):
                return asyncio.run(
                    mcp_server._post_agent_research({"model": "perplexity/sonar"})
                )

    def test_posts_to_agent_endpoint_with_bearer_and_timeout(self):
        captured: dict = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": 1}

        class _FakeClient:
            async def post(self, url, **kwargs):
                captured["url"] = url
                captured["kwargs"] = kwargs
                return _FakeResponse()

        data = self._run_helper(_FakeClient())
        self.assertEqual(data, {"ok": 1})
        self.assertEqual(captured["url"], "https://api.perplexity.ai/v1/responses")
        self.assertEqual(
            captured["kwargs"]["headers"]["Authorization"], "Bearer test-key"
        )
        self.assertEqual(captured["kwargs"]["json"], {"model": "perplexity/sonar"})
        # Sandbox runs exceed the shared client's 30s default; a longer
        # per-request timeout must be passed explicitly.
        self.assertGreaterEqual(captured["kwargs"]["timeout"], 300)

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
        # Network-layer failures (connect errors, timeouts) are the most
        # likely failure mode on minutes-long sandbox runs — they must
        # return the structured envelope, not crash the tool call.
        class _FakeClient:
            async def post(self, url, **kwargs):
                raise mcp_server.httpx.RequestError("read timeout after 600s")

        data = self._run_helper(_FakeClient())
        self.assertEqual(data["status"], "failed")
        self.assertIn("read timeout after 600s", data["error"])

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


if __name__ == "__main__":
    unittest.main()
