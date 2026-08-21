#!/usr/bin/env python3
"""Tests for the coding_agent / coding_agent_result MCP surface.

`mcp_server.py` owns REGISTRATION AND DISPATCH ONLY — the loop, the sandbox
and the review diff live in `coding_agent/`. So does this file's subject:
argument validation, the background job registry, and the one thing that is
genuinely mcp_server's to solve — turning an `AgentResult` that legitimately
carries surrogate-escaped bytes into a well-formed JSON response.

Self-contained: stubs out the third-party imports (mcp, openai, httpx,
google.auth) and the Keychain lookup so the test can import mcp_server
without the full runtime environment, exactly as test_local_delegate.py,
test_agent_research.py, test_redact.py and test_session_mgmt.py each do.
No docker and no ollama: `run_coding_agent` is either driven with a fake
`SandboxOps` or patched out entirely.

Run:
    uv run --with pytest --with pathspec pytest test_coding_agent_mcp.py -q
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pathspec

from coding_agent import loop as L
from coding_agent.basetree import BaseTree
from coding_agent.loop import AgentResult, StopReason
from coding_agent.walk import Entry

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
                "mcp_server_under_test_coding_agent_mcp", SERVER_PATH
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            # The allowlist is resolved from the environment at import time;
            # a shell exporting AI_TOOLS_OLLAMA_MODELS would replace the
            # built-in tags this suite asserts against. Same scrub as
            # test_local_delegate.py, plus the sandbox image override.
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AI_TOOLS_OLLAMA_MODELS", None)
                os.environ.pop("AI_TOOLS_OLLAMA_DEFAULT_MODEL", None)
                os.environ.pop("AI_TOOLS_CODING_AGENT_IMAGE", None)
                spec.loader.exec_module(module)
    return module


mcp_server = _load_mcp_server()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reject_constant(name: str):
    """What a STRICT RFC 8259 parser does with `NaN`/`Infinity`.

    Python's `json.loads` accepts them by default, so a test that merely
    round-trips through it would pass against the bug it is meant to catch.
    """
    raise AssertionError(f"response carries a non-JSON constant: {name}")


def _result(**overrides) -> AgentResult:
    """An AgentResult with every field set, overridable per test."""
    fields: dict = {
        "stop_reason": StopReason.completed,
        "turns": 1,
        "elapsed_seconds": 0.5,
        "diff": "",
        "diff_truncated": False,
        "diff_full_path": None,
        "changed_files": [],
        "unreadable": [],
        "last_command": None,
        "transcript": [],
        "model": "m",
        "cleanup_problems": [],
    }
    fields.update(overrides)
    return AgentResult(**fields)


def _returns(result: AgentResult, seen: dict | None = None):
    """Patch `coding_agent.run_coding_agent` to hand back `result`.

    The dispatch does `from coding_agent import run_coding_agent` at call
    time, so it resolves the attribute on the real package module.
    """

    async def runner(**kw):
        if seen is not None:
            seen.update(kw)
        return result

    return mock.patch("coding_agent.run_coding_agent", runner)


def _raises(exc: BaseException):
    async def runner(**kw):
        raise exc

    return mock.patch("coding_agent.run_coding_agent", runner)


def _never_returns(started: asyncio.Event | None = None):
    async def runner(**kw):
        if started is not None:
            started.set()
        await asyncio.sleep(3600)

    return mock.patch("coding_agent.run_coding_agent", runner)


class _Ops:
    """The `SandboxOps` seam with no docker and no git: the worktree is a
    directory the test already populated, and teardown is a no-op so the
    test can still look at it afterwards."""

    def __init__(self, worktree: str) -> None:
        self.worktree = worktree

    def create_worktree(self, repo, base_ref):
        return self.worktree

    async def start_container(self, worktree, image, **kw):
        return "cid"

    async def destroy_container(self, container):
        return None

    def teardown_worktree(self, repo, worktree):
        return []


def _base_tree(files: dict[str, str] | None = None) -> BaseTree:
    files = files or {}
    entries = {path: Entry(path, "file", body.encode()) for path, body in files.items()}
    spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

    def raw(path: str) -> bool:
        return bool(spec.check_file(path).include)

    return BaseTree(entries=entries, ignore=raw, tracked=frozenset(entries))


def _say(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": []}


def _real_loop(ops: _Ops, base: BaseTree, turns=(None,)):
    """Patch in the REAL `run_coding_agent`, driven by a fake sandbox.

    Used where the point is that the AgentResult the dispatch serialises came
    out of the real byte walk and the real diff renderer — not a hand-built
    dataclass that only looks like one.
    """
    scripted = [t if t is not None else _say("done") for t in turns]

    async def chat(payload, timeout_s):
        return {"message": scripted.pop(0)}

    async def runner(**kw):
        kw.pop("chat", None)
        with mock.patch.object(L, "read_base_tree", return_value=base):
            return await L.run_coding_agent(
                chat=chat, sandbox_factory=lambda: ops, **kw
            )

    return mock.patch("coding_agent.run_coding_agent", runner)


class _McpCase(unittest.TestCase):
    """A scratch directory that passes the repo check, plus registry and
    single-slot hygiene between tests."""

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp(prefix="ca-mcp-repo-")
        os.mkdir(os.path.join(self.repo, ".git"))
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.addCleanup(mcp_server._coding_jobs.clear)
        # NOT addCleanup(self.assertFalse, ...): addCleanup binds arguments
        # NOW, so that form would assert setUp's state and never fail.
        self.addCleanup(self._assert_slot_free)

    def _assert_slot_free(self) -> None:
        self.assertFalse(L._SLOT.locked(), "the coding_agent single slot leaked")

    def args(self, **over) -> dict:
        base = {"task": "t", "repo": self.repo, "background": False}
        base.update(over)
        return base

    def text(self, name: str, arguments: dict) -> str:
        return asyncio.run(mcp_server.call_tool(name, arguments))[0].text

    def payload(self, name: str, arguments: dict) -> dict:
        return json.loads(self.text(name, arguments))

    def assertNoLoneSurrogates(self, parsed) -> None:
        """The whole property, in one line.

        `json.dumps` does NOT raise on a lone surrogate — it emits a `\\udce9`
        escape, which is invalid JSON per RFC 8259 and raises the moment the
        parsed value is re-encoded to UTF-8. That re-encode is what every
        consumer downstream of this response eventually does.
        """
        json.dumps(parsed, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# The brief's own tests, verbatim
# ---------------------------------------------------------------------------


class McpRegistration(unittest.TestCase):
    def test_tools_are_registered(self):
        names = {t.name for t in asyncio.run(mcp_server.list_tools())}
        self.assertIn("coding_agent", names)
        self.assertIn("coding_agent_result", names)

    def test_missing_task_errors_without_touching_docker(self):
        out = asyncio.run(mcp_server.call_tool("coding_agent", {"repo": "/tmp"}))
        self.assertIn("task is required", out[0].text)

    def test_model_must_be_allowlisted(self):
        out = asyncio.run(
            mcp_server.call_tool(
                "coding_agent",
                {"task": "x", "repo": "/tmp", "model": "evil:latest"},
            )
        )
        self.assertIn("not allowlisted", out[0].text)


# ---------------------------------------------------------------------------
# The description is a safety surface
# ---------------------------------------------------------------------------


class DescriptionCarriesTheHonestyClauses(unittest.TestCase):
    """The description is what a future Claude reads when deciding whether to
    call this. Spec §8's two clauses belong there, not only in the README."""

    def setUp(self) -> None:
        tools = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
        self.desc = tools["coding_agent"].description
        self.poll = tools["coding_agent_result"].description

    def test_says_nothing_is_applied_and_the_caller_is_the_gate(self):
        self.assertIn("NOTHING IS APPLIED", self.desc)
        self.assertIn("gate", self.desc)

    def test_clause_one_unreliable_gate_and_review_matters_most(self):
        self.assertIn("MEASURED-UNRELIABLE", self.desc)
        self.assertIn("defect gates", self.desc)
        self.assertIn("matters MOST", self.desc)

    def test_clause_two_the_image_drifts_and_nothing_installs_at_runtime(self):
        self.assertIn("WILL drift", self.desc)
        self.assertIn("no network", self.desc)
        self.assertIn("installed at runtime", self.desc)

    def test_says_the_slot_is_single_and_rejects_rather_than_queues(self):
        self.assertIn("One run at a time", self.desc)
        self.assertIn("rejected, not queued", self.desc)

    def test_the_poll_tool_points_at_the_fields_a_reviewer_must_not_miss(self):
        self.assertIn("unreadable", self.poll)
        self.assertIn("cleanup_problems", self.poll)
        self.assertIn("single-collect", self.poll)


class ToolSchema(unittest.TestCase):
    def setUp(self) -> None:
        tools = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
        self.schema = tools["coding_agent"].inputSchema

    def test_model_enum_is_the_shared_allowlist(self):
        """A schema enum that disagrees with the validator is a bug: the
        client would offer a tag the server then refuses."""
        prop = self.schema["properties"]["model"]
        self.assertEqual(prop["enum"], list(mcp_server.OLLAMA_DELEGATE_MODELS))
        self.assertIn(prop["default"], prop["enum"])

    def test_default_model_is_the_31b_review_tag(self):
        self.assertEqual(
            self.schema["properties"]["model"]["default"], "gemma4:31b-nvfp4"
        )

    def test_budgets_are_bounded_at_both_ends(self):
        turns = self.schema["properties"]["max_turns"]
        seconds = self.schema["properties"]["max_seconds"]
        self.assertEqual((turns["minimum"], turns["maximum"]), (1, 60))
        self.assertEqual((seconds["minimum"], seconds["maximum"]), (1, 1800))

    def test_background_defaults_to_true(self):
        self.assertTrue(self.schema["properties"]["background"]["default"])
        self.assertEqual(self.schema["required"], ["task", "repo"])


# ---------------------------------------------------------------------------
# Surrogate escapes at the response boundary
# ---------------------------------------------------------------------------


class SurrogatesReachTheResponse(_McpCase):
    """coding_agent decodes on-disk bytes with `surrogateescape` ON PURPOSE —
    scrubbing in the diff layer would falsify what the human reviews. So they
    have to be handled HERE, at the response, and handled without dropping
    the path: a file that silently vanishes from the review is the failure
    this exists to prevent."""

    def setUp(self) -> None:
        super().setUp()
        self.wt = tempfile.mkdtemp(prefix="ca-mcp-wt-")
        self.addCleanup(shutil.rmtree, self.wt, ignore_errors=True)

    def _run_real(self) -> dict:
        with _real_loop(_Ops(self.wt), _base_tree()):
            return self.payload("coding_agent", self.args())

    def test_invalid_utf8_file_bytes_survive_as_wellformed_json(self):
        """A REAL file holding REAL invalid UTF-8 — measured to be exactly
        what a --network=none container CAN author on the bind-mounted
        worktree of a macOS host (the filename vector cannot; see below)."""
        Path(self.wt, "notes.txt").write_bytes(b"caf\xe9 binary\n")
        parsed = self._run_real()
        self.assertNoLoneSurrogates(parsed)
        self.assertEqual(parsed["changed_files"], ["notes.txt"])
        # the byte is still IN the diff the human reads, rendered visibly
        self.assertIn(r"caf\xe9 binary", parsed["diff"])

    def test_invalid_utf8_filename_stays_visible(self):
        raw_name = b"bad\xff.txt"
        target = os.path.join(os.fsencode(self.wt), raw_name)
        try:
            with open(target, "wb") as fh:
                fh.write(b"x\n")
            on_disk = True
        except OSError:
            # APFS/HFS+ reject invalid-UTF-8 names outright (EILSEQ) and so,
            # measured, does a docker bind mount backed by one. The vector is
            # unreachable on THIS host but reachable on Linux, where the same
            # string arrives in AgentResult — so drive the boundary with it
            # rather than skipping the property outright.
            on_disk = False

        if on_disk:
            parsed = self._run_real()
            self.assertNoLoneSurrogates(parsed)
            self.assertEqual(parsed["changed_files"], [r"bad\xff.txt"])
            self.assertIn(r"bad\xff.txt", parsed["diff"])
            return

        decoded = os.fsdecode(raw_name)
        self.assertIn("\udcff", decoded)  # control: really is a lone surrogate
        with _returns(
            _result(
                changed_files=[decoded],
                diff=f"diff --git a/{decoded} b/{decoded}\n",
                unreadable=[{"path": decoded, "reason": "PermissionError"}],
            )
        ):
            parsed = self.payload("coding_agent", self.args())
        self.assertNoLoneSurrogates(parsed)
        self.assertEqual(parsed["changed_files"], [r"bad\xff.txt"])
        self.assertEqual(parsed["unreadable"][0]["path"], r"bad\xff.txt")

    def test_surrogates_nested_in_the_transcript_are_scrubbed_too(self):
        """The transcript is a list of dicts of model-chosen values — a
        top-level-strings-only scrub would leave those untouched."""
        with _returns(
            _result(
                transcript=[
                    {"turn": 1, "args": {"path": os.fsdecode(b"p\xff")}},
                    {"turn": 2, "out": [os.fsdecode(b"\xfe")]},
                ],
                last_command={"cmd": os.fsdecode(b"cat \xfd"), "exit": 1},
            )
        ):
            parsed = self.payload("coding_agent", self.args())
        self.assertNoLoneSurrogates(parsed)
        self.assertEqual(parsed["transcript"][0]["args"]["path"], r"p\xff")
        self.assertEqual(parsed["transcript"][1]["out"], [r"\xfe"])
        self.assertEqual(parsed["last_command"]["cmd"], r"cat \xfd")

    def test_legitimate_unicode_is_left_exactly_as_it_is(self):
        """The scrub must not be an ASCII filter: mangling a real filename
        into mojibake is its own kind of hiding place."""
        with _returns(
            _result(diff="+café — naïve 日本語\n", changed_files=["café.py"])
        ):
            parsed = self.payload("coding_agent", self.args())
        self.assertEqual(parsed["diff"], "+café — naïve 日本語\n")
        self.assertEqual(parsed["changed_files"], ["café.py"])

    def test_a_non_finite_float_cannot_poison_the_whole_response(self):
        """SF-5 / NF-5. Same bug class as the lone surrogate, in the branch
        that was missed.

        `_as_dict` deliberately supports tool `arguments` arriving as a JSON
        *string* and parses them with `json.loads`, which accepts
        `NaN`/`Infinity` by default. `_summarize_args` passes floats through
        untouched, `_json_safe` only walked str/dict/list, and `json.dumps`
        then emitted a bare `NaN` — which a strict RFC 8259 parser rejects,
        taking the ENTIRE payload with it. The human loses the diff, not just
        the number.
        """
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with _returns(
                    _result(
                        transcript=[
                            {"turn": 1, "tool": "read_file", "args": {"path": bad}}
                        ]
                    )
                ):
                    text = self.text("coding_agent", self.args())
                # MECHANISM: a bare NaN/Infinity token is exactly what a
                # strict parser refuses, so its absence is the property.
                for token in ("NaN", "Infinity", "-Infinity"):
                    self.assertNotIn(f": {token}", text)
                parsed = json.loads(text, parse_constant=_reject_constant)
                rendered = parsed["transcript"][0]["args"]["path"]
                self.assertIsInstance(rendered, str)
                self.assertIn("non-finite", rendered)

    def test_ordinary_numbers_are_left_exactly_as_they_are(self):
        """The coercion must not touch the numbers a normal result carries —
        a stringified `turns` or `elapsed_seconds` would break every consumer."""
        with _returns(_result(turns=7, elapsed_seconds=1.25)):
            parsed = self.payload("coding_agent", self.args())
        self.assertEqual(parsed["turns"], 7)
        self.assertEqual(parsed["elapsed_seconds"], 1.25)
        self.assertIsInstance(parsed["diff_truncated"], bool)

    def test_a_surrogate_outside_the_surrogateescape_range_is_handled(self):
        """surrogateescape only owns U+DC80-U+DCFF; a model can put any lone
        surrogate in a tool argument and it reaches the transcript."""
        with _returns(_result(diff="\ud800 lead surrogate\n")):
            parsed = self.payload("coding_agent", self.args())
        self.assertNoLoneSurrogates(parsed)
        self.assertIn("lead surrogate", parsed["diff"])

    def test_the_polled_result_is_scrubbed_the_same_way(self):
        """Background and sync must not diverge — the scrub belongs to both."""

        async def scenario():
            with _returns(_result(changed_files=[os.fsdecode(b"bad\xff.txt")])):
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
                job_id = json.loads(out[0].text)["job_id"]
                await mcp_server._coding_jobs[job_id]["task"]
                out = await mcp_server.call_tool(
                    "coding_agent_result", {"job_id": job_id}
                )
            return json.loads(out[0].text)

        parsed = asyncio.run(scenario())
        self.assertNoLoneSurrogates(parsed)
        self.assertEqual(parsed["changed_files"], [r"bad\xff.txt"])


# ---------------------------------------------------------------------------
# The job registry
# ---------------------------------------------------------------------------


class RegistryIsSeparateFromLocalDelegate(_McpCase):
    def test_the_two_registries_are_distinct_objects(self):
        self.assertIsNot(mcp_server._coding_jobs, mcp_server._delegate_jobs)

    def test_a_coding_run_neither_fills_nor_is_blocked_by_the_delegate_slots(self):
        """_DELEGATE_JOB_CAP (=4) is local_delegate's shared inference budget.
        coding_agent is single-slot and must not spend one of those, nor be
        refused when all four are taken."""

        async def scenario():
            async def never():
                await asyncio.sleep(3600)

            loop = asyncio.get_running_loop()
            for i in range(mcp_server._DELEGATE_JOB_CAP):
                mcp_server._delegate_jobs[f"{i:032x}"] = {
                    "task": loop.create_task(never()),
                    "started": 0.0,
                }
            try:
                with _never_returns():
                    out = await mcp_server.call_tool(
                        "coding_agent", self.args(background=True)
                    )
                started = json.loads(out[0].text)
                return started, len(mcp_server._delegate_jobs)
            finally:
                for job in list(mcp_server._delegate_jobs.values()):
                    job["task"].cancel()
                for job in list(mcp_server._coding_jobs.values()):
                    job["task"].cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await job["task"]
                mcp_server._delegate_jobs.clear()

        started, delegate_count = asyncio.run(scenario())
        self.assertEqual(started["status"], "started")
        self.assertEqual(delegate_count, mcp_server._DELEGATE_JOB_CAP)
        self.assertEqual(len(mcp_server._coding_jobs), 1)


class BackgroundJobs(_McpCase):
    def _finished_job(self, loop, started: float) -> str:
        async def done():
            return _result()

        job_id = os.urandom(16).hex
        task = loop.create_task(done())
        mcp_server._coding_jobs[job_id] = {"task": task, "started": started}
        return job_id

    def test_background_is_the_default_and_returns_a_32_hex_job_id(self):
        async def scenario():
            # Deliberately a run that COMPLETES: if `background` stopped
            # defaulting to true this would return the result inline rather
            # than hanging the suite on a run that never finishes.
            with _returns(_result()):
                out = await mcp_server.call_tool(
                    "coding_agent", {"task": "t", "repo": self.repo}
                )
                body = json.loads(out[0].text)
                for job in list(mcp_server._coding_jobs.values()):
                    await job["task"]
            return body

        body = asyncio.run(scenario())
        self.assertEqual(body["status"], "started")
        self.assertRegex(body["job_id"], r"^[0-9a-f]{32}$")

    def test_a_running_job_polls_as_running_with_elapsed_seconds(self):
        async def scenario():
            with _never_returns():
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
            job_id = json.loads(out[0].text)["job_id"]
            out = await mcp_server.call_tool("coding_agent_result", {"job_id": job_id})
            body = json.loads(out[0].text)
            for job in list(mcp_server._coding_jobs.values()):
                job["task"].cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job["task"]
            return body

        body = asyncio.run(scenario())
        self.assertEqual(body["status"], "running")
        self.assertIsInstance(body["elapsed_s"], int)

    def test_results_are_single_collect(self):
        async def scenario():
            with _returns(_result(turns=7)):
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
                job_id = json.loads(out[0].text)["job_id"]
                await mcp_server._coding_jobs[job_id]["task"]
                first = await mcp_server.call_tool(
                    "coding_agent_result", {"job_id": job_id}
                )
                second = await mcp_server.call_tool(
                    "coding_agent_result", {"job_id": job_id}
                )
            return first[0].text, second[0].text

        first, second = asyncio.run(scenario())
        self.assertEqual(json.loads(first)["turns"], 7)
        self.assertIn("single-collect", second)
        self.assertEqual(mcp_server._coding_jobs, {})

    def test_a_malformed_job_id_is_rejected_by_the_VALIDATOR_not_the_lookup(self):
        """Asserting "Error:" appeared was decorative: replacing
        `_CODING_JOB_ID_RE` with `^.*$` left the whole suite green, because a
        malformed id then fell through to the `Unknown job_id` branch and
        produced an "Error:" of its own. The test pinned the message, not the
        guard.

        The guard is worth pinning on its own: the unknown-id branch
        interpolates the caller's string into the human's context with
        `{job_id!r}`, so the validator is what keeps an arbitrary-length,
        arbitrary-charset value out of it. The two errors are now told apart.
        """
        for bad in (None, "", "not-hex", "abc", 17, "0" * 31, "0" * 33, "A" * 32):
            with self.subTest(job_id=bad):
                out = self.text("coding_agent_result", {"job_id": bad})
                self.assertIn("Error:", out)
                self.assertIn("must be the 32-hex id", out)
                self.assertNotIn("Unknown job_id", out)

    def test_a_well_formed_but_unknown_job_id_reaches_the_lookup(self):
        """The other side of the same guard: a value that PASSES validation
        must get the lookup's answer, not the validator's. Without this, a
        validator that rejected everything would satisfy the test above."""
        out = self.text("coding_agent_result", {"job_id": "a" * 32})
        self.assertIn("Unknown job_id", out)
        self.assertNotIn("must be the 32-hex id", out)

    def test_under_the_retention_bound_nothing_is_evicted(self):
        """The eviction slice must be guarded: an unguarded `done_ids[:excess]`
        with a NEGATIVE excess drops the NEWEST jobs instead of nothing."""

        async def scenario():
            loop = asyncio.get_running_loop()
            ids = [
                self._finished_job(loop, float(i))
                for i in range(mcp_server._CODING_DONE_RETAINED - 1)
            ]
            for jid in ids:
                await mcp_server._coding_jobs[jid]["task"]
            with _never_returns():
                await mcp_server.call_tool("coding_agent", self.args(background=True))
            survivors = set(mcp_server._coding_jobs) & set(ids)
            for job in list(mcp_server._coding_jobs.values()):
                job["task"].cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job["task"]
            return survivors, set(ids)

        survivors, ids = asyncio.run(scenario())
        self.assertEqual(survivors, ids)

    def test_over_the_retention_bound_the_oldest_completed_jobs_are_dropped(self):
        async def scenario():
            loop = asyncio.get_running_loop()
            n = mcp_server._CODING_DONE_RETAINED + 2
            ids = [self._finished_job(loop, float(i)) for i in range(n)]
            for jid in ids:
                await mcp_server._coding_jobs[jid]["task"]
            with _never_returns():
                await mcp_server.call_tool("coding_agent", self.args(background=True))
            survivors = [jid for jid in ids if jid in mcp_server._coding_jobs]
            for job in list(mcp_server._coding_jobs.values()):
                job["task"].cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job["task"]
            return survivors, ids

        survivors, ids = asyncio.run(scenario())
        self.assertEqual(survivors, ids[2:])  # the two OLDEST went

    def test_background_and_sync_return_the_same_shape(self):
        async def scenario():
            with _returns(_result()):
                sync = await mcp_server.call_tool("coding_agent", self.args())
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
                job_id = json.loads(out[0].text)["job_id"]
                await mcp_server._coding_jobs[job_id]["task"]
                polled = await mcp_server.call_tool(
                    "coding_agent_result", {"job_id": job_id}
                )
            return json.loads(sync[0].text), json.loads(polled[0].text)

        sync, polled = asyncio.run(scenario())
        self.assertEqual(sorted(sync), sorted(polled))
        self.assertEqual(
            sorted(sync),
            [
                "changed_files",
                "cleanup_problems",
                "diff",
                "diff_full_path",
                "diff_truncated",
                "elapsed_seconds",
                "last_command",
                "model",
                "stop_reason",
                "transcript",
                "turns",
                "unreadable",
            ],
        )
        self.assertEqual(sync["stop_reason"], "completed")


# ---------------------------------------------------------------------------
# Cancellation and failure
# ---------------------------------------------------------------------------


class CancellationAndFailure(_McpCase):
    def test_a_cancelled_background_run_is_reported_not_raised(self):
        """`run_coding_agent` re-raises CancelledError once its OWN cleanup
        has finished. CancelledError is a BaseException, so `except Exception`
        does not catch it — it escapes the collector and asyncio then marks
        the MCP request's own task cancelled."""

        async def scenario():
            started = asyncio.Event()
            with _never_returns(started):
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
            job_id = json.loads(out[0].text)["job_id"]
            await started.wait()
            task = mcp_server._coding_jobs[job_id]["task"]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            out = await mcp_server.call_tool("coding_agent_result", {"job_id": job_id})
            return json.loads(out[0].text)

        body = asyncio.run(scenario())
        self.assertEqual(body["status"], "cancelled")
        self.assertIn("tore the sandbox down", body["error"])

    def test_a_failing_background_run_reports_the_reason(self):
        async def scenario():
            with _raises(RuntimeError(L._SLOT_BUSY)):
                out = await mcp_server.call_tool(
                    "coding_agent", self.args(background=True)
                )
                job_id = json.loads(out[0].text)["job_id"]
                with contextlib.suppress(RuntimeError):
                    await mcp_server._coding_jobs[job_id]["task"]
                out = await mcp_server.call_tool(
                    "coding_agent_result", {"job_id": job_id}
                )
            return json.loads(out[0].text)

        body = asyncio.run(scenario())
        self.assertEqual(body["status"], "failed")
        self.assertIn("RuntimeError", body["error"])
        self.assertIn("already in progress", body["error"])

    def test_a_failing_sync_run_reports_instead_of_raising_at_the_protocol(self):
        with _raises(RuntimeError(L._SLOT_BUSY)):
            body = self.payload("coding_agent", self.args())
        self.assertEqual(body["status"], "failed")
        self.assertIn("already in progress", body["error"])

    def test_sync_cancellation_propagates_rather_than_becoming_a_status(self):
        """Our own request being cancelled must stay a cancellation: turning
        it into a `failed` envelope would leave the caller believing the run
        merely errored while the loop's teardown decides the container's
        fate."""

        async def scenario():
            with _raises(asyncio.CancelledError()):
                await mcp_server.call_tool("coding_agent", self.args())

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Argument validation (nothing below reaches the sandbox)
# ---------------------------------------------------------------------------


class ArgumentValidation(_McpCase):
    def setUp(self) -> None:
        super().setUp()
        # Any dispatch that reaches this has skipped a validation gate.
        guard = mock.MagicMock(side_effect=AssertionError("run_coding_agent reached"))
        patcher = mock.patch("coding_agent.run_coding_agent", guard)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertRejected(self, arguments: dict, needle: str) -> None:
        out = self.text("coding_agent", arguments)
        self.assertIn(needle, out)
        self.assertEqual(mcp_server._coding_jobs, {})

    def test_task_must_be_a_non_empty_string(self):
        for bad in (None, "", "   ", 5, ["x"]):
            with self.subTest(task=bad):
                self.assertRejected(self.args(task=bad), "task is required")

    def test_repo_must_be_an_absolute_path_to_a_git_repository(self):
        relative = os.path.relpath(self.repo)
        not_a_repo = tempfile.mkdtemp(prefix="ca-mcp-plain-")
        self.addCleanup(shutil.rmtree, not_a_repo, ignore_errors=True)
        for bad in (None, "", 5, relative, not_a_repo, "/nope/nowhere"):
            with self.subTest(repo=bad):
                self.assertRejected(self.args(repo=bad), "absolute path to a git")

    def test_a_linked_worktree_whose_dot_git_is_a_FILE_is_accepted(self):
        linked = tempfile.mkdtemp(prefix="ca-mcp-linked-")
        self.addCleanup(shutil.rmtree, linked, ignore_errors=True)
        Path(linked, ".git").write_text("gitdir: /elsewhere/.git/worktrees/w\n")
        with _returns(_result(turns=3)):
            body = self.payload("coding_agent", self.args(repo=linked))
        self.assertEqual(body["turns"], 3)

    def test_max_turns_is_range_checked_not_silently_clamped(self):
        """`min(int(x), 60)` accepts 0 and -1 (an instantly-dead run) and
        raises ValueError out of the MCP handler on "abc"; bool is an int
        subclass, so True became a one-turn budget."""
        for bad in ("abc", 0, -1, 61, 3.5, True, None):
            with self.subTest(max_turns=bad):
                self.assertRejected(
                    self.args(max_turns=bad), "max_turns must be an integer"
                )

    def test_max_seconds_is_range_checked(self):
        for bad in ("abc", 0, -5, 1801, 12.5, False, None):
            with self.subTest(max_seconds=bad):
                self.assertRejected(
                    self.args(max_seconds=bad), "max_seconds must be an integer"
                )

    def test_base_ref_must_not_look_like_a_git_option(self):
        for bad in ("--upload-pack=touch /tmp/x", "-q", "", "  ", 5, None, {"a": 1}):
            with self.subTest(base_ref=bad):
                self.assertRejected(self.args(base_ref=bad), "base_ref must be")

    def test_background_must_be_a_boolean(self):
        for bad in ("true", 1, None, []):
            with self.subTest(background=bad):
                self.assertRejected(
                    self.args(background=bad), "background must be a JSON boolean"
                )

    def test_the_model_error_names_the_allowlist(self):
        out = self.text("coding_agent", self.args(model="evil:latest"))
        self.assertIn("not allowlisted", out)
        for tag in mcp_server.OLLAMA_DELEGATE_MODELS:
            self.assertIn(tag, out)


# ---------------------------------------------------------------------------
# What the dispatch hands the package
# ---------------------------------------------------------------------------


class DispatchWiring(_McpCase):
    def _seen(self, **over) -> dict:
        seen: dict = {}
        with _returns(_result(), seen):
            self.text("coding_agent", self.args(**over))
        return seen

    def test_defaults_are_forwarded_verbatim(self):
        seen = self._seen()
        self.assertEqual(seen["task"], "t")
        self.assertEqual(seen["repo"], self.repo)
        self.assertEqual(seen["base_ref"], "HEAD")
        self.assertEqual(seen["model"], "gemma4:31b-nvfp4")
        self.assertEqual(seen["max_turns"], 25)
        self.assertEqual(seen["max_seconds"], 600.0)
        self.assertIsInstance(seen["max_seconds"], float)

    def test_the_pinned_sandbox_image_is_wired(self):
        self.assertEqual(self._seen()["image"], "ai-tools-coding-agent:latest")

    def test_the_chat_callable_is_the_ollama_poster(self):
        self.assertIs(self._seen()["chat"], mcp_server._post_ollama_chat)

    def test_an_explicit_allowlisted_model_wins(self):
        self.assertEqual(
            self._seen(model="qwen3.8:27b-nvfp4")["model"], "qwen3.8:27b-nvfp4"
        )

    def test_no_sandbox_factory_is_injected(self):
        """The dispatch must not choose the sandbox; the package does."""
        self.assertNotIn("sandbox_factory", self._seen())


class SandboxImageResolution(unittest.TestCase):
    def _with(self, value):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        if value is None:
            os.environ.pop(mcp_server._CODING_AGENT_IMAGE_ENV_VAR, None)
        else:
            os.environ[mcp_server._CODING_AGENT_IMAGE_ENV_VAR] = value
        return mcp_server._coding_agent_image()

    def test_default_is_the_pinned_tag(self):
        self.assertEqual(self._with(None), "ai-tools-coding-agent:latest")

    def test_an_override_is_honoured(self):
        self.assertEqual(
            self._with("registry.local/agent:2026-08"), "registry.local/agent:2026-08"
        )

    def test_a_flag_like_or_split_override_falls_back(self):
        """The tag sits in `docker run`'s argv AFTER the flags, so a leading
        '-' or an embedded space would smuggle docker options into the
        sandbox launch."""
        for bad in ("--privileged", "-v/:/host", "", "   ", "img --privileged"):
            with self.subTest(value=bad):
                self.assertEqual(self._with(bad), "ai-tools-coding-agent:latest")


class DefaultModelResolution(unittest.TestCase):
    def test_it_is_the_31b_tag_and_is_allowlisted(self):
        model = mcp_server._coding_agent_default_model()
        self.assertEqual(model, "gemma4:31b-nvfp4")
        self.assertIn(model, mcp_server.OLLAMA_DELEGATE_MODELS)

    def test_it_never_shells_out(self):
        """The brief resolved this with `subprocess.run(["local-model",
        "review"], timeout=5)` — a blocking call on the event loop whose
        answer comes from a binary outside this repo. That binary IS
        installed here, which is exactly what made the path look green."""
        with mock.patch("subprocess.run", side_effect=AssertionError("shelled out")):
            self.assertEqual(
                mcp_server._coding_agent_default_model(), "gemma4:31b-nvfp4"
            )

    def test_an_allowlist_without_the_preferred_tag_falls_back_to_its_first(self):
        with mock.patch.object(
            mcp_server, "OLLAMA_DELEGATE_MODELS", ("only:tag", "other:tag")
        ):
            with mock.patch.object(
                mcp_server, "OLLAMA_DELEGATE_DEFAULT_MODEL", "only:tag"
            ):
                self.assertEqual(mcp_server._coding_agent_default_model(), "only:tag")


# ---------------------------------------------------------------------------
# Open-file limit
# ---------------------------------------------------------------------------


class NofileLimit(unittest.TestCase):
    """launchd hands this process a soft RLIMIT_NOFILE of 256 — verified with
    `launchctl limit maxfiles` (256 / unlimited). An interactive shell reports
    a far larger `ulimit -n`, but that is the SHELL's raised value, not the
    one the launchd-spawned server gets. The coding_agent walk holds one
    directory descriptor per level on top of the httpx pool and every
    subprocess pipe."""

    def _fake_resource(self, soft, hard, *, refuse=()):
        calls: list = []

        def setrlimit(key, value):
            calls.append(value)
            if value[0] in refuse:
                raise ValueError("kernel says no")

        return calls, types.SimpleNamespace(
            RLIMIT_NOFILE=7,
            RLIM_INFINITY=-1,
            getrlimit=lambda key: (soft, hard),
            setrlimit=setrlimit,
        )

    def test_a_finite_hard_limit_is_taken(self):
        calls, fake = self._fake_resource(256, 4096)
        with mock.patch.object(mcp_server, "resource", fake):
            self.assertEqual(mcp_server._raise_nofile_limit(), (4096, 4096))
        self.assertEqual(calls, [(4096, 4096)])

    def test_an_infinite_hard_limit_is_attempted_then_stepped_down(self):
        """RLIM_INFINITY is a sentinel, not a comparable maximum (-1 on Linux),
        and some kernels refuse an infinite soft limit."""
        calls, fake = self._fake_resource(256, -1, refuse=(-1,))
        with mock.patch.object(mcp_server, "resource", fake):
            self.assertEqual(mcp_server._raise_nofile_limit(), (8192, -1))
        self.assertEqual(calls, [(-1, -1), (8192, -1)])

    def test_an_already_high_limit_is_left_alone(self):
        calls, fake = self._fake_resource(65536, 65536)
        with mock.patch.object(mcp_server, "resource", fake):
            self.assertIsNone(mcp_server._raise_nofile_limit())
        self.assertEqual(calls, [])

    def test_a_refusing_kernel_is_never_fatal(self):
        calls, fake = self._fake_resource(256, 4096, refuse=(4096, 8192))
        with mock.patch.object(mcp_server, "resource", fake):
            self.assertIsNone(mcp_server._raise_nofile_limit())
        self.assertEqual(len(calls), 2)

    def test_windows_has_no_resource_module(self):
        with mock.patch.object(mcp_server, "resource", None):
            self.assertIsNone(mcp_server._raise_nofile_limit())

    def test_it_really_lifts_a_launchd_sized_limit_on_this_os(self):
        r = mcp_server.resource
        if r is None:
            self.skipTest("no resource module on this platform")
        soft, hard = r.getrlimit(r.RLIMIT_NOFILE)
        if hard != r.RLIM_INFINITY and hard <= 256:
            self.skipTest("hard limit leaves no headroom above 256")
        self.addCleanup(r.setrlimit, r.RLIMIT_NOFILE, (soft, hard))
        r.setrlimit(r.RLIMIT_NOFILE, (256, hard))  # what launchd hands us
        self.assertEqual(r.getrlimit(r.RLIMIT_NOFILE)[0], 256)  # control
        self.assertIsNotNone(mcp_server._raise_nofile_limit())
        new_soft = r.getrlimit(r.RLIMIT_NOFILE)[0]
        self.assertTrue(
            new_soft == r.RLIM_INFINITY or new_soft > 256, f"soft is still {new_soft}"
        )

    def test_it_is_wired_into_main_and_not_merely_defined(self):
        seen: list = []

        @contextlib.asynccontextmanager
        async def fake_stdio():
            yield (None, None)

        with mock.patch.object(mcp_server, "stdio_server", fake_stdio):
            with mock.patch.object(
                mcp_server, "_raise_nofile_limit", lambda: seen.append(True)
            ):
                asyncio.run(mcp_server.main())
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
