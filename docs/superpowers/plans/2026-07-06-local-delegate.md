# local_delegate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `local_delegate` / `local_delegate_result` MCP tools that send tasks to a local Ollama qwen3.6 model (sync or background), per the approved spec at `docs/superpowers/specs/2026-07-06-local-delegate-design.md`.

**Architecture:** Two new tools in the existing single-file server `mcp_server.py`, following the `agent_research` / `agent_research_result` precedent: native Ollama `POST /api/chat` through the existing shared async httpx client with explicit per-request timeouts; an in-memory (never disk) background-job registry with a hard cap; server-side model allowlist; fail-closed validation on every parameter.

**Tech Stack:** Python ≥3.12, `httpx` (already a dep), `unittest` self-contained test files (repo pattern), `uv run` PEP 723, ruff format.

## Global Constraints

- Branch: `feat/local-delegate` (already exists; spec committed there). Never push to `main`; finish with a PR via `gh pr create`.
- All work in `/Users/jasonvassallo/Documents/Code/ai-tools-mcp`.
- No new dependencies; no disk writes for prompts/results/job state; no new secrets.
- Specific exception types only — never bare `except Exception`.
- Everything logged or embedded in an error payload passes through `redact_secrets`.
- Model allowlist (exact strings): `qwen3.6:35b-a3b-coding-nvfp4` (default), `qwen3.6:35b-a3b-coding-nvfp4-32k`, `qwen3.6:35b-a3b-coding-nvfp4-256k`.
- **[v1.1]** Endpoint resolution: ordered **chain** — env `AI_TOOLS_OLLAMA_URLS` (comma-separated; singular `AI_TOOLS_OLLAMA_URL` honored as a one-item chain for compat), else default chain `http://localhost:11434`, `https://ollama-mbp.djvassallo.com`; Keychain service `OLLAMA_URL` (account = current user) appended if set. Per-call: probe each endpoint's `GET /api/tags` (2 s timeout), pick the first serving the requested tag; cache (model → endpoint) 60 s. Localhost may be http; **non-localhost must be https** (rejected at parse). Remote endpoints authenticate with Cloudflare Access service-token headers from Keychain (`OLLAMA_CF_ACCESS_CLIENT_ID` / `OLLAMA_CF_ACCESS_CLIENT_SECRET`); creds absent → endpoint **skipped**, never called bare.
- Sync timeout default 300 s, cap 600 s; background ceiling 1800 s; job cap 4; **[v1.1]** `keep_alive` default **omitted** (inherit the server's `OLLAMA_KEEP_ALIVE`); when provided it must match `^(0|[1-9][0-9]{0,3}(s|m|h))$`.
- **[v1.1] Execution order:** Tasks 1, 2 were built to v1.0 and already landed. Remaining order: **Task 8 → Task 9 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7.** Tasks 8/9 (appended at the end of this file) retrofit the v1.1 amendment onto Tasks 1/2's code; Tasks 4/5/6 below are already edited to v1.1.
- Run tests with: `uv run --with pytest pytest test_local_delegate.py -q` (matches `.github/workflows/tests.yml`). Format-check with `uv tool run ruff format --check <files>`.
- Every commit message ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (and the Claude-Session URL line if the executor has one).

---

### Task 1: Constants + endpoint resolution (`_resolve_ollama_url`)

**Files:**
- Modify: `mcp_server.py` (new section after the agent-research helpers, i.e. after `_render_agent_research`, ~line 1108; new imports at top)
- Create: `test_local_delegate.py` (repo root)

**Interfaces:**
- Produces (used by Tasks 2–5):
  - `OLLAMA_DELEGATE_MODELS: tuple[str, ...]`, `OLLAMA_DELEGATE_DEFAULT_MODEL: str`
  - `_OLLAMA_DEFAULT_URL: str`, `_OLLAMA_URL_ENV_VAR: str`, `_OLLAMA_URL_KEYCHAIN_SERVICE: str`
  - `_DELEGATE_KEEP_ALIVE_RE`, `_DELEGATE_KEEP_ALIVE_DEFAULT = "5m"`
  - `_DELEGATE_TIMEOUT_DEFAULT_S = 300`, `_DELEGATE_TIMEOUT_MAX_S = 600`
  - `_DELEGATE_BG_CEILING_S = 1800.0`, `_DELEGATE_JOB_CAP = 4`, `_DELEGATE_JOB_ID_RE`
  - `def _resolve_ollama_url() -> str` — returns base URL without trailing slash; raises `ValueError` on a non-http(s) configured URL. Blocking (may shell out to `security`); callers in async context wrap with `asyncio.to_thread`.

- [ ] **Step 1: Create `test_local_delegate.py` with the repo's stub-import harness and the endpoint-resolution tests**

Copy the harness **verbatim** from `test_agent_research.py`: the module docstring pattern, imports, `_build_stub_modules()` (lines 43–146 there), and `_load_mcp_server()` — changing only the spec name string to `"mcp_server_under_test_local_delegate"` (per-file unique, see the comment in the original). Then add:

```python
mcp_server = _load_mcp_server()


def _call(name: str, arguments: dict) -> list:
    return asyncio.run(mcp_server.call_tool(name, arguments))


class TestResolveOllamaUrl(unittest.TestCase):
    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_URL": "http://jvmacmini:11434/"}):
            self.assertEqual(mcp_server._resolve_ollama_url(), "http://jvmacmini:11434")

    def test_keychain_second(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TOOLS_OLLAMA_URL", None)
            with mock.patch.object(
                mcp_server, "get_api_key_from_keychain", return_value="https://mini.tail:11434"
            ) as kc:
                self.assertEqual(mcp_server._resolve_ollama_url(), "https://mini.tail:11434")
        kc.assert_called_once_with("OLLAMA_URL", getpass.getuser())

    def test_default_localhost_when_neither(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_TOOLS_OLLAMA_URL", None)
            with mock.patch.object(
                mcp_server, "get_api_key_from_keychain", side_effect=ValueError("not found")
            ):
                self.assertEqual(mcp_server._resolve_ollama_url(), "http://localhost:11434")

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
```

Add `import getpass` and `import os` to the test file's imports.

- [ ] **Step 2: Run — verify FAIL**

Run: `cd /Users/jasonvassallo/Documents/Code/ai-tools-mcp && uv run --with pytest pytest test_local_delegate.py -q`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute '_resolve_ollama_url'`

- [ ] **Step 3: Implement constants + `_resolve_ollama_url` in `mcp_server.py`**

Add `import getpass` and `import urllib.parse` to the stdlib import block at the top (keep alphabetical). Then, after `_render_agent_research` (before `@server.list_tools()`), add:

```python
# ─── Local delegate (Ollama) ──────────────────────────────────────────
#
# Third tool family: delegate tasks to a LOCAL Ollama model. Inverts the
# data-flow of every other family — exists precisely so input text can
# stay on-device (plus quota offload, second opinions, background/batch
# work). The server only CALLS an already-running Ollama; it never reads
# files, pulls models, or manages the Ollama service (least privilege).
#
# Native /api/chat (not the OpenAI-compat endpoint) because only the
# native API accepts `think` — qwen3.6 is a thinking model and
# think:false is required for fast structured work.

OLLAMA_DELEGATE_MODELS: tuple[str, ...] = (
    "qwen3.6:35b-a3b-coding-nvfp4",
    "qwen3.6:35b-a3b-coding-nvfp4-32k",
    "qwen3.6:35b-a3b-coding-nvfp4-256k",
)
OLLAMA_DELEGATE_DEFAULT_MODEL = OLLAMA_DELEGATE_MODELS[0]

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_URL_ENV_VAR = "AI_TOOLS_OLLAMA_URL"
_OLLAMA_URL_KEYCHAIN_SERVICE = "OLLAMA_URL"

# `0` (unload immediately) or 1-9999 seconds/minutes/hours. Strict so a
# malformed value cannot smuggle arbitrary JSON into the Ollama request.
_DELEGATE_KEEP_ALIVE_RE = re.compile(r"^(0|[1-9][0-9]{0,3}(s|m|h))$")
_DELEGATE_KEEP_ALIVE_DEFAULT = "5m"

# Shared-client default is 30s; delegate calls pass explicit per-request
# timeouts (same mechanism as _AGENT_API_TIMEOUT_SECONDS).
_DELEGATE_TIMEOUT_DEFAULT_S = 300
_DELEGATE_TIMEOUT_MAX_S = 600
_DELEGATE_BG_CEILING_S = 1800.0

# jvmacmini runs num_parallel=1 — queuing beyond a few jobs would lie to
# the caller; fail fast instead.
_DELEGATE_JOB_CAP = 4
_DELEGATE_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _resolve_ollama_url() -> str:
    """Resolve the Ollama base URL: env var → Keychain → localhost.

    Blocking (Keychain lookup shells out to `security`) — async callers
    wrap in asyncio.to_thread. Raises ValueError for a configured URL
    that is not plain http(s) (fail closed rather than guess).
    """
    url = os.environ.get(_OLLAMA_URL_ENV_VAR, "").strip()
    if not url:
        try:
            url = get_api_key_from_keychain(
                _OLLAMA_URL_KEYCHAIN_SERVICE, getpass.getuser()
            )
        except ValueError:
            url = _OLLAMA_DEFAULT_URL
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid Ollama URL {redact_secrets(url)!r}: must be http(s)://host[:port]"
        )
    return url.rstrip("/")
```

- [ ] **Step 4: Run — verify PASS**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: 5 passed

- [ ] **Step 5: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat: local delegate constants + Ollama endpoint resolution"
```

---

### Task 2: `_post_ollama_chat` HTTP helper

**Files:**
- Modify: `mcp_server.py` (append to the Local-delegate section from Task 1)
- Modify: `test_local_delegate.py`

**Interfaces:**
- Consumes: `_resolve_ollama_url`, `_get_http_client`, `_http_error_payload`, `redact_secrets`
- Produces: `async def _post_ollama_chat(payload: dict[str, Any], timeout_s: float) -> dict[str, Any]` — returns Ollama's JSON on success, or `{"status": "failed", "error": str}` (never raises for network/HTTP/parse failures)

- [ ] **Step 1: Add failing tests**

The stub `httpx` module in the harness needs error types with the right subclass relationship. In `_build_stub_modules()` of `test_local_delegate.py`, replace the two fake error classes with:

```python
    class _FakeRequestError(Exception):
        pass

    class _FakeConnectError(_FakeRequestError):
        pass

    class _FakeHTTPStatusError(Exception):
        def __init__(self, message="", *, request=None, response=None):
            super().__init__(message)
            self.request = request
            self.response = response
```

and export them: `"httpx": _make("httpx", AsyncClient=_FakeAsyncClient, HTTPStatusError=_FakeHTTPStatusError, RequestError=_FakeRequestError, ConnectError=_FakeConnectError)`. Then add the test class:

```python
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
        patcher = mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_URL": "http://localhost:11434"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, client, payload=None, timeout_s=300.0):
        with _with_client(client):
            return asyncio.run(
                mcp_server._post_ollama_chat(payload or {"model": "m"}, timeout_s)
            )

    def test_happy_path_posts_to_api_chat_with_timeout(self):
        client = _FakeClient(response=_FakeResponse(json_data={"message": {"content": "hi"}}))
        out = self._post(client, payload={"model": "m", "stream": False}, timeout_s=42.0)
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
        client = _FakeClient(response=_FakeResponse(status_code=404, text="model not found"))
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
```

Note: `_http_error_payload` calls `exc.response.status_code` and `.text` — `_FakeResponse` provides both.

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: new tests ERROR with `AttributeError: ... '_post_ollama_chat'`

- [ ] **Step 3: Implement**

Append to the Local-delegate section in `mcp_server.py`:

```python
async def _post_ollama_chat(
    payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST to the local Ollama /api/chat endpoint.

    Same structured-error contract as _post_agent_research: network, HTTP,
    and parse failures return {"status": "failed", "error": ...} instead of
    raising. No auth header — the endpoint is localhost by default; a remote
    endpoint's auth story is transport-level (Tailscale ACLs), not app-level.
    No retries: a local server is either up or not.
    """
    try:
        base_url = await asyncio.to_thread(_resolve_ollama_url)
    except ValueError as exc:
        return {"status": "failed", "error": redact_secrets(str(exc))}
    client = await _get_http_client()
    try:
        response = await client.post(
            f"{base_url}/api/chat", json=payload, timeout=timeout_s
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        failure = _http_error_payload(exc)
        if exc.response.status_code == 404:
            model = payload.get("model", "")
            failure["error"] += (
                f" — model may not be pulled on this host; try: ollama pull {model}"
            )
        return failure
    except httpx.ConnectError:
        # Most likely real-world failure; make the message actionable.
        return {
            "status": "failed",
            "error": (
                f"Ollama not running at {base_url} — is the LaunchAgent up? "
                "(launchctl kickstart -k gui/$UID/com.jasonvassallo.ollama)"
            ),
        }
    except httpx.RequestError as exc:
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        # response.json() on a non-JSON 200 body.
        return {
            "status": "failed",
            "error": f"invalid JSON from Ollama: {redact_secrets(str(exc))}",
        }
```

(`httpx.ConnectError` must be caught **before** `httpx.RequestError` — it is a subclass.)

- [ ] **Step 4: Run — verify PASS**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: 12 passed

- [ ] **Step 5: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat: _post_ollama_chat helper with structured failure envelopes"
```

---

### Task 3: Answer rendering + in-memory background-job registry

**Files:**
- Modify: `mcp_server.py` (append to Local-delegate section)
- Modify: `test_local_delegate.py`

**Interfaces:**
- Consumes: `_post_ollama_chat(payload, timeout_s)`, constants from Task 1
- Produces (used by Task 4):
  - `def _render_delegate_answer(data: dict[str, Any]) -> list[TextContent]`
  - `def _start_delegate_job(payload: dict[str, Any]) -> str` — returns 32-hex job id; raises `ValueError` at cap; must be called with a running event loop
  - `def _collect_delegate_job(job_id: str | None) -> dict[str, Any]` — `{"status": "running", "elapsed_s": int}` while running; on completion returns the job's result dict **and deletes the entry** (single-collect); raises `ValueError` for malformed/unknown ids
  - module-level `_delegate_jobs: dict[str, dict[str, Any]]`

- [ ] **Step 1: Add failing tests**

```python
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
                await asyncio.sleep(0.05)  # let the wait_for-wrapped task finish
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
                gate.set()
                await asyncio.sleep(0.05)
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
```

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: ERRORs — `_render_delegate_answer` / `_delegate_jobs` missing

- [ ] **Step 3: Implement**

Ensure `import time` and `import uuid` are present in the stdlib import block (add if absent). Append:

```python
def _render_delegate_answer(data: dict[str, Any]) -> list[TextContent]:
    """Render an Ollama /api/chat response (or failure envelope) as MCP text.

    message.thinking is deliberately discarded — the caller needs the
    answer, not the model's scratchpad. Output passes through
    redact_secrets for the same never-emit-secret-shapes contract as
    every other family.
    """
    if data.get("status") == "failed":
        return [
            TextContent(type="text", text=f"Error: {data.get('error', 'unknown failure')}")
        ]
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return [TextContent(type="text", text="Error: Ollama returned no content")]
    model = redact_secrets(str(data.get("model", "")))
    return [
        TextContent(
            type="text",
            text=f"## Local Delegate ({model})\n\n{redact_secrets(content)}",
        )
    ]


# In-memory only, deliberately: delegated input may be exactly the
# sensitive text kept off cloud APIs — it does not belong on disk. Jobs
# die with the MCP server process; the polling Claude session dies with
# it too, so nothing durable is lost.
_delegate_jobs: dict[str, dict[str, Any]] = {}


def _start_delegate_job(payload: dict[str, Any]) -> str:
    """Launch a background delegate call; return its job id.

    Raises ValueError when _DELEGATE_JOB_CAP jobs are already running —
    the 32 GB host runs num_parallel=1, so queuing more would lie to the
    caller; failing fast is honest.
    """
    running = sum(1 for job in _delegate_jobs.values() if not job["task"].done())
    if running >= _DELEGATE_JOB_CAP:
        raise ValueError(
            f"Delegate job cap ({_DELEGATE_JOB_CAP}) reached — collect finished "
            "jobs via local_delegate_result or wait for one to complete."
        )
    job_id = uuid.uuid4().hex
    coro = asyncio.wait_for(
        _post_ollama_chat(payload, _DELEGATE_BG_CEILING_S),
        timeout=_DELEGATE_BG_CEILING_S,
    )
    _delegate_jobs[job_id] = {
        "task": asyncio.get_running_loop().create_task(coro),
        "started": time.monotonic(),
    }
    return job_id


def _collect_delegate_job(job_id: str | None) -> dict[str, Any]:
    """Poll/collect a background job. Completed jobs are single-collect:
    the registry entry is deleted on retrieval so memory stays clean."""
    if not isinstance(job_id, str) or not _DELEGATE_JOB_ID_RE.fullmatch(job_id):
        raise ValueError("job_id must be the 32-hex id returned by local_delegate.")
    job = _delegate_jobs.get(job_id)
    if job is None:
        raise ValueError(
            f"Unknown job_id {job_id!r} — results are single-collect and jobs "
            "do not survive an MCP server restart."
        )
    task = job["task"]
    if not task.done():
        return {
            "status": "running",
            "elapsed_s": int(time.monotonic() - job["started"]),
        }
    del _delegate_jobs[job_id]
    try:
        return task.result()
    except (TimeoutError, asyncio.CancelledError):
        # asyncio.wait_for raises TimeoutError (== asyncio.TimeoutError on
        # 3.12) past the ceiling; treat cancellation the same way.
        return {
            "status": "failed",
            "error": (
                f"background job exceeded the {int(_DELEGATE_BG_CEILING_S)}s "
                "ceiling and was cancelled"
            ),
        }
```

- [ ] **Step 4: Run — verify PASS**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: 23 passed

- [ ] **Step 5: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat: delegate answer rendering + in-memory background job registry"
```

---

### Task 4: Tool schemas + `call_tool` dispatch

**Files:**
- Modify: `mcp_server.py` — two places: append two `Tool(...)` entries in `list_tools()` (after the `delete_session` Tool, keeping family grouping: put them after the gemini tools / before `list_sessions` is also fine — choose **after `gemini_deep_research_result`**, before `list_sessions`, so research→delegate→sessions reads in family order); add two `if name == ...:` blocks in `call_tool()` (before the session handlers)
- Modify: `test_local_delegate.py`

**Interfaces:**
- Consumes: everything produced by Tasks 1–3 (exact names/signatures as defined there), plus Task 8's `_delegate_default_model() -> str`
- Produces: MCP tools `local_delegate`, `local_delegate_result` (the stable public surface)

- [ ] **Step 1: Add failing tests**

```python
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
        self.assertEqual(payload["messages"], [{"role": "user", "content": "do the thing"}])
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], False)
        self.assertNotIn("keep_alive", payload)  # v1.1: omitted → inherit server OLLAMA_KEEP_ALIVE
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
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "you are terse"})
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
                    await mcp_server.call_tool("local_delegate_result", {"job_id": job_id})

        asyncio.run(scenario())

    def test_result_unknown_id_is_clean_error(self):
        out = _call("local_delegate_result", {"job_id": "b" * 32})
        self.assertIn("Error", out[0].text)

    def test_result_missing_id_is_clean_error(self):
        out = _call("local_delegate_result", {})
        self.assertIn("Error", out[0].text)
```

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: FAILs — tools not listed, `Unknown tool` errors from `call_tool`

- [ ] **Step 3: Implement the two `Tool(...)` schema entries**

Insert after the `gemini_deep_research_result` Tool entry in `list_tools()`:

```python
        Tool(
            name="local_delegate",
            description=(
                "Delegate a task to the LOCAL Ollama qwen3.6 coding model — "
                "input text never leaves the machine (unlike every research "
                "tool here, which calls hosted APIs). Use for: private/"
                "sensitive text that must stay on-device; cheap mechanical "
                "work (summaries, boilerplate, drafts, bulk transforms) that "
                "doesn't need frontier quality; an independent second opinion "
                "on code or text; or long background jobs (pass "
                "background=true, poll local_delegate_result). No web access "
                "— for research use the research tools instead. The model is "
                "strong at code and structured transforms but far below "
                "frontier models on hard reasoning: keep tasks well-scoped."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The task. Include any needed file content inline — "
                            "the server never reads the filesystem."
                        ),
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt framing the task.",
                    },
                    "model": {
                        "type": "string",
                        "enum": list(OLLAMA_DELEGATE_MODELS),
                        "default": _delegate_default_model(),
                        "description": (
                            "Server-side allowlist. The default (base tag) "
                            "inherits each serving host's context window "
                            "(64k on JVMBPro, 32k on jvmacmini); -32k/-256k "
                            "pin explicit windows (-256k = several GB of KV "
                            "cache, JVMBPro only). The endpoint chain is "
                            "probed per call; the first endpoint serving "
                            "the tag wins."
                        ),
                    },
                    "think": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Enable the model's thinking mode. Off by default "
                            "for speed; enable for reasoning-heavy asks."
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "true: return a job_id immediately; poll "
                            "local_delegate_result. false: wait for the answer."
                        ),
                    },
                    "keep_alive": {
                        "type": "string",
                        "description": (
                            "Optional: how long Ollama keeps the model loaded "
                            "after the call ('0' = unload immediately — use "
                            "after a big -256k job). Omit to inherit the "
                            "server's OLLAMA_KEEP_ALIVE. Pattern: 0 or "
                            "<1-9999><s|m|h>."
                        ),
                    },
                    "timeout_s": {
                        "type": "integer",
                        "default": 300,
                        "description": "Sync timeout in seconds (1-600).",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="local_delegate_result",
            description=(
                "Poll/collect a background local_delegate job by job_id. "
                "Returns running status with elapsed seconds, or the answer. "
                "Results are single-collect: once retrieved the job is gone. "
                "Jobs live in server memory only and do not survive restarts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The 32-hex job id returned by local_delegate.",
                    },
                },
                "required": ["job_id"],
            },
        ),
```

- [ ] **Step 4: Implement the `call_tool` dispatch blocks**

Insert before the `list_sessions` handler in `call_tool()`:

```python
    if name == "local_delegate":
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return [
                TextContent(
                    type="text",
                    text="Error: prompt is required and must be a non-empty string.",
                )
            ]
        model = arguments.get("model", _delegate_default_model())
        if model not in OLLAMA_DELEGATE_MODELS:
            allowed = ", ".join(OLLAMA_DELEGATE_MODELS)
            return [
                TextContent(type="text", text=f"Error: model must be one of: {allowed}")
            ]
        think = arguments.get("think", False)
        if not isinstance(think, bool):
            return [TextContent(type="text", text="Error: think must be a JSON boolean.")]
        background = arguments.get("background", False)
        if not isinstance(background, bool):
            return [
                TextContent(type="text", text="Error: background must be a JSON boolean.")
            ]
        keep_alive = arguments.get("keep_alive")
        if keep_alive is not None and (
            not isinstance(keep_alive, str)
            or not _DELEGATE_KEEP_ALIVE_RE.fullmatch(keep_alive)
        ):
            return [
                TextContent(
                    type="text",
                    text="Error: keep_alive must match 0 or <1-9999><s|m|h> (e.g. '5m', '0').",
                )
            ]
        timeout_s = arguments.get("timeout_s", _DELEGATE_TIMEOUT_DEFAULT_S)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int)
            or not 1 <= timeout_s <= _DELEGATE_TIMEOUT_MAX_S
        ):
            return [
                TextContent(
                    type="text",
                    text=(
                        "Error: timeout_s must be an integer between 1 and "
                        f"{_DELEGATE_TIMEOUT_MAX_S}."
                    ),
                )
            ]
        system = arguments.get("system")
        if system is not None and not isinstance(system, str):
            return [TextContent(type="text", text="Error: system must be a string.")]

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "think": think,
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        if background:
            try:
                job_id = _start_delegate_job(payload)
            except ValueError as exc:
                return [TextContent(type="text", text=f"Error: {exc}")]
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"job_id": job_id, "status": "started"}),
                )
            ]

        data = await _post_ollama_chat(payload, float(timeout_s))
        return _render_delegate_answer(data)

    if name == "local_delegate_result":
        try:
            outcome = _collect_delegate_job(arguments.get("job_id"))
        except ValueError as exc:
            return [TextContent(type="text", text=f"Error: {exc}")]
        if outcome.get("status") == "running":
            return [TextContent(type="text", text=json.dumps(outcome))]
        return _render_delegate_answer(outcome)
```

Also delete the now-unused `_DELEGATE_KEEP_ALIVE_DEFAULT` constant from the Task-1 constants block (v1.1: no server-side keep_alive default).

- [ ] **Step 5: Run — verify PASS (full suite, all files)**

Run: `uv run --with pytest pytest test_local_delegate.py test_agent_research.py test_redact.py test_session_mgmt.py -q`
Expected: all pass (delegate file ~40 tests; zero regressions elsewhere)

- [ ] **Step 6: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat: local_delegate + local_delegate_result MCP tools"
```

---

### Task 5: Non-fatal Ollama line in `run_check()` + CI wiring

**Files:**
- Modify: `mcp_server.py:207-227` (`run_check`)
- Modify: `.github/workflows/tests.yml:72,75` (add `test_local_delegate.py` to both lists)
- Modify: `test_local_delegate.py`

**Interfaces:**
- Consumes: `_resolve_ollama_chain`, `_ollama_auth_headers`, `_OLLAMA_DEFAULT_MODEL_ENV_VAR` (Tasks 8/9)
- Produces: `--check` output gains one `ok:`/`warn:` line **per chain endpoint** (+ optional env-default-model warn); **never increments `errors`**

- [ ] **Step 1: Add failing test**

```python
class TestRunCheckOllamaLine(unittest.TestCase):
    def _run_check_output(self, get_side_effect=None, json_version="0.9.0"):
        fake_resp = mock.Mock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.json.return_value = {"version": json_version}
        fake_requests = types.SimpleNamespace(
            get=mock.Mock(return_value=fake_resp, side_effect=get_side_effect),
            RequestException=Exception,
        )

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
                        os.environ, {"AI_TOOLS_OLLAMA_URL": "http://localhost:11434"}
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
```

Add `import contextlib` and `import io` to the test file imports. Note `create=True` on the `requests` patch: the stub-import harness may not give `mcp_server` a real `requests` attribute.

`run_check` catches `requests.RequestException`; the fake sets `RequestException=Exception` so the generic side_effect is caught.

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q -k RunCheck`
Expected: FAIL — no ollama line in output

- [ ] **Step 3: Implement**

Check whether `import requests` exists at the top of `mcp_server.py` (it is in the PEP 723 deps for google-auth transport; the module may only import `google.auth.transport.requests`). If plain `requests` is not imported, add `import requests` to the third-party import block. Then in `run_check()`, immediately before `sys.exit(errors)`:

```python
    # Non-fatal: local_delegate family. Ollama being down must not fail
    # installs or preflights of the hosted tool families — delegate calls
    # themselves fail closed at call time.
    try:
        chain = _resolve_ollama_chain()
    except ValueError as e:
        print(
            "warn: ollama endpoint chain invalid (local_delegate unavailable): "
            f"{redact_secrets(str(e))}"
        )
        chain = []
    for endpoint in chain:
        try:
            headers = _ollama_auth_headers(endpoint)
            if headers is None:
                print(
                    "warn: ollama endpoint skipped (no Cloudflare Access creds "
                    f"in Keychain): {endpoint}"
                )
                continue
            resp = requests.get(f"{endpoint}/api/version", headers=headers, timeout=3)
            resp.raise_for_status()
            version = resp.json().get("version", "?")
            print(f"ok: ollama reachable at {endpoint} (version {version})")
        except (ValueError, requests.RequestException) as e:
            print(
                f"warn: ollama not reachable at {endpoint} "
                f"(local_delegate may fall back): {redact_secrets(str(e))}"
            )
    env_default = os.environ.get(_OLLAMA_DEFAULT_MODEL_ENV_VAR, "").strip()
    if env_default and env_default not in OLLAMA_DELEGATE_MODELS:
        print(
            f"warn: {_OLLAMA_DEFAULT_MODEL_ENV_VAR}={env_default!r} not in "
            f"allowlist; using {OLLAMA_DELEGATE_DEFAULT_MODEL}"
        )
```

- [ ] **Step 4: Update `.github/workflows/tests.yml`**

Line 72: append ` test_local_delegate.py` to the pytest file list.
Line 75: append ` test_local_delegate.py` to the ruff format --check file list.

- [ ] **Step 5: Run — verify PASS + live smoke**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: all pass

Live smoke (Ollama runs on this machine): `uv run mcp_server.py --check`
Expected output includes `ok: ollama reachable at http://localhost:11434 (version ...)` and exit code unchanged by the ollama line.

- [ ] **Step 6: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py .github/workflows/tests.yml
git commit -m "feat: non-fatal ollama reachability line in --check; CI runs delegate tests"
```

---

### Task 6: Docs & packaging

**Files:**
- Modify: `README.md` (charter, tool counts, stable surface, provider mapping)
- Modify: `mcp_server.py:12-35` (module docstring family list)
- Modify: `mcpb/manifest.json` (tool declarations + `description`/`long_description` counts)
- Create: `commands/local-delegate.md`
- Modify: `skills/using-ai-research/SKILL.md` (routing: when to stay local)

**Interfaces:**
- Consumes: final tool names/params from Task 4 (documentation must match the schemas exactly)
- Produces: user-facing docs; no code

- [ ] **Step 1: README.md**

Make these edits (prose may be adapted, facts must be exact):

1. Replace the charter bullets:
   - `- It is for hosted API-backed MCP tooling.` → `- It exposes hosted AI providers and the machine's local Ollama server behind one MCP surface.`
   - `- It is not a local-model repo.` → `- No model weights live in this repo — the local family only calls an already-running Ollama.`
   - `- It currently exposes eleven tools across two families:` → `- It currently exposes thirteen tools across three families:` and add to the family list: `  - Local delegate: \`local_delegate\` / \`local_delegate_result\` (Ollama, on-device)`
2. Stable Public Surface: add a `Tool names (local delegate):` block listing both names.
3. Provider Mapping: add a `### \`local_delegate\` / \`local_delegate_result\`` section documenting: provider (**local-first Ollama endpoint chain** — default `http://localhost:11434` → `https://ollama-mbp.djvassallo.com` (Cloudflare-Access-gated); override via `AI_TOOLS_OLLAMA_URLS` comma-separated env (singular `AI_TOOLS_OLLAMA_URL` honored for compat), Keychain `OLLAMA_URL` appended; per-call `/api/tags` probe picks the first endpoint serving the tag, cached 60 s; remote endpoints require https + CF Access service-token creds in Keychain, else skipped); model allowlist (three qwen3.6 tags; default base tag inherits each host's window — 64k JVMBPro / 32k jvmacmini; env `AI_TOOLS_OLLAMA_DEFAULT_MODEL` may pick a different allowlisted tag); purpose (privacy / quota offload / second opinion / background jobs); latency (seconds-to-minutes, background via `job_id` + poll); privacy note (**input stays on your machines** — on-device when localhost serves the model, otherwise only your own Access-gated endpoint, never a third-party API; nothing written to disk; jobs in-memory and single-collect); `think` semantics and `keep_alive` (omit to inherit the server's `OLLAMA_KEEP_ALIVE`).
4. In the "How It Works" bullet list, add: `- calls the local Ollama server (native /api/chat) for the local_delegate family`.
5. Update the closing "Together these complement..." paragraph: add "use `local_delegate` when the input must stay on-device or the task is cheap mechanical work".

- [ ] **Step 2: Module docstring**

In the `mcp_server.py` docstring, change "four families of tools" to "five families of tools" and insert after the gemini bullet:

```
- ``local_delegate`` / ``local_delegate_result``: local-first Ollama
  delegation — send a task to the qwen3.6 coding model (native
  /api/chat, think off by default) via an ordered endpoint chain:
  localhost first, then the user's own Cloudflare-Access-gated
  remote. Input text never leaves the user's machines; background
  jobs are in-memory and single-collect.
```

- [ ] **Step 3: mcpb/manifest.json**

- In `description`: "…— exposed as 11 MCP tools." → "…, local Ollama delegation, and local conversation-session persistence — exposed as 13 MCP tools."
- In `long_description`: "Four families of tools:" → "Five families of tools:" and insert as family (4): `(4) local_delegate sends a task to the machine's local Ollama qwen3.6 model — on-device, input never leaves the machine — synchronously or in the background via local_delegate_result polling.` (renumber sessions to (5)).
- In the `tools` array, add two entries following the existing entry shape (name + description matching the Tool schema descriptions from Task 4, abbreviated to one sentence each).
- **[v1.1]** Add `user_config` entries mapped to env (follow the manifest's existing `user_config`/`env` conventions — read the file first; MCPB v0.3 `${user_config.KEY}` interpolation in the server's `mcp_config.env` block):
  - `ollama_endpoints` — type string, title "Ollama endpoint chain", description "Comma-separated ordered list of Ollama endpoints, local first", default `http://localhost:11434,https://ollama-mbp.djvassallo.com`, required false → env `AI_TOOLS_OLLAMA_URLS`
  - `default_model` — type string, title "Default delegate model", description "Must be one of the allowlisted qwen3.6 tags", default `qwen3.6:35b-a3b-coding-nvfp4`, required false → env `AI_TOOLS_OLLAMA_DEFAULT_MODEL`
- Validate: `python3 -c "import json; json.load(open('mcpb/manifest.json'))"`.

- [ ] **Step 4: commands/local-delegate.md**

Follow the frontmatter/body shape of `commands/agent-research.md` (read it first). Content:

```markdown
---
description: Delegate a task to the local Ollama qwen3.6 model (on-device, private)
argument-hint: <task, e.g. "summarize this diff: ...">
---

Use the `local_delegate` MCP tool to run this task on the local Ollama model:

$ARGUMENTS

Guidance:
- Include any needed file content inline in the prompt — the server never reads files.
- Default model is the base qwen3.6 coding tag; pass model=...-256k only for genuinely huge inputs, and keep_alive="0" to unload afterwards.
- Pass think=true only for reasoning-heavy asks (slower).
- For long jobs pass background=true, then poll with the `local_delegate_result` tool.
- Output quality is below frontier models — treat results as a draft to verify, not a final answer.
```

- [ ] **Step 5: skills/using-ai-research routing update**

Read `skills/using-ai-research/SKILL.md` first, then add a routing rule to its decision guidance: *keep it local with `local_delegate` when the input is sensitive/private (must not reach any hosted API), when the task is cheap mechanical transformation of text you already have, or when you want an independent local second opinion; use the hosted research tools when the task needs the web.* Match the file's existing format (table or bullets).

- [ ] **Step 6: Verify + commit**

```bash
python3 -c "import json; json.load(open('mcpb/manifest.json'))"
uv run --with pytest pytest test_local_delegate.py test_agent_research.py test_redact.py test_session_mgmt.py -q
git add README.md mcp_server.py mcpb/manifest.json commands/local-delegate.md skills/using-ai-research/
git commit -m "docs: local delegate family — README charter, manifest, command, routing skill"
```

---

### Task 7: Live smoke test, local gate, PR

**Files:**
- No new files; verification + PR only

- [ ] **Step 1: Live end-to-end smoke (Ollama is running on this machine)**

```bash
cd /Users/jasonvassallo/Documents/Code/ai-tools-mcp
# NB: a bare `uv run python3 -` does NOT resolve mcp_server.py's PEP 723
# inline deps (that only happens when the script file is the run target),
# so supply them explicitly:
uv run --with 'openai>=1.0.0' --with 'mcp>=1.0.0' --with 'httpx>=0.27' \
       --with 'google-auth>=2.30' --with 'requests>=2.31' python3 - <<'EOF'
import asyncio, importlib.util
spec = importlib.util.spec_from_file_location("m", "mcp_server.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = asyncio.run(m.call_tool("local_delegate", {"prompt": "Reply with exactly: DELEGATE-OK"}))
print(out[0].text)
EOF
```

Expected: `## Local Delegate (qwen3.6:35b-a3b-coding-nvfp4)` header and `DELEGATE-OK` in the body, in well under 60 s. (This exercises the real Ollama path once — the unit suite never does.)

- [ ] **Step 2: Full local gate**

```bash
uv tool run ruff format --check mcp_server.py test_redact.py test_session_mgmt.py test_agent_research.py test_local_delegate.py
uv run --with pytest pytest test_local_delegate.py test_agent_research.py test_redact.py test_session_mgmt.py -q
uv run mcp_server.py --check
```

Expected: format clean; all tests pass; `--check` shows the ollama `ok:` line.

- [ ] **Step 3: Pre-PR reviews per the repo's standing pipeline**

Quick-scan the diff for credentials first (`git diff main...HEAD | grep -iE 'key|token|secret|password'` — expect only variable names). Then `semgrep scan --config=auto` and `coderabbit review --base main --plain` per the user's Stage-1 pipeline. Address findings.

- [ ] **Step 4: Push + PR**

```bash
git push -u origin feat/local-delegate
gh pr create --title "feat: local_delegate — on-device Ollama delegation tool family" \
  --body "$(cat <<'EOF'
Adds the local-delegate tool family per the approved spec
(docs/superpowers/specs/2026-07-06-local-delegate-design.md):

- `local_delegate`: send a task to the local Ollama qwen3.6 coding model
  (native /api/chat, think:false default, server-side model allowlist,
  sync or background)
- `local_delegate_result`: poll/collect background jobs (in-memory,
  single-collect, cap 4, 30-min ceiling)
- Endpoints (v1.1): local-first ordered chain (default localhost →
  https://ollama-mbp.djvassallo.com behind Cloudflare Access), per-call
  /api/tags probe + 60s cache, CF Access service-token headers from
  Keychain for remote https (creds absent → endpoint skipped), non-https
  remote rejected
- Non-fatal ollama reachability line in `--check`
- README charter updated: hosted **and** local behind one MCP surface

Privacy: input text never leaves the machine and is never written to disk.
Tests: test_local_delegate.py (~45 unit tests, all network mocked) + live
smoke against local Ollama.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then run the remaining Stage-2 gate (pr-agent local via its wrapper if configured for this repo) and wait for all GitHub bot reviewers before merge, per the standing review pipeline.

---

### Task 8: [v1.1] Endpoint chain, per-model probe + cache, default-model env (reworks Task 1/2 code)

**Files:**
- Modify: `mcp_server.py` — in the Local-delegate constants/helpers section: replace `_resolve_ollama_url` (and `_OLLAMA_DEFAULT_URL`) with the chain machinery below; rework `_post_ollama_chat` to select per call
- Modify: `test_local_delegate.py` — replace `TestResolveOllamaUrl` with `TestResolveOllamaChain`; add `TestDelegateDefaultModel`, `TestSelectOllamaEndpoint`; rework `TestPostOllamaChat` (see Step 1)

**Interfaces:**
- Consumes: Task 1 constants (`_OLLAMA_URL_ENV_VAR`, `_OLLAMA_URL_KEYCHAIN_SERVICE`, allowlist), `get_api_key_from_keychain`, `_get_http_client`, `_http_error_payload`, `redact_secrets`
- Produces (used by Tasks 4/5/9):
  - `_OLLAMA_URLS_ENV_VAR = "AI_TOOLS_OLLAMA_URLS"`, `_OLLAMA_DEFAULT_MODEL_ENV_VAR = "AI_TOOLS_OLLAMA_DEFAULT_MODEL"`
  - `_OLLAMA_DEFAULT_CHAIN: tuple[str, ...] = ("http://localhost:11434", "https://ollama-mbp.djvassallo.com")`
  - `_OLLAMA_PROBE_TIMEOUT_S = 2.0`, `_OLLAMA_PROBE_CACHE_TTL_S = 60.0`
  - `def _is_localhost_endpoint(url: str) -> bool`
  - `def _validate_ollama_endpoint(url: str) -> str` — raises `ValueError` (bad scheme/netloc, or plain-http non-localhost)
  - `def _resolve_ollama_chain() -> list[str]` — blocking (Keychain); validated, deduped, ordered
  - `def _delegate_default_model() -> str` — env override honored only if allowlisted, else base tag
  - `def _ollama_auth_headers(endpoint: str) -> dict[str, str] | None` — **stub in this task**: `{}` for localhost, `None` for remote (None = SKIP, never call bare; Task 9 replaces with real CF Access lookup)
  - `async def _select_ollama_endpoint(model: str) -> str` — raises `ValueError` naming every endpoint tried when none serves the tag
  - `_ollama_endpoint_cache: dict[str, tuple[str, float]]` (model → (endpoint, expires))
  - `_post_ollama_chat(payload, timeout_s)` — same signature, now selects endpoint internally and sends auth headers
  - **Removes:** `_resolve_ollama_url`, `_OLLAMA_DEFAULT_URL`

- [ ] **Step 1: Rework/replace tests**

Delete class `TestResolveOllamaUrl`. Rework `TestPostOllamaChat`: delete its `setUp` env patch; instead each test patches endpoint selection directly — add to the class:

```python
    def _with_selection(self, endpoint="http://localhost:11434"):
        return mock.patch.object(
            mcp_server,
            "_select_ollama_endpoint",
            mock.AsyncMock(return_value=endpoint),
        )
```

wrap each `self._post(...)` call site in `with self._with_selection():` (the redaction test passes its JWT-bearing URL as `endpoint=` instead of via env); replace `test_bad_configured_url_is_failure_envelope` with:

```python
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
```

update the happy-path test to also assert `client.calls[0][1]["headers"] == {}`, and add:

```python
    def test_connect_error_drops_cache_entry(self):
        mcp_server._ollama_endpoint_cache["m"] = ("http://localhost:11434", 10**12)
        client = _FakeClient(exc=mcp_server.httpx.ConnectError("refused"))
        with self._with_selection():
            self._post(client, payload={"model": "m"})
        self.assertNotIn("m", mcp_server._ollama_endpoint_cache)
```

Then add the new test classes:

```python
_MODEL = "qwen3.6:35b-a3b-coding-nvfp4"


def _no_keychain(service, account):
    raise ValueError("not found")


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

    def test_garbage_url_rejected(self):
        with self.assertRaises(ValueError):
            self._chain({"AI_TOOLS_OLLAMA_URLS": "http://"})


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
        with mock.patch.dict(os.environ, {"AI_TOOLS_OLLAMA_DEFAULT_MODEL": "llama3:8b"}):
            self.assertEqual(
                mcp_server._delegate_default_model(),
                mcp_server.OLLAMA_DELEGATE_DEFAULT_MODEL,
            )


class _FakeTagsClient:
    """Programmable fake for _select_ollama_endpoint probes."""

    def __init__(self, tags_by_url=None, exc_by_url=None):
        self.tags_by_url = tags_by_url or {}
        self.exc_by_url = exc_by_url or {}
        self.get_calls: list = []

    async def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        base = url.removesuffix("/api/tags")
        if base in self.exc_by_url:
            raise self.exc_by_url[base]
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
        client = _FakeTagsClient(tags_by_url={self.EP1: ["other:1b"], self.EP2: [_MODEL]})
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
```

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: ERRORs — `_resolve_ollama_chain`, `_select_ollama_endpoint`, `_delegate_default_model` missing; reworked TestPostOllamaChat failing on `_select_ollama_endpoint` patch target

- [ ] **Step 3: Implement**

In the Local-delegate section of `mcp_server.py`: delete `_OLLAMA_DEFAULT_URL` and `_resolve_ollama_url`; add after the existing constants:

```python
_OLLAMA_URLS_ENV_VAR = "AI_TOOLS_OLLAMA_URLS"
_OLLAMA_DEFAULT_MODEL_ENV_VAR = "AI_TOOLS_OLLAMA_DEFAULT_MODEL"

# v1.1 (spec amendment): local-first endpoint chain. The remote default is
# the user's own Cloudflare-Access-gated JVMBPro tunnel — never a
# third-party service.
_OLLAMA_DEFAULT_CHAIN: tuple[str, ...] = (
    "http://localhost:11434",
    "https://ollama-mbp.djvassallo.com",
)
_OLLAMA_PROBE_TIMEOUT_S = 2.0
_OLLAMA_PROBE_CACHE_TTL_S = 60.0
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_localhost_endpoint(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in _LOCALHOST_HOSTS


def _validate_ollama_endpoint(url: str) -> str:
    """Validate one chain entry; fail closed on anything not plain http(s).

    Loopback may be http; every other host must be https (v1.1 rule — a
    remote endpoint is only ever the Access-gated tunnel).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid Ollama endpoint {redact_secrets(url)!r}: must be "
            "http(s)://host[:port]"
        )
    if parsed.scheme == "http" and not _is_localhost_endpoint(url):
        raise ValueError(
            f"Refusing plain-http non-localhost Ollama endpoint "
            f"{redact_secrets(url)!r} — remote endpoints must be https"
        )
    return url.rstrip("/")


def _resolve_ollama_chain() -> list[str]:
    """Ordered Ollama endpoint chain (v1.1).

    Env `AI_TOOLS_OLLAMA_URLS` (comma-separated) wins; singular
    `AI_TOOLS_OLLAMA_URL` is honored as a one-item chain for v1 compat;
    otherwise the default local-first chain. A Keychain `OLLAMA_URL`
    endpoint is appended when present. Every entry is validated; dupes
    dropped preserving order. Blocking (Keychain) — async callers wrap
    in asyncio.to_thread.
    """
    raw = os.environ.get(_OLLAMA_URLS_ENV_VAR, "").strip()
    if raw:
        entries = [e.strip() for e in raw.split(",") if e.strip()]
    else:
        single = os.environ.get(_OLLAMA_URL_ENV_VAR, "").strip()
        entries = [single] if single else list(_OLLAMA_DEFAULT_CHAIN)
    try:
        keychain_url = get_api_key_from_keychain(
            _OLLAMA_URL_KEYCHAIN_SERVICE, getpass.getuser()
        ).strip()
        if keychain_url:
            entries.append(keychain_url)
    except ValueError:
        pass  # optional config — absence is the common case
    chain: list[str] = []
    for entry in entries:
        validated = _validate_ollama_endpoint(entry)
        if validated not in chain:
            chain.append(validated)
    return chain


def _delegate_default_model() -> str:
    """Default model tag; env override honored only if allowlisted.

    Falls back silently to the base tag (run_check surfaces a warn) so a
    typo'd Desktop setting cannot break tool listing.
    """
    env_model = os.environ.get(_OLLAMA_DEFAULT_MODEL_ENV_VAR, "").strip()
    if env_model in OLLAMA_DELEGATE_MODELS:
        return env_model
    return OLLAMA_DELEGATE_DEFAULT_MODEL


def _ollama_auth_headers(endpoint: str) -> dict[str, str] | None:
    """Auth headers for an endpoint; None means SKIP (never call bare).

    Stub for this task: localhost needs no auth ({}); remote endpoints
    report None until the Cloudflare Access Keychain wiring lands in the
    next task.
    """
    if _is_localhost_endpoint(endpoint):
        return {}
    return None


_ollama_endpoint_cache: dict[str, tuple[str, float]] = {}


async def _select_ollama_endpoint(model: str) -> str:
    """First endpoint in the chain whose /api/tags lists `model`.

    Results are cached per model for _OLLAMA_PROBE_CACHE_TTL_S. Raises
    ValueError naming every endpoint tried and what each reported —
    actionable and fail-closed.
    """
    cached = _ollama_endpoint_cache.get(model)
    if cached is not None and time.monotonic() < cached[1]:
        return cached[0]
    chain = await asyncio.to_thread(_resolve_ollama_chain)
    client = await _get_http_client()
    attempts: list[str] = []
    for endpoint in chain:
        headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
        if headers is None:
            attempts.append(
                f"{endpoint}: skipped (Cloudflare Access credentials not in Keychain)"
            )
            continue
        try:
            response = await client.get(
                f"{endpoint}/api/tags",
                headers=headers,
                timeout=_OLLAMA_PROBE_TIMEOUT_S,
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            tags = [
                str(m.get("name") or m.get("model") or "")
                for m in models
                if isinstance(m, dict)
            ]
        except httpx.HTTPStatusError as exc:
            attempts.append(f"{endpoint}: HTTP {exc.response.status_code}")
            continue
        except httpx.RequestError:
            attempts.append(f"{endpoint}: unreachable")
            continue
        except ValueError:
            attempts.append(f"{endpoint}: invalid JSON from /api/tags")
            continue
        if model in tags:
            _ollama_endpoint_cache[model] = (
                endpoint,
                time.monotonic() + _OLLAMA_PROBE_CACHE_TTL_S,
            )
            return endpoint
        present = ", ".join(sorted(t for t in tags if t)) or "no models"
        attempts.append(f"{endpoint}: model not present (has {present})")
    detail = "; ".join(attempts) or "empty endpoint chain"
    raise ValueError(
        f"No Ollama endpoint serves {model!r}: {redact_secrets(detail)}"
    )
```

Then rework `_post_ollama_chat` to (docstring updated accordingly):

```python
async def _post_ollama_chat(
    payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST /api/chat to the first chain endpoint serving payload['model'].

    Same structured-error contract as _post_agent_research: selection,
    network, HTTP, and parse failures return {"status": "failed", ...}
    instead of raising. Remote https endpoints get Cloudflare Access
    service-token headers (absent creds → skipped at selection; the None
    check here is defense in depth). No retries: an endpoint is either
    serving or not.
    """
    model = str(payload.get("model", ""))
    try:
        endpoint = await _select_ollama_endpoint(model)
    except ValueError as exc:
        return {"status": "failed", "error": redact_secrets(str(exc))}
    headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
    if headers is None:
        return {
            "status": "failed",
            "error": f"no credentials for {redact_secrets(endpoint)}",
        }
    client = await _get_http_client()
    try:
        response = await client.post(
            f"{endpoint}/api/chat", json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        failure = _http_error_payload(exc)
        if exc.response.status_code == 404:
            failure["error"] += (
                f" — model may not be pulled on this host; try: ollama pull {model}"
            )
        return failure
    except httpx.ConnectError:
        # Answered the probe moments ago but refused the POST — drop the
        # cached resolution so the next call re-probes the chain.
        _ollama_endpoint_cache.pop(model, None)
        return {
            "status": "failed",
            "error": (
                f"Ollama not running at {redact_secrets(endpoint)} — is the "
                "LaunchAgent up? "
                "(launchctl kickstart -k gui/$UID/com.jasonvassallo.ollama)"
            ),
        }
    except httpx.RequestError as exc:
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        return {
            "status": "failed",
            "error": f"invalid JSON from Ollama: {redact_secrets(str(exc))}",
        }
```

- [ ] **Step 4: Run — verify PASS**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: all pass (~31 tests). Then full suite: `uv run --with pytest pytest test_local_delegate.py test_agent_research.py test_redact.py test_session_mgmt.py -q` — zero regressions.

- [ ] **Step 5: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat(v1.1): local-first Ollama endpoint chain with per-model probe + cache"
```

---

### Task 9: [v1.1] Cloudflare Access service-token auth for remote endpoints

**Files:**
- Modify: `mcp_server.py` — add the two Keychain service constants; replace the Task-8 stub `_ollama_auth_headers` with the real implementation
- Modify: `test_local_delegate.py` — add `TestOllamaAuthHeaders`; add POST-header and probe-header tests

**Interfaces:**
- Consumes: `_is_localhost_endpoint`, `get_api_key_from_keychain`, Task 8's selection/POST paths (which already pass `headers` through)
- Produces:
  - `_CF_ACCESS_ID_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_ID"`
  - `_CF_ACCESS_SECRET_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_SECRET"`
  - `_ollama_auth_headers(endpoint) -> dict[str, str] | None` (final): `{}` localhost; CF Access header pair for remote https; `None` when either cred absent

- [ ] **Step 1: Add failing tests**

```python
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

    def test_probe_sends_cf_headers_to_remote(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLS_OLLAMA_URLS": "https://remote.example", "AI_TOOLS_OLLAMA_URL": ""},
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
```

- [ ] **Step 2: Run — verify FAIL**

Run: `uv run --with pytest pytest test_local_delegate.py -q -k OllamaAuthHeaders`
Expected: `test_remote_with_creds_gets_cf_access_headers` and the probe/POST header tests FAIL (stub returns None for remote)

- [ ] **Step 3: Implement**

Add constants next to `_OLLAMA_URL_KEYCHAIN_SERVICE`:

```python
_CF_ACCESS_ID_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_ID"
_CF_ACCESS_SECRET_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_SECRET"
```

Replace the stub `_ollama_auth_headers` with:

```python
def _ollama_auth_headers(endpoint: str) -> dict[str, str] | None:
    """Auth headers for an Ollama endpoint; None means SKIP, never call bare.

    localhost → {} (no auth). Non-localhost https → Cloudflare Access
    service-token headers read from the Keychain per call (never cached,
    never logged). Either credential absent → None (fail closed): the
    caller treats the endpoint as unavailable rather than calling an
    Access-gated host unauthenticated.
    """
    if _is_localhost_endpoint(endpoint):
        return {}
    user = getpass.getuser()
    try:
        client_id = get_api_key_from_keychain(_CF_ACCESS_ID_KEYCHAIN_SERVICE, user)
        client_secret = get_api_key_from_keychain(
            _CF_ACCESS_SECRET_KEYCHAIN_SERVICE, user
        )
    except ValueError:
        return None
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": client_secret,
    }
```

- [ ] **Step 4: Run — verify PASS**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: all pass (Task 8's `test_remote_without_creds_is_skipped_with_reason` still passes — its keychain mock raises for every service, so creds resolve to None). Then the full 4-file suite — zero regressions.

- [ ] **Step 5: Format + commit**

```bash
uv tool run ruff format mcp_server.py test_local_delegate.py
git add mcp_server.py test_local_delegate.py
git commit -m "feat(v1.1): Cloudflare Access service-token auth for remote Ollama endpoints"
```
