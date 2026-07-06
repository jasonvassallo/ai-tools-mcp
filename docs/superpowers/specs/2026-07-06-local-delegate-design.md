# local_delegate — design

**Date:** 2026-07-06
**Status:** approved (brainstorming session)
**Repo:** ai-tools-mcp

## Purpose

Add a third tool family to `mcp_server.py`: delegation of tasks to the local
Ollama `qwen3.6:35b-a3b-coding-nvfp4` model. Four use cases, all served by one
generic surface:

1. **Privacy** — input text that must never leave the machine (the existing
   research tools all send text to hosted APIs; this tool exists so text can
   stay on-device).
2. **Quota/cost offload** — cheap mechanical work (summaries, boilerplate,
   drafts, bulk transforms) on free local compute.
3. **Second opinion** — independent local review of code or text.
4. **Background/batch worker** — long jobs (including big-context via the
   `-256k` tag) that run while the session continues.

This deliberately changes the repo charter: the README's "not a local-model
repo" line is replaced with "hosted **and local** AI behind one MCP surface;
no model weights in-repo". No weights, no model management — the tool only
*calls* an already-running Ollama server.

## Environment (context, not managed by this repo)

- JVMBPro (M5 Pro, 64 GB): Ollama LaunchAgent, tags `qwen3.6:35b-a3b-coding-nvfp4`
  (base), `-32k`, `-256k` — all sharing weights (~21 GB), q8_0 KV, flash-attn.
- jvmacmini (M2 Pro, 32 GB): same base model, 32k context, loopback-only bind,
  `num_parallel=1`.
- qwen3.6 is a *thinking* model; `think:false` is required for fast structured
  work. Only Ollama's **native** `/api/chat` accepts `think` — the
  OpenAI-compat endpoint does not (this is why the pr-agent wrapper needs a
  proxy; this tool avoids that entire problem class by speaking native API).

## Tool surface

Two tools, mirroring the `agent_research` / `agent_research_result` precedent.

### `local_delegate`

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string, **required** | — | The task. The caller (Claude) inlines any file content; the server never reads the filesystem. |
| `system` | string | none | Optional system prompt. |
| `model` | string | `qwen3.6:35b-a3b-coding-nvfp4` | **Server-side allowlist**: base, `-32k`, `-256k` tags only. Anything else → error listing allowed values. |
| `think` | bool | `false` | Passed natively to `/api/chat`. |
| `background` | bool | `false` | `false`: block until the answer returns. `true`: return a `job_id` immediately. |
| `keep_alive` | string | `"5m"` | Passed through to Ollama; `"0"` unloads immediately after the call (useful after a big `-256k` job on the 32 GB box). Validated against a strict pattern (`0` or `<int><s|m|h>`). |
| `timeout_s` | int | 300 | Sync read timeout; capped at 600. |

Returns: the model's answer as plain `TextContent` (sync), or
`{job_id, status: "started"}` (background).

When `think=true`, the response's `message.thinking` field is **discarded** —
only `message.content` is returned. The caller doesn't need the scratchpad.

### `local_delegate_result`

| Param | Type | Notes |
|---|---|---|
| `job_id` | string, **required** | uuid4 hex; validated by pattern before lookup. |

Returns one of:
- `running` + elapsed seconds
- the completed answer → registry entry **deleted after retrieval** (single-collect)
- the error payload → entry deleted likewise
- unknown id → clean error

## Endpoint resolution (fail-closed order)

1. `AI_TOOLS_OLLAMA_URL` environment variable, if set
2. macOS Keychain entry, service `OLLAMA_URL` (optional; config not credential,
   but Keychain keeps it out of dotfiles)
3. Default `http://localhost:11434`

Constraints: URL must parse as `http` or `https`; anything else is rejected.
Non-localhost endpoints are permitted (that is the "configurable endpoint"
decision) — remote auth is transport-level (e.g. Tailscale ACLs), not
app-level. No auth headers in v1.

## Background job model

- In-memory only: `dict[str, Job]` + `asyncio.create_task`. **Nothing is
  written to disk** — deliberate, because delegated input may be exactly the
  sensitive text kept off cloud APIs. Jobs die with the MCP server process;
  acceptable because the polling Claude session dies with it too.
- Job id: `uuid.uuid4().hex`.
- **Concurrency cap: 4** running jobs; a fifth `background=true` call errors
  immediately (the 32 GB mini runs `num_parallel=1` — queuing would lie to the
  caller; failing fast is honest).
- Hard ceiling: 30 min per background job, after which the job is marked
  `error: timeout` (the underlying HTTP call is cancelled).

## Data flow

**Sync:** `call_tool("local_delegate")` → validate params → resolve endpoint →
`POST {endpoint}/api/chat` with
`{model, messages: [system?, user], think, stream: false, keep_alive}`
(no `options` block — context length is owned by the model tags, not the caller)
→ extract `message.content` → `TextContent`.

**Background:** same request inside a task; registry `running → done|error`;
collected via `local_delegate_result` as above.

Uses the existing shared async httpx client (`_get_http_client()`); no new
dependencies. The shared client's default timeout is 30 s, so delegate calls
pass an **explicit per-request timeout** (`timeout_s`, or the 30-min ceiling
for background jobs) — same mechanism the agent-research POST already uses.

## Error handling (fail-closed, matching existing patterns)

- Connection refused/unreachable → actionable message: "Ollama not running at
  {endpoint} — is the LaunchAgent up?"
- Non-200 → structured error payload (status + body excerpt), run through the
  existing redaction.
- Model not in allowlist → error listing the three allowed tags.
- Allowlisted model not pulled on the host → surface Ollama's 404 with an
  `ollama pull <tag>` hint.
- Job cap exceeded, unknown job id, timeout → explicit errors; never silent.
- **No retries in v1** — a local server is either up or not; retrying a 21 GB
  model load multiplies pain.

## Security posture

- Input text goes only to the resolved Ollama endpoint (localhost by default).
- No disk writes for prompts, results, or job state.
- No new secrets; `OLLAMA_URL` Keychain entry is optional config.
- Redaction (`redact_secrets`) applied to anything logged or embedded in error
  payloads.
- Least privilege: the server does not read files, list models, pull models,
  or manage Ollama in any way.

## Testing

`test_local_delegate.py` at repo root (pattern of the existing `test_*.py`
files); all network mocked, no live Ollama in CI:

- allowlist: valid tags, invalid tag, default
- endpoint resolution precedence: env > keychain > default (mocked)
- request body: `think` propagation, system message included/omitted,
  `keep_alive` pass-through and pattern validation, `timeout_s` cap
- sync: happy path, connection refused, non-200
- background lifecycle: start → running → collect → entry gone
- job cap: 5th job rejected
- `local_delegate_result`: unknown id, malformed id
- timeout paths (sync and background)

## Docs & packaging updates (same PR)

- README: charter reframed (hosted **and** local), tool count 11 → 13,
  provider-mapping section for `local_delegate` (+ guidance vs. the research
  tools), stable-surface list gains both tool names.
- `commands/local-delegate.md` slash command.
- `skills/using-ai-research/`: routing updated — when to keep work local
  (privacy/cheap/second-opinion/batch) vs. Perplexity/Gemini (needs web).
- `mcpb/manifest.json`: declare the two new tools **and** update the
  `description`/`long_description` prose ("11 MCP tools", "Four families")
  to match the new counts.
- `run_check()` (`--check`): add a **non-fatal** Ollama reachability line
  (`ok:`/`warn:`, never counted in `errors`) — Ollama being down must not
  fail installs or preflights of the hosted tool families. Surfaced at
  SessionStart automatically via the existing `hooks/preflight.sh`.

## Out of scope (v1)

Fleet routing between machines, streaming responses, app-level auth headers,
disk-persisted jobs, free-form model strings, server-side file reading,
returning thinking traces, model management (pull/stop/ps).
