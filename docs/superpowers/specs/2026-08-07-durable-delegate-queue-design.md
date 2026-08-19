# Durable delegate queue — design

**Date:** 2026-08-07
**Status:** approved (user-approved 2026-08-07)
**Repo:** ai-tools-mcp
**Ships in:** v1.6.0

## Purpose

`local_delegate(background=true)` jobs currently live only in the MCP
server's memory: they die with the process (each Claude session spawns
and kills its own MCP server), and results are single-collect. That
makes long background jobs fragile in exactly the situations they exist
for. This design adds a **durable submit/poll queue** — a standalone
service on JVMBPro in front of its Ollama — that background delegate
jobs prefer, with the in-memory path retained as an explicit fallback.

## Decision record: jobs on disk, reversed with mitigation

The repo's historical rule (2026-07-06 local_delegate design) was
deliberate: *"In-memory only … delegated input may be exactly the
sensitive text kept off cloud APIs — it does not belong on disk."*

**That rule is intentionally reversed for this service, with user
approval given 2026-08-07.** The mitigation is encryption at rest:

- Job payloads and results are stored as **AES-256-GCM** blobs (fresh
  random 96-bit nonce per encryption, nonce stored alongside the
  ciphertext). Only operational metadata is plaintext: job id, status,
  timestamps, model name, attempt count, error class.
- The 32-byte key lives in the macOS **System** keychain (service
  `DELEGATE_QUEUE_KEY`, account `jasonvassallo`, base64-encoded),
  read once at service startup via `/usr/bin/security`. **Fail
  closed:** no key (or malformed key) → the service refuses to start.
  Key material and payload contents are never logged.
- SQLite database at `~/.local/state/delegate-queue/queue.db`, WAL
  mode, directory 0700, db files 0600.

The data still never leaves the user's machines: the queue runs on
JVMBPro, works against the local Ollama, and its only remote exposure
is the user's own Cloudflare-Access-gated tunnel hostname.

## Architecture

### A. Service: `queue_server.py` (repo root)

Standalone PEP 723 script (own dependency set is allowed — it runs as a
LaunchAgent on JVMBPro, not per-MCP-launch): stdlib plus a pinned
`cryptography`.

- HTTP JSON API on `127.0.0.1:11438` (`QUEUE_PORT`), upstream Ollama at
  `http://127.0.0.1:11434` (`QUEUE_UPSTREAM`). Loopback callers are
  trusted; remote exposure/auth happens at the edge (Cloudflare Access
  on the tunnel hostname), not in this service.
- Endpoints:
  - `POST /v1/jobs` — validated Ollama chat payload, ~2 MB cap →
    `{"job_id": "q<uuid4hex>", "status": "queued"}`. The `q` prefix
    distinguishes durable ids from the legacy 32-hex in-memory ids and
    is what routes `local_delegate_result` polls.
  - `GET /v1/jobs/<id>` — status (queued|running|done|failed|cancelled)
    + elapsed/attempts/error class.
  - `GET /v1/jobs/<id>/result` — the result envelope. **Does NOT
    delete**: results persist until TTL. This is the deliberate
    contrast with the legacy single-collect store.
  - `DELETE /v1/jobs/<id>` — cancel (a running job's in-flight upstream
    call finishes but its outcome is discarded).
  - `GET /healthz` — liveness + queue depth.
- Worker: max 2 concurrent jobs, POST to Ollama `/api/chat`, 1800 s
  per-job ceiling. Startup crash recovery: `running` → `queued`;
  attempts counter, max 2 attempts then `failed`. TTL purge of
  finished jobs after 72 h (at startup and on an hourly timer).
- Error-envelope discipline mirrors the MCP server: advisory info as
  JSON fields, never prefixes; payloads never logged; worker error text
  carries transport/HTTP metadata only.
- Implementation note: the approved sketch said "async loop"; the
  shipped worker is a 2-thread pool over the same SQLite store because
  the stdlib HTTP layer (`http.server`) is thread-based and stdlib
  asyncio has no HTTP server. Concurrency semantics (2 in flight,
  ceiling, attempts, recovery) are identical.

### B. Client integration: `mcp_server.py` (stdlib-only additions)

- Queue endpoint chain: default
  `("http://localhost:11438", "https://queue-mbp.djvassallo.com")`,
  env override `AI_TOOLS_QUEUE_URLS` (CSV). Every entry passes the same
  fail-closed validation as the Ollama chain (`_validate_ollama_endpoint`:
  loopback may be http, remote must be https, URL userinfo rejected).
- Auth reuses `_ollama_auth_headers` — the SAME Cloudflare Access
  service token (`OLLAMA_CF_ACCESS_CLIENT_ID`/`_SECRET` Keychain items)
  gates the queue hostname. Credentials unavailable → the remote
  endpoint is skipped entirely, never called bare.
- `local_delegate(background=true)`: probe `/healthz` (~2 s) down the
  chain; on a healthy endpoint, submit there and return the queue
  job_id in the existing JSON envelope plus `"queue": "durable"`. No
  reachable queue → the in-memory path runs unchanged and the envelope
  carries a `"warning"` noting the non-durable fallback (composed with
  any think/model advisory in the same field).
- `local_delegate_result`: ids matching `^q[0-9a-f]{32}$` poll the
  queue service (running → the existing running envelope with
  `"queue": "durable"`; done → the stored chat response through the
  existing renderer; failed/cancelled → the failure envelope; queue
  unreachable → a clean *retryable* error, since the job is persisted
  server-side). Legacy 32-hex ids take the untouched in-memory path.
- Both tool descriptions document durable vs fallback semantics.

### C. Deploy artifacts (committed, NOT executed)

`deploy/jvmbpro-delegate-queue/`:

- `com.jasonvassallo.delegate-queue.plist` — LaunchAgent (not
  LaunchDaemon: it belongs to the user GUI session alongside the Ollama
  LaunchAgent it calls, matching the ollama-nothink-proxy house style),
  `uv run` via absolute `/opt/homebrew/bin/uv`, RunAtLoad, KeepAlive
  SuccessfulExit=false, ThrottleInterval 10, ProcessType Background,
  logs to `~/Library/Logs/delegate-queue.log`.
- `DEPLOY.md` — key provisioning (one pipeline into the v3
  `keychain-write` wrapper, secret never printed), install/bootstrap,
  then the edge steps in gate-before-front-door order: Access
  self-hosted app with a `non_identity` service-token policy reusing the
  existing Ollama token, cloudflared ingress snippet carrying that app's
  AUD tag (before the catch-all; needs sudo), and only then the proxied
  DNS CNAME `queue-mbp` → `<TUNNEL_UUID>.cfargotunnel.com` via the
  Cloudflare API — the record comes last because it is what exposes the
  hostname, and the service itself has no auth. Ends with the 403-bare /
  200-with-token verification. Concrete tunnel/zone/token identifiers
  are placeholders in the runbook; this repo is public.

## Fallback story

The queue is an *upgrade*, never a gate: if no queue endpoint is
reachable (laptop asleep, tunnel down, service not yet deployed, creds
missing), background jobs run exactly as before — in-memory,
single-collect — and say so in the envelope. Sync delegate calls never
touch the queue at all.

## Non-goals

- **No multi-user auth.** Loopback callers are trusted; edge auth is
  Cloudflare Access (service token), and that is the only remote gate.
- **No LAN/Tailscale transport ladder.** Reaching the queue over
  LAN/Tailnet instead of the tunnel is a separate task.
- No model management, no weights, no scheduling policy beyond FIFO.
- No queue for the research (hosted-API) tool families.

## Testing

- `test_queue_server.py` (real `cryptography`, loopback-only): crypto
  round-trip and fresh-nonce, fail-closed key loading, payload
  validation, encrypted-at-rest assertion on raw db bytes, claim/
  attempts/crash-recovery, TTL purge, persistent-until-TTL (result
  collectable repeatedly), cancel semantics incl. discarding a late
  outcome, size cap, and an end-to-end HTTP loop against a fake
  upstream.
- `test_local_delegate.py` additions: queue chain resolution and
  validation, healthz selection with CF-header gating (remote skipped
  without creds), submit/poll mapping, queue-first envelope, fallback
  warning, and `q`-id vs legacy-id routing.
- CI runs both via
  `uv run --with pytest --with cryptography pytest … -q`.
