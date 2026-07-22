# ai-tools-mcp

`ai-tools-mcp` is a small MCP server that exposes hosted AI providers behind a stable local MCP surface.

This repository is intentionally narrow in scope:

- It exposes hosted AI providers and the machine's local Ollama server behind one MCP surface.
- No model weights live in this repo — the local family only calls an already-running Ollama.
- It currently exposes thirteen tools across three families:
  - Research: `quick_research` (Perplexity Sonar), `deep_research` (Perplexity Sonar Pro), `agent_research` / `agent_research_result` (Perplexity Agent API, Search-as-Code), `gemini_deep_research_start`, `gemini_deep_research_result`
  - Local delegate: `local_delegate` / `local_delegate_result` (Ollama, on-device)
  - Sessions: `list_sessions`, `save_session`, `load_session`, `update_session`, `delete_session`

The same `mcp_server.py` is shipped three ways: standalone MCP server (installer registers it directly in `~/.claude/.mcp.json`), Claude Code plugin (`.claude-plugin/` + commands/skills/hooks), and Claude Desktop extension (`.mcpb` archive). Pick whichever fits your client.

## Stable Public Surface

The following identifiers are meant to stay stable unless intentionally changed:

- MCP server key: `ai-tools-mcp`
- Tool names (research):
  - `quick_research`
  - `deep_research`
  - `agent_research`
  - `agent_research_result`
  - `gemini_deep_research_start`
  - `gemini_deep_research_result`
- Tool names (local delegate):
  - `local_delegate`
  - `local_delegate_result`
- Tool names (sessions):
  - `list_sessions`
  - `save_session`
  - `load_session`
  - `update_session`
  - `delete_session`

## Provider Mapping

### `quick_research`

- Provider: Perplexity
- Model: `sonar`
- Purpose: fast, concise, citation-backed answers for well-scoped questions
- Latency: a few seconds (synchronous)
- Use when: a single-source answer with citations is enough; cheaper than `deep_research`

### `deep_research`

- Provider: Perplexity
- Model: `sonar-pro`
- Purpose: deep research with multi-source synthesis, cross-referencing, and citations
- Latency: seconds (synchronous)
- Use when: the answer should come back inline in the current session and spans multiple sources

### `agent_research` / `agent_research_result`

- Provider: Perplexity Agent API (`/v1/responses`) with the `sandbox` tool ("Search as Code")
- Models: `anthropic/claude-sonnet-4-6` (default) or `perplexity/sonar`, server-side allowlist (the Agent API does not offer `sonar-pro`)
- Purpose: bulk/enumerable research — the upstream agent writes and runs code in a Perplexity-hosted container, searching programmatically so every item in a list gets resolved (chat synthesis samples a few and generalizes)
- Latency: one to several minutes; synchronous by default, or pass `background=true` to get a `response_id` immediately and poll `agent_research_result`
- Cost: per-model tokens + $0.03 per sandbox container + per-invocation charges for searches made inside the sandbox; higher and less predictable per request than `deep_research`
- Use when: the task is "for each of these N items, find X", needs computation over search results, or must produce a structured dataset. For a single research question use `deep_research` instead.

### `gemini_deep_research_start` / `gemini_deep_research_result`

- Provider: Google Gemini Deep Research (`/v1beta/interactions`)
- Models: `deep-research-preview-04-2026` (fast) and `deep-research-max-preview-04-2026` (max)
- Purpose: long-running, citation-dense reports drawing on many sources
- Latency: minutes (up to 60); asynchronous, polled via the `_result` tool
- Use when: you need a standalone, multi-page report — not a quick answer

### `local_delegate` / `local_delegate_result`

- Provider: **local-first Ollama endpoint chain** — default `http://localhost:11434` → `https://ollama-mbp.djvassallo.com` → `https://ollama.djvassallo.com` (both Cloudflare-Access-gated); override via `AI_TOOLS_OLLAMA_URLS` comma-separated env (singular `AI_TOOLS_OLLAMA_URL` honored for compat), Keychain `OLLAMA_URL` appended; per-call `/api/tags` probe picks the first endpoint serving the tag, cached 60s; remote endpoints require https + CF Access service-token creds (env vars or macOS Keychain — see Credentials), else skipped
- The `https://ollama-mbp.djvassallo.com` remote entry is the repo owner's own Access-gated host — a **placeholder** for everyone else; set `AI_TOOLS_OLLAMA_URLS` (or the Desktop extension's `ollama_endpoints` setting) to your own endpoint(s) instead of relying on the default chain
- Models: server-side allowlist, default = `gemma4:12b-nvfp4` followed by four qwen3.6 tags (the qwen base tag inherits each host's window — 64k JVMBPro / 32k jvmacmini — while `-32k`/`-64k`/`-256k` pin the window explicitly regardless of host; all four tags share one weight blob on disk); override the allowlist per machine via `AI_TOOLS_OLLAMA_MODELS` comma-separated env or the extension's `ollama_models` setting (first entry becomes the default model; blank/garbage fails closed to the built-ins). Omitted-model calls resolve local-first across the allowlist — the first chain endpoint serving any allowlisted tag picks the model, so a missing default falls back with an advisory instead of failing or silently going remote when a local option exists; `AI_TOOLS_OLLAMA_DEFAULT_MODEL` may pick a different allowlisted tag
- Purpose: privacy / quota offload / second opinion / background jobs
- Latency: seconds-to-minutes, synchronous by default, or pass `background=true` to get a `job_id` and poll `local_delegate_result`
- Privacy: **input stays on your machines** — on-device when localhost serves the model, otherwise only your own Access-gated endpoint, never a third-party API; nothing written to disk; jobs are in-memory and single-collect
- `think`: off by default (faster); set `true` only for reasoning-heavy asks
- `keep_alive`: omit to inherit the server's `OLLAMA_KEEP_ALIVE`

Together these complement Claude's built-in `WebSearch`: use `WebSearch` for
quick lookups, `deep_research` for thorough inline investigation,
`agent_research` for bulk/enumerable tasks where coverage matters, the
`gemini_deep_research_*` pair when the deliverable IS the report, and
`local_delegate` when the input must stay on-device or the task is cheap
mechanical work.

## How It Works

The server is a single Python script (`mcp_server.py`) with PEP 723 inline dependency metadata. `uv run` resolves and caches dependencies automatically — no virtualenv, no `requirements.txt`, no build step.

It:

- starts an MCP server named `ai-tools-mcp`
- reads API credentials from environment variables or the macOS Keychain (env wins; env is the only source on non-macOS hosts)
- calls the Perplexity API through the `openai` Python client
- calls the local Ollama server (native /api/chat) for the local_delegate family
- returns plain text MCP responses

There are no local model weights, no persistent background service, and no embedded secrets in the repo. (The local_delegate family only calls an already-running Ollama server; background jobs are in-process asyncio tasks.)

## Repository Layout

Source:
- `mcp_server.py`: Self-contained MCP server with PEP 723 inline dependency metadata (single source of truth — both packaging formats wrap this same file)

Standalone install:
- `install.sh`: macOS installer (Keychain setup, Claude Code registration, preflight check)
- `scripts/launch_ai_tools_mcp.sh`: Legacy launcher (virtualenv-based, for Claude Desktop)
- `scripts/configure_claude_ai_tools_mcp.sh`: Claude Desktop registration helper
- `scripts/uv_sync_projects.sh`: Separate local helper for syncing Python projects with `uv`

Claude Code plugin (loaded via `claude --plugin-dir .`):
- `.claude-plugin/plugin.json`: Plugin manifest (name, version, author)
- `.mcp.json`: MCP server registration (points at `mcp_server.py` via `${CLAUDE_PLUGIN_ROOT}`)
- `commands/`: Eleven slash commands (`/ai-tools-mcp:quick-research`, `:deep-research`, `:agent-research`, `:gemini-start`, `:gemini-result`, `:local-delegate`, `:sessions`, `:save-session`, `:load-session`, `:update-session`, `:delete-session`)
- `skills/using-ai-research/`: When-to-use routing skill (WebSearch vs. Perplexity vs. Gemini)
- `skills/session-workflows/`: Save/load/rename/delete patterns
- `hooks/hooks.json` + `hooks/preflight.sh`: `SessionStart` hook that runs `--check` and surfaces credential health to Claude

Claude Desktop extension (built into `dist/ai-tools-mcp.mcpb`):
- `mcpb/manifest.json`: DXT/MCPB v0.3 manifest (server type, tool declarations, user_config)
- `scripts/build_mcpb.sh`: Build script — copies `mcp_server.py` into `mcpb/server/` and zips
- `dist/ai-tools-mcp.mcpb`: Build output (gitignored)

## Installation

Three options, depending on your Claude client:

### A. Standalone MCP server (Claude Code via `~/.claude/.mcp.json`)

The original install path — runs the bare MCP server with no plugin wrapper.

```bash
./install.sh
```

This will:

1. Check prerequisites (`uv`, `jq`, macOS Keychain)
2. Install `mcp_server.py` to `~/.local/share/ai-tools-mcp/`
3. Prompt for API tokens and store them in the macOS Keychain
4. Register the MCP server in `~/.claude/.mcp.json`
5. Run a preflight check to verify dependencies and configuration

Safe to run multiple times — updates existing config without clobbering.

### B. Claude Code plugin (commands + skills + hooks + MCP server)

Bundles the MCP server with slash commands (`/ai-tools-mcp:deep-research <q>`, `/ai-tools-mcp:sessions`, etc.), routing skills (when to use Perplexity vs. Gemini vs. WebSearch, session-management workflows), and a `SessionStart` preflight hook that verifies Perplexity Keychain + ADC are healthy before your first query.

Test locally — Claude Code loads the plugin directly from this directory:

```bash
claude --plugin-dir /path/to/ai-tools-mcp
```

The plugin manifest lives at `.claude-plugin/plugin.json`. To make it permanently installable via `/plugin install`, publish through a marketplace (see [plugin-marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)).

### C. Claude Desktop extension (.mcpb, drag-to-install)

Build the single-file `.mcpb` archive:

```bash
./scripts/build_mcpb.sh
```

Then drag `dist/ai-tools-mcp.mcpb` into Claude Desktop → Settings → Extensions. The extension's user-config UI exposes the `uv_path` field (defaults to `/opt/homebrew/bin/uv` for Apple-Silicon Homebrew installs).

First-run note: macOS may show a one-time Keychain access prompt when the server reads your Perplexity key — approve it.

### D. Windows (Claude Code CLI, Claude Code desktop app, or Claude Desktop)

Everything runs from env vars on Windows — no Keychain, no installer script.

## Benchmarks

`benchmarks/local-model-bench/` holds the ground-truth A/B harness behind the delegate-model defaults (planted-defect review corpus, machine-graded delegate tasks, blind adversarial judging). See its [README](benchmarks/local-model-bench/README.md) and [RESULTS](benchmarks/local-model-bench/RESULTS.md) for the 2026-07 measurements that set gemma4 as the default.

1. **Prerequisites:** [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`), and for `local_delegate` a local [Ollama for Windows](https://ollama.com/download/windows) with a model pulled — on a 32 GB CPU-only box use `ollama pull qwen2.5-coder:14b` (~9 GB; the qwen3-coder line starts at 30B and will not fit alongside office apps).
2. **Credentials:** set the env vars from the Credentials table above via the System Environment Variables GUI. For Gemini tools, install the Google Cloud SDK and run `gcloud auth application-default login` then `gcloud auth application-default set-quota-project YOUR_PROJECT` (ADC is fully portable; no key files).
3. **Claude Code (CLI and the desktop app share one config):**

   ```powershell
   git clone https://github.com/jasonvassallo/ai-tools-mcp
   # Use the ABSOLUTE path to uv (run `where uv`), not bare `uv` — the
   # desktop app spawns servers with a minimal PATH and won't find it.
   claude mcp add ai-tools-mcp --scope user --env GOOGLE_CLOUD_PROJECT=YOUR_PROJECT --env UV_PRERELEASE=if-necessary-or-explicit -- C:\path\to\uv.exe run C:\path\to\ai-tools-mcp\mcp_server.py
   ```

   `--env GOOGLE_CLOUD_PROJECT=...` is baked into the registration because GUI-launched servers don't inherit your shell environment: user-credential ADC files carry no project id, and google-auth's fallback discovery (which consults the gcloud CLI) isn't reliable from a GUI-spawned process — pinning the project in the registration sidesteps both. Optional per-machine env (append more `--env` flags or set them alongside the credentials): `AI_TOOLS_OLLAMA_MODELS=qwen2.5-coder:14b,gemma4:12b-nvfp4,qwen3.6:35b-a3b-coding-nvfp4,qwen3.6:35b-a3b-coding-nvfp4-256k` — the small model serves locally; the gemma4 and qwen3.6 tags miss the local probe (the box is CPU-only — nvfp4 is Apple-silicon-served) and fall through the remote chain to ollama-mbp when the MBP is awake. Implicit calls still resolve to the local small model first by design; name gemma explicitly when quality matters more than locality.
4. **Claude Desktop:** install the `.mcpb` as in (C), then in the extension settings set `uv_path` to your `uv.exe` (find it with `where uv` — the default is a macOS Homebrew path) and `ollama_models` as above.
5. **Platform note:** all 13 tools work on Windows. `update_session`/`delete_session` lock via `msvcrt.locking` byte-range locks there (`fcntl.flock` on POSIX) — same lockfile, same serialization guarantees.
6. **Verify:** `uv run C:\path\to\mcp_server.py --check` — hosted-tool credentials must pass; the Ollama line is non-fatal (`warn:` when the local server is down and calls will use the remote chain).

### Troubleshooting: "Server disconnected" in a desktop app

The two Claude desktop apps register this server differently, so start by
identifying which one is complaining:

- **Claude Code desktop app** shares the CLI's registration (`claude mcp
  list/add/remove`, stored in `~/.claude.json`) — its fixes are the first two
  bullets below.
- **Claude Desktop** runs the `.mcpb` extension (shown as "AI Tools MCP") —
  see the Desktop-specific items at the end. A banner naming lowercase
  `ai-tools-mcp` in Claude Desktop usually means a **legacy `mcpServers` entry
  in `claude_desktop_config.json`** is also registered — the third bullet.

If the app shows *"MCP ai-tools-mcp: Server disconnected"* while `claude mcp
list` in a **terminal** shows the same server ✔ Connected, the cause is almost
always one of these — all about how GUI apps launch servers, not about the
server itself:

- **Bare command name / minimal PATH.** GUI-launched apps spawn MCP servers
  with a stripped-down PATH that omits `/opt/homebrew/bin`, `~/.local/bin`,
  `%LOCALAPPDATA%\...`, etc. A registration using bare `uv` connects from your
  shell but fails in the app. Fix: re-register with the **absolute** command
  path (find it with `which uv` on macOS/Linux, `where uv` on Windows) and bake
  required env in.

  macOS / Linux:

  ```bash
  claude mcp remove ai-tools-mcp
  claude mcp add ai-tools-mcp --scope user \
    --env GOOGLE_CLOUD_PROJECT=YOUR_PROJECT \
    --env UV_PRERELEASE=if-necessary-or-explicit \
    -- /absolute/path/to/uv run /absolute/path/to/ai-tools-mcp/mcp_server.py
  ```

  Windows (PowerShell):

  ```powershell
  claude mcp remove ai-tools-mcp
  claude mcp add ai-tools-mcp --scope user --env GOOGLE_CLOUD_PROJECT=YOUR_PROJECT --env UV_PRERELEASE=if-necessary-or-explicit -- C:\path\to\uv.exe run C:\path\to\ai-tools-mcp\mcp_server.py
  ```

- **The app wasn't fully restarted.** Closing the window (⌘W, or the red
  traffic-light / the ✕ button) does **not** reload config — the app and its
  server processes keep running. After any registration change, **fully quit**
  and reopen:
  - **macOS:** ⌘Q, then confirm it's gone from the Dock (or `pgrep -f Claude`
    returns nothing) before reopening.
  - **Windows:** right-click the Claude icon in the system tray → Quit (or End
    task in Task Manager), then reopen — the tray process outlives a closed
    window.

  This is the single most common reason a corrected registration still shows
  the old error.

- **A registration points at a deleted path (survives every restart).** If a
  server was ever installed via a copy (e.g. `install.sh`'s
  `~/.local/share/ai-tools-mcp/`) and that copy was later removed, any config
  still referencing it fails on every launch. Check **all** of:
  `~/.claude.json` (`mcpServers`), `~/.claude/.mcp.json`, and — for Claude
  Desktop — `claude_desktop_config.json`'s `mcpServers` block. Delete stale
  entries (the `.mcpb` extension makes a config-file entry redundant in
  Claude Desktop anyway), then fully quit and reopen.

Claude Desktop `.mcpb`-specific notes: a same-id local `.mcpb` dragged in on
top of an installed one is a **no-op** — Desktop keeps the old copy. Uninstall
the extension in Settings → Extensions first, then install the new `.mcpb`,
then fully quit and reopen. Launch problems for the extension itself are
usually the `uv_path` setting (must be an absolute path valid on *this*
machine).

## Requirements

### System

- macOS
- `uv` installed and on `PATH`
- `jq` installed (for installer only)

### Credentials (environment variables or macOS Keychain)

Every credential resolves **environment-first, then macOS Keychain** — a
non-empty env var wins; on non-macOS hosts (Windows/Linux, where there is no
`security(1)`) env vars are the only source. A miss in both raises one error
naming both remedies.

| Credential | Env var | Keychain (service / account) | Needed for |
|---|---|---|---|
| Perplexity API key | `PERPLEXITY_API_KEY` | `api_tokens` / `perplexity` | quick/deep/agent research |
| CF Access client id | `OLLAMA_CF_ACCESS_CLIENT_ID` | `OLLAMA_CF_ACCESS_CLIENT_ID` / `$USER` | remote Ollama endpoints |
| CF Access client secret | `OLLAMA_CF_ACCESS_CLIENT_SECRET` | `OLLAMA_CF_ACCESS_CLIENT_SECRET` / `$USER` | remote Ollama endpoints |
| Extra Ollama endpoint (optional) | `OLLAMA_URL` | `OLLAMA_URL` / `$USER` | appended to the chain |

macOS setup (the installer handles the Perplexity one automatically):

```bash
security add-generic-password -s 'api_tokens' -a 'perplexity' -w 'YOUR_PERPLEXITY_API_KEY'
```

Windows setup: set per-user environment variables through **Settings →
System → About → Advanced system settings → Environment Variables** (the GUI
keeps secrets out of shell history). Never put them in files in the repo.
Trade-off note: a persisted user env var is readable by any process running
as the same user — weaker isolation than the macOS Keychain. Accepted,
documented design choice for non-macOS hosts.

Naming gotcha: `OLLAMA_URL` (above) **appends one extra endpoint** to the
chain; the similarly named `AI_TOOLS_OLLAMA_URL`/`AI_TOOLS_OLLAMA_URLS`
**replace the entire chain**. Use the `AI_TOOLS_*` vars to control ordering.

### Google Cloud Application Default Credentials (ADC)

The Gemini Deep Research tools authenticate via **ADC**, not a static API
key. Set this up once with:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT
```

The server reads ADC from the standard location
(`~/.config/gcloud/application_default_credentials.json`) and refreshes
short-lived bearer tokens transparently via the `google-auth` library. The
billing project is auto-detected from ADC.

The preflight check (`uv run mcp_server.py --check`) verifies both the
Perplexity key and ADC, refreshing a token to confirm credentials are live.

## Running

The recommended way to run is via `uv run`, which reads the PEP 723 inline metadata and manages dependencies automatically:

```bash
uv run mcp_server.py
```

No virtualenv creation or dependency installation needed — `uv` handles it from its global cache.

### Preflight Check

Validate that dependencies resolve and the Keychain entry exists without starting the server:

```bash
uv run mcp_server.py --check
```

## Claude Code Registration

The installer handles this. For manual setup, add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "ai-tools-mcp": {
      "command": "uv",
      "args": ["run", "/path/to/ai-tools-mcp/mcp_server.py"],
      "env": {
        "UV_PRERELEASE": "if-necessary-or-explicit"
      }
    }
  }
}
```

## Claude Desktop Registration

For Claude Desktop (uses the legacy virtualenv launcher):

```bash
./scripts/configure_claude_ai_tools_mcp.sh
```

## Tool Inputs

### `deep_research`

Input schema:

- `query`: required string
- `max_tokens`: optional integer, default `2048`

Output behavior:

- returns a formatted text block with research results
- relies on Perplexity response content to include citations
- response is routed through a redactor that masks secret-shape strings
  (Google API keys, OAuth tokens, JWTs, private-key blocks)

### `agent_research`

Input schema:

- `query`: required string — the bulk research task, with items and per-item
  fields stated explicitly
- `model`: optional `"anthropic/claude-sonnet-4-6" | "perplexity/sonar"`
  (default `"anthropic/claude-sonnet-4-6"`; server-side allowlist — the Agent
  API can route to many third-party models, but this tool refuses anything
  outside the allowlist so a malformed or injected request cannot select an
  arbitrary upstream model)
- `max_output_tokens`: optional integer in `[256, 8192]`, default `4096`
- `background`: optional boolean, default `false` — when `true`, returns
  `{response_id, status, hint}` immediately instead of waiting; poll with
  `agent_research_result`. Prefer `background=true` for large fan-outs:
  synchronous calls block for up to 10 minutes, and MCP clients that enforce
  their own shorter tool-call timeouts will kill the call before the server's
  600s ceiling — background mode is the safe path, not just an optimization

Output behavior (synchronous, and `agent_research_result` on completion):

- returns a formatted text block: the agent's answer, then a metadata footer
  (model, sandbox execution count, itemized cost in USD as reported by the API)
- failed sandbox executions (non-zero exit codes) are listed with truncated
  stderr snippets for diagnosis
- a non-`completed` upstream status (e.g. truncation) is flagged inline
- all model-emitted text is routed through the same secret-redactor as
  `deep_research`

### `agent_research_result`

Input schema:

- `response_id`: required string (must match `^[A-Za-z0-9_-]{1,128}$`;
  enforced at the tool boundary — and re-validated inside the HTTP helper —
  to prevent the authenticated request from being redirected to an
  attacker-controlled host)

Output behavior:

- the formatted answer block (above) when `status` is `completed` (or
  `incomplete` with partial output, flagged inline)
- `{status, hint}` while `queued` / `in_progress` — poll again in ~30s
- `{status: "failed", error}` when the task failed, was cancelled, or the
  HTTP call errored

### `gemini_deep_research_start`

Input schema:

- `query`: required string
- `mode`: optional `"fast" | "max"` (default `"fast"`)
- `collaborative_planning`: optional boolean (default `false`)
- `thinking_summaries`: optional `"auto" | "none"` (default `"auto"`)

Output behavior:

- returns JSON `{interaction_id, status, model, hint}`
- the task runs in the background on Google's side; poll with the result tool

### `gemini_deep_research_result`

Input schema:

- `interaction_id`: required string (must match `^[A-Za-z0-9_-]{1,128}$`;
  this is enforced at the tool boundary to prevent the authenticated request
  from being redirected to an attacker-controlled host)

Output behavior:

- `{status: "completed", output_text, steps_count, steps_summary}` when done
- `{status: "failed", error}` on failure
- `{status: "in_progress", hint}` while still running — poll again in ~30s
- `output_text` and `error` are routed through the same secret-redactor as
  `deep_research`

## Development Notes

- This repo is intentionally small and single-purpose.
- Keep the public MCP surface stable.
- Avoid adding unrelated automation or local-model functionality here.
- If functionality drifts beyond hosted MCP tooling, it should likely live in a different repo.

## Secret Hygiene

This repo is intended to be safe to publish publicly. The current design keeps secrets out of source control by using the macOS Keychain.

Rules for working in this repo:

- never commit raw API keys, session tokens, or bearer tokens
- never commit `.env` files
- never commit certificate or private key files
- keep local debug dumps out of Git
- prefer Keychain lookups over environment-file storage for this project

## License

No license file has been added yet. If you plan to accept contributions or want reuse clarity, add one explicitly.
