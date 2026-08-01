# ai-tools-mcp

This repository contains a small MCP server that exposes hosted AI APIs and the machine's local Ollama server behind one MCP surface. No model weights live here.

## Purpose

- Expose `deep_research` for Perplexity Sonar Pro deep research with multi-source synthesis and citations.
- Expose `agent_research` for Perplexity Agent API Search-as-Code — a hosted sandbox agent that searches programmatically for bulk/enumerable research tasks.
- Expose `gemini_deep_research_start` / `_result` for Google Gemini Deep Research — long-running, citation-dense reports drawn from many sources.
- Complement Claude's built-in WebSearch (quick lookups) with thorough-research tiers (fast inline via Perplexity, programmatic bulk via the Agent API sandbox, multi-minute report via Gemini).

## Stable Public Surface

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
- Tool names (session management):
  - `list_sessions`
  - `save_session`
  - `load_session`
  - `update_session`
  - `delete_session`

Keep those tool names stable unless a change is explicitly requested.

## Packaging Formats

The same `mcp_server.py` is wrapped three ways. When making changes, update all three:

- Standalone: `install.sh` registers it directly in `~/.claude/.mcp.json`
- Claude Code plugin: `.claude-plugin/plugin.json` + `.mcp.json` + `commands/` + `skills/` + `hooks/`
- Claude Desktop extension: `mcpb/manifest.json` (built into `dist/ai-tools-mcp.mcpb` by `scripts/build_mcpb.sh`)

Every surface that launches or preflight-checks the server via `uv run`
pins `UV_PRERELEASE=if-necessary-or-explicit` in its env (the three
launch entries above, `hooks/preflight.sh`, `install.sh`'s check, and
the README's manual `claude mcp add` examples), and the repo-level
`uv.toml` pins the same policy for repo-cwd runs. There is no single
source of truth, so if the prerelease policy ever changes, find every
copy with `grep -r UV_PRERELEASE` and update them together (see the
comments in `uv.toml` for why the env pin is what actually protects
installed runs). CI's `prerelease-guard` job in
`.github/workflows/tests.yml` enforces the pins on the executable
surfaces and the `uv.toml` policy line, so partial edits fail CI
instead of drifting silently — update its expectations alongside any
policy change.

## CI Gates on `main`

`semgrep/ci` is a required status check. `claude-gate`
(`.github/workflows/claude-gate.yml`, a cheap binary Claude Haiku 4.5
merge gate) is not yet required in branch protection — that's a
deliberate, separate follow-up step, pending here until it produces a
real passing run. `claude-gate` carries no `trivial`-label guard: its
`if:` condition only excludes forks, draft PRs, and PRs whose
triggering actor is `dependabot[bot]` — it does not check PR
authorship, so another bot's same-repo PR (e.g. Renovate) still runs
it, and a human reopening a Dependabot PR also passes the actor check.
It exists so a `trivial`-labeled PR still has at least one enforced
check: `trivial` guards only `claude-review` and `semgrep/ci`.
CodeRabbit's automatic reviews are disabled fleet-wide regardless of
label (`.coderabbit.yaml`), and Greptile is gated by the separate
`greptile` label, not `trivial` — a PR carrying both labels still gets
a Greptile review.
It runs on plain `pull_request`, which means a same-repo PR editing this
workflow file could in principle neuter its own gate check — a
`pull_request_target` trigger would close that hole, but is a confirmed
no-go here: `claude-code-action`'s OIDC token exchange rejects
`pull_request_target`-sourced tokens outright (401 on every retry,
reproduced on PR #55). See the comment in `claude-gate.yml` before
trying that again. The accepted trade-off is narrow: it requires
existing write access to this repo, and forks are already excluded.

## Provider Mapping

- `quick_research`:
  - Provider: Perplexity
  - Model: `sonar`
- `deep_research`:
  - Provider: Perplexity
  - Model: `sonar-pro`
- `agent_research` / `agent_research_result`:
  - Provider: Perplexity Agent API (`/v1/responses`) with the `sandbox` tool
  - Models: `anthropic/claude-sonnet-4-6` (default) or `perplexity/sonar` — server-side allowlist (the Agent API does not offer `sonar-pro`)
  - Synchronous by default (runs take one to several minutes), or `background=true` returns a `response_id` to poll via `agent_research_result`; billed per-model tokens + per-container fee + per-search charges.
- `gemini_deep_research_start` / `gemini_deep_research_result`:
  - Provider: Google Gemini Deep Research (`/v1beta/interactions`)
  - Models: `deep-research-preview-04-2026` (fast), `deep-research-max-preview-04-2026` (max)
  - Asynchronous; start returns an `interaction_id`, result polls until `status="completed"`.
- `local_delegate` / `local_delegate_result`:
  - Provider: local-first Ollama endpoint chain (localhost first, then the user's own Access-gated remote) — on-device/own-infra, never a third-party API

## Dependencies

`mcp_server.py` uses PEP 723 inline script metadata — no virtualenv or project config needed. `uv run mcp_server.py` handles everything.

Credentials resolve env var → Windows Credential Manager → macOS Keychain
(first non-empty wins). The vault tier is Windows-only and applies only to
the services listed in `_CRED_VAULT_TARGETS` (currently the Cloudflare
Access service token); it is read in-process via ctypes/`advapi32.CredReadW`
because Claude Desktop launches the packaged `.mcpb` outside any shell and
inherits no profile. Keep env first — that is what makes an install
upgrade-safe and a vault cutover reversible. Never log a credential value;
`_resolve_credential` returns a source label for that purpose.

- macOS Keychain: service `api_tokens`, account `perplexity` (required)
- Windows Credential Manager: generic targets `ai-tools-mcp-cf-access/client-id` and `.../client-secret`
- Google Cloud ADC at `~/.config/gcloud/application_default_credentials.json` (required for Gemini Deep Research). Configure with `gcloud auth application-default login`. Billing project is auto-detected.

## Running

```bash
uv run mcp_server.py
```

## Claude Code Registration

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "ai-tools-mcp": {
      "command": "uv",
      "args": ["run", "/path/to/mcp_server.py"],
      "env": {
        "UV_PRERELEASE": "if-necessary-or-explicit"
      }
    }
  }
}
```

## Guardrails

- Keep this repo focused on hosted API-backed MCP tooling, plus the
  `local_delegate` family, which only *calls* an already-running Ollama.
- Local-model features here are limited to calling an already-running Ollama
  server (the `local_delegate` family) — no model weights, no inference
  engines, no model management.
- Do not add email-triage code here.
- Model weights, inference engines, and deeper local-AI projects still
  belong in `local_llm_integration`, not this repo.
- Never publish secrets, API keys, tokens, `.env` files, certificates, or private keys to GitHub from this repo.
