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
pins `UV_PRERELEASE=if-necessary` in its env (the three
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

`semgrep/ci` and `claude-gate`
(`.github/workflows/claude-gate.yml`, a cheap binary Claude Haiku 4.5
merge gate) are both required status checks. `claude-gate` was promoted
to required after it landed on `main` untouched and produced real
passing runs (#60 in 53s and #63 in 39s; #64 then passed in 47s, after
the promotion); branch protection returned both contexts when it was
last read live on 2026-08-09. `claude-gate` carries no `trivial`-label
guard: its
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
trying that again. `claude-code-action` does have its own partial
anti-tamper check instead: it refuses to run when the invoking PR's
copy of `claude-gate.yml` differs from `main`'s (confirmed live on
#55). Be precise about what that does and does not cause. The action
**self-skips**, and a self-skip is not a step failure — the action's
own step still reports `success`. What turns the check red is the
separate "Enforce verdict (fail closed)" step finding no verdict file.
Verified on #66, job 93321756563: gate step `success`, enforce step
`failure`. So an edit that KEEPS the enforcement step fails its own
required check, while an edit that DELETES that step — or the action
step — passes with nothing run. That stops a PR from rewriting the
prompt/model to self-approve, but not one that removes the machinery
outright. The accepted trade-off is narrow:
exploiting either gap requires existing write access to this repo
(forks are already excluded), and a write-access collaborator has
simpler ways to bypass CI than editing this file.

### Landing a change to `claude-gate.yml`

An edit that keeps the enforcement step fails its own required check,
so such a PR needs an administrator exception. **Prefer the narrowest
lever** — temporarily disable `enforce_admins`, merge that one PR
pinned to its exact head, then re-enable it:

```
gh api -X DELETE repos/OWNER/REPO/branches/main/protection/enforce_admins
gh pr merge N --squash --match-head-commit SHA --admin
gh api -X POST   repos/OWNER/REPO/branches/main/protection/enforce_admins
```

Be clear about the trade: disabling `enforce_admins` is narrower in
**who** (admins only, for the window) but broader in **what** — for
that window admins are exempt from *every* main protection, including
`allow_force_pushes: false` and `allow_deletions: false`, not just the
status checks. It is preferred because the alternative provably leaves
zero enforced CI on a `trivial` PR, not because it is unconditionally
safer. Keep the window to a single pinned merge.

Do **not** reach first for dropping `claude-gate` from the required
contexts. That relaxes `main` repo-wide for the whole window, and if
the PR is labelled `trivial` it is worse than useless: `semgrep/ci`
skips on that label and a skipped required check counts as satisfied,
so the PR would merge with ZERO enforced CI — precisely the #52/#53
hole this job exists to close. If the contexts must be rewritten
anyway, use the `checks` form with an explicit `app_id`; a bare
`contexts` array lets any app satisfy the context, which is a silent
and permanent weakening.

Whichever lever is used: get explicit administrator approval first,
record the exception in the PR body, restore protection
unconditionally — even if the merge fails — verify the restore by
re-reading the endpoint and diffing it against the captured prior
state rather than trusting a PATCH exit code, and prove the gate is
operational again on a later PR. A protection read only shows the
context is required; it does not show the job still runs and can fail
closed.

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
        "UV_PRERELEASE": "if-necessary"
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
