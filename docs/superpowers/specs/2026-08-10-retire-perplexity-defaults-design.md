# Retire Perplexity/Sonar defaults — design

Date: 2026-08-10. Approved by Jason in-session.

## Goal

No Perplexity-billed tool may be selected implicitly. Default recent-web
lookup moves to Gemini with Google Search grounding. Explicitly-invoked
Perplexity Agent research stays available on the user's key.

## Context and verified facts

- OpenClaw (`~/.openclaw/openclaw.json` on JVMacMini) had
  `tools.web.search.provider=perplexity`; ai-tools-mcp's `quick_research` /
  `deep_research` were Sonar / Sonar Pro.
- The keychain value stored as `GOOGLE_API_KEY` (System keychain,
  `ai.openclaw`) is an OAuth-family token (`AQ.A…`), rejected by
  `generativelanguage.googleapis.com` with `ACCESS_TOKEN_TYPE_UNSUPPORTED`.
  It is NOT an AI Studio API key. Left untouched.
- ADC (`ai-orchestrator-482302`) works against Vertex `aiplatform` with
  `googleSearch` grounding (locations `global` and `us-central1`), and the
  server already loads ADC lazily (`_load_adc`, PR #57 quota fallback).
- On Vertex, `gemini-flash-latest` and `gemini-3.6-flash` resolve;
  `gemini-flash-3.6` does not exist. On the AI Studio key,
  `gemini-flash-latest` works but `gemini-2.5-flash` is 404
  ("no longer available to new users") — so the model must be pinned
  explicitly everywhere.

## Part A — OpenClaw (config only, no code)

1. A new AI Studio API key (uid `f34659c0-…`, display name
   `ai-tools-gemini-search`) was minted on `ai-orchestrator-482302`,
   restricted to `generativelanguage.googleapis.com`, and staged in the
   LOGIN keychain as `ai.openclaw` / `GEMINI_API_KEY` (zero-argv write).
2. USER STEP (root required):
   `sudo ~/.local/bin/migrate-token-to-system-keychain.sh ai.openclaw GEMINI_API_KEY`
3. `~/.local/bin/apply-openclaw-gemini-search.sh` (staged, fail-closed on
   the System-keychain precondition) then:
   - backs up `openclaw.json`
   - `tools.web.search.provider` → `gemini`
   - `plugins.entries.google.config.webSearch` → keychain secretRef
     `GEMINI_API_KEY` + `model: gemini-flash-latest` (highest credential
     precedence, so the broken `models.providers.google.apiKey` fallback
     never engages)
   - `plugins.allow`: add `google`, remove `perplexity`
   - `plugins.entries.perplexity.enabled` → `false` (entry kept for easy
     restore)
   - restarts `system/ai.openclaw.gateway`, prints verify + rollback lines

## Part B — ai-tools-mcp (this repo)

- New `_vertex_generate_content(model, system, user_text, max_tokens)`
  helper: POST
  `https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent`
  with `tools: [{"googleSearch": {}}]`, auth via the existing
  `_get_bearer_token()` ADC path and the shared lazy `httpx.AsyncClient`.
  Location `global` per the verified-cheapest Vertex note.
- `quick_research` and `deep_research` keep their names (AGENTS.md stable
  surface), both move to `gemini-flash-latest`, differing by system prompt
  exactly as the Sonar/Sonar-Pro split did. The defaults were planned as
  1024 / 2048 (carried over from Sonar) but **shipped as 8192 / 16384**:
  Stage 3 adversarial review demonstrated live that this model bills its
  thinking tokens against `maxOutputTokens` (measured 666–1851), so the
  carried-over budgets truncated ordinary answers mid-sentence. The
  renderer now also reads `finishReason` so a truncated or otherwise
  abnormal response is named rather than returned as a clean answer. Responses render synthesized text plus a
  Sources section from `groundingMetadata.groundingChunks`
  (title + URI) and `webSearchQueries`; all output passes
  `redact_secrets` as before.
- `agent_research` / `agent_research_result` stay on the Perplexity Agent
  API and the user's `PERPLEXITY_API_KEY` — explicit invocation only, per
  Jason's direction. No gating env var.
- Dead code removed (standing rule): `_get_perplexity_client`,
  `_perplexity_client_cache`, `from openai import OpenAI`, and the
  `openai>=1.0.0` script dependency (the Agent API path uses httpx + its
  own key fetch, not the OpenAI client).
- Docs: module docstring, README, AGENTS.md tool descriptions.
- Lockstep version bump: `.claude-plugin/plugin.json` and
  `mcpb/manifest.json` 1.5.5 → 1.6.0. (Planned from 1.5.4; `main`
  advanced to 1.5.5 via PR #65 before this branch merged.)
- Tests: new coverage for `_vertex_generate_content` (mocked transport,
  grounding-chunk rendering, empty-candidates fail-closed) and updated
  quick/deep tests; Perplexity-client tests removed with the client.
- After merge: refresh `~/.local/share/ai-tools-mcp/mcp_server.py` and
  keep the `.bak` convention.

## Billing note (accepted)

This consolidates vendors, not spend: Google Search grounding is metered
on both surfaces (per grounded request on the Gemini API for OpenClaw;
GCP billing for Vertex in ai-tools-mcp). Accepted by Jason in-session.

## Out of scope

- email_triage PR #28 (closed, do not reopen)
- transport-chain, remote queue, review-pipeline recovery follow-ups
- the Qwen LaunchAgent symlink cleanup
