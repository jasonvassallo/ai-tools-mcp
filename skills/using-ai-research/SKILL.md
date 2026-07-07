---
name: using-ai-research
description: Choose the right research tool for a question — built-in WebSearch for quick lookups, Perplexity quick_research (Sonar) for fast citation-backed answers, Perplexity deep_research (Sonar Pro) for inline multi-source synthesis, Perplexity agent_research (Search-as-Code) for bulk/enumerable research, Gemini Deep Research for long-form reports, or local_delegate (Ollama) for sensitive/private input or local text work. Use whenever the user asks a research question, requests citations, needs to investigate a topic across multiple sources, has sensitive/private input that must stay on-device, or wants to perform cheap local text work (summarize, reformat, boilerplate).
---

# Choosing a Research Tool

The `ai-tools-mcp` server exposes four hosted research APIs and one local-only tool (`local_delegate`) alongside Claude's built-in `WebSearch`. Pick based on **latency**, **depth**, **whether the task enumerates many items**, **whether the report itself is the deliverable**, and **whether the input is allowed to leave the machine at all**.

## Decision Tree

```text
Does the input need to stay off every hosted API — sensitive/private text
  that must not reach a third party — or is the task cheap mechanical work
  on text you already have (summarize, reformat, boilerplate), or do you
  want an independent local second opinion?
  → Use local_delegate (local Ollama, on-device or your own Access-gated
    endpoint — never a third-party API). No web access: if the task
    actually needs research from the web, use a hosted tool below instead.

Is the question a simple factual lookup ("when did X happen", "what is Y's price")?
  → Use built-in WebSearch. Don't burn paid tokens on this.

Is the question well-scoped and a single-source answer with citations enough?
  Need LLM synthesis on top of a search but don't need cross-source reasoning?
  → Use quick_research (Perplexity Sonar). Returns in a few seconds, cheap.

Does the answer need multi-source synthesis or cross-referencing?
  Comparison, ambiguous query, tradeoff/architecture investigation?
  → Use deep_research (Perplexity Sonar Pro). Returns in seconds.

Is the task bulk/enumerable — "for each of these N CVEs/packages/vendors,
  find X" — or does it need computation over search results or a
  structured dataset (CSV/JSON-shaped answer)?
  → Use agent_research (Perplexity Agent API, Search-as-Code). A hosted
    sandbox agent writes code that searches per item, so every item gets
    resolved instead of a sampled few. Takes minutes: call synchronously
    for small fan-outs, or pass background=true and poll
    agent_research_result (~every 30s) for large ones.

Is the report itself the deliverable — a multi-page citation-dense document
  the user will read or share, not just background for the conversation?
  → Use gemini_deep_research_start. Runs for minutes (up to 60).
  → Then poll with gemini_deep_research_result every ~30 seconds.
```

## Latency vs. Depth Tradeoff

| Tool | Provider | Latency | Use when |
|------|----------|---------|----------|
| `WebSearch` | Built-in | <1s | Factual lookup, single answer |
| `quick_research` | Perplexity Sonar | ~2–5s | Well-scoped Q, citations needed, single source OK |
| `deep_research` | Perplexity Sonar Pro | ~5–15s | Multi-source synthesis, cross-referencing |
| `agent_research` | Perplexity Agent API | 1–10 min | Bulk/enumerable tasks, computation, structured output |
| `gemini_deep_research_*` | Google Gemini | 5–60 min | Long-form report is the deliverable |
| `local_delegate` | Local Ollama | seconds–minutes | Input must stay on-device, cheap mechanical work, or a local second opinion |

## Anti-Patterns

- **Don't** use `deep_research` when `quick_research` suffices — Sonar Pro is meaningfully pricier per token than Sonar. Reach for it only when you actually need the multi-source synthesis.
- **Don't** use `quick_research` (or `deep_research`) when `WebSearch` suffices — wastes paid API tokens on cached factoids.
- **Don't** use `agent_research` for a single research question — the per-container fee and orchestration latency are pure overhead there; `deep_research` is faster and cheaper. Reach for it only when the task enumerates items or needs computation.
- **Don't** start `gemini_deep_research_start` for a question the user expects answered in this turn. They'll wait minutes and may abandon.
- **Don't** poll `gemini_deep_research_result` more often than ~30 seconds. Status-completed transitions don't happen faster.
- **Don't** drop citations from any Perplexity output — the value is in source attribution. Pass them through verbatim.
- **Don't** use `local_delegate` when the task needs the web — it has no web access; use one of the hosted research tools above instead.
- **Don't** send sensitive/private input to any hosted tool above — route it through `local_delegate` instead.

## Secret Redaction

`deep_research`, `agent_research`, and `gemini_deep_research_result` route output through a redactor that masks Google API keys, OAuth tokens, JWTs, and PEM private-key blocks. This catches secrets that scraped pages might include. You do not need to re-redact; trust the tool boundary.

## When the Gemini Task Needs Action

If `gemini_deep_research_result` returns `status: "requires_action"`, the agent is paused waiting on user input (typically when `collaborative_planning=true` was set on start). Tell the user — don't keep polling.
