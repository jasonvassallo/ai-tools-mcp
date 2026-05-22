---
name: using-ai-research
description: Choose the right research tool for a question — built-in WebSearch for quick lookups, Perplexity quick_research (Sonar) for fast citation-backed answers, Perplexity deep_research (Sonar Pro) for inline multi-source synthesis, Gemini Deep Research for long-form reports. Use whenever the user asks a research question, requests citations, or needs to investigate a topic across multiple sources.
---

# Choosing a Research Tool

The `ai-tools-mcp` server exposes three hosted research APIs alongside Claude's built-in `WebSearch`. Pick based on **latency**, **depth**, and **whether the report itself is the deliverable**.

## Decision Tree

```text
Is the question a simple factual lookup ("when did X happen", "what is Y's price")?
  → Use built-in WebSearch. Don't burn paid tokens on this.

Is the question well-scoped and a single-source answer with citations enough?
  Need LLM synthesis on top of a search but don't need cross-source reasoning?
  → Use quick_research (Perplexity Sonar). Returns in a few seconds, cheap.

Does the answer need multi-source synthesis or cross-referencing?
  Comparison, ambiguous query, tradeoff/architecture investigation?
  → Use deep_research (Perplexity Sonar Pro). Returns in seconds.

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
| `gemini_deep_research_*` | Google Gemini | 5–60 min | Long-form report is the deliverable |

## Anti-Patterns

- **Don't** use `deep_research` when `quick_research` suffices — Sonar Pro is meaningfully pricier per token than Sonar. Reach for it only when you actually need the multi-source synthesis.
- **Don't** use `quick_research` (or `deep_research`) when `WebSearch` suffices — wastes paid API tokens on cached factoids.
- **Don't** start `gemini_deep_research_start` for a question the user expects answered in this turn. They'll wait minutes and may abandon.
- **Don't** poll `gemini_deep_research_result` more often than ~30 seconds. Status-completed transitions don't happen faster.
- **Don't** drop citations from any Perplexity output — the value is in source attribution. Pass them through verbatim.

## Secret Redaction

Both `deep_research` and `gemini_deep_research_result` route output through a redactor that masks Google API keys, OAuth tokens, JWTs, and PEM private-key blocks. This catches secrets that scraped pages might include. You do not need to re-redact; trust the tool boundary.

## When the Gemini Task Needs Action

If `gemini_deep_research_result` returns `status: "requires_action"`, the agent is paused waiting on user input (typically when `collaborative_planning=true` was set on start). Tell the user — don't keep polling.
