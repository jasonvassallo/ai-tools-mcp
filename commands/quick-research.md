---
description: Run a fast Gemini grounded-search query (concise, citation-backed, smaller answer budget than deep_research)
argument-hint: <research-query>
---

Use the `quick_research` MCP tool from the `ai-tools-mcp` server to investigate:

$ARGUMENTS

This uses Gemini Flash with Google Search grounding and a concise-answer prompt. Return the answer directly with citations preserved verbatim. If the question turns out to need multi-source synthesis or cross-referencing, suggest the user re-run with `/ai-tools-mcp:deep-research` instead.
