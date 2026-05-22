---
description: Run a fast Perplexity Sonar query (concise, single-source citations, cheaper than deep_research)
argument-hint: <research-query>
---

Use the `quick_research` MCP tool from the `ai-tools-mcp` server to investigate:

$ARGUMENTS

This uses the smaller/faster Sonar model rather than Sonar Pro. Return the answer directly with citations preserved verbatim. If the question turns out to need multi-source synthesis or cross-referencing, suggest the user re-run with `/ai-tools-mcp:deep-research` instead.
