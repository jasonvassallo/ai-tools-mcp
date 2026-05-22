---
description: Kick off a long-running Google Gemini Deep Research task (returns an interaction_id to poll)
argument-hint: <research-query> [--mode fast|max]
---

Use the `gemini_deep_research_start` MCP tool from the `ai-tools-mcp` server to start an asynchronous deep-research task.

Query: $ARGUMENTS

Default to `mode="fast"` unless the user explicitly asks for `max`. Return the `interaction_id` to the user and remind them to poll with `/ai-tools-mcp:gemini-result <id>` in ~30 seconds. These tasks take several minutes (up to 60 max).
