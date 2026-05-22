---
description: List all saved conversation sessions (most recent first)
---

Use the `list_sessions` MCP tool from the `ai-tools-mcp` server. Present the returned table as-is — it is already formatted as Markdown with session id, name, message count, and last-modified timestamp.

If the list is empty, tell the user there are no saved sessions and remind them they can save the current one with `/ai-tools-mcp:save-session <name>`.
