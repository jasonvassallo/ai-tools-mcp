---
description: Save the current conversation as a named session (~/.claude/sessions/<uuid>.json)
argument-hint: <session-name>
---

Use the `save_session` MCP tool from the `ai-tools-mcp` server with:
- `name="$ARGUMENTS"` (use "Untitled" if empty)
- `messages` = the current conversation history as an array of `{role, content}` objects

The server redacts secret-shape strings before writing to disk. Return the new `session_id` to the user so they can reference it later via `/ai-tools-mcp:load-session <id>`.
