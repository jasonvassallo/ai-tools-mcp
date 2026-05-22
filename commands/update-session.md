---
description: Rename a saved session (bumps last_modified)
argument-hint: <session-uuid> <new-name>
---

Parse `$ARGUMENTS` as two parts: the first whitespace-separated token is the session UUID, the rest is the new name.

Use the `update_session` MCP tool from the `ai-tools-mcp` server with `session_id=<uuid>` and `name=<new-name>`. The session's `last_modified` timestamp bumps automatically.
