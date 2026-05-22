---
description: Load a saved conversation session by UUID
argument-hint: <session-uuid>
---

Use the `load_session` MCP tool from the `ai-tools-mcp` server with `session_id=$ARGUMENTS`.

If the session is found, display its name, timestamps, metadata (if any), and conversation history. If the id is invalid or the session doesn't exist, the tool will return a clean error — surface it to the user and suggest `/ai-tools-mcp:sessions` to see available ids.
