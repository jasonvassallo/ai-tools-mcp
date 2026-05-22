---
description: Permanently delete a saved session by UUID (no undo)
argument-hint: <session-uuid>
---

Before calling the tool, confirm with the user that they want to permanently delete session `$ARGUMENTS`. There is no undo — the file is unlinked from disk.

If confirmed, use the `delete_session` MCP tool from the `ai-tools-mcp` server with `session_id=$ARGUMENTS`.
