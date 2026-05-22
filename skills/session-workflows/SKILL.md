---
name: session-workflows
description: Workflows for saving, loading, listing, renaming, and deleting Claude conversation sessions via the ai-tools-mcp server. Use when the user wants to checkpoint a conversation, resume a prior one, or manage their session history.
---

# Session Management Workflows

Sessions persist to `~/.claude/sessions/<uuid>.json` with the shape:

```json
{
  "session_id": "uuid",
  "name": "string",
  "created_at": "ISO-8601 UTC",
  "last_modified": "ISO-8601 UTC",
  "messages": [{"role": "user|assistant|system", "content": "..."}],
  "metadata": {}
}
```

Secret-shape strings (Google API keys, OAuth tokens, JWTs, PEM blocks) are redacted at the boundary in `save_session` and `update_session` — they do not land on disk in plaintext.

## Save the current conversation

1. Reconstruct the message history as `[{role, content}, ...]`. Include both user and assistant turns; skip tool-result blocks unless they're load-bearing context.
2. Pick a short descriptive `name` (the user may supply one — use it; otherwise propose one based on the topic).
3. Call `save_session` with `name`, `messages`, and optional `metadata` (e.g., `{"project": "X", "purpose": "Y"}`).
4. Return the new `session_id` so the user can reference it.

## Resume a saved conversation

1. If the user knows the id, call `load_session` directly.
2. If not, call `list_sessions` first, show them the table, and ask which to resume.
3. After loading, summarize what the session covered before continuing — the user may have stepped away for days.

## Rename a session

Use `update_session` with `session_id` and the new `name`. The `last_modified` timestamp bumps automatically. To explicitly clear a name, pass `name=""` (empty string, not omitted).

## Delete a session

**Always confirm before calling `delete_session`** — there is no undo, the file is unlinked from disk. The lockfile (`<uuid>.lock`) is left behind intentionally to avoid races with concurrent updates; this is benign.

## Concurrency notes

`update_session` and `delete_session` serialize via a per-session POSIX flock advisory lockfile at `~/.claude/sessions/<uuid>.lock`. Non-cooperating deleters (manual `rm`, foreign tools) can still slip past; `update_session` has a best-effort existence re-check inside the lock to catch this. If the user reports a "session was deleted concurrently during update" error, it means a non-cooperating process beat the update — just retry with a fresh `load_session` first.

## Platform note

Session management requires POSIX (macOS/Linux). On Windows, `update_session` and `delete_session` raise `OSError` immediately because `fcntl` is unavailable. The other tools (`list_sessions`, `save_session`, `load_session`) work on any platform.
