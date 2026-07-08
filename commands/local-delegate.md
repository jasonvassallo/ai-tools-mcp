---
description: Delegate a task to the local Ollama qwen3.6 model (on-device, private)
argument-hint: <task, e.g. "summarize this diff: ...">
---

Use the `local_delegate` MCP tool to run this task on the local Ollama model:

$ARGUMENTS

Guidance:
- Include any needed file content inline in the prompt — the server never reads files.
- Default model is the base qwen3.6 coding tag; pass model=...-256k only for genuinely huge inputs, and keep_alive="0" to unload afterward.
- Pass think=true only for reasoning-heavy asks (slower).
- For long jobs pass background=true, then poll with the `local_delegate_result` tool.
- Output quality is below frontier models — treat results as a draft to verify, not a final answer.
