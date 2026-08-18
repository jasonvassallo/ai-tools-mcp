---
description: Delegate a task to a local Ollama model (on-device, private)
argument-hint: <task, e.g. "summarize this diff: ...">
---

Use the `local_delegate` MCP tool to run this task on the local Ollama model:

$ARGUMENTS

Guidance:
- Include any needed file content inline in the prompt — the server never reads files.
- Default model is gemma4:12b-nvfp4 (stronger on short mechanical work); pass model=gemma4:31b-nvfp4
  for review and long-context code work (MBP only; there are no -256k / per-context tags any
  more — every call runs at the serving host's window), and keep_alive="0"
  to unload afterward.
- Pass think=true only for reasoning-heavy asks (slower).
- For long jobs pass background=true, then poll with the `local_delegate_result` tool.
- Output quality is below frontier models — treat results as a draft to verify, not a final answer.
