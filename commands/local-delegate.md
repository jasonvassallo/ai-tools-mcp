---
description: Delegate a task to a local Ollama model (on-device, private)
argument-hint: <task, e.g. "summarize this diff: ...">
---

Use the `local_delegate` MCP tool to run this task on the local Ollama model:

$ARGUMENTS

Guidance:
- Include any needed file content inline in the prompt — the server never reads files.
- Default model is gemma4:12b-nvfp4 (stronger on short mechanical work); pass a qwen3.6 tag for
  long-context code work, model=...-256k only for genuinely huge inputs. keep_alive can be
  omitted on qwen tags — they default to "0", and any "0" (defaulted or explicit) also evicts
  the runner before the call, which is the contamination protection; only pass a non-zero
  keep_alive when you deliberately want the model kept warm (that skips the protection).
- Pass think=true only for reasoning-heavy asks (slower).
- For long jobs pass background=true, then poll with the `local_delegate_result` tool.
- Output quality is below frontier models — treat results as a draft to verify, not a final answer.
