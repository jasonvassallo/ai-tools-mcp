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
  more — every call runs at the serving host's window).
- Leave keep_alive alone for the gemma4 tags: they are kept WARM on their hosts by design, and
  keep_alive="0" would unload the 31B after every review and force a cold reload next time.
  Only qwen tags default to keep_alive="0" (repeat-call contamination mitigation), and the
  server applies that automatically — you don't need to pass it.
- Keep think=false for the gemma4 tags. With thinking on they spend the generation in
  `message.thinking`, and this tool deliberately discards that field and returns
  "Error: Ollama returned no content" — so a think=true review against gemma4:31b yields
  no answer. Only enable think=true on a model whose content you have confirmed survives it.
- For long jobs pass background=true, then poll with the `local_delegate_result` tool.
- Output quality is below frontier models — treat results as a draft to verify, not a final answer.
