---
description: Poll a Gemini Deep Research task by interaction_id (returns status or final report)
argument-hint: <interaction_id>
---

Use the `gemini_deep_research_result` MCP tool from the `ai-tools-mcp` server with `interaction_id=$ARGUMENTS`.

Interpret the response:
- `status="completed"` → present `output_text` with citations preserved, summarize `steps_summary` briefly
- `status="in_progress"` → tell the user to poll again in ~30 seconds
- `status="requires_action"` → tell the user the agent is awaiting input (collaborative planning); next step is theirs
- `status="failed"|"cancelled"|"incomplete"|"budget_exceeded"` → report the `error` field verbatim, suggest restarting with `/ai-tools-mcp:gemini-start`
