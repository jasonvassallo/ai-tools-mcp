---
description: Run a Perplexity Agent API Search-as-Code task (hosted sandbox agent, bulk/enumerable research, synchronous, minutes)
argument-hint: <bulk-research-task>
---

Use the `agent_research` MCP tool from the `ai-tools-mcp` server to investigate:

$ARGUMENTS

This tool is for bulk/enumerable research ("for each of these N items, find X"), computation over search results, or structured-dataset answers. Before calling it, restate the task so the items to enumerate and the fields to resolve per item are explicit. If the task is actually a single research question, use `deep_research` instead and say why.

Expect the call to take one to several minutes. For large fan-outs (dozens of items or more), pass `background: true` instead and poll the `agent_research_result` tool with the returned response_id every ~30 seconds.

Present the agent's answer with citations preserved verbatim, report the cost figure from the metadata footer, and surface any sandbox execution warnings rather than hiding them.
