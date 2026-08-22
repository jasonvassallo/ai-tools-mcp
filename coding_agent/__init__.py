"""Sandboxed agentic coding on a local model.

The public surface `mcp_server.py` imports. Everything else in the package is
internal: the byte-walk, the base-tree reader, the docker/git lifecycle and
the four model-facing tools.
"""

from __future__ import annotations

from .loop import AgentResult, StopReason, run_coding_agent

__all__ = ["AgentResult", "StopReason", "run_coding_agent"]
