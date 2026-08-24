"""Repo-root pytest configuration.

Registers marks used by the `coding_agent` test suite so pytest recognises
them whether or not the plugin backing a mark is actually installed.

`timeout` backs `@pytest.mark.timeout(seconds)`, applied to a handful of
tests in test_coding_agent_security.py, test_coding_agent_tools.py and
test_coding_agent_walk.py whose ONLY regression signal is a HUNG process
rather than a red assertion (a FIFO opened without O_NONBLOCK blocks
forever with no writer ever attached; see those tests' docstrings for the
mechanism each one pins). The mark is enforced only when `pytest-timeout` is
installed (`uv run --with pytest-timeout ...`, or add it to any `--with
pytest ...` invocation) — this repo has no `pyproject.toml`, deliberately
(dependencies are declared per-invocation for the test runner, exactly like
`--with pathspec`), so there is nowhere else to register it. Without the
plugin, `@pytest.mark.timeout(...)` still parses because it is registered
here; it just does not enforce anything, and every test still runs — the
suite never hard-requires the plugin.
"""

from __future__ import annotations

from typing import Any


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "timeout(seconds): fail this test if it runs longer than `seconds` "
        "wall-clock (requires the pytest-timeout plugin; a no-op without it, "
        "not an error).",
    )
