"""Orchestration loop for coding_agent (spec §4, §7).

The loop runs on the HOST. Tool calls dispatch to tools.py; only
run_command crosses into the container. Stop conditions are mechanical
and the model cannot extend its own budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class StopReason(str, Enum):
    completed = "completed"
    max_turns = "max_turns"
    max_seconds = "max_seconds"
    no_progress = "no_progress"
    error = "error"


@dataclass
class Budget:
    max_turns: int
    max_seconds: float
    started: float = field(default_factory=time.monotonic)

    def exceeded_time(self) -> bool:
        return (time.monotonic() - self.started) >= self.max_seconds


class ProgressTracker:
    """Progress == walk-hash changed OR (cmd, exit) pair changed OR write_file
    was called this turn. Not adversarially robust by design (spec §5.1) —
    the hard caps are the real budget; this catches *stuck*, not *malicious*.

    Deliberately NOT output-hash based: command output routinely contains
    timestamps, durations, and paths that differ every run, which would make
    every turn look like progress and defeat the detector entirely.

    A turn is measured against the immediately preceding turn. The very
    first observation has no preceding turn to compare against, so it
    counts as showing no change — spec §7's "N turns with no change" is
    read literally, not as "N turns after a baseline is established".
    """

    def __init__(self, no_progress_turns: int = 5) -> None:
        self._n = no_progress_turns
        self._idle = 0
        self._last_hash: str | None = None
        self._last_cmd: tuple[str, int] | None = None

    def observe(
        self, *, tree_hash: str, last_cmd: tuple[str, int] | None, wrote_file: bool
    ) -> bool:
        if self._last_hash is None:
            progressed = False  # no prior turn to compare against yet
        else:
            progressed = (
                wrote_file
                or tree_hash != self._last_hash
                or (last_cmd is not None and last_cmd != self._last_cmd)
            )
        self._last_hash = tree_hash
        if last_cmd is not None:
            self._last_cmd = last_cmd
        self._idle = 0 if progressed else self._idle + 1
        return progressed

    @property
    def stalled(self) -> bool:
        return self._idle >= self._n
