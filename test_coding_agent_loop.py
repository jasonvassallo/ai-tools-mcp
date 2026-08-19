#!/usr/bin/env python3
"""Control-flow tests for coding_agent.loop (no docker, no ollama).
Run:  uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q
"""
from __future__ import annotations

import unittest

from coding_agent.loop import ProgressTracker


class NoProgressDetector(unittest.TestCase):
    """Spec §7: stalls after N turns with no walk-hash change, no
    (cmd, exit) change, and no write_file. Deliberately NOT output-hash
    based (timestamps would flip it every turn)."""

    def test_stalls_after_n_identical_turns(self):
        t = ProgressTracker(no_progress_turns=3)
        for _ in range(3):
            t.observe(tree_hash="H", last_cmd=("pytest", 1), wrote_file=False)
        self.assertTrue(t.stalled)

    def test_hash_change_is_progress(self):
        t = ProgressTracker(no_progress_turns=3)
        t.observe(tree_hash="H1", last_cmd=None, wrote_file=False)
        t.observe(tree_hash="H1", last_cmd=None, wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H2", last_cmd=None, wrote_file=False))
        self.assertFalse(t.stalled)

    def test_exit_flip_same_command_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=("pytest", 1), wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=("pytest", 0), wrote_file=False))

    def test_different_command_same_exit_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=("pytest a", 0), wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=("pytest b", 0), wrote_file=False))

    def test_write_file_alone_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=None, wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=None, wrote_file=True))
        self.assertFalse(t.stalled)


if __name__ == "__main__":
    unittest.main()
