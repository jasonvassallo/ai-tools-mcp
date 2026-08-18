#!/usr/bin/env python3
"""Unit tests for coding_agent.walk (no docker, no git).
Run:  uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from coding_agent.walk import snapshot_tree, tree_hash


class SnapshotBasics(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "a.py").write_text("x = 1\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.txt").write_text("hi\n")

    def test_regular_files_are_read(self):
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["a.py"].data, b"x = 1\n")
        self.assertEqual(snap.entries["sub/b.txt"].data, b"hi\n")

    def test_ignore_callback_gates_new_paths(self):
        snap = snapshot_tree(str(self.root), lambda p: p.endswith(".txt"))
        self.assertIn("a.py", snap.entries)
        self.assertNotIn("sub/b.txt", snap.entries)

    def test_hash_is_deterministic_and_content_sensitive(self):
        h1 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        h2 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertEqual(h1, h2)
        (self.root / "a.py").write_text("x = 2\n")
        self.assertNotEqual(
            h1, tree_hash(snapshot_tree(str(self.root), lambda p: False))
        )


if __name__ == "__main__":
    unittest.main()
