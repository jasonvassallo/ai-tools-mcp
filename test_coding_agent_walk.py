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


def _open_fd_count() -> int:
    """Number of descriptors this process holds open (macOS/Linux /dev/fd)."""
    return len(os.listdir("/dev/fd"))


class DescriptorHygiene(unittest.TestCase):
    """The race fix walks through OPEN directory descriptors instead of
    re-resolving paths, so descriptor accounting is now a correctness property
    of the walk, not an implementation detail: a leak exhausts the host
    process, and holding one descriptor per QUEUED directory would blow the
    open-file limit on a wide tree (macOS ships a 256 soft limit by default).
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_repeated_walks_leak_no_descriptors(self):
        (self.root / "a.py").write_text("x = 1\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.txt").write_text("hi\n")
        (self.root / "sub" / "deeper").mkdir()
        (self.root / "sub" / "deeper" / "c.txt").write_text("c\n")
        (self.root / "ignored").mkdir()
        (self.root / "ignored" / "d.txt").write_text("d\n")
        os.symlink("/etc/passwd", self.root / "lnk")
        os.symlink("sub", self.root / "dirlnk")
        os.mkfifo(self.root / "trap.fifo")

        def ignore(p: str) -> bool:
            return p == "ignored/"

        snapshot_tree(str(self.root), ignore)  # warm any lazy imports
        before = _open_fd_count()
        for _ in range(50):
            snapshot_tree(str(self.root), ignore)
        self.assertEqual(_open_fd_count(), before)

    def test_descriptor_use_is_bounded_by_depth_not_breadth(self):
        for i in range(200):
            d = self.root / f"d{i:03d}"
            d.mkdir()
            (d / "f.txt").write_text("x\n")
        peak = 0

        def ignore(p: str) -> bool:
            nonlocal peak
            peak = max(peak, _open_fd_count())
            return False

        baseline = _open_fd_count()
        snap = snapshot_tree(str(self.root), ignore)
        self.assertEqual(len(snap.entries), 200)
        # Depth here is 2, so a handful of descriptors. A walk that held one
        # per queued directory would sit ~200 above the baseline.
        self.assertLess(peak - baseline, 10, f"peak={peak} baseline={baseline}")

    def test_no_descriptors_leak_when_the_ignore_callable_raises(self):
        """Task 3 owns `is_ignored`; a bug in it must not leak the host
        process's descriptors. This is the path where the walk unwinds with
        frames still on the stack, so it exercises the outer `finally`.
        """

        class Boom(Exception):
            pass

        cur = self.root
        for i in range(40):
            cur = cur / f"lvl{i:02d}"
            cur.mkdir()
            (cur / "f.txt").write_text("x\n")

        def raise_deep(p: str) -> bool:
            if p.count("/") >= 35:  # ~36 directory descriptors are open here
                raise Boom(p)
            return False

        snapshot_tree(str(self.root), lambda p: False)  # warm up
        before = _open_fd_count()
        for _ in range(50):
            with self.assertRaises(Boom):
                snapshot_tree(str(self.root), raise_deep)
            with self.assertRaises(Boom):
                snapshot_tree(str(self.root), lambda p: (_ for _ in ()).throw(Boom(p)))
        self.assertEqual(_open_fd_count(), before)

    def test_deep_tree_is_walked_to_the_bottom(self):
        cur = self.root
        for i in range(60):
            cur = cur / f"lvl{i:02d}"
            cur.mkdir()
        (cur / "bottom.txt").write_text("deep\n")
        before = _open_fd_count()
        snap = snapshot_tree(str(self.root), lambda p: False)
        expected = "/".join(f"lvl{i:02d}" for i in range(60)) + "/bottom.txt"
        self.assertEqual(snap.entries[expected].data, b"deep\n")
        self.assertEqual(_open_fd_count(), before)


class RootResolution(unittest.TestCase):
    def test_root_that_is_itself_a_symlink_is_not_followed(self):
        """Deliberate, documented consequence of opening the root O_NOFOLLOW.

        The root is the host-chosen worktree path — a container cannot rename
        its own mount point — so following it would be safe, but refusing is
        uniform and its failure mode is loud: an empty snapshot makes every
        tracked path surface as a deletion in the two-sided diff rather than
        silently reading somewhere else. Only the FINAL component is affected,
        so an ordinary path under a symlinked ancestor (macOS /tmp, /var) is
        unaffected.
        """
        tmp = Path(tempfile.mkdtemp())
        real = tmp / "real"
        real.mkdir()
        (real / "a.py").write_text("x = 1\n")
        link = tmp / "link"
        os.symlink(str(real), link)
        self.assertIn("a.py", snapshot_tree(str(real), lambda p: False).entries)
        self.assertEqual(snapshot_tree(str(link), lambda p: False).entries, {})


if __name__ == "__main__":
    unittest.main()
