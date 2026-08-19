#!/usr/bin/env python3
"""Unit tests for coding_agent.walk and coding_agent.basetree (no docker).

The basetree cases DO run git, but only ever as `git -C <repo>` against a
throwaway repository the test itself created — never with a worktree as cwd
or gitdir (spec §6.5).

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent.basetree import make_ignore, read_base_tree
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


def _mkrepo(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "t@e.com"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "T"], check=True)


def _commit(root: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", message], check=True)


class BaseTreeRead(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _mkrepo(self.repo)
        (self.repo / ".gitignore").write_text("*.secret\nignored-dir/\n")
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg" / ".gitignore").write_text("*.log\n")
        (self.repo / "app.py").write_text("print(1)\n")
        (self.repo / "app.secret").write_text("tracked-despite-pattern\n")  # force-add
        (self.repo / "ignored-dir").mkdir()
        (self.repo / "ignored-dir" / "keep.py").write_text("kept\n")  # force-added
        os.symlink("app.py", self.repo / "link")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A", "-f"], check=True)
        _commit(self.repo, "base")

    def test_reads_files_symlinks_and_tracked_set(self):
        base = read_base_tree(str(self.repo), "HEAD")
        self.assertEqual(base.entries["app.py"].data, b"print(1)\n")
        self.assertEqual(base.entries["link"].kind, "symlink")
        self.assertEqual(base.entries["link"].data, b"app.py")
        self.assertIn("app.secret", base.tracked)
        self.assertIn("ignored-dir/keep.py", base.tracked)

    def test_ignore_is_tracked_aware_so_gate_cannot_be_blinded(self):
        """Pass 4: uniform ignore hid edits to TRACKED files matching a base
        pattern. git's rule: ignore never applies to tracked files."""
        base = read_base_tree(str(self.repo), "HEAD")
        ign = make_ignore(base)
        self.assertFalse(ign("app.secret"))  # tracked -> shown
        self.assertFalse(ign("ignored-dir/keep.py"))  # tracked -> shown
        self.assertFalse(ign("ignored-dir/"))  # contains a tracked file -> walked
        self.assertTrue(ign("new.secret"))  # untracked + matches -> ignored
        self.assertTrue(ign("ignored-dir/new.py"))  # untracked under ignored dir
        self.assertTrue(ign("pkg/debug.log"))  # NESTED .gitignore honoured
        self.assertFalse(ign("pkg/main.py"))

    def test_repo_local_info_exclude_is_read_from_the_named_repo(self):
        """`rev-parse --git-path` answers RELATIVE to the repo, so the answer
        must be resolved against `repo`, not against whatever directory this
        process happens to be sitting in. Resolving it against the process cwd
        reads some OTHER repository's exclude file — a stranger's patterns
        silently hiding paths is the under-showing direction.
        """
        (self.repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (self.repo / ".git" / "info" / "exclude").write_text("*.scratch\n")
        ign = make_ignore(read_base_tree(str(self.repo), "HEAD"))
        self.assertTrue(ign("notes.scratch"))
        self.assertFalse(ign("notes.py"))

    def test_a_submodule_gitlink_is_skipped_not_fetched_as_a_blob(self):
        """`ls-tree -r` still lists mode 160000 gitlinks. Their sha names a
        COMMIT, so asking for it as a blob fails; the read must skip them
        rather than abort the whole base-tree read.
        """
        head = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{head},vendor/sub",
            ],
            check=True,
        )
        _commit(self.repo, "gitlink")
        base = read_base_tree(str(self.repo), "HEAD")
        self.assertNotIn("vendor/sub", base.entries)
        self.assertIn("app.py", base.entries)


class IgnorePrecedence(unittest.TestCase):
    """git composes ignore files by precedence, not by first match.

    `$GIT_DIR/info/exclude` is weakest, then ignore files shallowest-first,
    and within one file the LAST matching pattern wins. Stopping at the first
    spec that matches discards every re-inclusion, so a file the user's own
    git would show as untracked never reaches the diff. The model reads the
    repo's ignore files like anyone else, so a blind spot spelled out in them
    is one it can aim a new file at.
    """

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _mkrepo(self.repo)
        (self.repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (self.repo / ".git" / "info" / "exclude").write_text("*.md\n")
        (self.repo / ".gitignore").write_text("*.log\n!README.md\n")
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg" / ".gitignore").write_text("!keep.log\n")
        (self.repo / "app.py").write_text("print(1)\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A", "-f"], check=True)
        _commit(self.repo, "base")

    def test_deeper_and_later_patterns_override_shallower_ones(self):
        ign = make_ignore(read_base_tree(str(self.repo), "HEAD"))
        self.assertTrue(ign("debug.log"))  # root pattern, unopposed
        self.assertTrue(ign("pkg/other.log"))  # root pattern reaches subdirs
        self.assertFalse(ign("pkg/keep.log"))  # nested negation wins
        self.assertTrue(ign("notes.md"))  # info/exclude applies
        self.assertFalse(ign("README.md"))  # .gitignore outranks info/exclude


class IgnoreComposesWithWalk(unittest.TestCase):
    """The seam between Task 2 and Task 3, which neither proves alone.

    `walk.py` PRUNES a directory whose query returns True — it never descends,
    so nothing beneath it reaches the snapshot at all. A tracked file inside an
    ignored directory is therefore invisible unless `make_ignore` answers False
    for the DIRECTORY query too. Unit-testing the predicate proves the answer;
    only walking a real tree proves the two halves actually compose.
    """

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _mkrepo(self.repo)
        (self.repo / ".gitignore").write_text("ignored-dir/\n")
        (self.repo / "app.py").write_text("print(1)\n")
        (self.repo / "ignored-dir").mkdir()
        (self.repo / "ignored-dir" / "keep.py").write_text("kept\n")  # force-added
        subprocess.run(["git", "-C", str(self.repo), "add", "-A", "-f"], check=True)
        _commit(self.repo, "base")
        # What the sandbox sees: the same checkout, plus the model's edits.
        self.work = Path(tempfile.mkdtemp())
        (self.work / ".gitignore").write_text("ignored-dir/\n")
        (self.work / "app.py").write_text("print(1)\n")
        (self.work / "ignored-dir").mkdir()
        (self.work / "ignored-dir" / "keep.py").write_text("BACKDOOR\n")
        (self.work / "ignored-dir" / "scratch.py").write_text("noise\n")

    def test_edit_to_a_tracked_file_under_an_ignored_dir_reaches_the_snapshot(self):
        base = read_base_tree(str(self.repo), "HEAD")
        snap = snapshot_tree(str(self.work), make_ignore(base))
        self.assertIn("ignored-dir/keep.py", snap.entries)
        self.assertEqual(snap.entries["ignored-dir/keep.py"].data, b"BACKDOOR\n")
        # Over-showing stays bounded: a genuinely NEW path under the ignored
        # directory is still ignored, so walking that directory is not a
        # blanket un-ignore.
        self.assertNotIn("ignored-dir/scratch.py", snap.entries)


if __name__ == "__main__":
    unittest.main()
