#!/usr/bin/env python3
"""Unit tests for coding_agent.walk and coding_agent.basetree (no docker).

The basetree cases DO run git, but only ever as `git -C <repo>` against a
throwaway repository the test itself created — never with a worktree as cwd
or gitdir (spec §6.5).

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from coding_agent.basetree import make_ignore, read_base_tree
from coding_agent.walk import snapshot_tree, tree_hash, unified_diff


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

    def test_a_readable_tree_reports_nothing_unreadable(self):
        """`unreadable` is the gate's "these paths exist and I could not read
        them" channel. On an ordinary tree it must be EMPTY, or the signal is
        noise and a reviewer learns to skip it."""
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.unreadable, ())
        self.assertEqual(sorted(snap.entries), ["a.py", "sub/b.txt"])

    def test_special_files_are_skipped_without_being_called_unreadable(self):
        """A FIFO/socket/device is DELIBERATELY not in the diff (spec §6.5) —
        it is not a regular file and there is nothing to show. Reporting it as
        unreadable would say "a change is hidden here" when none is."""
        os.mkfifo(self.root / "trap.fifo")
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertNotIn("trap.fifo", snap.entries)
        self.assertEqual(snap.unreadable, ())


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
        snap = snapshot_tree(str(link), lambda p: False)
        self.assertEqual(snap.entries, {})
        # "Loud" is now literal: the root names ITSELF as unreadable, so the
        # gate can tell a refused root from a genuinely empty worktree instead
        # of inferring it from a wall of deletions.
        self.assertEqual([u.path for u in snap.unreadable], ["."])


def _mkrepo(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "t@e.com"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "T"], check=True)


def _commit(root: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", message], check=True)


class BaseTreeGitArgvIsAnchoredToTheRealRepo(unittest.TestCase):
    """The security spine, at `basetree.py`'s door.

    `sandbox.py` has an argv-level pin on every git call it makes. `basetree`
    is the OTHER module that runs git, and it had none — the shape was
    verified by hand during the build and nothing held it there. That matters
    because the failure this prevents is not a crash: pointing git at a
    directory the model controls is how two adversarial passes reached host
    RCE, and `git -C <worktree>` differs from `git -C <repo>` by one argument.

    A `cwd=`, an ambient `GIT_DIR`, or an inherited `PATH` would each
    reintroduce it without touching the argv, so all three are asserted too.
    """

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        _mkrepo(self.repo)
        (self.repo / "a.py").write_text("x = 1\n")
        (self.repo / ".gitignore").write_text("*.log\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)
        _commit(self.repo, "base")

    def _record(self):
        real = subprocess.run
        seen: list[tuple[list[str], dict]] = []

        def recorder(argv, **kwargs):
            seen.append((list(argv), dict(kwargs)))
            return real(argv, **kwargs)

        with unittest.mock.patch("subprocess.run", side_effect=recorder):
            base = read_base_tree(str(self.repo), "HEAD")
        return base, seen

    def test_every_git_call_is_git_dash_C_the_real_repo(self):
        base, seen = self._record()
        self.assertIn("a.py", base.entries)  # the read really happened
        # ls-tree + one cat-file per blob + rev-parse for info/exclude
        self.assertGreaterEqual(len(seen), 3)
        for argv, _ in seen:
            self.assertEqual(argv[:3], ["git", "-C", str(self.repo)], argv)

    def test_no_git_call_carries_a_cwd_or_an_inheritable_environment(self):
        _, seen = self._record()
        for argv, kwargs in seen:
            self.assertIsNone(kwargs.get("cwd"), f"git ran with a cwd: {argv}")
            env = kwargs.get("env")
            self.assertIsInstance(env, dict, f"git inherited the environment: {argv}")
            assert isinstance(env, dict)
            # An ambient GIT_DIR/GIT_WORK_TREE/GIT_COMMON_DIR would repoint a
            # `-C`-anchored call at the worktree without changing the argv.
            for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
                self.assertNotIn(var, env, argv)
            # No PATH: git resolves through os.defpath, so an ambient PATH
            # cannot substitute a different binary.
            self.assertNotIn("PATH", env, argv)
            # And the read must be deterministic: a user's global config —
            # core.excludesFile above all — must not change what the human
            # sees in the review diff.
            self.assertEqual(env.get("GIT_CONFIG_NOSYSTEM"), "1", argv)
            self.assertNotIn("HOME", env, argv)

    def test_the_env_REPLACES_rather_than_extends_the_ambient_one(self):
        """`env={**_GIT_ENV}` and `env={**os.environ, **_GIT_ENV}` look alike
        and are not: only the first keeps an ambient GIT_DIR out."""
        with unittest.mock.patch.dict(
            os.environ,
            {"GIT_DIR": "/tmp/attacker", "GIT_WORK_TREE": "/tmp/attacker"},
            clear=False,
        ):
            base, seen = self._record()
        self.assertIn("a.py", base.entries)
        for argv, kwargs in seen:
            self.assertNotIn("GIT_DIR", kwargs["env"], argv)
            self.assertNotIn("GIT_WORK_TREE", kwargs["env"], argv)


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


class ExecutableBitIsRecordedByTheWalk(unittest.TestCase):
    """The walk half of mode visibility. `chmod +x` changes no bytes, so
    unless the walk carries the bit there is nothing downstream can render.

    Only the OWNER execute bit is read, which is exactly what git records
    (`ce_mode_from_stat`: `mode & 0100 ? 100755 : 100644`). Tracking group or
    other would report a mode change for a file git itself calls unchanged.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "plain.txt").write_text("hi\n")
        (self.root / "tool.sh").write_text("#!/bin/sh\n")
        os.chmod(self.root / "tool.sh", 0o755)
        os.symlink("plain.txt", self.root / "link")

    def test_modes_match_what_git_would_store(self):
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["plain.txt"].mode, "100644")
        self.assertFalse(snap.entries["plain.txt"].executable)
        self.assertEqual(snap.entries["tool.sh"].mode, "100755")
        self.assertTrue(snap.entries["tool.sh"].executable)
        self.assertEqual(snap.entries["link"].mode, "120000")

    def test_a_group_only_execute_bit_is_not_reported_as_executable(self):
        os.chmod(self.root / "plain.txt", 0o644 | 0o010)
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["plain.txt"].mode, "100644")

    def test_the_no_progress_hash_moves_when_a_file_becomes_executable(self):
        """§7's stuck-detector must see the same changes the diff does, or a
        turn that only flips a mode bit reads as no progress."""
        before = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        os.chmod(self.root / "plain.txt", 0o755)
        after = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(after.entries["plain.txt"].data, b"hi\n")  # bytes unchanged
        self.assertNotEqual(before, tree_hash(after))


class ModeIsConsistentAcrossBothSides(unittest.TestCase):
    """The seam that makes the mode fix safe rather than noisy.

    A walk that reports the execute bit against a base tree that does not
    would call every committed 100755 file a mode change on turn one. Only a
    REAL `ls-tree` read compared against a REAL checkout proves the two
    encodings agree; asserting them against each other proves nothing.

    The repository and the walked directory are SEPARATE, as they are in
    production: git only ever runs against `self.repo`, and only the copy is
    ever walked or modified (§6.5 — no host-side git in the worktree, ever).
    """

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _mkrepo(self.repo)
        (self.repo / "plain.txt").write_text("hi\n")
        (self.repo / "tool.sh").write_text("#!/bin/sh\necho hi\n")
        os.chmod(self.repo / "tool.sh", 0o755)
        os.symlink("plain.txt", self.repo / "link")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)
        _commit(self.repo, "base")
        self.base = read_base_tree(str(self.repo), "HEAD")
        # What the sandbox is handed: the same content, no repo metadata.
        # copytree's copy2 preserves the mode bits, which is the point.
        self.work = Path(tempfile.mkdtemp()) / "work"
        shutil.copytree(
            self.repo, self.work, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )

    def _diff(self):
        snap = snapshot_tree(str(self.work), make_ignore(self.base))
        return unified_diff(
            self.base, snap, max_bytes=1 << 20, spill_dir=tempfile.mkdtemp()
        )

    def test_the_base_tree_carries_gits_own_mode(self):
        self.assertEqual(self.base.entries["tool.sh"].mode, "100755")
        self.assertEqual(self.base.entries["plain.txt"].mode, "100644")
        self.assertEqual(self.base.entries["link"].mode, "120000")

    def test_an_untouched_checkout_produces_no_diff_at_all(self):
        d = self._diff()
        self.assertEqual(d.changed_files, [])
        self.assertEqual(d.text, "")

    def test_chmod_plus_x_is_the_only_thing_reported(self):
        os.chmod(self.work / "plain.txt", 0o755)
        d = self._diff()
        self.assertEqual(d.changed_files, ["plain.txt"])
        self.assertIn("old mode 100644", d.text)
        self.assertIn("new mode 100755", d.text)
        self.assertNotIn("@@", d.text)  # no content hunk: the bytes are identical
        self.assertNotIn("tool.sh", d.text)  # already 100755 on both sides

    def test_an_edit_and_a_mode_change_are_both_reported(self):
        (self.work / "plain.txt").write_text("hi\nthere\n")
        os.chmod(self.work / "plain.txt", 0o755)
        d = self._diff()
        self.assertEqual(d.changed_files, ["plain.txt"])
        self.assertIn("old mode 100644", d.text)
        self.assertIn("new mode 100755", d.text)
        self.assertIn("+there", d.text)


if __name__ == "__main__":
    unittest.main()
