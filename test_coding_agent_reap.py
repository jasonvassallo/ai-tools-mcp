#!/usr/bin/env python3
"""Tests for coding_agent.reap (issue #79): the sweep that reaps spilled
diffs and crash-orphaned worktree parents from the host temp dir.

Every test here builds its own scratch directory with `tempfile.mkdtemp` and
tears it down with `addCleanup(shutil.rmtree, ..., ignore_errors=True)` —
`sweep()`'s default target is the REAL `tempfile.gettempdir()`, but every
call below passes an explicit `spill_dir` so nothing here ever touches it.

Run:  python3 -m unittest test_coding_agent_reap -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from coding_agent import reap as R
from coding_agent.sandbox import create_worktree, teardown_worktree

_HOUR = 60 * 60


def _write(path: str, content: bytes = b"x") -> None:
    with open(path, "wb") as fh:
        fh.write(content)


def _set_mtime(path: str, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def _diff_spill(
    directory: str, name: str, *, age_s: float, size: int, now: float
) -> str:
    path = os.path.join(directory, name)
    _write(path, b"d" * size)
    _set_mtime(path, now - age_s)
    return path


def _wt_parent(directory: str, name: str = "coding-agent-wt-abc123") -> str:
    path = os.path.join(directory, name)
    os.mkdir(path)
    return path


def _wt_child(parent: str, name: str = "wt-deadbeefcafe") -> str:
    path = os.path.join(parent, name)
    os.mkdir(path)
    return path


def _git_marker(child: str, gitdir: str | None) -> None:
    """Write `child`'s `.git` file. `gitdir=None` leaves it absent."""
    if gitdir is not None:
        with open(os.path.join(child, ".git"), "w", encoding="utf-8") as fh:
            fh.write(f"gitdir: {gitdir}\n")


class _ScratchCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="ca-reap-test-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.now = time.time()


# ---------------------------------------------------------------------------
# Diff spills: TTL
# ---------------------------------------------------------------------------


class DiffSpillTTL(_ScratchCase):
    def test_an_old_spill_is_removed(self):
        path = _diff_spill(
            self.dir,
            "coding-agent-diff-old.patch",
            age_s=25 * _HOUR,
            size=10,
            now=self.now,
        )
        stats = R.sweep(self.dir, now=self.now)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(stats["diffs_removed"], 1)
        self.assertEqual(stats["diff_bytes_freed"], 10)

    def test_a_young_spill_is_kept(self):
        path = _diff_spill(
            self.dir,
            "coding-agent-diff-new.patch",
            age_s=1 * _HOUR,
            size=10,
            now=self.now,
        )
        stats = R.sweep(self.dir, now=self.now)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(stats["diffs_removed"], 0)

    def test_the_boundary_is_approximately_ttl_seconds_old(self):
        # A few seconds of margin on each side of the TTL, not the exact
        # boundary — `os.utime` and the `now - mtime` subtraction go through
        # enough float rounding that asserting the literal edge would be
        # testing float precision, not the reaper.
        just_over = _diff_spill(
            self.dir,
            "coding-agent-diff-over.patch",
            age_s=R._TTL_SECONDS + 5,
            size=1,
            now=self.now,
        )
        just_under = _diff_spill(
            self.dir,
            "coding-agent-diff-under.patch",
            age_s=R._TTL_SECONDS - 5,
            size=1,
            now=self.now,
        )
        R.sweep(self.dir, now=self.now)
        self.assertFalse(os.path.exists(just_over))
        self.assertTrue(os.path.exists(just_under))

    def test_non_matching_entries_are_never_touched(self):
        """Wrong prefix, wrong suffix, and a directory that happens to be
        named like a spill — none of these are ours to remove."""
        wrong_prefix = os.path.join(self.dir, "some-other-diff-x.patch")
        wrong_suffix = os.path.join(self.dir, "coding-agent-diff-x.txt")
        a_directory = os.path.join(self.dir, "coding-agent-diff-dir.patch")
        for p in (wrong_prefix, wrong_suffix):
            _write(p)
            _set_mtime(p, self.now - 100 * _HOUR)
        os.mkdir(a_directory)
        _set_mtime(a_directory, self.now - 100 * _HOUR)

        stats = R.sweep(self.dir, now=self.now)

        self.assertTrue(os.path.exists(wrong_prefix))
        self.assertTrue(os.path.exists(wrong_suffix))
        self.assertTrue(os.path.exists(a_directory))
        self.assertEqual(stats["diffs_removed"], 0)


# ---------------------------------------------------------------------------
# Diff spills: count / byte caps, independent of the TTL
# ---------------------------------------------------------------------------


class DiffSpillCaps(_ScratchCase):
    def test_the_count_cap_removes_oldest_first(self):
        # All well inside the TTL, so only the count cap can explain removal.
        paths = [
            _diff_spill(
                self.dir,
                f"coding-agent-diff-{i}.patch",
                age_s=(10 - i) * 60,  # paths[0] is oldest
                size=1,
                now=self.now,
            )
            for i in range(5)
        ]
        stats = R.sweep(self.dir, now=self.now, max_count=3, max_bytes=10**9)
        self.assertEqual(stats["diffs_removed"], 2)
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(os.path.exists(paths[1]))
        for p in paths[2:]:
            self.assertTrue(os.path.exists(p), p)

    def test_the_byte_cap_removes_oldest_first(self):
        paths = [
            _diff_spill(
                self.dir,
                f"coding-agent-diff-{i}.patch",
                age_s=(10 - i) * 60,
                size=100,
                now=self.now,
            )
            for i in range(4)
        ]
        # 4 * 100 = 400 bytes; cap at 250 must drop the oldest two (200
        # freed) to land at 200 <= 250, not stop after just one.
        stats = R.sweep(self.dir, now=self.now, max_count=10**9, max_bytes=250)
        self.assertEqual(stats["diffs_removed"], 2)
        self.assertEqual(stats["diff_bytes_freed"], 200)
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(os.path.exists(paths[1]))
        self.assertTrue(os.path.exists(paths[2]))
        self.assertTrue(os.path.exists(paths[3]))

    def test_a_run_within_both_caps_removes_nothing(self):
        for i in range(3):
            _diff_spill(
                self.dir,
                f"coding-agent-diff-{i}.patch",
                age_s=60,
                size=10,
                now=self.now,
            )
        stats = R.sweep(self.dir, now=self.now, max_count=10, max_bytes=10**6)
        self.assertEqual(stats["diffs_removed"], 0)
        self.assertEqual(len(os.listdir(self.dir)), 3)


# ---------------------------------------------------------------------------
# Worktree parents: staleness
# ---------------------------------------------------------------------------


class WorktreeStaleness(_ScratchCase):
    """`_worktree_is_stale` in isolation — no TTL involved, just the shape
    check `sweep()` gates on liveness with."""

    def test_missing_git_marker_is_stale(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        _git_marker(child, None)
        self.assertTrue(R._worktree_is_stale(parent))

    def test_a_recorded_gitdir_that_no_longer_exists_is_stale(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        _git_marker(child, os.path.join(self.dir, "nonexistent-repo-worktrees-entry"))
        self.assertTrue(R._worktree_is_stale(parent))

    def test_a_recorded_gitdir_that_exists_is_NOT_stale(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        real_gitdir = os.path.join(self.dir, "looks-like-a-real-repo-worktrees-entry")
        os.mkdir(real_gitdir)
        _git_marker(child, real_gitdir)
        self.assertFalse(R._worktree_is_stale(parent))

    def test_an_empty_parent_is_stale(self):
        parent = _wt_parent(self.dir)
        self.assertTrue(R._worktree_is_stale(parent))

    def test_a_git_marker_that_is_itself_a_directory_is_NOT_stale(self):
        """Never a shape `sandbox.create_worktree` produces — a linked
        worktree's `.git` is always a file. Ambiguous, so kept."""
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        os.mkdir(os.path.join(child, ".git"))
        self.assertFalse(R._worktree_is_stale(parent))

    def test_unparseable_git_marker_content_is_NOT_stale(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        with open(os.path.join(child, ".git"), "w", encoding="utf-8") as fh:
            fh.write("not a gitdir line\n")
        self.assertFalse(R._worktree_is_stale(parent))

    def test_more_than_one_child_is_NOT_stale(self):
        """Never a shape this module's own writer produces either — don't
        guess at an unfamiliar layout."""
        parent = _wt_parent(self.dir)
        _wt_child(parent, "wt-one")
        _wt_child(parent, "wt-two")
        self.assertFalse(R._worktree_is_stale(parent))

    def test_a_child_not_named_wt_dash_is_NOT_stale(self):
        parent = _wt_parent(self.dir)
        os.mkdir(os.path.join(parent, "something-else"))
        self.assertFalse(R._worktree_is_stale(parent))


# ---------------------------------------------------------------------------
# Worktree parents: sweep() wiring (TTL + staleness together)
# ---------------------------------------------------------------------------


class WorktreeSweep(_ScratchCase):
    def test_a_stale_worktree_older_than_the_ttl_is_removed(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        _git_marker(child, None)  # stale: no .git marker
        _set_mtime(parent, self.now - 25 * _HOUR)

        stats = R.sweep(self.dir, now=self.now)

        self.assertFalse(os.path.lexists(parent))
        self.assertEqual(stats["worktrees_removed"], 1)

    def test_a_stale_worktree_younger_than_the_ttl_is_kept(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        _git_marker(child, None)
        _set_mtime(parent, self.now - 1 * _HOUR)

        stats = R.sweep(self.dir, now=self.now)

        self.assertTrue(os.path.lexists(parent))
        self.assertEqual(stats["worktrees_removed"], 0)

    def test_a_live_looking_worktree_older_than_the_ttl_is_kept(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        real_gitdir = os.path.join(self.dir, "real-gitdir-entry")
        os.mkdir(real_gitdir)
        _git_marker(child, real_gitdir)
        _set_mtime(parent, self.now - 100 * _HOUR)

        stats = R.sweep(self.dir, now=self.now)

        self.assertTrue(os.path.lexists(parent))
        self.assertEqual(stats["worktrees_removed"], 0)


# ---------------------------------------------------------------------------
# Real git integration: sandbox.create_worktree's actual on-disk shape
# ---------------------------------------------------------------------------


def _init_repo() -> str:
    repo = tempfile.mkdtemp(prefix="ca-reap-repo-")
    subprocess.run(["git", "-C", repo, "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "T"], check=True)
    with open(os.path.join(repo, "f"), "w", encoding="utf-8") as fh:
        fh.write("1\n")
    subprocess.run(["git", "-C", repo, "add", "f"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "b"], check=True)
    return repo


class RealWorktreeShape(unittest.TestCase):
    """`_worktree_is_stale`'s parsing is exercised against a real
    `git worktree add` layout, not just hand-built fixtures — the same
    reason `test_coding_agent_sandbox.py` builds real repos."""

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            shutil.rmtree(path, ignore_errors=True)

    def test_a_freshly_created_real_worktree_is_NOT_stale(self):
        wt = create_worktree(self.repo, "HEAD")
        self.strays.append(wt)
        parent = os.path.dirname(wt)
        self.assertFalse(R._worktree_is_stale(parent))
        teardown_worktree(self.repo, wt)

    def test_a_worktree_whose_registration_is_gone_is_reaped(self):
        """`git worktree remove` deletes BOTH the registration and the
        directory in one step (verified directly, not assumed), so it
        cannot produce the state this module reaps. What actually leaves a
        `.git` file pointing at nothing is the repo-side admin data
        (`.git/worktrees/<name>`) going away by some OTHER path — the repo
        relocated, that admin dir got corrupted or manually pruned — while
        the linked worktree directory under $TMPDIR is left exactly as
        `create_worktree` made it. Clause 2: "the recorded gitdir no longer
        exists"."""
        wt = create_worktree(self.repo, "HEAD")
        self.strays.append(wt)
        parent = os.path.dirname(wt)

        with open(os.path.join(wt, ".git"), encoding="utf-8") as fh:
            gitdir = fh.readline()[len("gitdir:") :].strip()
        self.assertTrue(
            os.path.isdir(gitdir), "test assumption: the real admin dir exists"
        )
        shutil.rmtree(gitdir)  # only the repo-side registration goes away
        self.assertTrue(
            os.path.exists(os.path.join(wt, ".git")),
            "the worktree directory itself must be untouched",
        )
        self.assertTrue(R._worktree_is_stale(parent))

        _set_mtime(parent, time.time() - 25 * _HOUR)
        stats = R.sweep(os.path.dirname(parent), now=time.time())

        self.assertFalse(os.path.lexists(parent))
        self.assertGreaterEqual(stats["worktrees_removed"], 1)


# ---------------------------------------------------------------------------
# Best-effort: never raises
# ---------------------------------------------------------------------------


class SweepNeverRaises(_ScratchCase):
    def test_a_nonexistent_target_dir_does_not_raise(self):
        stats = R.sweep(os.path.join(self.dir, "does-not-exist"), now=self.now)
        self.assertEqual(
            stats, {"diffs_removed": 0, "diff_bytes_freed": 0, "worktrees_removed": 0}
        )

    def test_a_target_that_is_a_file_not_a_directory_does_not_raise(self):
        target = os.path.join(self.dir, "not-a-dir")
        _write(target)
        stats = R.sweep(target, now=self.now)
        self.assertEqual(stats["diffs_removed"], 0)

    def test_an_unremovable_diff_is_skipped_not_raised(self):
        path = _diff_spill(
            self.dir,
            "coding-agent-diff-locked.patch",
            age_s=25 * _HOUR,
            size=5,
            now=self.now,
        )
        with mock.patch.object(R.os, "remove", side_effect=OSError("EPERM")):
            stats = R.sweep(self.dir, now=self.now)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(stats["diffs_removed"], 0)

    def test_an_unremovable_worktree_is_skipped_not_raised(self):
        parent = _wt_parent(self.dir)
        child = _wt_child(parent)
        _git_marker(child, None)
        _set_mtime(parent, self.now - 25 * _HOUR)
        with mock.patch.object(R.shutil, "rmtree", side_effect=OSError("EPERM")):
            stats = R.sweep(self.dir, now=self.now)
        self.assertTrue(os.path.lexists(parent))
        self.assertEqual(stats["worktrees_removed"], 0)

    def test_an_entry_that_vanishes_mid_scan_does_not_raise(self):
        path = _diff_spill(
            self.dir,
            "coding-agent-diff-flaky.patch",
            age_s=25 * _HOUR,
            size=5,
            now=self.now,
        )

        real_stat = os.DirEntry.stat

        def flaky_stat(self, *a, **kw):  # noqa: ANN001 - matches the bound method it patches
            os.unlink(path)
            return real_stat(self, *a, **kw)

        with mock.patch.object(os.DirEntry, "stat", flaky_stat):
            stats = R.sweep(self.dir, now=self.now)
        self.assertEqual(stats["diffs_removed"], 0)

    def test_a_completely_broken_scan_still_returns_the_stats_shape(self):
        with mock.patch.object(R.os, "scandir", side_effect=RuntimeError("boom")):
            stats = R.sweep(self.dir, now=self.now)
        self.assertEqual(
            stats, {"diffs_removed": 0, "diff_bytes_freed": 0, "worktrees_removed": 0}
        )


# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------


class DefaultTarget(unittest.TestCase):
    def test_no_spill_dir_argument_targets_the_real_tempdir(self):
        """Confirms the default without ever touching the real dir: patch
        `tempfile.gettempdir` to point at a scratch directory instead."""
        scratch = tempfile.mkdtemp(prefix="ca-reap-default-")
        self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        path = _diff_spill(
            scratch,
            "coding-agent-diff-x.patch",
            age_s=25 * _HOUR,
            size=1,
            now=time.time(),
        )
        with mock.patch.object(R.tempfile, "gettempdir", return_value=scratch):
            stats = R.sweep()
        self.assertFalse(os.path.exists(path))
        self.assertEqual(stats["diffs_removed"], 1)


if __name__ == "__main__":
    unittest.main()
