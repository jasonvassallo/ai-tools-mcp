#!/usr/bin/env python3
"""Security regression tests for coding_agent.

Every test here pins an attack that was DEMONSTRATED (not hypothesised)
during adversarial review of the design spec, 2026-08-16. Read
docs/superpowers/specs/2026-08-16-coding-agent-design.md §6.5/§6.6/§10.

Run:  uv run --with pytest --with pytest-timeout --with pathspec pytest test_coding_agent_security.py -q

`pytest-timeout` bounds the tests below whose regression signal is a HUNG
process rather than a red assertion — most directly
`test_file_swapped_for_fifo_after_classification_does_not_hang`, which pins
`_FILE_FLAGS`'s `O_NONBLOCK` in `coding_agent/walk.py`. It is requested here,
not required: absent the plugin, `@pytest.mark.timeout(...)` is an unknown
mark pytest warns about and ignores, and every test still runs — it just
loses that backstop. See `conftest.py` for the marker registration that
silences the warning either way.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import unittest.mock
from pathlib import Path

import pytest

from coding_agent import walk as walkmod
from coding_agent.basetree import BaseTree, make_ignore, read_base_tree
from coding_agent.walk import (
    DiffResult,
    Entry,
    TreeSnapshot,
    Unreadable,
    snapshot_tree,
    tree_hash,
    unified_diff,
)


class SymlinkExfiltration(unittest.TestCase):
    """Pass 3: a naive os.walk+open().read() dereferenced a planted
    /work/leak -> ~/.ssh/id_ed25519 and read the host secret INTO the diff."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.secret = Path(self.tmp) / "outside" / "id_ed25519"
        self.secret.parent.mkdir()
        self.secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETBYTES\n")
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()
        (self.root / "ok.py").write_text("print(1)\n")
        os.symlink(str(self.secret), self.root / "leak_abs")
        os.symlink("../outside/id_ed25519", self.root / "leak_rel")
        os.symlink(str(self.secret.parent), self.root / "leak_dir")

    def test_symlink_targets_are_recorded_as_text_never_dereferenced(self):
        snap = snapshot_tree(str(self.root), lambda p: False)
        for name in ("leak_abs", "leak_rel", "leak_dir"):
            e = snap.entries[name]
            self.assertEqual(e.kind, "symlink")
            self.assertNotIn(b"SECRETBYTES", e.data)
        # the secret's bytes appear NOWHERE in the snapshot
        blob = b"".join(e.data for e in snap.entries.values())
        self.assertNotIn(b"SECRETBYTES", blob)
        # a symlinked directory is not descended
        self.assertNotIn("leak_dir/id_ed25519", snap.entries)

    def test_hash_invariant_to_target_contents_sensitive_to_target_string(self):
        h1 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.secret.write_bytes(b"DIFFERENT SECRET")  # contents change
        h2 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertEqual(h1, h2, "hash must not depend on symlink TARGET contents")
        os.unlink(self.root / "leak_abs")
        os.symlink("/somewhere/else", self.root / "leak_abs")  # target STRING changes
        h3 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertNotEqual(h1, h3)

    @pytest.mark.timeout(15)
    def test_special_files_are_skipped_not_read(self):
        """Doubly guarded today: `_visit`'s `lstat` classifies the FIFO and
        returns before `_read_regular` is ever called, and `_read_regular`'s
        own `os.open` carries `O_NONBLOCK` besides. Losing EITHER guard alone
        still leaves the other standing; losing both at once is exactly the
        "a future maintainer simplifies away a defence-in-depth layer" shape
        this build kept finding (see `DefenceInDepthLayersAreIndividuallyPinned`
        below), so this still gets the same backstop as the tests that pin
        each half on its own.
        """
        os.mkfifo(self.root / "trap.fifo")  # open().read() here would block forever
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertNotIn("trap.fifo", snap.entries)

    def test_filter_attribute_escape_cannot_run(self):
        """Spec §10 test 2. The filter vector needs host git to honour an
        in-tree .gitattributes plus a repo-local [filter] config. Nothing in
        the walk invokes git, so the payload can never fire. Written rather
        than 'subsumed' because it is the REGRESSION GUARD if host git is ever
        reintroduced (ruled 2026-08-17)."""
        marker = Path(self.tmp) / "PWNED_FILTER"
        (self.root / ".gitattributes").write_text("*.py filter=pwn\n")
        (self.root / ".git").mkdir(exist_ok=True)
        (self.root / ".git" / "config").write_text(
            '[filter "pwn"]\n\tclean = sh -c "touch ' + str(marker) + '"\n'
        )
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertFalse(marker.exists(), "a filter driver executed during the walk")
        # .gitattributes is DATA to the walk, never configuration
        self.assertEqual(snap.entries[".gitattributes"].data, b"*.py filter=pwn\n")

    def test_concurrent_writer_toctou_cannot_win(self):
        """Spec §10 test 3. A writer re-mangling .git mid-run defeated the
        earlier check-then-invoke design (15 host executions in 20s). With no
        invocation at all there is no window. Guards a future reintroduction."""
        import threading

        marker = Path(self.tmp) / "PWNED_TOCTOU"
        stop = threading.Event()

        def remangle() -> None:
            g = self.root / ".git"
            while not stop.is_set():
                try:
                    if g.is_dir():
                        (g / "config").write_text(
                            '[core]\n\tfsmonitor = sh -c "touch ' + str(marker) + '"\n'
                        )
                    else:
                        g.mkdir(exist_ok=True)
                except OSError:
                    pass

        t = threading.Thread(target=remangle, daemon=True)
        t.start()
        try:
            for _ in range(50):
                snapshot_tree(str(self.root), lambda p: False)
        finally:
            stop.set()
            t.join(timeout=2)
        self.assertFalse(marker.exists(), "TOCTOU payload executed during the walk")

    def test_top_level_dot_git_is_opaque(self):
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text(
            "[core]\n\tfsmonitor = /tmp/pwn.sh\n"
        )
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertFalse(
            any(p == ".git" or p.startswith(".git/") for p in snap.entries)
        )


_SECRET_MARKER = b"STOLEN_SECRET_BYTES"
_SECRET_BLOB = b"-----BEGIN OPENSSH PRIVATE KEY-----\n" + _SECRET_MARKER + b"\n"


def _swap_on_first_lstat(victim_abs: str, link_target: str, as_fifo: bool = False):
    """Return an os.lstat replacement that swaps `victim_abs` for a symlink
    exactly ONCE, on the first lstat of that name.

    This makes the TOCTOU window DETERMINISTIC instead of raced: the swap lands
    between the classifying lstat and whatever the walk does with that
    classification. It matches both call shapes, `os.lstat(fullpath)` (the
    vulnerable pass-5 walk) and `os.lstat(name, dir_fd=fd)` (the fixed walk),
    so the same test is a valid counterfactual against either.
    """
    real_lstat = os.lstat
    target_name = os.path.basename(victim_abs)
    state = {"swapped": False}

    def fake_lstat(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if state["swapped"]:
            return st
        try:
            name = os.path.basename(os.fspath(path))
        except TypeError:
            return st
        if name != target_name:
            return st
        state["swapped"] = True
        if stat.S_ISDIR(st.st_mode):
            os.rmdir(victim_abs)
            os.symlink(link_target, victim_abs)
        elif as_fifo:
            # `link_target` is an existing FIFO; rename it over the victim so
            # the name the walk is about to open is a FIFO, not a symlink.
            os.rename(link_target, victim_abs)
        else:
            staged = victim_abs + ".staged"
            os.symlink(link_target, staged)
            os.rename(staged, victim_abs)
        return st

    return fake_lstat, state


class WalkRaceCannotDereference(unittest.TestCase):
    """Pass 5 (2026-08-17): `snapshot_tree` classified each path with `os.lstat`
    and then ACTED on that classification by PATH — `open(full, "rb")` for a
    file, and `os.listdir` on a stack-queued path for a directory. Neither used
    O_NOFOLLOW and neither re-verified the object, so a process racing the walk
    swapped the path for a symlink and made the HOST read a host file.

    Reachable in the real loop: the end-of-loop diff is protected because the
    container is destroyed first (spec 6.5 rule 3), but the PER-TURN no-progress
    hash runs while the container is alive, and spec 6.5 rule 1 says that window
    cannot be closed by killing the container. Verified exploitable: a
    separate-PROCESS swapper leaked a host private key on walk #13.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.secret_dir = Path(self.tmp) / "outside"
        self.secret_dir.mkdir()
        self.secret = self.secret_dir / "HOST_SECRET"
        self.secret.write_bytes(_SECRET_BLOB)
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()

    def _assert_no_secret(self, snap, why):
        for path, entry in snap.entries.items():
            self.assertNotIn(
                _SECRET_MARKER,
                entry.data,
                f"{why}: host secret bytes reached the snapshot via {path!r}",
            )

    def test_file_swapped_for_symlink_after_classification_is_never_read(self):
        victim = self.root / "victim.txt"
        victim.write_bytes(b"benign\n")
        fake, state = _swap_on_first_lstat(str(victim), str(self.secret))
        with unittest.mock.patch("os.lstat", fake):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertTrue(state["swapped"], "the interposition never fired")
        self._assert_no_secret(snap, "file vector")
        entry = snap.entries.get("victim.txt")
        if entry is not None:
            self.assertNotEqual(
                entry.kind,
                "file",
                "a symlink planted after classification was read as a file",
            )

    def test_directory_swapped_for_symlink_after_classification_is_never_listed(self):
        victim = self.root / "subdir"
        victim.mkdir()
        (self.root / "zzz_keep.txt").write_bytes(b"keep\n")
        fake, state = _swap_on_first_lstat(str(victim), str(self.secret_dir))
        with unittest.mock.patch("os.lstat", fake):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertTrue(state["swapped"], "the interposition never fired")
        self._assert_no_secret(snap, "directory vector")
        self.assertNotIn(
            "subdir/HOST_SECRET",
            snap.entries,
            "the walk descended through a symlink swapped in after lstat",
        )

    @pytest.mark.timeout(20)
    def test_file_swapped_for_fifo_after_classification_does_not_hang(self):
        """O_NOFOLLOW alone does not close the swap — it only refuses SYMLINKS.

        A FIFO renamed over a regular file between the classifying lstat and
        the open passes O_NOFOLLOW and then BLOCKS a plain open forever, which
        takes the host walk (and the human gate with it) offline. O_NONBLOCK is
        what closes that half.

        Regression signal, be warned: if O_NONBLOCK is ever dropped from
        _FILE_FLAGS this test HANGS rather than failing red — the same property
        the original FIFO test has, and the reason the repo would want
        pytest-timeout before this suite runs unattended in CI.

        The swap forces the walk PAST the classification-skip that would
        otherwise catch a FIFO before ever opening it (see
        `test_special_files_are_skipped_not_read`), so this is the one test
        in the suite that isolates `O_NONBLOCK` as the SOLE guard — no
        assertion below is reachable if it regresses; `os.open` blocks first.
        `@pytest.mark.timeout` is the only thing that turns that into a red
        failure naming this test rather than a hung, silent CI job.
        """
        victim = self.root / "victim.txt"
        victim.write_bytes(b"benign\n")
        fifo = str(victim) + ".fifo"
        os.mkfifo(fifo)
        fake, state = _swap_on_first_lstat(str(victim), fifo, as_fifo=True)
        with unittest.mock.patch("os.lstat", fake):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertTrue(state["swapped"], "the interposition never fired")
        entry = snap.entries.get("victim.txt")
        if entry is not None:
            self.assertNotEqual(entry.kind, "file", "a FIFO was read as a file")

    def _race(self, attacker_src: str, max_walks: int, budget_s: float) -> int:
        """Run `attacker_src` as a SEPARATE PROCESS while walking repeatedly.

        The attacker MUST be a process, never a thread. Two earlier probes used
        a Python thread and saw no leak across 620 walks: the GIL throttles the
        swapper into losing a race it wins easily as a process. A thread-based
        version of this test would be green while pinning nothing.
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", attacker_src],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        walks = 0
        leaks: list[tuple[int, str, str]] = []
        deadline = time.monotonic() + budget_s
        try:
            while walks < max_walks and time.monotonic() < deadline:
                walks += 1
                try:
                    snap = snapshot_tree(str(self.root), lambda p: False)
                except OSError as exc:
                    # A racing writer must never take the gate offline either.
                    self.fail(f"the walk aborted on walk #{walks}: {exc!r}")
                for path, entry in snap.entries.items():
                    if _SECRET_MARKER in entry.data:
                        leaks.append((walks, path, entry.kind))
                if leaks:
                    break
        finally:
            proc.kill()
            proc.wait(timeout=10)
        self.assertGreaterEqual(
            walks, 50, "too few walks for the probe to mean anything"
        )
        self.assertEqual(
            leaks, [], f"host secret reached the snapshot within {walks} walks"
        )
        return walks

    def test_separate_process_file_swapper_never_leaks_host_bytes(self):
        """File vector. The check-to-use window is lstat -> open, microseconds
        wide, so the attacker flips as fast as it can rather than holding a
        state: more flips, more chances to land inside the window. Measured
        against the pass-5 walk this leaks on walk #28.
        """
        victim = self.root / "aaa.txt"
        victim.write_bytes(b"benign\n")
        benign = self.root / ".benign"
        benign.write_bytes(b"benign\n")
        for i in range(60):
            (self.root / f"pad{i:03d}.txt").write_bytes(b"x" * 32)

        attacker = textwrap.dedent(f"""
            import os
            victim = {str(victim)!r}
            benign = {str(benign)!r}
            secret = {str(self.secret)!r}
            staged = victim + ".staged"
            while True:
                # ATTACK: a real symlink renamed over the victim. os.symlink
                # never follows, unlike os.link, which on macOS resolves its
                # source and would silently make this a HARDLINK probe instead.
                try:
                    os.symlink(secret, staged)
                    os.rename(staged, victim)
                except OSError:
                    pass
                # RESTORE: hardlink of a benign in-tree file, no secret involved.
                try:
                    os.link(benign, staged)
                    os.rename(staged, victim)
                except OSError:
                    pass
        """)
        self._race(attacker, max_walks=400, budget_s=6.0)

    def test_separate_process_dir_swapper_never_leaks_host_bytes(self):
        """Directory vector, the wider of the two: the pass-5 walk queued the
        directory's PATH on a stack and only re-resolved it with `os.listdir`
        when it was popped — after every sibling file had been read. Here the
        attacker HOLDS each state briefly, because this window is wide enough
        that dwell time beats flip rate. Measured against the pass-5 walk this
        leaks `aaa_dir/HOST_SECRET` on walk #696.

        `aaa_dir` sorts first so it is classified first and popped last.
        """
        victim = self.root / "aaa_dir"
        victim.mkdir()
        for i in range(60):
            (self.root / f"pad{i:03d}.txt").write_bytes(b"x" * 32)

        attacker = textwrap.dedent(f"""
            import os
            victim = {str(victim)!r}
            secret_dir = {str(self.secret_dir)!r}
            HOLD = 2000
            while True:
                try:
                    os.rmdir(victim)
                    os.symlink(secret_dir, victim)
                except OSError:
                    pass
                for _ in range(HOLD):
                    pass
                try:
                    os.unlink(victim)
                    os.mkdir(victim)
                except OSError:
                    pass
                for _ in range(HOLD):
                    pass
        """)
        self._race(attacker, max_walks=1500, budget_s=8.0)

    def test_separate_process_ancestor_swapper_never_leaks_host_bytes(self):
        """MID-TREE ANCESTOR vector, added by the pass-6 adversarial review.

        The two vectors above swap a LEAF. This one swaps a NON-EMPTY directory
        that the walk is going to descend THROUGH, using `rename` to move the
        real directory aside — which works where the `rmdir` of the directory
        vector cannot, because `rmdir` refuses a non-empty directory. Measured
        against the pass-5 walk it is the FASTEST of the three, leaking
        `aaa_mid/HOST_SECRET` on walk #10 (vs #26 and #168); against the fixed
        walk it produced 50,442 confirmed symlink plants and zero leaks.

        fd-pinning is what defeats it, and nothing else in this suite pins that
        property against a non-empty directory.
        """
        victim = self.root / "aaa_mid"
        victim.mkdir()
        (victim / "inner").mkdir()
        (victim / "inner" / "keep.txt").write_bytes(b"keep\n")
        (victim / "leaf.txt").write_bytes(b"leaf\n")
        for i in range(60):
            (self.root / f"pad{i:03d}.txt").write_bytes(b"x" * 32)

        attacker = textwrap.dedent(f"""
            import os
            victim = {str(victim)!r}
            secret_dir = {str(self.secret_dir)!r}
            aside = victim + ".aside"
            HOLD = 2000
            while True:
                # ATTACK: move the real (NON-EMPTY) directory aside and plant a
                # symlink to the host directory at the name the walk is about
                # to descend through.
                try:
                    os.rename(victim, aside)
                    os.symlink(secret_dir, victim)
                except OSError:
                    pass
                for _ in range(HOLD):
                    pass
                # RESTORE.
                try:
                    os.unlink(victim)
                    os.rename(aside, victim)
                except OSError:
                    pass
                for _ in range(HOLD):
                    pass
        """)
        self._race(attacker, max_walks=1500, budget_s=8.0)


class DeepTreeCannotHideAChange(unittest.TestCase):
    """Found while fixing the pass-5 race, not previously recorded.

    The pass-5 walk addressed every entry by absolute PATH, so once a tree grew
    past the operating system's PATH_MAX (1024 on macOS) `os.listdir` failed
    with ENAMETOOLONG, the guarded `except OSError: continue` swallowed it, and
    the ENTIRE subtree vanished from the snapshot with no error — a change the
    human gate exists to review, silently absent from the diff. Measured: a
    120-deep tree of 20-character names produced `entries=0`.

    The fd-based walk addresses each child through its parent's descriptor, so
    no single path is ever assembled and PATH_MAX never applies. Descriptor use
    is the tree's depth (measured: depth 60 -> 61 descriptors).

    CORRECTED 2026-08-19: an earlier version of this docstring cited a
    1,048,576 soft RLIMIT_NOFILE "on this host". That figure was measured in an
    interactive shell and is wrong for the deployment process — launchd hands
    the MCP server **256** and nothing raises it. So this fix MOVED the silent
    boundary from ~50 levels (PATH_MAX) to ~252 (descriptors) rather than
    removing it, and this test inherits the AMBIENT limit, so it cannot see
    that boundary at all. `DescriptorBudgetCannotHideASubtree` pins the limit
    itself and covers the descriptor case; this class covers PATH_MAX only.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_file_below_path_max_still_reaches_the_snapshot(self):
        name = "d" * 20
        depth = 120
        parent = os.open(str(self.root), os.O_RDONLY | os.O_DIRECTORY)
        try:
            for _ in range(depth):
                os.mkdir(name, dir_fd=parent)
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent)
                os.close(parent)
                parent = child
            fd = os.open("buried.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=parent)
            try:
                os.write(fd, b"A CHANGE THE GATE MUST SEE\n")
            finally:
                os.close(fd)
        finally:
            os.close(parent)

        expected = "/".join([name] * depth) + "/buried.txt"
        self.assertGreater(len(expected), 1024, "the fixture must exceed PATH_MAX")
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertIn(expected, snap.entries, "a deep subtree vanished from the diff")
        self.assertEqual(snap.entries[expected].data, b"A CHANGE THE GATE MUST SEE\n")


class UnreadablePathsAreVisibleToTheGate(unittest.TestCase):
    """Pass 6 (2026-08-19): `chmod 000 backdoor.py` — ONE command, no race, no
    live process, no timing — made the file VANISH from the snapshot with no
    exception and no signal. `chmod` is persistent STATE, so destroying the
    container does not undo it: the file is still missing when the FINAL diff
    the human reviews is built. A model can write a backdoor, chmod it 000, and
    the reviewer never learns the file exists.

    It was a REGRESSION introduced by the pass-5 race fix. The pre-fix walk's
    unguarded `open()` raised PermissionError LOUDLY on this fixture; guarding
    the open traded a noisy failure for a silent under-show — the direction
    §6.5 names as the dangerous one.

    The fix is not to raise again (a racing writer must not take the gate
    offline, §5.2) but to record the path in `TreeSnapshot.unreadable`, so the
    gate can say "this path exists and could not be read" instead of quietly
    showing fewer files. Every test below pins VISIBILITY, not availability.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()
        self._locked: list[Path] = []

    def tearDown(self):
        for p in reversed(self._locked):
            try:
                os.chmod(p, 0o755)
            except OSError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _lock(self, p: Path) -> None:
        """chmod 000, and assert the MECHANISM: an existing object, owned by
        the walking user, with every permission bit off. Not a vanished path,
        not a symlink, not somebody else's file."""
        os.chmod(p, 0o000)
        self._locked.append(p)
        st = os.lstat(p)
        self.assertEqual(stat.S_IMODE(st.st_mode), 0o000)
        self.assertEqual(st.st_uid, os.geteuid())
        self.assertFalse(stat.S_ISLNK(st.st_mode))

    @staticmethod
    def _pairs(snap) -> list[tuple[str, str]]:
        return [(u.path, u.reason) for u in snap.unreadable]

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_a_mode_000_file_is_recorded_not_silently_dropped(self):
        (self.root / "innocent.py").write_text("print('hello')\n")
        victim = self.root / "backdoor.py"
        victim.write_text("BACKDOOR = True\n")
        self._lock(victim)
        self.assertTrue(stat.S_ISREG(os.lstat(victim).st_mode))

        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["innocent.py"].data, b"print('hello')\n")
        self.assertNotIn("backdoor.py", snap.entries)
        self.assertEqual(self._pairs(snap), [("backdoor.py", "EACCES")])

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_a_mode_000_directory_says_a_whole_subtree_is_hidden(self):
        (self.root / "innocent.py").write_text("ok\n")
        (self.root / "src" / "deep").mkdir(parents=True)
        (self.root / "src" / "deep" / "payload.py").write_text("PAYLOAD\n")
        self._lock(self.root / "src")

        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertNotIn("src/deep/payload.py", snap.entries)
        # TRAILING SLASH: this is a whole subtree of unknown size, not one file.
        self.assertEqual(self._pairs(snap), [("src/", "EACCES")])

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_a_file_created_already_unreadable_moves_the_no_progress_hash(self):
        """`entries` alone cannot see this. A file that was NEVER readable never
        appears in `entries`, so a hash over `entries` is byte-identical before
        and after the model creates it — the §7 detector would score the turn as
        no progress and the chmod-000 hide would be invisible to it as well as
        to the diff. The two walks below have IDENTICAL `entries`, so only
        `unreadable` can move the hash.
        """
        (self.root / "innocent.py").write_text("ok\n")
        first = snapshot_tree(str(self.root), lambda p: False)
        before = tree_hash(first)

        victim = self.root / "backdoor.py"
        victim.write_text("BACKDOOR = True\n")
        self._lock(victim)
        second = snapshot_tree(str(self.root), lambda p: False)

        self.assertEqual(sorted(first.entries), sorted(second.entries))
        self.assertNotEqual(before, tree_hash(second), "the hash ignored a change")

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_a_permanently_unreadable_path_does_not_flap_the_hash(self):
        """The other half of the ruling. A path that is permanently unreadable
        contributes the SAME (path, reason) every turn, so the hash is a
        constant — it must not report progress on every turn forever, which
        would defeat §7's detector on any tree containing one 000 file.
        """
        (self.root / "innocent.py").write_text("ok\n")
        locked = self.root / "locked.py"
        locked.write_text("locked\n")
        self._lock(locked)
        h1 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        h2 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        h3 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertEqual(h1, h2)
        self.assertEqual(h2, h3)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_an_unreadable_root_reports_itself_instead_of_looking_empty(self):
        """An empty snapshot makes every tracked path surface as a deletion.
        That is loud, but it does not say WHY. The root records itself as `.`
        so the gate can tell "the worktree is empty" from "the worktree could
        not be opened"."""
        (self.root / "a.py").write_text("x\n")
        self._lock(self.root)
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries, {})
        self.assertEqual(self._pairs(snap), [(".", "EACCES")])

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_unreadable_is_sorted_and_deterministic(self):
        for name in ("zeta.py", "alpha.py", "middle.py"):
            (self.root / name).write_text("x\n")
            self._lock(self.root / name)
        (self.root / "sub").mkdir()
        self._lock(self.root / "sub")
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(
            self._pairs(snap),
            [
                ("alpha.py", "EACCES"),
                ("middle.py", "EACCES"),
                ("sub/", "EACCES"),
                ("zeta.py", "EACCES"),
            ],
        )
        again = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.unreadable, again.unreadable)

    def test_the_hash_ignores_transient_reasons_and_keeps_permanent_ones(self):
        """MEASURED ruling, not a taste call. Hashing EVERY unreadable reason
        makes the §7 no-progress detector useless in the presence of an HONEST
        racing writer — a test run creating and deleting temp files. Over 1512
        walks against a separate churning process:

            flaps with entries only ......  0.6%
            flaps with ALL reasons hashed . 87.1%
            flaps with stable reasons only  0.6%

        A permanent reason is a property of the TREE and must move the hash (it
        is the only way a created-already-000 file is visible to §7). A
        transient one is a property of someone else's timing and must not.
        """
        empty = TreeSnapshot({})
        for transient in ("RACED", "ENOENT", "ELOOP", "ESTALE"):
            with self.subTest(reason=transient):
                self.assertEqual(
                    tree_hash(empty),
                    tree_hash(TreeSnapshot({}, (Unreadable("a.py", transient),))),
                )
        for permanent in ("EACCES", "EPERM", "EMFILE", "OVERSIZE", "BUDGET", "XDEV"):
            with self.subTest(reason=permanent):
                self.assertNotEqual(
                    tree_hash(empty),
                    tree_hash(TreeSnapshot({}, (Unreadable("a.py", permanent),))),
                )

    def test_a_path_that_keeps_vanishing_is_still_REPORTED(self):
        """The hash ignores it; the gate must not. `unreadable` is what the
        reviewer reads, and "this path was there when I listed the directory
        and gone when I looked at it" is worth saying — the final diff is built
        after the container is destroyed, so a transient there is anomalous.
        """
        (self.root / "a.py").write_text("x\n")
        (self.root / "ghost.py").write_text("y\n")
        real_lstat = os.lstat

        def flaky(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == "ghost.py":
                raise FileNotFoundError(errno.ENOENT, "vanished", "ghost.py")
            return real_lstat(path, *args, **kwargs)

        with unittest.mock.patch("os.lstat", flaky):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(sorted(snap.entries), ["a.py"])
        self.assertEqual(self._pairs(snap), [("ghost.py", "ENOENT")])

    def test_a_directory_the_walk_gives_up_on_is_reported_AS_A_DIRECTORY(self):
        """Found by review of the pass-6 fix itself. The exhausted-retries
        fall-through records the bare `rel`, because it is reached with no
        classification in hand — so a directory the walk lost the race for
        three times was reported as `victim_dir`, not `victim_dir/`. Visible,
        but the SCOPE was understated: a reviewer reads "one file could not be
        read" where an entire subtree of unknown size is hidden. Understating
        what is hidden is the same failure direction as hiding it.

        ELOOP is the real mechanism here, not a synthetic error: it is exactly
        what O_NOFOLLOW raises when a symlink has been swapped in over the
        directory since it was classified.
        """
        victim = self.root / "victim_dir"
        victim.mkdir()
        (victim / "payload.py").write_text("PAYLOAD\n")
        (self.root / "a.py").write_text("x\n")
        real_open = os.open
        state = {"refused": 0}

        def refuse_the_dir(path, flags, *args, **kwargs):
            if kwargs.get("dir_fd") is not None and path == "victim_dir":
                state["refused"] += 1
                raise OSError(errno.ELOOP, "symlink swapped in", "victim_dir")
            return real_open(path, flags, *args, **kwargs)

        with unittest.mock.patch("os.open", refuse_the_dir):
            snap = snapshot_tree(str(self.root), lambda p: False)
        # MECHANISM: it really was refused, and refused on every attempt.
        self.assertEqual(state["refused"], walkmod._RECLASSIFY_ATTEMPTS)
        self.assertIn("a.py", snap.entries)
        self.assertNotIn("victim_dir/payload.py", snap.entries)
        self.assertEqual(self._pairs(snap), [("victim_dir/", "ELOOP")])

    def test_a_directory_too_large_to_LIST_is_recorded_not_a_crash(self):
        """`os.listdir` builds the whole name list before returning it, so a
        directory with enough children exhausts host memory — and MemoryError
        is not an OSError, so it used to escape `snapshot_tree` entirely and
        take the review gate offline. That is the §5.2 failure ("a racing
        writer must not take the gate offline") arriving through the memory
        axis instead of the timing one, for a condition the sandbox creates
        with `touch`. The byte budgets do not bound it: they cap CONTENT, and
        this is a cost per NAME.
        """
        (self.root / "a.py").write_text("x\n")
        (self.root / "huge").mkdir()
        (self.root / "huge" / "f.txt").write_text("y\n")
        real_listdir = os.listdir
        calls = {"n": 0}

        def hungry(arg):
            calls["n"] += 1
            if calls["n"] >= 2:  # the root lists fine; the child does not
                raise MemoryError("cannot allocate the name list")
            return real_listdir(arg)

        before = len(os.listdir("/dev/fd"))
        with unittest.mock.patch("os.listdir", hungry):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["a.py"].data, b"x\n")
        self.assertEqual(self._pairs(snap), [("huge/", "ENOMEM")])
        self.assertEqual(len(os.listdir("/dev/fd")), before, "descriptor leaked")

    def test_a_root_too_large_to_LIST_is_recorded_not_a_crash(self):
        (self.root / "a.py").write_text("x\n")
        before = len(os.listdir("/dev/fd"))
        with unittest.mock.patch("os.listdir", side_effect=MemoryError("no room")):
            snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries, {})
        self.assertEqual(self._pairs(snap), [(".", "ENOMEM")])
        self.assertEqual(len(os.listdir("/dev/fd")), before, "descriptor leaked")

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 path anyway")
    def test_an_ignored_unreadable_path_is_not_reported(self):
        """`unreadable` must not become a second channel that leaks ignored
        paths into the gate's view. `lstat` succeeds on a 000 file — only the
        OPEN fails — so the ignore decision is made exactly as it is for a
        readable file, with the same query string."""
        victim = self.root / "secrets.env"
        victim.write_text("x\n")
        self._lock(victim)
        (self.root / "shown.py").write_text("y\n")
        snap = snapshot_tree(str(self.root), lambda p: p.endswith(".env"))
        self.assertIn("shown.py", snap.entries)
        self.assertEqual(self._pairs(snap), [])


def _sparse(path: Path, size: int) -> None:
    """Create a file with `size` APPARENT bytes and no allocated blocks."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


def _max_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


class HostMemoryCannotBeExhaustedByTheSandbox(unittest.TestCase):
    """Pass 6: `dd if=/dev/zero of=/work/big.bin bs=1 count=0 seek=8589934592`
    costs the container ZERO bytes of disk and drove the host walk to 3006 MB
    RSS in 0.3s (measured in a child process under a 2500 MB kill ceiling;
    nothing in walk.py would have stopped it). `seek=1099511627776` — 1 TiB —
    is exactly as cheap for the attacker, and the old `_read_all` accumulated
    1 MiB chunks in a list before `b"".join`, so peak was roughly 2x apparent
    size.

    §5.1's diff size cap CANNOT help: it truncates the RENDERED diff, and by
    then the bytes are already resident in `TreeSnapshot.entries`. The cap has
    to be at the read.

    An over-cap file is never TRUNCATED into something that looks like real
    content — that would hand the reviewer a misleading diff of a file it did
    not fully read. It is recorded in `unreadable` instead.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_sparse_file_above_the_cap_is_recorded_not_read(self):
        (self.root / "real.py").write_text("print(1)\n")
        big = self.root / "big.bin"
        _sparse(big, 8 * (1 << 30))
        st = os.lstat(big)
        # MECHANISM: 8 GiB APPARENT, zero blocks ALLOCATED. If this fixture
        # ever stops being sparse the test is measuring disk, not the cap.
        self.assertEqual(st.st_size, 8 * (1 << 30))
        self.assertEqual(st.st_blocks, 0)
        self.assertGreater(st.st_size, walkmod._MAX_FILE_BYTES)

        before = _max_rss_bytes()
        snap = snapshot_tree(str(self.root), lambda p: False)
        grew = _max_rss_bytes() - before

        self.assertEqual(snap.entries["real.py"].data, b"print(1)\n")
        self.assertNotIn("big.bin", snap.entries)
        self.assertEqual(
            [(u.path, u.reason) for u in snap.unreadable], [("big.bin", "OVERSIZE")]
        )
        self.assertLess(grew, 256 << 20, f"the walk grew RSS by {grew} bytes")

    def test_a_file_that_grows_past_the_cap_mid_read_is_not_truncated(self):
        """The pre-check is `fstat().st_size` on the OPEN descriptor, which a
        writer can beat by appending afterwards and which a synthetic
        filesystem can simply lie about. The read loop is therefore bounded
        too, and hitting that bound is an OVERSIZE record — never a short read
        recorded as if it were the whole file."""
        victim = self.root / "grows.bin"
        victim.write_bytes(b"x" * 64)
        cap = 4096
        real_read = os.read
        state = {"grown": False}

        def growing_read(fd: int, n: int) -> bytes:
            if not state["grown"]:
                state["grown"] = True
                with open(victim, "ab") as fh:
                    fh.write(b"y" * (cap * 4))
            return real_read(fd, n)

        with unittest.mock.patch.object(walkmod, "_MAX_FILE_BYTES", cap):
            with unittest.mock.patch("os.read", growing_read):
                snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertTrue(state["grown"], "the interposition never fired")
        self.assertNotIn("grows.bin", snap.entries)
        self.assertEqual(
            [(u.path, u.reason) for u in snap.unreadable], [("grows.bin", "OVERSIZE")]
        )

    def test_many_under_cap_files_cannot_exhaust_memory_in_aggregate(self):
        """A PER-FILE cap alone is not enough. `entries` holds every file at
        once, so N files of (cap - 1) bytes cost N x cap of host RAM, and N
        sparse files are as cheap to create as one. The shipped constants are
        patched DOWN here so the test costs megabytes instead of gigabytes; the
        mechanism — a running total that stops reading and RECORDS what it did
        not read — is the shipped one.
        """
        names = [f"f{i:02d}.bin" for i in range(16)]
        for name in names:
            _sparse(self.root / name, 1 << 19)  # 512 KiB apparent, 0 allocated
        self.assertEqual(sum(os.lstat(self.root / n).st_blocks for n in names), 0)

        file_cap = 1 << 20  # every file is UNDER this
        total_cap = 4 << 20  # but 16 x 512 KiB = 8 MiB is over this
        with unittest.mock.patch.object(walkmod, "_MAX_FILE_BYTES", file_cap):
            with unittest.mock.patch.object(walkmod, "_MAX_TOTAL_BYTES", total_cap):
                snap = snapshot_tree(str(self.root), lambda p: False)

        read = sum(len(e.data) for e in snap.entries.values())
        self.assertGreater(read, 0, "the budget starved the whole walk")
        self.assertLessEqual(read, total_cap, "the total budget was not enforced")
        self.assertTrue(any(u.reason == "BUDGET" for u in snap.unreadable))
        # NOTHING VANISHES: every file is either shown or named.
        named = {u.path for u in snap.unreadable}
        for name in names:
            self.assertTrue(
                name in snap.entries or name in named, f"{name} vanished silently"
            )

    def test_the_shipped_caps_are_bounded_and_ordered(self):
        self.assertLessEqual(walkmod._MAX_FILE_BYTES, walkmod._MAX_TOTAL_BYTES)
        self.assertLessEqual(walkmod._MAX_TOTAL_BYTES, 1 << 30)


class DescriptorBudgetCannotHideASubtree(unittest.TestCase):
    """Pass 6 corrected a 4000x measurement error. `progress.md` and
    `task-2-fix-report.md` both recorded the host's soft RLIMIT_NOFILE as
    1,048,576, which was measured in an INTERACTIVE SHELL. The process that
    actually runs this code is the ai-tools-mcp server under launchd, and
    `launchctl limit maxfiles` gives it **256**; nothing in the server raises
    it. "Silent subtree drop past RLIMIT_NOFILE" therefore costs ~250 mkdirs,
    not a million — cheap enough to matter.

    This test PINS the limit rather than inheriting the ambient one. The older
    `DeepTreeCannotHideAChange` inherits it, so it stays green in a shell while
    saying nothing about the deployed configuration.

    The requirement is the same as for a mode-000 path: the walk may fail to
    read a subtree, but it may not do so SILENTLY.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()
        self.depth = 150
        parent = os.open(str(self.root), os.O_RDONLY | os.O_DIRECTORY)
        try:
            for _ in range(self.depth):
                os.mkdir("d", dir_fd=parent)
                child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent)
                os.close(parent)
                parent = child
                fd = os.open("f.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=parent)
                os.close(fd)
        finally:
            os.close(parent)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _walk_under_fd_limit(self, spare: int):
        """Walk with RLIMIT_NOFILE pinned to (currently open + `spare`).

        The limit is restored BEFORE any assertion runs, so a failing assertion
        cannot be reported through a descriptor the test just took away.
        """
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        budget = len(os.listdir("/dev/fd")) + spare
        resource.setrlimit(resource.RLIMIT_NOFILE, (budget, hard))
        try:
            return snapshot_tree(str(self.root), lambda p: False)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    @staticmethod
    def _deepest(snap) -> int:
        return max((p.count("/") for p in snap.entries), default=0)

    def test_a_subtree_lost_to_the_descriptor_limit_is_recorded(self):
        snap = self._walk_under_fd_limit(spare=40)
        deepest = self._deepest(snap)
        self.assertLess(
            deepest, self.depth, "the fixture did not exhaust the descriptor budget"
        )
        reasons = {u.reason for u in snap.unreadable}
        self.assertTrue(
            reasons & {"EMFILE", "ENFILE"},
            f"a subtree vanished with no record; unreadable={snap.unreadable!r}",
        )
        # And every record names a DIRECTORY, so the gate reads it as a subtree.
        cut = [u.path for u in snap.unreadable if u.reason in ("EMFILE", "ENFILE")]
        self.assertTrue(all(p.endswith("/") for p in cut), cut)

    def test_the_cut_off_tracks_the_descriptor_limit_and_nothing_else(self):
        """MECHANISM. If the truncation depth did not move with the limit, this
        test would be pinning some other boundary (PATH_MAX, a hard-coded depth
        cap, an ENAMETOOLONG) while claiming to pin fd exhaustion."""
        shallow = self._deepest(self._walk_under_fd_limit(spare=25))
        deeper = self._deepest(self._walk_under_fd_limit(spare=75))
        self.assertGreater(deeper, shallow + 20, f"{shallow=} {deeper=}")
        self.assertLess(deeper, self.depth)


class RepoDirectoryOpacityIsIdentityBased(unittest.TestCase):
    """Pass 6: `snapshot_tree` guarded the repo directory with a
    case-SENSITIVE text compare, and the deployment volume is case-INSENSITIVE
    APFS. A case-only rename of the real one dodged the guard entirely.

    Direct impact is over-showing, which §6.5 calls harmless — but it CHAINS:
    the model can flood the diff with pack and object bytes, and §5.1 truncates
    the diff past a byte limit, so over-showing plus a truncation cap becomes
    under-showing BY DISPLACEMENT — the real change scrolls off the end. It is
    also a delivery vector for the sparse-file DoS under a name reviewers are
    trained to skip.

    The guard is now identity-based (`st_ino`/`st_dev`, compared against the
    SAME lstat that classifies each child, so no new TOCTOU window opens). That
    is correct on both kinds of volume: a case-only rename is the same inode
    and stays opaque, while on a case-SENSITIVE volume an upper-case name is a
    different inode and is shown — which is what a real checkout would do.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_case_renamed_repo_directory_stays_opaque(self):
        (self.root / "app.py").write_text("print(1)\n")
        meta = self.root / ".git"
        meta.mkdir()
        (meta / "config").write_text("[core]\n\tfsmonitor = /tmp/pwn\n")
        (meta / "objects").mkdir()
        (meta / "objects" / "pack.idx").write_bytes(b"PACKBYTES" * 64)
        os.rename(meta, self.root / ".GIT")
        if not meta.exists():
            self.skipTest("case-sensitive volume: the upper-case name is a new object")
        self.assertIn(".GIT", os.listdir(self.root))

        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertIn("app.py", snap.entries)
        leaked = [p for p in snap.entries if p.lower().startswith(".git")]
        self.assertEqual(leaked, [], "a case-renamed repo directory reached the diff")

    def test_a_worktree_metadata_FILE_is_still_opaque(self):
        """In a WORKTREE — which is what this tool walks — the metadata entry
        is a regular FILE containing a pointer, not a directory. Identity
        matching must cover that shape too."""
        (self.root / "app.py").write_text("print(1)\n")
        (self.root / ".git").write_text("gitdir: /elsewhere/worktrees/wt\n")
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertIn("app.py", snap.entries)
        self.assertNotIn(".git", snap.entries)

    def test_a_hardlink_to_the_metadata_file_does_not_hide_a_chosen_name(self):
        """Identity matching, taken alone, is itself an under-showing bug. A
        worktree's metadata entry is an ordinary FILE, so the sandbox can
        `ln` it to any name — and every one of those names would then share
        its inode and vanish from the diff silently, at a name the SANDBOX
        picks. The guard therefore requires the name (in any casing) as well
        as the identity.
        """
        (self.root / ".git").write_text("gitdir: /elsewhere/worktrees/wt\n")
        decoy = self.root / "notes.txt"
        os.link(self.root / ".git", decoy)
        # MECHANISM: the decoy really is the SAME inode, so a pure identity
        # test would have skipped it.
        self.assertEqual(os.lstat(decoy).st_ino, os.lstat(self.root / ".git").st_ino)

        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertNotIn(".git", snap.entries)
        self.assertIn("notes.txt", snap.entries, "a sandbox-chosen name vanished")


_NO_SPAWN_CHILD = r'''
"""Child for the §10.1 zero-spawn assertion.

An audit hook cannot be REMOVED once installed, so this cannot run inside the
pytest process without slowing and perturbing every later test. It also must
not run in a thread of one: the counter has to see the whole interpreter.
"""
import json, os, shutil, subprocess, sys, tempfile

SPAWN_EVENTS = (
    "subprocess.Popen", "os.exec", "os.posix_spawn", "os.posix_spawnp",
    "os.system", "os.spawn", "os.fork", "os.forkpty", "pty.spawn",
)
seen = []
def hook(event, args):
    if event.startswith(SPAWN_EVENTS):
        seen.append(event)
sys.addaudithook(hook)

sys.path.insert(0, sys.argv[1])
from coding_agent.basetree import BaseTree
from coding_agent.walk import Entry, snapshot_tree, tree_hash, unified_diff

root = sys.argv[2]
base = BaseTree(
    entries={"app.py": Entry("app.py", "file", b"print(1)\n")},
    ignore=lambda p: False,
    tracked=frozenset({"app.py"}),
)
snap = snapshot_tree(root, lambda p: False)
digest = tree_hash(snap)
spill_dir = tempfile.mkdtemp()
try:
    diff = unified_diff(base, snap, max_bytes=1 << 20, spill_dir=spill_dir)
finally:
    shutil.rmtree(spill_dir, ignore_errors=True)
during = len(seen)

# CONTROL: the hook must be able to COUNT. Without this, "0 spawns" is
# indistinguishable from "the hook never fired at all".
subprocess.run([sys.executable, "-c", "pass"], check=True)
print(json.dumps({
    "during": during,
    "after_control": len(seen),
    "entries": sorted(snap.entries),
    "diff_len": len(diff.text),
    "digest": digest[:12],
}))
'''


class NoHostSubprocessRunsOverTheWalkOrTheDiff(unittest.TestCase):
    """§10 item 1, literal half. The spec asks for an assertion that the
    HOST-SIDE read path spawns no `git` — indeed no subprocess at all.

    `sandbox.py`'s git calls are pinned at argv level in
    `test_coding_agent_sandbox.py`, but that covers the calls this design
    ACCEPTS. The property the spec actually names is about the path that runs
    over a directory the model controls: `snapshot_tree` -> `tree_hash` ->
    `unified_diff`. Those three replaced `git add -A`/`git diff` because
    adversarial review reached host RCE through a sandbox-authored `.git`, and
    nothing pinned that they stayed subprocess-free. A refactor that reached
    for `git check-ignore`, `git hash-object` or `file(1)` for convenience
    would reinstate the whole class silently.

    A CPython audit hook is the right instrument because it observes the
    interpreter rather than the source: it counts `subprocess.Popen`,
    `os.exec*`, `os.posix_spawn*`, `os.system` and `os.fork` however they are
    reached, including from inside a dependency.
    """

    @pytest.mark.timeout(30)
    def test_the_walk_hash_and_diff_spawn_no_subprocess_over_a_hostile_tree(self):
        """The hostile tree below includes a FIFO (`pipe`), walked by a CHILD
        interpreter through the exact `snapshot_tree` -> `tree_hash` ->
        `unified_diff` pipeline the real loop uses. `subprocess.run` had no
        `timeout=` at all until this pass: if that child ever blocked inside
        the walk — a `_FILE_FLAGS`/`O_NONBLOCK` regression, or a classification
        regression that stopped skipping the FIFO before opening it — this
        test would hang the parent right along with it, silently, forever.
        `timeout=` below turns that into `subprocess.TimeoutExpired`, caught
        and re-raised as a named, informative failure; `@pytest.mark.timeout`
        is the outer backstop in case the raise itself is somehow never
        reached.
        """
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = tmp / "work"
        root.mkdir()
        (root / "app.py").write_text("print(2)\n")
        # Every input that has ever tempted this code back towards git.
        meta = root / ".git"
        meta.mkdir()
        (meta / "config").write_text(
            "[core]\n\tfsmonitor = /tmp/pwn.sh\n\thooksPath = /tmp/hooks\n"
            '[filter "evil"]\n\tclean = /tmp/pwn.sh\n\tsmudge = /tmp/pwn.sh\n'
        )
        (root / ".gitattributes").write_text("* filter=evil\n")
        (root / ".gitignore").write_text("*.log\n")
        os.symlink("/etc/passwd", root / "outward")
        os.mkfifo(root / "pipe")
        nested = root / "sub" / ".git"
        nested.mkdir(parents=True)
        (nested / "config").write_text("[core]\n\tfsmonitor = /tmp/pwn.sh\n")

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _NO_SPAWN_CHILD,
                    str(Path(__file__).parent),
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                "the child walking the hostile tree (which includes a FIFO) "
                "did not return within 20s -- this is what a blocking, "
                "non-O_NONBLOCK open (or a lost FIFO classification-skip) "
                f"looks like; partial stdout={exc.stdout!r} stderr={exc.stderr!r}"
            )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        got = json.loads(proc.stdout.strip().splitlines()[-1])

        self.assertEqual(got["during"], 0, "the host read path spawned a process")
        # MECHANISM: the hook demonstrably counts, so the zero above is a
        # measurement and not a silently inert instrument.
        self.assertGreater(
            got["after_control"],
            0,
            "the audit hook never fired: the 0 above proves nothing",
        )
        # ...and the read really happened over the hostile tree.
        self.assertIn("app.py", got["entries"])
        self.assertGreater(got["diff_len"], 0)


class DefenceInDepthLayersAreIndividuallyPinned(unittest.TestCase):
    """Final review 2026-08-19: five layered defences whose layers were only
    pinned IN PAIRS.

    The mutation survey found that removing `O_NOFOLLOW` from the file read
    left the whole suite green (the `st_ino`/`st_dev` check absorbs it), that
    removing the `st_ino`/`st_dev` check left it green too (`O_NOFOLLOW`
    absorbs it), and that removing BOTH leaked a host secret into the snapshot
    on walk #238. The same held for the child-directory descent, and for the
    two independent implementations of `.git` opacity.

    A layer that no test can distinguish from its partner is a layer a future
    maintainer deletes as redundant, with a green suite as the receipt — and
    the suite would stay green right up until somebody removed the second one.
    Each test below therefore turns a mutation that SURVIVED into one that
    does not, by asserting the MECHANISM (which refusal, from which guard) and
    not merely the outcome (no leak).

    Everything here is deterministic. The races these guards defend against
    were demonstrated with separate-process probes during the build; a probe
    that must win a race to prove anything is a probe that reports safety when
    it simply lost, so the window is opened by hand instead and the guard is
    interrogated at the point it fires.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "work"
        self.root.mkdir()
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        self.secret = self.outside / "id_rsa"
        self.secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\nSTOLEN\n")
        self.fd = os.open(str(self.root), os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)
        self.root_dev = os.fstat(self.fd).st_dev

    # -- layer 1 of 2 on the FILE read: O_NOFOLLOW --

    def test_a_symlink_swapped_in_after_classification_is_refused_by_O_NOFOLLOW(self):
        """Mutation W1 (drop `O_NOFOLLOW` from `_FILE_FLAGS`) SURVIVED the
        whole suite: the identity check refuses the read anyway, so no secret
        leaks and nothing goes red. The two guards are distinguishable only by
        WHICH refusal you get — `ELOOP` means the kernel never opened the
        symlink, `RACED` means it did and the identity check caught it after
        the fact. Assert the first, and W1 dies.
        """
        victim = self.root / "victim"
        victim.write_bytes(b"honest content\n")
        classified = os.lstat("victim", dir_fd=self.fd)  # a REGULAR file
        # The race window, opened by hand: the name now resolves elsewhere.
        victim.unlink()
        os.symlink(str(self.secret), victim)

        # MECHANISM, both halves — a probe that measures the wrong object is
        # this project's most repeated mistake (a "symlink" probe that was
        # really a hardlink; a race probe throttled by the GIL).
        self.assertTrue(stat.S_ISLNK(os.lstat("victim", dir_fd=self.fd).st_mode))
        self.assertEqual(victim.read_bytes(), self.secret.read_bytes())
        self.assertTrue(stat.S_ISREG(classified.st_mode))
        # ...and dropping ONLY this guard really does open the host secret.
        followed = os.open(
            "victim", walkmod._FILE_FLAGS & ~os.O_NOFOLLOW, dir_fd=self.fd
        )
        try:
            self.assertIn(b"STOLEN", os.read(followed, 4096))
        finally:
            os.close(followed)

        got = walkmod._read_regular(
            self.fd,
            "victim",
            "victim",
            classified,
            self.root_dev,
            walkmod._Budget(1 << 20),
        )
        self.assertIsInstance(got, walkmod._Refusal)
        self.assertEqual(
            got.reason,
            "ELOOP",
            "the open followed the symlink; only the identity check refused it",
        )

    # -- layer 2 of 2 on the FILE read: the st_ino/st_dev identity check --

    def test_a_different_regular_file_renamed_over_the_name_is_refused_by_identity(
        self,
    ):
        """Mutation W3 (`_same_object` -> `return True`) SURVIVED: every probe
        the suite runs plants a SYMLINK, and `O_NOFOLLOW` refuses those before
        identity is consulted. This plants a REGULAR FILE instead, which
        `O_NOFOLLOW` is deliberately blind to — so the identity check is the
        only thing standing between the walk and content it never classified.
        """
        victim = self.root / "victim"
        victim.write_bytes(b"honest content\n")
        classified = os.lstat("victim", dir_fd=self.fd)
        impostor = self.root / ".impostor"
        impostor.write_bytes(b"BAIT-NOT-THE-CLASSIFIED-FILE\n")
        os.rename(impostor, victim)  # atomic swap, no symlink anywhere

        # MECHANISM: a different inode, and O_NOFOLLOW cannot possibly object.
        now = os.lstat("victim", dir_fd=self.fd)
        self.assertTrue(stat.S_ISREG(now.st_mode))
        self.assertNotEqual(now.st_ino, classified.st_ino)
        probe = os.open("victim", walkmod._FILE_FLAGS, dir_fd=self.fd)
        os.close(probe)  # O_NOFOLLOW opens it happily: identity is the guard

        got = walkmod._read_regular(
            self.fd,
            "victim",
            "victim",
            classified,
            self.root_dev,
            walkmod._Budget(1 << 20),
        )
        self.assertIsInstance(got, walkmod._Refusal)
        self.assertEqual(got.reason, walkmod._RACED)
        self.assertNotIsInstance(got, Entry)

    # -- layer 1 of 2 on the DIRECTORY descent: O_NOFOLLOW --

    def test_a_symlinked_directory_swapped_in_is_refused_by_O_NOFOLLOW(self):
        """Mutation W23b (drop `O_NOFOLLOW` at the child-descent `os.open`)
        SURVIVED: the descent's own identity check absorbs it. Same asymmetry
        as the file read, same fix — assert which guard refused, not merely
        "no leak".

        The refusal here is `ENOTDIR`, not `ELOOP`, and that was MEASURED
        rather than assumed: with `O_NOFOLLOW` set, macOS evaluates
        `O_DIRECTORY` against the symlink itself, which is not a directory.
        Linux answers `ELOOP` for the same call. Either way it is a KERNEL
        refusal at open time, which is exactly what distinguishes it from the
        identity check's `RACED` after a successful open.
        """
        sub = self.root / "sub"
        sub.mkdir()
        classified = os.lstat("sub", dir_fd=self.fd)  # a REAL directory
        sub.rmdir()
        os.symlink(str(self.outside), sub)

        # MECHANISM, three ways: it really is a symlink, it really reaches the
        # host directory holding the secret, and dropping ONLY this guard
        # really does open that directory — so the assertion below is
        # measuring O_NOFOLLOW and not some unrelated refusal.
        self.assertTrue(stat.S_ISLNK(os.lstat("sub", dir_fd=self.fd).st_mode))
        self.assertIn("id_rsa", os.listdir(str(sub)))
        self.assertTrue(stat.S_ISDIR(classified.st_mode))
        followed = os.open("sub", walkmod._DIR_FLAGS & ~os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            self.assertNotEqual(os.fstat(followed).st_ino, classified.st_ino)
        finally:
            os.close(followed)

        got = walkmod._open_child_dir(self.fd, "sub", "sub/", classified, self.root_dev)
        self.assertIsInstance(got, walkmod._Refusal)
        self.assertIn(
            got.reason,
            {"ENOTDIR", "ELOOP"},
            "the descent followed the symlink; only the identity check refused it",
        )
        self.assertNotEqual(got.reason, walkmod._RACED)

    # -- the two independent implementations of `.git` opacity --

    def test_visit_refuses_to_treat_a_non_metadata_object_as_metadata(self):
        """Mutation W11c (drop the `st_ino`/`st_dev` half of `_visit`'s `.git`
        check) SURVIVED, because `snapshot_tree`'s outer prefix-skip hides any
        top-level `.git` by name before `_visit` is ever consulted. The layer
        is real all the same: `_visit`'s rule is "opaque because it IS the
        repository's metadata", and without identity it degrades to "opaque
        because it is CALLED .git" — a rule the sandbox can satisfy, since a
        linked worktree's `.git` is an ordinary file it can overwrite.

        Asserted at `_visit`, which owns the decision, precisely because the
        outer skip masks it one level up. If that skip is ever removed as
        redundant, this layer becomes load-bearing with no warning.
        """
        meta = self.root / ".git"
        meta.write_text("SANDBOX-AUTHORED, not the repository's metadata\n")
        stale = (-1, -1)  # the identity the real metadata entry HAD

        got = walkmod._visit(
            self.fd,
            ".git",
            ".git",
            lambda p: False,
            self.root_dev,
            walkmod._Budget(1 << 20),
            stale,
        )
        self.assertIsNotNone(
            got.entry, "a sandbox-authored file vanished as 'metadata'"
        )
        self.assertIn(b"SANDBOX-AUTHORED", got.entry.data)

    # -- §6.5 rule 2: stay on the root's device --

    def test_a_file_on_another_device_is_refused_as_XDEV(self):
        """`walk._XDEV` had NO test at all — deleting the check in
        `_read_regular` left the whole suite green, and the token appeared in
        the suite only as a string inside a `_STABLE_REASONS` list.

        Asserted by handing the guard a `root_dev` that is not this object's
        device, which is exactly the comparison it makes. A real second
        filesystem would be a truer fixture but would mean creating and
        mounting a volume from a unit test; the guard compares two integers,
        so this varies the one that matters, and the control below holds it
        equal to prove the refusal comes from THIS check and nothing else.
        """
        victim = self.root / "victim"
        victim.write_bytes(b"content\n")
        classified = os.lstat("victim", dir_fd=self.fd)

        refused = walkmod._read_regular(
            self.fd,
            "victim",
            "victim",
            classified,
            classified.st_dev + 1,  # "the root is on some other device"
            walkmod._Budget(1 << 20),
        )
        self.assertIsInstance(refused, walkmod._Refusal)
        self.assertEqual(refused.reason, walkmod._XDEV)
        self.assertTrue(refused.terminal, "a device boundary cannot be retried away")

        # CONTROL: the very same call with the real device READS the file, so
        # the refusal above is the device check and not an unrelated failure.
        allowed = walkmod._read_regular(
            self.fd,
            "victim",
            "victim",
            classified,
            classified.st_dev,
            walkmod._Budget(1 << 20),
        )
        self.assertIsInstance(allowed, Entry)
        self.assertEqual(allowed.data, b"content\n")

    def test_a_directory_on_another_device_is_refused_as_XDEV(self):
        """The descent carries its own copy of the check, and its own
        mutation: dropping either one alone left the suite green."""
        sub = self.root / "sub"
        sub.mkdir()
        classified = os.lstat("sub", dir_fd=self.fd)

        refused = walkmod._open_child_dir(
            self.fd, "sub", "sub/", classified, classified.st_dev + 1
        )
        self.assertIsInstance(refused, walkmod._Refusal)
        self.assertEqual(refused.reason, walkmod._XDEV)

        allowed = walkmod._open_child_dir(
            self.fd, "sub", "sub/", classified, classified.st_dev
        )
        self.assertIsInstance(allowed, walkmod._Frame)
        assert isinstance(allowed, walkmod._Frame)
        os.close(allowed.fd)

    def test_reclassification_is_bounded_and_the_path_is_then_recorded(self):
        """SF-8. `_RECLASSIFY_ATTEMPTS` was unpinned — mutation `W20`
        survived. The bound is what stops a writer that keeps losing the race
        for the walk from spinning it forever, and it is only half the
        property: giving up must RECORD the path, because "indistinguishable
        from never existed" is the under-show §6.5 calls dangerous.

        The recorded reason is the LAST refusal seen (`ENOENT` here), not the
        generic `_RACED` — `_RACED` is the initialiser and survives only when
        no attempt produced a reason of its own. That is the more informative
        of the two and is asserted as such rather than papered over.
        """
        (self.root / "flaky").write_bytes(b"x\n")
        real_lstat = os.lstat
        attempts = {"n": 0}

        def racing_lstat(path, *args, **kwargs):
            # ENOENT is deliberately retryable — a racing writer produces it
            # and a later attempt can still win — so this models a writer
            # that never settles rather than a permanent failure.
            if path == "flaky" and kwargs.get("dir_fd") is not None:
                attempts["n"] += 1
                raise OSError(errno.ENOENT, "vanished")
            return real_lstat(path, *args, **kwargs)

        with unittest.mock.patch.object(walkmod.os, "lstat", racing_lstat):
            snap = snapshot_tree(str(self.root), lambda p: False)

        self.assertEqual(
            attempts["n"],
            walkmod._RECLASSIFY_ATTEMPTS,
            "the walk did not stop at its own retry bound",
        )
        self.assertEqual(
            [(u.path, u.reason) for u in snap.unreadable],
            [("flaky", "ENOENT")],
            "the walk gave up on a path and said nothing",
        )
        self.assertNotIn("flaky", snap.entries)

    def test_XDEV_is_a_stable_reason_so_it_moves_the_hash(self):
        """A device boundary is a property of the tree, not a moment inside
        somebody else's race, so it must be one of the reasons `tree_hash`
        keeps — otherwise a subtree that disappears across a mount point is
        invisible to the no-progress detector as well as to the diff."""
        self.assertIn(walkmod._XDEV, walkmod._STABLE_REASONS)
        moved = tree_hash(TreeSnapshot({}, (Unreadable("mnt/", walkmod._XDEV),)))
        self.assertNotEqual(moved, tree_hash(TreeSnapshot({})))

    def test_the_outer_prefix_skip_keeps_git_opaque_when_identity_no_longer_matches(
        self,
    ):
        """Mutation W18 (drop the outer prefix-skip) SURVIVED, because
        `_visit`'s own check covers the ordinary case. Its unique job is the
        case `_visit` cannot cover: the recorded metadata identity is read
        ONCE, at the top of the walk, and the sandbox can replace the entry
        afterwards — after which `_visit` sees a name it matches and an
        identity it does not, and would show the file.

        `_meta_identity` is interposed to produce that post-swap state
        deterministically rather than by racing a real swap; the test above is
        the control proving `_visit` really does return an entry in it.
        """
        (self.root / "app.py").write_text("print(1)\n")
        (self.root / ".git").write_text("gitdir: /elsewhere/worktrees/wt\n")

        with unittest.mock.patch.object(
            walkmod, "_meta_identity", return_value=(-1, -1)
        ):
            snap = snapshot_tree(str(self.root), lambda p: False)

        self.assertIn("app.py", snap.entries)
        self.assertNotIn(".git", snap.entries)
        self.assertEqual([u.path for u in snap.unreadable], [])


# ---------------------------------------------------------------------------
# Task 4: THE REVIEW DIFF.
#
# `unified_diff` renders the only artifact a human sees before deciding whether
# to apply an untrusted model's work, so every failure mode below is a way to
# make a real change absent from — or unreadable in — that artifact. Spec
# §5.1/§6.5: over-showing is noise, UNDER-showing defeats the design.
# ---------------------------------------------------------------------------


# One spill dir for the whole file rather than a fresh `tempfile.mkdtemp()`
# per `_diff()`/`unified_diff()` call (there are dozens across this file's
# tests) — each spill write picks its own unique filename inside it, so
# sharing is safe, and this way there is exactly one directory for
# `tearDownModule` to remove instead of dozens left behind after every run.
_SPILL_DIR: str | None = None


def setUpModule() -> None:
    global _SPILL_DIR
    _SPILL_DIR = tempfile.mkdtemp(prefix="ca-test-security-spill-")


def tearDownModule() -> None:
    if _SPILL_DIR:
        shutil.rmtree(_SPILL_DIR, ignore_errors=True)


def _base(**files: bytes) -> BaseTree:
    ents = {p: Entry(p, "file", d) for p, d in files.items()}
    return BaseTree(entries=ents, ignore=lambda p: False, tracked=frozenset(ents))


def _base_of(*entries: Entry) -> BaseTree:
    ents = {e.path: e for e in entries}
    return BaseTree(entries=ents, ignore=lambda p: False, tracked=frozenset(ents))


def _diff(base: BaseTree, snap: TreeSnapshot, max_bytes: int = 1 << 20) -> DiffResult:
    return unified_diff(base, snap, max_bytes=max_bytes, spill_dir=_SPILL_DIR)


def _starting(text: str, prefix: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(prefix)]


class DiffStructureInjection(unittest.TestCase):
    """Pass 4: a symlink TARGET is attacker-controlled and round-trips
    newlines + '--- a/' bytes. It must render like git's 120000 blob:
    every target line '+'-prefixed, so no forged header lands in column 0."""

    def test_forged_header_in_symlink_target_cannot_reach_column_zero(self):
        forged = "harmless\n--- a/etc/shadow\n+++ b/etc/shadow\n@@ -0,0 +1 @@\n+root:INJECTED"
        snap = TreeSnapshot({"leak": Entry("leak", "symlink", forged.encode())})
        d = unified_diff(_base(), snap, max_bytes=1 << 20, spill_dir=_SPILL_DIR)
        for line in d.text.splitlines():
            if line.startswith(("--- ", "+++ ", "@@ ")):
                # only OUR headers may appear at column 0
                self.assertTrue(
                    line.startswith(
                        ("--- /dev/null", "--- a/leak", "+++ b/leak", "@@ -0,0 +")
                    ),
                    f"forged header reached column 0: {line!r}",
                )
        self.assertIn("+--- a/etc/shadow", d.text)  # rendered as content, '+'-prefixed

    def test_the_at_counts_match_the_lines_actually_emitted(self):
        """The other half of §10 test 4b. A '+'-prefix with a LYING hunk count
        is still structure injection: a patch tool — or a reader counting
        lines — resynchronises on the count, and everything after the short
        count reads as the next file's content."""
        snap = TreeSnapshot({"leak": Entry("leak", "symlink", b"a\nb\nc\nd\ne")})
        d = _diff(_base(), snap)
        self.assertEqual(_starting(d.text, "@@ "), ["@@ -0,0 +1,5 @@"])
        added = [
            ln
            for ln in d.text.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")
        ]
        self.assertEqual(len(added), 5)

    def test_forged_header_in_a_FILENAME_cannot_reach_column_zero(self):
        """The same injection through the OTHER attacker-controlled string.

        A path is as controllable as a symlink target — `os.listdir` returns
        whatever bytes are on disk and `ls-tree -z` carries them through — and
        a newline in a filename lands the rest of the name in column 0 of the
        diff, where a forged `--- a/…` block reads as a whole extra file. The
        forged block is what a reviewer would scrutinise while the REAL change
        two files down is dismissed as part of it, so this is under-showing by
        misdirection. git's answer is to C-quote such a path onto one line.
        """
        evil = "notes.txt\n--- a/etc/shadow\n+++ b/etc/shadow\n@@ -1 +1 @@\n-root:x\n+root:INJECTED"
        snap = TreeSnapshot({evil: Entry(evil, "file", b"benign\n")})
        d = _diff(_base(), snap)
        # exactly ONE file is being described, so exactly one of each header
        self.assertEqual(len(_starting(d.text, "--- ")), 1)
        self.assertEqual(len(_starting(d.text, "+++ ")), 1)
        self.assertEqual(len(_starting(d.text, "@@ ")), 1)
        self.assertEqual(len(_starting(d.text, "diff --git ")), 1)
        self.assertNotIn("\n--- a/etc/shadow", d.text)
        self.assertIn("\\n", d.text)  # the newline is escaped, git-style
        # the raw path still reaches the caller intact, unquoted
        self.assertEqual(d.changed_files, [evil])


class NoNewlineAtEofCannotSwallowTheNextHeader(unittest.TestCase):
    """A file whose last line has no trailing newline used to run straight
    into the NEXT file's header, taking it out of column 0 — one file with no
    final newline hid the following file's entire section from a reader
    scanning for headers. git's `\\ No newline at end of file` marker is what
    keeps every header on its own line."""

    def test_a_file_without_a_trailing_newline_does_not_eat_the_next_file(self):
        snap = TreeSnapshot(
            {
                "a.txt": Entry("a.txt", "file", b"no trailing newline"),
                "b.txt": Entry("b.txt", "file", b"second file\n"),
            }
        )
        d = _diff(_base(), snap)
        self.assertIn("\\ No newline at end of file", d.text)
        self.assertIn("\ndiff --git a/b.txt b/b.txt\n", d.text)
        self.assertIn("\n+++ b/b.txt\n", d.text)
        self.assertIn("+second file", d.text)


class ModeBitsAreVisible(unittest.TestCase):
    """`chmod +x` is a real, persistent change to code a human already
    approved: the same bytes become something the shell will RUN. It carries
    no content delta at all, so a diff keyed on content alone shows nothing —
    the quietest under-show in the module."""

    def test_chmod_plus_x_with_identical_content_appears_in_the_diff(self):
        same = b"#!/bin/sh\necho hello\n"
        snap = TreeSnapshot(
            {"tool.sh": Entry("tool.sh", "file", same, executable=True)}
        )
        d = _diff(_base(**{"tool.sh": same}), snap)
        self.assertIn("tool.sh", d.changed_files)
        self.assertIn("old mode 100644", d.text)
        self.assertIn("new mode 100755", d.text)

    def test_dropping_the_execute_bit_is_shown_too(self):
        same = b"#!/bin/sh\n"
        base = _base_of(Entry("tool.sh", "file", same, executable=True))
        snap = TreeSnapshot({"tool.sh": Entry("tool.sh", "file", same)})
        d = _diff(base, snap)
        self.assertIn("old mode 100755", d.text)
        self.assertIn("new mode 100644", d.text)

    def test_an_already_executable_file_is_not_a_false_positive(self):
        """The reason this had to land on BOTH sides at once: a walk that
        reports the execute bit against a base tree that does not would call
        every 100755 file in the repo a mode change, and a diff that cries
        wolf on every entry is one a reviewer stops reading."""
        same = b"#!/bin/sh\n"
        base = _base_of(Entry("tool.sh", "file", same, executable=True))
        snap = TreeSnapshot(
            {"tool.sh": Entry("tool.sh", "file", same, executable=True)}
        )
        d = _diff(base, snap)
        self.assertEqual(d.changed_files, [])
        self.assertEqual(d.text, "")

    def test_a_new_file_states_its_mode(self):
        snap = TreeSnapshot(
            {"run.sh": Entry("run.sh", "file", b"id\n", executable=True)}
        )
        d = _diff(_base(), snap)
        self.assertIn("new file mode 100755", d.text)


class GateCannotBeBlinded(unittest.TestCase):
    def test_tracked_file_replaced_by_special_file_surfaces_as_deletion(self):
        base = _base(**{"foo.py": b"print(1)\n"})
        snap = TreeSnapshot({})  # walk skipped the FIFO that replaced foo.py
        d = unified_diff(base, snap, max_bytes=1 << 20, spill_dir=_SPILL_DIR)
        self.assertIn("foo.py", d.changed_files)
        self.assertIn("--- a/foo.py", d.text)
        self.assertIn("+++ /dev/null", d.text)

    def test_new_untracked_file_appears(self):
        snap = TreeSnapshot(
            {
                "tests/test_new.py": Entry(
                    "tests/test_new.py", "file", b"def test_x(): pass\n"
                )
            }
        )
        d = unified_diff(_base(), snap, max_bytes=1 << 20, spill_dir=_SPILL_DIR)
        self.assertIn("tests/test_new.py", d.changed_files)
        self.assertIn("+def test_x(): pass", d.text)

    def test_a_tracked_file_replaced_by_a_symlink_shows_as_a_typechange(self):
        """§10 test 4c's typechange half. The dangerous shape is a config or
        key file swapped for a symlink pointing somewhere else: the CONTENT
        the reviewer reads (`/etc/passwd`) looks like a one-line edit unless
        the 100644 -> 120000 transition is stated."""
        base = _base(**{"config.json": b'{"a": 1}\n'})
        snap = TreeSnapshot(
            {"config.json": Entry("config.json", "symlink", b"/etc/passwd")}
        )
        d = _diff(base, snap)
        self.assertIn("config.json", d.changed_files)
        self.assertIn("old mode 100644", d.text)
        self.assertIn("new mode 120000", d.text)
        self.assertIn('-{"a": 1}', d.text)
        self.assertIn("+/etc/passwd", d.text)

    def test_a_symlink_replaced_by_a_file_is_a_typechange_too(self):
        base = _base_of(Entry("link", "symlink", b"app.py"))
        snap = TreeSnapshot({"link": Entry("link", "file", b"import os\n")})
        d = _diff(base, snap)
        self.assertIn("old mode 120000", d.text)
        self.assertIn("new mode 100644", d.text)

    def test_content_identical_across_a_typechange_is_still_shown(self):
        """The nastiest shape of all: a file containing the text `app.py`
        replaced by a SYMLINK to app.py. Both sides hold identical bytes, so
        a content-only comparison emits nothing whatsoever."""
        base = _base(**{"link": b"app.py"})
        snap = TreeSnapshot({"link": Entry("link", "symlink", b"app.py")})
        d = _diff(base, snap)
        self.assertIn("link", d.changed_files)
        self.assertIn("old mode 100644", d.text)
        self.assertIn("new mode 120000", d.text)


class UnreadablePathsReachTheHuman(unittest.TestCase):
    """`TreeSnapshot.unreadable` exists because `chmod 000 backdoor.py` made a
    file VANISH from the snapshot with no signal at all. Populating the field
    only moved the silence one layer up: a record nothing renders leaves the
    human reading a diff in which those paths simply do not exist."""

    def test_unreadable_paths_are_stated_in_the_text_and_carried_on_the_result(self):
        snap = TreeSnapshot(
            {"visible.txt": Entry("visible.txt", "file", b"hi\n")},
            (Unreadable("backdoor.py", "EACCES"), Unreadable("src/", "EACCES")),
        )
        d = _diff(_base(), snap)
        # structured, for a gate that must fail closed without parsing text
        self.assertEqual(d.unreadable, snap.unreadable)
        # and in the text, for the human who only reads the diff
        self.assertIn("2 path(s) could not be read", d.text)
        self.assertIn("backdoor.py", d.text)
        self.assertIn("src/", d.text)
        self.assertIn("EACCES", d.text)

    def test_the_warning_is_first_so_the_size_cap_cannot_cut_it_off(self):
        """Truncation removes the TAIL. A warning rendered after the file
        sections is exactly the thing an oversized diff drops, and an attacker
        picks the diff's size."""
        snap = TreeSnapshot(
            {"big.txt": Entry("big.txt", "file", b"x" * 5000)},
            (Unreadable("backdoor.py", "EACCES"),),
        )
        d = _diff(_base(), snap, max_bytes=400)
        self.assertTrue(d.truncated)
        self.assertTrue(d.text.startswith("coding_agent: WARNING"), d.text[:80])
        self.assertIn("1 path(s) could not be read", d.text)
        self.assertIn("backdoor.py", d.text)

    def test_a_flood_of_unreadable_paths_cannot_displace_the_diff(self):
        """The block is bounded for the same reason it is first: an unbounded
        list of sandbox-chosen names would push the real change out of the
        visible window. The COUNT stays exact and the full list stays on the
        result, so bounding the render loses nothing."""
        many = tuple(Unreadable(f"hidden{i:04d}.py", "EACCES") for i in range(500))
        snap = TreeSnapshot({"real.py": Entry("real.py", "file", b"import os\n")}, many)
        d = _diff(_base(), snap)
        self.assertIn("500 path(s) could not be read", d.text)
        self.assertIn("+import os", d.text)
        self.assertLess(d.text.index("+import os"), 2000)
        self.assertEqual(len(d.unreadable), 500)

    def test_a_clean_snapshot_says_nothing(self):
        snap = TreeSnapshot({"a.py": Entry("a.py", "file", b"x = 1\n")})
        d = _diff(_base(), snap)
        self.assertNotIn("could not be read", d.text)
        self.assertEqual(d.unreadable, ())

    def test_an_unreadable_path_cannot_forge_a_diff_header_either(self):
        evil = "ok.py\n--- a/etc/shadow\n+++ b/etc/shadow"
        snap = TreeSnapshot({}, (Unreadable(evil, "EACCES"),))
        d = _diff(_base(), snap)
        self.assertEqual(_starting(d.text, "--- "), [])
        self.assertEqual(_starting(d.text, "+++ "), [])
        for line in d.text.splitlines():
            self.assertTrue(line == "" or line.startswith("coding_agent:"), repr(line))


def _apply_check(repo: Path, patch_text: str) -> tuple[int, str]:
    """`git apply --check --numstat` of a patch against a REAL repository.

    The point of this feature is that the emitted text is a patch a human may
    act on, so "is this a valid, applicable deletion?" has exactly one honest
    answer: ask git. Returns (rc, numstat stdout).
    """
    patch = repo.parent / "emitted.patch"
    patch.write_text(patch_text, encoding="utf-8", errors="surrogateescape")
    proc = subprocess.run(
        ["git", "apply", "--check", "--numstat", str(patch)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,  # a REFUSED patch is a valid answer, not an error
    )
    return proc.returncode, proc.stdout


class UnreadablePathsAreNeverRenderedAsDeletions(unittest.TestCase):
    """MF-1, final review 2026-08-19. THE regression this class exists for.

    `unified_diff` compares `base ∪ snapshot` and rendered base-present /
    snapshot-absent as `deleted file mode`. That was sound until the pass-6
    read caps landed: a path is now also absent when the walk REFUSED to read
    it. The result was a diff that made affirmatively false statements about
    files nobody touched, and — for text content — an APPLICABLE patch:

        9 MiB untouched tracked file -> 1,887,436-line removal hunk,
        `git apply --check` rc = 0, and the model's real one-line edit pushed
        out of the returned window by the size cap.

    Adversarially, 33 sparse files created in 0.00s and 164 KB of real disk
    drained the 256 MiB snapshot budget and converted 37 of 40 tracked files
    into deletions that did not happen — the gate being blinded, which is the
    single failure this whole design exists to prevent (§6.5: over-showing is
    noise, UNDER-showing is the danger).

    Both directions are pinned here. Suppressing the deletion is only half a
    fix if a GENUINE deletion stops rendering too.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "-q", "-b", "main"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "t@e.com"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "T"], check=True
        )

    def _commit_and_copy(self) -> tuple[BaseTree, Path]:
        """Commit the repo, then hand back a `.git`-less COPY to walk.

        Separate directories, as in production: git only ever runs against the
        repo, and only the copy is ever walked or modified.
        """
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-q", "-m", "base"], check=True
        )
        work = self.root / "work"
        shutil.copytree(
            self.repo, work, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        return read_base_tree(str(self.repo), "HEAD"), work

    def _diff_of(self, base: BaseTree, work: Path, max_bytes: int = 512 * 1024):
        snap = snapshot_tree(str(work), make_ignore(base))
        spill = self.root / "spill"
        spill.mkdir(exist_ok=True)
        return snap, unified_diff(base, snap, max_bytes=max_bytes, spill_dir=str(spill))

    # -- the non-adversarial trigger: any tracked file over _MAX_FILE_BYTES --

    def test_an_oversize_untouched_tracked_file_is_not_reported_as_deleted(self):
        (self.repo / "small.py").write_text("x = 1\n")
        (self.repo / "big.txt").write_text("line\n" * (9 * 1024 * 1024 // 5))
        base, work = self._commit_and_copy()
        (work / "small.py").write_text("x = 2\n")

        snap, d = self._diff_of(base, work)

        # The walk really did refuse it — otherwise this test proves nothing.
        self.assertEqual(
            [(u.path, u.reason) for u in snap.unreadable], [("big.txt", "OVERSIZE")]
        )
        self.assertNotIn("big.txt", snap.entries)
        # ...and the diff does not call that a deletion.
        self.assertNotIn("deleted file mode", d.text)
        self.assertNotIn("--- a/big.txt", d.text)
        # The path is still CONSPICUOUS — twice, and marked as not-a-deletion.
        self.assertIn("NOT COMPARED", d.text)
        self.assertIn("NOT deleted", d.text)
        # The HEAD block's own marker, asserted on that line specifically: it
        # is what separates "a committed file I cannot compare" from "the
        # model made something I cannot read", and a reader who sees a tracked
        # path listed as unreadable and finds no section below has to be told
        # which reading is true.
        head = [
            ln for ln in d.text.splitlines() if ln.startswith('coding_agent:   "big')
        ]
        self.assertEqual(len(head), 1, d.text)
        self.assertIn("tracked in the base tree", head[0])
        # And the model's real edit is back in the returned window: before the
        # fix, 1.9 million removal hunks displaced it entirely.
        self.assertIn("+x = 2", d.text)
        self.assertFalse(d.truncated)

    def test_the_emitted_patch_cannot_delete_an_unreadable_file(self):
        """`git apply --check` is the only honest answer to "would this
        really delete it?" — and before the fix the answer was yes, rc=0."""
        (self.repo / "small.py").write_text("x = 1\n")
        (self.repo / "big.txt").write_text("line\n" * (9 * 1024 * 1024 // 5))
        base, work = self._commit_and_copy()
        (work / "small.py").write_text("x = 2\n")

        _, d = self._diff_of(base, work)
        rc, numstat = _apply_check(self.repo, d.text)

        self.assertEqual(rc, 0, numstat)
        self.assertNotIn("big.txt", numstat)
        self.assertIn("small.py", numstat)  # the real edit still applies

    def test_a_genuinely_deleted_file_still_renders_as_deleted(self):
        """The other direction. Trading one under-show for another would be a
        worse bug than the one being fixed: a real deletion that renders as
        nothing is exactly the gate-blinding this suppression could cause."""
        (self.repo / "keep.py").write_text("x = 1\n")
        (self.repo / "gone.py").write_text("y = 2\n")
        base, work = self._commit_and_copy()
        (work / "gone.py").unlink()

        snap, d = self._diff_of(base, work)

        self.assertEqual(snap.unreadable, ())  # nothing was refused
        self.assertIn("deleted file mode 100644", d.text)
        self.assertIn("gone.py", d.changed_files)
        rc, numstat = _apply_check(self.repo, d.text)
        self.assertEqual(rc, 0, numstat)
        self.assertIn("gone.py", numstat)  # the deletion is real and applies

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 directory anyway")
    def test_an_unreadable_directory_does_not_delete_its_whole_subtree(self):
        """The same defect one level up, and the reason the suppression is a
        PREFIX test. `Unreadable.path` carries a trailing slash for a
        directory; one `chmod 000` on a directory hid every base path under it
        and rendered all of them as deletions."""
        (self.repo / "top.py").write_text("a = 1\n")
        (self.repo / "sub").mkdir()
        (self.repo / "sub" / "one.py").write_text("b = 1\n")
        (self.repo / "sub" / "two.py").write_text("c = 1\n")
        base, work = self._commit_and_copy()
        os.chmod(work / "sub", 0o000)
        self.addCleanup(os.chmod, work / "sub", 0o755)

        snap, d = self._diff_of(base, work)

        self.assertEqual([u.path for u in snap.unreadable], ["sub/"])
        self.assertNotIn("deleted file mode", d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 2)  # one.py and two.py
        self.assertIn("sub/one.py", d.text)
        self.assertIn("sub/two.py", d.text)
        self.assertEqual(d.changed_files, [])

    # -- the adversarial trigger: drain the whole-snapshot budget --

    def test_draining_the_read_budget_fabricates_no_deletions(self):
        """33 sparse files, 0.00s to create, 164 KB of real disk. Before the
        fix this turned 37 of 40 tracked files into deletions that did not
        happen while removing the model's actual edit from the diff."""
        for i in range(40):
            (self.repo / f"mod{i:02d}.py").write_text(f"v = {i}\n")
        base, work = self._commit_and_copy()
        (work / "mod07.py").write_text("v = 7  # THE REAL EDIT\n")
        for i in range(33):  # sort first: '!' precedes every letter
            with open(work / f"!{i:03d}.dat", "wb") as fh:
                fh.truncate((8 << 20) - 1)

        snap, d = self._diff_of(base, work)

        # The attack really landed — the budget really was drained.
        drained = [u for u in snap.unreadable if u.reason == "BUDGET"]
        self.assertGreater(len(drained), 30, [u.path for u in snap.unreadable])
        self.assertNotIn("mod07.py", snap.entries)
        # ...and produced ZERO false deletions.
        self.assertNotIn("deleted file mode", d.text)
        rc, numstat = _apply_check(self.repo, d.text)
        self.assertEqual(rc, 0, numstat)
        for i in range(40):
            self.assertNotIn(f"mod{i:02d}.py", numstat)
        # The path whose edit cannot be shown is named, and named honestly.
        self.assertIn("mod07.py", d.text)
        self.assertNotIn("mod07.py", d.changed_files)

    def test_an_exhausted_budget_is_its_own_loud_signal(self):
        """A drained budget and a changed file are different statements, and
        only the first tells the
        reviewer that an UNKNOWN number of changes are missing. One OVERSIZE
        file costs one named file; an exhausted budget costs everything the
        walk had not reached yet."""
        snap = TreeSnapshot(
            {"real.py": Entry("real.py", "file", b"x = 1\n")},
            (Unreadable("a.dat", "BUDGET"), Unreadable("b.py", "BUDGET")),
        )
        d = _diff(_base(), snap)
        self.assertIn("read budget was EXHAUSTED", d.text)
        self.assertIn("THIS DIFF IS INCOMPLETE", d.text)
        self.assertIn("2 path(s) hit it", d.text)
        # First, so the size cap cannot cut it off.
        self.assertTrue(d.text.startswith("coding_agent: WARNING: the whole"))

    def test_an_oversize_file_alone_does_not_claim_the_budget_ran_out(self):
        """The distinct signal is worthless if it fires on the ordinary case."""
        snap = TreeSnapshot(
            {"real.py": Entry("real.py", "file", b"x = 1\n")},
            (Unreadable("big.bin", "OVERSIZE"),),
        )
        d = _diff(_base(), snap)
        self.assertNotIn("EXHAUSTED", d.text)
        self.assertIn("1 path(s) could not be read", d.text)

    # -- the structured half: what a programmatic gate sees --

    def test_an_uncomparable_path_is_not_asserted_to_have_changed(self):
        """`changed_files` is a POSITIVE assertion, consumed by callers that
        never read the prose. A path nobody could read did not necessarily
        change, and listing it makes the field untrustworthy in the other
        direction — 69 entries of which one was real, in the demonstrated
        drain. Completeness is carried by `unreadable`, which is non-empty
        exactly when this fires, so a gate that fails closed still can."""
        snap = TreeSnapshot(
            {"real.py": Entry("real.py", "file", b"x = 2\n")},
            (Unreadable("hidden.py", "EACCES"),),
        )
        base = _base(**{"real.py": b"x = 1\n", "hidden.py": b"secret\n"})
        d = _diff(base, snap)
        self.assertEqual(d.changed_files, ["real.py"])
        self.assertEqual(len(d.unreadable), 1)  # the gate's fail-closed signal
        self.assertIn("hidden.py", d.text)  # still visible to the human

    def test_an_unreadable_path_absent_from_the_base_is_still_reported(self):
        """A file the model CREATED and made unreadable has no base side, so
        it can never be mistaken for a deletion — but it must still reach the
        human, and it must not be marked as tracked."""
        snap = TreeSnapshot({}, (Unreadable("new_backdoor.py", "EACCES"),))
        d = _diff(_base(), snap)
        self.assertIn("new_backdoor.py", d.text)
        self.assertNotIn("tracked in the base tree", d.text)
        self.assertNotIn("NOT COMPARED", d.text)

    def test_the_not_compared_marker_cannot_forge_diff_structure(self):
        """Same injection surface as every other path this module renders: the
        marker embeds a base path, and a base path can hold a newline."""
        evil = "ok.py\ndiff --git a/etc/shadow b/etc/shadow\ndeleted file mode 100644"
        base = _base(**{evil: b"x\n"})
        snap = TreeSnapshot({}, (Unreadable(evil, "EACCES"),))
        d = _diff(base, snap)
        self.assertEqual(_starting(d.text, "diff --git "), [])
        self.assertEqual(_starting(d.text, "deleted file mode"), [])
        for line in d.text.splitlines():
            self.assertTrue(line == "" or line.startswith("coding_agent:"), repr(line))


class DiffSizeCap(unittest.TestCase):
    def test_binary_is_annotated_not_embedded(self):
        snap = TreeSnapshot(
            {
                "blob.bin": Entry("blob.bin", "file", b"\x00\x01\x02" * 100),
            }
        )
        d = _diff(_base(), snap)
        self.assertIn("Binary files differ: blob.bin", d.text)
        self.assertIn("300 bytes", d.text)
        self.assertNotIn("\x00", d.text)

    def test_large_text_truncates_with_a_spill_file(self):
        big = b"x" * 5000
        snap = TreeSnapshot({"big.txt": Entry("big.txt", "file", big)})
        d = unified_diff(_base(), snap, max_bytes=1000, spill_dir=_SPILL_DIR)
        self.assertTrue(d.truncated)
        self.assertIsNotNone(d.full_path)
        self.assertGreater(os.path.getsize(d.full_path), 1000)

    def test_truncation_cannot_hide_THAT_a_later_file_changed(self):
        """This is the brief's size-cap test, split in two because the single
        bundled version could not pass — and the reason it could not is a real
        under-show, not a test artifact.

        'big.txt' sorts before 'blob.bin', so 5000 bytes of `x` displaced the
        binary annotation past a 1000-byte cap and the second file vanished
        from the review entirely. One `dd` of an early-sorting file hides every
        change after it. The cap now truncates DETAIL only: the complete list
        of changed paths, with mode transitions, is appended after the marker.
        """
        snap = TreeSnapshot(
            {
                "blob.bin": Entry("blob.bin", "file", b"\x00\x01\x02" * 100),
                "big.txt": Entry("big.txt", "file", b"x" * 5000),
            }
        )
        d = unified_diff(_base(), snap, max_bytes=1000, spill_dir=_SPILL_DIR)
        self.assertTrue(d.truncated)
        self.assertEqual(d.changed_files, ["big.txt", "blob.bin"])
        tail = d.text.split("[... diff truncated")[1]
        self.assertIn("2 file(s) changed", tail)
        self.assertIn("blob.bin", tail)
        self.assertIn("- -> 100644", tail)
        self.assertNotIn("\x00", d.text)

    def test_the_appended_list_is_bounded_and_states_the_true_count(self):
        """Bounded for the same reason it exists: an unbounded list of
        sandbox-chosen names is itself a way to flood the reviewer."""
        snap = TreeSnapshot(
            {f"f{i:04d}.py": Entry(f"f{i:04d}.py", "file", b"x\n") for i in range(500)}
        )
        d = unified_diff(_base(), snap, max_bytes=1000, spill_dir=_SPILL_DIR)
        self.assertIn("500 file(s) changed", d.text)
        self.assertIn("and 400 more", d.text)
        self.assertEqual(len(d.changed_files), 500)

    def test_the_spill_file_holds_the_untruncated_diff(self):
        snap = TreeSnapshot({"big.txt": Entry("big.txt", "file", b"line\n" * 400)})
        d = _diff(_base(), snap, max_bytes=500)
        self.assertTrue(d.truncated)
        with open(d.full_path, encoding="utf-8") as fh:
            full = fh.read()
        self.assertEqual(full.count("+line"), 400)
        self.assertIn("full diff at", d.text)
        self.assertIn(d.full_path, d.text)

    def test_an_untruncated_diff_writes_no_spill_file(self):
        snap = TreeSnapshot({"a.py": Entry("a.py", "file", b"x = 1\n")})
        d = _diff(_base(), snap)
        self.assertFalse(d.truncated)
        self.assertIsNone(d.full_path)


if __name__ == "__main__":
    unittest.main()
