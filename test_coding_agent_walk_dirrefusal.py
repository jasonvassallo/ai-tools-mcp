#!/usr/bin/env python3
"""Regression tests for the pass-8 gate-blinding defect (`F-1`).

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_walk_dirrefusal.py -q

THE DEFECT. `MF-1` stopped `unified_diff` rendering a path the walk REFUSED to
read as `deleted file mode`, and for a directory it did so with a prefix test
keyed on the trailing slash `Unreadable.path` carries. The eighth adversarial
pass showed the slash is not always there to key on.

`_visit` learns a child's type from the `lstat` that classifies it, and sets
the slash only after that call succeeds. A directory that is readable but not
SEARCHABLE — any mode with `r` and without `x`: `0400 0444 0600 0644 0666` —
lets the parent `open` and `os.listdir` succeed and denies `fstatat` on every
child. The failure therefore lands on the very `lstat` that would have said
"directory", `EACCES` is (correctly) terminal, and the record is emitted with
no slash. `_not_compared`'s `elif u.path in base.entries` never matches a
directory, nothing beneath it was blocked, and every tracked file under it
rendered as a deletion that `git apply --check` accepted.

`0644` is the default file mode. A stray `chmod -R 644 .` reaches this without
anybody being hostile.

The shipped regression test pinned `chmod 000`, which fails at `open` — a point
where the walk ALREADY knows the entry is a directory. It exercised the branch
that worked and never the branch that did not.

THE FIX is in `_not_compared`: every record covers `path` AND the `path + "/"`
prefix, whether or not the walk was able to learn it was a directory. Tests
below pin both directions, because trading a false deletion for a vanished real
one would be the worse bug.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent.basetree import BaseTree, make_ignore, read_base_tree
from coding_agent.walk import (
    Entry,
    TreeSnapshot,
    Unreadable,
    snapshot_tree,
    unified_diff,
)

# Every directory mode that is readable but not searchable. Measured, not
# guessed: see `_assert_untypable`, which fails the test if any of them stops
# producing the branch this file exists for.
UNTYPABLE_MODES = (0o400, 0o444, 0o600, 0o644, 0o666)

# Modes where the walk CAN classify the children, kept as the vacuity control:
# without them, a test asserting "the backdoor is not rendered as a deletion"
# would pass just as well against a fixture that never contained a backdoor.
SEARCHABLE_MODES = (0o500, 0o755)

_MAX_BYTES = 512 * 1024


def _restore_modes(root: Path) -> None:
    """Make `root` removable again.

    `os.walk` is top-down and yields a directory's names BEFORE descending into
    them, so relaxing each mode at the moment it is yielded is what lets the
    walk — and `shutil.rmtree` after it — reach anything underneath.
    """
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            try:
                os.chmod(os.path.join(dirpath, name), 0o755)
            except OSError:
                pass


class _RealRepo(unittest.TestCase):
    """A real git repository, and a `.git`-less COPY of it to walk.

    Separate directories, exactly as in production: git only ever runs against
    the repo (always as `git -C <repo>`, never with the worktree as cwd or
    gitdir — spec §6.5's security spine), and only the copy is ever walked or
    chmodded.

    `_new_case` gives every parameterised iteration its own repo and worktree,
    so a fixture left at `0o444` by one mode cannot leak into the next.
    """

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        # LIFO: this runs BEFORE the rmtree above, which cannot otherwise
        # descend through a directory a test left at 0o444.
        self.addCleanup(_restore_modes, self.root)
        self._cases = 0
        self.repo = self._new_case()

    def _new_case(self) -> Path:
        self._cases += 1
        self.repo = self.root / f"case{self._cases}" / "repo"
        self.repo.mkdir(parents=True)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "t@e.com")
        self._git("config", "user.name", "T")
        return self.repo

    def _git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *args], check=True)

    def _commit_and_copy(self) -> tuple[BaseTree, Path]:
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base")
        work = self.repo.parent / "work"
        shutil.copytree(
            self.repo, work, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        return read_base_tree(str(self.repo), "HEAD"), work

    def _diff_of(self, base: BaseTree, work: Path):
        snap = snapshot_tree(str(work), make_ignore(base))
        spill = self.repo.parent / "spill"
        spill.mkdir(exist_ok=True)
        return snap, unified_diff(
            base, snap, max_bytes=_MAX_BYTES, spill_dir=str(spill)
        )

    def _apply_check(self, patch_text: str) -> tuple[int, str]:
        """`git apply --check --numstat` against the REAL repository.

        "Would this patch really delete that file?" has exactly one honest
        answer, and it is git's. Callers MUST restore any restrictive mode on
        the fixture first: git cannot read through a `dr--r--r--` directory
        either, and the resulting non-zero rc would masquerade as the patch
        being refused when it is only the fixture getting in the way.
        """
        patch = self.repo.parent / "emitted.patch"
        patch.write_text(patch_text, encoding="utf-8", errors="surrogateescape")
        proc = subprocess.run(
            ["git", "apply", "--check", "--numstat", str(patch)],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
            check=False,  # a REFUSED patch is an answer, not an error
        )
        return proc.returncode, proc.stdout

    def _assert_untypable(self, parent: Path, child: str) -> None:
        """Prove the fixture exercises the lstat-failed branch, not chmod-000.

        Asserting the MECHANISM rather than inferring it from the outcome: this
        whole file is about two branches that produce similar-looking damage
        for different reasons, and a test that cannot tell them apart would
        report the wrong one as fixed.
        """
        fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.assertIn(child, os.listdir(fd))  # the name IS enumerable
            with self.assertRaises(OSError) as caught:
                os.lstat(child, dir_fd=fd)  # ...and cannot be classified
            self.assertEqual(caught.exception.errno, errno.EACCES)
        finally:
            os.close(fd)

    def _backdoor_case(self) -> tuple[BaseTree, Path]:
        """Tracked `outer/inner/{a,b,c}.py` + `top.py`, with `a.py` backdoored."""
        self._new_case()
        (self.repo / "outer" / "inner").mkdir(parents=True)
        for name in ("a", "b", "c"):
            (self.repo / "outer" / "inner" / f"{name}.py").write_text(f"{name} = 1\n")
        (self.repo / "top.py").write_text("t = 1\n")
        base, work = self._commit_and_copy()
        (work / "outer" / "inner" / "a.py").write_text("BACKDOOR = True\n")
        return base, work


class UntypableDirectoryDoesNotFabricateDeletions(_RealRepo):
    """`F-1`, eighth adversarial pass, 2026-08-19.

    Fixture throughout: tracked `outer/inner/{a,b,c}.py`. The model backdoors
    the tracked file `outer/inner/a.py` and then runs ONE command,
    `chmod <mode> outer`. No race, no timing, no live process, and a mode bit
    is persistent state — destroying the container does not undo it, so this
    survives into the FINAL diff a human reviews.
    """

    def test_a_read_without_execute_directory_deletes_nothing(self):
        """The headline defect, over every mode that reaches it.

        Before the fix each mode below produced `unreadable=[('outer/inner',
        'EACCES')]` — note the absent slash — three `deleted file mode`
        sections for files nobody deleted, and a patch git accepted (rc=0,
        `0 1` against each of the three).
        """
        for mode in UNTYPABLE_MODES:
            with self.subTest(mode=oct(mode)):
                base, work = self._backdoor_case()
                os.chmod(work / "outer", mode)
                self._assert_untypable(work / "outer", "inner")

                snap, d = self._diff_of(base, work)

                # The precondition the fix has to survive: the record carries
                # NO trailing slash, because the walk never got to find out it
                # was a directory.
                self.assertEqual(
                    [(u.path, u.reason) for u in snap.unreadable],
                    [("outer/inner", "EACCES")],
                )
                # Nothing beneath it is claimed as deleted...
                self.assertNotIn("deleted file mode", d.text)
                for name in ("a", "b", "c"):
                    self.assertNotIn(f"--- a/outer/inner/{name}.py", d.text)
                # ...and `changed_files` makes no false positive assertion.
                self.assertEqual(d.changed_files, [])
                # The readable sibling is untouched and correctly still read.
                self.assertIn("top.py", snap.entries)

    def test_the_emitted_patch_cannot_delete_the_hidden_subtree(self):
        """The only honest test of "would it really delete them": ask git.

        Before the fix: rc=0 with `0 1 outer/inner/{a,b,c}.py` — an applicable
        patch that destroys three untouched tracked files.
        """
        base, work = self._backdoor_case()
        os.chmod(work / "outer", 0o644)
        self._assert_untypable(work / "outer", "inner")
        _, d = self._diff_of(base, work)

        # Restore BEFORE asking git; see `_apply_check`.
        _restore_modes(self.root)
        rc, numstat = self._apply_check(d.text)

        # Nothing in the text is patch structure at all, so there is nothing
        # for git to apply and nothing it could be made to delete.
        self.assertNotEqual(rc, 0, numstat)
        for name in ("a", "b", "c"):
            self.assertNotIn(f"outer/inner/{name}.py", numstat)

    def test_the_backdoored_file_is_not_reported_as_deleted(self):
        """The consequence that matters, stated in its own right.

        The file the model actually backdoored was rendered as DELETED rather
        than edited, and `changed_files` — documented as "a positive assertion
        that a path changed" — asserted a deletion that never happened. A
        reviewer reading that diff approves a run whose backdoor is nowhere in
        it, believing the file was merely removed.

        It cannot render as an EDIT: its bytes are unreachable by construction,
        since the `lstat` that failed is the same one the read would have gone
        through. The honest answer, and the one the `chmod 000` branch has
        given since `MF-1`, is an explicit "NOT COMPARED ... it is NOT deleted;
        this diff cannot say whether it changed".
        """
        base, work = self._backdoor_case()
        os.chmod(work / "outer", 0o644)
        self._assert_untypable(work / "outer", "inner")
        snap, d = self._diff_of(base, work)

        self.assertNotIn("outer/inner/a.py", d.changed_files)
        self.assertNotIn("deleted file mode", d.text)
        self.assertNotIn("outer/inner/a.py", snap.entries)
        # Named, at its own sorted position, as uncomparable rather than gone.
        marker = [
            ln
            for ln in d.text.splitlines()
            if ln.startswith("coding_agent: NOT COMPARED:") and "outer/inner/a.py" in ln
        ]
        self.assertEqual(len(marker), 1, d.text)
        self.assertIn("EACCES", marker[0])

    def test_the_hidden_subtree_stays_conspicuous(self):
        """Suppressing the deletion is only safe while the human is still told.

        Silently omitting the subtree would be the same under-showing failure
        wearing a different hat, so both signals are pinned: the head-block
        warning (rendered FIRST, where the size cap cannot reach it) and one
        explicit line per hidden tracked path, at that path's own sorted
        position in the body.
        """
        base, work = self._backdoor_case()
        os.chmod(work / "outer", 0o644)
        _, d = self._diff_of(base, work)

        self.assertIn("1 path(s) could not be read", d.text)
        self.assertIn('"outer/inner" (EACCES)', d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 3)
        for name in ("a", "b", "c"):
            self.assertIn(f"outer/inner/{name}.py", d.text)
        self.assertIn("it is NOT deleted", d.text)
        # The reason names the directory that actually failed, so the line does
        # not imply the file itself was tried and refused.
        self.assertIn('EACCES under "outer/inner/"', d.text)
        self.assertFalse(d.truncated)

    def test_a_searchable_directory_still_shows_the_backdoor(self):
        """VACUITY CONTROL. Without this, every assertion above would hold
        just as well against a fixture that never contained a backdoor.

        At `0500`/`0755` the walk classifies the children normally, so the same
        fixture and the same edit must produce an ordinary, applicable diff.
        """
        for mode in SEARCHABLE_MODES:
            with self.subTest(mode=oct(mode)):
                base, work = self._backdoor_case()
                os.chmod(work / "outer", mode)

                snap, d = self._diff_of(base, work)

                self.assertEqual(snap.unreadable, ())
                self.assertEqual(d.changed_files, ["outer/inner/a.py"])
                self.assertIn("+BACKDOOR = True", d.text)
                self.assertNotIn("NOT COMPARED", d.text)
                _restore_modes(self.root)
                rc, numstat = self._apply_check(d.text)
                self.assertEqual(rc, 0, numstat)
                self.assertIn("outer/inner/a.py", numstat)

    def test_the_chmod_000_branch_is_unchanged(self):
        """CONTROL for the branch that already worked.

        `chmod 000` fails at `open`, by which point the walk knows the entry is
        a directory, so the record keeps its slash. The fix reorganises the
        loop that consumes those records, so the previously-correct branch is
        re-pinned here rather than assumed.
        """
        base, work = self._backdoor_case()
        os.chmod(work / "outer", 0o000)
        _, d = self._diff_of(base, work)

        snap, d = self._diff_of(base, work)
        self.assertEqual([u.path for u in snap.unreadable], ["outer/"])
        self.assertNotIn("deleted file mode", d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 3)
        self.assertEqual(d.changed_files, [])
        self.assertIn('EACCES under "outer/"', d.text)

    def test_nothing_under_an_untypable_record_was_ever_read(self):
        """PREMISE PIN, not a fix pin — it passes with or without the fix.

        The suppression is only safe because it cannot hide a comparison the
        walk actually made, and that rests on one fact: nothing beneath a path
        whose `lstat` failed can be in `entries`, because the walk never
        obtained a descriptor to it. If a later change ever made the walk reach
        through such a path, the prefix block would start suppressing real
        content and this test is what fires.
        """
        base, work = self._backdoor_case()
        os.chmod(work / "outer", 0o644)
        snap, _ = self._diff_of(base, work)

        blind = [u.path for u in snap.unreadable]
        self.assertEqual(blind, ["outer/inner"])
        for path in snap.entries:
            for prefix in blind:
                self.assertFalse(path.startswith(prefix + "/"), path)


class UnreadableRootDoesNotFabricateDeletions(_RealRepo):
    """Attack B from the review: `chmod 0444` on the worktree ROOT.

    The root itself opens and lists fine, so `snapshot_tree` never takes its
    `Unreadable(".")` early return. Every top-level child fails to classify
    instead, producing a mix the fix has to handle in one pass: `README.md` is
    a base entry and matches exactly, while `src` and `tests` are directories
    whose records carry no slash and cover base paths beneath them.
    """

    def test_an_unreadable_root_deletes_nothing(self):
        (self.repo / "README.md").write_text("# hi\n")
        (self.repo / "src" / "deep").mkdir(parents=True)
        (self.repo / "src" / "app.py").write_text("a = 1\n")
        (self.repo / "src" / "deep" / "core.py").write_text("c = 1\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_app.py").write_text("t = 1\n")
        base, work = self._commit_and_copy()
        (work / "src" / "app.py").write_text("BACKDOOR = True\n")
        os.chmod(work, 0o444)
        self._assert_untypable(work, "src")

        snap, d = self._diff_of(base, work)

        self.assertEqual(
            sorted(u.path for u in snap.unreadable), ["README.md", "src", "tests"]
        )
        self.assertNotIn("deleted file mode", d.text)
        self.assertEqual(d.changed_files, [])
        # All four tracked paths accounted for: one by exact match, three by
        # the two subtree prefixes.
        self.assertEqual(d.text.count("NOT COMPARED"), 4)
        for path in ("README.md", "src/app.py", "src/deep/core.py", "tests/test_app.py"):
            self.assertIn(path, d.text)
        # The exact hit names the file itself; the covered ones name the
        # directory that failed. The second pass in `_not_compared` is what
        # keeps the more precise reason for `README.md`.
        readme = [
            ln
            for ln in d.text.splitlines()
            if ln.startswith("coding_agent: NOT COMPARED:") and "README.md" in ln
        ]
        self.assertEqual(len(readme), 1, d.text)
        self.assertIn("(EACCES)", readme[0])
        self.assertNotIn("under", readme[0])

    def test_the_root_patch_cannot_delete_anything(self):
        (self.repo / "keep.py").write_text("k = 1\n")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "app.py").write_text("a = 1\n")
        base, work = self._commit_and_copy()
        os.chmod(work, 0o444)
        _, d = self._diff_of(base, work)

        _restore_modes(self.root)
        rc, numstat = self._apply_check(d.text)
        self.assertNotEqual(rc, 0, numstat)
        self.assertNotIn("src/app.py", numstat)
        self.assertNotIn("keep.py", numstat)


class GenuineDeletionsSurvive(_RealRepo):
    """The other direction, and the reason this is not a one-sided fix.

    Suppressing too much is the same class of failure as suppressing too
    little: a real deletion that renders as nothing blinds the reviewer just as
    effectively. Every test here puts a genuine deletion in the SAME diff as an
    untypable subtree, so an over-broad suppression cannot hide behind a
    fixture that never contained one.
    """

    def test_a_real_deletion_still_renders_and_applies(self):
        (self.repo / "outer" / "inner").mkdir(parents=True)
        (self.repo / "outer" / "inner" / "hidden.py").write_text("h = 1\n")
        (self.repo / "gone.py").write_text("g = 1\n")
        (self.repo / "edited.py").write_text("e = 1\n")
        base, work = self._commit_and_copy()
        (work / "outer" / "inner" / "hidden.py").write_text("BACKDOOR = True\n")
        (work / "gone.py").unlink()
        (work / "edited.py").write_text("e = 2\n")
        os.chmod(work / "outer", 0o644)
        self._assert_untypable(work / "outer", "inner")

        snap, d = self._diff_of(base, work)

        # Exactly ONE deletion is rendered, and it is the real one.
        self.assertEqual(d.text.count("deleted file mode"), 1)
        self.assertIn("deleted file mode 100644", d.text)
        self.assertIn("--- a/gone.py", d.text)
        self.assertIn("gone.py", d.changed_files)
        # The ordinary edit is untouched by any of this.
        self.assertIn("+e = 2", d.text)
        self.assertIn("edited.py", d.changed_files)
        # The hidden path is suppressed, not deleted.
        self.assertNotIn("outer/inner/hidden.py", d.changed_files)
        self.assertIn("NOT COMPARED", d.text)
        self.assertEqual([u.path for u in snap.unreadable], ["outer/inner"])

        _restore_modes(self.root)
        rc, numstat = self._apply_check(d.text)
        self.assertEqual(rc, 0, numstat)
        self.assertIn("gone.py", numstat)  # the deletion is real and applies
        self.assertIn("edited.py", numstat)
        self.assertNotIn("hidden.py", numstat)

    def test_a_readable_sibling_subtree_is_still_compared(self):
        """An unreadable `outer` must not suppress `outerly/`.

        `startswith` on a bare path would blur the two; the fix always appends
        the separator, so the prefix can only ever match a genuine descendant.
        """
        (self.repo / "outer" / "inner").mkdir(parents=True)
        (self.repo / "outer" / "inner" / "hidden.py").write_text("h = 1\n")
        (self.repo / "outerly").mkdir()
        (self.repo / "outerly" / "sibling.py").write_text("s = 1\n")
        base, work = self._commit_and_copy()
        (work / "outerly" / "sibling.py").write_text("s = 2\n")
        os.chmod(work / "outer", 0o644)

        _, d = self._diff_of(base, work)

        self.assertEqual(d.changed_files, ["outerly/sibling.py"])
        self.assertIn("+s = 2", d.text)
        self.assertIn("diff --git a/outerly/sibling.py", d.text)
        # ...and it is compared, not merely mentioned in a suppression line.
        self.assertNotIn('NOT COMPARED: "outerly/sibling.py"', d.text)

        _restore_modes(self.root)
        rc, numstat = self._apply_check(d.text)
        self.assertEqual(rc, 0, numstat)
        self.assertIn("outerly/sibling.py", numstat)


class EveryUntypableCauseIsCovered(unittest.TestCase):
    """Permissions are one way to lose the type; they are not the only way.

    `_visit` records an unslashed path whenever it never got a successful
    `lstat` on the child: a terminal errno on the classifying call
    (`EACCES`/`EPERM`/`EMFILE`/`ENFILE`), or the attempt loop running out while
    the child kept moving (`ENOENT`, `ESTALE`, `ELOOP`, `ENAMETOOLONG`,
    `RACED`). The fix is deliberately blind to WHICH — it keys on the shape of
    the record, not on a list of causes that could go stale as new errnos start
    reaching it.

    Driven through the public `unified_diff` with synthetic snapshots, so the
    coverage claim does not depend on being able to manufacture every errno on
    this filesystem.
    """

    REASONS = (
        "EACCES",
        "EPERM",
        "EMFILE",
        "ENFILE",
        "ENOENT",
        "ESTALE",
        "ELOOP",
        "ENAMETOOLONG",
        "ENOMEM",
        "RACED",
        "OVERSIZE",
        "BUDGET",
        "XDEV",
        "OSERROR",
        "ERRNO9999",
    )

    def setUp(self) -> None:
        self.spill = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.spill, ignore_errors=True)

    def _base(self) -> BaseTree:
        entries = {
            "elsewhere.py": Entry("elsewhere.py", "file", b"e = 1\n"),
            "outer/inner/a.py": Entry("outer/inner/a.py", "file", b"a = 1\n"),
            "outer/inner/deep/b.py": Entry("outer/inner/deep/b.py", "file", b"b = 1\n"),
        }
        return BaseTree(entries, lambda p: False, frozenset(entries))

    def _diff(self, snap: TreeSnapshot):
        return unified_diff(
            self._base(), snap, max_bytes=_MAX_BYTES, spill_dir=self.spill
        )

    def test_any_reason_covers_the_subtree(self):
        for reason in self.REASONS:
            with self.subTest(reason=reason):
                d = self._diff(
                    TreeSnapshot(
                        {"elsewhere.py": Entry("elsewhere.py", "file", b"e = 2\n")},
                        (Unreadable("outer/inner", reason),),
                    )
                )
                self.assertNotIn("deleted file mode", d.text)
                self.assertEqual(d.text.count("NOT COMPARED"), 2)
                self.assertIn("outer/inner/a.py", d.text)
                self.assertIn("outer/inner/deep/b.py", d.text)
                self.assertIn(f'{reason} under "outer/inner/"', d.text)
                # The unrelated real edit is still compared, every time.
                self.assertEqual(d.changed_files, ["elsewhere.py"])
                self.assertIn("+e = 2", d.text)

    def test_an_unreadable_leaf_file_is_still_an_exact_match(self):
        """The fix adds a prefix branch; it must not cost the exact one.

        The other two base paths are present in the snapshot, so the only
        absence in this diff is the one the record explains — otherwise a
        `deleted file mode` assertion would be answered by files that really
        are missing and the test would prove nothing about the exact branch.
        """
        d = self._diff(
            TreeSnapshot(
                {
                    "outer/inner/a.py": Entry("outer/inner/a.py", "file", b"a = 1\n"),
                    "outer/inner/deep/b.py": Entry(
                        "outer/inner/deep/b.py", "file", b"b = 1\n"
                    ),
                },
                (Unreadable("elsewhere.py", "EACCES"),),
            )
        )
        self.assertNotIn("deleted file mode", d.text)
        line = [
            ln
            for ln in d.text.splitlines()
            if ln.startswith("coding_agent: NOT COMPARED:") and "elsewhere.py" in ln
        ]
        self.assertEqual(len(line), 1, d.text)
        self.assertIn("(EACCES)", line[0])
        self.assertNotIn("under", line[0])  # the precise reason, not a covering one

    def test_a_record_naming_no_base_path_blocks_nothing(self):
        """A file the model CREATED and made unreadable has no base side, so
        the prefix branch must find nothing and suppress nothing."""
        d = self._diff(
            TreeSnapshot(
                {
                    "elsewhere.py": Entry("elsewhere.py", "file", b"e = 1\n"),
                    "outer/inner/a.py": Entry("outer/inner/a.py", "file", b"a = 1\n"),
                    "outer/inner/deep/b.py": Entry(
                        "outer/inner/deep/b.py", "file", b"b = 1\n"
                    ),
                },
                (Unreadable("new_backdoor", "EACCES"),),
            )
        )
        self.assertNotIn("NOT COMPARED", d.text)
        self.assertIn("new_backdoor", d.text)  # still reported to the human
        self.assertEqual(d.changed_files, [])  # nothing actually differs

    def test_a_record_cannot_block_a_name_it_merely_prefixes(self):
        """`outer/inner` must not cover `outer/innerly.py`."""
        entries = {
            "outer/inner/a.py": Entry("outer/inner/a.py", "file", b"a = 1\n"),
            "outer/innerly.py": Entry("outer/innerly.py", "file", b"i = 1\n"),
        }
        base = BaseTree(entries, lambda p: False, frozenset(entries))
        snap = TreeSnapshot(
            {"outer/innerly.py": Entry("outer/innerly.py", "file", b"i = 2\n")},
            (Unreadable("outer/inner", "EACCES"),),
        )
        d = unified_diff(base, snap, max_bytes=_MAX_BYTES, spill_dir=self.spill)
        self.assertEqual(d.changed_files, ["outer/innerly.py"])
        self.assertIn("+i = 2", d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 1)
        self.assertIn("outer/inner/a.py", d.text)

    def test_the_most_specific_record_supplies_the_reason(self):
        """Two records covering one base path: the precise one must win.

        `unified_diff` is public and its output must not depend on which of two
        records a caller listed first. The walk itself cannot produce this pair
        — a path whose `lstat` failed is a path it never descended into — so it
        is built by hand, and asserted BOTH ways round to prove the answer
        comes from `TreeSnapshot.unreadable` being sorted rather than from the
        order the tuple happens to arrive in.
        """
        records = tuple(
            sorted(
                (
                    Unreadable("outer/inner/a.py", "OVERSIZE"),
                    Unreadable("outer/", "EACCES"),
                )
            )
        )
        # MECHANISM: the precedence is a consequence of the sort, so assert the
        # sort actually puts the covering ancestor first. If `Unreadable`'s
        # ordering ever stopped doing that, this is what fires — rather than
        # the reason strings quietly going wrong.
        self.assertEqual([u.path for u in records], ["outer/", "outer/inner/a.py"])

        d = self._diff(
            TreeSnapshot(
                {"elsewhere.py": Entry("elsewhere.py", "file", b"e = 1\n")}, records
            )
        )
        self.assertNotIn("deleted file mode", d.text)
        line = [
            ln
            for ln in d.text.splitlines()
            if ln.startswith("coding_agent: NOT COMPARED:") and "outer/inner/a.py" in ln
        ]
        self.assertEqual(len(line), 1, d.text)
        # The file's own reason, not the directory's covering one.
        self.assertIn("(OVERSIZE)", line[0])
        self.assertNotIn("EACCES", line[0])
        # ...while its sibling is still covered by the directory.
        self.assertIn('EACCES under "outer/"', d.text)

    def test_the_root_record_still_blocks_everything(self):
        """`snapshot_tree`'s early returns name the root `.`; that short
        circuit precedes the new prefix logic and must survive it."""
        d = self._diff(TreeSnapshot({}, (Unreadable(".", "EACCES"),)))
        self.assertNotIn("deleted file mode", d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 3)
        self.assertIn("EACCES on the worktree root", d.text)
        self.assertEqual(d.changed_files, [])


class VanishingDirectoryIsNotADeletion(_RealRepo):
    """The same unslashed record, reached WITHOUT any permission bit.

    A directory that disappears between `os.listdir` and the classifying
    `lstat` exhausts `_RECLASSIFY_ATTEMPTS` on `ENOENT` and falls through to
    `Unreadable(rel + tail, reason)` with `tail` still `""` — the identical
    shape, from a completely different cause.

    The window is opened BY HAND through the injected `is_ignored` callable, in
    this process, with no thread and no racing writer: a probe that has to win
    a race reports safety when it merely lost, which is this build's most
    repeated lesson. The hook fires on the walk's own call for an earlier
    sibling, so the ordering is a property of the walk (names are popped in
    ascending order), not of the scheduler.

    Suppressing the deletion here is right even though the directory really is
    gone. The walk saw the name in `listdir` and then could not classify it, so
    "deleted" is a positive assertion it cannot support — the racer may have
    moved it, swapped it, or be mid-atomic-save. The artifact that matters is
    built after the container is destroyed (§6.5 rule 3), when nothing races: a
    genuine deletion is then simply absent from `listdir` and renders normally,
    as `GenuineDeletionsSurvive` pins. And the suppression is loud, not silent.
    """

    def test_a_directory_that_vanishes_mid_walk_deletes_nothing(self):
        (self.repo / "outer").mkdir()
        (self.repo / "outer" / "aaa.py").write_text("a = 1\n")
        (self.repo / "outer" / "zdir").mkdir()
        (self.repo / "outer" / "zdir" / "deep.py").write_text("d = 1\n")
        base, work = self._commit_and_copy()

        real = make_ignore(base)
        fired: list[str] = []

        def hook(query: str) -> bool:
            # Ascending order, so `outer/aaa.py` is classified before `zdir`.
            if query == "outer/aaa.py" and not fired:
                fired.append(query)
                shutil.rmtree(work / "outer" / "zdir")
            return real(query)

        snap = snapshot_tree(str(work), hook)
        spill = self.repo.parent / "spill"
        spill.mkdir(exist_ok=True)
        d = unified_diff(base, snap, max_bytes=_MAX_BYTES, spill_dir=str(spill))

        # MECHANISM: the window really did open, and the record really is the
        # untypable shape — no slash, and a race errno rather than EACCES.
        self.assertEqual(fired, ["outer/aaa.py"])
        self.assertEqual(
            [(u.path, u.reason) for u in snap.unreadable], [("outer/zdir", "ENOENT")]
        )
        # ...and the tracked file beneath it is not claimed as deleted.
        self.assertNotIn("deleted file mode", d.text)
        self.assertNotIn("outer/zdir/deep.py", d.changed_files)
        self.assertIn("outer/zdir/deep.py", d.text)
        self.assertIn('ENOENT under "outer/zdir/"', d.text)
        # The sibling the walk DID reach is still compared normally.
        self.assertIn("outer/aaa.py", snap.entries)

        rc, numstat = self._apply_check(d.text)
        self.assertNotEqual(rc, 0, numstat)
        self.assertNotIn("deep.py", numstat)


class SymlinkRecordsKeepTheirShape(_RealRepo):
    """A symlink refusal is recorded with `rel`, never `rel + tail`.

    Worth pinning because the fix appends a separator to every record: a
    symlink is a leaf, so its record must keep matching the base tree exactly
    and must not start covering a subtree that does not exist. `lstat` succeeds
    on a symlink whatever its mode, so the refusal side is asserted through
    `unified_diff` rather than by manufacturing a `readlink` failure.
    """

    def test_a_symlink_path_is_matched_exactly(self):
        (self.repo / "link").symlink_to("target")
        (self.repo / "link_dir").mkdir()
        (self.repo / "link_dir" / "under.py").write_text("u = 1\n")
        base, work = self._commit_and_copy()
        spill = self.repo.parent / "spill"
        spill.mkdir(exist_ok=True)

        snap = TreeSnapshot(
            {"link_dir/under.py": Entry("link_dir/under.py", "file", b"u = 2\n")},
            (Unreadable("link", "EACCES"),),
        )
        d = unified_diff(base, snap, max_bytes=_MAX_BYTES, spill_dir=str(spill))

        self.assertNotIn("deleted file mode", d.text)
        self.assertIn('NOT COMPARED: "link" (EACCES)', d.text)
        self.assertEqual(d.text.count("NOT COMPARED"), 1)
        # `link_dir/under.py` is NOT under `link/`, and is compared normally.
        self.assertEqual(d.changed_files, ["link_dir/under.py"])
        self.assertIn("+u = 2", d.text)

    def test_the_walk_records_a_symlink_without_a_trailing_slash(self):
        """The producer side of the same property."""
        (self.repo / "link").symlink_to("target")
        base, work = self._commit_and_copy()
        snap, _ = self._diff_of(base, work)
        self.assertEqual(snap.entries["link"].kind, "symlink")
        self.assertEqual(snap.entries["link"].mode, "120000")
        self.assertNotIn("link/", snap.entries)


if __name__ == "__main__":
    unittest.main()
