#!/usr/bin/env python3
"""F-2 regression tests: DISPLACEMENT of a real change by sandbox-created noise.

The sign-off review (2026-08-19, eighth adversarial pass) demonstrated, and
this controller reproduced at the SHIPPED constants, that a model which
backdoors a TRACKED file and then writes N decoy files whose names sort first
can push that file out of every artifact a human reads:

    decoys=0    truncated=False  BACKDOOR visible: True   'src.py' in text: True
    decoys=600  truncated=True   BACKDOOR visible: False  'src.py' in text: False

720 KB of ordinary writes, no race, no privilege. At 600 decoys the path was
absent from `DiffResult.text` AND — via `loop._capped_paths` keeping the first
500 of the same sorted list — from `AgentResult.changed_files`, so it appeared
nowhere in the MCP response at all. `_CHANGED_HEAD` meanwhile asserted, in the
same text, that "a truncated diff cannot hide THAT a path changed".

The fix is an ordering, not a bigger cap: base-tree paths are rendered and
listed FIRST, in the diff body, in the appended path list, and in both capped
result lists. The base tree is read from the host's repository, so the sandbox
cannot author an entry in it — which is why this is structural rather than
merely expensive. Every test below is written so that it FAILS on the
pre-fix code (verified, one by one, against a stashed copy) rather than
passing because the attack did not fire; the "CONTROL" assertions exist to
prove the attack really did fire.

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_displacement.py -q

Like test_coding_agent_walk.py, the cases here DO run git — but only ever as
`git -C <throwaway repo>` against a repository the test itself created, and
never with the walked directory as cwd or gitdir (spec §6.5). The walked tree
is always a `.git`-less COPY.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent import loop as L
from coding_agent import walk as W
from coding_agent.basetree import BaseTree, make_ignore, read_base_tree
from coding_agent.walk import DiffResult, Entry, snapshot_tree, unified_diff

# Big enough that 600 of them overrun `loop._DIFF_MAX_BYTES` (512 KiB), which
# is the review's own measurement: ~720 KB of ordinary file writes.
_DECOY_BYTES = 1200
_BACKDOOR = "BACKDOOR = 'exfil'\n"


def _decoy_name(i: int) -> str:
    """`!` (0x21) sorts before every letter, digit and `.` in a byte sort, so
    a decoy always lands ahead of a real source path in the same directory.
    The attack does not depend on this — any name that sorts first will do —
    it just makes the test deterministic."""
    return f"!{i:04d}.txt"


class _RealRepoCase(unittest.TestCase):
    """A real git repo for the base tree, a separate copy for the walk."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.spill = self.root / "spill"
        self.spill.mkdir()
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "t@e.com")
        self._git("config", "user.name", "T")

    def _git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *args], check=True)

    def commit_and_copy(self) -> tuple[BaseTree, Path]:
        # `--allow-empty` so a case can ask for an EMPTY base tree, which is
        # the state that proves the ordering costs nothing when there is no
        # tracked side at all.
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base", "--allow-empty")
        work = self.root / "work"
        shutil.copytree(
            self.repo, work, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        return read_base_tree(str(self.repo), "HEAD"), work

    def decoys(self, work: Path, n: int) -> None:
        for i in range(n):
            (work / _decoy_name(i)).write_text("x" * _DECOY_BYTES + "\n")

    def diff(self, base: BaseTree, work: Path, **kw: int) -> DiffResult:
        snap = snapshot_tree(str(work), make_ignore(base))
        return unified_diff(
            base,
            snap,
            max_bytes=kw.get("max_bytes", L._DIFF_MAX_BYTES),
            spill_dir=str(self.spill),
        )

    def capped(self, base: BaseTree, d: DiffResult) -> list[str]:
        """`changed_files` exactly as `loop._teardown` builds it."""
        return L._capped_paths(
            list(d.changed_files),
            where=(
                f"the complete diff is at {d.full_path}"
                if d.full_path is not None
                else "the diff text above names every changed path"
            ),
            keep_first=L._base_reach(base),
        )


class ADecoyFloodCannotHideAChangeToATrackedFile(_RealRepoCase):
    """THE regression this file exists for.

    A tracked file the model edits must remain NAMED in the human-visible
    artifact no matter how many files the model creates, at any cost, whatever
    their names. Everything else here is a corollary of that sentence.
    """

    def test_the_backdoored_tracked_file_survives_at_and_beyond_the_shipped_caps(
        self,
    ) -> None:
        (self.repo / "src.py").write_text("def f():\n    return 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("def f():\n    return 1\n" + _BACKDOOR)

        for n in (600, 2000):
            with self.subTest(decoys=n):
                for stale in work.glob("!*.txt"):
                    stale.unlink()
                self.decoys(work, n)
                d = self.diff(base, work)

                # CONTROLS. Without these the test could pass because the
                # attack never fired — the exact failure mode this project has
                # shipped before (a security test that passed with AND without
                # the bug it targeted).
                self.assertTrue(d.truncated, "the flood must overrun the cap")
                self.assertIn(_decoy_name(0), d.text, "the decoys must be present")
                self.assertEqual(len(d.changed_files), n + 1)

                self.assertIn("src.py", d.text)  # THAT it changed
                self.assertIn(_BACKDOOR.strip(), d.text)  # ...and WHAT changed
                self.assertIn("src.py", self.capped(base, d))

    def test_the_tracked_section_is_rendered_before_any_created_one(self) -> None:
        """The body order is the reason the backdoor's own hunk survives, not
        just its name: truncation removes the TAIL, so the group that must not
        be lost has to be the head."""
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("x = 2\n")
        self.decoys(work, 3)
        d = self.diff(base, work)

        self.assertFalse(d.truncated)  # CONTROL: ordering, not truncation
        self.assertLess(d.text.index("a/src.py"), d.text.index(_decoy_name(0)))
        self.assertIn("section below is a path this run CREATED", d.text)
        # The boundary is prose, never patch structure.
        for line in d.text.splitlines():
            if "CREATED" in line or "BASE TREE holds" in line:
                self.assertTrue(line.startswith("coding_agent:"), repr(line))

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file anyway")
    def test_a_flood_cannot_evict_the_not_compared_line_either(self) -> None:
        """`NOT COMPARED` lives in the BODY, so a flood that truncates the
        body could cut it and leave only the bounded head-block record — which
        a 500-name flood then displaces too. Both records are for a path the
        BASE TREE holds, so the same ordering saves it."""
        (self.repo / "secret.py").write_text("k = 1\n")
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        os.chmod(work / "secret.py", 0o000)
        self.decoys(work, 600)
        d = self.diff(base, work)

        self.assertTrue(d.truncated)  # CONTROL
        self.assertIn("NOT COMPARED", d.text)
        self.assertIn("secret.py", d.text)
        self.assertIn("EACCES", d.text)


class CreatedFilesAreStillReported(_RealRepoCase):
    """Suppressing displacement is only half a fix if a genuine new file stops
    being shown. Over-showing is noise; the ordering must not turn into
    under-showing in the other direction."""

    def test_a_new_untracked_file_is_named_and_rendered(self) -> None:
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "zzz_new.py").write_text("payload = 1\n")
        d = self.diff(base, work)

        self.assertFalse(d.truncated)
        self.assertEqual(d.changed_files, ["zzz_new.py"])
        self.assertIn("new file mode 100644", d.text)
        self.assertIn("+payload = 1", d.text)
        self.assertIn("zzz_new.py", self.capped(base, d))

    def test_created_paths_still_fill_the_appended_list(self) -> None:
        """Tracked paths go first; they do not go ALONE. Whatever budget the
        tracked side leaves is spent on created ones."""
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("x = 2\n")
        self.decoys(work, 600)
        d = self.diff(base, work)

        tail = d.text.split("[... diff truncated")[1]
        listed = [ln for ln in tail.splitlines() if "(- -> 100644)" in ln]
        self.assertEqual(len(listed), W._CHANGED_SHOWN - 1)  # 1 slot went to src.py
        self.assertIn(_decoy_name(0), tail)

    def test_a_diff_with_no_base_tree_is_unchanged_by_the_ordering(self) -> None:
        """No tracked side means no boundary line: a created-only diff renders
        exactly as it did before, which is what keeps the ordering from
        costing bytes on the ordinary path."""
        base, work = self.commit_and_copy()  # empty base tree
        (work / "a.py").write_text("x = 1\n")
        d = self.diff(base, work)
        self.assertNotIn("BASE TREE holds", d.text)
        self.assertTrue(d.text.startswith("diff --git a/a.py"), d.text[:60])


_HEAD_RE = re.compile(
    r"^coding_agent: (\d+) file\(s\) changed in total\.", re.MULTILINE
)
_MORE_RE = re.compile(
    r"\.\.\. and (\d+) more not listed above \((\d+) tracked in the\n"
    r"coding_agent:\s+base tree, (\d+) created by this run\)"
)
_CUT_RE = re.compile(
    r"WARNING: (\d+) TRACKED path\(s\) changed, more than the (\d+) this\n"
    r"coding_agent: list can hold, so (\d+) changed tracked path\(s\) are NOT named"
)
# One rendered entry of the appended list: a path, then a mode transition.
# Anchored on the transition so the `... and N more` sentence, which also
# starts with the same indent, cannot be mistaken for a listed path.
_ENTRY_RE = re.compile(
    r"^coding_agent:   (.*) \((?:-|\d{6}) -> (?:-|\d{6})\)$", re.MULTILINE
)


class EveryClaimTheTextMakesIsTrue(_RealRepoCase):
    """The reason F-2 was rated blocking rather than medium: the old text
    asserted "a truncated diff cannot hide THAT a path changed" while doing
    exactly that. A control whose documentation states a guarantee it does not
    provide is worse than one that says nothing, because the reader stops
    looking. So the numbers are checked against reality, not just for
    presence."""

    def assert_claims_hold(self, base: BaseTree, d: DiffResult) -> None:
        tracked = [p for p in d.changed_files if p in base.entries]
        created = [p for p in d.changed_files if p not in base.entries]
        if not d.truncated:
            # The strongest claim an untruncated diff makes is implicit: every
            # changed path has a section of its own.
            for p in d.changed_files:
                self.assertIn(f"diff --git a/{p} ", d.text)
            self.assertNotIn("file(s) changed in total", d.text)
            return

        tail = d.text.split("[... diff truncated")[1]
        head = _HEAD_RE.search(d.text)
        assert head is not None, d.text[-400:]
        self.assertEqual(int(head.group(1)), len(d.changed_files))

        shown_tracked = min(len(tracked), W._CHANGED_SHOWN)
        shown_created = min(len(created), W._CHANGED_SHOWN - shown_tracked)
        cut_tracked = len(tracked) - shown_tracked
        cut_created = len(created) - shown_created
        cut = _CUT_RE.search(d.text)
        if cut_tracked:
            assert cut is not None, "a cut tracked list must SAY it was cut"
            self.assertEqual(
                [int(g) for g in cut.groups()],
                [len(tracked), shown_tracked, cut_tracked],
            )
        else:
            self.assertIsNone(cut, "no warning when nothing tracked was cut")
            for p in tracked:  # the whole point: all of them, by name
                self.assertIn(p, tail)

        # "listed FIRST" is a claim like any other, so it is checked like one:
        # every tracked entry precedes every created entry in the block, and
        # the block holds exactly the entries the budget allows.
        listed = [p.strip('"') for p in _ENTRY_RE.findall(tail)]
        self.assertEqual(len(listed), shown_tracked + shown_created)
        flags = [p in base.entries for p in listed]
        self.assertEqual(flags, sorted(flags, reverse=True), listed[:5])
        self.assertEqual(flags.count(True), shown_tracked)

        more = _MORE_RE.search(d.text)
        if cut_tracked + cut_created:
            assert more is not None, d.text[-400:]
            n, t, c = (int(g) for g in more.groups())
            self.assertEqual((t, c), (cut_tracked, cut_created))
            self.assertEqual(n, t + c)
        else:
            self.assertIsNone(more)

        # The retired claims must not come back by copy-paste.
        self.assertNotIn("cannot hide THAT a path changed", d.text)
        self.assertNotIn("the list below is complete", d.text)
        # Any spill path the text names must be a file that exists.
        for named in re.findall(r"/[^\s]*coding-agent-diff-[^\s]+\.patch", d.text):
            self.assertTrue(os.path.exists(named), named)

    def test_claims_hold_with_no_tracked_change(self) -> None:
        base, work = self.commit_and_copy()
        self.decoys(work, 600)
        d = self.diff(base, work)
        self.assertTrue(d.truncated)
        self.assert_claims_hold(base, d)

    def test_claims_hold_with_one_tracked_change_under_a_flood(self) -> None:
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("x = 2\n")
        self.decoys(work, 600)
        d = self.diff(base, work)
        self.assertTrue(d.truncated)
        self.assert_claims_hold(base, d)

    def test_claims_hold_when_the_tracked_list_ITSELF_overruns_the_cap(self) -> None:
        """The residue the fix cannot remove, stated instead of hidden. Only
        the base tree can supply more than `_CHANGED_SHOWN` tracked paths, and
        its size is the human's own repository — the one quantity the sandbox
        has no way to inflate."""
        for i in range(W._CHANGED_SHOWN + 50):
            (self.repo / f"trk{i:03d}.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        for i in range(W._CHANGED_SHOWN + 50):
            (work / f"trk{i:03d}.py").write_text("x = 2\n")
        self.decoys(work, 600)
        d = self.diff(base, work)

        self.assertTrue(d.truncated)
        self.assert_claims_hold(base, d)
        self.assertIn("150 TRACKED path(s) changed", d.text)
        self.assertIn("50 changed tracked path(s) are NOT named", d.text)
        # Loud, and ABOVE the list it qualifies rather than under it.
        tail = d.text.split("[... diff truncated")[1]
        self.assertLess(tail.index("TRACKED path(s) changed"), tail.index("trk000"))

    def test_claims_hold_on_an_ordinary_untruncated_diff(self) -> None:
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("x = 2\n")
        (work / "new.py").write_text("y = 1\n")
        d = self.diff(base, work)
        self.assertFalse(d.truncated)
        self.assert_claims_hold(base, d)

    def test_the_reordered_body_is_still_a_patch_git_accepts(self) -> None:
        """The boundary line is new text in the middle of the body. `git
        apply` must still parse the whole thing, or the reviewer's own
        verification step breaks."""
        (self.repo / "src.py").write_text("x = 1\n")
        base, work = self.commit_and_copy()
        (work / "src.py").write_text("x = 2\n")
        (work / "new.py").write_text("y = 1\n")
        d = self.diff(base, work)

        patch = self.root / "emitted.patch"
        patch.write_text(d.text, encoding="utf-8", errors="surrogateescape")
        proc = subprocess.run(
            ["git", "apply", "--check", "--numstat", str(patch)],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("src.py", proc.stdout)
        self.assertIn("new.py", proc.stdout)


def _base_of(*paths: str) -> BaseTree:
    return BaseTree(
        entries={p: Entry(p, "file", b"x\n") for p in paths},
        ignore=lambda p: False,
        tracked=frozenset(paths),
    )


class TheResultListsAreCappedBaseTreeFirst(unittest.TestCase):
    """The same hole one layer up. Fixing the diff text alone would leave
    `AgentResult.changed_files` truncating by sort position, which is where
    the 600-decoy reproduction dropped `src.py` from the response entirely."""

    def test_a_tracked_path_that_sorts_last_still_survives_the_cap(self) -> None:
        items = [_decoy_name(i) for i in range(L._RESULT_PATHS + 100)] + ["src.py"]
        out = L._capped_paths(items, keep_first=L._base_reach(_base_of("src.py")))

        self.assertIn("src.py", out)
        self.assertEqual(len(out), L._RESULT_PATHS + 1)  # + the overflow marker
        self.assertIn(f"and {len(items) - L._RESULT_PATHS} more", out[-1])
        # Survivors keep the caller's order; only WHICH ones survive changes.
        self.assertEqual(out[0], _decoy_name(0))
        self.assertEqual(out[:2], sorted(out[:2]))
        # CONTROL, in the same test so it cannot drift: capped POSITIONALLY —
        # which is what shipped — the identical list loses the tracked path.
        # This is the measured pre-fix behaviour, kept as the A/B that proves
        # `keep_first` is the thing doing the work.
        self.assertNotIn("src.py", L._capped_paths(items))

    def test_an_unreadable_record_covering_tracked_files_survives_the_cap(self) -> None:
        """An `unreadable` record is not always a file: `src/` — or `src/deep`,
        when the classifying lstat is what failed — is the record that hides
        every tracked path beneath it."""
        records = [
            {"path": _decoy_name(i), "reason": "EACCES"}
            for i in range(L._RESULT_PATHS + 5)
        ] + [{"path": "src/deep", "reason": "EACCES"}]
        reaches = L._base_reach(_base_of("src/deep/util.py"))
        out = L._capped_paths(records, keep_first=lambda r: reaches(r["path"]))

        self.assertIn({"path": "src/deep", "reason": "EACCES"}, out)
        self.assertEqual(out[-1]["reason"], "TRUNCATED")

    def test_base_reach_answers_for_directories_and_the_root_not_just_files(
        self,
    ) -> None:
        reaches = L._base_reach(_base_of("src/deep/util.py", "top.py"))
        for yes in ("src/deep/util.py", "top.py", "src/", "src", "src/deep", "."):
            self.assertTrue(reaches(yes), yes)
        for no in ("srcly.py", "src/deepish", "decoy/", "top.py.orig"):
            self.assertFalse(reaches(no), no)

    def test_the_overflow_marker_states_where_the_complete_data_really_is(
        self,
    ) -> None:
        """H-3. The old marker claimed "the diff text and its spill file are
        complete" unconditionally — false for the text whenever the diff was
        truncated, and naming a spill file that does not exist when it was
        not."""
        items = [f"f{i:04d}.py" for i in range(L._RESULT_PATHS + 2)]
        spilled = L._capped_paths(items, where="the complete diff is at /tmp/x.patch")
        self.assertIn("the complete diff is at /tmp/x.patch", spilled[-1])
        default = L._capped_paths(items)
        self.assertIn("not in this response", default[-1])
        self.assertNotIn("complete", default[-1])


if __name__ == "__main__":
    unittest.main()
