#!/usr/bin/env python3
"""Unit tests for coding_agent.tools — the four tools the model sees and the
host-side path guard (spec §5 "Tools the model sees", §6 path safety).

PLACEMENT NOTE. The Task-6 brief says to append these to
`test_coding_agent_security.py`. That file was owned by a concurrent agent when
this landed, so **spec §10 security regression 4d (the write-side and read-side
symlinked-parent escape) lives HERE**. An auditor tracing §10 coverage should
look in this file for 4d; a prior task made the same deviation for §10 item 5,
which is in `test_coding_agent_sandbox.py`.

No docker and no git are needed by anything in this file.

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_tools.py -q
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.tools import (
    MAX_DEPTH,
    MAX_LIST,
    MAX_READ,
    TOOL_SCHEMAS,
    PathEscape,
    safe_path,
    tool_list_files,
    tool_read_file,
    tool_run_command,
    tool_write_file,
)


def _assert_is_symlink(case: unittest.TestCase, path: Path) -> None:
    """MECHANISM control: prove the thing we planted really is a symlink.

    Without this, "the write was rejected" could be true because the fixture
    never built the attack — a confident false negative of exactly the kind
    this project has shipped before.
    """
    st = os.lstat(path)
    case.assertTrue(
        stat.S_ISLNK(st.st_mode), f"fixture is broken: {path} is not a symlink"
    )


class _Tree(unittest.TestCase):
    """`tmp/work` is the worktree root; `tmp/outside` is the host filesystem
    the model must never reach; `tmp/work-evil` is the sibling that a
    prefix check without a trailing separator would let through."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup)
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        self.root = self.tmp / "work"
        self.root.mkdir()
        self.evil = self.tmp / "work-evil"
        self.evil.mkdir()
        self.real_root = os.path.realpath(str(self.root))

    def _cleanup(self) -> None:
        subprocess.run(["chmod", "-R", "u+rwX", str(self.tmp)], check=False)
        subprocess.run(["rm", "-rf", str(self.tmp)], check=False)


# --------------------------------------------------------------------------
# safe_path: the string guard
# --------------------------------------------------------------------------


class SafePathRejectsTheObviousEscapes(_Tree):
    def test_dotdot_and_absolute_are_rejected(self) -> None:
        for bad in ("../x", "/etc/passwd", "a/../../x", "sub/../../../etc/passwd"):
            for for_write in (True, False):
                with self.subTest(rel=bad, for_write=for_write):
                    with self.assertRaises(PathEscape):
                        safe_path(str(self.root), bad, for_write=for_write)

    def test_empty_path_is_rejected(self) -> None:
        for for_write in (True, False):
            with self.assertRaises(PathEscape):
                safe_path(str(self.root), "", for_write=for_write)

    def test_nul_byte_is_a_PathEscape_not_a_bare_ValueError(self) -> None:
        """A caller that catches PathEscape must not be bypassed by a plain
        ValueError from deep inside posixpath."""
        for bad in ("a\0b", "\0", "sub/\0"):
            for for_write in (True, False):
                with self.subTest(rel=bad, for_write=for_write):
                    with self.assertRaises(PathEscape):
                        safe_path(str(self.root), bad, for_write=for_write)

    def test_a_non_string_path_is_rejected_not_a_TypeError(self) -> None:
        for bad in (None, 7, ["x"], {"path": "x"}):
            with self.subTest(rel=bad):
                with self.assertRaises(PathEscape):
                    safe_path(str(self.root), bad, for_write=True)  # type: ignore[arg-type]

    def test_a_sibling_sharing_the_root_prefix_is_rejected(self) -> None:
        """`work-evil` string-starts-with `work`."""
        (self.evil / "loot").write_text("secret\n")
        for for_write in (True, False):
            with self.subTest(for_write=for_write):
                with self.assertRaises(PathEscape):
                    safe_path(str(self.root), "../work-evil/loot", for_write=for_write)

    def test_a_symlink_to_the_prefix_sharing_sibling_is_rejected(self) -> None:
        """The case that actually exercises the trailing os.sep.

        A literal `../work-evil/loot` is refused by the lexical layer before
        the prefix check is reached, so it does not test that check at all.
        Reaching it needs a path with no surviving `..` whose *resolution* is
        the sibling — i.e. a symlink the model planted inside the worktree.
        Then `.../work-evil/loot`.startswith(`.../work`) is true and only the
        separator keeps it out.
        """
        (self.evil / "loot").write_text("secret\n")
        os.symlink("../work-evil", self.root / "sub")
        _assert_is_symlink(self, self.root / "sub")
        self.assertTrue(
            os.path.realpath(str(self.root / "sub")).startswith(self.real_root)
        )
        for for_write in (True, False):
            with self.subTest(for_write=for_write):
                with self.assertRaises(PathEscape):
                    safe_path(str(self.root), "sub/loot", for_write=for_write)
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), "sub/loot", "pwned")
        self.assertEqual((self.evil / "loot").read_text(), "secret\n")

    def test_dot_reads_the_root_and_never_writes_it(self) -> None:
        self.assertEqual(
            safe_path(str(self.root), ".", for_write=False), self.real_root
        )
        with self.assertRaises(PathEscape):
            safe_path(str(self.root), ".", for_write=True)

    def test_new_file_in_real_subdir_is_allowed(self) -> None:
        (self.root / "real").mkdir()
        p = safe_path(str(self.root), "real/new.py", for_write=True)
        self.assertTrue(p.startswith(self.real_root + os.sep))

    def test_new_file_in_a_not_yet_existing_subdir_is_allowed(self) -> None:
        p = safe_path(str(self.root), "brand/new/deep.py", for_write=True)
        self.assertEqual(p, os.path.join(self.real_root, "brand/new/deep.py"))

    def test_a_path_that_leaves_the_root_before_coming_back_is_rejected(self) -> None:
        """This is the lexical layer's own job, and nothing else does it.

        `../outside/back_in/x.py`, where `back_in` is a symlink pointing INTO
        the worktree, resolves to a path inside the root — so the realpath
        prefix check admits it. The descriptor walk would then have had to step
        out of the root through `..` to get there, which is a primitive this
        module must not hand the model at all. Any surviving `..` component is
        therefore refused before either of the other layers runs.
        """
        (self.root / "inner").mkdir()
        os.symlink(str(self.root / "inner"), self.outside / "back_in")
        rel = f"../{self.outside.name}/back_in/x.py"

        # MECHANISM: prove the symbolic layer really would have admitted it.
        norm = os.path.normpath(os.path.join(self.real_root, rel))
        self.assertTrue(
            os.path.realpath(os.path.dirname(norm)).startswith(self.real_root + os.sep)
        )

        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), rel, "pwned")
        self.assertFalse((self.root / "inner" / "x.py").exists())


# --------------------------------------------------------------------------
# §10.4d — the symlinked-parent escape, write side and read side
# --------------------------------------------------------------------------


class WriteFilePathEscape(_Tree):
    """A symlinked PARENT (`ln -s /etc sub`) makes realpath(dirname(sub/passwd))
    resolve to /etc. The parent must be realpath'd and prefix-checked."""

    def setUp(self) -> None:
        super().setUp()
        os.symlink(str(self.outside), self.root / "sub")

    def test_MECHANISM_the_planted_parent_really_is_a_symlink(self) -> None:
        _assert_is_symlink(self, self.root / "sub")
        self.assertEqual(
            os.path.realpath(str(self.root / "sub")),
            os.path.realpath(str(self.outside)),
        )

    def test_write_through_symlinked_parent_is_rejected(self) -> None:
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), "sub/passwd", "pwned")
        self.assertFalse((self.outside / "passwd").exists())

    def test_read_through_symlinked_parent_is_rejected(self) -> None:
        (self.outside / "passwd").write_text("root:x:0:0\n")
        with self.assertRaises(PathEscape):
            tool_read_file(str(self.root), "sub/passwd")

    def test_listing_through_a_symlinked_parent_is_rejected(self) -> None:
        (self.outside / "passwd").write_text("root:x:0:0\n")
        with self.assertRaises(PathEscape):
            tool_list_files(str(self.root), "sub")

    def test_dotdot_and_absolute_are_rejected(self) -> None:
        for bad in ("../x", "/etc/passwd", "a/../../x"):
            with self.assertRaises(PathEscape):
                safe_path(str(self.root), bad, for_write=True)

    def test_new_file_in_real_subdir_is_allowed(self) -> None:
        (self.root / "real").mkdir()
        p = safe_path(str(self.root), "real/new.py", for_write=True)
        self.assertTrue(p.startswith(os.path.realpath(str(self.root)) + os.sep))


class SymlinkedFinalComponentWrite(_Tree):
    """The twin the brief's parent-only rule misses.

    `write_file("link", …)` where `link` is already a symlink has an innocent
    parent — the parent IS the root — so a parent-only realpath passes it, and
    the write then follows the symlink out of the tree. The final component
    must be checked too, and the actual open must be O_NOFOLLOW so the check
    cannot be raced.
    """

    def setUp(self) -> None:
        super().setUp()
        self.target = self.outside / "target"
        self.target.write_text("original\n")
        os.symlink(str(self.target), self.root / "link")

    def test_MECHANISM_the_planted_final_component_really_is_a_symlink(self) -> None:
        _assert_is_symlink(self, self.root / "link")

    def test_MECHANISM_a_plain_open_really_would_write_through_it(self) -> None:
        """Proves the escape is live on this filesystem: without the guard,
        an ordinary open() of the same path rewrites the outside file."""
        with open(self.root / "link", "w", encoding="utf-8") as fh:
            fh.write("plain-open-got-through\n")
        self.assertEqual(self.target.read_text(), "plain-open-got-through\n")

    def test_write_to_a_symlinked_final_component_is_rejected(self) -> None:
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), "link", "pwned")
        self.assertEqual(self.target.read_text(), "original\n")

    def test_write_to_a_DANGLING_outward_symlink_is_rejected(self) -> None:
        os.symlink(str(self.outside / "not-yet"), self.root / "later")
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), "later", "pwned")
        self.assertFalse((self.outside / "not-yet").exists())

    def test_write_through_an_INWARD_symlink_is_refused_but_is_not_an_escape(
        self,
    ) -> None:
        """Deliberately stricter than "stay inside": these tools never write
        through a symlink at all, matching the byte walk's never-dereference
        rule (§6.5). The model can name the target directly.

        It is an `error:` and NOT a PathEscape, because the path does stay
        inside the worktree — a repo that legitimately contains a symlink must
        not raise a security signal on every ordinary write."""
        (self.root / "realfile").write_text("keep\n")
        os.symlink("realfile", self.root / "alias")
        out = tool_write_file(str(self.root), "alias", "pwned")
        self.assertIn("error:", out)
        self.assertIn("symlink", out)
        self.assertEqual((self.root / "realfile").read_text(), "keep\n")

    def test_a_symlinked_directory_component_inside_the_tree_is_refused(self) -> None:
        (self.root / "real").mkdir()
        os.symlink("real", self.root / "alias")
        out = tool_write_file(str(self.root), "alias/x.py", "pwned")
        self.assertIn("error:", out)
        self.assertFalse((self.root / "real" / "x.py").exists())


class ReadFileNeverLeavesTheTree(_Tree):
    def test_an_absolute_target_symlink_is_rejected(self) -> None:
        secret = self.outside / "id_ed25519"
        secret.write_text("PRIVATE KEY\n")
        os.symlink(str(secret), self.root / "leak_abs")
        _assert_is_symlink(self, self.root / "leak_abs")
        with self.assertRaises(PathEscape):
            tool_read_file(str(self.root), "leak_abs")

    def test_a_relative_escape_symlink_is_rejected(self) -> None:
        secret = self.outside / "id_ed25519"
        secret.write_text("PRIVATE KEY\n")
        os.symlink("../outside/id_ed25519", self.root / "leak_rel")
        with self.assertRaises(PathEscape):
            tool_read_file(str(self.root), "leak_rel")

    def test_an_inward_symlink_returns_an_error_not_the_target_contents(self) -> None:
        (self.root / "realfile").write_text("secret-but-in-tree\n")
        os.symlink("realfile", self.root / "alias")
        out = tool_read_file(str(self.root), "alias")
        self.assertIn("error:", out)
        self.assertNotIn("secret-but-in-tree", out)

    def test_a_fifo_reports_an_error_and_does_not_hang(self) -> None:
        os.mkfifo(self.root / "pipe")
        started = time.monotonic()
        out = tool_read_file(str(self.root), "pipe")
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("error:", out)

    def test_a_directory_reports_an_error(self) -> None:
        (self.root / "d").mkdir()
        self.assertIn("error:", tool_read_file(str(self.root), "d"))

    def test_a_missing_file_reports_an_error(self) -> None:
        self.assertIn("error:", tool_read_file(str(self.root), "nope.py"))

    def test_a_regular_file_is_returned(self) -> None:
        (self.root / "a.py").write_text("x = 1\n")
        self.assertEqual(tool_read_file(str(self.root), "a.py"), "x = 1\n")

    def test_a_binary_file_is_refused(self) -> None:
        (self.root / "b.bin").write_bytes(b"\x7fELF\0\0\0\0rest")
        self.assertIn("binary", tool_read_file(str(self.root), "b.bin"))

    def test_an_oversized_file_is_truncated_with_a_marker(self) -> None:
        (self.root / "big.txt").write_text("a" * (MAX_READ + 4096))
        out = tool_read_file(str(self.root), "big.txt")
        self.assertIn("truncated", out)
        self.assertLess(len(out), MAX_READ + 200)

    def test_MECHANISM_an_oversized_file_is_never_fully_read_into_memory(self) -> None:
        """Truncating the *result* is not the same property as bounding the
        *read*: a cap applied after the fact still pulls the whole file into
        the MCP server's address space. Count the bytes the syscall returns."""
        (self.root / "big.txt").write_text("a" * (MAX_READ * 3))
        real_read = os.read
        total = 0

        def counting(fd: int, n: int) -> bytes:
            nonlocal total
            chunk = real_read(fd, n)
            total += len(chunk)
            return chunk

        with mock.patch("os.read", counting):
            tool_read_file(str(self.root), "big.txt")
        self.assertLessEqual(total, MAX_READ + 1)


# --------------------------------------------------------------------------
# write_file: the happy paths still work
# --------------------------------------------------------------------------


class WriteFileHappyPath(_Tree):
    def test_a_new_file_is_created(self) -> None:
        out = tool_write_file(str(self.root), "a.py", "x = 1\n")
        self.assertIn("wrote", out)
        self.assertEqual((self.root / "a.py").read_text(), "x = 1\n")

    def test_missing_parent_directories_are_created(self) -> None:
        tool_write_file(str(self.root), "pkg/sub/mod.py", "y = 2\n")
        self.assertEqual((self.root / "pkg/sub/mod.py").read_text(), "y = 2\n")

    def test_an_existing_file_is_fully_overwritten(self) -> None:
        (self.root / "a.py").write_text("a very long previous body\n")
        tool_write_file(str(self.root), "a.py", "short\n")
        self.assertEqual((self.root / "a.py").read_text(), "short\n")

    def test_writing_over_a_directory_reports_an_error(self) -> None:
        (self.root / "d").mkdir()
        self.assertIn("error:", tool_write_file(str(self.root), "d", "x"))

    def test_the_root_itself_cannot_be_written(self) -> None:
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), ".", "x")


# --------------------------------------------------------------------------
# The check-then-use window: a SEPARATE PROCESS flips the parent
# --------------------------------------------------------------------------


_SWAPPER = textwrap.dedent(
    """
    import os, sys, time
    root, outside, trigger, done = sys.argv[1:5]
    sub = os.path.join(root, "sub")
    deadline = time.time() + 30
    while not os.path.exists(trigger):
        if time.time() > deadline:
            sys.exit(1)
        time.sleep(0.002)
    # Rename rather than remove, so the original directory stays reachable:
    # a guard that pinned a descriptor writes into `sub_moved`, a guard that
    # re-resolves the name writes through the symlink into `outside`.
    os.rename(sub, os.path.join(root, "sub_moved"))
    os.symlink(outside, sub)
    with open(done, "w") as fh:
        fh.write("swapped")
    """
)


class ParentSwapAfterTheCheck(_Tree):
    """The check-then-use window between resolving the parent and opening the
    file, made deterministic.

    A tight opportunistic race is a coin flip that reports a confident false
    negative when the coin lands wrong, so the window is instead widened to a
    real handshake and both implementations get the *same* widened window. The
    attacker is a separate PROCESS: a thread loses this to the GIL.
    """

    def setUp(self) -> None:
        super().setUp()
        (self.root / "sub").mkdir()
        self.trigger = self.tmp / "go"
        self.done = self.tmp / "swapped"
        script = self.tmp / "swap.py"
        script.write_text(_SWAPPER)
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(self.root),
                str(self.outside),
                str(self.trigger),
                str(self.done),
            ]
        )
        self.addCleanup(self.proc.wait)
        self.addCleanup(self.proc.kill)

    def _seam(self) -> None:
        """Called from inside the victim, between the check and the open."""
        self.trigger.write_text("go")
        deadline = time.monotonic() + 20
        while not self.done.exists():
            if time.monotonic() > deadline:
                raise AssertionError("the attacker process never swapped")
            time.sleep(0.002)

    def _assert_swap_happened(self) -> None:
        _assert_is_symlink(self, self.root / "sub")
        self.assertTrue((self.root / "sub_moved").is_dir())

    def test_MECHANISM_a_check_then_open_really_is_defeated_by_this_swap(self) -> None:
        """The brief's algorithm, reproduced here so the control stays valid
        after the shipped module is hardened. If this does not escape, the
        hardened assertion below proves nothing."""
        real_root = os.path.realpath(str(self.root))
        joined = os.path.normpath(os.path.join(real_root, "sub/passwd"))
        parent_real = os.path.realpath(os.path.dirname(joined))
        candidate = os.path.join(parent_real, os.path.basename(joined))
        self.assertTrue(candidate.startswith(real_root + os.sep))

        self._seam()

        os.makedirs(os.path.dirname(candidate), exist_ok=True)
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write("pwned")

        self._assert_swap_happened()
        self.assertEqual((self.outside / "passwd").read_text(), "pwned")

    def test_tool_write_file_lands_on_the_pinned_inode_not_through_the_swap(
        self,
    ) -> None:
        from coding_agent import tools as tools_mod

        real_descend = tools_mod._descend

        def hooked(real_root: str, parts, *, create: bool) -> int:  # type: ignore[no-untyped-def]
            fd = real_descend(real_root, parts, create=create)
            self._seam()  # the parent is pinned; now swap the name
            return fd

        with mock.patch.object(tools_mod, "_descend", hooked):
            out = tool_write_file(str(self.root), "sub/passwd", "pwned")

        self._assert_swap_happened()
        self.assertFalse((self.outside / "passwd").exists())
        self.assertEqual((self.root / "sub_moved" / "passwd").read_text(), "pwned")
        self.assertIn("wrote", out)


# --------------------------------------------------------------------------
# list_files
# --------------------------------------------------------------------------


class ListFiles(_Tree):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "a.py").write_text("x\n")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "b.py").write_text("y\n")

    def test_files_are_listed_relative_to_the_root(self) -> None:
        out = tool_list_files(str(self.root)).splitlines()
        self.assertEqual(out, ["a.py", "pkg/b.py"])

    def test_a_subdirectory_listing_is_still_root_relative(self) -> None:
        self.assertEqual(
            tool_list_files(str(self.root), "pkg").splitlines(), ["pkg/b.py"]
        )

    def test_an_empty_directory_says_so(self) -> None:
        (self.root / "empty").mkdir()
        self.assertEqual(tool_list_files(str(self.root), "empty"), "(empty)")

    def test_the_git_metadata_is_never_listed(self) -> None:
        (self.root / ".git").write_text("gitdir: /elsewhere\n")
        (self.root / "nested").mkdir()
        (self.root / "nested" / ".git").mkdir()
        (self.root / "nested" / ".git" / "config").write_text("[core]\n")
        out = tool_list_files(str(self.root))
        self.assertNotIn(".git", out)

    def test_a_symlinked_directory_is_listed_but_not_descended(self) -> None:
        """Both halves, because the one-sided version could not fail.

        `assertNotIn("loot.txt", out)` passes whether the symlink is listed
        and not descended — what this module documents — or absent from the
        listing altogether, which is what it actually did: `os.fwalk`
        classifies names with `entry.is_dir()`, which FOLLOWS symlinks, so a
        symlink-to-directory landed in `dirnames`, was refused by the samestat
        check, and was never in `filenames` to be printed. A path that exists
        and is not shown is an under-show, and a docstring that says otherwise
        is the "the tool's own description is a lie" class.
        """
        (self.outside / "loot.txt").write_text("secret\n")
        os.symlink(str(self.outside), self.root / "peek")
        os.symlink(str(self.outside / "loot.txt"), self.root / "peekfile")
        os.symlink("nowhere", self.root / "dangling")
        _assert_is_symlink(self, self.root / "peek")

        lines = tool_list_files(str(self.root)).splitlines()

        # LISTED — all three shapes, including the one pointing at a directory.
        for name in ("peek", "peekfile", "dangling"):
            self.assertIn(name, lines, f"{name} vanished from the listing")
        # NOT DESCENDED, and nothing reached through.
        self.assertNotIn("loot.txt", "\n".join(lines))
        self.assertNotIn("peek/loot.txt", lines)

    def test_a_symlink_the_ignore_predicate_excludes_is_not_listed(self) -> None:
        """The symlink is queried WITHOUT a trailing slash, as `walk.py`
        queries one — the two sides must ask the predicate the same question
        or the diff and the model's view of the tree disagree."""
        os.symlink(str(self.outside), self.root / "peek")
        asked: list[str] = []

        def ign(path: str) -> bool:
            asked.append(path)
            return path == "peek"

        lines = tool_list_files(str(self.root), ".", ignore=ign).splitlines()
        self.assertNotIn("peek", lines)
        self.assertIn("peek", asked, "the symlink was never offered to `ignore`")
        self.assertNotIn("peek/", asked, "a symlink was queried as a directory")

    def test_a_newline_in_a_filename_cannot_forge_a_listing_line(self) -> None:
        (self.root / "ev\nil.py").write_text("x\n")
        out = tool_list_files(str(self.root))
        self.assertEqual(len(out.splitlines()), 3)
        self.assertIn("\\n", out)

    def test_the_ignore_callback_hides_files_and_prunes_directories(self) -> None:
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "junk.js").write_text("j\n")

        def ign(path: str) -> bool:
            return path in ("node_modules/", "a.py")

        out = tool_list_files(str(self.root), ".", ignore=ign).splitlines()
        self.assertEqual(out, ["pkg/b.py"])

    def test_the_listing_is_capped(self) -> None:
        many = self.root / "many"
        many.mkdir()
        for i in range(MAX_LIST + 20):
            (many / f"f{i:05d}.txt").write_text("x")
        out = tool_list_files(str(self.root))
        self.assertIn("capped", out)
        self.assertLessEqual(len(out.splitlines()), MAX_LIST + 1)

    def test_the_listing_stops_descending_at_MAX_DEPTH(self) -> None:
        """SF-8. `MAX_DEPTH` was unpinned — mutation `T12` survived. It is the
        companion to `MAX_LIST`: one bounds a wide tree, this bounds a deep
        one, and the sandbox chooses both. `mkdir -p a/a/a/...` is cheap.
        """
        deep = self.root
        for _ in range(MAX_DEPTH + 6):
            deep = deep / "d"
            deep.mkdir()
        (deep / "bottom.txt").write_text("x\n")
        shallow = self.root
        for _ in range(MAX_DEPTH - 1):
            shallow = shallow / "d"
        (shallow / "reachable.txt").write_text("x\n")

        lines = tool_list_files(str(self.root)).splitlines()

        # MECHANISM: the walk really does get deep, so the absence below is
        # the cap and not a listing that stopped for some other reason.
        self.assertIn(str(shallow.relative_to(self.root) / "reachable.txt"), lines)
        self.assertNotIn(str(deep.relative_to(self.root) / "bottom.txt"), lines)
        self.assertTrue(lines)
        self.assertLessEqual(max(ln.count("/") for ln in lines), MAX_DEPTH)


# --------------------------------------------------------------------------
# run_command and the schemas
# --------------------------------------------------------------------------


class RunCommandDelegates(unittest.TestCase):
    def test_the_command_is_forwarded_verbatim_and_the_flag_dropped(self) -> None:
        seen: dict[str, object] = {}

        async def fake(container: str, cmd: str, *, timeout_s: float):  # type: ignore[no-untyped-def]
            seen.update(container=container, cmd=cmd, timeout_s=timeout_s)
            return 3, "out", True

        raw = "pytest -q && echo 'do not quote me' | tee /tmp/x"
        with mock.patch("coding_agent.tools.exec_in_container", fake):
            rc, out = asyncio.run(tool_run_command("cid", raw, timeout_s=12.5))
        self.assertEqual((rc, out), (3, "out"))
        self.assertEqual(seen, {"container": "cid", "cmd": raw, "timeout_s": 12.5})

    def test_a_timeout_is_returned_not_raised(self) -> None:
        async def fake(container: str, cmd: str, *, timeout_s: float):  # type: ignore[no-untyped-def]
            return 124, "partial\n[... command timed out after 5s ...]\n", False

        with mock.patch("coding_agent.tools.exec_in_container", fake):
            rc, out = asyncio.run(tool_run_command("cid", "sleep 99", timeout_s=5))
        self.assertEqual(rc, 124)
        self.assertIn("timed out", out)


class ToolSchemas(unittest.TestCase):
    def test_exactly_the_four_tools_are_offered(self) -> None:
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        self.assertEqual(
            names, ["list_files", "read_file", "write_file", "run_command"]
        )

    def test_the_payload_is_json_serialisable(self) -> None:
        self.assertEqual(json.loads(json.dumps(TOOL_SCHEMAS)), TOOL_SCHEMAS)

    def test_every_tool_declares_its_required_arguments(self) -> None:
        required = {
            s["function"]["name"]: s["function"]["parameters"].get("required", [])
            for s in TOOL_SCHEMAS
        }
        self.assertEqual(
            required,
            {
                "list_files": [],
                "read_file": ["path"],
                "write_file": ["path", "content"],
                "run_command": ["cmd"],
            },
        )


class PlatformSupportsTheHardening(unittest.TestCase):
    """The syscall-level guard is not optional. If a platform ever lacks
    openat(), this test goes red rather than the guard silently degrading to
    the check-then-use it replaces."""

    def test_dir_fd_relative_opens_are_available(self) -> None:
        self.assertIn(os.open, os.supports_dir_fd)
        self.assertIn(os.mkdir, os.supports_dir_fd)
        self.assertTrue(hasattr(os, "O_NOFOLLOW"))
        self.assertTrue(hasattr(errno, "ELOOP"))


if __name__ == "__main__":
    unittest.main()
