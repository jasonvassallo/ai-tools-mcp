#!/usr/bin/env python3
"""Security regression tests for coding_agent.

Every test here pins an attack that was DEMONSTRATED (not hypothesised)
during adversarial review of the design spec, 2026-08-16. Read
docs/superpowers/specs/2026-08-16-coding-agent-design.md §6.5/§6.6/§10.

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path

from coding_agent.walk import snapshot_tree, tree_hash


class SymlinkExfiltration(unittest.TestCase):
    """Pass 3: a naive os.walk+open().read() dereferenced a planted
    /work/leak -> ~/.ssh/id_ed25519 and read the host secret INTO the diff."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
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

    def test_special_files_are_skipped_not_read(self):
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


if __name__ == "__main__":
    unittest.main()
