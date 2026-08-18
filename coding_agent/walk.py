"""Byte-level tree snapshot for coding_agent.

SECURITY SPINE (spec §6.5): this module NEVER invokes git and NEVER
dereferences a symlink. It replaced `git add -A`/`git diff` after adversarial
review demonstrated host RCE via a sandbox-controlled `.git`; a later pass
demonstrated that a naive os.walk+read() then leaked host secrets through
planted symlinks. Both are pinned by test_coding_agent_security.py.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "file" | "symlink"
    data: bytes  # file bytes, or the symlink TARGET string as utf-8


@dataclass(frozen=True)
class TreeSnapshot:
    entries: dict[str, Entry]


def snapshot_tree(root: str, is_ignored: Callable[[str], bool]) -> TreeSnapshot:
    """Walk `root` with lstat semantics.

    - symlinks are recorded as their target TEXT and never followed
      (git's 120000 mode); symlinked directories are not descended
    - only regular files are read; FIFOs/sockets/devices are skipped
    - the top-level `.git` entry is opaque (never read, never listed)
    - `is_ignored(relpath)` gates NEW paths only; the caller enforces
      tracked-awareness (basetree paths are always included) — see diff()
    """
    root = os.path.abspath(root)
    entries: dict[str, Entry] = {}

    def rel(p: str) -> str:
        return os.path.relpath(p, root).replace(os.sep, "/")

    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            names = sorted(os.listdir(cur))
        except OSError:
            continue
        for name in names:
            full = os.path.join(cur, name)
            r = rel(full)
            if r == ".git" or r.startswith(".git/"):
                continue
            try:
                st = os.lstat(full)  # lstat: never follows
            except OSError:
                continue
            mode = st.st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(full)
                if not is_ignored(r):
                    entries[r] = Entry(
                        r, "symlink", target.encode("utf-8", "surrogateescape")
                    )
                continue  # never descend a symlinked dir
            if stat.S_ISDIR(mode):
                if not is_ignored(r + "/"):
                    stack.append(full)
                continue
            if not stat.S_ISREG(mode):
                continue  # FIFO / socket / device: skip
            if is_ignored(r):
                continue
            with open(full, "rb") as fh:
                entries[r] = Entry(r, "file", fh.read())
    return TreeSnapshot(entries)


def tree_hash(snap: TreeSnapshot) -> str:
    h = hashlib.sha256()
    for path in sorted(snap.entries):
        e = snap.entries[path]
        h.update(e.kind.encode())
        h.update(b"\0")
        h.update(path.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
        h.update(hashlib.sha256(e.data).digest())
    return h.hexdigest()
