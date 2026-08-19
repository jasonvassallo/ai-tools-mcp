"""Byte-level tree snapshot for coding_agent.

SECURITY SPINE (spec §6.5): this module NEVER invokes git and NEVER
dereferences a symlink. It replaced `git add -A`/`git diff` after adversarial
review demonstrated host RCE via a sandbox-controlled `.git`; a later pass
demonstrated that a naive os.walk+read() then leaked host secrets through
planted symlinks; a FIFTH pass demonstrated that classifying with `lstat` and
then acting on that classification BY PATH leaked them again through a race —
a separate process swapped the path for a symlink inside the window and the
host read a planted private key into the snapshot on walk #28 (file vector)
and walk #696 (directory vector). All three are pinned by
test_coding_agent_security.py.

Out of scope, by spec §6.5 and not by oversight: a HARDLINK is indistinguishable
from a regular file to both `lstat` and `fstat`, so nothing below defends
against one. Safety there rests on the container mount model — host paths are
not in the sandbox's mount namespace, and `/work` is a different device from
the image and tmpfs, so `link()` is EXDEV. Adding a second bind-mount reopens
it. (Beware when probing this: `os.link` FOLLOWS a symlink source on macOS, so
a probe that links a symlink is measuring the hardlink vector, not this one.)

The race fix in one sentence: the walk holds every directory OPEN and reaches
each child through that descriptor, every open is `O_NOFOLLOW`, and every
opened object's `st_ino`/`st_dev` must equal the `lstat` that classified it —
so a path is never re-resolved after it has been checked.

Why the race had to be closed HERE rather than by the caller: the end-of-loop
diff is already safe because the container is destroyed before the host reads
(§6.5 rule 3), but the per-turn no-progress hash runs with the container
alive, and §6.5 rule 1 states that window cannot be closed by killing it.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass

# O_NOFOLLOW is the whole fix: it makes the open FAIL (ELOOP) when the final
# component has been swapped for a symlink since it was classified, instead of
# silently resolving that symlink in the HOST namespace. O_DIRECTORY does the
# same job for a descent. O_NONBLOCK covers the other half of a swap: a FIFO
# or device dropped in after classification would hang a blocking open
# forever; POSIX gives O_NONBLOCK no effect on regular files, so it costs
# nothing on the path that matters.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_READ_CHUNK = 1 << 20

# A racing writer need not be hostile — the model's own background process, or
# an editor's atomic save, can invalidate a classification honestly. Re-classify
# a bounded number of times so a real change is recorded rather than silently
# dropped ("under-showing a real change is the danger", §6.5). Bounded, so no
# writer can spin the walk.
_RECLASSIFY_ATTEMPTS = 3


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "file" | "symlink"
    data: bytes  # file bytes, or the symlink TARGET string as utf-8


@dataclass(frozen=True)
class TreeSnapshot:
    entries: dict[str, Entry]


@dataclass
class _Frame:
    """One directory the walk is enumerating. Owns `fd` until it is popped."""

    fd: int
    prefix: str  # "" at the root, otherwise "sub/dir/"
    names: list[str]  # remaining children, reverse-sorted so pop() ascends


def _close_quietly(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _same_object(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_ino == b.st_ino and a.st_dev == b.st_dev


def _pending_names(dir_fd: int) -> list[str]:
    return sorted(os.listdir(dir_fd), reverse=True)


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, _READ_CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _open_child_dir(
    dir_fd: int, name: str, prefix: str, classified: os.stat_result
) -> _Frame | None:
    """Open `name` as a directory THROUGH its parent's descriptor.

    Returns None when the object is no longer the directory that was
    classified — ELOOP because O_NOFOLLOW refused a swapped-in symlink,
    ENOTDIR, or an st_ino/st_dev mismatch. The caller re-classifies.
    """
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except OSError:
        return None
    keep = False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_object(opened, classified):
            return None
        frame = _Frame(fd, prefix, _pending_names(fd))
        keep = True
        return frame
    except OSError:
        return None
    finally:
        if not keep:
            _close_quietly(fd)


def _read_regular(
    dir_fd: int, name: str, rel: str, classified: os.stat_result
) -> Entry | None:
    """Read `name` THROUGH its parent's descriptor, or return None.

    The bytes are read only after `fstat` on the OPEN descriptor confirms the
    object is still a regular file and still the same inode on the same device
    as the `lstat` that classified it, so the descriptor cannot address
    anything the walk did not choose to read.
    """
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_object(opened, classified):
            return None
        return Entry(rel, "file", _read_all(fd))
    except OSError:
        return None
    finally:
        _close_quietly(fd)


def _visit(
    dir_fd: int, name: str, rel: str, is_ignored: Callable[[str], bool]
) -> tuple[Entry | None, _Frame | None]:
    """Classify ONE child of the open directory `dir_fd`, and act on it.

    Every syscall here is `dir_fd`-relative, so no ancestor directory can be
    swapped underneath the walk. Returns at most one of (entry, child frame);
    the caller owns the returned frame's descriptor.
    """
    # Re-classifying must not re-query the injected callable: the query string
    # is the same, so the answer is the same, and Task 3's matcher should see
    # the same call pattern it would see on a quiescent tree.
    decided: dict[str, bool] = {}

    def ignored(query: str) -> bool:
        if query not in decided:
            decided[query] = is_ignored(query)
        return decided[query]

    for _attempt in range(_RECLASSIFY_ATTEMPTS):
        try:
            classified = os.lstat(name, dir_fd=dir_fd)  # lstat: never follows
        except OSError:
            return None, None
        mode = classified.st_mode
        if stat.S_ISLNK(mode):
            # Deliberately NOT ino/dev re-verified, unlike the two branches
            # below. `readlink` never dereferences, whatever now occupies the
            # name, so no host CONTENT can reach the snapshot through here —
            # the worst a winning racer achieves is having a different link's
            # target TEXT recorded, which is data the sandbox already controls.
            # There is no descriptor to fstat either: O_NOFOLLOW would refuse
            # to open a symlink at all, which is the point of it.
            try:
                target = os.readlink(name, dir_fd=dir_fd)
            except OSError:
                continue  # swapped or vanished; re-classify
            if ignored(rel):
                return None, None
            data = target.encode("utf-8", "surrogateescape")
            return Entry(rel, "symlink", data), None
        if stat.S_ISDIR(mode):
            if ignored(rel + "/"):
                return None, None
            frame = _open_child_dir(dir_fd, name, rel + "/", classified)
            if frame is not None:
                return None, frame
            continue  # lost the race; re-classify
        if not stat.S_ISREG(mode):
            return None, None  # FIFO / socket / device: skip
        if ignored(rel):
            return None, None
        entry = _read_regular(dir_fd, name, rel, classified)
        if entry is not None:
            return entry, None
    # KNOWN GAP, escalated rather than papered over. Losing the race
    # _RECLASSIFY_ATTEMPTS times in a row drops this entry from the snapshot
    # with no signal — indistinguishable from "never existed". Confidentiality
    # holds (no host bytes are ever read on a losing attempt), but it is a
    # narrow, timing-gated instance of the under-showing failure §6.5 calls the
    # dangerous direction. It cannot be surfaced from here without changing
    # TreeSnapshot, which downstream tasks are written against; the fix belongs
    # with whoever owns the §9 error path — a dropped-path count carried
    # alongside the diff, so the gate can say "N paths could not be read
    # consistently" instead of silently showing fewer.
    return None, None


def snapshot_tree(root: str, is_ignored: Callable[[str], bool]) -> TreeSnapshot:
    """Walk `root` with lstat semantics.

    - symlinks are recorded as their target TEXT and never followed
      (git's 120000 mode); symlinked directories are not descended
    - only regular files are read; FIFOs/sockets/devices are skipped
    - the top-level `.git` entry is opaque (never read, never listed)
    - `is_ignored(relpath)` gates NEW paths only; the caller enforces
      tracked-awareness (basetree paths are always included) — see diff().
      DIRECTORIES are queried with a TRAILING SLASH, files and symlinks
      without; Task 3's callable depends on that distinction.

    Descriptors: each frame on the stack owns exactly one open directory, and a
    child's descriptor is opened only when the walk descends into it, so the
    number of open descriptors is the tree's DEPTH, never its breadth. Every
    frame is closed when it is exhausted, and the `finally` closes whatever is
    still on the stack if anything raises.
    """
    root = os.path.abspath(root)
    entries: dict[str, Entry] = {}

    try:
        root_fd = os.open(root, _DIR_FLAGS)
    except OSError:
        return TreeSnapshot(entries)
    try:
        stack = [_Frame(root_fd, "", _pending_names(root_fd))]
    except OSError:
        _close_quietly(root_fd)
        return TreeSnapshot(entries)

    try:
        while stack:
            frame = stack[-1]
            if not frame.names:
                stack.pop()
                _close_quietly(frame.fd)
                continue
            name = frame.names.pop()
            rel = frame.prefix + name
            if rel == ".git" or rel.startswith(".git/"):
                continue
            entry, child = _visit(frame.fd, name, rel, is_ignored)
            if child is not None:
                stack.append(child)
            if entry is not None:
                entries[rel] = entry
    finally:
        for pending in stack:
            _close_quietly(pending.fd)
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
