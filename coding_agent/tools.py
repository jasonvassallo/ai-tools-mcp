"""The four tools the model sees, and the host-side path guard (spec §5, §6).

`list_files` / `read_file` / `write_file` run **on the host**, straight against
the worktree path; only `run_command` crosses into the container (spec §5).
That asymmetry is deliberate and settled — and it is why this module, not the
container, is the boundary for host file I/O. The model is untrusted, so
`safe_path` is the only thing standing between a model-supplied relative path
and the operator's filesystem.

Three layers enforce containment. None of them is decoration, and each catches
something the others do not:

1. **Lexical** — normalise `root/rel`, take it back relative to the root, and
   refuse if any component survives as `..`. Pure string work, no filesystem,
   so it holds even where the filesystem would lie.
2. **Symbolic** — `realpath`, and for a not-yet-existing file `realpath` of the
   **parent** (there is nothing to resolve at the leaf, while the parent may be
   `ln -s /etc sub`), then prefix-check against `realpath(root) + os.sep`. The
   trailing separator is half of that check: without it the sibling directory
   `work-evil` passes `startswith(".../work")`. The **leaf** is checked too —
   a parent-only rule passes `write_file("link", …)` where `link` is already a
   symlink out of the tree, and the write then follows it.
3. **Syscall** — layers 1 and 2 are a check-then-use, and the sandbox can leave
   a background process flipping `sub` between a real directory and a symlink
   (§6.5 proved that pattern lands). So the I/O never re-resolves a path by
   name: it walks the components from a descriptor on the root with
   `O_NOFOLLOW` at every step, and opens the leaf relative to the pinned parent
   descriptor. A swap after the walk cannot redirect the write, because the
   descriptor names an inode, not a path.

Deliberately stricter than "stay inside": these tools never read or write
*through* a symlink at all, even one that resolves inside the worktree. That
matches the byte walk's never-dereference rule (§6.5), removes a whole
check-then-use class, and costs the model nothing — it can name the target. An
in-tree symlink is refused with an ordinary `error:` result, not a
`PathEscape`; only leaving the worktree is an escape.

What this module does NOT defend against, stated plainly: nothing here bounds
what `run_command` does to the *contents* of the mount. It cannot — and it
does not need to, because the container's only writable path is the worktree
and every byte of it is re-read by the §6.5 walk before a human sees the diff.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable, Sequence
from typing import Any

from .sandbox import exec_in_container
from .walk import _quote_path

MAX_READ = 256 * 1024
MAX_LIST = 2000

# A listing is a convenience, not the gate — the diff is (§5.1). So a tree deep
# enough to exhaust the fd table or the C stack is simply not descended past
# this point rather than being allowed to turn `list_files` into a crash.
MAX_DEPTH = 64

_READ_CHUNK = 1 << 20
_NEW_FILE_MODE = 0o644
_NEW_DIR_MODE = 0o755
_GIT_META = ".git"

# The same flag discipline `walk.py` uses, for the same reasons. O_NOFOLLOW
# makes the open FAIL (ELOOP) when the component is a symlink instead of
# resolving it in the HOST namespace; O_DIRECTORY does that job for a descent;
# O_NONBLOCK means a FIFO dropped in place of a regular file returns an error
# instead of hanging the server forever, and is a no-op on regular files.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK

# What the kernel says when O_NOFOLLOW refuses a symlink. Linux and macOS use
# ELOOP; some BSDs use EMLINK. Both mean "a path component was a symlink".
_SYMLINK_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK})

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Recursively list files under a directory of the repository "
                "(respects .gitignore)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite (or create) a text file in the repository with the "
                "given content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the repository root (tests, linters). "
                "No network is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


class PathEscape(ValueError):
    """A model-supplied path that does not resolve inside the worktree.

    Raised, never returned as a tool result string, so a caller cannot mistake
    an attempted escape for an ordinary failed read. Ordinary I/O failures —
    missing file, directory, binary, FIFO, and the refusal to traverse a
    symlink that *does* stay inside — come back as an `error: …` string the
    model can act on, because an agentic loop that crashes on a typo is
    useless. `PathEscape` is the one exception these tools raise.

    The line is drawn at "leaves the worktree", not at "touched a symlink", so
    that a caller can treat a `PathEscape` as a security event. A repository
    that legitimately contains an in-tree symlink would otherwise raise one on
    every ordinary read and train the operator to ignore it.
    """


def _close_quietly(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _errno_name(exc: OSError) -> str:
    code = exc.errno
    if code is None:
        return "OSERROR"
    return errno.errorcode.get(code, f"ERRNO{code}")


def _why(exc: OSError) -> str:
    """The model-facing reason an open failed.

    A symlink reaching this layer at all has already been cleared by the
    symbolic layer as resolving *inside* the worktree — an outward one raised
    `PathEscape` before any syscall ran. The one exception is a component
    swapped for a symlink *after* that check, and it is deliberately reported
    the same way: telling the two apart would mean re-resolving the name, which
    is re-opening the very window `O_NOFOLLOW` just closed. Either way the
    operation is refused, which is the property that matters.
    """
    if exc.errno in _SYMLINK_ERRNOS:
        return "path component is a symlink and is not traversed"
    return _errno_name(exc)


def _escapes(candidate: str, real_root: str) -> bool:
    """The prefix check, with the half that is easy to drop.

    `+ os.sep` is load-bearing: without it a sibling directory named
    `work-evil` string-starts-with `.../work` and every path under it passes.

    On a case-insensitive or unicode-normalising filesystem (APFS, HFS+) this
    comparison is stricter than the kernel's: `../WORK/x` names a file inside
    the root and is nonetheless rejected, because `realpath` does not
    canonicalise case. That is the fail-CLOSED direction — a legitimate path
    refused, never a foreign one admitted — and is left as-is.
    """
    return candidate != real_root and not candidate.startswith(real_root + os.sep)


def _validate(root: str, rel: str, *, for_write: bool) -> tuple[str, list[str], str]:
    """The lexical and symbolic layers. Returns (real_root, parts, candidate).

    `parts` are the components **as the model named them** — never the
    symlink-resolved ones — because they are what the syscall layer walks, and
    walking the resolved form would silently re-follow every symlink `realpath`
    had already followed.
    """
    if not isinstance(rel, str):
        raise PathEscape(f"path must be a string, got {type(rel).__name__}")
    if not rel:
        raise PathEscape("path must be non-empty")
    if "\0" in rel:
        # realpath() happily carries a NUL through; open() then raises a bare
        # ValueError from inside posixpath, which a caller catching PathEscape
        # would not catch. Refuse it here, as the escape attempt it is.
        raise PathEscape("path contains a NUL byte")
    if os.path.isabs(rel):
        raise PathEscape(f"path must be relative: {_quote_path(rel, force=True)}")

    real_root = os.path.realpath(root)
    norm = os.path.normpath(os.path.join(real_root, rel))
    inside = os.path.relpath(norm, real_root)
    parts = [] if inside == os.curdir else inside.split(os.sep)
    if any(p in ("", os.curdir, os.pardir) for p in parts):
        raise PathEscape(f"path escapes the worktree: {_quote_path(rel, force=True)}")

    if for_write:
        if not parts:
            raise PathEscape("refusing to write to the worktree root itself")
        candidate = os.path.join(
            os.path.realpath(os.path.dirname(norm)), os.path.basename(norm)
        )
        # The LEAF, not just the parent. A parent-only rule passes
        # `write_file("link", …)` where `link` is already `ln -s /etc/passwd`:
        # its parent is innocently the root, and the open then follows it out.
        # Resolving the leaf is only meaningful for a symlink — for anything
        # else `candidate` is already canonical.
        if os.path.islink(candidate) and _escapes(
            os.path.realpath(candidate), real_root
        ):
            raise PathEscape(
                f"refusing to write through a symlink out of the worktree: "
                f"{_quote_path(rel, force=True)}"
            )
    else:
        candidate = os.path.realpath(norm)

    if _escapes(candidate, real_root):
        raise PathEscape(f"path escapes the worktree: {_quote_path(rel, force=True)}")
    return real_root, parts, candidate


def safe_path(root: str, rel: str, *, for_write: bool) -> str:
    """Resolve `rel` inside `root`, or raise `PathEscape`.

    Returns the absolute host path the guard would allow. Callers doing I/O
    should use the tool functions rather than this string: the string is a
    check-then-use, and only the tool functions weld the check to the open.
    """
    return _validate(root, rel, for_write=for_write)[2]


def _descend(real_root: str, parts: Sequence[str], *, create: bool) -> int:
    """Open `real_root/parts…` as a directory, one `O_NOFOLLOW` step at a time.

    Returns a descriptor the caller must close. Every step goes through the
    previous descriptor, so no ancestor can be swapped out from under the
    walk: a component replaced by a symlink after this returns still cannot
    redirect anything, because the descriptor names the inode.
    """
    fd = os.open(real_root, _DIR_FLAGS)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, _NEW_DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    pass  # already there — or a symlink, which the open refuses
            nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            _close_quietly(fd)
            fd = nxt
    except OSError:
        _close_quietly(fd)
        raise
    return fd


def _read_capped(fd: int, limit: int) -> bytes:
    """Read at most `limit` bytes, without trusting `st_size`."""
    buf = bytearray()
    while len(buf) < limit:
        chunk = os.read(fd, min(_READ_CHUNK, limit - len(buf)))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def tool_list_files(
    root: str, rel: str = ".", *, ignore: Callable[[str], bool] | None = None
) -> str:
    """Recursive listing of `rel`, as paths relative to the worktree root.

    `ignore` is the tracked-aware predicate from `basetree.make_ignore` —
    directories are queried with a trailing `/`, matching `walk.py`. It is
    optional only so the guard can be tested on its own; the loop passes it,
    which is what makes the tool's "respects .gitignore" description true.
    """
    real_root, parts, _ = _validate(root, rel or os.curdir, for_write=False)
    is_ignored = ignore if ignore is not None else (lambda _p: False)
    prefix = "".join(part + "/" for part in parts)
    try:
        base_fd = _descend(real_root, parts, create=False)
    except OSError as exc:
        return f"error: cannot list {_quote_path(rel)}: {_why(exc)}"

    out: list[str] = []
    try:
        # `follow_symlinks=False` makes fwalk lstat each name and then confirm,
        # via samestat against the fstat of the descriptor it opened, that it
        # descended the same object it classified — the identity check walk.py
        # spells out by hand. A symlinked directory is therefore listed but
        # never descended.
        for dirpath, dirnames, filenames, _dfd in os.fwalk(
            dir_fd=base_fd, follow_symlinks=False
        ):
            here = (
                "" if dirpath == os.curdir else dirpath[len(os.curdir + os.sep) :] + "/"
            )
            depth = here.count("/")
            keep = []
            for d in sorted(dirnames):
                if d == _GIT_META or is_ignored(prefix + here + d + "/"):
                    continue
                if depth >= MAX_DEPTH:
                    continue
                keep.append(d)
            dirnames[:] = keep
            for name in sorted(filenames):
                full = prefix + here + name
                if name == _GIT_META or is_ignored(full):
                    continue
                # A filename is as attacker-controlled as a symlink target: a
                # newline in one would drop the rest of itself into column 0 of
                # the listing the model (and later the human) reads.
                out.append(_quote_path(full))
                if len(out) >= MAX_LIST:
                    out.append(f"[... listing capped at {MAX_LIST} entries ...]")
                    return "\n".join(out)
    finally:
        _close_quietly(base_fd)
    return "\n".join(out) or "(empty)"


def tool_read_file(root: str, rel: str) -> str:
    real_root, parts, _ = _validate(root, rel, for_write=False)
    quoted = _quote_path(rel)
    if not parts:
        return f"error: not a regular file: {quoted}"
    try:
        dir_fd = _descend(real_root, parts[:-1], create=False)
    except OSError as exc:
        return f"error: cannot read {quoted}: {_why(exc)}"
    try:
        fd = os.open(parts[-1], _READ_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        return f"error: cannot read {quoted}: {_why(exc)}"
    finally:
        _close_quietly(dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return f"error: not a regular file: {quoted}"
        data = _read_capped(fd, MAX_READ + 1)
    except OSError as exc:
        return f"error: cannot read {quoted}: {_errno_name(exc)}"
    finally:
        _close_quietly(fd)
    if b"\0" in data[:8192]:
        return f"error: binary file, not shown: {quoted}"
    text = data[:MAX_READ].decode("utf-8", "replace")
    return text + ("\n[... truncated ...]" if len(data) > MAX_READ else "")


def tool_write_file(root: str, rel: str, content: str) -> str:
    real_root, parts, _ = _validate(root, rel, for_write=True)
    quoted = _quote_path(rel)
    try:
        # Encode BEFORE the open: O_TRUNC has already destroyed the previous
        # contents by the time a late encoding failure could be noticed.
        data = content.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return f"error: content is not encodable as UTF-8: {quoted}"
    try:
        dir_fd = _descend(real_root, parts[:-1], create=True)
    except OSError as exc:
        return f"error: cannot create parent of {quoted}: {_why(exc)}"
    try:
        fd = os.open(parts[-1], _WRITE_FLAGS, _NEW_FILE_MODE, dir_fd=dir_fd)
    except OSError as exc:
        return f"error: cannot write {quoted}: {_why(exc)}"
    finally:
        _close_quietly(dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return f"error: not a regular file: {quoted}"
        _write_all(fd, data)
    except OSError as exc:
        return f"error: cannot write {quoted}: {_errno_name(exc)}"
    finally:
        _close_quietly(fd)
    return f"wrote {len(content)} chars to {quoted}"


async def tool_run_command(
    container: str, cmd: str, *, timeout_s: float
) -> tuple[int, str]:
    """The only tool that crosses into the container.

    `cmd` is forwarded verbatim — `exec_in_container` passes it as its own argv
    element, so quoting or escaping it here would double-escape it. The
    truncation flag is dropped: the marker is already in the text the model
    sees.
    """
    rc, out, _truncated = await exec_in_container(container, cmd, timeout_s=timeout_s)
    return rc, out
