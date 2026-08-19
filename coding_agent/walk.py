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

THE OTHER HALF OF §6.5 — added in pass 6 (2026-08-19). Confidentiality is not
the only property this walk owes the gate: it must not UNDER-SHOW. `chmod 000
backdoor.py` inside the container — one command, no race, no live process —
made the file vanish from the snapshot with no exception and no signal, and a
mode bit is persistent STATE, so destroying the container does not undo it: the
file was still missing from the FINAL diff a human reviews. That was a
REGRESSION; the pass-5 walk's unguarded `open()` at least crashed loudly.

Hence `TreeSnapshot.unreadable`. Where a path EXISTS but cannot be turned into
an entry, the walk records (path, reason) instead of quietly showing fewer
files. It never raises — a racing writer must not take the gate offline
(§5.2) — but nothing is dropped in silence any more. Three failure classes had
been collapsed into one indistinguishable `return None`:

    lost a race to a concurrent writer  transient  retry may win  -> retry, then record
    EACCES/EPERM (a mode bit)           PERMANENT  retry cannot   -> record at once
    EMFILE/ENFILE (descriptor limit)    PERMANENT  retry cannot   -> record at once

Retrying a permanent errno three times was itself the tell that the code did
not know the difference.

Reading is capped for the same reason a FIFO is not opened: an 8 GiB SPARSE
file costs the container ZERO bytes of disk and drove this walk to 3 GB of host
RSS in under a second. §5.1's diff size cap is DOWNSTREAM and cannot help — the
memory is spent before a diff is rendered. An over-cap file is recorded, never
truncated into something that reads like real content. The cap is per-file AND
per-snapshot, because a per-file cap alone leaves N files just under it costing
N x cap (measured: 400 sparse 8 MiB files reached 2627 MB before being killed
with only the per-file cap, and 132 MB with both).

OPEN, NOT CLOSED — the BREADTH axis of the same DoS. The byte budgets bound
CONTENT. Nothing bounds the NUMBER of paths: `os.listdir` materialises a whole
directory's names, and `entries`/`unreadable` hold one record per child.
Measured on this host: 249 bytes of host RSS per empty file, so 1,000,000
files cost ~237 MiB and take ~48s to create, and 10,000,000 cost ~2.4 GiB. It
is far slower and costlier for the attacker than the sparse-file vector (which
is instant and free) and its extreme end now degrades to an `ENOMEM` record
instead of a crash, but it is NOT bounded. Closing it means replacing
`os.listdir` with a bounded `os.scandir` loop plus a recorded overflow marker,
which is a redesign of the enumeration path rather than a local fix.
"""

from __future__ import annotations

import errno
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

# Per-file and whole-snapshot read budgets. Both are needed: a per-file cap
# alone leaves N files of (cap - 1) bytes costing N x cap, and N sparse files
# are as cheap for the sandbox to create as one.
#
# 8 MiB per file sits far above anything a human reviews line by line and above
# the large-but-legitimate files a repo actually carries (lockfiles, generated
# sources), while bounding one file's transient cost to ~17 MiB. 256 MiB for
# the whole snapshot bounds `entries`, which is held in memory in full.
# Exceeding either is RECORDED, so the failure is loud, not a shorter diff.
_MAX_FILE_BYTES = 8 << 20
_MAX_TOTAL_BYTES = 256 << 20

# A racing writer need not be hostile — the model's own background process, or
# an editor's atomic save, can invalidate a classification honestly. Re-classify
# a bounded number of times so a real change is recorded rather than silently
# dropped ("under-showing a real change is the danger", §6.5). Bounded, so no
# writer can spin the walk.
_RECLASSIFY_ATTEMPTS = 3

# Errnos no retry can ever win, so the walk records them at once instead of
# spending _RECLASSIFY_ATTEMPTS pretending they are a race. Deliberately
# minimal: ELOOP, ENOTDIR, ENOENT and ESTALE are all things a racing writer
# produces and a later attempt can still resolve, so they stay retryable.
_PERMANENT_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EMFILE, errno.ENFILE})

# Non-errno reasons. Kept as short stable tokens because Task 4 renders them.
_RACED = "RACED"  # re-classified _RECLASSIFY_ATTEMPTS times and never settled
_OVERSIZE = "OVERSIZE"  # bigger than _MAX_FILE_BYTES; not read, not truncated
_BUDGET = "BUDGET"  # _MAX_TOTAL_BYTES was already spent on earlier files
_XDEV = "XDEV"  # not on the worktree's device (§6.5 rule 2)
_ENOMEM = "ENOMEM"  # the host ran out of memory listing this directory

# Reasons that describe a STABLE property of the tree — a mode bit, a size, a
# descriptor budget — as opposed to a moment inside somebody else's race. Only
# these enter `tree_hash`; see its docstring for the measurement that settled
# it. Exactly the reasons `_refuse` marks terminal, plus the caps.
_STABLE_REASONS = frozenset(
    {errno.errorcode[code] for code in _PERMANENT_ERRNOS} | {_OVERSIZE, _BUDGET, _XDEV}
)

# The repo-metadata entry at the top of the worktree, opaque by §6.5.
_META = ".git"
# How the root itself is named when the root is what could not be read.
_ROOT = "."


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "file" | "symlink"
    data: bytes  # file bytes, or the symlink TARGET string as utf-8


@dataclass(frozen=True, order=True)
class Unreadable:
    """A path that EXISTS in the worktree but is absent from `entries`.

    `path` carries a TRAILING SLASH when the object is a directory, so the gate
    can tell "one file is hidden" from "a subtree of unknown size is hidden".
    The root itself is named `.`.

    `reason` is an errno name (`EACCES`, `EMFILE`, ...) or one of the tokens
    `RACED`, `OVERSIZE`, `BUDGET`, `XDEV`. Some reasons are STABLE properties of
    the tree and some are a moment inside somebody else's race; `tree_hash`
    distinguishes them, the gate need not.
    """

    path: str
    reason: str


@dataclass(frozen=True)
class TreeSnapshot:
    entries: dict[str, Entry]
    # Sorted, so two walks of the same tree produce byte-identical output.
    # Defaulted, so the three-field callers written before pass 6 still build.
    unreadable: tuple[Unreadable, ...] = ()


@dataclass
class _Frame:
    """One directory the walk is enumerating. Owns `fd` until it is popped."""

    fd: int
    prefix: str  # "" at the root, otherwise "sub/dir/"
    names: list[str]  # remaining children, reverse-sorted so pop() ascends


@dataclass(frozen=True)
class _Refusal:
    """Why a child did not become an entry.

    `terminal` means the answer cannot change on a retry — a permission bit, a
    descriptor limit, a size cap — so `_visit` records it immediately. A
    non-terminal refusal means the object moved underneath the walk and the
    caller should re-classify.
    """

    reason: str
    terminal: bool


@dataclass(frozen=True)
class _Outcome:
    """What `_visit` produced for one child: at most one of the three."""

    entry: Entry | None = None
    frame: _Frame | None = None
    unreadable: Unreadable | None = None


@dataclass
class _Budget:
    """The snapshot's remaining read allowance, in bytes."""

    remaining: int

    def would_exceed(self, want: int) -> bool:
        return want > self.remaining

    def spend(self, got: int) -> None:
        self.remaining = max(0, self.remaining - got)


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


def _refuse(exc: OSError) -> _Refusal:
    return _Refusal(_errno_name(exc), exc.errno in _PERMANENT_ERRNOS)


def _same_object(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_ino == b.st_ino and a.st_dev == b.st_dev


def _pending_names(dir_fd: int) -> list[str]:
    return sorted(os.listdir(dir_fd), reverse=True)


def _read_capped(fd: int, limit: int) -> bytes | None:
    """Read at most `limit` bytes; None when the file holds more than that.

    Bounded independently of any `st_size` pre-check, because `st_size` can be
    beaten by a writer appending after the `fstat` and a synthetic filesystem
    can simply lie about it. An overrunning file is NEVER returned truncated: a
    short read recorded as if it were the whole file hands the reviewer a
    diff that looks real and is not, which is worse than an honest "not read".
    """
    buf = bytearray()
    while len(buf) <= limit:
        chunk = os.read(fd, _READ_CHUNK)
        if not chunk:
            return bytes(buf)
        buf += chunk
    return None


def _open_child_dir(
    dir_fd: int, name: str, prefix: str, classified: os.stat_result, root_dev: int
) -> _Frame | _Refusal:
    """Open `name` as a directory THROUGH its parent's descriptor.

    Returns a `_Refusal` when the object is no longer the directory that was
    classified — ELOOP because O_NOFOLLOW refused a swapped-in symlink,
    ENOTDIR, or an st_ino/st_dev mismatch — or when it cannot be listed at all.
    """
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        return _refuse(exc)
    keep = False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or not _same_object(opened, classified):
            return _Refusal(_RACED, False)
        if opened.st_dev != root_dev:
            return _Refusal(_XDEV, True)
        frame = _Frame(fd, prefix, _pending_names(fd))
        keep = True
        return frame
    except MemoryError:
        # `os.listdir` builds the WHOLE name list before returning it, so a
        # directory with enough children exhausts host memory here. Nothing
        # else in this module catches a non-OSError, and letting it out takes
        # the review gate offline — the failure mode §5.2 forbids — for a
        # condition the sandbox can create. Record it like any other refusal.
        return _Refusal(_ENOMEM, True)
    except OSError as exc:
        return _refuse(exc)
    finally:
        if not keep:
            _close_quietly(fd)


def _read_regular(
    dir_fd: int,
    name: str,
    rel: str,
    classified: os.stat_result,
    root_dev: int,
    budget: _Budget,
) -> Entry | _Refusal:
    """Read `name` THROUGH its parent's descriptor, or say why not.

    The bytes are read only after `fstat` on the OPEN descriptor confirms the
    object is still a regular file, still the same inode on the same device as
    the `lstat` that classified it, and still on the worktree's own device, so
    the descriptor cannot address anything the walk did not choose to read.
    """
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        return _refuse(exc)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_object(opened, classified):
            return _Refusal(_RACED, False)
        if opened.st_dev != root_dev:
            return _Refusal(_XDEV, True)
        if opened.st_size > _MAX_FILE_BYTES:
            return _Refusal(_OVERSIZE, True)
        if budget.would_exceed(opened.st_size):
            return _Refusal(_BUDGET, True)
        # The read is bounded by the REMAINING budget as well as by the
        # per-file cap, so a file whose st_size reads smaller than the file
        # actually is cannot spend more than the snapshot has left.
        limit = min(_MAX_FILE_BYTES, budget.remaining)
        data = _read_capped(fd, limit)
        if data is None:
            return _Refusal(_OVERSIZE if limit == _MAX_FILE_BYTES else _BUDGET, True)
        budget.spend(len(data))
        return Entry(rel, "file", data)
    except OSError as exc:
        return _refuse(exc)
    finally:
        _close_quietly(fd)


def _visit(
    dir_fd: int,
    name: str,
    rel: str,
    is_ignored: Callable[[str], bool],
    root_dev: int,
    budget: _Budget,
    meta_id: tuple[int, int] | None,
) -> _Outcome:
    """Classify ONE child of the open directory `dir_fd`, and act on it.

    Every syscall here is `dir_fd`-relative, so no ancestor directory can be
    swapped underneath the walk. Returns at most one of (entry, child frame,
    unreadable record); the caller owns the returned frame's descriptor.
    """
    # Re-classifying must not re-query the injected callable: the query string
    # is the same, so the answer is the same, and Task 3's matcher should see
    # the same call pattern it would see on a quiescent tree.
    decided: dict[str, bool] = {}

    def ignored(query: str) -> bool:
        if query not in decided:
            decided[query] = is_ignored(query)
        return decided[query]

    reason = _RACED
    # Appended to `rel` when the walk records a refusal, so a directory the
    # walk gave up on is reported as a hidden SUBTREE. Tracked across attempts
    # because the fall-through below is reached with no classification in hand.
    tail = ""
    for _attempt in range(_RECLASSIFY_ATTEMPTS):
        try:
            classified = os.lstat(name, dir_fd=dir_fd)  # lstat: never follows
        except OSError as exc:
            refusal = _refuse(exc)
            if refusal.terminal:
                return _Outcome(unreadable=Unreadable(rel + tail, refusal.reason))
            reason = refusal.reason  # ENOENT, ESTALE, ... : it may come back
            continue
        if (
            meta_id is not None
            # Identity ALONE would make any hardlink to the metadata entry
            # opaque too, and a worktree's metadata entry is an ordinary file
            # the sandbox can `ln`. That would be a silent omission at a name
            # of the sandbox's choosing — the failure this pass exists to stop.
            # Both halves are required: the name (in any case, for the
            # case-insensitive volume) AND the identity.
            and name.lower() == _META
            and (classified.st_ino, classified.st_dev) == meta_id
        ):
            return _Outcome()  # the repo's own metadata, under any casing
        mode = classified.st_mode
        tail = "/" if stat.S_ISDIR(mode) else ""
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
            except OSError as exc:
                refusal = _refuse(exc)
                if refusal.terminal:
                    return _Outcome(unreadable=Unreadable(rel, refusal.reason))
                reason = refusal.reason  # swapped or vanished; re-classify
                continue
            if ignored(rel):
                return _Outcome()
            data = target.encode("utf-8", "surrogateescape")
            return _Outcome(entry=Entry(rel, "symlink", data))
        if stat.S_ISDIR(mode):
            if ignored(rel + "/"):
                return _Outcome()
            opened_dir = _open_child_dir(dir_fd, name, rel + "/", classified, root_dev)
            if isinstance(opened_dir, _Frame):
                return _Outcome(frame=opened_dir)
            if opened_dir.terminal:
                # TRAILING SLASH: a whole subtree of unknown size is hidden.
                return _Outcome(unreadable=Unreadable(rel + "/", opened_dir.reason))
            reason = opened_dir.reason  # lost the race; re-classify
            continue
        if not stat.S_ISREG(mode):
            # FIFO / socket / device. Deliberately absent from the diff and NOT
            # reported as unreadable: there is no content a diff could show, so
            # naming it would claim a change is hidden when none is.
            return _Outcome()
        if ignored(rel):
            return _Outcome()
        read = _read_regular(dir_fd, name, rel, classified, root_dev, budget)
        if isinstance(read, Entry):
            return _Outcome(entry=read)
        if read.terminal:
            return _Outcome(unreadable=Unreadable(rel, read.reason))
        reason = read.reason
    # Out of attempts. Recorded rather than dropped: "indistinguishable from
    # never existed" is the under-showing failure §6.5 calls the dangerous
    # direction, and it is the gate — not this module — that decides what a
    # path it could not read consistently means for the diff. `tail` carries
    # the trailing slash when the last successful classification saw a
    # directory, so giving up on a subtree is not reported as one lost file.
    return _Outcome(unreadable=Unreadable(rel + tail, reason))


def _meta_identity(root_fd: int) -> tuple[int, int] | None:
    """(st_ino, st_dev) of the root's own repo-metadata entry, or None.

    The name comparison in `snapshot_tree` is case-SENSITIVE and the deployment
    volume is case-INSENSITIVE APFS, so a case-only rename walked straight past
    it. That is over-showing, which §6.5 calls harmless — but §5.1 truncates the
    diff past a byte limit, so a flood of pack bytes becomes under-showing BY
    DISPLACEMENT: the real change scrolls off the end.

    Identity is the right test and is correct on both kinds of volume. A
    case-only rename keeps the inode and stays opaque; on a case-SENSITIVE
    volume an upper-case name is a genuinely different object with a different
    inode and is shown, exactly as a real checkout would treat it. It is also
    race-free: the comparison uses the SAME `lstat` that classifies the child,
    so it opens no window a second stat would.

    Identity is necessary but NOT sufficient — see `_visit`, which requires the
    name to match too, so a hardlink to the metadata file cannot make a name of
    the sandbox's choosing disappear.
    """
    try:
        st = os.lstat(_META, dir_fd=root_fd)
    except OSError:
        return None
    return (st.st_ino, st.st_dev)


def snapshot_tree(root: str, is_ignored: Callable[[str], bool]) -> TreeSnapshot:
    """Walk `root` with lstat semantics.

    - symlinks are recorded as their target TEXT and never followed
      (git's 120000 mode); symlinked directories are not descended
    - only regular files are read; FIFOs/sockets/devices are skipped
    - the top-level repo-metadata entry is opaque (never read, never listed),
      by name AND by inode identity so a case-only rename cannot expose it
    - the walk stays on the root's device (§6.5 rule 2)
    - `is_ignored(relpath)` gates NEW paths only; the caller enforces
      tracked-awareness (basetree paths are always included) — see diff().
      DIRECTORIES are queried with a TRAILING SLASH, files and symlinks
      without; Task 3's callable depends on that distinction.
    - a path that exists but cannot be read is recorded in `unreadable`, never
      dropped in silence. IGNORED paths are not recorded: `lstat` succeeds on a
      mode-000 file, so the ignore decision is made from the same classification
      as for a readable one, and `unreadable` never becomes a side channel that
      shows the gate paths the ignore rules excluded.

    Never raises for a filesystem condition, including the two the sandbox can
    steer into an out-of-memory: a directory too large to list and a file too
    large to read are both recorded, not raised. A racing writer must not be
    able to take the review gate offline (§5.2). Only the injected callable,
    and host resource exhaustion outside those two paths, can propagate.

    Descriptors: each frame on the stack owns exactly one open directory, and a
    child's descriptor is opened only when the walk descends into it, so the
    number of open descriptors is the tree's DEPTH, never its breadth. Every
    frame is closed when it is exhausted, and the `finally` closes whatever is
    still on the stack if anything raises. Running out of them anyway (the
    deployed process inherits a soft RLIMIT_NOFILE of 256 from launchd, not the
    1,048,576 an interactive shell reports) surfaces as EMFILE records, not as
    a subtree that quietly is not there.
    """
    root = os.path.abspath(root)
    entries: dict[str, Entry] = {}
    unreadable: list[Unreadable] = []

    try:
        root_fd = os.open(root, _DIR_FLAGS)
    except OSError as exc:
        return TreeSnapshot(entries, (Unreadable(_ROOT, _errno_name(exc)),))

    opened_root = False
    try:
        root_st = os.fstat(root_fd)
        names = _pending_names(root_fd)
        meta_id = _meta_identity(root_fd)
        opened_root = True
    except MemoryError:
        # `os.listdir` materialises the whole name list; a root with enough
        # children exhausts host memory. Degrade to a record rather than
        # letting the gate crash. See `_open_child_dir` for the same case.
        return TreeSnapshot(entries, (Unreadable(_ROOT, _ENOMEM),))
    except OSError as exc:
        return TreeSnapshot(entries, (Unreadable(_ROOT, _errno_name(exc)),))
    finally:
        # Runs on every exit above, so nothing escapes holding the root
        # descriptor — including an exception this function does not name.
        if not opened_root:
            _close_quietly(root_fd)

    budget = _Budget(_MAX_TOTAL_BYTES)
    stack = [_Frame(root_fd, "", names)]
    try:
        while stack:
            frame = stack[-1]
            if not frame.names:
                stack.pop()
                _close_quietly(frame.fd)
                continue
            name = frame.names.pop()
            rel = frame.prefix + name
            if rel == _META or rel.startswith(_META + "/"):
                continue
            outcome = _visit(
                frame.fd,
                name,
                rel,
                is_ignored,
                root_st.st_dev,
                budget,
                meta_id if frame.prefix == "" else None,
            )
            if outcome.frame is not None:
                stack.append(outcome.frame)
            if outcome.entry is not None:
                entries[rel] = outcome.entry
            if outcome.unreadable is not None:
                unreadable.append(outcome.unreadable)
    finally:
        for pending in stack:
            _close_quietly(pending.fd)
    return TreeSnapshot(entries, tuple(sorted(unreadable)))


def tree_hash(snap: TreeSnapshot) -> str:
    """Content hash of a snapshot, for §7's no-progress detector.

    STABLE `unreadable` reasons are hashed alongside `entries`, and transient
    ones are not. Both halves of that were measured, not assumed.

    Why hash them at all: a file the model creates ALREADY unreadable never
    appears in `entries`, so a hash over `entries` alone is byte-identical
    before and after. The chmod-000 hide would be invisible to the detector as
    well as to the diff, and "make a change the progress detector cannot see"
    is the same under-showing failure one layer up.

    Why NOT the transient ones: a path that vanished between `listdir` and
    `lstat` is a fact about somebody else's timing, not about the tree. An
    honest racing writer — a test run creating and deleting temp files, which
    is the ordinary case, not an attack — makes those records appear on most
    walks. Measured over 1512 walks against a separate churning process:

        flaps with entries only ......  0.6%   (9 of 1511)
        flaps with ALL reasons hashed . 87.1%  (1316 of 1511)
        flaps with stable reasons only  0.6%

    Hashing every reason therefore reports progress on nearly every turn and
    takes §7's stuck-detector off the air, which is the failure PR #25 (82
    commits / 64h) is this fleet's proof of. Nothing is lost from the gate's
    view: `unreadable` still LISTS the transient paths, and the diff a human
    reviews is built after the container is destroyed, when nothing is racing —
    so a path that is genuinely unreadable then is permanent and IS hashed.
    """
    h = hashlib.sha256()
    for path in sorted(snap.entries):
        e = snap.entries[path]
        h.update(e.kind.encode())
        h.update(b"\0")
        h.update(path.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
        h.update(hashlib.sha256(e.data).digest())
    for u in snap.unreadable:  # already sorted, so this is deterministic
        if u.reason not in _STABLE_REASONS:
            continue
        h.update(b"unreadable")  # never a valid Entry.kind, so no collision
        h.update(b"\0")
        h.update(u.path.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
        h.update(hashlib.sha256(u.reason.encode()).digest())
    return h.hexdigest()
