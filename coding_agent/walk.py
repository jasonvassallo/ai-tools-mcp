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

import difflib
import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time cycle otherwise: basetree imports this module
    from .basetree import BaseTree

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

# The three modes a git tree can hold for something this walk records. git
# keeps exactly ONE permission bit — the owner execute bit — and normalises
# everything else away (`ce_mode_from_stat`: `mode & 0100 ? 100755 : 100644`),
# so these are the only values either side of the diff can produce.
_MODE_FILE = "100644"
_MODE_EXEC = "100755"
_MODE_SYMLINK = "120000"

# The repo-metadata entry at the top of the worktree, opaque by §6.5.
_META = ".git"
# How the root itself is named when the root is what could not be read.
_ROOT = "."


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "file" | "symlink"
    data: bytes  # file bytes, or the symlink TARGET string as utf-8
    # The owner execute bit, and NOTHING else about st_mode — the one
    # permission bit git tracks. Carried because `chmod +x` changes no bytes:
    # a diff keyed on content alone shows nothing at all while code the human
    # already approved silently becomes something the shell will run. Recorded
    # on BOTH sides in the same change (here from the walk's existing `fstat`,
    # in basetree.py from `ls-tree`'s mode column) — a one-sided version would
    # report every committed 100755 file as a mode change on turn one, and a
    # diff that cries wolf on every entry is one a reviewer stops reading.
    # Defaulted so the three-field constructions written before this build.
    executable: bool = False

    @property
    def mode(self) -> str:
        """The git tree mode for this entry: 100644, 100755 or 120000.

        Derived rather than stored, so `Entry(p, "symlink", target)` cannot be
        built claiming a regular file's mode.
        """
        if self.kind == "symlink":
            return _MODE_SYMLINK
        return _MODE_EXEC if self.executable else _MODE_FILE


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
        # The mode comes from the fstat that already verified the identity of
        # this descriptor — no extra syscall, and no second path resolution a
        # racing writer could answer differently.
        return Entry(rel, "file", data, bool(opened.st_mode & stat.S_IXUSR))
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
        # The mode joins the hash for the same reason it joins the diff: a
        # turn whose only change is `chmod +x` would otherwise read as no
        # progress. It cannot add flap — a mode bit is a stable property of
        # the tree, not a moment inside somebody else's race, and an entry
        # that appears or vanishes under a racing writer already moves the
        # hash through `entries`. Measured: see the report for this task.
        h.update(e.mode.encode())
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


# ---------------------------------------------------------------------------
# THE REVIEW DIFF (spec §5.1)
#
# Nothing the sandboxed model produces is ever applied automatically, so the
# text rendered below is the ONLY artifact a human sees before deciding. That
# makes it a security control rather than a convenience: a real change that
# does not appear here defeats the entire design. Over-showing is noise;
# UNDER-showing is the dangerous direction (§6.5), and every rule in this
# section exists because some way of hiding a change was demonstrated.
#
# The comparison is TWO-SIDED over `base-tree ∪ worktree-walk`, so a tracked
# file the model deletes — or replaces with a special file the walk skips —
# still surfaces, as a deletion rather than as silence.
# ---------------------------------------------------------------------------

_TRUNC = "\n[... diff truncated by coding_agent size cap; full diff at {path} ...]\n"

# The marker for a final line with no newline, spelled as git spells it.
# Without it that line runs straight into the NEXT file's header, taking the
# header out of column 0 and hiding a whole file section from anyone scanning
# for one.
_NO_NEWLINE = "\\ No newline at end of file\n"

# The `unreadable` block. Rendered FIRST, and bounded, both deliberately:
# truncation removes the TAIL, so a warning printed after the file sections is
# exactly what an oversized diff drops — and the sandbox chooses the diff's
# size. Bounded because an unlimited list of sandbox-chosen names would push
# the real change out of the visible window instead (under-showing by
# displacement). The COUNT is always exact and `DiffResult.unreadable` always
# carries the complete list, so bounding the render costs nothing.
_UNREADABLE_HEAD = (
    "coding_agent: WARNING: {n} path(s) could not be read and are NOT in this diff:\n"
)
_UNREADABLE_MORE = (
    "coding_agent:   ... and {n} more (complete list on DiffResult.unreadable)\n"
)
_UNREADABLE_SHOWN = 20

# The footer a TRUNCATED diff carries. Truncation is the sandbox's to trigger
# — one `dd` of a file that sorts early buys 5000 bytes of `xxxx` ahead of
# every other section — so a cap that drops the tail also drops the EXISTENCE
# of every change after it. (Found by the brief's own size-cap test: `big.txt`
# sorts before `blob.bin`, and a 1000-byte cap removed the binary annotation
# entirely.) The cap therefore truncates DETAIL only: every changed path is
# named after the marker, with its mode transition, whatever the cap did to
# the hunks. Bounded for the same reason the unreadable block is, and appended
# AFTER the slice so the cap cannot eat it either.
_CHANGED_HEAD = (
    "coding_agent: {n} file(s) changed in total; the list below is complete "
    "to {shown},\ncoding_agent: so a truncated diff cannot hide THAT a path "
    "changed:\n"
)
_CHANGED_MORE = (
    "coding_agent:   ... and {n} more (complete list on DiffResult.changed_files)\n"
)
_CHANGED_SHOWN = 100

# C-style escapes for a quoted path, the same set a quoted path uses upstream.
_ESCAPES = {
    ord("\\"): "\\\\",
    ord('"'): '\\"',
    0x07: "\\a",
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0B: "\\v",
    0x0C: "\\f",
    0x0D: "\\r",
}


@dataclass(frozen=True)
class DiffResult:
    text: str  # unified diff, possibly truncated
    truncated: bool
    changed_files: list[str]  # sorted; RAW paths, never quoted
    full_path: str | None  # where the untruncated diff was written
    # The snapshot's unreadable records, verbatim. `text` states them too, but
    # a gate that must fail closed should not have to parse prose to do it.
    unreadable: tuple[Unreadable, ...] = ()


def _quote_path(path: str, *, force: bool = False) -> str:
    """C-quote a path when it holds anything that could break a line.

    A path is as attacker-controlled as a symlink target — `os.listdir`
    returns whatever bytes are on disk and `ls-tree -z` carries them through —
    so a filename containing a newline drops the rest of itself into column 0
    of the diff, where a forged `--- a/… / +++ / @@` block reads as an entire
    extra file. That is the same structure injection §6.5 pins for symlink
    targets, arriving through the other string. Quoting collapses any such
    path onto ONE line.

    Non-ASCII is left as itself (upstream would octal-escape it): unreadable
    mojibake in a path a human must recognise is its own kind of hiding place,
    and only control characters can break the line structure.
    """
    if not force and not any(ch in '"\\' or ch < " " or ch == "\x7f" for ch in path):
        return path
    out = ['"']
    for ch in path:
        esc = _ESCAPES.get(ord(ch))
        if esc is not None:
            out.append(esc)
        elif ch < " " or ch == "\x7f":
            out.append(f"\\{ord(ch):03o}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def _lines(data: bytes) -> list[str]:
    return data.decode("utf-8", "surrogateescape").splitlines(keepends=True)


def _render_symlink(data: bytes) -> list[str]:
    """The 120000 rendering: the target text as CONTENT lines, so a forged
    '--- a/' inside the target becomes '+--- a/' — never column 0."""
    lines = _lines(data)
    return lines if lines else [""]


def _entry_lines(e: Entry | None) -> list[str]:
    if e is None:
        return []
    if e.kind == "symlink":
        return _render_symlink(e.data)
    return _lines(e.data)


def _emit(diff_lines: Iterable[str], out: list[str]) -> None:
    """Append difflib's output, marking any line that carries no newline.

    difflib prefixes every content line and computes its own `@@` counts, so
    the structure it emits is trustworthy — except that it reproduces a final
    line without a newline verbatim, which would splice the next header onto
    the end of it.
    """
    for line in diff_lines:
        out.append(line if line.endswith("\n") else line + "\n" + _NO_NEWLINE)


def _mode_lines(b: Entry | None, s: Entry | None) -> list[str]:
    """The mode header for one path. Also the ONLY output for a file whose
    content is unchanged and whose mode is not."""
    if b is None:
        return [] if s is None else [f"new file mode {s.mode}\n"]
    if s is None:
        return [f"deleted file mode {b.mode}\n"]
    if b.mode != s.mode:
        return [f"old mode {b.mode}\n", f"new mode {s.mode}\n"]
    return []


def _binary_side(e: Entry | None) -> bool:
    # A symlink is never binary: its data is the target TEXT, and rendering it
    # as "Binary files differ" would hide where it points.
    return e is not None and e.kind == "file" and _is_binary(e.data)


def _sizes(b: Entry | None, s: Entry | None) -> str:
    was = len(b.data) if b is not None else 0
    now = len(s.data) if s is not None else 0
    return f"{was} -> {now} bytes"


def _transition(b: Entry | None, s: Entry | None) -> str:
    """`100644 -> 100755`, with `-` for a side where the path is absent."""
    was = b.mode if b is not None else "-"
    now = s.mode if s is not None else "-"
    return f"{was} -> {now}"


def _changed_block(summary: list[str]) -> str:
    if not summary:
        return ""
    shown = summary[:_CHANGED_SHOWN]
    out = [_CHANGED_HEAD.format(n=len(summary), shown=len(shown))]
    out.extend(shown)
    if len(shown) != len(summary):
        out.append(_CHANGED_MORE.format(n=len(summary) - len(shown)))
    return "".join(out)


def _unreadable_block(unreadable: tuple[Unreadable, ...], limit: int | None) -> str:
    """The "these paths exist and I could not read them" preamble.

    Every line is prefixed, and every path force-quoted, so nothing in this
    block can be mistaken for — or forged into — diff structure.
    """
    if not unreadable:
        return ""  # silence when clean, or the signal becomes noise
    shown = unreadable if limit is None else unreadable[:limit]
    out = [_UNREADABLE_HEAD.format(n=len(unreadable))]
    out.extend(
        f"coding_agent:   {_quote_path(u.path, force=True)} ({u.reason})\n"
        for u in shown
    )
    if len(shown) != len(unreadable):
        out.append(_UNREADABLE_MORE.format(n=len(unreadable) - len(shown)))
    out.append("\n")
    return "".join(out)


def unified_diff(
    base: BaseTree, snap: TreeSnapshot, *, max_bytes: int, spill_dir: str
) -> DiffResult:
    """Render `base` vs `snap` as a unified diff a human can act on.

    What each rule is defending against:

    - a path is compared on MODE and BYTES, never bytes alone, so `chmod +x`
      on untouched content is a change (and so is a file <-> symlink
      typechange whose two sides happen to hold identical bytes);
    - a symlink is rendered as a 120000 blob — its target as content lines,
      every one `+`-prefixed with an accurate `@@` count — because the target
      string is sandbox-controlled and round-trips forged headers;
    - a path that needs it is C-quoted, closing the same injection through the
      filename;
    - binary content is ANNOTATED with its sizes, never embedded, so one
      `dd`-written blob cannot flood the reviewer's context;
    - the text is capped, with the untruncated diff spilled to a file, because
      the length of the diff is the sandbox's to choose — and a truncated diff
      still names every changed path, so the cap can cost the reviewer the
      DETAIL of a change but never the knowledge that one happened;
    - paths the walk could not read are stated FIRST, so the cap cannot cut
      the warning off.

    `max_bytes` bounds the RETURNED text only; `spill_dir` must be a host
    directory the sandbox cannot reach.
    """
    paths = sorted(set(base.entries) | set(snap.entries))
    body: list[str] = []
    changed: list[str] = []
    summary: list[str] = []
    for p in paths:
        b, s = base.entries.get(p), snap.entries.get(p)
        # Mode equality implies kind equality (a symlink is 120000 and nothing
        # else is), so these two comparisons are the whole test for "same".
        if b is not None and s is not None and b.mode == s.mode and b.data == s.data:
            continue
        changed.append(p)
        summary.append(
            f"coding_agent:   {_quote_path(p, force=True)} ({_transition(b, s)})\n"
        )
        left, right = _quote_path("a/" + p), _quote_path("b/" + p)
        body.append(f"diff --git {left} {right}\n")
        body.extend(_mode_lines(b, s))
        if _binary_side(b) or _binary_side(s):
            body.append(f"Binary files differ: {_quote_path(p)} ({_sizes(b, s)})\n")
            continue
        _emit(
            difflib.unified_diff(
                _entry_lines(b),
                _entry_lines(s),
                fromfile=left if b is not None else "/dev/null",
                tofile=right if s is not None else "/dev/null",
                lineterm="\n",
            ),
            body,
        )
    joined = "".join(body)
    text = _unreadable_block(snap.unreadable, _UNREADABLE_SHOWN) + joined
    full_path: str | None = None
    truncated = False
    if len(text.encode("utf-8", "surrogateescape")) > max_bytes:
        fd, full_path = tempfile.mkstemp(
            prefix="coding-agent-diff-", suffix=".patch", dir=spill_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as fh:
            # The SPILL carries every unreadable path, not the bounded render:
            # nothing is lost, it is only the in-context text that is capped.
            fh.write(_unreadable_block(snap.unreadable, None) + joined)
        text = (
            text.encode("utf-8", "surrogateescape")[:max_bytes].decode(
                "utf-8", "ignore"
            )
            + _TRUNC.format(path=full_path)
            + _changed_block(summary)
        )
        truncated = True
    return DiffResult(
        text=text,
        truncated=truncated,
        changed_files=changed,
        full_path=full_path,
        unreadable=snap.unreadable,
    )
