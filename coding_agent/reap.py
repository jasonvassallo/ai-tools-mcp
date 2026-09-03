"""Best-effort reaper for host-side coding_agent temp litter (issue #79).

Two kinds of entries accumulate in the host's `tempfile.gettempdir()` with no
owner for their eventual death:

- `coding-agent-diff-<hex>.patch` files — the full diff `walk.unified_diff`
  spills there whenever a run's diff is over `loop._DIFF_MAX_BYTES`. Outliving
  the sandbox teardown is INTENTIONAL (§6.5: the human reads the complete
  diff after the worktree is gone; see
  `test_the_loop_ships_its_own_diff_cap_and_spills_outside_the_worktree`), so
  this module never removes one on any schedule tighter than the TTL below.
- `coding-agent-wt-<hex>/` directories — the private `mkdtemp` parent
  `sandbox.create_worktree` makes for each linked worktree. Normal teardown
  (`sandbox.teardown_worktree`) removes it; a crash before teardown runs does
  not, and nothing else ever revisits it.

`sweep()` is the entry point. It is meant to be called cheaply and often — at
MCP server start and at the top of every `coding_agent` run (see
`loop.run_coding_agent` and `mcp_server.main`) — and is written so that doing
so is safe: one `os.scandir` of the target directory, then per-entry
`os.stat`/`os.remove`/`shutil.rmtree` calls that are each wrapped so a single
unremovable or unreadable entry is skipped rather than aborting the sweep.
`sweep()` itself never raises; a caller on the event loop should still run it
via `asyncio.to_thread`, since a large temp dir makes the scandir a real
(if bounded) blocking call.

This module touches ONLY entries matching the two exact prefixes/suffixes
below. Nothing else in the temp dir is inspected, let alone removed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

# Naming this module is allowed to touch. Must match `walk.py`'s
# `tempfile.mkstemp(prefix=..., suffix=...)` call and `sandbox._WT_PREFIX`
# exactly.
_DIFF_PREFIX = "coding-agent-diff-"
_DIFF_SUFFIX = ".patch"
_WT_PREFIX = "coding-agent-wt-"

# How long a spilled diff or an orphaned worktree parent is left alone before
# this module will remove it. 24h: long enough that a human polling
# `coding_agent_result` on a background run has ample time to read the
# spilled path (mentioned in the diff's own truncation marker and in the
# result's `diff_full_path`), short enough that a crash-looping host does not
# accumulate litter for more than a day between sweeps.
_TTL_SECONDS = 24 * 60 * 60

# Independent of the TTL: a burst of over-cap diffs inside one TTL window
# must not be able to fill the host temp dir. Oldest-first once either bound
# is exceeded. No equivalent byte cap applies to worktree parents — sizing
# those would mean walking each one recursively, which is the same
# fan-out hazard #77 was about; worktree cleanup here is TTL + liveness only.
_MAX_SPILL_COUNT = 200
_MAX_SPILL_BYTES = 256 * 1024 * 1024  # 256 MiB


def _remove_file(path: str) -> bool:
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def _scan(spill_dir: str) -> tuple[list[os.DirEntry], list[os.DirEntry]]:
    """One pass over `spill_dir`. Returns (diff files, worktree dirs), each
    matched purely by name/type — nothing here reads file content."""
    diffs: list[os.DirEntry] = []
    worktrees: list[os.DirEntry] = []
    try:
        with os.scandir(spill_dir) as it:
            for entry in it:
                name = entry.name
                try:
                    if (
                        name.startswith(_DIFF_PREFIX)
                        and name.endswith(_DIFF_SUFFIX)
                        and entry.is_file(follow_symlinks=False)
                    ):
                        diffs.append(entry)
                    elif name.startswith(_WT_PREFIX) and entry.is_dir(
                        follow_symlinks=False
                    ):
                        worktrees.append(entry)
                except OSError:
                    continue  # vanished mid-scan; nothing to reap
    except OSError:
        pass  # no such dir, or unreadable — nothing this sweep can do
    return diffs, worktrees


def _reap_diffs(
    diffs: list[os.DirEntry],
    now: float,
    ttl_seconds: float,
    max_count: int,
    max_bytes: int,
) -> tuple[int, int]:
    """Delete diff spills older than the TTL, then enforce the count/byte
    caps oldest-first over what is left. Returns (files removed, bytes freed).
    """
    removed = 0
    freed = 0
    survivors: list[tuple[str, float, int]] = []
    for entry in diffs:
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue  # vanished under us; nothing to reap
        if now - st.st_mtime >= ttl_seconds:
            if _remove_file(entry.path):
                removed += 1
                freed += st.st_size
            continue
        survivors.append((entry.path, st.st_mtime, st.st_size))

    survivors.sort(key=lambda item: item[1])  # oldest first
    count = len(survivors)
    total_bytes = sum(size for _, _, size in survivors)
    i = 0
    while i < count and (count > max_count or total_bytes > max_bytes):
        path, _mtime, size = survivors[i]
        i += 1
        if _remove_file(path):
            removed += 1
            freed += size
            count -= 1
            total_bytes -= size
    return removed, freed


def _read_gitdir(git_marker: str) -> str | None:
    """The path recorded in a linked worktree's `.git` FILE ("gitdir: ...",
    the shape `sandbox.create_worktree` always makes), or None if the marker
    is missing, is a directory (never this shape), or does not parse.
    """
    try:
        with open(git_marker, encoding="utf-8", errors="strict") as fh:
            first_line = fh.readline()
    except (OSError, UnicodeDecodeError):
        return None
    if not first_line.startswith("gitdir:"):
        return None
    target = first_line[len("gitdir:") :].strip()
    return target or None


def _worktree_is_stale(parent_path: str) -> bool:
    """True only when this `coding-agent-wt-*` parent can be CONFIRMED to
    hold no live linked worktree. Conservative by construction: any shape
    other than exactly what `sandbox.create_worktree`/`teardown_worktree`
    produce keeps the directory.

    - empty parent (the worktree itself is already gone; only the private
      `mkdtemp` shell survived, e.g. a `teardown_worktree` whose final
      `os.rmdir` step failed) -> stale.
    - exactly one `wt-*` subdirectory whose `.git` marker is missing -> stale
      (clause 1: "the .git file/dir is missing").
    - exactly one `wt-*` subdirectory whose `.git` marker names a gitdir that
      no longer exists on disk -> stale (clause 2: "the recorded gitdir no
      longer exists").
    - anything else (unexpected extra entries, an unparseable `.git`, a
      `.git` that is itself a directory, a listing that cannot be read) ->
      NOT stale; this module does not guess.
    """
    try:
        children = list(os.scandir(parent_path))
    except OSError:
        return False  # can't even list it — don't guess
    if not children:
        return True
    if len(children) != 1:
        return False
    child = children[0]
    try:
        is_wt_dir = child.name.startswith("wt-") and child.is_dir(follow_symlinks=False)
    except OSError:
        return False
    if not is_wt_dir:
        return False
    git_marker = os.path.join(child.path, ".git")
    if not os.path.lexists(git_marker):
        return True
    gitdir = _read_gitdir(git_marker)
    if gitdir is None:
        return False  # unparseable / a directory — conservative: keep
    return not os.path.exists(gitdir)


def _reap_worktrees(
    worktrees: list[os.DirEntry], now: float, ttl_seconds: float
) -> int:
    """Remove `coding-agent-wt-*` parents older than the TTL and confirmed
    stale by `_worktree_is_stale`. Returns the count removed."""
    removed = 0
    for entry in worktrees:
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if now - st.st_mtime < ttl_seconds:
            continue
        if not _worktree_is_stale(entry.path):
            continue
        shutil.rmtree(entry.path, ignore_errors=True)
        if not os.path.lexists(entry.path):
            removed += 1
    return removed


def sweep(
    spill_dir: str | None = None,
    *,
    now: float | None = None,
    ttl_seconds: float = _TTL_SECONDS,
    max_count: int = _MAX_SPILL_COUNT,
    max_bytes: int = _MAX_SPILL_BYTES,
) -> dict[str, int]:
    """Best-effort cleanup of host-side coding_agent temp litter.

    `spill_dir` defaults to `tempfile.gettempdir()` — the same directory
    `loop.py` passes to `walk.unified_diff` as `spill_dir` and `sandbox.py`
    passes to `tempfile.mkdtemp` for a worktree's private parent, so the
    default target always matches where this litter is actually written.

    NEVER raises: every failure below is caught and the entry it came from
    is skipped. A caller on an event loop should still run this via
    `asyncio.to_thread` — see `loop.run_coding_agent` and `mcp_server.main`.

    Returns a stats dict ({"diffs_removed", "diff_bytes_freed",
    "worktrees_removed"}); callers are not required to use it.
    """
    stats = {"diffs_removed": 0, "diff_bytes_freed": 0, "worktrees_removed": 0}
    try:
        target = spill_dir if spill_dir is not None else tempfile.gettempdir()
        clock = now if now is not None else time.time()
        diffs, worktrees = _scan(target)
        removed, freed = _reap_diffs(diffs, clock, ttl_seconds, max_count, max_bytes)
        stats["diffs_removed"] = removed
        stats["diff_bytes_freed"] = freed
        stats["worktrees_removed"] = _reap_worktrees(worktrees, clock, ttl_seconds)
    except Exception:  # noqa: BLE001, S110 — best-effort sweep must never raise
        pass
    return stats
