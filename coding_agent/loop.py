"""Orchestration loop for coding_agent (spec §4, §7).

The loop runs on the HOST. Tool calls dispatch to tools.py; only
run_command crosses into the container. Stop conditions are mechanical
and the model cannot extend its own budget.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, cast

from . import sandbox as _sb
from .basetree import BaseTree, make_ignore, read_base_tree
from .tools import (
    TOOL_SCHEMAS,
    PathEscape,
    tool_list_files,
    tool_read_file,
    tool_run_command,
    tool_write_file,
)
from .walk import snapshot_tree, tree_hash, unified_diff


class StopReason(str, Enum):
    completed = "completed"
    max_turns = "max_turns"
    max_seconds = "max_seconds"
    no_progress = "no_progress"
    error = "error"


@dataclass
class Budget:
    max_turns: int
    max_seconds: float
    started: float = field(default_factory=time.monotonic)

    def exceeded_time(self) -> bool:
        return (time.monotonic() - self.started) >= self.max_seconds

    def remaining(self) -> float:
        """Seconds left on the wall clock; negative once the budget is spent."""
        return self.max_seconds - (time.monotonic() - self.started)

    def bounded(self, ceiling: float, floor: float) -> float:
        """`ceiling`, cut down to what is left of the wall clock.

        Every awaited call inside the loop goes through here. The budget check
        at the top of a turn only decides whether to START one more turn; what
        that turn is then ALLOWED TO SPEND is this. Without it, `max_seconds`
        is not a cap but a check performed at turn boundaries — measured at
        `max_seconds=1.0` giving a 3.0 s run, with a worst case of
        `300 + 32 x 300` per turn, because the model call and each command
        carried their own unconditional 300 s.

        `floor` keeps the last slice usable: clamping to a remaining 0.01 s
        would time every call out and turn an expiring budget into an error
        instead of a clean `max_seconds` stop.
        """
        return max(floor, min(ceiling, self.remaining()))


class ProgressTracker:
    """Progress == walk-hash changed OR (cmd, exit) pair changed OR write_file
    was called this turn. Not adversarially robust by design (spec §5.1) —
    the hard caps are the real budget; this catches *stuck*, not *malicious*.

    Deliberately NOT output-hash based: command output routinely contains
    timestamps, durations, and paths that differ every run, which would make
    every turn look like progress and defeat the detector entirely.

    A turn is measured against the immediately preceding turn. The very
    first observation has no preceding turn to compare against, so it
    counts as showing no change — spec §7's "N turns with no change" is
    read literally, not as "N turns after a baseline is established".
    """

    def __init__(self, no_progress_turns: int = 5) -> None:
        self._n = no_progress_turns
        self._idle = 0
        self._last_hash: str | None = None
        self._last_cmd: tuple[str, int] | None = None

    def observe(
        self, *, tree_hash: str, last_cmd: tuple[str, int] | None, wrote_file: bool
    ) -> bool:
        if self._last_hash is None:
            progressed = False  # no prior turn to compare against yet
        else:
            progressed = (
                wrote_file
                or tree_hash != self._last_hash
                or (last_cmd is not None and last_cmd != self._last_cmd)
            )
        self._last_hash = tree_hash
        if last_cmd is not None:
            self._last_cmd = last_cmd
        self._idle = 0 if progressed else self._idle + 1
        return progressed

    @property
    def stalled(self) -> bool:
        return self._idle >= self._n


# --------------------------------------------------------------------------
# The single slot (spec §6 item 4)
# --------------------------------------------------------------------------

# A DEDICATED single-slot lock. Deliberately not `_DELEGATE_JOB_CAP` (=4,
# and shared with local_delegate): four concurrent 31B inference streams,
# each with its own container, is not honest capacity on this box.
#
# It is only ever acquired UNCONTENDED — the `locked()` check above every
# acquisition rejects a second caller outright — which is what makes
# rejection, not queueing, the behaviour: a queued caller would sit on an
# open MCP request for up to half an hour. That also keeps the lock free of
# asyncio's loop binding (`Lock.acquire`'s fast path never calls
# `_get_loop()`), so the module survives being used from more than one
# `asyncio.run()` over a process's life — which every test file does.
#
# `locked()`-then-`acquire()` is atomic here because nothing between them
# awaits: an uncontended acquire completes without yielding to the loop.
_SLOT = asyncio.Lock()
_SLOT_BUSY = "coding_agent: a run is already in progress on this host"

_T = TypeVar("_T")

# Cleanup's own ceiling (§6 item 3). Sized above sandbox.destroy_container's
# 30s + 5s grace so a slow `docker rm -f` does not eat the worktree removal.
_CLEANUP_BOUND_S = 60.0
# How long we keep re-awaiting a shielded cleanup that cancellations keep
# interrupting. Measured in TIME, not in attempts: a caller that cancels in a
# loop consumes one attempt per cancel, so an attempt COUNT would have given
# up after a few tens of milliseconds — measured, and the reason this is a
# deadline. The teardown task bounds itself at _CLEANUP_BOUND_S, so this only
# has to outlast that.
_CLEANUP_GRACE_S = 5.0

_DIFF_MAX_BYTES = 512 * 1024
_CHAT_TIMEOUT_S = 300.0
_CMD_TIMEOUT_S = 300.0
# Floor for the budget-clamped per-command timeout: a command must get a
# usable slice even at the very end of the wall clock. The same floor applies
# to the model call — see `Budget.bounded`.
_MIN_CMD_TIMEOUT_S = 5.0

_CONTAINER_CPUS = "4"
_CONTAINER_MEMORY = "16g"
_CONTAINER_PIDS = 512

# Everything the model controls that lands in the RETURNED transcript is
# bounded (§5.1: "the diff and transcript are size-capped").
_RESULT_HEAD = 400
_ARG_HEAD = 200
_ARG_KEYS = 8
_OUTPUT_TAIL = 2000
# ...including the two structured path lists. Chosen to sit above
# `walk._CHANGED_SHOWN` (100), so a result never names fewer paths than the
# diff text itself already does. WHICH 500 survive is the load-bearing half:
# see `_capped_paths`, which fills both lists base-tree-first for the same
# reason `walk` renders its body that way.
_RESULT_PATHS = 500
# One assistant turn may not queue unbounded work or unbounded transcript.
_MAX_TOOL_CALLS_PER_TURN = 32

# The IN-FLIGHT conversation, which is a different thing from the RETURNED
# transcript and was the only model-controlled accumulation left unbounded.
# Every tool result is appended in full and the whole list is re-serialised
# into the Ollama payload on EVERY turn, so the ceiling was
# `60 turns x 32 calls x MAX_READ (256 KB)` ~= 490 MB, re-encoded each turn.
# 1 MiB is already well past any context window this feature targets, so the
# bound costs nothing a real run would notice.
_MAX_CONVERSATION_CHARS = 1 << 20
# What an over-budget tool result is cut down to. Matches `_OUTPUT_TAIL`'s
# scale: enough to keep the shape of what happened, not enough to matter.
_ELIDED_TOOL_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a coding agent working in a sandboxed copy of a repository. You have "
    "list_files, read_file, write_file, and run_command (no network). Work until the "
    "task's success criterion is met, then reply with a short summary and NO tool calls."
)


class SandboxOps(Protocol):
    """The lifecycle seam `run_coding_agent` drives.

    Satisfied by `coding_agent.sandbox` itself; injectable so control-flow
    tests need neither docker nor git. NOTE: `run_command` does NOT come
    through here — `tools.tool_run_command` calls `sandbox.exec_in_container`
    directly — so a test that scripts a `run_command` turn must patch
    `coding_agent.tools.exec_in_container` as well, or it will shell out to a
    real docker daemon.
    """

    def create_worktree(self, repo: str, base_ref: str) -> str: ...

    async def start_container(
        self, worktree: str, image: str, *, cpus: str, memory: str, pids: int
    ) -> str: ...

    async def destroy_container(self, container: str) -> None: ...

    def teardown_worktree(self, repo: str, worktree: str) -> list[str]: ...


@dataclass
class AgentResult:
    stop_reason: StopReason
    turns: int
    elapsed_seconds: float
    diff: str
    diff_truncated: bool
    diff_full_path: str | None
    changed_files: list[str]
    # Paths that EXIST in the worktree but the walk could not read, as
    # {"path": ..., "reason": ...}. Carried all the way out on purpose: a
    # `chmod 000` file that silently vanishes from the diff is a demonstrated
    # backdoor-hiding vector, and a warning the human never sees is the same
    # silent under-show with extra steps.
    unreadable: list[dict[str, str]]
    last_command: dict[str, Any] | None
    transcript: list[dict[str, Any]]
    model: str
    cleanup_problems: list[str]


@dataclass
class _Cleanup:
    """What teardown produces: the result's host-read half, plus residue."""

    problems: list[str] = field(default_factory=list)
    diff: str = ""
    diff_truncated: bool = False
    diff_full_path: str | None = None
    changed_files: list[str] = field(default_factory=list)
    unreadable: list[dict[str, str]] = field(default_factory=list)


def _default_ops() -> SandboxOps:
    return _sb


def _as_dict(value: Any) -> dict[str, Any]:
    """Model-supplied `arguments` as a dict, whatever it actually was.

    Ollama sends a dict; other servers (and a confused model) send a JSON
    string, a list, or a bare scalar. None of those may reach a tool as a
    positional index or crash the run — spec §9 says a malformed tool call is
    fed BACK as a tool error, not turned into `stop_reason: error`.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _summarize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Bound the model's own arguments before they enter the transcript.

    `write_file(content=<1 MB>)` would otherwise put a megabyte of
    model-chosen bytes into the MCP response — the transcript is size-capped
    for the same reason the diff is.
    """
    out: dict[str, Any] = {}
    for key, value in list(args.items())[:_ARG_KEYS]:
        if isinstance(value, str):
            out[key] = (
                f"{value[:_ARG_HEAD]}… [{len(value)} chars]"
                if len(value) > _ARG_HEAD
                else value
            )
        elif value is None or isinstance(value, (bool, int, float)):
            out[key] = value
        else:
            rendered = repr(value)
            out[key] = (
                f"{rendered[:_ARG_HEAD]}…" if len(rendered) > _ARG_HEAD else rendered
            )
    if len(args) > _ARG_KEYS:
        out["…"] = f"[{len(args) - _ARG_KEYS} more argument(s) omitted]"
    return out


async def _teardown(
    ops: SandboxOps,
    *,
    repo: str,
    worktree: str,
    container: str | None,
    base: BaseTree,
    ignore: Callable[[str], bool],
    out: _Cleanup,
) -> None:
    """Destroy, THEN read, THEN remove. Ordering and layering are normative.

    §6.5 rule 3: the container is `docker rm -f`'d **before** any host read
    that builds the returned result. A live container can race the walk that
    produces the diff — writing a file the human never sees, or removing one
    they would have — and a destroyed one cannot. (The per-turn no-progress
    hash necessarily runs with the container alive; that is accepted, and
    mitigated inside walk.py. The FINAL read is not.)

    §6.6: every layer is unconditional. The `rm -rf` of the literal recorded
    path sits in a `finally`, so it runs whether the container destroy
    failed, whether the walk failed, and whether cleanup's own ceiling
    cancelled us mid-way. Nothing here raises: teardown that can raise is
    teardown that can be skipped.
    """
    try:
        try:
            if container is not None:
                await ops.destroy_container(container)
        except Exception as exc:  # noqa: BLE001 — a teardown step may not raise
            # The diff below is still computed — losing it entirely on a
            # docker hiccup would cost the human the only record of what the
            # model did — but it is no longer race-free, and saying so is the
            # difference between a caveated artifact and a quiet under-show.
            out.problems.append(
                f"docker rm -f failed: {type(exc).__name__}: {exc} — container "
                f"{container} may still be running, so the diff may be incomplete"
            )
        # ---- the host read. Never before the line above. ----
        try:
            snap = await asyncio.to_thread(snapshot_tree, worktree, ignore)
            diff = unified_diff(
                base,
                snap,
                max_bytes=_DIFF_MAX_BYTES,
                # A host temp dir; the sandbox mounts only the worktree, so
                # the spilled full diff is outside everything it can reach.
                spill_dir=tempfile.gettempdir(),
            )
            out.diff = diff.text
            out.diff_truncated = diff.truncated
            out.diff_full_path = diff.full_path
            # Both lists are capped BASE-TREE-FIRST, the same asymmetry
            # `walk.unified_diff` orders its body and its appended path list
            # by: a path the sandbox created cannot evict one the host's own
            # repository holds. Capping these by sort position alone would
            # have left that hole open one layer above the diff text.
            reaches = _base_reach(base)
            out.changed_files = _capped_paths(
                list(diff.changed_files),
                where=(
                    f"the complete diff is at {diff.full_path}"
                    if diff.full_path is not None
                    else "the diff text above names every changed path"
                ),
                keep_first=reaches,
            )
            out.unreadable = _capped_paths(
                [
                    {"path": item.path, "reason": item.reason}
                    for item in diff.unreadable
                ],
                # NOT the same sentence as above: the diff's warning block is
                # itself bounded (`walk._UNREADABLE_SHOWN`), so an untruncated
                # diff does NOT name every unreadable path the way it names
                # every changed one. Only the spill file does.
                where=(
                    f"the complete list is in the diff at {diff.full_path}"
                    if diff.full_path is not None
                    else "the complete list is NOT in this response - the diff's "
                    "own warning block is bounded too"
                ),
                keep_first=lambda rec: reaches(rec["path"]),
            )
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            out.problems.append(
                f"could not read the worktree to build the diff: "
                f"{type(exc).__name__}: {exc}"
            )
    finally:
        _remove_worktree(ops, repo, worktree, out)


def _tool_call_arg_chars(m: dict[str, Any]) -> int:
    """Bytes an assistant message's `tool_calls` arguments add to the
    payload — the request-side counterpart to `content`, which the original
    ceiling analysis above missed: a single `write_file` call embeds its
    whole `content` argument as a JSON string here, so a model that writes a
    large file grows the conversation exactly like a large tool RESULT does,
    through a field `_trim_conversation` used to never look at."""
    calls = m.get("tool_calls")
    if not isinstance(calls, list):
        return 0
    total = 0
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            total += len(str(function.get("arguments") or ""))
    return total


def _trim_conversation(messages: list[dict[str, Any]]) -> int:
    """Bound the in-flight conversation by shrinking oversized tool results
    and `tool_calls` arguments, oldest message first regardless of which
    kind it is.

    Returns the number of fields it elided.

    Shrinking content in place rather than dropping whole messages is
    deliberate: the assistant/tool message pairing is protocol, and evicting
    one side of a `tool_calls` pair is how a "memory fix" turns into a model
    backend rejecting the payload. This changes only bytes — never the message
    count, the roles, or the `tool_calls` array's shape (id/type/name/call
    count are untouched; only an oversized `arguments` string is cut).

    Oldest first, because the recent turns are the ones the model is actually
    reasoning about; the early ones have already done their work.
    """
    total = sum(
        len(str(m.get("content") or "")) + _tool_call_arg_chars(m) for m in messages
    )
    if total <= _MAX_CONVERSATION_CHARS:
        return 0
    elided = 0
    for m in messages:
        if total <= _MAX_CONVERSATION_CHARS:
            break
        role = m.get("role")
        if role == "tool":
            content = str(m.get("content") or "")
            if len(content) <= _ELIDED_TOOL_CHARS:
                continue
            cut = content[:_ELIDED_TOOL_CHARS] + (
                f"\n[... {len(content) - _ELIDED_TOOL_CHARS} chars elided from an "
                f"earlier tool result to bound this conversation ...]"
            )
            total -= len(content) - len(cut)
            m["content"] = cut
            elided += 1
        elif role == "assistant":
            calls = m.get("tool_calls")
            if not isinstance(calls, list):
                continue  # the system prompt, the task, or a plain reply turn
            for call in calls:
                if total <= _MAX_CONVERSATION_CHARS:
                    break
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                args = str(function.get("arguments") or "")
                if len(args) <= _ELIDED_TOOL_CHARS:
                    continue
                cut = args[:_ELIDED_TOOL_CHARS] + (
                    f"...[{len(args) - _ELIDED_TOOL_CHARS} chars elided from an "
                    f"earlier tool call to bound this conversation]"
                )
                total -= len(args) - len(cut)
                function["arguments"] = cut
                elided += 1
    return elided


def _base_reach(base: BaseTree) -> Callable[[str], bool]:
    """True for a path whose fate the BASE TREE has a stake in.

    The exact base paths, plus every directory an entry lives under, plus the
    worktree root — because an `unreadable` record is not always a file. A
    record for `src/` (or, when the classifying `lstat` is what failed, for
    `src/deep`) is the thing hiding every tracked path beneath it, and losing
    THAT record to a flood of sandbox-named ones is the same under-show as
    losing the file itself.

    The set is derived ONCE from `base.entries`, which `basetree` read from
    the host's own repository with git; the sandbox cannot add a member to it
    at any cost, which is the whole reason this is a usable priority signal
    and a name-sort is not.
    """
    dirs = {p[:i] for p in base.entries for i, ch in enumerate(p) if ch == "/"}

    def reaches(path: str) -> bool:
        bare = path[:-1] if path.endswith("/") else path
        return bare == "." or bare in base.entries or bare in dirs

    return reaches


def _capped_paths(
    items: list[_T],
    *,
    where: str = "the remainder is not in this response",
    keep_first: Callable[[_T], bool] | None = None,
) -> list[_T]:
    """Bound a sandbox-sized list, keeping an EXACT count of what was dropped.

    Every other model-controlled field in the result is already bounded — the
    diff at 512 KB, `result_head` at 400, arguments at 8x200, `output_tail` at
    2000 — but `changed_files` and `unreadable` were carried verbatim, and
    both are as long as the sandbox chooses. Measured: 200,000 unreadable
    paths keep the diff TEXT correctly at 1,220 bytes while the result JSON
    reaches 11.8 MB, and that JSON lands in the caller's context. The breadth
    DoS is accepted as an availability trade-off for host MEMORY; this is the
    other half of it, and it is the half that reaches a human.

    WHICH items survive is a security question, not a formatting one, and the
    pass-8 answer to it is `keep_first`. Truncating by position alone let 600
    files named `!0000.txt`… evict a backdoored TRACKED path from
    `changed_files` — measured, at the shipped `_RESULT_PATHS` — leaving the
    path in no field of the response at all. Callers pass a predicate that is
    true for paths the BASE TREE reaches; those are selected first, the rest
    fill what is left, and the survivors are emitted in the caller's original
    (sorted) order, so nothing but the choice of survivors changes.

    `where` is the caller's statement of where the complete data really is,
    and it is a parameter precisely because the honest answer differs per
    field and per run: the untruncated diff names every changed path but only
    a bounded HEAD of the unreadable list, and a spill file exists only when
    the diff was truncated. The default assumes none of that. The marker used
    to assert "the diff text and its spill file are complete", which was false
    in both halves whenever the diff was truncated, and named a spill file
    that does not exist when it was not.

    The overflow marker is a value of the same SHAPE as the list's elements,
    so a consumer iterating either field sees a well-formed entry rather than
    a type error — and the count is exact, which is the property that keeps
    this from being one more silent truncation.
    """
    if len(items) <= _RESULT_PATHS:
        return items
    dropped = len(items) - _RESULT_PATHS
    if keep_first is None:
        kept = items[:_RESULT_PATHS]
    else:
        first = [i for i, item in enumerate(items) if keep_first(item)]
        rest = [i for i, item in enumerate(items) if not keep_first(item)]
        kept = [items[i] for i in sorted([*first, *rest][:_RESULT_PATHS])]
    note = f"[... and {dropped} more; {where} ...]"
    if kept and isinstance(kept[0], dict):
        return [*kept, cast("_T", {"path": note, "reason": "TRUNCATED"})]
    return [*kept, cast("_T", note)]


def _remove_worktree(ops: SandboxOps, repo: str, worktree: str, out: _Cleanup) -> None:
    """The last layer, on its own so the abandonment path can also run it.

    Synchronous by nature — `rm -rf` of the literal recorded path plus the
    real repo's prune — which is what lets it be the fallback when the async
    teardown cannot be awaited: sync code cannot be cancelled. Idempotent, so
    running it twice costs nothing.

    DELIBERATELY NOT moved to `asyncio.to_thread`, unlike the other blocking
    host calls in this module. Being uncancellable is the entire reason this
    layer exists: it is what the repeated-cancellation path runs when the
    shielded teardown cannot be awaited to completion, and an `await` there
    would reintroduce exactly the cancellation point that would leak the
    worktree. It blocks the event loop, and that is the cheaper of the two
    costs — a leaked worktree is permanent, a blocked loop is not.
    """
    try:
        out.problems.extend(ops.teardown_worktree(repo, worktree))
    except Exception as exc:  # noqa: BLE001 — the last layer, still bounded
        out.problems.append(
            f"worktree teardown failed: {type(exc).__name__}: {exc} "
            f"(worktree may remain at {worktree})"
        )


async def _run_cleanup(
    ops: SandboxOps,
    *,
    repo: str,
    worktree: str,
    container: str | None,
    base: BaseTree,
    ignore: Callable[[str], bool],
    out: _Cleanup,
) -> None:
    """`asyncio.shield(asyncio.wait_for(cleanup, T))`, awaited to COMPLETION.

    Shielded so an outer ceiling's `CancelledError` cannot cancel teardown —
    that would turn the runaway-bounding timeout into a leak path. Bounded so
    a wedged docker daemon cannot wedge teardown forever.

    The subtlety `shield` hides, and the reason for the retry loop: shield
    protects the *inner* task, not the awaiter. The awaiter resumes the
    instant it is cancelled while the teardown is still in flight — so a
    single `await shield(...)` + `except CancelledError: pass` returns a
    result built from a worktree nobody has read yet and a container still
    running: an empty diff and an empty `cleanup_problems`, i.e. "the model
    changed nothing and cleanup was clean" for a run where neither is true.
    Measured, not theorised. So we re-await the same task until it is done.

    Cancellation is never swallowed: once teardown has finished, the
    `CancelledError` is re-raised so the caller's `cancel()` means what it says.
    """
    cleanup = asyncio.ensure_future(
        asyncio.wait_for(
            _teardown(
                ops,
                repo=repo,
                worktree=worktree,
                container=container,
                base=base,
                ignore=ignore,
                out=out,
            ),
            timeout=_CLEANUP_BOUND_S,
        )
    )
    deadline = time.monotonic() + _CLEANUP_BOUND_S + _CLEANUP_GRACE_S
    cancelled: BaseException | None = None
    while True:
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as exc:
            cancelled = exc
            if not cleanup.done():
                # shield absorbed it; the teardown is still running
                if time.monotonic() < deadline:
                    continue
                # Giving up on an unfinished teardown would leak the
                # worktree, so its last layer is run HERE instead — sync
                # code cannot be cancelled, and it is idempotent.
                out.problems.append(
                    "cleanup could not be awaited to completion (repeated "
                    "cancellation); the container may still be running"
                )
                _remove_worktree(ops, repo, worktree, out)
        except TimeoutError:
            # wait_for cancelled the teardown at its ceiling. Its `finally`
            # still ran the unconditional worktree removal on the way out.
            out.problems.append(
                f"cleanup exceeded its {_CLEANUP_BOUND_S:g}s bound; "
                f"later layers may not have completed"
            )
        except Exception as exc:  # noqa: BLE001 — a result is owed regardless
            out.problems.append(f"cleanup failed: {type(exc).__name__}: {exc}")
        break
    if cancelled is not None:
        raise cancelled


async def run_coding_agent(
    *,
    task: str,
    repo: str,
    base_ref: str,
    model: str,
    max_turns: int,
    max_seconds: float,
    image: str,
    chat: Callable[[dict[str, Any], float], Awaitable[dict[str, Any]]],
    sandbox_factory: Callable[[], SandboxOps] | None = None,
) -> AgentResult:
    """Run one bounded agentic coding session and return what it produced.

    NOTHING is applied: the worktree is destroyed on the way out and the
    caller gets a transcript plus a unified diff to review. `repo` is never
    written to, and no host-side `git` ever runs with the worktree as cwd or
    gitdir (§6.5) — `basetree` reads the base ref from the real repo's object
    store, and `sandbox`'s git runs are all `git -C <repo>`.

    `chat` is `mcp_server._post_ollama_chat`, injected so tests need no model.
    `sandbox_factory` swaps the docker/git lifecycle (see `SandboxOps`).

    Raises `RuntimeError` immediately if another run holds the single slot,
    and re-raises `CancelledError` (after cleanup completes) if cancelled.
    """
    if _SLOT.locked():
        raise RuntimeError(_SLOT_BUSY)
    async with _SLOT:
        ops = (sandbox_factory or _default_ops)()
        started = time.monotonic()
        budget = Budget(max_turns=max_turns, max_seconds=max_seconds, started=started)
        # The REAL repo, before anything exists to clean up. Trusted side.
        #
        # OFF THE EVENT LOOP, like every other blocking host call here. This
        # one is the expensive one: `read_base_tree` runs one `git cat-file`
        # subprocess PER TRACKED BLOB, measured at 9.4 ms per file on this
        # repo — ~47 s for an ordinary 5,000-file checkout, ~3 min at 20,000.
        # Run inline it blocks the shared MCP event loop for that whole
        # window, during which nothing else the server offers responds,
        # INCLUDING `coding_agent_result` — so `background=true`'s "poll for
        # progress" promise would not hold precisely while the run was
        # starting. `mcp_server.py` already uses `asyncio.to_thread` for its
        # other blocking work; this makes the coding path match.
        base: BaseTree = await asyncio.to_thread(read_base_tree, repo, base_ref)
        ignore = make_ignore(base)
        # The literal path, recorded once. Teardown rm -rf's this string and
        # never re-derives it through the sandbox-controlled worktree.
        worktree = await asyncio.to_thread(ops.create_worktree, repo, base_ref)
        container: str | None = None
        transcript: list[dict[str, Any]] = []
        last_cmd: dict[str, Any] | None = None
        stop = StopReason.error
        turns = 0
        elapsed = 0.0
        tracker = ProgressTracker()
        cleanup = _Cleanup()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        try:
            container = await ops.start_container(
                worktree,
                image,
                cpus=_CONTAINER_CPUS,
                memory=_CONTAINER_MEMORY,
                pids=_CONTAINER_PIDS,
            )
            while True:
                if turns >= budget.max_turns:
                    stop = StopReason.max_turns
                    break
                if budget.exceeded_time():
                    stop = StopReason.max_seconds
                    break
                turns += 1
                # Bounded BEFORE the payload is built, so the cap applies to
                # what actually goes over the wire rather than to what is
                # kept afterwards.
                _trim_conversation(messages)
                response = await chat(
                    {
                        "model": model,
                        "messages": messages,
                        "tools": TOOL_SCHEMAS,
                        "stream": False,
                        "think": False,
                    },
                    # Clamped by what is left of the wall clock, for the same
                    # reason `run_command` is: the budget check above decides
                    # whether to start a turn, not how long that turn may run.
                    budget.bounded(_CHAT_TIMEOUT_S, _MIN_CMD_TIMEOUT_S),
                )
                if response.get("status") == "failed":
                    transcript.append(
                        {
                            "turn": turns,
                            "error": str(response.get("error", "model call failed"))[
                                :_RESULT_HEAD
                            ],
                        }
                    )
                    stop = StopReason.error
                    break
                message = response.get("message")
                if not isinstance(message, dict):
                    # A non-failure response with no message is a broken
                    # endpoint, not a model that finished. Calling it
                    # `completed` would read as "the model says it is done".
                    transcript.append(
                        {
                            "turn": turns,
                            "error": f"model response had no message: "
                            f"{sorted(response)[:8]}",
                        }
                    )
                    stop = StopReason.error
                    break
                raw_calls = message.get("tool_calls")
                calls = list(raw_calls) if isinstance(raw_calls, list) else []
                messages.append(
                    {
                        key: value
                        for key, value in message.items()
                        if key in ("role", "content", "tool_calls")
                    }
                    or {"role": "assistant", "content": ""}
                )
                if not calls:
                    stop = StopReason.completed
                    break
                dropped = len(calls) - _MAX_TOOL_CALLS_PER_TURN
                calls = calls[:_MAX_TOOL_CALLS_PER_TURN]
                wrote = False
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    function = function if isinstance(function, dict) else {}
                    name = function.get("name")
                    args = _as_dict(function.get("arguments"))
                    escaped = False
                    try:
                        if name == "list_files":
                            where = args.get("path", ".")
                            result = tool_list_files(
                                worktree,
                                where if isinstance(where, str) else ".",
                                # Without this the tool's own advertised
                                # "respects .gitignore" is simply false.
                                ignore=ignore,
                            )
                        elif name == "read_file":
                            path = args.get("path")
                            result = (
                                tool_read_file(worktree, path)
                                if isinstance(path, str)
                                else "error: read_file needs a string 'path'"
                            )
                        elif name == "write_file":
                            # §7(c) is "any write_file call at all in the
                            # turn", read literally — a rejected one counts
                            # too. The hash already covers writes that
                            # landed; this signal exists for the ones that
                            # change no bytes.
                            wrote = True
                            path = args.get("path")
                            content = args.get("content", "")
                            if not isinstance(path, str):
                                result = "error: write_file needs a string 'path'"
                            elif not isinstance(content, str):
                                result = "error: write_file needs a string 'content'"
                            else:
                                result = tool_write_file(worktree, path, content)
                        elif name == "run_command":
                            command = args.get("cmd")
                            if not isinstance(command, str):
                                result = "error: run_command needs a string 'cmd'"
                            elif container is None:
                                result = "error: no sandbox container is running"
                            else:
                                code, output = await tool_run_command(
                                    container,
                                    command,
                                    # Clamped by what is left of the wall
                                    # clock, so `sleep 100000` cannot push the
                                    # run far past the caller's max_seconds.
                                    timeout_s=budget.bounded(
                                        _CMD_TIMEOUT_S, _MIN_CMD_TIMEOUT_S
                                    ),
                                )
                                last_cmd = {
                                    "cmd": command,
                                    "exit": code,
                                    "output_tail": output[-_OUTPUT_TAIL:],
                                }
                                result = f"[exit {code}]\n{output}"
                        else:
                            result = f"error: unknown tool {name!r}"
                    except PathEscape as exc:
                        # The one exception the tools raise, and it means
                        # "left the worktree" — a security event, not a typo.
                        # Fed back so the loop continues under its own caps,
                        # but flagged in the transcript the human reads.
                        escaped = True
                        result = f"error: PathEscape: {exc}"
                    except (KeyError, TypeError, OSError) as exc:
                        result = f"error: {type(exc).__name__}: {exc}"
                    entry: dict[str, Any] = {
                        "turn": turns,
                        "tool": name,
                        "args": _summarize_args(args),
                        "result_head": result[:_RESULT_HEAD],
                    }
                    if escaped:
                        entry["path_escape"] = True
                    transcript.append(entry)
                    messages.append({"role": "tool", "content": result})
                if dropped > 0:
                    note = (
                        f"error: {dropped} further tool call(s) in this turn were "
                        f"dropped; at most {_MAX_TOOL_CALLS_PER_TURN} run per turn"
                    )
                    transcript.append(
                        {"turn": turns, "tool": None, "args": {}, "result_head": note}
                    )
                    messages.append({"role": "tool", "content": note})
                # Host byte-walk, never `git add -A` in the worktree (§6.5/§7a).
                # Off the event loop: this one runs once per TURN, so on a
                # large tree it is the repeat offender even though each walk
                # is cheaper than the base-tree read.
                snapshot_hash = tree_hash(
                    await asyncio.to_thread(snapshot_tree, worktree, ignore)
                )
                pair = (
                    (str(last_cmd["cmd"]), int(last_cmd["exit"])) if last_cmd else None
                )
                tracker.observe(
                    tree_hash=snapshot_hash, last_cmd=pair, wrote_file=wrote
                )
                if tracker.stalled:
                    stop = StopReason.no_progress
                    break
        except Exception as exc:  # noqa: BLE001 — surfaced in the result, cleanup still runs
            transcript.append(
                {"turn": turns, "error": f"{type(exc).__name__}: {exc}"[:_RESULT_HEAD]}
            )
            stop = StopReason.error
        finally:
            # How long the AGENT ran — measured before teardown, so a slow
            # `docker rm -f` is not reported as the model thinking.
            elapsed = round(time.monotonic() - started, 1)
            await _run_cleanup(
                ops,
                repo=repo,
                worktree=worktree,
                container=container,
                base=base,
                ignore=ignore,
                out=cleanup,
            )
        return AgentResult(
            stop_reason=stop,
            turns=turns,
            elapsed_seconds=elapsed,
            diff=cleanup.diff,
            diff_truncated=cleanup.diff_truncated,
            diff_full_path=cleanup.diff_full_path,
            changed_files=cleanup.changed_files,
            unreadable=cleanup.unreadable,
            last_command=last_cmd,
            transcript=transcript,
            model=model,
            cleanup_problems=cleanup.problems,
        )
