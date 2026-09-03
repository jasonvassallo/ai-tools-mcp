#!/usr/bin/env python3
"""Control-flow tests for coding_agent.loop.

Everything except `ContainerGoneBeforeHostRead` runs without docker and
without ollama. That one class needs a real daemon and proves — with an
observation made by a SEPARATE PROCESS — the property the in-process
ordering test can only assert about its own bookkeeping.

Run:  uv run --with pytest --with pytest-timeout --with pathspec pytest test_coding_agent_loop.py -q

`--with pytest-timeout` is requested for parity with the rest of the suite's
`Run:` lines (see test_coding_agent_security.py and test_coding_agent_tools.py
for the tests it actually bounds), not because anything in this file relies on
it — every `asyncio.wait_for`/`asyncio.Event` pairing below already carries its
own explicit deadline. It is optional here too: without the plugin this file
runs exactly as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pathspec

from coding_agent import loop as L
from coding_agent.basetree import BaseTree
from coding_agent.loop import ProgressTracker
from coding_agent.walk import Entry

# ---------------------------------------------------------------------------
# $TMPDIR hygiene (issue #79 follow-through, not the reaper itself — that is
# covered end-to-end in test_coding_agent_reap.py against per-test dirs).
#
# `run_coding_agent` spills over-cap diffs to the REAL host temp dir by
# design (`spill_dir=tempfile.gettempdir()` is pinned by
# `test_the_loop_ships_its_own_diff_cap_and_spills_outside_the_worktree`), so
# a test in THIS file cannot redirect it without changing the thing #79
# reported. Before the fix here, an unmodified run of this file left two
# `coding-agent-diff-*.patch` files (~1.6 MB) behind — the tests that hit the
# over-cap path elsewhere in the suite have no `addCleanup` of their own.
# Rather than chase down every call site, account for the module as a whole:
# snapshot what already matches the pattern before any test runs, and remove
# whatever this module added by the time it is done.
# ---------------------------------------------------------------------------


def _diff_spill_names() -> set[str]:
    try:
        return {
            e.name
            for e in os.scandir(tempfile.gettempdir())
            if e.name.startswith("coding-agent-diff-") and e.name.endswith(".patch")
        }
    except OSError:
        return set()


_PRE_EXISTING_DIFF_SPILLS: set[str] = set()


def setUpModule() -> None:
    global _PRE_EXISTING_DIFF_SPILLS
    _PRE_EXISTING_DIFF_SPILLS = _diff_spill_names()


def tearDownModule() -> None:
    for name in _diff_spill_names() - _PRE_EXISTING_DIFF_SPILLS:
        with contextlib.suppress(OSError):
            os.unlink(os.path.join(tempfile.gettempdir(), name))


class NoProgressDetector(unittest.TestCase):
    """Spec §7: stalls after N turns with no walk-hash change, no
    (cmd, exit) change, and no write_file. Deliberately NOT output-hash
    based (timestamps would flip it every turn)."""

    def test_stalls_after_n_identical_turns(self):
        t = ProgressTracker(no_progress_turns=3)
        for _ in range(3):
            t.observe(tree_hash="H", last_cmd=("pytest", 1), wrote_file=False)
        self.assertTrue(t.stalled)

    def test_hash_change_is_progress(self):
        t = ProgressTracker(no_progress_turns=3)
        t.observe(tree_hash="H1", last_cmd=None, wrote_file=False)
        t.observe(tree_hash="H1", last_cmd=None, wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H2", last_cmd=None, wrote_file=False))
        self.assertFalse(t.stalled)

    def test_exit_flip_same_command_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=("pytest", 1), wrote_file=False)
        self.assertTrue(
            t.observe(tree_hash="H", last_cmd=("pytest", 0), wrote_file=False)
        )

    def test_different_command_same_exit_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=("pytest a", 0), wrote_file=False)
        self.assertTrue(
            t.observe(tree_hash="H", last_cmd=("pytest b", 0), wrote_file=False)
        )

    def test_write_file_alone_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=None, wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=None, wrote_file=True))
        self.assertFalse(t.stalled)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingOps:
    """The `SandboxOps` seam, instrumented.

    `calls` records the observable teardown ORDER — `destroy:done` is appended
    only once the container is really gone, so a test can assert the host read
    happened after it and not merely after the destroy was *requested*.
    """

    def __init__(
        self,
        worktree: str,
        *,
        destroy_delay: float = 0.0,
        destroy_exc: BaseException | None = None,
        # What `sandbox.destroy_container` returns when it could not confirm
        # the removal. None is the confirmed case, so every existing fake
        # keeps meaning exactly what it meant before.
        destroy_report: str | None = None,
        start_exc: BaseException | None = None,
        teardown_problems: tuple[str, ...] = (),
        teardown_exc: BaseException | None = None,
    ) -> None:
        self.worktree = worktree
        self.calls: list[str] = []
        self.container_alive = False
        self.destroy_delay = destroy_delay
        self.destroy_exc = destroy_exc
        self.destroy_report = destroy_report
        self.start_exc = start_exc
        self.teardown_problems = teardown_problems
        self.teardown_exc = teardown_exc

    def create_worktree(self, repo, base_ref):
        self.calls.append("create_worktree")
        return self.worktree

    async def start_container(self, worktree, image, **kw):
        self.calls.append("start_container")
        if self.start_exc is not None:
            raise self.start_exc
        self.container_alive = True
        return "cid"

    async def destroy_container(self, container):
        self.calls.append("destroy:start")
        if self.destroy_delay:
            await asyncio.sleep(self.destroy_delay)
        if self.destroy_exc is not None:
            raise self.destroy_exc
        if self.destroy_report is not None:
            # An UNCONFIRMED removal: the container is deliberately left
            # `alive`, because that is precisely what the report means.
            self.calls.append("destroy:unconfirmed")
            return self.destroy_report
        self.container_alive = False
        self.calls.append("destroy:done")
        return None

    def teardown_worktree(self, repo, worktree):
        self.calls.append("teardown_worktree")
        if self.teardown_exc is not None:
            raise self.teardown_exc
        return list(self.teardown_problems)


def _scripted_chat(turns):
    """A fake `_post_ollama_chat` replaying `turns` (assistant message dicts)."""
    it = iter(turns)

    async def chat(payload, timeout_s):
        return {"message": next(it)}

    return chat


def _say(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


def _call(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def _base_tree(files: dict[str, str] | None = None) -> BaseTree:
    """A BaseTree whose ignore rules come from a TRACKED .gitignore."""
    files = files or {}
    entries = {path: Entry(path, "file", body.encode()) for path, body in files.items()}
    text = files.get(".gitignore", "")
    spec = pathspec.PathSpec.from_lines("gitwildmatch", text.splitlines())

    def raw(path: str) -> bool:
        return bool(spec.check_file(path).include)

    return BaseTree(entries=entries, ignore=raw, tracked=frozenset(entries))


class _LoopCase(unittest.TestCase):
    """Base: a scratch repo path, a scratch worktree, and a runner."""

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp(prefix="ca-loop-repo-")
        self.wt = tempfile.mkdtemp(prefix="ca-loop-wt-")
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.wt, ignore_errors=True)
        # NOT `addCleanup(self.assertFalse, L._SLOT.locked())` — addCleanup
        # binds its arguments NOW, so that form asserts the state at setUp and
        # can never fail. The slot has to be read when the cleanup runs.
        self.addCleanup(self._assert_slot_free)
        # issue #79: `run_coding_agent` now sweeps the REAL host temp dir
        # (`reap.sweep`'s default target) at the top of every run. Patched
        # to a no-op here so the hundreds of `run_loop` calls in this file
        # never touch the developer's actual $TMPDIR; `reap.py`'s own
        # behavior is covered end-to-end in test_coding_agent_reap.py
        # against per-test dirs, and `ReaperRunsBeforeTheSandbox` below
        # restores the real thing just long enough to prove the call site.
        self.reap_patcher = mock.patch.object(L._reap, "sweep", autospec=True)
        self.reap_mock = self.reap_patcher.start()
        self.addCleanup(self.reap_patcher.stop)

    def _assert_slot_free(self) -> None:
        self.assertFalse(L._SLOT.locked(), "the single slot leaked")

    def write(self, rel: str, body: str) -> None:
        path = Path(self.wt, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def run_loop(self, turns, *, ops=None, base=None, **kw):
        ops = ops or _RecordingOps(self.wt)
        chat = kw.pop("chat", None) or _scripted_chat(turns)
        with mock.patch.object(L, "read_base_tree", return_value=base or _base_tree()):
            result = asyncio.run(
                L.run_coding_agent(
                    task=kw.pop("task", "t"),
                    repo=self.repo,
                    base_ref="HEAD",
                    model="m",
                    max_turns=kw.pop("max_turns", 10),
                    max_seconds=kw.pop("max_seconds", 60),
                    image="img",
                    chat=chat,
                    sandbox_factory=lambda: ops,
                    **kw,
                )
            )
        return result, ops


# ---------------------------------------------------------------------------
# The reaper is actually invoked (issue #79)
# ---------------------------------------------------------------------------


class ReaperRunsBeforeTheSandbox(_LoopCase):
    """`run_coding_agent` sweeps host temp-dir litter from earlier runs
    before it creates its OWN worktree — a crash on run N must not compound
    across run N+1, N+2, ... `reap.py`'s own sweep logic is exercised
    end-to-end in test_coding_agent_reap.py; this only pins that `loop.py`
    actually calls it, and calls it before `ops.create_worktree`."""

    def test_the_loop_invokes_the_reaper_before_create_worktree(self):
        ops = _RecordingOps(self.wt)
        seen_ops_calls_at_reap_time: list[list[str]] = []
        self.reap_mock.side_effect = lambda *a, **kw: (
            seen_ops_calls_at_reap_time.append(list(ops.calls))
        )

        self.run_loop([_say("done")], ops=ops)

        self.reap_mock.assert_called_once()
        # `ops.calls` had nothing in it yet when the reaper ran — proof the
        # sweep happened strictly before `create_worktree`, not just
        # somewhere in the run.
        self.assertEqual(seen_ops_calls_at_reap_time, [[]])
        self.assertIn("create_worktree", ops.calls)


# ---------------------------------------------------------------------------
# The normative ordering (§6.5 rule 3)
# ---------------------------------------------------------------------------


class TeardownOrdering(_LoopCase):
    """The container is destroyed BEFORE any host read that builds the result.

    The obvious version of this test — assert `destroy_container` is recorded
    before `teardown_worktree` — is VACUOUS: it passes just as happily with
    the host read moved in front of the destroy, because the read is not in
    that record at all. So the read itself is instrumented here, and what is
    asserted is that the container was already GONE when it ran.
    """

    def _run_watching_the_read(self, turns, ops=None, base=None):
        ops = ops or _RecordingOps(self.wt)
        alive_at_read: list[bool] = []
        real = L.snapshot_tree

        def spy(root, is_ignored):
            alive_at_read.append(ops.container_alive)
            ops.calls.append("host_read")
            return real(root, is_ignored)

        with mock.patch.object(L, "snapshot_tree", spy):
            result, _ = self.run_loop(turns, ops=ops, base=base)
        return result, ops, alive_at_read

    def test_the_final_host_read_runs_after_the_container_is_gone(self):
        self.write("a.py", "x = 1\n")
        _, ops, alive = self._run_watching_the_read([_say("done")])
        self.assertEqual(
            ops.calls,
            [
                "create_worktree",
                "start_container",
                "destroy:start",
                "destroy:done",
                "host_read",
                "teardown_worktree",
            ],
        )
        self.assertFalse(alive[-1], "the container was still alive at the host read")

    def test_the_per_turn_hash_read_is_the_only_one_taken_while_it_lives(self):
        """The accepted exception, pinned so it cannot silently grow.

        §6.5 rule 3 governs the read that builds the RESULT. The no-progress
        hash necessarily runs mid-loop with the container alive; that is the
        one and only read that may.
        """
        self.write("a.py", "x = 1\n")
        _, _, alive = self._run_watching_the_read(
            [_call("list_files", {"path": "."}), _say("done")]
        )
        self.assertEqual(alive, [True, False])

    def test_the_blocking_host_reads_do_not_block_the_event_loop(self):
        """SF-2 / NF-3. `read_base_tree` runs one `git cat-file` subprocess per
        tracked blob — 9.4 ms per file measured — so a 5,000-file repo blocked
        the shared MCP event loop for ~47 s with nothing else able to respond,
        `coding_agent_result` included. `background=true`'s "poll for
        progress" promise does not survive that.

        Asserted by running a heartbeat coroutine CONCURRENTLY with the run
        and counting its ticks: a loop blocked inside a synchronous call
        cannot tick. The control below proves the heartbeat can be starved,
        so a non-zero count here is a measurement and not a formality.
        """
        self.write("a.py", "x = 1\n")
        for target in ("read_base_tree", "create_worktree", "snapshot_tree"):
            with self.subTest(blocking_call=target):
                ticks, calls = self._ticks_during_run(target, block=0.25)
                # Scaled by how many times the slowed call actually ran, so a
                # `snapshot_tree` whose PER-TURN uses went back on the loop
                # cannot hide behind the teardown one still yielding. The
                # heartbeat ticks 10x per `block`; 7 is slack for scheduling.
                self.assertGreaterEqual(calls, 1, "the slowed call never ran")
                self.assertGreaterEqual(
                    ticks,
                    7 * calls,
                    f"the event loop was blocked while {target} ran "
                    f"({calls} call(s), {ticks} tick(s))",
                )

    def test_MECHANISM_a_synchronous_host_read_really_would_starve_the_loop(self):
        """The control. Without it, "the heartbeat ticked" could just mean the
        heartbeat is cheap and the calls are fast."""
        self.write("a.py", "x = 1\n")
        ticks, _ = self._ticks_during_run(None, block=0.25)
        self.assertLessEqual(
            ticks, 1, "the heartbeat is not sensitive enough to prove anything"
        )

    def _ticks_during_run(self, target: str | None, *, block: float):
        """Run the loop with ONE host call slowed, counting heartbeat ticks.

        Returns (ticks, times the slowed call ran).

        `target=None` is the control: the same `time.sleep` is taken directly
        in the coroutine, the way every one of these calls used to be reached.
        Same wall time either way; only the loop's ability to tick differs.
        """
        ops = _RecordingOps(self.wt)
        base = _base_tree()
        real_snapshot = L.snapshot_tree
        real_create = ops.create_worktree
        calls = {"n": 0}

        def slow_read(repo, ref):
            calls["n"] += 1
            time.sleep(block)
            return base

        def slow_create(repo, ref):
            calls["n"] += 1
            time.sleep(block)
            return real_create(repo, ref)

        def slow_snapshot(root, is_ignored):
            calls["n"] += 1
            time.sleep(block)
            return real_snapshot(root, is_ignored)

        counter = {"n": 0}

        async def heartbeat():
            while True:
                await asyncio.sleep(block / 10)
                counter["n"] += 1

        async def scenario():
            beat = asyncio.get_running_loop().create_task(heartbeat())
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(
                            L,
                            "read_base_tree",
                            slow_read
                            if target == "read_base_tree"
                            else lambda *a: base,
                        )
                    )
                    if target == "create_worktree":
                        ops.create_worktree = slow_create  # type: ignore[method-assign]
                    if target == "snapshot_tree":
                        stack.enter_context(
                            mock.patch.object(L, "snapshot_tree", slow_snapshot)
                        )
                    if target is None:
                        calls["n"] += 1
                        time.sleep(block)  # the pre-fix shape, inline
                    await self._bare_run(ops)
            finally:
                beat.cancel()
            return counter["n"], calls["n"]

        return asyncio.run(scenario())

    async def _bare_run(self, ops):
        # Three turns, so `snapshot_tree` runs its PER-TURN reads as well as
        # the one that builds the diff — otherwise the teardown read alone
        # yields enough for the assertion and the per-turn hop goes unpinned.
        return await L.run_coding_agent(
            task="t",
            repo=self.repo,
            base_ref="HEAD",
            model="m",
            max_turns=3,
            max_seconds=60,
            image="img",
            chat=_scripted_chat([_call("list_files", {"path": "."}), _say("done")]),
            sandbox_factory=lambda: ops,
        )

    def test_the_result_carries_the_diff_the_post_destroy_read_produced(self):
        self.write("a.py", "x = 1\n")
        result, _, _ = self._run_watching_the_read([_say("done")])
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertEqual(result.changed_files, ["a.py"])
        self.assertIn("+x = 1", result.diff)
        self.assertEqual(result.cleanup_problems, [])
        self.assertIsNone(result.diff_full_path)
        self.assertFalse(result.diff_truncated)


# ---------------------------------------------------------------------------
# Cleanup: layered, shielded, bounded (§6 item 3, §6.6)
# ---------------------------------------------------------------------------


class CleanupIsUnconditional(_LoopCase):
    def test_a_failing_docker_rm_still_removes_the_worktree_and_is_reported(self):
        ops = _RecordingOps(self.wt, destroy_exc=OSError("daemon went away"))
        result, ops = self.run_loop([_say("done")], ops=ops)
        self.assertIn("teardown_worktree", ops.calls)
        self.assertEqual(len(result.cleanup_problems), 1)
        self.assertIn("docker rm -f failed", result.cleanup_problems[0])
        self.assertIn("diff may be incomplete", result.cleanup_problems[0])

    def test_an_UNCONFIRMED_docker_rm_is_caveated_on_the_result(self):
        """`docker rm -f` reports failure by EXIT CODE far more often than by
        raising, and the reported case is the dangerous one: §6.5 rule 3 says
        the container is dead before the host read, so a removal that only
        looked successful means the diff below it raced a live container. The
        human has to be told, or they review an under-show believing it whole.
        """
        ops = _RecordingOps(self.wt, destroy_report="docker rm -f exited 1")
        result, ops = self.run_loop([_say("done")], ops=ops)
        self.assertIn("destroy:unconfirmed", ops.calls)
        self.assertEqual(len(result.cleanup_problems), 1)
        self.assertIn("docker rm -f exited 1", result.cleanup_problems[0])
        self.assertIn("diff may be incomplete", result.cleanup_problems[0])
        # Still unconditional: the worktree goes regardless (§6.6).
        self.assertIn("teardown_worktree", ops.calls)

    def test_a_CONFIRMED_docker_rm_caveats_nothing(self):
        """The other half — otherwise the caveat above could be unconditional
        and this suite could not tell the difference."""
        result, ops = self.run_loop([_say("done")])
        self.assertIn("destroy:done", ops.calls)
        self.assertEqual(result.cleanup_problems, [])

    def test_a_failing_worktree_teardown_is_reported_not_raised(self):
        ops = _RecordingOps(self.wt, teardown_exc=OSError("EBUSY"))
        result, _ = self.run_loop([_say("done")], ops=ops)
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertTrue(
            any("worktree teardown failed" in p for p in result.cleanup_problems)
        )

    def test_residue_reported_by_the_sandbox_reaches_the_caller(self):
        ops = _RecordingOps(self.wt, teardown_problems=("worktree still registered",))
        result, _ = self.run_loop([_say("done")], ops=ops)
        self.assertEqual(result.cleanup_problems, ["worktree still registered"])

    def test_cleanup_that_blows_its_ceiling_still_removes_the_worktree(self):
        """A hung `docker rm -f` may not cost the `rm -rf`, and may not
        replace the caller's result with a TimeoutError."""
        ops = _RecordingOps(self.wt, destroy_delay=30.0)
        with mock.patch.object(L, "_CLEANUP_BOUND_S", 0.05):
            result, ops = self.run_loop([_say("done")], ops=ops)
        self.assertIn("teardown_worktree", ops.calls)
        self.assertNotIn("destroy:done", ops.calls)
        self.assertTrue(
            any("exceeded its" in p for p in result.cleanup_problems),
            result.cleanup_problems,
        )
        self.assertEqual(result.stop_reason, L.StopReason.completed)

    def test_an_exception_inside_the_loop_still_tears_down(self):
        async def boom(payload, timeout_s):
            raise RuntimeError("ollama exploded")

        result, ops = self.run_loop([], chat=boom)
        self.assertEqual(result.stop_reason, L.StopReason.error)
        self.assertIn("ollama exploded", result.transcript[0]["error"])
        self.assertIn("destroy:done", ops.calls)
        self.assertIn("teardown_worktree", ops.calls)

    def test_a_container_that_never_started_still_tears_down_the_worktree(self):
        ops = _RecordingOps(self.wt, start_exc=RuntimeError("docker run failed"))
        result, ops = self.run_loop([], ops=ops)
        self.assertEqual(result.stop_reason, L.StopReason.error)
        self.assertNotIn("destroy:start", ops.calls)
        self.assertIn("teardown_worktree", ops.calls)
        self.assertEqual(result.cleanup_problems, [])


class CancellationDoesNotSkipCleanup(_LoopCase):
    """`asyncio.shield` protects the teardown TASK, not its awaiter.

    A single `await shield(...)` plus `except CancelledError: pass` returns
    while the teardown is still in flight — an empty diff and an empty
    `cleanup_problems` for a run that changed files and left residue. So the
    shielded task is re-awaited to completion, and the cancellation is then
    re-raised rather than swallowed.
    """

    def test_cancel_during_teardown_completes_it_then_propagates(self):
        self.write("a.py", "x = 1\n")
        ops = _RecordingOps(self.wt, destroy_delay=0.25, teardown_problems=("residue",))
        seen: dict[str, object] = {}

        async def go():
            with mock.patch.object(L, "read_base_tree", return_value=_base_tree()):
                task = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=_scripted_chat([_say("done")]),
                        sandbox_factory=lambda: ops,
                    )
                )
                # let the loop finish and get INTO the shielded teardown
                while "destroy:start" not in ops.calls:
                    await asyncio.sleep(0.01)
                task.cancel()
                try:
                    seen["result"] = await task
                except asyncio.CancelledError:
                    seen["cancelled"] = True

        asyncio.run(go())
        self.assertTrue(seen.get("cancelled"), "cancellation was swallowed")
        self.assertNotIn("result", seen)
        self.assertEqual(
            ops.calls[-3:], ["destroy:start", "destroy:done", "teardown_worktree"]
        )
        self.assertFalse(L._SLOT.locked())

    def test_a_repeat_canceller_does_not_get_the_teardown_abandoned(self):
        """The re-await budget has to be measured in TIME, not attempts.

        A caller that cancels in a loop consumes one attempt per cancel, so a
        small attempt count gives up in tens of milliseconds and returns with
        the teardown still in flight — which is the whole failure the shield
        exists to prevent, just moved behind a bound.
        """
        ops = _RecordingOps(self.wt, destroy_delay=0.30)

        async def go():
            with mock.patch.object(L, "read_base_tree", return_value=_base_tree()):
                task = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=_scripted_chat([_say("done")]),
                        sandbox_factory=lambda: ops,
                    )
                )
                while "destroy:start" not in ops.calls:
                    await asyncio.sleep(0.005)

                async def pester():
                    for _ in range(100):
                        task.cancel()
                        await asyncio.sleep(0.01)

                nag = asyncio.create_task(pester())
                with self.assertRaises(asyncio.CancelledError):
                    await task
                nag.cancel()

        asyncio.run(go())
        self.assertIn("destroy:done", ops.calls)
        self.assertIn("teardown_worktree", ops.calls)

    def test_an_abandoned_teardown_still_removes_the_worktree_synchronously(self):
        """The one path where the async teardown really is given up on.

        Forced here by making the re-await deadline already past. The `rm -rf`
        layer is sync and idempotent, so it is run inline rather than left to
        a task that may never be scheduled again — a leaked worktree of a repo
        an untrusted model wrote into is not an acceptable outcome (§9).
        """
        ops = _RecordingOps(self.wt, destroy_delay=5.0)
        at_return: list[str] = []

        async def go():
            with (
                mock.patch.object(L, "_CLEANUP_GRACE_S", -1000.0),
                mock.patch.object(L, "read_base_tree", return_value=_base_tree()),
            ):
                task = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=_scripted_chat([_say("done")]),
                        sandbox_factory=lambda: ops,
                    )
                )
                while "destroy:start" not in ops.calls:
                    await asyncio.sleep(0.005)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                # Snapshot INSIDE the loop. Asserting on ops.calls after
                # asyncio.run() returns would pass with no fallback at all:
                # shutting the loop down cancels the abandoned teardown task,
                # whose own `finally` then removes the worktree. That is the
                # interpreter tidying up, not this function keeping a promise.
                at_return.extend(ops.calls)

        asyncio.run(go())
        self.assertNotIn("destroy:done", at_return)  # genuinely abandoned
        self.assertIn("teardown_worktree", at_return)

    def test_cancel_before_teardown_still_tears_down(self):
        ops = _RecordingOps(self.wt)

        async def slow_chat(payload, timeout_s):
            await asyncio.sleep(5)
            return {"message": _say("done")}

        async def go():
            with mock.patch.object(L, "read_base_tree", return_value=_base_tree()):
                task = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=slow_chat,
                        sandbox_factory=lambda: ops,
                    )
                )
                while "start_container" not in ops.calls:
                    await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(go())
        self.assertIn("destroy:done", ops.calls)
        self.assertIn("teardown_worktree", ops.calls)


# ---------------------------------------------------------------------------
# The single slot (§6 item 4)
# ---------------------------------------------------------------------------


class SingleSlot(_LoopCase):
    def test_a_concurrent_second_call_is_rejected_while_the_first_still_runs(self):
        """Rejected, NOT queued. The proof that it is not queued: the first
        run is still in flight at the moment the second one gives up, and the
        second never created a worktree."""
        first_ops = _RecordingOps(self.wt)
        second_ops = _RecordingOps(self.wt)
        timing: dict[str, object] = {}

        async def go():
            entered = asyncio.Event()
            release = asyncio.Event()

            async def held_chat(payload, timeout_s):
                entered.set()
                await release.wait()
                return {"message": _say("done")}

            with mock.patch.object(L, "read_base_tree", return_value=_base_tree()):
                first = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=held_chat,
                        sandbox_factory=lambda: first_ops,
                    )
                )
                await entered.wait()
                started = time.monotonic()
                # wait_for is the deadlock guard, not the assertion: a
                # QUEUEING implementation would block here until the first run
                # released the slot, which this test deliberately never does
                # until afterwards. It then surfaces as TimeoutError, which is
                # not a RuntimeError, so the test fails instead of hanging.
                with self.assertRaises(RuntimeError) as caught:
                    await asyncio.wait_for(
                        L.run_coding_agent(
                            task="t2",
                            repo=self.repo,
                            base_ref="HEAD",
                            model="m",
                            max_turns=5,
                            max_seconds=60,
                            image="img",
                            chat=_scripted_chat([_say("done")]),
                            sandbox_factory=lambda: second_ops,
                        ),
                        timeout=2.0,
                    )
                timing["waited"] = time.monotonic() - started
                timing["first_done"] = first.done()
                timing["message"] = str(caught.exception)
                release.set()
                await first

        asyncio.run(go())
        self.assertFalse(timing["first_done"], "the second call waited for the first")
        self.assertLess(timing["waited"], 0.5)
        self.assertIn("already in progress", timing["message"])
        self.assertEqual(second_ops.calls, [], "the rejected call created something")

    def test_the_slot_is_free_again_after_the_first_run_finishes(self):
        self.run_loop([_say("one")])
        self.assertFalse(L._SLOT.locked())
        result, _ = self.run_loop([_say("two")])
        self.assertEqual(result.stop_reason, L.StopReason.completed)

    def test_the_slot_is_released_on_every_exit_path(self):
        async def boom(payload, timeout_s):
            raise RuntimeError("boom")

        paths = {
            "completed": {"turns": [_say("done")]},
            "loop raised": {"turns": [], "chat": boom},
            "start_container raised": {
                "turns": [],
                "ops": _RecordingOps(self.wt, start_exc=RuntimeError("no docker")),
            },
            "destroy raised": {
                "turns": [_say("done")],
                "ops": _RecordingOps(self.wt, destroy_exc=OSError("gone")),
            },
            "teardown raised": {
                "turns": [_say("done")],
                "ops": _RecordingOps(self.wt, teardown_exc=OSError("EBUSY")),
            },
        }
        for label, kwargs in paths.items():
            with self.subTest(path=label):
                self.run_loop(kwargs.pop("turns"), **kwargs)
                self.assertFalse(L._SLOT.locked())

    def test_the_slot_is_released_after_a_cancellation(self):
        ops = _RecordingOps(self.wt)

        async def slow_chat(payload, timeout_s):
            await asyncio.sleep(5)
            return {"message": _say("done")}

        async def go():
            with mock.patch.object(L, "read_base_tree", return_value=_base_tree()):
                task = asyncio.create_task(
                    L.run_coding_agent(
                        task="t",
                        repo=self.repo,
                        base_ref="HEAD",
                        model="m",
                        max_turns=5,
                        max_seconds=60,
                        image="img",
                        chat=slow_chat,
                        sandbox_factory=lambda: ops,
                    )
                )
                while "start_container" not in ops.calls:
                    await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(go())
        self.assertFalse(L._SLOT.locked())

    def test_rejection_happens_before_the_base_tree_is_even_read(self):
        reads: list[str] = []

        async def go():
            async with L._SLOT:
                with mock.patch.object(
                    L, "read_base_tree", side_effect=lambda r, b: reads.append(r)
                ):
                    with self.assertRaises(RuntimeError):
                        await L.run_coding_agent(
                            task="t",
                            repo=self.repo,
                            base_ref="HEAD",
                            model="m",
                            max_turns=1,
                            max_seconds=5,
                            image="img",
                            chat=_scripted_chat([]),
                            sandbox_factory=lambda: _RecordingOps(self.wt),
                        )

        asyncio.run(go())
        self.assertEqual(reads, [])


# ---------------------------------------------------------------------------
# Stop conditions (§7)
# ---------------------------------------------------------------------------


class StopConditions(_LoopCase):
    def test_completed_when_the_model_stops_calling_tools(self):
        result, _ = self.run_loop([_say("all done")])
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertEqual(result.turns, 1)

    def test_max_turns(self):
        turns = [_call("list_files", {})] * 20
        result, _ = self.run_loop(turns, max_turns=3)
        self.assertEqual(result.stop_reason, L.StopReason.max_turns)
        self.assertEqual(result.turns, 3)

    def test_max_seconds_is_checked_before_the_first_model_call(self):
        result, _ = self.run_loop([_call("list_files", {})] * 20, max_seconds=0.0)
        self.assertEqual(result.stop_reason, L.StopReason.max_seconds)
        self.assertEqual(result.turns, 0)

    def test_max_seconds_stops_a_run_already_under_way(self):
        async def slow(payload, timeout_s):
            await asyncio.sleep(0.12)
            return {"message": _call("list_files", {})}

        result, _ = self.run_loop([], chat=slow, max_seconds=0.25, max_turns=50)
        self.assertEqual(result.stop_reason, L.StopReason.max_seconds)
        self.assertGreaterEqual(result.turns, 1)
        self.assertLess(result.turns, 50)

    def test_the_model_call_gets_a_budget_clamped_timeout_not_a_flat_300s(self):
        """SF-1 / NF-2. `max_seconds` was documented as "a hard wall-clock cap"
        in both the README and the tool schema, and was not one.

        `run_command`'s timeout was clamped by `budget.remaining()`; the MODEL
        call was handed `_CHAT_TIMEOUT_S = 300.0` unconditionally, after the
        budget check had already passed. The check at the top of a turn only
        decides whether to start one more turn — it says nothing about what
        that turn may then spend. Measured worst case for a single turn:
        `300 + 32 x 300`.
        """
        seen: list[float] = []

        # Chosen to sit BETWEEN the floor and the ceiling, so the clamp is the
        # thing being observed rather than the floor: a `max_seconds` under
        # `_MIN_CMD_TIMEOUT_S` would hand out 5.0 every time and prove nothing.
        budget_s = 30.0
        self.assertLess(L._MIN_CMD_TIMEOUT_S, budget_s)
        self.assertLess(budget_s, L._CHAT_TIMEOUT_S)

        async def recording(payload, timeout_s):
            seen.append(timeout_s)
            await asyncio.sleep(0.02)
            return {"message": _call("list_files", {})}

        self.run_loop([], chat=recording, max_seconds=budget_s, max_turns=3)

        self.assertEqual(len(seen), 3)
        self.assertLess(
            max(seen),
            L._CHAT_TIMEOUT_S,
            "the model call was handed the flat ceiling, not the remaining budget",
        )
        self.assertLessEqual(max(seen), budget_s)
        # ...and never below the floor, or an expiring budget would time the
        # last call out instead of stopping cleanly on max_seconds.
        self.assertGreaterEqual(min(seen), L._MIN_CMD_TIMEOUT_S)
        # The clamp tightens as the clock runs down.
        self.assertLess(seen[-1], seen[0])

    def test_a_generous_budget_leaves_the_model_call_at_its_full_ceiling(self):
        """The other side: the clamp must not shorten a call that has plenty
        of budget left, or it becomes a latency bug dressed as a safety fix."""
        seen: list[float] = []

        async def recording(payload, timeout_s):
            seen.append(timeout_s)
            return {"message": _say("done")}

        self.run_loop([], chat=recording, max_seconds=3600)
        self.assertEqual(seen, [L._CHAT_TIMEOUT_S])

    def test_no_progress(self):
        """Five read-only turns that change nothing. The default N is 5, so
        turn 5 is the one that trips it."""
        result, _ = self.run_loop([_call("list_files", {})] * 20, max_turns=50)
        self.assertEqual(result.stop_reason, L.StopReason.no_progress)
        self.assertEqual(result.turns, 5)

    def test_a_response_without_a_message_is_an_error_not_completed(self):
        """A broken endpoint must not read as "the model says it is done"."""

        async def empty(payload, timeout_s):
            return {"model": "m", "done": True}

        result, _ = self.run_loop([], chat=empty)
        self.assertEqual(result.stop_reason, L.StopReason.error)
        self.assertIn("no message", result.transcript[0]["error"])

    def test_a_failed_model_call_stops_with_error_and_keeps_the_transcript(self):
        async def failing(payload, timeout_s):
            return {"status": "failed", "error": "Ollama not running at http://x"}

        result, _ = self.run_loop([], chat=failing)
        self.assertEqual(result.stop_reason, L.StopReason.error)
        self.assertIn("Ollama not running", result.transcript[0]["error"])

    def test_the_result_survives_the_serialization_task_9_will_apply(self):
        """`mcp_server` returns `json.dumps(res.__dict__, default=str)`. Pin
        the field set and its serializability here, where it is cheap to fix,
        rather than in the MCP dispatch."""
        self.write("a.py", "x = 1\n")
        result, _ = self.run_loop([_say("done")])
        payload = json.loads(json.dumps(result.__dict__, default=str))
        self.assertEqual(
            sorted(payload),
            [
                "changed_files",
                "cleanup_problems",
                "diff",
                "diff_full_path",
                "diff_truncated",
                "elapsed_seconds",
                "last_command",
                "model",
                "stop_reason",
                "transcript",
                "turns",
                "unreadable",
            ],
        )
        self.assertEqual(payload["stop_reason"], "completed")
        self.assertEqual(payload["model"], "m")

    def test_elapsed_seconds_measures_the_loop_not_the_teardown(self):
        """Bounded on BOTH sides, and the lower bound is the point.

        The upper bound alone was decorative: deleting the `elapsed = round(
        time.monotonic() - started, 2)` assignment outright, so the field kept
        its `0.0` initialiser, left the whole loop+mcp suite green. A constant
        zero satisfies "less than the teardown delay" perfectly. The chat
        callable is therefore made to take a measurable amount of time, so the
        assertion can tell a measurement from an initialiser.
        """
        # The field is rounded to ONE decimal, so the floor asserted below is
        # deliberately looser than the sleep: 0.15 s cannot round to less than
        # 0.1, and it cannot round to 0.0 at all.
        ops = _RecordingOps(self.wt, destroy_delay=0.30)

        async def slow_chat(payload, timeout_s):
            await asyncio.sleep(0.15)
            return {"message": _say("done")}

        result, _ = self.run_loop([], ops=ops, chat=slow_chat)
        self.assertGreaterEqual(
            result.elapsed_seconds,
            0.1,
            "elapsed_seconds is not measuring the loop",
        )
        # ...and still excludes the 0.30 s teardown that ran after it.
        self.assertLess(result.elapsed_seconds, 0.30)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class ToolDispatch(_LoopCase):
    def test_list_files_is_given_the_tracked_aware_ignore_predicate(self):
        """Without it the tool's own advertised "respects .gitignore" is a
        lie, and the model burns turns on build junk."""
        self.write(".gitignore", "build/\n")
        self.write("src/a.py", "x = 1\n")
        self.write("build/ARTIFACT.bin", "junk\n")
        result, _ = self.run_loop(
            [_call("list_files", {"path": "."}), _say("done")],
            base=_base_tree({".gitignore": "build/\n"}),
        )
        listing = result.transcript[0]["result_head"]
        self.assertIn("src/a.py", listing)
        self.assertNotIn("build/ARTIFACT.bin", listing)

    def test_read_and_write_round_trip_through_the_worktree(self):
        self.write("a.py", "x = 1\n")
        result, _ = self.run_loop(
            [
                _call("read_file", {"path": "a.py"}),
                _call("write_file", {"path": "a.py", "content": "x = 2\n"}),
                _say("done"),
            ]
        )
        self.assertIn("x = 1", result.transcript[0]["result_head"])
        self.assertEqual(Path(self.wt, "a.py").read_text(), "x = 2\n")

    def test_run_command_records_the_last_command(self):
        async def fake_exec(container, cmd, *, timeout_s):
            return 1, "3 failed\n", False

        with mock.patch("coding_agent.tools.exec_in_container", fake_exec):
            result, _ = self.run_loop(
                [_call("run_command", {"cmd": "pytest -q"}), _say("done")]
            )
        self.assertEqual(
            result.last_command,
            {"cmd": "pytest -q", "exit": 1, "output_tail": "3 failed\n"},
        )
        self.assertIn("[exit 1]", result.transcript[0]["result_head"])

    def test_the_command_timeout_is_clamped_by_what_is_left_of_max_seconds(self):
        seen: list[float] = []

        async def fake_exec(container, cmd, *, timeout_s):
            seen.append(timeout_s)
            return 0, "", False

        with mock.patch("coding_agent.tools.exec_in_container", fake_exec):
            self.run_loop(
                [_call("run_command", {"cmd": "sleep 100000"}), _say("done")],
                max_seconds=20,
            )
        self.assertLessEqual(seen[0], 20.0)
        self.assertGreater(seen[0], 0.0)

    def test_an_unknown_tool_name_is_fed_back_not_fatal(self):
        result, _ = self.run_loop([_call("rm_rf", {"path": "/"}), _say("done")])
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertIn("unknown tool", result.transcript[0]["result_head"])


class MalformedToolCallsAreFedBack(_LoopCase):
    """Spec §9: "Model returns malformed tool calls -> fed back as a tool
    error result". Every one of these otherwise raises AttributeError or
    TypeError out of the dispatch and ends the whole run as `error`."""

    def _one_bad_turn(self, message):
        return self.run_loop([message, _say("done")])

    def test_arguments_as_a_list(self):
        result, _ = self._one_bad_turn(_call("read_file", ["a.py"]))
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertIn("needs a string 'path'", result.transcript[0]["result_head"])

    def test_arguments_as_a_json_string(self):
        self.write("a.py", "x = 1\n")
        result, _ = self._one_bad_turn(_call("read_file", '{"path": "a.py"}'))
        self.assertIn("x = 1", result.transcript[0]["result_head"])

    def test_arguments_as_unparseable_text(self):
        result, _ = self._one_bad_turn(_call("read_file", "not json at all"))
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertIn("needs a string 'path'", result.transcript[0]["result_head"])

    def test_write_file_with_non_string_content(self):
        result, _ = self._one_bad_turn(
            _call("write_file", {"path": "a.py", "content": 5})
        )
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertIn("needs a string 'content'", result.transcript[0]["result_head"])

    def test_a_tool_call_that_is_not_even_a_dict(self):
        message = {"role": "assistant", "content": "", "tool_calls": ["read_file"]}
        result, _ = self._one_bad_turn(message)
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertIn("unknown tool None", result.transcript[0]["result_head"])

    def test_tool_calls_that_is_not_a_list_reads_as_a_final_message(self):
        message = {"role": "assistant", "content": "done", "tool_calls": "nope"}
        result, _ = self.run_loop([message])
        self.assertEqual(result.stop_reason, L.StopReason.completed)

    def test_run_command_without_a_cmd(self):
        result, _ = self._one_bad_turn(_call("run_command", {}))
        self.assertIn("needs a string 'cmd'", result.transcript[0]["result_head"])


class TranscriptIsBounded(_LoopCase):
    """§5.1: "the diff and transcript are size-capped". The model chooses the
    arguments, so the arguments are as unbounded as it likes."""

    def test_a_huge_write_file_argument_does_not_land_in_the_transcript(self):
        payload = "A" * 1_000_000
        result, _ = self.run_loop(
            [_call("write_file", {"path": "big.txt", "content": payload}), _say("done")]
        )
        rendered = result.transcript[0]["args"]["content"]
        self.assertLess(len(rendered), 400)
        self.assertIn("[1000000 chars]", rendered)
        # the file itself was still written in full
        self.assertEqual(len(Path(self.wt, "big.txt").read_text()), 1_000_000)

    def test_a_turn_may_not_queue_unbounded_tool_calls(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}] * 200,
        }
        result, _ = self.run_loop([message, _say("done")])
        executed = [e for e in result.transcript if e.get("tool") == "list_files"]
        self.assertEqual(len(executed), L._MAX_TOOL_CALLS_PER_TURN)
        self.assertIn("were dropped", result.transcript[-1]["result_head"])

    def test_the_loop_ships_its_own_diff_cap_and_spills_outside_the_worktree(self):
        """SF-8. The size-cap MECHANISM is tested in `walk` with an explicit
        `max_bytes`; the VALUES the loop actually passes were unpinned —
        `_DIFF_MAX_BYTES` could be deleted and nothing went red.

        `spill_dir` is the more serious half. `unified_diff`'s docstring says
        it "must be a host directory the sandbox cannot reach", and repointing
        it at the WORKTREE left the suite green — after which teardown
        `rm -rf`s the spill and `diff_full_path` names a file that no longer
        exists, so the human is handed a path to nothing precisely when the
        diff was too big to read inline.
        """
        # Comfortably over the cap, and text so it is not binary-annotated.
        self.write("huge.txt", "line of source\n" * 60_000)

        result, _ = self.run_loop([_say("done")])

        self.assertTrue(result.diff_truncated)
        self.assertLessEqual(
            len(result.diff.encode("utf-8", "surrogateescape")),
            L._DIFF_MAX_BYTES + 4096,  # slack for the marker + changed block
        )
        self.assertIsNotNone(result.diff_full_path)
        spill = result.diff_full_path or ""
        # The spill survives teardown, which is the whole point of it being
        # outside the worktree.
        self.assertTrue(os.path.exists(spill), "the spill file was torn down with it")
        self.addCleanup(lambda: os.path.exists(spill) and os.unlink(spill))
        self.assertFalse(
            os.path.realpath(spill).startswith(os.path.realpath(self.wt) + os.sep),
            f"the full diff was spilled INSIDE the sandbox's worktree: {spill}",
        )
        self.assertTrue(
            os.path.realpath(spill).startswith(
                os.path.realpath(tempfile.gettempdir()) + os.sep
            ),
            spill,
        )

    def test_the_loop_ships_its_own_transcript_and_output_caps(self):
        """`_RESULT_HEAD` and `_OUTPUT_TAIL` had the same gap as
        `_DIFF_MAX_BYTES`: the mechanism was tested, the shipped value was
        not."""
        long_output = "y" * 50_000

        async def fake_exec(container, cmd, *, timeout_s):
            return 0, long_output, False

        with mock.patch("coding_agent.tools.exec_in_container", fake_exec):
            result, _ = self.run_loop(
                [_call("run_command", {"cmd": "yes"}), _say("done")]
            )

        entry = next(e for e in result.transcript if e.get("tool") == "run_command")
        self.assertLessEqual(len(entry["result_head"]), L._RESULT_HEAD)
        self.assertIsNotNone(result.last_command)
        assert result.last_command is not None
        self.assertEqual(len(result.last_command["output_tail"]), L._OUTPUT_TAIL)
        # The TAIL, not the head — the exit status and the last error a
        # command printed are what a reviewer needs.
        self.assertTrue(long_output.endswith(result.last_command["output_tail"]))

    def test_the_in_flight_conversation_is_bounded_across_turns(self):
        """SF-4 / NF-8. The RETURNED transcript was bounded; the conversation
        SENT to the model was not. Every tool result was appended in full and
        the whole list re-serialised on every turn, so the ceiling was
        `60 turns x 32 calls x MAX_READ` ~= 490 MB, re-encoded each turn.

        Asserted on the PAYLOAD the chat callable actually receives, not on
        the list afterwards — the bound is only worth anything if it applies
        to what goes over the wire.
        """
        big = "B" * (600 * 1024)
        self.write("big.txt", big)
        sizes: list[int] = []

        async def watching(payload, timeout_s):
            n = len(sizes)
            sizes.append(
                sum(len(str(m.get("content") or "")) for m in payload["messages"])
            )
            if n >= 8:
                return {"message": _say("done")}
            # read_file grows the conversation by MAX_READ each turn;
            # write_file keeps the no-progress detector satisfied so the run
            # reaches enough turns to cross the bound.
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "big.txt"},
                            }
                        },
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": f"t{n}.txt", "content": str(n)},
                            }
                        },
                    ],
                }
            }

        result, _ = self.run_loop([], chat=watching, max_turns=12)

        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertGreater(len(sizes), 3)
        self.assertLessEqual(
            max(sizes),
            L._MAX_CONVERSATION_CHARS + len(big),
            "the payload grew without bound across turns",
        )
        # MECHANISM: the conversation really did exceed the bound, so the
        # trimming was exercised rather than never reached.
        self.assertGreater(sum(sizes), L._MAX_CONVERSATION_CHARS)

    def test_trimming_shrinks_only_tool_content_never_the_message_structure(self):
        """The assistant/tool pairing is protocol. Dropping one side of a
        `tool_calls` pair is how a memory fix becomes a backend rejecting the
        payload, so this changes bytes only — never the count, the roles, or
        the tool_calls."""
        # Each protected message is deliberately LARGER than
        # `_ELIDED_TOOL_CHARS`, so it is the ROLE check that spares it and not
        # the "already small enough" shortcut — otherwise dropping the role
        # check would leave this test green.
        big = 50_000
        self.assertGreater(big, L._ELIDED_TOOL_CHARS)
        messages = [
            {"role": "system", "content": "S" * big},
            {"role": "user", "content": "U" * big},
            {
                "role": "assistant",
                "content": "A" * big,
                "tool_calls": [{"function": {}}],
            },
            {"role": "tool", "content": "T" * (L._MAX_CONVERSATION_CHARS + 5000)},
            {"role": "tool", "content": "recent"},
        ]
        before = [(m["role"], "tool_calls" in m) for m in messages]

        elided = L._trim_conversation(messages)

        self.assertEqual(elided, 1)
        self.assertEqual([(m["role"], "tool_calls" in m) for m in messages], before)
        self.assertEqual(messages[0]["content"], "S" * big)  # system untouched
        self.assertEqual(messages[1]["content"], "U" * big)  # task untouched
        self.assertEqual(messages[2]["content"], "A" * big)  # tool_calls turn intact
        self.assertEqual(messages[4]["content"], "recent")  # newest untouched
        self.assertIn("chars elided", messages[3]["content"])
        self.assertLess(len(messages[3]["content"]), L._ELIDED_TOOL_CHARS + 200)

    def test_a_conversation_under_the_bound_is_left_alone(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "tool", "content": "T" * 50_000},
        ]
        self.assertEqual(L._trim_conversation(messages), 0)
        self.assertEqual(len(messages[1]["content"]), 50_000)

    def test_a_large_tool_call_argument_counts_toward_the_bound_and_gets_elided(self):
        """The original ceiling analysis (`_MAX_CONVERSATION_CHARS`'s own
        comment) sized the bound off tool RESULTS only. A `write_file` call
        embeds its whole `content` argument as JSON in the assistant
        message's `tool_calls`, which `content` never carries — so a model
        that writes large files could grow the payload past the bound
        without `_trim_conversation` ever noticing, let alone shrinking it."""
        big_arg = "X" * (L._MAX_CONVERSATION_CHARS + 5000)
        messages = [
            {"role": "system", "content": "S"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": big_arg},
                    }
                ],
            },
            {"role": "tool", "content": "recent"},
        ]

        elided = L._trim_conversation(messages)

        self.assertEqual(elided, 1)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(len(messages[1]["tool_calls"]), 1)  # call itself intact
        call = messages[1]["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")  # id/type/name untouched
        self.assertEqual(call["function"]["name"], "write_file")
        shrunk = call["function"]["arguments"]
        self.assertLess(len(shrunk), len(big_arg))
        self.assertIn("chars elided", shrunk)
        self.assertEqual(messages[2]["content"], "recent")  # newest untouched

    def test_an_object_valued_tool_call_argument_stays_an_object_when_elided(self):
        """The sibling test above passes `arguments` as a STRING, which is the
        OpenAI shape. Ollama — the only server this tool actually talks to —
        sends an OBJECT (`_as_dict`'s own docstring says so), and that path was
        untested: the trim rendered the dict with `str()` and assigned the
        truncated result back, turning `arguments` into a Python repr. That is
        a shape change `_trim_conversation` promises never to make, and the
        repr does not survive a re-parse, so `_as_dict` would recover `{}` —
        every argument silently lost. Pins the object shape, the surviving
        keys, and that the value is still JSON."""
        big = "X" * (L._MAX_CONVERSATION_CHARS + 5000)
        messages = [
            {"role": "system", "content": "S"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            # Ollama's real shape: a dict, not a JSON string.
                            "arguments": {"path": "a.py", "content": big, "mode": "w"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "recent"},
        ]

        elided = L._trim_conversation(messages)

        self.assertEqual(elided, 1)
        shrunk = messages[1]["tool_calls"][0]["function"]["arguments"]
        # The shape guarantee: still an object, not a stringified repr.
        self.assertIsInstance(shrunk, dict)
        self.assertEqual(set(shrunk), {"path", "content", "mode"})
        self.assertEqual(shrunk["path"], "a.py")  # untouched keys survive whole
        self.assertEqual(shrunk["mode"], "w")  # including non-cut ones
        # The cut lands INSIDE the largest value, and announces itself.
        self.assertLess(len(shrunk["content"]), len(big))
        self.assertIn("chars elided", shrunk["content"])
        # Still round-trips, so `_as_dict` recovers every argument.
        self.assertEqual(json.loads(json.dumps(shrunk))["path"], "a.py")
        self.assertEqual(L._as_dict(shrunk)["path"], "a.py")
        # And the bound it exists to enforce was actually reached.
        total = sum(
            len(str(m.get("content") or "")) + L._tool_call_arg_chars(m)
            for m in messages
        )
        self.assertLessEqual(total, L._MAX_CONVERSATION_CHARS)

    def test_a_list_valued_tool_call_argument_keeps_its_type_too(self):
        """The third shape, and the one the object fix above did not reach.
        `_as_dict`'s docstring names a JSON string, an OBJECT, a LIST and a
        bare scalar as things a server can send. String and object each got a
        type-preserving path; a list fell into the old `else` and came back a
        `str` — the same shape change, one branch over.

        Left whole now. That costs bytes on a call that is already unusable
        (`_as_dict` yields `{}` for a list), and buys an unconditional type
        guarantee, which is the only kind worth documenting.
        """
        big = ["Y" * (L._MAX_CONVERSATION_CHARS + 5000)]
        messages = [
            {"role": "system", "content": "S"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": big},
                    }
                ],
            },
            {"role": "tool", "content": "recent"},
        ]

        L._trim_conversation(messages)

        kept = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(kept, list)  # NOT a str, which is the whole point
        self.assertEqual(kept, big)
        self.assertNotIn("chars elided", json.dumps(kept))

    def test_a_scalar_valued_tool_call_argument_keeps_its_type_too(self):
        """The fourth shape `_as_dict` names. Bounded by construction — a
        scalar cannot be oversized — so this pins the branch, not a cut."""
        messages = [
            {"role": "system", "content": "S" * (L._MAX_CONVERSATION_CHARS + 5000)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": 12345},
                    }
                ],
            },
        ]

        L._trim_conversation(messages)

        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], 12345)

    def test_changed_files_and_unreadable_are_bounded_with_an_exact_count(self):
        """SF-3 / NF-4. Every other model-controlled field was bounded; these
        two were carried verbatim, and both are as long as the sandbox likes.
        Measured at 200,000 unreadable paths: the diff TEXT stays correctly at
        1,220 bytes while the result JSON reaches 11.8 MB and lands in the
        caller's context. The accepted breadth-DoS trade-off was reasoned
        about as a HOST MEMORY bound; this is the half that reaches a human.
        """
        n = L._RESULT_PATHS * 2
        for i in range(n):
            self.write(f"f{i:05d}.py", f"x = {i}\n")

        result, _ = self.run_loop([_say("done")])

        self.assertEqual(len(result.changed_files), L._RESULT_PATHS + 1)
        overflow = result.changed_files[-1]
        self.assertIn(f"and {n - L._RESULT_PATHS} more", overflow)
        # The marker used to read "the diff text and its spill file are
        # complete". In THIS state there is no spill file at all — 1000 tiny
        # files do not reach the 512 KB cap — so the old wording named an
        # artifact that does not exist, and asserted completeness for a diff
        # text that had been truncated in the states where one does. Both
        # halves are now stated from the run's actual state (F-2 / H-3).
        self.assertFalse(result.diff_truncated)
        self.assertIsNone(result.diff_full_path)
        self.assertIn("names every changed path", overflow)
        self.assertNotIn("spill", overflow)
        # The kept entries are still real paths, in order, not a summary.
        self.assertEqual(result.changed_files[0], "f00000.py")

    def test_the_unreadable_overflow_marker_keeps_the_field_s_own_shape(self):
        """`unreadable` is a list of dicts a gate reads programmatically, so
        the marker must be a dict too — a bare string here would make a
        consumer that fails closed crash instead."""
        marker = L._capped_paths(
            [{"path": f"p{i}", "reason": "EACCES"} for i in range(L._RESULT_PATHS + 3)]
        )[-1]
        self.assertIsInstance(marker, dict)
        self.assertEqual(marker["reason"], "TRUNCATED")
        self.assertIn("and 3 more", marker["path"])

    def test_a_list_under_the_bound_is_returned_untouched(self):
        """No marker, no copy semantics change, on the ordinary path."""
        small = [{"path": "a", "reason": "EACCES"}]
        self.assertIs(L._capped_paths(small), small)
        self.assertEqual(L._capped_paths([]), [])


class SecuritySignalsReachTheHuman(_LoopCase):
    def test_a_path_escape_is_flagged_in_the_transcript_and_does_not_kill_the_run(self):
        result, ops = self.run_loop(
            [_call("read_file", {"path": "../../../etc/passwd"}), _say("done")]
        )
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        self.assertTrue(result.transcript[0]["path_escape"])
        self.assertIn("PathEscape", result.transcript[0]["result_head"])
        self.assertIn("teardown_worktree", ops.calls)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file anyway")
    def test_an_unreadable_path_is_carried_into_the_result(self):
        """A `chmod 000` file used to vanish from the diff in silence. The
        walk records it; the diff surfaces it; the RESULT must carry it, or
        the human never learns a path was hidden."""
        self.write("hidden.py", "backdoor = True\n")
        os.chmod(Path(self.wt, "hidden.py"), 0o000)
        self.addCleanup(
            os.chmod, Path(self.wt, "hidden.py"), stat.S_IRUSR | stat.S_IWUSR
        )
        result, _ = self.run_loop([_say("done")])
        self.assertEqual(result.unreadable, [{"path": "hidden.py", "reason": "EACCES"}])
        self.assertIn("hidden.py", result.diff)


# ---------------------------------------------------------------------------
# The same ordering claim, measured from OUTSIDE the process
# ---------------------------------------------------------------------------

_TEST_IMAGE = os.environ.get("AI_TOOLS_CODING_AGENT_TEST_IMAGE", "alpine:3")


def _docker_ready() -> str | None:
    """A skip reason, or None. Never pulls: a test suite must not reach the
    network."""
    try:
        probe = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "docker CLI not available"
    if probe.returncode != 0:
        return "no docker daemon"
    present = subprocess.run(
        ["docker", "image", "inspect", _TEST_IMAGE],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if present.returncode != 0:
        return f"image {_TEST_IMAGE} not present locally"
    return None


@unittest.skipIf(_docker_ready() is not None, _docker_ready() or "")
class ContainerGoneBeforeHostRead(unittest.TestCase):
    """The ordering claim, checked by a SEPARATE PROCESS against a real daemon.

    `docker ps` is asked, at the instant of each host read, whether the
    container id still exists. The mid-loop read is the control: if it did not
    see a live container the probe would be measuring nothing, and the
    post-destroy observation would be worthless.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp(prefix="ca-loop-realrepo-")
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        for argv in (
            ["init", "-q", "-b", "main"],
            ["config", "user.email", "t@example.com"],
            ["config", "user.name", "T"],
        ):
            subprocess.run(["git", "-C", self.repo, *argv], check=True)
        Path(self.repo, "calc.py").write_text("def add(a, b):\n    return a - b\n")
        subprocess.run(["git", "-C", self.repo, "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", self.repo, "commit", "-q", "-m", "base"], check=True
        )

    def test_docker_ps_sees_nothing_at_the_read_that_builds_the_diff(self):
        sightings: list[str] = []
        container: list[str] = []
        real_snapshot = L.snapshot_tree

        class RealOpsRecordingTheId:
            """The REAL sandbox module; only the container id is intercepted."""

            create_worktree = staticmethod(L._sb.create_worktree)
            destroy_container = staticmethod(L._sb.destroy_container)
            teardown_worktree = staticmethod(L._sb.teardown_worktree)

            @staticmethod
            async def start_container(worktree, image, **kw):
                cid = await L._sb.start_container(worktree, image, **kw)
                container.append(cid)
                return cid

        def probe(root, is_ignored):
            found = subprocess.run(
                ["docker", "ps", "-q", "--no-trunc", "--filter", f"id={container[0]}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.strip()
            sightings.append(found)
            return real_snapshot(root, is_ignored)

        ops = RealOpsRecordingTheId()
        with mock.patch.object(L, "snapshot_tree", probe):
            result = asyncio.run(
                L.run_coding_agent(
                    task="t",
                    repo=self.repo,
                    base_ref="HEAD",
                    model="m",
                    max_turns=5,
                    max_seconds=120,
                    image=_TEST_IMAGE,
                    chat=_scripted_chat(
                        [_call("list_files", {"path": "."}), _say("done")]
                    ),
                    sandbox_factory=lambda: ops,
                )
            )
        self.assertEqual(len(sightings), 2, sightings)
        # control: the probe CAN see a live container
        self.assertEqual(sightings[0], container[0])
        # the claim: it is gone before the read that builds the result
        self.assertEqual(sightings[1], "")
        self.assertEqual(result.cleanup_problems, [])
        self.assertEqual(result.stop_reason, L.StopReason.completed)
        # and nothing leaked
        listing = subprocess.run(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
        self.assertNotIn(container[0], listing)


if __name__ == "__main__":
    unittest.main()
