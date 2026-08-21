#!/usr/bin/env python3
"""Docker-required integration tests for coding_agent. Skipped when the daemon
is down or the sandbox image is not built (never pulled: a test suite must not
reach the network).

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_integration.py -q

Everything here drives the REAL loop against a REAL git repo, a REAL throwaway
worktree and a REAL container built from the real sandbox image. Only the model
is fake — `chat` is a scripted callable, so no Ollama is needed — because the
thing under test is the loop and its lifecycle, not the model's judgement.

Two conventions the rest of the file leans on:

- **Nothing is believed on this process's own say-so.** "The container is
  gone" is asked of `docker inspect`, not inferred from the fact that we
  called `destroy_container`. Where the point of a test is that something
  really happened (a container was really killed, a command really failed
  before the fix landed), there is an explicit CONTROL observation proving the
  probe could have seen the other answer.
- **The sandbox ops are the real module.** `_RealOpsRecordingIds` forwards
  every call to `coding_agent.sandbox` verbatim and only remembers the
  container id and worktree path on the way through, so the tests can name the
  exact objects to look for afterwards instead of pattern-matching the whole
  host.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from coding_agent import StopReason, run_coding_agent
from coding_agent import loop as _loop
from coding_agent import sandbox as _sb

# The real sandbox image (spec §8.2), not a stand-in: these tests run pytest
# and git INSIDE the container, which alpine cannot do.
IMAGE = "ai-tools-coding-agent:latest"

# The fixture's bug and its fix. The fix is deliberately a DIFFERENT NUMBER OF
# BYTES from the line it replaces, and that is load-bearing rather than
# cosmetic — see `_FIXED`'s comment below.
_BROKEN = "def add(a, b):\n    return a - b\n"
_FIXED = (
    'def add(a, b):\n    """Return the sum of a and b."""\n    return a + b\n'
    # A same-length replacement USED TO BE a COIN FLIP, not a pass. CPython
    # validates a cached .pyc against (int(source mtime), source size); a
    # same-length rewrite that lands in the same integer second leaves both
    # fields matching, so the second `python -m pytest` run imports the OLD
    # bytecode and the "fixed" test still fails. Measured on this host, n=4
    # per cell, before Task 12's fix below existed:
    #   same length, no delay  -> ['PASS','STALE','STALE','STALE']
    #   same length, 1.5s wait -> ['STALE','STALE','STALE','STALE']
    #   this fix,    no delay  -> ['PASS','PASS','PASS','PASS']
    #   this fix,    1.5s wait -> ['PASS','PASS','PASS','PASS']
    # (A wait does NOT help: it elapses after the write, so it cannot change
    # which second the write landed in.) Task 12 closed the underlying bug at
    # its root by adding PYTHONDONTWRITEBYTECODE=1 to the sandbox IMAGE
    # (scripts/coding-agent-image/Dockerfile) — deliberately NOT to the §6.2
    # docker argv, which stays pinned by test_coding_agent_sandbox.py and the
    # spec. No .pyc is ever written now, so there is no stale cache left to
    # read regardless of edit length or timing — see
    # test_the_container_never_writes_pyc_bytecode_but_still_writes_pytest_cache
    # below for the live regression check. The different-length fixture is
    # kept anyway as a belt-and-suspenders regression guard for this test:
    # it is still the more realistic fix to model, and it keeps this test
    # meaningful even against a future image variant that reintroduces
    # bytecode caching.
)

_TEST_FILE = "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n"
# Committed, so `basetree.make_ignore` picks it up: it keeps the bytecode and
# pytest caches the container creates out of the diff a human reviews.
_GITIGNORE = "__pycache__/\n.pytest_cache/\n"


def _skip_reason() -> str | None:
    """A skip reason, or None when a real run is possible."""
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
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if present.returncode != 0:
        return f"image {IMAGE} not built locally (scripts/coding-agent-image/build.sh)"
    return None


_SKIP = _skip_reason()


def _docker(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def _container_exists(cid: str) -> bool:
    """Docker's own answer, not our bookkeeping."""
    return _docker("inspect", "--type", "container", cid).returncode == 0


def _container_running(cid: str) -> bool:
    probe = _docker("inspect", "--type", "container", "-f", "{{.State.Running}}", cid)
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _containers_from_the_sandbox_image() -> str:
    """Every container — running or exited — created from the sandbox image.

    Scoped to the image on purpose. A bare `docker ps -a | grep 'sleep
    infinity'` would also fail on somebody else's unrelated container, and a
    leak check that can cry wolf gets muted.
    """
    return _docker(
        "ps",
        "-a",
        "--no-trunc",
        "--filter",
        f"ancestor={IMAGE}",
        "--format",
        "{{.ID}} {{.Command}} {{.Status}}",
    ).stdout.strip()


def _worktree_parents() -> set[str]:
    """The private `coding-agent-wt-*` parents currently under $TMPDIR."""
    return {str(p) for p in Path(tempfile.gettempdir()).glob("coding-agent-wt-*")}


def _repo() -> str:
    """A real git repo whose one test FAILS at HEAD."""
    r = tempfile.mkdtemp(prefix="ca-integration-repo-")
    subprocess.run(["git", "-C", r, "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", r, "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", r, "config", "user.name", "T"], check=True)
    Path(r, "calc.py").write_text(_BROKEN)
    Path(r, "test_calc.py").write_text(_TEST_FILE)
    Path(r, ".gitignore").write_text(_GITIGNORE)
    subprocess.run(["git", "-C", r, "add", "-A"], check=True)
    subprocess.run(["git", "-C", r, "commit", "-q", "-m", "base"], check=True)
    return r


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": []}


def _script(*msgs: dict[str, Any]) -> Any:
    """A scripted `chat`. Running off the end is an explicit failure rather
    than the `RuntimeError: coroutine raised StopIteration` a bare iterator
    would produce, so an over-running loop is legible."""
    it = iter(msgs)

    async def chat(payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        try:
            return {"message": next(it)}
        except StopIteration:  # pragma: no cover - only on an over-running loop
            raise AssertionError(
                f"the loop asked for turn {len(msgs) + 1}; the script has {len(msgs)}"
            ) from None

    return chat


class _RealOpsRecordingIds:
    """The REAL `coding_agent.sandbox`, with the ids remembered on the way past.

    Every method forwards verbatim — the git worktree, the `docker run` argv,
    the teardown layering are all the product's. Nothing is stubbed; this
    exists only so a test can name the exact container and worktree to go
    looking for afterwards.
    """

    def __init__(self) -> None:
        self.worktree: str | None = None
        self.container: str | None = None

    def create_worktree(self, repo: str, base_ref: str) -> str:
        self.worktree = _sb.create_worktree(repo, base_ref)
        return self.worktree

    async def start_container(self, worktree: str, image: str, **kw: Any) -> str:
        self.container = await _sb.start_container(worktree, image, **kw)
        return self.container

    async def destroy_container(self, container: str) -> None:
        await _sb.destroy_container(container)

    def teardown_worktree(self, repo: str, worktree: str) -> list[str]:
        return _sb.teardown_worktree(repo, worktree)


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class DockerIntegration(unittest.TestCase):
    """Shared fixture: a repo, a recorder, and cleanup that runs even when an
    assertion explodes half way through a run."""

    maxDiff = None

    def setUp(self) -> None:
        self.repo = _repo()
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.ops = _RealOpsRecordingIds()
        self.parents_before = _worktree_parents()
        # Registered LAST so they run FIRST: belt and braces, so a failed
        # assertion cannot leave a container or a worktree behind for the
        # next test (or the next developer) to trip over.
        self.addCleanup(self._assert_slot_free)
        self.addCleanup(self._force_clean)

    def _force_clean(self) -> None:
        if self.ops.container:
            _docker("rm", "-f", self.ops.container)
        if self.ops.worktree:
            _sb.teardown_worktree(self.repo, self.ops.worktree)

    def _assert_slot_free(self) -> None:
        # Not `addCleanup(self.assertFalse, _loop._SLOT.locked())`: that binds
        # the value at registration time and can therefore never fail.
        self.assertFalse(_loop._SLOT.locked(), "the single slot leaked")

    # -- shared post-conditions ------------------------------------------

    def assertNothingLeaked(self) -> None:
        """Container, worktree, private parent — gone, per docker and the fs."""
        self.assertIsNotNone(self.ops.container)
        cid = self.ops.container or ""
        self.assertFalse(
            _container_exists(cid), f"container {cid[:12]} survived the run"
        )
        self.assertEqual(
            _containers_from_the_sandbox_image(),
            "",
            "a container from the sandbox image is still on this host",
        )
        wt = self.ops.worktree or ""
        self.assertFalse(os.path.lexists(wt), f"worktree still on disk: {wt}")
        self.assertFalse(
            os.path.lexists(os.path.dirname(wt)),
            f"the private worktree parent survived: {os.path.dirname(wt)}",
        )
        self.assertEqual(
            _worktree_parents() - self.parents_before,
            set(),
            "a coding-agent-wt-* directory was left under $TMPDIR",
        )

    def assertRepoUntouched(self) -> None:
        """The user's own checkout is read-only to this feature."""
        self.assertEqual(Path(self.repo, "calc.py").read_text(), _BROKEN)
        status = subprocess.run(
            ["git", "-C", self.repo, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(status, "", "the user's repo has uncommitted changes")
        listing = subprocess.run(
            ["git", "-C", self.repo, "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertNotIn("coding-agent-wt-", listing, "a worktree is still registered")

    def commands(self, result: Any) -> list[str]:
        """`result_head` of every run_command entry, in order."""
        return [
            entry["result_head"]
            for entry in result.transcript
            if entry.get("tool") == "run_command"
        ]

    def run_agent(self, chat: Any, **kw: Any) -> Any:
        params: dict[str, Any] = {
            "task": "make the tests pass",
            "repo": self.repo,
            "base_ref": "HEAD",
            "model": "scripted",
            "max_turns": 10,
            "max_seconds": 300,
            "image": IMAGE,
            "chat": chat,
            "sandbox_factory": lambda: self.ops,
        }
        params.update(kw)
        return asyncio.run(run_coding_agent(**params))

    # -- the tests --------------------------------------------------------

    def test_fix_failing_test_end_to_end_and_everything_is_gone_after(self) -> None:
        """The whole feature, once, for real: the model runs the suite, sees it
        fail, edits a file on the host side of the mount, re-runs the suite in
        the container, and the human gets a diff of exactly what changed."""
        result = self.run_agent(
            _script(
                _call("run_command", cmd="python -m pytest -q"),
                _call("write_file", path="calc.py", content=_FIXED),
                _call("run_command", cmd="python -m pytest -q"),
                _say("fixed"),
            )
        )

        self.assertEqual(result.stop_reason, StopReason.completed)
        self.assertEqual(result.turns, 4)

        # CONTROL: the first run really did fail, and failed by executing the
        # BROKEN function (`add(2, 3)` returning -1). Without this the final
        # `exit 0` would be equally consistent with a suite that passed all
        # along — i.e. with the write never reaching the container at all.
        # `result_head` is capped at 400 chars, so this asserts on pytest's
        # assertion-rewrite output rather than on its trailing summary line.
        first, second = self.commands(result)
        self.assertTrue(first.startswith("[exit 1]"), first)
        self.assertIn("assert -1 == 5", first)
        self.assertTrue(second.startswith("[exit 0]"), second)
        self.assertIn("1 passed", second)

        self.assertEqual(result.last_command["cmd"], "python -m pytest -q")
        self.assertEqual(result.last_command["exit"], 0)

        # The review artifact. `changed_files` is asserted EXACTLY: the
        # container writes .pytest_cache/ into the mount, and the point of
        # the ignore predicate is that the human never sees it. (Before Task
        # 12 added PYTHONDONTWRITEBYTECODE=1 to the sandbox image, it wrote
        # __pycache__/ too; that env var means .pyc is never written now, so
        # __pycache__ never exists to filter — see the control test below.)
        self.assertEqual(result.changed_files, ["calc.py"])
        self.assertIn("+    return a + b", result.diff)
        self.assertIn('+    """Return the sum of a and b."""', result.diff)
        self.assertIn("-    return a - b", result.diff)
        self.assertNotIn("__pycache__", result.diff)
        self.assertFalse(result.diff_truncated)
        self.assertEqual(result.unreadable, [])

        # ... and .pytest_cache really was there to be ignored (proven by the
        # control test below), so the assertion above is about the filter and
        # not about an empty dir.
        self.assertEqual(result.cleanup_problems, [])
        self.assertNothingLeaked()
        self.assertRepoUntouched()

    def test_the_container_never_writes_pyc_bytecode_but_still_writes_pytest_cache(
        self,
    ) -> None:
        """The other half of the assertion above, which can only be observed
        while the worktree still exists. Read from inside the container, so
        it is the sandbox's own view of the mount and not a host-side
        inference. Two things at once:

        1. `.pytest_cache` really IS created by `python -m pytest` — the
           CONTROL proving the ignore-predicate assertion above is about a
           real filter, not an already-empty dir.
        2. `__pycache__` is never created at all, because Task 12 added
           PYTHONDONTWRITEBYTECODE=1 to the sandbox image: this is the live
           regression check for that fix, run against the real image and a
           real `python -m pytest` invocation rather than inferred from the
           Dockerfile text.
        """
        result = self.run_agent(
            _script(
                _call("run_command", cmd="python -m pytest -q"),
                _call("run_command", cmd="ls __pycache__ .pytest_cache"),
                _say("done"),
            )
        )
        listing = self.commands(result)[1]
        # `ls` given one missing and one present path lists what it can and
        # exits nonzero for the miss — exactly the mixed outcome here.
        self.assertTrue(listing.startswith("[exit 2]"), listing)
        self.assertIn("cannot access '__pycache__'", listing)
        self.assertIn("No such file or directory", listing)
        self.assertIn("CACHEDIR.TAG", listing)
        # Neither existed in the review — .pytest_cache was filtered,
        # __pycache__ was never written.
        self.assertEqual(result.changed_files, [])
        self.assertEqual(result.cleanup_problems, [])
        self.assertNothingLeaked()

    def test_git_inside_the_container_never_reports_dubious_ownership(self) -> None:
        """The §6.7 spike settled `--user` for raw WRITES only; it never ran
        git in the container, so `fatal: detected dubious ownership` was left
        open. Answered here, against the real image and the real bind mount.

        Two findings, both asserted:

        1. There is no ownership mismatch for git to complain about. Measured
           mechanism (probed at `--user 1000:1000`, `501:20` and `0:0`): a
           Docker Desktop bind mount presents every file as owned by whatever
           uid:gid the container runs as, so `owner == euid` holds by
           construction and the image's `safe.directory /work` never has to
           fire. NOTE for whoever reads this on Linux, where bind mounts pass
           real uids through: there the config IS load-bearing, so this
           finding is a reason to leave it in place, not to delete it. The
           `dubious ownership` assertion below is a tripwire for that platform
           and for a changed mount driver — on Docker Desktop it cannot fail.
        2. git nonetheless cannot operate on /work — for an unrelated reason.
           A linked worktree's `.git` is a FILE pointing at
           `<repo>/.git/worktrees/<name>`, and only the worktree is mounted,
           so the gitdir it names does not exist inside the container. Every
           git command in /work dies with `not a git repository`, including
           `git config --system --list` (git resolves the repository before it
           reads config, so safe.directory is not even reachable from /work).

        That is a security-positive accident — the sandbox cannot reach the
        repo's object store, config or hooks — but it is NOT what the
        Dockerfile's safe.directory comment claims to enable ("using git on
        its own worktree"), and pinning it here means a future change that
        starts mounting the gitdir has to come past this test.
        """
        result = self.run_agent(
            _script(
                _call("run_command", cmd="id -u; id -g"),
                _call("run_command", cmd="stat -c '%u:%g' /work /work/calc.py"),
                _call("run_command", cmd="git status --porcelain; echo rc=$?"),
                _call(
                    "run_command",
                    cmd="cd /tmp && git init -q fresh && cd fresh && : > a "
                    "&& git status --porcelain; echo rc=$?",
                ),
                _say("done"),
            )
        )
        ids, ownership, in_work, in_tmp = self.commands(result)

        # (1) the container runs as the host user ...
        self.assertEqual(ids, f"[exit 0]\n{os.getuid()}\n{os.getgid()}\n")
        # ... and the mount is owned by that same user, so there is no
        # mismatch for git to call dubious.
        want = f"{os.getuid()}:{os.getgid()}"
        self.assertEqual(ownership, f"[exit 0]\n{want}\n{want}\n")

        # (2) git in /work fails, and NOT on ownership.
        self.assertIn("rc=128", in_work)
        self.assertIn("not a git repository", in_work)
        self.assertNotIn("dubious ownership", in_work.lower())

        # CONTROL: git itself works in this image, so the failure above is
        # about the missing gitdir and not about a broken git.
        self.assertIn("rc=0", in_tmp)
        self.assertIn("?? a", in_tmp)
        self.assertNotIn("fatal", in_tmp)

        self.assertEqual(result.cleanup_problems, [])
        self.assertNothingLeaked()
        self.assertRepoUntouched()

    def test_container_crash_midloop_still_cleans_up(self) -> None:
        """`docker rm -f` the sandbox out from under a running loop.

        The kill targets the id THIS run created, never `--filter ancestor=`:
        a broad filter would also reap a container belonging to somebody
        else's session on the same host.
        """
        alive_before: list[bool] = []
        alive_after: list[bool] = []

        async def crashing_chat(payload: dict[str, Any], timeout_s: float) -> Any:
            cid = self.ops.container or ""
            alive_before.append(_container_running(cid))
            _docker("rm", "-f", cid)
            alive_after.append(_container_running(cid))
            return {"message": _call("run_command", cmd="true")}

        result = self.run_agent(crashing_chat, task="t", max_turns=3, max_seconds=60)

        # CONTROL first: the kill was not a no-op on an already-dead
        # container, and it landed. Docker's answer, not ours.
        self.assertTrue(alive_before[0], "the container was not running to begin with")
        self.assertFalse(alive_after[0], "docker rm -f did not remove the container")

        # The loop SAW the crash and fed it back rather than dying on it.
        # The count is asserted FIRST: a `for` over an empty list passes
        # every assertion inside it, which is exactly the shape of a test
        # that proves nothing.
        heads = self.commands(result)
        self.assertEqual(len(heads), 3, heads)
        for head in heads:
            self.assertNotIn("[exit 0]", head, head)
            self.assertIn("No such container", head)
        # ... ran its full budget of turns on a dead sandbox ...
        self.assertEqual(result.turns, 3)
        self.assertEqual(result.stop_reason, StopReason.max_turns)
        # ... and cleaned up anyway.
        self.assertEqual(result.cleanup_problems, [])
        self.assertNothingLeaked()
        self.assertRepoUntouched()

    def test_wall_clock_ceiling_mid_command_still_removes_both(self) -> None:
        """`sleep 30` under a 2s ceiling. The command is killed at the
        budget-clamped per-command timeout instead of running to completion,
        and both the container and the worktree still go."""
        started = time.monotonic()
        result = asyncio.run(
            asyncio.wait_for(
                run_coding_agent(
                    task="t",
                    repo=self.repo,
                    base_ref="HEAD",
                    model="scripted",
                    max_turns=5,
                    max_seconds=2,
                    image=IMAGE,
                    chat=_script(_call("run_command", cmd="sleep 30")),
                    sandbox_factory=lambda: self.ops,
                ),
                timeout=90,
            )
        )
        wall = time.monotonic() - started

        self.assertEqual(result.stop_reason, StopReason.max_seconds)
        self.assertEqual(result.turns, 1)
        # The command was killed mid-flight, not waited out ...
        self.assertEqual(result.last_command["exit"], 124)
        self.assertIn("timed out", result.last_command["output_tail"])
        # ... which is the whole claim: `sleep 30` under a 2s ceiling must not
        # cost 30 seconds. Measured by the TEST's clock as well as the
        # result's own, so the bound does not rest on the code's self-report.
        # The floor is _MIN_CMD_TIMEOUT_S (5s), so ~5s is the expected shape.
        self.assertLess(wall, 20.0, f"the ceiling did not bound the run: {wall:.1f}s")
        self.assertLess(result.elapsed_seconds, 20.0)
        self.assertGreaterEqual(result.elapsed_seconds, 2.0)

        self.assertEqual(result.cleanup_problems, [])
        self.assertNothingLeaked()
        self.assertRepoUntouched()


if __name__ == "__main__":
    unittest.main()
