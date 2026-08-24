#!/usr/bin/env python3
"""Sandbox lifecycle tests for coding_agent (spec §6.2, §6.6, §6.7).

The git-only classes MUST pass everywhere. The docker classes skip when no
daemon answers, so this file is still meaningful on a machine without Docker.

Run:  uv run --with pytest --with pytest-timeout --with pathspec pytest test_coding_agent_sandbox.py -q

`--with pytest-timeout` is requested for parity with the rest of the suite's
`Run:` lines (see test_coding_agent_security.py and test_coding_agent_tools.py
for the tests it actually bounds), not because anything in this file relies on
it — every `exec_in_container`/`asyncio.wait_for` call below already carries
its own explicit `timeout_s`/deadline, deliberately (see
`ShieldedCleanupSurvivesCancellation`). It is optional here too: without the
plugin this file runs exactly as before. A slow-but-healthy real-docker test
is intentionally NOT marked, so it can never be killed spuriously by a
per-test bound sized for the git-only classes.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import unittest.mock
import uuid
from pathlib import Path

from coding_agent.sandbox import (
    _OUTPUT_CAP,
    _TRUNC,
    SANDBOX_USER_FLAG,
    create_worktree,
    destroy_container,
    exec_in_container,
    start_container,
    teardown_worktree,
)


def _init_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="ca-test-repo-"))
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "f").write_text("1\n")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "b"], check=True)
    return repo


def _registered(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class MangledGitCleanup(unittest.TestCase):
    """Pass 1: after the sandbox replaces the worktree's .git FILE with a
    directory, `git worktree remove --force` FAILS (validation error 10) and
    the worktree stays on disk AND registered. Cleanup must not depend on it."""

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)

    def _mangle(self, wt: str) -> None:
        os.remove(os.path.join(wt, ".git"))
        os.mkdir(os.path.join(wt, ".git"))
        with open(os.path.join(wt, ".git", "config"), "w") as fh:
            fh.write("[core]\n\tfsmonitor = /tmp/pwn.sh\n")

    def test_cleanup_leaves_zero_residue_after_dot_git_tampering(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._mangle(wt)
        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))

    def test_MECHANISM_git_worktree_remove_really_fails_on_this_git(self) -> None:
        """Without this the class above proves nothing: if `worktree remove`
        happened to succeed here, the layered cleanup would never be exercised
        and a git-dependent implementation would pass. Pins the demonstrated
        failure (git 2.55.0: 'validation failed ... error code 10')."""
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._mangle(wt)
        proc = subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "remove", "--force", wt],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0, "git removed a tampered worktree")
        self.assertIn("validation failed", proc.stderr)
        # ... and the residue the layered cleanup exists to mop up:
        self.assertTrue(os.path.exists(wt))
        self.assertIn(wt, _registered(self.repo))

        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))


class NoHostGitEverRunsInTheWorktree(unittest.TestCase):
    """The security spine (§6.5). Adversarial review DEMONSTRATED host RCE by
    pointing git at a worktree the model controls; this pins that teardown —
    the one place still tempted to do it — never does."""

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.scratch = Path(tempfile.mkdtemp(prefix="ca-test-pwn-"))
        self.marker = self.scratch / "marker"
        self.payload = self.scratch / "pwn.sh"
        self.payload.write_text(f"#!/bin/sh\necho PWNED > {self.marker}\nexit 1\n")
        self.payload.chmod(0o755)
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _arm(self, wt: str) -> None:
        """What a merely-buggy model does by accident and a hostile one does
        on purpose: `git init` in place, then a config key git EXECUTES."""
        os.remove(os.path.join(wt, ".git"))
        subprocess.run(["git", "init", "-q", wt], check=True)
        subprocess.run(
            ["git", "-C", wt, "config", "core.fsmonitor", str(self.payload)], check=True
        )

    # Every hook name git will resolve out of `core.hooksPath`. Which git
    # commands fire which of these was MEASURED, not assumed, because the
    # obvious short list made the test vacuous: `post-commit`/`pre-commit`
    # need a commit, so a teardown that ran `git -C <worktree> status`
    # executed nothing and the test passed anyway. The two that matter for a
    # plausible teardown bug are `post-index-change` (fires on a bare `git
    # status` whenever the index needs rewriting) and `reference-transaction`
    # (fires on `git reset --hard`).
    _HOOK_NAMES = (
        "applypatch-msg",
        "pre-applypatch",
        "post-applypatch",
        "pre-commit",
        "pre-merge-commit",
        "prepare-commit-msg",
        "commit-msg",
        "post-commit",
        "pre-rebase",
        "post-checkout",
        "post-merge",
        "pre-push",
        "pre-auto-gc",
        "post-rewrite",
        "post-index-change",
        "reference-transaction",
    )

    def _arm_hooks(self, wt: str) -> None:
        """§10 item 2's second named vector, `core.hooksPath`.

        Substantively subsumed by `_arm` — the asserted property is "no host
        git runs here at all", and one executing config key proves that as
        well as another — but the spec names this one specifically and it is a
        DIFFERENT execution mechanism inside git (a hook binary resolved from
        a directory, not a command string invoked for a config value), so a
        change could plausibly close one route and leave the other.

        The worktree is left with a STALE INDEX on purpose: that is what makes
        even a read-only-looking `git status` rewrite the index and fire
        `post-index-change`, so the fixture is armed against the shape a
        teardown bug actually takes rather than only against `git commit`.

        MEASURED (not assumed): backdating `f`'s mtime while also changing
        its content is NOT reliable — content that actually differs from
        what's staged makes git report "modified" without necessarily
        rewriting the index (that rewrite, when it happens, comes from an
        unrelated subsystem racing its own staleness check, e.g. the
        untracked-cache extension). A standalone harness running that
        content-changing variant repeatedly reproduced this test's flake
        rate directly; the same harness ran the content-UNCHANGED variant
        below 300/300 clean. The fix is to leave `f`'s content untouched and
        only backdate its mtime: that puts git on the deterministic path —
        recorded stat stale, content re-verified as UNCHANGED, so git must
        persist a fixed-up stat into the index before it can trust the cache
        next time, which is an unconditional index write.
        """
        hooks = self.scratch / "hooks"
        hooks.mkdir(exist_ok=True)
        for name in self._HOOK_NAMES:
            hook = hooks / name
            hook.write_text(f"#!/bin/sh\necho PWNED > {self.marker}\nexit 0\n")
            hook.chmod(0o755)
        os.remove(os.path.join(wt, ".git"))
        subprocess.run(["git", "init", "-q", wt], check=True)
        subprocess.run(
            ["git", "-C", wt, "config", "core.hooksPath", str(hooks)], check=True
        )
        subprocess.run(["git", "-C", wt, "add", "-A"], capture_output=True, check=False)
        # Backdate the already-staged `f` (content untouched) so its recorded
        # stat is stale; git must re-verify and rewrite the index unconditionally.
        os.utime(os.path.join(wt, "f"), (0, 0))
        self.marker.unlink(missing_ok=True)  # arming itself must not count

    def test_MECHANISM_the_planted_payload_really_executes_if_git_runs_there(
        self,
    ) -> None:
        """Without this control the test below is vacuous — it would pass just
        as well against a payload that never could have run."""
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._arm(wt)
        subprocess.run(["git", "-C", wt, "add", "-A"], capture_output=True, check=False)
        self.assertTrue(
            self.marker.exists(),
            "the payload never fired: this fixture proves nothing",
        )
        self.assertEqual(self.marker.read_text().strip(), "PWNED")
        teardown_worktree(str(self.repo), wt)

    def test_teardown_of_an_armed_worktree_executes_nothing(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._arm(wt)
        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(self.marker.exists(), "host RCE: the payload executed")
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))

    def test_MECHANISM_a_hooksPath_payload_really_executes_if_git_runs_there(
        self,
    ) -> None:
        """The control for the `core.hooksPath` case, for the same reason the
        `core.fsmonitor` one has one: a payload that could never have fired
        makes the test below pass for free.

        The command is a bare `git status` — the most innocuous-looking thing
        a teardown could be tempted to run inside the worktree, and enough to
        execute a hook against this fixture. An earlier version of this
        control used `git commit`, which passed while a `git status` mutation
        of the teardown went undetected.
        """
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._arm_hooks(wt)
        self.assertFalse(self.marker.exists(), "arming itself fired the hook")
        subprocess.run(["git", "-C", wt, "status"], capture_output=True, check=False)
        self.assertTrue(
            self.marker.exists(),
            "the hook never fired: this fixture proves nothing",
        )
        self.assertEqual(self.marker.read_text().strip(), "PWNED")
        teardown_worktree(str(self.repo), wt)

    def test_teardown_of_a_hooksPath_armed_worktree_executes_nothing(self) -> None:
        """§10 item 2. Same property as the `core.fsmonitor` case, through
        git's other execution mechanism."""
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._arm_hooks(wt)
        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(self.marker.exists(), "host RCE: a hook executed")
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))

    def test_every_git_subprocess_targets_the_real_repo_never_the_worktree(
        self,
    ) -> None:
        """The argv-level pin. The marker tests above catch execution; this
        catches the *shape* that would eventually produce it, including a
        `cwd=` or `GIT_DIR=` aimed at the worktree, which no payload happens
        to be armed for today."""
        real_run = subprocess.run
        seen: list[tuple[list[str], dict[str, object]]] = []

        def recorder(argv: list[str], **kwargs: object) -> object:
            seen.append((list(argv), dict(kwargs)))
            return real_run(argv, **kwargs)  # type: ignore[arg-type]

        # The patch windows cover ONLY the sandbox's own calls — arming the
        # worktree legitimately runs `git -C <worktree>` and is done outside.
        with unittest.mock.patch("subprocess.run", side_effect=recorder):
            wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self._arm(wt)
        with unittest.mock.patch("subprocess.run", side_effect=recorder):
            teardown_worktree(str(self.repo), wt)

        self.assertGreaterEqual(len(seen), 4, "expected add + remove/prune/list")
        for argv, kwargs in seen:
            self.assertEqual(argv[:2], ["git", "-C"], f"unanchored git: {argv}")
            self.assertEqual(argv[2], str(self.repo), f"git aimed off-repo: {argv}")
            self.assertNotIn(wt, argv[:3])
            self.assertIsNone(kwargs.get("cwd"), f"git ran with a cwd: {argv}")
            env = kwargs.get("env")
            self.assertIsInstance(env, dict)
            assert isinstance(env, dict)
            for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
                self.assertNotIn(var, env)


class WorktreePathIsNeverASymlink(unittest.TestCase):
    """walk.snapshot_tree opens the root O_NOFOLLOW, so a symlinked final
    component yields an EMPTY snapshot — "the model changed nothing" shown to
    a human whose repo was in fact rewritten. Preventing that is this
    module's job, and it is verified rather than assumed."""

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_created_path_is_a_canonical_real_directory(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        self.assertTrue(os.path.isabs(wt))
        self.assertFalse(os.path.islink(wt))
        self.assertEqual(os.path.realpath(wt), wt, "path has a symlinked ancestor")
        self.assertTrue(stat_isdir(wt))
        # the private parent is ours alone, so nothing can race a symlink in
        self.assertEqual(stat_mode(os.path.dirname(wt)) & 0o777, 0o700)
        teardown_worktree(str(self.repo), wt)

    def test_the_snapshot_of_a_created_worktree_is_NOT_empty(self) -> None:
        from coding_agent.walk import snapshot_tree

        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        snap = snapshot_tree(wt, lambda p: False)
        self.assertIn("f", snap.entries)
        self.assertEqual(list(snap.unreadable), [])

        # MECHANISM: the same walk through a symlinked final component is
        # empty. This is what create_worktree's guarantee is buying.
        link = os.path.join(os.path.dirname(wt), "as-symlink")
        os.symlink(wt, link)
        blind = snapshot_tree(link, lambda p: False)
        self.assertEqual(blind.entries, {})
        # O_DIRECTORY|O_NOFOLLOW on a symlink-to-directory refuses with
        # ENOTDIR on macOS and ELOOP on Linux; the empty snapshot — the part
        # that would be shown to a human as "no changes" — is the same.
        self.assertEqual(len(blind.unreadable), 1)
        self.assertIn(blind.unreadable[0].reason, ("ELOOP", "ENOTDIR"))
        os.unlink(link)
        teardown_worktree(str(self.repo), wt)


class TeardownSurvivesHostileFilesystemStates(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _init_repo()
        self.scratch = Path(tempfile.mkdtemp(prefix="ca-test-fs-"))
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            _chmod_tree_open(path)
            shutil.rmtree(path, ignore_errors=True)
        _chmod_tree_open(str(self.scratch))
        shutil.rmtree(self.scratch, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)

    @unittest.skipIf(
        os.geteuid() == 0,
        "root bypasses the mode-000 directory check that plain rmtree relies "
        "on failing, so the leak this fixture demonstrates cannot occur",
    )
    def test_MECHANISM_a_plain_rmtree_leaks_a_mode_000_directory(self) -> None:
        """`chmod 000 somedir` inside the sandbox is enough to defeat a plain
        `rm -rf`, which would leak the entire worktree — including whatever
        the model wrote — into the user's temp space."""
        victim = self.scratch / "victim"
        (victim / "sub").mkdir(parents=True)
        (victim / "sub" / "f").write_text("x\n")
        (victim / "sub").chmod(0o000)
        shutil.rmtree(victim, ignore_errors=True)
        self.assertTrue(victim.exists(), "rmtree coped; this fixture proves nothing")

    @unittest.skipIf(
        os.geteuid() == 0,
        "root bypasses the mode-000 directory check, so teardown succeeds "
        "trivially without exercising the _chmod_tree_open recovery path "
        "this test exists to pin",
    )
    def test_teardown_removes_a_worktree_with_a_mode_000_directory(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)
        sub = Path(wt) / "sub"
        sub.mkdir()
        (sub / "f").write_text("x\n")
        sub.chmod(0o000)
        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))

    def test_a_recorded_path_that_is_a_symlink_is_unlinked_not_followed(self) -> None:
        """Defence in depth for the rm -rf primitive: if the recorded path
        were ever a symlink, teardown must delete the LINK. Following it would
        delete whatever it aims at — the spec's reason for using the literal
        recorded path in the first place."""
        outside = self.scratch / "precious"
        outside.mkdir()
        (outside / "keep").write_text("do not delete\n")
        holder = self.scratch / (
            "coding-agent-wt-fake"
        )  # name-guarded parent, as created
        holder.mkdir()
        link = holder / "wt-deadbeef"
        os.symlink(str(outside), link)

        problems = teardown_worktree(str(self.repo), str(link))
        self.assertEqual(problems, [])
        self.assertFalse(os.path.lexists(link))
        self.assertTrue((outside / "keep").exists(), "followed a symlink and deleted")

    def test_teardown_of_an_already_gone_worktree_is_clean(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.assertEqual(teardown_worktree(str(self.repo), wt), [])
        self.assertEqual(teardown_worktree(str(self.repo), wt), [])

    def test_an_implausible_path_is_refused_not_removed(self) -> None:
        for bad in ("relative/path", "/toplevel", ""):
            problems = teardown_worktree(str(self.repo), bad)
            self.assertEqual(len(problems), 1)
            self.assertIn("implausible", problems[0])

    def test_the_private_parent_directory_does_not_leak(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        parent = os.path.dirname(wt)
        self.assertTrue(os.path.isdir(parent))
        self.assertEqual(teardown_worktree(str(self.repo), wt), [])
        self.assertFalse(os.path.exists(parent))


class _Shim:
    """A fake `docker` on PATH. Lets the flag/output/timeout paths be pinned
    on a machine with no daemon, while still exercising the real argv
    assembly and the real streaming reader."""

    def __init__(self, body: str) -> None:
        self.dir = tempfile.mkdtemp(prefix="ca-test-bin-")
        self.log = os.path.join(self.dir, "argv.log")
        exe = os.path.join(self.dir, "docker")
        with open(exe, "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                f'for a in "$@"; do printf "%s\\0" "$a" >> "{self.log}"; done\n'
                f'printf "\\1" >> "{self.log}"\n'
                'case "$*" in *"kill -9"*) exit 0 ;; esac\n' + body
            )
        os.chmod(exe, 0o755)

    def invocations(self) -> list[list[str]]:
        if not os.path.exists(self.log):
            return []
        with open(self.log, "rb") as fh:
            raw = fh.read()
        return [
            [a.decode() for a in rec.split(b"\0") if a]
            for rec in raw.split(b"\1")
            if rec
        ]

    def patched(self) -> unittest.mock._patch_dict:
        return unittest.mock.patch.dict(
            os.environ, {"PATH": self.dir + os.pathsep + os.environ.get("PATH", "")}
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class DockerArgvIsExactlyTheSpec(unittest.TestCase):
    """§6.2 step 2 is a security surface, not a style choice: every one of
    these flags is load-bearing, so the argv is pinned literally.

    AMENDED 2026-08-19 (final review SF-6/NF-7): `--cap-drop=ALL` and
    `--security-opt=no-new-privileges` are ADDITIONS to §6.2's literal argv.
    Measured in a live container without them, `NoNewPrivs: 0` and
    `CapBnd: 00000000a80425fb` — Docker's full default bounding set — while
    the image carries 11 setuid-root binaries the untrusted model can reach.
    See `sandbox._docker_run_argv`'s docstring for the full measurement and
    the toolchain equivalence check.
    """

    def test_docker_run_argv(self) -> None:
        shim = _Shim("echo deadbeefcafe\n")
        self.addCleanup(shim.cleanup)
        fixed_run_id = "0" * 32
        with (
            shim.patched(),
            unittest.mock.patch(
                "coding_agent.sandbox.uuid.uuid4",
                return_value=uuid.UUID(int=0),
            ),
        ):
            cid = asyncio.run(
                start_container("/wt", "img:tag", cpus="2", memory="4g", pids=256)
            )
        self.assertEqual(cid, "deadbeefcafe")
        argv = shim.invocations()[0]
        self.assertEqual(
            argv,
            [
                "run", "-d", "--rm", "--init", "--network=none",
                "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--label", f"com.jasonvassallo.ai-tools-mcp.coding-agent-run-id={fixed_run_id}",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/home/agent",
                "-e", "HOME=/home/agent", "-e", "TMPDIR=/tmp",
                "-e", "UV_CACHE_DIR=/tmp/uv", "-e", "PIP_CACHE_DIR=/tmp/pip",
                "-e", "NPM_CONFIG_CACHE=/tmp/npm",
                "--cpus", "2", "--memory", "4g", "--pids-limit", "256",
                "-v", "/wt:/work:rw", "-w", "/work", "img:tag",
                "sleep", "infinity",
            ],
        )  # fmt: skip

    def test_the_user_flag_is_the_spike_s_decision_not_a_hardcoded_uid(self) -> None:
        """Task 0 refuted the spec's premise and chose least privilege
        expressed at runtime; a literal '501:20' would be silently wrong under
        any other account."""
        self.assertEqual(SANDBOX_USER_FLAG[0], "--user")
        self.assertEqual(SANDBOX_USER_FLAG[1], f"{os.getuid()}:{os.getgid()}")

    def test_a_failing_docker_run_raises_with_the_daemon_s_message(self) -> None:
        shim = _Shim("echo 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            with self.assertRaises(RuntimeError) as caught:
                asyncio.run(
                    start_container("/wt", "img", cpus="1", memory="1g", pids=64)
                )
        self.assertIn("Cannot connect", str(caught.exception))

    def test_the_model_s_command_is_a_separate_argv_element(self) -> None:
        """It is never interpolated into the wrapper script, so no quoting of
        the model's command is required — or possible to get wrong."""
        shim = _Shim("exit 0\n")
        self.addCleanup(shim.cleanup)
        nasty = '"; touch /tmp/pwned; echo "'
        with shim.patched():
            asyncio.run(exec_in_container("cid", nasty, timeout_s=10))
        argv = shim.invocations()[0]
        self.assertEqual(argv[:2], ["exec", "cid"])
        self.assertIn(nasty, argv)
        self.assertEqual(argv[-1], nasty)


class ExecOutputCannotExhaustHostMemory(unittest.TestCase):
    """`yes > /dev/stdout` inside the sandbox must not be able to take the
    host process down — the output is capped AND streamed, never accumulated
    and then trimmed."""

    def test_output_past_the_cap_is_truncated_and_flagged(self) -> None:
        shim = _Shim("dd if=/dev/zero bs=65536 count=256 2>/dev/null | tr '\\0' 'x'\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            rc, out, truncated = asyncio.run(
                exec_in_container("cid", "flood", timeout_s=60)
            )
        self.assertEqual(rc, 0)
        self.assertTrue(truncated)
        self.assertEqual(len(out), _OUTPUT_CAP + len(_TRUNC))
        self.assertTrue(out.endswith(_TRUNC))

    def test_MECHANISM_16MiB_of_output_is_never_held_in_memory(self) -> None:
        import tracemalloc

        shim = _Shim("dd if=/dev/zero bs=65536 count=256 2>/dev/null | tr '\\0' 'x'\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            tracemalloc.start()
            asyncio.run(exec_in_container("cid", "flood", timeout_s=60))
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        # 16 MiB was written; a communicate()-style read peaks above it.
        self.assertLess(peak, 4 << 20, f"peak {peak} bytes: output was accumulated")

    def test_short_output_is_returned_whole_and_not_flagged(self) -> None:
        shim = _Shim("echo hello\nexit 3\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            rc, out, truncated = asyncio.run(
                exec_in_container("cid", "echo hello", timeout_s=10)
            )
        self.assertEqual((rc, out, truncated), (3, "hello\n", False))


class ExecTimeoutKeepsTheSandboxAlive(unittest.TestCase):
    def test_timeout_returns_124_with_partial_output_and_kills_in_container(
        self,
    ) -> None:
        shim = _Shim("echo partial\nsleep 30\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            rc, out, truncated = asyncio.run(
                exec_in_container("cid", "hang", timeout_s=0.5)
            )
        self.assertEqual(rc, 124)
        self.assertFalse(truncated)
        # the output that explains the hang is kept, not thrown away
        self.assertIn("partial", out)
        self.assertIn("timed out", out)
        # ... and a kill was actually issued INSIDE the container
        kills = [inv for inv in shim.invocations() if any("kill -9" in a for a in inv)]
        self.assertEqual(len(kills), 1, "no in-container kill was attempted")
        self.assertEqual(kills[0][:2], ["exec", "cid"])

    def test_the_kill_never_targets_everything(self) -> None:
        """`kill -9 -1` would also take `sleep infinity`; tini would exit and
        the container would vanish mid-run, turning one slow command into a
        dead sandbox for every turn that follows."""
        shim = _Shim("sleep 30\n")
        self.addCleanup(shim.cleanup)
        with shim.patched():
            asyncio.run(exec_in_container("cid", "hang", timeout_s=0.4))
        script = " ".join(shim.invocations()[1])
        self.assertIn("kill -9", script)
        self.assertNotIn("kill -9 -1", script)


class ShieldedCleanupSurvivesCancellation(unittest.TestCase):
    """§6 item 3: cleanup runs in a `finally` wrapped in
    `asyncio.shield(asyncio.wait_for(cleanup, T))`. The loop owns that
    wrapping; what is pinned here is that THESE primitives survive it — a
    wall-clock ceiling that cancels the run must not become a leak path.

    The `docker rm -f` step is stood in for by a short sleep so the test is
    daemon-independent; what matters for cancellation is that it suspends.
    """

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.strays: list[str] = []

    def tearDown(self) -> None:
        for path in self.strays:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)

    def _cleanup_coro(self, wt: str) -> object:
        async def cleanup() -> list[str]:
            await destroy_container("coding-agent-no-such-container")
            await asyncio.sleep(0.05)  # a busy daemon's `docker rm -f`
            return teardown_worktree(str(self.repo), wt)

        return cleanup()

    def test_shielded_cleanup_completes_after_the_task_is_cancelled(self) -> None:
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)

        async def scenario() -> list[str]:
            inner: list[asyncio.Task[list[str]]] = []

            async def body() -> None:
                try:
                    await asyncio.sleep(30)  # the runaway the ceiling cancels
                finally:
                    task = asyncio.ensure_future(
                        asyncio.wait_for(self._cleanup_coro(wt), 30)  # type: ignore[arg-type]
                    )
                    inner.append(task)
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(task)

            outer = asyncio.create_task(body())
            await asyncio.sleep(0.05)
            outer.cancel()
            await asyncio.sleep(0)  # let it reach the finally
            outer.cancel()  # an impatient supervisor cancels again
            with contextlib.suppress(asyncio.CancelledError):
                await outer
            return await inner[0]

        problems = asyncio.run(scenario())
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(wt))
        self.assertNotIn(wt, _registered(self.repo))

    def test_MECHANISM_without_the_shield_the_same_cancellation_leaks(self) -> None:
        """Why the shield is mandatory and not stylistic."""
        wt = create_worktree(str(self.repo), "HEAD")
        self.strays.append(wt)

        async def scenario() -> None:
            async def body() -> None:
                try:
                    await asyncio.sleep(30)
                finally:
                    # MUTANT-BY-CONSTRUCTION: unshielded
                    await asyncio.wait_for(self._cleanup_coro(wt), 30)  # type: ignore[arg-type]

            outer = asyncio.create_task(body())
            await asyncio.sleep(0.05)
            outer.cancel()
            await asyncio.sleep(0)
            outer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await outer

        asyncio.run(scenario())
        self.assertTrue(os.path.exists(wt), "unshielded cleanup happened to survive")
        self.assertEqual(teardown_worktree(str(self.repo), wt), [])


class DestroyContainerReportsAnUnconfirmedRemoval(unittest.TestCase):
    """§6.5 rule 3 puts `docker rm -f` BEFORE the host read so a live
    container cannot race the diff a human is about to trust. That ordering
    is worth nothing unless a removal that did NOT happen is distinguishable
    from one that did: otherwise `_teardown` snapshots regardless and hands
    back an uncaveated diff. So every outcome is reported, and pinned here.
    """

    def _shim(self, body: str) -> _Shim:
        shim = _Shim(body)
        self.addCleanup(shim.cleanup)
        return shim

    def _destroy(self, shim: _Shim, **kw: object) -> str | None:
        with shim.patched():
            return asyncio.run(destroy_container("cid", **kw))  # type: ignore[arg-type]

    def test_a_confirmed_removal_reports_nothing(self) -> None:
        self.assertIsNone(self._destroy(self._shim("exit 0\n")))

    def test_a_nonzero_rm_is_reported(self) -> None:
        problem = self._destroy(self._shim("exit 1\n"))
        self.assertIsNotNone(problem, "a failed `docker rm -f` looked like success")
        self.assertIn("exited 1", problem or "")

    def test_a_timed_out_rm_is_reported_and_still_bounded(self) -> None:
        started = time.monotonic()
        problem = self._destroy(self._shim("sleep 5\n"), timeout_s=0.2)
        elapsed = time.monotonic() - started
        self.assertIn("did not finish", problem or "")
        # Killing the docker CLIENT proves nothing about the daemon, so the
        # timeout is reported — but it is still bounded, not merely noticed.
        self.assertLess(elapsed, 4.0, "destroy_container outran its own bound")

    def test_a_docker_that_cannot_be_SPAWNED_is_reported(self) -> None:
        empty = tempfile.mkdtemp(prefix="ca-test-nobin-")
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        with unittest.mock.patch.dict(os.environ, {"PATH": empty}):
            problem = asyncio.run(destroy_container("cid"))
        self.assertIn("could not start", problem or "")

    def test_the_argv_is_still_exactly_rm_dash_f(self) -> None:
        """The report is added; the command is not changed."""
        shim = self._shim("exit 1\n")
        with shim.patched():
            asyncio.run(destroy_container("cid"))
        self.assertEqual(shim.invocations(), [["rm", "-f", "cid"]])


_TEST_IMAGE = os.environ.get("AI_TOOLS_CODING_AGENT_TEST_IMAGE", "alpine:3")


def _docker_ready() -> str | None:
    """Returns a skip reason, or None when a real run is possible. The image
    is never pulled: a test suite must not reach the network."""
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
class RealContainerLifecycle(unittest.TestCase):
    """The docker half of §10's integration row. Skipped, never silently
    weakened, when no daemon answers."""

    def setUp(self) -> None:
        self.repo = _init_repo()
        self.wt = create_worktree(str(self.repo), "HEAD")
        self.addCleanup(lambda: teardown_worktree(str(self.repo), self.wt))
        self.addCleanup(lambda: shutil.rmtree(self.repo, ignore_errors=True))
        self.cid = asyncio.run(
            start_container(self.wt, _TEST_IMAGE, cpus="1", memory="512m", pids=128)
        )
        # belt and braces: nothing leaks even if an assertion below explodes
        self.addCleanup(
            lambda: subprocess.run(
                ["docker", "rm", "-f", self.cid], capture_output=True, check=False
            )
        )

    def _run(self, cmd: str, timeout_s: float = 30.0) -> tuple[int, str, bool]:
        return asyncio.run(exec_in_container(self.cid, cmd, timeout_s=timeout_s))

    def test_the_isolation_flags_are_real_not_just_present_on_the_command_line(
        self,
    ) -> None:
        rc, out, _ = self._run("id -u")
        self.assertEqual((rc, out.strip()), (0, str(os.getuid())))

        # --network=none. NOT asserted via `ls /sys/class/net`: on Docker
        # Desktop that lists the LinuxKit VM's tunnel stubs (gre0, sit0,
        # tunl0, ...) even inside an isolated netns, so it looks alarming
        # while proving nothing. The netns facts that do mean something:
        rc, out, _ = self._run("ip -o addr")
        self.assertEqual(rc, 0)
        addressed = [ln for ln in out.splitlines() if " inet" in ln]
        self.assertTrue(addressed)
        for line in addressed:
            self.assertIn(
                " lo ", line, f"a non-loopback interface has an address: {line}"
            )
        self.assertEqual(self._run("ip route")[1].strip(), "", "a route exists")
        # ... and the behaviour those facts predict
        rc, out, _ = self._run(
            "wget -q -T 2 -O /dev/null http://1.1.1.1/", timeout_s=20
        )
        self.assertNotEqual(rc, 0, "the sandbox reached the network")

        # --read-only root, with writable /tmp and $HOME (or the first test
        # run in the sandbox dies on EROFS and looks like a model failure)
        rc, _out, _ = self._run("touch /nope-readonly")
        self.assertNotEqual(rc, 0, "the container root was writable")
        self.assertEqual(self._run("touch /tmp/ok && touch $HOME/ok")[0], 0)
        self.assertEqual(self._run("test -w /work")[0], 0)

    def test_the_bind_mount_round_trips_to_the_host(self) -> None:
        rc, _out, _ = self._run("printf 'made-by-sandbox\\n' > /work/new.py")
        self.assertEqual(rc, 0)
        self.assertEqual(Path(self.wt, "new.py").read_text(), "made-by-sandbox\n")
        # ... and the host can read what it wrote back (the §6.7 finding:
        # host-side ownership is the real user regardless of container uid)
        self.assertEqual(Path(self.wt, "new.py").stat().st_uid, os.getuid())

    def test_exit_codes_and_combined_output(self) -> None:
        rc, out, truncated = self._run("echo to-stdout; echo to-stderr >&2; exit 7")
        self.assertEqual(rc, 7)
        self.assertFalse(truncated)
        self.assertIn("to-stdout", out)
        self.assertIn("to-stderr", out)

    def test_a_timed_out_command_dies_but_the_container_survives(self) -> None:
        rc, out, _ = self._run("echo starting; sleep 60", timeout_s=3.0)
        self.assertEqual(rc, 124)
        self.assertIn("starting", out)
        self.assertIn("timed out", out)

        # the container is still usable for the next turn
        self.assertEqual(self._run("echo alive")[:2], (0, "alive\n"))
        # ... and the timed-out command is gone
        _rc, procs, _ = self._run("ps")
        # MECHANISM: `ps` really does show full command lines here, so the
        # absence below is evidence and not an artefact of a truncated table.
        self.assertIn("sleep infinity", procs)
        self.assertNotIn("sleep 60", procs)

    def test_a_backgrounded_child_of_a_timed_out_command_is_killed_too(
        self,
    ) -> None:
        """The kill targets the process GROUP. `docker exec` makes the exec'd
        shell a group leader (verified on this host: pid == pgrp), so the
        whole tree goes — which is what `--init`'s reaping alone would not
        give, since a live `sleep` is not a zombie."""
        rc, out, _ = self._run("sleep 61 & echo spawned; sleep 62", timeout_s=3.0)
        self.assertEqual(rc, 124)
        self.assertIn("spawned", out, "the background child never started")
        _rc, procs, _ = self._run("ps")
        self.assertIn("sleep infinity", procs)
        self.assertNotIn("sleep 62", procs)
        self.assertNotIn("sleep 61", procs, "a backgrounded child outlived the kill")

    def test_output_over_the_cap_is_truncated_from_a_real_container(self) -> None:
        rc, out, truncated = self._run(
            "dd if=/dev/zero bs=65536 count=64 2>/dev/null | tr '\\0' 'x'"
        )
        self.assertEqual(rc, 0)
        self.assertTrue(truncated)
        self.assertEqual(len(out), _OUTPUT_CAP + len(_TRUNC))

    def test_destroy_then_teardown_leaves_no_container_and_no_worktree(self) -> None:
        self._run("printf 'x\\n' > /work/leftover.py")
        asyncio.run(destroy_container(self.cid))
        listing = subprocess.run(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertNotIn(self.cid, listing, "the container survived docker rm -f")

        problems = teardown_worktree(str(self.repo), self.wt)
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(self.wt))
        self.assertNotIn(self.wt, _registered(self.repo))

    def test_destroying_an_already_gone_container_is_silent(self) -> None:
        asyncio.run(destroy_container(self.cid))
        asyncio.run(destroy_container(self.cid))  # must not raise


def stat_isdir(path: str) -> bool:
    import stat as _stat

    return _stat.S_ISDIR(os.lstat(path).st_mode)


def stat_mode(path: str) -> int:
    return os.lstat(path).st_mode


def _chmod_tree_open(root: str) -> None:
    for dirpath, dirnames, _files in os.walk(root, topdown=True):
        for name in dirnames:
            child = os.path.join(dirpath, name)
            if not os.path.islink(child):
                with contextlib.suppress(OSError):
                    os.chmod(child, 0o700)


if __name__ == "__main__":
    unittest.main()
