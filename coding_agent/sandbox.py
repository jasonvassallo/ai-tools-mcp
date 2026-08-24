"""Worktree + container lifecycle for coding_agent (spec §6.2, §6.6).

Rules enforced here:

- Every git command is `git -C <repo>` against the user's REAL, TRUSTED
  repository. No host-side git ever runs with the sandbox-controlled
  worktree as cwd or gitdir — not for cleanup, not for anything (§6.5).
  Two adversarial passes reached host RCE through a worktree the model
  controls (`core.fsmonitor` via a swapped `.git`, `filter.*` clean/smudge
  via an in-tree `.gitattributes`), and a "check for tampering, then run
  git" design was defeated as a TOCTOU. There is no safe way to point git
  at this directory, so nothing here does.

- Teardown is layered and does NOT depend on `git worktree remove`
  succeeding: it is `docker rm -f` (the caller's ordering), then a
  best-effort `git -C <repo> worktree remove --force`, then an
  UNCONDITIONAL `rm -rf` of the LITERAL path recorded at creation, then
  `git -C <repo> worktree prune`, then verification. The literal path is
  load-bearing: it is never re-derived from git and never re-resolved
  through the worktree, so a planted symlink cannot redirect the delete.

- The container is destroyed BEFORE any host read of the worktree that
  builds the returned result (§6.5 rule 3). The loop enforces that
  ordering; this module provides the primitives and never reads the
  worktree itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field

# Decided by the §6.7 spike (docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md).
# The spike REFUTED the spec's original claim: all three variants wrote the
# bind mount, and Docker Desktop's virtiofs bridge gave the host user
# ownership of container-written files regardless of the uid the container
# declared. The tie-break was therefore least privilege — variant (c), the
# only one that does not run the untrusted model's commands as container
# root. Kept as the runtime expression rather than the observed literal
# "501:20": nothing in the finding depends on those digits, and a literal
# would be silently wrong under any other account.
SANDBOX_USER_FLAG: list[str] = ["--user", f"{os.getuid()}:{os.getgid()}"]

# No PATH: git is resolved through os.defpath (/bin:/usr/bin), so an ambient
# PATH cannot substitute a different `git`. No HOME either, which together
# with GIT_CONFIG_NOSYSTEM keeps behaviour independent of the operator's
# global config. Identical to basetree.py's, deliberately.
_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}

_OUTPUT_CAP = 64 * 1024
_TRUNC = "\n[... output truncated by coding_agent ...]\n"
_READ_CHUNK = 64 * 1024

# Every worktree this module creates lives alone inside a private 0700
# directory carrying this prefix. Teardown will rmdir that directory, but
# only when it is empty and only when its name still carries the prefix.
_WT_PREFIX = "coding-agent-wt-"

_DESTROY_TIMEOUT_S = 30.0
_KILL_GRACE_S = 5.0
_START_TIMEOUT_S = 60.0
_REAP_TIMEOUT_S = 10.0
# A local git operation should never take this long. Without a bound, a
# stalled git (held index.lock, a network filesystem gone away) hangs
# create_worktree/teardown_worktree forever — every OTHER subprocess in this
# module already carries an explicit timeout; this was the one that didn't.
_GIT_TIMEOUT_S = 120.0

# Every container this module starts carries this label, valued with a
# per-call run_id. Its only reader is `_reap_labelled`: a container id is
# ordinarily learned from `docker run`'s own stdout, but a client that timed
# out waiting for that response never gets one. The label makes such a
# container findable anyway.
_RUN_LABEL_KEY = "com.jasonvassallo.ai-tools-mcp.coding-agent-run-id"

# `sh -c SCRIPT name a b` sets $0=name, $1=a, $2=b. The model's command is
# passed as its own argv element and is never interpolated into the script
# text, so no quoting of `cmd` is required or performed. `echo $$` records
# the pid of the shell that then `exec`s the command, which is what the
# timeout path kills.
_EXEC_WRAPPER = 'echo $$ > "$1" 2>/dev/null; exec sh -c "$2"'

# Kill the timed-out command's process GROUP first (which takes anything it
# spawned) and fall back to the single pid. Never `kill -1`: that would also
# take `sleep infinity`, tini would exit, and the container would vanish
# mid-run, turning one slow command into a dead sandbox.
_KILL_WRAPPER = (
    'p=$(cat "$1" 2>/dev/null) || exit 0; [ -n "$p" ] || exit 0; '
    'kill -9 -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; rm -f "$1"; exit 0'
)


@dataclass
class Sandbox:
    repo: str
    base_ref: str
    worktree: str
    container: str | None
    image: str


@dataclass
class _Capture:
    """A capped, streaming output buffer.

    Owned by the caller rather than by the reading coroutine so that a
    cancelled read (the timeout path) still surfaces what was captured
    before the deadline. Bytes past the cap are read and DISCARDED, never
    accumulated: the pipe still has to be drained or the container-side
    command blocks forever on a full pipe, but `yes > /dev/stdout` must not
    be able to exhaust host memory.
    """

    cap: int
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    truncated: bool = False

    def feed(self, chunk: bytes) -> None:
        room = self.cap - self.size
        if room > 0:
            keep = chunk[:room]
            self.chunks.append(keep)
            self.size += len(keep)
        if len(chunk) > room:
            self.truncated = True

    def text(self) -> str:
        out = b"".join(self.chunks).decode("utf-8", "replace")
        return out + _TRUNC if self.truncated else out


def _reject_option_like(name: str, value: str) -> None:
    """A value that looks like a flag would be parsed as one by git.

    `repo` and `base_ref` arrive from the MCP caller; neither is ever meant
    to start with `-`, and letting one through would let a caller inject an
    option into a git command line.
    """
    if value.startswith("-"):
        raise ValueError(f"coding_agent: {name} must not start with '-': {value!r}")


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=check,
        capture_output=True,
        text=True,
        env={**_GIT_ENV},
        timeout=_GIT_TIMEOUT_S,
    )


def _git_quiet(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    """`_git` that cannot raise — teardown must not abort on layer 1."""
    try:
        return _git(repo, *args, check=False)
    except OSError as exc:  # git missing, fork failure, ...
        return subprocess.CompletedProcess(
            args=["git", "-C", repo, *args], returncode=127, stdout="", stderr=str(exc)
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=["git", "-C", repo, *args],
            returncode=124,
            stdout="",
            stderr=f"timed out after {_GIT_TIMEOUT_S:g}s: {exc}",
        )


def create_worktree(repo: str, base_ref: str) -> str:
    """Detached throwaway worktree of `base_ref`. Returns the ABSOLUTE path —
    callers must record it verbatim; teardown rm -rf's this literal string.

    The returned path is fully realpath-canonical and its FINAL COMPONENT is
    never a symlink. That is not cosmetic: `walk.snapshot_tree` opens the
    walk root with `O_NOFOLLOW`, so a symlinked final component fails ELOOP
    and yields an EMPTY snapshot — which would reach the human as "the model
    changed nothing." Guaranteed two ways, both cheap:

    1. the worktree is created inside a private 0700 `mkdtemp` directory
       under a name only this process knows, so nothing else — the sandbox
       included, since only the worktree itself is bind-mounted — can race a
       symlink into that name, and git itself creates the leaf as a real
       directory;
    2. it is verified anyway after creation (lstat says directory, not
       symlink; realpath is a fixed point), and a violation fails closed.
    """
    _reject_option_like("repo", repo)
    _reject_option_like("base_ref", base_ref)
    parent = os.path.realpath(tempfile.mkdtemp(prefix=_WT_PREFIX))
    path = os.path.join(parent, f"wt-{uuid.uuid4().hex[:12]}")
    try:
        _git(repo, "worktree", "add", "--detach", path, base_ref)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(
            f"coding_agent: git worktree add failed: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(
            f"coding_agent: git worktree add did not return within {_GIT_TIMEOUT_S:g}s"
        ) from exc
    except OSError as exc:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(f"coding_agent: git worktree add failed: {exc}") from exc

    try:
        st = os.lstat(path)
        canonical = os.path.realpath(path) == path
    except OSError as exc:
        teardown_worktree(repo, path)
        raise RuntimeError(f"coding_agent: worktree not usable: {exc}") from exc
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode) or not canonical:
        teardown_worktree(repo, path)
        raise RuntimeError(
            f"coding_agent: refusing a worktree path that is not a canonical "
            f"real directory (a symlinked root walks as EMPTY): {path}"
        )
    return path


def _docker_run_argv(
    worktree: str, image: str, *, cpus: str, memory: str, pids: int, run_id: str
) -> list[str]:
    """Spec §6.2 step 2. `--init` runs tini as pid 1 so processes the model
    backgrounds are reaped; `--read-only` needs a writable $HOME and not just
    /tmp, or the first in-sandbox test run dies on EROFS and looks like a
    model failure.

    AMENDS §6.2 (final review, SF-6/NF-7): `--cap-drop=ALL` and
    `--security-opt=no-new-privileges` are additions to the spec's literal
    argv, not part of it. Measured in a live container built from the shipped
    argv WITHOUT them:

        NoNewPrivs: 0      CapBnd: 00000000a80425fb   (Docker's full default)

    and the image carries 11 setuid-root binaries the untrusted model can
    reach (`/usr/bin/{su,mount,umount,passwd,newgrp,chsh,chfn,gpasswd,expiry,
    chage}`, `/usr/sbin/unix_chkpwd`). `--user` already gives an unprivileged
    `CapEff`, but an intact BOUNDING set is exactly what lets a setuid-root
    binary hand them back. With both flags:

        NoNewPrivs: 1      CapBnd: 0000000000000000

    Escalation from there needs a separate CVE in one of those binaries, and
    container-root on Docker Desktop/macOS is still confined to the Linux VM,
    so this is defence in depth rather than a fix for a live hole. It was
    added because it costs nothing: python, git, uv, ruff, mypy, node, npm and
    shellcheck were exercised in a real container under both variants — a
    failing pytest, an edit, a passing pytest, a `uv run` subprocess, tmpfs
    and $HOME writes — with identical results.

    `test_coding_agent_integration.py` pins this argv literally; the two flags
    were added there in the same change.

    `--label` (also not in §6.2's literal argv) exists so a container the
    daemon created but whose id our client never received — it timed out
    waiting on `docker run`'s own response, §6.7's `_START_TIMEOUT_S` case —
    can still be found and reaped by `_reap_labelled`, without which that
    container would run forever unrecognized by anything.
    """
    return [
        "docker",
        "run",
        "-d",
        "--rm",
        "--init",
        "--network=none",
        # Not in §6.2's literal argv — see the docstring above.
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--label",
        f"{_RUN_LABEL_KEY}={run_id}",
        *SANDBOX_USER_FLAG,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home/agent",
        "-e",
        "HOME=/home/agent",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "UV_CACHE_DIR=/tmp/uv",
        "-e",
        "PIP_CACHE_DIR=/tmp/pip",
        "-e",
        "NPM_CONFIG_CACHE=/tmp/npm",
        "--cpus",
        cpus,
        "--memory",
        memory,
        "--pids-limit",
        str(pids),
        "-v",
        f"{worktree}:/work:rw",
        "-w",
        "/work",
        image,
        "sleep",
        "infinity",
    ]


async def _reap_labelled(run_id: str) -> None:
    """Best-effort cleanup for a container `start_container` never got an id
    for. Bounded, and silent by necessity rather than by preference: this
    runs on an already-failing path with no result to attach a caveat to, so
    `destroy_container`'s report is deliberately dropped here. It must never
    raise or hang that path further.
    """
    with contextlib.suppress(OSError):
        lister = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            f"label={_RUN_LABEL_KEY}={run_id}",
            "--format",
            "{{.ID}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            out, _ = await asyncio.wait_for(lister.communicate(), _REAP_TIMEOUT_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                lister.kill()
            return
        for cid in out.decode(errors="replace").split():
            await destroy_container(cid, timeout_s=_REAP_TIMEOUT_S)


async def start_container(
    worktree: str, image: str, *, cpus: str, memory: str, pids: int
) -> str:
    """Bounded by `_START_TIMEOUT_S`, like every other await in this module
    (§ file docstring intent: nothing here can hang the event loop
    indefinitely). A wedged daemon otherwise leaves this coroutine waiting
    forever with no way for the caller's own timeout machinery to reach it,
    since it hasn't returned a container id yet for that machinery to act on.

    Both failure paths call `_reap_labelled`: the daemon can create and start
    the container as part of handling `docker run`'s single API call before
    responding, so a client that times out or otherwise never gets a usable
    id can still leave one running. Labelling every run makes it findable
    without that id.
    """
    run_id = uuid.uuid4().hex
    proc = await asyncio.create_subprocess_exec(
        *_docker_run_argv(
            worktree, image, cpus=cpus, memory=memory, pids=pids, run_id=run_id
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), _START_TIMEOUT_S)
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _KILL_GRACE_S)
        await _reap_labelled(run_id)
        raise RuntimeError(
            f"docker run did not return within {_START_TIMEOUT_S:g}s"
        ) from exc
    if proc.returncode != 0:
        await _reap_labelled(run_id)
        raise RuntimeError(f"docker run failed: {err.decode(errors='replace').strip()}")
    return out.decode().strip()


async def _pump(proc: asyncio.subprocess.Process, cap: _Capture) -> int:
    stream = proc.stdout
    if stream is not None:
        while True:
            chunk = await stream.read(_READ_CHUNK)
            if not chunk:
                break
            cap.feed(chunk)
    return await proc.wait()


async def _kill_in_container(container: str, pidfile: str) -> None:
    with contextlib.suppress(OSError):
        killer = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            container,
            "sh",
            "-c",
            _KILL_WRAPPER,
            "coding_agent",
            pidfile,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(killer.wait(), _KILL_GRACE_S)


async def exec_in_container(
    container: str, cmd: str, *, timeout_s: float
) -> tuple[int, str, bool]:
    """Run `cmd` in the running container. Returns (exit, combined output,
    truncated). Never raises for anything the sandbox can cause.

    A timeout returns exit 124 with whatever the command had already printed
    — losing it would hide exactly the output that explains the hang — and
    kills the command's process group INSIDE the container, leaving the
    container itself alive for the next turn.
    """
    pidfile = f"/tmp/.coding-agent-exec-{uuid.uuid4().hex}"
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        _EXEC_WRAPPER,
        "coding_agent",
        pidfile,
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    cap = _Capture(_OUTPUT_CAP)
    try:
        rc = await asyncio.wait_for(_pump(proc, cap), timeout=timeout_s)
    except TimeoutError:
        await _kill_in_container(container, pidfile)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _KILL_GRACE_S)
        text = cap.text() + f"\n[... command timed out after {timeout_s:g}s ...]\n"
        return 124, text, cap.truncated
    return rc, cap.text(), cap.truncated


async def destroy_container(
    container: str, *, timeout_s: float = _DESTROY_TIMEOUT_S
) -> str | None:
    """`docker rm -f`, bounded so a wedged daemon cannot hold teardown open.

    Still never raises — teardown runs in a `finally`, and a teardown step
    that can raise is a teardown step that can be skipped. It REPORTS
    instead: `None` once the daemon has confirmed the removal, otherwise one
    line saying why the removal could not be confirmed.

    Reporting rather than raising is also what `_reap_labelled` needs: it
    calls this in a loop on an already-failing path, where an exception
    raised for one container would abandon every container after it.

    The return value is load-bearing, not decorative. §6.5 rule 3 puts the
    destroy BEFORE the host read precisely so a live container cannot race
    the diff a human is about to trust; when the removal is unconfirmed that
    ordering did not hold, and the caller owes the human a caveat rather
    than a diff presented as race-free.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            container,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        return f"docker rm -f could not start: {type(exc).__name__}: {exc}"
    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _KILL_GRACE_S)
        # Killing the CLIENT settles nothing about the DAEMON: the removal
        # may still be in flight, may have failed, or may never have been
        # accepted. Unknown is reported as unknown.
        return f"docker rm -f did not finish within {timeout_s:g}s"
    if rc != 0:
        return f"docker rm -f exited {rc}"
    return None


def _force_writable(root: str) -> None:
    """Restore search/write permission on directories under `root`.

    `rm -rf` cannot unlink entries out of a mode-000 directory, so a sandbox
    that ran `chmod 000 somedir` would otherwise leak the whole worktree.
    Everything here is owned by the host user (the §6.7 spike measured that
    even a root-in-container write lands owned by the real macOS user), so
    the chmod succeeds. Symlinked directories are skipped, never chmod'd:
    following one would change the mode of something OUTSIDE the worktree.
    """
    with contextlib.suppress(OSError):
        os.chmod(root, 0o700)
    for dirpath, dirnames, _files in os.walk(root, topdown=True):
        for name in dirnames:
            child = os.path.join(dirpath, name)
            if os.path.islink(child):
                continue
            with contextlib.suppress(OSError):
                os.chmod(child, 0o700)


def _remove_tree(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.islink(path):
        # Unlink the LINK. rmtree refuses a symlink root outright, and
        # following one would delete something outside the worktree.
        with contextlib.suppress(OSError):
            os.unlink(path)
        return
    shutil.rmtree(path, ignore_errors=True)
    if os.path.lexists(path):
        _force_writable(path)
        shutil.rmtree(path, ignore_errors=True)


def teardown_worktree(repo: str, worktree: str) -> list[str]:
    """Layered, unconditional. Returns a list of residue problems ([] == clean).

    `worktree` must be the literal path `create_worktree` returned. It is
    never re-derived from git and never re-resolved through the worktree's
    own contents, so nothing the sandbox planted can redirect step 2.
    """
    problems: list[str] = []
    if not os.path.isabs(worktree) or os.path.dirname(worktree) in ("", "/"):
        # An `rm -rf` primitive should refuse a path that could only be a
        # caller bug. Never reachable from create_worktree's own output.
        return [f"refusing to remove an implausible worktree path: {worktree}"]

    # 1. best-effort; FAILS after .git tampering (validation error 10) —
    #    expected, and the reason nothing below is conditional on it. Runs
    #    against the REAL repo so no git ever touches the worktree (§6.5).
    _git_quiet(repo, "worktree", "remove", "--force", worktree)
    # 2. unconditional: the LITERAL recorded path.
    _remove_tree(worktree)
    # 3. drop the registration.
    _git_quiet(repo, "worktree", "prune")
    # 4. verify.
    if os.path.lexists(worktree):
        problems.append(f"worktree path still exists: {worktree}")
    listing = _git_quiet(repo, "worktree", "list", "--porcelain")
    if listing.returncode != 0:
        problems.append(
            f"could not verify worktree registration in {repo}: "
            f"{listing.stderr.strip()}"
        )
    elif worktree in listing.stdout:
        problems.append(f"worktree still registered in {repo}")
    # 5. the private parent this module created. Pure string work on the
    #    recorded path, name-guarded, and `rmdir` refuses a non-empty
    #    directory — so it can only ever remove our own empty leftover.
    parent = os.path.dirname(worktree)
    if os.path.basename(parent).startswith(_WT_PREFIX):
        with contextlib.suppress(OSError):
            os.rmdir(parent)
    return problems
