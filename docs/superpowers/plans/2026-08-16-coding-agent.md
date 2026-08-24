# coding_agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `coding_agent` / `coding_agent_result` MCP tools specified in `docs/superpowers/specs/2026-08-16-coding-agent-design.md` — a local model runs an autonomous coding loop inside a network-less Docker container over a throwaway git worktree, and returns a diff that Claude gates.

**Architecture:** The orchestration loop runs in the MCP server process on the host; it calls Ollama, executes each requested tool call (three file tools on the host, `run_command` inside the container via `docker exec`), and stops on hard caps or no-progress. **No host-side `git` ever runs inside the worktree** — the diff and no-progress hash are byte-level filesystem walks compared against the base tree read from the *real* repo's object store. The container is destroyed before any host read of the worktree. Nothing is applied to the user's repo.

**Tech Stack:** Python 3.12 (stdlib `asyncio`, `subprocess`, `os`, `hashlib`, `difflib`), `pathspec` (gitignore evaluation, NEW dependency), Docker Desktop for Mac (`docker` CLI, `--init --network=none`), git 2.55 (real-repo reads only). Tests: `unittest` via `uv run --with pytest pytest`, matching `test_local_delegate.py`.

## Global Constraints

Copied from the spec; every task's requirements include these.

- **No host-side `git` command may ever run with the worktree as cwd or gitdir.** All git reads use `git -C <repo>` against the real repository (§6.5 rule 1, 2). This is the security spine; a task that violates it is wrong regardless of tests.
- The container is `docker rm -f`'d **before** any host read of the worktree that builds the returned result (§6.5 rule 3).
- Diff/hash walk: `lstat` every entry; symlink → target TEXT rendered as git `120000` (every target line `+`-prefixed, accurate `@@` counts), never dereferenced, never descended; regular files only (skip FIFO/socket/device); top-level `.git` opaque (§6.5 rule 2).
- Ignore handling is **tracked-aware**: every base-tree path appears in the diff regardless of ignore; ignore applies only to new untracked paths (§6.5 rule 2, §5.1).
- `write_file`/`read_file` path safety on the host: `realpath` of the **parent** for not-yet-existing files, prefix-check `realpath(target).startswith(realpath(root) + os.sep)` (§6, Gemini pass 2).
- Cleanup is layered and shielded+bounded: `docker rm -f` → best-effort `git -C <repo> worktree remove --force` → unconditional `rm -rf <literal recorded path>` → `git -C <repo> worktree prune` → verify (§6.6). `asyncio.shield(asyncio.wait_for(cleanup, T))`.
- Exactly one coding-agent container at a time via a **dedicated single-slot lock**; a second call is **rejected**, not queued (§6 item 4). Do NOT reuse `_DELEGATE_JOB_CAP` (=4, shared).
- Termination: first of `max_turns` (default 25, hard max 60), `max_seconds` (default 600, hard max 1800), no-progress (`N=5` turns with no change in walk-hash, `(cmd, exit)` pair, or `write_file`), `completed`, error (§7).
- Docker run flags (§6.2): `--rm --init --network=none <USER-FLAG per §6.7 spike> --read-only --tmpfs /tmp --tmpfs /home/agent -e HOME=/home/agent -e TMPDIR=/tmp -e UV_CACHE_DIR=/tmp/uv -e PIP_CACHE_DIR=/tmp/pip -e NPM_CONFIG_CACHE=/tmp/npm --cpus <cap> --memory <cap> --pids-limit <cap> -v <worktree>:/work:rw -w /work <image> sleep infinity`.
- New code lives in a new package `coding_agent/`, NOT in `mcp_server.py` (3,285 lines). `mcp_server.py` gains only tool registration + dispatch.
- Repo conventions: `ruff format` + `ruff check .` clean (this IS the repo's
  convention and is clean at baseline). **`mypy --strict` applies to
  `coding_agent/` ONLY** — ruled 2026-08-17: mypy is not a repo convention
  (no config, not in CI, not in AGENTS.md) and `mcp_server.py` carries 37
  pre-existing `--strict` errors. Never run `--strict` against
  `mcp_server.py`; never 'fix' those 37 — that is out of scope.
  Tests are stdlib `unittest` run via `uv run --with pytest pytest <file> -q`.
- Version bump `mcpb/manifest.json` **and** `.claude-plugin/plugin.json` in lockstep (currently both `1.5.4` → this ships as `1.6.0`, a new tool = minor bump).
- Delivery tier: **T2** per the review-pipeline convergence policy (introduces arbitrary execution into a server that had none).

---

## File Structure

| File | Responsibility | Why its own file |
|---|---|---|
| `coding_agent/__init__.py` | Public entry points: `run_coding_agent(...)`, `collect_result(job_id)`; re-exports | Thin surface for `mcp_server.py` to import |
| `coding_agent/walk.py` | **The byte-walk**: `snapshot_tree(root, ignore) -> TreeSnapshot`, `tree_hash`, `unified_diff(base, snap)`, symlink `120000` rendering, tracked-aware ignore | Every demonstrated exploit (passes 1–4) lives here; must be holdable in one head |
| `coding_agent/basetree.py` | Read the base ref's tree from the **real** repo: `read_base_tree(repo, ref) -> BaseTree` via `git -C <repo> ls-tree -r -z` + `cat-file`; capture nested `.gitignore` rules | Isolates the *only* place git runs, and it runs against the trusted repo |
| `coding_agent/sandbox.py` | Worktree + container lifecycle: `create_worktree`, `start_container`, `exec_in_container`, `destroy_container`, `teardown_worktree` (layered) | Docker/git process management, no model logic |
| `coding_agent/tools.py` | The 4 tool schemas as Ollama tool defs; host-side handlers `list_files`/`read_file`/`write_file` with path safety; `run_command` delegating to sandbox | Model-facing contract + the host path guard |
| `coding_agent/loop.py` | Orchestration: build messages, call Ollama, dispatch tools, stop conditions, transcript, result assembly, single-slot lock, background job registry | Control flow only; imports the above |
| `test_coding_agent_walk.py` | Unit tests for walk/basetree (no docker) | Pure functions, fast |
| `test_coding_agent_security.py` | **§10 security regression tests** — one per demonstrated attack | A reviewer reads this file alone and sees the threat model |
| `test_coding_agent_loop.py` | Stop conditions, no-progress detector, lock, result shape (fake Ollama, fake sandbox) | Control-flow tests |
| `test_coding_agent_integration.py` | Docker-required; skipped if daemon absent | End-to-end + cleanup-on-crash + ceiling-mid-command |
| `scripts/coding-agent-image/Dockerfile` + `build.sh` + `check.sh` | The pinned toolchain image and the `--check` drift report (§8.2) | Operability artifact |
| `docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md` | §6.7 spike result | Decides the `<USER-FLAG>` before container code is finalized |

---

### Task 0: §6.7 spike — settle `--user` on Docker Desktop for macOS (FIRST, before any container code)

**Files:**
- Create: `docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md`

**Interfaces:**
- Produces: a decided `USER_FLAG: list[str]` value (`[]`, `["--user", "0:0"]`, or `["--user", "<uid>:<gid>"]`) that Task 5 hardcodes as `SANDBOX_USER_FLAG`.

Why first: the spec's `--user <host-uid>` rationale is Linux-native behaviour asserted for a VM-backed platform (virtiofs synthesizes ownership). Container code that bakes in the wrong flag will fail on first real use.

- [ ] **Step 1: Confirm the daemon is up and note the mount driver**

Run:
```bash
docker info 2>/dev/null | grep -iE 'Operating System|OSType|Server Version|virtiofs|gRPC'; id -u; id -g
```
Expected: `Operating System: Docker Desktop`, a server version, and your uid/gid (501/20 on this Mac). Record all four in the spike doc.

- [ ] **Step 2: Create a scratch git worktree to mount, exactly as the tool will**

Run:
```bash
cd ~/Documents/Code/ai-tools-mcp && git worktree add /tmp/ca-spike-wt HEAD >/dev/null && ls -la /tmp/ca-spike-wt | head -3
```
Expected: worktree created; files owned by your uid.

- [ ] **Step 3: Probe write + git behaviour under three `--user` choices**

Run this once per variant, substituting `USERFLAG` with (a) nothing, (b) `--user 0:0`, (c) `--user 501:20`:
```bash
docker run --rm --init --network=none USERFLAG --read-only --tmpfs /tmp --tmpfs /home/agent -e HOME=/home/agent -v /tmp/ca-spike-wt:/work:rw -w /work python:3.12-slim sh -c '
  echo "== id ==";      id
  echo "== stat ==";    stat -c "%u:%g %n" /work/mcp_server.py
  echo "== write ==";   (echo probe > /work/.spike-write && echo WRITE_OK) || echo WRITE_FAIL
  echo "== git ==";     (apt-get -qq update >/dev/null 2>&1 || true); which git >/dev/null || echo "(no git in slim; skip)"
' 2>&1
```
Expected: record, per variant, the container `id`, the synthesized ownership `stat` shows, and `WRITE_OK`/`WRITE_FAIL`. (Ignore the git line for `python:3.12-slim`; git-in-container is tested with the real image in Task 11.)

- [ ] **Step 4: Decide and record**

Write `docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md` containing: the four facts from Step 1, the per-variant table from Step 3, and one line: `DECISION: SANDBOX_USER_FLAG = <exact list>` chosen by this rule — prefer the variant that gives `WRITE_OK` **and** does not run as container-root; if only `--user 0:0` writes, that is acceptable (root-in-container is not host root under Docker Desktop) and must be stated. Clean up: `git worktree remove --force /tmp/ca-spike-wt`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md
git commit -m "spike(coding-agent): settle --user on Docker Desktop/macOS before container code (§6.7)"
```

---

### Task 1: Prerequisite — fix the stale delegate allowlist + declare `pathspec`

**Files:**
- Modify: `mcp_server.py:1591-1597` (allowlist), `mcp_server.py:2-19` (PEP 723 deps)
- Test: `test_local_delegate.py` (existing allowlist assertions)

**Interfaces:**
- Produces: `_OLLAMA_BUILTIN_DELEGATE_MODELS` containing `gemma4:31b-nvfp4` (which `coding_agent` resolves against as its ceiling).

Why in this delivery: the allowlist still lists four qwen3.6 tags **deleted from every machine on 2026-08-16** and lacks the model this tool uses. Latent bug; the spec (§12) makes fixing it a prerequisite.

- [ ] **Step 1: Write the failing test**

Append to `test_local_delegate.py` inside an existing `unittest.TestCase` class that imports `mcp_server` (e.g. next to the other allowlist tests):
```python
    def test_builtin_allowlist_reflects_installed_fleet(self):
        models = mcp_server._OLLAMA_BUILTIN_DELEGATE_MODELS
        self.assertIn("gemma4:12b-nvfp4", models)
        self.assertIn("gemma4:31b-nvfp4", models)
        self.assertIn("qwen3.8:27b-nvfp4", models)
        for m in models:
            self.assertNotIn("qwen3.6", m, f"retired tag still allowlisted: {m}")
        self.assertEqual(models[0], "gemma4:12b-nvfp4")  # default preserved
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest test_local_delegate.py -q -k allowlist_reflects`
Expected: FAIL — `gemma4:31b-nvfp4` not in tuple.

- [ ] **Step 3: Update the allowlist and the dependency block**

In `mcp_server.py` replace the tuple body:
```python
_OLLAMA_BUILTIN_DELEGATE_MODELS: tuple[str, ...] = (
    "gemma4:12b-nvfp4",   # local_delegate default; cli-updates gate; Signal
    "gemma4:31b-nvfp4",   # JVMBPro review/delegate default; coding_agent
    "qwen3.8:27b-nvfp4",  # coding-assistant role (Qwen Code CLI)
)
```
In the PEP 723 block add one line after `"requests>=2.31",`:
```python
#     "pathspec>=0.12",
```

- [ ] **Step 4: Run the whole file to verify it passes and nothing else broke**

Run: `uv run --with pytest pytest test_local_delegate.py -q`
Expected: all pass. If any *other* test asserted a `qwen3.6` tag, update that assertion to `qwen3.8:27b-nvfp4` — those tags no longer exist anywhere.

- [ ] **Step 5: Commit**

```bash
git add mcp_server.py test_local_delegate.py
git commit -m "fix(local_delegate): allowlist reflects the installed fleet; add pathspec dep

qwen3.6 tags were deleted from every machine 2026-08-16; gemma4:31b-nvfp4
is the live default and coding_agent resolves against this list."
```

---

### Task 2: `walk.py` — snapshot a tree WITHOUT following symlinks (the exfil fix, test-first)

**Files:**
- Create: `coding_agent/__init__.py` (empty for now), `coding_agent/walk.py`
- Test: `test_coding_agent_security.py`, `test_coding_agent_walk.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Entry:
      path: str          # POSIX relative path
      kind: str          # "file" | "symlink"
      data: bytes        # file bytes, or symlink TARGET as utf-8 bytes
  @dataclass(frozen=True)
  class TreeSnapshot:
      entries: dict[str, Entry]
  def snapshot_tree(root: str, is_ignored: Callable[[str], bool]) -> TreeSnapshot
  def tree_hash(snap: TreeSnapshot) -> str   # sha256 hex, deterministic
  ```

- [ ] **Step 1: Write the failing security test (§10 test 4 — symlink read-exfil)**

Create `test_coding_agent_security.py`:
```python
#!/usr/bin/env python3
"""Security regression tests for coding_agent.

Every test here pins an attack that was DEMONSTRATED (not hypothesised)
during adversarial review of the design spec, 2026-08-16. Read
docs/superpowers/specs/2026-08-16-coding-agent-design.md §6.5/§6.6/§10.

Run:  uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q
"""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path

from coding_agent.walk import snapshot_tree, tree_hash


class SymlinkExfiltration(unittest.TestCase):
    """Pass 3: a naive os.walk+open().read() dereferenced a planted
    /work/leak -> ~/.ssh/id_ed25519 and read the host secret INTO the diff."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.secret = Path(self.tmp) / "outside" / "id_ed25519"
        self.secret.parent.mkdir()
        self.secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETBYTES\n")
        self.root = Path(self.tmp) / "work"
        self.root.mkdir()
        (self.root / "ok.py").write_text("print(1)\n")
        os.symlink(str(self.secret), self.root / "leak_abs")
        os.symlink("../outside/id_ed25519", self.root / "leak_rel")
        os.symlink(str(self.secret.parent), self.root / "leak_dir")

    def test_symlink_targets_are_recorded_as_text_never_dereferenced(self):
        snap = snapshot_tree(str(self.root), lambda p: False)
        for name in ("leak_abs", "leak_rel", "leak_dir"):
            e = snap.entries[name]
            self.assertEqual(e.kind, "symlink")
            self.assertNotIn(b"SECRETBYTES", e.data)
        # the secret's bytes appear NOWHERE in the snapshot
        blob = b"".join(e.data for e in snap.entries.values())
        self.assertNotIn(b"SECRETBYTES", blob)
        # a symlinked directory is not descended
        self.assertNotIn("leak_dir/id_ed25519", snap.entries)

    def test_hash_invariant_to_target_contents_sensitive_to_target_string(self):
        h1 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.secret.write_bytes(b"DIFFERENT SECRET")           # contents change
        h2 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertEqual(h1, h2, "hash must not depend on symlink TARGET contents")
        os.unlink(self.root / "leak_abs")
        os.symlink("/somewhere/else", self.root / "leak_abs")   # target STRING changes
        h3 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertNotEqual(h1, h3)

    def test_special_files_are_skipped_not_read(self):
        os.mkfifo(self.root / "trap.fifo")   # open().read() here would block forever
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertNotIn("trap.fifo", snap.entries)

    def test_filter_attribute_escape_cannot_run(self):
        """Spec §10 test 2. The filter vector needs host git to honour an
        in-tree .gitattributes plus a repo-local [filter] config. Nothing in
        the walk invokes git, so the payload can never fire. Written rather
        than 'subsumed' because it is the REGRESSION GUARD if host git is ever
        reintroduced (ruled 2026-08-17)."""
        marker = Path(self.tmp) / "PWNED_FILTER"
        (self.root / ".gitattributes").write_text("*.py filter=pwn\n")
        (self.root / ".git").mkdir(exist_ok=True)
        (self.root / ".git" / "config").write_text(
            '[filter "pwn"]\n\tclean = sh -c "touch ' + str(marker) + '"\n'
        )
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertFalse(marker.exists(), "a filter driver executed during the walk")
        # .gitattributes is DATA to the walk, never configuration
        self.assertEqual(snap.entries[".gitattributes"].data, b"*.py filter=pwn\n")

    def test_concurrent_writer_toctou_cannot_win(self):
        """Spec §10 test 3. A writer re-mangling .git mid-run defeated the
        earlier check-then-invoke design (15 host executions in 20s). With no
        invocation at all there is no window. Guards a future reintroduction."""
        import threading

        marker = Path(self.tmp) / "PWNED_TOCTOU"
        stop = threading.Event()

        def remangle() -> None:
            g = self.root / ".git"
            while not stop.is_set():
                try:
                    if g.is_dir():
                        (g / "config").write_text(
                            '[core]\n\tfsmonitor = sh -c "touch '
                            + str(marker)
                            + '"\n'
                        )
                    else:
                        g.mkdir(exist_ok=True)
                except OSError:
                    pass

        t = threading.Thread(target=remangle, daemon=True)
        t.start()
        try:
            for _ in range(50):
                snapshot_tree(str(self.root), lambda p: False)
        finally:
            stop.set()
            t.join(timeout=2)
        self.assertFalse(marker.exists(), "TOCTOU payload executed during the walk")

    def test_top_level_dot_git_is_opaque(self):
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("[core]\n\tfsmonitor = /tmp/pwn.sh\n")
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertFalse(any(p == ".git" or p.startswith(".git/") for p in snap.entries))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q`
Expected: FAIL — `ModuleNotFoundError: coding_agent`.

- [ ] **Step 3: Implement `walk.py`**

Create `coding_agent/__init__.py` (empty) and `coding_agent/walk.py`:
```python
"""Byte-level tree snapshot for coding_agent.

SECURITY SPINE (spec §6.5): this module NEVER invokes git and NEVER
dereferences a symlink. It replaced `git add -A`/`git diff` after adversarial
review demonstrated host RCE via a sandbox-controlled `.git`; a later pass
demonstrated that a naive os.walk+read() then leaked host secrets through
planted symlinks. Both are pinned by test_coding_agent_security.py.
"""
from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str        # "file" | "symlink"
    data: bytes      # file bytes, or the symlink TARGET string as utf-8


@dataclass(frozen=True)
class TreeSnapshot:
    entries: dict[str, Entry]


def snapshot_tree(root: str, is_ignored: Callable[[str], bool]) -> TreeSnapshot:
    """Walk `root` with lstat semantics.

    - symlinks are recorded as their target TEXT and never followed
      (git's 120000 mode); symlinked directories are not descended
    - only regular files are read; FIFOs/sockets/devices are skipped
    - the top-level `.git` entry is opaque (never read, never listed)
    - `is_ignored(relpath)` gates NEW paths only; the caller enforces
      tracked-awareness (basetree paths are always included) — see diff()
    """
    root = os.path.abspath(root)
    entries: dict[str, Entry] = {}

    def rel(p: str) -> str:
        return os.path.relpath(p, root).replace(os.sep, "/")

    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            names = sorted(os.listdir(cur))
        except OSError:
            continue
        for name in names:
            full = os.path.join(cur, name)
            r = rel(full)
            if r == ".git" or r.startswith(".git/"):
                continue
            try:
                st = os.lstat(full)          # lstat: never follows
            except OSError:
                continue
            mode = st.st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(full)
                if not is_ignored(r):
                    entries[r] = Entry(r, "symlink", target.encode("utf-8", "surrogateescape"))
                continue                     # never descend a symlinked dir
            if stat.S_ISDIR(mode):
                if not is_ignored(r + "/"):
                    stack.append(full)
                continue
            if not stat.S_ISREG(mode):
                continue                     # FIFO / socket / device: skip
            if is_ignored(r):
                continue
            with open(full, "rb") as fh:
                entries[r] = Entry(r, "file", fh.read())
    return TreeSnapshot(entries)


def tree_hash(snap: TreeSnapshot) -> str:
    h = hashlib.sha256()
    for path in sorted(snap.entries):
        e = snap.entries[path]
        h.update(e.kind.encode()); h.update(b"\0")
        h.update(path.encode("utf-8", "surrogateescape")); h.update(b"\0")
        h.update(hashlib.sha256(e.data).digest())
    return h.hexdigest()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q`
Expected: 6 passed.

- [ ] **Step 5: Add a plain unit test file for the happy path**

Create `test_coding_agent_walk.py`:
```python
#!/usr/bin/env python3
"""Unit tests for coding_agent.walk (no docker, no git).
Run:  uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from coding_agent.walk import snapshot_tree, tree_hash


class SnapshotBasics(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "a.py").write_text("x = 1\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.txt").write_text("hi\n")

    def test_regular_files_are_read(self):
        snap = snapshot_tree(str(self.root), lambda p: False)
        self.assertEqual(snap.entries["a.py"].data, b"x = 1\n")
        self.assertEqual(snap.entries["sub/b.txt"].data, b"hi\n")

    def test_ignore_callback_gates_new_paths(self):
        snap = snapshot_tree(str(self.root), lambda p: p.endswith(".txt"))
        self.assertIn("a.py", snap.entries)
        self.assertNotIn("sub/b.txt", snap.entries)

    def test_hash_is_deterministic_and_content_sensitive(self):
        h1 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        h2 = tree_hash(snapshot_tree(str(self.root), lambda p: False))
        self.assertEqual(h1, h2)
        (self.root / "a.py").write_text("x = 2\n")
        self.assertNotEqual(h1, tree_hash(snapshot_tree(str(self.root), lambda p: False)))


if __name__ == "__main__":
    unittest.main()
```
Run: `uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q` → 3 passed.

- [ ] **Step 6: Lint/type and commit**

Run: `ruff format coding_agent test_coding_agent_*.py && ruff check coding_agent test_coding_agent_*.py && mypy --strict coding_agent`
Expected: clean.
```bash
git add coding_agent/__init__.py coding_agent/walk.py test_coding_agent_security.py test_coding_agent_walk.py
git commit -m "feat(coding-agent): byte-walk tree snapshot that never follows symlinks (spec §6.5)

Pins the pass-3 demonstrated read-exfil: symlinks recorded as target text,
never dereferenced or descended; special files skipped; top-level .git opaque."
```

---

### Task 3: `basetree.py` — read the base ref from the REAL repo + nested `.gitignore` (git only here, only against the trusted repo)

**Files:**
- Create: `coding_agent/basetree.py`
- Test: `test_coding_agent_walk.py` (append)

**Interfaces:**
- Consumes: `Entry` from Task 2.
- Produces:
  ```python
  @dataclass(frozen=True)
  class BaseTree:
      entries: dict[str, Entry]        # path -> Entry (kind file|symlink)
      ignore: Callable[[str], bool]    # tracked-UNaware raw pathspec matcher
      tracked: frozenset[str]          # every path in the base tree
  def read_base_tree(repo: str, ref: str) -> BaseTree
  ```
  and the tracked-aware predicate callers must use:
  ```python
  def make_ignore(base: BaseTree) -> Callable[[str], bool]:
      # returns True ONLY for paths NOT in base.tracked that match ignore rules
  ```

Why git is allowed here: `repo` is the user's own trusted checkout, passed as the MCP `repo` parameter — never derived from the sandbox-controlled worktree. Every command is `git -C <repo>`.

- [ ] **Step 1: Write the failing tests (incl. §10 test 4c — tracked file not blinded)**

Append to `test_coding_agent_walk.py`:
```python
import subprocess
from coding_agent.basetree import make_ignore, read_base_tree


def _mkrepo(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "T"], check=True)


class BaseTreeRead(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _mkrepo(self.repo)
        (self.repo / ".gitignore").write_text("*.secret\nignored-dir/\n")
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg" / ".gitignore").write_text("*.log\n")
        (self.repo / "app.py").write_text("print(1)\n")
        (self.repo / "app.secret").write_text("tracked-despite-pattern\n")   # force-added
        (self.repo / "ignored-dir").mkdir()
        (self.repo / "ignored-dir" / "keep.py").write_text("kept\n")           # force-added
        os.symlink("app.py", self.repo / "link")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A", "-f"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "base"], check=True)

    def test_reads_files_symlinks_and_tracked_set(self):
        base = read_base_tree(str(self.repo), "HEAD")
        self.assertEqual(base.entries["app.py"].data, b"print(1)\n")
        self.assertEqual(base.entries["link"].kind, "symlink")
        self.assertEqual(base.entries["link"].data, b"app.py")
        self.assertIn("app.secret", base.tracked)
        self.assertIn("ignored-dir/keep.py", base.tracked)

    def test_ignore_is_tracked_aware_so_gate_cannot_be_blinded(self):
        """Pass 4: uniform ignore hid edits to TRACKED files matching a base
        pattern. git's rule: ignore never applies to tracked files."""
        base = read_base_tree(str(self.repo), "HEAD")
        ign = make_ignore(base)
        self.assertFalse(ign("app.secret"))            # tracked -> shown
        self.assertFalse(ign("ignored-dir/keep.py"))   # tracked -> shown
        self.assertTrue(ign("new.secret"))             # untracked + matches -> ignored
        self.assertTrue(ign("ignored-dir/new.py"))     # untracked under ignored dir
        self.assertTrue(ign("pkg/debug.log"))          # NESTED .gitignore honoured
        self.assertFalse(ign("pkg/main.py"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q -k BaseTree`
Expected: FAIL — `ModuleNotFoundError: coding_agent.basetree`.

- [ ] **Step 3: Implement `basetree.py`**

```python
"""Read the base ref's tree from the user's REAL repository.

The ONLY module in coding_agent that runs git — and only ever as
`git -C <repo>` against the trusted checkout named by the MCP `repo`
parameter. It must NEVER be pointed at the sandbox worktree (spec §6.5).
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import pathspec

from .walk import Entry

_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}


@dataclass(frozen=True)
class BaseTree:
    entries: dict[str, Entry]
    ignore: Callable[[str], bool]
    tracked: frozenset[str]


def _git(repo: str, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, env={**_GIT_ENV}
    ).stdout


def read_base_tree(repo: str, ref: str) -> BaseTree:
    raw = _git(repo, "ls-tree", "-r", "-z", ref)
    entries: dict[str, Entry] = {}
    ignore_sources: list[tuple[str, str]] = []      # (dir_prefix, text)
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, path_b = rec.split(b"\t", 1)
        mode, _typ, sha = meta.decode().split(" ")
        path = path_b.decode("utf-8", "surrogateescape")
        data = _git(repo, "cat-file", "blob", sha)
        if mode == "120000":
            entries[path] = Entry(path, "symlink", data)
        elif mode in ("100644", "100755"):
            entries[path] = Entry(path, "file", data)
        # 160000 (submodule) and 040000 are ignored: not files
        if path == ".gitignore" or path.endswith("/.gitignore"):
            prefix = path[: -len(".gitignore")]
            ignore_sources.append((prefix, data.decode("utf-8", "replace")))
    # also honour .git/info/exclude of the REAL repo (host-owned)
    try:
        excl = _git(repo, "rev-parse", "--git-path", "info/exclude").decode().strip()
        with open(excl, encoding="utf-8", errors="replace") as fh:
            ignore_sources.append(("", fh.read()))
    except (subprocess.CalledProcessError, OSError):
        pass
    specs = [(p, pathspec.PathSpec.from_lines("gitwildmatch", t.splitlines()))
             for p, t in ignore_sources]

    def raw_ignore(path: str) -> bool:
        for prefix, spec in specs:
            if prefix and not path.startswith(prefix):
                continue
            sub = path[len(prefix):] if prefix else path
            if spec.match_file(sub):
                return True
        return False

    return BaseTree(entries=entries, ignore=raw_ignore, tracked=frozenset(entries))


def make_ignore(base: BaseTree) -> Callable[[str], bool]:
    """Tracked-aware ignore: NEVER ignores a path in the base tree."""
    def ign(path: str) -> bool:
        p = path.rstrip("/")
        if p in base.tracked:
            return False
        # a directory prefix containing tracked files must be walked
        if path.endswith("/") and any(t.startswith(path) for t in base.tracked):
            return False
        return base.ignore(path)
    return ign
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_walk.py -q`
Expected: all pass (3 + 2).

- [ ] **Step 5: Lint/type and commit**

Run: `ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent`
```bash
git add coding_agent/basetree.py test_coding_agent_walk.py
git commit -m "feat(coding-agent): read base tree from the REAL repo; tracked-aware nested-gitignore (spec §6.5)

Pins pass-4: ignore never applies to tracked paths, so an edit to a tracked
file matching a base pattern cannot be hidden from the gate."
```

---

### Task 4: `walk.py` — the unified diff with git-`120000` symlink rendering + size cap (§10 tests 4b, 4c-typechange)

**Files:**
- Modify: `coding_agent/walk.py`
- Test: `test_coding_agent_security.py`, `test_coding_agent_walk.py`

**Interfaces:**
- Consumes: `TreeSnapshot`, `BaseTree`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class DiffResult:
      text: str                 # unified diff, possibly truncated
      truncated: bool
      changed_files: list[str]  # sorted
      full_path: str | None     # where the untruncated diff was written when truncated
  def unified_diff(base: BaseTree, snap: TreeSnapshot, *, max_bytes: int, spill_dir: str) -> DiffResult
  ```

- [ ] **Step 1: Write the failing security tests**

Append to `test_coding_agent_security.py`:
```python
from coding_agent.basetree import BaseTree
from coding_agent.walk import Entry, TreeSnapshot, unified_diff


def _base(**files: bytes) -> BaseTree:
    ents = {p: Entry(p, "file", d) for p, d in files.items()}
    return BaseTree(entries=ents, ignore=lambda p: False, tracked=frozenset(ents))


class DiffStructureInjection(unittest.TestCase):
    """Pass 4: a symlink TARGET is attacker-controlled and round-trips
    newlines + '--- a/' bytes. It must render like git's 120000 blob:
    every target line '+'-prefixed, so no forged header lands in column 0."""

    def test_forged_header_in_symlink_target_cannot_reach_column_zero(self):
        forged = "harmless\n--- a/etc/shadow\n+++ b/etc/shadow\n@@ -0,0 +1 @@\n+root:INJECTED"
        snap = TreeSnapshot({"leak": Entry("leak", "symlink", forged.encode())})
        d = unified_diff(_base(), snap, max_bytes=1 << 20, spill_dir=tempfile.mkdtemp())
        for line in d.text.splitlines():
            if line.startswith(("--- ", "+++ ", "@@ ")):
                # only OUR headers may appear at column 0
                self.assertTrue(
                    line.startswith(("--- /dev/null", "--- a/leak", "+++ b/leak", "@@ -0,0 +")),
                    f"forged header reached column 0: {line!r}",
                )
        self.assertIn("+--- a/etc/shadow", d.text)   # rendered as content, '+'-prefixed


class GateCannotBeBlinded(unittest.TestCase):
    def test_tracked_file_replaced_by_special_file_surfaces_as_deletion(self):
        base = _base(**{"foo.py": b"print(1)\n"})
        snap = TreeSnapshot({})     # walk skipped the FIFO that replaced foo.py
        d = unified_diff(base, snap, max_bytes=1 << 20, spill_dir=tempfile.mkdtemp())
        self.assertIn("foo.py", d.changed_files)
        self.assertIn("--- a/foo.py", d.text)
        self.assertIn("+++ /dev/null", d.text)

    def test_new_untracked_file_appears(self):
        snap = TreeSnapshot({"tests/test_new.py": Entry("tests/test_new.py", "file", b"def test_x(): pass\n")})
        d = unified_diff(_base(), snap, max_bytes=1 << 20, spill_dir=tempfile.mkdtemp())
        self.assertIn("tests/test_new.py", d.changed_files)
        self.assertIn("+def test_x(): pass", d.text)


class DiffSizeCap(unittest.TestCase):
    def test_binary_is_annotated_not_embedded_and_large_text_truncates_with_spill(self):
        big = b"x" * 5000
        snap = TreeSnapshot({
            "blob.bin": Entry("blob.bin", "file", b"\x00\x01\x02" * 100),
            "big.txt": Entry("big.txt", "file", big),
        })
        spill = tempfile.mkdtemp()
        d = unified_diff(_base(), snap, max_bytes=1000, spill_dir=spill)
        self.assertIn("Binary files differ: blob.bin", d.text)
        self.assertNotIn("\x00", d.text)
        self.assertTrue(d.truncated)
        self.assertIsNotNone(d.full_path)
        self.assertGreater(os.path.getsize(d.full_path), 1000)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q`
Expected: FAIL — `unified_diff` not defined.

- [ ] **Step 3: Implement `unified_diff` in `walk.py`**

Append to `coding_agent/walk.py`:
```python
import difflib
import os as _os
import tempfile as _tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basetree import BaseTree

_TRUNC = "\n[... diff truncated by coding_agent size cap; full diff at {path} ...]\n"


@dataclass(frozen=True)
class DiffResult:
    text: str
    truncated: bool
    changed_files: list[str]
    full_path: str | None


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def _lines(data: bytes) -> list[str]:
    return data.decode("utf-8", "surrogateescape").splitlines(keepends=True)


def _render_symlink(data: bytes) -> list[str]:
    """git's 120000 rendering: the target text as content lines, so a
    forged '--- a/' inside the target becomes '+--- a/' — never column 0."""
    lines = _lines(data)
    return lines if lines else [""]


def _entry_lines(e: Entry | None) -> list[str]:
    if e is None:
        return []
    if e.kind == "symlink":
        return _render_symlink(e.data)
    return _lines(e.data)


def unified_diff(
    base: BaseTree, snap: TreeSnapshot, *, max_bytes: int, spill_dir: str
) -> DiffResult:
    paths = sorted(set(base.entries) | set(snap.entries))
    out: list[str] = []
    changed: list[str] = []
    for p in paths:
        b, s = base.entries.get(p), snap.entries.get(p)
        if b is not None and s is not None and b.kind == s.kind and b.data == s.data:
            continue
        changed.append(p)
        if (b is not None and b.kind == "file" and _is_binary(b.data)) or (
            s is not None and s.kind == "file" and _is_binary(s.data)
        ):
            out.append(f"Binary files differ: {p}\n")
            continue
        fromfile = f"a/{p}" if b is not None else "/dev/null"
        tofile = f"b/{p}" if s is not None else "/dev/null"
        out.extend(
            difflib.unified_diff(
                _entry_lines(b), _entry_lines(s), fromfile=fromfile, tofile=tofile, lineterm="\n"
            )
        )
    text = "".join(out)
    full_path: str | None = None
    truncated = False
    if len(text.encode("utf-8", "surrogateescape")) > max_bytes:
        fd, full_path = _tempfile.mkstemp(prefix="coding-agent-diff-", suffix=".patch", dir=spill_dir)
        with _os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(text)
        text = text.encode("utf-8", "surrogateescape")[:max_bytes].decode("utf-8", "ignore") + _TRUNC.format(path=full_path)
        truncated = True
    return DiffResult(text=text, truncated=truncated, changed_files=changed, full_path=full_path)
```
Note for the implementer: `difflib.unified_diff` emits `--- a/x`, `+++ b/x`, `@@ … @@` headers itself and `+`/`-`/` `-prefixes every content line — that is exactly the property test 4b asserts. Do not hand-roll headers.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py test_coding_agent_walk.py -q`
Expected: all pass.

- [ ] **Step 5: Lint/type and commit**

```bash
ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent
git add coding_agent/walk.py test_coding_agent_security.py
git commit -m "feat(coding-agent): unified diff with git-120000 symlink rendering, typechange visibility, size cap

Pins pass-4 header injection (target text can never reach column 0) and the
tracked-file-replaced-by-special-file deletion case; binary annotated not
embedded; oversize text truncates with a spill file."
```

---

### Task 5: `sandbox.py` — worktree + container lifecycle with layered cleanup (§6.2, §6.6)

**Files:**
- Create: `coding_agent/sandbox.py`
- Test: `test_coding_agent_security.py` (mangled-`.git` cleanup, git-only), `test_coding_agent_integration.py` (docker)

**Interfaces:**
- Consumes: `SANDBOX_USER_FLAG` decided in Task 0.
- Produces:
  ```python
  SANDBOX_USER_FLAG: list[str]           # from Task 0 spike, e.g. [] or ["--user","0:0"]
  @dataclass
  class Sandbox:
      repo: str; base_ref: str; worktree: str; container: str | None; image: str
  def create_worktree(repo: str, base_ref: str) -> str          # returns absolute path (record it!)
  async def start_container(worktree: str, image: str, *, cpus: str, memory: str, pids: int) -> str  # container id
  async def exec_in_container(container: str, cmd: str, *, timeout_s: float) -> tuple[int, str]  # (exit, combined output, truncated)
  async def destroy_container(container: str) -> None
  def teardown_worktree(repo: str, worktree: str) -> list[str]  # returns residue problems; [] == clean
  ```

- [ ] **Step 1: Write the failing security test (§10 test 5 — mangled `.git` cleanup, no docker)**

Append to `test_coding_agent_security.py`:
```python
import subprocess
from coding_agent.sandbox import create_worktree, teardown_worktree


class MangledGitCleanup(unittest.TestCase):
    """Pass 1: after the sandbox replaces the worktree's .git FILE with a
    directory, `git worktree remove --force` FAILS (validation error 10) and
    the worktree stays on disk AND registered. Cleanup must not depend on it."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "-C", str(self.repo), "init", "-q", "-b", "main"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "t@e.com"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "T"], check=True)
        (self.repo / "f").write_text("1\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "f"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "b"], check=True)

    def test_cleanup_leaves_zero_residue_after_dot_git_tampering(self):
        wt = create_worktree(str(self.repo), "HEAD")
        # sandbox mangles .git into a hostile directory
        os.remove(os.path.join(wt, ".git"))
        os.mkdir(os.path.join(wt, ".git"))
        with open(os.path.join(wt, ".git", "config"), "w") as fh:
            fh.write("[core]\n\tfsmonitor = /tmp/pwn.sh\n")
        problems = teardown_worktree(str(self.repo), wt)
        self.assertEqual(problems, [])
        self.assertFalse(os.path.exists(wt))
        listing = subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, check=True
        ).stdout
        self.assertNotIn(wt, listing)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q -k Mangled`
Expected: FAIL — `coding_agent.sandbox` missing.

- [ ] **Step 3: Implement `sandbox.py`**

```python
"""Worktree + container lifecycle for coding_agent (spec §6.2, §6.6).

Rules enforced here:
- every git command is `git -C <repo>` (the trusted real repo), never run
  with the worktree as cwd — spec §6.5
- teardown is layered and does not depend on `git worktree remove`
  succeeding: rm -rf the LITERAL recorded path, then prune, then verify
- the container is destroyed before any host read of the worktree; the
  loop enforces ordering, this module provides the primitives
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

# Decided by the §6.7 spike (docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md).
# Replace with the recorded DECISION value before merging.
SANDBOX_USER_FLAG: list[str] = []

_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
_OUTPUT_CAP = 64 * 1024
_TRUNC = "\n[... output truncated by coding_agent ...]\n"


@dataclass
class Sandbox:
    repo: str
    base_ref: str
    worktree: str
    container: str | None
    image: str


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo, *args], check=check, capture_output=True, text=True, env={**_GIT_ENV}
    )


def create_worktree(repo: str, base_ref: str) -> str:
    """Detached throwaway worktree of `base_ref`. Returns the ABSOLUTE path —
    callers must record it verbatim; teardown rm -rf's this literal string."""
    path = tempfile.mkdtemp(prefix="coding-agent-wt-")
    os.rmdir(path)  # git wants to create it
    _git(repo, "worktree", "add", "--detach", path, base_ref)
    return os.path.abspath(path)


async def start_container(
    worktree: str, image: str, *, cpus: str, memory: str, pids: int
) -> str:
    cmd = [
        "docker", "run", "-d", "--rm", "--init", "--network=none", *SANDBOX_USER_FLAG,
        "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/home/agent",
        "-e", "HOME=/home/agent", "-e", "TMPDIR=/tmp",
        "-e", "UV_CACHE_DIR=/tmp/uv", "-e", "PIP_CACHE_DIR=/tmp/pip", "-e", "NPM_CONFIG_CACHE=/tmp/npm",
        "--cpus", cpus, "--memory", memory, "--pids-limit", str(pids),
        "-v", f"{worktree}:/work:rw", "-w", "/work", image, "sleep", "infinity",
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {err.decode(errors='replace').strip()}")
    return out.decode().strip()


async def exec_in_container(container: str, cmd: str, *, timeout_s: float) -> tuple[int, str, bool]:
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", container, "sh", "-c", cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        rc = proc.returncode if proc.returncode is not None else 1
    except asyncio.TimeoutError:
        proc.kill()
        # --init (tini) reaps whatever the model backgrounded; kill the pgroup too
        await asyncio.create_subprocess_exec("docker", "exec", container, "sh", "-c", "kill -9 -1 2>/dev/null || true")
        return 124, "[command timed out]", False
    text = out.decode("utf-8", "replace")
    truncated = len(text) > _OUTPUT_CAP
    if truncated:
        text = text[:_OUTPUT_CAP] + _TRUNC
    return rc, text, truncated


async def destroy_container(container: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await proc.communicate()


def teardown_worktree(repo: str, worktree: str) -> list[str]:
    """Layered, unconditional. Returns a list of residue problems ([] == clean)."""
    problems: list[str] = []
    # 1. best-effort; FAILS after .git tampering (validation error 10) — expected
    _git(repo, "worktree", "remove", "--force", worktree, check=False)
    # 2. unconditional: the LITERAL recorded path, never re-resolved through the worktree
    shutil.rmtree(worktree, ignore_errors=True)
    # 3. drop the registration
    _git(repo, "worktree", "prune", check=False)
    # 4. verify
    if os.path.lexists(worktree):
        problems.append(f"worktree path still exists: {worktree}")
    listing = _git(repo, "worktree", "list", "--porcelain", check=False).stdout
    if worktree in listing:
        problems.append(f"worktree still registered in {repo}")
    return problems
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q -k Mangled`
Expected: PASS.

- [ ] **Step 5: Set `SANDBOX_USER_FLAG` from the Task 0 decision, lint, commit**

Edit the `SANDBOX_USER_FLAG` line to the exact `DECISION` value from `docs/superpowers/spikes/2026-08-16-docker-desktop-uid.md`.
```bash
ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent
git add coding_agent/sandbox.py test_coding_agent_security.py
git commit -m "feat(coding-agent): sandbox lifecycle with layered cleanup independent of git worktree remove (spec §6.6)

Pins pass-1: a mangled .git makes 'git worktree remove' fail; rm -rf of the
literal recorded path + prune leaves zero residue on disk or in the real
repo's registration."
```

---

### Task 6: `tools.py` — the 4 tool schemas + host-side path safety (symlinked-parent write escape)

**Files:**
- Create: `coding_agent/tools.py`
- Test: `test_coding_agent_security.py`

**Interfaces:**
- Consumes: `exec_in_container` from Task 5.
- Produces:
  ```python
  TOOL_SCHEMAS: list[dict]                       # Ollama "tools" payload
  class PathEscape(ValueError): ...
  def safe_path(root: str, rel: str, *, for_write: bool) -> str   # absolute, or raises PathEscape
  def tool_list_files(root: str, rel: str) -> str
  def tool_read_file(root: str, rel: str) -> str
  def tool_write_file(root: str, rel: str, content: str) -> str
  async def tool_run_command(container: str, cmd: str, *, timeout_s: float) -> tuple[int, str]
  ```

- [ ] **Step 1: Write the failing security test (Gemini pass 2 — symlinked parent)**

Append to `test_coding_agent_security.py`:
```python
from coding_agent.tools import PathEscape, safe_path, tool_write_file


class WriteFilePathEscape(unittest.TestCase):
    """A symlinked PARENT (`ln -s /etc sub`) makes realpath(dirname(sub/passwd))
    resolve to /etc. The parent must be realpath'd and prefix-checked."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.outside = self.tmp / "outside"; self.outside.mkdir()
        self.root = self.tmp / "work"; self.root.mkdir()
        os.symlink(str(self.outside), self.root / "sub")

    def test_write_through_symlinked_parent_is_rejected(self):
        with self.assertRaises(PathEscape):
            tool_write_file(str(self.root), "sub/passwd", "pwned")
        self.assertFalse((self.outside / "passwd").exists())

    def test_dotdot_and_absolute_are_rejected(self):
        for bad in ("../x", "/etc/passwd", "a/../../x"):
            with self.assertRaises(PathEscape):
                safe_path(str(self.root), bad, for_write=True)

    def test_new_file_in_real_subdir_is_allowed(self):
        (self.root / "real").mkdir()
        p = safe_path(str(self.root), "real/new.py", for_write=True)
        self.assertTrue(p.startswith(os.path.realpath(str(self.root)) + os.sep))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q -k PathEscape`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `tools.py`**

```python
"""The four tools the model sees, and the host-side path guard (spec §5, §6).

list_files/read_file/write_file run ON THE HOST against the worktree path.
Only run_command crosses into the container.
"""
from __future__ import annotations

import os

from .sandbox import exec_in_container

MAX_READ = 256 * 1024
MAX_LIST = 2000

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {"name": "list_files", "description": "Recursively list files under a directory of the repository (respects .gitignore).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Overwrite (or create) a text file in the repository with the given content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the repository root (tests, linters). No network is available.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
]


class PathEscape(ValueError):
    pass


def safe_path(root: str, rel: str, *, for_write: bool) -> str:
    """Resolve `rel` inside `root`. For a not-yet-existing file, the PARENT is
    realpath'd (a symlinked parent otherwise escapes). Prefix-check against
    realpath(root) + os.sep."""
    if not rel or os.path.isabs(rel):
        raise PathEscape(f"path must be relative and non-empty: {rel!r}")
    real_root = os.path.realpath(root)
    joined = os.path.normpath(os.path.join(real_root, rel))
    if for_write:
        parent_real = os.path.realpath(os.path.dirname(joined))
        candidate = os.path.join(parent_real, os.path.basename(joined))
    else:
        candidate = os.path.realpath(joined)
    if not (candidate == real_root or candidate.startswith(real_root + os.sep)):
        raise PathEscape(f"path escapes the worktree: {rel!r}")
    return candidate


def tool_list_files(root: str, rel: str = ".") -> str:
    base = safe_path(root, rel or ".", for_write=False)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [d for d in sorted(dirnames) if d != ".git"]
        for f in sorted(filenames):
            out.append(os.path.relpath(os.path.join(dirpath, f), root))
            if len(out) >= MAX_LIST:
                out.append(f"[... listing capped at {MAX_LIST} entries ...]")
                return "\n".join(out)
    return "\n".join(out) or "(empty)"


def tool_read_file(root: str, rel: str) -> str:
    p = safe_path(root, rel, for_write=False)
    if os.path.islink(p) or not os.path.isfile(p):
        return f"error: not a regular file: {rel}"
    with open(p, "rb") as fh:
        data = fh.read(MAX_READ + 1)
    if b"\0" in data[:8192]:
        return f"error: binary file, not shown: {rel}"
    text = data[:MAX_READ].decode("utf-8", "replace")
    return text + ("\n[... truncated ...]" if len(data) > MAX_READ else "")


def tool_write_file(root: str, rel: str, content: str) -> str:
    p = safe_path(root, rel, for_write=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"wrote {len(content)} chars to {rel}"


async def tool_run_command(container: str, cmd: str, *, timeout_s: float) -> tuple[int, str]:
    rc, out, _trunc = await exec_in_container(container, cmd, timeout_s=timeout_s)
    return rc, out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_security.py -q`
Expected: all pass.

- [ ] **Step 5: Lint/type and commit**

```bash
ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent
git add coding_agent/tools.py test_coding_agent_security.py
git commit -m "feat(coding-agent): tool schemas + host-side path guard incl. symlinked-parent write escape"
```

---

### Task 7: `loop.py` — stop conditions + no-progress detector (pure logic, fake everything)

**Files:**
- Create: `coding_agent/loop.py` (part 1: `StopReason`, `ProgressTracker`, `Budget`)
- Test: `test_coding_agent_loop.py`

**Interfaces:**
- Produces:
  ```python
  class StopReason(str, Enum): completed | max_turns | max_seconds | no_progress | error
  @dataclass
  class Budget: max_turns: int; max_seconds: float; started: float
  class ProgressTracker:
      def __init__(self, no_progress_turns: int = 5) -> None
      def observe(self, *, tree_hash: str, last_cmd: tuple[str, int] | None, wrote_file: bool) -> bool  # True == progress this turn
      @property stalled(self) -> bool
  ```

- [ ] **Step 1: Write the failing tests**

Create `test_coding_agent_loop.py`:
```python
#!/usr/bin/env python3
"""Control-flow tests for coding_agent.loop (no docker, no ollama).
Run:  uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q
"""
from __future__ import annotations

import unittest

from coding_agent.loop import ProgressTracker


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
        self.assertTrue(t.observe(tree_hash="H", last_cmd=("pytest", 0), wrote_file=False))

    def test_different_command_same_exit_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=("pytest a", 0), wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=("pytest b", 0), wrote_file=False))

    def test_write_file_alone_is_progress(self):
        t = ProgressTracker(no_progress_turns=2)
        t.observe(tree_hash="H", last_cmd=None, wrote_file=False)
        self.assertTrue(t.observe(tree_hash="H", last_cmd=None, wrote_file=True))
        self.assertFalse(t.stalled)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the pure parts of `loop.py`**

Create `coding_agent/loop.py`:
```python
"""Orchestration loop for coding_agent (spec §4, §7).

The loop runs on the HOST. Tool calls dispatch to tools.py; only
run_command crosses into the container. Stop conditions are mechanical
and the model cannot extend its own budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


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


class ProgressTracker:
    """Progress == walk-hash changed OR (cmd, exit) pair changed OR write_file
    was called this turn. Not adversarially robust by design (spec §5.1) —
    the hard caps are the real budget; this catches *stuck*, not *malicious*."""

    def __init__(self, no_progress_turns: int = 5) -> None:
        self._n = no_progress_turns
        self._idle = 0
        self._last_hash: str | None = None
        self._last_cmd: tuple[str, int] | None = None

    def observe(self, *, tree_hash: str, last_cmd: tuple[str, int] | None, wrote_file: bool) -> bool:
        progressed = (
            wrote_file
            or (self._last_hash is not None and tree_hash != self._last_hash)
            or (last_cmd is not None and last_cmd != self._last_cmd)
        )
        if self._last_hash is None:
            progressed = True     # first observation establishes a baseline
        self._last_hash, self._last_cmd = tree_hash, (last_cmd or self._last_cmd)
        self._idle = 0 if progressed else self._idle + 1
        return progressed

    @property
    def stalled(self) -> bool:
        return self._idle >= self._n
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint/type and commit**

```bash
ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent
git add coding_agent/loop.py test_coding_agent_loop.py
git commit -m "feat(coding-agent): stop reasons, budget, and no-progress detector (spec §7)"
```

---

### Task 8: `loop.py` — the orchestration loop with ordered teardown + single-slot lock

**Files:**
- Modify: `coding_agent/loop.py`, `coding_agent/__init__.py`
- Test: `test_coding_agent_loop.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7; `_post_ollama_chat(payload, timeout_s) -> dict` from `mcp_server` (injected as a callable so tests can fake it).
- Produces:
  ```python
  @dataclass
  class AgentResult:
      stop_reason: StopReason; turns: int; elapsed_seconds: float; diff: str; diff_truncated: bool
      diff_full_path: str | None; changed_files: list[str]; last_command: dict | None
      transcript: list[dict]; model: str; cleanup_problems: list[str]
  async def run_coding_agent(*, task: str, repo: str, base_ref: str, model: str,
      max_turns: int, max_seconds: float, image: str,
      chat: Callable[[dict, float], Awaitable[dict]],   # mcp_server._post_ollama_chat
      sandbox_factory=None) -> AgentResult
  ```
  and the lock: `_SLOT = asyncio.Lock()`; a second concurrent call raises `RuntimeError("coding_agent: a run is already in progress on this host")` — **rejected, not queued** (`if _SLOT.locked(): raise`).

- [ ] **Step 1: Write the failing tests (ordering + lock + result shape) with a fake sandbox and fake chat**

Append to `test_coding_agent_loop.py`:
```python
import asyncio
import os
import tempfile
from pathlib import Path
from unittest import mock

from coding_agent import loop as L


class _FakeSandboxOps:
    """Records call ORDER so we can assert destroy_container precedes the host read."""
    def __init__(self, worktree: str):
        self.worktree = worktree; self.calls: list[str] = []
    def create_worktree(self, repo, base_ref):
        self.calls.append("create_worktree"); return self.worktree
    async def start_container(self, worktree, image, **kw):
        self.calls.append("start_container"); return "cid"
    async def exec_in_container(self, container, cmd, *, timeout_s):
        self.calls.append(f"exec:{cmd}"); return (0, "ok\n", False)
    async def destroy_container(self, container):
        self.calls.append("destroy_container")
    def teardown_worktree(self, repo, worktree):
        self.calls.append("teardown_worktree"); return []


def _scripted_chat(turns):
    """Return a fake _post_ollama_chat that replays `turns` (list of message dicts)."""
    it = iter(turns)
    async def chat(payload, timeout_s):
        return {"message": next(it)}
    return chat


class LoopOrdering(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp()); self.wt = tempfile.mkdtemp()
        (Path(self.wt) / "a.py").write_text("x=1\n")

    def _run(self, turns, **kw):
        ops = _FakeSandboxOps(self.wt)
        with mock.patch.object(L, "read_base_tree", return_value=L.BaseTree(entries={}, ignore=lambda p: False, tracked=frozenset())):
            res = asyncio.run(L.run_coding_agent(
                task="t", repo=str(self.repo), base_ref="HEAD", model="m",
                max_turns=kw.get("max_turns", 10), max_seconds=kw.get("max_seconds", 60), image="img",
                chat=_scripted_chat(turns), sandbox_factory=lambda: ops))
        return res, ops

    def test_container_destroyed_before_host_read_and_result_shape(self):
        res, ops = self._run([{"role": "assistant", "content": "done", "tool_calls": []}])
        self.assertEqual(res.stop_reason, L.StopReason.completed)
        i_destroy = ops.calls.index("destroy_container"); i_teardown = ops.calls.index("teardown_worktree")
        self.assertLess(i_destroy, i_teardown)
        # host read (diff) happens after destroy: the result already carries the diff
        self.assertIn("a.py", res.changed_files)
        self.assertEqual(res.cleanup_problems, [])

    def test_max_turns_stops(self):
        turns = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}]}] * 20
        res, _ = self._run(turns, max_turns=3)
        self.assertEqual(res.stop_reason, L.StopReason.max_turns); self.assertEqual(res.turns, 3)

    def test_second_concurrent_run_is_rejected_not_queued(self):
        async def go():
            async with L._SLOT:
                with self.assertRaises(RuntimeError):
                    await L.run_coding_agent(task="t", repo=str(self.repo), base_ref="HEAD", model="m",
                        max_turns=1, max_seconds=5, image="img", chat=_scripted_chat([]),
                        sandbox_factory=lambda: _FakeSandboxOps(self.wt))
        asyncio.run(go())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q -k Loop`
Expected: FAIL — `run_coding_agent` not defined.

- [ ] **Step 3: Implement the loop**

Append to `coding_agent/loop.py`:
```python
import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from typing import Any

from . import sandbox as _sb
from .basetree import BaseTree, make_ignore, read_base_tree
from .tools import (TOOL_SCHEMAS, PathEscape, tool_list_files, tool_read_file,
                    tool_run_command, tool_write_file)
from .walk import snapshot_tree, tree_hash, unified_diff

_SLOT = asyncio.Lock()
_CLEANUP_BOUND_S = 60.0
_DIFF_MAX_BYTES = 512 * 1024
_CMD_TIMEOUT_S = 300.0

SYSTEM_PROMPT = (
    "You are a coding agent working in a sandboxed copy of a repository. You have "
    "list_files, read_file, write_file, and run_command (no network). Work until the "
    "task's success criterion is met, then reply with a short summary and NO tool calls."
)


@dataclass
class AgentResult:
    stop_reason: StopReason
    turns: int
    elapsed_seconds: float
    diff: str
    diff_truncated: bool
    diff_full_path: str | None
    changed_files: list[str]
    last_command: dict[str, Any] | None
    transcript: list[dict[str, Any]]
    model: str
    cleanup_problems: list[str]


def _default_ops() -> Any:
    return _sb


async def run_coding_agent(
    *, task: str, repo: str, base_ref: str, model: str, max_turns: int, max_seconds: float,
    image: str, chat: Callable[[dict[str, Any], float], Awaitable[dict[str, Any]]],
    sandbox_factory: Callable[[], Any] | None = None,
) -> AgentResult:
    if _SLOT.locked():
        raise RuntimeError("coding_agent: a run is already in progress on this host")
    async with _SLOT:
        ops = (sandbox_factory or _default_ops)()
        started = time.monotonic()
        budget = Budget(max_turns=max_turns, max_seconds=max_seconds, started=started)
        base: BaseTree = read_base_tree(repo, base_ref)          # REAL repo, trusted
        ignore = make_ignore(base)
        worktree = ops.create_worktree(repo, base_ref)             # record the literal path
        container: str | None = None
        transcript: list[dict[str, Any]] = []
        last_cmd: dict[str, Any] | None = None
        stop = StopReason.error
        turns = 0
        tracker = ProgressTracker()
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": task}]
        try:
            container = await ops.start_container(worktree, image, cpus="4", memory="16g", pids=512)
            while True:
                if turns >= budget.max_turns:
                    stop = StopReason.max_turns; break
                if budget.exceeded_time():
                    stop = StopReason.max_seconds; break
                turns += 1
                resp = await chat({"model": model, "messages": messages, "tools": TOOL_SCHEMAS,
                                   "stream": False, "think": False}, _CMD_TIMEOUT_S)
                if resp.get("status") == "failed":
                    transcript.append({"turn": turns, "error": resp}); stop = StopReason.error; break
                msg = resp.get("message") or {}
                calls = msg.get("tool_calls") or []
                messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} or {"role": "assistant", "content": ""})
                if not calls:
                    stop = StopReason.completed; break
                wrote = False
                for tc in calls:
                    fn = (tc.get("function") or {}); name = fn.get("name"); args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except json.JSONDecodeError: args = {}
                    try:
                        if name == "list_files":
                            result = tool_list_files(worktree, args.get("path", "."))
                        elif name == "read_file":
                            result = tool_read_file(worktree, args["path"])
                        elif name == "write_file":
                            result = tool_write_file(worktree, args["path"], args.get("content", "")); wrote = True
                        elif name == "run_command":
                            rc, out = await tool_run_command(container, args["cmd"], timeout_s=_CMD_TIMEOUT_S)
                            last_cmd = {"cmd": args["cmd"], "exit": rc, "output_tail": out[-2000:]}
                            result = f"[exit {rc}]\n{out}"
                        else:
                            result = f"error: unknown tool {name!r}"
                    except (PathEscape, KeyError, TypeError, OSError) as exc:
                        result = f"error: {type(exc).__name__}: {exc}"
                    transcript.append({"turn": turns, "tool": name, "args": args, "result_head": result[:400]})
                    messages.append({"role": "tool", "content": result})
                snap_hash = tree_hash(snapshot_tree(worktree, ignore))       # host byte-walk, NO git
                lc = (last_cmd["cmd"], last_cmd["exit"]) if last_cmd else None
                tracker.observe(tree_hash=snap_hash, last_cmd=lc, wrote_file=wrote)
                if tracker.stalled:
                    stop = StopReason.no_progress; break
        except Exception as exc:  # noqa: BLE001 — surfaced in the result, cleanup still runs
            transcript.append({"turn": turns, "error": f"{type(exc).__name__}: {exc}"}); stop = StopReason.error
        finally:
            # ORDER IS NORMATIVE (§6.5 rule 3): kill the container BEFORE any host
            # read used to build the result; shielded AND bounded (§6 item 3).
            problems: list[str] = []
            diff_text, diff_trunc, diff_full, changed = "", False, None, []
            async def _teardown() -> None:
                nonlocal problems, diff_text, diff_trunc, diff_full, changed
                if container:
                    await ops.destroy_container(container)
                try:
                    snap = snapshot_tree(worktree, ignore)
                    d = unified_diff(base, snap, max_bytes=_DIFF_MAX_BYTES, spill_dir=tempfile.gettempdir())
                    diff_text, diff_trunc, diff_full, changed = d.text, d.truncated, d.full_path, d.changed_files
                finally:
                    problems = ops.teardown_worktree(repo, worktree)
            try:
                await asyncio.shield(asyncio.wait_for(_teardown(), timeout=_CLEANUP_BOUND_S))
            except asyncio.CancelledError:
                pass   # one CancelledError is delivered at the shield boundary; teardown continues
        return AgentResult(stop_reason=stop, turns=turns, elapsed_seconds=round(time.monotonic() - started, 1),
                           diff=diff_text, diff_truncated=diff_trunc, diff_full_path=diff_full,
                           changed_files=changed, last_command=last_cmd, transcript=transcript,
                           model=model, cleanup_problems=problems)
```
And make `coding_agent/__init__.py`:
```python
from .loop import AgentResult, StopReason, run_coding_agent  # noqa: F401
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q`
Expected: all pass (5 + 3).

- [ ] **Step 5: Lint/type and commit**

```bash
ruff format coding_agent && ruff check coding_agent && mypy --strict coding_agent
git add coding_agent/loop.py coding_agent/__init__.py test_coding_agent_loop.py
git commit -m "feat(coding-agent): orchestration loop — container destroyed before host read, shielded+bounded teardown, single-slot lock (spec §4, §6, §7)"
```

---

### Task 9: `mcp_server.py` — register `coding_agent` + `coding_agent_result` (dispatch only) and background jobs

**Files:**
- Modify: `mcp_server.py` (tool list near line 2411; dispatch near line 3009)
- Test: `test_coding_agent_loop.py`

**Interfaces:**
- Consumes: `run_coding_agent`, `AgentResult` from Task 8; the existing `_delegate_jobs` **pattern** (copy its shape into a separate `_coding_jobs` dict — do NOT share the dict or `_DELEGATE_JOB_CAP`).
- Produces: MCP tools `coding_agent(task, repo, base_ref="HEAD", model=None, max_turns=25, max_seconds=600, background=True)` and `coding_agent_result(job_id)`.

- [ ] **Step 1: Write the failing test**

Append to `test_coding_agent_loop.py` (this file already stubs nothing; import `mcp_server` the way `test_local_delegate.py` does — copy its `_build_stub_modules` + `mock.patch.dict(sys.modules, ...)` import block verbatim into a `setUpModule` here):
```python
class McpRegistration(unittest.TestCase):
    def test_tools_are_registered(self):
        names = {t.name for t in asyncio.run(mcp_server.list_tools())}
        self.assertIn("coding_agent", names); self.assertIn("coding_agent_result", names)

    def test_missing_task_errors_without_touching_docker(self):
        out = asyncio.run(mcp_server.call_tool("coding_agent", {"repo": "/tmp"}))
        self.assertIn("task is required", out[0].text)

    def test_model_must_be_allowlisted(self):
        out = asyncio.run(mcp_server.call_tool("coding_agent", {"task": "x", "repo": "/tmp", "model": "evil:latest"}))
        self.assertIn("not allowlisted", out[0].text)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py -q -k Mcp`
Expected: FAIL — tools not registered.

- [ ] **Step 3: Register + dispatch in `mcp_server.py`**

Near the other `Tool(...)` entries (after `local_delegate_result`) add:
```python
        Tool(
            name="coding_agent",
            description=(
                "Hand a BOUNDED, MECHANICAL coding task to a local model that works "
                "autonomously (files + shell + tests) inside a network-less Docker "
                "sandbox over a throwaway copy of `repo`. Returns a transcript and a "
                "unified diff; NOTHING is applied — you review and apply the diff. Best "
                "for verifiable work (make a test pass, rename across files, mechanical "
                "refactors); local models are measured-unreliable as defect gates, so "
                "give a concrete success criterion. Toolchains must already exist in the "
                "sandbox image (no network to install). background=true by default; poll "
                "coding_agent_result."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "repo": {"type": "string", "description": "Absolute path to a git repo on this host."},
                    "base_ref": {"type": "string", "default": "HEAD"},
                    "model": {"type": "string", "description": "Allowlisted tag; default = the review role model."},
                    "max_turns": {"type": "integer", "default": 25, "maximum": 60},
                    "max_seconds": {"type": "integer", "default": 600, "maximum": 1800},
                    "background": {"type": "boolean", "default": True},
                },
                "required": ["task", "repo"],
            },
        ),
        Tool(
            name="coding_agent_result",
            description="Poll a background coding_agent job by job_id (single-collect).",
            inputSchema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        ),
```
Near the top-level helpers add a separate registry and default model resolver:
```python
_coding_jobs: dict[str, dict[str, Any]] = {}
_CODING_AGENT_IMAGE = os.environ.get("AI_TOOLS_CODING_AGENT_IMAGE", "ai-tools-coding-agent:latest")


def _coding_agent_default_model() -> str:
    """The machine's review-role model via `local-model review`, falling back
    to gemma4:31b-nvfp4; must be allowlisted either way."""
    try:
        out = subprocess.run(["local-model", "review"], capture_output=True, text=True, timeout=5)
        cand = out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        cand = ""
    return cand or "gemma4:31b-nvfp4"
```
In `call_tool`, add dispatch (mirroring `local_delegate`'s validation style):
```python
    if name == "coding_agent":
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return [TextContent(type="text", text="Error: task is required and must be a non-empty string.")]
        repo = arguments.get("repo")
        if not isinstance(repo, str) or not os.path.isdir(os.path.join(repo, ".git")) and not os.path.isfile(os.path.join(repo, ".git")):
            return [TextContent(type="text", text="Error: repo must be an absolute path to a git repository on this host.")]
        model = arguments.get("model") or _coding_agent_default_model()
        if model not in _resolve_delegate_models():
            return [TextContent(type="text", text=f"Error: model {model!r} is not allowlisted; allowed: {', '.join(_resolve_delegate_models())}")]
        max_turns = min(int(arguments.get("max_turns", 25)), 60)
        max_seconds = min(float(arguments.get("max_seconds", 600)), 1800.0)
        from coding_agent import run_coding_agent  # local import keeps server import cheap
        coro = run_coding_agent(task=task, repo=repo, base_ref=str(arguments.get("base_ref", "HEAD")),
                                model=model, max_turns=max_turns, max_seconds=max_seconds,
                                image=_CODING_AGENT_IMAGE, chat=_post_ollama_chat)
        if arguments.get("background", True):
            job_id = uuid.uuid4().hex
            _coding_jobs[job_id] = {"task": asyncio.get_running_loop().create_task(coro), "started": time.monotonic()}
            return [TextContent(type="text", text=json.dumps({"job_id": job_id, "status": "started"}))]
        res = await coro
        return [TextContent(type="text", text=json.dumps(res.__dict__, default=str))]

    if name == "coding_agent_result":
        job_id = arguments.get("job_id")
        job = _coding_jobs.get(job_id) if isinstance(job_id, str) else None
        if job is None:
            return [TextContent(type="text", text="Error: unknown job_id (results are single-collect and do not survive a restart).")]
        t = job["task"]
        if not t.done():
            return [TextContent(type="text", text=json.dumps({"status": "in_progress"}))]
        _coding_jobs.pop(job_id, None)
        try:
            res = t.result()
        except Exception as exc:  # noqa: BLE001
            return [TextContent(type="text", text=json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))]
        return [TextContent(type="text", text=json.dumps(res.__dict__, default=str))]
```

- [ ] **Step 4: Run all tests**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_loop.py test_local_delegate.py -q`
Expected: all pass.

- [ ] **Step 5: Lint/type and commit**

```bash
ruff format . && ruff check . && mypy --strict coding_agent
git add mcp_server.py test_coding_agent_loop.py
git commit -m "feat(mcp): register coding_agent + coding_agent_result (dispatch only; loop lives in coding_agent/)"
```

---

### Task 10: The sandbox image + `--check` drift report (§8.2)

**Files:**
- Create: `scripts/coding-agent-image/Dockerfile`, `scripts/coding-agent-image/build.sh`, `scripts/coding-agent-image/check.sh`

**Interfaces:**
- Produces: image tag `ai-tools-coding-agent:latest` (matches `_CODING_AGENT_IMAGE` default in Task 9).

- [ ] **Step 1: Write the Dockerfile (pinned toolchain; git needed IN the container for the model's own use)**

`scripts/coding-agent-image/Dockerfile`:
```dockerfile
FROM python:3.12-slim
# Pinned toolchain the agent may need. NO network at runtime, so everything
# must be here (spec §8.2). Bump deliberately; check.sh reports drift.
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm shellcheck \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.* pytest==8.* ruff==0.13.* mypy==1.18.*
RUN git config --system --add safe.directory /work \
    && useradd -m -u 1000 agent
WORKDIR /work
```

- [ ] **Step 2: Write `build.sh` and `check.sh`**

`build.sh`:
```bash
#!/bin/sh
set -eu
cd "$(dirname "$0")"
docker build -t ai-tools-coding-agent:latest .
```
`check.sh` (the §8.2 drift report):
```bash
#!/bin/sh
# Report which expected tools the current image is missing.
set -u
IMG="${1:-ai-tools-coding-agent:latest}"
for t in git python3 uv pytest ruff mypy node npm shellcheck; do
  if docker run --rm --network=none "$IMG" sh -c "command -v $t >/dev/null"; then
    printf '  ok      %s\n' "$t"
  else
    printf '  MISSING %s\n' "$t"
  fi
done
```
`chmod +x` both.

- [ ] **Step 3: Build and check**

Run: `scripts/coding-agent-image/build.sh && scripts/coding-agent-image/check.sh`
Expected: all `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/coding-agent-image
git commit -m "feat(coding-agent): pinned sandbox image + --check drift report (spec §8.2)"
```

---

### Task 11: Docker integration tests — real loop, container crash, ceiling mid-command (skipped when daemon absent)

**Files:**
- Create: `test_coding_agent_integration.py`

**Interfaces:**
- Consumes: everything. Uses a **fake Ollama** (scripted `chat`) so no model is needed; docker IS needed.

- [ ] **Step 1: Write the tests**

```python
#!/usr/bin/env python3
"""Docker-required integration tests. Skipped when the daemon is down.
Run:  uv run --with pytest --with pathspec pytest test_coding_agent_integration.py -q
"""
from __future__ import annotations

import asyncio, os, shutil, subprocess, tempfile, unittest
from pathlib import Path

from coding_agent import StopReason, run_coding_agent
from coding_agent import sandbox as sb

def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False

def _repo() -> str:
    r = tempfile.mkdtemp()
    subprocess.run(["git", "-C", r, "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", r, "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", r, "config", "user.name", "T"], check=True)
    Path(r, "calc.py").write_text("def add(a, b):\n    return a - b\n")
    Path(r, "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    subprocess.run(["git", "-C", r, "add", "-A"], check=True)
    subprocess.run(["git", "-C", r, "commit", "-q", "-m", "base"], check=True)
    return r

def _script(*msgs):
    it = iter(msgs)
    async def chat(payload, timeout_s):
        return {"message": next(it)}
    return chat

@unittest.skipUnless(_docker_ok(), "docker daemon not available")
class RealLoop(unittest.TestCase):
    def test_fix_failing_test_end_to_end_and_everything_is_gone_after(self):
        repo = _repo()
        chat = _script(
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_command", "arguments": {"cmd": "python -m pytest -q"}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_command", "arguments": {"cmd": "python -m pytest -q"}}}]},
            {"role": "assistant", "content": "fixed", "tool_calls": []},
        )
        res = asyncio.run(run_coding_agent(task="make tests pass", repo=repo, base_ref="HEAD", model="m",
                                           max_turns=10, max_seconds=300, image="ai-tools-coding-agent:latest", chat=chat))
        self.assertEqual(res.stop_reason, StopReason.completed)
        self.assertIn("calc.py", res.changed_files)
        self.assertIn("+    return a + b", res.diff)
        self.assertEqual(res.last_command["exit"], 0)
        self.assertEqual(res.cleanup_problems, [])
        # container gone
        ps = subprocess.run(["docker", "ps", "-a", "--format", "{{.Command}}"], capture_output=True, text=True).stdout
        self.assertNotIn("sleep infinity", ps)
        # user's repo untouched
        self.assertIn("return a - b", Path(repo, "calc.py").read_text())

    def test_container_crash_midloop_still_cleans_up(self):
        repo = _repo()
        async def crashing_chat(payload, timeout_s):
            # kill the container out from under the loop, then ask it to run a command
            cid = subprocess.run(["docker", "ps", "-q", "--filter", "ancestor=ai-tools-coding-agent:latest"], capture_output=True, text=True).stdout.split()
            for c in cid: subprocess.run(["docker", "rm", "-f", c], capture_output=True)
            return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_command", "arguments": {"cmd": "true"}}}]}}
        res = asyncio.run(run_coding_agent(task="t", repo=repo, base_ref="HEAD", model="m", max_turns=3, max_seconds=60,
                                           image="ai-tools-coding-agent:latest", chat=crashing_chat))
        self.assertEqual(res.cleanup_problems, [])
        self.assertIn(res.stop_reason, (StopReason.error, StopReason.max_turns, StopReason.no_progress))

    def test_wall_clock_ceiling_mid_command_still_removes_both(self):
        repo = _repo()
        chat = _script({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_command", "arguments": {"cmd": "sleep 30"}}}]})
        res = asyncio.run(asyncio.wait_for(run_coding_agent(task="t", repo=repo, base_ref="HEAD", model="m", max_turns=5, max_seconds=2,
                                                            image="ai-tools-coding-agent:latest", chat=chat), timeout=90))
        self.assertEqual(res.cleanup_problems, [])
        ps = subprocess.run(["docker", "ps", "-a", "--format", "{{.Command}}"], capture_output=True, text=True).stdout
        self.assertNotIn("sleep infinity", ps)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Run: `uv run --with pytest --with pathspec pytest test_coding_agent_integration.py -q`
Expected: 3 passed (or 3 skipped if the daemon is down — then start Docker Desktop and re-run; these MUST pass before merge).

- [ ] **Step 3: Commit**

```bash
git add test_coding_agent_integration.py
git commit -m "test(coding-agent): docker integration — real loop, crash cleanup, ceiling mid-command"
```

---

### Task 12: Version bump, README, and the T2 delivery

**Files:**
- Modify: `mcpb/manifest.json:5`, `.claude-plugin/plugin.json:4` (`1.5.4` → `1.6.0`), `README.md` (new tool section), `mcpb/manifest.json` (tools list + `user_config` for `AI_TOOLS_CODING_AGENT_IMAGE`)

- [ ] **Step 1: Bump both versions in lockstep**

Edit both files: `"version": "1.6.0"`.

- [ ] **Step 2: README section (copy the two honesty clauses verbatim from spec §8.1/§8.2)**

Under the tools list add a `### coding_agent` section: what it does, that nothing is applied, the two clauses (mechanical/verifiable work only + toolchain-in-image cost), `AI_TOOLS_CODING_AGENT_IMAGE`, and `scripts/coding-agent-image/check.sh`.

- [ ] **Step 3: Full suite + rebuild bundle**

Run: `uv run --with pytest --with pathspec pytest -q` then rebuild the mcpb bundle per README §D and verify: `unzip -p dist/ai-tools-mcp.mcpb manifest.json | grep '"version"'` → `1.6.0`.

- [ ] **Step 4: Commit and hand to the review pipeline as T2**

```bash
git add mcpb/manifest.json .claude-plugin/plugin.json README.md
git commit -m "chore(release): ai-tools-mcp 1.6.0 — coding_agent (sandboxed local-model coding loop, Claude gates the diff)"
```
Then invoke the `review-pipeline` skill: classify **T2** in the PR description (introduces arbitrary execution + a container boundary), freeze the diff, run Stage 1 (16 tests + `test_coding_agent_*` + Semgrep + CodeRabbit exact), Normal-tier PR, Stage 2, Stage 3 with the focused adversarial pass T2 prescribes.

---

## Self-Review

**Spec coverage** — every load-bearing section maps to a task: §4/§5 tools + dispatch (T6, T8, T9); §5.1 diff incl. new files, size cap, tracked-aware (T3, T4); §6.2 docker flags (T5, T10); §6.5 rules 1–5 (T2, T3, T4, T5, T8 ordering); §6.6 layered cleanup (T5); §6.7 spike (T0); §7 termination (T7, T8); §8 honesty clauses (T9 description, T12 README); §10 every regression test — fsmonitor/opaque-.git (T2), filter/.gitattributes (T2, written as a regression guard), TOCTOU (T2, written as a regression guard), symlink exfil (T2), header injection (T4), tracked-not-blinded (T3), typechange (T4), symlinked-parent write (T6), mangled-.git cleanup (T5), crash + ceiling (T11); §12 allowlist prerequisite (T1) + lockstep bump (T12).

**Placeholder scan** — one deliberate: `SANDBOX_USER_FLAG` in T5 is set to `[]` with an instruction to replace it with T0's recorded DECISION. That is a data dependency on the spike, not a TODO; the step says exactly what to write.

**Type consistency** — `Entry(path, kind, data)` (T2) is consumed identically in T3/T4; `TreeSnapshot.entries` dict-of-Entry throughout; `BaseTree(entries, ignore, tracked)` (T3) is what T4/T8 construct in tests; `unified_diff(base, snap, *, max_bytes, spill_dir) -> DiffResult(text, truncated, changed_files, full_path)` (T4) matches T8's use; `run_coding_agent(..., chat, sandbox_factory)` (T8) matches T9's call (T9 omits `sandbox_factory`, defaulting to the real sandbox); `AgentResult` fields serialized via `__dict__` in T9. `exec_in_container` returns a 3-tuple in T5; `tool_run_command` (T6) unpacks it and returns a 2-tuple, which T8 uses — consistent.
