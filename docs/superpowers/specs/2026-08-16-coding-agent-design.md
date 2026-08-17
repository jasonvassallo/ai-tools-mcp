# `coding_agent` — sandboxed agentic coding on a local model

**Status:** design approved in conversation 2026-08-16, pending spec review.
**Author model:** Claude (Fable 5, design); implementation intended for Opus 5.

## 1. What this is, in one paragraph

A new ai-tools-mcp tool that lets Claude (or Codex) hand a **bounded, mechanical
coding task** to a local model, which then works **autonomously** — reading and
writing files, running commands and tests — inside a **throwaway, network-less
Docker container** holding a disposable copy of the repo. When it stops, it
returns the complete transcript, the final unified diff, test output, and *why*
it stopped. **Nothing is applied to the real repository.** Claude reviews the
diff and decides. The point is to delegate the *grind* (rename across files,
add tests until green, mechanical refactors) while keeping every keystroke
auditable and every side-effect confined.

## 1.1 Provenance — three claims in this doc live outside this repo

A reader checking only this repository will not find these; they are true, but
sourced elsewhere. Stated up front so nobody mistakes them for typos:

- **`gemma4:31b-nvfp4`** is not in this repo's `_OLLAMA_BUILTIN_DELEGATE_MODELS`
  allowlist as of this writing (that still lists `gemma4:12b-nvfp4` + the
  retired `qwen3.6` tags). It is the model **currently deployed** as the
  `local_delegate` default on JVMBPro via the per-machine `AI_TOOLS_OLLAMA_MODELS`
  override, and its **1.000 agentic-benchmark score** comes from a purpose-built
  8-scenario tool-use harness run 2026-08-11..16 (`agentic_bench.py`, session
  scratch — not committed here). Adding it to the built-in allowlist is a
  prerequisite of implementation and is called out in §12.
- **"PR #25 (82 commits / 64h)"** refers to `jasonvassallo/review-pipeline#25`,
  **not** this repo's PR #25. It is the incident that produced review-pipeline's
  convergence policy (`skills/review-pipeline/policy/convergence-policy.md` in
  that repo).
- **`LOCAL_REVIEW_MODEL` / "role config"** is `~/.config/local-models.env` +
  the `local-model` helper, added to the operator's machines 2026-08-16. It is
  host configuration, not repo code — which is the point: model choice is
  operational, not a code change.

## 2. Why not extend `local_delegate`

`local_delegate` is deliberately single-shot prompt→text with **no filesystem
access**. Its tool description promises "input text never leaves the machine"
and "the server never reads the filesystem" — that is a security guarantee
callers rely on, not a limitation to remove. Widening it would silently
change what an existing tool is allowed to do. `coding_agent` is a **separate
tool with a separate, louder contract**: it *does* execute, and it says so.

The two also serve different asks. `local_delegate` = "think about this text."
`coding_agent` = "go do this to the code and show me what you did."

## 3. Decisions made, and the reasoning that bound them

| Decision | Choice | Why |
|---|---|---|
| Capability | Full loop: files + shell + tests | The only tier where "agentic" means autonomous; anything less is `local_delegate` with file context |
| Oversight | Sandbox-autonomous; **Claude gates the diff** | Per-step approval would make Claude do all the thinking with a network hop per action — it defeats delegating. Oversight comes from *where the gate sits*, not how often Claude is consulted |
| Isolation | Docker, `--network=none` | Strongest boundary available and already installed. Also enables the loop-on-host architecture (§4) |
| Termination | Hard caps **and** no-progress detection | A stochastic loop with shell access does not reliably self-terminate; PR #25 (82 commits / 64h) is this fleet's proof. The agent cannot elect to continue |
| Model | `gemma4:31b-nvfp4`, resolved via role config | Best-measured on the wide review corpus (78 runs/arm: lowest false-flag, tersest findings, fastest) **and** the only arm to recover from a tool error on the 8-scenario agentic harness. Honest weight: the agentic result is small-n and *indicative*, not decisive on its own; the wide-corpus run is the load-bearing evidence. Because the model is a role-config value, this choice is revisable without a code change |

## 4. Architecture — the loop runs on the host; the sandbox only executes

```
Claude ──MCP──▶ ai-tools-mcp (host process)
                  │  orchestration loop:
                  │   1. call Ollama (gemma4:31b) with task + tool schemas
                  │   2. receive requested tool calls
                  │   3. execute EACH inside the container
                  │   4. feed results back; repeat until a stop condition
                  │
                  └──▶ docker run --network=none … (per task)
                         bind-mount: throwaway git worktree of the repo
                         tools: list_files / read_file / write_file / run_command
```

**The single most important property:** because the loop lives in the MCP
server, the container needs **no network at all**. The model never talks to
Ollama from inside; the server does. So `--network=none` is absolute — no
exfiltration, no remote-code-fetch, no phone-home. The alternative (agent
process inside the sandbox calling Ollama) would force a network hole open
and then require policing what it reaches.

**But the container is NOT the whole boundary — and an earlier draft of this
spec wrongly said it was.** The orchestrator runs `git` **on the host**, inside
a worktree the sandboxed model fully controls (§5.1 diff, §7 hash, cleanup).
git is a config/hook/filter-driven execution engine, not a passive reader, and
`.git` in a worktree is a plain *file* inside the bind mount. Adversarial
review **demonstrated on this host** that the model can replace it and get
code run as the operator via a single `git add -A`. That threat is handled in
§6.5 and is a hard requirement, not a hardening nice-to-have.

**Second property, nearly free:** the working copy is a `git worktree`, so
**only tracked files enter the sandbox**. `.env`, credentials, ignored build
artifacts and caches never do. This is a real secondary win, not incidental.

### Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `coding_agent.py` (new module) | Orchestration loop, stop conditions, transcript, diff extraction | Ollama client (`_post_ollama_chat`), sandbox runner, role resolver |
| Sandbox runner | Create/destroy worktree + container; execute one tool call inside; enforce container-level limits | `git`, `docker` |
| Tool schemas | The 4 tools the model sees, as Ollama tool definitions | — |
| Job registry | Background launch + single-collect polling | **reuse** `_delegate_jobs` pattern (`_launch`/`_collect`), including its done-job eviction and job cap |
| Role resolver | Which model runs the loop | `local-model review` / `LOCAL_REVIEW_MODEL`, with the existing allowlist as the ceiling |

`coding_agent.py` is a **new file**, not more `mcp_server.py` — that file is
3,285 lines and this is a genuinely separate concern with its own failure
modes and tests. `mcp_server.py` gains only the tool registration and a
dispatch call.

## 5. Tool contract

### `coding_agent`

| param | type | notes |
|---|---|---|
| `task` | string, required | The instruction. Should name success criteria (e.g. "until `pytest tests/test_x.py` passes") |
| `repo` | string, required | Absolute path to a git repo on this host |
| `base_ref` | string, default `HEAD` | Commit/branch the worktree is created from |
| `model` | string, optional | Must be allowlisted; defaults to the `review` role |
| `max_turns` | int, default 25, hard max 60 | |
| `max_seconds` | int, default 600, hard max 1800 | Wall clock for the whole loop |
| `background` | bool, default **true** | These runs take minutes; sync is allowed but discouraged |

Returns (or, in background mode, `coding_agent_result` returns on collect):

```json
{
  "stop_reason": "completed | max_turns | max_seconds | no_progress | error",
  "turns": 17,
  "elapsed_seconds": 412,
  "diff": "<unified diff, worktree vs base_ref, INCLUDING new files — see §5.1>",
  "changed_files": ["src/x.py", "tests/test_x.py"],
  "last_command": {"cmd": "pytest tests/test_x.py", "exit": 0, "output_tail": "..."},
  "transcript": [ {"turn":1,"tool":"read_file","args":{...},"result_head":"..."}, ... ],
  "model": "gemma4:31b-nvfp4"
}
```

`completed` means the model emitted a final message with no further tool
calls. It is **not** a claim of correctness — see §8.

### 5.1 The diff must include files the agent CREATED

A model that adds
`tests/test_new.py` and stops "completed" must not hand Claude a diff missing
the very file it created. **But it is computed by reading file bytes, never by
running git in the worktree** — see §6.5, which is the security spine of this
tool and overrides any git-command phrasing here. Concretely:

- The host walks the worktree filesystem directly, applies a host-owned copy
  of the base ref's `.gitignore` (captured at creation, never re-read from the
  sandbox), and diffs those bytes against the base ref's tree read from the
  **real repo's object store** (located from the `repo` param, outside the
  mount). Untracked additions, deletions and renames all appear because the
  comparison is byte-level, not index-based.
- No staging touches the user's index (no git runs at all).
- `changed_files` is derived from the same byte comparison, so the two never
  disagree.
- **The diff and transcript are size-capped.** `read_file` already caps and
  rejects binaries, but without a cap here a model that writes one large or
  binary file gets it staged and returned inline, blowing up the MCP response
  and Claude's context. Binary blobs are annotated (`Binary files differ`, path
  + size) rather than embedded; the text diff is truncated with a marker past
  a configured byte limit, exactly as `run_command` output already is; and the
  full diff is written to a file the caller can fetch, so nothing is lost.
- The no-progress detector in §7 is real but **not adversarially robust**: a
  model that writes one throwaway byte per turn defeats it. Against a
  non-cooperative loop, only `max_turns` and `max_seconds` bound anything.
  It is a defense against *stuck*, not against *malicious*; the caps are the
  actual budget control.

### `coding_agent_result`

Identical semantics to `local_delegate_result`: poll by `job_id`, single-collect,
jobs do not survive a server restart.

### Tools the model sees — and WHERE each one runs

| tool | runs | behaviour | limits |
|---|---|---|---|
| `list_files(path=".")` | **host** | recursive listing, respects `.gitignore` | capped entries |
| `read_file(path)` | **host** | returns text | size cap; binary rejected |
| `write_file(path, content)` | **host** | full overwrite | path must resolve inside the worktree; **no symlink escape** (resolve real path, then prefix-check) |
| `run_command(cmd)` | **container** | `sh -c` via `docker exec` | per-command timeout; output truncated with a marker; **no network exists to reach** |

**Only `run_command` crosses into the container.** The three file tools need no
execution — just I/O against the worktree path — so they run in the MCP server
process on the host, where the path-safety check lives. This is deliberate and
better than routing everything through `docker exec`: it is faster, it keeps
the host-side symlink/prefix guard as the *primary* enforcement for file I/O
(not a mirror of the mount boundary), and it means the container's only job is
executing commands — a smaller surface to reason about. The container still
sees every file the host tools wrote, because they share the bind-mounted
worktree.

Consequence to state plainly: the model *can* also touch files indirectly via
`run_command` (`sed -i`, `echo > file`). That is fine — those writes are still
confined to the mount and still appear in the byte-level diff (§5.1/§6.5). The
host-side guard protects the *host* filesystem, which `run_command` cannot
reach at all.

## 6. Isolation, precisely

Per task:

1. `git worktree add <tmp> <base_ref>` — throwaway, detached.
2. `docker run --rm --network=none --user <host-uid>:<host-gid> --read-only`
   `--tmpfs /tmp --tmpfs /home/agent -e HOME=/home/agent -e TMPDIR=/tmp`
   `--cpus <cap> --memory <cap> --pids-limit <cap>`
   `-v <worktree>:/work:rw -w /work <image> sleep infinity`
   (a long-lived container per task; every `run_command` is a `docker exec`
   into it, avoiding per-command startup cost).

   Two details here are load-bearing, both learned the cheap way from review:

   - **`--user` is the HOST uid:gid, not an arbitrary non-root user.** The
     worktree is created by the host user; a container running as some other
     uid cannot write to it, and `git` inside refuses to operate at all
     (`fatal: detected dubious ownership in repository at '/work'`). Matching
     the uid removes both. The image additionally sets
     `git config --system --add safe.directory /work` as belt-and-braces.
   - **`--read-only` needs a writable `$HOME`, not just `/tmp`.** `pytest`
     writes `.pytest_cache`, `ruff`/`mypy`/`uv`/`npm` write under
     `~/.cache`, `~/.local`, `~/.npm`. With only `/tmp` writable, the first
     test run fails on `EROFS` inside the sandbox and looks like a model
     failure. So `$HOME` is a tmpfs too, and cache dirs are steered there
     (`UV_CACHE_DIR`, `NPM_CONFIG_CACHE`, `PIP_CACHE_DIR` → under `/tmp`).
     The read-only root is kept: it still blocks tampering with the image's
     toolchain, which is the property that mattered.
3. On any stop condition **or** any error: `docker rm -f`, then worktree
   teardown per §6.6. Cleanup runs in a `finally` **wrapped in
   `asyncio.shield(asyncio.wait_for(cleanup, T_cleanup))`** — shielded so the
   outer wall-clock ceiling's `CancelledError` cannot cancel the teardown
   itself (that would turn the runaway-bounding timeout into a leak path), and
   *its own* bounded so a hung `docker rm` cannot wedge teardown forever. The
   awaiting coroutine catches the one `CancelledError` delivered at the shield
   boundary.
4. Concurrency: **exactly one** coding-agent container at a time, enforced by
   a **dedicated single-slot lock** — *not* by reusing `_DELEGATE_JOB_CAP`.
   That cap is 4 and is shared with `local_delegate`; reusing it would permit
   four containers, each with a 31B inference stream, on a box that has honest
   capacity for one. The registry pattern is reused; the cap is not. A second
   concurrent `coding_agent` is **rejected with a clear message** (matching
   `_launch`'s existing over-cap behaviour), not queued.

Path safety in `write_file`/`read_file` is enforced **on the host side** by
resolving the real path — **for a not-yet-existing file, resolve the parent
directory** — and prefix-checking against the worktree root, *in addition to*
the container's own mount boundary. Belt and braces, because a mount boundary
that is the only check is a single point of failure.

### 6.5 Host-side git runs on HOSTILE repository contents — hard requirement

This is the threat the first draft missed, and it is the most serious one in
the design. Adversarial review **demonstrated it on JVMBPro (git 2.55.0)**,
not reasoned about it:

> Inside the sandbox, replace the worktree's `.git` *file* with a `.git`
> *directory* whose `config` sets `core.fsmonitor = <worktree>/pwn.sh`.
> On the host, run the spec's own §5.1 step, `git add -A`.
> Result: the planted script executed **as the operator, on the host, outside
> any container.**

`core.fsmonitor` is one of a class: `core.hooksPath`, `filter.*.clean/smudge`
via `.gitattributes`, `diff.*.textconv`, `core.sshCommand`, `core.pager` —
all reachable once the model controls `.git/config`, which is trivial because
`.git` lives inside the mount. `--network=none`, `--user`, `--read-only` and
`safe.directory` all govern the *container*; the vulnerable `git` runs on the
*host*. A merely-buggy model (`git init`, `rm -rf .git`) can trip this by
accident; a poisoned repo or prompt-injected task trips it on purpose.

**The rule that actually closes this — mandatory, not a menu.** A second
adversarial pass **demonstrated** that a per-call env/`-c` allowlist does NOT
work: the `filter.*` clean/smudge vector uses an *attacker-named* config key
(`.gitattributes: secret filter=pwn` + repo-local `[filter "pwn"] clean=…`),
and you cannot `-c filter.<name>=` a name you don't know; and
`-c core.attributesFile=/dev/null` does **not** disable an *in-tree*
`.gitattributes` (verified — it only overrides the user/global file). It also
demonstrated that "tamper-detect then call git" is a check-then-use TOCTOU: a
process left running inside the container re-mangles `.git` in the window
between the check and git's own `open(".git")` — **15 host-code executions in
20 seconds through the full neutralized env.** So:

1. **No host-side `git` command is ever executed inside the worktree.** Not
   for the end-of-loop diff (§5.1), not for the per-turn no-progress hash
   (§7a), not for cleanup registration reads. This is structural and absolute;
   it is the only thing that closes the class, because the config-driven
   execution surface of `git` cannot be enumerated and neutralized per-call,
   and the per-turn hash cannot be protected by killing the container (the
   container must stay alive between turns).
2. **The diff and the content hash are computed by reading FILE BYTES, not by
   invoking git in the worktree.** The host walks the worktree path directly
   (a plain filesystem read — no git, no config, no hooks) and diffs against
   the base ref's tree read from the **user's real repository object store**.
   The diff is a true **two-sided comparison of `base-tree ∪ worktree-walk`**,
   so a tracked file the model deletes or replaces with a skipped special
   file surfaces as a **deletion/typechange**, never silently vanishing.
   **Ignore handling must be tracked-aware, or the gate can be blinded.**
   git's real rule is that `.gitignore` does **not** apply to already-tracked
   files — and a uniform pattern match does not know that. **Demonstrated in
   the fourth adversarial pass:** a model overwriting a *tracked* file that
   matches a base ignore pattern (a force-added file under an ignored dir, a
   committed `*.cfg`/`config.json` predating the rule) produces a change git
   would show but a naive ignore-filtered walk would hide — defeating the very
   gate this whole tool exists for. So: **every path present in the base tree
   appears in the diff regardless of ignore rules; ignore rules (a host-owned
   copy of the base ref's `.gitignore` + `.git/info/exclude` +
   `core.excludesFile`, captured at creation, never re-read from the sandbox)
   apply ONLY to paths not in the base tree.** Bias toward under-ignoring:
   over-showing an ignored artifact is harmless to a gate; under-showing a
   real change is the danger. That store is located from the **`repo` MCP parameter** (host-known
   input), *never* from the worktree's `.git`/`commondir` pointer (sandbox-
   controlled), and every base-tree read uses the real-repo gitdir as cwd
   (`git -C <repo> …`, never `git -C <worktree> …`, which would re-anchor
   config resolution on the hostile worktree). The store lives outside the
   `:/work` mount, so the sandbox cannot reach or alter it.
   **The walk must NOT follow symlinks or read special files — this is the one
   protection removing git took away.** git stored a symlink as a harmless
   `120000` blob (its target *text*); a naive `os.walk` + `open().read()` on
   the same tree instead *dereferences* a planted `/work/leak ->
   ~/.ssh/id_ed25519` and reads the host secret's bytes straight into the diff
   Claude sees — **demonstrated** in the third adversarial pass, and it inverts
   §4's "credentials stay out of the sandbox" into "credentials get pulled
   back in." So the walk: `lstat` every entry; represent a symlink by its
   **target string**, exactly as git's `120000` mode does, and never
   dereference it or descend a symlinked directory; read **only regular
   files** (skip FIFOs/sockets/devices — a FIFO would hang `open().read()`
   forever, a DoS; skipping also moots a `mknod` raw-disk device node under a
   root-in-container `--user 0:0`); stay on the worktree's device; and exclude
   the top-level `.git` entry as opaque. **The symlink target string is
   attacker-controlled** (a sandbox `ln -s "<forged diff header>" leak` round-
   trips arbitrary newlines and `--- a/…` bytes — demonstrated), so it must be
   rendered into the unified diff **exactly as git renders a `120000` blob:
   every target line carries a leading `+`, so no target byte can land in
   column 0 as a spurious `---`/`+++`/`@@` header, and the `@@` line count
   matches the emitted lines.** A hand-rolled encoding that embeds the target
   raw reintroduces diff-structure injection.
   (Note the belt-and-braces framing honestly: on the macOS target the
   worktree and `~/.ssh` share one APFS volume, so "stay on-device" protects
   nothing about host secrets — the real guard is never-follow-symlinks +
   only-walking-real-directories, so the walk never *constructs* a path
   outside the root. Hardlinks are `lstat`-indistinguishable from a regular
   file and the walk has no defense against one; safety there rests entirely
   on the container mount model — host secrets are not in the container's
   mount namespace to be `link()`ed, and `/work` is a different device from
   image/tmpfs so a cross-device link fails. Document this dependency so a
   future extra bind-mount does not silently reopen it.)
3. **The container is destroyed (`docker rm -f`, confirmed no process can still
   write `/work`) BEFORE any host-side read of the worktree used to build the
   returned result.** The end-of-loop sequence is: stop condition → kill
   container → read bytes → assemble diff → teardown worktree. Ordering is
   normative, not incidental.
4. **The worktree's `.git` pointer is used only by the model, inside the
   container.** The host treats everything under the mount, `.git` included,
   as untrusted bytes.
5. **Regression tests (git only, no docker) that MUST pass:** the exact
   `core.fsmonitor` escape; the `filter`/`.gitattributes` escape; and a
   **concurrent-writer TOCTOU** harness (a loop re-mangling `.git` while the
   host builds the result) — each asserting **no planted payload executes**,
   which holds trivially once rule (1) is met because no `git` runs at all.

This supersedes the earlier draft's "(1)–(3) hardening, may move to (4)"
framing, which the second pass proved unsafe: the hardening path shipped a
demonstrable operator-level RCE. There is no (1)–(3) path. §5.1's
`git add -A`/`git diff --cached` wording is **overridden by this section** —
where they conflict, this section wins, and §5.1 is implemented as the
byte-snapshot of rule (2).

### 6.6 Cleanup must not depend on `git worktree remove` succeeding

Also **demonstrated**: after the same `.git` tampering, `git worktree remove
--force` fails (`validation failed, cannot remove working tree ... error code
10`) and the worktree stays on disk **and** stays registered in the user's real
repo — directly contradicting §9's promise. Cleanup is therefore layered and
each layer is unconditional:

1. `docker rm -f <container>` (shielded + bounded, per §6 item 3).
2. attempt `git worktree remove --force` — but run it **in the REAL repo**
   (`git -C <repo> worktree remove --force <path>`), never inside the worktree,
   so §6.5's no-git-in-the-worktree rule holds. This step is best-effort; a
   tampered `.git` makes it fail, which is expected.
3. **regardless of 2's result:** `rm -rf <worktree-path>` where
   `<worktree-path>` is the **literal absolute path recorded at creation**,
   never a path re-resolved through the (sandbox-controlled) worktree — so it
   cannot be redirected by a planted symlink. Then `git -C <repo> worktree
   prune` in the real repo.
4. verify both are gone (path absent on disk **and** absent from
   `git -C <repo> worktree list`); any residue is surfaced to the caller as an
   error, never swallowed.

§10's tests must include a run with a **deliberately mangled `.git`** and
assert zero residue — the container-crash case alone does not cover this.

### 6.7 Docker Desktop on macOS — the uid claim is [plausible], not [verified]

§6.2's `--user <host-uid>` rationale is **Linux-native Docker behaviour stated
as fact**. The deployment target is a Mac running Docker Desktop, which runs a
LinuxKit VM and shares bind mounts through virtiofs/gRPC-FUSE that
*synthesizes* ownership to the container's runtime uid. On that platform the
mount is typically writable regardless of `--user`, host uid 501 maps to no
container user, and macOS gid 20 (`staff`) lands on an unrelated Linux group.
So the "dubious ownership" failure the text predicts may simply not occur
there — and the confident framing would mislead whoever debugs the first
permission error. **The plan must include a spike on the real host** to
establish which of `--user <host-uid>`, `--user 0:0` (root-in-container is not
host root under Docker Desktop), or no `--user` is correct **before** the uid
text is treated as settled. Until then it is [plausible].

## 7. Termination — mechanical, not aspirational

The loop stops on the **first** of:

- `max_turns` reached
- `max_seconds` reached (wall clock, checked before every model call)
- **no-progress:** `N` consecutive turns (default 5) in which **none** of
  three signals moved:
  (a) the content hash of the worktree — computed by the **same host-side
  byte walk as §5.1/§6.5, never by `git add -A`** (running git in the worktree
  every turn is exactly the escape §6.5 forbids). It covers tracked *and* new
  untracked files (host-owned `.gitignore` honoured), because a model whose
  whole job was to create a new file must not be scored "no progress" for it;
  (b) the **(command, exit status) pair** of the most recent `run_command` —
  a *different* command, or the same command flipping fail↔pass, both count;
  (c) any `write_file` call at all in the turn.
  "Progress" is defined purely mechanically. It deliberately does **not**
  hash command *output*: test runners print timestamps and durations, so an
  output hash would change every run and silently disable the detector. It
  also does not try to recognise "tests" by name. What it catches is the
  "productively-looking but stuck" pattern — re-reading the same files,
  re-running the same failing command — that turn/time caps alone let burn
  the whole budget. What it tolerates: a few consecutive read-only
  investigation turns are normal early in a task; `N=5` is sized so that
  legitimate exploration does not trip it, and it is a tunable, not a
  constant
- the model returns a final message with no tool calls (`completed`)
- an unrecoverable error (docker/worktree failure)

Whatever exists at that moment is returned with the reason. **The agent has no
way to extend its own budget.** This mirrors the convergence policy's
mechanical budgets and exists for the same reason.

## 8. Two honesty clauses — read these before relying on the tool

### 8.1 Local models are measured-unreliable as defect gates

This fleet's own benchmarks (2026-07-31 336-call production-size A/B; 2026-08-13
wide corpus) established that **no local model caught planted CRITICAL
credential-leak or injection regressions**, and that they silently APPROVE
large diffs with zero findings. That evidence is exactly why the Qodo step is
labelled "advisory smoke-check, not a defect gate."

**Consequence for this tool:** `coding_agent` is well-suited to **mechanical,
externally-verifiable work** — where "done" is a test going green, a symbol
renamed everywhere, a pattern applied across files — and **poorly suited to
tasks requiring judgment** (security-sensitive changes, subtle logic, anything
where the model must decide whether the result is *right* rather than whether
it *runs*). The diff gate in Claude's hands is not decoration: it is the
control that compensates for this, and it matters *most* precisely on the
tasks where the local model is weakest. Task descriptions should therefore
carry a **verifiable success criterion** whenever possible; the tool should
nudge (not block) when one is absent.

### 8.2 Toolchain duplication is the real operating cost

`--network=none` is the boundary that makes everything else safe — and it
means the container **cannot `pip install`, `npm ci`, `uv sync`, or fetch
anything**. Every toolchain the agent might need (Python + `uv` + `pytest` +
`ruff`, Node, shell utilities, `git`) must **already exist in the image**.

That image is a maintenance surface. It will drift from what is installed on
the host, and the first time the agent needs a tool the image lacks, the task
fails inside the sandbox rather than "just working" the way a shell on the
host would. This is a **deliberate trade**: friction on the operator's side in
exchange for a boundary that does not depend on trusting the model. It should
be stated plainly in the tool description and in the README, not discovered
by the first user. Mitigation, in scope for v1: a `coding-agent-image`
build script in `scripts/` that pins the toolchain, and a `--check` mode that
reports which expected tools the current image is missing.

## 9. Error handling

- Docker absent / daemon down → tool errors immediately with a one-line
  remedy; nothing is created.
- Worktree creation fails (dirty base ref, bad path) → error, no container.
- Container dies mid-loop → `stop_reason: error`, transcript up to that point
  is still returned, cleanup runs.
- Model returns malformed tool calls → fed back as a tool error result (the
  agentic benchmark showed gemma4:31b recovers from these); counts toward
  no-progress if repeated.
- Any exception in the loop → `finally` cleanup, then error surfaced. **A
  leaked container or worktree is a bug, not an acceptable outcome.**

## 10. Testing

- **Unit (no docker):** stop-condition logic (each of the five, in isolation
  and racing); no-progress detector against synthetic transcripts; path-escape
  rejection for `write_file`/`read_file` including symlink, `..`, and
  not-yet-existing-file (parent-resolution) cases; transcript/diff assembly
  including **new untracked files present in the diff**; diff size cap and
  binary-blob handling; single-slot container lock.
- **Security regression (git only, no docker needed — MUST pass):**
  1. **`core.fsmonitor` escape:** mangle the worktree's `.git` into a directory
     with a hostile `config`, run the host-side diff/hash path, and assert
     **no `git` subprocess was spawned against the worktree** and **no payload
     marker executed** — the byte-walk treats a mangled `.git` as opaque bytes
     to skip and still returns a diff. (It does **not** "detect tampering and
     error": under §6.5 there is nothing to detect and a mangled `.git` from a
     merely-buggy model is harmless. Asserting an error here would push the
     implementer back toward the check-then-use gate §6.5 proved unsafe.)
  2. the same with `core.hooksPath`, and with a `filter` named in an in-tree
     `.gitattributes` (the vector the env-allowlist could not stop);
  3. a **concurrent-writer TOCTOU** harness — a loop re-mangling `/work/.git`
     while the host builds the result — asserting no payload runs (trivially
     true once rule 1 holds: no `git` runs at all);
  4. **symlink read-exfiltration:** plant `/work/leak_abs ->
     ~/.ssh/id_ed25519`, a relative `..`-escape symlink, and a symlinked
     directory pointing outside the mount; assert the diff shows only the
     symlink **target text**, never the target's **contents**, that the
     content hash is invariant to the target file's contents (changes only
     when the target *string* changes), and that a planted FIFO does not hang
     the walk;
  4b. **diff-structure injection:** a symlink whose target is a forged
     `--- a/… / +++ / @@` block cannot introduce a column-0 diff header and
     the `@@` counts stay accurate;
  4c. **gate cannot be blinded (the load-bearing one):** an edit to a
     *tracked* file that matches a base `.gitignore` pattern **still appears**
     in the diff; and a tracked file replaced by a skipped special file
     (`rm foo.py; mkfifo foo.py`) surfaces as a deletion/typechange, not a
     silent disappearance;
  5. **cleanup with a deliberately mangled `.git`** (§6.6): assert zero
     residue on disk *and* zero stale entries in `git -C <repo> worktree list`.
  These exist because every one of these escapes was **demonstrated** across
  three adversarial passes, not hypothesised.
- **Integration (docker required, skipped if absent):** a real loop against a
  fixture repo with a deliberately failing test and a fake Ollama that emits a
  scripted tool-call sequence — asserts the diff, the `completed` stop, and
  **that the container and worktree are gone afterwards**. Plus: one run that
  simulates a container crash; one that fires the **outer wall-clock ceiling
  mid-`run_command`** and asserts the shielded cleanup still removed both.
- **Platform spike (manual, before implementation):** the Docker-Desktop-on-
  macOS uid behaviour in §6.7 — settle `--user` empirically on JVMBPro.
- **Live smoke (manual, documented):** one real task against `gemma4:31b`.
- Follows the repo's existing conventions (`test_local_delegate.py` style,
  ruff, mypy strict).

## 11. Explicitly out of scope for v1

- Applying the diff automatically (Claude gates it — by design, forever).
- Network access of any kind inside the sandbox.
- Multiple concurrent coding-agent containers.
- Running on JVMacMini (32GB, one model resident, `MAX_LOADED_MODELS=1` — no
  honest capacity).
- Per-step / checkpointed approval modes (rejected in design; §3).
- Any change to `local_delegate`'s contract.

## 12. Versioning and delivery

Ships as a minor version bump of ai-tools-mcp (new tool = shipped behaviour),
`mcpb/manifest.json` **and** `.claude-plugin/plugin.json` in lockstep, through
the review pipeline as a **T2** change. **Prerequisite in the same delivery:**
add `gemma4:31b-nvfp4` (and `qwen3.8:27b-nvfp4`) to
`_OLLAMA_BUILTIN_DELEGATE_MODELS` and retire the deleted `qwen3.6` tags from
it — the allowlist is the ceiling this tool resolves against, and today it
does not contain the model this spec chooses. That change is a code edit to
a shared default and must be reviewed as such, not slipped in (it introduces arbitrary execution and
a container boundary into a server that previously had none — that is a
security-boundary change under the convergence policy, and it should get the
focused adversarial review T2 prescribes).
