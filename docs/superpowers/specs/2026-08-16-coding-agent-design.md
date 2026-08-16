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
| Model | `gemma4:31b-nvfp4`, resolved via role config | Scored 1.000 on the 8-scenario agentic benchmark and was the **only** model that reliably recovered from a tool error — the failure mode that matters most in a loop |

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
exfiltration, no remote-code-fetch, no phone-home — with nothing to argue
about. The alternative (agent process inside the sandbox calling Ollama)
would force a network hole open and then require policing what it reaches.

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
  "diff": "<unified diff, worktree vs base_ref>",
  "changed_files": ["src/x.py", "tests/test_x.py"],
  "last_command": {"cmd": "pytest tests/test_x.py", "exit": 0, "output_tail": "..."},
  "transcript": [ {"turn":1,"tool":"read_file","args":{...},"result_head":"..."}, ... ],
  "model": "gemma4:31b-nvfp4"
}
```

`completed` means the model emitted a final message with no further tool
calls. It is **not** a claim of correctness — see §8.

### `coding_agent_result`

Identical semantics to `local_delegate_result`: poll by `job_id`, single-collect,
jobs do not survive a server restart.

### Tools the model sees (inside the sandbox)

| tool | behaviour | limits |
|---|---|---|
| `list_files(path=".")` | recursive listing, respects `.gitignore` | capped entries |
| `read_file(path)` | returns text | size cap; binary rejected |
| `write_file(path, content)` | full overwrite | path must resolve inside the worktree; **no symlink escape** (resolve real path, then prefix-check) |
| `run_command(cmd)` | `sh -c` inside the container | per-command timeout; output truncated with a marker; **no network exists to reach** |

## 6. Isolation, precisely

Per task:

1. `git worktree add <tmp> <base_ref>` — throwaway, detached.
2. `docker run --rm --network=none --user <non-root> --read-only`
   `--tmpfs /tmp --cpus <cap> --memory <cap> --pids-limit <cap>`
   `-v <worktree>:/work:rw -w /work <image> sleep infinity`
   (a long-lived container per task; each tool call is `docker exec` into it,
   avoiding per-command startup cost).
3. On any stop condition **or** any error: `docker rm -f`, then
   `git worktree remove --force`. Cleanup is unconditional (`finally`).
4. Concurrency: **one** coding-agent container at a time on this host, on top
   of the existing delegate job cap. The 64GB box already keeps a 31B model
   resident; a second container plus a second inference stream is not honest
   capacity.

Path safety in `write_file`/`read_file` is enforced **on the host side** by
resolving the real path and prefix-checking against the worktree root, *in
addition to* the container's own mount boundary. Belt and braces, because a
mount boundary that is the only check is a single point of failure.

## 7. Termination — mechanical, not aspirational

The loop stops on the **first** of:

- `max_turns` reached
- `max_seconds` reached (wall clock, checked before every model call)
- **no-progress:** `N` consecutive turns (default 5) in which **neither** of
  two signals moved: (a) the content hash of the worktree's tracked files, and
  (b) the exit status of the most recent `run_command`. "Progress" is defined
  purely mechanically — a file changed, or a command that was failing now
  exits 0 (or vice-versa). This deliberately avoids trying to detect "tests"
  by name or output-parsing: it catches the "productively-looking but stuck"
  pattern (re-reading the same files, re-running the same failing command)
  that turn/time caps alone let burn the whole budget, without any brittle
  heuristic about what counts as a test
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
  rejection for `write_file`/`read_file` including symlink and `..` cases;
  transcript/diff assembly; job registry reuse.
- **Integration (docker required, skipped if absent):** a real loop against a
  fixture repo with a deliberately failing test and a fake Ollama that emits a
  scripted tool-call sequence — asserts the diff, the `completed` stop, and
  **that the container and worktree are gone afterwards**. Plus one run that
  simulates a container crash and asserts cleanup.
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
