# Spike: settle `--user` on Docker Desktop for macOS (§6.7)

**Date:** 2026-08-18
**Status:** VERIFIED (re-run after Docker Desktop repair; a prior attempt on
2026-08-17 was BLOCKED because the Docker Desktop app bundle was broken — see
`task-0-report.md` history / `progress.md` for the repair log)

## Why this spike exists

The design spec originally asserted the container must run with
`--user <host-uid>:<host-gid>` so it can write a bind-mounted git worktree,
citing git's `fatal: detected dubious ownership` error otherwise. Adversarial
review flagged this as Linux-native Docker behaviour asserted for a macOS
target: Docker Desktop runs a LinuxKit VM and shares bind mounts via
virtiofs, which may synthesize file ownership rather than enforcing raw host
uid/gid matching. This spike empirically settles which `--user` value (if
any) `coding_agent/sandbox.py` should pass to `docker run`.

## Step 1 — daemon status and host identity

```
$ docker info 2>/dev/null | grep -iE 'Operating System|OSType|Server Version|virtiofs|gRPC'
 Server Version: 29.7.2
 Operating System: Docker Desktop
 OSType: linux

$ id -u
501
$ id -g
20
```

Four facts recorded:
- **Operating System:** Docker Desktop
- **Server Version:** 29.7.2
- **Host uid:** 501
- **Host gid:** 20 (`staff`)

Note: `virtiofs` and `gRPC` did not appear as literal lines in `docker info`
output on this Docker Desktop/server version — those implementation details
aren't surfaced by `docker info` directly (the mount driver is virtiofs by
Docker Desktop's current default on macOS, but that's a Docker Desktop
Settings fact, not something `docker info` prints). This doesn't affect the
spike's conclusions, which are based on direct behavioral probes, not on
`docker info` self-reporting the mount driver.

## Deviation from the brief (authorized)

The brief's Step 2 says to run `git worktree add /tmp/ca-spike-wt HEAD` off
this repo. That was **not** done. Two reasons, both given by the task
authorizer: (1) this session is worktree-isolated behind a PreToolUse gate
that blocks git operations targeting paths outside the current worktree; (2)
a concurrent agent session is committing to this same repo and shared git
state should not be touched.

**Substitution used instead:** a fully independent scratch repo —
`mkdir /tmp/ca-spike-repo && git init` inside it, with a committed
`mcp_server.py` file — used as the bind-mount target for all three `docker
run` probes. Cleaned up (`rm -rf`) after each probe and again at the end.

**Does the substitution affect validity?** No, for the specific things this
spike measures. The failure mode under test (`dubious ownership` / write
failure due to uid mismatch between the bind-mounted directory's host owner
and the container's declared user) is a property of the **bind mount and the
container's own filesystem-level uid**, not of whether the mounted directory
happens to be a git worktree versus a plain repo. `git worktree add` and
`git init` produce the same kind of thing for this purpose: a directory with
regular files owned by the host user, checked into git. The one thing a
scratch `git init` repo does *not* reproduce is `git`'s own worktree
metadata (`.git` as a file pointing at a linked worktree's gitdir elsewhere)
— but the brief itself says to ignore the git-behavior probe entirely for
this task ("Ignore the git line for `python:3.12-slim`; git-in-container is
tested with the real image in Task 11"), so that gap doesn't matter here.

One incidental artifact of using `/tmp` instead of the real repo path: `/tmp`
on this Mac has BSD group-inheritance behavior that gave newly created files
group `wheel` (0) instead of the process's primary group `staff` (20) — see
"Surprising / extra observations" below. This is a quirk of `/tmp`
specifically, not of the git-repo-vs-worktree substitution; the real target
directory (`~/Documents/Code/ai-tools-mcp/...`) was separately confirmed to
use group `staff` as expected (see below).

## Step 3 — per-variant probe results

Ran the brief's probe (id / stat / write) against all three `--user`
variants, against the scratch repo described above. `apt-get`/git-in-container
lines were dropped per the brief's own instruction to ignore them for this
image. A host-side ownership check (not in the brief, added for completeness
— see below) was appended after each run.

| Variant | Container `id` | Container `stat` of tracked file | Write result | Host-side owner of file the container just wrote |
|---|---|---|---|---|
| (a) no `--user` | `uid=0(root) gid=0(root)` | `0:0` | **WRITE_OK** | `501:0` → `jasonvassallo:wheel` |
| (b) `--user 0:0` | `uid=0(root) gid=0(root)` | `0:0` | **WRITE_OK** | `501:0` → `jasonvassallo:wheel` |
| (c) `--user 501:20` | `uid=501 gid=20(dialout)` | `501:20` | **WRITE_OK** | `501:0` → `jasonvassallo:wheel` |

Raw output, variant (a) — no `--user`:
```
== id ==
uid=0(root) gid=0(root) groups=0(root)
== stat ==
0:0 /work/mcp_server.py
== write ==
WRITE_OK
```

Raw output, variant (b) — `--user 0:0`:
```
== id ==
uid=0(root) gid=0(root) groups=0(root)
== stat ==
0:0 /work/mcp_server.py
== write ==
WRITE_OK
```

Raw output, variant (c) — `--user 501:20`:
```
== id ==
uid=501 gid=20(dialout) groups=20(dialout)
== stat ==
501:20 /work/mcp_server.py
== write ==
WRITE_OK
```

Note on variant (c): the container has no `/etc/passwd` entry for uid 501,
so `id` prints the bare number with no name — expected and harmless (Linux
permission checks operate on the numeric uid, not a passwd entry). gid 20
resolves to `dialout` inside the container's `/etc/group`, **not** `staff` —
exactly the "macOS gid 20 (staff) lands on an unrelated Linux group" concern
raised in the task brief. It's cosmetic here (the numeric gid is what governs
permissions, and no group-name-based logic was exercised), but any future
code that resolves the *name* rather than the *number* would get the wrong
answer.

## Result: the spec's original claim is REFUTED

**All three variants gave `WRITE_OK`.** Docker Desktop's virtiofs bind mount
does not require the container's declared uid/gid to match the host owner in
order to write — the "dubious ownership" write-failure scenario the spec's
`--user <host-uid>` rationale was designed to prevent does not occur here
under any of the three tested variants, including plain root with no `--user`
flag at all.

### Extra finding beyond the brief's ask: host-side ownership is stable regardless of container uid

Not asked for by the brief, but directly relevant to *why* the flag choice
doesn't matter for write success, and important for whoever consumes the
worktree after the container exits: after each variant, the check was
extended to `stat` the newly-written file from the **host** side (outside
the container). In all three cases — including the container running as
literal root (uid 0) — the host saw the new file as owned by `501`
(`jasonvassallo`), the real macOS user, never `0`/root and never floating to
whatever uid the container declared. Docker Desktop's virtiofs bridge
process runs as the host user, and that's what actually creates the file on
the host side; the uid/gid the container sees is a synthesized view for the
container's own permission checks, decoupled from what lands on the host
filesystem. Practical consequence: **no `--user` choice can leave the host
user locked out of files the container wrote** on this platform. This is the
mechanism, not just the anecdote, behind why all three variants wrote
successfully.

## Surprising / extra observations

- **Group mismatch under `/tmp`:** the scratch repo's directory itself
  showed host-side files with group `wheel` (0), not `staff` (20) — even
  though `id -g` for the invoking shell is 20. This is BSD directory
  group-inheritance in `/tmp` (new files inherit the parent directory's
  group, not the creating process's primary group), unrelated to Docker.
  Confirmed this is a `/tmp`-specific quirk, not representative of the real
  target directory: `stat` on `~/Documents/Code/ai-tools-mcp` itself (the
  actual repo Task 5 will operate against) shows `501:20`
  (`jasonvassallo:staff`) as expected. Anyone re-running this spike under a
  different scratch path should not be alarmed by a `wheel` group showing up
  under `/tmp` — it isn't a Docker/virtiofs effect.
- **`docker info` doesn't print `virtiofs`/`gRPC` lines** on this Docker
  Desktop/server version, contrary to the brief's expected grep output. Not
  a blocker — the spike's conclusions rest on the behavioral write/stat
  probes, not on `docker info` self-description of its mount driver.
- **This spike fully refutes rather than confirms the spec's original
  premise** — the [plausible]-marked claim that `--user <host-uid>:<host-gid>`
  is *necessary* to avoid a write failure was wrong for this platform; write
  succeeded in every configuration tested, including plain container-root.

## Decision

Selection rule (from the brief): prefer the variant that gives `WRITE_OK`
**and** does not run as container-root; accept `--user 0:0` only if that's
the *only* configuration that writes.

All three variants gave `WRITE_OK`, so the tie-break is "does not run as
container-root." Variant (c) — `--user <host-uid>:<host-gid>` — is the only
one of the three that avoids running the sandboxed process as root inside
the container. Although this spike shows host-side file ownership is
unaffected by which variant is chosen (see finding above), running as a
non-root, unprivileged uid inside the container is still the better default
on security-in-depth grounds (a compromised or misbehaving agent process
gets a plain-uid's capabilities, not root's, inside the sandbox's own
namespace) with zero measured downside — variant (c) writes exactly as
reliably as (a)/(b).

The value is expressed as a runtime-computed expression rather than the
literal digits `501:20` observed on this Mac, because nothing in this
spike's findings depends on those specific numbers — the mechanism verified
here (virtiofs bridges through the real host user regardless of declared
container uid; the container-declared uid only needs to be *some*
non-privileged value to get the least-privilege benefit) generalizes to
whoever's account actually runs `coding_agent` at the time. A value hardcoded
to `"501:20"` would be silently wrong if this tool is ever run under a
different macOS account. `os.getuid()`/`os.getgid()` read the real invoking
user at the moment `sandbox.py` builds the `docker run` command, which is
the direct generalization of what variant (c) tested here.

```
DECISION: SANDBOX_USER_FLAG = ["--user", f"{os.getuid()}:{os.getgid()}"]
```

If Task 5's implementer wants a pure static list literal instead (e.g. for
constant-folding, testability, or because `sandbox.py` doesn't otherwise
import `os` at module scope), the equivalent fixed value **for this specific
Mac and this specific user account only** is `["--user", "501:20"]` — but
that substitution should be made deliberately, not silently, since it
reintroduces the exact single-machine assumption this decision avoids.

## Cleanup

`/tmp/ca-spike-repo` (the scratch repo substituted for the brief's
`/tmp/ca-spike-wt` worktree) was removed with `rm -rf` after the probes
completed. No `git worktree` was ever created, so no `git worktree remove`
was needed.
