# syntax=docker/dockerfile:1
#
# coding_agent sandbox image (spec §8.2).
#
# The container this image backs is started by coding_agent/sandbox.py with
# --network=none and --read-only. Once it is running, NOTHING can be
# installed or fetched: no pip install, no npm install, no apt-get, no `uv
# run --with X` (that needs PyPI). Every tool the untrusted local model might
# need for mechanical coding work -- running tests, formatting, type-
# checking, shell-linting, using git on its own worktree -- must already
# exist in this image at build time.
#
# That makes this image a maintenance surface, not a one-time artifact: it
# WILL drift from whatever toolchain the host project actually uses.
# scripts/coding-agent-image/check.sh is the accepted mitigation -- it
# reports which of the tools below are missing from a built image (and the
# installed version of the ones that are present), so drift shows up as a
# report the operator can read, not as a confusing failure discovered mid-run
# inside a sandbox the human is trying to review.
#
# Reproducibility: the base image is pinned by tag AND digest (not the
# floating `python:3.12-slim`, which silently repoints to whatever the
# newest 3.12.x point release is whenever it's next pulled), and every pip
# package below is pinned to an exact version, not a wildcard range. A
# version bump is therefore a deliberate, visible diff to this file, not
# something that happens underneath a rebuild. Pinned as of 2026-08-19; see
# scripts/coding-agent-image's task report for how these were chosen and
# for the apt-package drift risk that is knowingly NOT closed the same way.
FROM python:3.12.14-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

# Debian's bookworm apt archive advances in place -- these are NOT
# version-pinned the way the base image and pip packages are, so a rebuild
# months from now can silently pick up newer git/node/npm/shellcheck. That
# is an accepted, documented drift risk (see the task report), not an
# oversight: pinning exact .deb versions ages out of the mirror and turns
# into build breakage rather than reproducibility. check.sh's per-tool
# version print is the mitigation -- an operator diffing two check.sh runs
# sees exactly which tool moved.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        nodejs \
        npm \
        shellcheck \
    && rm -rf /var/lib/apt/lists/*

# uv, pytest, ruff, mypy: pinned to an exact release, not a wildcard range.
# ruff is pinned to match the version this repo's own CLAUDE.md-mandated
# `ruff format` / `ruff check .` gate runs on the host (confirmed via `ruff
# --version` at image-authoring time), so a diff the sandboxed model
# produces and the host's own lint pass agree on formatting/lint verdicts.
RUN pip install --no-cache-dir \
        uv==0.12.5 \
        pytest==9.1.1 \
        ruff==0.16.3 \
        mypy==2.3.1

# CPython's __pycache__ .pyc cache is invalidated by (mtime, size), not by
# content hash. When the sandboxed model fixes a bug with a same-length edit
# within the same wall-clock second the original file was compiled, that
# invalidation check can pass on stale bytecode: pytest re-runs against the
# OLD compiled code and reports the (already-fixed) failure again. The model
# then "re-fixes" code that was already correct, burning turns against its
# no-progress budget on this tool's core use case -- fix a failing test,
# rerun pytest, see it pass. Measured n=4 per cell during Task 11: a
# same-length edit landed on stale bytecode roughly half the time, and an
# added delay does NOT help, because the delay elapses AFTER the write, not
# before the next read. Disabling .pyc writes entirely removes the
# invalidation check instead of trying to outrun its resolution.
ENV PYTHONDONTWRITEBYTECODE=1

# git refuses to operate on a repository whose directory owner doesn't match
# the running uid ("detected dubious ownership"). This does NOT fix git
# inside this sandbox on Docker Desktop for macOS -- Task 11 measured two
# separate reasons git is unusable there regardless of this config:
#   (a) "dubious ownership" is UNREACHABLE at any --user on that platform,
#       because a bind mount presents files as owned by the container's own
#       uid, so the ownership check that safe.directory would waive never
#       fires in the first place; and
#   (b) even so, every git command still dies `fatal: not a git repository`,
#       because a linked worktree's .git is a FILE naming
#       <repo>/.git/worktrees/<name>, and only the worktree itself -- not
#       that path inside the main repo's .git -- is bind-mounted into
#       /work. There is nothing for git to find at the location the file
#       points to.
# The config is kept anyway because it IS load-bearing on Linux hosts: a
# real (non-Docker-Desktop-macOS) bind mount passes the host's real uid/gid
# through, so the ownership mismatch this waives is reachable there. Must be
# a SYSTEM config (/etc/gitconfig via --system), not a per-user
# ~/.gitconfig: the runtime uid has no /etc/passwd entry and therefore no
# resolvable home directory for a user-level config to live in.
RUN git config --system --add safe.directory /work

# A named, non-root fallback account for anyone running this image directly
# (`docker run -it ai-tools-coding-agent:latest bash`) without the --user
# flag coding_agent/sandbox.py always passes at runtime. Real coding_agent
# runs never use uid 1000 -- sandbox.py overrides it with the caller's own
# host uid:gid -- so nothing depends on this specific uid; it exists only so
# the image isn't root-only by default for manual/debugging use.
#
# The home directory is deliberately NOT /home/agent, even though the user
# is named "agent". The real runtime always mounts --tmpfs /home/agent
# (coding_agent's Global Constraints), and empirically, when a tmpfs target
# path already exists as a real directory baked into the image, Docker
# inherits THAT directory's mode for the tmpfs mount instead of tmpfs's own
# safe default (1777, world-writable+sticky, same as /tmp). `useradd -m`
# creates /home/agent at mode 755 owned by uid 1000 -- so a naive `useradd -m
# -u 1000 agent` here would make every real coding_agent run's --tmpfs
# /home/agent mount inherit mode 755 owned by uid 1000, which the runtime's
# actual --user <host-uid>:<host-gid> (almost never uid 1000) cannot write
# to. That would silently break anything that touches $HOME -- `git config
# --global`, or any tool that lazily creates ~/.cache or ~/.config -- only
# inside the sandbox, not in this build step, so it would surface as a
# confusing in-container failure rather than a build error. Pointing this
# user's home at /var/lib/agent instead means /home/agent is never created
# in the image, so the runtime's tmpfs mount there gets Docker's normal
# 1777 default. Verified by rebuilding with each variant and diffing `stat
# -c '%a %U:%G' /home/agent` after `--tmpfs /home/agent` -- see the task
# report for the two mode/owner sets observed.
RUN useradd -m -u 1000 -d /var/lib/agent agent

WORKDIR /work
