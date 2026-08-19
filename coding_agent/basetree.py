"""Read the base ref's tree from the user's REAL repository.

The ONLY module in coding_agent that runs git — and only ever as
`git -C <repo>` against the trusted checkout named by the MCP `repo`
parameter. It must NEVER be pointed at the sandbox worktree (spec §6.5):
pointing git at a directory the model can write is how two adversarial
passes reached host RCE, because a `.git` the model controls carries
hooks and config that git executes.

Nothing here reads, or is influenced by, the worktree: `repo` is the
host-supplied path, and every ignore rule comes from the committed base
tree plus the host's own `$GIT_DIR/info/exclude`. The sandbox cannot
author an ignore rule.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import pathspec

from .walk import Entry

# No PATH: git is resolved through os.defpath (/bin:/usr/bin), so an ambient
# PATH cannot substitute a different `git`. No HOME either, which together
# with GIT_CONFIG_NOSYSTEM keeps the read deterministic — a user's global
# config (core.excludesFile in particular) must not change what the human
# sees in the review diff.
_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}

# `ls-tree -r` yields blobs (100644/100755 regular, 120000 symlink) and
# 160000 gitlinks. A gitlink's sha names a COMMIT, so it must be filtered
# out BEFORE `cat-file blob` is asked for it, or reading the base tree of
# any repo with a submodule aborts outright.
_BLOB_MODES = {"100644": "file", "100755": "file", "120000": "symlink"}

# The mode column is also where the executable bit lives, and reading it here
# is half of a fix that is only correct as a whole: `walk.py` reports the bit
# from the worktree, and if this side did not, every file committed 100755
# would look like a fresh `chmod +x` in the very first diff. The two halves
# landed together, and `test_coding_agent_walk.py` pins them against a real
# checkout of a real repository rather than against each other.
_EXEC_MODE = "100755"

_GITIGNORE = ".gitignore"


@dataclass(frozen=True)
class BaseTree:
    entries: dict[str, Entry]
    ignore: Callable[[str], bool]  # tracked-UNaware raw pathspec matcher
    tracked: frozenset[str]


def _git(repo: str, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, env={**_GIT_ENV}
    ).stdout


def _matcher(sources: list[tuple[str, str]]) -> Callable[[str], bool]:
    """Compose ignore files the way git composes them.

    `sources` is (dir_prefix, text), lowest precedence first. git evaluates
    shallower ignore files first and lets deeper ones override, and within a
    single file the LAST matching pattern wins — which is how `!keep.log` in
    `pkg/.gitignore` re-includes a file that a root `*.log` excluded. Taking
    the first match instead would ignore that re-inclusion and drop the file
    from the diff: under-showing, the dangerous direction (§6.5).
    """
    specs = [
        (prefix, pathspec.PathSpec.from_lines("gitwildmatch", text.splitlines()))
        for prefix, text in sorted(sources, key=lambda src: src[0].count("/"))
    ]

    def raw_ignore(path: str) -> bool:
        verdict = False
        for prefix, spec in specs:
            if prefix and not path.startswith(prefix):
                continue  # this ignore file governs a different subtree
            sub = path[len(prefix) :]
            if not sub:
                continue  # a .gitignore never governs its own directory
            result = spec.check_file(sub)
            if result.include is not None:
                verdict = result.include
        return verdict

    return raw_ignore


def _repo_local_excludes(repo: str) -> list[tuple[str, str]]:
    """`$GIT_DIR/info/exclude` of the REAL repo — host-owned, not committed.

    Lowest precedence of everything considered here, so it is returned to be
    placed first. The answer is joined onto `repo` because resolving it
    against this process's cwd would read some OTHER repository's exclude
    file, letting a stranger's patterns hide paths from the gate.

    `os.path.join` is correct for BOTH shapes `rev-parse --git-path` returns,
    and that is worth stating because only one of them is obvious. In an
    ordinary checkout the answer is RELATIVE (`.git/info/exclude`) and the
    join does what it looks like. In a LINKED WORKTREE — which the MCP
    surface accepts as `repo`, and which this project is developed in — it is
    ABSOLUTE (`/…/ai-tools-mcp/.git/info/exclude`, verified), and
    `os.path.join` discards its left operand. Both land on the right file; a
    reader who assumes only the relative case will be surprised by the second.
    """
    try:
        rel = _git(repo, "rev-parse", "--git-path", "info/exclude")
        path = os.path.join(repo, rel.decode("utf-8", "surrogateescape").strip())
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [("", fh.read())]
    except (subprocess.CalledProcessError, OSError):
        return []


def read_base_tree(repo: str, ref: str) -> BaseTree:
    raw = _git(repo, "ls-tree", "-r", "-z", ref)
    entries: dict[str, Entry] = {}
    ignore_sources: list[tuple[str, str]] = []  # (dir_prefix, text)
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, path_b = rec.split(b"\t", 1)
        mode, _typ, sha = meta.decode().split(" ")
        kind = _BLOB_MODES.get(mode)
        if kind is None:
            continue  # 160000 gitlink (and 040000 tree): not a file, not a blob
        path = path_b.decode("utf-8", "surrogateescape")
        data = _git(repo, "cat-file", "blob", sha)
        entries[path] = Entry(path, kind, data, mode == _EXEC_MODE)
        # git does not honour a SYMLINKED .gitignore, so neither does this.
        if kind == "file" and (path == _GITIGNORE or path.endswith("/" + _GITIGNORE)):
            prefix = path[: -len(_GITIGNORE)]
            ignore_sources.append((prefix, data.decode("utf-8", "replace")))
    sources = _repo_local_excludes(repo) + ignore_sources
    return BaseTree(
        entries=entries, ignore=_matcher(sources), tracked=frozenset(entries)
    )


def make_ignore(base: BaseTree) -> Callable[[str], bool]:
    """Tracked-aware ignore: NEVER ignores a path in the base tree.

    This is where git's rule — ignore never applies to a TRACKED file —
    becomes code. Adversarial pass 4 used a uniform ignore predicate to hide
    edits to tracked files from the human's review diff; both early returns
    below exist to stop that, and neither is redundant.
    """

    def ign(path: str) -> bool:
        p = path.rstrip("/")
        if p in base.tracked:
            return False
        # A directory prefix containing tracked files must be walked. Load
        # bearing, and NOT subsumed by the check above: walk.py PRUNES a
        # directory whose query answers True, so an ignored directory holding
        # a force-added tracked file would never be descended into and that
        # file would never be queried at all — it would surface as a spurious
        # deletion, and an edit to it would be invisible to the gate.
        if path.endswith("/") and any(t.startswith(path) for t in base.tracked):
            return False
        return base.ignore(path)

    return ign
