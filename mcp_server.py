#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=1.0.0",
#     "mcp>=1.0.0",
#     "httpx>=0.27",
#     "google-auth>=2.30",
#     "requests>=2.31",
# ]
# ///
"""
MCP server providing four families of tools:

- ``quick_research`` / ``deep_research``: Perplexity Sonar / Sonar Pro
  — inline research with citations. ``quick_research`` uses the smaller
  Sonar model for fast, concise, well-scoped answers; ``deep_research``
  uses Sonar Pro for multi-source synthesis when the question spans
  sources or needs cross-referencing.
- ``agent_research`` / ``agent_research_result``: Perplexity Agent API
  with the ``sandbox`` tool ("Search as Code") — the upstream agent
  writes and runs code in a Perplexity-hosted container, searching
  programmatically from inside that code. For bulk/enumerable research,
  computation over search results, and structured datasets. Runs take
  minutes; call synchronously or pass ``background=true`` and poll
  ``agent_research_result`` by response_id.
- ``gemini_deep_research_start`` / ``_result``: Gemini Deep Research —
  long-running (minutes, up to 60), citation-dense reports via
  Google's hosted research agent. Asynchronous: ``_start`` returns an
  interaction_id; poll ``_result`` until terminal status.
- ``list_sessions`` / ``save_session`` / ``load_session`` /
  ``update_session`` / ``delete_session``: local conversation-session
  persistence backed by ``~/.claude/sessions/``.

Designed to complement Claude's built-in WebSearch tool (quick factual
lookups, single-answer questions).

PLATFORM: macOS / POSIX only. The session helpers use ``fcntl.flock``
(Unix-specific) and the Perplexity-key lookup uses macOS's ``security``
CLI. Importing this module on Windows succeeds (so docs/inspection
tools work) but invoking ``update_session`` / ``delete_session`` will
raise OSError with a clear message (per PR #4 round-9 review, Gemini
medium L17: "fcntl module is Unix-specific"), and invoking
``deep_research`` will raise the keychain-lookup error from the
lazy ``_get_perplexity_client`` accessor (per PR #4 round-10 review,
Codex P2 L38: keychain lookup must be deferred so ``import mcp_server``
succeeds even when the ``security`` CLI is unavailable).

For the Gemini tools, Application Default Credentials (ADC) are
likewise loaded lazily on first ``gemini_*`` call rather than at
module import. This means the MCP server can start and the
Perplexity-backed ``deep_research`` and session tools can be used
even on a machine without ``gcloud auth application-default login``
having been run; only the ``gemini_*`` tools will fail when invoked.
"""

import asyncio
import getpass
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import google.auth
import google.auth.transport.requests
import httpx

# fcntl is POSIX-only; on Windows the import fails. We catch
# ImportError so the module can still be imported (e.g. for docs,
# tool discovery, or the deep_research path which doesn't need
# locking) — _session_lock raises a clean error if invoked on a
# non-POSIX platform (per PR #4 round-9 review, Gemini medium L17).
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised via mocked import
    fcntl = None  # type: ignore[assignment]

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI


# Patterns for secret-shape strings that may appear in scraped web content
# returned by upstream search providers. Applied at the response boundary so
# secrets do not get persisted in client transcripts.
#
# Order matters: the JWT pattern would otherwise eat substrings of nothing
# else here, but private-key blocks are matched first because they may
# contain other matchable substrings inside the body.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY_BLOCK]",
    ),
    (re.compile(r"ya29\.[A-Za-z0-9_-]+"), "[REDACTED_GOOGLE_OAUTH_ACCESS]"),
    (re.compile(r"1//0[A-Za-z0-9_-]{30,}"), "[REDACTED_GOOGLE_OAUTH_REFRESH]"),
    (re.compile(r"AIza[A-Za-z0-9_-]{20,}"), "[REDACTED_GOOGLE_API_KEY]"),
    # JWT minimums relaxed from {30,30,20} to {10,10,10} per PR #1 review
    # (Gemini): minimal valid header `{"alg":"HS256"}` encodes to 20 chars,
    # which the original {30,} requirement missed.
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "[REDACTED_JWT]",
    ),
    # Apple app-specific password format (xxxx-xxxx-xxxx-xxxx) was originally
    # included but removed per PR #1 review (Codex P2): the regex
    # \b[a-z]{4}-[a-z]{4}-[a-z]{4}-[a-z]{4}\b false-positives on ordinary
    # research-prose phrases like "real-time-data-flow" or
    # "zero-shot-text-only", silently mangling deep_research output. ASPs
    # leak via local-config / IMAP-debug paths this MCP doesn't touch, so
    # for prose-content redaction the safer trade is to drop the pattern.
)


def redact_secrets(value: Any) -> Any:
    """Recursively mask secret-shape substrings in arbitrary nested data.

    Walks strings, lists, tuples, and dicts; leaves other types untouched.
    Pure-stdlib (uses ``re``); no new dependencies.
    """
    if isinstance(value, str):
        for pattern, replacement in _REDACTION_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        # Collision-handling preserves all entries when two distinct
        # original keys redact to the same value:
        #   - String keys: append "#N" suffix.
        #   - Tuple keys (e.g. (api_key_1, "x") and (api_key_2, "x") both
        #     becoming ("[REDACTED_..]", "x") after recursion): append a
        #     "#N" string element to the tuple.
        #   - Other hashable types: fall through to last-write-wins (rare).
        out: dict[Any, Any] = {}
        for k, v in value.items():
            new_k = redact_secrets(k)
            new_v = redact_secrets(v)
            if new_k in out:
                if isinstance(new_k, str):
                    i = 2
                    while f"{new_k}#{i}" in out:
                        i += 1
                    new_k = f"{new_k}#{i}"
                elif isinstance(new_k, tuple):
                    i = 2
                    while (*new_k, f"#{i}") in out:
                        i += 1
                    new_k = (*new_k, f"#{i}")
            out[new_k] = new_v
        return out
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(v) for v in value)
    return value


def get_api_key_from_keychain(service: str, account: str) -> str:
    """Retrieve API key from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Keychain item not found. Add with:\n"
            f"  security add-generic-password -s '{service}' -a '{account}' -w 'YOUR_API_KEY'"
        )
    return result.stdout.strip()


_ADC_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def _load_adc() -> tuple[Any, str]:
    """Load Google Cloud Application Default Credentials.

    Returns (credentials, billing_project). Raises a clear error if ADC is not
    configured. The credentials object is refreshable — tokens are minted
    lazily per request via `_get_bearer_token`.
    """
    try:
        creds, project = google.auth.default(scopes=list(_ADC_SCOPES))
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise ValueError(
            "Google Cloud Application Default Credentials not found. "
            "Run: gcloud auth application-default login"
        ) from exc
    if not project:
        raise ValueError(
            "Could not determine billing project from ADC. Run: "
            "gcloud auth application-default set-quota-project YOUR_PROJECT"
        )
    return creds, project


def run_check() -> None:
    """Validate configuration and exit. Used by install.sh to verify setup."""
    errors = 0
    try:
        get_api_key_from_keychain("api_tokens", "perplexity")
        print("ok: perplexity key found in keychain")
    except ValueError as e:
        print(f"fail: {e}")
        errors += 1

    try:
        creds, project = _load_adc()
        # Force a refresh so a stale/expired ADC fails the check here rather
        # than at first tool call.
        creds.refresh(google.auth.transport.requests.Request())
        print(f"ok: google ADC valid (billing project: {project})")
    except (ValueError, Exception) as e:  # noqa: BLE001 - report any auth issue
        print(f"fail: {e}")
        errors += 1

    sys.exit(errors)


# ─── Session management storage ───────────────────────────────────────
#
# Sessions are persisted as ``~/.claude/sessions/<uuid>.json`` with shape:
#     {
#       "session_id": str,
#       "name": str,
#       "created_at": ISO-8601 UTC,
#       "last_modified": ISO-8601 UTC,
#       "messages": [{"role": ..., "content": ...}, ...],
#       "metadata": {...}
#     }
# Note: directory creation happens lazily inside save_session/update_session
# rather than at module load time. This avoids a side effect during import
# (per PR #3 follow-up review, Gemini medium): test suites import this
# module to introspect helpers — they should NOT have the user's real
# ~/.claude/sessions/ created as a side effect of the import.
SESSIONS_DIR = Path.home() / ".claude" / "sessions"


def _atomic_temp_for(target: Path) -> Path:
    """Create a unique-per-call empty temp file in ``target``'s directory.

    Used by save_session/update_session for atomic writes via
    ``os.replace(temp, target)``. ``tempfile.mkstemp`` ensures:

    1. **Uniqueness**: O_EXCL + randomized name → each concurrent
       writer gets its own inode. Without this, two writers sharing
       a fixed ``<sid>.json.tmp`` path can have one's still-open fd
       end up writing into the post-replace final file (per PR #4
       round-5 review, Codex P2 L392: "Use unique temp files").
    2. **Mode**: 0o600 by default on POSIX → session content stays
       owner-only at every moment, even before the rename lands.
    3. **Atomic creation**: no race window between ``open`` and
       ``write`` where another process could see a half-formed file.

    Returns the temp file's Path; the caller is responsible for
    writing to it and ``os.replace``-ing into place (or unlinking
    on error).
    """
    fd, path_str = tempfile.mkstemp(
        suffix=target.suffix + ".tmp",
        prefix=target.stem + ".",
        dir=str(target.parent),
    )
    # Close the descriptor — we'll open the path again with the
    # standard ``open()`` in the caller. mkstemp's role is just to
    # reserve a unique path with the right mode.
    os.close(fd)
    return Path(path_str)


@contextmanager
def _session_lock(session_file: Path):
    """Cooperative per-session lockfile (POSIX flock).

    Used by ``update_session`` / ``delete_session`` to serialize their
    critical sections against each other. This closes the resurrection
    race Codex flagged on PR #4 round 5 (P2 L429): without the lock,
    a concurrent ``delete_session`` could land between
    ``update_session``'s existence check and its ``os.replace``,
    leaving ``os.replace`` to recreate the just-deleted file.

    Lockfile path: ``<session_file>.lock`` (sibling). We pick
    ``.lock`` instead of ``.json.lock`` so ``list_sessions``'
    ``glob("*.json")`` doesn't accidentally enumerate it as a
    session. The lockfile persists across operations — we don't
    unlink on release (would create its own race with another
    waiter). 0o600 mode keeps it owner-only.

    LIMITATION: this is **advisory** locking — it only protects
    callers that go through ``update_session`` / ``delete_session``.
    A non-cooperating deleter (manual ``rm``, foreign tool not
    using this API) can still slip past the lock; ``update_session``
    keeps an existence re-check before its atomic write as a
    best-effort safeguard for that case, but the residual hairline
    race against non-cooperating processes can only be fully closed
    with platform-specific syscalls (``renameat2 RENAME_EXCHANGE``
    on Linux, ``renamex_np`` on macOS), which aren't portably
    exposed in stdlib Python.
    """
    if fcntl is None:
        # Non-POSIX platform (Windows). The session-mgmt path requires
        # advisory locking that fcntl provides; fail clearly rather
        # than fall through to a no-op lock that would silently let
        # the resurrection race re-open. (Per PR #4 round-9 review,
        # Gemini medium L17: "fcntl module is Unix-specific".)
        raise OSError(
            "Session management requires POSIX (macOS/Linux). "
            "fcntl is unavailable on this platform."
        )
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = session_file.with_suffix(".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Defensive: if mcp_server.fcntl was monkey-patched to None
        # mid-context (test fixtures, runtime mutation), the unlock
        # call would raise AttributeError and mask the real exception
        # this finally is trying to clean up after. Guard explicitly
        # rather than relying on the early check at the top of the
        # context manager (per PR #4 round-11 review, Gemini medium
        # L258).
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                # Lock release on a closed-or-already-released fd is
                # benign; we close the fd next anyway.
                pass
        os.close(fd)


def get_session_file(session_id: str) -> Path:
    """Return the on-disk path for a session id.

    Validates that ``session_id`` is a valid UUID to prevent path
    traversal (per PR #3 review, Codex P1). Without this check, an
    attacker-controlled session_id like ``"/tmp/victim"`` or
    ``"../../../etc/passwd"`` would resolve to a .json file OUTSIDE
    ``SESSIONS_DIR``, allowing the load/update/delete MCP tools to
    read, overwrite, or unlink arbitrary local files.
    """
    try:
        # Parse + canonicalize: uuid.UUID accepts braced and urn:uuid:
        # forms, so we re-stringify the parsed object to get the
        # canonical 36-char hyphenated lowercase form. The path then
        # provably contains only [0-9a-f-] — no path separators or
        # other shell-interesting characters can survive.
        # Note: the type hint says ``str`` but Python doesn't enforce
        # it at runtime; the except clause below catches the TypeError
        # that uuid.UUID raises on non-string inputs (per PR #4
        # follow-up review, Gemini nitpick L172).
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        raise ValueError(
            f"Invalid session_id: must be a valid UUID, got {session_id!r}"
        )
    return SESSIONS_DIR / f"{parsed}.json"


def list_sessions() -> list[dict[str, Any]]:
    """List all sessions, most-recently-modified first."""
    sessions: list[dict[str, Any]] = []
    if not SESSIONS_DIR.exists():
        return sessions

    for session_file in SESSIONS_DIR.glob("*.json"):
        # Skip stray ``.json`` files whose stem isn't a valid UUID.
        # Without this, a manually-dropped ``notes.json`` or backup
        # file in SESSIONS_DIR would be parsed as if it were a
        # session and either error out (skipped below) or appear
        # in the listing under a misleading id. UUID validation
        # mirrors get_session_file's check (per PR #4 round-7
        # review, Gemini medium L270: "filter for files whose names
        # are valid UUIDs").
        try:
            uuid.UUID(session_file.stem)
        except (ValueError, AttributeError, TypeError):
            continue
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # Defensive: skip files where the parsed JSON is not a session
        # object or where "messages" is present but not a list. Without
        # these guards, a syntactically-valid JSON file shaped like
        # ``[]`` or ``{"messages": "not-a-list"}`` would crash
        # data.get() or len() and abort the entire listing instead of
        # just skipping the bad file. Note: relying on the broad
        # except (json.JSONDecodeError, OSError, AttributeError,
        # TypeError) does NOT cover the "messages is a string" case
        # because len("string") returns 10, not a TypeError. Explicit
        # isinstance guards are clearer and correct
        # (per PR #3 follow-up review, Gemini medium + Codex P3).
        if not isinstance(data, dict):
            continue
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            continue
        sessions.append(
            {
                "session_id": session_file.stem,
                # str() coercion (per PR #4 review, Gemini medium): if a
                # session file has numeric or null values for these fields,
                # the Markdown render path (.replace, .upper, etc.) would
                # crash. Coerce here at the boundary.
                "name": str(data.get("name") or "Untitled"),
                "created_at": str(data.get("created_at") or ""),
                "last_modified": str(data.get("last_modified") or ""),
                "message_count": len(messages),
            }
        )

    sessions.sort(key=lambda x: x.get("last_modified") or "", reverse=True)
    return sessions


def save_session(
    name: str = "Untitled",
    messages: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a conversation session to ``SESSIONS_DIR``.

    SECURITY: ``messages`` and ``metadata`` are passed through
    ``redact_secrets`` before being persisted to disk. This is the same
    exposure shape that motivated PR #1 (tool result persisted with
    secrets) — session content may originate from upstream tool output
    or user-pasted material that included secret-shape strings, and we
    do not want those landing on disk in plaintext where they will be
    read back into future conversations.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # Redact name too (per PR #3 review, Codex P2). User-typed names or
    # AI-generated titles can contain secret-shape strings; without this
    # they would land on disk in plaintext while messages/metadata are
    # protected.
    safe_name = redact_secrets(name)
    safe_messages = redact_secrets(messages or [])
    safe_metadata = redact_secrets(metadata or {})

    session_data = {
        "session_id": session_id,
        "name": safe_name,
        "created_at": now,
        "last_modified": now,
        "messages": safe_messages,
        "metadata": safe_metadata,
    }

    session_file = get_session_file(session_id)
    # Ensure the sessions directory exists before writing (lazy mkdir so
    # imports stay side-effect-free for test isolation).
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic write via per-call unique temp file + os.replace(). Using
    # tempfile.mkstemp instead of a fixed ``<sid>.json.tmp`` name avoids
    # the concurrent-update race Codex flagged on PR #4 (round-5 review):
    # with a shared temp path, two writers' fds bind to the same inode
    # and the second writer's bytes can land in the final session file
    # after the first writer's os.replace(). mkstemp uses O_EXCL +
    # randomized name so each writer gets its own inode. Mode is 0o600
    # by default on POSIX, so session content stays owner-only (per
    # PR #4 review, CodeRabbit Major).
    temp_file = _atomic_temp_for(session_file)
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        os.replace(temp_file, session_file)
    except Exception:
        # Clean up the temp file on any error so we don't leave .tmp
        # litter in SESSIONS_DIR.
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "success": True,
        "session_id": session_id,
        "name": safe_name,
        "message_count": len(session_data["messages"]),
    }


def load_session(session_id: str) -> dict[str, Any]:
    """Load a previously saved session by id.

    Wraps the file read in try/except (per PR #3 follow-up review, Gemini
    medium): avoids a TOCTOU race between exists()/open() and surfaces
    corrupted JSON as a clean ValueError instead of bubbling JSONDecodeError
    up to the MCP layer.
    """
    session_file = get_session_file(session_id)
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Session not found: {session_id}") from e
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Session file invalid or unreadable: {session_id}") from e

    # Defensive: a syntactically-valid JSON file shaped like ``[]`` or
    # ``"string"`` would crash the .get() calls below. Raise a clean
    # ValueError so the MCP layer sees a consistent error contract
    # (per PR #3 follow-up review, Gemini medium L275).
    if not isinstance(data, dict):
        raise ValueError(f"Session file shape is not a JSON object: {session_id}")

    # Use `or` fallback (not get-default) so JSON null round-trips to a
    # usable value rather than None — same reason as the list_sessions
    # name fix (per PR #3 follow-up review, Gemini medium L284).
    return {
        "session_id": str(data.get("session_id") or session_id),
        "name": str(data.get("name") or "Untitled"),
        "created_at": str(data.get("created_at") or ""),
        "last_modified": str(data.get("last_modified") or ""),
        # Normalize non-list "messages" to [] so the load_session render
        # path can't be tripped by truthy non-list values (e.g. 1 or
        # "string") in malformed/manually-edited session files
        # (per PR #4 review, Codex P2). The trailing ``or []`` /
        # ``or {}`` was redundant — the ternary already returns the
        # empty container on type-mismatch (per PR #4 follow-up review,
        # CodeRabbit + Gemini nitpick L319/L324).
        "messages": (
            data.get("messages") if isinstance(data.get("messages"), list) else []
        ),
        "metadata": (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        ),
    }


def update_session(session_id: str, name: str | None = None) -> dict[str, Any]:
    """Update mutable session metadata (name) and bump ``last_modified``.

    Wraps the read+write critical section in a per-session advisory
    lockfile (``_session_lock``) so concurrent ``delete_session``
    calls can't slip in between our read and our atomic-write,
    resurrecting a just-deleted session. (Per PR #4 round-6 review,
    Codex P2 L429.)

    Wraps the file read in try/except (per PR #3 follow-up review,
    Gemini medium): avoids a TOCTOU race and handles corrupted JSON
    cleanly.
    """
    session_file = get_session_file(session_id)

    with _session_lock(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Session not found: {session_id}") from e
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Session file invalid or unreadable: {session_id}") from e

        # Defensive: same shape guard as load_session (per PR #3
        # follow-up review, Gemini medium L301).
        if not isinstance(data, dict):
            raise ValueError(f"Session file shape is not a JSON object: {session_id}")

        if name is not None:
            # `is not None` (not truthy check) so callers can pass
            # name="" to explicitly clear the name (per PR #4
            # follow-up review, CodeRabbit nitpick L360).
            data["name"] = redact_secrets(name)
        data["last_modified"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        # Best-effort safeguard against non-cooperating deleters
        # (manual ``rm``, foreign tools not using this API) that
        # bypass the advisory lock. Cooperating deleters are
        # serialized by the surrounding ``_session_lock`` and
        # cannot reach this point with a deleted session.
        if not session_file.exists():
            raise ValueError(
                f"Session was deleted concurrently during update: {session_id}"
            )

        # Atomic write: per-call mkstemp + os.replace so a crash
        # mid-write doesn't truncate the live session file, and so
        # concurrent updates don't corrupt each other via a shared
        # temp inode (per PR #4 follow-up review, Codex P2 L360 /
        # L362 / L392 + Gemini med L270/L362).
        temp_file = _atomic_temp_for(session_file)
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, session_file)
        except Exception:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
            raise

    return {
        "success": True,
        "session_id": session_id,
        "name": data["name"],
        "last_modified": data["last_modified"],
    }


def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session file by id.

    Wraps the unlink in the same per-session advisory lockfile
    (``_session_lock``) used by ``update_session`` so a concurrent
    ``update_session`` can't resurrect the file via its atomic
    ``os.replace`` (per PR #4 round-6 review, Codex P2 L429).

    Catches FileNotFoundError from unlink() directly to avoid the
    TOCTOU race window between exists() and unlink() (per PR #4
    follow-up review, CodeRabbit nitpick L377).
    """
    session_file = get_session_file(session_id)
    with _session_lock(session_file):
        try:
            session_file.unlink()
        except FileNotFoundError:
            raise ValueError(f"Session not found: {session_id}") from None
    return {"success": True, "session_id": session_id}


if "--check" in sys.argv:
    run_check()

# Perplexity client is constructed lazily so the module imports
# cleanly even when the keychain CLI is unavailable (e.g. on Windows
# or in a container without the macOS ``security`` binary). The round-9
# fcntl fix made the lock helpers Windows-tolerant, but the eager call
# to ``get_api_key_from_keychain`` here still shelled out at import
# time and raised on non-macOS, defeating the "module loads cleanly on
# non-POSIX" goal. Per PR #4 round-10 review, Codex P2 L38: defer the
# lookup so tool discovery and non-deep-research code paths (including
# the session helpers) can load without a keychain dependency. The
# error surfaces only when ``deep_research`` is actually invoked.
_perplexity_client_cache: OpenAI | None = None


def _get_perplexity_client() -> OpenAI:
    """Lazy accessor for the Perplexity client.

    Builds and caches a single ``OpenAI`` client on first call. Raises
    whatever ``get_api_key_from_keychain`` raises (typically
    ``ValueError`` for a missing key, or ``FileNotFoundError`` if the
    macOS ``security`` CLI itself is absent). Per PR #4 round-10
    review, Codex P2 L38.
    """
    global _perplexity_client_cache
    if _perplexity_client_cache is None:
        _perplexity_client_cache = OpenAI(
            api_key=get_api_key_from_keychain("api_tokens", "perplexity"),
            base_url="https://api.perplexity.ai",
        )
    return _perplexity_client_cache


# Gemini Deep Research configuration. The /interactions endpoint is a separate
# surface from the standard Generative Language API and is not yet covered by
# the google-genai SDK at time of writing — call it directly via httpx.
#
# Authentication uses Google Cloud Application Default Credentials (ADC)
# rather than a static API key. Tokens are short-lived (~1 hour) and refreshed
# transparently by the google-auth library.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODELS = {
    "fast": "deep-research-preview-04-2026",
    "max": "deep-research-max-preview-04-2026",
}
# Strict allowlist: interaction IDs from tool parameters are concatenated into
# the request URL. Reject anything that could perform path traversal or escape
# the API host, since the ADC bearer token is attached to every request.
_INTERACTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# ADC is loaded LAZILY on first Gemini tool call rather than at module import.
# This means the MCP server can start (and the Perplexity-backed deep_research
# tool can be used) even on a machine without gcloud ADC configured — only the
# gemini_* tools will fail when invoked. Module-level eager-load was crashing
# the entire server at startup if ADC was missing or slow to fetch.
_gemini_credentials: Any = None
_gemini_billing_project: str | None = None
_gemini_token_lock = asyncio.Lock()

# Terminal states for a Gemini Deep Research interaction. Anything not in this
# set is treated as still-in-progress so the client knows to keep polling.
# Includes "cancelled" (user-cancelled or quota-cancelled) per Gemini API docs
# alongside the obvious "completed"/"failed", plus "incomplete" (run ended
# without a final answer — e.g. tool/agent failure mid-stream) and
# "budget_exceeded" (token or compute budget exhausted). Status strings are
# matched case-insensitively at the comparison site.
#
# Note: "requires_action" is intentionally NOT in this set — it's a distinct
# non-terminal state where the agent is awaiting user input (typically when
# collaborative_planning=true). It is handled with its own branch in the
# result tool so the caller knows it's actionable, not stuck.
_GEMINI_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "incomplete", "budget_exceeded"}
)

# Module-level lazy singleton httpx.AsyncClient. Created on first Gemini call
# and reused across all subsequent calls so we get connection pooling / keep-
# alive against the Gemini API host. Initialized under a lock so a burst of
# concurrent tool calls doesn't race on first use.
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Return the module-level shared httpx client, creating it on first use.

    follow_redirects is disabled so the ADC bearer token cannot be forwarded
    to another host via a redirect response.
    """
    global _http_client
    if _http_client is None:
        async with _http_client_lock:
            if _http_client is None:
                _http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=False)
    return _http_client


def _http_error_payload(exc: httpx.HTTPStatusError) -> dict[str, Any]:
    """Build a structured failure dict from an httpx HTTPStatusError.

    Keeps a short body snippet (≤500 chars) so the caller has enough context to
    diagnose without bloating the MCP response. Runs the snippet through
    `redact_secrets` because Gemini error bodies have, on occasion, echoed
    request headers or query content.
    """
    status_code = exc.response.status_code
    try:
        body = exc.response.text or ""
    except Exception:  # noqa: BLE001 - never let body extraction shadow the real error
        body = ""
    snippet = redact_secrets(body[:500])
    return {"status": "failed", "error": f"{status_code}: {snippet}"}


async def _get_bearer_token() -> str:
    """Return a fresh ADC bearer token, loading credentials on first call and
    refreshing on a worker thread if expired. The lock serializes concurrent
    init/refresh attempts from parallel tool calls.
    """
    global _gemini_credentials, _gemini_billing_project
    async with _gemini_token_lock:
        if _gemini_credentials is None:
            # Defer the blocking ADC lookup to a worker thread — google.auth.default
            # can do file I/O and (rarely) network calls under the hood.
            _gemini_credentials, _gemini_billing_project = await asyncio.to_thread(
                _load_adc
            )
        if not _gemini_credentials.valid:
            await asyncio.to_thread(
                _gemini_credentials.refresh,
                google.auth.transport.requests.Request(),
            )
        token = _gemini_credentials.token
    if not token:
        raise RuntimeError("ADC refresh returned empty token")
    return token


async def _gemini_headers() -> dict[str, str]:
    # _get_bearer_token populates _gemini_billing_project as a side effect of
    # first-time ADC load, so call it first to ensure the project is available.
    token = await _get_bearer_token()
    return {
        "Authorization": f"Bearer {token}",
        # Required when using OAuth (not API key) so the request is billed and
        # quota-attributed to the user's project rather than the credential's
        # home project.
        "x-goog-user-project": _gemini_billing_project,
        "Content-Type": "application/json",
    }


def _validate_interaction_id(interaction_id: str) -> str:
    """Reject interaction IDs that could redirect the authenticated request.

    The interaction_id is concatenated into the URL of an authenticated HTTP
    call; an attacker-controlled value containing ``/``, ``..``, or a scheme
    could cause the Gemini API key to be sent to an unintended host.
    """
    if not isinstance(interaction_id, str) or not _INTERACTION_ID_RE.fullmatch(
        interaction_id
    ):
        raise ValueError(
            "interaction_id must match ^[A-Za-z0-9_-]{1,128}$ — refusing to "
            "send authenticated request with untrusted path segment."
        )
    return interaction_id


async def _post_gemini_interaction(payload: dict[str, Any]) -> dict[str, Any]:
    """POST a Deep Research interaction. URL is fully static; no tool input.

    On HTTP, network, or JSON-decode error, returns a structured
    ``{"status": "failed", "error": ...}`` dict instead of raising so the
    MCP client gets a graceful error envelope rather than an opaque
    exception. The shared httpx client gives us connection pooling across
    calls.
    """
    headers = await _gemini_headers()
    client = await _get_http_client()
    try:
        # Auth is server-sourced (ADC bearer token via _gemini_headers), never a
        # caller-supplied credential. The request host is the hardcoded HTTPS
        # constant GEMINI_API_BASE and no tool parameter is interpolated into the
        # URL, so the credential cannot be redirected to an attacker host. The
        # mcp-auth-passthrough-taint rule cannot see that the host is static.
        response = await client.post(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            f"{GEMINI_API_BASE}/interactions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)
    except httpx.RequestError as exc:
        # Connect errors and read timeouts must keep the structured-envelope
        # contract instead of crashing the tool call — same treatment the
        # Agent API helpers got in PR #16 review (Qodo bug #2 / CodeRabbit
        # major). ADC/credential errors deliberately propagate: the
        # _gemini_headers() lookup sits outside this try block. Exception
        # text is redacted like _http_error_payload's body snippet — the
        # same "never emit secret-shapes" contract on every error path.
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        # response.json() on a non-JSON 200 body (json.JSONDecodeError is a
        # ValueError subclass). Only the json parse can raise ValueError
        # inside this try block.
        return {
            "status": "failed",
            "error": f"invalid JSON from Deep Research API: {redact_secrets(str(exc))}",
        }


async def _get_gemini_interaction(interaction_id: str) -> dict[str, Any]:
    """GET a Deep Research interaction by ID.

    The interaction_id MUST have already passed _validate_interaction_id; this
    helper re-validates as defense in depth so the URL cannot escape the API
    host even if a future caller forgets. Same structured-error contract as
    `_post_gemini_interaction`.
    """
    safe_id = _validate_interaction_id(interaction_id)
    headers = await _gemini_headers()
    client = await _get_http_client()
    try:
        # Auth is server-sourced (ADC bearer token via _gemini_headers). The only
        # caller-influenced URL segment, safe_id, has passed _validate_interaction_id
        # (^[A-Za-z0-9_-]{1,128}$) — re-validated here as defense in depth — so it
        # cannot contain '/', '.', ':', a scheme, or a host. The credential cannot
        # be redirected off GEMINI_API_BASE. The taint rule does not recognize the
        # regex allowlist as a sanitizer.
        response = await client.get(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            f"{GEMINI_API_BASE}/interactions/{safe_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)
    except httpx.RequestError as exc:
        # Same structured-envelope contract as _post_gemini_interaction.
        # The validation ValueError from _validate_interaction_id and any
        # ADC/credential error are raised before this try block and cannot
        # be swallowed below.
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        # response.json() decode failure only — see above.
        return {
            "status": "failed",
            "error": f"invalid JSON from Deep Research API: {redact_secrets(str(exc))}",
        }


# --- Perplexity Agent API (Search-as-Code) -------------------------------
#
# agent_research drives Perplexity's Agent API with the `sandbox` tool
# enabled: the upstream model writes and executes code in a Perplexity-
# hosted container, calling search programmatically from inside that
# code. This wins on bulk/enumerable research ("for each of N items,
# find X") where one-shot synthesis (quick_research / deep_research)
# under-covers the item list; it loses on single questions, where the
# fixed per-container fee and orchestration latency are pure overhead.

_AGENT_RESEARCH_URL = "https://api.perplexity.ai/v1/responses"

# Server-side model allowlist. The Agent API can route to third-party
# frontier models; the `model` argument is an enum over this tuple so a
# prompt-injected or malformed request cannot select an arbitrary
# (expensive) upstream model. Default is the strongest allowlisted
# orchestrator: the sandbox agent writes code against scraped web
# content, and weaker models are more susceptible to prompt injection.
AGENT_RESEARCH_MODELS: tuple[str, ...] = (
    "anthropic/claude-sonnet-4-6",
    "perplexity/sonar",
)
AGENT_RESEARCH_DEFAULT_MODEL = AGENT_RESEARCH_MODELS[0]

_AGENT_MAX_OUTPUT_TOKENS_MIN = 256
_AGENT_MAX_OUTPUT_TOKENS_MAX = 8192
_AGENT_MAX_OUTPUT_TOKENS_DEFAULT = 4096

# Sandbox runs routinely take minutes (container spin-up + iterative
# code execution + per-item searches) — far beyond the shared client's
# 30s default, so the POST passes an explicit per-request timeout.
_AGENT_API_TIMEOUT_SECONDS = 600.0

# stderr from failed sandbox executions is surfaced for diagnosis but
# truncated: it is model-generated-code output over scraped web content,
# i.e. doubly untrusted, and must not flood the MCP response.
_SANDBOX_STDERR_SNIPPET_LEN = 300

_AGENT_RESEARCH_INSTRUCTIONS = (
    "You are a research agent with a code sandbox. When the task involves "
    "many items, calculations, or structured output, write code in the "
    "sandbox to enumerate every item and search programmatically rather "
    "than sampling a few and generalizing. Cite sources for factual claims "
    "and state clearly when an item could not be resolved."
)


async def _post_agent_research(payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the Perplexity Agent API responses endpoint.

    Same structured-error contract as `_post_gemini_interaction`: on HTTP
    error returns ``{"status": "failed", "error": ...}`` instead of raising
    so the MCP client gets a graceful envelope. The Keychain lookup runs on
    a worker thread because `security` is a blocking subprocess call.
    """
    api_key = await asyncio.to_thread(
        get_api_key_from_keychain, "api_tokens", "perplexity"
    )
    client = await _get_http_client()
    try:
        # Auth is server-sourced (Keychain lookup above), never a caller-
        # supplied credential. The request host is the hardcoded HTTPS
        # constant _AGENT_RESEARCH_URL and no tool parameter is interpolated
        # into the URL, so the credential cannot be redirected to an
        # attacker host. The mcp-auth-passthrough-taint rule cannot see
        # that the host is static.
        response = await client.post(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            _AGENT_RESEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=_AGENT_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)
    except httpx.RequestError as exc:
        # Connect errors and read timeouts are the most likely failure mode
        # on minutes-long sandbox runs — keep the structured-envelope
        # contract instead of crashing the tool call (per PR #16 review,
        # Qodo bug #2 / CodeRabbit major). The keychain ValueError
        # deliberately propagates: credential-setup errors raise across all
        # tool families (_get_perplexity_client and _gemini_headers behave
        # the same) and the lookup sits outside this try block. Exception
        # text is redacted like _http_error_payload's body snippet — the
        # same "never emit secret-shapes" contract on every error path.
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        # response.json() on a non-JSON 200 body (json.JSONDecodeError is a
        # ValueError subclass). Only the json parse can raise ValueError
        # inside this try block.
        return {
            "status": "failed",
            "error": f"invalid JSON from Agent API: {redact_secrets(str(exc))}",
        }


# Same allowlist shape as _INTERACTION_ID_RE and for the same reason: the
# response id is interpolated into the URL of an authenticated GET, so a
# value containing '/', '..', or a scheme could redirect the Perplexity
# key to an unintended host. Live ids look like
# "resp_79b0f91b-e4c6-44e9-86cf-8ab09e9c88d0" — well within the pattern.
_AGENT_RESPONSE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_agent_response_id(response_id: str | None) -> str:
    """Reject response IDs that could redirect the authenticated request."""
    if response_id is None:
        # Distinct message for the common caller mistake — the regex
        # contract below would be confusing for a simply-missing argument.
        raise ValueError("response_id is required.")
    if not isinstance(response_id, str) or not _AGENT_RESPONSE_ID_RE.fullmatch(
        response_id
    ):
        raise ValueError(
            "response_id must match ^[A-Za-z0-9_-]{1,128}$ — refusing to "
            "send authenticated request with untrusted path segment."
        )
    return response_id


async def _get_agent_response(response_id: str) -> dict[str, Any]:
    """GET an Agent API response by ID (poll for background runs).

    The response_id MUST have already passed _validate_agent_response_id;
    this helper re-validates as defense in depth so the URL cannot escape
    the API host even if a future caller forgets. Same structured-error
    contract as `_post_agent_research`.
    """
    safe_id = _validate_agent_response_id(response_id)
    api_key = await asyncio.to_thread(
        get_api_key_from_keychain, "api_tokens", "perplexity"
    )
    client = await _get_http_client()
    try:
        # Auth is server-sourced (Keychain). The only caller-influenced URL
        # segment, safe_id, has passed _validate_agent_response_id
        # (^[A-Za-z0-9_-]{1,128}$) — re-validated here as defense in depth —
        # so it cannot contain '/', '.', ':', a scheme, or a host. The taint
        # rule does not recognize the regex allowlist as a sanitizer.
        # No explicit timeout (unlike the POST helper's 600s): this GET is
        # a status poll that returns immediately whatever the run's state,
        # so the shared client's 30s default is correct here.
        response = await client.get(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            f"{_AGENT_RESEARCH_URL}/{safe_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)
    except httpx.RequestError as exc:
        # Same structured-envelope contract as _post_agent_research (per
        # PR #16 review). Keychain/validation ValueErrors are raised before
        # this try block and cannot be swallowed below.
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        # response.json() decode failure only — see above.
        return {
            "status": "failed",
            "error": f"invalid JSON from Agent API: {redact_secrets(str(exc))}",
        }


def _render_agent_research(data: dict[str, Any]) -> list[TextContent]:
    """Format a completed Agent API response as the agent_research result.

    Shared by the synchronous agent_research path and the
    agent_research_result poll tool so both render identically.

    Response shape (verified against a live request on 2026-06-09):
    output[] mixes `sandbox_results` items (code, per-command results with
    exit_code/stdout/stderr) and `message` items (content[] of output_text).
    usage.cost carries an itemized USD breakdown.
    """
    output_items = data.get("output") or []
    answer_parts: list[str] = []
    sandbox_runs = 0
    failed_execs: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for chunk in item.get("content") or []:
                # isinstance(str) guard, not just truthiness: the response
                # is untrusted input, and a non-string `text` would crash
                # the "\n\n".join below (per PR #16 review, Qodo bug #3).
                if (
                    isinstance(chunk, dict)
                    and chunk.get("type") == "output_text"
                    and isinstance(chunk.get("text"), str)
                    and chunk["text"]
                ):
                    answer_parts.append(chunk["text"])
        elif item.get("type") == "sandbox_results":
            sandbox_runs += 1
            for exec_result in item.get("results") or []:
                if not isinstance(exec_result, dict):
                    continue
                exit_code = exec_result.get("exit_code")
                if exit_code not in (0, None):
                    stderr = str(exec_result.get("stderr") or "")
                    snippet = stderr[:_SANDBOX_STDERR_SNIPPET_LEN]
                    if len(stderr) > _SANDBOX_STDERR_SNIPPET_LEN:
                        snippet += "…"
                    failed_execs.append(f"exit_code={exit_code} — {snippet}")

    if not answer_parts:
        return [
            TextContent(
                type="text",
                text=(
                    "Error: Agent API returned no assistant message for agent_research"
                ),
            )
        ]

    # Redact secret-shape patterns: the answer synthesizes scraped web
    # content, and failed-execution stderr is sandbox output over that
    # same untrusted content.
    answer = redact_secrets("\n\n".join(answer_parts))

    usage = data.get("usage") or {}
    cost = usage.get("cost") or {}
    total_cost = cost.get("total_cost")
    currency = cost.get("currency", "USD")
    # `model` and `status` are API-emitted strings rendered verbatim —
    # redact like every other response field (per PR #16 review).
    meta_bits = [
        f"model: {redact_secrets(str(data.get('model', 'unknown')))}",
        f"sandbox executions: {sandbox_runs}",
    ]
    if total_cost is not None:
        meta_bits.append(f"cost: {total_cost} {currency}")

    lines = ["## Agent Research (Search-as-Code)", ""]
    status = data.get("status", "unknown")
    if status != "completed":
        # e.g. "incomplete" when max_output_tokens truncated the run —
        # surface it so the caller knows coverage may be partial.
        lines.extend([f"> ⚠️ upstream status: {redact_secrets(str(status))}", ""])
    lines.extend([answer, "", "---", f"*{' · '.join(meta_bits)}*"])
    if failed_execs:
        lines.extend(["", "### Sandbox execution warnings", ""])
        lines.extend(f"- {redact_secrets(detail)}" for detail in failed_execs)

    return [TextContent(type="text", text="\n".join(lines))]


# ─── Local delegate (Ollama) ──────────────────────────────────────────
#
# Third tool family: delegate tasks to a LOCAL Ollama model. Inverts the
# data-flow of every other family — exists precisely so input text can
# stay on-device (plus quota offload, second opinions, background/batch
# work). The server only CALLS an already-running Ollama; it never reads
# files, pulls models, or manages the Ollama service (least privilege).
#
# Native /api/chat (not the OpenAI-compat endpoint) because only the
# native API accepts `think` — qwen3.6 is a thinking model and
# think:false is required for fast structured work.

OLLAMA_DELEGATE_MODELS: tuple[str, ...] = (
    "qwen3.6:35b-a3b-coding-nvfp4",
    "qwen3.6:35b-a3b-coding-nvfp4-32k",
    "qwen3.6:35b-a3b-coding-nvfp4-256k",
)
OLLAMA_DELEGATE_DEFAULT_MODEL = OLLAMA_DELEGATE_MODELS[0]

_OLLAMA_URL_ENV_VAR = "AI_TOOLS_OLLAMA_URL"
_OLLAMA_URL_KEYCHAIN_SERVICE = "OLLAMA_URL"
_CF_ACCESS_ID_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_ID"
_CF_ACCESS_SECRET_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_SECRET"

# `0` (unload immediately) or 1-9999 seconds/minutes/hours. Strict so a
# malformed value cannot smuggle arbitrary JSON into the Ollama request.
_DELEGATE_KEEP_ALIVE_RE = re.compile(r"^(0|[1-9][0-9]{0,3}(s|m|h))$")
_DELEGATE_KEEP_ALIVE_DEFAULT = "5m"

# Shared-client default is 30s; delegate calls pass explicit per-request
# timeouts (same mechanism as _AGENT_API_TIMEOUT_SECONDS).
_DELEGATE_TIMEOUT_DEFAULT_S = 300
_DELEGATE_TIMEOUT_MAX_S = 600
_DELEGATE_BG_CEILING_S = 1800.0

# jvmacmini runs num_parallel=1 — queuing beyond a few jobs would lie to
# the caller; fail fast instead.
_DELEGATE_JOB_CAP = 4
_DELEGATE_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


_OLLAMA_URLS_ENV_VAR = "AI_TOOLS_OLLAMA_URLS"
_OLLAMA_DEFAULT_MODEL_ENV_VAR = "AI_TOOLS_OLLAMA_DEFAULT_MODEL"

# v1.1 (spec amendment): local-first endpoint chain. The remote default is
# the user's own Cloudflare-Access-gated JVMBPro tunnel — never a
# third-party service.
_OLLAMA_DEFAULT_CHAIN: tuple[str, ...] = (
    "http://localhost:11434",
    "https://ollama-mbp.djvassallo.com",
)
_OLLAMA_PROBE_TIMEOUT_S = 2.0
_OLLAMA_PROBE_CACHE_TTL_S = 60.0
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_localhost_endpoint(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in _LOCALHOST_HOSTS


def _validate_ollama_endpoint(url: str) -> str:
    """Validate one chain entry; fail closed on anything not plain http(s).

    Loopback may be http; every other host must be https (v1.1 rule — a
    remote endpoint is only ever the Access-gated tunnel).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid Ollama endpoint {redact_secrets(url)!r}: must be "
            "http(s)://host[:port]"
        )
    if parsed.scheme == "http" and not _is_localhost_endpoint(url):
        raise ValueError(
            f"Refusing plain-http non-localhost Ollama endpoint "
            f"{redact_secrets(url)!r} — remote endpoints must be https"
        )
    return url.rstrip("/")


def _resolve_ollama_chain() -> list[str]:
    """Ordered Ollama endpoint chain (v1.1).

    Env `AI_TOOLS_OLLAMA_URLS` (comma-separated) wins; singular
    `AI_TOOLS_OLLAMA_URL` is honored as a one-item chain for v1 compat;
    otherwise the default local-first chain. A Keychain `OLLAMA_URL`
    endpoint is appended when present. Every entry is validated; dupes
    dropped preserving order. Blocking (Keychain) — async callers wrap
    in asyncio.to_thread.
    """
    raw = os.environ.get(_OLLAMA_URLS_ENV_VAR, "").strip()
    if raw:
        entries = [e.strip() for e in raw.split(",") if e.strip()]
    else:
        single = os.environ.get(_OLLAMA_URL_ENV_VAR, "").strip()
        entries = [single] if single else list(_OLLAMA_DEFAULT_CHAIN)
    try:
        keychain_url = get_api_key_from_keychain(
            _OLLAMA_URL_KEYCHAIN_SERVICE, getpass.getuser()
        ).strip()
        if keychain_url:
            entries.append(keychain_url)
    except ValueError:
        pass  # optional config — absence is the common case
    chain: list[str] = []
    for entry in entries:
        validated = _validate_ollama_endpoint(entry)
        if validated not in chain:
            chain.append(validated)
    return chain


def _delegate_default_model() -> str:
    """Default model tag; env override honored only if allowlisted.

    Falls back silently to the base tag (run_check surfaces a warn) so a
    typo'd Desktop setting cannot break tool listing.
    """
    env_model = os.environ.get(_OLLAMA_DEFAULT_MODEL_ENV_VAR, "").strip()
    if env_model in OLLAMA_DELEGATE_MODELS:
        return env_model
    return OLLAMA_DELEGATE_DEFAULT_MODEL


def _ollama_auth_headers(endpoint: str) -> dict[str, str] | None:
    """Auth headers for an Ollama endpoint; None means SKIP, never call bare.

    localhost → {} (no auth). Non-localhost https → Cloudflare Access
    service-token headers read from the Keychain per call (never cached,
    never logged). Either credential absent → None (fail closed): the
    caller treats the endpoint as unavailable rather than calling an
    Access-gated host unauthenticated.
    """
    if _is_localhost_endpoint(endpoint):
        return {}
    user = getpass.getuser()
    try:
        client_id = get_api_key_from_keychain(_CF_ACCESS_ID_KEYCHAIN_SERVICE, user)
        client_secret = get_api_key_from_keychain(
            _CF_ACCESS_SECRET_KEYCHAIN_SERVICE, user
        )
    except ValueError:
        return None
    if not client_id or not client_secret:
        # A Keychain item can exist with an empty password — `security`
        # returns "" with returncode 0 (no ValueError). Treat that the same
        # as "absent" so we fail closed instead of calling the Access-gated
        # host with a malformed header.
        return None
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": client_secret,
    }


_ollama_endpoint_cache: dict[str, tuple[str, float]] = {}


async def _select_ollama_endpoint(model: str) -> str:
    """First endpoint in the chain whose /api/tags lists `model`.

    Results are cached per model for _OLLAMA_PROBE_CACHE_TTL_S. Raises
    ValueError naming every endpoint tried and what each reported —
    actionable and fail-closed.
    """
    cached = _ollama_endpoint_cache.get(model)
    if cached is not None and time.monotonic() < cached[1]:
        return cached[0]
    chain = await asyncio.to_thread(_resolve_ollama_chain)
    client = await _get_http_client()
    attempts: list[str] = []
    for endpoint in chain:
        headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
        if headers is None:
            attempts.append(
                f"{endpoint}: skipped (Cloudflare Access credentials not in Keychain)"
            )
            continue
        try:
            response = await client.get(
                f"{endpoint}/api/tags",
                headers=headers,
                timeout=_OLLAMA_PROBE_TIMEOUT_S,
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            tags = [
                str(m.get("name") or m.get("model") or "")
                for m in models
                if isinstance(m, dict)
            ]
        except httpx.HTTPStatusError as exc:
            attempts.append(f"{endpoint}: HTTP {exc.response.status_code}")
            continue
        except httpx.RequestError:
            attempts.append(f"{endpoint}: unreachable")
            continue
        except ValueError:
            attempts.append(f"{endpoint}: invalid JSON from /api/tags")
            continue
        if model in tags:
            _ollama_endpoint_cache[model] = (
                endpoint,
                time.monotonic() + _OLLAMA_PROBE_CACHE_TTL_S,
            )
            return endpoint
        present = ", ".join(sorted(t for t in tags if t)) or "no models"
        attempts.append(f"{endpoint}: model not present (has {present})")
    detail = "; ".join(attempts) or "empty endpoint chain"
    raise ValueError(f"No Ollama endpoint serves {model!r}: {redact_secrets(detail)}")


async def _post_ollama_chat(
    payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST /api/chat to the first chain endpoint serving payload['model'].

    Same structured-error contract as _post_agent_research: selection,
    network, HTTP, and parse failures return {"status": "failed", ...}
    instead of raising. Remote https endpoints get Cloudflare Access
    service-token headers (absent creds → skipped at selection; the None
    check here is defense in depth). No retries: an endpoint is either
    serving or not.
    """
    model = str(payload.get("model", ""))
    try:
        endpoint = await _select_ollama_endpoint(model)
    except ValueError as exc:
        return {"status": "failed", "error": redact_secrets(str(exc))}
    headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
    if headers is None:
        return {
            "status": "failed",
            "error": f"no credentials for {redact_secrets(endpoint)}",
        }
    client = await _get_http_client()
    try:
        response = await client.post(
            f"{endpoint}/api/chat", json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        failure = _http_error_payload(exc)
        for secret in headers.values():
            # Value-aware scrub: an Access-gated host's error body can echo
            # request headers; redact_secrets has no CF-token pattern, but we
            # hold the exact values, so scrub them precisely.
            if secret:
                failure["error"] = failure["error"].replace(
                    secret, "[REDACTED_CF_ACCESS]"
                )
        if exc.response.status_code == 404:
            failure["error"] += (
                f" — model may not be pulled on this host; try: ollama pull {model}"
            )
        return failure
    except httpx.ConnectError:
        # Answered the probe moments ago but refused the POST — drop the
        # cached resolution so the next call re-probes the chain.
        _ollama_endpoint_cache.pop(model, None)
        return {
            "status": "failed",
            "error": (
                f"Ollama not running at {redact_secrets(endpoint)} — is the "
                "LaunchAgent up? "
                "(launchctl kickstart -k gui/$UID/com.jasonvassallo.ollama)"
            ),
        }
    except httpx.RequestError as exc:
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }
    except ValueError as exc:
        return {
            "status": "failed",
            "error": f"invalid JSON from Ollama: {redact_secrets(str(exc))}",
        }


def _render_delegate_answer(data: dict[str, Any]) -> list[TextContent]:
    """Render an Ollama /api/chat response (or failure envelope) as MCP text.

    message.thinking is deliberately discarded — the caller needs the
    answer, not the model's scratchpad. Output passes through
    redact_secrets for the same never-emit-secret-shapes contract as
    every other family.
    """
    if data.get("status") == "failed":
        return [
            TextContent(
                type="text", text=f"Error: {data.get('error', 'unknown failure')}"
            )
        ]
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return [TextContent(type="text", text="Error: Ollama returned no content")]
    model = redact_secrets(str(data.get("model", "")))
    return [
        TextContent(
            type="text",
            text=f"## Local Delegate ({model})\n\n{redact_secrets(content)}",
        )
    ]


# In-memory only, deliberately: delegated input may be exactly the
# sensitive text kept off cloud APIs — it does not belong on disk. Jobs
# die with the MCP server process; the polling Claude session dies with
# it too, so nothing durable is lost.
_delegate_jobs: dict[str, dict[str, Any]] = {}


def _start_delegate_job(payload: dict[str, Any]) -> str:
    """Launch a background delegate call; return its job id.

    Raises ValueError when _DELEGATE_JOB_CAP jobs are already running —
    the 32 GB host runs num_parallel=1, so queuing more would lie to the
    caller; failing fast is honest.
    """
    running = sum(1 for job in _delegate_jobs.values() if not job["task"].done())
    if running >= _DELEGATE_JOB_CAP:
        raise ValueError(
            f"Delegate job cap ({_DELEGATE_JOB_CAP}) reached — collect finished "
            "jobs via local_delegate_result or wait for one to complete."
        )
    job_id = uuid.uuid4().hex
    coro = asyncio.wait_for(
        _post_ollama_chat(payload, _DELEGATE_BG_CEILING_S),
        timeout=_DELEGATE_BG_CEILING_S,
    )
    _delegate_jobs[job_id] = {
        "task": asyncio.get_running_loop().create_task(coro),
        "started": time.monotonic(),
    }
    return job_id


def _collect_delegate_job(job_id: str | None) -> dict[str, Any]:
    """Poll/collect a background job. Completed jobs are single-collect:
    the registry entry is deleted on retrieval so memory stays clean."""
    if not isinstance(job_id, str) or not _DELEGATE_JOB_ID_RE.fullmatch(job_id):
        raise ValueError("job_id must be the 32-hex id returned by local_delegate.")
    job = _delegate_jobs.get(job_id)
    if job is None:
        raise ValueError(
            f"Unknown job_id {job_id!r} — results are single-collect and jobs "
            "do not survive an MCP server restart."
        )
    task = job["task"]
    if not task.done():
        return {
            "status": "running",
            "elapsed_s": int(time.monotonic() - job["started"]),
        }
    del _delegate_jobs[job_id]
    try:
        return task.result()
    except (TimeoutError, asyncio.CancelledError):
        # asyncio.wait_for raises TimeoutError (== asyncio.TimeoutError on
        # 3.12) past the ceiling; treat cancellation the same way.
        return {
            "status": "failed",
            "error": (
                f"background job exceeded the {int(_DELEGATE_BG_CEILING_S)}s "
                "ceiling and was cancelled"
            ),
        }


# Create MCP server
server = Server("ai-tools-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="quick_research",
            description=(
                "Quick research using Perplexity Sonar (the smaller, faster, "
                "cheaper sibling of Sonar Pro). Returns a concise answer with "
                "citations in a few seconds. Use when: the query is well-scoped "
                "and a single-source answer with citations is enough, you've "
                "already tried built-in WebSearch and need LLM synthesis on top, "
                "or you want a citation-backed answer without paying for Sonar "
                "Pro's deeper multi-source reasoning. For ambiguous queries, "
                "cross-source comparisons, or architectural tradeoff "
                "investigations, use `deep_research` (Sonar Pro) instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response (default: 1024)",
                        "default": 1024,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="deep_research",
            description=(
                "Deep research using Perplexity Sonar Pro with multi-source "
                "synthesis and citations. Use instead of built-in WebSearch when: "
                "the answer spans multiple sources, requires cross-referencing, "
                "involves comparing tradeoffs/architectures/approaches, "
                "the query is ambiguous and benefits from AI-powered search reasoning, "
                "or you need comprehensive coverage with source citations. "
                "Do NOT use for simple factual lookups (use built-in WebSearch for those). "
                "For well-scoped single-source questions where a quick citation-backed "
                "answer suffices, use `quick_research` (Sonar) instead — it is faster "
                "and cheaper."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic requiring deep investigation",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response (default: 2048)",
                        "default": 2048,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="agent_research",
            description=(
                "Search-as-Code research via the Perplexity Agent API: an agent "
                "writes and runs code in a hosted sandbox, searching the web "
                "programmatically from inside that code. Use ONLY when the task "
                "is bulk/enumerable ('for each of these N CVEs/packages/vendors, "
                "find X'), needs computation over search results, or must produce "
                "a structured dataset — code loops cover every item where chat "
                "synthesis samples a few and generalizes. For a single research "
                "question use `deep_research` instead (faster, cheaper); for "
                "quick lookups use `quick_research`. Runs take one to several "
                "minutes: call synchronously (default) to wait inline, or pass "
                "background=true to get a response_id immediately and poll "
                "`agent_research_result`. Costs include a per-container fee "
                "plus per-search charges, so per-request cost is higher and "
                "less predictable than deep_research."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The bulk research task. Enumerate the items and the "
                            "fields to resolve per item explicitly."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "enum": list(AGENT_RESEARCH_MODELS),
                        "default": AGENT_RESEARCH_DEFAULT_MODEL,
                        "description": (
                            "Orchestrator model (server-side allowlist). Default "
                            "is the strongest option; perplexity/sonar is the "
                            "cheap alternative for simple enumerations."
                        ),
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "minimum": _AGENT_MAX_OUTPUT_TOKENS_MIN,
                        "maximum": _AGENT_MAX_OUTPUT_TOKENS_MAX,
                        "default": _AGENT_MAX_OUTPUT_TOKENS_DEFAULT,
                        "description": (
                            "Maximum output tokens (default: "
                            f"{_AGENT_MAX_OUTPUT_TOKENS_DEFAULT})"
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Run in the background: returns a response_id "
                            "immediately; poll agent_research_result for the "
                            "answer. Use for large fan-outs that would "
                            "otherwise block the session for minutes."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="agent_research_result",
            description=(
                "Poll a background agent_research task by response_id. Returns "
                "the formatted answer when the task completes, a poll-again "
                "hint while it is queued or in progress, and a structured "
                "error if the task failed or was cancelled. Poll roughly every "
                "30 seconds — sandbox runs typically take one to several "
                "minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "response_id": {
                        "type": "string",
                        "description": (
                            "The response_id returned by agent_research with "
                            "background=true."
                        ),
                    },
                },
                "required": ["response_id"],
            },
        ),
        Tool(
            name="gemini_deep_research_start",
            description=(
                "Start a Gemini Deep Research task (asynchronous). Returns an "
                "interaction_id you must poll with gemini_deep_research_result. "
                "Tasks run for several minutes and up to 60 minutes. Use when "
                "you need a citation-dense, multi-page report drawing on many "
                "sources. For quick inline research that completes in seconds, "
                "use `deep_research` (Perplexity) instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "max"],
                        "default": "fast",
                        "description": (
                            "fast = deep-research-preview (speed/efficiency); "
                            "max = deep-research-max-preview (maximum comprehensiveness)."
                        ),
                    },
                    "collaborative_planning": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable collaborative planning mode.",
                    },
                    "thinking_summaries": {
                        "type": "string",
                        "enum": ["auto", "none"],
                        "default": "auto",
                        "description": "Whether the agent should emit thinking summaries.",
                    },
                    "previous_interaction_id": {
                        "type": "string",
                        "description": (
                            "Optional ID of a prior interaction to continue from. "
                            "Must match ^[A-Za-z0-9_-]{1,128}$."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="gemini_deep_research_result",
            description=(
                "Retrieve the status or final result of a Gemini Deep Research "
                "task started with gemini_deep_research_start. Returns "
                "{status, output_text, steps_summary} when status='completed', "
                "{status: 'failed'|'cancelled'|'incomplete'|'budget_exceeded', "
                "error} on terminal non-success, {status: 'requires_action', "
                "hint} when the agent is awaiting user input, or "
                "{status: 'in_progress', hint} while running. Poll roughly "
                "every 30 seconds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "interaction_id": {
                        "type": "string",
                        "description": "ID returned by gemini_deep_research_start.",
                    },
                },
                "required": ["interaction_id"],
            },
        ),
        Tool(
            name="list_sessions",
            description=(
                "List all saved conversation sessions, most recent first. "
                "Returns session id, name, created_at, last_modified, and "
                "message count for each."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="save_session",
            description=(
                "Persist the current conversation context to a new session "
                "file. Returns the new session id. Pass the full conversation "
                "history as the 'messages' array. Secret-shape strings in "
                "messages and metadata are redacted before write."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A descriptive name for the session",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Array of message objects from the conversation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant", "system"],
                                },
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional free-form metadata for the session",
                    },
                },
                # All optional: implementation defaults to "Untitled" + []
                # (per PR #4 review, CodeRabbit Major: schema must align
                # with implementation defaults).
                "required": [],
            },
        ),
        Tool(
            name="load_session",
            description=(
                "Load a previously saved session by its id. Returns the "
                "full conversation history and metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to load",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="update_session",
            description=(
                "Update a saved session's name and bump its last_modified timestamp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to update",
                    },
                    "name": {
                        "type": "string",
                        "description": "New name for the session",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="delete_session",
            description=("Delete a saved session permanently. Use with caution."),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to delete",
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "quick_research":
        query = arguments.get("query")
        max_tokens = arguments.get("max_tokens", 1024)

        # Same lazy-client + redaction path as deep_research; only the
        # model and system prompt differ. Sonar is smaller/faster than
        # Sonar Pro — the system prompt asks for brevity to match the
        # use case rather than coaxing the smaller model into mimicking
        # Sonar Pro's depth.
        #
        # asyncio.to_thread wrapper: the openai client's chat.completions
        # .create is a synchronous blocking call. Running it bare inside
        # an async def would block the asyncio event loop for the duration
        # of the request (seconds-to-tens-of-seconds for Sonar). Per
        # PR #11 review, Gemini high: wrap in asyncio.to_thread so other
        # coroutines can progress. Same fix applied to deep_research below.
        response = await asyncio.to_thread(
            _get_perplexity_client().chat.completions.create,
            model="sonar",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise research assistant. Answer the user's "
                        "question directly, with citations. Prefer a single "
                        "well-sourced answer over a survey of perspectives. "
                        "Skip caveats unless they materially change the answer."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
        )

        # Defensive: per PR #11 review, Gemini medium — response.choices
        # *should* always be non-empty per the API contract but a malformed
        # or truncated response would raise IndexError on choices[0].
        choices = response.choices or []
        if not choices:
            return [
                TextContent(
                    type="text",
                    text="Error: Perplexity returned no choices for quick_research",
                )
            ]
        message = choices[0].message
        content = redact_secrets(message.content or "")
        result = f"## Quick Research\n\n{content}"

        return [TextContent(type="text", text=result)]

    if name == "deep_research":
        query = arguments.get("query")
        max_tokens = arguments.get("max_tokens", 2048)

        # Lazy client construction: per PR #4 round-10 review, Codex
        # P2 L38, the keychain lookup is deferred to here so the module
        # imports cleanly on non-macOS even though the ``security`` CLI
        # is unavailable.
        #
        # asyncio.to_thread wrapper: same rationale as quick_research above
        # (per PR #11 review, Gemini high). Extending the fix to this
        # pre-existing call site rather than leave the codebase in a
        # half-fixed state where only the newer function is event-loop-safe.
        response = await asyncio.to_thread(
            _get_perplexity_client().chat.completions.create,
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a thorough research assistant. Provide comprehensive, "
                        "well-sourced answers that synthesize information across multiple "
                        "sources. Include relevant details, comparisons, and caveats. "
                        "Always cite your sources."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
        )

        # Defensive: same empty-choices guard as quick_research (per PR #11
        # review, Gemini medium).
        choices = response.choices or []
        if not choices:
            return [
                TextContent(
                    type="text",
                    text="Error: Perplexity returned no choices for deep_research",
                )
            ]
        message = choices[0].message
        # Redact secret-shape patterns from scraped web content before the
        # response leaves this server. Perplexity's synthesis can include raw
        # API keys / JWTs / private-key blocks lifted from indexed pages.
        content = redact_secrets(message.content or "")
        result = f"## Research Results\n\n{content}"

        return [TextContent(type="text", text=result)]

    if name == "agent_research":
        # Fail-closed validation before any network traffic, mirroring the
        # gemini_* handlers: a structured {"status": "failed"} envelope so
        # the MCP client gets a parseable error rather than an exception.
        try:
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string.")

            model = arguments.get("model", AGENT_RESEARCH_DEFAULT_MODEL)
            if model not in AGENT_RESEARCH_MODELS:
                raise ValueError(
                    f"model must be one of {sorted(AGENT_RESEARCH_MODELS)}; "
                    f"got {model!r}."
                )

            max_output_tokens = arguments.get(
                "max_output_tokens", _AGENT_MAX_OUTPUT_TOKENS_DEFAULT
            )
            # Strict type check — bool is an int subclass in Python, so
            # `True` would otherwise slip through as 1 (same trap as the
            # collaborative_planning flag in gemini_deep_research_start).
            if (
                not isinstance(max_output_tokens, int)
                or isinstance(max_output_tokens, bool)
                or not (
                    _AGENT_MAX_OUTPUT_TOKENS_MIN
                    <= max_output_tokens
                    <= _AGENT_MAX_OUTPUT_TOKENS_MAX
                )
            ):
                raise ValueError(
                    "max_output_tokens must be an integer in "
                    f"[{_AGENT_MAX_OUTPUT_TOKENS_MIN}, "
                    f"{_AGENT_MAX_OUTPUT_TOKENS_MAX}]; got {max_output_tokens!r}."
                )

            # Strict bool check — `bool("false")` is True in Python, so a
            # JSON-stringified flag would silently flip the meaning (same
            # contract as collaborative_planning on gemini_deep_research_start).
            background = arguments.get("background", False)
            if not isinstance(background, bool):
                raise ValueError(
                    "background must be a JSON boolean (true/false), "
                    "not a string or number."
                )
        except ValueError as exc:
            err = {"status": "failed", "error": str(exc)}
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        payload: dict[str, Any] = {
            "model": model,
            "input": query,
            "tools": [{"type": "sandbox"}],
            "max_output_tokens": max_output_tokens,
            "instructions": _AGENT_RESEARCH_INSTRUCTIONS,
        }
        if background:
            payload["background"] = True

        data = await _post_agent_research(payload)
        # "failed" covers both the helper's HTTP-failure envelope and an
        # upstream terminal failure; "cancelled" gets the same envelope so
        # the sync path matches agent_research_result for that status.
        post_status = data.get("status")
        if post_status in ("failed", "cancelled"):
            err = {
                "status": "failed",
                "error": redact_secrets(
                    str(data.get("error") or f"agent task {post_status}")
                ),
            }
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        if background:
            response_id = data.get("id")
            if not isinstance(response_id, str) or not _AGENT_RESPONSE_ID_RE.fullmatch(
                response_id
            ):
                # Fail loudly: a null/malformed id breaks the poll contract
                # since the result tool can't be called without a valid id.
                raise RuntimeError(
                    "Agent API background start did not include a valid "
                    f"response id; got {response_id!r}."
                )
            result = {
                "response_id": response_id,
                # API-emitted string — redact like the renderer does for
                # the same field (per PR #16 review).
                "status": redact_secrets(str(data.get("status", "queued"))),
                "model": model,
                "hint": (
                    "Poll agent_research_result with this response_id. "
                    "Sandbox runs typically take one to several minutes."
                ),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        return _render_agent_research(data)

    if name == "agent_research_result":
        try:
            safe_id = _validate_agent_response_id(arguments.get("response_id"))
        except ValueError as exc:
            err = {"status": "failed", "error": str(exc)}
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        data = await _get_agent_response(safe_id)
        if data.get("status") == "failed" and "output" not in data:
            # Either the HTTP-failure envelope from the helper or an
            # upstream terminal failure with no output to render.
            err = {
                "status": "failed",
                "error": redact_secrets(str(data.get("error") or "agent task failed")),
            }
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        status = data.get("status", "unknown")
        if status in ("queued", "in_progress"):
            result = {
                "status": status,
                "hint": "Still running. Poll again in ~30 seconds.",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if status in ("cancelled", "failed"):
            # Also catches a terminal "failed" that arrived WITH an output
            # key and therefore fell through the no-output guard above —
            # the two checks together cover both upstream failure shapes.
            err = {
                "status": "failed",
                "error": redact_secrets(
                    str(data.get("error") or f"agent task {status}")
                ),
            }
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        # "completed", and "incomplete" with partial output — the renderer
        # flags any non-completed status inline.
        return _render_agent_research(data)

    if name == "gemini_deep_research_start":
        try:
            query = arguments["query"]
            mode = arguments.get("mode", "fast")
            if mode not in GEMINI_MODELS:
                raise ValueError(
                    f"mode must be one of {sorted(GEMINI_MODELS)}; got {mode!r}"
                )

            # Strict bool check — `bool("false")` is True in Python, so a
            # JSON-stringified flag would silently flip the meaning.
            collaborative_planning = arguments.get("collaborative_planning", False)
            if not isinstance(collaborative_planning, bool):
                raise ValueError(
                    "collaborative_planning must be a JSON boolean "
                    "(true/false), not a string or number."
                )

            thinking_summaries = arguments.get("thinking_summaries", "auto")
            if thinking_summaries not in {"auto", "none"}:
                raise ValueError(
                    "thinking_summaries must be 'auto' or 'none'; "
                    f"got {thinking_summaries!r}."
                )

            payload: dict[str, Any] = {
                "agent": GEMINI_MODELS[mode],
                "input": query,
                "background": True,
                "agent_config": {
                    "type": "deep-research",
                    "thinking_summaries": thinking_summaries,
                    "collaborative_planning": collaborative_planning,
                },
            }

            # Optional continuation. Validate with the same allowlist used for
            # interaction_id since it's also concatenated into the request body
            # and (more importantly) used by the upstream API for routing.
            previous_interaction_id = arguments.get("previous_interaction_id")
            if previous_interaction_id is not None:
                payload["previous_interaction_id"] = _validate_interaction_id(
                    previous_interaction_id
                )

            data = await _post_gemini_interaction(payload)

            # If the helper returned a structured failure, surface it directly
            # — no point trying to extract an id from an error envelope.
            if data.get("status") == "failed":
                return [TextContent(type="text", text=json.dumps(data, indent=2))]

            interaction_id = data.get("id")
            if not isinstance(interaction_id, str) or not _INTERACTION_ID_RE.fullmatch(
                interaction_id
            ):
                # Fail loudly: a null/malformed id breaks the poll contract
                # since the result tool can't be called without a valid id.
                raise RuntimeError(
                    "Gemini start response did not include a valid interaction id; "
                    f"got {interaction_id!r}."
                )

            result = {
                "interaction_id": interaction_id,
                "status": data.get("status", "in_progress"),
                "model": GEMINI_MODELS[mode],
                "hint": (
                    "Poll gemini_deep_research_result with this interaction_id. "
                    "Tasks take several minutes; up to 60 minutes max."
                ),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except ValueError as exc:
            err = {"status": "failed", "error": str(exc)}
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

    if name == "gemini_deep_research_result":
        try:
            safe_id = _validate_interaction_id(arguments["interaction_id"])
        except ValueError as exc:
            err = {"status": "failed", "error": str(exc)}
            return [TextContent(type="text", text=json.dumps(err, indent=2))]

        data = await _get_gemini_interaction(safe_id)
        status = data.get("status", "unknown")
        result: dict[str, Any] = {"status": status}

        # Normalize for terminal-status comparison; the API has used mixed case
        # historically (e.g. "Completed") — be liberal in what we accept.
        normalized_status = status.lower() if isinstance(status, str) else "unknown"

        if normalized_status == "completed":
            # Route all model-emitted text through the redactor — Deep Research
            # can lift API keys, JWTs, and private-key blocks from the open web.
            result["output_text"] = redact_secrets(data.get("output_text", ""))
            steps = data.get("steps") or []
            result["steps_count"] = len(steps)
            # Some upstream payloads have included non-dict step entries (raw
            # strings, nulls) — guard so a single malformed step doesn't crash
            # the entire result handler.
            result["steps_summary"] = [
                s.get("type") for s in steps if isinstance(s, dict)
            ]
        elif normalized_status == "requires_action":
            # Distinct non-terminal state: the agent has paused mid-run and is
            # waiting on user input (typically when collaborative_planning is
            # enabled). The caller should re-issue the interaction with the
            # required action attached rather than continuing to poll.
            result["hint"] = (
                "Agent is awaiting user input; collaborative-planning "
                "approval may be needed."
            )
        elif normalized_status in _GEMINI_TERMINAL_STATUSES:
            # "failed", "cancelled", "incomplete", "budget_exceeded", and any
            # future terminal status. Use whatever error/message field the
            # upstream provided.
            result["error"] = redact_secrets(
                data.get("error") or data.get("message") or f"task {normalized_status}"
            )
        else:
            result["hint"] = "Still running. Poll again in ~30 seconds."

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "list_sessions":
        sessions = list_sessions()
        if not sessions:
            return [
                TextContent(
                    type="text", text="## Saved Sessions\n\nNo saved sessions found.\n"
                )
            ]
        lines = [
            "## Saved Sessions",
            "",
            "| Session ID | Name | Messages | Last Modified |",
            "|------------|------|----------|---------------|",
        ]
        for s in sessions:
            # Sanitize session names for the Markdown table:
            # - escape pipe (|) so it doesn't break column boundaries
            # - replace newlines with spaces so a multi-line name
            #   doesn't collapse the table (per PR #4 follow-up review,
            #   Gemini medium L583).
            safe_name = s["name"].replace("|", "&#124;").replace("\n", " ")
            lines.append(
                f"| `{s['session_id']}` | {safe_name} | {s['message_count']} | {s['last_modified']} |"
            )
        return [TextContent(type="text", text="\n".join(lines) + "\n")]

    if name == "save_session":
        session_name = arguments.get("name", "Untitled")
        messages = arguments.get("messages", [])
        metadata = arguments.get("metadata", {})
        result = save_session(name=session_name, messages=messages, metadata=metadata)
        return [
            TextContent(
                type="text",
                text=f"Session saved: {result['session_id']} ({result['message_count']} messages)",
            )
        ]

    if name == "load_session":
        session_id = arguments.get("session_id")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            session = load_session(session_id)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        lines = [
            f"## Session: {session['name']}",
            "",
            f"**Created:** {session['created_at']}",
            f"**Last Modified:** {session['last_modified']}",
            "",
        ]
        # Surface saved metadata in the rendered output. Without this,
        # callers can save metadata via save_session but cannot retrieve
        # it via load_session — the helper returns it but the tool
        # surface used to drop it (per PR #4 round-7 review, Codex P2
        # L760: "Include saved metadata in load_session output").
        # Render the metadata as pretty JSON inside a fenced block so
        # nested objects/arrays survive the markdown trip without the
        # ambiguity of a flat key:value dump.
        metadata = session.get("metadata") or {}
        if metadata:
            lines.append("### Metadata")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(metadata, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
        lines.append("### Conversation History")
        lines.append("")
        for msg in session["messages"]:
            # Defensive: skip non-dict entries (corrupted/manually-edited
            # files) and coerce role to str before .upper() in case it
            # is null or numeric (per PR #3 follow-up review,
            # Gemini medium L569).
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown").upper()
            # str() coerce content too: handles null/numeric/non-string
            # values (renders as empty rather than literal "None")
            # (per PR #4 follow-up review, Gemini medium L626).
            content = str(msg.get("content") or "")
            lines.append(f"**{role}:** {content}")
            lines.append("")
        return [TextContent(type="text", text="\n".join(lines))]

    if name == "update_session":
        session_id = arguments.get("session_id")
        new_name = arguments.get("name")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            result = update_session(session_id, name=new_name)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        return [
            TextContent(
                type="text",
                text=f"Session updated: {result['session_id']} (name={result['name']})",
            )
        ]

    if name == "delete_session":
        session_id = arguments.get("session_id")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            delete_session(session_id)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        return [TextContent(type="text", text=f"Session deleted: {session_id}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
