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
MCP server providing three families of tools:

- ``quick_research`` / ``deep_research``: Perplexity Sonar / Sonar Pro
  — inline research with citations. ``quick_research`` uses the smaller
  Sonar model for fast, concise, well-scoped answers; ``deep_research``
  uses Sonar Pro for multi-source synthesis when the question spans
  sources or needs cross-referencing.
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
import json
import os
import re
import subprocess
import sys
import tempfile
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

    On HTTP error, returns a structured ``{"status": "failed", "error": ...}``
    dict instead of raising so the MCP client gets a graceful error envelope
    rather than an opaque exception. The shared httpx client gives us
    connection pooling across calls.
    """
    headers = await _gemini_headers()
    client = await _get_http_client()
    try:
        response = await client.post(
            f"{GEMINI_API_BASE}/interactions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)


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
        response = await client.get(
            f"{GEMINI_API_BASE}/interactions/{safe_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return _http_error_payload(exc)


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
            return [TextContent(type="text", text="Error: Perplexity returned no choices for quick_research")]
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
            return [TextContent(type="text", text="Error: Perplexity returned no choices for deep_research")]
        message = choices[0].message
        # Redact secret-shape patterns from scraped web content before the
        # response leaves this server. Perplexity's synthesis can include raw
        # API keys / JWTs / private-key blocks lifted from indexed pages.
        content = redact_secrets(message.content or "")
        result = f"## Research Results\n\n{content}"

        return [TextContent(type="text", text=result)]

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
