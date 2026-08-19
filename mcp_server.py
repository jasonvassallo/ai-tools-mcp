#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=1.0.0",
#     "mcp>=1.0.0,<2",
#     "httpx>=0.27",
#     "google-auth>=2.30",
#     "requests>=2.31",
#     # transitive floors per pip-audit 2026-07-07 (starlette, python-multipart,
#     # pyjwt, cryptography, pydantic-settings, idna)
#     "starlette>=1.3.1",
#     "python-multipart>=0.0.31",
#     "pyjwt>=2.13.0",
#     "cryptography>=48.0.1",
#     "pydantic-settings>=2.14.2",
#     "idna>=3.15",
# ]
# ///
"""
MCP server providing five families of tools:

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
- ``local_delegate`` / ``local_delegate_result``: local-first Ollama
  delegation — send a task to a local model (native /api/chat, think
  off by default) via an ordered endpoint chain:
  localhost first, then the user's own Cloudflare-Access-gated
  remote. Input text never leaves the user's machines; background
  jobs are in-memory and single-collect.
- ``list_sessions`` / ``save_session`` / ``load_session`` /
  ``update_session`` / ``delete_session``: local conversation-session
  persistence backed by ``~/.claude/sessions/``.

Designed to complement Claude's built-in WebSearch tool (quick factual
lookups, single-answer questions).

PLATFORM: macOS / POSIX and Windows. The session helpers lock via
``fcntl.flock`` on POSIX and ``msvcrt.locking`` byte-range locks on
Windows (same dedicated lockfile, byte 0 by convention), so all
thirteen tools work on both families; a platform with neither module
gets a clean ValueError from ``update_session`` / ``delete_session``.
Credentials resolve env-first everywhere; the macOS Keychain is the
fallback where ``security(1)`` exists (per PR #4 round-10 review,
Codex P2 L38: lookups are deferred so ``import mcp_server`` succeeds
on any platform). On Windows the Cloudflare Access service token also
resolves from Credential Manager, between those two tiers — the
preferred store there, since a persisted env var is plaintext and
Claude Desktop launches the packaged extension outside any shell and
so cannot be handed one by a profile script.

For the Gemini tools, Application Default Credentials (ADC) are
likewise loaded lazily on first ``gemini_*`` call rather than at
module import. This means the MCP server can start and the
Perplexity-backed ``deep_research`` and session tools can be used
even on a machine without ``gcloud auth application-default login``
having been run; only the ``gemini_*`` tools will fail when invoked.
"""

import asyncio
import atexit
import errno
import functools
import getpass
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, NamedTuple

import google.auth
import google.auth.transport.requests
import httpx
import requests

# fcntl is POSIX-only; on Windows the import fails. We catch
# ImportError so the module can still be imported (e.g. for docs,
# tool discovery, or the deep_research path which doesn't need
# locking) — _session_lock falls back to msvcrt on Windows, or raises
# a clean error when neither is available (per PR #4 round-9 review,
# Gemini medium L17).
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised via mocked import
    fcntl = None  # type: ignore[assignment]

# msvcrt is the Windows counterpart (byte-range locking); absent on
# POSIX. _session_lock uses whichever module is available, so
# update_session / delete_session work on both platform families.
try:
    import msvcrt
except ImportError:  # pragma: no cover - absent on POSIX; mocked in tests
    msvcrt = None  # type: ignore[assignment]

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
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


# v1.2 (issue #20): credentials resolve environment-first, then macOS
# Keychain. The env step is what makes non-macOS hosts (Windows) work —
# there is no security(1) there, so credentials are supplied via per-user
# environment variables instead. Most services map to an env var named
# after the Keychain service; the exceptions live here.
_CRED_ENV_OVERRIDES: dict[tuple[str, str], str] = {
    ("api_tokens", "perplexity"): "PERPLEXITY_API_KEY",
}

# Keychain service names for the Cloudflare Access service token that gates
# the remote Ollama endpoints. Defined here rather than beside the delegate
# code that consumes them because _CRED_VAULT_TARGETS keys off them.
_CF_ACCESS_ID_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_ID"
_CF_ACCESS_SECRET_KEYCHAIN_SERVICE = "OLLAMA_CF_ACCESS_CLIENT_SECRET"

# Windows Credential Manager targets (generic credentials, user-scoped).
#
# Why a third tier at all: a persisted HKCU env var is plaintext, readable
# by any process running as the same user. The vault is the stronger store,
# but a shell profile cannot bridge to it — Claude Desktop launches the
# packaged .mcpb extension outside any shell, so it inherits nothing. The
# read therefore has to happen in-process, which is what this tier does.
#
# Only services listed here are ever looked up in the vault; everything
# else keeps the exact two-tier behaviour it had before.
_CRED_VAULT_TARGETS: dict[str, str] = {
    _CF_ACCESS_ID_KEYCHAIN_SERVICE: "ai-tools-mcp-cf-access/client-id",
    _CF_ACCESS_SECRET_KEYCHAIN_SERVICE: "ai-tools-mcp-cf-access/client-secret",
}

# Source labels. Safe to print — they name a tier, never a value.
_CRED_SOURCE_ENV = "env"
_CRED_SOURCE_VAULT = "windows-credential-manager"
_CRED_SOURCE_KEYCHAIN = "macos-keychain"

_CRED_TYPE_GENERIC = 1  # wincred.h CRED_TYPE_GENERIC


def _cred_env_var(service: str, account: str) -> str:
    override = _CRED_ENV_OVERRIDES.get((service, account))
    if override:
        return override
    return re.sub(r"[^A-Z0-9]", "_", service.upper())


def _decode_credential_blob(blob: bytes) -> str:
    """Decode a CredentialBlob to text. Never raises; never logs the value.

    Windows tooling writes generic-credential blobs as UTF-16LE — the
    sibling CredVault.psm1 (``[Text.Encoding]::Unicode``), ``cmdkey``, and
    Python's ``keyring`` all agree — but a blob written by some other tool
    may be UTF-8. An undecodable blob yields "" so the caller fails closed
    rather than forwarding garbage as a credential.

    The discriminator is the SECOND byte, not "contains a NUL anywhere":
    an ASCII character in UTF-16LE always has 0x00 as its high byte, so
    ``blob[1] == 0`` identifies UTF-16LE reliably. "Contains a NUL" is not
    a safe test — a NUL-terminated UTF-8 blob of even length (``b"abc\\x00"``,
    which some C callers write via ``strlen + 1``) contains one, decodes as
    UTF-16LE *without* raising, and would yield CJK garbage.

    Caveat, deliberate: this assumes the UTF-16LE case starts with an ASCII
    character. A UTF-16LE blob whose FIRST character is non-ASCII (e.g. CJK)
    is misread as UTF-8. That trade is right for the only credentials this
    decodes — CF Access service tokens are ASCII — but do not reuse this on
    secrets that may lead with non-ASCII text.
    """
    if not blob:
        return ""
    looks_utf16 = len(blob) >= 2 and blob[1] == 0
    encodings = ("utf-16-le", "utf-8") if looks_utf16 else ("utf-8", "utf-16-le")
    for encoding in encodings:
        try:
            return blob.decode(encoding).rstrip("\x00")
        except UnicodeDecodeError:
            continue
    return ""


@functools.lru_cache(maxsize=1)
def _windows_credential_api() -> tuple[Any, Any, Any] | None:
    """Bind advapi32's CredReadW/CredFree plus the CREDENTIALW layout.

    Returns None on any non-Windows host, which is what keeps macOS/Linux
    resolution byte-for-byte unchanged. Bound lazily because
    ``ctypes.wintypes`` does not import off Windows and this module is
    imported on Linux by CI; cached because the delegate reads both CF
    credentials on every endpoint probe.

    ctypes over a package (keyring/pywin32) deliberately: the PEP 723
    dependency block is resolved on every ``uv run`` of the server, so a
    new wheel is a real cost on every launch and every install surface,
    and this needs exactly two calls from a stdlib module that is already
    present on the only platform that uses it.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:  # pragma: no cover - ctypes is stdlib on Windows
        return None

    class CREDENTIALW(ctypes.Structure):
        # Field order/types mirror wincred.h; ctypes applies the platform's
        # natural alignment, which is the layout the API expects.
        _fields_ = (
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_char)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        )

    try:
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        cred_read = advapi32.CredReadW
        cred_read.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(CREDENTIALW)),
        )
        cred_read.restype = wintypes.BOOL
        cred_free = advapi32.CredFree
        cred_free.argtypes = (ctypes.c_void_p,)
        cred_free.restype = None
    except (AttributeError, OSError):  # pragma: no cover - advapi32 is always present
        return None
    return cred_read, cred_free, CREDENTIALW


def _read_windows_credential(target: str) -> str | None:
    """Read one generic credential from Windows Credential Manager.

    Returns the stored secret, ``""`` when the entry exists but is empty,
    or None when the target is absent or this host has no vault. Never
    raises and never logs the value — the caller decides what to do with
    a miss.
    """
    api = _windows_credential_api()
    if api is None:
        return None
    import ctypes

    cred_read, cred_free, credential_t = api
    pcred = ctypes.POINTER(credential_t)()
    if not cred_read(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pcred)):
        # ERROR_NOT_FOUND (1168) or any other failure — both mean "nothing
        # usable here", so fall through to the next tier rather than raise.
        return None
    try:
        cred = pcred.contents
        size = int(cred.CredentialBlobSize)
        blob = ctypes.string_at(cred.CredentialBlob, size) if size > 0 else b""
    finally:
        cred_free(pcred)
    return _decode_credential_blob(blob)


def _resolve_credential(service: str, account: str) -> tuple[str, str]:
    """Resolve a credential and report which tier supplied it.

    Order: environment variable → Windows Credential Manager (Windows
    only, and only for services in ``_CRED_VAULT_TARGETS``) → macOS
    Keychain. Env stays first so an existing install keeps working
    unchanged and a vault cutover is reversible by putting the env var
    back.

    Returns ``(value, source)``. The value goes to the caller and must
    never be logged; ``source`` is a tier name and is safe to print.
    """
    env_var = _cred_env_var(service, account)
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val, _CRED_SOURCE_ENV

    vault_target = _CRED_VAULT_TARGETS.get(service)
    if vault_target:
        # No-ops off Windows (_read_windows_credential returns None there),
        # so non-Windows resolution is exactly what it was before.
        vault_val = (_read_windows_credential(vault_target) or "").strip()
        if vault_val:
            return vault_val, _CRED_SOURCE_VAULT

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        result = None  # non-macOS: no security(1) binary
    if result is not None and result.returncode == 0:
        # Unconditional, including an empty password: callers that must
        # fail closed on a blank credential already check for it
        # (_ollama_auth_headers), and changing that here would move the
        # decision away from the caller that documents why.
        return result.stdout.strip(), _CRED_SOURCE_KEYCHAIN

    message = (
        f"Credential not found. Set the {env_var} environment variable "
        f"(required on non-macOS), or add it to the macOS Keychain with:\n"
        f"  security add-generic-password -s '{service}' -a '{account}' -w 'YOUR_API_KEY'"
    )
    if vault_target:
        # ASCII only: this text reaches a Windows console via run_check's
        # `print(f"fail: {e}")`, where a redirected stdout is cp1252.
        message += (
            f"\nOn Windows the preferred store is Credential Manager: add a "
            f"generic credential targeting '{vault_target}' (see README, "
            f"'Windows: Credential Manager vault')."
        )
    raise ValueError(message)


def get_api_key_from_keychain(service: str, account: str) -> str:
    """Retrieve a credential: env var, then Windows vault, then Keychain.

    Thin wrapper over ``_resolve_credential`` for callers that only need
    the value. Empty/whitespace env values are ignored (fail closed, never
    treat blank as a credential). A miss in every available source raises
    naming each remedy that applies to this host.
    """
    return _resolve_credential(service, account)[0]


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
    project = project or getattr(creds, "quota_project_id", None)
    if not project:
        raise ValueError(
            "Could not determine billing project from ADC. Run: "
            "gcloud auth application-default set-quota-project YOUR_PROJECT"
        )
    return creds, project


def _report_cf_access_credentials() -> None:
    """Print the resolution tier for each CF Access credential (--check only).

    Values are never printed — only the tier that supplied them.

    When an env var and a vault entry both hold a value for the same
    credential and they DISAGREE, say so. Env wins by precedence, so a
    stale env var silently shadows a rotated vault entry, and the only
    other symptom is a 403 from the Access-gated endpoint that looks
    identical to "no credentials at all". Comparing here costs nothing and
    leaks nothing.
    """
    user = getpass.getuser()
    for label, service in (
        ("client id", _CF_ACCESS_ID_KEYCHAIN_SERVICE),
        ("client secret", _CF_ACCESS_SECRET_KEYCHAIN_SERVICE),
    ):
        try:
            value, source = _resolve_credential(service, user)
        except ValueError:
            value, source = "", ""
        if not value:
            # An existing-but-empty Keychain item resolves to ("", "keychain")
            # rather than raising (see _resolve_credential). Reporting that as
            # `ok` would be a FALSE SUCCESS on the one surface people use to
            # verify a cutover, while _ollama_auth_headers rejects the blank
            # and skips every remote endpoint. Treat blank as missing here.
            print(
                f"warn: cloudflare access {label} not found "
                "(remote ollama endpoints will be skipped)"
            )
            continue
        print(f"ok: cloudflare access {label} found ({source})")

        target = _CRED_VAULT_TARGETS.get(service)
        if source != _CRED_SOURCE_ENV or not target:
            continue
        vaulted = (_read_windows_credential(target) or "").strip()
        if vaulted and vaulted != value:
            # ASCII only (cp1252 console on a redirected Windows stdout).
            print(
                f"warn: cloudflare access {label} differs between the "
                f"{_cred_env_var(service, user)} env var (winning) and vault "
                f"target {target!r}. If remote endpoints return 403, the env "
                "var is the stale copy: clear it to fall through to the vault."
            )


def run_check() -> None:
    """Validate configuration and exit. Used by install.sh to verify setup."""
    errors = 0
    try:
        _, source = _resolve_credential("api_tokens", "perplexity")
        print(f"ok: perplexity key found ({source})")
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

    # Report which tier supplied each Cloudflare Access credential. Only
    # the tier name is printed, never a value — so `--check` is the safe
    # way to confirm a Credential Manager cutover actually took effect
    # (the installed .mcpb is a packaged copy, so "it works in the CLI"
    # proves nothing about Desktop).
    _report_cf_access_credentials()

    # Non-fatal: local_delegate family. Ollama being down must not fail
    # installs or preflights of the hosted tool families — delegate calls
    # themselves fail closed at call time.
    try:
        chain = _resolve_ollama_chain()
    except ValueError as e:
        print(
            "warn: ollama endpoint chain invalid (local_delegate unavailable): "
            f"{redact_secrets(str(e))}"
        )
        chain = []
    for endpoint in chain:
        try:
            headers = _ollama_auth_headers(endpoint)
            if headers is None:
                print(
                    "warn: ollama endpoint skipped (no Cloudflare Access creds "
                    f"in env or Keychain): {endpoint}"
                )
                continue
            # allow_redirects=False: a CF Access service-token header must
            # never follow a redirect off-host — same rationale as the
            # shared httpx client's follow_redirects=False.
            resp = requests.get(
                f"{endpoint}/api/version",
                headers=headers,
                timeout=3,
                allow_redirects=False,
            )
            resp.raise_for_status()
            version = resp.json().get("version", "?")
            print(f"ok: ollama reachable at {endpoint} (version {version})")
        except (ValueError, requests.RequestException) as e:
            print(
                f"warn: ollama not reachable at {endpoint} "
                f"(local_delegate may fall back): {redact_secrets(str(e))}"
            )
    env_default = os.environ.get(_OLLAMA_DEFAULT_MODEL_ENV_VAR, "").strip()
    if env_default and env_default not in OLLAMA_DELEGATE_MODELS:
        print(
            f"warn: {_OLLAMA_DEFAULT_MODEL_ENV_VAR}={env_default!r} not in "
            f"allowlist; using {OLLAMA_DELEGATE_DEFAULT_MODEL}"
        )

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
    """Cooperative per-session lockfile (POSIX flock / Windows msvcrt).

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
    if fcntl is None and msvcrt is None:
        # Neither POSIX flock nor Windows byte-range locking exists on
        # this platform. Fail clearly rather than fall through to a
        # no-op lock that would silently let the resurrection race
        # re-open. ValueError (not OSError) so the tool dispatcher's
        # existing `except ValueError` surfaces this as a clean tool
        # error instead of an internal exception (Codex P2, PR #22).
        raise ValueError(
            "update_session/delete_session require OS advisory file "
            "locking (fcntl on POSIX, msvcrt on Windows), and this "
            "platform provides neither — these two tools are "
            "unavailable here; all other tools work."
        )
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = session_file.with_suffix(".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            # Windows: msvcrt.locking locks a byte RANGE starting at the
            # current file position — seek to 0 so every cooperating
            # process locks the same byte (whole-file convention on a
            # dedicated lockfile). LK_LOCK internally retries ~10 times
            # a second apart, then raises OSError with EACCES/EDEADLK
            # while the region is still held; loop on exactly those
            # errnos to emulate flock's indefinite LOCK_EX blocking,
            # and propagate anything else (EBADF etc.) as a real error.
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
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
        elif msvcrt is not None:
            try:
                # Same seek-to-0 convention as acquisition: unlock the
                # exact byte range that was locked.
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                # Benign for the same reason as the flock branch.
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
    # ``"string"`` would crash the .get() calls below. Preserve the public
    # ValueError contract for malformed persisted content, distinct from a
    # missing or unreadable session.
    # (per PR #3 follow-up review, Gemini medium L275).
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004 - persisted-content validation is a value error
            f"Session file shape is not a JSON object: {session_id}"
        )

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
            raise ValueError(  # noqa: TRY004 - persisted-content validation is a value error
                f"Session file shape is not a JSON object: {session_id}"
            )

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
    whatever ``get_api_key_from_keychain`` raises — since v1.2 that is
    always ``ValueError`` (a missing key, or a missing ``security(1)``
    binary on non-macOS, both folded into the same actionable error).
    Per PR #4 round-10 review, Codex P2 L38.
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


def _http_error_payload(
    exc: httpx.HTTPStatusError, *, scrub: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Build a structured failure dict from an httpx HTTPStatusError.

    Keeps a short body snippet (≤500 chars) so the caller has enough context to
    diagnose without bloating the MCP response. Runs the snippet through
    `redact_secrets` because Gemini error bodies have, on occasion, echoed
    request headers or query content.

    `scrub` is an optional set of exact secret values (e.g. live header
    values a caller holds) to strip from the body — value-aware, precise
    replacement for secret shapes `redact_secrets`'s patterns don't cover.
    Applied to the FULL body, before the 500-char truncation: a secret
    straddling the cutoff would otherwise leave an un-scrubbed fragment.
    """
    status_code = exc.response.status_code
    try:
        body = exc.response.text or ""
    except Exception:  # noqa: BLE001 - never let body extraction shadow the real error
        body = ""
    for secret in scrub:
        if secret:
            body = body.replace(secret, "[REDACTED_CF_ACCESS]")
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
# native API accepts `think` — and think:false is required for fast
# structured work on thinking models. Whether a given tag can think is NOT
# hardcoded here: it is read per call from the endpoint's /api/show
# `capabilities` list (measured 2026-07-20: BOTH built-in defaults report
# 'thinking' — an earlier comment claiming gemma4 does not was wrong, an
# assertion that had never been tested). Non-thinking override tags get
# `think` stripped plus an advisory instead of Ollama's hard 400.
#
# Default is gemma4:12b-nvfp4. Measured 2026-07-20 over a 16-task machine-graded
# delegate benchmark (3 trials, the no-options/default-temperature regime
# this tool actually sends):
#
#   gemma4:12b-nvfp4   mean 0.917    0% cross-task contamination
#   qwen3.6 35B  mean 0.732   20% cross-task contamination
#
# "Cross-task contamination" = the model returns the completion belonging
# to a DIFFERENT recently-seen prompt. It is 0% on a prompt's first call and
# ~25% on repeat calls, so it hits exactly the short, structurally-similar
# codegen/transform prompts this tool is used for. Pinning temperature 0 cuts
# it to 6% but does NOT recover the score (0.733) — the failures just become
# deterministic, which is why the default is a model change and not an
# options change. Root-caused 2026-07-31 (48-call repeat protocol, both qwen
# families): long-lived runner state under OLLAMA_KEEP_ALIVE=-1 — it
# reproduces across q8_0 AND f16 KV cache, MoE AND dense qwen, Ollama 0.31.1
# AND 0.32.5. Unloading BETWEEN calls eliminates it (0/96 measured on a
# protocol where every call began on a fresh runner); gemma is immune under
# the identical config (0/141 lifetime). Hence the qwen-conditional
# keep_alive default below. keep_alive:"0" on a request does NOT protect
# that request itself — it is a post-response TTL, and a call landing on an
# already-dirty resident runner contaminates at the control rate (measured
# 2026-08-08: 16.1%/25.0% vs 25.8%/25.0% control across both qwen families,
# 63 paired probes) — which is why _evict_ollama_runner drops the runner
# BEFORE every protected call.
#
# The measurement above is against qwen3.6:35b-a3b, which no longer exists on
# the endpoint (retired 2026-08-17). The surviving qwen tag, qwen3.8:27b-nvfp4,
# has NOT been benchmarked here and its "long-context advantage" is gone with
# the per-context tag variants: every call now runs at the serving host's
# window regardless of tag. Prefer gemma4:31b-nvfp4 for review and
# long-context work (see the tool schema description and the allowlist comment
# below); neither family can be trusted to count or aggregate over long inputs
# (both scored 0.33 on that task).
_OLLAMA_MODELS_ENV_VAR = "AI_TOOLS_OLLAMA_MODELS"
_OLLAMA_BUILTIN_DELEGATE_MODELS: tuple[str, ...] = (
    "gemma4:12b-nvfp4",
    # 2026-08-17: the qwen3.6 tags were removed from the endpoint. There are no
    # per-context-window tag variants any more -- both remaining large tags
    # advertise context_length 262144 (verified via /api/show on ollama-mbp
    # 2026-08-18: gemma4.context_length=262144, qwen3_5.context_length=262144 --
    # `qwen3_5` there is Ollama's model_info ARCHITECTURE key for the qwen3.8:27b
    # tag, not a typo: the family key is independent of the release tag name;
    # this is the model's advertised window, NOT a quality benchmark -- qwen3.8
    # remains unbenchmarked here), but local_delegate sends no options.num_ctx,
    # so a call runs at the serving host's OLLAMA_CONTEXT_LENGTH (64k on JVMBPro,
    # 32k on jvmacmini). gemma4:31b is the reviewer/long-context tier (kept warm on the
    # MBP); qwen3.8:27b is the surviving qwen-family tag, still subject to the
    # qwen keep_alive:"0" contamination default below.
    "gemma4:31b-nvfp4",
    "qwen3.8:27b-nvfp4",
)


def _resolve_delegate_models() -> tuple[str, ...]:
    """v1.2 (issue #20): allowlist overridable per machine via env/user_config.

    Comma-separated, order-preserving, deduplicated; the first entry becomes
    the default model. Blank or effectively-empty values fall back to the
    built-in tags (fail closed — a typo'd setting cannot yield an empty
    allowlist that rejects everything).
    """
    raw = os.environ.get(_OLLAMA_MODELS_ENV_VAR, "").strip()
    if not raw:
        return _OLLAMA_BUILTIN_DELEGATE_MODELS
    models = tuple(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    return models or _OLLAMA_BUILTIN_DELEGATE_MODELS


OLLAMA_DELEGATE_MODELS: tuple[str, ...] = _resolve_delegate_models()
OLLAMA_DELEGATE_DEFAULT_MODEL = OLLAMA_DELEGATE_MODELS[0]

_OLLAMA_URL_ENV_VAR = "AI_TOOLS_OLLAMA_URL"
_OLLAMA_URL_KEYCHAIN_SERVICE = "OLLAMA_URL"
# _CF_ACCESS_{ID,SECRET}_KEYCHAIN_SERVICE live with the credential
# resolver near the top of this file — _CRED_VAULT_TARGETS keys off them.

# `0` (unload immediately) or 1-9999 seconds/minutes/hours. Strict so a
# malformed value cannot smuggle arbitrary JSON into the Ollama request.
_DELEGATE_KEEP_ALIVE_RE = re.compile(r"^(0|[1-9][0-9]{0,3}(s|m|h))$")

# Models whose omitted keep_alive defaults to "0" (unload after the call)
# instead of inheriting the server's OLLAMA_KEEP_ALIVE. This is the
# repeat-call contamination mitigation (see the allowlist comment above):
# a resident qwen runner returns other prompts' answers on ~15-25% of repeat
# calls; unloading between calls measured 0/96 contaminated. Prefix match so
# every qwen tag — built-in or env-overridden — is covered. An explicit
# caller keep_alive always wins (deliberate warm-pinning stays possible).
_KEEP_ALIVE_ZERO_MODEL_PREFIXES: tuple[str, ...] = ("qwen",)


def _keep_alive_zero_default_applies(model: str) -> bool:
    """True when an omitted keep_alive should default to "0" for this tag.

    Matches on the final path component, case-folded: the env-overridable
    allowlist accepts arbitrary tags (e.g. "hf.co/acme/qwen-model",
    "Qwen3:latest"), where a bare whole-tag prefix check would silently
    skip the mitigation (PR #60 review findings).
    """
    name = model.rsplit("/", 1)[-1].lower()
    return name.startswith(_KEEP_ALIVE_ZERO_MODEL_PREFIXES)


# Shared-client default is 30s; delegate calls pass explicit per-request
# timeouts (same mechanism as _AGENT_API_TIMEOUT_SECONDS).
_DELEGATE_TIMEOUT_DEFAULT_S = 300
_DELEGATE_TIMEOUT_MAX_S = 600
_DELEGATE_BG_CEILING_S = 1800.0

# jvmacmini runs num_parallel=1 — queuing beyond a few jobs would lie to
# the caller; fail fast instead.
_DELEGATE_JOB_CAP = 4
# Completed jobs the caller never collects via local_delegate_result would
# otherwise accumulate forever on a long-lived server — bound the tail.
_DELEGATE_DONE_RETAINED = 16
_DELEGATE_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


_OLLAMA_URLS_ENV_VAR = "AI_TOOLS_OLLAMA_URLS"
_OLLAMA_DEFAULT_MODEL_ENV_VAR = "AI_TOOLS_OLLAMA_DEFAULT_MODEL"

# v1.1 (spec amendment): local-first endpoint chain. The remote defaults are
# the user's own Cloudflare-Access-gated tunnels — never a third-party
# service. Order: local → JVMBPro (ollama-mbp: gemma4:31b/12b, qwen3.8;
# 64k host window; laptop, may be off) → jvmacmini (ollama.djvassallo.com:
# gemma4:12b-nvfp4 only, 32k host window, always-on server).
_OLLAMA_DEFAULT_CHAIN: tuple[str, ...] = (
    "http://localhost:11434",
    "https://ollama-mbp.djvassallo.com",
    "https://ollama.djvassallo.com",
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
    # Display-safe form for every error branch below: never echo userinfo
    # back to the caller. redact_secrets only masks known secret SHAPES
    # (JWT, Google API key, ...) — an arbitrary password like "hunter2"
    # has no shape it can match, so it would otherwise reach the MCP
    # error verbatim. Masking the netloc's userinfo here closes that gap
    # regardless of which branch raises. (IPv6 hosts lose their brackets
    # in this display-only string — acceptable, this is never re-parsed.)
    if parsed.username is not None or parsed.password is not None:
        masked_netloc = "***@" + (parsed.hostname or "")
        if parsed.port:
            masked_netloc += f":{parsed.port}"
        display = urllib.parse.urlunparse(parsed._replace(netloc=masked_netloc))
    else:
        display = url
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid Ollama endpoint {redact_secrets(display)!r}: must be "
            "http(s)://host[:port]"
        )
    if parsed.scheme == "http" and not _is_localhost_endpoint(url):
        raise ValueError(
            f"Refusing plain-http non-localhost Ollama endpoint "
            f"{redact_secrets(display)!r} — remote endpoints must be https"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"Refusing Ollama endpoint with embedded credentials "
            f"{redact_secrets(display)!r} — auth belongs in the Keychain "
            "(CF Access service token), never in the URL"
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
        # Not found in env or Keychain (get_api_key_from_keychain folds the
        # non-macOS missing-security(1) case into ValueError) — this entry
        # is optional config, so degrade gracefully instead of crashing.
        pass
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
    service-token headers resolved per call (never cached, never logged)
    from env, the Windows Credential Manager vault, or the Keychain —
    whichever answers first. Either credential absent → None (fail
    closed): the caller treats the endpoint as unavailable rather than
    calling an Access-gated host unauthenticated.
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
        # Not found in any source (get_api_key_from_keychain folds the
        # non-macOS missing-security(1) case into ValueError) — degrade
        # gracefully (remote endpoint skipped, never called bare) instead
        # of crashing.
        return None
    if not client_id or not client_secret:
        # A Keychain or vault item can exist with an empty password —
        # `security` returns "" with returncode 0 (no ValueError), and
        # CredReadW succeeds with CredentialBlobSize 0. Treat both the same
        # as "absent" so we fail closed instead of calling the
        # Access-gated host with a malformed header.
        return None
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": client_secret,
    }


_ollama_endpoint_cache: dict[str, tuple[str, float]] = {}
_implicit_resolution_cache: dict[str, tuple[str, str, str, float]] = {}
_ollama_capability_cache: dict[tuple[str, str], tuple[frozenset[str], float]] = {}
_OLLAMA_CAPABILITY_CACHE_TTL_S = 300.0


async def _probe_endpoint_tags(endpoint: str, attempts: list[str]) -> list[str] | None:
    """Model tags served by `endpoint`, or None with the reason in `attempts`.

    Single shared probe for both selection paths so the auth handling and
    its security rationale live in exactly one place.
    """
    headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
    if headers is None:
        attempts.append(
            f"{endpoint}: skipped (Cloudflare Access credentials not in "
            "env, Windows Credential Manager, or Keychain)"
        )
        return None
    client = await _get_http_client()
    try:
        # Auth is server-sourced (CF Access service-token creds from env
        # or Keychain via _ollama_auth_headers), never a caller-supplied
        # credential. The endpoint comes from operator config (env /
        # user_config / Keychain / built-in defaults), validated
        # https-only for non-localhost with URL userinfo rejected — no
        # tool parameter can redirect the credential to an attacker
        # host. Same false-positive class as the PR #15 suppressions.
        response = await client.get(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            f"{endpoint}/api/tags",
            headers=headers,
            timeout=_OLLAMA_PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        attempts.append(f"{endpoint}: HTTP {exc.response.status_code}")
        return None
    except httpx.RequestError:
        attempts.append(f"{endpoint}: unreachable")
        return None
    except ValueError:
        attempts.append(f"{endpoint}: invalid JSON from /api/tags")
        return None
    models = data.get("models", []) if isinstance(data, dict) else []
    return [
        str(m.get("name") or m.get("model") or "")
        for m in models
        if isinstance(m, dict)
    ]


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
    attempts: list[str] = []
    for endpoint in chain:
        tags = await _probe_endpoint_tags(endpoint, attempts)
        if tags is None:
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


async def _resolve_implicit_model() -> tuple[str, str, str]:
    """(model, endpoint, advisory_note) for a call that omitted `model`.

    Walks the endpoint chain once and picks the first endpoint serving ANY
    allowlisted tag; among that endpoint's tags, allowlist order decides.
    Chain order already encodes the operator's locality preference
    (localhost first by default), which yields two properties the naive
    "resolve allowlist[0]" default lacked (both flagged on PR #32):

    - An install that has only the qwen tags pulled locally keeps resolving
      locally instead of silently shipping prompt text to a remote host
      just because the tool-wide default isn't present there (CWE-200
      concern — the whole point of this tool is that input stays local
      when a local option exists).
    - A host where the default tag isn't served by anything reachable
      falls back to the next allowlisted tag instead of hard-failing.

    Any substitution (resolved model != allowlist[0]) is reported in the
    returned note. Explicit `model=` calls never come through here — they
    keep strict fail-loud semantics via _select_ollama_endpoint. Raises
    ValueError when no endpoint serves any allowlisted tag.
    """
    default = _delegate_default_model()
    cached = _implicit_resolution_cache.get(default)
    if cached is not None and time.monotonic() < cached[3]:
        return cached[0], cached[1], cached[2]
    chain = await asyncio.to_thread(_resolve_ollama_chain)
    attempts: list[str] = []
    ordered = (default, *(m for m in OLLAMA_DELEGATE_MODELS if m != default))
    for endpoint in chain:
        tags = await _probe_endpoint_tags(endpoint, attempts)
        if tags is None:
            continue
        for tag in ordered:
            if tag not in tags:
                continue
            note = ""
            if tag != default:
                where = (
                    "localhost"
                    if _is_localhost_endpoint(endpoint)
                    else redact_secrets(endpoint)
                )
                note = (
                    f"Note: default model {default} is not served by any "
                    f"reachable endpoint checked before {where}; using {tag} "
                    f"({where}) instead.\n\n"
                )
            expiry = time.monotonic() + _OLLAMA_PROBE_CACHE_TTL_S
            _ollama_endpoint_cache[tag] = (endpoint, expiry)
            _implicit_resolution_cache[default] = (tag, endpoint, note, expiry)
            return tag, endpoint, note
        present = ", ".join(sorted(t for t in tags if t)) or "no models"
        attempts.append(f"{endpoint}: no allowlisted model (has {present})")
    detail = "; ".join(attempts) or "empty endpoint chain"
    raise ValueError(
        f"No Ollama endpoint serves any allowlisted model: {redact_secrets(detail)}"
    )


async def _model_capabilities(endpoint: str, model: str) -> frozenset[str] | None:
    """Capabilities `endpoint` reports for `model` via /api/show, or None.

    None means "could not determine" (endpoint too old to report
    capabilities, transient failure, unexpected shape) and callers MUST
    treat it as neutral: no advisory, request payload untouched — a wrong
    "your flag was ignored" note is worse than letting Ollama surface its
    own error. Cached per (endpoint, model); capabilities are a property
    of the tag so a longer TTL than the serving probe is safe.
    """
    key = (endpoint, model)
    cached = _ollama_capability_cache.get(key)
    now = time.monotonic()
    if cached is not None and now < cached[1]:
        return cached[0]
    headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
    if headers is None:
        return None
    client = await _get_http_client()
    try:
        # Same server-sourced-auth / operator-configured-endpoint rationale
        # as _probe_endpoint_tags above.
        response = await client.post(  # nosemgrep: python.mcp.mcp-auth-passthrough-taint.mcp-auth-passthrough-taint
            f"{endpoint}/api/show",
            json={"model": model},
            headers=headers,
            timeout=_OLLAMA_PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError):
        return None
    caps = data.get("capabilities") if isinstance(data, dict) else None
    if not isinstance(caps, list):
        return None
    result = frozenset(str(c) for c in caps)
    _ollama_capability_cache[key] = (result, now + _OLLAMA_CAPABILITY_CACHE_TTL_S)
    return result


# An eviction is a control-plane call, not inference: it either drops a
# resident runner or no-ops. Bounded well under the delegate timeout so a
# wedged Ollama cannot stall the caller's real request behind it.
_OLLAMA_EVICT_TIMEOUT_S = 30.0

# Whatever the eviction costs, the chat still gets a usable slice: a caller
# timeout consumed entirely by a pathological eviction would turn a working
# call into an instant timeout, which is worse than the contamination.
_OLLAMA_MIN_CHAT_TIMEOUT_S = 5.0

# stdout is the MCP protocol stream, so this goes to stderr, which the
# host captures per-server (Claude Code writes one JSON record per stderr
# LINE into mcp-logs-ai-tools-mcp/*.jsonl — verified), hence the
# single-line key=value format.
_evict_log = logging.getLogger("ai_tools_mcp.delegate.evict")
# Level and propagation are set UNCONDITIONALLY, outside the handler guard
# below, because logging.getLogger() is process-global: anything that
# touched this name first — a host, the SDK, an earlier import of this
# module in the same process — leaves the guard skipped, and with it
# everything the guard would have configured. propagate is the dangerous
# half: left True, every record also climbs to the root logger, whose
# handlers this module does not own and one of which may well write to
# stdout, i.e. straight into the MCP protocol stream.
_evict_log.setLevel(logging.INFO)
_evict_log.propagate = False
if not _evict_log.handlers:
    # The stderr write happens on the listener's thread, never on the
    # caller's: QueueHandler.emit() is an enqueue. The caller here is the
    # asyncio event loop, and a StreamHandler write+flush into a stderr pipe
    # the host has stopped draining blocks the WHOLE loop — every other MCP
    # request included — for as long as the pipe stays full, outside every
    # timeout this module sets. QueueHandler is the stdlib answer to exactly
    # that.
    _evict_sink = logging.StreamHandler(sys.stderr)
    _evict_sink.setFormatter(
        logging.Formatter("%(asctime)s ai-tools-mcp %(levelname)s %(message)s")
    )
    _evict_listener = QueueListener(queue.SimpleQueue(), _evict_sink)
    _evict_listener.start()  # daemon thread, so it never holds up exit
    # ...but a daemon thread is killed mid-queue at exit, so drain on the
    # way out rather than losing the last records.
    atexit.register(_evict_listener.stop)
    _evict_handler = QueueHandler(_evict_listener.queue)
    # The conventional slot (logging.config.dictConfig sets the same one).
    # It is how a reader — and the test that proves the write is off the
    # event loop — gets from the logger to the handler that really writes.
    _evict_handler.listener = _evict_listener
    _evict_log.addHandler(_evict_handler)

_evict_stats = {"ok": 0, "absent": 0, "failed": 0}


class _EvictOutcome(NamedTuple):
    """What one eviction attempt achieved.

    `reason` == "" means the runner was provably dropped; anything else is a
    short, server-authored description of why the protection did not run.
    `benign` marks the reasons that are an expected operating condition
    rather than a broken mitigation, so the operator's log does not cry wolf
    about them.
    """

    reason: str = ""
    benign: bool = False


def _record_eviction_outcome(
    outcome: _EvictOutcome, *, model: str, endpoint: str, elapsed_ms: int
) -> None:
    """Count and log one eviction attempt.

    Written at eviction time, not at return time: if the chat then fails,
    _post_ollama_chat returns an error envelope that renders as a bare
    error, and a background job that is never collected is evicted by
    _DELEGATE_DONE_RETAINED and never rendered at all. In both cases this
    line is the only surviving record.

    `ollama-preunload` is a stable grep anchor — tests assert the literal
    token, so a rename fails loudly rather than silently orphaning the
    operator's search. Kept to ONE line: a newline would fragment the
    host's per-line JSON record, hence %r on the one remote-influenced
    field. Emitting is non-blocking by construction (see the QueueHandler
    above); this runs on the event loop.
    """
    where = (
        "localhost" if _is_localhost_endpoint(endpoint) else redact_secrets(endpoint)
    )
    if not outcome.reason:
        _evict_stats["ok"] += 1
        result, level = "OK", logging.INFO
    elif outcome.benign:
        # Not a failure of anything: there was no runner to drop. WARNING
        # here would fire on every call to a tag this host never pulled.
        _evict_stats["absent"] += 1
        result, level = "NO-RUNNER", logging.INFO
    else:
        _evict_stats["failed"] += 1
        result, level = "FAILED", logging.WARNING
    _evict_log.log(
        level,
        "ollama-preunload result=%s reason=%r model=%s endpoint=%s "
        "elapsed_ms=%d ok_total=%d absent_total=%d fail_total=%d",
        result,
        outcome.reason,
        redact_secrets(model),
        where,
        elapsed_ms,
        _evict_stats["ok"],
        _evict_stats["absent"],
        _evict_stats["failed"],
    )


async def _evict_ollama_runner(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout_s: float,
) -> _EvictOutcome:
    """Best-effort: drop `model`'s resident runner before a protected call.

    `keep_alive` is a POST-*response* TTL, so `keep_alive:"0"` does not
    protect the request that carries it: a protected call landing on an
    already-resident, already-dirty qwen runner is exposed to exactly the
    cross-task contamination that default exists to prevent. Measured on
    JVMBPro 2026-08-08 (Ollama 0.32.6, q8_0 KV, OLLAMA_KEEP_ALIVE=-1) —
    see the benchmark note in _KEEP_ALIVE_ZERO_MODEL_PREFIXES' comment.
    Evicting first is what actually puts the call on a fresh runner.

    Empty prompt + keep_alive 0 is Ollama's documented unload idiom, and on
    a PULLED-but-not-resident tag it is a measured 22 ms no-op that returns
    done_reason:"unload" without loading anything. An UN-PULLED tag is a
    different answer entirely: HTTP 404 {"error":"model ... not found"},
    measured twice on Ollama 0.32.7 — which is why 404 is classified below
    as "nothing was resident" rather than as a broken mitigation. So in the
    steady state — where the qwen keep_alive default already unloads after
    every call — this costs a round trip and no reload. A reload (~2.5 s
    warm page cache) is paid only when another client had warmed the model,
    which is exactly the exposed case.

    Concurrency (measured 2026-08-10, Ollama 0.32.7): an eviction does NOT
    abort another call's in-flight generation — that completes normally —
    but the eviction is not queued behind the busy runner either: it lands
    immediately, rewriting the runner's expiry, and it destroys any warm pin
    another caller held on the SAME tag (a different tag is untouched).
    Overlapping protected jobs are NOT protected against each other: job B's
    eviction can fire before job A has even loaded the runner, so B's chat
    then lands on A's dirty runner. Not a regression — before the eviction
    existed, overlapping qwen jobs shared a dirty runner with no mitigation
    at all — and a cut-off eviction now at least surfaces as a non-empty
    outcome reason instead of silence.

    Bounded by `timeout_s` in WALL CLOCK, via asyncio.wait_for. httpx's
    float timeout is per-phase — connect, write, read and pool each get that
    value — so passing it alone would let a pathological eviction overshoot
    and erode the caller's total budget that the pre_unload path reserves
    against (CodeRabbit MINOR on 4a46cc2). The httpx timeout is kept as the
    inner per-phase bound; wait_for is the outer total.

    Never raises. A failed OR timed-out eviction leaves exactly today's
    behaviour; it must not turn into a failure of the caller's real request.

    Returns an _EvictOutcome whose `reason` is "" when the runner was
    provably dropped, else a SHORT description of why the protection did not
    run. Silence used to mean "success" here: the call checked neither
    status nor body, so a 403 from a stale Access token, a 404, a 5xx, or a
    200 that merely declined to unload all looked identical to a real
    eviction — a permanently broken mitigation would have been undetectable
    (demonstrated live: a host answering 403 on /api/generate and 200 on
    /api/chat returned a clean, unprotected answer). A real eviction answers
    200 with done_reason "unload", so that is the success condition. The
    reason string is server-authored and carries no header values, no
    response body and no exception text — only a status code, an exception
    CLASS name, and a length-capped done_reason.
    """

    async def _attempt() -> Any:
        # raise_for_status() and .json() live INSIDE the wait_for bound with
        # the POST they belong to, so the budget covers the whole attempt
        # rather than stopping at the last await.
        response = await client.post(
            f"{endpoint}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            headers=headers,
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()

    try:
        body = await asyncio.wait_for(_attempt(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return _EvictOutcome(f"timed out after {timeout_s:.1f}s")
    except httpx.HTTPStatusError as exc:
        # Read the status defensively: httpx builds these with a response,
        # but nothing here guarantees the instance this arm catches has one,
        # and an AttributeError would escape a helper whose entire contract
        # is that it never raises.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            # An un-pulled tag, measured on Ollama 0.32.7. Nothing was
            # resident, so nothing could be contaminated, and the chat that
            # follows fails on its own 404 with the `ollama pull` hint. Say
            # what it is instead of dressing it up as a broken mitigation —
            # but still report it, because a proxy that 404s /api/generate
            # while serving /api/chat leaves a real answer unprotected.
            return _EvictOutcome(
                "model not pulled on this endpoint (HTTP 404)", benign=True
            )
        if status is None:
            return _EvictOutcome("endpoint rejected the request (no HTTP status)")
        return _EvictOutcome(f"endpoint answered HTTP {status}")
    except httpx.RequestError as exc:
        return _EvictOutcome(f"request failed ({type(exc).__name__})")
    except ValueError:
        return _EvictOutcome("endpoint returned a non-JSON body")
    if not isinstance(body, dict):
        # No Ollama version answers /api/generate with a JSON array or
        # scalar, but a proxy in front of one can; .get() on it would raise
        # straight out of the never-raises helper.
        return _EvictOutcome("endpoint returned a non-object JSON body")
    done_reason = body.get("done_reason")
    if done_reason != "unload":
        # done_reason is the ONLY remote-controlled fragment that reaches
        # the log or the answer, so it gets the same value-aware scrub
        # _post_ollama_chat's HTTPStatusError arm uses: this file documents
        # at _http_error_payload that redact_secrets has no CF-token
        # pattern, but we hold the exact header values. Scrubbed BEFORE the
        # 40-char truncation so a secret straddling the cutoff cannot leave
        # an un-scrubbed fragment. Status codes, exception class names and
        # the timeout float are server-authored and need no scrub.
        shown = str(done_reason)
        for secret in headers.values():
            if secret:
                shown = shown.replace(secret, "[REDACTED_CF_ACCESS]")
        return _EvictOutcome(
            f"runner not unloaded (done_reason={redact_secrets(shown)[:40]!r})"
        )
    return _EvictOutcome()


async def _post_ollama_chat(
    payload: dict[str, Any], timeout_s: float, pre_unload: bool = False
) -> tuple[dict[str, Any], str]:
    """POST /api/chat to the first chain endpoint serving payload['model'].

    Same structured-error contract as _post_agent_research: selection,
    network, HTTP, and parse failures return {"status": "failed", ...}
    instead of raising. Remote https endpoints get Cloudflare Access
    service-token headers (absent creds → skipped at selection; the None
    check here is defense in depth). No retries: an endpoint is either
    serving or not.

    Returns (body, evict_warning). The warning travels BESIDE the body, not
    inside it: the body is the upstream host's own JSON, so anything read
    back out of it is attacker-authored by construction — a host that put
    an "_evict_warning" key in its /api/chat response could otherwise forge
    the harness-authored banner _render_delegate_answer prints, on any call,
    protected or not. This second element is the only eviction signal a
    renderer may believe, and nothing upstream can reach it. It is also why
    the body is never mutated on the way through: an in-place annotation
    would be visible to every later reader of that dict.
    """
    model = str(payload.get("model", ""))
    try:
        endpoint = await _select_ollama_endpoint(model)
    except ValueError as exc:
        return {"status": "failed", "error": redact_secrets(str(exc))}, ""
    headers = await asyncio.to_thread(_ollama_auth_headers, endpoint)
    if headers is None:
        return {
            "status": "failed",
            "error": f"no credentials for {redact_secrets(endpoint)}",
        }, ""
    client = await _get_http_client()
    # Non-empty only when a protected call ran UNPROTECTED. Returned to the
    # caller below so a silently-failing mitigation is visible instead of
    # indistinguishable from a working one.
    evict_warning = ""
    # Here, not at the call site: this is the one place that has resolved
    # the endpoint, so the eviction and the chat provably hit the SAME host
    # (no second resolution that could drift), and both the sync and the
    # background delegate paths route through it.
    if pre_unload:
        # timeout_s is the documented ceiling for the WHOLE delegate call, so
        # the chat's minimum slice is reserved up front rather than restored
        # afterwards. Restoring a floor after the fact hands back budget the
        # eviction already spent, which lets the total reach ~2x the caller's
        # ceiling on small timeouts (CodeRabbit MAJOR, reproduced at
        # timeout_s=1). Reserving first makes the sum bounded by construction.
        chat_reserve = min(_OLLAMA_MIN_CHAT_TIMEOUT_S, timeout_s)
        evict_budget = min(_OLLAMA_EVICT_TIMEOUT_S, timeout_s - chat_reserve)
        # A budget too small to afford an eviction skips it and runs
        # unprotected rather than overrunning. Moot in practice: no qwen tag
        # completes a generation in the seconds that implies.
        if evict_budget > 0:
            started = time.monotonic()
            outcome = await _evict_ollama_runner(
                client, endpoint, model, headers, evict_budget
            )
            spent = time.monotonic() - started
            # Recorded HERE rather than at return: the banner is missable in
            # two real cases (the chat fails, so the answer is a bare error;
            # or a background job is never collected and never rendered at
            # all). This stderr line is the only record that survives all
            # four outcome paths.
            _record_eviction_outcome(
                outcome,
                model=model,
                endpoint=endpoint,
                elapsed_ms=int(spent * 1000),
            )
            evict_warning = outcome.reason
            timeout_s = max(chat_reserve, timeout_s - spent)
        else:
            skipped = _EvictOutcome("skipped: timeout_s too small to afford it")
            _record_eviction_outcome(
                skipped, model=model, endpoint=endpoint, elapsed_ms=0
            )
            evict_warning = skipped.reason
    try:
        response = await client.post(
            f"{endpoint}/api/chat", json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        return response.json(), evict_warning
    except httpx.HTTPStatusError as exc:
        # Value-aware scrub: an Access-gated host's error body can echo
        # request headers; redact_secrets has no CF-token pattern, but we
        # hold the exact header values, so scrub them precisely — before
        # _http_error_payload truncates the body, so a secret straddling
        # the truncation cutoff can't leave an un-scrubbed fragment.
        failure = _http_error_payload(exc, scrub=tuple(headers.values()))
        if exc.response.status_code == 404:
            failure["error"] += (
                f" — model may not be pulled on this host; try: ollama pull {model}"
            )
        return failure, evict_warning
    except httpx.ConnectError:
        # Answered the probe moments ago but refused the POST — drop the
        # cached resolution so the next call re-probes the chain. The
        # implicit-resolution cache pins (model, endpoint) pairs too, so it
        # must go as well or omitted-model calls would keep chasing the dead
        # endpoint for up to a TTL (Codex P2 on PR #32). It is a dict of at
        # most one entry per configured default — clearing wholesale is
        # cheaper than matching endpoints and can never be wrong.
        _ollama_endpoint_cache.pop(model, None)
        _implicit_resolution_cache.clear()
        return {
            "status": "failed",
            "error": (
                f"Ollama not running at {redact_secrets(endpoint)} — is the "
                "LaunchAgent up? "
                "(launchctl kickstart -k gui/$UID/com.jasonvassallo.ollama)"
            ),
        }, evict_warning
    except httpx.RequestError as exc:
        return {
            "status": "failed",
            "error": f"request error: {redact_secrets(str(exc))}",
        }, evict_warning
    except ValueError as exc:
        return {
            "status": "failed",
            "error": f"invalid JSON from Ollama: {redact_secrets(str(exc))}",
        }, evict_warning


def _render_delegate_answer(
    data: dict[str, Any], prefix: str = "", evict_warning: str = ""
) -> list[TextContent]:
    """Render an Ollama /api/chat response (or failure envelope) as MCP text.

    message.thinking is deliberately discarded — the caller needs the
    answer, not the model's scratchpad. Output passes through
    redact_secrets for the same never-emit-secret-shapes contract as
    every other family.

    `prefix` carries caller advisories (e.g. a dropped `think` flag). It is
    server-authored, never model output, so it is emitted verbatim ahead of
    the answer — and deliberately not prepended to error returns, which have
    their own shape callers may match on.

    `evict_warning` is the same kind of thing: server-authored, handed in by
    the caller from _post_ollama_chat's out-of-band return. It is NOT read
    out of `data`. `data` is the upstream host's own JSON, so a key looked
    up in it is a key the host can set — and a warning banner an upstream
    host can author is worse than no banner at all, because it is the one
    piece of this output a reader is meant to trust as the harness's own
    voice. One surfacing point covers both delegate paths: the sync call
    renders here, and a background job's collected result renders here too.
    """
    if data.get("status") == "failed":
        return [
            TextContent(
                type="text",
                text=f"Error: {redact_secrets(str(data.get('error', 'unknown failure')))}",
            )
        ]
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return [TextContent(type="text", text="Error: Ollama returned no content")]
    model = redact_secrets(str(data.get("model", "")))
    banner = ""
    if evict_warning:
        # Quantified and actionable on purpose: a bare "protection failed"
        # does not change how the reading model treats the answer, whereas
        # the measured rate plus two concrete checks does.
        banner = (
            "> Warning: the qwen contamination pre-unload did NOT run "
            f"({redact_secrets(evict_warning)}).\n"
            "> This answer may be a different prompt's output — measured at "
            "~20-25% on repeat calls against a stale qwen runner.\n"
            "> Check it actually answers the prompt, and re-run to compare.\n\n"
        )
    return [
        TextContent(
            type="text",
            text=(
                f"{prefix}{banner}## Local Delegate ({model})\n\n"
                f"{redact_secrets(content)}"
            ),
        )
    ]


# In-memory only, deliberately: delegated input may be exactly the
# sensitive text kept off cloud APIs — it does not belong on disk. Jobs
# die with the MCP server process; the polling Claude session dies with
# it too, so nothing durable is lost.
_delegate_jobs: dict[str, dict[str, Any]] = {}


def _start_delegate_job(payload: dict[str, Any], pre_unload: bool = False) -> str:
    """Launch a background delegate call; return its job id.

    Raises ValueError when _DELEGATE_JOB_CAP jobs are already running —
    the 32 GB host runs num_parallel=1, so queuing more would lie to the
    caller; failing fast is honest.
    """
    # Bound the never-collected tail: a caller that starts jobs and never
    # calls local_delegate_result would otherwise leak one dict entry per
    # job forever on a long-lived server. Retain only the newest
    # _DELEGATE_DONE_RETAINED completed jobs, evicting older ones
    # oldest-first (by "started"). This does not change single-collect
    # semantics — anything actually retrieved via local_delegate_result is
    # deleted there as before; eviction here only reclaims memory for
    # completed jobs nobody ever asks for.
    done_ids = sorted(
        (jid for jid, job in _delegate_jobs.items() if job["task"].done()),
        key=lambda jid: _delegate_jobs[jid]["started"],
    )
    excess = len(done_ids) - _DELEGATE_DONE_RETAINED
    if excess > 0:
        for jid in done_ids[:excess]:
            task = _delegate_jobs.pop(jid)["task"]
            try:
                # task.exception() marks the exception (if any) as
                # retrieved, so asyncio never logs "exception was never
                # retrieved" for an evicted-but-failed job. It raises
                # CancelledError instead of returning for a cancelled
                # task — swallow that the same way.
                task.exception()
            except asyncio.CancelledError:
                pass

    running = sum(1 for job in _delegate_jobs.values() if not job["task"].done())
    if running >= _DELEGATE_JOB_CAP:
        raise ValueError(
            f"Delegate job cap ({_DELEGATE_JOB_CAP}) reached — collect finished "
            "jobs via local_delegate_result or wait for one to complete."
        )
    job_id = uuid.uuid4().hex
    coro = asyncio.wait_for(
        _post_ollama_chat(payload, _DELEGATE_BG_CEILING_S, pre_unload=pre_unload),
        timeout=_DELEGATE_BG_CEILING_S,
    )
    _delegate_jobs[job_id] = {
        "task": asyncio.get_running_loop().create_task(coro),
        "started": time.monotonic(),
    }
    return job_id


def _collect_delegate_job(job_id: str | None) -> tuple[dict[str, Any], str]:
    """Poll/collect a background job. Completed jobs are single-collect:
    the registry entry is deleted on retrieval so memory stays clean.

    Returns (outcome, evict_warning), passing through the pair
    _post_ollama_chat returned inside the task. The warning has to ride
    alongside rather than inside the outcome for the same reason it does
    there: the outcome of a completed job IS the upstream host's JSON body.
    Envelopes this function authors itself carry no warning — no eviction
    result is known for a job that is still running or was cancelled.
    """
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
        }, ""
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
        }, ""


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
            name="local_delegate",
            description=(
                "Delegate a task to a LOCAL Ollama model — "
                "input text never leaves the machine (unlike every research "
                "tool here, which calls hosted APIs). Use for: private/"
                "sensitive text that must stay on-device; cheap mechanical "
                "work (summaries, boilerplate, drafts, bulk transforms) that "
                "doesn't need frontier quality; an independent second opinion "
                "on code or text; or long background jobs (pass "
                "background=true, poll local_delegate_result). No web access "
                "— for research use the research tools instead. The model is "
                "strong at code and structured transforms but far below "
                "frontier models on hard reasoning: keep tasks well-scoped."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The task. Include any needed file content inline — "
                            "the server never reads the filesystem."
                        ),
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt framing the task.",
                    },
                    "model": {
                        "type": "string",
                        "enum": list(OLLAMA_DELEGATE_MODELS),
                        "default": _delegate_default_model(),
                        "description": (
                            "Server-side allowlist; the authoritative default "
                            "is the `default` field above (it follows the "
                            "allowlist's first entry, which AI_TOOLS_OLLAMA_"
                            "MODELS can override per machine). Out of the box "
                            "that is gemma4:12b-nvfp4 — it outscored the (since-"
                            "retired) qwen3.6 tags on mechanical delegate work "
                            "(0.92 vs 0.73) and is the safer pick for short "
                            "repeated prompts; qwen3.8:27b-nvfp4 is unbenchmarked. Prefer "
                            "gemma4:31b-nvfp4 for review and long-context code "
                            "work (served by the MBP only). Neither is reliable "
                            "at counting or aggregating over long inputs. There "
                            "is no per-request context window: every call runs "
                            "at the serving host's OLLAMA_CONTEXT_LENGTH (64k on "
                            "JVMBPro, 32k on jvmacmini) — no -32k/-64k/-256k "
                            "tag variants exist any more. "
                            "Explicit tags resolve strictly: the endpoint "
                            "chain is probed per call and the first endpoint "
                            "serving the tag wins, else the call fails. "
                            "Omitting this field resolves local-first across "
                            "the allowlist instead — the first endpoint "
                            "serving ANY allowlisted tag picks the model, so "
                            "prompts stay on-device whenever a local option "
                            "exists and a missing default falls back with an "
                            "advisory note rather than failing."
                        ),
                    },
                    "think": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Enable the model's thinking mode. Off by default "
                            "for speed. KEEP IT OFF for the gemma4 tags "
                            "(including the gemma4:31b-nvfp4 reviewer): with "
                            "thinking on they put the generation in "
                            "message.thinking, which this tool discards, so the "
                            "call returns 'Error: Ollama returned no content'. "
                            "Enable it only on a model whose content you have "
                            "confirmed survives it. Every built-in allowlist tag "
                            "reports the 'thinking' capability; if an overridden "
                            "tag does not, the server disables the flag and "
                            "prefixes an advisory instead of letting Ollama "
                            "reject the call. Qwen thinking can likewise consume "
                            "the whole output budget on large inputs and return "
                            "no answer at all."
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "true: return a job_id immediately; poll "
                            "local_delegate_result. false: wait for the answer."
                        ),
                    },
                    "keep_alive": {
                        "type": "string",
                        "description": (
                            "Optional: how long Ollama keeps the model loaded "
                            "after the call ('0' = unload immediately — use "
                            "after a big one-off job). Omitted: qwen tags "
                            "default to '0'. A '0' on a qwen tag (defaulted "
                            "OR explicit) also EVICTS the runner before the "
                            "call — the contamination mitigation; the TTL "
                            "alone cannot protect the request carrying it. A "
                            "non-zero value keeps the model warm and skips "
                            "that protection. Other models inherit the "
                            "server's OLLAMA_KEEP_ALIVE. Pattern: 0 or "
                            "<1-9999><s|m|h>."
                        ),
                    },
                    "timeout_s": {
                        "type": "integer",
                        "default": 300,
                        "description": (
                            "Sync timeout in seconds (1-600). The qwen "
                            "contamination pre-unload is wall-clock bounded and "
                            "deducted from this budget; at 5s or less it is "
                            "skipped rather than allowed to eat it, so such a "
                            "call runs unprotected. Note httpx applies the "
                            "remainder per network phase, so this is a phase "
                            "bound rather than a hard total."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="local_delegate_result",
            description=(
                "Poll/collect a background local_delegate job by job_id. "
                "Returns running status with elapsed seconds, or the answer. "
                "Results are single-collect: once retrieved the job is gone. "
                "Jobs live in server memory only and do not survive restarts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The 32-hex job id returned by local_delegate.",
                    },
                },
                "required": ["job_id"],
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
                raise TypeError(
                    "max_output_tokens must be an integer in "
                    f"[{_AGENT_MAX_OUTPUT_TOKENS_MIN}, "
                    f"{_AGENT_MAX_OUTPUT_TOKENS_MAX}]; got {max_output_tokens!r}."
                )

            # Strict bool check — `bool("false")` is True in Python, so a
            # JSON-stringified flag would silently flip the meaning (same
            # contract as collaborative_planning on gemini_deep_research_start).
            background = arguments.get("background", False)
            if not isinstance(background, bool):
                raise TypeError(
                    "background must be a JSON boolean (true/false), "
                    "not a string or number."
                )
        except (TypeError, ValueError) as exc:
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
                raise TypeError(
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
        except (TypeError, ValueError) as exc:
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

    if name == "local_delegate":
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return [
                TextContent(
                    type="text",
                    text="Error: prompt is required and must be a non-empty string.",
                )
            ]
        model_arg = arguments.get("model")
        if model_arg is not None and model_arg not in OLLAMA_DELEGATE_MODELS:
            allowed = ", ".join(OLLAMA_DELEGATE_MODELS)
            return [
                TextContent(type="text", text=f"Error: model must be one of: {allowed}")
            ]
        model = model_arg if model_arg is not None else _delegate_default_model()
        think = arguments.get("think", False)
        if not isinstance(think, bool):
            return [
                TextContent(type="text", text="Error: think must be a JSON boolean.")
            ]
        background = arguments.get("background", False)
        if not isinstance(background, bool):
            return [
                TextContent(
                    type="text", text="Error: background must be a JSON boolean."
                )
            ]
        keep_alive = arguments.get("keep_alive")
        if keep_alive is not None and (
            not isinstance(keep_alive, str)
            or not _DELEGATE_KEEP_ALIVE_RE.fullmatch(keep_alive)
        ):
            return [
                TextContent(
                    type="text",
                    text="Error: keep_alive must match 0 or <1-9999><s|m|h> (e.g. '5m', '0').",
                )
            ]
        timeout_s = arguments.get("timeout_s", _DELEGATE_TIMEOUT_DEFAULT_S)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int)
            or not 1 <= timeout_s <= _DELEGATE_TIMEOUT_MAX_S
        ):
            return [
                TextContent(
                    type="text",
                    text=(
                        "Error: timeout_s must be an integer between 1 and "
                        f"{_DELEGATE_TIMEOUT_MAX_S}."
                    ),
                )
            ]
        system = arguments.get("system")
        if system is not None and not isinstance(system, str):
            return [TextContent(type="text", text="Error: system must be a string.")]

        # Implicit-model calls resolve local-first across the whole allowlist
        # (see _resolve_implicit_model); explicit model= keeps strict
        # fail-loud semantics and resolves at send time as before.
        implicit_note = ""
        endpoint_hint: str | None = None
        if model_arg is None:
            try:
                model, endpoint_hint, implicit_note = await _resolve_implicit_model()
            except ValueError as exc:
                return [
                    TextContent(type="text", text=f"Error: {redact_secrets(str(exc))}")
                ]

        # think=true on a model without the 'thinking' capability draws a hard
        # 400 from current Ollama. Ask /api/show instead of hardcoding model
        # families (an earlier hardcoded list wrongly claimed gemma4 cannot
        # think): strip the flag and say so, keeping the call useful. An
        # indeterminate capability read changes nothing — fail neutral, let
        # Ollama speak for itself.
        think_note = ""
        if think:
            endpoint_for_caps = endpoint_hint
            if endpoint_for_caps is None:
                try:
                    endpoint_for_caps = await _select_ollama_endpoint(model)
                except ValueError:
                    endpoint_for_caps = None  # send path will surface the real error
            caps = (
                await _model_capabilities(endpoint_for_caps, model)
                if endpoint_for_caps is not None
                else None
            )
            if caps is not None and "thinking" not in caps:
                think = False
                think_note = (
                    f"Note: think=true was disabled — {model} does not report "
                    "the 'thinking' capability. Choose a tag that does (every "
                    "built-in allowlist tag reports it) if you need reasoning.\n\n"
                )
        advisory = implicit_note + think_note

        # After final model resolution (implicit calls may have re-resolved
        # `model` above): qwen tags default to keep_alive "0" — the proven
        # repeat-call contamination mitigation. Explicit caller values win.
        #
        # The pre-unload follows the EFFECTIVE zero TTL, not how it was
        # chosen. Gating it on "we applied the default" left the repo's own
        # documented route unprotected: commands/local-delegate.md tells
        # callers to pass keep_alive="0" for long-context qwen work, so the
        # most likely qwen caller opted itself out of the very protection
        # this exists to provide (Codex P1 + Gemini, PR #65, both confirmed
        # against that file). An explicit NON-zero value still opts out:
        # YOUR value is sent unchanged and YOUR call is not pre-evicted.
        # That is a statement about this request only, not a residency
        # guarantee — any other caller's "0" on the same tag still unloads
        # it (true since PR #60's post-response default; the eviction just
        # moves the same collision earlier).
        if keep_alive is None and _keep_alive_zero_default_applies(model):
            keep_alive = "0"
        pre_unload = keep_alive == "0" and _keep_alive_zero_default_applies(model)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "think": think,
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        if background:
            try:
                job_id = _start_delegate_job(payload, pre_unload=pre_unload)
            except ValueError as exc:
                return [TextContent(type="text", text=f"Error: {exc}")]
            # Carried as a JSON field, NOT prepended: this envelope is
            # json.loads()-ed by callers, so a bare prefix would break parsing.
            envelope = {"job_id": job_id, "status": "started"}
            if advisory:
                envelope["warning"] = advisory.strip()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(envelope),
                )
            ]

        data, evict_warning = await _post_ollama_chat(
            payload, float(timeout_s), pre_unload=pre_unload
        )
        return _render_delegate_answer(
            data, prefix=advisory, evict_warning=evict_warning
        )

    if name == "local_delegate_result":
        try:
            outcome, evict_warning = _collect_delegate_job(arguments.get("job_id"))
        except ValueError as exc:
            return [TextContent(type="text", text=f"Error: {exc}")]
        if outcome.get("status") == "running":
            return [TextContent(type="text", text=json.dumps(outcome))]
        return _render_delegate_answer(outcome, evict_warning=evict_warning)

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


if "--check" in sys.argv:
    run_check()


if __name__ == "__main__":
    asyncio.run(main())
