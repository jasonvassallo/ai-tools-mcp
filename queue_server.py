#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cryptography>=48.0.1",
# ]
# ///
"""Durable submit/poll queue for local Ollama delegation.

Standalone LaunchAgent service (NOT part of the MCP server process): it
binds a small HTTP JSON API on loopback, persists submitted Ollama
``/api/chat`` payloads to SQLite, and works them off against the local
Ollama server with bounded concurrency. Remote access happens only at
the edge, via a Cloudflare-Access-gated tunnel hostname that forwards to
this loopback port — transport auth is deliberately NOT this service's
concern; loopback callers are trusted.

Durability is the whole point: unlike the MCP server's in-memory job
store (jobs die with the process, results are single-collect), jobs and
results here survive restarts and results persist until a TTL sweep
removes them. That reverses the repo's historical "no delegated jobs on
disk" rule — approved by the user 2026-08-07 with encryption at rest as
the mitigation (see docs/superpowers/specs/2026-08-07-durable-delegate-
queue-design.md).

Encryption at rest: job payloads and results are AES-256-GCM blobs with
a fresh random nonce per encryption; only operational metadata (job id,
status, timestamps, model name, attempt count, error class) is stored
as plaintext columns. The 32-byte key is read once at startup from the
macOS SYSTEM keychain (generic password, service ``DELEGATE_QUEUE_KEY``,
account ``jasonvassallo``, value base64) — the System keychain because
this runs headless and the login keychain is GUI-only on these machines.
FAIL CLOSED: the service refuses to start without a valid key, and never
logs key material or payload/result content — log lines carry job ids,
statuses, and error classes only.

API (all JSON):
- ``POST /v1/jobs``          submit a validated Ollama chat payload
                             (≤ ~2 MB) → ``{"job_id": "q<uuid4hex>",
                             "status": "queued"}``. The ``q`` prefix
                             distinguishes durable queue ids from the
                             legacy 32-hex in-memory ids.
- ``GET /v1/jobs/<id>``      status: queued|running|done|failed|
                             cancelled, plus elapsed seconds/attempts.
- ``GET /v1/jobs/<id>/result``  the result envelope. Does NOT delete —
                             results persist until the TTL sweep (72 h
                             after finishing), the deliberate contrast
                             with the legacy single-collect store.
- ``DELETE /v1/jobs/<id>``   cancel a queued/running job.
- ``GET /healthz``           liveness + queue depth.

Worker: 2 concurrent jobs max, POSTing to the upstream Ollama
``/api/chat`` with an 1800 s per-job ceiling. On startup any job found
``running`` reverts to ``queued`` (crash recovery); each claim bumps an
attempts counter and a job that has already burned 2 attempts fails
instead of requeueing. Finished jobs are purged 72 h after completion
(sweep at startup and on a timer).

Concurrency note: the approved design sketch said "async loop"; this
implementation uses a small thread pool over the same SQLite store
because the HTTP layer is stdlib ``http.server`` (threads) and stdlib
asyncio has no HTTP server — semantics (max 2 in flight, ceiling,
attempts, recovery) are identical.

Config (env): ``QUEUE_PORT`` (default 11438), ``QUEUE_UPSTREAM``
(default http://127.0.0.1:11434), ``QUEUE_DB`` (default
~/.local/state/delegate-queue/queue.db; test override).
"""

from __future__ import annotations

import base64
import contextlib
import getpass
import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("delegate-queue")

DEFAULT_PORT = 11438
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_DB = Path.home() / ".local" / "state" / "delegate-queue" / "queue.db"

KEY_SERVICE = "DELEGATE_QUEUE_KEY"
KEY_ACCOUNT = "jasonvassallo"

MAX_BODY_BYTES = 2 * 1024 * 1024  # ~2 MB submit cap
# Backpressure: submits beyond this many already-queued jobs draw a 429
# instead of growing the backlog without bound. Queued jobs are exempt
# from the TTL sweep, so the cap is what bounds on-disk backlog growth.
MAX_QUEUED = 64
MAX_ATTEMPTS = 2
JOB_TIMEOUT_S = 1800.0
TTL_S = 72 * 3600.0
SWEEP_INTERVAL_S = 3600.0
WORKER_CONCURRENCY = 2
WORKER_POLL_S = 0.5

_JOB_ID_RE = re.compile(r"^q[0-9a-f]{32}$")
# Mirrors the MCP server's keep_alive validation so a malformed value
# cannot smuggle arbitrary JSON into the upstream request.
_KEEP_ALIVE_RE = re.compile(r"^(0|[1-9][0-9]{0,3}(s|m|h))$")
# Top-level payload keys forwarded to Ollama. Anything else is rejected
# at submit time — the queue only ever replays validated chat payloads.
_ALLOWED_PAYLOAD_KEYS = frozenset(
    {"model", "messages", "think", "stream", "keep_alive", "options"}
)

_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


class QueueError(ValueError):
    """Client-caused error carrying an HTTP status code."""

    def __init__(self, http_status: int, message: str):
        super().__init__(message)
        self.http_status = http_status


def load_key_from_system_keychain() -> bytes:
    """Read the AES-256 key from the System keychain. FAIL CLOSED.

    ``security find-generic-password -s DELEGATE_QUEUE_KEY -a
    jasonvassallo -w`` must return the base64 of exactly 32 bytes. Any
    miss, decode failure, or wrong length raises — the service must not
    start with a missing or malformed key, and neither the raised error
    nor any log line ever contains key material.
    """
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                KEY_SERVICE,
                "-a",
                KEY_ACCOUNT,
                "-w",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "security(1) not found — this service requires the macOS "
            "System keychain for its encryption key."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Encryption key not found in the System keychain (service "
            f"{KEY_SERVICE!r}, account {KEY_ACCOUNT!r}). Provision it per "
            "deploy/jvmbpro-delegate-queue/DEPLOY.md; refusing to start."
        )
    raw = result.stdout.strip()
    try:
        key = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Keychain item {KEY_SERVICE!r} is not valid base64; refusing to start."
        ) from exc
    if len(key) != 32:
        raise RuntimeError(
            f"Keychain item {KEY_SERVICE!r} decodes to {len(key)} bytes, "
            "expected 32 (AES-256); refusing to start."
        )
    return key


def encrypt_blob(key: bytes, data: dict[str, Any]) -> tuple[bytes, bytes]:
    """AES-256-GCM encrypt a JSON-serializable dict → (nonce, ciphertext).

    A fresh 96-bit random nonce per call — never reused, stored alongside
    the ciphertext.
    """
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(data, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce, ciphertext


def decrypt_blob(key: bytes, nonce: bytes, ciphertext: bytes) -> dict[str, Any]:
    """Decrypt and parse a blob written by ``encrypt_blob``.

    Raises ValueError on tampering/decryption failure — deliberately
    without echoing any blob content.
    """
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise ValueError("stored blob failed authenticated decryption") from exc
    data = json.loads(plaintext.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004 - persisted-content validation is a value error
            "stored blob is not a JSON object"
        )
    return data


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate a submitted Ollama chat payload; return the sanitized copy.

    Fail closed on anything that is not a plain chat request: unknown
    top-level keys, missing/empty model, malformed messages, non-bool
    think, keep_alive not matching the strict pattern. ``stream`` is
    forced to False — the queue stores complete responses.
    """
    if not isinstance(payload, dict):
        raise QueueError(400, "payload must be a JSON object")
    unknown = set(payload) - _ALLOWED_PAYLOAD_KEYS
    if unknown:
        raise QueueError(400, f"unknown payload keys: {', '.join(sorted(unknown))}")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise QueueError(400, "model must be a non-empty string")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise QueueError(400, "messages must be a non-empty list")
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise QueueError(400, "each message must be {role: str, content: str}")
    think = payload.get("think", False)
    if not isinstance(think, bool):
        raise QueueError(400, "think must be a boolean")
    keep_alive = payload.get("keep_alive")
    if keep_alive is not None and (
        not isinstance(keep_alive, str) or not _KEEP_ALIVE_RE.fullmatch(keep_alive)
    ):
        raise QueueError(400, "keep_alive must match 0 or <1-9999><s|m|h>")
    options = payload.get("options")
    if options is not None and not isinstance(options, dict):
        raise QueueError(400, "options must be an object")
    sanitized = dict(payload)
    sanitized["stream"] = False
    return sanitized


class QueueStore:
    """SQLite-backed encrypted job store. Thread-safe via per-call
    connections; WAL mode so readers never block the worker's writes."""

    def __init__(self, db_path: Path | str, key: bytes):
        self.db_path = Path(db_path)
        self._key = key
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        # mode= closes the umask-sized window where a freshly created
        # directory is briefly group/world-readable; the chmod still
        # corrects pre-existing directories.
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.db_path.parent, 0o700)
        with contextlib.closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN
                        ('queued','running','done','failed','cancelled')),
                    model TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error_class TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    payload_nonce BLOB NOT NULL,
                    payload_ct BLOB NOT NULL,
                    result_nonce BLOB,
                    result_ct BLOB
                )
                """
            )
        self._tighten_file_modes()

    def _tighten_file_modes(self) -> None:
        # SQLite creates the db/-wal/-shm files with the process umask;
        # payload ciphertext still should not be world-readable.
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                with contextlib.suppress(OSError):
                    os.chmod(path, 0o600)

    # ── submit / read ────────────────────────────────────────────────

    def submit(self, payload: dict[str, Any]) -> str:
        sanitized = validate_payload(payload)
        job_id = "q" + uuid.uuid4().hex
        nonce, ciphertext = encrypt_blob(self._key, sanitized)
        with contextlib.closing(self._connect()) as conn:
            # BEGIN IMMEDIATE so the depth check and the insert are one
            # write transaction — two racing submits cannot both observe
            # MAX_QUEUED - 1 and overshoot the cap.
            conn.execute("BEGIN IMMEDIATE")
            try:
                (queued,) = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status='queued'"
                ).fetchone()
                if queued >= MAX_QUEUED:
                    raise QueueError(
                        429,
                        f"queue is full ({queued} jobs queued, cap "
                        f"{MAX_QUEUED}); retry later",
                    )
                conn.execute(
                    "INSERT INTO jobs (job_id, status, model, created_at,"
                    " payload_nonce, payload_ct) VALUES (?,?,?,?,?,?)",
                    (
                        job_id,
                        "queued",
                        str(sanitized["model"]),
                        time.time(),
                        nonce,
                        ciphertext,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        self._tighten_file_modes()
        log.info("job %s queued (model %s)", job_id, sanitized["model"])
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT status, model, attempts, error_class, created_at,"
                " started_at, finished_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        status, model, attempts, error_class, created, started, finished = row
        now = time.time()
        if status == "queued" or started is None:
            elapsed = now - created
        elif finished is None:
            elapsed = now - started
        else:
            elapsed = finished - started
        out: dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "model": model,
            "attempts": attempts,
            "elapsed_s": int(max(elapsed, 0)),
        }
        if error_class:
            out["error_class"] = error_class
        return out

    def result(self, job_id: str) -> dict[str, Any] | None:
        """Decrypted result envelope for a terminal job, or None.

        None means "no result yet" for a queued/running job; a missing
        job raises. Results are NOT deleted on read — they persist until
        the TTL sweep (the deliberate contrast with the in-memory
        store's single-collect contract).
        """
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT status, result_nonce, result_ct, error_class, attempts"
                " FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise QueueError(404, f"unknown job_id {job_id!r}")
        status, nonce, ciphertext, error_class, attempts = row
        if status == "cancelled":
            return {"status": "failed", "error": "job was cancelled"}
        if status not in _TERMINAL_STATUSES:
            return None
        if nonce is None or ciphertext is None:
            # Jobs failed administratively (attempts cap at claim time or
            # crash recovery) never got a result envelope written; say
            # what actually happened instead of a generic shrug.
            if error_class == "attempts_exhausted":
                error = (
                    f"job failed after exhausting {attempts} of "
                    f"{MAX_ATTEMPTS} attempts (crashed or was interrupted "
                    "each time before finishing)"
                )
            elif error_class:
                error = f"job finished without a result ({error_class})"
            else:
                error = "job finished without a result"
            return {"status": "failed", "error": error}
        return decrypt_blob(self._key, nonce, ciphertext)

    # ── cancel ───────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> str:
        """Cancel a queued/running job; returns the resulting status.

        A running job is marked cancelled immediately; the worker's
        in-flight upstream call cannot be aborted mid-request, but its
        eventual outcome is discarded (``finish`` refuses to overwrite a
        cancelled row). Cancelling an already-terminal job is a 409.

        BEGIN IMMEDIATE makes the status check and the update one write
        transaction; without it, ``finish`` could commit ``done`` between
        this method's SELECT and its UPDATE and the cancel would then
        overwrite a real result. The status guard on the UPDATE is
        belt-and-suspenders for the same race.
        """
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise QueueError(404, f"unknown job_id {job_id!r}")
                status = row[0]
                if status in _TERMINAL_STATUSES:
                    raise QueueError(409, f"job is already {status}")
                cur = conn.execute(
                    "UPDATE jobs SET status='cancelled', finished_at=?"
                    " WHERE job_id = ? AND status IN ('queued','running')",
                    (time.time(), job_id),
                )
                if cur.rowcount != 1:
                    raise QueueError(409, "job reached a terminal state first")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        log.info("job %s cancelled (was %s)", job_id, status)
        return "cancelled"

    # ── worker interface ─────────────────────────────────────────────

    def claim_next(self) -> tuple[str, dict[str, Any]] | None:
        """Atomically claim the oldest queued job → (job_id, payload).

        The claim bumps the attempts counter; a queued job that has
        already burned MAX_ATTEMPTS (crash-recovery replays) is failed
        instead of claimed.
        """
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT job_id, attempts, payload_nonce, payload_ct"
                    " FROM jobs WHERE status='queued'"
                    " ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                job_id, attempts, nonce, ciphertext = row
                if attempts >= MAX_ATTEMPTS:
                    conn.execute(
                        "UPDATE jobs SET status='failed',"
                        " error_class='attempts_exhausted', finished_at=?"
                        " WHERE job_id = ?",
                        (time.time(), job_id),
                    )
                    conn.execute("COMMIT")
                    log.warning("job %s failed: attempts exhausted", job_id)
                    return None
                conn.execute(
                    "UPDATE jobs SET status='running', attempts=attempts+1,"
                    " started_at=? WHERE job_id = ?",
                    (time.time(), job_id),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        try:
            payload = decrypt_blob(self._key, nonce, ciphertext)
        except ValueError:
            self.finish(
                job_id,
                {"status": "failed", "error": "stored payload unreadable"},
                error_class="payload_corrupt",
                failed=True,
            )
            return None
        return job_id, payload

    def finish(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        error_class: str | None = None,
        failed: bool = False,
    ) -> bool:
        """Record a job's outcome. Returns False when the row was
        cancelled (or removed) in the meantime — the outcome is then
        discarded, honoring the cancel."""
        nonce, ciphertext = encrypt_blob(self._key, result)
        status = "failed" if failed else "done"
        with contextlib.closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, error_class=?, finished_at=?,"
                " result_nonce=?, result_ct=? WHERE job_id=?"
                " AND status='running'",
                (status, error_class, time.time(), nonce, ciphertext, job_id),
            )
            recorded = cur.rowcount == 1
        if recorded:
            log.info(
                "job %s %s%s",
                job_id,
                status,
                f" ({error_class})" if error_class else "",
            )
        else:
            log.info("job %s outcome discarded (cancelled or gone)", job_id)
        return recorded

    # ── maintenance ──────────────────────────────────────────────────

    def recover_running(self) -> int:
        """Crash recovery: running → queued (or failed past the attempts
        cap). Called once at startup, before any worker begins."""
        with contextlib.closing(self._connect()) as conn, conn:
            failed = conn.execute(
                "UPDATE jobs SET status='failed',"
                " error_class='attempts_exhausted', finished_at=?"
                " WHERE status='running' AND attempts >= ?",
                (time.time(), MAX_ATTEMPTS),
            ).rowcount
            requeued = conn.execute(
                "UPDATE jobs SET status='queued', started_at=NULL"
                " WHERE status='running'"
            ).rowcount
        if failed or requeued:
            log.info(
                "crash recovery: %d requeued, %d failed (attempts cap)",
                requeued,
                failed,
            )
        return requeued

    def purge_expired(self, now: float | None = None) -> int:
        """Delete terminal jobs older than TTL_S past their finish."""
        cutoff = (now if now is not None else time.time()) - TTL_S
        with contextlib.closing(self._connect()) as conn, conn:
            purged = conn.execute(
                "DELETE FROM jobs WHERE status IN ('done','failed','cancelled')"
                " AND finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            ).rowcount
        if purged:
            log.info("ttl sweep: purged %d job(s)", purged)
        return purged

    def counts(self) -> dict[str, int]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall()
        return {status: count for status, count in rows}


class QueueWorker:
    """Thread pool working queued jobs against the upstream Ollama."""

    def __init__(
        self,
        store: QueueStore,
        upstream: str,
        *,
        concurrency: int = WORKER_CONCURRENCY,
        poll_interval_s: float = WORKER_POLL_S,
        job_timeout_s: float = JOB_TIMEOUT_S,
    ):
        self.store = store
        self.upstream = upstream.rstrip("/")
        self.concurrency = concurrency
        self.poll_interval_s = poll_interval_s
        self.job_timeout_s = job_timeout_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self.concurrency):
            thread = threading.Thread(
                target=self._run, name=f"queue-worker-{i}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)

    def _run(self) -> None:
        # The loop guard is the last line of defense: NO exception class
        # may terminate a worker thread (a dead worker silently strands
        # every future queued job). _work_one already converts job-level
        # failures into failed rows; anything that still escapes — a
        # store/SQLite error in claim_next, a finish() failure — is
        # logged and the loop backs off and continues.
        while not self._stop.is_set():
            try:
                claimed = self.store.claim_next()
                if claimed is None:
                    self._stop.wait(self.poll_interval_s)
                    continue
                job_id, payload = claimed
                self._work_one(job_id, payload)
            except Exception:  # noqa: BLE001 - worker must outlive any single job
                log.exception("worker iteration failed")
                self._stop.wait(self.poll_interval_s)

    def _work_one(self, job_id: str, payload: dict[str, Any]) -> None:
        # Error text deliberately never includes payload or response
        # content — only transport/HTTP metadata.
        request = urllib.request.Request(
            f"{self.upstream}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - operator-configured loopback upstream
                request, timeout=self.job_timeout_s
            ) as response:
                body = response.read()
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError(  # noqa: TRY004 - upstream-content validation is a value error
                    "upstream returned non-object JSON"
                )
        except urllib.error.HTTPError as exc:
            self.store.finish(
                job_id,
                {"status": "failed", "error": f"upstream HTTP {exc.code}"},
                error_class="http_error",
                failed=True,
            )
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            self.store.finish(
                job_id,
                {
                    "status": "failed",
                    "error": f"upstream unreachable or timed out: "
                    f"{type(reason).__name__}",
                },
                error_class="unreachable",
                failed=True,
            )
            return
        except ValueError:
            self.store.finish(
                job_id,
                {"status": "failed", "error": "invalid JSON from upstream"},
                error_class="bad_response",
                failed=True,
            )
            return
        except Exception as exc:  # noqa: BLE001 - see below
            # Terminal backstop: the chain above names the EXPECTED
            # failures, but urllib can raise outside it — e.g.
            # http.client.IncompleteRead (a truncated upstream response)
            # is an HTTPException, not an OSError — and any such escape
            # previously killed the worker thread for good. Mark the job
            # failed with the exception class (never its message, which
            # could echo response content) and keep the worker alive.
            self.store.finish(
                job_id,
                {
                    "status": "failed",
                    "error": f"unexpected worker error: {type(exc).__name__}",
                },
                error_class="worker_error",
                failed=True,
            )
            return
        self.store.finish(job_id, data)


def _make_handler(store: QueueStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "delegate-queue"
        protocol_version = "HTTP/1.1"
        # Finite per-connection socket timeout. StreamRequestHandler
        # applies this via settimeout() in setup(), and
        # handle_one_request() treats the resulting socket.timeout as
        # terminal — the connection closes instead of parking one of
        # ThreadingHTTPServer's threads forever behind a client that
        # stalls mid-request (reachable through the Access tunnel, so
        # not a purely loopback concern). 60 s comfortably covers both
        # loopback and tunnel-relayed callers.
        timeout = 60

        # ── plumbing ─────────────────────────────────────────────────

        def log_message(self, format: str, *args: Any) -> None:
            # Default handler logs full request lines to stderr; ours
            # carry only paths/ids (never payloads), which is fine, but
            # route through logging for consistency.
            log.debug("http: " + format, *args)

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _job_id_from(self, path: str, suffix: str = "") -> str | None:
            prefix = "/v1/jobs/"
            if not path.startswith(prefix) or not path.endswith(suffix):
                return None
            job_id = path[len(prefix) : len(path) - len(suffix)]
            if not _JOB_ID_RE.fullmatch(job_id):
                return None
            return job_id

        # ── verbs ────────────────────────────────────────────────────

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                if self.path == "/healthz":
                    counts = store.counts()
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "queued": counts.get("queued", 0),
                            "running": counts.get("running", 0),
                        },
                    )
                    return
                job_id = self._job_id_from(self.path, "/result")
                if job_id is not None and self.path.endswith("/result"):
                    envelope = store.result(job_id)
                    if envelope is None:
                        status = store.status(job_id) or {}
                        self._send_json(
                            200,
                            {
                                "job_id": job_id,
                                "status": status.get("status", "running"),
                                "elapsed_s": status.get("elapsed_s", 0),
                            },
                        )
                        return
                    self._send_json(200, {"job_id": job_id, "result": envelope})
                    return
                job_id = self._job_id_from(self.path)
                if job_id is not None:
                    status = store.status(job_id)
                    if status is None:
                        self._send_json(404, {"error": "unknown job_id"})
                        return
                    self._send_json(200, status)
                    return
                self._send_json(404, {"error": "not found"})
            except QueueError as exc:
                self._send_json(exc.http_status, {"error": str(exc)})
            except Exception:  # noqa: BLE001 - boundary: never crash the listener
                log.exception("GET handler error")
                self._send_json(500, {"error": "internal error"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                if self.path != "/v1/jobs":
                    # The request body (if any) is never read on this
                    # path. Under HTTP/1.1 keep-alive those unread bytes
                    # would be parsed as the START of the next request
                    # (reproduced: a follow-up GET drew a 501), so the
                    # connection must close after the error response.
                    self.close_connection = True
                    self._send_json(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    # A non-numeric header is a client error, not the
                    # 500 the boundary except below would turn it into.
                    # The body length is unknowable, so the unread bytes
                    # would desynchronize a keep-alive connection —
                    # close it (same reproduction as above).
                    self.close_connection = True
                    self._send_json(400, {"error": "invalid Content-Length"})
                    return
                if length < 0:
                    # Negative is as malformed as non-numeric, with the
                    # same unread-body desync — close.
                    self.close_connection = True
                    self._send_json(400, {"error": "invalid Content-Length"})
                    return
                if length == 0:
                    # A declared-zero body has nothing unread: any bytes
                    # that follow are, per protocol, the next request —
                    # keep-alive stays safe here.
                    self._send_json(400, {"error": "empty body"})
                    return
                if length > MAX_BODY_BYTES:
                    # Drain the oversize body first so the client reliably
                    # receives the 413 instead of a broken pipe mid-send.
                    # Bounded by Content-Length; loopback callers are
                    # trusted, so this is a UX nicety, not a DoS surface.
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(remaining, 65536))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    self._send_json(
                        413,
                        {"error": f"body exceeds {MAX_BODY_BYTES} bytes"},
                    )
                    return
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send_json(400, {"error": "body is not valid JSON"})
                    return
                job_id = store.submit(payload)
                self._send_json(200, {"job_id": job_id, "status": "queued"})
            except QueueError as exc:
                self._send_json(exc.http_status, {"error": str(exc)})
            except Exception:  # noqa: BLE001 - boundary: never crash the listener
                log.exception("POST handler error")
                self._send_json(500, {"error": "internal error"})

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                job_id = self._job_id_from(self.path)
                if job_id is None:
                    self._send_json(404, {"error": "not found"})
                    return
                status = store.cancel(job_id)
                self._send_json(200, {"job_id": job_id, "status": status})
            except QueueError as exc:
                self._send_json(exc.http_status, {"error": str(exc)})
            except Exception:  # noqa: BLE001 - boundary: never crash the listener
                log.exception("DELETE handler error")
                self._send_json(500, {"error": "internal error"})

    return Handler


def make_server(store: QueueStore, port: int) -> ThreadingHTTPServer:
    """Loopback-only HTTP server; port 0 picks an ephemeral port (tests)."""
    return ThreadingHTTPServer(("127.0.0.1", port), _make_handler(store))


def _sweep_loop(store: QueueStore, stop: threading.Event) -> None:
    while not stop.wait(SWEEP_INTERVAL_S):
        try:
            store.purge_expired()
        except Exception:  # noqa: BLE001 - sweep must never kill the service
            log.exception("ttl sweep failed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if getpass.getuser() != KEY_ACCOUNT:
        log.warning(
            "running as %r but the keychain item account is %r",
            getpass.getuser(),
            KEY_ACCOUNT,
        )
    try:
        key = load_key_from_system_keychain()
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    port = int(os.environ.get("QUEUE_PORT") or DEFAULT_PORT)
    upstream = os.environ.get("QUEUE_UPSTREAM") or DEFAULT_UPSTREAM
    db_path = Path(os.environ.get("QUEUE_DB") or DEFAULT_DB)

    store = QueueStore(db_path, key)
    store.recover_running()
    store.purge_expired()

    worker = QueueWorker(store, upstream)
    worker.start()

    stop = threading.Event()
    sweeper = threading.Thread(target=_sweep_loop, args=(store, stop), daemon=True)
    sweeper.start()

    server = make_server(store, port)
    log.info(
        "delegate-queue listening on 127.0.0.1:%d (upstream %s, db %s)",
        port,
        upstream,
        db_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
