#!/usr/bin/env python3
"""Unit tests for queue_server.py (the durable delegate queue service).

Unlike the other test files, this one imports its module directly with a
REAL ``cryptography`` — stubbing AES-GCM would make the crypto
round-trip tests meaningless. CI (and the local command) therefore adds
``--with cryptography``:

    uv run --with pytest --with cryptography pytest test_queue_server.py -q

No network beyond loopback: the end-to-end tests bind ephemeral loopback
ports for both the queue API and a fake upstream Ollama. The keychain is
never touched — key-loading tests mock ``subprocess.run``, and every
store gets a random in-test key.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import importlib.util
import json
import secrets
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent

# Unique per-file module name — same collision-proofing convention as the
# other test files (see test_redact.py for the rationale).
_spec = importlib.util.spec_from_file_location(
    "queue_server_under_test", HERE / "queue_server.py"
)
assert _spec is not None and _spec.loader is not None
queue_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(queue_server)


def _key() -> bytes:
    return secrets.token_bytes(32)


def _store(tmp: Path, key: bytes | None = None) -> queue_server.QueueStore:
    return queue_server.QueueStore(tmp / "queue.db", key or _key())


_PAYLOAD = {
    "model": "gemma4:12b-nvfp4",
    "messages": [{"role": "user", "content": "summarize the attached notes"}],
    "think": False,
    "stream": False,
}


class TestCrypto(unittest.TestCase):
    def test_round_trip(self):
        key = _key()
        nonce, ct = queue_server.encrypt_blob(key, _PAYLOAD)
        self.assertEqual(queue_server.decrypt_blob(key, nonce, ct), _PAYLOAD)

    def test_fresh_nonce_per_encryption(self):
        key = _key()
        n1, c1 = queue_server.encrypt_blob(key, _PAYLOAD)
        n2, c2 = queue_server.encrypt_blob(key, _PAYLOAD)
        self.assertNotEqual(n1, n2)
        self.assertNotEqual(c1, c2)

    def test_ciphertext_does_not_contain_plaintext(self):
        nonce, ct = queue_server.encrypt_blob(_key(), _PAYLOAD)
        self.assertNotIn(b"summarize the attached notes", nonce + ct)

    def test_wrong_key_fails_closed(self):
        nonce, ct = queue_server.encrypt_blob(_key(), _PAYLOAD)
        with self.assertRaises(ValueError):
            queue_server.decrypt_blob(_key(), nonce, ct)

    def test_tampered_ciphertext_fails_closed(self):
        key = _key()
        nonce, ct = queue_server.encrypt_blob(key, _PAYLOAD)
        tampered = bytes([ct[0] ^ 0xFF]) + ct[1:]
        with self.assertRaises(ValueError):
            queue_server.decrypt_blob(key, nonce, tampered)


class TestKeyLoading(unittest.TestCase):
    """FAIL CLOSED: no key, no service — and never echo key material."""

    def _load(self, returncode=0, stdout=""):
        proc = mock.Mock(returncode=returncode, stdout=stdout)
        with mock.patch.object(
            queue_server.subprocess, "run", return_value=proc
        ) as run:
            key = queue_server.load_key_from_system_keychain()
        return key, run

    def test_valid_key_loads(self):
        raw = secrets.token_bytes(32)
        key, run = self._load(stdout=base64.b64encode(raw).decode() + "\n")
        self.assertEqual(key, raw)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/security")
        self.assertIn("DELEGATE_QUEUE_KEY", argv)

    def test_missing_item_refuses_to_start(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._load(returncode=44)
        self.assertIn("refusing to start", str(ctx.exception))

    def test_invalid_base64_refuses(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._load(stdout="not-base64!!!")
        self.assertIn("base64", str(ctx.exception))

    def test_wrong_length_refuses_without_echoing_material(self):
        material = base64.b64encode(b"short-key").decode()
        with self.assertRaises(RuntimeError) as ctx:
            self._load(stdout=material)
        message = str(ctx.exception)
        self.assertIn("expected 32", message)
        self.assertNotIn(material, message)
        self.assertNotIn("short-key", message)

    def test_missing_security_binary_refuses(self):
        with mock.patch.object(
            queue_server.subprocess, "run", side_effect=FileNotFoundError
        ):
            with self.assertRaises(RuntimeError):
                queue_server.load_key_from_system_keychain()


class TestValidatePayload(unittest.TestCase):
    def test_valid_payload_passes_and_forces_stream_false(self):
        out = queue_server.validate_payload({**_PAYLOAD, "stream": True})
        self.assertIs(out["stream"], False)
        self.assertEqual(out["model"], _PAYLOAD["model"])

    def test_rejections(self):
        cases = [
            "not-a-dict",
            {},
            {**_PAYLOAD, "evil_key": 1},
            {**_PAYLOAD, "model": ""},
            {**_PAYLOAD, "model": 42},
            {**_PAYLOAD, "messages": []},
            {**_PAYLOAD, "messages": ["hi"]},
            {**_PAYLOAD, "messages": [{"role": 1, "content": "x"}]},
            {**_PAYLOAD, "think": "yes"},
            {**_PAYLOAD, "keep_alive": "5 m"},
            {**_PAYLOAD, "keep_alive": "99999s"},
            {**_PAYLOAD, "options": "fast"},
        ]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(queue_server.QueueError):
                    queue_server.validate_payload(bad)

    def test_keep_alive_zero_and_units_accepted(self):
        for good in ("0", "5m", "30s", "2h"):
            with self.subTest(good=good):
                out = queue_server.validate_payload({**_PAYLOAD, "keep_alive": good})
                self.assertEqual(out["keep_alive"], good)


class TestQueueStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.key = _key()
        self.store = _store(self.tmp, self.key)

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.store.db_path)

    def _set(self, job_id, **cols):
        assignments = ", ".join(f"{k}=?" for k in cols)
        with contextlib.closing(self._db()) as conn, conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id=?",  # noqa: S608 - test-only, fixed column names
                (*cols.values(), job_id),
            )

    def test_submit_returns_q_prefixed_id_and_queued_status(self):
        job_id = self.store.submit(_PAYLOAD)
        self.assertRegex(job_id, r"^q[0-9a-f]{32}$")
        status = self.store.status(job_id)
        self.assertEqual(status["status"], "queued")
        self.assertEqual(status["attempts"], 0)
        self.assertEqual(status["model"], _PAYLOAD["model"])

    def test_payload_is_encrypted_at_rest(self):
        self.store.submit(_PAYLOAD)
        raw = self.store.db_path.read_bytes()
        for path in (
            Path(str(self.store.db_path) + "-wal"),
            Path(str(self.store.db_path) + "-shm"),
        ):
            if path.exists():
                raw += path.read_bytes()
        self.assertNotIn(b"summarize the attached notes", raw)

    def test_claim_marks_running_and_bumps_attempts(self):
        job_id = self.store.submit(_PAYLOAD)
        claimed_id, payload = self.store.claim_next()
        self.assertEqual(claimed_id, job_id)
        self.assertEqual(payload["model"], _PAYLOAD["model"])
        status = self.store.status(job_id)
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["attempts"], 1)
        self.assertIsNone(self.store.claim_next())  # nothing else queued

    def test_result_persists_until_ttl_not_single_collect(self):
        # The deliberate contrast with the in-memory store: collecting a
        # result does NOT delete it.
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        answer = {"model": "m", "message": {"content": "the answer"}}
        self.assertTrue(self.store.finish(job_id, answer))
        self.assertEqual(self.store.result(job_id), answer)
        self.assertEqual(self.store.result(job_id), answer)  # again — still there
        self.assertEqual(self.store.status(job_id)["status"], "done")

    def test_result_none_while_pending_and_404_when_unknown(self):
        job_id = self.store.submit(_PAYLOAD)
        self.assertIsNone(self.store.result(job_id))
        with self.assertRaises(queue_server.QueueError):
            self.store.result("q" + "e" * 32)

    def test_crash_recovery_requeues_running_jobs(self):
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self.assertEqual(self.store.status(job_id)["status"], "running")
        # Simulate a crash+restart: recover_running is the startup step.
        self.assertEqual(self.store.recover_running(), 1)
        status = self.store.status(job_id)
        self.assertEqual(status["status"], "queued")
        self.assertEqual(status["attempts"], 1)  # attempt is not refunded

    def test_attempts_cap_fails_job_after_two_crashes(self):
        job_id = self.store.submit(_PAYLOAD)
        for expected_attempts in (1, 2):
            claimed = self.store.claim_next()
            self.assertIsNotNone(claimed)
            self.assertEqual(self.store.status(job_id)["attempts"], expected_attempts)
            self.store.recover_running()
        # Third claim finds attempts exhausted → failed, not claimed.
        self.assertIsNone(self.store.claim_next())
        status = self.store.status(job_id)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_class"], "attempts_exhausted")

    def test_recovery_fails_running_job_already_at_cap(self):
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self._set(job_id, attempts=queue_server.MAX_ATTEMPTS)
        self.store.recover_running()
        self.assertEqual(self.store.status(job_id)["status"], "failed")

    def test_ttl_purges_old_terminal_jobs_only(self):
        old_done = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self.store.finish(old_done, {"message": {"content": "x"}})
        self._set(old_done, finished_at=time.time() - queue_server.TTL_S - 60)
        fresh_done = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self.store.finish(fresh_done, {"message": {"content": "y"}})
        still_queued = self.store.submit(_PAYLOAD)
        # An ancient queued job must never be purged — it has not run yet.
        self._set(still_queued, created_at=time.time() - queue_server.TTL_S - 60)

        self.assertEqual(self.store.purge_expired(), 1)
        self.assertIsNone(self.store.status(old_done))
        self.assertIsNotNone(self.store.status(fresh_done))
        self.assertEqual(self.store.status(still_queued)["status"], "queued")

    def test_cancel_queued_job(self):
        job_id = self.store.submit(_PAYLOAD)
        self.assertEqual(self.store.cancel(job_id), "cancelled")
        self.assertEqual(self.store.status(job_id)["status"], "cancelled")
        self.assertIsNone(self.store.claim_next())  # not claimable anymore
        # Its "result" is a failed envelope naming the cancellation.
        self.assertEqual(
            self.store.result(job_id),
            {"status": "failed", "error": "job was cancelled"},
        )

    def test_cancel_running_job_discards_late_outcome(self):
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self.assertEqual(self.store.cancel(job_id), "cancelled")
        # The worker's in-flight call eventually finishes — its outcome
        # must be discarded, not resurrect the job.
        self.assertFalse(self.store.finish(job_id, {"message": {"content": "late"}}))
        self.assertEqual(self.store.status(job_id)["status"], "cancelled")

    def test_cancel_terminal_job_conflicts(self):
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        self.store.finish(job_id, {"message": {"content": "x"}})
        with self.assertRaises(queue_server.QueueError) as ctx:
            self.store.cancel(job_id)
        self.assertEqual(ctx.exception.http_status, 409)

    def test_cancel_unknown_job_404(self):
        with self.assertRaises(queue_server.QueueError) as ctx:
            self.store.cancel("q" + "f" * 32)
        self.assertEqual(ctx.exception.http_status, 404)

    def test_submit_validates_payload(self):
        with self.assertRaises(queue_server.QueueError):
            self.store.submit({"model": "m"})  # no messages

    def test_submit_429_when_queue_full(self):
        with mock.patch.object(queue_server, "MAX_QUEUED", 2):
            self.store.submit(_PAYLOAD)
            self.store.submit(_PAYLOAD)
            with self.assertRaises(queue_server.QueueError) as ctx:
                self.store.submit(_PAYLOAD)
        self.assertEqual(ctx.exception.http_status, 429)
        self.assertIn("queue is full", str(ctx.exception))
        # Only QUEUED jobs count against the cap: work one off and the
        # next submit is accepted again.
        with mock.patch.object(queue_server, "MAX_QUEUED", 2):
            job_id, _ = self.store.claim_next()
            self.store.finish(job_id, {"message": {"content": "x"}})
            self.store.submit(_PAYLOAD)

    def test_attempts_exhausted_result_names_the_cause(self):
        job_id = self.store.submit(_PAYLOAD)
        for _ in (1, 2):
            self.store.claim_next()
            self.store.recover_running()
        self.assertIsNone(self.store.claim_next())  # fails the job
        result = self.store.result(job_id)
        self.assertEqual(result["status"], "failed")
        self.assertIn("attempts", result["error"])
        self.assertNotEqual(result["error"], "job finished without a result")

    def test_cancel_finish_race_never_overwrites_a_terminal_row(self):
        # cancel() and a late finish() racing from two threads must
        # serialize: whichever commits first wins, and the loser must
        # never overwrite the winner (a cancel that lands after finish
        # used to overwrite a real 'done' result). BEGIN IMMEDIATE in
        # both methods makes the interleaving atomic; run a handful of
        # iterations to exercise both orderings.
        for _ in range(10):
            job_id = self.store.submit(_PAYLOAD)
            self.store.claim_next()
            outcomes: dict[str, object] = {}
            barrier = threading.Barrier(2)

            def do_finish(job_id=job_id, outcomes=outcomes, barrier=barrier):
                barrier.wait()
                outcomes["finish_recorded"] = self.store.finish(
                    job_id, {"message": {"content": "late"}}
                )

            def do_cancel(job_id=job_id, outcomes=outcomes, barrier=barrier):
                barrier.wait()
                try:
                    self.store.cancel(job_id)
                    outcomes["cancelled"] = True
                except queue_server.QueueError as exc:
                    outcomes["cancelled"] = False
                    outcomes["cancel_status"] = exc.http_status

            threads = [
                threading.Thread(target=do_finish),
                threading.Thread(target=do_cancel),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            status = self.store.status(job_id)["status"]
            self.assertIn(status, ("done", "cancelled"))
            if status == "done":
                # finish won — cancel must have 409ed, and the stored
                # result must be intact.
                self.assertTrue(outcomes["finish_recorded"])
                self.assertFalse(outcomes["cancelled"])
                self.assertEqual(outcomes["cancel_status"], 409)
                self.assertEqual(
                    self.store.result(job_id)["message"]["content"], "late"
                )
            else:
                # cancel won — the late outcome must have been discarded.
                self.assertTrue(outcomes["cancelled"])
                self.assertFalse(outcomes["finish_recorded"])

    def test_db_files_are_owner_only(self):
        self.store.submit(_PAYLOAD)
        self.assertEqual(self.store.db_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.store.db_path.parent.stat().st_mode & 0o777, 0o700)


class TestWorkerResilience(unittest.TestCase):
    """No exception class may kill a worker thread (PR #61 P1): a dead
    worker silently strands every future queued job."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = _store(Path(self._tmp.name))
        self.worker = queue_server.QueueWorker(
            self.store, "http://127.0.0.1:9", poll_interval_s=0.01
        )

    def test_incomplete_read_fails_job_instead_of_raising(self):
        # http.client.IncompleteRead is an HTTPException, NOT an OSError,
        # so it escaped the original except chain and killed the thread.
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        with mock.patch.object(
            queue_server.urllib.request,
            "urlopen",
            side_effect=http.client.IncompleteRead(b""),
        ):
            self.worker._work_one(job_id, _PAYLOAD)  # must not raise
        status = self.store.status(job_id)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_class"], "worker_error")
        result = self.store.result(job_id)
        self.assertEqual(result["status"], "failed")
        # The error names the exception class, never response content.
        self.assertIn("IncompleteRead", result["error"])

    def test_arbitrary_exception_fails_job_instead_of_raising(self):
        job_id = self.store.submit(_PAYLOAD)
        self.store.claim_next()
        with mock.patch.object(
            queue_server.urllib.request,
            "urlopen",
            side_effect=RuntimeError("secret detail"),
        ):
            self.worker._work_one(job_id, _PAYLOAD)  # must not raise
        result = self.store.result(job_id)
        self.assertEqual(result["status"], "failed")
        self.assertIn("RuntimeError", result["error"])
        # Exception MESSAGES are never echoed into the stored result.
        self.assertNotIn("secret detail", result["error"])

    def test_worker_loop_survives_a_claim_next_crash(self):
        # Even a failure OUTSIDE the per-job path (e.g. SQLite raising in
        # claim_next) must not end the loop: the thread logs, backs off,
        # and keeps polling.
        calls: list[int] = []
        proceed = threading.Event()

        def flaky_claim():
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("transient store failure")
            proceed.set()
            return None

        with mock.patch.object(self.store, "claim_next", side_effect=flaky_claim):
            self.worker.concurrency = 1
            self.worker.start()
            try:
                self.assertTrue(
                    proceed.wait(timeout=5),
                    "worker thread died after the first claim_next crash",
                )
            finally:
                self.worker.stop()
        self.assertGreaterEqual(len(calls), 2)


class _FakeUpstream:
    """Minimal fake Ollama: POST /api/chat → canned JSON (or an error)."""

    def __init__(self, response=None, status=200):
        self.response = response or {"model": "m", "message": {"content": "up!"}}
        self.status = status
        self.requests: list = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps(outer.response).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _http(method: str, url: str, body: dict | None = None):
    """Tiny JSON HTTP helper → (status_code, parsed_body)."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class TestEndToEnd(unittest.TestCase):
    """Full loop on loopback: HTTP submit → worker → upstream → result."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = _store(Path(self._tmp.name))
        self.upstream = _FakeUpstream()
        self.addCleanup(self.upstream.stop)
        self.worker = queue_server.QueueWorker(
            self.store, self.upstream.url, poll_interval_s=0.02
        )
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.server = queue_server.make_server(self.store, 0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _await_status(self, job_id: str, wanted: str, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, status = _http("GET", f"{self.base}/v1/jobs/{job_id}")
            if status.get("status") == wanted:
                return status
            time.sleep(0.02)
        raise AssertionError(f"job never reached {wanted!r}: {status}")

    def test_healthz(self):
        code, body = _http("GET", f"{self.base}/healthz")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")

    def test_submit_work_collect_repeatedly(self):
        code, body = _http("POST", f"{self.base}/v1/jobs", _PAYLOAD)
        self.assertEqual(code, 200)
        job_id = body["job_id"]
        self.assertRegex(job_id, r"^q[0-9a-f]{32}$")
        self._await_status(job_id, "done")
        for _ in range(2):  # persistent, not single-collect
            code, result = _http("GET", f"{self.base}/v1/jobs/{job_id}/result")
            self.assertEqual(code, 200)
            self.assertEqual(result["result"]["message"]["content"], "up!")
        # The worker forwarded the sanitized payload upstream.
        self.assertEqual(self.upstream.requests[0]["model"], _PAYLOAD["model"])
        self.assertIs(self.upstream.requests[0]["stream"], False)

    def test_upstream_http_error_fails_job_with_error_class(self):
        self.upstream.status = 500
        _, body = _http("POST", f"{self.base}/v1/jobs", _PAYLOAD)
        status = self._await_status(body["job_id"], "failed")
        self.assertEqual(status["error_class"], "http_error")
        _, result = _http("GET", f"{self.base}/v1/jobs/{body['job_id']}/result")
        self.assertEqual(result["result"]["status"], "failed")
        self.assertIn("HTTP 500", result["result"]["error"])

    def test_invalid_payload_400(self):
        code, body = _http("POST", f"{self.base}/v1/jobs", {**_PAYLOAD, "evil": True})
        self.assertEqual(code, 400)
        self.assertIn("unknown payload keys", body["error"])

    def test_non_numeric_content_length_400(self):
        # urllib always sends a correct Content-Length, so drive a raw
        # socket to deliver the malformed header. Must be a 400 client
        # error, not a 500 from the boundary handler.
        with socket.create_connection(
            ("127.0.0.1", self.server.server_address[1]), timeout=5
        ) as sock:
            sock.sendall(
                b"POST /v1/jobs HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Length: abc\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            # Read to EOF rather than taking a single recv(): the handler
            # writes status/headers and body separately, so one recv can
            # return the headers alone and lose the body assertion below.
            # `Connection: close` above guarantees the server closes.
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks)
        self.assertIn(b" 400 ", response.split(b"\r\n", 1)[0])
        self.assertIn(b"invalid Content-Length", response)

    def test_handler_has_finite_socket_timeout(self):
        # StreamRequestHandler applies Handler.timeout via settimeout();
        # None would let a stalled client park a server thread forever.
        timeout = self.server.RequestHandlerClass.timeout
        self.assertIsNotNone(timeout)
        self.assertGreater(timeout, 0)

    def test_oversize_body_413(self):
        big = {**_PAYLOAD}
        big["messages"] = [
            {"role": "user", "content": "x" * (queue_server.MAX_BODY_BYTES + 10)}
        ]
        code, body = _http("POST", f"{self.base}/v1/jobs", big)
        self.assertEqual(code, 413)
        self.assertIn("exceeds", body["error"])

    def test_unknown_and_malformed_ids_404(self):
        code, _ = _http("GET", f"{self.base}/v1/jobs/q{'a' * 32}")
        self.assertEqual(code, 404)
        code, _ = _http("GET", f"{self.base}/v1/jobs/not-an-id")
        self.assertEqual(code, 404)
        code, _ = _http("GET", f"{self.base}/v1/jobs/../../etc/passwd")
        self.assertEqual(code, 404)

    def test_cancel_endpoint(self):
        # Stop the worker so the job stays queued long enough to cancel.
        self.worker.stop()
        _, body = _http("POST", f"{self.base}/v1/jobs", _PAYLOAD)
        code, cancelled = _http("DELETE", f"{self.base}/v1/jobs/{body['job_id']}")
        self.assertEqual(code, 200)
        self.assertEqual(cancelled["status"], "cancelled")
        code, again = _http("DELETE", f"{self.base}/v1/jobs/{body['job_id']}")
        self.assertEqual(code, 409)


if __name__ == "__main__":
    unittest.main()
