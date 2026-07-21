#!/usr/bin/env python3
"""Ground-truth benchmark corpus for qwen3.6-coding vs gemma4:12b.

Two task families mirroring the two real consumers:

  REVIEW   -> review-pipeline / Qodo pr-agent step (code review on a diff)
  DELEGATE -> ai-tools-mcp local_delegate (mechanical work, transforms, codegen)

Every REVIEW task carries ground truth: BUGGY tasks name the planted defect,
CLEAN tasks assert there is no defect at all. Recall comes from the BUGGY set;
the hallucination rate that is actually in dispute comes from the CLEAN set.

Every DELEGATE task is machine-gradable (exec, parse, or exact-match) so no
LLM judge sits in the loop for that half.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"

# ── Review-task prompt, modelled on what pr-agent actually sends ────────
REVIEW_SYSTEM = (
    "You are a senior code reviewer performing a pull-request review. "
    "Report only defects you can point to in the diff. Do not speculate "
    "about code you cannot see. If the diff is correct, say so.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "VERDICT: <APPROVE|REQUEST_CHANGES>\n"
    "FINDINGS:\n"
    "- [SEVERITY] <one-line description>   (repeat per finding; "
    "write '- NONE' if there are no defects)\n"
    "SEVERITY is one of CRITICAL, MAJOR, MINOR."
)

REVIEW_USER = "Review this diff.\n\n```diff\n{diff}\n```"


def R(tid, kind, lang, gt, diff, note=""):
    return dict(
        id=tid,
        family="review",
        kind=kind,
        lang=lang,
        ground_truth=gt,
        diff=diff,
        note=note,
    )


REVIEW = [
    # ─────────────── BUGGY: a real, unambiguous, planted defect ───────────────
    R(
        "R01",
        "buggy",
        "python",
        "path traversal: user-supplied path joined to base without rejecting '..'",
        r'''
--- a/store.py
+++ b/store.py
@@ -12,6 +12,14 @@ BASE = "/var/lib/appdata"

+def read_artifact(name: str) -> bytes:
+    """Read an artifact by name from the artifact store."""
+    target = os.path.join(BASE, name)
+    with open(target, "rb") as fh:
+        return fh.read()
+
''',
    ),
    R(
        "R02",
        "buggy",
        "python",
        "off-by-one: range(1, max_retries) performs max_retries-1 attempts, not max_retries",
        r"""
--- a/fetch.py
+++ b/fetch.py
@@ -3,6 +3,17 @@ import time

+def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
+    last_exc = None
+    for attempt in range(1, max_retries):
+        try:
+            return http_get(url, timeout=10)
+        except TimeoutError as exc:
+            last_exc = exc
+            time.sleep(2 ** attempt)
+    raise last_exc
+
""",
    ),
    R(
        "R03",
        "buggy",
        "python",
        "mutable default argument: dict default is shared across all calls",
        r"""
--- a/cache.py
+++ b/cache.py
@@ -1,3 +1,12 @@

+def memo_lookup(key: str, seen: dict = {}) -> int:
+    if key in seen:
+        return seen[key]
+    value = expensive_compute(key)
+    seen[key] = value
+    return value
+
""",
    ),
    R(
        "R04",
        "buggy",
        "bash",
        "unquoted variable in rm -rf: empty or whitespace value expands destructively",
        r"""
--- a/cleanup.sh
+++ b/cleanup.sh
@@ -4,6 +4,11 @@ set -euo pipefail

+purge_workdir() {
+  local DIR="$1"
+  echo "purging $DIR"
+  rm -rf $DIR/*
+}
+
""",
    ),
    R(
        "R05",
        "buggy",
        "python",
        "shell injection: user input interpolated into a shell=True command string",
        r'''
--- a/git_tools.py
+++ b/git_tools.py
@@ -20,6 +20,12 @@ import subprocess

+def checkout_branch(branch: str) -> None:
+    """Check out the requested branch in the working tree."""
+    subprocess.run(
+        f"git checkout {branch}", shell=True, check=True
+    )
+
''',
    ),
    R(
        "R06",
        "buggy",
        "python",
        "fail-open authorization: missing/None token returns True (authorized) instead of denying",
        r'''
--- a/authz.py
+++ b/authz.py
@@ -30,6 +30,16 @@ def _lookup(token): ...

+def is_authorized(token: str | None, scope: str) -> bool:
+    """Return True if the token grants the requested scope."""
+    if token is None:
+        return True
+    record = _lookup(token)
+    if record is None:
+        return False
+    return scope in record.scopes
+
''',
    ),
    R(
        "R07",
        "buggy",
        "python",
        "dict mutated while being iterated: RuntimeError at runtime",
        r"""
--- a/sessions.py
+++ b/sessions.py
@@ -44,6 +44,12 @@ import time

+def reap_expired(sessions: dict[str, Session]) -> int:
+    removed = 0
+    for sid, sess in sessions.items():
+        if sess.expires_at < time.time():
+            del sessions[sid]
+            removed += 1
+    return removed
+
""",
    ),
    R(
        "R08",
        "buggy",
        "python",
        "resource leak: file handle never closed when the parse raises",
        r"""
--- a/loader.py
+++ b/loader.py
@@ -8,6 +8,14 @@ import json

+def load_config(path: str) -> dict:
+    fh = open(path, "r", encoding="utf-8")
+    data = json.load(fh)
+    fh.close()
+    return data
+
""",
    ),
    # ─────────────── CLEAN: correct code. Any CRITICAL/MAJOR = hallucination ───────────────
    R(
        "R09",
        "clean",
        "python",
        "NONE - correct retry loop with bounded backoff and specific exception types",
        r"""
--- a/fetch.py
+++ b/fetch.py
@@ -3,6 +3,18 @@ import time

+def fetch_with_retry(url: str, max_retries: int = 3) -> Response:
+    last_exc: Exception | None = None
+    for attempt in range(max_retries):
+        try:
+            return http_get(url, timeout=10)
+        except (TimeoutError, ConnectionError) as exc:
+            last_exc = exc
+            if attempt < max_retries - 1:
+                time.sleep(min(2 ** attempt, 30))
+    assert last_exc is not None
+    raise last_exc
+
""",
        note="looks reviewable (loop, sleep, retries) but is correct",
    ),
    R(
        "R10",
        "clean",
        "bash",
        "NONE - correct: strict mode, quoted expansions, guarded empty input",
        r"""
--- a/cleanup.sh
+++ b/cleanup.sh
@@ -4,6 +4,15 @@ set -euo pipefail

+purge_workdir() {
+  local dir="${1:?usage: purge_workdir <dir>}"
+  if [[ ! -d "$dir" ]]; then
+    echo "not a directory: $dir" >&2
+    return 1
+  fi
+  find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
+}
+
""",
        note="quoted, guarded, uses find -exec",
    ),
    R(
        "R11",
        "clean",
        "python",
        "NONE - pure rename plus type annotations, zero behaviour change",
        r"""
--- a/metrics.py
+++ b/metrics.py
@@ -10,8 +10,8 @@ from typing import Iterable

-def calc(vals):
-    t = sum(vals)
-    n = len(vals)
-    return t / n if n else 0.0
+def mean(values: Iterable[float]) -> float:
+    total = sum(values)
+    count = len(values)
+    return total / count if count else 0.0

""",
        note="cosmetic refactor only",
    ),
    R(
        "R12",
        "clean",
        "python",
        "NONE - correct; at most a MINOR style nit is defensible",
        r"""
--- a/report.py
+++ b/report.py
@@ -15,6 +15,15 @@ from decimal import Decimal

+def format_total(rows: list[Row]) -> str:
+    total = Decimal("0")
+    for row in rows:
+        total += Decimal(str(row.amount))
+    return f"${total:,.2f}"
+
""",
        note="Decimal used correctly for money; only nits available",
    ),
]

# ── Delegate tasks: every one machine-gradable ─────────────────────────

# Synthetic credential fixtures for the redaction tasks, assembled at
# runtime so no token-shaped literal exists in this file. The values are
# fake; the concatenation exists purely so GitHub push protection and
# secret scanners have nothing credential-shaped to match on.
FAKE_SK = "sk-live-" + "4f9a2b7c8d1e6053"
FAKE_PW = "hunter2-" + "correct-horse"
FAKE_GHP = "ghp_" + "A1b2C3d4E5f6G7h8" + "I9j0K1l2M3n4O5p6Q7r8"


def D(tid, grader, prompt, system="", **kw):
    d = dict(id=tid, family="delegate", grader=grader, prompt=prompt, system=system)
    d.update(kw)
    return d


DELEGATE = [
    D(
        "D01",
        "json_schema",
        system="You output JSON only. No prose, no markdown fences.",
        prompt=(
            "Extract the fields from this log line into JSON with exactly the "
            "keys: timestamp, level, service, message, latency_ms (number).\n\n"
            "2026-07-19T14:03:22Z ERROR [billing-api] upstream refused "
            "connection after 4213ms"
        ),
        expect={
            "timestamp": "2026-07-19T14:03:22Z",
            "level": "ERROR",
            "service": "billing-api",
            "latency_ms": 4213,
        },
    ),
    D(
        "D02",
        "pytest",
        system="You output a single Python code block and nothing else.",
        prompt=(
            "Write a Python function `chunk_ranges(total: int, size: int) -> "
            "list[tuple[int, int]]` that splits range(total) into inclusive-"
            "start/exclusive-end chunks of at most `size`.\n"
            "chunk_ranges(10, 4) == [(0,4),(4,8),(8,10)]\n"
            "chunk_ranges(0, 4) == []\n"
            "chunk_ranges(3, 10) == [(0,3)]\n"
            "Raise ValueError if size <= 0. Output only the function."
        ),
        tests=[
            "assert chunk_ranges(10, 4) == [(0,4),(4,8),(8,10)]",
            "assert chunk_ranges(0, 4) == []",
            "assert chunk_ranges(3, 10) == [(0,3)]",
            "assert chunk_ranges(8, 4) == [(0,4),(4,8)]",
            "assert chunk_ranges(1, 1) == [(0,1)]",
            "try:\n    chunk_ranges(5, 0)\n    raise AssertionError('no ValueError')\nexcept ValueError:\n    pass",
        ],
    ),
    D(
        "D03",
        "json_schema",
        system="You output JSON only. No prose, no markdown fences.",
        prompt=(
            "Convert these shell exports into a single flat JSON object "
            "mapping name to value. Preserve values exactly.\n\n"
            'export API_HOST="api.internal.example"\n'
            "export API_PORT=8443\n"
            'export RETRY_BUDGET="5"\n'
            'export FEATURE_FLAGS="a,b,c"\n'
            "export DEBUG=false\n"
        ),
        expect={
            "API_HOST": "api.internal.example",
            "API_PORT": 8443,
            "RETRY_BUDGET": "5",
            "FEATURE_FLAGS": "a,b,c",
        },
    ),
    D(
        "D04",
        "constraint",
        system="Follow the output constraints exactly.",
        prompt=(
            "Summarize the following into EXACTLY 5 bullet points. Each bullet "
            "must start with '- ' and be 15 words or fewer. No preamble, no "
            "closing line.\n\n"
            "The service migrated from a single Postgres primary to a sharded "
            "topology with four shards keyed by tenant id. Read replicas were "
            "added per shard to absorb analytics traffic that previously "
            "competed with transactional writes. The migration ran online using "
            "logical replication, with a dual-write window of six hours and a "
            "cutover guarded by a feature flag. Latency at p99 dropped from "
            "840ms to 210ms. Two incidents occurred: a replication slot filled "
            "the primary's disk, and a stale connection pool routed writes to a "
            "decommissioned host for eleven minutes."
        ),
        constraints={"bullets": 5, "max_words": 15},
    ),
    D(
        "D05",
        "shellcheck",
        system="You output a single bash code block and nothing else.",
        prompt=(
            "Write a bash script that takes one argument (a directory), exits "
            "non-zero with a message on stderr if the argument is missing or "
            "not a directory, and otherwise prints the count of regular files "
            "directly inside it (not recursive). Use strict mode. Output only "
            "the script."
        ),
    ),
    D(
        "D06",
        "redact",
        system="You output the redacted text only.",
        prompt=(
            "Redact every secret value in the config below by replacing the "
            "value with the literal string REDACTED. Keep all keys, structure, "
            "and non-secret values unchanged. Output the full config.\n\n"
            "host = api.internal.example\n"
            "port = 8443\n"
            f"api_key = {FAKE_SK}\n"
            "timeout_s = 30\n"
            f"db_password = {FAKE_PW}\n"
            "region = us-east-1\n"
            f"github_token = {FAKE_GHP}\n"
            "log_level = info\n"
        ),
        secrets=[FAKE_SK, FAKE_PW, FAKE_GHP],
        keep=["api.internal.example", "8443", "30", "us-east-1", "info"],
    ),
    D(
        "D07",
        "regex",
        system='You output JSON only: {"pattern": "..."}. No prose.',
        prompt=(
            "Write a Python regular expression that matches a semantic version "
            "tag of the form vMAJOR.MINOR.PATCH, anchored to the whole string, "
            "where each part is one or more digits with no leading zeros "
            "(except the digit 0 itself).\n"
            "Must match: v1.0.0, v0.1.2, v10.20.30\n"
            "Must NOT match: 1.0.0, v01.0.0, v1.0, v1.0.0-rc1\n"
            'Return {"pattern": "<the regex>"} and nothing else.'
        ),
        match=["v1.0.0", "v0.1.2", "v10.20.30"],
        nomatch=["1.0.0", "v01.0.0", "v1.0", "v1.0.0-rc1", "vx.y.z"],
    ),
    D(
        "D08",
        "pytest",
        system="You output a single Python code block and nothing else.",
        prompt=(
            "Write a Python function `redact_secrets(text: str) -> str` that "
            "replaces the VALUE of any line matching `key = value` where the "
            "key contains 'token', 'password', 'secret', or 'key' "
            "(case-insensitive) with the literal REDACTED, preserving the "
            "`key = ` prefix exactly. Other lines pass through unchanged. "
            "Output only the function."
        ),
        tests=[
            "assert redact_secrets('api_key = abc123') == 'api_key = REDACTED'",
            "assert redact_secrets('host = example.com') == 'host = example.com'",
            "assert redact_secrets('DB_PASSWORD = hunter2') == 'DB_PASSWORD = REDACTED'",
            "assert redact_secrets('a = 1\\nsecret = xyz') == 'a = 1\\nsecret = REDACTED'",
            "assert redact_secrets('') == ''",
        ],
    ),
]


def main():
    TASKS.mkdir(parents=True, exist_ok=True)
    for t in REVIEW:
        t["system"] = REVIEW_SYSTEM
        t["prompt"] = REVIEW_USER.format(diff=t["diff"].strip())
        (TASKS / f"{t['id']}.json").write_text(json.dumps(t, indent=2))
    for t in DELEGATE:
        (TASKS / f"{t['id']}.json").write_text(json.dumps(t, indent=2))
    n_buggy = sum(1 for t in REVIEW if t["kind"] == "buggy")
    n_clean = sum(1 for t in REVIEW if t["kind"] == "clean")
    print(
        f"wrote {len(REVIEW)} review tasks ({n_buggy} buggy, {n_clean} clean) "
        f"+ {len(DELEGATE)} delegate tasks -> {TASKS}"
    )


if __name__ == "__main__":
    main()
