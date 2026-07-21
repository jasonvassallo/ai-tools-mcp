#!/usr/bin/env python3
"""Tier-2 delegate tasks. Tier 1 saturated (0.98 vs 1.00), so these are built
to discriminate, and each maps to a use case named in the local_delegate tool
description: bulk transforms, on-device redaction, long-context extraction,
boilerplate/codegen, and instruction adherence.

Still 100% machine-gradable.
"""

import json
import pathlib
import random

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"

# ── E01: long-context extraction over a synthetic log ──────────────────
rng = random.Random(20260719)
SERVICES = ["billing-api", "auth-svc", "search-idx", "mailer", "gateway"]
LEVELS = ["INFO"] * 12 + ["WARN"] * 4 + ["ERROR"] * 3
log_lines, err_by_svc, lat = [], {s: 0 for s in SERVICES}, []
for i in range(160):
    svc = rng.choice(SERVICES)
    lvl = rng.choice(LEVELS)
    ms = rng.randint(5, 900)
    lat.append(ms)
    if lvl == "ERROR":
        err_by_svc[svc] += 1
    log_lines.append(
        f"2026-07-1{rng.randint(0, 9)}T{rng.randint(10, 23)}:"
        f"{rng.randint(10, 59)}:{rng.randint(10, 59)}Z {lvl} [{svc}] "
        f"request handled in {ms}ms"
    )
LOG = "\n".join(log_lines)
TOP_ERR = max(err_by_svc, key=lambda s: err_by_svc[s])
N_ERR = sum(1 for ln in log_lines if " ERROR " in ln)

# ── E03: bulk transform with an exception rule ─────────────────────────
BULK = [
    ("user_id", "int"),
    ("created_at", "timestamp"),
    ("email", "string"),
    ("is_active", "bool"),
    ("balance_cents", "int"),
    ("tenant_id", "uuid"),
    ("updated_at", "timestamp"),
    ("nickname", "string"),
    ("login_count", "int"),
    ("deleted_at", "timestamp"),
    ("is_admin", "bool"),
    ("avatar_url", "string"),
    ("shard_key", "uuid"),
    ("score", "int"),
    ("last_seen", "timestamp"),
    ("bio", "string"),
    ("is_verified", "bool"),
    ("referrer_id", "uuid"),
]
# rule: SQL type per logical type, EXCEPT any name ending in _at -> TIMESTAMPTZ
SQLMAP = {
    "int": "BIGINT",
    "string": "TEXT",
    "bool": "BOOLEAN",
    "uuid": "UUID",
    "timestamp": "TIMESTAMPTZ",
}
BULK_EXPECT = {}
for name, typ in BULK:
    BULK_EXPECT[name] = "TIMESTAMPTZ" if name.endswith("_at") else SQLMAP[typ]
BULK_INPUT = "\n".join(f"{n}: {t}" for n, t in BULK)

# ── E05: redaction at scale, with near-secret decoys ───────────────────
# Runtime-assembled synthetic credentials — see corpus.py for rationale.
FAKE_SK2 = "sk-live-" + "9f2a71bc0d4e8352"
FAKE_GHP2 = "ghp_" + "Z9y8X7w6V5u4T3s2" + "R1q0P9o8N7m6L5k4"
FAKE_AKIA = "AKIA" + "IOSFODNN7EXAMPLE"  # AWS's canonical docs example key
FAKE_PW2 = "hunter2-" + "battery-staple"
FAKE_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" + ".abc.def"

E05_SECRETS = [FAKE_SK2, FAKE_GHP2, FAKE_AKIA, FAKE_PW2, FAKE_JWT]
E05_KEEP = [
    "api.internal.example",
    "us-east-1",
    "8443",
    "30",
    "public_key_algorithm",
    "keyboard_shortcut",
    "monkey_patch",
    "info",
]
E05_CONFIG = f"""host = api.internal.example
port = 8443
region = us-east-1
openai_api_key = {FAKE_SK2}
timeout_s = 30
github_token = {FAKE_GHP2}
aws_access_key_id = {FAKE_AKIA}
db_password = {FAKE_PW2}
session_jwt = {FAKE_JWT}
public_key_algorithm = ed25519
keyboard_shortcut = cmd+shift+k
monkey_patch = enabled
log_level = info
"""


def D(tid, grader, prompt, system="", **kw):
    d = dict(
        id=tid, family="delegate", tier=2, grader=grader, prompt=prompt, system=system
    )
    d.update(kw)
    return d


TIER2 = [
    D(
        "E01",
        "json_schema",
        system="You output JSON only. No prose, no markdown fences.",
        prompt=(
            "Below is an application log. Return JSON with exactly these keys:\n"
            '  "total_lines" (number of log lines),\n'
            '  "error_count" (lines with level ERROR),\n'
            '  "top_error_service" (the service with the most ERROR lines).\n\n'
            f"{LOG}"
        ),
        expect={"total_lines": 160, "error_count": N_ERR, "top_error_service": TOP_ERR},
    ),
    D(
        "E02",
        "pytest",
        system="You output a single Python code block and nothing else.",
        prompt=(
            "Write `compare_semver(a: str, b: str) -> int` returning -1, 0 or 1 "
            "comparing two semantic versions of the form MAJOR.MINOR.PATCH with "
            "an optional prerelease suffix after '-'.\n"
            "Rules: numeric parts compare numerically; a version WITH a "
            "prerelease is LOWER than the same version without one; prerelease "
            "identifiers compare dot-separated, numeric identifiers compare "
            "numerically and rank lower than alphanumeric ones.\n"
            "Output only the function."
        ),
        tests=[
            "assert compare_semver('1.0.0', '1.0.1') == -1",
            "assert compare_semver('1.2.0', '1.10.0') == -1",
            "assert compare_semver('2.0.0', '2.0.0') == 0",
            "assert compare_semver('1.0.0-alpha', '1.0.0') == -1",
            "assert compare_semver('1.0.0', '1.0.0-alpha') == 1",
            "assert compare_semver('1.0.0-alpha.1', '1.0.0-alpha.2') == -1",
            "assert compare_semver('1.0.0-1', '1.0.0-alpha') == -1",
            "assert compare_semver('1.0.0-alpha.beta', '1.0.0-alpha.1') == 1",
        ],
    ),
    D(
        "E03",
        "json_schema",
        system="You output JSON only. No prose, no markdown fences.",
        prompt=(
            "Map each field below to a Postgres column type. Base mapping: "
            "int->BIGINT, string->TEXT, bool->BOOLEAN, uuid->UUID, "
            "timestamp->TIMESTAMPTZ.\n"
            "EXCEPTION: any field whose NAME ends in `_at` must be TIMESTAMPTZ "
            "regardless of its stated type.\n"
            "Return a flat JSON object mapping field name to type. Include every "
            "field.\n\n" + BULK_INPUT
        ),
        expect=BULK_EXPECT,
    ),
    D(
        "E04",
        "forbid",
        system="Follow the constraints literally.",
        prompt=(
            "Write a Python function `parse_port(raw: str) -> int` that converts "
            "a string to an int and returns it.\n\n"
            "HARD CONSTRAINTS — the reviewer will reject any violation:\n"
            "1. The function body must be EXACTLY ONE line (plus the def line).\n"
            "2. Do NOT add try/except, validation, range checks, logging, or "
            "docstrings.\n"
            "3. Do NOT add type guards or raise any exception yourself.\n"
            "Output only the code, no explanation."
        ),
        must_not=["try:", "except", "raise", "assert", '"""', "if ", "logging"],
        must=["def parse_port", "int("],
    ),
    D(
        "E05",
        "redact",
        system="You output the redacted config only.",
        prompt=(
            "Redact every SECRET VALUE in the config below, replacing the value "
            "with the literal REDACTED. A secret is a credential: api keys, "
            "tokens, passwords, access key ids, JWTs.\n"
            "Do NOT redact non-secret values, even if the key name merely "
            "contains a word like 'key' or 'patch'. Keep all keys and structure.\n"
            "Output the full config.\n\n" + E05_CONFIG
        ),
        secrets=E05_SECRETS,
        keep=E05_KEEP,
    ),
    D(
        "E06",
        "pytest",
        system="You output a single Python code block and nothing else.",
        prompt=(
            "Write `parse_hunk_header(line: str) -> tuple[int, int, int, int]` "
            "that parses a unified-diff hunk header of the form\n"
            "  @@ -oldStart,oldLen +newStart,newLen @@ optional trailing text\n"
            "returning (oldStart, oldLen, newStart, newLen).\n"
            "When a length is omitted (e.g. `@@ -3 +3 @@`) it defaults to 1.\n"
            "Raise ValueError on any line that is not a hunk header.\n"
            "Output only the function."
        ),
        tests=[
            "assert parse_hunk_header('@@ -12,6 +12,14 @@') == (12,6,12,14)",
            "assert parse_hunk_header('@@ -3 +3 @@') == (3,1,3,1)",
            "assert parse_hunk_header('@@ -1,2 +3 @@ def foo():') == (1,2,3,1)",
            "assert parse_hunk_header('@@ -0,0 +1,5 @@') == (0,0,1,5)",
            "try:\n    parse_hunk_header('not a hunk')\n    raise AssertionError('no ValueError')\nexcept ValueError:\n    pass",
            "try:\n    parse_hunk_header('--- a/x.py')\n    raise AssertionError('no ValueError')\nexcept ValueError:\n    pass",
        ],
    ),
    D(
        "E07",
        "constraint",
        system="Follow the output constraints exactly.",
        prompt=(
            "Rewrite the following commit message as EXACTLY 3 bullet points, "
            "each starting with '- ' and 12 words or fewer. No preamble, no "
            "trailing line.\n\n"
            "Refactored the ingestion worker so that batches are committed "
            "transactionally rather than per-row, which removes the partial-write "
            "failure mode we hit last quarter. Also swapped the retry backoff "
            "from linear to exponential with jitter, capped at thirty seconds, "
            "and added a dead-letter queue for payloads that fail five times. "
            "Throughput went from 1.2k to 8.4k rows per second in staging."
        ),
        constraints={"bullets": 3, "max_words": 12},
    ),
    D(
        "E08",
        "pytest",
        system="You output a single Python code block and nothing else.",
        prompt=(
            "Write `merge_ranges(ranges: list[tuple[int,int]]) -> "
            "list[tuple[int,int]]` that merges overlapping or touching "
            "half-open intervals and returns them sorted by start.\n"
            "Touching means (1,3) and (3,5) merge into (1,5). Empty input "
            "returns []. Do not mutate the input list.\n"
            "Output only the function."
        ),
        tests=[
            "assert merge_ranges([]) == []",
            "assert merge_ranges([(1,3),(3,5)]) == [(1,5)]",
            "assert merge_ranges([(5,7),(1,3)]) == [(1,3),(5,7)]",
            "assert merge_ranges([(1,10),(2,3)]) == [(1,10)]",
            "assert merge_ranges([(1,2),(4,5),(2,4)]) == [(1,5)]",
            "src=[(3,4),(1,2)]\nmerge_ranges(src)\nassert src == [(3,4),(1,2)], 'input mutated'",
        ],
    ),
]


def main():
    for t in TIER2:
        (TASKS / f"{t['id']}.json").write_text(json.dumps(t, indent=2))
    print(f"wrote {len(TIER2)} tier-2 delegate tasks")
    print(f"  E01 ground truth: lines=160 errors={N_ERR} top={TOP_ERR}")
    print(f"  E03 fields={len(BULK_EXPECT)}  E05 secrets={len(E05_SECRETS)}")


if __name__ == "__main__":
    main()
