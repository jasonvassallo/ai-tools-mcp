#!/usr/bin/env python3
"""Does pinning temperature=0 fix qwen's cross-task contamination?

local_delegate currently sends no options block, so it inherits the model's
default temperature. If temp 0 removes the contamination, the fix is a
one-line payload change rather than a model swap — which keeps qwen's coding
strength instead of trading it away.

Replays the delegate corpus, 3 trials (contamination only ever appeared on
trials 2-3, once the cache held many conversations).
"""

import json
import pathlib
import sys
import time
import urllib.request

import grade_machine as G

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"
# Loopback-only by design: this harness benchmarks the LOCAL Ollama at
# 127.0.0.1 (constant, never caller-supplied); TLS adds nothing on
# loopback and the production server separately refuses plain-http
# non-localhost endpoints. Suppressions are per-line and annotated.
# nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
ENDPOINT = "http://127.0.0.1:11434/api/chat"
QWEN = "qwen3.6:35b-a3b-coding-nvfp4"
GEMMA = "gemma4:12b-nvfp4"

SIG = {
    "D02": ["chunk_ranges"],
    "D05": ["#!/usr/bin/env bash"],
    "D08": ["redact_secrets"],
    "E02": ["compare_semver"],
    "E03": ["TIMESTAMPTZ"],
    "E04": ["parse_port"],
    "E06": ["parse_hunk_header"],
    "E08": ["merge_ranges"],
}


def call(model, system, prompt, temperature):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    # nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for _ in range(2):
        try:
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read()).get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    return f"__ERROR__ {last}"


def main():
    model = QWEN if (len(sys.argv) < 2 or sys.argv[1] == "qwen") else GEMMA
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    tasks = [json.loads(p.read_text()) for p in sorted(TASKS.glob("[DE]*.json"))]

    rows = []
    for trial in (1, 2, 3):
        for t in tasks:
            txt = call(model, t.get("system", ""), t["prompt"], temp)
            score, note = G.GRADERS[t["grader"]](t, txt)
            own = SIG.get(t["id"], [])
            foreign = [
                o for o, s in SIG.items() if o != t["id"] and any(x in txt for x in s)
            ]
            contaminated = bool(foreign) and (not own or not any(x in txt for x in own))
            rows.append(
                {
                    "task": t["id"],
                    "trial": trial,
                    "score": score,
                    "contaminated": contaminated,
                    "foreign": foreign,
                    "note": note,
                }
            )
            flag = "  <<< CONTAMINATED " + str(foreign) if contaminated else ""
            print(f"  t{trial} {t['id']} score={score:.2f}{flag}", flush=True)

    tag = "qwen" if model == QWEN else "gemma"
    (HERE / f"temp{temp}_{tag}.json").write_text(json.dumps(rows, indent=2))
    n = len(rows)
    print(f"\n=== {tag} @ temperature={temp} ===")
    print(f"mean score      : {sum(r['score'] for r in rows) / n:.3f}")
    print(f"contamination   : {sum(r['contaminated'] for r in rows)}/{n}")
    for tr in (1, 2, 3):
        sub = [r for r in rows if r["trial"] == tr]
        print(
            f"  trial {tr}: score={sum(r['score'] for r in sub) / len(sub):.3f} "
            f"contaminated={sum(r['contaminated'] for r in sub)}/{len(sub)}"
        )


if __name__ == "__main__":
    main()
