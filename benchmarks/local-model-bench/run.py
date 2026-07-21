#!/usr/bin/env python3
"""Sequential A/B harness against the local Ollama.

Sequential ON PURPOSE: both models are pinned in VRAM on the same GPU, so
running them concurrently would make the latency numbers meaningless. One
call at a time, alternating arms, so thermal/memory state is shared evenly.

Arms:
  qwen-nothink   qwen3.6:35b-a3b-coding-nvfp4   think=False  <- production
  gemma          gemma4:12b-nvfp4               think=False
  qwen-think     qwen3.6:35b-a3b-coding-nvfp4   think=True   <- diagnostic
"""

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"
OUT = HERE / "out"
# Loopback-only by design: this harness benchmarks the LOCAL Ollama at
# 127.0.0.1 (constant, never caller-supplied); TLS adds nothing on
# loopback and the production server separately refuses plain-http
# non-localhost endpoints. Suppressions are per-line and annotated.
# nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
ENDPOINT = "http://127.0.0.1:11434/api/chat"

QWEN = "qwen3.6:35b-a3b-coding-nvfp4"
GEMMA = "gemma4:12b-nvfp4"

ARMS = [
    ("qwen-nothink", QWEN, False),
    ("gemma", GEMMA, False),
    ("qwen-think", QWEN, True),
]

TRIALS_REVIEW = 3
TRIALS_DELEGATE = 3
# think=True is a diagnostic arm: review tasks only, fewer trials.
TRIALS_THINK = 2


def call(model: str, system: str, prompt: str, think: bool, timeout: int = 600):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": msgs, "stream": False, "think": think}
    data = json.dumps(body).encode()
    # nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        # Some models reject the `think` field outright; retry without it.
        if think is False and "think" in detail.lower():
            body.pop("think")
            # nosemgrep: python.lang.security.audit.insecure-transport.urllib.insecure-request-object.insecure-request-object
            req2 = urllib.request.Request(
                ENDPOINT,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.monotonic()
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                payload = json.loads(resp.read())
        else:
            return {
                "error": f"HTTP {exc.code}: {detail}",
                "wall_s": round(time.monotonic() - t0, 2),
            }
    except Exception as exc:  # noqa: BLE001 - harness must not die mid-run
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.monotonic() - t0, 2),
        }

    wall = time.monotonic() - t0
    msg = payload.get("message", {})
    return {
        "content": msg.get("content", ""),
        "thinking": msg.get("thinking", "") or "",
        "wall_s": round(wall, 2),
        "eval_count": payload.get("eval_count"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_duration_ns": payload.get("eval_duration"),
        "load_duration_ns": payload.get("load_duration"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task_files = sorted(TASKS.glob("*.json"))
    if not task_files:
        sys.exit("no tasks; run corpus.py first")
    tasks = [json.loads(p.read_text()) for p in task_files]

    jobs = []
    for t in tasks:
        for arm, model, think in ARMS:
            if think and t["family"] != "review":
                continue  # diagnostic arm is review-only
            if think:
                n = TRIALS_THINK
            else:
                n = TRIALS_REVIEW if t["family"] == "review" else TRIALS_DELEGATE
            for trial in range(1, n + 1):
                jobs.append((t, arm, model, think, trial))

    # Interleave arms so drift over the run hits every arm equally.
    jobs.sort(key=lambda j: (j[4], j[0]["id"], j[1]))

    total = len(jobs)
    print(f"[harness] {len(tasks)} tasks, {total} calls, sequential", flush=True)
    done = 0
    for task, arm, model, think, trial in jobs:
        dest = OUT / f"{task['id']}__{arm}__t{trial}.json"
        done += 1
        if dest.exists():
            print(f"[{done}/{total}] skip {dest.name} (cached)", flush=True)
            continue
        res = call(model, task.get("system", ""), task["prompt"], think)
        rec = {
            "task_id": task["id"],
            "family": task["family"],
            "arm": arm,
            "model": model,
            "think": think,
            "trial": trial,
            **res,
        }
        dest.write_text(json.dumps(rec, indent=2))
        status = "ERR" if "error" in res else f"{res['wall_s']}s"
        print(f"[{done}/{total}] {task['id']} {arm} t{trial} -> {status}", flush=True)

    print("[harness] complete", flush=True)


if __name__ == "__main__":
    main()
