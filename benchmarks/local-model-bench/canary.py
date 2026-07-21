#!/usr/bin/env python3
"""Canary probe: does a long-lived Ollama runner degrade with request count?

One fixed, trivial prompt with exactly one correct answer, fired repeatedly.
Any answer that is not the expected function is a degradation event. Run against
either model to see whether the effect is model-specific.
"""

import json
import pathlib
import sys
import time

import run as R

HERE = pathlib.Path(__file__).parent

PROMPT = (
    "Write a Python function `add_two(a: int, b: int) -> int` that returns "
    "the sum of its two arguments. Output only the function, nothing else."
)
SYSTEM = "You output a single Python code block and nothing else."


def ok(text: str) -> bool:
    """Execute the candidate and check behavior (Codex+Gemini): substring
    matching flagged stylistic variants (`return b + a`, intermediate
    variables) as degradation events."""
    import subprocess
    import sys
    import tempfile

    t = text.replace("```python", "").replace("```", "")
    prog = t + "\nassert add_two(2, 3) == 5\nassert add_two(-1, 1) == 0\nprint('OK')\n"
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "cand.py"
        f.write_text(prog)
        try:
            r = subprocess.run(
                [sys.executable, str(f)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and r.stdout.strip().endswith("OK")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else R.QWEN
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    tag = "qwen" if "qwen" in model else "gemma"
    recs = []
    bad = 0
    t_start = time.monotonic()
    for i in range(1, n + 1):
        r = R.call(model, SYSTEM, PROMPT, False, timeout=240)
        c = r.get("content", "") or ""
        good = ok(c)
        if not good:
            bad += 1
        recs.append(
            {
                "i": i,
                "ok": good,
                "wall_s": r.get("wall_s"),
                "eval_count": r.get("eval_count"),
                "uptime_s": round(time.monotonic() - t_start, 1),
                "sample": c[:160],
            }
        )
        if not good:
            print(f"  [{i}] DEGRADED: {c[:110]!r}", flush=True)
        elif i % 10 == 0:
            print(f"  [{i}] ok ({bad} bad so far)", flush=True)
    (HERE / f"canary_{tag}.json").write_text(json.dumps(recs, indent=2))
    print(f"{tag}: {n - bad}/{n} correct, {bad} degradation events")


if __name__ == "__main__":
    main()
