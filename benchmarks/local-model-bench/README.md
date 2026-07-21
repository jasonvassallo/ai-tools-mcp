# local-model-bench

Ground-truth A/B harness for the local Ollama models behind `local_delegate`
and the review tooling. This is the corpus + harness that produced the
measurements cited in the `local_delegate` default-model change (PR #32) and
in `~/.claude/skills/review-pipeline/SKILL.md`'s model/mode decision note.
Committed so those numbers are reproducible instead of "measured off-repo."

## Why it exists

Model choices here kept being made from impressions ("qwen hallucinates",
"gemma is the better reviewer", "gemma can't think") that flipped or died on
contact with measurement. The harness's design goal is that **every score is
machine-checkable or blind-judged against planted ground truth** — no
grading by vibes.

## Layout

| file | role |
|---|---|
| `corpus.py` | Tier-1 tasks: 12 review diffs (8 with a planted defect, 4 clean controls) + 8 delegate tasks |
| `corpus2.py` | Tier-2 delegate tasks (harder: long-context extraction, exception-rule transforms, redaction with decoys, instruction adherence) |
| `run.py` | Sequential A/B runner — one call at a time, arms interleaved, per-trial caching |
| `grade_machine.py` | Deterministic graders (exec generated code, parse JSON, run regexes, redaction leak checks) — no LLM judge in this file |
| `make_units.py` | Emits **blinded** judging units (model identity stripped; mapping in `units/_key.json` which judges never read) |
| `agg_judge.py` | Joins external blind-judge verdicts back to arms and aggregates |
| `temp0.py` | Replays the delegate corpus at a pinned temperature (used to rule out temp-0 as a contamination fix) |
| `canary.py` | Fixed-prompt degradation probe for long-lived runners |

## Quickstart

```bash
python3 corpus.py && python3 corpus2.py   # generate tasks/ (not committed)
python3 run.py                            # sequential; writes out/*.json
python3 grade_machine.py                  # -> machine_scores.jsonl
python3 make_units.py                     # -> units/ for blind judging
```

Assumes Ollama at `http://127.0.0.1:11434` and the model tags named in
`run.py` (`QWEN` / `GEMMA` constants — edit to compare other pairs).

## Design decisions that matter

- **Sequential on purpose.** Both models share one GPU; concurrent calls
  would corrupt the latency comparison. Arms are interleaved so drift over
  the run hits every arm equally.
- **Clean controls are load-bearing.** A reviewer scoring 100% on buggy
  diffs may just be a model that never approves anything — only the clean
  set exposes that (it did: one model request-changes'd 100% of correct
  diffs).
- **Judges are blind and two-sided.** Review findings are judged with model
  identity stripped, three adversarial judges per claim on clean diffs, and
  judges are explicitly told the benchmark author's "clean" label may be
  wrong (twice, it was).
- **Repeat trials are the point.** The cross-task contamination failure
  (a model returning a *different recently-seen prompt's* answer) is 0% on
  first exposure and ~25% on repeats — a trials=1 benchmark cannot see it.
- **Credential fixtures are runtime-assembled.** The redaction tasks need
  token-shaped strings; they are concatenated at runtime so no
  secret-scanner-matchable literal exists in the source. All values are fake.

## Security note

`grade_machine.py` executes model-generated Python/bash in a subprocess to
grade it. Run this harness only against models you operate; it is a grader,
not a sandbox.

## Limitations

Small n (per-cell trials 2–3), one hardware/serving config per run, corpus
written by the same author who planted the defects. Treat absolute scores as
config-specific; the *comparative* deltas and failure modes are the durable
part. Raw outputs, judge journals, and the runs behind the 2026-07-19/20
numbers are archived (with logs) in
`~/Documents/security-audits/model-bench-2026-07-19/` — not committed, since
raw task/output JSON contains the assembled token-shaped fixtures.

See [RESULTS.md](RESULTS.md) for the 2026-07-19/20 findings.
