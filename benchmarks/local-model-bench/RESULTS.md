# Results — 2026-07-19/20, JVMBPro

**Environment:** Apple M5 Pro, 64 GB. Ollama 0.31.1 (`--mlx-engine`), both
models VRAM-resident. Serving env: `OLLAMA_KV_CACHE_TYPE=q8_0`,
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1`,
`OLLAMA_CONTEXT_LENGTH=65536`.
**Models:** `qwen3.6:35b-a3b-coding-nvfp4` (21 GB) vs `gemma4:12b-nvfp4`
(7.7 GB). 288 harness calls + ~150 replay/probe calls; 227 blind
adversarial judgements (Claude workflow, 3 judges per clean-diff claim).

## 1. Code review (template format: VERDICT + severity findings)

Blind-judged; qwen figures are from a fresh runner after a degraded-runner
run was discarded as unfair.

| arm | finds planted bug | approves clean | verdict acc | unjustified MAJOR | speculative/run | median |
|---|---|---|---|---|---|---|
| gemma4:12b | **1.00** | **0.00** | 0.67 | **13** | 0.00 | 1.2 s |
| qwen think:false | 0.88 | 0.67 | 0.89 | 1 | 0.21 | 1.0 s |
| qwen think:true | 0.94 | 0.88 | 0.96 | 1 | 0.12 | 19.1 s |

Gemma's perfect detection is partly an artifact: it issued REQUEST_CHANGES
on 100% of correct diffs (17 claims; blind judges upheld 2). Opposite
failure modes: gemma over-flags, qwen occasionally speculates beyond the
diff.

**think:true does not survive real diffs.** Replayed on real committed
diffs (temperature 0, num_predict 4096):

| diff size | think:false | think:true |
|---|---|---|
| 671 lines | verdict ✓ 78.9 s | **no verdict** — 13,099 thinking chars, budget exhausted |
| 312 lines | verdict ✓ 12.6 s | **no verdict** — 12,723 thinking chars, budget exhausted |
| 230 lines | verdict ✓ 61.5 s | verdict ✓ 36.9 s |

The toy-diff 0.96 above does not generalize; `think:false` is load-bearing
for review work.

**Binary CLEAN-gate contract is a different game.** Replaying the
cli-updates gate payload (whole-file context, reply-exactly-`CLEAN`,
temp 0) over real merged clean commits: **7/7 PASS — gemma 4/4, qwen 3/3.**
Both models gate safely under that contract; severity calibration is where
they diverge.

## 2. Delegation (machine-graded, 16 tasks × 3 trials)

| config | mean score | cross-task contamination |
|---|---|---|
| gemma @ default temp | **0.917** | **0 / 45** |
| qwen @ default temp | 0.732 | 9 / 45 (20%) |
| qwen @ temperature 0 | 0.733 | 3 / 48 (6.2%) |

First-exposure capability is close (qwen 0.896 / gemma 0.958); the served
gap is the contamination.

**Contamination = returning a different recently-seen prompt's completion**
(observed verbatim: a `parse_port` request answered with a redaction
config). By trial, pooled across two full qwen runs:

| exposure | rate |
|---|---|
| trial 1 (prompt novel) | 0 / 30 |
| trial 2 | 8 / 30 (27%) |
| trial 3 | 7 / 30 (23%) |

Review prompts (each embedding a unique diff): **0 / 120 contaminated** —
the failure needs many short, structurally similar prompts, i.e. exactly
the `local_delegate` workload. Temperature 0 reduces the rate but leaves
the score flat: the failures become deterministic rather than absent.
Suspected serving-side cause (**unproven**): q8_0 KV cache + KEEP_ALIVE=-1
prefix-cache collision. Did not reproduce with only 2 alternating prompts
or big/small context churn; needs the many-distinct-task workload.

Shared weakness: counting/aggregating over a 160-line log — qwen 0.33,
gemma 0.33. Don't use either for tallies over long inputs.

## 3. Capability corrections the data forced

- **gemma4:12b-nvfp4 IS a thinking model** — `/api/show` reports
  `['completion','tools','thinking']` and `think:true` yields a real trace.
  An earlier shipped claim to the contrary was asserted, never tested.
- **Temp-0 was predicted (from 2/27 calls) to fix qwen's contamination.**
  The full run falsified the prediction. Both corrections are recorded so
  they don't get re-learned the hard way.

## Decisions taken on this data

1. `local_delegate` default → `gemma4:12b-nvfp4` (PR #32), with local-first
   implicit resolution and capability-driven `think` handling.
2. review-pipeline Qodo step → stays `qwen3.6` + `think:false` (encoded in
   SKILL.md with the numbers above).
3. cli-updates CLEAN gate → stays gemma at temp 0 (validated 4/4; either
   model would work under that contract).
