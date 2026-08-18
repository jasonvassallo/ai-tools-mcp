---
name: choosing-local-model-context
description: Size the context window (num_ctx) for a local_delegate call and pick the host that can serve it. Use whenever calling local_delegate with anything other than a trivially small prompt, when a delegate call fails because the prompt exceeded the model window, when choosing the model param explicitly, or when deciding whether a task can run on the always-on endpoint vs the laptop-only one. Note the per-context model tags (-32k/-64k/-256k) NO LONGER EXIST — and local_delegate exposes NO num_ctx parameter, so the window is whatever the serving host's OLLAMA_CONTEXT_LENGTH is.
---

# Choosing a Context Window for local_delegate

## What changed (2026-08-17) — read this before trusting any older note

The `qwen3.6:*` tags this skill used to be about **are gone from every endpoint we
can measure.** There are no longer any per-context-window tag variants of anything:
no `-32k`, no `-64k`, no `-256k`.

**And there is no `num_ctx` knob on `local_delegate` either.** Verified against
`mcp_server.py` (2026-08-18, Greptile + gemma4 review finding, confirmed by grep): the
tool's request carries `model`, `messages`, `think`, `stream: false` and (conditionally)
`keep_alive`, and nothing else — no `options` block, so `options.num_ctx` is never sent. **Every `local_delegate` call runs at
the serving host's default window** (`OLLAMA_CONTEXT_LENGTH` on that Ollama server — the
tool description states it: 64k on JVMBPro, 32k on jvmacmini). The window is chosen by
the *operator on the host*, not by the caller per request. Two OTHER tools do set
`options.num_ctx` themselves, and neither is `local_delegate`: `scripts/pr_review_local.sh`
in this repo pins a **16384 floor** (`PR_AGENT_NUM_CTX`) with no diff-based sizing; and
`review.ps1` — which lives in `~\.ollama-qodo\`, NOT in this repository — auto-sizes to the
diff (capped 262144, `LargeCutoffKB=150`).

Both endpoints were probed directly on 2026-08-17. **They do not serve the same
things**, and that is the routing fact that matters:

| Endpoint | Machine | Serves | Availability |
|---|---|---|---|
| `ollama-mbp.djvassallo.com` | MBP (laptop) | `gemma4:31b-nvfp4` (kept warm), `gemma4:12b-nvfp4`, `qwen3.8:27b-nvfp4`, `muse-glimmer:30b-nvfp4-dflash`, `granite3.2-vision:2b`, `minicpm-v` | **may be asleep/closed** |
| `ollama.djvassallo.com` | jvmacmini | **`gemma4:12b-nvfp4` ONLY** (kept warm for the signal openclaw agent) | **always on** |

`gemma4:31b-nvfp4` reports `parameter_size` 31.7B and `context_length` **262144**, and
`/api/show` lists **no** `num_ctx` parameter — the tag pins no window, so the host's
`OLLAMA_CONTEXT_LENGTH` governs `local_delegate` calls (and `options.num_ctx` governs
callers like `review.ps1` that set it).

**The consequence to internalize: the 31b exists on exactly ONE host, and that host is a
laptop.** If the MBP is asleep there is no 31b anywhere — review and long-context work have
no home, and the honest fallback is `gemma4:12b-nvfp4` on the always-on mini (or
your small local model — e.g. `qwen2.5-coder:14b` on a constrained Windows box) with a
note in your reply that you dropped a tier. Never silently
substitute.

**CF Access tokens are per-application.** The `ollama-mbp-cf-access/*` pair reaches only the
MBP app — it returns **403** against `ollama.djvassallo.com`. The `ai-tools-mcp-cf-access/*`
pair reaches **both**. Use that one when you need to probe the mini.

**Always name a model by its BARE tag** — `gemma4:31b-nvfp4`, never
`ollama-mbp/gemma4:31b-nvfp4`. The host comes from the endpoint chain, and Ollama parses a
host-prefixed name as a *model* called `ollama-mbp/gemma4` (measured: 404).

**Verify before you trust.** The tag list written into any skill or config here has been
wrong twice. Query `/api/tags` (creds in Credential Manager) rather than believing this table.

## The one fact that changes the math

With flash attention + q8_0 KV quantization (how every host here runs), the KV cache
**grows with tokens actually used, not with the window size** — in a measurement taken
on these hosts (2026-07, flash attention + q8_0 KV), a 5k-token task on a host configured
for 262144 cost about the same RAM as on one configured for 32768. Treat that as a
**version-specific observation, not a guarantee**: Ollama documents that memory grows
with `OLLAMA_CONTEXT_LENGTH` and scales with `OLLAMA_NUM_PARALLEL`; flash attention and
q8_0 *reduce* KV-cache usage, they do not remove the constraint. So for a *small* prompt
there is little per-call memory reason to prefer a smaller-window host — but re-measure
before relying on it after an Ollama upgrade.

The configured window IS a **memory constraint at the host level**: it bounds the *peak*
KV cache the host must be able to reach, and parallelism multiplies that. This repo's own
history recorded the old `-256k` tag needing several GB of KV cache. So: pick the host by
whether the prompt FITS its window and whether it is awake; and if you are the one raising
`OLLAMA_CONTEXT_LENGTH` on a host, size it to that host's memory, not to the model's
advertised maximum.

## Sizing rule of thumb

Tokens ≈ characters ÷ 3.5 for code, ÷ 4 for prose. Budget = prompt + inlined files +
expected answer + thinking tokens (if `think:true`, add 1–4k).

The `~28k` and `~60k` figures below are **conservative safety thresholds, not the host
limits** (those are 32k on the mini and 64k on the MBP). The gap is deliberate headroom
for the answer plus any thinking — a prompt sized right up to the window leaves nothing
for the model to reply with, and Ollama then truncates the *input* silently.

- **≤ ~28k tokens** → anything works. Most delegations live here: a few files, a
  summary, boilerplate, a focused review.
- **~28k–60k** → `gemma4:31b-nvfp4` on the MBP, whose host default is 64k. Multi-file
  context, long diffs, big log analysis. Will NOT fit the mini's 32k.
- **> ~60k** → does not fit `local_delegate` safely at the current host defaults (no
  per-request window; the MBP host default is 64k) — the tool itself is not the limit, the
  host's `OLLAMA_CONTEXT_LENGTH` is. Either raise `OLLAMA_CONTEXT_LENGTH` on the MBP's Ollama —
  the model advertises 262144 — or route the job through `~\.ollama-qodo\review.ps1`
  (outside this repo), which sets `options.num_ctx` itself, auto-sized to the diff.
  `scripts/pr_review_local.sh` here does NOT help: it pins `num_ctx` at 16384.
  Do not tell a caller to "raise num_ctx" on `local_delegate`; the parameter does not exist.

## think flag

`gemma4:31b-nvfp4` is thinking-capable (`/api/show` lists `thinking`). Send
**`think:false`** for review and delegation work — with thinking on, the generation is
spent in the `thinking` field and the response comes back empty. A small local
coder like `qwen2.5-coder:14b` has
no thinking support at all; the server strips a `think:true` it can't honor and says so.

## keep_alive: which tags stay resident

`local_delegate` defaults an omitted `keep_alive` to `"0"` (unload after the call)
for any tag whose final path component starts with **`qwen`**, case-folded
(`_KEEP_ALIVE_ZERO_MODEL_PREFIXES` in `mcp_server.py`; matching the last path
component means `hf.co/acme/qwen-model` and `Qwen3:latest` are covered too). That
is the cross-task contamination mitigation: leaving a qwen instance resident
measured contaminated answers across repeat calls, and unloading between calls
measured 0/96. An explicit caller `keep_alive` always wins, so deliberate
warm-pinning stays possible.

**This still matters after the retag, for a different tag.** The qwen3.6 tags it
was written for are gone, but `qwen3.8:27b-nvfp4` is present on `ollama-mbp` and
matches the same prefix — so it unloads per call unless you ask otherwise. The
gemma4 tags do NOT match, which is what lets `gemma4:31b-nvfp4` stay warm on the
MBP and `gemma4:12b-nvfp4` stay warm on the mini for the signal openclaw agent.
Do not "fix" that asymmetry without re-running the contamination measurement.

Note `keep_alive: "0"` unloads only when a call *finishes*, so overlapping calls
can briefly hold two runners at once. Each tag is its own runner instance, so a
second warm-pinned tag on the same host loads a second full copy of the weights —
pin at most one tag per host.

## Constrained machines (32 GB Windows boxes, CPU-only)

The 31B does not fit locally on TSDPUR012 / TSLPUR110, and the gemma4 tags are
macOS/MLX builds that cannot even be pulled on Windows (`ollama pull` → 412). The
allowlist is overridden per machine via `AI_TOOLS_OLLAMA_MODELS` with a small local
model first — the one you actually pulled, e.g. `qwen2.5-coder:14b` per the README's
Windows setup (`AI_TOOLS_OLLAMA_MODELS=qwen2.5-coder:14b,gemma4:12b-nvfp4,gemma4:31b-nvfp4`)
— so short mechanical calls run on-device and anything naming a remote tag misses the
local probe and falls through the endpoint chain. Whatever tag you put first MUST be one
`ollama list` shows locally, or implicit calls skip it and go remote. Keep
`think:false` on these boxes and expect ~4–8 tok/s from a dense 12–14B q4.

See the `local-delegate-routing` skill for which model to force per task type.

## When a call fails on length

Two different failures look alike from the outside, and moving to a bigger host fixes
only one of them:

- **Context-window overflow** — Ollama returns a context-length error, or the prompt is
  silently truncated at the *input* side. `local_delegate` cannot raise the window per
  call: move the job to the larger-window host (MBP 64k over the mini's 32k), raise
  `OLLAMA_CONTEXT_LENGTH` on that host, or use `~\.ollama-qodo\review.ps1`.
- **Generation-limit truncation** — the *output* stops early: mid-sentence, mid-list,
  or mid-code-block. Underneath, Ollama reported `done_reason: length` — but
  `local_delegate` returns only `message.content` and **discards `done_reason` and
  `eval_count`**, so through this tool you can only infer it from the shape of the text.
  (Calling Ollama directly, e.g. `review.ps1`, you can read those fields.) Note
  `done_reason: length` is ambiguous on its own: it fires when the `num_predict` cap is
  hit AND when the *remaining* context budget runs out mid-generation. Disambiguate by
  the prompt size — a small prompt that stops early is `num_predict` (a bigger host
  changes nothing: ask for a shorter answer, split the task, or reduce `think`); a
  prompt near the host's window that stops early is the context budget, and belongs in
  the bullet above. `local_delegate` exposes no `num_predict` either. Surfacing
  `done_reason` in the tool's response would make this diagnosable instead of guessed;
  that is a code change, not a doc one.

Never trim the user's input to force a fit without saying so.
