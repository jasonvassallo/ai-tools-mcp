---
name: choosing-local-model-context
description: Pick the right qwen3.6 context-window tag (32k, 64k, or 256k) for a local_delegate call, and the host it implies. Applies only when explicitly choosing a qwen tag — the tool-wide default is gemma4:12b-nvfp4. Use whenever calling local_delegate with anything other than a trivially small prompt, when a delegate call fails because the prompt exceeded the model window, when choosing the model param explicitly, or when deciding whether a task can run on the always-on 32k endpoint vs the laptop-only 256k one.
---

# Choosing a Context Window for local_delegate

**The default model is `gemma4:12b-nvfp4`, not qwen.** This skill is about the
qwen tags, so it only applies once you've decided to pass `model=` explicitly.
Reach for a qwen tag when the prompt is large or the work is genuinely
code-heavy; leave the default alone for short mechanical tasks, where gemma
scored better (0.92 vs 0.73) and never returned another prompt's answer.
Neither model can be trusted to count or aggregate over long inputs.

The qwen model (`qwen3.6:35b-a3b-coding-nvfp4`, 35B MoE, nvfp4) is served under
three tags that differ **only** in max context window, and the endpoint chain
means the tag you pick also decides **which machine can serve you**.

## The one fact that changes the math

With flash attention + q8_0 KV quantization (how every host here runs), the
KV cache **grows with tokens actually used, not with the window size**. A 5k
token task on the 256k tag costs the same RAM as on the 32k tag. The window
is a *cap*, not a preallocation. So you never pick a smaller tag to "save
memory" on a per-call basis — you pick tags to match **host availability**
and **worst-case bounding on small machines**.

## Tag ↔ host map

| Tag | Window | Served by | Notes |
|---|---|---|---|
| `qwen3.6:35b-a3b-coding-nvfp4` (base) | host's default: **64k** on JVMBPro / **32k** on jvmacmini | localhost (JVMBPro), `ollama-mbp.djvassallo.com` (64k), `ollama.djvassallo.com` (jvmacmini, 32k, always-on) | Default **qwen** tag (gemma4:12b-nvfp4 is the tool-wide default). Window depends on which host answers. |
| `-32k` | 32,768 | JVMBPro only (tag exists there) | Rarely needed — prefer base. |
| `-256k` | 262,144 | JVMBPro only (localhost or `ollama-mbp`) | Not on the mini (32 GB — a full window would exceed the machine). By default, no qwen tag stays resident between `local_delegate` calls; see the explicit `keep_alive` opt-in below. |

## Sizing rule of thumb

Tokens ≈ characters ÷ 3.5 for code, ÷ 4 for prose. Budget = prompt + inlined
files + expected answer + thinking tokens (if `think:true`, add 1–4k).

- **≤ ~28k tokens total** → any tag/host works, including the always-on
  32k mini endpoint. Most "simple/easy coding task" delegations live here:
  a few files, a summary, boilerplate, a focused review.
- **~28k–60k** → base tag, but it must land on a 64k host (JVMBPro local or
  `ollama-mbp`) — the mini would truncate. Multi-file context, long diffs,
  big log analysis.
- **> ~60k** → `-256k`, explicitly. Whole-repo dumps, giant transcripts.
  JVMBPro must be on.

## Constrained machines (e.g. 32 GB Windows desktop, CPU-only, office apps open)

When the host can't fit the 35B qwen (~20 GB loaded), the allowlist is
overridden per machine (`ollama_models` extension setting /
`AI_TOOLS_OLLAMA_MODELS`) with a small local model first — e.g.
`qwen2.5-coder:14b` (q4, ~9 GB; the qwen3-coder line starts at 30B and does
not fit). Routing then works itself out: calls for the small tag run
locally; calls that name a qwen3.6 tag miss the local probe and fall
through the endpoint chain to `ollama-mbp` (64k/256k) or the always-on
`ollama` (32k) endpoint. On CPU-only hosts keep `think:false` (thinking is
slow there) and expect ~4–8 tok/s from a dense 12–14B q4.

## Host-specific etiquette

- **Qwen tags unload after every call by default.** When the caller omits
  `keep_alive`, `local_delegate` sends `keep_alive: "0"` for any qwen tag, so
  no qwen instance is left resident. That is the contamination mitigation: a
  long-lived qwen runner returns *other prompts'* answers on ~15–25% of
  repeat calls, and unloading between calls measured 0/96 contaminated. Other
  models — including the `gemma4:12b-nvfp4` default — still inherit the
  server's `OLLAMA_KEEP_ALIVE`. The cost is latency: every qwen call now pays
  a cold load — ~20 GB for the qwen3.6 35B tags, less for a smaller override
  tag such as the ~9 GB `qwen2.5-coder:14b` above, which the `qwen` prefix
  also matches.
- **Pass `keep_alive` explicitly to keep one warm — an explicit value always
  wins.** `keep_alive: "5m"` is the deliberate opt-out when repeated cold
  loads would dominate the work. Know what you are re-enabling: the measured
  trigger is **many short, structurally similar prompts through one resident
  runner**, and *distinctness is not the protective factor* — 16 genuinely
  distinct delegation tasks came back 0/30 contaminated on first exposure but
  8/30 and 7/30 on re-runs, while review prompts each embedding a unique diff
  measured 0/120. Size and dissimilarity are what protect you. So warm-pinning
  is defensible for a handful of large, unlike calls and unsafe for a batch of
  small similar ones, however distinct their subjects.
- **Don't leave two qwen tags warm on JVMBPro.** Each tag is its own runner
  instance, so a second warm-pinned tag loads a *second* ~20 GB copy of the
  same weights (weights on disk are shared, loaded GPU memory is not). Under
  the default, *sequential* calls leave no resident copy to collide with, so
  tag choice is free — pick purely by the window the task needs. Two caveats:
  `keep_alive: "0"` unloads only when a call *finishes*, so overlapping calls
  against two tags (`background=true` allows up to four in flight) can still
  hold two runners at once; and if you do warm-pin, pin exactly one tag per
  host and reuse it.
- **The mini endpoint (`ollama.djvassallo.com`) is the always-on fallback.**
  It serves only the base tag at 32k. "Always-on" means the host and its
  Ollama server are always up — not that the model stays resident: a
  `local_delegate` call unloads it afterwards like any other qwen tag, unless
  you passed `keep_alive` yourself. If the task fits 32k, it works even when
  the laptop is closed.
- If Ollama returns a context-length error or output is silently truncated,
  step up one tier and retry — never trim the user's input to force a fit
  without saying so.
