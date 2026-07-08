---
name: choosing-local-model-context
description: Pick the right qwen3.6 context-window tag (32k, 64k, or 256k) for a local_delegate call, and the host it implies. Use whenever calling local_delegate with anything other than a trivially small prompt, when a delegate call fails because the prompt exceeded the model window, when choosing the model param explicitly, or when deciding whether a task can run on the always-on 32k endpoint vs the laptop-only 256k one.
---

# Choosing a Context Window for local_delegate

One model (`qwen3.6:35b-a3b-coding-nvfp4`, 35B MoE, nvfp4) is served under
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
| `qwen3.6:35b-a3b-coding-nvfp4` (base) | host's default: **64k** on JVMBPro / **32k** on jvmacmini | localhost (JVMBPro), `ollama-mbp.djvassallo.com` (64k), `ollama.djvassallo.com` (jvmacmini, 32k, always-on) | Default. Window depends on which host answers. |
| `-32k` | 32,768 | JVMBPro only (tag exists there) | Rarely needed — prefer base. |
| `-256k` | 262,144 | JVMBPro only (localhost or `ollama-mbp`) | **Kept warm/pinned on JVMBPro.** Not on the mini (32 GB — a full window would exceed the machine). |

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

- **On JVMBPro, prefer `-256k` for local calls.** It is the instance already
  pinned warm; calling the base tag there loads a *second* ~20 GB copy of
  the same weights (different tag = different runner instance — weights on
  disk are shared, loaded GPU memory is not).
- **The mini endpoint (`ollama.djvassallo.com`) is the always-on fallback.**
  It serves only the base tag at 32k, pinned warm. If the task fits 32k, it
  works even when the laptop is closed.
- If Ollama returns a context-length error or output is silently truncated,
  step up one tier and retry — never trim the user's input to force a fit
  without saying so.
