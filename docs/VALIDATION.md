# How this recipe validates claims (methodology)

Every throughput, latency, pool-size, and integrity number in this
repo's README and FIXLOG was produced under the same discipline
(hardware facts like cable/drain/MTU notes aside). Reproduce it and
your numbers are comparable to ours; skip any part and they aren't.
The bring-up crash ladders live in `FIXLOG.md`; this file is the
measurement method behind the numbers.

## 1. Read the landmine ledger first

`FIXLOG.md` is the ledger of everything already tried and its outcome —
the crash rungs, the tuning notes, the measured rejections. Before
changing a setting here, check it: a lever with a recorded result
(some tuning notes end in explicit "don't retry" verdicts with numbers)
is settled unless your stack changed (new image lineage, new
checkpoint). One variable per boot; a boot that changes two things
proves nothing either way.

## 2. Harness before boots

Stage a benchmark script on the head node **before** touching any knob,
stdlib-only so it runs anywhere:

- **Anti-contamination**: every prefill-class prompt embeds a fresh
  random nonce. Without this, prefix caching replays your "new" prompt
  and reports cache hits as prefill — we have measured same-payload
  "prefill" at 10–15× its true rate this way.
- **Workload split for acceptance-bound metrics.** With speculative
  decoding, speed depends on draft acceptance, which depends on what
  the model is writing. Always measure separately: a high-draftability
  stream (count-to-N or code), free prose, N-way parallel aggregate
  (4/8/16), prefill at a realistic long-context size, and the
  per-position acceptance histogram from the engine's Prometheus
  metrics (`vllm:spec_decode_num_accepted_tokens_per_pos_total`). A
  single headline number hides all of this; publish the split.
- **Corruption battery next to every speed run**: exact-repeat prompts
  at temperature 0 (any non-ASCII or drift = fail), a strict-JSON
  request, and a nonce'd needle-in-haystack at a prompt length beyond
  whatever was previously exercised. A speed win on a lane that
  corrupts output is not a win.
- **One sacrificial long-prefill request after the server reports
  healthy** — the first long prefill pays ~2× TTFT in JIT compilation.
  It must never land inside a measurement.
- **Quote the engine's own pool** (`GPU KV cache size` from the boot
  log), never an estimate.

## 3. Boot hygiene

Workers first, head last. Drop filesystem caches on every rank before
boot (the driver OOMs during load otherwise). Never `docker restart` —
fresh `docker run` every time. Poll `/health` with a bound generous for
the stack: speculative builds spend 12–15+ minutes autotuning and
priming thousands of kernels before first light, and declaring a hang
at minute 8 throws away a good boot.

## 4. Record as you go

Every cycle's full output is teed to a timestamped file; on failure,
all three ranks' logs are captured before anything else happens. Keep
a running experiment log with, per cycle: the exact config delta, boot
facts (pool, time-to-healthy), **all** the numbers per workload, the
decision, and the trigger — what caused what. End the window with a
summary table and a "do not re-derive" trigger map. Numbers that exist
only in a terminal scrollback don't exist.

## 5. Honest reporting rules

- A repeated-boot jitter band (±10% on the KV pool, ±3–5% on throughput
  samples here) means ranges, not points — and A/B verdicts need
  non-overlapping samples, not vibes.
- Speed settings interact with capture sizes and pool sizes; when a
  faster setting costs memory, publish both variants and the workload
  each suits (see the capture-size landmine in the README; FIXLOG carries the tuning note).
- Where our measurement disagrees with the model card's official
  recommendation, we publish both and say which weights the measurement
  was taken on.
- "Not tested / not claimed" is a section in the README, not a
  footnote. Sustained-load windows, contexts beyond what was exercised,
  and untested quantizations stay unclaimed.

## 5a. The acceptance gate is a mixed battery

A boot is not "accepted" because single-class probes pass. The gate
this recipe actually uses before calling a lane healthy is a **mixed
battery**: interactive streams, chat-class requests, a reranker and an
embedder hitting their services, and ordinary background tenants all
in flight **at the same time**, with host `MemAvailable` sampled
throughout. A lane that only survives isolated benchmarks is not
validated for the two-surface operating mode this recipe targets
(coding harness + chat simultaneously). Build the mixed battery from
the harness classes above and run it beside every adoption decision.

## CSF-build validation (third ladder)

Layered on the same harness; adds format-level proofs beneath the
behavioral battery:

- **Codec round-trip**: decode→re-encode reproduces the published plane
  bytes exactly (bases, selector bits, exception ordering) — the codec
  is re-derived from the loader's own decode, so agreement is a proof,
  not a coincidence.
- **Converter proof**: on a real same-geometry checkpoint, compressed
  planes decode bit-exact to the source scales and every pass-through
  tensor is byte-identical; inventories match the reference layout.
- **Cross-base behavioral check**: fixed temp-0 prompts against the fp8
  and CSF builds — same weights through different quantization paths.
  Byte-identity is *not* expected (different kernels round
  differently); the gate is rephrase-only differences and zero
  non-printable characters.
- **Same-day paired A/B** for any adoption claim: both boots the same
  day, identical settings, one variable, 3-sample non-overlapping bands
  (boot-to-boot pool jitter is ±6–8% — single-boot deltas are noise).
- **Determinism gate**: normalized text (a benign leading-space
  alternation exists under spec decode + prefix caching — see FIXLOG).
