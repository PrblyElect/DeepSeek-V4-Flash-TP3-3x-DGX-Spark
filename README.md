---
license: mit
base_model: Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX
library_name: vllm
tags:
- serving-recipe
- vllm
- dgx-spark
- tensor-parallel
- speculative-decoding
---

# DeepSeek-V4-Flash at TP=3 across 3× NVIDIA DGX Spark — speed & capacity

**TL;DR** — Serve DeepSeek-V4-Flash (284B total / 13B active, native
mixed FP4+FP8) tensor-parallel across three DGX Sparks on a 200G CX-7
RoCE triangle. Two validated launch profiles from one repo — interactive
decode at ~100+ tok/s on code-class streams (speed, DSpark k=5), or a
~7.5M-token KV pool ≈ 7 simultaneous full-1M-context sessions
(capacity) — plus a measured CSF build (fastest single-stream; one
upstream release away). Digest-pinned images, three crash-ladder
FIXLOGs, acceptance gates, and a measured landmine ledger are included.
Bring-up is a handful of commands — see **Quick start** below. Both
profiles are sized to share the boxes: host headroom for co-tenant
services is part of the spec (**Headroom by design**).

**Two validated launch profiles — plus a measured CSF build — for the
abliterated "SuperDeepseek MQ" build of the 0731-lineage checkpoint,
tensor-parallel across three DGX Sparks (GB10) on one CX-7 RoCE
triangle (200G per leg). The two profiles serve from the same container
names and port; you pick per deployment:**

- **SPEED** (default) — karmic-kraken-lineage image with DSpark
  speculative decoding: single-stream decode is draft-acceptance-bound
  and therefore workload-dependent — **~100–108 tok/s on high-draftability
  streams (count/code class), ~45–50 tok/s on free prose** — with
  aggregate climbing to **~200 tok/s at the 24-stream cap**. Nothing is
  installed at container start: the kernels are baked into the image, so
  the boot fetches nothing.
- **CAPACITY** — chthonic-lineage image, zero spec decode, maximum KV:
  **~7.5–7.6M-token pool ≈ 7.2 simultaneous full-1M-context
  sessions** (at `MAX_NUM_BATCHED_TOKENS=4096`, see landmine 14) —
  the largest published pool for this model on any 2–3-Spark
  deployment identified so far. Its runtime dependencies (Ray, the pinned
  b12x build) are installed from a local wheelhouse, prepared once per
  box — the boot itself still fetches nothing.
- **CSF build (compressed-scales)** — the same speed-profile serving
  shape on a CSF-packed copy of the checkpoint: **measured fastest of
  the three** (~5–10% single-stream prose decode over the fp8 speed
  profile in a same-day paired A/B — the +10% reading came against a
  control at the bottom of its band — code-class top-of-band, pool
  parity-plus), with a no-spec capacity-style CSF variant reaching
  ~3.6–4.0M pool at 8192 (~5.6–5.8M at 4096, per landmine 14's
  no-spec rule). Ships as a complete offline toolchain + loader patches
  (`patches-csf/`, `tools/csf/`) with **one prerequisite outside this
  repo**: a b12x runtime with the CSF kernel path — that path landed on
  the b12x master branch 2026-10-06, and with the two documented b12x
  fixes applied it boots and serves, but at roughly half the beta
  build's spec throughput (rung 29); see the CSF section below.

On the evidence gathered here: the capacity profile is the first TP=3 serving of
this checkpoint at full native weight, and the speed profile the first
DSpark-at-TP=3 serving of the 0731-lineage V4 checkpoint (the published
TP=3 DSpark lanes run V4.1 — credited below). The machinery both profiles
build on is credited below and in `docs/FIXLOG.md`.

**Why SPEED is the default.** The default is the profile with the fewest
things that must be true before an engine can start. The speed profile
installs nothing at container start: the b12x kernels and NCCL are baked
into the image, there is no pip step, and there is no upstream fetch on
the boot path. The capacity profile needs one extra, *offline* step — the
wheelhouse pre-flight below — but its boot is equally fetch-free. Pick
capacity when the KV pool is the constraint, not because it boots more
easily.

Prerequisites: three DGX Sparks wired as a full CX-7 RoCE triangle (one
/24 per leg, MTU 9000) plus a management LAN, the checkpoint (~170 GB)
and the digest-pinned image staged on every box, and Docker with GPU
support. The full pinned stack and hardware detail live in the
**Hardware** and **Stack** sections below.

## Quick start

From your workstation, after staging the checkpoint on all three boxes
(one-time, see Bring-up step 1):

```sh
./launch/ds4flash-tp3-up preflight
./launch/ds4flash-tp3-up pull-images
./launch/ds4flash-tp3-up stage-env            # speed profile (default)
./launch/ds4flash-tp3-up start --yes          # workers→head; :8889 in ~12–15 min
# capacity instead:
ENV_SRC=config/cluster.env.capacity ./launch/ds4flash-tp3-up {preflight|pull-images|stage-env|wheelhouse|start --yes}
```

The capacity profile needs one extra step before its first boot —
`wheelhouse` — because it installs Ray and the pinned b12x build from a
local wheelhouse instead of from the network. The speed profile has no
such step.

Then verify the acceptance gates (Bring-up step 3) and read the
landmines before your second boot. Full detail below.

Validated over a two-day first-light window (its soak/boot counts
predate the detailed FIXLOG record): the capacity profile through 9/9 clean cold
boots of the eager → PIECEWISE → FULL cudagraph ladder; the speed profile
through the ten-rung bring-up ladder (FIXLOG rungs 14–23), the acceptance
battery, and a 10-minute 120-request soak. Every headline number below was
measured on this hardware; the fp8 profiles' numbers on the pinned images,
and the CSF ladder's on the pinned image plus this repo's port kit as a
dev overlay (a *released* CSF-capable runtime is the missing piece — CSF
section).

## The numbers (measured on this hardware, these images)

| Metric | Speed profile | Capacity profile |
|---|---|---|
| Single-stream decode | **~100–108 tok/s** count/code-class · ~45–50 free prose (acceptance-bound: 74%→11% across k=5 draft positions on prose) | **~29–30 tok/s** (zero spec decode; re-verified live at 4096) |
| Aggregate decode | ~88–98 @ 4 · ~140–155 @ 8 · ~176 @ 12 · ~186 @ 20 · **~199–208 @ 24-way** (best-day pair; ~199–201 at the shipped gmu 0.79 — the `max_num_seqs` cap) | **~150 tok/s @ 16-way** (3-repeat stable, same harness class as the speed numbers; older 4/8-way figures were harness-ambiguous and retired) |
| Prefill | ~1.0–1.06K tok/s @ 38–43K prompt | **~1.5–1.6K tok/s** @ ~38K prompt (the 4096 lever; was ~0.9K at 8192) |
| KV pool (fp8 KV, native 1M ctx, 24 seqs) | ~2.3–2.6M ≈ **2.2–2.5 full-1M sessions** at gmu 0.79 (±6–8% boot-to-boot jitter; capture-48 variant: ~2.7–2.8M ≈ 2.6–2.7 — see landmine 15) | **~7.5–7.6M at gmu 0.80** (two-boot verified within 1%; ±10% boot-to-boot jitter class) ≈ **~7.2 sessions** (~29–30 @ 256K) |
| Cold boot to serving | ~12–15 min (spec autotune + kernel priming) | ~7–10 min (Ray bring-up) |
| Long-context retrieval | needle-in-haystack correct at ~21–22K and ~48K tokens | needle correct at 21K and ~59K |
| Output integrity | strict-JSON valid; temp-0 exact repeats clean; 10-min soak, 120 requests, **0 corrupted** | strict-JSON byte-valid; 0 CJK/corruption incl. FULL graphs; 15-min mixed soak, ~340 rounds, **0 errors** |
| Host headroom | head box ~6 GiB steady, floor ≥5 GiB through a 500K-token cold prefill; ~5–7 GiB co-tenant budget on the head, more on workers | head box ~4 GiB steady, floor ~3.2 GiB through a 500K-token cold prefill; ~2.5 GiB co-tenant on the head, 11–12 GiB on workers |

The speed profile's decode numbers are the honest workload split: DSpark
accelerates streams it can draft well (code, counting, structured output —
where per-position draft acceptance stays high) roughly 3.5× over this
lane's no-spec baseline, and free prose much less. If your traffic is
prose-heavy and latency-critical, the capacity profile at ~29–30 tok/s may
be the more predictable choice.

Prefill is the honest weak axis on both profiles — see the comparison
below for what other lanes publish, and the roadmap for the known levers.

### Headroom by design — the profiles share the boxes

These are working-stack recipes, not whole-box benchmarks. The shipped
defaults sit deliberately below the measured ceilings so the same three
Sparks keep carrying the rest of a live stack beside the lane —
rerankers, embeddings, document processing, TTS/STT, web chat
frontends. The table's "Host headroom" row is a budget, not a leftover:
at the shipped pins the head box keeps ~5–7 GiB (speed) / ~2.5 GiB
(capacity) with workers at 11–12 GiB, and the measured head-box floor
through a 500K-token cold prefill is ≥3.1 GiB (landmine 17). Spend it
deliberately: pushing `GPU_MEM_UTIL` past the shipped pins trades the
budget away fast (the speed profile loses ~14% of its 16-way aggregate
at 0.80 — FIXLOG), and every gmu ceiling in the FIXLOG was measured, not
guessed. If a box must carry heavier co-tenants, step `GPU_MEM_UTIL`
down before touching the pool pins — the FIXLOG ladders record what
each step buys.

### Not tested / not claimed

Multi-*hour* soaks; the spec-decode + CUDA-graph corruption windows the
community documented under sustained load ([eugr
#363](https://github.com/eugr/spark-vllm-docker/issues/363) garble
lineage; [#391](https://github.com/eugr/spark-vllm-docker/issues/391)
reports a sparse-MLA Xid only after ~32 h on an older nightly) — the
battery and 10-minute soak here are clean, sustained windows are not proven;
contexts above ~48K *exercised* on the speed profile / above 59K on
capacity (1M is configured and pool-proven, not output-proven); any
packaging except the native mixed FP4+FP8 (re-quants untested);
nvfp4 KV (deliberate — see "Full quality"); the thirteen image-baked
engine-file digests in `launch/tp3-kit/engine-identity.json`, which the
manifest ships unfilled — the gate checks the build stamp and the four
bind-mounted patch digests and reports the rest as `NOT PINNED` until a
reader fills them against a live container (see "Proving which engine
answered").

## The CSF build (fastest measured; one upstream release away)

The third ladder (FIXLOG rungs 24–29) ports the CSF compressed-scales
serving path to TP=3 and races it against the fp8 speed profile at
identical settings (single-variable base swap):

| | fp8 speed profile | **CSF speed build** |
|---|---|---|
| Single-stream decode | ~45–50 prose · ~100–108 code-class | **~48–50 prose (~5–10% paired-A/B win over fp8 speed; the +10% reading came against a bottom-of-band control) · ~105–107 code-class** |
| KV pool | ~2.3–2.6M | ~2.4–2.6M (parity-plus; gmu 0.80 measured cliff-free on this base) |
| Prefill | ~1.0–1.06K tok/s | top of the same band |
| Quality | battery clean | battery clean; cross-base text diffs are same-meaning rephrase class only; ~370K-token cold prefill coherent |

The geometry that wins is the 2048→2112 pad (64/rank): it unlocks
b12x's compact W4A8 decode kernels. The wider 2304 pad boots on
unpatched runtimes but pays ~+12 GB phantom expert weights plus ~+12.5%
MoE compute and measured slower on every axis — see
`patches-csf/README.md` for both, and for the small b12x compile
wait-race fix the 2112 geometry needs (root-caused; fix intent
documented).

**What you can run today vs. what waits:** the converter
(`tools/csf/`) runs on stock PyTorch today and round-trip-validates its
output; serving needs a b12x build with the CSF kernel path — that path
landed on the b12x master branch on 2026-10-06, so the next release or
Spark nightly that bakes it completes the chain (`patches-csf/README.md`
has a one-line readiness check, and the wait-race and `_instantiate`
retry fixes it documents are both required for the 2112 geometry). You can also **extract a working
CSF-capable b12x from the author's public beta channel today** — a
digest-pinned, copy-paste procedure in `patches-csf/README.md` ("Getting
a CSF-capable b12x today"). Master alone is still **not sufficient** —
and this is now measured, not cautious: master plus the two documented
b12x fixes (compile wait-race + `_instantiate` retry) boots cleanly at
pool parity, but its draft/verify path
delivers roughly **half** the beta build's spec throughput — code-class
decode lands at ~half the beta band, prose at ~three-quarters (FIXLOG
rung 29). A release must also match the beta lineage's draft/verify
kernels; until one does, verify any candidate runtime with a small
boot, not just the symbol check. The k-depth answer
carries over from the fp8 profile: k=5 stays optimal — k=4 measured
faster prose but −10–11% on code-class streams (deep acceptance;
measured on both bases), closing
the spec-depth axis at {4, 5, 10}.

## Which profile for which job

- **Interactive coding, agent loops, tool-call round-trips** → **speed**.
  Spec decode pays best at low–medium concurrency: 80–90 tok/s streaming
  makes model latency roughly match a fast remote API.
- **Deep research, many parallel huge-context sessions** → **capacity**.
  ~7.2 full-1M windows vs 2.2–2.5 (capture-48 speed variant: 2.6–2.7);
  also the simpler stack: fork-native
  TP padding, one small dynamo patch, no spec-decode variables.
- Aggregate throughput favors speed at interactive depths (~90–155
  across 4–8-way); capacity holds ~150 @ 16-way — per-stream there is
  ~9.5 tok/s, batch class either way.
- Switching is one env file (`ENV_SRC=…`) + re-stage + lane reboot — same
  containers, port, served names; clients don't change.

## Full quality, deliberately

- **Full native weight, no re-quant.** The 0731 checkpoint ships in its
  trained/exported packaging — FP4 experts with FP8/BF16 on the sensitive
  tensors (~160 GB class) — and that is what this recipe serves. Every
  further step down (EXL3 3.5bpw TP=3 at 51.5 tok/s, GGUF Q2–Q4) is a
  lossy re-quant; NVIDIA's own experts-only 4-bit conversion table for
  this family lands within ±1.5 benchmark points of source, which is
  exactly why *staying at the native format* is the quality play — and
  why the third box exists: it spends the room on weights + KV instead
  of forcing a smaller quant.
- **Why not V4.1-Flash?** It is the stronger card on paper (GPQA-D 90.9
  vs 88.1, Terminal-Bench 2.1 90.6) but weighs ~510 GB natively — it
  does not fit 3× ~122 GiB without aggressive community re-quants whose
  quality is undocumented. Native 0731 now; V4.1 when a credible
  sub-300 GiB packaging exists.
- **fp8 KV, not nvfp4 — now measured, not just cautious.** An A/B of
  `nvfp4_ds_mla` on this stack: the pool came out *smaller* than fp8
  (this image's V4.0 record path is fatter per token; the lean 416-byte
  writer lives in another lineage) with speed identical — and the
  community separately documents a sustained-context garble class for
  4-bit KV on these checkpoints. fp8 wins on both axes here.
- **Abliterated weights are a choice, not a default**: the recipe takes
  any `deepseek_v4` checkpoint in this packaging; run the stock release
  if you prefer refusal behavior matching the model card. The validated
  abliterated build publishes its own regression evidence (refusal
  97.9%→4.2% with tool compliance 100%, capability minimum ≥ parent,
  matched-decode ratio 1.001× vs parent, and a documented
  1,028,621-token accepted prompt with needle retrieval) — pick your
  weights with your eyes open, both ways.
- **Sampling**: the official model card recommends temperature 1.0 /
  top_p 1.0 for the stock checkpoint. On the abliterated build, stray CJK
  leakage was measured at temp 1.0, so this recipe's clients pin
  temperature ≤ 0.7 / top_p 0.95 (0.6/0.95 is the sweet spot for
  thinking-mode work). If you serve stock weights, try the official
  setting first; on this build, don't. No repetition/frequency/presence
  penalties on either — they gut draft acceptance (speed) and add
  nothing measurable (capacity).

## Thinking mode (read this before your first request)

The speed profile's `deepseek_v4` reasoning parser **defaults thinking
ON**. Two consequences the first request will teach you otherwise:

1. `max_tokens` is consumed by `reasoning_content` first. Budget for it,
   or disable thinking when you don't want it:

```sh
curl http://HEAD:8889/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ds4flash",
  "messages": [{"role":"user","content":"Ship a fix."}],
  "max_tokens": 1024,
  "temperature": 0.6, "top_p": 0.95,
  "chat_template_kwargs": {"thinking": false}
}'
```

2. Keep thinking **on** for hard reasoning and research synthesis — it is
   the quality mode of this family. The card defines discrete effort
   modes (Non-think / Think-High / Think-Max, exposed as the
   `reasoning_effort` template kwarg — the canonical recipes pin
   `high` server-side) and **Think-Max requires a ≥384K context
   window** — one reason this recipe pins 1M `--max-model-len` even when
   your prompts are short: it keeps the model's deepest reasoning mode
   reachable. Turn thinking off for latency-bound coding loops where the
   model would otherwise burn your token budget re-deriving intent it
   can answer directly.

Both served names (`ds4flash`, `deepseek`) accept the same requests;
tool calls work through the `deepseek_v4` parser with
`--enable-auto-tool-choice`.

## Working with long contexts (coding & research)

- **TTFT expectations** (speed profile): a ~22K-token prompt costs ~22 s
  cold and ~1.0K tok/s thereafter; the *first* long prefill after boot
  pays ~2× TTFT in JIT — the launcher pre-warms with a tiny + ~15K
  request after `/health` so no real user eats it (`prewarm` re-runs it
  idempotently).
- **Keep system prompts and repo context stable per client.** Prefix
  caching makes a replayed ~15K prefix answer ~17× faster (measured
  0.9 s vs 15+ s); `--prefix-cache-retention-interval 4096` keeps recent
  prefixes warm. For repo-scale coding, front-load the stable repo
  material, keep the volatile question last.
- **Concurrency**: 1–4 interactive streams sit in the 80 tok/s class;
  8-way aggregate is ~140–155 tok/s. On capacity, 16-way aggregate is
  ~150 tok/s with full-1M headroom per stream.
- **Exercise your real context length before trusting it.** Retrieval was
  validated at ~21–22K (speed) and 59K (capacity); 1M is configured and
  the pool math holds, but output quality at 1M is unproven here.

## vs. two Sparks (TP=2) and other TP=3 lanes — honest scoreboard

- **Best published 2-Spark lane** ([tonyd2wild's DSpark/NVFP4-KV
  recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark),
  current head): 2.79M-token pool at gmu 0.85 (2.04M at the June
  checkpoint), 84.3 tok/s peak / 80.1 count-benchmark / 67.6 mean, ~182
  tok/s at 6-way, TTFT 4.1 s @8K → 29.5 s @100K (1.5–2.6K tok/s
  prefill). The speed profile: **count/code-class decode parity-plus
  (~100–108), pool parity in its capture-48 variant (~2.7–2.8M at fp8
  KV vs their nvfp4)**, and half the boxes' worth of prefill still
  ahead (1.0–1.06K vs 1.5–2.6K). Their lane also carries the
  garble-fallback note that motivated the fp8-KV choice here.
- **Best published 3-Spark pool/throughput** ([christopher_owen's DS4.1
  native TP=3](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438),
  final r5o): 2,845,543 tokens at 512K ctx (5.4 full windows), 64.8
  tok/s code / 78.5 JSON single-stream, 171–196 @ 8-way, prefill flat
  ~3.8K tok/s to 183K. The speed profile: **faster single-stream decode
  on draftable streams (~100–108 vs 64.8–78.5), 16-way aggregate in the
  same class (~170 vs 171–196)**, pool ~2.3–2.6M @ 1M ctx (his 2.85M @
  512K — the capture-48 variant matches it at 2× the context setting),
  and **~4× slower prefill** — his lane runs sequence-parallel prefill
  patches and UVM driver-level memory work deliberately left out here
  (outside NVIDIA's validated config). The capacity profile:
  **~2.6× his pool** at well under half the per-stream speed.
- **Lossy-quant TP=3** ([EXL3 3.5bpw](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/EXL3-TP3.md)):
  51.5 tok/s — a different weight class, see "Full quality, deliberately".
- **What a 4th box buys** ([MiaAI-Lab's SGLang TP=4
  lane](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)):
  87.7 tok/s @c1, ~343 @c16, prefill 4–5.9K, 8M-token pool — on a fourth
  Spark and a different engine.

Net: with three boxes you get **count/code-class decode at ~100+ tok/s
with 16-way headroom** (speed profile), **~2.7–2.8M fp8-KV tokens at 1M
context when you trade the 16-way ceiling back for pool** (capture-48
variant), or **~7.5M tokens / ~7.2 full-1M sessions** at well under half
the speed (capacity). Prefill is where the frontier still is — nobody's 3-box
prefill matches the best patched lanes, and neither does this profile.

## Hardware / network this was proven on

- 3× DGX Spark (GB10, sm_121, ~122 GiB usable unified memory each)
- Full CX-7 triangle: three QSFP cables, one /24 per leg
  (e.g. 10.77.0.x / 10.77.1.x / 10.77.2.x — whatever your mesh uses), RoCEv2 GID
  index 3, MTU 9000
- Management LAN for engine bootstrap (OOB), NCCL pinned to both
  on-board HCAs per rank (`rocep1s0f0,rocep1s0f1`); the twin-RoCE
  interface trick (`roceP2p1s0f*` alongside `rocep1s0f*`) is a community
  fabric lever not A/B'd here — see roadmap.

## Stack (pinned, per profile)

| Component | Speed (default) | Capacity |
|---|---|---|
| Container | `eugr/spark-vllm-b12x:nightly-20261004` @ `sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b` | `ghcr.io/tonyd2wild/vllm-m3-chthonic:nccl230u1` @ `sha256:cdeae56d23d0d3f35a21c67c9f3e47979725a1ef9cd65faa6134b91871497be2` |
| vLLM fork | karmic-kraken lineage, baked — distribution version `0.1.dev21553+gab86b7073.d20261004` (the commit `gab86b7073` plus the nightly's packaging date) | `local-inference-lab/vllm` `dev/chthonic-consecration`, baked — distribution version `0.11.2.dev279+chthonic.consecration.b12x.cu132` |
| b12x kernels | 1.5.0 baked | pinned commit `08e980c303b0b6291700a6b85aa09aa874fc27cb`, installed from the offline wheelhouse (pre-flight, no boot-time fetch; baked 0.20.0 is API-older) |
| NCCL | 2.32.x baked (no shims to fight) | 2.30.7 (image-built for sm_121; the boot banner must say 2.30.7, never the baked 2.30.4 shim) |
| Executor | `mp` (`--nnodes 3 --node-rank`), no Ray | Ray 2.55.1 + V2 executor, plasma store capped 1 GiB/node |
| TP=3 pad | 3 bind-mount patches (`patches-karmic/`) | fork-native virtual-TP + `patches/tilelang.py` for compile |
| Spec decode | DSpark k=5 probabilistic | none |

Checkpoint (both): any `deepseek_v4` checkpoint in the native mixed
FP4+FP8 packaging; validated with `Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX`
@ `2a7dd6a12d46f752e290c6255140a42600897864` (48-shard repack — `EXPECTED_SHARDS` assumes it).

The shard count is per-repository, and the three candidates do not
agree: `deepseek-ai/DeepSeek-V4-Flash` ships **46** shards,
`deepseek-ai/DeepSeek-V4-Flash-0731` ships **48** (it is the
`base_model` tagged on the pinned checkpoint above), and the pinned
repack ships **48**. Set `EXPECTED_SHARDS` to the one you stage.

## Bring-up

1. Stage the checkpoint on **all three** boxes (identical path), e.g.:

```sh
hf download Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX \
  --revision 2a7dd6a12d46f752e290c6255140a42600897864 \
  --local-dir /home/YOURUSER/models/superdeepseek-v4-flash-abliterated-mq-2xdgx-2a7dd6a12d46
# then create the preflight marker the helpers gate on:
echo '{"source":"huggingface","repo":"Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX"}' \
  > /home/YOURUSER/models/superdeepseek-v4-flash-abliterated-mq-2xdgx-2a7dd6a12d46/_DOWNLOAD_COMPLETE.json
```

   Staging a different DeepSeek repo? The shard count is per-repository —
   `deepseek-ai/DeepSeek-V4-Flash` ships 46 shards, and
   `deepseek-ai/DeepSeek-V4-Flash-0731` ships 48 — so set
   `EXPECTED_SHARDS` to the count you actually staged, or repack to 48
   to match the validated layout.
2. Edit the profile env: `config/cluster.env` (speed, default) or
   `config/cluster.env.capacity`, then run the **Quick start** sequence
   above (`preflight` → `pull-images` → `stage-env` → `start --yes`;
   the capacity profile adds `wheelhouse` before the first boot).

3. Acceptance gates. Speed: the two boot-log pad lines
   ("Padded DeepSeek V4.1 attention for TP3: heads 64 -> 72 …" and
   "Padded DeepSeek V4 MoE intermediate …: 2048 -> 2112"), DSpark enabled,
   `/v1/models` answers, a thinking-off generation probe is coherent.
   Capacity: the virtual-TP pad lines, `NCCL version 2.30.7`, the same
   probes. Both: `Maximum concurrency` / `GPU KV cache size` lines give
   the honest pool.

### Boot-time installs are gone (capacity profile)

The capacity profile used to install two things *while the container was
starting*: Ray from PyPI, and a pinned b12x build straight from a GitHub
URL. That put a network fetch on the boot path — the boot depended on
PyPI and on an upstream host being reachable, and on whatever those
served that day.

Both fetches are now off the boot path:

1. **Pre-flight, once per box.** `launch/prepare-wheelhouse.sh` fetches
   the b12x source at the pinned commit (`B12X_COMMIT` in
   `config/cluster.env.capacity`), verifies that commit id, builds the
   wheel *inside the pinned image* (so the wheel matches the target
   platform and interpreter), downloads the Ray wheels, and records every
   file's sha256 in `wheelhouse.json`.
2. **Boot.** The container mounts that directory read-only and installs
   with pip's index switched off (`--no-index --find-links=/wheelhouse`),
   but only after `launch/tp3-kit/verify-wheelhouse.py` has re-checked
   every sha256 and the pinned commit. A wheelhouse that was edited or
   truncated fails the boot instead of half-installing into it.

```sh
# once per box, from the workstation (all three boxes):
ENV_SRC=config/cluster.env.capacity ./launch/ds4flash-tp3-up wheelhouse
```

The network is needed at pre-flight time, never at boot. The speed
profile has no wheelhouse step at all: it installs nothing.

### Proving which engine answered

A digest-pinned image says which manifest was pulled. It does not say
which **bytes** ran: the recipe bind-mounts patched engine files over the
image's own, and the capacity profile installs a pinned b12x build on top
of the baked one. `launch/tp3-kit/` is the kit that answers the question
from inside the container, before the engine starts:

- `engine-identity.json` — the manifest: a build stamp (the engine's own
  package version) and the sha256 of the engine-critical files.
- `boot-gate-identity.py` — the in-container gate. It runs before the engine
  starts, so a mismatched engine fails the boot instead of answering requests.
- `make-engine-identity.py` — the generator that fills the manifest from
  a container that is already up.

**What is filled in by default: 6 of the 19 claims.** The manifest ships
PARTLY FILLED, because a claim can only be pinned truthfully if its bytes
are knowable without an image:

- both **build stamps** are pinned — the distribution version string the
  pinned image reports (`0.1.dev21553+gab86b7073.d20261004` for the speed
  image, `0.11.2.dev279+chthonic.consecration.b12x.cu132` for the capacity
  one). It is the *distribution* version, not the commit string: a nightly's
  version carries its packaging date as well as its commit id, and the gate
  compares the string exactly.
- the **four bind-mounted patch digests** are pinned — those bytes ship in
  this repository, so their sha256 is computable at the desk:
  `patches-karmic/config.py`, `patches-karmic/v4-attention.py`,
  `patches-karmic/v4-model.py` and `patches/tilelang.py`.
- the **thirteen image-baked engine files** are still `<FILL-IN: ...>`
  markers: their bytes live inside the image, and an invented digest is
  worse than no digest.

So the gate is no longer a gate that cannot fail: with the shipped
manifest it verifies the build stamp and the four patch digests, and says
`NOT PINNED` for the thirteen it cannot check without an image.

**What the gate reports.** The default run is the boot path's run — a
claim whose value is still a `<FILL-IN: …>` marker is reported
`NOT PINNED` and does not fail, so a fresh clone boots. An engine-file
claim whose value is not a full lowercase sha256 is a **FAIL**, not a
`NOT PINNED`; a build stamp is pinned by holding a concrete version
string, and the two build stamps are reported `OK`. `--strict` turns every
unpinned claim into a failure, so a strict run can never pass on a
manifest that pins nothing, and `./launch/ds4flash-tp3-up identity` is
exactly that strict run. A default `ENGINE-IDENTITY PASS` that still
lists unpinned claims is **not** a verification: read the count in
parentheses on the PASS line.

Filling the rest needs a container that is already up, and is one command:

```sh
# on the box that owns the container, with the lane up:
python3 launch/tp3-kit/make-engine-identity.py --profile speed --container ds4flash-tp3-head --write
# re-stage the kit so the boxes carry the filled manifest, then ask again:
./launch/ds4flash-tp3-up stage-env
./launch/ds4flash-tp3-up identity
```

`--profile` is `speed` or `capacity` — the two lineages ship different
trees, so each has its own section and its own site-packages root. The
gate hashes the files **as they are inside the container**, so the
bind-mounted TP=3 patches are part of the claim: change a patch and the
gate fails until the manifest is regenerated. A re-pinned image fails
the gate until it is re-audited — that is the intended adoption gate,
not a bug.

The capacity profile's tilelang digest is the recipe's own patch, and the
capacity profile mounts it only when it has been staged. A gate failure
on that one file means the patch was not staged (`./launch/ds4flash-tp3-up
stage-env` stages it) or was overwritten: a container holding the image's
own copy runs a different routing than the recipe audited, and that is
worth failing on.

### Reading the container command without starting it

Both node helpers build the container command in **one** place — a
`run_args` array — and then execute exactly that array
(`docker run "${run_args[@]}"`). `PRINT_CMD=1` prints that array instead
of running it: one argument per line, then the same command as a single
copy-pasteable line. It starts no container, removes none, and touches
no GPU.

```sh
# on the box you are about to start, with that box's own staged env.
# stage-env installs the helpers under ~/.local/libexec/ds4flash-tp3/,
# which is NOT on PATH by default — use the full path:
PRINT_CMD=1 ~/.local/libexec/ds4flash-tp3/ds4flash-tp3-spark start         # capacity profile
PRINT_CMD=1 ~/.local/libexec/ds4flash-tp3/ds4flash-tp3-karmic-spark start  # speed profile
```

This is the check for "did every value in `config/cluster.env` actually
reach the container": the printed command *is* the command the box would
run — the image digest, the mounted patches and wheelhouse, the engine
env, the serve flags — not a copy of it that can drift from it.

Run it **on the box being started**, with that box's own environment set
the way the start path sets it. The role is derived from `hostname -s`
against `HEAD_HOST`/`W1_HOST`/`W2_HOST` in the staged env file, so the
printed `--node-rank`, `--master-addr`, `--node-ip-address` and container
name are the ones *that* box would use; the env file itself is the same
one the boot sources. The two profiles differ in more than the image:
the capacity command mounts the wheelhouse and the TP=3 padding patch,
and its `-c` string starts Ray and the offline pip install; the speed
command mounts the three `patches-karmic/` files, and its `-c` string is
the identity gate followed by `vllm serve`.

### Where `--block-size 256` comes from

256 is not inherited from a sibling model: it is the DS-V4-Flash
lineage's own answer, in two independent places in the engine source:

- the model's own sparse-MLA backend (`vllm/models/deepseek_v4/sparse_mla.py`)
  declares `256` as its supported kernel block size, and the B12X
  attention backend this profile selects inherits that declaration;
- the sparse-attention indexer backend
  (`vllm/v1/attention/backends/mla/indexer.py`) declares `256` too.

The engine's block-size policy (`vllm/platforms/interface.py`) derives
the block size from the resolved backend **only when the reader does not
pass `--block-size`**; passing it explicitly makes the request visible
in the command line instead of implied. The flag therefore restates what
the model asks for.

The sibling V4.1 lineage answers differently — its own backend and its
indexer declare a smaller block size on this GPU generation — so "the
other model uses 256" would not be an argument here. The value is pinned
per profile in the env file (`BLOCK_SIZE`, next to `TENSOR_PARALLEL_SIZE`).

Stated plainly: this is read out of the engine source of the pinned fork
lineage, not measured on a running lane. It does not move any number in
this README, and changing `BLOCK_SIZE` changes the KV geometry — and
therefore the pool — so leave it alone unless you are re-measuring.

### Short prompts under speculative decode

Upstream vLLM tracks a prefill that can be misread as a uniform-decode
batch: when a request's scheduled token count happens to equal
`uniform_decode_query_len × num_reqs`, the prefill can be dispatched
into the CUDA graph captured for speculative decode
([vllm#53051](https://github.com/vllm-project/vllm/issues/53051)). The
trigger is a prompt of exactly `uniform_decode_query_len` tokens, so a
short prompt is enough to reach it.

`launch/tp3-kit/spec-short-prompt-check.py` probes exactly that shape.
It resolves `uniform_decode_query_len` — from the running engine's own
command line, from a speculative config you pass, or from an explicit
`--query-len` — builds a prompt of exactly that many tokens using the
server's own tokenizer endpoint, sends it at temperature 0, and checks
that the answer is non-empty, printable, not degenerate, and — when the
prompt carries a marker — echoes that marker back.

```sh
# on the box, with the lane up:
python3 launch/tp3-kit/spec-short-prompt-check.py \
  --base-url http://127.0.0.1:8889 --model ds4flash \
  --container ds4flash-tp3-head
```

It is a read-only probe — one request. Run it in a boot window against
a lane being accepted, not against one carrying traffic.

What this profile already does about it: the speed profile selects the
model runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), and in the pinned fork
lineage that runner classifies a batch as uniform decode only when the
batch contains no prefill — so the shape alone does not misdispatch
there. Upstream filed the issue against the older runner (the report
carries an `mrv1-only` label). Both statements are read out of the
pinned fork's source, not measured on a lane; the check ships so the
property is *verified* on the engine build actually in use rather than
inferred from a flag, and so an engine swap is caught.

## The landmines (read before booting)

Full detail per rung in `docs/FIXLOG.md`. Tagged **[S]** speed,
**[C]** capacity, **[B]** both.

1. **[B] Pass `--model` explicitly** (the capacity helper does; the
   speed image's `vllm serve` takes the path positionally). The capacity
   image's api_server ignores a bare positional model path and silently
   boots `Qwen/Qwen3-0.6B` — which then fails TP divisibility (16 % 3)
   and sends you debugging a phantom model for hours.
2. **[S] The three `patches-karmic/` mounts are mandatory**, not an
   optimization: without them the boot dies at head divisibility
   (64 % 3), the MoE config assert (2048 % 3), or DSpark config
   validation — three different crashes, one patch set.
3. **[C] The b12x build must be the pinned commit** — the baked
   0.20.0 wheel lacks `B12XMHCScratchPlan.bind(expected_m=…)`. It is now
   installed from the offline wheelhouse (pre-flight, not boot): run
   `ENV_SRC=config/cluster.env.capacity ./launch/ds4flash-tp3-up wheelhouse`
   once per box. The boot refuses to install from a wheelhouse whose
   sha256s or pinned commit do not match what the pre-flight recorded.
4. **[C] Route DS4 projections into b12x**: `VLLM_USE_B12X_WO_PROJECTION=1`,
   `VLLM_USE_B12X_MHC=1`, `B12X_MOE_FORCE_A8=1` — else DeepGEMM's
   o_proj path asserts on the padded TP=3 shapes.
5. **[C] `patches/tilelang.py` is what unlocks cudagraphs** (a C++
   probe called inside the dynamo-traced region kills compile); eager
   (~6.4 tok/s) → FULL (~27–28 at bring-up, 8192-era) is entirely this patch + compile mode.
6. **[C] Override the image's baked NCCL poison** (`LD_PRELOAD` →
   2.30.7, empty the shim paths, `NCCL_IB_DISABLE=0`) — the image bakes
   `=1` + a 2.30.4 shim. The speed image has no such fight.
7. **[C] Ray, not mp** on that fork (`collective_rpc` follower assert on
   mp; V1 executor can't pickle the DS4 config; **V2 works**). The speed
   profile is the reverse: **mp, not Ray**.
8. **[B] One GPU owner per box.** Both helpers refuse to start beside
   any resident model container; the same `docker rm` hygiene applies
   after crashes (fresh `docker run` every boot, never `docker restart`).
9. **[S] A worker "collective_rpc should not be called on follower node"
   assertion is BENIGN** on mp+spec — read whether `Worker_TP` completed
   before diagnosing. (Same message, real dead end, on the capacity
   image's mp path.)
10. **[S] Spec boots take ~14 minutes** (autotune + ~2,800-kernel
    priming). The launcher polls `/health` to 25 min; don't declare a
    hang before 15.
11. **[S] `--load-format safetensors`** — the default loader rejects
    stock safetensors shards ("InstantTensor only supports pytorch
    format").
12. **[S] `B12X_W4A8_TINY_DECODE=0` is correctness-critical** (1 omits
    the SwiGLU clamp on this model class).
13. **[S] Thinking defaults ON** in this build's parser — see the
    thinking section, or your first low-`max_tokens` request looks
    "broken".
14. **[B] `MAX_NUM_BATCHED_TOKENS` has two truths — set it by profile.**
    On **no-spec profiles (capacity, and any capacity-style variant):
    4096** — measured in both directions from 8192: the halved prefill
    activation workspace converts to KV (+40–54% pool on the capacity
    profile), prefill gains ~1.7–2.2×, and concurrent decode during a
    big prefill gets ~2.6× fairer (chunked prefill at 8192 starves
    decode on this stack). On **spec-decode profiles (speed): 8192** —
    at 4096 the draft/verify pipeline starves inside the smaller token
    budget and code-class decode halves (the engine warns at boot about
    draft-token slots; believe it — and halving `max_num_seqs` does not
    recover it; measured both ways). 16384 remains a trap on both
    (speed: 0731 `sparse_mla` metadata crash; capacity: ~1.9M pool
    burned for zero prefill gain).
15. **[S] `--max-cudagraph-capture-size` is a cliff, not a dial** (FIXLOG
    tuning notes): at 48 the lane collapses past 8 concurrent streams
    (16-way measured at 14 tok/s); 144 = full 24-seq coverage and costs
    ~14% of the pool. The helper ships 144.
16. **[B] Memory hygiene between boots**: `sync; echo 3 >
    /proc/sys/vm/drop_caches` on every rank or the driver OOMs during
    load; reboots revert CX-7 MTU to 1500 (re-apply 9000); a NIC stuck
    at ~13 Gb/s or a GPU latched <1 GHz is fixed by cold power-drain
    (~90 s unplugged), not a warm reboot.
17. **[B] `GPU_MEM_UTIL` is profile-specific — don't copy 0.80 onto the
    speed profile**: A/B'd live with same-day paired boots, the speed
    profile at 0.80 loses **~14% of the 16-way aggregate** (the cliff
    sits between 0.79 and 0.80 on this image; single-stream, 4/8/24-way
    all unchanged). Capacity has no such cliff and takes 0.80 for ~+10%
    pool. Boot-to-boot pool jitter is ±6–8% (speed) / ±10% (capacity) —
    a rank whose page cache wasn't dropped feeds the profiler less free
    memory, and the pool is the min across ranks. Co-tenant budget at
    the shipped settings: ~5–7 GiB on the speed head box, ~2.5 GiB on
    the capacity head box, 11–12 GiB on workers — run OCR-class helpers
    on the workers. A 500K-token cold prefill was survivable on both
    shipped profiles with a ≥3.1 GiB head-box floor (measured low edge
    of the ~3.2 GiB band).

## Repo layout

- `launch/ds4flash-tp3-up` — workstation orchestrator (profile dispatch,
  preflight, stage-env, workers-then-head start with bounded health
  watch + JIT pre-warm, rank-log capture on failure)
- `launch/ds4flash-tp3-karmic-spark` — speed-profile node helper (mp
  bootstrap, 3-patch mounts, DSpark)
- `launch/ds4flash-tp3-spark` — capacity-profile node helper (Ray V2,
  offline wheelhouse install, `COMPILE_MODE` eager/piecewise/full ladder)
- `launch/prepare-wheelhouse.sh` — capacity-profile pre-flight: fetches
  the pinned b12x source, builds the wheel inside the pinned image,
  downloads the Ray wheels, records every sha256 (the boot installs from
  it with pip's index off)
- `launch/tp3-kit/` — the in-container kit: `engine-identity.json` (the
  manifest — pinned in part: both build stamps and the four bind-mounted
  patch digests; the image-baked file digests are left for a reader to
  fill in against a live container),
  `boot-gate-identity.py` (the gate that runs before the engine starts),
  `make-engine-identity.py` (the generator), `verify-wheelhouse.py`
  (the wheelhouse integrity check), `spec-short-prompt-check.py` (the
  short-prompt probe under speculative decode)
- `config/cluster.env` — speed profile, every pin in one file (default)
- `config/cluster.env.capacity` — capacity profile
- `patches-karmic/` — the TP=3 padding patches for the speed image
  (upstream Apache-2.0 files with marked edit blocks; see its README)
- `patches-csf/` — the CSF serving path: the pad-aware loader from
  PR #973 in the `local-inference-lab/vllm` fork, with marked TP=3 edit
  blocks, the b12x compile wait-race fix
  write-up, geometry guidance, and the twelve-file `vllm-overlay/` a
  CSF boot mounts (see its README)
- `tools/csf/` — the offline CSF converter toolchain (codec, converter,
  round-trip validator) — runs on stock PyTorch today
- `patches/tilelang.py` — dynamo-safe DeepGEMM probe for the capacity
  image (upstream vLLM file, Apache-2.0 — keep its header). It is based
  on the engine file as it stood **before** the upstream "Route DeepSeek
  hc head through TileLang op" commit, and it is bind-mounted over the
  image's own copy at boot — so a capacity boot runs the older routing.
  See `NOTICE`.
- `tools/validate-tp3-plan.py` — offline virtual-TP plan verifier for
  the *capacity* fork's plan machinery (runs against any checkout of
  the fork, or inside the pinned image)
- `docs/FIXLOG.md` — the three ladders (capacity, speed, CSF), rung by rung
- `docs/VALIDATION.md` — the measurement methodology behind every number
  in this repo (anti-contamination, workload split, corruption battery,
  recording discipline) — reuse it for your own A/B ladders
- `LICENSE` — MIT for this repository's own scripts and documentation
- `NOTICE` — the provenance notice for the modified upstream files
  (Apache-2.0), which `patches*/` remain under
- `.gitignore` — keeps logs and Python caches out of the tree

## What's next (roadmap, not needed for correctness)

- **CSF runtime release** is the one external dependency left: the
  converter and loader patches here are complete; the day a public b12x
  ships CSF symbols **with beta-lineage draft/verify throughput**, the
  CSF speed build (fastest measured — see its section) serves with no
  further porting. Readiness criteria and today's master-vs-beta gap:
  `patches-csf/README.md` and FIXLOG rung 29 — the symbol check alone is
  not readiness; verify any candidate runtime with a small boot.
- **Prefill** stays the open frontier, now with the packaging half
  answered: a CSF-packed copy of this checkpoint measures top-of-band
  (not the 1.5–2.6K the W4A16-prefill lanes publish — their advantage is
  the runtime's prefill kernel path, not the packing alone). The
  3.8K lane's remaining edge is the V4.1 checkpoint's cheaper prefill
  architecture, which no patch ports. Flag space on the image is
  measured empty. The indexer-split idea was investigated to resolution:
  the selection buffer is row-indexed (trivial slicing) but every
  published split rides on sequence-parallel prefill's row ownership,
  which this tree does not have — a standalone split is net-negative
  (per-rank scoring saved < the selection all-gather it must add).
  Reopens only with an SP-prefill port or an upstream landing.
- **nvfp4 KV** — tested and rejected: anti-capacity on this fork (pool
  shrank) with identical speed; see FIXLOG.
- **RoCE twin interfaces** — tested here and **rejected for now**: the
  twin devices carry no GID at the pinned index, so the ring dies at
  NCCL init (FIXLOG rung 22); making twins work means re-doing GID
  selection for the whole ring, a two-variable change.
- **DSpark draft depth** — axis closed at k=5: k=10 strictly worse on
  every axis; k=4 faster on prose but −11% on code-class streams
  (deep acceptance — FIXLOG rung 28).
- **`GPU_MEM_UTIL` above 0.79 on the speed profile** — tested and
  rejected (16-way −14%); capacity already ships 0.80. The ladder and
  the safety gates (500K-token cold prefill, co-tenant budget) are in
  the FIXLOG tuning notes.
- Multi-hour soak on the speed profile.
- Re-pin discipline: `eugr/spark-vllm-b12x` moves nightly
  (nightly-20261004 was the newest tag at validation, shared with
  `latest`; a newer tag has since shipped the same commit); every re-pin
  must re-diff `patches-karmic/` against the
  new image (procedure in `patches-karmic/README.md`).

## Credits & prior art

- **Jiunsong (Jun Song)** — [SuperDeepSeek, the abliterated V4-Flash
  build this recipe serves](https://huggingface.co/Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX)
  (MIT; the checkpoint is the foundation everything here stands on).
- **lukealonso / local-inference-lab** — B12X virtual-TP padding (the
  mechanism), the vLLM forks, b12x kernels.
- **tonyd2wild** — the `nccl230u1` image, the MiniMax-M3 TP=3 recipe the
  capacity profile inherits its Ray/NCCL/OOM fixes from, the [2-Spark
  DSpark/NVFP4-KV recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  used as the benchmark baseline (and whose k≤5-or-multiple rule and
  capture-size = seqs×(k+1) sizing this recipe follows), and the first
  DS4.1-Flash TP=3 lanes (native and
  [EXL3](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/EXL3-TP3.md)).
- **eugr** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  (the ARM64 image builder + changelog behind the speed profile's
  container), the DS4 b12x env set, mesh networking docs.
- **christopher_owen** — [DS4.1 native TP=3 + DSpark reference
  numbers](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438)
  and the NCCL ring conventions the speed profile's fabric pins follow.
- **jasl** — DS4 SM12x vLLM support (PR #41834), TP=4 mechanics.
- **provos** — the MTP+CUDA-graph corruption write-up behind this
  recipe's battery-before-trust stance on spec decode.
- **MiaAI-Lab** — DS4-on-Spark bug documentation, the original 2-Spark
  1M-context recipe line, and the 4-box SGLang reference numbers.
- A companion [Vision-Exp TP=3 recipe](https://github.com/PrblyElect/DeepSeek-V4-Flash-Vision-TP3-3x-DGX-Spark)
  (the same account) — native vision at TP=3, the SP-prefill port, and
  the scheduler fairness patches (decode deferral cadence + interactive
  admission) live there.

Contained here: the 0731/V4.0 TP=3 port itself — the three ladders in
`docs/FIXLOG.md` (29 rungs), the `patches-karmic/` set (V4→V4.1 pad delegation,
64-aligned MoE pad, the `DSparkDraftModel` gate, `allow_tp_padding`
marking), the wrong-default-model trap, the MHC dynamo wall + tilelang
patch, the o_proj→b12x routing, the pre-warm/prefix-cache workflow
numbers. Scripts: MIT (see `LICENSE`); upstream licenses apply.
