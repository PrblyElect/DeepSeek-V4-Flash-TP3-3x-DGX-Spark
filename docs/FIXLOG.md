# FIXLOG — the crash ladder from nothing to first light

Window of 2026-10-03/04 (extended through 10-06 by the CSF campaign). Three
ladders, twenty-nine rungs of distinct root causes, each diagnosed from
container logs and fixed forward. Recorded verbatim so nobody repeats this
grind. This is what "nobody has ever run this model at
TP=3" looks like from the inside — three times: once on the chthonic
lineage (capacity profile, rungs 1–13), once on the karmic lineage (speed
profile, rungs 14–23), and once on the CSF port (rungs 24–29).

Re-verified 2026-10-04 on the live lane: ~28 tok/s
single-stream, a KV pool at the low edge of the ~4.9–5.4M band (8192-era
band, pre-4096; boot
jitter; the re-verify landed there), NCCL 2.30.7 banner — every README
number re-checked against the running ring.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `missing .env.tp3-ds` on workers | staging script wrote a literal `~` path into the helper (`"~/..."` doesn't expand) | use `$HOME` in the rewritten path |
| 2 | head refused: "resident model container active: …" | overlap-grep pattern matched a non-model staging container | tighten pattern to real model containers only |
| 3 | workers exit instantly, `sleep: invalid option -- 'm'` | image ENTRYPOINT is `[sleep]`; serve args became sleep's args | `--entrypoint bash`, pass full `-c` command |
| 4 | `can't open file '//python3'` | `--entrypoint python3` **plus** `python3 -m ...` argv → double python | entrypoint + `-m vllm.entrypoints...` directly |
| 5 | pydantic: speculative method `dspark` invalid; `B12X` unknown backend | this fork has no DSpark method (mtp/eagle3/ngram only); backend enum is `B12X_MLA_SPARSE`/`B12X_ATTN` | spec-decode OFF for first light; correct enum name |
| 6 | **"attention heads (16) must be divisible by TP 3"** + derived max_pos 40960 | **the api_server ignored the bare positional model path and silently loaded its default `Qwen/Qwen3-0.6B`** (16 heads, 40960 ctx). Every upstream theory (compress-ratio division, hc_mult, sub-config mutation) was chasing the wrong model. | pass `--model /path` explicitly — then the real checkpoint parses (64 heads, 1M) and the virtual-TP pad engages: heads 64→72, groups 8→9, MoE 2048→2112, shared 2304, vocab 129408 |
| 7 | `AssertionError: collective_rpc should not be called on follower node` | this June-era fork's mp executor predates multi-node follower support | pivot to Ray (`--distributed-executor-backend ray`) — the path proven on this image |
| 8 | `ray: command not found` | Ray isn't in the image | `pip install ray==2.55.1` at container start (all nodes) |
| 9 | `ActorHandleNotFoundError: created in job 01000000, current job 02000000` | **secondary noise** — the V2 executor's teardown after a worker died; it masks the real error. Always chase the *first* worker ERROR block and persist Ray logs (`-v $cache/ray:/tmp/ray`) | discipline: read first-error-first; V2 executor is the one that works |
| 10 | V1 Ray: `pybind11 ... function_record ... is not pickleable` | V1 RayDistributedExecutor pickles the engine config; DS4's config carries a C++-binding object | force the V2 executor: `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` |
| 11 | `torch._dynamo.exc.Unsupported: can't handle functions not implemented in python` in `mhc_pre_tilelang` → `is_deep_gemm_supported` → `is_device_capability` | DS4's hyper-connection path calls a C++ binding inside the dynamo-traced region (DS4-only code; MiniMax never touches it) | **solved, not worked around**: `patches/tilelang.py` precomputes the probe at import as a module constant; bind-mount it and run `COMPILE_MODE=piecewise` → compile + PIECEWISE cudagraphs return (6.4 → 11 tok/s measured, corruption battery clean). `--enforce-eager` remains the one-flag fallback |
| 12 | DeepGEMM `layout.hpp:39 t.dim() == N` assertion in `deep_gemm_fp8_o_proj` → `fp8_einsum` | the DeepGEMM o_proj einsum path cannot take the virtual-TP-padded TP=3 shapes; the fork's pad is designed **for b12x kernels** | route DS4 projections into b12x: `VLLM_USE_B12X_WO_PROJECTION=1`, `VLLM_USE_B12X_MHC=1`, `B12X_MOE_FORCE_A8=1` (eugr's DS4 env set) |
| 13 | `B12XMHCScratchPlan.bind() got an unexpected keyword argument 'expected_m'` | baked b12x 0.20.0 is API-older than the fork's MHC caller (the M3 recipe hit the same class of skew with `copy_runtime_metadata`) | `pip install --force-reinstall --no-deps git+...b12x.git@08e980c...` at container start |

## The second ladder: karmic + DSpark (the speed profile, rungs 14–23)

Same window, successor-lineage image. The karmic-kraken lineage brings
DSpark speculative decode, async scheduling, baked b12x 1.5.0 and the V4.1
pad natively — but the 0731 V4 checkpoint still needed its own port, and
the mp executor replaced Ray with a different failure grammar.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 14 | `exec format error` from the project's own ghcr images | **every** ghcr tag of the karmic lineage is amd64-only (single-arch manifests, verified across the full tag list) — Sparks are aarch64 | the only prebuilt ARM64 karmic image found: `eugr/spark-vllm-b12x` nightlies on Docker Hub, digest-pinned |
| 15 | pydantic rejects the canonical `--attention-config` | this fork's AttentionConfig schema has no `use_fp4_indexer_cache` field (upstream-only) | omit the flag entirely |
| 16 | SpeculativeConfig validation dies on the raw `64 % 3` head check | the draft arch `DSparkDraftModel` wasn't in `MODELS_CONFIG_MAP`, so the draft config validated **unpadded** | one map line in `patches-karmic/config.py` — the entire DSpark-at-TP=3 gate |
| 17 | `VllmConfig` build: heads 64 not divisible by TP 3 | the fork pads V4.1 natively but not V4 | `DeepseekV4ForCausalLMConfig.update_model_config_for_parallelism` delegates to the V4.1 method (heads 64→72, o_groups 8→9) — boot log then prints the pad lines |
| 18 | `FusedMoEConfig`: moe_intermediate 2048 % 3 | same gap on the MoE axis — the routed-expert loader tolerates padded intermediates, but nothing padded the config | pad to a 64-per-rank-aligned multiple of TP (2048→2112 = 704/rank). A minimal pad (2049) **fails**: the B12X MXFP4 path requires per-rank intermediates divisible by 32 |
| 19 | `InstantTensor only supports pytorch format` at weight load | the default loader rejects stock safetensors-shard checkpoints | `--load-format safetensors` |
| 20 | worker asserts `collective_rpc should not be called on follower node`, a secondary dies | **benign** on mp multinode with spec — the real `Worker_TP` and head complete fine. The *same message* was a real dead end on the chthonic mp path (rung 7); opposite meaning on this stack | read the whole boot: if `Worker_TP` completes and the head serves, ignore the assertion |
| 21 | boot "hangs" ~10 min in `shm_broadcast` waits | spec boots take ~14 min: autotune + ~2,800-kernel priming + warmup | poll `/health` out to 15+ min before declaring a hang; the launcher's bounded watch does this |
| 22 | adding the RoCE "twin" interfaces (`roceP2p1s0f*`) to `NCCL_IB_HCA` kills the ring at init: `ibv_modify_qp failed ... No data available` on the twin, worker dies `NCCL error: unhandled system error` | the twin devices have **no GID at the pinned index** (empty local GID — link-local only); one `NCCL_IB_GID_INDEX` cannot span both device families. The community twin recipes run with the GID index **unset** + address-range selection instead | stay on `rocep1s0f0,rocep1s0f1` with GID 3. The twin lever (~200G/leg claims) requires re-doing GID selection for the whole ring — a two-variable change, not an A/B |
| 23 | `GPU_MEM_UTIL=0.80` on the speed profile silently costs throughput at one concurrency: 16-way aggregate drops ~14% (~148 vs ~173 tok/s, 3-sample means, same-day paired boots) while single-stream / 4 / 8 / 24-way stay in band | a per-batch-shape cliff between 0.79 and 0.80 on this image (not pool-size-driven: 0.78 boots at similar pools measured full speed); the knob's extra claim mostly never becomes resident KV on the spec profile anyway | speed ships 0.79; capacity (no spec, no graph pool) converts 0.80 into ~+10% pool and ships it — don't copy the value across profiles |

## Speed-profile tuning notes

- `B12X_W4A8_TINY_DECODE=0` is correctness-critical on this model class:
  1 omits the SwiGLU clamp. Never "optimize" it on.
- **P2P enabled (`NCCL_P2P_DISABLE=0`), A/B'd live**: ~3–4% faster
  prefill than `=1` on this stack (decode/aggregate unchanged within
  noise). Not the 2× the capacity profile measured when it dropped its
  *combined* P2P+SHM disable pins — different mechanism — but free, and
  both profiles now agree on P2P enabled.
- **`--max-cudagraph-capture-size` is a cliff, not a dial.** At 48
  (= 8 spec'd seqs × (k+1)) the lane is perfect ≤ 8 concurrent streams
  and collapses past it: 16-way measured **~14 tok/s aggregate** (~180 s
  of off-graph churn) vs **174 tok/s** at 144. 144 = 24 × (k+1) — full
  coverage at `max_num_seqs` — and costs ~14% of the KV pool
  (≈2.3–2.6M vs ≈2.7–2.8M tokens at gmu 0.79). The helper ships 144; drop
  to 48 only if you never exceed 8 concurrent streams and want the pool
  back. The full concurrency curve at 144: ~95 @ 4, ~150 @ 8, ~176 @ 12,
  ~170 @ 16, ~186 @ 20, **~199–208 @ 24** (best-day pair repeat-run
  stable; ~199–201 at the shipped gmu 0.79; 0 errors) —
  aggregate climbs to the `max_num_seqs` cap (16-way dips to ~170 within noise); per-stream
  at 24-way is ~8.6 tok/s (batch class, not interactive).
- **k=10 is strictly worse on this checkpoint — measured, don't retry:**
  decode −27–31% (prose 31–34, count 78–79 tok/s), 8-way aggregate
  83 vs 139, 16-way 103 vs 174, KV pool −27% (1.74M), and a ~50-min
  boot (~5,100 kernels to prime vs ~2,800). The per-position acceptance
  histogram explains it: ~74% at position 0 decaying to ~11% at position
  4 and ~2% at positions 5–9 — the extra drafted positions are dead
  weight whose verify compute costs more than their tokens return.
  k=5 is the optimum; the community "k ≤5 or a multiple of 5" rule is a
  divisibility constraint, not a recommendation.
- **Thinking defaults ON** in this build's `deepseek_v4` reasoning parser.
  Send `chat_template_kwargs: {"thinking": false}` or budget for
  `reasoning_content` inside `max_tokens` (see README).
- `MAX_NUM_BATCHED_TOKENS=16384` crashes 0731 `sparse_mla` metadata init —
  8192 stays (the capacity profile hit a *memory* trap at 16384 instead;
  same number, different failure).
- The NCCL channel set (8 channels, `NCCL_NET=IB`) follows the DS4.1
  TP=3 karmic lanes and was validated as a set; the capacity profile's
  4-channel pin was not A/B'd here.
- **The `GPU_MEM_UTIL` ladder (A/B'd live, same-day paired boots, one
  variable per boot)**: speed 0.78 → 0.79 → 0.80, capacity 0.80.
  Findings that will save you a day: (1) the speed profile's pool gain
  from the knob is small (+~150K tokens expected at 0.79; at 0.80 much
  of the extra claim is absorbed by the graph/allocator re-profile, not
  KV); (2) the capacity profile converts it almost fully — ~+10% pool
  at 0.80 with speed at/above the 0.78 bands; (3) the speed cliff is
  16-way-only at 0.80 (rung 23); (4) the pool jitters ±6–8% (speed) /
  ±10% (capacity) boot-to-boot because a rank with undropped page cache
  feeds the profiler less free memory and the pool is the min across
  ranks — never conclude from one boot; (5) steady-state host headroom
  is what matters, not post-boot readings: at the shipped settings the
  head box holds ~6 GiB steady (speed) / ~4 GiB (capacity), a
  500K-token cold prefill dips it to ≥5 GiB / ~3.2 GiB respectively,
  and a bounded co-tenant allocation held 5–7 GiB (speed) / 2.5 GiB
  (capacity) on the head with the lane serving mid-hold — put OCR-class
  helpers on the workers (11–12 GiB free).
- **`MAX_NUM_BATCHED_TOKENS` has two truths — measured in both
  directions.** On **spec-decode boots** 4096 is neutral-to-harmful for
  prefill (the community 5%-at-64K datapoint did not reproduce at 40K
  on the speed profile) and halves code-class decode: the DSpark
  draft/verify pipeline starves inside the smaller token budget — the
  engine itself warns at boot ("max_num_scheduled_tokens … based on the
  speculative decoding settings"), and measurement confirmed it
  (~100–108 → ~52 tok/s code-class). Spec profiles keep 8192. On
  **no-spec boots** 4096 is a four-effect win: the halved prefill
  activation workspace converts to KV (+40–54% pool on the capacity
  profile, ~4.9–5.4M-class → ~7.5–7.6M-class; similar on the CSF capacity-style
  variant), prefill gains ~1.7–2.2× (~0.9K→~1.6K on capacity,
  ~1.0K→~2.2K on the CSF variant), concurrent decode during a big
  prefill gets ~2.6× fairer (worst chunk gap ~8 → ~2 s under a ~58K
  prefill), and single-stream decode is flat-to-better. Every future
  change to this knob is a multi-effect variable — never treat it as
  single-axis.
- **The image's remaining config space is measured-empty for prefill**
  (spec-profile scope): `--linear-backend b12x` and `--load-format b12x`
  (which boots cleanly
  through the TP3 padding patches and lands top-of-band on pool — a fine
  alternative, just not faster). Spec-profile boots live in one
  ~1.0–1.06K prefill band. The 1.5–2.6K same-checkpoint lanes elsewhere
  run a different image and a CSF-packed checkpoint feeding W4A16
  kernels; that is a repack project, not a flag.
- **`nvfp4_ds_mla` KV on this fork is anti-capacity for this model**:
  measured pool **smaller** than fp8 at identical settings (≈2.23M vs
  ≈2.36–2.48M) with speed identical everywhere — this image's V4.0
  record path is fatter per token than fp8 (the lean 416-byte writer
  ships in a different lineage, and only for the V4.1 path). Together
  with the community's sustained-context garble class, fp8 KV is the
  measured choice here, not just the cautious one.
- Sustained-load budget: community reports on this image family include
  garble after long contexts ([eugr
  #363](https://github.com/eugr/spark-vllm-docker/issues/363)) and a
  sparse-MLA Xid only after ~32 h on an older nightly ([eugr
  #391](https://github.com/eugr/spark-vllm-docker/issues/391)). Our
  battery + 10-min soak are clean; run a multi-hour soak before trusting
  any lane of this class for unattended work.

## Order matters (the working sequence)

Ray head/worker daemons → b12x pin-upgrade → vLLM on the head connects to
the existing Ray cluster → placement group across 3 GPUs → actors load
shards (~1:50 for 50 shards/rank) → profile run in eager → KV pool →
`Application startup complete`.

## Operational notes

- Fresh `docker run` every boot; never `docker restart` (NCCL/Ray state).
- `sync; echo 3 > /proc/sys/vm/drop_caches` before boot on memory-tight boxes.
- A stale lane-switch process holding the advisory lock will block
  relaunch — kill it first.
- Workers-first launch order keeps the head from waiting on cold containers.
- NCCL banner must read **2.30.7** (capacity profile). If it reads 2.30.4,
  the image's baked `LD_PRELOAD` shim won — re-assert the override env.
  The speed profile has no such fight: it uses the image's baked 2.32.x.
- Sampling: temperature ≤ 0.7 / top_p 0.95 pinned client-side; no
  repetition/frequency/presence penalties (they crash or gut acceptance).
- First request after boot is slow (warmup); judge steady-state from the
  second request on.

## Post-first-light tuning notes

- The `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` pins inherited from
  older Spark recipes were **halving prefill throughput** (~50 s → ~25 s on
  a 21K prompt) with decode unchanged — dropping them doubled it. Those
  pins treat symptoms this stack doesn't have. Not in the table above
  because it cost a debugging hour to learn and exactly one line to fix.
- Channels 8→4 + `NCCL_ALGO=Ring` + `NCCL_NET_GDR_LEVEL=0`: A/B'd live —
  prefill and decode unchanged within noise on this lane. Kept (community-
  best pins on GB10 RoCE), but don't expect them to be a speed lever here;
  at 8K chunks this engine is compute-bound, not fabric-bound.
- **The first long-prompt request after boot pays ~2.2× TTFT** (~45 s vs
  ~20 s steady on a ~15K prompt) — prefill-graph JIT compiles on first
  use. The launcher pre-warms with one tiny AND one ~15K-token request so
  no real user eats it. Related: a replayed ~15K prefix answers in 0.9 s
  (~17×) thanks to prefix caching — keep client system prompts stable.
- `MAX_NUM_BATCHED_TOKENS=16384` is a trap on 121 GiB boxes: the activation
  workspace eats ~1.9M tokens of KV pool for zero measured prefill gain.
  8192 was the original sweet spot at this gmu; the no-spec profiles now
  ship 4096 (see the two-truths tuning note above — same workspace
  mechanism, measured in both directions).
- The KV pool sizing jitters ±10% across identical boots (profiling
  watermark) — quote a range, not a point, if you benchmark.

## The third ladder: the CSF (compressed-scales) campaign (rungs 24–29)

Serving the checkpoint in the CSF compressed-scales format (public
PR #973 loader lineage) at TP=3, then racing it against the fp8 speed
profile. Toolchain and loader edits: `patches-csf/`, `tools/csf/`.

| # | Symptom / goal | Root cause / finding | Fix / outcome |
|---|---|---|---|
| 24 | convert a native checkpoint to CSF offline | the plane codec (row base = max-pair-coverage, offset-1 u24 exceptions) had to be re-derived from the loader's own decode | `tools/csf/` codec + converter; round-trip validated (decode→re-encode reproduces published bytes; pass-through tensors byte-identical) |
| 25 | the PR-973 loader rejects the TP=3 virtual pad (moe intermediate 2048→2112) | it validates exact family geometry, and the pad machinery raises the config's intermediate | marked `TP3-recipe patch` blocks in `patches-csf/mxfp4_csf_loader.py`: accept the small rank-aligned pad when real geometry matches, slice real, zero-fill phantom rows/columns, remap exceptions, admit the padded TP size |
| 26 | at the 2112 pad, boots die in weight prep on the compact_micro n64 path ("unplanned CuTe program" under no-compilation); at 2304 they boot | b12x's obligation ready-path adopts a stale program record instead of re-deriving declared coverage, so the compact kernel lazily compiles where compilation is forbidden | three-line fix intent in `patches-csf/README.md` (re-derive via `_compile(required=True)`, union after — it overwrites the dict — then `_wait_programs`). Not env-fixable. 2304 = unpatched fallback |
| 27 | which base is faster at identical speed settings? | same-day paired A/B, single-variable base swap (k=5, gmu 0.79, cap 144): CSF/2112 measured ~5–10% faster single-stream prose decode (+10% in the paired A/B, whose fp8 control sat at the bottom of its historical band), code-class top-of-band, pool parity-plus (~2.4M), prefill top-of-band; battery clean, cross-base diffs rephrase-class only, ~370K-token cold prefill coherent | **CSF/2112 adopted as the fastest measured configuration** — pending the rung-26 runtime release (rung 24's converter works today). Capacity-style CSF variant (no spec) reaches ~3.6–4.0M pool at 8192, ~5.6–5.8M at 4096 |
| 28 | would a shorter draft (k=4) trade unused positions for step time? | +~5% prose (under the adoption bar) but −11% code/count class, non-overlapping: acceptance histograms show code-class content accepts deep, and k=4 amputates exactly those positions | k=5 stays; the spec-depth axis is closed {4, 5, 10} for mixed workloads |
| 29 | does b12x master + the documented wait-race fix serve CSF already? | paired same-stack race: it boots cleanly — zero tracebacks, pool at parity with the speed band (~2.3M), prefill healthy — and spec stays active with acceptance counted, but draft/verify throughput falls far short of the beta build: code-class lands at ~half the beta band, prose at ~three-quarters | master + boot fixes ≠ serving-grade spec; the missing piece is the beta lineage's draft/verify kernel path, not just the declarations. Re-test as a one-variable swap when a public build lands; until then the beta-extraction procedure remains the working path |

Two fork findings worth knowing on any profile: temp-0 output is not
bit-stable under spec decode + prefix caching (an occasional leading-space
alternation on the first token, semantically identical — gate determinism
checks on normalized text); and requests over `max_model_len` can be
silently swallowed by this fork's API server (no 400, no timeout — a
silent client wait on a huge prompt means over-limit, not a hang).
