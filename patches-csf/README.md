# patches-csf — serving this checkpoint in MXFP4-CSF at TP=3

These pieces complete the CSF (compressed FP4 scales) serving path for
DeepSeek-V4-Flash at TP=3, on top of the public CSF loader lineage
(vLLM PR #973, branch `feat/kk-fp4-csf` — take the unmodified branch
files from there; only the loader below carries TP=3 edits, marked
`TP3-recipe patch`).

## Prerequisite (read this first)

The CSF serving path needs a **b12x runtime with CSF support**. The
CSF prepare + compact-decode kernel path landed on the b12x **master
branch on 2026-10-06** (public commits; watch for the release or
container nightly that bakes it — verify with the symbol check below),
and the vLLM-side loader files live on the open PR #973 branch. The
converter in `tools/csf/` runs on stock `torch` + `numpy` +
`safetensors` today. Quick readiness check for any candidate runtime:

    python3 -c "import b12x; from b12x.moe.fused_moe import CsfScalePlanes" && echo ready

Two known gaps as of that landing: the compact-path (n % 128 == 64)
compile wait-race documented below is **not** fixed on master, and the
m1 decode plan's program **declaration** does not yet enumerate the
compact kernel (a weights-stage prepare on master dies with `moe.w4a8.
compact_micro ... unplanned CuTe program` even with the wait-race fix
applied — a newer runtime lineage carries the serving integration;
master catches up when its `_preparation` declarations land). A runtime
built from master alone is therefore **not yet sufficient** for the
2112 geometry: verify with a small boot, not just the symbol check.

## What's here

- `mxfp4_csf_loader.py` — the PR-973 expert loader with the marked
  `TP3-recipe patch` blocks: accepts the virtual-padded MoE intermediate
  this model needs at TP=3 (2048 → 2112, i.e. 64/rank) when the real
  geometry matches the checkpoint family, slices at real geometry,
  zero-fills phantom rows (w13) / columns (w2, incl. selector-axis
  extension), remaps exception indices, and admits the padded TP size
  into the supported-TP set. Drop it over the branch's file.
- `../tools/csf/` — the offline toolchain (independent of vLLM):
  - `csf_codec.py` — the plane codec reimplemented from the loader's own
    decode (row base = max-pair-coverage, offset-1 u24 exceptions).
  - `convert_to_csf.py` — converts a native DeepSeek-V4-Flash checkpoint
    directory (official or the same-geometry abliterated build) into the
    CSF serving layout (manifest + compressed expert scales; everything
    else passes through byte-identical). ~30–60 min per copy, CPU-bound.
  - `roundtrip_validate.py` — shard-level proof: decode→re-encode
    reproduces published bytes; pass-through tensors byte-identical.
- `vllm-overlay/` — **the twelve-file read-only vLLM overlay a CSF boot
  mounts, shipped in-tree** so a boot needs no assembly from the PR
  branch. Only `mxfp4_csf_loader.py` carries TP=3 edits (byte-identical
  to the copy at the top of this folder); the rest are the PR-973
  lineage files unmodified, plus the small registry/allowlist edits
  noted in the manifest below. Apache-2.0 headers retained.

## Assembly manifest (serving the CSF build end-to-end)

The loader above is the only file with TP=3 edits, but a serving boot
mounts a **twelve-file vLLM overlay** (read-only, whole-file) on top of
the pinned image, plus the b12x drop. The overlay ships in
`vllm-overlay/`; if you would rather assemble it from the PR-973 branch
yourself, this is the full mount list (paths relative
to the image's vLLM root):

    model_executor/model_loader/__init__.py          # registry: admit the CSF loaders
    model_executor/model_loader/csf_utils.py
    model_executor/model_loader/mxfp4_csf_loader.py  # this repo's TP3-patched copy
    model_executor/model_loader/nvfp4_csf_loader.py
    model_executor/layers/fused_moe/b12x.py
    model_executor/layers/quantization/__init__.py   # registry: admit the CSF quants
    model_executor/layers/quantization/mxfp4_csf.py
    model_executor/layers/quantization/nvfp4_csf.py
    model_executor/layers/quantization/kimi_mxfp4_csf.py
    models/deepseek_v4/mxfp4_csf.py
    models/deepseek_v4_1/mxfp4_csf.py
    config/model.py

The two `__init__.py` registry edits are small (admit-list lines), and
`config/model.py` carries the quant-method allowlist entries — a
~2-line diff each; everything else is the PR branch's file unmodified.
The b12x side needs the CSF-symbol runtime plus the two fixes described
below (wait-race + `_instantiate` retry). Serving flags:
`--load-format mxfp4_csf --quantization mxfp4_csf` with the same env,
pads, and DSpark settings as the fp8 speed profile.

## The b12x compile wait-race (apply when a CSF b12x ships)

Symptom: at the 2112 pad (n % 128 == 64) the compact_micro n64 kernel
path is selected for decode, and boots die during weight preparation
with a rejected "unplanned CuTe program" inside a no-compilation
instantiate. At 2304 (n % 128 == 0) the path never fires — which is why
2304 boots clean and loses anyway (see below).

Root cause: in b12x's `session.py`, `_install_obligation`'s ready path
adopts `plan.prepared.programs` — a historical record of one
materialization — as the obligation's program set. The compact ProgramKey
is not in that record, so the no-compilation instantiate lazily compiles
it and rejects.

Fix (three lines of intent): on the ready path, re-derive the declared
coverage with `_compile(..., required=True)`, union the result with the
recorded set (`_compile` overwrites the dict — union after, not
before), then `_wait_programs`. No-op for n % 128 == 0 paths.

Operational notes: the compile-cache key embeds patch-sensitive state —
treat key drift as a canary when iterating on b12x; a fixed boot
compiles compact per rank (selection caches share selections, not
artifacts); CuTe disk-persistence failures are silently suppressed, so
an unpersisted artifact is indistinguishable from never-compiled next
boot — wipe caches via container when in doubt.

## Getting a CSF-capable b12x today (extraction from the public beta channel)

You do not have to wait for the release: the author's beta channel on
`ghcr.io/local-inference-lab/vllm` already ships working CSF builds,
and the part you need is **pure Python — architecture-independent** —
so their amd64-only tags are fine on arm64 Sparks. Extract, never
execute:

```sh
docker create --name b12x-src \
  ghcr.io/local-inference-lab/vllm@sha256:6008f020c23065bef71390194f968af9f5ebbedf57807b1659f876632e4a1067
docker cp b12x-src:/opt/venv/lib/python3.12/site-packages/b12x ./b12x-beta
docker rm b12x-src
python3 -c "from b12x.moe.fused_moe import CsfScalePlanes, Mxfp4CsfWeights; print('ready')"
```

(The `docker cp` source path is that image's venv layout; if a newer
beta moves it, `docker export b12x-src | tar -t | grep 'b12x/__init__'`
finds the current one. The digest above is the build this recipe's CSF
numbers were measured on; beta tags churn, so pin by digest.)

On a fresh extraction, apply the two fixes documented in the sections
above before the
2112 geometry will serve: the compile wait-race (fix intent further
up) and the `_instantiate` retry — a ready diff ships in
this folder as `b12x-instantiate-retry.patch` (written against
master's `preparation/session.py`; on a beta extraction the same
27-line wrapper plus the `_instantiate` → `_instantiate_once` rename
applies, possibly with fuzz). Then mount the patched directory over
the image's `b12x` package the way the assembly manifest describes,
and **verify with a small boot — the symbol check alone is not
readiness**.

Watch the channel: `…-beta-spark-…` arm64 tags have started appearing,
which is the upstream signal that official Spark packaging of this
stack is close. When a released build passes the readiness check and
a boot, the extraction step can be deleted from this recipe.

## The second b12x patch the spec boot needed (apply alongside)

The wait-race fix above unblocked the no-spec boots, but the DSpark
(spec-decode) boot on the 2112 geometry hit the same "unplanned CuTe
program" rejection through a **different path**: the declaration error
surfaced inside `_instantiate` itself rather than at obligation
adoption. Fix intent: make `_instantiate` retry once on a declaration
error after re-deriving the declared coverage — the same
`_compile(..., required=True)` → union → `_wait_programs` sequence as
above, applied at the instantiate level. If a public b12x closes the
declaration gap upstream, this retry becomes inert; either way, do not
assume a runtime is spec-ready from the symbol check alone — boot it
with DSpark enabled and watch weight preparation.

## Geometry guidance (measured)

- **2112** (the 64/rank pad): unlocks compact W4A8 decode kernels
  (~+3–4% decode) and the smaller phantom footprint — **the recommended
  pad once the wait-race fix above is applied**. At speed-profile
  settings (DSpark k=5, gmu 0.79, capture 144) this geometry measured
  ~5–10% faster single-stream decode than the fp8 native path in a
  same-day paired A/B (+10% against a control at the bottom of its
  historical band), with pool parity-plus.
- **2304** (128/rank): boots on unpatched b12x, but the phantom expert
  weights (~+12 GB across ranks) plus ~+12.5% MoE compute made it lose
  on every axis at speed settings. Use only as the unpatched fallback,
  or for the no-spec capacity-style variant where its pool still pays.
