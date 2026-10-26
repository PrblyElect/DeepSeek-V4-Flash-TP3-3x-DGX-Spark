# patches-karmic/ — the TP=3 padding patches (SPEED profile)

Three files from the pinned image's own vLLM tree, each with a small
`--- TP3-recipe patch` edit block (search that marker to read every change
we made). The node helper bind-mounts them over the image's copies at
container start — no image rebuild, no source build on the Sparks. They are
upstream Apache-2.0 files; keep their headers and re-diff against the image
when you move to a newer nightly.

The 0731-lineage V4 checkpoint has 64 attention heads, 8 output groups and
a 2048 MoE intermediate — none divide by 3, and this fork pads V4.1
natively but not V4. These patches route V4 through the V4.1 machinery and
close the two gaps the fork leaves open:

| File | Mount target (in the pinned image) | What the patch does |
|---|---|---|
| `config.py` | `vllm/model_executor/models/config.py` | (1) `DeepseekV4ForCausalLMConfig` gains `update_model_config_for_parallelism`, delegating to the V4.1 pad (heads 64→72, o_groups 8→9 — the boot log then prints "Padded DeepSeek V4.1 attention for TP3"); (2) pads `moe_intermediate_size` up to a 64-per-rank-aligned multiple of TP — 2048→2112, because the B12X MXFP4 path additionally requires per-rank intermediates divisible by 32 (2049-style minimal pads fail it) — and records `original_moe_intermediate_size`; (3) `MODELS_CONFIG_MAP` gains `"DSparkDraftModel": DeepseekV4ForCausalLMConfig`, which is the entire DSpark-at-TP=3 gate: without it, SpeculativeConfig validates the draft arch against the *unpadded* V4 config and dies on the raw `64 % 3`. |
| `v4-attention.py` | `vllm/models/deepseek_v4/attention.py` | When the config is padded (`original_num_attention_heads != num_attention_heads`), marks `wq_b`/`wo_a`/`wo_b` params `allow_tp_padding: True` so the generic linear weight loaders zero-fill the pad columns instead of asserting on shard-size mismatch. |
| `v4-model.py` | `vllm/models/deepseek_v4/nvidia/model.py` | Same `allow_tp_padding` marking for the shared experts' params, keyed on `original_moe_intermediate_size` (the routed experts' custom loader already tolerates padded intermediates upstream). |

## Rebasing onto a newer nightly

1. `docker run --rm --entrypoint cat <image> <target-path> > /tmp/upstream.py`
   for each of the three files and diff against the copy here.
2. Re-apply the `--- TP3-recipe patch` blocks (they are small and
   self-contained; every hunk is marked).
3. If upstream grew its own V4 pad or a `DSparkDraftModel` map entry,
   drop the corresponding block — don't double-pad.
4. Re-verify the boot gates: the two "Padded DeepSeek …" log lines, DSpark
   enabled, and the acceptance battery from the README before serving
   anything real.
