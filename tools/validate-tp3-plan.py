#!/usr/bin/env python3
"""Validate the chthonic fork's B12X virtual-TP plan for DeepSeek-V4-Flash at TP=3.

Runs the fork's own vllm/config/virtual_tp.py against the checkpoint geometry
with stubbed vLLM surroundings (no torch, no GPU), so the exact plan the engine
will auto-apply at boot can be inspected and asserted offline.

Usage:
  validate-tp3-plan.py path/to/virtual_tp.py

  # or, against the pinned image's own baked code (no checkout needed):
  docker run --rm -v "$PWD/tools":/tools --entrypoint python3 \
    ghcr.io/tonyd2wild/vllm-m3-chthonic:nccl230u1 /tools/validate-tp3-plan.py

With no argument and no real vllm importable, prints usage and exits 2.
(The stubs are skipped when real vllm imports are available, so the docker
form exercises the image's actual module.)
"""

import argparse
import importlib.util
import math
import os
import sys
import types
from types import SimpleNamespace

CHECKPOINT = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "o_groups": 8,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "vocab_size": 129280,
    "num_hidden_layers": 43,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "kv_lora_rank": None,  # set below; DS4 carries q/o lora ranks instead
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "index_topk": 512,
}

TP = 3


def load_virtual_tp(path):
    """Import the target virtual_tp.py, stubbing the vLLM modules it imports
    only when they are not already importable (inside the real image they are)."""
    try:
        import vllm.config.virtual_tp as mod  # noqa: F401

        return sys.modules["vllm.config.virtual_tp"]
    except Exception:
        pass

    for name, attrs in (
        ("vllm", {}),
        ("vllm.envs", {"VLLM_USE_B12X_MOE": True, "VLLM_USE_B12X_MINIMAX_M3_MSA": False}),
        ("vllm.logger", {"init_logger": lambda *_: types.SimpleNamespace(
            warning=lambda *a, **k: print("[fork]", a[0] % a[1:] if len(a) > 1 else a[0]))}),
        ("vllm.platforms", {"current_platform": types.SimpleNamespace(
            is_cuda=lambda: True, has_device_capability=lambda n: True)}),
        ("vllm.v1", {}),
        ("vllm.v1.attention", {}),
        ("vllm.v1.attention.backends", {}),
        ("vllm.v1.attention.backends.registry", {}),
    ):
        mod = sys.modules.get(name) or types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    registry = sys.modules["vllm.v1.attention.backends.registry"]
    registry.AttentionBackendEnum = types.SimpleNamespace(
        B12X_MLA_SPARSE="B12X_MLA_SPARSE", B12X_ATTN="B12X_ATTN"
    )

    spec = importlib.util.spec_from_file_location("virtual_tp_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["virtual_tp_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_model_config():
    text = SimpleNamespace(**CHECKPOINT)
    hf_config = SimpleNamespace(**CHECKPOINT, text_config=text)
    cfg = SimpleNamespace(hf_config=hf_config, hf_text_config=text)
    cfg.get_model_arch_config = lambda: {"arch": "DeepseekV4ForCausalLM"}
    cfg.model_arch_config = None
    return cfg


def make_vllm_config(model_config, tp=TP, expert_parallel=False):
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=expert_parallel,
        ),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(backend="B12X_MLA_SPARSE"),
    )


def main():
    parser = argparse.ArgumentParser(
        prog="validate-tp3-plan.py",
        description="Validate the chthonic fork's B12X virtual-TP plan for "
                    "DeepSeek-V4-Flash at TP=3.",
        epilog="With no argument inside the pinned image, the image's own "
               "vllm.config.virtual_tp module is used.",
    )
    parser.add_argument(
        "path", nargs="?",
        help="path to the fork's virtual_tp.py (omit inside the pinned image)",
    )
    ns = parser.parse_args()

    path = ns.path
    if path is None:
        try:
            import vllm.config.virtual_tp  # noqa: F401
        except Exception:
            parser.print_usage(sys.stderr)
            print("error: no path given and vllm is not importable here", file=sys.stderr)
            return 2
        path = "vllm.config.virtual_tp"
    elif not os.path.isfile(path):
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1

    try:
        vt = load_virtual_tp(path)
    except Exception as exc:
        print(f"error: could not load {path}: {exc}", file=sys.stderr)
        return 1
    src = "real vllm import" if "virtual_tp_under_test" not in sys.modules else path
    print(f"virtual_tp under test: {src}\n")

    mc = make_model_config()
    vc = make_vllm_config(mc)
    vt.maybe_apply_b12x_virtual_tp_padding(vc)
    plan = getattr(mc.hf_text_config, "vllm_virtual_tp_plan", None)
    if plan is None:
        print("FAIL: no virtual TP plan was applied (padding did not engage).")
        return 1

    print("Plan the engine will apply at TP=3:")
    failures = []
    for name, axis in plan.items():
        if not isinstance(axis, dict):
            print(f"  {name}: {axis}")
            continue
        print(f"  {name}: {axis}")
        if axis.get("padded_size", 0) % TP != 0:
            failures.append(f"{name} padded_size {axis['padded_size']} not divisible by {TP}")
        if axis.get("original_size") == axis.get("padded_size") and name != "vocab_size":
            pass  # no-op axes are legal
    text = mc.hf_text_config
    heads = CHECKPOINT["num_attention_heads"]
    groups = CHECKPOINT["o_groups"]
    checks = [
        ("num_attention_heads padded", text.num_attention_heads % TP, 0),
        ("num_attention_heads recorded", getattr(text, "original_num_attention_heads", None), heads),
        ("moe_intermediate padded", text.moe_intermediate_size % TP, 0),
        ("o_groups padded", mc.hf_config.o_groups % TP, 0),
        ("heads-per-group preserved",
         text.num_attention_heads // mc.hf_config.o_groups,
         heads // groups),
    ]
    for label, got, want in checks:
        status = "ok" if got == want else "FAIL"
        print(f"  {label}: {got} (want {want}) {status}")
        if got != want:
            failures.append(label)

    est = {
        "attention FLOPs overhead":
            f"{(plan['attention_heads']['padded_size'] / heads - 1) * 100:.1f}%",
        "routed-expert params overhead":
            f"{(plan['moe_intermediate_size']['padded_size'] / CHECKPOINT['moe_intermediate_size'] - 1) * 100:.1f}%",
        "vocab embed overhead":
            f"{(plan['vocab_size']['padded_size'] / CHECKPOINT['vocab_size'] - 1) * 100:.2f}%",
    }
    print("\nPadding overhead estimates:")
    for k, v in est.items():
        print(f"  {k}: +{v.lstrip('+')}")

    if failures:
        print(f"\nRESULT: FAIL — {failures}")
        return 1
    print("\nRESULT: PASS — plan is TP=3-clean for DeepSeek-V4-Flash geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
