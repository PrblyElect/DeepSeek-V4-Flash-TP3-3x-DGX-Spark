#!/usr/bin/env python3
"""Convert a DeepSeek-V4-Flash native-checkpoint directory (official or the
abliterated MQ repack — identical inventory) into a lil-mxfp4-csf-checkpoint/1
container readable by the MXFP4-CSF loader.

Usage:
  convert_to_csf.py SRC_DIR DST_DIR [--shard model-000XX-of-00048.safetensors]

SRC_DIR: checkpoint dir with model.safetensors.index.json + shards.
DST_DIR: created; gets tensors/, manifest/build-contract (full run only),
         and serving-dir metadata (config.json patched with quantization_config).

Codec: row-base-offset1-u24-exceptions/1, re-derived from the loader's
own decode and validated byte-exact against a published reference
container (roundtrip_validate.py proves the same property on your data).
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csf_codec import SLAB, encode_plane

SCHEMA = "lil-mxfp4-csf-checkpoint/1"
CODEC = "row-base-offset1-u24-exceptions/1"
FAMILY = "deepseek_v4_flash"
# loader FAMILIES: deepseek_v4_flash -> 256 experts, h=4096, n=2048, layers range(43)
COMPRESSED_LAYERS = range(43)
PROJ_GEOM = {"w1": (2048, 128), "w3": (2048, 128), "w2": (4096, 64)}
EXPERT_SCALE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.\d+\.(w1|w3|w2)\.scale$")
CODEC_SUFFIX = ".mxfp4_csf_fixed"
EXC_SUFFIX = ".mxfp4_csf_exceptions"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_shard(src: Path, dst: Path, stats: dict) -> None:
    out = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            m = EXPERT_SCALE.match(name)
            if m and int(m.group(1)) in COMPRESSED_LAYERS:
                proj = m.group(2)
                rows, cols = PROJ_GEOM[proj]
                scale = f.get_tensor(name)
                assert list(scale.shape) == [rows, cols], (name, scale.shape)
                # scales are F8_E8M0; the E8M0 payload byte is the scale byte
                scale_u8 = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
                fixed, exc = encode_plane(scale_u8.numpy())
                out[name + CODEC_SUFFIX] = torch.from_numpy(fixed)
                out[name + EXC_SUFFIX] = torch.from_numpy(exc)
                stats["compressed_scale_count"] += 1
                stats["compressed_scale_bytes"] += int(fixed.nbytes + exc.nbytes)
                stats["original_scale_bytes"] += int(scale.numel())
            else:
                out[name] = f.get_tensor(name)
    save_file(out, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--shard", help="convert a single shard file only (testing/parallel)")
    ap.add_argument("--source-revision", default=None,
                    help="source checkpoint revision/dirname for provenance (default: src dir name)")
    ap.add_argument("--checkpoint-root", default=None,
                    help="absolute path of the container as seen by the SERVING container "
                         "(default: dst as given — set explicitly when converting through mounts)")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    if args.shard:
        (dst / "tensors").mkdir(parents=True, exist_ok=True)
        stats = {"compressed_scale_count": 0, "compressed_scale_bytes": 0, "original_scale_bytes": 0}
        s = args.shard
        convert_shard(src / s, dst / "tensors" / s, stats)
        print(f"converted {s}: +{stats['compressed_scale_count']} planes, "
              f"{stats['original_scale_bytes']/1e9:.3f}GB -> {stats['compressed_scale_bytes']/1e9:.3f}GB")
        return

    # full conversion
    (dst / "tensors").mkdir(parents=True, exist_ok=True)
    stats = {"compressed_scale_count": 0, "compressed_scale_bytes": 0, "original_scale_bytes": 0}
    shard_records = []
    for s in shards:
        src_path = src / s
        dst_path = dst / "tensors" / s
        src_bytes = src_path.stat().st_size
        src_sha = sha256_file(src_path)
        convert_shard(src_path, dst_path, stats)
        shard_records.append({
            "file": s,
            "source_file_bytes": src_bytes,
            "source_sha256": src_sha,
            "target_file_bytes": dst_path.stat().st_size,
            "target_sha256": sha256_file(dst_path),
        })
        print(f"converted {s} ({src_bytes/1e9:.2f}GB -> {dst_path.stat().st_size/1e9:.2f}GB)")

    # serving-dir metadata: copy + patch config.json
    for meta in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "model.safetensors.index.json"):
        if (src / meta).exists():
            shutil.copy2(src / meta, dst / meta)
    cfg = json.loads((src / "config.json").read_text())
    src_quant = cfg.get("quantization_config", {})
    # serving contract (models/deepseek_v4/mxfp4_csf.py from_config): the
    # source fp8 fields stay INLINE at top level; only the method is replaced
    cfg["quantization_config"] = {
        **src_quant,
        "quant_method": "mxfp4_csf",
        "format_version": 1,
        "checkpoint_root": args.checkpoint_root or str(dst.resolve()),
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))

    metadata_sha256 = {
        p.name: sha256_file(p) for p in sorted(dst.iterdir()) if p.is_file()
    }
    manifest = {
        "codec": CODEC,
        "compressed_scale_bytes": stats["compressed_scale_bytes"],
        "compressed_scale_count": stats["compressed_scale_count"],
        "compressed_weight_file_bytes": sum(r["target_file_bytes"] for r in shard_records),
        "family": FAMILY,
        "metadata_sha256": metadata_sha256,
        "original_scale_bytes": stats["original_scale_bytes"],
        "saved_weight_file_bytes": stats["original_scale_bytes"] - stats["compressed_scale_bytes"],
        "schema": SCHEMA,
        "shards": shard_records,
        "source_model_id": "Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX",
        "source_index_sha256": sha256_file(src / "model.safetensors.index.json"),
        "source_index_size_semantics": "tensor_payload",
        "source_payload_bytes": sum(r["source_file_bytes"] for r in shard_records),
        "source_revision": args.source_revision or src.name,
        "source_weight_file_bytes": sum(r["source_file_bytes"] for r in shard_records),
        "status": "converted",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    contract = {k: manifest[k] for k in (
        "codec", "compressed_scale_count", "family", "schema",
        "source_index_sha256", "source_index_size_semantics",
        "source_model_id", "source_payload_bytes", "source_revision")}
    contract["source_names"] = weight_map
    (dst / "build-contract.json").write_text(json.dumps(contract, indent=1))
    print(f"DONE: {stats['compressed_scale_count']} planes, "
          f"{stats['original_scale_bytes']/1e9:.2f}GB -> {stats['compressed_scale_bytes']/1e9:.2f}GB")


if __name__ == "__main__":
    main()
