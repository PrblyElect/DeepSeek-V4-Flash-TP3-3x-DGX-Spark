#!/usr/bin/env python3
"""Round-trip validation of a converted shard against its source shard:
every compressed scale plane must decode bit-exact to the source scale
bytes; every other tensor must pass through byte-identical."""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csf_codec import decode_plane

GEOM = {"w1": (2048, 128), "w3": (2048, 128), "w2": (4096, 64)}

src_path, dst_path = sys.argv[1], sys.argv[2]
src = safe_open(src_path, framework="pt", device="cpu")
dst = safe_open(dst_path, framework="pt", device="cpu")
sn, dn = set(src.keys()), set(dst.keys())

dropped = {k for k in sn - dn if k.endswith(".scale")}
added = {k for k in dn - sn if "_csf_" in k}
unexpected_drops = sn - dn - dropped
unexpected_adds = dn - sn - added
print(f"source tensors: {len(sn)} | converted: {len(dn)}")
print(f"dropped scales: {len(dropped)} | added csf tensors: {len(added)}")
print(f"unexpected drops: {sorted(unexpected_drops)[:4]}")
print(f"unexpected adds: {sorted(unexpected_adds)[:4]}")
assert not unexpected_drops and not unexpected_adds
assert len(added) == 2 * len(dropped)

ok = 0
n_exc = 0
for k in sorted(dropped):
    proj = k[: -len(".scale")].rsplit(".", 1)[-1]
    rows, cols = GEOM[proj]
    fixed = dst.get_tensor(k + ".mxfp4_csf_fixed").numpy()
    exc = dst.get_tensor(k + ".mxfp4_csf_exceptions").numpy()
    n_exc += int(len(exc))
    plane = decode_plane(fixed, exc, rows, cols)
    src_scale = src.get_tensor(k).view(torch.uint8).numpy()
    assert np.array_equal(plane, src_scale), f"round-trip mismatch: {k}"
    ok += 1

common = sn & dn
same = sum(1 for k in common if torch.equal(src.get_tensor(k), dst.get_tensor(k)))
print(f"round-trip: {ok}/{len(dropped)} scale planes BIT-EXACT (exceptions total: {n_exc})")
print(f"pass-through: {same}/{len(common)} tensors byte-identical")
assert ok == len(dropped) and same == len(common)
print("CONVERTED SHARD: FULLY VALIDATED")
