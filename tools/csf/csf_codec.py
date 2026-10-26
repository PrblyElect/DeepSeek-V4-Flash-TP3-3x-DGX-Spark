#!/usr/bin/env python3
"""row-base-offset1-u24-exceptions/1 codec — encode/decode MXFP4-CSF scale
planes. Spec derived from the loader (vllm .../model_loader/mxfp4_csf_loader.py
_slice_scale_plane) and validated byte-exact against a published container.

Layout per scale plane (rows x columns, uint8 UE8M0 bytes):
  fixed: uint8, rows//16 slabs of 16*(1+selectors) bytes, selectors=(cols+7)//8
         slab = 16 per-row base bytes, then per row a selector bitstream,
         one bit per column, LSB-first; bit k -> scale byte = base + k
  exceptions: uint32 words, ascending by position; low 24 bits =
         row*columns+col, high 8 bits = the actual scale byte
Decoding an exception position overrides the base/offset decode.
Bases must be <= 254. Unused tail selector bits must be zero.
"""
import numpy as np

SLAB = 16
MASK24 = (1 << 24) - 1


def decode_plane(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int) -> np.ndarray:
    assert fixed.dtype == np.uint8 and exceptions.dtype == np.uint32
    selectors = (cols + 7) // 8
    stream = fixed.reshape(rows // SLAB, SLAB * (1 + selectors))
    bases = stream[:, :SLAB]                                     # (slabs, 16)
    bits = np.unpackbits(
        stream[:, SLAB:].reshape(rows, selectors), axis=1, bitorder="little"
    )[:, :cols]                                                  # (rows, cols)
    plane = (bases.reshape(-1)[:, None] + bits).astype(np.uint8)
    words = exceptions.reshape(-1)
    if len(words):
        pos = (words & MASK24).astype(np.int64)
        val = (words >> 24).astype(np.uint8)
        plane.reshape(-1)[pos] = val
    return plane


def _row_bases(plane: np.ndarray, base_policy: str) -> np.ndarray:
    """Per-row base choice. 'coverage' = argmax pair-coverage {b, b+1}
    (ties -> lowest b) — minimizes exception count per row. 'min' = row
    minimum (kept for comparison experiments)."""
    rows = plane.shape[0]
    if base_policy == "min":
        b = plane.min(axis=1).astype(np.int64)
        return np.minimum(b, 254)
    if base_policy == "coverage":
        bases = np.empty(rows, dtype=np.int64)
        # scale bytes cluster tightly; only values present matter, but a
        # full bincount per row is still cheap at these plane sizes.
        for r in range(rows):
            hist = np.bincount(plane[r], minlength=256)
            cov = hist[:-1] + hist[1:]               # coverage of {b, b+1}
            bases[r] = int(np.argmax(cov))            # ties -> lowest b
        return np.minimum(bases, 254)
    raise ValueError(base_policy)


def encode_plane(plane: np.ndarray, base_policy: str = "coverage") -> tuple[np.ndarray, np.ndarray]:
    """Encode a (rows, cols) uint8 plane."""
    assert plane.dtype == np.uint8
    rows, cols = plane.shape
    assert rows % SLAB == 0
    selectors = (cols + 7) // 8
    fixed = np.zeros((rows // SLAB, SLAB * (1 + selectors)), dtype=np.uint8)
    bases_rows = _row_bases(plane, base_policy)
    bits = np.zeros((rows, cols), dtype=np.uint8)
    diff = plane.astype(np.int64) - bases_rows[:, None]
    np.copyto(bits, 1, where=diff == 1)
    bad = (diff < 0) | (diff > 1)
    rr, cc = np.nonzero(bad)
    exc_pos = (rr.astype(np.int64) * cols + cc).astype(np.uint32)
    exc_val = plane[rr, cc].astype(np.uint32)
    # at exception positions we emit bit 0 (decode overrides them anyway)
    fixed[:, :SLAB] = bases_rows.reshape(-1, SLAB).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    assert packed.shape[1] <= selectors
    fixed[:, SLAB:] = packed.reshape(rows // SLAB, SLAB * selectors)
    if len(exc_pos):
        order = np.argsort(exc_pos, kind="stable")
        words = (exc_val[order] << 24) | exc_pos[order]
    else:
        words = np.zeros(0, dtype=np.uint32)
    return fixed.reshape(-1), words


def validate_plane(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int) -> dict:
    """Round-trip check against the loader's own constraints."""
    plane = decode_plane(fixed, exceptions, rows, cols)
    f2, e2 = encode_plane(plane)
    return {
        "fixed_bytes_equal": bool(np.array_equal(fixed, f2)),
        "exceptions_equal": bool(np.array_equal(exceptions, e2)),
        "n_exceptions": int(len(exceptions)),
    }
