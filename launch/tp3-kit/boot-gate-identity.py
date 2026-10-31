#!/usr/bin/env python3
"""boot-gate-identity.py — prove WHICH engine answered, before it answers.

The image digest proves which manifest was pulled. It does not prove what
executes: the recipe bind-mounts patched engine files over the image's own
(vllm/models/deepseek_v4/attention.py, the TP=3 padding patches), and the
capacity profile installs a pinned b12x build on top of the baked one. A
reader who wants to know "is this the engine the recipe audited?" needs a
claim about the BYTES that run, not about a tag.

This gate makes that claim, from inside the container, before the engine
starts:

  * build stamp  — the engine's own package version (a commit id, not a tag)
  * file hashes  — sha256 of the engine-critical files, AS THEY ARE in this
    container (so a mounted patch changes the hash, which is the point)

The shipped manifest (engine-identity.json) is a TEMPLATE: the hashes are
NOT filled in, because they cannot be computed without an image. The gate
therefore reports NOT PINNED and passes, so a fresh clone boots. Fill the
manifest in (see make-engine-identity.py) and the gate becomes a real
adoption gate: a re-pinned image fails it until it is re-audited.

Exit codes: 0 = gate passed, 1 = gate failed, 2 = usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Where a vLLM install can live. The speed image ships under dist-packages,
# the capacity image under site-packages; the manifest names the right one
# per profile, and this list is only the fallback when it does not.
DIST_CANDIDATES = (
    "/usr/local/lib/python3.12/dist-packages",
    "/opt/venv/lib/python3.12/site-packages",
    "/usr/lib/python3/dist-packages",
)

HEX = set("0123456789abcdef")


def is_pinned(value) -> bool:
    """A value is pinned only if it is a full lowercase sha256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in HEX for c in value)
    )


def is_placeholder(value) -> bool:
    """A template marker ('<FILL-IN: ...>') means the claim is not filled in."""
    if not isinstance(value, str):
        return True
    text = value.strip()
    return not text or "FILL-IN" in text or (text.startswith("<") and text.endswith(">"))


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def package_version(name: str, root: str | None = None) -> str | None:
    """A distribution's version, resolved INSIDE `root` when one is given.

    WHY NOT importlib.metadata.version(): that function searches the WHOLE
    of sys.path, so a distribution installed anywhere else -- a stale
    site-packages, a PYTHONPATH entry, another project's venv -- can
    satisfy it. Putting the validated root on sys.path does NOT make the
    stamp come from that tree; it only makes it the FIRST place searched.
    A tree containing no vllm at all would still pass, off a copy from
    somewhere else entirely. Resolving against `path=[root]` asks the
    question this gate means to ask: what version is in THIS tree?

    With no root given, fall back to the ordinary whole-sys.path lookup.
    """
    try:
        from importlib.metadata import distributions, version
    except ImportError:  # pragma: no cover - python < 3.8
        return None
    try:
        if root:
            wanted = name.lower().replace("_", "-")
            for dist in distributions(path=[root]):
                try:
                    raw = dist.metadata["Name"]
                except Exception:
                    raw = None
                got = (raw or "").strip().lower().replace("_", "-")
                if got == wanted:
                    return dist.version
            return None
        return version(name)
    except Exception:
        return None


def detect_dist_root(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for root in DIST_CANDIDATES:
        if os.path.isfile(os.path.join(root, "vllm", "__init__.py")):
            return root
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=os.path.join(here, "engine-identity.json"))
    ap.add_argument("--profile", required=True,
                    help="which profile section to check (the manifest's key)")
    ap.add_argument("--dist-root", default=None,
                    help="override the site-packages root the paths resolve in")
    ap.add_argument("--strict", action="store_true",
                    help="treat an unfilled (template) manifest as a failure")
    args = ap.parse_args()

    failures: list[str] = []
    unpinned: list[str] = []

    if not os.path.exists(args.manifest):
        print(f"  FAIL  engine-identity manifest present ({args.manifest})")
        print("\nENGINE-IDENTITY FAIL")
        return 1

    try:
        with open(args.manifest) as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"  FAIL  engine-identity manifest readable ({exc.__class__.__name__})")
        print("\nENGINE-IDENTITY FAIL")
        return 1

    profiles = spec.get("profiles") or {}
    section = profiles.get(args.profile)
    if not isinstance(section, dict):
        print(f"  FAIL  manifest carries a '{args.profile}' profile section "
              f"(have: {', '.join(sorted(profiles)) or 'none'})")
        print("\nENGINE-IDENTITY FAIL")
        return 1

    dist_root = detect_dist_root(args.dist_root or section.get("dist_root"))
    if not dist_root:
        print("  FAIL  site-packages root resolvable "
              "(pass --dist-root, or set dist_root in the manifest)")
        print("\nENGINE-IDENTITY FAIL")
        return 1
    print(f"  OK    site-packages root {dist_root}")

    # Read the build stamp from the SAME tree the file hashes come from.
    # The stamp is resolved INSIDE the validated root (see package_version),
    # not off sys.path: a version found anywhere else on sys.path would
    # satisfy a tree that contains no such distribution at all.
    if dist_root not in sys.path:
        sys.path.insert(0, dist_root)

    # ---- build stamp: the engine's own version, not a tag that merely looks right
    # The KEY of each entry is the DISTRIBUTION NAME: it is looked up with
    # importlib.metadata.version(). `make-engine-identity.py` writes it that
    # way, and the manifest template shows the same shape.
    for pkg, expected in (section.get("build_stamp") or {}).items():
        if is_placeholder(expected):
            unpinned.append(f"build stamp {pkg}")
            print(f"  NOT PINNED  build stamp {pkg}")
            continue
        got = package_version(pkg, root=dist_root)
        if got is None:
            failures.append(f"build stamp {pkg}: package not importable")
            print(f"  FAIL  build stamp {pkg} (package not installed)")
        elif got != expected:
            failures.append(f"build stamp {pkg}: {got} != {expected}")
            print(f"  FAIL  build stamp {pkg} = {got!r} (expected {expected!r})")
        else:
            print(f"  OK    build stamp {pkg} = {got}")

    # ---- engine-critical files, as they are in THIS container
    for rel, expected in sorted((section.get("files") or {}).items()):
        path = os.path.join(dist_root, rel)
        if is_placeholder(expected):
            unpinned.append(rel)
            print(f"  NOT PINNED  engine file {rel}")
            continue
        if not is_pinned(expected):
            failures.append(f"engine file {rel}: expected digest is malformed")
            print(f"  FAIL  engine file {rel} (expected digest is not a sha256)")
            continue
        got = sha256_file(path)
        if got is None:
            failures.append(f"engine file {rel}: missing under {dist_root}")
            print(f"  FAIL  engine file {rel} (missing under {dist_root})")
        elif got != expected:
            failures.append(f"engine file {rel}: {got} != {expected}")
            print(f"  FAIL  engine file {rel} (sha256 {got} != {expected})")
        else:
            print(f"  OK    engine file {rel}")

    if unpinned:
        print(f"\n  {len(unpinned)} claim(s) are NOT pinned in this manifest — "
              "the engine identity is NOT proven.")
        print("  Fill them in with: python3 launch/tp3-kit/make-engine-identity.py "
              "--profile %s --container <container> --write" % args.profile)
        if args.strict:
            failures.append(f"{len(unpinned)} claim(s) not pinned (--strict)")

    # A manifest that pins NOTHING claims nothing. Under --strict that is a
    # failure, not a pass: the gate would be reporting a green verdict about
    # an engine it never looked at, which is the one verdict a strict
    # adoption check must never give.
    claim_count = (len(section.get("build_stamp") or {})
                   + len(section.get("files") or {}))
    if args.strict and claim_count == 0:
        failures.append(
            "the manifest pins NO claims at all (--strict): no build stamp and "
            "no file digests, so this gate would certify an engine it never "
            "looked at"
        )
        print("  FAIL  the manifest pins no claims at all — there is nothing "
              "here to certify (--strict)")

    if failures:
        print("\n  failures:")
        for f in failures:
            print(f"    - {f}")
        print("\nENGINE-IDENTITY FAIL")
        return 1

    print("\nENGINE-IDENTITY PASS"
          + (f" ({len(unpinned)} claim(s) still not pinned)" if unpinned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
