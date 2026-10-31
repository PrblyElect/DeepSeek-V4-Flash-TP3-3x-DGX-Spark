#!/usr/bin/env python3
"""verify-wheelhouse.py — check the offline wheelhouse before installing from it.

The capacity profile boots with NO network fetch: b12x and Ray are installed
from a wheelhouse that was prepared ahead of time (launch/prepare-wheelhouse.sh).
This script is the boot's first step, and it refuses to continue unless the
wheelhouse is exactly what the pre-flight recorded:

  * every file's sha256 matches the manifest written at pre-flight time;
  * the b12x source commit in the manifest is the pinned one;
  * a b12x wheel and a Ray wheel are actually there.

A wheelhouse that was edited, truncated or swapped after the pre-flight
fails here, before the engine starts, instead of halfway through a boot.

Usage (inside the container):
    python3 /opt/tp3-kit/verify-wheelhouse.py --wheelhouse /wheelhouse \\
        --b12x-commit "$B12X_COMMIT"

Exit codes: 0 = wheelhouse verified, 1 = verification failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheelhouse", default="/wheelhouse")
    ap.add_argument("--b12x-commit", default=os.environ.get("B12X_COMMIT", ""))
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    root = args.wheelhouse
    manifest_path = args.manifest or os.path.join(root, "wheelhouse.json")
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL  {msg}")

    if not os.path.isdir(root):
        fail(f"wheelhouse present ({root})")
        print("\nWHEELHOUSE FAIL")
        return 1
    print(f"  OK    wheelhouse directory {root}")

    if not os.path.isfile(manifest_path):
        fail(f"wheelhouse manifest present ({manifest_path}) — run "
             "launch/prepare-wheelhouse.sh on this box first")
        print("\nWHEELHOUSE FAIL")
        return 1

    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        fail(f"wheelhouse manifest readable ({exc.__class__.__name__})")
        print("\nWHEELHOUSE FAIL")
        return 1

    commit = manifest.get("b12x_commit") or ""
    if args.b12x_commit:
        if commit != args.b12x_commit:
            fail(f"b12x source commit {commit or 'unset'} != pinned {args.b12x_commit}")
        else:
            print(f"  OK    b12x source commit {commit}")
    elif not commit:
        fail("b12x source commit recorded in the manifest")
    else:
        print(f"  OK    b12x source commit {commit} (no pin supplied to compare)")

    files = manifest.get("files") or {}
    if not files:
        fail("wheelhouse manifest lists no files")
    for name, want in sorted(files.items()):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            fail(f"wheel file present ({name})")
            continue
        got = sha256_file(path)
        if got != want:
            fail(f"wheel file {name} (sha256 {got} != {want})")
        else:
            print(f"  OK    wheel file {name}")

    names = os.listdir(root)
    if not any(n.lower().startswith("b12x-") and n.endswith(".whl") for n in names):
        fail("a b12x wheel is in the wheelhouse")
    if not any(n.lower().startswith("ray-") and n.endswith(".whl") for n in names):
        fail("a ray wheel is in the wheelhouse")

    # Nothing may be in the wheelhouse that the pre-flight did not record.
    # This is the ADD case, and it is the one that matters: the boot
    # installs with `--find-links`, and pip resolves the HIGHEST version it
    # finds there. A newer wheel dropped into the directory after the
    # pre-flight would therefore be installed in preference to the audited
    # one, with every recorded file still hash-matching and nothing to
    # report. The pre-flight records a complete listing, so the check is
    # exact.
    recorded = set(files)
    manifest_name = os.path.basename(manifest_path)
    for name in sorted(names):
        if name in recorded or name == manifest_name:
            continue
        fail(f"unexpected file in the wheelhouse ({name}) — it was not "
             f"recorded by the pre-flight, and pip would prefer it")

    print("\nWHEELHOUSE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
