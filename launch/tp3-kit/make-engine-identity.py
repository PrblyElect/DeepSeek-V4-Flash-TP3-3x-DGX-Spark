#!/usr/bin/env python3
"""make-engine-identity.py — fill the engine-identity manifest from a LIVE box.

The manifest ships PARTLY FILLED: the build stamps and the digests of the
patch files this repository itself ships are pinned in the tree, while the
sha256 of the image-baked engine files cannot be computed without the image,
and a made-up digest would be worse than no digest. This tool computes the
rest for real, from a container that is already running the pinned image, and
writes them into the manifest.

What it records:

  * the sha256 of every engine-critical file AS IT IS inside the container
    — so a bind-mounted patch (the TP=3 padding patches) is part of the
    claim, not a gap in it;
  * the engine's own package version — the distribution version string, not a
    tag: it carries the commit id, and for a nightly build the packaging-date
    segment as well (the gate compares that string exactly).

It never starts a container and never touches a GPU: it hashes files and
reads package metadata inside a container that is already up.

Usage:
    python3 launch/tp3-kit/make-engine-identity.py --profile speed \\
        --container ds4flash-tp3-head --write

Run it on the box that owns the container (the reader's head node). The
filled manifest is then re-staged to the boxes with the rest of the kit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "engine-identity.json")

# Hashing + version read. Deliberately does NOT import vllm: importing the
# engine package can pull CUDA into the process, and the lane is already
# holding the GPU.
SNIPPET = r"""
import hashlib, json, os, sys

root = sys.argv[1]
paths = json.loads(sys.argv[2])
out = {"files": {}, "packages": {}}
for rel in paths:
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        out["files"][rel] = None
        continue
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    out["files"][rel] = h.hexdigest()
for pkg in ("vllm", "torch"):
    try:
        from importlib.metadata import version
        out["packages"][pkg] = version(pkg)
    except Exception:
        out["packages"][pkg] = None
print(json.dumps(out))
"""

DIST_CANDIDATES = (
    "/usr/local/lib/python3.12/dist-packages",
    "/opt/venv/lib/python3.12/site-packages",
)

HEX = set("0123456789abcdef")


def is_pinned(value) -> bool:
    """A value the gate will actually check: a full lowercase sha256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in HEX for c in value)
    )


def is_claim(value) -> bool:
    """A non-digest claim (a build stamp) the gate will actually check."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "FILL-IN" in text:
        return False
    return not (text.startswith("<") and text.endswith(">"))


def summarize(spec: dict) -> str:
    """A status line that says what is pinned and what still needs a container."""
    per = []
    total_p = total_c = 0
    for name in sorted(spec.get("profiles") or {}):
        section = spec["profiles"][name]
        pinned = claims = 0
        for value in (section.get("build_stamp") or {}).values():
            claims += 1
            pinned += 1 if is_claim(value) else 0
        for value in (section.get("files") or {}).values():
            claims += 1
            pinned += 1 if is_pinned(value) else 0
        per.append(f"{name} {pinned}/{claims}")
        total_p += pinned
        total_c += claims
    if total_c == 0:
        return "no claims in this manifest — nothing is pinned"
    word = "FULLY PINNED" if total_p == total_c else "PARTIALLY PINNED"
    return (
        f"{word} — {total_p}/{total_c} claims ({', '.join(per)}) — "
        f"filled by make-engine-identity.py; re-audit on every re-pin"
    )


def detect_dist_root(container: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    probe = ";".join(
        f'[ -f "{r}/vllm/__init__.py" ] && echo "{r}"'
        for r in DIST_CANDIDATES
    )
    try:
        res = subprocess.run(
            ["docker", "exec", container, "bash", "-c", probe],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"cannot probe {container}: {exc}")
    roots = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not roots:
        raise SystemExit(
            f"no vllm install found in {container} "
            f"(looked in {', '.join(DIST_CANDIDATES)})"
        )
    return roots[0]


def collect(container: str, dist_root: str, paths: list[str]) -> dict:
    cmd = ["docker", "exec", "-i", container, "python3", "-",
           dist_root, json.dumps(paths)]
    try:
        res = subprocess.run(cmd, input=SNIPPET, capture_output=True,
                             text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"cannot read {container}: {exc}")
    if res.returncode != 0:
        raise SystemExit(
            f"the probe inside {container} failed:\n{res.stderr.strip()}"
        )
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SystemExit(f"unreadable probe output from {container}:\n{res.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--container", required=True,
                    help="a container already running the pinned image")
    ap.add_argument("--dist-root", default=None)
    ap.add_argument(
        "--accept-drift",
        action="store_true",
        help="take the container's bytes over a pinned claim that has drifted "
        "(without this, a drifted pin is KEPT and the run exits non-zero)",
    )
    ap.add_argument("--write", action="store_true",
                    help="update the manifest in place (default: print only)")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        spec = json.load(fh)
    section = (spec.get("profiles") or {}).get(args.profile)
    if not isinstance(section, dict):
        raise SystemExit(
            f"the manifest has no '{args.profile}' profile section "
            f"(have: {', '.join(sorted(spec.get('profiles') or {}))})"
        )
    files = sorted((section.get("files") or {}).keys())
    if not files:
        raise SystemExit(f"the '{args.profile}' section lists no files")

    dist_root = detect_dist_root(args.container, args.dist_root)
    print(f"# profile {args.profile} · container {args.container} · {dist_root}")
    data = collect(args.container, dist_root, files)

    filled = dict(section)
    new_files = {}
    missing = []
    drift = []
    old_files = section.get("files") or {}
    downgraded = []
    for rel in files:
        got = (data.get("files") or {}).get(rel)
        was = old_files.get(rel)
        if got is None:
            missing.append(rel)
            if is_pinned(was):
                # A claim that WAS pinned is kept as it was: downgrading it to
                # a marker would silently un-pin it, and the gate would then
                # pass on a file it can no longer see.
                downgraded.append(rel)
                new_files[rel] = was
            else:
                new_files[rel] = f"<FILL-IN: {rel} not found at {dist_root}>"
            continue
        if is_pinned(was) and was != got:
            drift.append(f"{rel}: manifest {was[:12]}… → container {got[:12]}…")
            if not args.accept_drift:
                # KEEP the pinned value. Reporting the drift is not enough:
                # overwriting a pin with the container's bytes would let a
                # drifted engine be laundered into a passing --strict gate by
                # regenerating the manifest.
                new_files[rel] = was
                continue
        new_files[rel] = got
    filled["files"] = new_files
    filled["dist_root"] = dist_root

    stamp = {}
    for pkg, val in (data.get("packages") or {}).items():
        if val is None:
            continue
        # The KEY is the distribution name, because that is what the gate
        # looks up: boot-gate-identity.py reads each key of `build_stamp`
        # with importlib.metadata.version(). A key like "vllm_version" is a
        # field name, not a distribution, and the gate would report it as
        # "package not installed" forever.
        if pkg == "vllm":
            stamp["vllm"] = val
    if stamp:
        old_stamp = (section.get("build_stamp") or {}).get("vllm")
        if is_claim(old_stamp) and old_stamp != stamp.get("vllm"):
            drift.append(
                f"build_stamp.vllm: manifest {old_stamp} → container {stamp.get('vllm')}"
            )
            if not args.accept_drift:
                filled["build_stamp"] = dict(section.get("build_stamp") or {})
            else:
                filled["build_stamp"] = stamp
        else:
            filled["build_stamp"] = stamp

    print(json.dumps(filled, indent=2, sort_keys=True))
    if missing:
        print(f"\n# NOT FOUND inside the container: {', '.join(missing)}")
        print("# (the manifest keeps a FILL-IN marker for them)")
    if downgraded:
        print("\n# DOWNGRADED to a FILL-IN marker (these claims were pinned):")
        for row in downgraded:
            print(f"#   {row}")
    if drift:
        print("\n# CHANGED against the manifest's pinned value — the container holds")
        print("# DIFFERENT bytes than the manifest claims:")
        for row in drift:
            print(f"#   {row}")
        if args.accept_drift:
            print("# --accept-drift: the container's bytes WILL replace the pin.")
        else:
            print("# the PINNED values are KEPT. Re-audit, then re-run with")
            print("# --accept-drift to take the container's bytes deliberately.")

    if args.write:
        spec["profiles"][args.profile] = filled
        spec["status"] = summarize(spec)
        with open(args.manifest, "w") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\n# written: {args.manifest} (profile {args.profile})")
    else:
        print("\n# dry run — pass --write to update the manifest")

    # A drifted pin is not a successful run: the exit status has to carry it,
    # because a scripted generate-then-verify cannot read stdout.
    if drift and not args.accept_drift:
        print("\n# EXIT 3: a pinned claim drifted and was NOT overwritten.")
        return 3
    if downgraded:
        print("\n# EXIT 4: a previously pinned claim was downgraded to a marker.")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
