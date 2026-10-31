#!/usr/bin/env python3
"""make-engine-identity.py — fill the engine-identity manifest from a LIVE box.

The manifest ships as a TEMPLATE: the sha256 of the engine-critical files
cannot be computed without the image, and a made-up digest would be worse
than no digest. This tool computes them for real, from a container that is
already running the pinned image, and writes them into the manifest.

What it records:

  * the sha256 of every engine-critical file AS IT IS inside the container
    — so a bind-mounted patch (the TP=3 padding patches) is part of the
    claim, not a gap in it;
  * the engine's own package version — a commit id, not a tag.

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
    for rel in files:
        got = (data.get("files") or {}).get(rel)
        if got is None:
            missing.append(rel)
            new_files[rel] = f"<FILL-IN: {rel} not found at {dist_root}>"
        else:
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
        filled["build_stamp"] = stamp

    print(json.dumps(filled, indent=2, sort_keys=True))
    if missing:
        print(f"\n# NOT FOUND inside the container: {', '.join(missing)}")
        print("# (the manifest keeps a FILL-IN marker for them)")

    if args.write:
        spec["profiles"][args.profile] = filled
        spec["status"] = "filled by make-engine-identity.py — re-audit on every re-pin"
        with open(args.manifest, "w") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\n# written: {args.manifest} (profile {args.profile})")
    else:
        print("\n# dry run — pass --write to update the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
