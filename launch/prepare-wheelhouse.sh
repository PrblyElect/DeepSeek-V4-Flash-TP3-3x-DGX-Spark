#!/usr/bin/env bash
# prepare-wheelhouse.sh — build the OFFLINE wheelhouse the capacity profile
# boots from.
#
# The capacity profile used to install its two runtime dependencies while
# the container was starting: a pip install of Ray from PyPI and a pip
# install of a b12x build straight from a GitHub URL. Both are network
# fetches on the boot path, which makes the boot depend on the network, on
# the upstream host being up, and on whatever that host serves that day.
#
# This script moves both fetches OFF the boot path. It is an operator
# pre-flight: run it once per box, before the lane's first boot. Afterwards
# the boot installs from a local directory with pip's index switched off,
# after verifying that the directory still holds what this script recorded.
#
# The b12x SOURCE is pinned by commit id, and the pin is verified twice:
# the fetch is checked against the pinned commit, and the built wheel's
# sha256 is recorded in the manifest the boot verifies.
#
# Run it ON the box that will boot the lane (the wheels must match the
# target platform), as the user that owns the lane:
#
#   ~/ds4flash-tp3/wheelhouse-prep.sh          # staged by stage-env
#   ./launch/prepare-wheelhouse.sh --env-file config/cluster.env.capacity
#
# Requirements on the box: docker (the pinned image is already local), git
# or curl, and a shell. Network access is needed HERE, at pre-flight time
# (PyPI for the Ray wheels and the wheel build backend) — never at boot.
set -euo pipefail

env_file="${DS4FLASH_ENV:-$HOME/ds4flash-tp3/.env.tp3-ds}"
wheelhouse="${WHEELHOUSE_DIR:-$HOME/ds4flash-tp3/wheelhouse}"
ray_version="${RAY_VERSION:-2.55.1}"
keep_src=0

usage() {
  cat >&2 <<'EOF'
usage: prepare-wheelhouse.sh [--env-file FILE] [--wheelhouse DIR] [--keep]

  --env-file FILE   profile env to read the image pin and the b12x commit
                    from (default: the staged .env.tp3-ds on this box)
  --wheelhouse DIR  where the wheels and the manifest are written
                    (default: ~/ds4flash-tp3/wheelhouse)
  --keep            keep the fetched b12x source tree for inspection
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)   env_file="$2"; shift 2 ;;
    --wheelhouse) wheelhouse="$2"; shift 2 ;;
    --keep)       keep_src=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f "$env_file" ]] || { echo "missing env file: $env_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

: "${IMAGE:?the env file must set IMAGE}"
: "${B12X_COMMIT:?the env file must set B12X_COMMIT (the pinned b12x commit)}"
image="${IMAGE}${IMAGE_DIGEST:+@$IMAGE_DIGEST}"

b12x_repo="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"

echo "== wheelhouse pre-flight =="
echo "  env file:   $env_file"
echo "  image:      $image"
echo "  b12x:       $b12x_repo @ $B12X_COMMIT"
echo "  ray:        $ray_version"
echo "  wheelhouse: $wheelhouse"

docker image inspect "$image" >/dev/null 2>&1 || {
  echo "the pinned image is not on this box: $image" >&2
  echo "pull it first (the lane's pull step does that)" >&2
  exit 1
}

mkdir -p "$wheelhouse"
src="$(mktemp -d)"
cleanup() {
  [[ "$keep_src" == 1 ]] || rm -rf "$src"
}
trap cleanup EXIT

# ---- 1. fetch the pinned b12x source and prove its identity --------------
echo "-- fetching b12x source at the pinned commit"
if command -v git >/dev/null 2>&1; then
  git -C "$src" init -q
  git -C "$src" remote add origin "$b12x_repo"
  git -C "$src" fetch -q --depth 1 origin "$B12X_COMMIT"
  git -C "$src" checkout -q --detach FETCH_HEAD
  got="$(git -C "$src" rev-parse HEAD)"
  [[ "$got" == "$B12X_COMMIT" ]] || {
    echo "fetched commit $got != pinned $B12X_COMMIT" >&2; exit 1; }
  echo "   verified commit $got"
  src_tree="$src"
else
  echo "   git not available — fetching the source tarball at the pinned commit"
  tarball="$src/b12x.tar.gz"
  curl -fsSL --max-time 300 \
    "https://codeload.github.com/lukealonso/b12x/tar.gz/$B12X_COMMIT" -o "$tarball"
  src_tree="$src/tree"
  mkdir -p "$src_tree"
  tar -xzf "$tarball" -C "$src_tree" --strip-components=1
  echo "   tarball identity is the commit in its URL ($B12X_COMMIT)"
fi

# ---- 2. build the wheels, INSIDE the pinned image ------------------------
# The wheels must match the target platform and the target interpreter, so
# they are produced by the same image the lane boots. That container needs
# network access (PyPI) — this is the only network use in the whole flow.
echo "-- building wheels inside $image"
docker run --rm --entrypoint bash \
  -v "$src_tree":/src:ro \
  -v "$wheelhouse":/wheelhouse \
  "$image" -lc '
    set -euo pipefail
    python3 -m pip wheel --no-deps --wheel-dir /wheelhouse /src
    python3 -m pip download --only-binary=:all: --dest /wheelhouse "ray=='"$ray_version"'"
  '

# ---- 3. record what the boot must find ----------------------------------
echo "-- recording the manifest"
python3 - "$wheelhouse" "$B12X_COMMIT" "$image" <<'PY'
import hashlib, json, os, sys

root, commit, image = sys.argv[1], sys.argv[2], sys.argv[3]
files = {}
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if not os.path.isfile(path) or name == "wheelhouse.json":
        continue
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    files[name] = h.hexdigest()

manifest = {
    "schema": 1,
    "b12x_commit": commit,
    "files": files,
}
# The image is NOT recorded in the manifest: nothing in the boot path
# checks it, and a key that implies a guarantee it does not provide is
# worse than no key. It is digest-pinned in the profile env file and
# checked by docker, and the pre-flight prints which image it built for.
print(f"wheelhouse built for image {image}")
with open(os.path.join(root, "wheelhouse.json"), "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"   {len(files)} file(s) recorded in {root}/wheelhouse.json")
for name in files:
    print(f"     {name}")
PY

echo "== wheelhouse ready: $wheelhouse =="
echo "The boot now installs from it with pip's index switched off; it fetches nothing."
