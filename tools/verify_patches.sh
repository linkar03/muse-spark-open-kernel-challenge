#!/usr/bin/env bash
# Check the serving patches: hashes match SHA256SUMS (the hashes the judge recorded for each job)
# and every patch applies cleanly to SGLang at the pinned commit.
#
#   tools/verify_patches.sh                      # shallow-fetches the pinned commit into a temp dir
#   SGLANG_DIR=/path/to/sglang tools/verify_patches.sh   # uses an existing checkout (left untouched)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=297e54fd105e974cdf401e5b4d20253a22c16f90
SGLANG_DIR="${SGLANG_DIR:-}"

echo "== hashes"
(cd "$ROOT/serving/patches" && sha256sum -c SHA256SUMS)

if [ -z "$SGLANG_DIR" ]; then
  SGLANG_DIR="$(mktemp -d)/sglang"
  echo "== fetching sgl-project/sglang at $COMMIT into $SGLANG_DIR"
  git init -q "$SGLANG_DIR"
  git -C "$SGLANG_DIR" remote add origin https://github.com/sgl-project/sglang
  git -C "$SGLANG_DIR" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SGLANG_DIR" checkout -q FETCH_HEAD
fi

HEAD_NOW="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
if [ "$HEAD_NOW" != "$COMMIT" ]; then
  echo "error: $SGLANG_DIR is at $HEAD_NOW, expected $COMMIT" >&2
  exit 1
fi

echo "== git apply --check (nothing is modified)"
for patch in "$ROOT"/serving/patches/*.diff; do
  git -C "$SGLANG_DIR" apply --check "$patch"
  echo "applies cleanly: $(basename "$patch")"
done
echo "all patches verified"
