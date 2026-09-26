#!/usr/bin/env bash
# check_patch.sh — does patches/dinov3_dapt.patch reproduce repos/dinov3 exactly the way the GCP VM will
# use it? Clones the upstream base commit fresh (LF line endings, like the VM), applies the patch, and
# compares every touched file with the local working tree, ignoring CR/LF differences (the Windows
# checkout has core.autocrlf=true, so a direct `git apply -R --check` on it fails spuriously).
#   bash experiments/check_patch.sh        (WSL)
set -euo pipefail
DT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$DT/patches/dinov3_dapt.patch"
BASE="$(git -C "$DT/repos/dinov3" rev-parse HEAD)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
git clone -q --no-checkout "$DT/repos/dinov3" "$T/d"
git -C "$T/d" -c core.autocrlf=false checkout -q "$BASE"
git -C "$T/d" apply --check "$PATCH"
git -C "$T/d" apply "$PATCH"
echo "patch applies cleanly on upstream $BASE"
n=0
while read -r f; do
  if ! diff -q --strip-trailing-cr "$T/d/$f" "$DT/repos/dinov3/$f" >/dev/null; then
    echo "DIFF $f"; n=$((n + 1))
  fi
done < <(grep -E '^\+\+\+ b/' "$PATCH" | sed 's#^+++ b/##')
echo "files differing from the working tree (ignoring CRLF): $n"
[ "$n" -eq 0 ]
