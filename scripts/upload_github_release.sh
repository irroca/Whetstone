#!/usr/bin/env bash
# Upload an artifact group staged in results/artifacts/GROUP/ (README.md, SHA256SUMS and every file
# SHA256SUMS lists) to the release GROUP of the private GitHub artifacts repo, then check each asset
# against SHA256SUMS. Files already uploaded completely are skipped, so rerun it until it prints
# GITHUB OK: uploads to GitHub drop with 408s, and a file can only be retried whole. Needs
# `gh auth login`. docs/artifacts.md says how a group is laid out.
#
# usage: scripts/upload_github_release.sh GROUP [TITLE]    (TITLE only when the release is new)
set -uo pipefail
REPO=irroca/whetstone-artifacts
group=${1:?usage: scripts/upload_github_release.sh GROUP [TITLE]}
cd "$(dirname "$0")/../results/artifacts/$group" || exit 1
[ -f SHA256SUMS ] && [ -f README.md ] || { echo "results/artifacts/$group needs README.md and SHA256SUMS"; exit 1; }
tmp=$(mktemp -d)

if ! gh release view "$group" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$group" --repo "$REPO" --title "${2:-$group}" --notes-file README.md || exit 1
fi

worker() {  # file...: upload each file, retrying up to 10 times
  local f tries
  for f in "$@"; do
    tries=0
    until gh release upload "$group" "$f" --repo "$REPO" --clobber >/dev/null 2>&1; do
      tries=$((tries + 1))
      echo "$(date +%T) retry $tries: $f"
      [ $tries -lt 10 ] || { echo "$(date +%T) GAVE UP: $f"; break; }
      sleep 10
    done
    echo "$(date +%T) done: $f"
  done
}

done_list=$(gh release view "$group" --repo "$REPO" --json assets --jq '.assets[] | select(.state == "uploaded") | .name')
todo=()
for f in README.md SHA256SUMS $(awk '$2 != "README.md" {print $2}' SHA256SUMS); do
  grep -qx "$f" <<< "$done_list" || todo+=("$f")
done
echo "$(date +%T) uploading ${#todo[@]} files"
if [ ${#todo[@]} -gt 0 ]; then  # bash 3.2 (macOS) treats expanding an empty array as unset under -u
  a=() b=() c=()
  i=0
  for f in "${todo[@]}"; do
    case $((i % 3)) in 0) a+=("$f") ;; 1) b+=("$f") ;; 2) c+=("$f") ;; esac
    i=$((i + 1))
  done
  [ ${#a[@]} -eq 0 ] || worker "${a[@]}" &
  [ ${#b[@]} -eq 0 ] || worker "${b[@]}" &
  [ ${#c[@]} -eq 0 ] || worker "${c[@]}" &
  wait
fi

echo "$(date +%T) verifying against SHA256SUMS"
gh release view "$group" --repo "$REPO" --json assets \
  --jq '.assets[] | select(.state == "uploaded") | [(.digest | sub("^sha256:"; "")), .name] | @tsv' |
  awk -F'\t' '$2 != "SHA256SUMS" {print $1 "  " $2}' | sort -k2 > "$tmp/github"
sort -k2 SHA256SUMS > "$tmp/expected"
if diff "$tmp/github" "$tmp/expected" >/dev/null; then
  echo "GITHUB OK: every file in SHA256SUMS is uploaded with a matching digest"
  status=0
else
  echo "GITHUB INCOMPLETE OR MISMATCHED (left: GitHub, right: SHA256SUMS):"
  diff "$tmp/github" "$tmp/expected"
  status=1
fi
rm -rf "$tmp"
exit $status
