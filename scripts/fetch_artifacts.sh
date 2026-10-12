#!/usr/bin/env bash
# Pull the large artifacts (weights, the tokenized corpus, raw run records) back from the private
# artifact repositories, verify every file against the group's SHA256SUMS, and lay them out where
# the training scripts and docs expect them. docs/artifacts.md says what every file is.
#
# usage: scripts/fetch_artifacts.sh [--source modelscope|github] [--with-optimizer] GROUP...
#   GROUP             pretrain-v2, mixture-v2, ablation-v1, or mixture-v2-eval: mixture-v2 without
#                     the 12.3GB training split, i.e. what evaluating a post-trained model needs
#   --source          modelscope (default; fast in China, needs `pip install modelscope` and
#                     `modelscope login`) or github (needs `gh auth login`)
#   --with-optimizer  also fetch pretrain-v2's epoch_1_checkpoint.pth (1.2GB), which only
#                     continuing the pretraining run itself needs
set -euo pipefail
REPO=irroca/whetstone-artifacts
STAGE=results/artifacts
cd "$(dirname "$0")/.."

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
nbytes() { wc -c < "$1" | tr -d ' '; }

matches() {  # name pattern...: whether name matches one of the glob patterns
  local name=$1 p
  shift
  for p in "$@"; do
    case $name in $p) return 0 ;; esac
  done
  return 1
}

# gh release download restarts a file from zero after a dropped connection, and release downloads
# from China run at 70-150KB/s and drop. curl resumes from the partial file; --fail keeps an error
# response out of it, which gh api would append. The token goes in on stdin, not in argv.
# A shorter file is taken for a partial one, so SHA256SUMS and README.md, which grow when a group
# gains a file, are always fetched whole: a fresh SHA256SUMS is what catches any stale download.
github_download() {  # group pattern...: the matching release assets, resuming partial files
  local group=$1 assets name want url dest tries
  shift
  assets=$(gh release view "$group" --repo "$REPO" --json assets \
    --jq '.assets[] | select(.state == "uploaded") | [.name, .size, .apiUrl] | @tsv')
  while IFS=$'\t' read -r name want url <&3; do
    [ -n "$name" ] && matches "$name" "$@" || continue
    dest=$STAGE/$group/$name
    case $name in SHA256SUMS|README.md) rm -f "$dest" ;; esac
    tries=0
    while [ ! -e "$dest" ] || [ "$(nbytes "$dest")" -lt "$want" ]; do
      tries=$((tries + 1))
      [ $tries -le 20 ] || { echo "$dest: incomplete after 20 attempts; rerun to resume"; return 1; }
      gh auth token | sed 's/.*/header = "Authorization: Bearer &"/' |
        curl -sS -L --fail -C - --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
          -H 'Accept: application/octet-stream' -K - -o "$dest" "$url" || sleep 5
    done
    [ "$(nbytes "$dest")" -eq "$want" ] || { echo "$dest: longer than the release asset; delete it and rerun"; return 1; }
  done 3<<< "$assets"
}

fetch() {  # group pattern...: download the group's matching files into $STAGE/group, then verify
  local group=$1 p
  local -a args
  args=()
  shift
  mkdir -p "$STAGE/$group"
  if [ "$source" = github ]; then
    github_download "$group" SHA256SUMS README.md "$@"
  else
    for p in SHA256SUMS README.md "$@"; do args+=("$group/$p"); done
    modelscope download "$REPO" --repo-type model --local-dir "$STAGE" --include "${args[@]}"
  fi
  verify "$group" "$@"
}

# Neither source reports a requested file it does not have, so every file that SHA256SUMS lists
# and the patterns ask for must be present, not just checked when it is.
verify() {  # group pattern...: the group's requested files are all there and match SHA256SUMS
  local group=$1 dir=$STAGE/$1 want name bad=0
  shift
  [ -e "$dir/SHA256SUMS" ] || { echo "$group: SHA256SUMS was not downloaded"; return 1; }
  for name in "$@"; do
    case $name in *[*?[]*) continue ;; esac
    [ -e "$dir/$name" ] || awk -v n="$name" '$2 == n {f = 1} END {exit !f}' "$dir/SHA256SUMS" ||
      { echo "not in $group: $name"; bad=1; }
  done
  while read -r want name; do
    matches "$name" README.md "$@" || continue
    if [ ! -e "$dir/$name" ]; then
      echo "missing: $dir/$name"
      bad=1
    elif [ "$(sha256 "$dir/$name" | cut -d' ' -f1)" != "$want" ]; then
      echo "checksum mismatch: $dir/$name (delete it and rerun)"
      bad=1
    fi
  done < "$dir/SHA256SUMS"
  [ $bad = 1 ] && return 1
  echo "$group: downloads verified"
}

place() {  # src dst: hard-link a verified download into the checkout layout
  mkdir -p "$(dirname "$2")"
  [ "$1" -ef "$2" ] || ln -f "$1" "$2"
}

source=modelscope
with_optimizer=0
groups=""
while [ $# -gt 0 ]; do
  case $1 in
    --source) source=${2:-}; shift ;;
    --with-optimizer) with_optimizer=1 ;;
    pretrain-v2|mixture-v2|mixture-v2-eval|ablation-v1) groups="$groups $1" ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done
case $source in
  modelscope) command -v modelscope >/dev/null || { echo "needs the ModelScope CLI: pip install modelscope && modelscope login"; exit 1; } ;;
  github) command -v gh >/dev/null || { echo "needs the GitHub CLI (https://cli.github.com), then gh auth login"; exit 1; } ;;
  *) echo "--source must be modelscope or github"; exit 2 ;;
esac
[ -n "$groups" ] || { sed -n '2,12p' "$0"; exit 2; }

for group in $groups; do
  case $group in
    pretrain-v2)
      if [ $with_optimizer = 1 ]; then
        fetch pretrain-v2 'pretrain_final.*' '*.tar.gz' epoch_1_checkpoint.pth
      else
        fetch pretrain-v2 'pretrain_final.*' '*.tar.gz'
      fi
      for name in pretrain_final.pth pretrain_final.config.json epoch_1_checkpoint.pth; do
        if [ -e "$STAGE/pretrain-v2/$name" ]; then place "$STAGE/pretrain-v2/$name" "results/pretrain_v2/$name"; fi
      done
      tar -xzf "$STAGE/pretrain-v2/pretrain_v2-records.tar.gz"
      ;;
    ablation-v1)
      fetch ablation-v1 '*'
      for f in "$STAGE"/ablation-v1/*.pretrain_final.*; do
        name=$(basename "$f")
        place "$f" "results/ablation_v1/${name%%.*}/${name#*.}"
      done
      tar -xzf "$STAGE/ablation-v1/ablation_v1-support.tar.gz"
      tar -xzf "$STAGE/ablation-v1/ablation_v1-records.tar.gz"
      ;;
    mixture-v2|mixture-v2-eval)
      if [ $group = mixture-v2 ]; then
        command -v zstd >/dev/null || { echo "mixture-v2 needs zstd (apt-get install zstd, brew install zstd)"; exit 1; }
        fetch mixture-v2 '*'
      else
        fetch mixture-v2 'val.*' 'holdout.*' manifest.json mixture_v2-support.tar.gz
      fi
      for name in train.idx train.meta.json val.bin val.idx val.meta.json val.jsonl \
                  holdout.bin holdout.idx holdout.meta.json holdout.jsonl manifest.json; do
        if [ -e "$STAGE/mixture-v2/$name" ]; then place "$STAGE/mixture-v2/$name" "datasets/mixture_v2/$name"; fi
      done
      tar -xzf "$STAGE/mixture-v2/mixture_v2-support.tar.gz"
      [ $group = mixture-v2 ] || continue
      sums=$PWD/$STAGE/mixture-v2/train.bin.sha256
      if ! (cd datasets/mixture_v2 && sha256 -c --status "$sums" 2>/dev/null); then
        echo "decompressing train.bin (19.8GB)"
        cat "$STAGE"/mixture-v2/train.bin.zst.part-* | zstd -d -f -o datasets/mixture_v2/train.bin
        (cd datasets/mixture_v2 && sha256 -c "$sums")
      fi
      echo "train.bin verified; the compressed parts can go: rm $STAGE/mixture-v2/train.bin.zst.part-*"
      ;;
  esac
done
