#!/bin/zsh
# Uploads datasets/mixture_v2 from a Mac to an AutoDL instance; run finish_upload.sh there next.
# One SSH connection to AutoDL tops out at 1-2.5 MB/s, so train.bin goes up zstd-compressed in
# 256MiB parts over 8 connections. Every round first asks the server what it holds and uploads only
# what is missing or the wrong size: the gateway has dropped every connection at once, so a round
# can end early, and rerunning the script resumes the same way. Done only when nothing is missing.
#   AUTODL_SSH="-F $HOME/.ssh/autodl.conf myhost" zsh scripts/autodl/upload.sh
here=${0:A:h}
S=(ssh ${=AUTODL_SSH:--F $HOME/.ssh/whetstone_autodl.conf h800})
dir=/root/autodl-tmp/mixture_v2
part=$(( 256 * 1048576 ))
cd ${here:h:h}/datasets/mixture_v2 || exit 1

if [ ! -f train.bin.zst ]; then
  zstd -T0 -3 -q train.bin -o train.bin.zst.tmp && mv train.bin.zst.tmp train.bin.zst || exit 1
fi
n=$(( ($(stat -f %z train.bin.zst) + part - 1) / part ))
if [ ! -f train.bin.zst.parts.sha256 ]; then
  for i in $(seq 0 $(( n - 1 ))); do
    h=$(dd if=train.bin.zst bs=1m skip=$(( i * 256 )) count=256 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
    printf "%s  part.%03d\n" $h $i
  done > train.bin.zst.parts.sha256.tmp && mv train.bin.zst.parts.sha256.tmp train.bin.zst.parts.sha256
fi
small=(SHA256SUMS manifest.json train.meta.json val.meta.json holdout.meta.json
       train.idx val.idx holdout.idx val.bin holdout.bin holdout.jsonl)
echo "=== start $(date '+%m-%d %H:%M:%S'): $n parts"

complete=0
for round in {1..30}; do
  # One connection per round: each new one costs 10-25 s while the instance's 0.5 CPU is busy.
  state=$($S "mkdir -p $dir/parts && cd $dir && stat -c '%n %s' $small parts/parts.sha256 2>/dev/null; ls parts; echo END")
  if [[ $state != *END ]]; then
    echo "$(date '+%H:%M:%S') round $round: server unreachable"; sleep 30; continue
  fi
  lines=(${(f)state})
  missing=()
  for i in $(seq 0 $(( n - 1 ))); do
    (( ${lines[(Ie)$(printf "part.%03d" $i)]} )) || missing+=($i)
  done
  stale=()
  for f in $small; do
    (( ${lines[(Ie)$f $(stat -f %z $f)]} )) || stale+=($f)
  done
  (( ${lines[(Ie)parts/parts.sha256 $(stat -f %z train.bin.zst.parts.sha256)]} )) || stale+=(parts.sha256)
  echo "=== round $round $(date '+%H:%M:%S'): ${#missing} of $n parts, ${#stale} small files to upload"
  if (( ${#missing} + ${#stale} == 0 )); then complete=1; break; fi

  ( for f in $stale; do
      src=$f dst=$dir/$f
      [ $f = parts.sha256 ] && src=train.bin.zst.parts.sha256 dst=$dir/parts/parts.sha256
      if $S "cat > $dst.tmp && mv $dst.tmp $dst" < $src; then
        echo "$(date '+%H:%M:%S') $f done"
      else
        echo "$(date '+%H:%M:%S') $f failed"
      fi
    done ) &
  (( ${#missing} )) && print -l $missing | xargs -P 8 -n 1 zsh $here/upload_part.sh
  wait
  sleep 10
done
(( complete )) && echo "=== UPLOAD DONE $(date '+%m-%d %H:%M:%S')" || { echo "=== UPLOAD INCOMPLETE"; exit 1; }
