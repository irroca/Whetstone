#!/bin/bash
# On the instance, after upload.sh: checks every part, joins and decompresses train.bin, verifies
# every file against SHA256SUMS (written on the Mac right after tokenizing), then deletes the parts.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/mixture_v2
n=$(wc -l < parts/parts.sha256)
(cd parts && sha256sum --quiet -c parts.sha256)
echo "all $n parts match"
cat $(printf 'parts/part.%03d ' $(seq 0 $((n - 1)))) | zstd -d -q -f -o train.bin.tmp
mv train.bin.tmp train.bin
sha256sum -c SHA256SUMS
rm -rf parts
df -h /root/autodl-tmp | tail -1
echo "FINISH EXIT=0"
