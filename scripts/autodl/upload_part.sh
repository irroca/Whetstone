#!/bin/zsh
# Uploads part $1 of train.bin.zst for upload.sh. The server renames it into place only once its size
# is right: a dropped connection gives `cat` an EOF, and the part would otherwise look complete. The
# temporary name carries this process's pid, so a writer left over from a dropped connection cannot
# interleave with a retry. Failure exits 1, never ssh's 255, which makes xargs drop every later part.
i=$1
part=$(( 256 * 1048576 ))
dir=/root/autodl-tmp/mixture_v2/parts
cd ${0:A:h:h:h}/datasets/mixture_v2 || exit 1
size=$(stat -f %z train.bin.zst)
left=$(( size - i * part ))
expected=$(( left < part ? left : part ))
name=$(printf "part.%03d" $i)
tmp=$dir/$name.tmp.$$
dd if=train.bin.zst bs=1m skip=$(( i * 256 )) count=256 2>/dev/null |
  ssh ${=AUTODL_SSH:--F $HOME/.ssh/whetstone_autodl.conf h800} \
    "rm -f $dir/$name.tmp*; cat > $tmp && [ \$(stat -c %s $tmp) = $expected ] && mv $tmp $dir/$name"
code=$?
echo "$(date '+%H:%M:%S') $name exit=$code"
(( code == 0 )) || exit 1
