#!/bin/bash
# Resumable download of one public CAMELS IllustrisTNG CV snapshot + group catalogue (users.flatironinstitute.org).
#   RAWROOT=<dir> ./download_camels.sh BOX SNAP      -> $RAWROOT/CV_<BOX>/{snapshot,groups}_<SNAP>.hdf5, done_<BOX>_<SNAP>
# (2026-10-08 runs used RAWROOT = <session scratchpad>/impact/raw; files are not kept in the repository.)
B=$1; s=$2
RAWROOT=${RAWROOT:?set RAWROOT}
D=$RAWROOT/CV_$B
mkdir -p $D; cd $D
U=https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_$B
for f in groups_$s.hdf5 snapshot_$s.hdf5; do
  want=$(curl -s -m 60 -I $U/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  for i in $(seq 1 40); do
    have=$(stat -c %s $f.part 2>/dev/null || echo 0)
    [ -s $f ] && break
    [ "$have" = "$want" ] && { mv $f.part $f; break; }
    curl -s -S -C - --speed-limit 50000 --speed-time 60 -o $f.part $U/$f
    sleep 1
  done
  [ -s $f.part ] && [ "$(stat -c %s $f.part)" = "$want" ] && mv $f.part $f
done
if [ -s groups_$s.hdf5 ] && [ -s snapshot_$s.hdf5 ]; then echo "done $B $s $(stat -c %s snapshot_$s.hdf5) $(date)" > done_${B}_$s; else echo "FAILED $B $s" > done_${B}_$s; fi
