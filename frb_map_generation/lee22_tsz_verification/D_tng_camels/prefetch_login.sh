#!/bin/bash
# Login-node prefetch of raw CAMELS snapshots for run_direct_multi.pbs (faster link than the compute nodes).
# Downloads to *.part and renames only when complete; the PBS job either finds the complete file or downloads
# it itself (a rename over a file the job is still writing is harmless: the job opens the name afterwards).
W=/lustre/work/kristero10/lee22_tsz_verification_20260929_D
for js in "$@"; do
  BOX=${js%%:*}; s=${js##*:}
  RAW=$W/raw/CV_$BOX; mkdir -p $RAW
  [ -s $W/direct/direct_CV_${BOX}_$s.npz ] && continue
  [ -e $RAW/snapshot_$s.hdf5 ] && continue
  URL=https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_$BOX
  [ -s $RAW/groups_$s.hdf5 ] || curl -s -S --retry 3 -o $RAW/groups_$s.hdf5.part $URL/groups_$s.hdf5 && mv -n $RAW/groups_$s.hdf5.part $RAW/groups_$s.hdf5 2>/dev/null
  curl -s -S --retry 3 -o $RAW/snapshot_$s.hdf5.part $URL/snapshot_$s.hdf5 && { [ -s $W/direct/direct_CV_${BOX}_$s.npz ] && rm -f $RAW/snapshot_$s.hdf5.part || mv $RAW/snapshot_$s.hdf5.part $RAW/snapshot_$s.hdf5; }
  echo "prefetched CV_$BOX $s $(date)"
  # keep at most ~3 prefetched snapshots on disk ahead of the job
  while [ $(ls $W/raw/CV_*/snapshot_*.hdf5 2>/dev/null | wc -l) -ge 3 ]; do sleep 30; done
done
echo PREFETCH_DONE
