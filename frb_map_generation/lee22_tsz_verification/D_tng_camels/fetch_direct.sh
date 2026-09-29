#!/bin/bash
# Copy all direct_CV_*.npz from idark (base64 over ssh.exe, md5-verified; scp.exe writes to the Windows temp dir failed).
R=/lustre/work/kristero10/lee22_tsz_verification_20260929_D/direct
L=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/direct
SSH="/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes idark"
mkdir -p $L
for f in $($SSH "ls $R" | tr -d '\r'); do
  rm_md5=$($SSH "md5sum $R/$f" | tr -d '\r' | cut -d' ' -f1)
  if [ -s $L/$f ] && [ "$(md5sum $L/$f | cut -d' ' -f1)" = "$rm_md5" ]; then continue; fi
  $SSH "base64 -w0 $R/$f" | tr -d '\r\n' | base64 -d > $L/$f
  [ "$(md5sum $L/$f | cut -d' ' -f1)" = "$rm_md5" ] && echo "ok $f" || echo "MD5 MISMATCH $f"
done
