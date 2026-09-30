#!/bin/bash
# Copy konietzka_defs_CV_*.npz from idark (base64 over ssh.exe, md5-verified).
R=/lustre/work/kristero10/lee22_konietzka_check_20260930/out
L=/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_konietzka_check_20260930/camels_defs
SSH="/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes idark"
mkdir -p $L
for f in $($SSH "ls $R" | tr -d '\r'); do
  rm_md5=$($SSH "md5sum $R/$f" | tr -d '\r' | cut -d' ' -f1)
  if [ -s $L/$f ] && [ "$(md5sum $L/$f | cut -d' ' -f1)" = "$rm_md5" ]; then continue; fi
  $SSH "base64 -w0 $R/$f" | tr -d '\r\n' | base64 -d > $L/$f
  [ "$(md5sum $L/$f | cut -d' ' -f1)" = "$rm_md5" ] && echo "ok $f" || echo "MD5 MISMATCH $f"
done
