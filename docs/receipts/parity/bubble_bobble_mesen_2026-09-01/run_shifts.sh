#!/bin/zsh
set -u
cd /Users/stits/Documents/macos-emulation-and-training
OUT=<SCRATCH>
TAPE=$OUT/bb_continuous_tape.bin
for k in 2 1 3 -1; do
  echo "=== shift=$k  start: $(date '+%F %T %Z')"
  python3 - "$k" "$TAPE" <<'PY'
import sys
k=int(sys.argv[1]); t=open(sys.argv[2],'rb').read()
if k>=0: t2=b'\x00'*k+t
else: t2=t[-k:]
open('/tmp/cv_tape.bin','wb').write(t2)
import hashlib; print(f"tape len={len(t2)} sha256={hashlib.sha256(t2).hexdigest()}")
PY
  rm -f /tmp/mesen_cv_ram.bin /tmp/mesen_cv_tape.status.txt
  /usr/bin/time -p nice -n 15 /Applications/Mesen.app/Contents/MacOS/Mesen --testRunner --noaudio --novideo --noinput scripts/tracing/mesen_cv_tape_dump.lua "roms/Bubble Bobble (USA).nes" --timeout=90 > $OUT/mesen_shift${k}.log 2>&1
  echo "exit=$? status: $(cat /tmp/mesen_cv_tape.status.txt 2>/dev/null)"
  grep -E "^(real|user|sys)" $OUT/mesen_shift${k}.log
  cp /tmp/mesen_cv_ram.bin $OUT/mesen_bb_shift${k}.bin
  ls -l $OUT/mesen_bb_shift${k}.bin | awk '{print "bytes="$5}'
  shasum -a 256 $OUT/mesen_bb_shift${k}.bin
done
cp $OUT/cv_tape.bin.orig /tmp/cv_tape.bin
echo "restored /tmp/cv_tape.bin: $(shasum -a 256 /tmp/cv_tape.bin)"
echo "=== end: $(date '+%F %T %Z')"
