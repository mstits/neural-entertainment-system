import numpy as np
from pathlib import Path
D=Path(__file__).parent; RAM=2048
LIVES,ROUND,PX,PY,ENEM=0x2E,0x401,0x203,0x200,0x496
ref=np.load(D/"bb_nescore_reference.npz"); fidx,rram,labels=ref["frame_idx"],ref["ram"],ref["labels"].astype(str)
m=np.fromfile(D/"mesen_bb_shift2.bin",dtype=np.uint8).reshape(-1,RAM)
rows=fidx-1+2
ms=m[rows]; d=np.count_nonzero(ms!=rram,axis=1)
print("shift+2 per-segment (row=F-1+2):")
for seg in ("materialize","round1","seam1","round2","seam2","round3"):
    sel=labels==seg
    if not sel.any(): continue
    dd=d[sel]; lm=(ms[sel,LIVES]==rram[sel,LIVES]).sum(); rm=(ms[sel,ROUND]==rram[sel,ROUND]).sum()
    print(f"  {seg:11s} n={sel.sum():4d} exact={(dd==0).sum():3d} median={np.median(dd):5.0f} p90={np.percentile(dd,90):5.0f} max={dd.max():4d} lives_match={lm}/{sel.sum()} round_match={rm}/{sel.sum()}")
print("\nround1 rows, first 40 + around fork: raw_frame | diff | nes(x,y,enem,score) | mesen(x,y,enem,score)")
sc=lambda r: ''.join(str(int(b)) for b in r[0x445:0x44B])
fork=None
for i in range(len(fidx)):
    if labels[i] not in ("materialize","round1"): break
    r=rram[i]; q=ms[i]
    dx=abs(int(r[PX])-int(q[PX])); dy=abs(int(r[PY])-int(q[PY]))
    flag = (dx>4 or dy>4 or r[ENEM]!=q[ENEM] or sc(r)!=sc(q))
    if fork is None and flag: fork=i
    if i<40 or (fork is not None and abs(i-fork)<=12):
        print(f"  {fidx[i]:5d} | {d[i]:4d} | ({r[PX]:3d},{r[PY]:3d},{r[ENEM]},{sc(r)}) | ({q[PX]:3d},{q[PY]:3d},{q[ENEM]},{sc(q)}) {'<-- FORK' if i==fork else ''}")
print(f"\nGAMEPLAY FORK (x/y >4px or enemies-remaining or score differ): row {fork} raw_frame={fidx[fork]} label={labels[fork]} diff={d[fork]}")
a=np.nonzero(ms[fork]!=rram[fork])[0]; print("  differing addrs at fork:", [hex(x) for x in a[:40]])
# diff at fork-1
a0=np.nonzero(ms[fork-1]!=rram[fork-1])[0]; print(f"  row before fork (raw_frame {fidx[fork-1]}) diff={d[fork-1]} addrs:", [hex(x) for x in a0[:40]])
# per-frame Mesen enemies-remaining / x / y around the nes_core round-1 clear (3108) and Mesen's (3452)
print("\nMesen shift+2 per-frame $0496 (enemies) sampled every 32 frames 1816..3500:")
print("  "+" ".join(f"{f}:{m[f-1+2][ENEM]}" for f in range(1816,3500,32)))
print("nes_core ref $0496 at action rows 1816..3200:")
print("  "+" ".join(f"{fidx[i]}:{rram[i][ENEM]}" for i in range(0,len(fidx)) if fidx[i]<=3200 and i%8==0))
