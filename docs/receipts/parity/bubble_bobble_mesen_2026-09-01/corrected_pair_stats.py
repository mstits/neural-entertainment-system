"""Corrected pair: nes_core fs=1 (env.reset lineage, no flags, no load_state) on the corrected tape
vs Mesen 2 at the +2 input phase on the same corrected tape. Mesen frame j applies tape byte j-2;
nes_core step i applies byte i on emulated frame i+1 -> nes_core row i (after byte i) <-> Mesen row i+2 (after byte i)
[same byte] or Mesen row i+1 [same emulated-frame count since power-on]. Report both, plus a +/-1-frame tolerant diff."""
import numpy as np
from pathlib import Path
D=Path(__file__).parent; RAM=2048
LIVES,ROUND,PX,PY,ENEM=0x2E,0x401,0x203,0x200,0x496
o=np.load(D/"ours_fs1_corrected.npy"); m=np.fromfile(D/"mesen_bb_shift2_materialized.bin",dtype=np.uint8).reshape(-1,RAM)
N=o.shape[0]; print(f"nes_core frames={N}  mesen frames={m.shape[0]}")
segs={"prefix":(0,1816),"materialize":(1816,1820),"round1":(1820,3112),"seam1":(3112,3148),"round2":(3148,5188),"seam2":(5188,5224),"round3":(5224,6964)}
for s in (1,2):
    ms=m[np.arange(N)+s]; d=np.count_nonzero(ms!=o,axis=1)
    print(f"\n== mapping mesen_row=i+{s}: exact frames={(d==0).sum()}/{N} median={np.median(d):.0f} p90={np.percentile(d,90):.0f} max={d.max()}")
    print(f"   lives agree {(ms[:,LIVES]==o[:,LIVES]).sum()}/{N}  round agree {(ms[:,ROUND]==o[:,ROUND]).sum()}/{N}  player x agree {(ms[:,PX]==o[:,PX]).sum()}/{N}  player y agree {(ms[:,PY]==o[:,PY]).sum()}/{N}  enemies-left agree {(ms[:,ENEM]==o[:,ENEM]).sum()}/{N}")
    for k,(a,b) in segs.items():
        dd=d[a:b]; print(f"   {k:11s} n={b-a:4d} exact={(dd==0).sum():4d} median={np.median(dd):4.0f} p90={np.percentile(dd,90):4.0f} max={dd.max():4d}")
    nz=np.nonzero(d)[0]; i=nz[0]; a=np.nonzero(ms[i]!=o[i])[0]
    print(f"   first differing frame: nes_core raw_frame={i+1} n={d[i]} addrs={[hex(x) for x in a]}")
    print(f"      nes_core={o[i][a].tolist()}\n      mesen   ={ms[i][a].tolist()}")
    # persistent addrs: differ on >90% of frames
    per=np.count_nonzero(ms!=o,axis=0); pers=np.nonzero(per>0.9*N)[0]
    print(f"   addresses differing on >90% of frames ({len(pers)}): {[hex(x) for x in pers]}")
    # frames where lives or round disagree
    bad=np.nonzero((ms[:,LIVES]!=o[:,LIVES])|(ms[:,ROUND]!=o[:,ROUND]))[0]
    print(f"   frames with lives/round disagreement: {[(int(f)+1,int(o[f][LIVES]),int(ms[f][LIVES]),int(o[f][ROUND]),int(ms[f][ROUND])) for f in bad[:12]]}")
    # gameplay fork test
    sc=lambda r: bytes(r[0x445:0x44B]); fork=None
    for i in range(1816,N):
        r=o[i]; q=ms[i]
        if abs(int(r[PX])-int(q[PX]))>4 or abs(int(r[PY])-int(q[PY]))>4 or r[ENEM]!=q[ENEM] or sc(r)!=sc(q): fork=i; break
    print(f"   gameplay fork (x/y>4px, enemies-left, or score differ): {'none in 6,964 frames' if fork is None else f'raw_frame {fork+1}'}")
# +/-1-frame tolerant (SMB report convention): per frame, min diff over mesen rows i+1..i+3 around the same-byte row
ms1=m[np.arange(N)+1]; ms2=m[np.arange(N)+2]; ms3=m[np.arange(N)+3]
d=np.minimum.reduce([np.count_nonzero(ms1!=o,axis=1),np.count_nonzero(ms2!=o,axis=1),np.count_nonzero(ms3!=o,axis=1)])
print(f"\n== +/-1-frame tolerant: exact={(d==0).sum()}/{N} median={np.median(d):.0f} p90={np.percentile(d,90):.0f} max={d.max()} last-frame={d[-1]}")
for k,(a,b) in segs.items():
    dd=d[a:b]; print(f"   {k:11s} median={np.median(dd):4.0f} p90={np.percentile(dd,90):4.0f} max={dd.max():4d} exact={(dd==0).sum()}")
# transition frame table
print("\n== transition frames (nes_core raw_frame = step+1; Mesen frame = row+1):")
for addr,nm in ((LIVES,'lives'),(ROUND,'round')):
    co=o[:,addr]; cm=m[:,addr]
    eo=[(int(i)+2,int(co[i]),int(co[i+1])) for i in np.nonzero(co[1:]!=co[:-1])[0]]
    em=[(int(i)+2,int(cm[i]),int(cm[i+1])) for i in np.nonzero(cm[1:]!=cm[:-1])[0]]
    print(f"   {nm}: nes_core {eo}\n   {nm}: mesen    {em}")
