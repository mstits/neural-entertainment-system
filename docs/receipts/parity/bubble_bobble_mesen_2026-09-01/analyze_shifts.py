import numpy as np, sys
from pathlib import Path
D = Path(__file__).parent
RAM=2048; LIVES=0x002E; ROUND=0x0401
ref = np.load(D/"bb_nescore_reference.npz")
fidx, rram, labels = ref["frame_idx"], ref["ram"], ref["labels"]
if labels.dtype.kind=='S': labels = labels.astype(str)
print(f"nes_core reference: rows={rram.shape[0]} first raw_frame={fidx[0]} last={fidx[-1]}")
def timeline(m, name):
    ev=[]
    for addr,nm in ((LIVES,'lives'),(ROUND,'round')):
        col=m[:,addr]; ch=np.nonzero(col[1:]!=col[:-1])[0]
        ev += [(int(i)+2, nm, int(col[i]), int(col[i+1])) for i in ch]   # raw_frame = row+1
    ev.sort()
    return ev
# nes_core timeline from the reference rows (action granularity)
print("\nnes_core reference lives/round transitions (raw_frame after step; action granularity):")
prev=None
for f,r,l in zip(fidx,rram,labels):
    cur=(int(r[LIVES]),int(r[ROUND]))
    if prev is not None and cur!=prev: print(f"  raw_frame={f} {prev}->{cur} [{l}]")
    prev=cur
print(f"  end: raw_frame={fidx[-1]} lives={rram[-1][LIVES]} round={rram[-1][ROUND]}")
for k in (0,1,2,3,-1):
    m=np.fromfile(D/f"mesen_bb_shift{k}.bin",dtype=np.uint8).reshape(-1,RAM)
    print(f"\n===== Mesen shift={k:+d}: {m.shape[0]} frames; final lives={m[-1][LIVES]} round={m[-1][ROUND]}")
    ev=timeline(m,k)
    for e in ev:
        if e[1]=='lives' or (e[1]=='round'): print(f"  raw_frame={e[0]:5d} {e[1]} {e[2]}->{e[3]}")
    # compare vs reference: nes_core byte F-1 ran on emulated frame F (reset advance = frame 0).
    # In shift-k tape, byte F-1 is applied at Mesen frame F-1+k -> dump row F-1+k. sweep s around that.
    best=None
    for s in range(-3,4):
        rows = fidx-1+k+s
        v=(rows>=0)&(rows<m.shape[0])
        ms=m[rows[v]]; rr=rram[v]
        d=np.count_nonzero(ms!=rr,axis=1)
        lm=int(np.count_nonzero(ms[:,LIVES]==rr[:,LIVES])); rm=int(np.count_nonzero(ms[:,ROUND]==rr[:,ROUND]))
        line=f"  cmp row=F-1{k:+d}{s:+d}: n={v.sum()} exact={int((d==0).sum())} median={np.median(d):.0f} p90={np.percentile(d,90):.0f} max={d.max()} lives_match={lm}/{v.sum()} round_match={rm}/{v.sum()}"
        print(line)
        if best is None or np.median(d)<best[0]: best=(np.median(d),s,line)
    print(f"  BEST offset s={best[1]:+d}: {best[2].strip()}")
