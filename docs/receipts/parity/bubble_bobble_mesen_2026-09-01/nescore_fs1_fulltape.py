"""nes_core frame_skip=1, hw_flags none, env.reset() lineage, NO load_state: step all 6,960 tape
bytes one real frame each (the full-render path Mesen also runs). Per-frame RAM. Then compare to
(a) the fs=4 native reference rows (solver lineage) and (b) Mesen shift+2 per frame."""
import sys, numpy as np
from pathlib import Path
REPO=Path("/Users/stits/Documents/macos-emulation-and-training"); sys.path.insert(0,str(REPO))
import nes_core
D=Path(__file__).parent; RAM=2048; LIVES,ROUND,PX,PY,ENEM=0x2E,0x401,0x203,0x200,0x496
tape=(D/"bb_continuous_tape.bin").read_bytes()
env=nes_core.NESEnvironment(rom_path=str(REPO/"roms"/"Bubble Bobble (USA).nes"),frame_skip=1); env.reset()
ours=np.zeros((len(tape),RAM),np.uint8)
for i,b in enumerate(tape):
    env.step(int(b)); ours[i]=np.frombuffer(bytes(env.get_ram_range(0,RAM)),np.uint8)
np.save(D/"ours_fs1_fulltape.npy",ours)
def tl(a,name):
    print(f"{name}: final lives={a[-1][LIVES]} round={a[-1][ROUND]}")
    for addr,nm in ((LIVES,'lives'),(ROUND,'round')):
        col=a[:,addr]; ch=np.nonzero(col[1:]!=col[:-1])[0]
        for i in ch: print(f"   raw_frame={i+2:5d} {nm} {col[i]}->{col[i+1]}")
tl(ours,"nes_core fs=1 x4 full tape (no flags, env.reset lineage)")
ref=np.load(D/"bb_nescore_reference.npz"); fidx,rram,labels=ref["frame_idx"],ref["ram"],ref["labels"].astype(str)
print("\n(a) vs fs=4 native reference rows (solver lineage), row = ours[F-1]:")
o=ours[fidx-1]; d=np.count_nonzero(o!=rram,axis=1)
for seg in ("materialize","round1","seam1","round2","seam2","round3"):
    s=labels==seg; dd=d[s]
    print(f"  {seg:11s} n={s.sum():4d} exact={(dd==0).sum():3d} median={np.median(dd):5.0f} max={dd.max():4d} lives_match={(o[s,LIVES]==rram[s,LIVES]).sum()}/{s.sum()} round_match={(o[s,ROUND]==rram[s,ROUND]).sum()}/{s.sum()}")
nz=np.nonzero(d)[0]
if len(nz):
    i=nz[0]; a=np.nonzero(o[i]!=rram[i])[0]
    print(f"  first differing row: raw_frame={fidx[i]} label={labels[i]} n={d[i]} addrs={[hex(x) for x in a[:30]]}")
    print(f"     fs4 vals={rram[i][a[:30]].tolist()}\n     fs1 vals={o[i][a[:30]].tolist()}")
print("\n(b) vs Mesen shift+2 per frame:")
m=np.fromfile(D/"mesen_bb_shift2.bin",dtype=np.uint8).reshape(-1,RAM)
for s in (1,2):
    rows=np.arange(len(tape))+s; ms=m[rows]; d=np.count_nonzero(ms!=ours,axis=1)
    seg=lambda lo,hi: d[lo:hi]
    print(f"  mesen_row=i+{s}: overall exact={(d==0).sum()} median={np.median(d):.0f} max={d.max()} | prefix median={np.median(seg(0,1816)):.0f} | round1 median={np.median(seg(1816,3108)):.0f} max={seg(1816,3108).max()} | round2 median={np.median(seg(3144,5184)):.0f} | round3 median={np.median(seg(5220,6960)):.0f}")
    print(f"     lives agree on {(ms[:,LIVES]==ours[:,LIVES]).sum()}/{len(tape)} frames, round on {(ms[:,ROUND]==ours[:,ROUND]).sum()}/{len(tape)}")
    # gameplay fork: x/y > 4 px, or enemies/score differ, in round1 onward
    sc=lambda r: bytes(r[0x445:0x44B])
    fork=None
    for i in range(1816,len(tape)):
        r=ours[i]; q=ms[i]
        if abs(int(r[PX])-int(q[PX]))>4 or abs(int(r[PY])-int(q[PY]))>4 or r[ENEM]!=q[ENEM] or sc(r)!=sc(q):
            fork=i; break
    if fork is not None:
        a=np.nonzero(ms[fork]!=ours[fork])[0]
        print(f"     gameplay fork at raw_frame={fork+1}: ours(x,y,enem)=({ours[fork][PX]},{ours[fork][PY]},{ours[fork][ENEM]}) mesen=({ms[fork][PX]},{ms[fork][PY]},{ms[fork][ENEM]}) diff={d[fork]} addrs={[hex(x) for x in a[:30]]}")
    else: print("     no gameplay fork by that test")
