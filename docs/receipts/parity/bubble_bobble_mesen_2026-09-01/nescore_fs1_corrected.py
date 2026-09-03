"""nes_core frame_skip=1, no flags, env.reset() lineage, NO load_state, on the CORRECTED tape
(prefix + 4-frame NOOP materialize block + actions). Compare to the fs=4 solver-lineage rows,
whose true frame index is label+4 (the materialize step's 4 frames were not counted)."""
import sys, numpy as np
from pathlib import Path
REPO=Path("/Users/stits/Documents/macos-emulation-and-training"); sys.path.insert(0,str(REPO))
import nes_core
D=Path(__file__).parent; RAM=2048; LIVES,ROUND=0x2E,0x401
tape=(D/"bb_tape_materialized.bin").read_bytes()
env=nes_core.NESEnvironment(rom_path=str(REPO/"roms"/"Bubble Bobble (USA).nes"),frame_skip=1); env.reset()
ours=np.zeros((len(tape),RAM),np.uint8)
for i,b in enumerate(tape):
    env.step(int(b)); ours[i]=np.frombuffer(bytes(env.get_ram_range(0,RAM)),np.uint8)
np.save(D/"ours_fs1_corrected.npy",ours)
print(f"nes_core fs=1 corrected tape: final lives={ours[-1][LIVES]} round={ours[-1][ROUND]}")
for addr,nm in ((LIVES,'lives'),(ROUND,'round')):
    col=ours[:,addr]; ch=np.nonzero(col[1:]!=col[:-1])[0]
    for i in ch: print(f"   raw_frame={i+2:5d} {nm} {col[i]}->{col[i+1]}")
ref=np.load(D/"bb_nescore_reference.npz"); fidx,rram,labels=ref["frame_idx"],ref["ram"],ref["labels"].astype(str)
print("\nvs fs=4 native solver lineage, row = ours[label+4-1]:")
o=ours[fidx+4-1]; d=np.count_nonzero(o!=rram,axis=1)
for seg in ("materialize","round1","seam1","round2","seam2","round3"):
    s=labels==seg; dd=d[s]
    print(f"  {seg:11s} n={s.sum():4d} exact={(dd==0).sum():4d} median={np.median(dd):4.0f} max={dd.max():4d} lives_match={(o[s,LIVES]==rram[s,LIVES]).sum()}/{s.sum()} round_match={(o[s,ROUND]==rram[s,ROUND]).sum()}/{s.sum()}")
nz=np.nonzero(d)[0]
if len(nz):
    i=nz[0]; a=np.nonzero(o[i]!=rram[i])[0]
    print(f"  first differing row: label={fidx[i]} (true frame {fidx[i]+4}) seg={labels[i]} n={d[i]} addrs={[hex(x) for x in a[:30]]}")
else: print("  BYTE-EXACT on all 1287 rows: fs=1 full-render == fs=4 skip-render lineage")
