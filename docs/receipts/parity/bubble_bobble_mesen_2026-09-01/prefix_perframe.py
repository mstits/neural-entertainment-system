"""Fresh nes_core frame_skip=1 dump of the 1,816-frame cold-boot prefix (env.reset() lineage,
same as scripts/capture_start_state.py) -> per-frame RAM; then per-frame compare vs each
shifted Mesen dump to locate the first divergence and the residue under each phase."""
import sys, numpy as np
from pathlib import Path
REPO=Path("/Users/stits/Documents/macos-emulation-and-training"); sys.path.insert(0,str(REPO))
import nes_core
D=Path(__file__).parent; RAM=2048; LIVES=0x2E; ROUND=0x401
prefix=(Path("/Users/stits/Documents/personal_os/reports/macos-emulation-and-training/2026-09-01-outstanding/receipts-bb-mesen-parity/bb_prefix.bin")).read_bytes()
tape=(D/"bb_continuous_tape.bin").read_bytes()
assert tape[:1816]==prefix and len(prefix)==1816
rom=str(REPO/"roms"/"Bubble Bobble (USA).nes")
env=nes_core.NESEnvironment(rom_path=rom,frame_skip=1); env.reset()
ours=np.zeros((1816,RAM),np.uint8)
for i,b in enumerate(prefix):
    env.step(int(b)); ours[i]=np.frombuffer(bytes(env.get_ram_range(0,RAM)),np.uint8)
blob=bytes(env.save_state())
ss=(REPO/"roms"/"Bubble Bobble (USA)_start.state.bin").read_bytes()
print(f"[nes_core fs=1 prefix] start_state byte-exact: {blob==ss}")
np.save(D/"ours_prefix_fs1.npy",ours)
def first_change(col,val):
    i=np.nonzero(col==val)[0]; return int(i[0])+1 if len(i) else None
print(f"[nes_core] lives->3 at raw_frame {first_change(ours[:,LIVES],3)}, round->1 at raw_frame {first_change(ours[:,ROUND],1)}")
for k in (0,1,2,3,-1):
    m=np.fromfile(D/f"mesen_bb_shift{k}.bin",dtype=np.uint8).reshape(-1,RAM)
    print(f"\n== Mesen shift={k:+d}: lives->3 at raw_frame {first_change(m[:,LIVES],3)}, round->1 at raw_frame {first_change(m[:,ROUND],1)}")
    for s in range(-1,3):
        rows=np.arange(1816)+k+s  # nes_core step i (byte i) ran on emulated frame i+1; Mesen applies byte i at frame i+k; try +s
        v=(rows>=0)&(rows<m.shape[0]); ms=m[rows[v]]; oo=ours[v]
        d=np.count_nonzero(ms!=oo,axis=1)
        nz=np.nonzero(d)[0]
        first=int(nz[0])+1 if len(nz) else None
        addrs=np.nonzero(ms[nz[0]]!=oo[nz[0]])[0] if len(nz) else []
        print(f"  cmp mesen_row=i+{k}{s:+d}: n={v.sum()} exact={(d==0).sum()} median={np.median(d):.0f} max={d.max()} first_div_frame={first} n_at_first={d[nz[0]] if len(nz) else 0} addrs={[hex(a) for a in addrs[:20]]}")
        # diff at prefix end
        print(f"      diff at last prefix row: {d[-1]}; diff at frame 257: {d[256]}; at 680: {d[679]}")
