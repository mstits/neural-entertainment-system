import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
sys.path.insert(0, str(REPO))

from scripts.discover_observables import Discoverer

ROM = str(REPO / "roms/DuckTales 2 (USA).nes")
STATE = REPO / "roms/DuckTales 2 (USA)_start.state.bin"

disc = Discoverer(ROM, str(STATE), frame_skip=4, forward="right", seed=1)
drives = disc.death_drives()

for d in drives:
    log = d["log"]
    churn = (np.diff(log.astype(np.int16), axis=0) != 0).sum(1)
    thr = disc.reset_threshold(log)
    big = np.where(churn > thr)[0]
    print(f"rep={d['rep']} reset_thr={thr:.0f} median_churn={np.median(churn):.0f} "
          f"max_churn={churn.max()} mass-rewrite steps (>{thr:.0f}): {big.tolist()}")
    for addr in (11, 142, 9, 176, 340):
        col = log[:, addr].astype(int)
        diffs = np.diff(col)
        idx = np.nonzero(diffs)[0]
        print(f"    addr={addr}: change steps = {idx.tolist()}")
disc.close()
