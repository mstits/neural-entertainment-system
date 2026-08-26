import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
sys.path.insert(0, str(REPO))

from scripts.discover_observables import Discoverer
from nes_core import Pool

ROM = str(REPO / "roms/DuckTales 2 (USA).nes")
STATE = REPO / "roms/DuckTales 2 (USA)_start.state.bin"
RAM_SIZE = 0x800
NOOP = 0x00

CANDIDATES = [11, 142, 9, 176, 340]

# --- true root value (pure idle, no driving at all) ---
pool = Pool(rom_path=ROM, num_workers=1, frame_skip=4)
pool.set_headless(True)
pool.set_skip_preprocess(True)
pool.reset_all()
pool.load_worker_state(0, STATE.read_bytes())

def step(mask):
    r = pool.step_all(np.array([mask], dtype=np.uint8))
    return np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]

log = np.empty((150, RAM_SIZE), dtype=np.uint8)
for t in range(150):
    log[t] = step(NOOP)

print("=== TRUE ROOT (pure idle, 150 NOOP steps, no driving at all) ===")
for addr in CANDIDATES:
    vals = log[:, addr]
    print(f"addr={addr} (0x{addr:04X}): min={vals.min()} max={vals.max()} "
          f"first5={vals[:5].tolist()} last5={vals[-5:].tolist()} "
          f"nchanges={(np.diff(vals.astype(int)) != 0).sum()}")

# --- behavior across the death-drive probes (real play) ---
print()
print("=== ACROSS 5 DEATH-DRIVE ROLLOUTS (700 steps each, real play) ===")
disc = Discoverer(ROM, str(STATE), frame_skip=4, forward="right", seed=1)
drives = disc.death_drives()
for addr in CANDIDATES:
    print(f"--- addr={addr} (0x{addr:04X}) ---")
    for d in drives:
        col = d["log"][:, addr]
        diffs = np.diff(col.astype(int))
        nchanges = int((diffs != 0).sum())
        print(f"  rep={d['rep']} start={col[0]} min={col.min()} max={col.max()} "
              f"nchanges={nchanges} first10={col[:10].tolist()}")
disc.close()
