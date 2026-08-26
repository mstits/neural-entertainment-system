import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
sys.path.insert(0, str(REPO))

from nes_core import Pool

ROM = str(REPO / "roms/DuckTales 2 (USA).nes")
STATE = REPO / "roms/DuckTales 2 (USA)_start.state.bin"
RAM_SIZE = 0x800
NOOP = 0x00

pool = Pool(rom_path=ROM, num_workers=1, frame_skip=4)
pool.set_headless(True)
pool.set_skip_preprocess(True)
pool.reset_all()
pool.load_worker_state(0, STATE.read_bytes())

def step(mask):
    r = pool.step_all(np.array([mask], dtype=np.uint8))
    return np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]

# Peek immediately: one NOOP step right after load (minimal disturbance)
ram = step(NOOP)
print("addr 0x000B right after load + 1 NOOP step:", ram[0x000B])

# Also peek across the first 150 idle steps (the settle window) to see
# whether it's zero throughout or becomes nonzero once gameplay starts.
pool.load_worker_state(0, STATE.read_bytes())
log = np.empty((150, RAM_SIZE), dtype=np.uint8)
for t in range(150):
    log[t] = step(NOOP)

vals = log[:, 0x000B]
print("0x000B over first 150 NOOP steps: min=%d max=%d first10=%s last10=%s" % (
    vals.min(), vals.max(), vals[:10].tolist(), vals[-10:].tolist()))
