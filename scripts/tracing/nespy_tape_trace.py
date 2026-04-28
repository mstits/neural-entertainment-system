"""Run the 3354-frame cave tape through instrumented nes-py.
Trace goes to stderr; we redirect externally."""
import sys, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nes_py

ROM = str(REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes")
TAPE = str(REPO / "roms" / "zelda_start_419.state.bin")

env = nes_py.NESEnv(ROM)
env.reset()
env.step(0)  # phase compensation
tape = list(open(TAPE, "rb").read())
for b in tape[:1020]:  # through f1019, covers our target f1011
    env.step(int(b) & 0xFF)
print("Done.")
