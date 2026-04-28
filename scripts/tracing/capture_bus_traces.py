"""Capture matched bus-access traces from both nes_core and
instrumented nes-py over a Zelda cave-tape frame window.

Use env vars:
  NES_TRACE_BUS=1                   (nes_core)
  NES_TRACE_BUS_FMIN=1005
  NES_TRACE_BUS_FMAX=1013
  NESPY_TRACE_BUS=1                 (nes-py)
  NESPY_TRACE_BUS_FMIN=1006         (phase-comp: nes-py is 1 frame ahead)
  NESPY_TRACE_BUS_FMAX=1014

Invoke via `scripts/tracing/diff_bus_traces.sh` which runs with the
right env vars set and redirects each stderr to a file.
"""
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WHICH = os.environ.get("WHICH", "").strip()
if WHICH not in ("ours", "theirs"):
    raise SystemExit("set WHICH=ours or WHICH=theirs")

ROM = str(REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes")
TAPE = str(REPO / "roms" / "zelda_start_419.state.bin")
buttons = list(open(TAPE, "rb").read())

# How many tape frames to replay. Include enough to get past the target
# window (no point running the whole tape when we only want f1005-1015).
N = int(os.environ.get("N", "1020"))

if WHICH == "ours":
    import nes_core
    env = nes_core.NESEnvironment(rom_path=ROM, frame_skip=1)
    env.reset()
    for b in buttons[:N]:
        env.step(int(b) & 0xFF)
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import nes_py
    env = nes_py.NESEnv(ROM)
    env.reset()
    env.step(0)  # phase-comp
    for b in buttons[:N]:
        env.step(int(b) & 0xFF)

print(f"{WHICH}: done", file=sys.stderr)
