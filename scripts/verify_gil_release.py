#!/usr/bin/env python3
"""Post-rebuild check that nes_core releases the GIL during env.step().

Run (AFTER the next attributed maturin rebuild installs the new .so):
    .venv/bin/python scripts/verify_gil_release.py [path/to/rom.nes]

A probe thread times its own loop latency while a worker thread hammers
env.step(). If step() releases the GIL during emulation, the probe keeps
getting scheduled and its max stall stays sub-millisecond. If step() holds
the GIL for the whole emulation batch (the pre-fix behaviour), the probe
starves and max stall ~= one step's emulation time (many ms at frame_skip=8).
"""
import sys, glob, time, threading
import nes_core

rom = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("roms/*.nes"))[0]
env = nes_core.NESEnvironment(rom, frame_skip=8)
env.reset()
stop = threading.Event()

def worker():
    for _ in range(3000):
        env.step(0)
    stop.set()

t = threading.Thread(target=worker); t.start()
max_stall, iters, last = 0.0, 0, time.perf_counter()
while not stop.is_set():
    now = time.perf_counter()
    max_stall = max(max_stall, now - last)
    last, iters = now, iters + 1
t.join()
print(f"probe iters={iters}  max_stall={max_stall*1e3:.2f} ms  "
      f"(GIL released if stall << a step's emu time and iters is large)")
