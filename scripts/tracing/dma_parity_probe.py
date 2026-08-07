"""Extract per-frame OAM-DMA ($4014) stall events — start CYC, stall
length, start parity — from a CPU trace, ours or Mesen's.

The two trace conventions differ: Mesen logs the STA $4014 once and the
next instruction's CYC jumps ~515/516; our tracer's step_one_instruction
returns once per STALL CYCLE, so the same event appears as a ~513-line
run of the SAME PC with CYC += 1 each line. This tool detects both and
normalizes to (startCYC, stall_cycles, parity).

Why parity matters (the v14 candidate-#2 mechanism, CONFIRMED on CV
2026-08-07): the DMA halt is 513/514 cycles depending on the even/odd
cycle of the $4014 write. A +-1-cycle NMI-service offset vs Mesen flips
that parity, toggling DMA duration, permanently shifting the next
frame's alignment — the feedback loop that turned a 1-cycle residual
into the frame-3435 CV state fork (parities run anti-phase from F:3423).
The phi-2 flag removes the jitter VARIANCE but leaves a constant
+1-cycle service offset, which parity (mod 2) still sees — closing that
constant is the follow-up lever.
"""
from __future__ import annotations

import argparse
import re

LINE = re.compile(r"([0-9A-F]{4}) .*CYC:(\d+)")


def stalls(path: str, lo: int, hi: int):
    events = []
    prev_cyc = prev_pc = None
    run_start, run_len = None, 0
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        pc, cyc = m.group(1), int(m.group(2))
        if prev_cyc is not None:
            delta = cyc - prev_cyc
            if lo <= delta <= hi:                 # Mesen: one big jump
                events.append((prev_cyc, delta))
            if delta == 1 and pc == prev_pc:      # ours: delta-1 run
                if run_len == 0:
                    run_start = prev_cyc
                run_len += 1
            else:
                if run_len >= 400:
                    events.append((run_start, run_len))
                run_len = 0
        prev_cyc, prev_pc = cyc, pc
    if run_len >= 400:
        events.append((run_start, run_len))
    return sorted(events)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--lo", type=int, default=500)
    ap.add_argument("--hi", type=int, default=560)
    args = ap.parse_args()
    for path in args.trace:
        print(f"--- {path}")
        for start, length in stalls(path, args.lo, args.hi):
            print(f"  start={start} stall={length} parity={start % 2}")
    return 0


if __name__ == "__main__":
    main()
