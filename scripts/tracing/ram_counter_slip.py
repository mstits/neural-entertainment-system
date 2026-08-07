"""Jitter-proof frame-slip detector: track the per-frame offset between a
game's own free-running frame counter (CV: $001A) in our RAM tape vs
Mesen's.

In lockstep the offset is CONSTANT (sample-point skew gives a constant,
never a drifting, offset). A behavioral fork that costs or gains a frame
shifts the offset and it STAYS shifted; +-cycle jitter can never move it.

INTERPRETATION RULES (learned 2026-08-07, the false-regression session):
  - offset chaos / |offset| >= 2 => REAL machine fork, full stop.
  - a constant +-1 span is AMBIGUOUS: it can be a real one-frame slip OR
    a sample-attribution flip (the counter increment crossing to the
    other side of one dump's per-frame sample point while the machines
    stay cycle-locked). Certify with NMI-entry CYC alignment from
    instruction traces before believing it: aligned entries (+-3 CYC)
    at the span boundary => attribution artifact, not a fork. Both CV
    -1 spans (642, 3423) certified as attribution this way.

Find the counter address with the companion discovery rule: the byte
where value[f+1] == value[f] + 1 (mod 256) for ~99% of frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FRAME = 2048


def load_col(path: str, addr: int) -> np.ndarray:
    raw = Path(path).read_bytes()
    n = len(raw) // FRAME
    return np.frombuffer(raw[:n * FRAME], dtype=np.uint8).reshape(n, FRAME)[:, addr]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ours")
    ap.add_argument("mesen")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x1A,
                    help="counter address (default CV's $001A)")
    ap.add_argument("--min-run", type=int, default=8,
                    help="ignore offset blips shorter than this "
                         "(counter pauses on lag/load frames)")
    ap.add_argument("--discover", action="store_true",
                    help="instead of diffing, list frame-counter "
                         "candidates in OURS and exit")
    args = ap.parse_args()

    if args.discover:
        raw = Path(args.ours).read_bytes()
        n = len(raw) // FRAME
        a = np.frombuffer(raw[:n * FRAME], dtype=np.uint8).reshape(n, FRAME)
        inc = ((a[1:] - a[:-1]) & 0xFF) == 1
        rate = inc.mean(axis=0)
        for addr in np.argsort(rate)[::-1][:8]:
            if rate[addr] < 0.5:
                break
            print(f"${addr:04X}: increments {rate[addr] * 100:.2f}% of frames")
        return 0

    a = load_col(args.ours, args.addr)
    b = load_col(args.mesen, args.addr)
    n = min(len(a), len(b))
    off = (a[:n].astype(int) - b[:n].astype(int)) & 0xFF

    segs = []
    s = 0
    for f in range(1, n + 1):
        if f == n or off[f] != off[s]:
            segs.append((s, f, int(off[s])))
            s = f
    stable = []
    for seg in segs:
        if seg[1] - seg[0] < args.min_run:
            continue
        if stable and stable[-1][2] == seg[2]:      # same offset, blip-split
            stable[-1] = (stable[-1][0], seg[1], seg[2])
        else:
            stable.append(seg)
    print(f"[counter ${args.addr:04X}] {n} frames, {len(segs)} raw segments, "
          f"{len(stable)} stable (>= {args.min_run}, same-offset merged):")
    for s, e, o in stable:
        signed = o - 256 if o > 128 else o
        print(f"  frames {s:5d}-{e - 1:5d} ({e - s:5d} long): offset {signed:+d}")
    if len(stable) <= 1:
        print("[verdict] no counter slip — frame-level LOCKSTEP end to end")
    else:
        first = stable[1]
        mag = min(first[2], 256 - first[2])
        kind = ("AMBIGUOUS (+-1: certify with trace CYC alignment "
                "before calling it a fork)" if mag <= 1 else "REAL fork")
        print(f"[verdict] first persistent offset change at frame {first[0]} "
              f"— {kind}")
    return 0


if __name__ == "__main__":
    main()
