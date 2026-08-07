"""First-divergence finder for per-frame RAM tapes (ours vs Mesen).

Compares two 2048-bytes/frame dumps with a ±1-frame roll tolerance and
reports the DIVERGENCE ONSET, defined by persistence, not density: the
first frame F where >= `persist` of the following `window` frames still
differ under the best per-frame roll. Mean byte-density is deliberately
NOT used — our step-boundary vs Mesen's endFrame sampling skew makes a
~15-40 bytes/frame noise floor that masquerades as divergence (three
separate forensics sessions were burned by this; see the 2026-07-31
lockstep notes)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FRAME = 2048


def load(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    n = len(raw) // FRAME
    return np.frombuffer(raw[:n * FRAME], dtype=np.uint8).reshape(n, FRAME)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ours")
    ap.add_argument("mesen")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--persist", type=int, default=20)
    ap.add_argument("--min-bytes", type=int, default=3,
                    help="persistent byte ADDRESSES required to call onset "
                         "(a real fork cascades; 1-2 lone addresses are "
                         "usually a skew-phase counter)")
    args = ap.parse_args()

    a, b = load(args.ours), load(args.mesen)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"[diff] comparing {n} frames", flush=True)

    # Per-BYTE skew-tolerant diff: ours[f,i] must match mesen at the
    # same frame or ±1 (our step-boundary sample vs Mesen's endFrame
    # sample legitimately splits mid-update bytes across a frame edge).
    d = (a != b)
    d &= (a != np.roll(b, 1, axis=0))
    d &= (a != np.roll(b, -1, axis=0))
    d[0] = d[-1] = False               # roll wrap edges

    per_frame = d.sum(axis=1)
    print(f"[diff] skew-noise profile: median {int(np.median(per_frame))} "
          f"differing bytes/frame, p90 {int(np.percentile(per_frame, 90))}, "
          f"max {int(per_frame.max())}")

    # Persistence per byte address over a sliding window, then onset =
    # first frame where >= min_bytes addresses are persistently wrong.
    w, p = args.window, args.persist
    cs = np.cumsum(d, axis=0, dtype=np.int32)
    wins = cs[w:] - cs[:-w]            # window sums starting at f+1
    persistent = wins >= p             # (n-w, 2048)
    counts = persistent.sum(axis=1)
    hits = np.flatnonzero(counts >= args.min_bytes)
    if len(hits) == 0:
        print(f"[diff] NO persistent divergence in {n} frames — LOCKSTEP")
    else:
        f = int(hits[0]) + 1
        addrs = np.flatnonzero(persistent[hits[0]])[:12]
        print(f"[diff] divergence onset: frame {f} "
              f"({int(counts[hits[0]])} persistent addresses; first: "
              f"{[hex(int(x)) for x in addrs]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
