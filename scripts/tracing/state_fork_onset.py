"""State-fork onset finder for ours-vs-Mesen RAM tapes: the layered
benign-mask method.

The raw per-byte roll-tolerant diff drowns in ~113 bytes/frame of
sample-point skew (our step-boundary sample vs Mesen's endFrame) plus
power-on fossils (bytes the game never rewrote, still holding the two
emulators' different RAM init patterns). Both classes are benign. The
fix: every byte that EVER differs (under +-1-frame roll) before
--cert-end is excluded — --cert-end must be a frame up to which
instruction traces have CERTIFIED cycle-lockstep, so everything before
it is benign by construction. Onset = first frame >= cert-end where
>= min-bytes of the remaining clean bytes differ persistently
(>= persist of the next window frames).

CV finding (2026-08-07): timing lockstep (NMI-entry CYC +-3) holds for
the ENTIRE 14,401-frame tape, but state forks at frame 3435 (object
block $390-$39D + $2E/$40/$FD), phi-2 flag on or off. Mechanism:
OAM-DMA 513/514 start-parity anti-phase from F:3423 (see
dma_parity_probe.py). Timing alignment is necessary, NOT sufficient.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--cert-end", type=int, required=True,
                    help="frame up to which traces certify lockstep; "
                         "all byte-diffs before it are treated as benign")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--persist", type=int, default=20)
    ap.add_argument("--min-bytes", type=int, default=3)
    args = ap.parse_args()

    a, b = load(args.ours), load(args.mesen)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = (a != b)
    d &= (a != np.roll(b, 1, axis=0))
    d &= (a != np.roll(b, -1, axis=0))
    d[0] = d[-1] = False

    benign = d[:args.cert_end].any(axis=0)
    dc = d[:, ~benign]
    addr_map = np.flatnonzero(~benign)
    print(f"[mask] {int(benign.sum())} bytes benign before f{args.cert_end} "
          f"(skew + fossils), {len(addr_map)} clean bytes kept")

    w, p = args.window, args.persist
    cs = np.cumsum(dc, axis=0, dtype=np.int32)
    wins = cs[w:] - cs[:-w]
    persistent = wins >= p
    counts = persistent.sum(axis=1)
    idx = np.arange(len(counts))
    hits = np.flatnonzero((counts >= args.min_bytes) & (idx >= args.cert_end))

    anyf = np.flatnonzero(dc.sum(axis=1) * (np.arange(n) >= args.cert_end))
    if len(anyf):
        f0 = int(anyf[0])
        addrs = [hex(int(addr_map[x])) for x in np.flatnonzero(dc[f0])[:8]]
        print(f"[first] first post-cert clean-byte diff: frame {f0} {addrs}")
    if len(hits):
        f = int(hits[0]) + 1
        addrs = [hex(int(addr_map[x]))
                 for x in np.flatnonzero(persistent[hits[0]])[:12]]
        print(f"[onset] STATE-FORK onset: frame {f} "
              f"({int(counts[hits[0]])} persistent addrs: {addrs})")
    else:
        print(f"[onset] NO persistent post-cert divergence in {n} frames "
              f"— state lockstep to tape end")
    return 0


if __name__ == "__main__":
    main()
