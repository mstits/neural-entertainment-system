"""Find the high byte that pairs with a wrapping progress byte.

`discover_observables` detects that a progress candidate wraps and prints
`wrapPair=0` when it cannot pair it — then recommends the low byte
anyway. The solver reads that byte RAW, so everything past the first wrap
is invisible to it. Three profiles shipped that way: Rygar (21 distinct
values in 1200 steps), Kung Fu (91, unpaired), Ninja Gaiden (2). Rygar's
cost 73 minutes of search that produced 116 cells.

The search is mechanical and uses no game knowledge: hold forward, note
every step where the low byte drops hard, and look for an address that
increments at those moments and stays put otherwise.

Finding a candidate is not enough, so every one is VALIDATED by
reconstructing `lo + (hi << 8)` and re-running the 3-probe protocol on
the combined value. A pair is accepted only if the combination is
strongly monotone under forward, flat under NOOP, and materially richer
than the low byte alone. That last test is what stops a plausible-looking
counter — an animation frame, a score digit — from being adopted.

    .venv/bin/python scripts/find_wrap_pair.py --profile configs/rygar.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

WRAP_DROP = 100          # a fall this large is a wrap, not a wiggle
MIN_MONO = 0.98          # combined value must be near-perfectly monotone
MIN_GAIN = 4.0           # and this many times richer than lo alone


def find_wrap_steps(trace: list[int], drop: int = WRAP_DROP) -> list[int]:
    """Indices where the low byte fell hard enough to be a wrap."""
    return [i for i in range(1, len(trace)) if trace[i] < trace[i - 1] - drop]


def score_candidate(col: list[int], wrap_steps: list[int]) -> dict:
    """How well does this address behave like a high byte?

    It must step up at wraps and be quiet between them. `quiet` matters as
    much as `hits`: a byte that changes constantly will coincide with some
    wraps by luck.
    """
    hits = 0
    for w in wrap_steps:
        if 0 < w < len(col) - 1 and col[w + 1] == col[w - 1] + 1:
            hits += 1
    changes = sum(1 for a, b in zip(col, col[1:]) if a != b)
    return {"hits": hits, "of": len(wrap_steps), "changes": changes,
            "quiet": changes <= max(4, len(wrap_steps) * 3),
            "span": max(col) - min(col)}


def combine(lo: list[int], hi: list[int]) -> list[int]:
    return [l + (h << 8) for l, h in zip(lo, hi)]


def monotone_fraction(seq: list[int]) -> float:
    if len(seq) < 2:
        return 0.0
    d = [b - a for a, b in zip(seq, seq[1:])]
    return sum(1 for x in d if x >= 0) / len(d)


def validate_pair(lo_fwd: list[int], hi_fwd: list[int],
                  lo_noop: list[int], hi_noop: list[int]) -> dict:
    """Accept a pair only if the COMBINED value behaves like progress."""
    comb_f = combine(lo_fwd, hi_fwd)
    comb_n = combine(lo_noop, hi_noop)
    mono = monotone_fraction(comb_f)
    net_f = comb_f[-1] - comb_f[0]
    net_n = comb_n[-1] - comb_n[0]
    distinct_comb = len(set(comb_f))
    distinct_lo = len(set(lo_fwd))
    gain = distinct_comb / max(1, distinct_lo)
    reasons = []
    if mono < MIN_MONO:
        reasons.append(f"combined monotone fraction {mono:.3f} < {MIN_MONO}")
    if net_f <= 0:
        reasons.append(f"combined does not rise under forward (net {net_f})")
    if abs(net_n) > max(2, 0.02 * abs(net_f or 1)):
        reasons.append(f"combined moves under NOOP (net {net_n})")
    if gain < MIN_GAIN:
        reasons.append(f"only {gain:.1f}x richer than the low byte alone "
                       f"({distinct_comb} vs {distinct_lo} distinct)")
    return {"accepted": not reasons, "monotone": round(mono, 4),
            "net_forward": net_f, "net_noop": net_n,
            "distinct_combined": distinct_comb, "distinct_lo": distinct_lo,
            "gain": round(gain, 2), "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--forward", default="right")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, yaml, nes_core
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load((REPO / args.profile).read_text())
    solve = prof.get("solve") or {}
    lo_addr = (solve.get("progress") or {}).get("lo")
    rom = solve.get("rom") or prof.get("rom_path")
    if lo_addr is None:
        raise SystemExit("profile has no solve.progress.lo")
    space = prof["action_space"]
    bm = action_space_to_bitmasks(space)
    fwd_i = next((i for i, a in enumerate(space) if a == [args.forward]), 1)

    def roll(mask: int) -> list:
        pool = nes_core.Pool(rom_path=str(REPO / rom), num_workers=1,
                             frame_skip=int(prof.get("frame_skip", 4)))
        pool.set_headless(True); pool.set_skip_preprocess(True)
        pool.load_worker_state(0, (REPO / prof["start_state_path"]).read_bytes())
        a = np.array([mask], dtype=np.uint8)
        return [np.frombuffer(bytes(pool.step_all(a)[0][2]),
                              dtype=np.uint8).copy() for _ in range(args.steps)]

    fwd, noop = roll(bm[fwd_i]), roll(bm[0])
    lo_f = [int(r[lo_addr]) for r in fwd]
    lo_n = [int(r[lo_addr]) for r in noop]
    wraps = find_wrap_steps(lo_f)
    print(f"  low byte ${lo_addr:04X}: {len(set(lo_f))} distinct, "
          f"{len(wraps)} wrap events")
    if not wraps:
        print("  no wraps — the low byte does not need a pair")
        return 0

    ranked = []
    for addr in range(0x0000, 0x0800):
        if addr == lo_addr:
            continue
        col = [int(r[addr]) for r in fwd]
        if max(col) == min(col):
            continue
        sc = score_candidate(col, wraps)
        if sc["hits"] >= max(2, 0.5 * len(wraps)) and sc["quiet"]:
            ranked.append((addr, sc))
    ranked.sort(key=lambda t: (-t[1]["hits"], t[1]["changes"]))
    print(f"  {len(ranked)} structural candidate(s)")

    accepted = None
    for addr, sc in ranked[:10]:
        v = validate_pair(lo_f, [int(r[addr]) for r in fwd],
                          lo_n, [int(r[addr]) for r in noop])
        flag = "ACCEPT" if v["accepted"] else "reject"
        print(f"    ${addr:04X} hits {sc['hits']}/{sc['of']} -> {flag}"
              + ("" if v["accepted"] else f"  ({v['reasons'][0]})"))
        if v["accepted"] and accepted is None:
            accepted = (addr, sc, v)

    if accepted:
        addr, sc, v = accepted
        print(f"\n  PAIR FOUND: progress: {{lo: 0x{lo_addr:04X}, "
              f"hi: 0x{addr:04X}}}")
        print(f"    combined {v['distinct_combined']} distinct "
              f"({v['gain']}x the low byte), monotone {v['monotone']}, "
              f"net forward {v['net_forward']}, net noop {v['net_noop']}")
    else:
        print("\n  NO VALID PAIR. The low byte wraps and nothing in RAM "
              "tracks the wraps, so this game has no usable scalar progress "
              "signal — it needs a different observable, not a repair.")
    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "profile": args.profile, "lo": lo_addr, "wraps": len(wraps),
            "accepted": ({"hi": accepted[0], **accepted[2]} if accepted
                         else None)}, indent=2) + "\n")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
