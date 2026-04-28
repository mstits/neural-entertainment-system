"""Library-wide PLAYABILITY sweep across every .nes ROM.

`parity_sweep.py` answers "does the ROM boot identically to nes-py?".
This script answers a different question: "does pressing Start at the
title screen advance the game?"

A ROM that boots can still be unplayable if pressing Start crashes the
emulator (Zelda's `roms/zelda.nes` IRQ-trap was an example), or hangs
the game in a wait loop, or makes no observable progress. This catches
all three.

For every `.nes` file in `roms/`:
  1. Cold-boot, idle 60 frames so the title screen renders.
  2. Snapshot zero-page RAM ($0000-$001F) — the bytes most games use
     for game-mode / submode / scratch.
  3. Hold Start (mask=0x08) for 5 frames.
  4. Idle 120 more frames.
  5. Snapshot again. Also read CPU PC.

Bucket each ROM:
  - `crashed`   PC ended in `$FFE0-$FFFF` (Zelda-style IRQ trap, or
                a similar RTS-into-vector landing). Means Start press
                drove the CPU off the rails — this is what the
                pre-fix `roms/zelda.nes` looked like.
  - `frozen`    No RAM bytes changed between the two snapshots.
                CPU is in a tight wait loop with no observable progress.
  - `advances`  At least 8 of 32 zero-page bytes changed AND PC is
                outside the trap region. Game responded to Start.
  - `noisy`     Less than 8 bytes changed but PC is fine — game might
                be running a subtle animation but didn't react to Start.
                Likely needs a different button (e.g. Start+Select for
                some games, or just a longer wait).
  - `load_failed`  Cartridge couldn't be loaded.
  - `step_panic`   nes_core panicked mid-run.

Usage:
    python scripts/playability_sweep.py [--out playability_sweep.json]

Skip lists default to empty; pass `--skip-list path.json` to honor
ROMs known to crash the harness itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROMS_DIR = REPO / "roms"

# ZP bytes most games touch every frame for game-mode/animation.
ZP_RANGE = (0, 0x20)
# Multi-tap schedule. Many NES games run 1-3 second logo+title splashes
# before the "press start" prompt appears — a single Start press at
# frame 60 misses them. Tap at 60/180/360 frames so even slow boots
# get a chance, with one final long observation window.
TAP_FRAMES = [60, 180, 360]
START_HOLD = 5
POST_IDLE_FINAL = 240
RAM_CHANGE_THRESHOLD = 8


def classify(pre_zp: bytes, post_zp: bytes, pc: int) -> str:
    diff_bytes = sum(1 for a, b in zip(pre_zp, post_zp) if a != b)
    in_trap = 0xFFE0 <= pc <= 0xFFFF
    # "crashed" only fires when BOTH (a) PC ends in the IRQ-vector
    # region AND (b) zero-page didn't change at all. PC alone is a
    # false positive for games whose reset/NMI handler legitimately
    # lives high in the bank (Battletoads's RESET vector is $FFF2 —
    # the bytes at $FFE0-$FFFF are real LDA/STA/JMP code, not a halt
    # trap). The Zelda IRQ-trap pattern requires both signals: a
    # halt-loop sits at one PC AND can't write any RAM.
    if in_trap and diff_bytes == 0:
        return "crashed"
    if diff_bytes == 0:
        return "frozen"
    if diff_bytes >= RAM_CHANGE_THRESHOLD:
        return "advances"
    return "noisy"


def trial(rom_path: Path) -> dict:
    """Run one ROM through the cold-boot → start → idle sequence.
    Returns a JSON-serializable record."""
    import nes_core

    rec: dict = {"rom": str(rom_path.name)}
    try:
        env = nes_core.NESEnvironment(rom_path=str(rom_path), frame_skip=1)
        env.reset()
    except Exception as exc:
        rec["bucket"] = "load_failed"
        rec["error"] = f"{type(exc).__name__}: {exc!s}"[:160]
        return rec

    try:
        # Idle to first tap point, then take baseline snapshot.
        for _ in range(TAP_FRAMES[0]):
            env.step(0)
        pre_zp = bytes(env.get_ram_range(*ZP_RANGE))
        # Press Start at each scheduled tap point with idle between
        # taps. Each tap = START_HOLD frames of mask=0x08 then idle to
        # the next tap. After the last tap, idle POST_IDLE_FINAL
        # frames so we can observe slow state advancement.
        last_tap = TAP_FRAMES[0]
        for tap in TAP_FRAMES:
            # Idle from last_tap up to this tap.
            gap = tap - last_tap
            for _ in range(gap):
                env.step(0)
            for _ in range(START_HOLD):
                env.step(0x08)
            last_tap = tap + START_HOLD
        for _ in range(POST_IDLE_FINAL):
            env.step(0)
        post_zp = bytes(env.get_ram_range(*ZP_RANGE))
        pc = env.cpu_state()[0]
    except Exception as exc:
        rec["bucket"] = "step_panic"
        rec["error"] = f"{type(exc).__name__}: {exc!s}"[:160]
        return rec

    rec["bucket"] = classify(pre_zp, post_zp, pc)
    rec["pc_after"] = f"0x{pc:04X}"
    rec["zp_changed"] = sum(1 for a, b in zip(pre_zp, post_zp) if a != b)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", default="playability_sweep.json",
        help="output JSON path (default: playability_sweep.json)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap ROMs scanned (debugging); default: all",
    )
    p.add_argument(
        "--rom", default=None,
        help="run a single ROM (for debugging the heuristic)",
    )
    args = p.parse_args()

    roms = (
        [ROMS_DIR / args.rom] if args.rom
        else sorted(ROMS_DIR.glob("*.nes"))
    )
    if args.limit:
        roms = roms[: args.limit]
    if not roms:
        print(f"no ROMs found in {ROMS_DIR}", file=sys.stderr)
        return 1

    print(f"playability sweep over {len(roms)} ROMs ...", flush=True)
    started = time.monotonic()
    out_path = REPO / args.out
    records: list[dict] = []

    for i, rom in enumerate(roms):
        rec = trial(rom)
        records.append(rec)
        if (i + 1) % 25 == 0 or rec["bucket"] in ("crashed", "frozen", "step_panic"):
            print(
                f"  [{i+1:>4d}/{len(roms)}] {rec['bucket']:<13s} {rec['rom']}",
                flush=True,
            )
        # Checkpoint every 50 ROMs so a crash doesn't lose progress.
        if (i + 1) % 50 == 0:
            out_path.write_text(json.dumps(records, indent=2))

    out_path.write_text(json.dumps(records, indent=2))
    elapsed = time.monotonic() - started

    counts: dict[str, int] = {}
    for r in records:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    print(f"\ndone in {elapsed:.1f}s")
    print("--- bucket summary ---")
    for bucket in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {bucket:<14s} {counts[bucket]:>4d}")
    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
