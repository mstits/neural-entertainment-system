"""Per-frame RAM dump of an input tape through our core, for ANY ROM:
the "ours" side of the Mesen lockstep comparison (2048 bytes/frame,
the same layout `mesen_cv_tape_dump.lua` and `mesen_dump_generic.lua`
emit). Generalizes `nes_core_cv_ram_dump.py`, whose ROM is hard-coded
to Castlevania; with default flags the two produce byte-identical
output on the same tape.

TAPE FORMAT. One input-bitmask byte per REAL frame (the Mesen Lua side
polls once per frame, so that is the only format it can consume). If a
tape came out of a solve lineage running `frame_skip=N`, it holds one
byte per ACTION instead; pass `--frame-skip N --one-byte-per-action` so
this script steps it the way the lineage did, or expand it to one byte
per real frame first and leave the defaults alone. The two are byte-
exact for the runs measured so far (Bubble Bobble rounds 1-3, 1,287
rows, 2026-09-01 ruling receipt `nescore_fs1_corrected.log`), but they
are different tape files and mixing them silently drops or duplicates
frames, so the flag is required rather than guessed.

ROW 0 IS THE ROOT FRAME. Tape byte 0 is the power-on (or post-load)
frame itself: its RAM is dumped as row 0 and the byte is never stepped.
Every offset this script takes is an offset into the tape FILE, so the
first position that is "after the root frame" is 1, not 0. Offsets that
would land in front of the root frame are refused rather than silently
consuming it.

THE TWO TAPE-CONSTRUCTION RULES. A tape assembled from a banked solve
has to carry both of these before any reference emulator sees it. They
are conventions the repo already implements; skipping either one moves
every later input relative to the game and reads as an emulator
divergence that is not there (2026-09-01: it cost a wrong NOT HELD
verdict on Bubble Bobble).

  1. Input phase. `env.reset()` advances one frame before the first
     input lands (`nes_core/src/python.rs:340`), and the Mesen Lua
     applies a byte on the frame after `inputPolled` fires. So a tape
     whose lineage is `env.reset()` is fed to Mesen two frames late
     (`--emit-phase 2`), and one whose lineage is `reset_no_advance()`
     one frame late (`--emit-phase 1`, equivalently the `--mesen-align`
     shift applied to our own dump instead). `--emit-phase` has no
     default: `--emit-tape` without it is refused, because a guessed
     phase is the whole failure class this script exists to close.
  2. Materialize block. `TapePlayer.play()` runs one no-op step between
     the root state and the first action ("the no-op materializes RAM",
     `src/training/tape_replay.py:249`), and so does the solver. That
     is `frame_skip` real frames, 4 on the default profile. A tape
     spliced straight from a cold-boot prefix into action 0 is missing
     them: `--materialize 4 --materialize-at <prefix length>` puts them
     back. A `--load-state` root wants `--materialize-at 1`, the
     default: straight after the root frame, which is tape byte 0.

`--emit-tape` writes the exact bytes the reference side should be fed,
so the phase and the materialize block are applied once, here, instead
of being rebuilt by hand per run.

BEFORE FEEDING ANY REFERENCE EMULATOR, prove the tape reproduces its
own lineage. Dump it twice through this script, once as the lineage ran
it and once at `frame_skip=1`, and diff the two with
`diff_ram_tapes.py`; a tape that does not reproduce its own lineage
byte-for-byte cannot say anything about a second emulator. It takes
about ten seconds and it is the check whose absence produced the
withdrawn Bubble Bobble result.

TRAJECTORY SCOPE. A tape is only meaningful under the hw-flag set it
was solved on. `nes_core_cv_ram_dump.py`'s five-flag set is NOT the CV
tape's build lineage (a 4-flag Pool), which is why that harness's
lockstep receipts contain zero block-3 frames; see its docstring. The
default here is no flags, matching most solve profiles' `solver_args`;
pass `--hw-flags` to match a profile that pins some.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

NOOP = 0


def available_hw_flags() -> tuple:
    """Flag names the INSTALLED core exposes, derived from
    `NESEnvironment`'s `set_hw_*` methods so the list cannot go stale
    against the wheel. Same derivation as
    `scripts/go_explore_solve.py:118`, read off the environment rather
    than the Pool because that is what this script steps."""
    return tuple(sorted(n[len("set_hw_"):]
                        for n in dir(nes_core.NESEnvironment)
                        if n.startswith("set_hw_")))


def resolve_hw_flags(cli: str) -> list:
    """`--hw-flags` as a comma list; empty or 'none' is the empty set.
    Tolerates the `set_hw_` method spelling and de-duplicates with
    order preserved, matching `go_explore_solve.resolve_hw_flags`."""
    raw = (cli or "").strip()
    names = ([] if raw.lower() in ("", "none")
             else [p.strip() for p in raw.split(",") if p.strip()])
    out, seen = [], set()
    for n in names:
        if n.startswith("set_hw_"):
            n = n[len("set_hw_"):]
        if n in seen:
            continue
        if n not in available_hw_flags():
            raise SystemExit(
                f"[nes_core_ram_dump] unknown hw flag {n!r}. The "
                f"installed nes_core exposes: "
                f"{', '.join(available_hw_flags())}.")
        seen.add(n)
        out.append(n)
    return out


def materialize_tape(tape: bytes, count: int, at: int) -> bytes:
    """Insert `count` no-op bytes after the first `at` tape bytes: rule
    2 above, `src/training/tape_replay.py:249`. `at` is an offset into
    the tape FILE, and tape byte 0 is the root frame this script never
    steps, so `at=1` is immediately after the root frame (what a
    `--load-state` root wants) and a cold-boot tape wants its prefix
    length. `at=0` inserts in FRONT of the root frame, which shifts the
    root byte into the stepped feed; `main()` refuses it."""
    if count < 0:
        raise ValueError("materialize count must be >= 0")
    if count == 0:
        return bytes(tape)
    if not 0 <= at <= len(tape):
        raise ValueError(f"materialize-at {at} outside tape of "
                         f"{len(tape)} bytes")
    return bytes(tape[:at]) + bytes([NOOP]) * count + bytes(tape[at:])


def align_feed(feed: bytes, frames: int) -> bytes:
    """Apply every byte `frames` frames later, keeping the row count so
    our dump still lines up with the reference's row for row. This is
    `nes_core_cv_ram_dump.py`'s `--mesen-align`, generalized past one
    frame."""
    if frames < 0:
        raise ValueError("align frames must be >= 0")
    if frames == 0:
        return bytes(feed)
    if frames >= len(feed):
        return bytes([NOOP]) * len(feed)
    return bytes([NOOP]) * frames + bytes(feed[:-frames])


def emit_tape(tape: bytes, phase: int) -> bytes:
    """The bytes a reference emulator should be fed: `phase` leading
    no-ops in front of the (already materialized) tape. Rule 1 above.
    Unlike `align_feed` this EXTENDS the tape, because the reference
    runs from its own power-on and has no rows to keep aligned yet."""
    if phase < 0:
        raise ValueError("emit phase must be >= 0")
    return bytes([NOOP]) * phase + bytes(tape)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rom", required=True, help="path to the .nes ROM")
    ap.add_argument("--tape", required=True,
                    help="raw byte tape: one byte per real frame, or one "
                         "byte per action with --one-byte-per-action")
    ap.add_argument("--out", default="/tmp/ours_ram.bin",
                    help="per-frame RAM dump, 2048 bytes per row")
    ap.add_argument("--load-state", default=None,
                    help="nes_core save_state() blob to load as the root "
                         "instead of cold boot + reset_no_advance()")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="native frame_skip for the environment (default "
                         "1: one tape byte is one real frame, what the "
                         "Mesen Lua emits). Set >1 only with "
                         "--one-byte-per-action.")
    ap.add_argument("--one-byte-per-action", action="store_true",
                    help="the tape holds one byte per ACTION (the core "
                         "repeats it frame_skip times), not one per frame")
    ap.add_argument("--hw-flags", default="",
                    help="comma list of hw flag names, or 'none' (default: "
                         "none, matching most solve profiles)")
    ap.add_argument("--mesen-align", type=int, nargs="?", const=1, default=0,
                    metavar="FRAMES",
                    help="apply our tape bytes FRAMES frames later "
                         "(bare flag: 1, the reset_no_advance lineage; "
                         "2 for an env.reset() lineage). Row count is "
                         "preserved. Use this OR --emit-phase, not both.")
    ap.add_argument("--materialize", type=int, default=0, metavar="FRAMES",
                    help="insert FRAMES no-op frames at --materialize-at "
                         "(the tape_replay.py:249 no-op step: frame_skip "
                         "real frames, 4 on the default profile)")
    ap.add_argument("--materialize-at", type=int, default=1, metavar="OFFSET",
                    help="tape-file offset the no-op block goes after "
                         "(default 1: straight after the root frame, which "
                         "is tape byte 0; a cold-boot tape wants its prefix "
                         "length). 0 is refused: it lands in front of the "
                         "root frame.")
    ap.add_argument("--emit-tape", default=None, metavar="PATH",
                    help="also write the materialized tape, phase-shifted "
                         "by --emit-phase, for the reference emulator")
    ap.add_argument("--emit-phase", type=int, default=None, metavar="FRAMES",
                    help="leading no-op frames on the emitted tape, required "
                         "with --emit-tape and never guessed (2 for an "
                         "env.reset() lineage, 1 for reset_no_advance; "
                         "see rule 1)")
    return ap


def validate_args(args: argparse.Namespace) -> None:
    """Every refusal this script makes, in one place so each one can be
    exercised without touching the filesystem or the emulator. Each
    guard is a phase or framing mistake that produces a plausible dump
    and a wrong verdict rather than an error."""
    if args.frame_skip > 1 and not args.one_byte_per_action:
        raise SystemExit(
            "[nes_core_ram_dump] --frame-skip >1 requires "
            "--one-byte-per-action: at frame_skip N a lineage tape holds "
            "one byte per action, a Mesen tape one byte per real frame, "
            "and guessing between them drops or duplicates frames.")
    if args.mesen_align and args.emit_phase:
        raise SystemExit(
            "[nes_core_ram_dump] --mesen-align and --emit-phase both "
            "shift the same phase, one on our dump and one on the "
            "reference's tape. Pick one.")
    if args.emit_tape is not None and args.emit_phase is None:
        raise SystemExit(
            "[nes_core_ram_dump] --emit-tape requires an explicit "
            "--emit-phase (2 for an env.reset() lineage, 1 for "
            "reset_no_advance, 0 to state that the tape is already "
            "phased). A guessed phase is the failure this script exists "
            "to close.")
    if args.materialize and args.materialize_at == 0:
        raise SystemExit(
            "[nes_core_ram_dump] --materialize-at 0 puts the no-op block "
            "in FRONT of tape byte 0, which is the root frame this "
            "script dumps as row 0 and never steps: one no-op is lost "
            "and the root byte becomes an input. Straight after the "
            "root frame is 1; a cold-boot tape wants its prefix length.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    hw_names = resolve_hw_flags(args.hw_flags)
    tape = materialize_tape(Path(args.tape).read_bytes(),
                            args.materialize, args.materialize_at)

    if args.emit_tape is not None:
        Path(args.emit_tape).write_bytes(emit_tape(tape, args.emit_phase))

    env = nes_core.NESEnvironment(rom_path=args.rom,
                                  frame_skip=args.frame_skip)
    for name in hw_names:  # BEFORE reset; see src/training/tape_replay.py
        getattr(env, f"set_hw_{name}")(True)

    if args.load_state is not None:
        env.reset()
        env.load_state(Path(args.load_state).read_bytes())
    else:
        env.reset_no_advance()

    # Tape byte 0 is the power-on (or post-load) frame itself: dump that
    # RAM as row 0, then step bytes 1..N.
    feed = align_feed(tape[1:], args.mesen_align)
    with open(args.out, "wb") as f:
        f.write(bytes(env.get_ram_range(0, 2048)))
        for byte in feed:
            env.step(int(byte))
            f.write(bytes(env.get_ram_range(0, 2048)))

    unit = "actions" if args.one_byte_per_action else "frames"
    print(f"[done] {len(tape)} {unit} -> {args.out} "
          f"(rom={Path(args.rom).name}, frame_skip={args.frame_skip}, "
          f"hw_flags={hw_names or 'none'}, "
          f"materialize={args.materialize}@{args.materialize_at}, "
          f"mesen_align={args.mesen_align})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
