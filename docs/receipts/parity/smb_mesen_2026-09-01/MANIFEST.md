# Receipt manifest: Super Mario Bros. cross-checked against Mesen 2, 2026-09-01

Backs the Super Mario Bros. row in `docs/proposals/parity_coverage_map.md`
and the Mesen sentence in `README.md`. The run itself was made read-only
against this repo; its raw dumps are 24 MB each and live outside the tree,
so this file hash-pins them instead of committing them.

## Reproducing it takes two runs, not one

The two sides carry different phase corrections, and
`nes_core_ram_dump.py` refuses `--mesen-align` together with
`--emit-phase` for exactly that reason. Run it twice.

Our side, aligned, which is the dump the numbers below were measured on:

    python scripts/tracing/nes_core_ram_dump.py \
        --rom "roms/Super Mario Bros. (World).nes" \
        --tape smb_banked_segment_3000x4.tape.bin \
        --out /tmp/ours_smb_aligned.bin --mesen-align 2

The reference side, the bytes to feed Mesen:

    python scripts/tracing/nes_core_ram_dump.py \
        --rom "roms/Super Mario Bros. (World).nes" \
        --tape smb_banked_segment_3000x4.tape.bin \
        --out /tmp/ours_smb_unaligned.bin \
        --emit-tape /tmp/mesen_smb_tape.bin --emit-phase 2

Phase 2 is the input phase this run measured as the only one that holds:
the `env.reset()` frame advance (`nes_core/src/python.rs:340`) plus the
one-frame `inputPolled` phase of `scripts/tracing/mesen_cv_tape_dump.lua`.
Phase 0 and phase 1 both kill Mario in 1-2 on both emulators.

Two frames of that phase land on our side rather than the reference's,
which is what `--mesen-align 2` applies. A single run at the default
`--mesen-align 0` leaves our side unshifted, and the comparison then
sits two frames out: measured median 419 differing bytes per frame under
the same one-frame tolerance, against the 17 below. That is the number
this manifest's first draft would have produced, and the reason the
recipe is two commands.

### Two conventions that do not line up, stated rather than assumed

The 2026-09-01 run built its ours side with `ours_smb_tape_dump.py`
(pinned below), which steps EVERY tape byte and emits one row per byte:
12,000 bytes, 12,000 rows. `nes_core_ram_dump.py` uses the
`nes_core_cv_ram_dump.py` convention instead, where tape byte 0 is the
root frame, dumped as row 0 and never stepped. So the two produce the
same row count from the same tape but not the same file, and the shift
that reproduces the alignment differs between them: the run applied
`[0] + segment[:-1]` to its own tape by hand, this script applies
`--mesen-align 2`.

The reference tapes differ in length as well. The run built its Mesen
tape in the length-preserving form `[0,0] + segment[:-2]`, 12,000 bytes;
`--emit-phase` extends instead, 12,002 bytes, so it carries two extra
trailing frames.

## What was measured

| Quantity | Value |
|---|---|
| Frames compared | 12,000 (9.6% of the banked flagship tape) |
| Lives, area, mode | agree on every compared frame |
| Level-transition frames | agree (`$0760` changes at 3452 / 3951 / 8306 / 10911 on both; `$0770` to 1 at 246 on both) |
| RAM bytes differing per frame, one-frame tolerance | median 17, p90 27, max 37; 596 frames zero; final frame zero |
| RAM bytes differing per frame, exact frame mapping | median 62, p90 83, max 219 |
| Where the differing bytes are | stack below the pointer, zero-page temps, OAM |

Both rows of numbers are the ALIGNED ours side
(`ours_smb_banked_segment_ram_aligned.bin`) against
`mesen_smb_banked_segment_shift2_ram.bin`. The unaligned dump
`ours_smb_banked_segment_ram.bin` is pinned as an input to that
alignment, not as the file the numbers came from.

## Hash-pinned inputs and outputs

`sha256`, files under
`personal_os/reports/macos-emulation-and-training/2026-09-01-ground-truth-execution/receipts/`:

    8d4b3d63dda80d0e84da9bfb8486ff976ec55bee64cdcd69eb755e51c22a9647  smb_banked_segment_3000x4.tape.bin (base tape, 12,000 bytes)
    febe46d03902dc93d3100a686b7329244b9315e6e74c497c6c5d754975bd47ea  shift-2 tape fed to Mesen, [0,0] + segment[:-2] (12,000 bytes)
    994a9260366d5712a89e2b2f2be953a2c4269bf6294953dc67c38f43c9e9d4f1  mesen_smb_banked_segment_shift2_ram.bin
    f3ffc76236911c015a67bb435c5cb3b4d92db0bdc50b3a7a1cae28d3cef40ffc  ours_smb_banked_segment_ram.bin (UNALIGNED)
    d3aa3c5ac30514c8e6ff0940e6337cd9d2d6f7cbe4424525ce628a62e96e6d56  ours_smb_banked_segment_ram_aligned.bin (the numbers above)
    1179426ab69fdeb31a07ea61eaa7f50fbc828f7431eb983c0b8676ed0712ea88  mesen_smb_banked_segment_shift2.log
    2d1c4d222fa93c50a019f4a89cd5b2084fd87992a833cd8617d3f95516476590  mesen_smb_banked_segment_shift2_transitions.log
    fb0b9115f43c517eb2edefeff1d2e7d2647612d5a6b961a993680d9b40808a5d  mesen_smb_banked_segment_analysis.log
    f04bb94e66a1e6592ed3dc2d60bdac325aa782e88bc72c537786b9e803ae3724  ours_smb_tape_dump.py

The shift-2 tape exists as a hash, not a file: it was fed to Mesen and
not kept. Rebuild it as `[0,0] + segment[:-2]` from the base tape above
and it hashes to `febe46d0...47ea`.

## Scope

CPU RAM only, no framebuffer comparison. The nes_core side of that run used a
scratch mirror of `nes_core_smb_walk_dump.py`, pinned above as
`ours_smb_tape_dump.py`, because no repo script then took a ROM and a tape;
`scripts/tracing/nes_core_ram_dump.py` is that script, and the byte-identity
gate in `tests/test_nes_core_ram_dump.py` ties it to the Castlevania harness
the scratch mirror was copied from. Mesen's exit code is blank in the run log:
completion is evidenced by the Lua status file and the dump hash above.
