# DO-42 receipt: mapper 37 (SMB + Tetris + NWC) boot fix, ROM census

2026-09-02. Receipt for the mapper 37 boot fix (`Mapper37::region` rewritten to the
NES-ZZ board's A16/A17 gating). Same form as `mapper228_boot_fix_2026-09-02.md` and
`mapper105_boot_fix_2026-09-02.md`.

Unlike those two rounds, the before-state is measured here, not argued: the pre-fix run
is kept as `mapper37_boot_fix_2026-09-02_base_HEAD.csv`.

## Method

`scripts/rom_library_scan.py` itself, unmodified, at `--frames 300 --static-frames 300
--workers 1`, the same invocation the mapper 105 round used.

The scanner imports the `nes_core` Python extension. To run it against a patched crate
without writing to the repo, two wheels were built with `maturin build --release` from
`rsync` copies of `nes_core/` in scratch, one at HEAD and one with the patch applied,
each unpacked to its own directory and put on `PYTHONPATH` ahead of `.venv`'s
site-packages. Both runs used the repo's `.venv/bin/python`. Nothing in the repo or its
`.venv` was modified.

`--roms-dir` was a scratch directory of five symlinks into the repo's `roms/`: the ROM
this fix moves, the mapper 47 multicart that shares `Mapper4::set_outer_region` (the
only other cart in the library on that code path), two plain MMC3 carts, and Jackal
(mapper 2, the other static ROM) as a control that this change does not move it.

## Result

```
base (HEAD df33765):  5/5 ok, static-check: 3 live, 2 static
fix   run1:           5/5 ok, static-check: 4 live, 1 static
fix   run2 (fresh):   5/5 ok, static-check: 4 live, 1 static
```

| ROM | mapper | base motion | fix motion | base hashes | fix hashes | fix first change |
|---|---:|---|---|---:|---:|---:|
| Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes | 37 | **static** | **live** | 1 | 13 | frame 7 |
| Super Spike V'Ball + Nintendo World Cup (USA).nes | 47 | live | live | 152 | 152 | frame 16 |
| Super Mario Bros. 3 (USA) (Rev A).nes | 4 | live | live | 143 | 143 | frame 9 |
| Kirby's Adventure (USA) (Rev A).nes | 4 | live | live | 118 | 118 | frame 5 |
| Jackal (USA).nes | 2 | static | static | 1 | n/a | n/a |

`diff` of the base and fix CSVs on `status,motion,static_distinct_hashes,
static_first_change_frame`, with the mapper 37 row excluded, is empty: the four control
ROMs are unchanged column-for-column. `diff` of run1 and run2 on the same columns is
empty: the measurement is deterministic across fresh interpreters.

Repeated at the scanner's own default depth (`--static-frames 3000`, all other
defaults): base 3 live / 2 static, fix 4 live / 1 static; mapper 37 goes 1 hash to 837
hashes with first change at frame 7; the four controls are unchanged there too.

## Artifact and ROM hashes

| artifact | sha256 |
|---|---|
| `roms/Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes` | `814763cdcc615c1988fd414a5a799a12fe2d4b902bdb4fb12b883c7e8f0bcac3` |
| `roms/Super Spike V'Ball + Nintendo World Cup (USA).nes` | `dbd8f793bd841cf967d499f80fd042b7abd40f7a6567db479ff081ccb984af4c` |
| base extension `nes_core.abi3.so` (HEAD build) | `a4366f8217db359c912a423c45eb0f4d56c30c35b4835e7788e81e1c47c26a75` |
| fix extension `nes_core.abi3.so` (patched build) | `f07603bee6643b2af085e3f7d5605506cfbfefed346b50edeca5de5fa5ea01fb` |

Header-stripped payload md5 of the mapper 37 ROM, which is the value the scanner records
in the `md5` column: `690ef25921a067c03218e473c3ba05f6`.

## What this receipt does not cover

The scan is five ROMs, not the 796-ROM library. It establishes that this ROM's motion
class changes and that four neighbours including the one shared-code-path cart do not.
The README's `794 of 796` follows from the previous full-library census plus this
one-row delta, not from a re-run of the full census.

The frame-motion probe used during the fix's development reported first change at frame
8 over 600 frames; the scanner reports frame 7. The two count frames from different
origins. The scanner's number is the one this receipt and the README stand on.
