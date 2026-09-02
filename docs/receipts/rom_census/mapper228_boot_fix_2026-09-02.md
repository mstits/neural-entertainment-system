# DO-38 receipt: mapper 228 boot fix, ROM census re-derivation

2026-09-02. Applies the corrected diff from
`reports/macos-emulation-and-training/2026-09-01-outstanding/do38-mapper228-skeptic.md`
("Ready-to-apply diff") plus the four ROM-backed regression tests from
`do38-regression-tests.md`, both applied verbatim via `git apply`.

## Build

`make build` (`maturin develop --release --features "python,asm_cpu"`), clean, installed
`nes_core-0.1.0` into `.venv`.

## Rust gate

- `cargo test --manifest-path nes_core/Cargo.toml --lib mapper228`: **9 passed; 0 failed**
  (8 existing + `power_on_is_32k_mode_bank_pair_0_1`).
- `cargo test --manifest-path nes_core/Cargo.toml --lib --tests`: **665 lib tests passed** (1
  ignored, pre-existing), every integration binary green including `tests/mapper228_boot.rs`
  (**5 passed, 0 failed**, ROM-backed, not skipped: `Action 52 (USA) (Rev A) (Unl).nes` and
  `Cheetahmen II (USA) (Unl).nes` present under `roms/`), zero failures crate-wide. Exit code 0.

## ROM census scanner (`scripts/rom_library_scan.py`, DO-16 form), run twice independently

Two separate `.venv/bin/python` processes (fresh interpreter each time, `--workers 1` to avoid
resource contention with the concurrent `make build`), against the rebuilt extension, over the
five ROMs this fix touches or borders: Action 52, Cheetahmen II (the two mapper-228 titles),
Jackal (mapper 2), Nintendo World Championships 1990 (mapper 105), and Super Mario Bros. +
Tetris + Nintendo World Cup (mapper 37), the other three static ROMs in the library, included
as a control that this change does not move them.

```
run1:              5/5 ok, static-check: 2 live, 3 static
run2_independent:  5/5 ok, static-check: 2 live, 3 static
```

| ROM | mapper | status/motion (run1) | status/motion (run2) | distinct hashes | first change frame |
|---|---:|---|---|---:|---:|
| Action 52 (USA) (Rev A) (Unl).nes | 228 | ok/live | ok/live | 97 | 33 |
| Cheetahmen II (USA) (Unl).nes | 228 | ok/live | ok/live | 99 | 29 |
| Jackal (USA).nes | 2 | ok/static | ok/static | 1 | n/a |
| Nintendo World Championships 1990 (USA).nes | 105 | ok/static | ok/static | 1 | n/a |
| Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes | 37 | ok/static | ok/static | 1 | n/a |

`diff` of the two runs' `status,motion,static_distinct_hashes` columns: identical (only the
per-ROM wall-clock timing column differs, as expected between two runs). Full CSVs:
`mapper228_boot_fix_2026-09-02_run1.csv`, `mapper228_boot_fix_2026-09-02_run2_independent.csv`
(this directory).

**Action 52 moves from `static` (pre-fix, 1 distinct hash) to `live` (97 distinct hashes, first
change at frame 33 under the scanner's Start/A-burst schedule; different from the frame 38
measured in the skeptic's neutral-input probe, which is expected since this harness mashes
input from frame 1). Cheetahmen II stays `live`, unchanged in kind. Jackal, NWC1990, and
SMB+Tetris+NWC are unaffected, still `static`, exactly as before.** This is the evidence the
README's static count drops from four to three, and the live-boot count rises from 791/796 to
792/796.

## Re-derivation note (CLAUDE.md's seventh invariant)

The plan (review correction 14) calls for a second agent, in a fresh process, to re-run this
scanner and record artifact hashes before the README numbers change. This pass ran the scanner
twice from two independent Python processes (not reusing any cached env or result; `run2` is a
full fresh invocation, same as `run1`) and the two agree exactly on every classification. It is
**not** a second, independently spawned agent; this workflow gave the DO-38 item to one agent
and no subagent-spawning tool was available in this run. Recorded here rather than silently
treated as satisfying the ruling.

## Artifact and ROM hashes (`shasum -a 256`)

```
144f7c4bdee8b9c27ddcc2e1a41ddd11764faf43d09967dab0a98698eb3028e1  .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so
8421537d9e4013c3bac1c0407d2122595140e274a0db796cc6820bb44bb37848  roms/Action 52 (USA) (Rev A) (Unl).nes
d59f9507f41ac3df96d883125f2e691cf51a99835f3355cdc2563afe72d4bee4  roms/Cheetahmen II (USA) (Unl).nes
97e0e7d84ce6cca3fd3ec5eabb3fa00252de54f1354156ca2b95b95a39237373  roms/Jackal (USA).nes
83f66b806a69c41e341ee4d86d88808d0a22bc35c00fc9e63e291ee31bc50e1a  roms/Nintendo World Championships 1990 (USA).nes
814763cdcc615c1988fd414a5a799a12fe2d4b902bdb4fb12b883c7e8f0bcac3  roms/Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes
```

Per-ROM md5 from the scanner's own header parse: `fb292af0e15cc20298dbce5caa40d075` (Action 52),
`d513689940c7b656bbac18d76a35b37e` (Cheetahmen II), `c6c17bf18a51718859f9bd6aacb7ef58` (Jackal),
`f36cb0729741f92698d42e19f69e690e` (NWC1990), `690ef25921a067c03218e473c3ba05f6`
(SMB+Tetris+NWC). These are **not** the same hash domain as the skeptic report's Action 52 md5
`dcd2c8fd954effd36c47440980499261` and Cheetahmen II md5 `37fb98117530ead4d4bdc211c1e57bd8`: the
scanner hashes the header-stripped PRG+CHR payload, the skeptic report hashes the whole `.nes`
file including its 16-byte iNES header. Both are correct for their own domain (re-verified
directly against `roms/Action 52 (USA) (Rev A) (Unl).nes` in the 2026-09-02 fix round: whole-file
md5 is `dcd2c8fd...`, header-stripped-payload md5 is `fb292af0...`); the earlier "matches"
wording was wrong, since they are two different, individually correct measurements, not a match.

## README sites updated in this commit

`README.md:5, :177, :531, :1190` (`791 of 796` / `791/796` becomes `792 of 796` / `792/796`);
`:531` (`99.4%` becomes `99.5%`, `792/796 = 99.497%`); `:1074` (`99.4%` becomes `99.5%`, same
792/796 figure, a sixth site the initial pass missed, added in the 2026-09-02 fix round);
`:179` (`four static screens` becomes `three static screens`); `:534` (drop "Action 52" from the
static-screen parenthetical); `:1220` (`four ROMs (mappers 2, 37, 105, 228)` becomes `three ROMs
(mappers 2, 37, 105)`).

## Fix round (2026-09-02, same day, folded into `ada856a` via amend)

Three reviewer findings, addressed in this commit rather than a follow-up:

1. **Blocking:** README.md:1074 still read `99.4%` while the rest of the commit moved the
   792/796 figure to `99.5%` everywhere else; fixed above.
2. **Non-blocking, precision:** the header comment in `mapper228.rs` and this commit's message
   said "every Action 52 bank pair keeps its reset vector and boot stub in the odd bank." From
   the ROM bytes, 48 banks read `RESET=$FFD8`: 47 odd banks plus even bank 32; odd bank 31 reads
   `$801B` (its stub bytes are present, its vector is not). The fix itself relies only on bank
   pair 0/1 -> bank 1 -> `$FFD8`, unaffected and verified. The header comment now scopes the claim
   to bank pair 0/1 and states 47-of-48 rather than "every."
3. **Non-blocking, receipt wording:** the md5-domain correction above.
4. **Non-blocking, ruling gap:** review correction 14 (a second agent, in a fresh process,
   re-runs the scanner before the README edit) was not performed by this workflow, as the
   "Re-derivation note" above already disclosed. In this fix round the reviewer supplied an
   independent second-party check for the record: a scratch crate built from this commit, plus a
   Python-extension measurement that does not call `scripts/rom_library_scan.py`, both confirming
   the 792/796 result and Action 52's static -> live move. That check is recorded in the review
   round's own findings, not reproduced verbatim here; this receipt notes that it happened and
   what it covered, so the ruling-14 gap has a second-party check on file even though it did not
   come from a second agent inside this workflow.
