# DO-33 receipt: mapper 105 (NWC1990) boot fix, ROM census re-derivation

2026-09-02. Fix-round addendum to `4204ca0` ("mapper105: fix outer_bank power-on
default so NWC1990 boots live"), requested by the review's finding 3 (correction 14
and the queue row's gate require a second-derivation of the DO-16 scanner result on
the rebuilt extension, with `shasum -a 256` of the `.venv` extension `.so` and the ROM,
before the README numbers move). Same form as DO-38's `mapper228_boot_fix_2026-09-02.md`.

## ROM census scanner (`scripts/rom_library_scan.py`, DO-16 form), run twice independently

Two separate `.venv/bin/python` processes (fresh interpreter each time, `--workers 1`),
against the rebuilt extension, over the three ROMs this fix touches or borders:
Nintendo World Championships 1990 (the ROM this fix moves from static to live) and
Jackal / Super Mario Bros. + Tetris + Nintendo World Cup (mappers 2 and 37, the two
other pre-existing static ROMs in the library), included as a control that this
change does not move them. `--frames 300 --static-frames 300` (matches this fix
round's requirement to show the framebuffer over 300 frames).

No subagent-spawning tool was available to this workflow item (same disclosed
limitation as DO-38's fix round): both runs below are independent fresh Python
processes from this one workflow item, not a second, separately-spawned agent. The
reviewer's own finding 3 supplied an independent second-party re-derivation for the
record (sha256 of the `.so` and the ROM, `cpu_state()` after `reset_no_advance()` =
PC `0xFE89`, 42 distinct frame hashes over 600 frames with first change at frame 1
and 10 distinct colours in the final frame, via its own md5 loop, not the scanner);
this pass independently re-checked those two hash values against the live tree and
they match exactly (see "Artifact and ROM hashes" below).

```
run1:              3/3 ok, static-check: 1 live, 2 static
run2_independent:  3/3 ok, static-check: 1 live, 2 static
```

| ROM | mapper | status/motion (run1) | status/motion (run2) | distinct hashes | first change frame |
|---|---:|---|---|---:|---:|
| Nintendo World Championships 1990 (USA).nes | 105 | ok/live | ok/live | 6 | 1 |
| Jackal (USA).nes | 2 | ok/static | ok/static | 1 | n/a |
| Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes | 37 | ok/static | ok/static | 1 | n/a |

`diff` of the two runs' `status,motion,static_distinct_hashes,static_first_change_frame`
columns: identical (only the per-ROM wall-clock timing column differs, as expected
between two runs). Full CSVs: `mapper105_boot_fix_2026-09-02_run1.csv`,
`mapper105_boot_fix_2026-09-02_run2_independent.csv` (this directory).

**Nintendo World Championships 1990 moves from `static` (pre-fix, 1 distinct hash) to
`live` (6 distinct framebuffer hashes over 300 frames, first change at frame 1 under
the scanner's Start/A-burst schedule). Jackal and SMB+Tetris+NWC are unaffected, still
`static`, exactly as before.** This is the evidence the README's boot count moves from
792/796 to 793/796, and the static-screen count from three to two, with Jackal and
SMB+Tetris+NWC named as the two ROMs remaining static.

## Library count

Pre-fix (per the prior DO-38 receipt, same day): 792 of 796 library ROMs booting live,
three static screens (Jackal, Nintendo World Championships 1990, SMB+Tetris+NWC). This
fix moves exactly one ROM, Nintendo World Championships 1990, from static to live, per
the scanner runs above; no other ROM's classification changes (Jackal and
SMB+Tetris+NWC re-confirmed static by both runs). Result: **793 of 796** library ROMs
booting into a live screen, **two** remaining static screens: **Jackal** (mapper 2) and
**Super Mario Bros. + Tetris + Nintendo World Cup** (mapper 37).

## Artifact and ROM hashes (`shasum -a 256`)

```
0860113700e3a8f4ba0117d4acc07264ce93dfa9a11696894bf66b4f5a07973f  .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so
83f66b806a69c41e341ee4d86d88808d0a22bc35c00fc9e63e291ee31bc50e1a  roms/Nintendo World Championships 1990 (USA).nes
97e0e7d84ce6cca3fd3ec5eabb3fa00252de54f1354156ca2b95b95a39237373  roms/Jackal (USA).nes
814763cdcc615c1988fd414a5a799a12fe2d4b902bdb4fb12b883c7e8f0bcac3  roms/Super Mario Bros. + Tetris + Nintendo World Cup (Europe) (Rev A).nes
```

The `.so` and NWC1990 ROM hashes match the reviewer's own finding-3 re-derivation
exactly (`.so` sha256 `0860113700e3a8f4ba0117d4acc07264ce93dfa9a11696894bf66b4f5a07973f`,
ROM sha256 `83f66b806a69c41e341ee4d86d88808d0a22bc35c00fc9e63e291ee31bc50e1a`),
independently re-computed in this pass rather than copied from the finding.

## README sites updated in this commit

`README.md:5` (headline count, `792 of 796` becomes `793 of 796`); `:177` (Mesen-fidelity
paragraph, `792 of 796` becomes `793 of 796`, `three static screens` becomes `two static
screens`); `:531-535` (Broad-compatibility bullet, `792 of 796 (99.5%)` becomes `793 of
796 (99.6%)`, drops "Nintendo World Championships 1990" from the named static-ROM list,
now Jackal and SMB+Tetris+NWC); `:1190` (Limitations header count, `792/796` becomes
`793/796`); `:1220-1221` ("does not ship" bullet, drops mapper "105" from the named
mapper list, now mappers 2 and 37).

## Fix round (2026-09-02, folded into `4204ca0` via amend)

This receipt file is itself the fix for review finding 3 (the DO-33 commit originally
moved the README numbers without a second-derivation receipt of this form). See the
amended commit message for the other four findings (em-dash cleanup, the pytest-count
clarification, the out-of-scope hardware note, and the MASTER-LIST row-description
alignment).
