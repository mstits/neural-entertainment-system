# Parity coverage map — nes_core vs nes-py across the full US library

Output of `scripts/parity_sweep.py --frames 120` against all 794 `.nes` files in `roms/`. Each ROM runs 120 cold-boot idle frames on both emulators; CPU RAM ($0000-$07FF) is diffed byte-by-byte. Raw data: `parity_sweep.json`.

## Summary

| Bucket | Count | % of comparable | Interpretation |
|--------|-------|-----------------|----------------|
| byte_exact | 23 | 5.2% | RAM matches nes-py byte-for-byte. Strongest correctness guarantee. |
| tight | 192 | 43.7% | 1-5 byte drift. Gameplay expected to be correct. |
| moderate | 150 | 34.2% | 6-50 byte drift. Likely playable; small cycle-accuracy edge cases. |
| loose | 64 | 14.6% | 51-500 byte drift. Playable but diverging visibly under scripted input. |
| wide | 10 | 2.3% | >500 byte drift. Likely broken for reference-driven scenarios. |
| theirs_unsupported | 354 | — | nes-py can't test (only 4 mappers supported). |
| ours_panic | 1 | — | Yoshi (USA) truncated ROM dump, not a real emulator bug. |

**Headline number:** of the 439 ROMs that nes-py can even test, **215 (49.0%) are byte-exact or within 5 bytes of nes-py.** Another 150 (34.2%) are within 50 bytes. Only 74 (16.9%) have significant divergence, and 354 additional ROMs only nes_core supports at all.

## byte-exact ROMs (23) — perfect parity

Games where nes_core produces identical CPU RAM to nes-py after 120 idle frames:

- Bad Street Brawler, Baseball, Bionic Commando, BreakThru, California Games
- Castlequest, Clash at Demonhead, Dr. Mario, Dragon Warrior III, Fester's Quest
- Final Fantasy, Godzilla 2, Golf Grand Slam, Kid Kool, Kung-Fu Heroes
- Lee Trevino's Fighting Golf, Magmax, Mega Man, Princess Tomato, Rally Bike
- Super Cars, Terra Cresta, Tetris

Mostly NROM (mapper 0) and MMC1 early titles. Shows the emulator IS functionally correct on a substantial chunk of the library.

## wide-divergence ROMs (10) — investigation priority

Candidates for the next structural fix. Share patterns (most use MMC1 with complex bank-switching patterns or mid-frame graphics tricks):

| ROM | Diff at 120f |
|-----|--------------|
| Casino Kid (USA) | 1569 |
| Casino Kid II (USA) | 1569 |
| Treasure Master (USA) | 743 |
| Break Time - The National Pool Tour (USA) | 694 |
| Fist of the North Star (USA) | 662 |
| New Ghostbusters II (Europe) | 645 |
| Dragon Warrior II (USA) | 602 |
| Alfred Chicken (USA) | 552 |
| Snow Brothers (USA) | 544 |
| Sesame Street 123 (USA) | 543 |

Casino Kid and Casino Kid II at identical 1569 is probably a common shared-code divergence — fix one, likely fixes the other.

## ours_panic (1)

**Yoshi (USA).nes** — truncated iNES header; already known per `project_full_library_compat.md`. Not a real emulator bug.

## theirs_unsupported (354) — nes-py gap

nes-py only supports NROM / CNROM / SxROM / UxROM (4 mappers). nes_core supports 36. The 354 ROMs here include every MMC3 game (Mario 3, Kirby's Adventure, Mega Man 2-6, Castlevania 3, every major MMC3 hit), and every unusual mapper (VRC6, Sunsoft 5B, MMC5 audio, etc.).

These are ROMs where nes_core IS the only reference. Can't diff against nes-py. Need separate validation (e.g. cross-reference against Mesen traces in a future session).

## Action items for next session

1. **Closing the structural CPU gap** (the NMI push timing covered in `project_lainess_reference_findings_2026-04-23.md`) should drop every "tight" ROM to byte-exact and shift the distribution left by a meaningful chunk. Estimate ~215 ROMs would move to byte-exact.

2. **Investigate the 10 wide-divergence ROMs** individually via the lockstep harness. Casino Kid 1+2 should be one fix.

3. **Add per-game replay tests** like `test_zelda_input_replay.py` for ROMs where gameplay-critical RAM bytes need to match. Start with the 23 byte_exact ROMs (easy wins) and extend.

4. **Nothing to do for the 354 theirs_unsupported ROMs** until we have a non-nes-py reference.

## Mesen 2 cross-checks under real input (banked tapes)

Everything above is nes_core against nes-py on 120 **idle** cold-boot frames.
This section is the other axis: a banked action tape replayed on Mesen 2 and
on nes_core, compared frame by frame. Reproduce a row with
`scripts/tracing/nes_core_ram_dump.py` (ROM, tape and input phase all on the
command line) plus `scripts/tracing/mesen_cv_tape_dump.lua` and
`scripts/tracing/diff_ram_tapes.py`.

Before feeding any tape to Mesen, replay it through nes_core at
`frame_skip=1` and show it byte-exact against its own solve lineage. A tape
that cannot reproduce its own lineage says nothing about a second emulator;
skipping that ten-second check produced a wrong divergence verdict on
2026-09-01. `tests/test_nes_core_ram_dump.py` holds the check.

| ROM | Tape | Gameplay-state parity | Receipt |
|---|---|---|---|
| Super Mario Bros. (World) | 12,000 frames of the banked flagship tape (9.6% of it) | HELD: lives, area, mode and every level-transition frame agree; RAM differs by median 17 / p90 27 / max 37 bytes per frame under a one-frame tolerance | `docs/receipts/parity/smb_mesen_2026-09-01/MANIFEST.md` |

Everything not in this table is idle-only (33 ROMs,
`tests/parity/test_mesen_lockstep.py`) or unchecked. CPU RAM only: no row
here compares framebuffers.
