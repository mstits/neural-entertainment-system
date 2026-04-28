# Split 03 — Mappers (Top 10)

## Context

Source proposal: [`../full_rust_refactor.md`](../full_rust_refactor.md)
Interview decisions: [`../deep_project_interview.md`](../deep_project_interview.md)
Manifest: [`../project-manifest.md`](../project-manifest.md)

Implements the `Mapper` trait stubbed by split 01 with the top 10
NES mappers, covering ~95% of the games this repo cares about. ROMs
with mappers outside this set refuse to load with a clear error.

## Key decisions inherited from interview

- **Top 10 hard-coded.** No "add as needed" mode. Lock scope.
- **Specifically:** NROM (0), MMC1 (1), UxROM (2), CNROM (3),
  MMC3 (4), MMC5-basic (5), AxROM (7), ColorDreams (11),
  GxROM (66), MMC2 (9). MMC5 is "basic" — the common cases used by
  Castlevania III etc., not the full feature set with extra audio
  channels.
- **Pluggable trait.** Even though the set is hard-coded, the trait
  layout makes it cheap to add mappers later if a target game needs
  one. The decision is "no add-as-needed during this refactor," not
  "make it impossible to extend."

## Deliverables

1. **`Mapper` trait** with `cpu_read/write` (PRG, $8000-$FFFF) and
   `ppu_read/write` (CHR, $0000-$1FFF) methods, plus `irq_pending()`
   and `tick(scanline_cycles)` for mappers that observe PPU activity
   (MMC3's IRQ counter is the main case).
2. **NROM (0)** — the simplest mapper, no banking. zelda.nes uses
   MMC1 not NROM, but smb1.nes is NROM and is the test bed.
3. **MMC1 (1)** — used by Zelda, Metroid. Serial-shifted register,
   PRG/CHR banking modes, mirroring control.
4. **UxROM (2)** — Castlevania, MegaMan 1.
5. **CNROM (3)** — minor games, simple CHR-only banking.
6. **MMC3 (4)** — Castlevania III, MegaMan 3-6, used by many in the
   collection. **IRQ timing is the critical risk** — gets the visual
   scroll-split effects in many games.
7. **MMC5-basic (5)** — Castlevania III's PRG/CHR banking subset
   (no audio, no fill mode, no ExGrafix).
8. **AxROM (7)** — Battletoads, etc. Single-screen mirroring.
9. **ColorDreams (11)** — minor.
10. **GxROM (66)** — Mario+Duck Hunt, Dragon Power.
11. **MMC2 (9)** — Punch-Out!! (latched CHR banking on PPU read).
12. **Mapper resolver:** factory function that takes the iNES mapper
    ID and returns `Box<dyn Mapper>` or a typed enum, or returns a
    descriptive `UnsupportedMapper` error.

## Dependencies

- **Provided by 01-foundation-fork-and-cpu:**
  - `Mapper` trait stub.
  - `RomCart` for PRG/CHR/mirroring metadata.
  - `Bus` integration point.
- **Provided by 02-ppu-renderer** (informational): the PPU calls
  `ppu_read` for CHR fetches. Ensure the trait surface matches the
  PPU's expected access pattern.

## Provides to other splits

- A booted-and-running emulator on the games in `roms/`. Specifically
  the integration split (05) needs zelda.nes, smb1.nes, contra.nes,
  castlevania.nes, megaman2.nes, metroid.nes to all reset cleanly.

## Risks for this split

- **MMC3 IRQ timing.** This is the canonical NES emulator bug. The
  IRQ counter clocks on PPU A12 transitions and the exact transition
  timing matters. Mitigation: blargg's `mmc3_test_2` ROM, plus
  visual check on Castlevania III (status bar must not jitter).
- **MMC1 register write timing.** The serial shift register is
  reset by writes within 2 CPU cycles of each other; getting this
  wrong corrupts saved state. Mitigation: blargg's MMC1 tests.
- **MMC5 scope creep.** MMC5 has many features beyond what we need.
  Lock to the basic case; document what's deliberately omitted.
- **Off-set ROMs in the repo** (Zelda 2, FF3, etc.) using mappers
  outside the top 10 — they'll refuse to load. Document supported
  mapper list in `docs/rust_nes_core.md` so users know.

## Acceptance criteria

1. blargg's `mmc3_test_2` passes.
2. All ROMs in `roms/` that use a top-10 mapper boot to title screen
   without crashing or visible corruption. Specifically: zelda.nes,
   smb1.nes, contra.nes, castlevania.nes, megaman2.nes, metroid.nes.
3. ROMs with unsupported mappers (e.g. ZELDA2.NES with mapper 1
   should work; FF3 with mapper 4 should work; truly unsupported
   like FF2_J.NES or some MMC5+audio games) return a clear error
   message naming the unsupported mapper number.
4. Each implemented mapper has a unit test exercising its banking
   logic with synthetic register writes.
